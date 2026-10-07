//! Stage 1 producer-build handoff: this module is split into a game-side
//! CPU half and a context-side GPU half with identical per-mesh semantics to
//! the historical single `stageInstancedMesh` path.
//!
//! - CPU half (`stageInstancesCpu` / `stageSegmentCpu`, sg-free, any
//!   non-pool thread under update-vs-prepare exclusion): pure per-instance
//!   world matrices/AABB (parallel at >=256 instances), combined bounds,
//!   visible count, matrix-bytes hash, transparent back-to-front sort by eye.
//!   Appends each mesh's segment into the caller-provided scratch (the P7
//!   back-slot `primary.instance_matrices` for the producer build)
//!   and records the small plain preview (`Mesh.instance_preview`); the
//!   matrix bytes live ONLY in the scratch segment
//!   `[scratch_lo, scratch_lo + count)`.
//! - GPU half (`stageInstancesGpu`, context thread only): buffer
//!   create/update/dedup/growth/retire exactly as before (FAILED/VALID
//!   handling, hash+count dedup gate), then publishes `instance_render`
//!   (bounds/count/`staged_frame`) only after the GPU half succeeded (or was
//!   skipped: no context, empty set).
//! - Latch (`stageInstancesLatch`, context side): consumes the slot-owned
//!   `staged_instances` records of one staged begin (frozen by
//!   `freezeStagedRecords` during the producer build) + the slot scratch.
//!   Each record carries its scratch segment, the staged matrices
//!   (bounds/hash/count), and the prior resolved state, so the latch performs
//!   NO live reads of any kind: no `instance_preview`, no `instance_render`,
//!   no mesh-list dereference, and it never touches `record.mesh` (the
//!   pointer rides along as an opaque identity token for the game-side
//!   commit below — a destroyed mesh's dangling pointer is never even
//!   compared here, so no UAF is possible). A mesh-list mutation between
//!   build and latch (reorder/destroy) does NOT fail-close here: the latch
//!   stages whatever the slot owns and mirrors the outcome into the record.
//!   Identity is enforced one step later, game-side, by the commit.
//!   Slices are bounds-checked so a contract violation (scratch reset
//!   between build and latch) can never slice out of bounds.
//! - Commit (`commitPublishedRecords`, game side, at the start of the NEXT
//!   producer build): applies the published slot's latch outcomes to
//!   the live meshes — the old guarded `instance_render` write-back, moved
//!   game-side and ordered after publish, never concurrent with the context.
//!   The O(1) aliveness/identity guard (`meshes[record.mesh_index] ==
//!   record.mesh` pointer compare without dereferencing the record mesh
//!   first, then the uid compare) plus the live skip re-checks (LOD child,
//!   GPU-pending, emptied) run here, where live reads are legal. On guard or
//!   skip failure the mesh keeps its previous complete state and the record
//!   stays as published (the patch already resolved the payload from it);
//!   the next funded build recomputes the mesh and the next latch consumes
//!   it. Fail-closed records (`staged_frame` never advanced by the GPU half)
//!   and stale generations (record frame != committed slot frame) never
//!   commit. Meshes with no record for the latched build (OOM-skipped
//!   segment, created after the build) never reach the commit — their
//!   previous complete `instance_render` stands, coherent, with no partial
//!   publish.
//! - Immediate (`stageInstances` / `stageInstancedMesh`, unchanged
//!   behavior): the direct CPU+GPU path for standalone immediate users
//!   (tests/tooling/one-off builds via `InstanceStageContext`), NOT a
//!   Scene renderer path.
//!
//! No per-field atomics (phase ownership), no second GPU buffer versioning,
//! no new dependencies. GPU handles stay borrowed under the P3 epochs.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const Mesh = @import("../mesh.zig").Mesh;
const InstancedMesh = @import("../mesh.zig").InstancedMesh;
const InstancePreviewState = @import("../mesh.zig").InstancePreviewState;
const InstanceRenderState = @import("../mesh.zig").InstanceRenderState;
const instancePairMode = @import("../mesh.zig").instancePairMode;
const normalizeInstancePingPong = @import("../mesh.zig").normalizeInstancePingPong;
const StagedInstanceRecord = @import("../mesh.zig").StagedInstanceRecord;
const material_mod = @import("../material.zig");
const Material = material_mod.Material;
const jobs = @import("../jobs.zig");
const gpu_thread = @import("../gpu_thread.zig");
const upload_meter = @import("../gpu_upload_meter.zig");
const gpu_retire = @import("gpu_retire.zig");

/// Mirrors render_queue.materialIsTransparent: a mesh is transparent when its
/// material opts into .blend alpha mode. Kept local so this module never
/// imports render_queue.zig (no import cycle with its staging caller).
fn isTransparentMaterial(mat: ?Material) bool {
    if (mat) |m| return m.isTransparent();
    return false;
}

/// Pure per-instance world matrix: TRS(instance) * source.base_matrix — the
/// same formula as InstancedMesh.computeWorldMatrix, but on a const pointer:
/// reads only instance TRS + source base_matrix, never touches the game
/// caches (InstancedMesh.cached_*/dirty/last_* stay valid for picking and
/// game APIs; Mesh regular cached_aabb is staging-untouched too).
fn instanceWorldMatrix(inst: *const InstancedMesh) Mat4 {
    const trs = Mat4.fromRotationTranslationScale(inst.position, inst.rotation, inst.scaling);
    return Mat4.mul(trs, inst.source_mesh.base_matrix);
}

/// Pure per-instance world AABB from an already-computed world matrix.
/// Mirrors InstancedMesh.updateCachedTransforms without writing it.
fn instanceWorldAABB(inst: *const InstancedMesh, world: Mat4) BoundingBox {
    return inst.source_mesh.local_bounding_box.transform(world);
}

const ParallelInstanceStage = struct {
    instances: []*InstancedMesh,
    span: usize,
    chunk_aabbs: []BoundingBox,
    chunk_visible_counts: []usize,
    /// Pre-sized to out_base + instances.len; chunk c owns segment
    /// [out_base + c*span, out_base + c*span+span) and packs its visible
    /// matrices densely at the segment start (a chunk holds at most span
    /// visibles, so the segment always fits). The serial tail compacts the
    /// segments into [out_base, out_base + total_visible). out_base is the
    /// append offset for the concatenated multi-mesh scratch (0 for the
    /// single-mesh immediate path).
    out_matrices: []Mat4,
    out_base: usize = 0,
    /// Parallel trail to `out_matrices`: chunk c owns the same segment in
    /// `out_uids` and packs visible instance uids densely at its start.
    out_uids: []u64 = &.{},
    uid_base: usize = 0,

    fn runChunks(stage: *ParallelInstanceStage, start: usize, end: usize) void {
        for (start..end) |chunk_id| {
            const rel_lo = chunk_id * stage.span;
            if (rel_lo >= stage.instances.len) {
                // Ceil-division span can overshoot on the tail chunk when
                // chunk_count divides unevenly: an empty chunk contributes
                // nothing (its slice would otherwise run lo > hi).
                stage.chunk_aabbs[chunk_id] = BoundingBox.zero;
                stage.chunk_visible_counts[chunk_id] = 0;
                continue;
            }
            const rel_hi = @min(rel_lo + stage.span, stage.instances.len);
            const lo = stage.out_base + rel_lo;
            const ulo = stage.uid_base + rel_lo;
            var aabb = BoundingBox.zero;
            var dst = lo;
            var udst = ulo;
            for (stage.instances[rel_lo..rel_hi]) |inst| {
                if (!inst.is_visible) continue;
                const world = instanceWorldMatrix(inst);
                stage.out_matrices[dst] = world;
                stage.out_uids[udst] = inst.uid;
                dst += 1;
                udst += 1;
                const box = instanceWorldAABB(inst, world);
                if (aabb.isValid()) {
                    aabb = aabb.merge(box);
                } else {
                    aabb = box;
                }
            }
            stage.chunk_aabbs[chunk_id] = aabb;
            stage.chunk_visible_counts[chunk_id] = dst - lo;
        }
    }
};

/// Test-only growth-failure injection (P5 5d): armed via
/// testArmGrowthFailOnce(), the next growth allocation synthesizes a nonzero
/// FAILED buffer (allocBuffer + failBuffer) instead of calling sg.makeBuffer,
/// driving the REAL `.FAILED` handling below (destroy the failed slot, keep
/// the previous complete published state, retire nothing). State is private:
/// the only writers/readers are the two context-thread-only accessors below,
/// never a freely mutable production switch. A zero id at consume time (dry
/// pool) falls through to the real makeBuffer so the failure mode stays
/// pool-exhaustion (covered by P5 5a), never a vacuous injection.
var test_inject_growth_fail_once: bool = false;
/// Exact buffer id the last armed injection failed (0 = none consumed yet).
/// Lets the test assert the failed handle itself reached INVALID, not just
/// that the pool has room.
var test_last_injected_fail_id: u32 = 0;

/// Arms one injected growth failure. Context thread only (asserted): the
/// staging GPU half never runs anywhere else.
pub fn testArmGrowthFailOnce() void {
    gpu_thread.assertOnContextThread();
    test_last_injected_fail_id = 0;
    test_inject_growth_fail_once = true;
}

/// Exact id the last armed injection failed, or 0 when no injection has been
/// consumed. Context thread only (asserted).
pub fn testLastInjectedFailId() u32 {
    gpu_thread.assertOnContextThread();
    return test_last_injected_fail_id;
}

/// Minimal per-frame input for instance staging, extracted from
/// FrameCullContext (see render_queue.zig). The producer build pre-stages
/// through this before the shadow pass so ShadowPass.prepare snapshots the
/// same frame's published render state (bounds/buffer/count) instead of the
/// previous frame's. The `staged_frame != frame_id` guard keeps staging once
/// per frame, shared by all view queues.
///
/// Carries the staging scratch list (`RenderQueues.instance_matrices`)
/// directly instead of `*RenderQueues`, so this module never imports
/// render_queue.zig and no import cycle can form.
pub const InstanceStageContext = struct {
    allocator: std.mem.Allocator,
    instance_matrices: *std.ArrayListUnmanaged(Mat4),
    thread_pool: ?*jobs.Pool,
    frame_id: u64,
    eye: Vec3,
    /// P5: retire queue for a grown-away old instance buffer (threaded by
    /// Scene; the same allocator funds the enqueue). Null marks a standalone
    /// low-level caller (tests, one-off tooling) and is a caller obligation:
    /// the caller asserts no live snapshot still references the old buffer,
    /// so it may be destroyed immediately — legal because the sg block below
    /// already requires the context thread. A null queue is NOT a guarantee
    /// about snapshots; it is the caller taking responsibility for them.
    retire_queue: ?*gpu_retire.GpuRetireQueue = null,
};

/// Per-frame instance-matrix staging for one mesh: pure per-instance
/// transforms/AABB (parallel at >=256 instances, single pass — no duplicate
/// transforms), combined bounds / visible count, transparent back-to-front
/// sort by eye, and sg instance-buffer create/update on the context thread.
/// Publishes the coherent result into `mesh.instance_render` atomically and
/// advances `staged_frame` only then: a failed stage (scratch OOM,
/// new-buffer failure) keeps the previous complete published state and may
/// retry on a later call, while a completed stage stays once-per-frame even
/// across shadow + N cameras.
///
/// Historical immediate path (standalone direct users): runs the CPU half
/// into the caller's scratch and immediately the GPU half. Behavior is
/// unchanged — see `stageSegmentCpu` + `stageInstancesGpu`.
pub fn stageInstancedMesh(sc: InstanceStageContext, mesh: *Mesh) void {
    // Stage-2A identity: lazy uid before any early-out (pending meshes get
    // one too; idempotent, no behavior change).
    _ = mesh.ensureUid();
    // Deferred-creation meshes have no vertex/index buffers yet; staging
    // instance data for them would produce a draw against invalid handles.
    if (mesh.gpu_pending) return;
    const st = &mesh.instance_render;
    if (st.staged_frame == sc.frame_id) return;

    sc.instance_matrices.clearRetainingCapacity();
    const cpu = stageSegmentCpu(sc.allocator, sc.instance_matrices, 0, sc.thread_pool, sc.eye, mesh) catch return;
    stageInstancesGpu(.{
        .allocator = sc.allocator,
        .retire_queue = sc.retire_queue,
        .frame_id = sc.frame_id,
    }, mesh, sc.instance_matrices.items, cpu);
}

/// CPU half of instance staging (stage 1, game side): pure per-instance
/// world matrices/AABB (parallel at >=256 instances, single pass),
/// combined bounds, matrix-bytes hash, and transparent back-to-front sort
/// by eye. Appends the mesh's segment at `lo` (the caller owns everything
/// before it) and returns the bounds/hash; the segment length (visible
/// count) is `scratch.items.len - lo` on success.
///
/// sg-free: safe on any non-pool thread under update-vs-prepare exclusion.
/// OOM fails clean: the partial segment is truncated back to `lo` (scratch
/// intact for earlier meshes) and the error propagates — the caller must
/// leave that mesh's preview (and `instance_render`) untouched so a later
/// build/latch may retry.
pub const CpuStageResult = struct {
    bounds: BoundingBox,
    hash: u64,
    /// Ordered identity layout of the staged segment: Wyhash over stable
    /// per-instance uids in exact upload order (source entry first when
    /// visible, post transparent re-sort). Feeds the velocity pairing gate.
    layout_hash: u64,
};

/// Hashes an ordered uid sequence (the velocity layout identity).
pub fn layoutHashFor(uids: []const u64) u64 {
    return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(uids));
}

pub fn stageSegmentCpu(
    allocator: std.mem.Allocator,
    scratch: *std.ArrayListUnmanaged(Mat4),
    lo: usize,
    thread_pool: ?*jobs.Pool,
    eye: Vec3,
    mesh: *Mesh,
) std.mem.Allocator.Error!CpuStageResult {
    std.debug.assert(lo <= scratch.items.len);
    var combined_aabb = BoundingBox.zero;
    // Stable identity FIRST (serial, before any parallel read): instance
    // uids feed the velocity layout hash, never addresses — a deleted and
    // recreated instance must not alias its predecessor.
    _ = mesh.ensureUid();
    for (mesh.instances.items) |inst| _ = inst.ensureUid();
    // Parallel uid trail: uids[i] is the identity of scratch.items[lo + i].
    // Allocated/freed here (one temp per mesh build); OOM restores the
    // scratch to `lo` and propagates, like every other staging OOM.
    var uids: std.ArrayListUnmanaged(u64) = .empty;
    defer uids.deinit(allocator);
    // Babylon semantics (`Mesh._renderWithInstances`): the instanced batch is
    // the visible instances PLUS one entry for the source mesh's own world
    // matrix (its instance buffer is sized `(visible + 1) * 16` floats), so a
    // mesh with instances is still drawn at its own transform unless it is
    // hidden. `mesh.is_visible` gates only that source entry — visible
    // instances keep drawing when the source is hidden.
    const self_visible = mesh.is_visible;
    const self_count: usize = if (self_visible) 1 else 0;
    const self_world = if (self_visible) mesh.getWorldMatrix() else Mat4.identity;
    const self_aabb: ?BoundingBox = if (self_visible) mesh.getWorldBoundingBox() else null;
    const use_parallel = if (thread_pool) |pool|
        pool.workerCount() > 0 and mesh.instances.items.len >= 256
    else
        false;
    if (use_parallel) {
        const pool = thread_pool.?;
        const chunk_count = @min((pool.workerCount() + 1) * 2, 64);
        const span = (mesh.instances.items.len + chunk_count - 1) / chunk_count;

        // Scratch upper bound first: OOM here stages nothing (scratch
        // unchanged — resize fails before mutating) and a later call may
        // retry. The uid trail resizes in lockstep (same OOM contract).
        try scratch.resize(allocator, lo + self_count + mesh.instances.items.len);
        try uids.resize(allocator, self_count + mesh.instances.items.len);
        errdefer scratch.items.len = lo;
        if (self_visible) {
            scratch.items[lo] = self_world;
            uids.items[0] = mesh.uid;
        }

        var chunk_aabbs_buf: [64]BoundingBox = undefined;
        var chunk_visible_buf: [64]usize = undefined;

        var stage = ParallelInstanceStage{
            .instances = mesh.instances.items,
            .span = span,
            .chunk_aabbs = chunk_aabbs_buf[0..chunk_count],
            .chunk_visible_counts = chunk_visible_buf[0..chunk_count],
            .out_matrices = scratch.items,
            .out_base = lo + self_count,
            .out_uids = uids.items,
            .uid_base = self_count,
        };

        pool.forkJoin(ParallelInstanceStage, &stage, ParallelInstanceStage.runChunks, chunk_count);

        // Serial tail: merge chunk AABBs, compact the per-chunk segments
        // down to [lo, lo + total_visible). Destinations never overtake
        // sources (offset_c <= c*span relative to lo: every earlier chunk
        // contributes at most span), so the forward copy is overlap-safe.
        // The source entry, when present, already occupies [lo, lo + self_count)
        // and is compacted to the head of the [lo, ...) segment. The uid
        // trail compacts identically, so uids[i] stays the identity of
        // scratch.items[lo + i].
        var total_visible: usize = self_count;
        if (self_aabb) |box| combined_aabb = box;
        for (0..chunk_count) |c| {
            const count = chunk_visible_buf[c];
            const src_lo = lo + self_count + c * span;
            if (count > 0) {
                std.mem.copyForwards(
                    Mat4,
                    scratch.items[lo + total_visible .. lo + total_visible + count],
                    scratch.items[src_lo .. src_lo + count],
                );
                std.mem.copyForwards(
                    u64,
                    uids.items[total_visible .. total_visible + count],
                    uids.items[self_count + c * span .. self_count + c * span + count],
                );
                total_visible += count;
            }
            if (chunk_aabbs_buf[c].isValid()) {
                if (combined_aabb.isValid()) {
                    combined_aabb = combined_aabb.merge(chunk_aabbs_buf[c]);
                } else {
                    combined_aabb = chunk_aabbs_buf[c];
                }
            }
        }
        scratch.items.len = lo + total_visible;
    } else {
        try scratch.ensureUnusedCapacity(allocator, self_count + mesh.instances.items.len);
        try uids.ensureTotalCapacity(allocator, self_count + mesh.instances.items.len);
        if (self_visible) {
            scratch.appendAssumeCapacity(self_world);
            uids.appendAssumeCapacity(mesh.uid);
        }
        for (mesh.instances.items) |inst| {
            if (!inst.is_visible) continue;
            const world = instanceWorldMatrix(inst);
            scratch.appendAssumeCapacity(world);
            uids.appendAssumeCapacity(inst.uid);
            const box = instanceWorldAABB(inst, world);
            if (combined_aabb.isValid()) {
                combined_aabb = combined_aabb.merge(box);
            } else {
                combined_aabb = box;
            }
        }
    }
    const segment = scratch.items[lo..];
    const uid_segment = uids.items[0..segment.len];
    std.debug.assert(uid_segment.len == segment.len);

    // Per-instance transparency sorting (OIT):
    // When the instanced mesh is transparent or a decal, sort its instance
    // matrices back-to-front relative to the camera eye. Farthest instances
    // render first, blending nearer instances over them correctly.
    // The uid trail sorts WITH the matrices (same permutation), so the
    // layout hash below always describes the exact upload order — a re-sort
    // changes the layout and zeroes the batch instead of mispairing.
    if (segment.len > 1 and (isTransparentMaterial(mesh.material) or mesh.is_decal)) {
        const Pair = struct {
            m: Mat4,
            u: u64,
        };
        const Ctx = struct {
            eye: Vec3,
            pub fn less(c: @This(), a: Pair, b: Pair) bool {
                const pos_a = Vec3.new(a.m.m[12], a.m.m[13], a.m.m[14]);
                const pos_b = Vec3.new(b.m.m[12], b.m.m[13], b.m.m[14]);
                const dist_a = pos_a.sub(c.eye).lengthSq();
                const dist_b = pos_b.sub(c.eye).lengthSq();
                if (dist_a != dist_b) {
                    return dist_a > dist_b; // back-to-front: farthest first
                }
                for (0..16) |i| {
                    if (a.m.m[i] != b.m.m[i]) return a.m.m[i] < b.m.m[i];
                }
                // Total order: never leave equal-but-distinct pairs
                // unordered, so serial and parallel paths agree exactly.
                return a.u < b.u;
            }
        };
        // Pair sort (fallible alloc, same OOM contract): matrices and uids
        // travel as one record, so the uid trail always describes the exact
        // upload order — never matrices alone.
        const pairs = allocator.alloc(Pair, segment.len) catch {
            scratch.items.len = lo;
            return error.OutOfMemory;
        };
        defer allocator.free(pairs);
        for (pairs, segment, uid_segment) |*p, m, u| p.* = .{ .m = m, .u = u };
        std.mem.sort(Pair, pairs, Ctx{ .eye = eye }, Ctx.less);
        for (pairs, segment, uid_segment) |p, *m, *u| {
            m.* = p.m;
            u.* = p.u;
        }
    }

    return .{
        .bounds = combined_aabb,
        .hash = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(segment)),
        .layout_hash = layoutHashFor(uid_segment),
    };
}

/// GPU half of instance staging (stage 1 latch, context thread only):
/// buffer create/update/dedup/growth/retire for the CPU-staged `matrices`
/// (FAILED/VALID handling, hash+count dedup gate — identical to the
/// historical inline path), then publishes `instance_render`
/// (bounds/count/`staged_frame`) only after the GPU half succeeded (or was
/// skipped: no context, empty set). Never reads live instance TRS — only
/// the staged slice plus the CPU result.
pub const PrevFrameSource = struct {
    /// Generation that staged this payload (`FrameDrawSlot.frame_id`).
    frame_id: u64,
    /// That slot's staged instance records (frozen at build time).
    records: []const StagedInstanceRecord,
    /// That slot's concatenated matrix scratch
    /// (`primary.instance_matrices`).
    scratch: []const Mat4,
};

/// Retained prior-generation lookup for velocity pairing (pure, sg-free):
/// finds the slot payload staged under `prior_frame` and, within it, the
/// record for `uid` with exactly `count` matrices; returns that record's
/// own scratch slice, bounds-checked. Anything missing or mismatched
/// (unknown generation, unknown uid, zero uid, count mismatch, OOB slice)
/// yields null — the caller falls back to zero motion. Never a live read,
/// never `record.mesh` (identity is uid-only, like the patch).
pub fn findPrevMatrices(
    prev_frames: []const PrevFrameSource,
    prior_frame: u64,
    uid: u64,
    count: usize,
) ?[]const Mat4 {
    if (prior_frame == std.math.maxInt(u64)) return null;
    if (uid == 0) return null;
    if (count == 0) return null;
    for (prev_frames) |pf| {
        if (pf.frame_id != prior_frame) continue;
        for (pf.records) |*pr| {
            if (pr.uid != uid) continue;
            if (pr.count != count) return null;
            const lo: usize = pr.scratch_lo;
            const end = lo + count;
            if (lo > pf.scratch.len or end > pf.scratch.len) return null;
            return pf.scratch[lo..end];
        }
    }
    return null;
}

pub const GpuStageContext = struct {
    allocator: std.mem.Allocator,
    frame_id: u64,
    /// P5: retire queue for a grown-away old instance buffer (threaded by
    /// Scene; the same allocator funds the enqueue). Null marks a standalone
    /// low-level caller (tests, one-off tooling) and is a caller obligation:
    /// the caller asserts no live snapshot still references the old buffer,
    /// so it may be destroyed immediately — legal because the sg block below
    /// already requires the context thread. A null queue is NOT a guarantee
    /// about snapshots; it is the caller taking responsibility for them.
    retire_queue: ?*gpu_retire.GpuRetireQueue = null,
    /// Retained slot payloads for the velocity pairing lookup (set once per
    /// latch by the prepare path from `scene.draws.slots`; empty by default
    /// so standalone callers compile unchanged). The latch resolves each
    /// record's prior generation into `prev_matrices` below.
    prev_frames: []const PrevFrameSource = &.{},
    /// Previous generation's staged matrices for the record currently being
    /// latched (set per-record by `stageInstancesLatch` via
    /// `findPrevMatrices`; null = unavailable). The pair branch only trusts
    /// it when its length equals the staged matrices length.
    prev_matrices: ?[]const Mat4 = null,
};

pub fn stageInstancesGpu(gctx: GpuStageContext, mesh: *Mesh, matrices: []const Mat4, cpu: CpuStageResult) void {
    // Deferred-creation meshes have no vertex/index buffers yet; staging
    // instance data for them would produce a draw against invalid handles.
    if (mesh.gpu_pending) return;
    stageInstancesGpuState(gctx, &mesh.instance_render, matrices, cpu);
}

/// Mesh-agnostic GPU core: buffer create/update/dedup/growth/retire for the
/// CPU-staged `matrices` (FAILED/VALID handling, hash+count dedup gate —
/// identical to the historical inline path), then publishes into `st`
/// (bounds/count/`staged_frame`) only after the GPU half succeeded (or was
/// skipped: no context, empty set). Never reads live instance TRS — only
/// the staged slice plus the CPU result. The latch seeds `st` from the
/// slot-owned record (never from live `instance_render`) and mirrors the
/// result back; the immediate helper passes `&mesh.instance_render` directly.
pub fn stageInstancesGpuState(gctx: GpuStageContext, st: *InstanceRenderState, matrices: []const Mat4, cpu: CpuStageResult) void {
    if (st.staged_frame == gctx.frame_id) return;
    // Adopt any legacy/foreign handle state into the buffers[] invariant
    // first (pure, sg-free): later branches may assume buffer aliases a
    // slot and prev aliases a known handle, so the old owned handle can
    // never be lost off the books.
    normalizeInstancePingPong(st);

    const active_count = matrices.len;

    if (active_count > 0 and sg.isvalid()) {
        // Владение GPU (P1): CPU-стейджинг выше может идти с воркеров
        // пула, но создание/обновление instance-буфера — только
        // context-поток. Staged begin выполняется на нём
        // (покрыт ассертом staged begin/render во всех build modes), это внутренний
        // трипвайр: при выносе prepare на update-поток sg-блок уедет за
        // handoff в quiesced drain.
        gpu_thread.assertOnContextThread();
        const items = matrices[0..active_count];
        if (st.buffer.id == 0 or st.capacity < active_count) {
            // Growth: the new buffer is staged FIRST (create + the single
            // allowed update of this sokol frame), and only then is the old
            // one retired — never destroyed immediately, so in-flight
            // snapshots keep reading valid geometry.
            //
            // Sokol pinned API: .dynamic_update buffers are created with
            // .size only (no initial .data) and get at most ONE
            // sg.updateBuffer per buffer per sokol frame; the fresh buffer
            // takes this frame's single update, the retired one takes none.
            const min_cap: usize = 16;
            const new_cap = std.math.ceilPowerOfTwo(usize, @max(active_count, @max(st.capacity * 2, min_cap))) catch @max(active_count, st.capacity * 2);
            // Test-only injection (P5 5d): synthesize a nonzero FAILED
            // buffer instead of the real makeBuffer, so the validity check
            // below exercises its `.FAILED` arm on the real path. The flag
            // auto-resets; a dry pool (id 0) falls through to the real call
            // so the mode stays pool-exhaustion, never vacuous.
            var injected_fail = false;
            var new_buf: sg.Buffer = undefined;
            if (test_inject_growth_fail_once) {
                test_inject_growth_fail_once = false;
                const fb = sg.allocBuffer();
                if (fb.id != 0) {
                    sg.failBuffer(fb);
                    test_last_injected_fail_id = fb.id;
                    new_buf = fb;
                    injected_fail = true;
                }
            }
            if (!injected_fail) {
                new_buf = sg.makeBuffer(.{
                    .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                    .size = new_cap * @sizeOf(Mat4),
                });
            }
            // A failed makeBuffer may still hand out a nonzero id in
            // FAILED resource state (id == 0 means pool exhaustion only),
            // so validity is checked via queryBufferState, not the id.
            if (new_buf.id == 0 or sg.queryBufferState(new_buf) != .VALID) {
                // Coherent failure: keep the previous complete published
                // state (new bounds/count never mix with old geometry),
                // release the failed pool slot, publish nothing — a later
                // call may retry.
                if (new_buf.id != 0) sg.destroyBuffer(new_buf);
                return;
            }
            sg.updateBuffer(new_buf, sg.asRange(items));
            // Учёт динамики: active_count матриц Mat4 (потокобезопасно — счётчик атомарный).
            upload_meter.record(active_count * @sizeOf(Mat4));
            const old0 = st.buffers[0];
            const old1 = st.buffers[1];
            if (gctx.retire_queue) |q| {
                if (old0.id != 0) q.retireBuffer(gctx.allocator, old0);
                if (old1.id != 0 and old1.id != old0.id) q.retireBuffer(gctx.allocator, old1);
            } else {
                if (old0.id != 0) sg.destroyBuffer(old0);
                if (old1.id != 0 and old1.id != old0.id) sg.destroyBuffer(old1);
            }
            st.buffers[0] = new_buf;
            st.buffers[1] = .{};
            st.active_slot = 0;
            st.buffer = new_buf;
            st.prev_buffer = new_buf;
            // Growth collapses history: prev aliases the fresh upload.
            st.prev_frame = gctx.frame_id;
            st.capacity = new_cap;
            st.hash = cpu.hash;
            st.uploaded_count = active_count;
            st.layout_hash = cpu.layout_hash;
        } else {
            if (active_count != st.uploaded_count or cpu.hash != st.hash) {
                // Pairing gate (see instancePairMode): same count AND same
                // ordered identity layout pairs by index; anything else
                // zeroes the whole batch — no slot flip, prev aliases cur.
                if (instancePairMode(st.uploaded_count, active_count, st.layout_hash, cpu.layout_hash) == .zero) {
                    sg.updateBuffer(st.buffer, sg.asRange(items));
                    st.prev_buffer = st.buffer;
                    st.prev_frame = gctx.frame_id;
                } else {
                    // .pair: SAME cur buffer id (committed renderer
                    // contract), updated in place. Real per-instance
                    // velocity comes from CPU history: the previous
                    // generation's staged matrices go into the OTHER slot
                    // BEFORE cur is overwritten (one update per buffer per
                    // frame — different buffers, legal).
                    const prev_items = gctx.prev_matrices;
                    const have_prev = prev_items != null and prev_items.?.len == active_count;
                    if (have_prev) {
                        // Ensure the other slot (rollback-safe create;
                        // FAILED ids are destroyed, never adopted).
                        var other = st.buffers[st.active_slot ^ 1];
                        if (other.id != 0 and sg.queryBufferState(other) != .VALID) {
                            sg.destroyBuffer(other);
                            other = .{};
                        }
                        if (other.id == 0) {
                            other = sg.makeBuffer(.{
                                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                                .size = st.capacity * @sizeOf(Mat4),
                            });
                            if (other.id == 0 or sg.queryBufferState(other) != .VALID) {
                                if (other.id != 0) sg.destroyBuffer(other);
                                other = .{};
                            }
                        }
                        // A legacy-aliased other slot (same id as cur) must
                        // not take a second update this frame: fall back to
                        // zero motion instead of clobbering cur.
                        if (other.id != 0 and other.id != st.buffer.id) {
                            st.buffers[st.active_slot ^ 1] = other;
                            sg.updateBuffer(other, sg.asRange(prev_items.?));
                            st.prev_buffer = other;
                            // The other slot now holds the previously
                            // published upload: its generation is the frame
                            // this state last published.
                            st.prev_frame = st.staged_frame;
                        } else {
                            // Second-slot creation failure (or alias):
                            // zero motion, never stale pairing.
                            st.prev_buffer = st.buffer;
                            st.prev_frame = gctx.frame_id;
                        }
                    } else {
                        // No prev source (no lookup, generation/uid/count
                        // mismatch, creation failure above): zero motion,
                        // never stale pairing.
                        st.prev_buffer = st.buffer;
                        st.prev_frame = gctx.frame_id;
                    }
                    sg.updateBuffer(st.buffer, sg.asRange(items));
                    // NOTE: st.active_slot and st.buffer are intentionally
                    // UNCHANGED here (no ping-pong flip): within-capacity
                    // updates keep buffer identity per the committed
                    // renderer contract.
                }
                // Учёт динамики: только при реальном изменении (dedup по хешу выше).
                upload_meter.record(active_count * @sizeOf(Mat4));
                st.hash = cpu.hash;
                st.uploaded_count = active_count;
                st.layout_hash = cpu.layout_hash;
            } else {
                st.prev_buffer = st.buffer;
                st.prev_frame = gctx.frame_id;
            }
        }
    }

    // Publish under phase ownership (P1: prepare/render hold phase_mutex on
    // the context thread; no concurrent update-phase TRS writes exist), not
    // real atomics: bounds/count/frame land together, only after the GPU
    // half above succeeded (or was skipped: no context, empty set).
    // hash/uploaded_count intentionally stay stale on the CPU-only and empty
    // paths — they describe the last real GPU upload, which the dedup gate
    // above must keep comparing against.
    st.bounds = cpu.bounds;
    st.count = @intCast(active_count);
    st.staged_frame = gctx.frame_id;
}

/// CPU build over every instance-bearing mesh (stage 1, game side): stages
/// each mesh's segment into the shared `scratch` (concatenated segments —
/// the P7 back-slot `primary.instance_matrices` for the producer build) and records its small plain preview
/// (`bounds`/`count`/`hash`/`build_seq`/`scratch_lo`). Skips LOD children,
/// GPU-pending meshes, and meshes with no instances, exactly like the
/// historical pre-stage. A mesh whose segment OOMs keeps its previous
/// preview (`build_seq` not advanced) — the latch will skip it and the
/// previous complete `instance_render` stands.
pub const CpuStageContext = struct {
    allocator: std.mem.Allocator,
    scratch: *std.ArrayListUnmanaged(Mat4),
    thread_pool: ?*jobs.Pool,
    eye: Vec3,
};

pub fn stageInstancesCpu(ctx: CpuStageContext, meshes: []const *Mesh, build_seq: u64) void {
    for (meshes) |mesh| {
        _ = mesh.ensureUid();
        if (mesh.is_lod_child) continue;
        if (mesh.gpu_pending) continue;
        if (mesh.instances.items.len == 0) continue;
        const lo = ctx.scratch.items.len;
        const cpu = stageSegmentCpu(ctx.allocator, ctx.scratch, lo, ctx.thread_pool, ctx.eye, mesh) catch continue;
        mesh.instance_preview = InstancePreviewState{
            .bounds = cpu.bounds,
            .count = @intCast(ctx.scratch.items.len - lo),
            .hash = cpu.hash,
            .layout_hash = cpu.layout_hash,
            .build_seq = build_seq,
            .scratch_lo = lo,
        };
    }
}

/// Freeze the slot-owned staged records for one build (stage 1, game side):
/// for every instance-bearing mesh with a fresh preview (`build_seq` from
/// the `stageInstancesCpu` pass just above), appends a `StagedInstanceRecord`
/// carrying the mesh pointer/uid/index, the scratch segment, and the prior
/// resolved state from `instance_render`. Same skip set as the CPU pass
/// (LOD children, GPU-pending, no instances, stale preview) plus OOM: a
/// record whose append fails is skipped like an OOM-skipped segment (the
/// latch has no record for it, the previous complete `instance_render`
/// stands, and the patch zeroes its provisional payload entries — coherent,
/// no partial publish). Appends in mesh-list order (strictly increasing
/// `mesh_index`); the caller reset the array first (two builds before a
/// latch: the second reset clears the first, newest wins).
pub fn freezeStagedRecords(
    allocator: std.mem.Allocator,
    records: *std.ArrayListUnmanaged(StagedInstanceRecord),
    meshes: []const *Mesh,
    build_seq: u64,
) void {
    records.ensureTotalCapacity(allocator, records.items.len + meshes.len) catch {};
    for (meshes, 0..) |mesh, i| {
        _ = mesh.ensureUid();
        if (mesh.is_lod_child) continue;
        if (mesh.gpu_pending) continue;
        if (mesh.instances.items.len == 0) continue;
        const pv = mesh.instance_preview;
        if (pv.build_seq != build_seq) continue;
        records.append(allocator, .{
            .mesh = mesh,
            .uid = mesh.uid,
            .mesh_index = @intCast(i),
            .scratch_lo = pv.scratch_lo,
            .count = pv.count,
            .bounds = pv.bounds,
            .hash = pv.hash,
            .layout_hash = pv.layout_hash,
            .uploaded_hash = mesh.instance_render.hash,
            .uploaded_layout = mesh.instance_render.layout_hash,
            .mesh_position = mesh.position,
            .buffer = mesh.instance_render.buffer,
            .prev_buffer = mesh.instance_render.prev_buffer,
            .buffers = mesh.instance_render.buffers,
            .active_slot = mesh.instance_render.active_slot,
            .capacity = mesh.instance_render.capacity,
            .uploaded_count = mesh.instance_render.uploaded_count,
            .prev_frame = mesh.instance_render.prev_frame,
            .staged_frame = mesh.instance_render.staged_frame,
        }) catch continue;
    }
}

/// Zero the mirrored (post-latch) half of a record fail-closed: null buffer,
/// zero capacities/counts, `staged_frame` back to never-staged, so
/// `patchInstanceRefs` zeroes every payload entry resolving to this record.
/// The frozen input half (identity/scratch_lo/bounds/hash/mesh_position) is
/// left intact for debuggability. Latch-created handles are cleared WITHOUT
/// retiring here: on the OOB-slice path the latch never ran (nothing was
/// created); on the GPU-half-failure path the core already destroyed the
/// failed handle and created nothing else.
fn failRecord(rec: *StagedInstanceRecord) void {
    rec.buffer = .{};
    rec.prev_buffer = .{};
    rec.buffers = .{ .{}, .{} };
    rec.active_slot = 0;
    rec.capacity = 0;
    rec.count = 0;
    rec.uploaded_count = 0;
    rec.uploaded_hash = 0;
    rec.uploaded_layout = 0;
    rec.prev_frame = std.math.maxInt(u64);
    rec.latch_created = .{ .{}, .{} };
    rec.staged_frame = std.math.maxInt(u64);
}

/// Handles the latch created while staging one record: post-latch handles
/// minus pre-latch handles (pure, sg-free). Growth contributes the fresh
/// buffer; the steady path contributes a fresh second slot, if any.
/// Dedup hits contribute nothing. At most two handles (fixed array, zero
/// padded) — the commit retires exactly these on abandon-skip paths.
pub fn latchCreatedBuffers(
    pre_buffer: sg.Buffer,
    pre_prev: sg.Buffer,
    pre_slots: [2]sg.Buffer,
    post: *const InstanceRenderState,
) [2]sg.Buffer {
    var created = [_]sg.Buffer{ .{}, .{} };
    var n: usize = 0;
    const pre = [_]sg.Buffer{ pre_buffer, pre_prev, pre_slots[0], pre_slots[1] };
    const post_handles = [_]sg.Buffer{ post.buffer, post.prev_buffer, post.buffers[0], post.buffers[1] };
    for (post_handles) |h| {
        if (h.id == 0) continue;
        var known = false;
        for (pre) |p| if (p.id == h.id) {
            known = true;
            break;
        };
        for (created[0..n]) |c| if (c.id == h.id) {
            known = true;
            break;
        };
        if (!known and n < created.len) {
            created[n] = h;
            n += 1;
        }
    }
    return created;
}

/// GPU latch over the slot-owned staged records (stage 1, context side):
/// one `stageInstancesGpuState` per record, reading its `[scratch_lo,
/// scratch_lo + count)` slice from the slot scratch and seeding the GPU
/// half from the record's PRIOR state (never from live `instance_preview`
/// or live `instance_render`). The final state lands ONLY in the record
/// mirror — the latch touches no live mesh: it never reads the mesh list,
/// never dereferences (or even compares) `record.mesh`, and never writes
/// `mesh.instance_render`. The game-side `commitPublishedRecords` applies
/// the mirrors to the live meshes at the next build; `patchInstanceRefs`
/// finalizes payloads from the mirrors in the same prepare.
///
/// Per record, in order:
/// - Already latched this frame (`staged_frame == frame_id` on entry):
///   nothing left to do — the mirror is final. (Same once-per-frame
///   contract as the old `staged_frame` dedup inside the GPU core.)
/// - Bounds-checked slice (truncated-scratch contract violation): skip with
///   fail-closed zeroing, previous state stands — never an OOB slice.
/// - GPU-half failure (growth `makeBuffer` failure: the core returns WITHOUT
///   publishing, `st.staged_frame` still holds the old frame): fail-close
///   the record mirror — the patch then zeroes every payload entry
///   resolving here, exactly like a skipped record, and the later commit
///   skips it (previous complete `instance_render` stands, retry next
///   build). No retirement happens. The `staged_frame != frame_id` check
///   below is the publication detector: the core advances `staged_frame` if
///   and only if it completely published, so a failed call is
///   indistinguishable from "never ran" and must take the exact same
///   no-publish path.
/// Meshes with no record for this build (OOM-skipped, created after the
/// build) never reach the latch: previous complete state stands.
pub fn stageInstancesLatch(
    gctx: GpuStageContext,
    records: []StagedInstanceRecord,
    scratch: *const std.ArrayListUnmanaged(Mat4),
) void {
    for (records) |*rec| {
        if (rec.staged_frame == gctx.frame_id) continue;
        const count: usize = rec.count;
        const end = rec.scratch_lo + count;
        if (rec.scratch_lo > scratch.items.len or end > scratch.items.len) {
            failRecord(rec);
            continue;
        }
        var st = InstanceRenderState{
            .buffer = rec.buffer,
            .prev_buffer = rec.prev_buffer,
            .buffers = rec.buffers,
            .active_slot = rec.active_slot,
            .capacity = rec.capacity,
            .count = rec.count,
            .bounds = rec.bounds,
            .hash = rec.uploaded_hash,
            .uploaded_count = rec.uploaded_count,
            .layout_hash = rec.uploaded_layout,
            .prev_frame = rec.prev_frame,
            .staged_frame = rec.staged_frame,
        };
        // Velocity pairing source: the previous generation's staged matrices
        // for this uid, resolved from retained slot payloads (never live).
        // Missing/mismatched => the pair branch falls back to zero motion.
        var rctx = gctx;
        rctx.prev_matrices = findPrevMatrices(gctx.prev_frames, rec.staged_frame, rec.uid, count);
        stageInstancesGpuState(rctx, &st, scratch.items[rec.scratch_lo..end], .{
            .bounds = rec.bounds,
            .hash = rec.hash,
            .layout_hash = rec.layout_hash,
        });
        if (st.staged_frame != gctx.frame_id) {
            // The GPU half did not publish (growth buffer failure): fail-close
            // the record mirror — the patch then zeroes every payload entry
            // resolving here, exactly like a skipped record, and the later
            // commit leaves `mesh.instance_render` completely untouched (old
            // buffer/capacity/count/hash/uploaded_count/staged_frame
            // preserved — previous complete state kept, retry next build).
            failRecord(rec);
            continue;
        }
        // Latch-created handles for the commit's abandon paths: post-latch
        // handles minus the pre-latch set the record was seeded with (read
        // BEFORE the mirror assignments below overwrite it). The commit
        // retires exactly these when it cannot adopt the mirror.
        const created = latchCreatedBuffers(rec.buffer, rec.prev_buffer, rec.buffers, &st);
        rec.buffer = st.buffer;
        rec.prev_buffer = st.prev_buffer;
        rec.buffers = st.buffers;
        rec.active_slot = st.active_slot;
        rec.capacity = st.capacity;
        rec.count = st.count;
        rec.bounds = st.bounds;
        // rec.hash stays the frozen staged-matrix hash (dedup compare input,
        // never mirrored); the post-latch uploaded hash lands below.
        rec.uploaded_hash = st.hash;
        rec.uploaded_count = st.uploaded_count;
        // Same split for the layout gate: staged input stays frozen, the
        // published layout lands here for the next build's pairing seed.
        rec.uploaded_layout = st.layout_hash;
        rec.prev_frame = st.prev_frame;
        rec.staged_frame = st.staged_frame;
        rec.latch_created = created;
    }
}

/// Game-side commit of published latch outcomes (stage 1, game side):
/// applies one committed slot's record mirrors to the live meshes — the
/// historical guarded `instance_render` write-back, moved game-side and
/// ordered after publish, never concurrent with the context. `commit_frame`
/// is the publishing slot's `frame_id` (the latch stamps its mirrors with
/// exactly that frame on success): records from any other generation —
/// stale builds, fail-closed mirrors (`staged_frame == maxInt`, which never
/// equals a real frame) — never commit.
///
/// Per record, in order:
/// - Generation check (`staged_frame == commit_frame`), else skip: the mesh
///   keeps its previous complete state.
/// - O(1) aliveness/identity guard: `mesh_index` in range,
///   `meshes[mesh_index] == record.mesh`, `uid` match. Evaluated without
///   dereferencing `record.mesh` first, so a destroyed (unlinked, possibly
///   freed) mesh skips on the pointer compare — no UAF. On mismatch the
///   mesh keeps its previous complete state.
/// - Today's skip conditions (LOD child, GPU-pending, emptied between build
///   and commit): skip, previous state stands. (Live re-checks, legal here —
///   this runs game-side under update-vs-prepare exclusion.)
/// - Already applied (`instance_render.staged_frame == commit_frame`):
///   skip — the values are already this exact publish (double build without
///   an intervening latch), so the write would be redundant.
/// - Abandon (`retire` handling): the two skip branches above (identity
///   guard, live re-checks) retire the record's latch-created handles
///   through `retire` (exactly once — the set is cleared), because no live
///   mesh can adopt them anymore. Fail-closed/stale/already-applied paths
///   retire nothing: the former never created anything, the latter's values
///   are already mesh-owned. A null `retire` keeps the historical
///   standalone behavior (caller obligation, same as the null retire_queue
///   in the GPU half).
/// Otherwise the mirror lands in `mesh.instance_render` verbatim — the
/// identical bytes the old latch wrote back synchronously, so sequential
/// build → latch → build usage commits exactly the same states (same
/// stats, same fail paths).
/// Retire sink for latch-created buffers the commit cannot adopt (game
/// side, sg-free: `retireBuffer` only enqueues under a spinlock, never
/// destroys — destruction stays context-owned in flush/deinit). Null marks
/// a standalone caller (tests, one-off tooling) under the same obligation
/// as the null `retire_queue` in the GPU half: no live snapshot may
/// reference the abandoned handles, and the leak is the caller's.
pub const CommitRetire = struct {
    allocator: std.mem.Allocator,
    queue: *gpu_retire.GpuRetireQueue,
};

/// Retires a record's latch-created handles exactly once (clears the set):
/// called only on abandon-skip paths, where the mesh keeps its previous
/// complete state and the mirror's fresh handles would otherwise strand.
/// Borrowed prior handles are never touched here — their lifetime stays
/// with the (possibly dead) owner's deinit/retireMesh path.
fn retireAbandonedLatchCreated(rec: *StagedInstanceRecord, retire: ?CommitRetire) void {
    const created = rec.latch_created;
    rec.latch_created = .{ .{}, .{} };
    const ctx = retire orelse return;
    for (created) |h| {
        if (h.id != 0) ctx.queue.retireBuffer(ctx.allocator, h);
    }
}

pub fn commitPublishedRecords(
    records: []StagedInstanceRecord,
    meshes: []const *Mesh,
    commit_frame: u64,
    retire: ?CommitRetire,
) void {
    for (records) |*rec| {
        if (rec.staged_frame == std.math.maxInt(u64)) continue;
        if (rec.staged_frame != commit_frame) continue;
        const idx: usize = rec.mesh_index;
        if (idx >= meshes.len or meshes[idx] != rec.mesh or meshes[idx].uid != rec.uid) {
            // Aliveness/identity guard failed (destroyed, unlinked, address
            // reuse): keep previous state AND retire the latch-created
            // handles, which no live mesh can now adopt.
            retireAbandonedLatchCreated(rec, retire);
            continue;
        }
        const mesh = rec.mesh;
        if (mesh.is_lod_child or mesh.gpu_pending or mesh.instances.items.len == 0) {
            // Live re-checks failed after a successful latch: same abandon
            // rule — the mesh keeps its previous complete state.
            retireAbandonedLatchCreated(rec, retire);
            continue;
        }
        if (mesh.instance_render.staged_frame == rec.staged_frame) continue;
        mesh.instance_render = .{
            .buffer = rec.buffer,
            .prev_buffer = rec.prev_buffer,
            .buffers = rec.buffers,
            .active_slot = rec.active_slot,
            .capacity = rec.capacity,
            .count = rec.count,
            .bounds = rec.bounds,
            .hash = rec.uploaded_hash,
            .uploaded_count = rec.uploaded_count,
            .layout_hash = rec.uploaded_layout,
            .prev_frame = rec.prev_frame,
            .staged_frame = rec.staged_frame,
        };
    }
}

/// Pre-stage every instance-bearing mesh once per frame. Skips LOD children,
/// GPU-pending meshes, and meshes with no instances; staging itself is
/// guarded per mesh by `staged_frame`, so calling this before the
/// view queues and again implicitly via submitInstancedMesh (render_queue.zig)
/// stays once-only.
pub fn stageInstances(sc: InstanceStageContext, meshes: []const *Mesh) void {
    for (meshes) |mesh| {
        _ = mesh.ensureUid();
        if (mesh.is_lod_child) continue;
        if (mesh.gpu_pending) continue;
        if (mesh.instances.items.len == 0) continue;
        stageInstancedMesh(sc, mesh);
    }
}

// ---- P5 regression: parallel chunker edge cases. ----

// Direct runChunks coverage at len 257 / 64 chunks (span 5, the production
// ceil-division formula): a fully hidden chunk contributes zero visibles
// with an invalid AABB, and ceil-division tail chunks past the instance
// list are empty without slicing out of bounds. No worker threads needed —
// the chunk function itself is exercised over the full chunk range.
test "P5: runChunks handles all-hidden chunks and empty tail chunks" {
    const ally = std.testing.allocator;
    var src = Mesh{
        .name = "chunk_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 257;
    const chunk_count: usize = 64;
    const span: usize = (n + chunk_count - 1) / chunk_count; // 5
    try std.testing.expectEqual(@as(usize, 5), span);

    const mem = try ally.alloc(InstancedMesh, n);
    defer ally.free(mem);
    const ptrs = try ally.alloc(*InstancedMesh, n);
    defer ally.free(ptrs);
    for (0..n) |i| {
        // Chunk 3 (indices 15..20) is fully hidden; everything else visible.
        const hidden = i >= 3 * span and i < 4 * span;
        mem[i] = InstancedMesh{
            .name = "c",
            .source_mesh = &src,
            .position = Vec3.new(@floatFromInt(i), 0, 0),
            .is_visible = !hidden,
        };
        ptrs[i] = &mem[i];
    }

    const out = try ally.alloc(Mat4, n);
    defer ally.free(out);
    const out_uids = try ally.alloc(u64, n);
    defer ally.free(out_uids);
    @memset(out_uids, 0);
    // Production contract: uids are assigned serially before any parallel
    // read (stageSegmentCpu does this); the chunker only trails them.
    for (ptrs) |inst| _ = inst.ensureUid();
    var aabbs_buf: [64]BoundingBox = undefined;
    var counts_buf: [64]usize = undefined;
    var stage = ParallelInstanceStage{
        .instances = ptrs,
        .span = span,
        .chunk_aabbs = aabbs_buf[0..chunk_count],
        .chunk_visible_counts = counts_buf[0..chunk_count],
        .out_matrices = out,
        .out_uids = out_uids,
    };
    ParallelInstanceStage.runChunks(&stage, 0, chunk_count);

    // Chunk 0: 5 visibles packed at the segment start, matrices exact, and
    // the uid trail carries the chunk's instance identities in the same order.
    try std.testing.expectEqual(@as(usize, 5), counts_buf[0]);
    for (0..5) |k| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(k)), out[k].m[12], 1e-6);
        try std.testing.expectEqual(ptrs[k].uid, out_uids[k]);
    }
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), aabbs_buf[0].min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), aabbs_buf[0].max.x, 1e-4);
    // Chunk 3: fully hidden → zero, invalid AABB (merges must skip it).
    try std.testing.expectEqual(@as(usize, 0), counts_buf[3]);
    try std.testing.expect(!aabbs_buf[3].isValid());
    // Chunk 51: partial tail of the list (indices 255..257) → 2 visibles.
    try std.testing.expectEqual(@as(usize, 2), counts_buf[51]);
    // Chunks 52..64: lo past the list → empty, no out-of-bounds slice.
    for (counts_buf[52..chunk_count]) |c| try std.testing.expectEqual(@as(usize, 0), c);
    for (aabbs_buf[52..chunk_count]) |b| try std.testing.expect(!b.isValid());
    // Total visibles: 257 − 5 hidden.
    var total: usize = 0;
    for (counts_buf[0..chunk_count]) |c| total += c;
    try std.testing.expectEqual(@as(usize, 252), total);
}

// ---- Stage 1 split: CPU/GPU halves, concatenated scratch, latch. ----

const StageSplitFixture = struct {
    mem: []InstancedMesh,
    ptrs: []*InstancedMesh,
    mesh: Mesh,

    /// `src` must outlive the fixture (test scope, like the render_queue
    /// fixtures): instances hold a raw pointer to it, so it can never be a
    /// field of this returned struct (use-after-return).
    fn init(ally: std.mem.Allocator, src: *Mesh, n: usize, hide_every: usize) !StageSplitFixture {
        var f = StageSplitFixture{
            .mem = try ally.alloc(InstancedMesh, n),
            .ptrs = try ally.alloc(*InstancedMesh, n),
            .mesh = Mesh{
                .name = "split_parent",
                .vertex_buffer = .{},
                .index_buffer = .{},
                .index_count = 3,
            },
        };
        errdefer ally.free(f.mem);
        errdefer ally.free(f.ptrs);
        for (0..n) |i| {
            const fi: f32 = @floatFromInt(i);
            f.mem[i] = InstancedMesh{
                .name = "s",
                .source_mesh = src,
                .position = Vec3.new(fi, fi * 0.5, 0),
                .is_visible = hide_every == 0 or i % hide_every != 0,
            };
            f.ptrs[i] = &f.mem[i];
        }
        f.mesh.instances = .{ .items = f.ptrs, .capacity = n };
        return f;
    }

    fn deinit(f: *StageSplitFixture, ally: std.mem.Allocator) void {
        ally.free(f.mem);
        ally.free(f.ptrs);
    }
};

fn splitSrcMesh() Mesh {
    return .{
        .name = "split_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
}

test "stage1: CPU+GPU halves equal the inline path (serial and parallel)" {
    const ally = std.testing.allocator;
    const eye = Vec3.new(0, 0, 100);

    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();

    // Two variants of the same 300-instance mesh: inline vs split+parallel.
    var src_a = splitSrcMesh();
    var fa = try StageSplitFixture.init(ally, &src_a, 300, 7);
    defer fa.deinit(ally);
    var src_b = splitSrcMesh();
    var fb = try StageSplitFixture.init(ally, &src_b, 300, 7);
    defer fb.deinit(ally);

    var scratch_inline: std.ArrayListUnmanaged(Mat4) = .empty;
    defer scratch_inline.deinit(ally);
    stageInstancedMesh(.{
        .allocator = ally,
        .instance_matrices = &scratch_inline,
        .thread_pool = null,
        .frame_id = 11,
        .eye = eye,
    }, &fa.mesh);

    var scratch_split: std.ArrayListUnmanaged(Mat4) = .empty;
    defer scratch_split.deinit(ally);
    const cpu = try stageSegmentCpu(ally, &scratch_split, 0, pool, eye, &fb.mesh);
    stageInstancesGpu(.{ .allocator = ally, .frame_id = 11 }, &fb.mesh, scratch_split.items, cpu);

    try std.testing.expectEqual(fa.mesh.instance_render.count, fb.mesh.instance_render.count);
    try std.testing.expectEqual(@as(u32, 258), fb.mesh.instance_render.count);
    try std.testing.expectEqual(fa.mesh.instance_render.bounds, fb.mesh.instance_render.bounds);
    try std.testing.expectEqual(@as(u64, 11), fb.mesh.instance_render.staged_frame);
    try std.testing.expectEqual(scratch_inline.items.len, scratch_split.items.len);
    for (scratch_inline.items, scratch_split.items) |m_a, m_b| {
        try std.testing.expectEqual(m_a, m_b);
    }
}

test "stage1: concatenated scratch + latch consume slot records, skip missing" {
    const ally = std.testing.allocator;
    const eye = Vec3.zero;

    var src_a = splitSrcMesh();
    var fa = try StageSplitFixture.init(ally, &src_a, 4, 0);
    defer fa.deinit(ally);
    var src_b = splitSrcMesh();
    var fb = try StageSplitFixture.init(ally, &src_b, 6, 3);
    defer fb.deinit(ally);
    const meshes = [_]*Mesh{ &fa.mesh, &fb.mesh };

    var scratch: std.ArrayListUnmanaged(Mat4) = .empty;
    defer scratch.deinit(ally);
    stageInstancesCpu(.{ .allocator = ally, .scratch = &scratch, .thread_pool = null, .eye = eye }, &meshes, 7);

    // Concatenated segments: [0,4) then [4,8); previews carry offsets.
    // (fb hides every 3rd of 6 → 4 visibles, not 6.)
    try std.testing.expectEqual(@as(usize, 10), scratch.items.len);
    try std.testing.expectEqual(@as(u32, 5), fa.mesh.instance_preview.count);
    try std.testing.expectEqual(@as(usize, 0), fa.mesh.instance_preview.scratch_lo);
    try std.testing.expectEqual(@as(u64, 7), fa.mesh.instance_preview.build_seq);
    try std.testing.expectEqual(@as(u32, 5), fb.mesh.instance_preview.count);
    try std.testing.expectEqual(@as(usize, 5), fb.mesh.instance_preview.scratch_lo);
    try std.testing.expectEqual(@as(u64, 7), fb.mesh.instance_preview.build_seq);

    // Records freeze one per fresh preview, in mesh order.
    var records: std.ArrayListUnmanaged(StagedInstanceRecord) = .empty;
    defer records.deinit(ally);
    freezeStagedRecords(ally, &records, &meshes, 7);
    try std.testing.expectEqual(@as(usize, 2), records.items.len);
    try std.testing.expectEqual(@as(u32, 0), records.items[0].mesh_index);
    try std.testing.expectEqual(@as(u32, 1), records.items[1].mesh_index);
    try std.testing.expectEqual(fa.mesh.uid, records.items[0].uid);
    try std.testing.expectEqual(@as(usize, 5), records.items[1].scratch_lo);

    stageInstancesLatch(.{ .allocator = ally, .frame_id = 21 }, records.items, &scratch);
    // The latch mirrors into the records only: live meshes stay exactly as
    // built (never-staged) — the write-back moved to the game-side commit.
    try std.testing.expectEqual(std.math.maxInt(u64), fa.mesh.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u32, 0), fa.mesh.instance_render.count);
    try std.testing.expectEqual(@as(u64, 21), records.items[0].staged_frame);
    try std.testing.expectEqual(@as(u64, 21), records.items[1].staged_frame);
    try std.testing.expectEqual(@as(u32, 5), records.items[1].count);

    // Commit (next game-side build) applies the published mirrors to live.
    commitPublishedRecords(records.items, &meshes, 21, null);
    try std.testing.expectEqual(@as(u32, 5), fa.mesh.instance_render.count);
    try std.testing.expectEqual(@as(u32, 5), fb.mesh.instance_render.count);
    try std.testing.expectEqual(@as(u64, 21), fa.mesh.instance_render.staged_frame);
    try std.testing.expectEqual(fa.mesh.instance_preview.bounds, fa.mesh.instance_render.bounds);
    try std.testing.expectEqual(fb.mesh.instance_preview.bounds, fb.mesh.instance_render.bounds);

    // No record for this build (OOM-skipped / post-build mesh): the commit
    // has nothing to apply — previous complete state stands, no partial
    // publish.
    const keep_bounds = fb.mesh.instance_render.bounds;
    _ = records.pop(); // drop B's record: same as a segment that never froze
    stageInstancesLatch(.{ .allocator = ally, .frame_id = 22 }, records.items, &scratch);
    commitPublishedRecords(records.items, &meshes, 22, null);
    try std.testing.expectEqual(@as(u64, 22), fa.mesh.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u64, 21), fb.mesh.instance_render.staged_frame);
    try std.testing.expectEqual(keep_bounds, fb.mesh.instance_render.bounds);
    try std.testing.expectEqual(@as(u32, 5), fb.mesh.instance_render.count);
}

test "stage1: latch ignores cleared live previews, consumes records only" {
    const ally = std.testing.allocator;
    const eye = Vec3.zero;

    var src_a = splitSrcMesh();
    var fa = try StageSplitFixture.init(ally, &src_a, 3, 0);
    defer fa.deinit(ally);
    var src_b = splitSrcMesh();
    var fb = try StageSplitFixture.init(ally, &src_b, 2, 0);
    defer fb.deinit(ally);
    const meshes = [_]*Mesh{ &fa.mesh, &fb.mesh };

    var scratch: std.ArrayListUnmanaged(Mat4) = .empty;
    defer scratch.deinit(ally);
    stageInstancesCpu(.{ .allocator = ally, .scratch = &scratch, .thread_pool = null, .eye = eye }, &meshes, 9);
    var records: std.ArrayListUnmanaged(StagedInstanceRecord) = .empty;
    defer records.deinit(ally);
    freezeStagedRecords(ally, &records, &meshes, 9);
    try std.testing.expectEqual(@as(usize, 2), records.items.len);

    // Live previews wiped between build and latch (the old latch would skip
    // both meshes): records still carry everything the latch needs, and the
    // latch publishes the mirrors without touching live meshes at all.
    fa.mesh.instance_preview = .{};
    fb.mesh.instance_preview = .{};

    stageInstancesLatch(.{ .allocator = ally, .frame_id = 33 }, records.items, &scratch);
    try std.testing.expectEqual(@as(u64, 33), records.items[0].staged_frame);
    try std.testing.expectEqual(@as(u64, 33), records.items[1].staged_frame);
    try std.testing.expectEqual(std.math.maxInt(u64), fa.mesh.instance_render.staged_frame);
    try std.testing.expectEqual(std.math.maxInt(u64), fb.mesh.instance_render.staged_frame);

    commitPublishedRecords(records.items, &meshes, 33, null);
    try std.testing.expectEqual(@as(u32, 4), fa.mesh.instance_render.count);
    try std.testing.expectEqual(@as(u32, 3), fb.mesh.instance_render.count);
    try std.testing.expectEqual(@as(u64, 33), fa.mesh.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u64, 33), fb.mesh.instance_render.staged_frame);
    try std.testing.expect(fa.mesh.instance_render.bounds.isValid());
    // A live instance TRS mutation after the build still cannot leak in: the
    // slice was frozen at build time.
    try std.testing.expectEqual(records.items[0].bounds, fa.mesh.instance_render.bounds);
}

test "stage1: latch stages from slot data alone; reorder is caught at commit, never dereferenced" {
    const ally = std.testing.allocator;
    const eye = Vec3.zero;

    var src_a = splitSrcMesh();
    var fa = try StageSplitFixture.init(ally, &src_a, 2, 0);
    defer fa.deinit(ally);
    var src_b = splitSrcMesh();
    var fb = try StageSplitFixture.init(ally, &src_b, 2, 0);
    defer fb.deinit(ally);
    const built = [_]*Mesh{ &fa.mesh, &fb.mesh };

    var scratch: std.ArrayListUnmanaged(Mat4) = .empty;
    defer scratch.deinit(ally);
    stageInstancesCpu(.{ .allocator = ally, .scratch = &scratch, .thread_pool = null, .eye = eye }, &built, 5);
    var records: std.ArrayListUnmanaged(StagedInstanceRecord) = .empty;
    defer records.deinit(ally);
    freezeStagedRecords(ally, &records, &built, 5);
    try std.testing.expectEqual(@as(usize, 2), records.items.len);

    // Reorder: B now sits at index 0 (A's record points at B), A is gone.
    // Destroy: index 1 is out of range for B's record. The latch takes no
    // mesh list at all, so it cannot consult it — both records publish from
    // slot data alone, live meshes untouched. (This is the proof no live-list
    // read remains: a guard here would have fail-closed both records.)
    const reordered = [_]*Mesh{&fb.mesh};
    stageInstancesLatch(.{ .allocator = ally, .frame_id = 44 }, records.items, &scratch);
    try std.testing.expectEqual(@as(u64, 44), records.items[0].staged_frame);
    try std.testing.expectEqual(@as(u64, 44), records.items[1].staged_frame);
    try std.testing.expectEqual(std.math.maxInt(u64), fa.mesh.instance_render.staged_frame);
    try std.testing.expectEqual(std.math.maxInt(u64), fb.mesh.instance_render.staged_frame);

    // The game-side commit runs the O(1) identity guard instead: A's record
    // (index 0, mesh A) meets B → pointer mismatch → skip; B's record
    // (index 1) is out of range → skip. Neither mesh publishes, previous
    // (never-staged) state stands, and the dangling record mesh is never
    // dereferenced (pointer compare only).
    commitPublishedRecords(records.items, &reordered, 44, null);
    try std.testing.expectEqual(std.math.maxInt(u64), fa.mesh.instance_render.staged_frame);
    try std.testing.expectEqual(std.math.maxInt(u64), fb.mesh.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u32, 0), fa.mesh.instance_render.count);
    try std.testing.expectEqual(@as(u32, 0), fb.mesh.instance_render.count);

    // The records themselves stay published (the patch already resolved the
    // payload from them); restoring the list lets a later commit apply them.
    commitPublishedRecords(records.items, &built, 44, null);
    try std.testing.expectEqual(@as(u64, 44), fa.mesh.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u64, 44), fb.mesh.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u32, 3), fa.mesh.instance_render.count);
    try std.testing.expectEqual(@as(u32, 3), fb.mesh.instance_render.count);
}

test "stage1: latch never touches record.mesh — dangling pointer proof" {
    const ally = std.testing.allocator;

    var src = splitSrcMesh();
    var f = try StageSplitFixture.init(ally, &src, 2, 0);
    defer f.deinit(ally);
    const meshes = [_]*Mesh{&f.mesh};

    var scratch: std.ArrayListUnmanaged(Mat4) = .empty;
    defer scratch.deinit(ally);
    stageInstancesCpu(.{ .allocator = ally, .scratch = &scratch, .thread_pool = null, .eye = Vec3.zero }, &meshes, 8);
    var records: std.ArrayListUnmanaged(StagedInstanceRecord) = .empty;
    defer records.deinit(ally);
    freezeStagedRecords(ally, &records, &meshes, 8);
    try std.testing.expectEqual(@as(usize, 1), records.items.len);

    // Simulate a mesh freed between build and latch: the pointer dangles.
    // Any dereference OR comparison of it on the latch path would fault or
    // fail-close; the latch does neither and publishes from slot data alone.
    records.items[0].mesh = @as(*Mesh, @ptrFromInt(0xdeadbee0));
    records.items[0].uid = 0xbadc0de;
    stageInstancesLatch(.{ .allocator = ally, .frame_id = 45 }, records.items, &scratch);
    try std.testing.expectEqual(@as(u64, 45), records.items[0].staged_frame);
    try std.testing.expectEqual(@as(u32, 3), records.items[0].count);

    // The commit only compares the pointer (never dereferences it): the
    // dangling record skips, the live mesh keeps its previous state.
    commitPublishedRecords(records.items, &meshes, 45, null);
    try std.testing.expectEqual(std.math.maxInt(u64), f.mesh.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u32, 0), f.mesh.instance_render.count);
}

test "stage1: truncated scratch skips the record, previous stands" {
    const ally = std.testing.allocator;
    const eye = Vec3.zero;

    var src = splitSrcMesh();
    var f = try StageSplitFixture.init(ally, &src, 3, 0);
    defer f.deinit(ally);
    const meshes = [_]*Mesh{&f.mesh};

    var scratch: std.ArrayListUnmanaged(Mat4) = .empty;
    defer scratch.deinit(ally);
    stageInstancesCpu(.{ .allocator = ally, .scratch = &scratch, .thread_pool = null, .eye = eye }, &meshes, 3);
    var records: std.ArrayListUnmanaged(StagedInstanceRecord) = .empty;
    defer records.deinit(ally);
    freezeStagedRecords(ally, &records, &meshes, 3);

    // Prime a previous complete publish (latch mirrors, commit applies),
    // then truncate the scratch (contract violation the latch must
    // survive): the out-of-range slice is skipped with fail-closed zeroing,
    // never an OOB slice; the later commit skips the fail-closed record so
    // the previous complete live state stands.
    stageInstancesLatch(.{ .allocator = ally, .frame_id = 50 }, records.items, &scratch);
    commitPublishedRecords(records.items, &meshes, 50, null);
    try std.testing.expectEqual(@as(u32, 4), f.mesh.instance_render.count);
    const primed = f.mesh.instance_render.bounds;
    scratch.clearRetainingCapacity();
    stageInstancesLatch(.{ .allocator = ally, .frame_id = 51 }, records.items, &scratch);
    commitPublishedRecords(records.items, &meshes, 51, null);
    try std.testing.expectEqual(@as(u32, 4), f.mesh.instance_render.count);
    try std.testing.expectEqual(primed, f.mesh.instance_render.bounds);
    try std.testing.expectEqual(@as(u64, 50), f.mesh.instance_render.staged_frame);
    try std.testing.expectEqual(std.math.maxInt(u64), records.items[0].staged_frame);
}

test "stage1: commit skips stale generations, fail-closed mirrors, and skips re-apply" {
    const ally = std.testing.allocator;

    var src = splitSrcMesh();
    var f = try StageSplitFixture.init(ally, &src, 2, 0);
    defer f.deinit(ally);
    const meshes = [_]*Mesh{&f.mesh};

    var scratch: std.ArrayListUnmanaged(Mat4) = .empty;
    defer scratch.deinit(ally);
    stageInstancesCpu(.{ .allocator = ally, .scratch = &scratch, .thread_pool = null, .eye = Vec3.zero }, &meshes, 4);
    var records: std.ArrayListUnmanaged(StagedInstanceRecord) = .empty;
    defer records.deinit(ally);
    freezeStagedRecords(ally, &records, &meshes, 4);
    stageInstancesLatch(.{ .allocator = ally, .frame_id = 60 }, records.items, &scratch);

    // Stale generation (a newer slot already published): never commits.
    commitPublishedRecords(records.items, &meshes, 61, null);
    try std.testing.expectEqual(std.math.maxInt(u64), f.mesh.instance_render.staged_frame);

    // Fail-closed mirror (GPU-half failure zeroed it): never commits, even
    // against its own generation.
    records.items[0].staged_frame = std.math.maxInt(u64);
    commitPublishedRecords(records.items, &meshes, 60, null);
    try std.testing.expectEqual(std.math.maxInt(u64), f.mesh.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u32, 0), f.mesh.instance_render.count);

    // The published generation commits exactly once: a repeat commit is a
    // no-op (already-applied skip), so double builds never regress.
    records.items[0].staged_frame = 60;
    records.items[0].count = 2;
    commitPublishedRecords(records.items, &meshes, 60, null);
    try std.testing.expectEqual(@as(u64, 60), f.mesh.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u32, 2), f.mesh.instance_render.count);
    f.mesh.instance_preview.count = 9; // live drift must not resurrect
    commitPublishedRecords(records.items, &meshes, 60, null);
    try std.testing.expectEqual(@as(u64, 60), f.mesh.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u32, 2), f.mesh.instance_render.count);

    // Live skip re-checks (the old latch-time re-reads, now game-side):
    // LOD children, GPU-pending meshes, and emptied instance lists keep
    // their previous complete state.
    f.mesh.is_lod_child = true;
    records.items[0].staged_frame = 61;
    commitPublishedRecords(records.items, &meshes, 61, null);
    try std.testing.expectEqual(@as(u64, 60), f.mesh.instance_render.staged_frame);
    f.mesh.is_lod_child = false;
    f.mesh.gpu_pending = true;
    commitPublishedRecords(records.items, &meshes, 61, null);
    try std.testing.expectEqual(@as(u64, 60), f.mesh.instance_render.staged_frame);
    f.mesh.gpu_pending = false;
    const saved = f.mesh.instances.items;
    f.mesh.instances.items = &.{};
    commitPublishedRecords(records.items, &meshes, 61, null);
    try std.testing.expectEqual(@as(u64, 60), f.mesh.instance_render.staged_frame);
    f.mesh.instances.items = saved;
    commitPublishedRecords(records.items, &meshes, 61, null);
    try std.testing.expectEqual(@as(u64, 61), f.mesh.instance_render.staged_frame);
}

test "stage1: failRecord zeroes the not-published mirror, keeps frozen input" {
    var mesh = Mesh{
        .name = "failrec",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
    };
    const frozen_bounds = BoundingBox.init(Vec3.new(1, 2, 3), Vec3.new(4, 5, 6));
    var rec = StagedInstanceRecord{
        .mesh = &mesh,
        .uid = 42,
        .mesh_index = 3,
        .scratch_lo = 5,
        .count = 7,
        .bounds = frozen_bounds,
        .hash = 123,
        .layout_hash = 124,
        .uploaded_hash = 456,
        .uploaded_layout = 457,
        .mesh_position = Vec3.new(9, 0, 0),
        .buffer = .{ .id = 9 },
        .capacity = 11,
        .uploaded_count = 13,
        .staged_frame = 77,
    };
    failRecord(&rec);
    // Not-published mirror: null buffer, zero counts, never-staged frame —
    // `staged_frame != frame_id` is what makes `patchInstanceRefs` zero
    // every payload entry resolving here (and what tells any residual
    // reader nothing new was published).
    try std.testing.expectEqual(@as(u32, 0), rec.buffer.id);
    try std.testing.expectEqual(@as(usize, 0), rec.capacity);
    try std.testing.expectEqual(@as(u32, 0), rec.count);
    try std.testing.expectEqual(@as(usize, 0), rec.uploaded_count);
    try std.testing.expectEqual(@as(u64, 0), rec.uploaded_hash);
    try std.testing.expectEqual(std.math.maxInt(u64), rec.staged_frame);
    // Frozen input half intact for debuggability.
    try std.testing.expect(rec.mesh == &mesh);
    try std.testing.expectEqual(@as(u64, 42), rec.uid);
    try std.testing.expectEqual(@as(u32, 3), rec.mesh_index);
    try std.testing.expectEqual(@as(usize, 5), rec.scratch_lo);
    try std.testing.expectEqual(@as(u64, 123), rec.hash);
    try std.testing.expectEqual(frozen_bounds, rec.bounds);
    try std.testing.expectEqual(Vec3.new(9, 0, 0), rec.mesh_position);
}

test "stage1: two record freezes, newest wins" {
    const ally = std.testing.allocator;
    const eye = Vec3.zero;

    var src = splitSrcMesh();
    var f = try StageSplitFixture.init(ally, &src, 2, 0);
    defer f.deinit(ally);
    const meshes = [_]*Mesh{&f.mesh};

    var scratch: std.ArrayListUnmanaged(Mat4) = .empty;
    defer scratch.deinit(ally);
    var records: std.ArrayListUnmanaged(StagedInstanceRecord) = .empty;
    defer records.deinit(ally);

    // First build.
    stageInstancesCpu(.{ .allocator = ally, .scratch = &scratch, .thread_pool = null, .eye = eye }, &meshes, 1);
    freezeStagedRecords(ally, &records, &meshes, 1);
    try std.testing.expectEqual(@as(usize, 1), records.items.len);
    const first_bounds = records.items[0].bounds;

    // Second build before any latch (slot reset first: newest wins).
    f.mem[1].position = Vec3.new(40, 0, 0);
    scratch.clearRetainingCapacity();
    records.clearRetainingCapacity();
    stageInstancesCpu(.{ .allocator = ally, .scratch = &scratch, .thread_pool = null, .eye = eye }, &meshes, 2);
    freezeStagedRecords(ally, &records, &meshes, 2);
    try std.testing.expectEqual(@as(usize, 1), records.items.len);
    try std.testing.expect(records.items[0].bounds.max.x > first_bounds.max.x + 10.0);

    stageInstancesLatch(.{ .allocator = ally, .frame_id = 60 }, records.items, &scratch);
    commitPublishedRecords(records.items, &meshes, 60, null);
    try std.testing.expectEqual(records.items[0].bounds, f.mesh.instance_render.bounds);
    try std.testing.expect(records.items[0].bounds.max.x > first_bounds.max.x + 10.0);
}

test "stage1: segment OOM truncates scratch and advances nothing" {
    const ally = std.testing.allocator;

    var src = splitSrcMesh();
    var f = try StageSplitFixture.init(ally, &src, 4, 0);
    defer f.deinit(ally);

    var scratch: std.ArrayListUnmanaged(Mat4) = .empty;
    defer scratch.deinit(ally);
    // Prime retained capacity, then refuse fresh allocs: the segment must
    // fail without touching the scratch or the preview.
    const warm = try stageSegmentCpu(ally, &scratch, 0, null, Vec3.zero, &f.mesh);
    _ = warm;
    scratch.clearRetainingCapacity();
    f.mesh.instance_preview = .{ .bounds = BoundingBox.init(Vec3.new(1, 2, 3), Vec3.new(4, 5, 6)), .count = 9, .hash = 1234, .build_seq = 5, .scratch_lo = 0 };

    var failing = std.testing.FailingAllocator.init(ally, .{ .fail_index = 0 });
    // Force real allocation pressure: drop retained capacity first.
    scratch.clearAndFree(ally);
    const res = stageSegmentCpu(failing.allocator(), &scratch, 0, null, Vec3.zero, &f.mesh);
    try std.testing.expectError(error.OutOfMemory, res);
    try std.testing.expectEqual(@as(usize, 0), scratch.items.len);

    // The build wrapper maps the error to skip: preview untouched.
    const meshes = [_]*Mesh{&f.mesh};
    stageInstancesCpu(.{ .allocator = failing.allocator(), .scratch = &scratch, .thread_pool = null, .eye = Vec3.zero }, &meshes, 6);
    try std.testing.expectEqual(@as(u64, 5), f.mesh.instance_preview.build_seq);
    try std.testing.expectEqual(@as(u32, 9), f.mesh.instance_preview.count);
    try std.testing.expectEqual(@as(usize, 0), scratch.items.len);

    // Recovery: a funded build recomputes the preview.
    stageInstancesCpu(.{ .allocator = ally, .scratch = &scratch, .thread_pool = null, .eye = Vec3.zero }, &meshes, 6);
    try std.testing.expectEqual(@as(u64, 6), f.mesh.instance_preview.build_seq);
    // 4 visible instances + the source mesh entry.
    try std.testing.expectEqual(@as(u32, 5), f.mesh.instance_preview.count);
    try std.testing.expectEqual(@as(usize, 5), scratch.items.len);
}

test "stage1: transparent sort is farthest-first under each path's own eye" {
    const ally = std.testing.allocator;

    // Documented eye divergence: the game-side build sorts with the live
    // (post-mutation) eye while the immediate helper sorts with the snapshot
    // eye. Both orders are correct back-to-front for their own eye — they
    // are NOT forced equal. Two instances symmetric about the origin make
    // the divergence exact: opposite eyes give reverse orders.
    var blend_mat = material_mod.PBRMaterial.init("s1_blend");
    blend_mat.alpha_mode = .blend;

    var src = splitSrcMesh();
    var mem: [2]InstancedMesh = .{
        .{ .name = "t0", .source_mesh = &src, .position = Vec3.new(-10, 0, 0) },
        .{ .name = "t1", .source_mesh = &src, .position = Vec3.new(10, 0, 0) },
    };
    var ptrs = [_]*InstancedMesh{ &mem[0], &mem[1] };
    var mesh = Mesh{
        .name = "s1_transparent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .material = .{ .pbr = &blend_mat },
        .instances = .{ .items = &ptrs, .capacity = 2 },
    };
    const meshes = [_]*Mesh{&mesh};

    // Build path with the live eye on the left: farthest (+10) first.
    const eye_live = Vec3.new(-100, 0, 0);
    var scratch_build: std.ArrayListUnmanaged(Mat4) = .empty;
    defer scratch_build.deinit(ally);
    stageInstancesCpu(.{ .allocator = ally, .scratch = &scratch_build, .thread_pool = null, .eye = eye_live }, &meshes, 1);
    try std.testing.expectEqual(@as(usize, 3), scratch_build.items.len);
    // (+10) farthest, then the source mesh entry (x=0), then (-10).
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), scratch_build.items[0].m[12], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), scratch_build.items[1].m[12], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -10.0), scratch_build.items[2].m[12], 1e-6);

    // Immediate helper path with the snapshot eye on the right: farthest
    // (-10) first — the reverse order. Correct for its own eye, divergent
    // from the build by design (counts/bounds never diverge, only order).
    const eye_snap = Vec3.new(100, 0, 0);
    var scratch_inline: std.ArrayListUnmanaged(Mat4) = .empty;
    defer scratch_inline.deinit(ally);
    stageInstancedMesh(.{
        .allocator = ally,
        .instance_matrices = &scratch_inline,
        .thread_pool = null,
        .frame_id = 31,
        .eye = eye_snap,
    }, &mesh);
    try std.testing.expectEqual(@as(usize, 3), scratch_inline.items.len);
    // Snapshot eye on the right: (-10) farthest, then the source entry, +10.
    try std.testing.expectApproxEqAbs(@as(f32, -10.0), scratch_inline.items[0].m[12], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), scratch_inline.items[1].m[12], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), scratch_inline.items[2].m[12], 1e-6);

    // Farthest-first check per eye (strict distances, no tiebreak needed).
    const d_live_0 = Vec3.new(scratch_build.items[0].m[12], 0, 0).sub(eye_live).lengthSq();
    const d_live_1 = Vec3.new(scratch_build.items[1].m[12], 0, 0).sub(eye_live).lengthSq();
    try std.testing.expect(d_live_0 > d_live_1);
    const d_snap_0 = Vec3.new(scratch_inline.items[0].m[12], 0, 0).sub(eye_snap).lengthSq();
    const d_snap_1 = Vec3.new(scratch_inline.items[1].m[12], 0, 0).sub(eye_snap).lengthSq();
    try std.testing.expect(d_snap_0 > d_snap_1);
}
