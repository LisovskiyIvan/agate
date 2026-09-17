const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const Mesh = @import("../mesh.zig").Mesh;
const InstancedMesh = @import("../mesh.zig").InstancedMesh;
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
    /// Pre-sized to instances.len; chunk c owns segment [c*span, c*span+span)
    /// and packs its visible matrices densely at the segment start (a chunk
    /// holds at most span visibles, so the segment always fits). The serial
    /// tail compacts the segments into [0, total_visible).
    out_matrices: []Mat4,

    fn runChunks(stage: *ParallelInstanceStage, start: usize, end: usize) void {
        for (start..end) |chunk_id| {
            const lo = chunk_id * stage.span;
            if (lo >= stage.instances.len) {
                // Ceil-division span can overshoot on the tail chunk when
                // chunk_count divides unevenly: an empty chunk contributes
                // nothing (its slice would otherwise run lo > hi).
                stage.chunk_aabbs[chunk_id] = BoundingBox.zero;
                stage.chunk_visible_counts[chunk_id] = 0;
                continue;
            }
            const hi = @min(lo + stage.span, stage.instances.len);
            var aabb = BoundingBox.zero;
            var dst = lo;
            for (stage.instances[lo..hi]) |inst| {
                if (!inst.is_visible) continue;
                const world = instanceWorldMatrix(inst);
                stage.out_matrices[dst] = world;
                dst += 1;
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

/// Minimal per-frame input for instance staging, extracted from
/// FrameCullContext (see render_queue.zig). Scene.prepareFrame pre-stages
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
pub fn stageInstancedMesh(sc: InstanceStageContext, mesh: *Mesh) void {
    // Deferred-creation meshes have no vertex/index buffers yet; staging
    // instance data for them would produce a draw against invalid handles.
    if (mesh.gpu_pending) return;
    const st = &mesh.instance_render;
    if (st.staged_frame == sc.frame_id) return;

    var combined_aabb = BoundingBox.zero;
    const use_parallel = if (sc.thread_pool) |pool|
        pool.workerCount() > 0 and mesh.instances.items.len >= 256
    else
        false;
    if (use_parallel) {
        const pool = sc.thread_pool.?;
        const chunk_count = @min((pool.workerCount() + 1) * 2, 64);
        const span = (mesh.instances.items.len + chunk_count - 1) / chunk_count;

        // Scratch upper bound first: OOM here publishes nothing (guard
        // unset, previous state intact) and a later call may retry.
        sc.instance_matrices.resize(sc.allocator, mesh.instances.items.len) catch return;

        var chunk_aabbs_buf: [64]BoundingBox = undefined;
        var chunk_visible_buf: [64]usize = undefined;

        var stage = ParallelInstanceStage{
            .instances = mesh.instances.items,
            .span = span,
            .chunk_aabbs = chunk_aabbs_buf[0..chunk_count],
            .chunk_visible_counts = chunk_visible_buf[0..chunk_count],
            .out_matrices = sc.instance_matrices.items,
        };

        pool.forkJoin(ParallelInstanceStage, &stage, ParallelInstanceStage.runChunks, chunk_count);

        // Serial tail: merge chunk AABBs, compact the per-chunk segments
        // down to [0, total_visible). Destinations never overtake sources
        // (offset_c <= c*span: every earlier chunk contributes at most span),
        // so the forward copy is overlap-safe.
        var total_visible: usize = 0;
        for (0..chunk_count) |c| {
            const count = chunk_visible_buf[c];
            const src_lo = c * span;
            if (count > 0) {
                std.mem.copyForwards(
                    Mat4,
                    sc.instance_matrices.items[total_visible .. total_visible + count],
                    sc.instance_matrices.items[src_lo .. src_lo + count],
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
        sc.instance_matrices.items.len = total_visible;
    } else {
        sc.instance_matrices.clearRetainingCapacity();
        for (mesh.instances.items) |inst| {
            if (!inst.is_visible) continue;
            const world = instanceWorldMatrix(inst);
            // OOM: publish nothing (guard unset, previous state intact).
            sc.instance_matrices.append(sc.allocator, world) catch return;
            const box = instanceWorldAABB(inst, world);
            if (combined_aabb.isValid()) {
                combined_aabb = combined_aabb.merge(box);
            } else {
                combined_aabb = box;
            }
        }
    }
    const active_count = sc.instance_matrices.items.len;

    // Per-instance transparency sorting (OIT):
    // When the instanced mesh is transparent or a decal, sort its instance
    // matrices back-to-front relative to the camera eye. Farthest instances
    // render first, blending nearer instances over them correctly.
    if (active_count > 1 and (isTransparentMaterial(mesh.material) or mesh.is_decal)) {
        const SortCtx = struct {
            eye: Vec3,
            pub fn sortFn(c: @This(), a: Mat4, b: Mat4) bool {
                const pos_a = Vec3.new(a.m[12], a.m[13], a.m[14]);
                const pos_b = Vec3.new(b.m[12], b.m[13], b.m[14]);
                const dist_a = pos_a.sub(c.eye).lengthSq();
                const dist_b = pos_b.sub(c.eye).lengthSq();
                if (dist_a != dist_b) {
                    return dist_a > dist_b; // back-to-front: farthest first
                }
                for (0..16) |i| {
                    if (a.m[i] != b.m[i]) return a.m[i] < b.m[i];
                }
                return false;
            }
        };
        std.mem.sort(Mat4, sc.instance_matrices.items[0..active_count], SortCtx{ .eye = sc.eye }, SortCtx.sortFn);
    }

    if (active_count > 0 and sg.isvalid()) {
        // Владение GPU (P1): CPU-стейджинг выше может идти с воркеров
        // пула, но создание/обновление instance-буфера — только
        // context-поток. prepareFrame сегодня выполняется на нём
        // (покрыт ассертом Scene.prepareFrame/render), это внутренний
        // трипвайр: при выносе prepare на update-поток sg-блок уедет за
        // handoff во flushPendingGpuUploads.
        gpu_thread.assertOnContextThread();
        const items = sc.instance_matrices.items[0..active_count];
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
            const new_cap = @max(active_count, st.capacity * 2);
            const new_buf = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = new_cap * @sizeOf(Mat4),
            });
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
            const old = st.buffer;
            if (old.id != 0) {
                if (sc.retire_queue) |q| {
                    q.retireBuffer(sc.allocator, old);
                } else {
                    // Standalone low-level caller without a queue (see the
                    // context docs): no cross-frame snapshot exists, and we
                    // are on the context thread — destroy immediately.
                    sg.destroyBuffer(old);
                }
            }
            st.buffer = new_buf;
            st.capacity = new_cap;
            st.hash = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(items));
            st.uploaded_count = active_count;
        } else {
            const h = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(items));
            if (active_count != st.uploaded_count or h != st.hash) {
                sg.updateBuffer(st.buffer, sg.asRange(items));
                // Учёт динамики: только при реальном изменении (dedup по хешу выше).
                upload_meter.record(active_count * @sizeOf(Mat4));
                st.hash = h;
                st.uploaded_count = active_count;
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
    st.bounds = combined_aabb;
    st.count = @intCast(active_count);
    st.staged_frame = sc.frame_id;
}

/// Pre-stage every instance-bearing mesh once per frame. Skips LOD children,
/// GPU-pending meshes, and meshes with no instances; staging itself is
/// guarded per mesh by `staged_frame`, so calling this before the
/// view queues and again implicitly via submitInstancedMesh (render_queue.zig)
/// stays once-only.
pub fn stageInstances(sc: InstanceStageContext, meshes: []const *Mesh) void {
    for (meshes) |mesh| {
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
    var aabbs_buf: [64]BoundingBox = undefined;
    var counts_buf: [64]usize = undefined;
    var stage = ParallelInstanceStage{
        .instances = ptrs,
        .span = span,
        .chunk_aabbs = aabbs_buf[0..chunk_count],
        .chunk_visible_counts = counts_buf[0..chunk_count],
        .out_matrices = out,
    };
    ParallelInstanceStage.runChunks(&stage, 0, chunk_count);

    // Chunk 0: 5 visibles packed at the segment start, matrices exact.
    try std.testing.expectEqual(@as(usize, 5), counts_buf[0]);
    for (0..5) |k| {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(k)), out[k].m[12], 1e-6);
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
