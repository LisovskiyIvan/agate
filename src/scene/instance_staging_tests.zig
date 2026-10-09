const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const Mesh = @import("../mesh.zig").Mesh;
const InstancedMesh = @import("../mesh.zig").InstancedMesh;
const InstancePreviewState = @import("../mesh.zig").InstancePreviewState;
const InstanceRenderState = @import("../mesh.zig").InstanceRenderState;
const StagedInstanceRecord = @import("../mesh.zig").StagedInstanceRecord;
const jobs = @import("../jobs.zig");

const staging = @import("instance_staging.zig");
const ParallelInstanceStage = staging.ParallelInstanceStage;
const InstanceStageContext = staging.InstanceStageContext;
const GpuStageContext = staging.GpuStageContext;
const CpuStageContext = staging.CpuStageContext;
const CpuStageResult = staging.CpuStageResult;
const stageSegmentCpu = staging.stageSegmentCpu;
const failRecord = staging.failRecord;
const stageInstancedMesh = staging.stageInstancedMesh;
const stageInstancesGpu = staging.stageInstancesGpu;
const stageInstancesGpuState = staging.stageInstancesGpuState;
const stageInstancesCpu = staging.stageInstancesCpu;
const freezeStagedRecords = staging.freezeStagedRecords;
const stageInstancesLatch = staging.stageInstancesLatch;
const commitPublishedRecords = staging.commitPublishedRecords;
const stageInstances = staging.stageInstances;
const testArmGrowthFailOnce = staging.testArmGrowthFailOnce;
const testLastInjectedFailId = staging.testLastInjectedFailId;
const material_mod = @import("../material.zig");

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
