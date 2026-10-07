//! Parallel cull pass: `ParallelCull` chunk state plus
//! `buildFrameQueuesParallel` (world-matrix warming, chunked data-parallel
//! cull, deterministic chunk-order merge reproducing the serial loop, serial
//! instanced tail; all fallible growth up front so OOM fails with
//! queues/stats untouched). Imports the `items`, `cull`, and `instances`
//! leaves plus engine modules — never the `build.zig` facade and never the
//! `frame` sibling (documented anti-cycle rule: the edge is one-directional
//! `frame` → `parallel`). `buildFrameQueuesParallel` is `pub` for the
//! `frame` sibling (same discipline as `cull.appendRenderItem`) but is
//! deliberately NOT re-exported by the facade, so the public surface is
//! identical to the pre-split file.
const std = @import("std");

const math = @import("math");
const Mat4 = math.Mat4;
const Frustum = math.Frustum;
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const Mesh = @import("../../../mesh.zig").Mesh;
const InstancedMesh = @import("../../../mesh.zig").InstancedMesh;
const material_mod = @import("../../../material.zig");
const Material = material_mod.Material;
const Texture = @import("../../../texture.zig").Texture;
const skeleton_mod = @import("../../../animation/skeleton.zig");
const visibility = @import("../../../visibility/mod.zig");
const jobs = @import("../../../jobs.zig");
const stats_mod = @import("../../stats.zig");
const SceneStats = stats_mod.SceneStats;
const instance_staging = @import("../../instance_staging.zig");
const items = @import("../items.zig");
const RenderQueues = items.RenderQueues;
const RenderMeshItem = items.RenderMeshItem;
const RenderInstancedBatch = items.RenderInstancedBatch;
const CulledMesh = items.CulledMesh;
const TransparentKind = items.TransparentKind;
const TransparentDrawEntry = items.TransparentDrawEntry;
const sortTransparentDrawOrder = items.sortTransparentDrawOrder;
const MAX_BONES = items.MAX_BONES;
const cull = @import("../cull.zig");
const FrameCullContext = cull.FrameCullContext;
const worldMatrixCached = cull.worldMatrixCached;
const appendRenderItem = cull.appendRenderItem;
const cullNonInstancedMesh = cull.cullNonInstancedMesh;
const instances = @import("../instances.zig");
const submitInstancedMesh = instances.submitInstancedMesh;

/// Parallel cull pass state. Chunks partition `ctx.meshes` into fixed
/// ranges; each chunk collects its records and stats independently (no
/// locks) and the caller merges them in chunk order, reproducing the serial
/// loop's queue order and stat totals exactly.
const ParallelCull = struct {
    ctx: FrameCullContext,
    frustum: Frustum,
    eye: Vec3,
    /// Meshes per chunk (ceil split; the last chunk may be short).
    span: usize,
    records: []std.ArrayListUnmanaged(CulledMesh),
    chunk_stats: []SceneStats,

    fn cullChunkRange(pass: *ParallelCull, start: usize, end: usize) void {
        for (start..end) |chunk_id| {
            const lo = chunk_id * pass.span;
            // chunk_count can exceed meshes.len (span floors at 1), so tail
            // chunks beyond the mesh list must be skipped: without this,
            // `lo..hi` would have start > end and panic.
            if (lo >= pass.ctx.meshes.len) continue;
            const hi = @min(lo + pass.span, pass.ctx.meshes.len);
            var local = SceneStats{};
            for (pass.ctx.meshes[lo..hi], lo..) |mesh, mesh_index| {
                if (mesh.is_lod_child) continue;
                if ((mesh.layer_mask & pass.ctx.culling_mask) == 0) continue;
                // Deferred-creation meshes have no buffers yet (same skip as
                // the serial loop above).
                if (mesh.gpu_pending) continue;
                // Instance-bearing meshes are staged serially in the merge
                // tail (sg buffer management); skip them here so they are
                // never culled as plain meshes and double-queued.
                if (mesh.instances.items.len > 0) continue;
                if (cullNonInstancedMesh(pass.ctx, pass.frustum, pass.eye, mesh, &local, mesh_index)) |culled| {
                    pass.records[chunk_id].appendAssumeCapacity(culled);
                }
            }
            pass.chunk_stats[chunk_id] = local;
        }
    }
};

pub fn buildFrameQueuesParallel(
    ctx: FrameCullContext,
    frustum: Frustum,
    eye: Vec3,
    pool: *jobs.Pool,
) !void {
    // World matrices cache per mesh with parent-chain recursion; lazy fill
    // from two workers could race on a shared parent's cache. Warm the
    // cache in mesh order first — pure TRS work, O(meshes), and a no-op
    // for meshes already tagged with this cache key (e.g. second camera of
    // a PIP render).
    for (ctx.meshes) |m| _ = worldMatrixCached(ctx.cache_key, m);

    const chunk_count = (pool.workerCount() + 1) * 4;
    const span = (ctx.meshes.len + chunk_count - 1) / chunk_count;

    // Reusable per-view scratch, retained across frames: reset clears the
    // previous call's lengths, ensure grows only when the chunk/span demand
    // exceeds what is already held. ALL fallible growth happens here,
    // before any queue/stats write and before forkJoin, so OOM still fails
    // with queues/stats untouched and the caller falls back to serial.
    const scratch = &ctx.queues.parallel_scratch;
    scratch.reset();
    try scratch.ensure(ctx.allocator, chunk_count, span);
    const records = scratch.records.items[0..chunk_count];
    const chunk_stats = scratch.chunk_stats.items[0..chunk_count];
    @memset(chunk_stats, .{});

    var pass = ParallelCull{
        .ctx = ctx,
        .frustum = frustum,
        .eye = eye,
        .span = span,
        .records = records,
        .chunk_stats = chunk_stats,
    };
    pool.forkJoin(ParallelCull, &pass, ParallelCull.cullChunkRange, chunk_count);

    // Pre-reserve capacity in destination queues to minimize reallocations
    var total_rendered: usize = 0;
    for (chunk_stats) |local| {
        total_rendered += local.rendered_meshes;
    }
    ctx.queues.items.ensureUnusedCapacity(ctx.allocator, total_rendered) catch {};

    // Deterministic merge: chunk order == mesh order, so the queues land
    // exactly as the serial loop would have filled them. Копии скинов/
    // шейдеров делаются здесь же, серийно и в том же порядке — параллельные
    // воркеры только заимствовали живые данные в скретч.
    for (records, chunk_stats) |*list, local| {
        ctx.stats.total_meshes += local.total_meshes;
        ctx.stats.rendered_meshes += local.rendered_meshes;
        ctx.stats.culled_meshes += local.culled_meshes;
        ctx.stats.occluded_meshes += local.occluded_meshes;
        for (list.items) |culled| appendRenderItem(ctx, culled);
    }

    // Instance-bearing meshes: serial submission (sg buffer management).
    for (ctx.meshes, 0..) |mesh, mesh_index| {
        if (mesh.is_lod_child or mesh.instances.items.len == 0) continue;
        if ((mesh.layer_mask & ctx.culling_mask) == 0) continue;
        submitInstancedMesh(ctx, frustum, mesh, mesh_index);
    }
}

test "parallel setup OOM fails cleanly without leaking" {
    const ally = std.testing.allocator;

    const count = 4;
    const meshes = try ally.alloc(Mesh, count);
    defer ally.free(meshes);
    const ptrs = try ally.alloc(*Mesh, count);
    defer ally.free(ptrs);
    for (0..count) |i| {
        meshes[i] = .{
            .name = "m",
            .vertex_buffer = .{},
            .index_buffer = .{},
            .index_count = 3,
        };
        ptrs[i] = &meshes[i];
    }

    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();

    // (2 workers + 1) * 4 = 12 per-chunk buffers: fail_index 0 hits the
    // records array itself, 1 the first chunk buffer, 7 a middle one.
    for ([_]usize{ 0, 1, 7 }) |fail_index| {
        var gpa = std.heap.DebugAllocator(.{}){};
        var failing = std.testing.FailingAllocator.init(gpa.allocator(), .{ .fail_index = fail_index });
        var queues = RenderQueues{};
        var stats = SceneStats{};
        var culler = visibility.OcclusionCuller.init();
        const result = buildFrameQueuesParallel(.{
            .allocator = failing.allocator(),
            .meshes = ptrs,
            .cache_key = 100 + fail_index,
            .view_proj = Mat4.identity,
            .eye = Vec3.zero,
            .cull_frustum = false,
            .cull_occlusion = false,
            .occlusion_culler = &culler,
            .stats = &stats,
            .queues = &queues,
            .default_white_id = 1,
        }, Frustum.fromViewProjection(Mat4.identity), Vec3.zero, pool);
        try std.testing.expectError(error.OutOfMemory, result);
        // Setup failed before any queue/stats write: the frame is untouched.
        try std.testing.expectEqual(@as(usize, 0), queues.items.items.len);
        try std.testing.expectEqual(@as(usize, 0), queues.transparent.items.len);
        try std.testing.expectEqual(@as(usize, 0), queues.transparent_order.items.len);
        try std.testing.expectEqual(@as(u32, 0), stats.total_meshes);
        try std.testing.expectEqual(@as(u32, 0), stats.rendered_meshes);
        try std.testing.expectEqual(@as(u32, 0), stats.culled_meshes);
        // The GPA deinit check is the leak proof: every buffer allocated
        // before the failure was freed by the setup cleanup.
        try std.testing.expectEqual(std.heap.Check.ok, gpa.deinit());
    }
}
