//! Frame-queue assembly: `buildFrameQueues` (occluder rasterization, serial
//! cull loop, parallel dispatch with serial fallback) plus the parallel
//! cull pass (`ParallelCull`, `buildFrameQueuesParallel`). Imports the
//! `items`, `cull`, and `instances` leaves plus engine modules — never the
//! `render_queue.zig` facade (documented anti-cycle rule). Nothing imports
//! this module except the facade.
const std = @import("std");

const math = @import("math");
const Mat4 = math.Mat4;
const Frustum = math.Frustum;
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const Mesh = @import("../../mesh.zig").Mesh;
const InstancedMesh = @import("../../mesh.zig").InstancedMesh;
const material_mod = @import("../../material.zig");
const Material = material_mod.Material;
const StandardMaterial = material_mod.StandardMaterial;
const Texture = @import("../../texture.zig").Texture;
const skeleton_mod = @import("../../animation/skeleton.zig");
const visibility = @import("../../visibility/mod.zig");
const jobs = @import("../../jobs.zig");
const stats_mod = @import("../stats.zig");
const SceneStats = stats_mod.SceneStats;
const instance_staging = @import("../instance_staging.zig");
const items = @import("items.zig");
const RenderQueues = items.RenderQueues;
const RenderMeshItem = items.RenderMeshItem;
const RenderInstancedBatch = items.RenderInstancedBatch;
const CulledMesh = items.CulledMesh;
const TransparentKind = items.TransparentKind;
const TransparentDrawEntry = items.TransparentDrawEntry;
const sortTransparentDrawOrder = items.sortTransparentDrawOrder;
const MAX_BONES = items.MAX_BONES;
const cull = @import("cull.zig");
const FrameCullContext = cull.FrameCullContext;
const worldMatrixCached = cull.worldMatrixCached;
const appendRenderItem = cull.appendRenderItem;
const cullNonInstancedMesh = cull.cullNonInstancedMesh;
const instances = @import("instances.zig");
const submitInstancedMesh = instances.submitInstancedMesh;

/// Phase -1 (occluder rasterization) and Phase 0 (frustum/occlusion
/// culling, LOD picking, instance-buffer management, queue fill) of the
/// render frame. Results land in ctx.queues and the stats counters.
///
/// Instance-bearing meshes stay serial in every path: they interleave with
/// sg buffer creation/update, which is single-context. Plain meshes cull
/// data-parallel when a pool is attached and the scene clears
/// `parallel_min_meshes`; the parallel pass produces the same records in
/// the same order as the serial loop (chunks partition the mesh list in
/// order and merge back in chunk order), so the choice between them is a
/// scheduling detail, never a behavior change.
pub fn buildFrameQueues(ctx: FrameCullContext) void {
    const frustum = Frustum.fromViewProjection(ctx.view_proj);
    const eye = ctx.eye;

    // Stage-2A identity: assign render uids serially before any parallel
    // read (binning/culling workers only read). Idempotent, no behavior change.
    for (ctx.meshes) |m| _ = m.ensureUid();

    // Phase -1: Occlusion Culling setup & occluder rasterization (serial:
    // the Hi-Z rasterizer is stateful).
    if (ctx.cull_occlusion) {
        ctx.occlusion_culler.beginFrame(ctx.view_proj);
        for (ctx.meshes) |m| {
            if (!m.is_lod_child and m.is_visible and m.is_occluder and ((m.layer_mask & ctx.culling_mask) != 0)) {
                const m_world = worldMatrixCached(ctx.cache_key, m);
                ctx.occlusion_culler.rasterizeOccluderMesh(
                    m.cpu_positions,
                    m.cpu_indices,
                    m.local_bounding_box,
                    m_world,
                );
            }
        }
        ctx.occlusion_culler.endOccluders();
        ctx.stats.occluders_count = ctx.occlusion_culler.occluder_count;
        ctx.stats.occluder_triangles = ctx.occlusion_culler.triangles_rasterized;
    }

    // Phase 0: cull + queue fill.
    const pool = ctx.thread_pool;
    const min_meshes = if (ctx.parallel_min_meshes == 0)
        (if (pool) |p| @max(@as(usize, 64), (p.workerCount() + 1) * 32) else 128)
    else
        ctx.parallel_min_meshes;
    if (pool != null and pool.?.workerCount() > 0 and ctx.meshes.len >= min_meshes) {
        const parallel_ran = blk: {
            buildFrameQueuesParallel(ctx, frustum, eye, pool.?) catch break :blk false;
            break :blk true;
        };
        if (parallel_ran) return;
        // Parallel setup OOM: it fails before any queue/stats write (only
        // benign world-matrix cache warming precedes the fallible allocs),
        // so fall through to the serial loop and still draw the frame.
    }

    for (ctx.meshes, 0..) |mesh, mesh_index| {
        if (mesh.is_lod_child) continue;
        if ((mesh.layer_mask & ctx.culling_mask) == 0) continue;
        // Deferred-creation meshes (off-context uploadGeometry) have no GPU
        // buffers until finishGpuUpload runs; drawing them would bind
        // invalid handles, so they stay out of the queues until finished.
        if (mesh.gpu_pending) continue;
        if (mesh.instances.items.len > 0) {
            submitInstancedMesh(ctx, frustum, mesh, mesh_index);
            continue;
        }
        if (cullNonInstancedMesh(ctx, frustum, eye, mesh, ctx.stats, mesh_index)) |culled| {
            appendRenderItem(ctx, culled);
        }
    }
}

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

fn buildFrameQueuesParallel(
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

test "shared LOD mesh preserves entity transforms without mutation" {
    const ally = std.testing.allocator;

    // Полностью инициализированный общий LOD-ребёнок (никаких undefined):
    // сентинел-хендлы геометрии отличают его от родителей.
    var shared_lod = Mesh{
        .name = "shared_lod",
        .vertex_buffer = .{ .id = 77 },
        .index_buffer = .{ .id = 78 },
        .index_count = 9,
        .position = Vec3.zero,
        .rotation = Vec3.zero,
        .scaling = Vec3.new(1.0, 1.0, 1.0),
        .base_matrix = Mat4.identity,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
        .culling_strategy = .always_render,
    };
    shared_lod.cached_aabb = shared_lod.local_bounding_box;
    shared_lod.cached_matrix = Mat4.identity;
    shared_lod.cached_frame = 0;

    var mesh1: Mesh = shared_lod;
    mesh1.name = "entity_a";
    mesh1.vertex_buffer = .{ .id = 11 };
    mesh1.index_buffer = .{ .id = 12 };
    mesh1.index_count = 3;
    mesh1.position = Vec3.new(10.0, 0.0, 0.0);
    mesh1.cached_frame = 0;
    try mesh1.lod_levels.append(ally, .{ .distance = 0.0, .mesh = &shared_lod });
    defer mesh1.lod_levels.deinit(ally);

    var mesh2: Mesh = shared_lod;
    mesh2.name = "entity_b";
    mesh2.vertex_buffer = .{ .id = 13 };
    mesh2.index_buffer = .{ .id = 14 };
    mesh2.index_count = 3;
    mesh2.position = Vec3.new(20.0, 0.0, 0.0);
    mesh2.cached_frame = 0;
    try mesh2.lod_levels.append(ally, .{ .distance = 0.0, .mesh = &shared_lod });
    defer mesh2.lod_levels.deinit(ally);

    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var stats = SceneStats{};
    var culler = visibility.OcclusionCuller.init();

    const meshes = [_]*Mesh{ &mesh1, &mesh2 };
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 42,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });

    // The shared child LOD mesh MUST NOT be mutated!
    try std.testing.expectEqual(@as(f32, 0.0), shared_lod.position.x);
    try std.testing.expectEqual(@as(usize, 2), queues.items.items.len);

    // P4: живых указателей в очередях нет — выбор общего LOD-ребёнка доказан
    // сентинел-хендлами: оба item несут геометрию shared_lod (77/78/9),
    // а не родителей (11/12, 13/14), но свои матрицы мира.
    for (queues.items.items) |it| {
        try std.testing.expectEqual(@as(u32, 77), it.vertex_buffer.id);
        try std.testing.expectEqual(@as(u32, 78), it.index_buffer.id);
        try std.testing.expectEqual(@as(u32, 9), it.index_count);
    }

    // But each entity keeps its own distinct world matrix!
    const m0_x = queues.items.items[0].model.m[12];
    const m1_x = queues.items.items[1].model.m[12];
    try std.testing.expect((m0_x == 10.0 and m1_x == 20.0) or (m0_x == 20.0 and m1_x == 10.0));
}

test "culling_mask filters out meshes with disjoint layer_mask" {
    const ally = std.testing.allocator;

    var mesh1 = Mesh{
        .name = "layer1",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .layer_mask = 0b01,
    };
    var mesh2 = Mesh{
        .name = "layer2",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .layer_mask = 0b10,
    };

    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var stats = SceneStats{};
    var culler = visibility.OcclusionCuller.init();

    const meshes = [_]*Mesh{ &mesh1, &mesh2 };

    // Cull with mask 0b01: only mesh1 should be queued
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 1,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .culling_mask = 0b01,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });
    try std.testing.expectEqual(@as(usize, 1), queues.items.items.len);
    try std.testing.expectEqual(@as(u32, 0), queues.items.items[0].mesh_index);

    // Cull with mask 0b10: only mesh2 should be queued
    queues.reset();
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 2,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .culling_mask = 0b10,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });
    try std.testing.expectEqual(@as(usize, 1), queues.items.items.len);
    try std.testing.expectEqual(@as(u32, 1), queues.items.items[0].mesh_index);
}

test "parallel cull produces serial-identical queues" {
    const ally = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const rand = prng.random();

    // Deterministic synthetic scene: mixed layer masks, visibility, and
    // placement so both culling decisions and sort keys vary.
    const count = 3000;
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
            .layer_mask = if (i % 7 == 0) 0b10 else 0b01,
        };
        if (i % 5 == 0) {
            // Near the origin: survives the identity-projection frustum.
            meshes[i].position = Vec3.new(rand.float(f32) * 0.2, rand.float(f32) * 0.2, rand.float(f32) * 0.2);
        } else {
            meshes[i].position = Vec3.new(rand.float(f32) * 100 - 50, rand.float(f32) * 100 - 50, rand.float(f32) * 100 - 50);
        }
        meshes[i].is_visible = i % 11 != 0;
        ptrs[i] = &meshes[i];
    }

    var culler = visibility.OcclusionCuller.init();
    var stats_a = SceneStats{};
    var queues_a = RenderQueues{};
    defer queues_a.deinit(ally);
    var stats_b = SceneStats{};
    var queues_b = RenderQueues{};
    defer queues_b.deinit(ally);

    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();

    // Serial pass (no pool attached).
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = ptrs,
        .cache_key = 1,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = true,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_a,
        .queues = &queues_a,
        .default_white_id = 1,
    });

    // Parallel pass (forced past the mesh threshold, 2 workers).
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = ptrs,
        .cache_key = 2,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = true,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_b,
        .queues = &queues_b,
        .default_white_id = 1,
        .thread_pool = pool,
        .parallel_min_meshes = 1,
    });

    // Identical stats…
    try std.testing.expectEqual(stats_a.total_meshes, stats_b.total_meshes);
    try std.testing.expectEqual(stats_a.rendered_meshes, stats_b.rendered_meshes);
    try std.testing.expectEqual(stats_a.culled_meshes, stats_b.culled_meshes);
    try std.testing.expectEqual(stats_a.occluded_meshes, stats_b.occluded_meshes);
    // …identical queue lengths…
    try std.testing.expectEqual(queues_a.items.items.len, queues_b.items.items.len);
    try std.testing.expect(stats_a.rendered_meshes > 0);
    try std.testing.expect(stats_a.culled_meshes > 0);
    // …and identical records in identical order (chunk merge order ==
    // serial mesh order, and the same world matrices feed both passes).
    for (queues_a.items.items, queues_b.items.items) |a, b| {
        try std.testing.expectEqual(a.mesh_index, b.mesh_index);
        try std.testing.expectEqual(a.model, b.model);
        try std.testing.expectEqual(a.distance_sq, b.distance_sq);
        try std.testing.expectEqual(a.texture_id, b.texture_id);
        try std.testing.expectEqual(a.transparent, b.transparent);
    }
}

test "parallel cull reuses scratch across calls" {
    const ally = std.testing.allocator;

    // Deterministic scene, forced onto the parallel path; no frustum or
    // occlusion culling so every mesh queues identically each frame.
    const count = 512;
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
        const f: f32 = @floatFromInt(i);
        meshes[i].position = Vec3.new(f * 0.01, 0.0, f * 0.005);
        ptrs[i] = &meshes[i];
    }

    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();

    var culler = visibility.OcclusionCuller.init();
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    const chunk_count = (pool.workerCount() + 1) * 4;
    const span = (count + chunk_count - 1) / chunk_count;

    // First parallel frame.
    var stats_a = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = ptrs,
        .cache_key = 1,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_a,
        .queues = &queues,
        .default_white_id = 1,
        .thread_pool = pool,
        .parallel_min_meshes = 1,
    });
    try std.testing.expectEqual(@as(usize, count), queues.items.items.len);
    const snapshot = try ally.alloc(RenderMeshItem, queues.items.items.len);
    defer ally.free(snapshot);
    @memcpy(snapshot, queues.items.items);

    // Scratch grew to the demand: outer capacity covers every chunk and
    // each per-chunk record buffer covers its full span.
    try std.testing.expectEqual(chunk_count, queues.parallel_scratch.records.items.len);
    try std.testing.expect(queues.parallel_scratch.records.capacity >= chunk_count);
    for (queues.parallel_scratch.records.items) |*r| {
        try std.testing.expect(r.capacity >= span);
    }
    const outer_cap = queues.parallel_scratch.records.capacity;
    const inner_caps = try ally.alloc(usize, chunk_count);
    defer ally.free(inner_caps);
    for (queues.parallel_scratch.records.items, 0..) |*r, i| inner_caps[i] = r.capacity;

    // Second parallel frame reuses the same queues (draw queues reset;
    // scratch retained) and must reproduce the first frame exactly.
    queues.reset();
    var stats_b = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = ptrs,
        .cache_key = 2,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_b,
        .queues = &queues,
        .default_white_id = 1,
        .thread_pool = pool,
        .parallel_min_meshes = 1,
    });
    try std.testing.expectEqual(stats_a.total_meshes, stats_b.total_meshes);
    try std.testing.expectEqual(stats_a.rendered_meshes, stats_b.rendered_meshes);
    try std.testing.expectEqual(stats_a.culled_meshes, stats_b.culled_meshes);
    try std.testing.expectEqual(stats_a.occluded_meshes, stats_b.occluded_meshes);
    try std.testing.expectEqual(snapshot.len, queues.items.items.len);
    for (snapshot, queues.items.items) |a, b| {
        try std.testing.expectEqual(a.mesh_index, b.mesh_index);
        try std.testing.expectEqual(a.model, b.model);
        try std.testing.expectEqual(a.distance_sq, b.distance_sq);
        try std.testing.expectEqual(a.texture_id, b.texture_id);
        try std.testing.expectEqual(a.transparent, b.transparent);
    }

    // No regrowth on the second call: identical demand reuses the retained
    // buffers (same capacities, leak-checked by the testing allocator via
    // queues.deinit).
    try std.testing.expectEqual(outer_cap, queues.parallel_scratch.records.capacity);
    for (queues.parallel_scratch.records.items, 0..) |*r, i| {
        try std.testing.expectEqual(inner_caps[i], r.capacity);
    }
}

test "stale AABB does not drive LOD selection" {
    const ally = std.testing.allocator;
    const unit_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1));

    var lod_far = Mesh{
        .name = "lod_far",
        .vertex_buffer = .{},
        .index_buffer = .{},
        // Маркер выбора LOD: очереди больше не несут живой указатель,
        // поэтому дальний уровень помечен отличным index_count.
        .index_count = 9,
        .local_bounding_box = unit_box,
        .culling_strategy = .always_render,
    };
    var base = Mesh{
        .name = "lod_base",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        // Fresh position is beyond the switch distance (far).
        .position = Vec3.new(100, 0, 0),
        .local_bounding_box = unit_box,
        .culling_strategy = .always_render,
    };
    try base.addLODLevel(ally, 50.0, &lod_far);
    defer base.lod_levels.deinit(ally);

    // Seed a stale AABB tagged with an older frame: as if the mesh sat at
    // the origin (near, distance 0 < 50) last frame, while its fresh
    // position is far (distance 100 >= 50). Reading it would pick the near
    // LOD (self); the fresh AABB must pick the far child.
    base.cached_aabb = unit_box;
    base.cached_matrix = Mat4.identity;
    base.cached_frame = 0;

    var queues = RenderQueues{};
    defer queues.deinit(ally);
    var stats = SceneStats{};
    var culler = visibility.OcclusionCuller.init();
    const meshes = [_]*Mesh{&base};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 1,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });

    try std.testing.expectEqual(@as(usize, 1), queues.items.items.len);
    try std.testing.expectEqual(@as(u32, 9), queues.items.items[0].index_count);
}

test "stale AABB LOD selection matches parallel path" {
    const ally = std.testing.allocator;
    const unit_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1));

    var lod_far_s = Mesh{
        .name = "lod_far_s",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 9,
        .local_bounding_box = unit_box,
        .culling_strategy = .always_render,
    };
    var base_s = Mesh{
        .name = "lod_base_s",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(100, 0, 0),
        .local_bounding_box = unit_box,
        .culling_strategy = .always_render,
    };
    try base_s.addLODLevel(ally, 50.0, &lod_far_s);
    defer base_s.lod_levels.deinit(ally);
    base_s.cached_aabb = unit_box;
    base_s.cached_matrix = Mat4.identity;
    base_s.cached_frame = 0;

    var lod_far_p = Mesh{
        .name = "lod_far_p",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 9,
        .local_bounding_box = unit_box,
        .culling_strategy = .always_render,
    };
    var base_p = Mesh{
        .name = "lod_base_p",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(100, 0, 0),
        .local_bounding_box = unit_box,
        .culling_strategy = .always_render,
    };
    try base_p.addLODLevel(ally, 50.0, &lod_far_p);
    defer base_p.lod_levels.deinit(ally);
    base_p.cached_aabb = unit_box;
    base_p.cached_matrix = Mat4.identity;
    base_p.cached_frame = 0;

    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();

    var culler = visibility.OcclusionCuller.init();

    // Serial pass (no pool attached).
    var stats_s = SceneStats{};
    var queues_s = RenderQueues{};
    defer queues_s.deinit(ally);
    const meshes_s = [_]*Mesh{&base_s};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes_s,
        .cache_key = 1,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_s,
        .queues = &queues_s,
        .default_white_id = 1,
    });

    // Parallel pass (forced past the mesh threshold, 2 workers): pre-warms
    // the world-matrix cache, so it always decided from the fresh AABB.
    var stats_p = SceneStats{};
    var queues_p = RenderQueues{};
    defer queues_p.deinit(ally);
    const meshes_p = [_]*Mesh{&base_p};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes_p,
        .cache_key = 1,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_p,
        .queues = &queues_p,
        .default_white_id = 1,
        .thread_pool = pool,
        .parallel_min_meshes = 1,
    });

    // Both paths must agree on the fresh (far) LOD despite the stale seed.
    try std.testing.expectEqual(@as(usize, 1), queues_s.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), queues_p.items.items.len);
    try std.testing.expectEqual(@as(u32, 9), queues_s.items.items[0].index_count);
    try std.testing.expectEqual(@as(u32, 9), queues_p.items.items[0].index_count);
}

test "transparent regular+instanced groups share one back-to-front order" {
    const ally = std.testing.allocator;
    const material = @import("../../material.zig");

    var blend_mat = material.StandardMaterial.init("blend");
    blend_mat.alpha_mode = .blend;
    const blend: Material = .{ .standard = &blend_mat };
    const unit_box = BoundingBox.init(Vec3.new(-0.5, -0.5, -0.5), Vec3.new(0.5, 0.5, 0.5));

    var regular = Mesh{
        .name = "regular",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(0, 0, 10),
        .local_bounding_box = unit_box,
        .culling_strategy = .always_render,
        .material = blend,
    };

    var src_far = Mesh{
        .name = "src_far",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = unit_box,
    };
    var src_near = Mesh{
        .name = "src_near",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = unit_box,
    };
    var inst_far = InstancedMesh{
        .name = "far_inst",
        .source_mesh = &src_far,
        .position = Vec3.new(0, 0, 15),
    };
    var inst_near = InstancedMesh{
        .name = "near_inst",
        .source_mesh = &src_near,
        .position = Vec3.new(0, 0, 5),
    };
    var far_ptrs = [_]*InstancedMesh{&inst_far};
    var near_ptrs = [_]*InstancedMesh{&inst_near};
    var far_parent = Mesh{
        .name = "far_group",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .material = blend,
        .culling_strategy = .always_render,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &far_ptrs, .capacity = 1 },
    };
    var near_parent = Mesh{
        .name = "near_group",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .material = blend,
        .culling_strategy = .always_render,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &near_ptrs, .capacity = 1 },
    };

    var queues = RenderQueues{};
    defer queues.deinit(ally);
    var stats = SceneStats{};
    var culler = visibility.OcclusionCuller.init();
    const meshes = [_]*Mesh{ &regular, &far_parent, &near_parent };
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 11,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });

    // One regular transparent plus two instanced transparent groups share a
    // single order list; per-instance batching is preserved (one entry per
    // group, not per instance).
    try std.testing.expectEqual(@as(usize, 1), queues.transparent.items.len);
    try std.testing.expectEqual(@as(usize, 2), queues.transparent_instanced.items.len);
    try std.testing.expectEqual(@as(usize, 3), queues.transparent_order.items.len);
    try std.testing.expectEqual(@as(u32, 3), stats.rendered_meshes);
    // Material snapshot survives the queue: regular item kept the blend
    // record (transparent flag + draw_record), без живых указателей.
    try std.testing.expect(queues.transparent.items[0].transparent);
    try std.testing.expectEqual(@as(f32, 1.0), queues.transparent.items[0].draw_record.base_color[3]);

    std.mem.sort(TransparentDrawEntry, queues.transparent_order.items, {}, sortTransparentDrawOrder);
    const ordered = queues.transparent_order.items;
    // Global back-to-front: far group (15^2=225), regular (10^2=100), near (5^2=25).
    try std.testing.expectApproxEqAbs(@as(f32, 225.0), ordered[0].distance_sq, 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), ordered[1].distance_sq, 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), ordered[2].distance_sq, 1e-2);
    try std.testing.expectEqual(TransparentKind.instanced, ordered[0].kind);
    try std.testing.expectEqual(TransparentKind.regular, ordered[1].kind);
    try std.testing.expectEqual(TransparentKind.instanced, ordered[2].kind);
    // P4: батчи без живых указателей — принадлежность проверяем по seq
    // (mesh_index исходника): far_parent — meshes[1], near_parent — meshes[2].
    try std.testing.expectEqual(@as(u32, 1), ordered[0].seq);
    try std.testing.expectEqual(@as(u32, 0), ordered[1].seq);
    try std.testing.expectEqual(@as(u32, 0), ordered[1].index);
    try std.testing.expectEqual(@as(u32, 2), ordered[2].seq);

    // Deterministic tie-break: exactly equal distances keep seq order.
    var ties = [_]TransparentDrawEntry{
        .{ .distance_sq = 4.0, .seq = 7, .kind = .regular, .index = 0, .is_decal = false },
        .{ .distance_sq = 4.0, .seq = 3, .kind = .instanced, .index = 0, .is_decal = false },
    };
    std.mem.sort(TransparentDrawEntry, &ties, {}, sortTransparentDrawOrder);
    try std.testing.expectEqual(@as(u32, 3), ties[0].seq);
    try std.testing.expectEqual(@as(u32, 7), ties[1].seq);

    // Strict weak ordering: near-equal but distinct distances sort by exact
    // distance, never by tie-break. An epsilon band would be non-transitive
    // here (first≈second and second≈third within 1e-4, yet first≉third).
    var near = [_]TransparentDrawEntry{
        .{ .distance_sq = 1.0, .seq = 0, .kind = .regular, .index = 0, .is_decal = false },
        .{ .distance_sq = 1.0 + 5e-5, .seq = 1, .kind = .regular, .index = 1, .is_decal = false },
        .{ .distance_sq = 1.0 + 1e-4, .seq = 2, .kind = .regular, .index = 2, .is_decal = false },
    };
    std.mem.sort(TransparentDrawEntry, &near, {}, sortTransparentDrawOrder);
    try std.testing.expectEqual(@as(u32, 2), near[0].index);
    try std.testing.expectEqual(@as(u32, 1), near[1].index);
    try std.testing.expectEqual(@as(u32, 0), near[2].index);
}

test "parallel cull mixed scene matches serial on all queues" {
    const ally = std.testing.allocator;
    const material = @import("../../material.zig");

    var blend_mat = material.StandardMaterial.init("mixed_blend");
    blend_mat.alpha_mode = .blend;
    const blend: Material = .{ .standard = &blend_mat };
    const unit_box = BoundingBox.init(Vec3.new(-0.5, -0.5, -0.5), Vec3.new(0.5, 0.5, 0.5));

    var opaque_a = Mesh{
        .name = "opaque_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(1, 0, 0),
        .local_bounding_box = unit_box,
    };
    var trans_a = Mesh{
        .name = "trans_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(5, 0, 0),
        .local_bounding_box = unit_box,
        .material = blend,
    };
    var src_opaque = Mesh{
        .name = "src_opaque",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = unit_box,
    };
    var inst_op_0 = InstancedMesh{ .name = "op0", .source_mesh = &src_opaque, .position = Vec3.new(2, 0, 0) };
    var inst_op_1 = InstancedMesh{ .name = "op1", .source_mesh = &src_opaque, .position = Vec3.new(3, 0, 0) };
    var op_ptrs = [_]*InstancedMesh{ &inst_op_0, &inst_op_1 };
    var inst_opaque = Mesh{
        .name = "inst_opaque",
        // Сентинел-хендлы батча: вместо живых указателей принадлежность
        // доказывается снимками геометрии родителя.
        .vertex_buffer = .{ .id = 51 },
        .index_buffer = .{ .id = 52 },
        .index_count = 30,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &op_ptrs, .capacity = 2 },
    };
    var tie_regular = Mesh{
        .name = "tie_regular",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(7, 0, 0),
        .local_bounding_box = unit_box,
        .material = blend,
    };
    var src_tie = Mesh{
        .name = "src_tie",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = unit_box,
    };
    // Same center as tie_regular (7,0,0), same non-decal flag: an exact
    // distance tie between a regular and an instanced transparent.
    var inst_tie = InstancedMesh{ .name = "tie_inst", .source_mesh = &src_tie, .position = Vec3.new(7, 0, 0) };
    var tie_ptrs = [_]*InstancedMesh{&inst_tie};
    var inst_trans_tie = Mesh{
        .name = "inst_trans_tie",
        .vertex_buffer = .{ .id = 61 },
        .index_buffer = .{ .id = 62 },
        .index_count = 33,
        .material = blend,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &tie_ptrs, .capacity = 1 },
    };
    var decal_a = Mesh{
        .name = "decal_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(3, 0, 0),
        .local_bounding_box = unit_box,
        .is_decal = true,
    };
    var opaque_b = Mesh{
        .name = "opaque_b",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(9, 0, 0),
        .local_bounding_box = unit_box,
    };
    var src_far = Mesh{
        .name = "src_far",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = unit_box,
    };
    var inst_far = InstancedMesh{ .name = "far_inst", .source_mesh = &src_far, .position = Vec3.new(15, 0, 0) };
    var far_ptrs = [_]*InstancedMesh{&inst_far};
    var inst_trans_far = Mesh{
        .name = "inst_trans_far",
        .vertex_buffer = .{ .id = 71 },
        .index_buffer = .{ .id = 72 },
        .index_count = 36,
        .material = blend,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &far_ptrs, .capacity = 1 },
    };

    const meshes = [_]*Mesh{ &opaque_a, &trans_a, &inst_opaque, &tie_regular, &inst_trans_tie, &decal_a, &opaque_b, &inst_trans_far };

    var culler = visibility.OcclusionCuller.init();
    var stats_a = SceneStats{};
    var queues_a = RenderQueues{};
    defer queues_a.deinit(ally);
    var stats_b = SceneStats{};
    var queues_b = RenderQueues{};
    defer queues_b.deinit(ally);

    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();

    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 21,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_a,
        .queues = &queues_a,
        .default_white_id = 1,
    });
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 22,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_b,
        .queues = &queues_b,
        .default_white_id = 1,
        .thread_pool = pool,
        .parallel_min_meshes = 1,
    });

    // No double-queueing of instanced meshes: identical stat totals.
    try std.testing.expectEqual(stats_a.total_meshes, stats_b.total_meshes);
    try std.testing.expectEqual(stats_a.rendered_meshes, stats_b.rendered_meshes);
    try std.testing.expectEqual(stats_a.culled_meshes, stats_b.culled_meshes);
    try std.testing.expectEqual(stats_a.occluded_meshes, stats_b.occluded_meshes);
    try std.testing.expect(stats_a.rendered_meshes > 0);

    // Opaque regulars: identical records in identical order.
    try std.testing.expectEqual(queues_a.items.items.len, queues_b.items.items.len);
    try std.testing.expectEqual(@as(usize, 2), queues_a.items.items.len);
    for (queues_a.items.items, queues_b.items.items) |a, b| {
        try std.testing.expectEqual(a.model, b.model);
        try std.testing.expectEqual(a.distance_sq, b.distance_sq);
        try std.testing.expectEqual(a.texture_id, b.texture_id);
        try std.testing.expectEqual(a.transparent, b.transparent);
        try std.testing.expectEqual(a.is_decal, b.is_decal);
        try std.testing.expectEqual(a.mesh_index, b.mesh_index);
    }

    // Transparent regulars (incl. the decal): identical records in order.
    try std.testing.expectEqual(queues_a.transparent.items.len, queues_b.transparent.items.len);
    try std.testing.expectEqual(@as(usize, 3), queues_a.transparent.items.len);
    for (queues_a.transparent.items, queues_b.transparent.items) |a, b| {
        try std.testing.expectEqual(a.model, b.model);
        try std.testing.expectEqual(a.distance_sq, b.distance_sq);
        try std.testing.expectEqual(a.is_decal, b.is_decal);
        try std.testing.expectEqual(a.mesh_index, b.mesh_index);
    }

    // Instanced groups: submitted once each, in mesh order, on both paths.
    // P4: принадлежность и порядок — по сентинел-снимкам геометрии
    // (inst_opaque → 51/52/30, tie → 61/62/33, far → 71/72/36), без указателей.
    try std.testing.expectEqual(queues_a.opaque_instanced.items.len, queues_b.opaque_instanced.items.len);
    try std.testing.expectEqual(@as(usize, 1), queues_a.opaque_instanced.items.len);
    for ([2]*const RenderQueues{ &queues_a, &queues_b }) |qs| {
        const b = qs.opaque_instanced.items[0];
        try std.testing.expectEqual(@as(u32, 51), b.vertex_buffer.id);
        try std.testing.expectEqual(@as(u32, 52), b.index_buffer.id);
        try std.testing.expectEqual(@as(u32, 30), b.index_count);
        try std.testing.expectEqual(@as(usize, 2), qs.transparent_instanced.items.len);
        const t0 = qs.transparent_instanced.items[0];
        try std.testing.expectEqual(@as(u32, 61), t0.vertex_buffer.id);
        try std.testing.expectEqual(@as(u32, 62), t0.index_buffer.id);
        try std.testing.expectEqual(@as(u32, 33), t0.index_count);
        try std.testing.expect(t0.transparent);
        const t1 = qs.transparent_instanced.items[1];
        try std.testing.expectEqual(@as(u32, 71), t1.vertex_buffer.id);
        try std.testing.expectEqual(@as(u32, 72), t1.index_buffer.id);
        try std.testing.expectEqual(@as(u32, 36), t1.index_count);
        try std.testing.expect(t1.transparent);
    }

    // Staged instance matrices: identical contents.
    try std.testing.expectEqual(queues_a.instance_matrices.items.len, queues_b.instance_matrices.items.len);
    try std.testing.expect(queues_a.instance_matrices.items.len > 0);
    for (queues_a.instance_matrices.items, queues_b.instance_matrices.items) |a, b| {
        try std.testing.expectEqual(a, b);
    }

    // Unified transparent order: raw insertion differs (parallel merges all
    // regulars before the instanced tail), so compare the sorted order —
    // the contract both paths must honor.
    try std.testing.expectEqual(queues_a.transparent_order.items.len, queues_b.transparent_order.items.len);
    try std.testing.expectEqual(@as(usize, 5), queues_a.transparent_order.items.len);
    std.mem.sort(TransparentDrawEntry, queues_a.transparent_order.items, {}, sortTransparentDrawOrder);
    std.mem.sort(TransparentDrawEntry, queues_b.transparent_order.items, {}, sortTransparentDrawOrder);
    for (queues_a.transparent_order.items, queues_b.transparent_order.items) |a, b| {
        try std.testing.expectEqual(a.distance_sq, b.distance_sq);
        try std.testing.expectEqual(a.seq, b.seq);
        try std.testing.expectEqual(a.kind, b.kind);
        try std.testing.expectEqual(a.index, b.index);
        try std.testing.expectEqual(a.is_decal, b.is_decal);
        if (a.kind == .regular) {
            try std.testing.expectEqual(
                queues_a.transparent.items[a.index].mesh_index,
                queues_b.transparent.items[b.index].mesh_index,
            );
            try std.testing.expectEqual(
                queues_a.transparent.items[a.index].model,
                queues_b.transparent.items[b.index].model,
            );
        } else {
            try std.testing.expectEqual(
                queues_a.transparent_instanced.items[a.index].visible_instance_count,
                queues_b.transparent_instanced.items[b.index].visible_instance_count,
            );
            try std.testing.expectEqual(
                queues_a.transparent_instanced.items[a.index].index_count,
                queues_b.transparent_instanced.items[b.index].index_count,
            );
        }
    }

    // Exact-distance tie (regular mesh 3 + instanced group 4 at 7^2 = 49):
    // the entries are bit-identical distances and mesh-index order wins on
    // both paths. Резолв order-записи ведёт ровно в tie-батч (61/62/33).
    const ordered = queues_a.transparent_order.items;
    try std.testing.expect(ordered[1].distance_sq == ordered[2].distance_sq);
    try std.testing.expectEqual(@as(f32, 49.0), ordered[1].distance_sq);
    try std.testing.expectEqual(TransparentKind.regular, ordered[1].kind);
    try std.testing.expectEqual(TransparentKind.instanced, ordered[2].kind);
    try std.testing.expectEqual(@as(u32, 3), ordered[1].seq);
    try std.testing.expectEqual(@as(u32, 4), ordered[2].seq);
    const tie_batch = queues_a.transparent_instanced.items[ordered[2].index];
    try std.testing.expectEqual(@as(u32, 61), tie_batch.vertex_buffer.id);
    try std.testing.expectEqual(@as(u32, 33), tie_batch.index_count);
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

/// Test allocator that fails the first N allocations, then delegates to
/// backing. Unlike FailingAllocator (which fails persistently at its index),
/// a single failure lets the serial fallback that follows a parallel setup
/// OOM allocate normally.
const FailFirstN = struct {
    backing: std.mem.Allocator,
    failures_left: usize,

    fn allocator(self: *FailFirstN) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = allocFn,
                .resize = resizeFn,
                .remap = remapFn,
                .free = freeFn,
            },
        };
    }

    fn allocFn(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *FailFirstN = @ptrCast(@alignCast(ctx));
        if (self.failures_left > 0) {
            self.failures_left -= 1;
            return null;
        }
        return self.backing.rawAlloc(len, alignment, ra);
    }

    fn resizeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) bool {
        const self: *FailFirstN = @ptrCast(@alignCast(ctx));
        return self.backing.rawResize(memory, alignment, new_len, ra);
    }

    fn remapFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) ?[*]u8 {
        const self: *FailFirstN = @ptrCast(@alignCast(ctx));
        return self.backing.rawRemap(memory, alignment, new_len, ra);
    }

    fn freeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const self: *FailFirstN = @ptrCast(@alignCast(ctx));
        return self.backing.rawFree(memory, alignment, ra);
    }
};

test "parallel setup OOM falls back to serial queues" {
    const ally = std.testing.allocator;
    const material = @import("../../material.zig");

    var blend_mat = material.StandardMaterial.init("fallback_blend");
    blend_mat.alpha_mode = .blend;
    const blend: Material = .{ .standard = &blend_mat };
    const unit_box = BoundingBox.init(Vec3.new(-0.5, -0.5, -0.5), Vec3.new(0.5, 0.5, 0.5));

    var plain_a = Mesh{
        .name = "plain_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(1, 0, 0),
        .local_bounding_box = unit_box,
    };
    var plain_b = Mesh{
        .name = "plain_b",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(5, 0, 0),
        .local_bounding_box = unit_box,
        .material = blend,
    };
    var src = Mesh{
        .name = "src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = unit_box,
    };
    var inst = InstancedMesh{ .name = "i", .source_mesh = &src, .position = Vec3.new(2, 0, 0) };
    var inst_ptrs = [_]*InstancedMesh{&inst};
    var inst_parent = Mesh{
        .name = "inst_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &inst_ptrs, .capacity = 1 },
    };
    const meshes = [_]*Mesh{ &plain_a, &plain_b, &inst_parent };

    var culler = visibility.OcclusionCuller.init();
    var stats_a = SceneStats{};
    var queues_a = RenderQueues{};
    defer queues_a.deinit(ally);
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 31,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_a,
        .queues = &queues_a,
        .default_white_id = 1,
    });

    // One failed allocation — the parallel records array — then the serial
    // fallback allocates normally through the same allocator.
    var limited = FailFirstN{ .backing = ally, .failures_left = 1 };
    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();
    var stats_b = SceneStats{};
    var queues_b = RenderQueues{};
    defer queues_b.deinit(limited.allocator());
    buildFrameQueues(.{
        .allocator = limited.allocator(),
        .meshes = &meshes,
        .cache_key = 32,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_b,
        .queues = &queues_b,
        .default_white_id = 1,
        .thread_pool = pool,
        .parallel_min_meshes = 1,
    });

    // The frame still drew: serial-identical queues, not empty ones.
    try std.testing.expect(stats_b.rendered_meshes > 0);
    try std.testing.expectEqual(stats_a.total_meshes, stats_b.total_meshes);
    try std.testing.expectEqual(stats_a.rendered_meshes, stats_b.rendered_meshes);
    try std.testing.expectEqual(stats_a.culled_meshes, stats_b.culled_meshes);
    try std.testing.expectEqual(queues_a.items.items.len, queues_b.items.items.len);
    try std.testing.expectEqual(queues_a.transparent.items.len, queues_b.transparent.items.len);
    try std.testing.expectEqual(queues_a.opaque_instanced.items.len, queues_b.opaque_instanced.items.len);
    try std.testing.expectEqual(queues_a.transparent_instanced.items.len, queues_b.transparent_instanced.items.len);
    try std.testing.expectEqual(queues_a.transparent_order.items.len, queues_b.transparent_order.items.len);
    for (queues_a.items.items, queues_b.items.items) |a, b| {
        try std.testing.expectEqual(a.mesh_index, b.mesh_index);
        try std.testing.expectEqual(a.model, b.model);
        try std.testing.expectEqual(a.distance_sq, b.distance_sq);
    }
    for (queues_a.transparent.items, queues_b.transparent.items) |a, b| {
        try std.testing.expectEqual(a.mesh_index, b.mesh_index);
        try std.testing.expectEqual(a.distance_sq, b.distance_sq);
    }
    for (queues_a.opaque_instanced.items, queues_b.opaque_instanced.items) |a, b| {
        try std.testing.expectEqual(a.visible_instance_count, b.visible_instance_count);
        try std.testing.expectEqual(a.index_count, b.index_count);
    }
    for (queues_a.transparent_order.items, queues_b.transparent_order.items) |a, b| {
        try std.testing.expectEqual(a.distance_sq, b.distance_sq);
        try std.testing.expectEqual(a.seq, b.seq);
        try std.testing.expectEqual(a.kind, b.kind);
        try std.testing.expectEqual(a.index, b.index);
    }
    try std.testing.expectEqual(queues_a.instance_matrices.items.len, queues_b.instance_matrices.items.len);
    for (queues_a.instance_matrices.items, queues_b.instance_matrices.items) |a, b| {
        try std.testing.expectEqual(a, b);
    }
}

test "transparent instanced mesh sorts its instance matrices strictly back-to-front" {
    const ally = std.testing.allocator;
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var mesh = Mesh{
        .name = "inst_sort_test",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 36,
        .index_type = .UINT16,
    };

    var trans_mat = StandardMaterial.init("trans_mat");
    trans_mat.alpha_mode = .blend;
    mesh.material = .{ .standard = &trans_mat };

    var inst0 = InstancedMesh{ .name = "i0", .source_mesh = &mesh, .position = Vec3.new(0, 0, 10) };
    var inst1 = InstancedMesh{ .name = "i1", .source_mesh = &mesh, .position = Vec3.new(0, 0, 30) };
    var inst2 = InstancedMesh{ .name = "i2", .source_mesh = &mesh, .position = Vec3.new(0, 0, 20) };
    var ptrs = [_]*InstancedMesh{ &inst0, &inst1, &inst2 };
    mesh.instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &ptrs, .capacity = 3 };

    var stats = SceneStats{};
    var culler = visibility.OcclusionCuller.init();

    const meshes = [_]*Mesh{&mesh};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 1,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });

    try std.testing.expectEqual(@as(usize, 3), queues.instance_matrices.items.len);
    // Back-to-front: z=30 (farthest), then z=20, then z=10 (nearest)
    try std.testing.expectEqual(@as(f32, 30.0), queues.instance_matrices.items[0].m[14]);
    try std.testing.expectEqual(@as(f32, 20.0), queues.instance_matrices.items[1].m[14]);
    try std.testing.expectEqual(@as(f32, 10.0), queues.instance_matrices.items[2].m[14]);

    // Opaque instanced mesh preserves original instance creation order
    var opaque_mat = StandardMaterial.init("opaque_mat");
    opaque_mat.alpha_mode = .@"opaque";
    mesh.material = .{ .standard = &opaque_mat };

    queues.reset();
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 2,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });

    try std.testing.expectEqual(@as(usize, 3), queues.instance_matrices.items.len);
    // Original insertion order: z=10, z=30, z=20
    try std.testing.expectEqual(@as(f32, 10.0), queues.instance_matrices.items[0].m[14]);
    try std.testing.expectEqual(@as(f32, 30.0), queues.instance_matrices.items[1].m[14]);
    try std.testing.expectEqual(@as(f32, 20.0), queues.instance_matrices.items[2].m[14]);
}

test "pre-staged instances feed buildFrameQueues batch" {
    const ally = std.testing.allocator;
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var src = Mesh{
        .name = "batch_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "inst0", .source_mesh = &src, .position = Vec3.new(0, 0, 0) };
    var inst1 = InstancedMesh{ .name = "inst1", .source_mesh = &src, .position = Vec3.new(5, 0, 0) };
    var inst2 = InstancedMesh{ .name = "inst2", .source_mesh = &src, .position = Vec3.new(10, 0, 0), .is_visible = false };
    var ptrs = [_]*InstancedMesh{ &inst0, &inst1, &inst2 };
    var parent = Mesh{
        .name = "batch_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &ptrs, .capacity = 3 },
    };
    const meshes = [_]*Mesh{&parent};
    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 9,
        .eye = Vec3.zero,
    }, &meshes);

    var stats = SceneStats{};
    var culler = visibility.OcclusionCuller.init();
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 9,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });

    try std.testing.expectEqual(@as(usize, 1), queues.opaque_instanced.items.len);
    try std.testing.expectEqual(@as(u32, 2), queues.opaque_instanced.items[0].visible_instance_count);
    try std.testing.expectEqual(@as(u32, 3), queues.opaque_instanced.items[0].index_count);
}

// Baseline bug this guards (live: candidate 19441 tris vs P4 baseline
// 105620): the pre-stage used to publish combined instance bounds into
// Mesh.cached_aabb, and the parallel cull's worldMatrixCached pre-warm of
// ALL meshes overwrote them with the tiny origin template — the frame guard
// then blocked any restage, so the offscreen group drew. With the isolated
// instance_render.bounds, a warmed origin cached_aabb must not move culling.
// Controlled identity-frustum proof: template box at the origin (inside),
// instances far outside (x=50) plus one inside-visible sentinel group, so a
// pass is impossible by merely dropping every instanced batch.
test "P5: warmed instance parents cull on staged bounds, serial and parallel" {
    const ally = std.testing.allocator;
    const origin_box = BoundingBox.init(Vec3.new(-0.5, -0.5, -0.5), Vec3.new(0.5, 0.5, 0.5));

    var src = Mesh{
        .name = "cull_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = origin_box,
    };
    var far_inst = InstancedMesh{ .name = "far", .source_mesh = &src, .position = Vec3.new(50, 0, 0) };
    var near_inst = InstancedMesh{ .name = "near", .source_mesh = &src, .position = Vec3.zero };
    var far_ptrs = [_]*InstancedMesh{&far_inst};
    var near_ptrs = [_]*InstancedMesh{&near_inst};
    var far_parent = Mesh{
        .name = "far_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = origin_box,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &far_ptrs, .capacity = 1 },
    };
    var near_parent = Mesh{
        .name = "near_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = origin_box,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &near_ptrs, .capacity = 1 },
    };
    // Sentinel GPU identity on the inside group (no sg context here, so the
    // upload half is skipped and staging preserves these untouched): the
    // surviving batch must carry exactly this handle.
    near_parent.instance_render.buffer = .{ .id = 77 };
    const meshes = [_]*Mesh{ &far_parent, &near_parent };

    // Scene-like pre-stage (frame 81): far bounds [49.5, 50.5], near bounds
    // [-0.5, 0.5], both count 1.
    var stage_queues = RenderQueues{};
    defer stage_queues.deinit(ally);
    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &stage_queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 81,
        .eye = Vec3.zero,
    }, &meshes);
    try std.testing.expectEqual(@as(u32, 1), far_parent.instance_render.count);
    try std.testing.expectEqual(@as(u32, 1), near_parent.instance_render.count);
    try std.testing.expectApproxEqAbs(@as(f32, 49.5), far_parent.instance_render.bounds.min.x, 1e-4);

    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();

    var serial_stats: ?SceneStats = null;
    var serial_batch: ?RenderInstancedBatch = null;
    var parallel_stats: ?SceneStats = null;
    var parallel_batch: ?RenderInstancedBatch = null;

    // Serial build (Scene-like: pre-stage definitive, identity frustum).
    {
        var queues = RenderQueues{};
        defer queues.deinit(ally);
        var stats = SceneStats{};
        var culler = visibility.OcclusionCuller.init();
        buildFrameQueues(.{
            .allocator = ally,
            .meshes = &meshes,
            .cache_key = 81,
            .view_proj = Mat4.identity,
            .eye = Vec3.zero,
            .cull_frustum = true,
            .cull_occlusion = false,
            .occlusion_culler = &culler,
            .stats = &stats,
            .queues = &queues,
            .default_white_id = 1,
            .instances_prepared = true,
        });
        try std.testing.expectEqual(@as(usize, 1), queues.opaque_instanced.items.len);
        serial_batch = queues.opaque_instanced.items[0];
        serial_stats = stats;
    }

    // Parallel build (parallel_min_meshes=1 forces the pre-warm path over
    // both instance parents on worker threads).
    {
        var queues = RenderQueues{};
        defer queues.deinit(ally);
        var stats = SceneStats{};
        var culler = visibility.OcclusionCuller.init();
        buildFrameQueues(.{
            .allocator = ally,
            .thread_pool = pool,
            .parallel_min_meshes = 1,
            .meshes = &meshes,
            .cache_key = 81,
            .view_proj = Mat4.identity,
            .eye = Vec3.zero,
            .cull_frustum = true,
            .cull_occlusion = false,
            .occlusion_culler = &culler,
            .stats = &stats,
            .queues = &queues,
            .default_white_id = 1,
            .instances_prepared = true,
        });
        try std.testing.expectEqual(@as(usize, 1), queues.opaque_instanced.items.len);
        parallel_batch = queues.opaque_instanced.items[0];
        parallel_stats = stats;
    }

    // The surviving batch is the inside sentinel group in both paths —
    // not an empty queue, and identical across paths.
    const sb = serial_batch orelse return error.TestUnexpectedResult;
    const pb = parallel_batch orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u32, 1), sb.visible_instance_count);
    try std.testing.expectEqual(@as(u32, 77), sb.instance_buffer.id);
    try std.testing.expectEqual(sb.visible_instance_count, pb.visible_instance_count);
    try std.testing.expectEqual(sb.instance_buffer.id, pb.instance_buffer.id);
    try std.testing.expectEqual(sb.index_count, pb.index_count);

    // Cull stats: the far group (1 instance) omitted, the near group (1
    // instance) drawn — equal in both paths.
    const ss = serial_stats orelse return error.TestUnexpectedResult;
    const ps = parallel_stats orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u32, 1), ss.culled_meshes);
    try std.testing.expectEqual(@as(u32, 1), ss.total_meshes);
    try std.testing.expectEqual(@as(u32, 1), ss.rendered_meshes);
    try std.testing.expectEqual(ss.culled_meshes, ps.culled_meshes);
    try std.testing.expectEqual(ss.total_meshes, ps.total_meshes);
    try std.testing.expectEqual(ss.rendered_meshes, ps.rendered_meshes);

    // The crux: the parallel pre-warm rewrote the regular cached_aabb to
    // the origin template, yet the staged far bounds survived for culling
    // and no restage happened (frame still 81, count still 1, not 0).
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), far_parent.cached_aabb.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), far_parent.cached_aabb.max.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 49.5), far_parent.instance_render.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 50.5), far_parent.instance_render.bounds.max.x, 1e-4);
    try std.testing.expectEqual(@as(u32, 1), far_parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 81), far_parent.instance_render.staged_frame);
}

// ---- P4 render-owned draw snapshot: регрессия владения. ----

// Очереди не хранят живых указателей: скриншот пережил мутацию TRS/
// материала и две публикации скелета, перезаписавшие исходный слот.
test "P4: queued snapshot survives source mutation and skeleton republication" {
    const ally = std.testing.allocator;
    const Skeleton = skeleton_mod.Skeleton;

    var std_mat = material_mod.StandardMaterial.init("snap");
    std_mat.diffuse_color = math.Color3.new(0.2, 0.4, 0.6);
    const mat: Material = .{ .standard = &std_mat };

    var blend_mat = material_mod.StandardMaterial.init("snap_blend");
    blend_mat.alpha_mode = .blend;
    blend_mat.diffuse_color = math.Color3.new(0.1, 0.2, 0.3);
    const blend: Material = .{ .standard = &blend_mat };

    const skel = try Skeleton.init(ally, 1);
    defer skel.deinit();
    skel.bones[0].local_position = Vec3.new(1, 0, 0);
    skel.update();

    const unit_box = BoundingBox.init(Vec3.new(-0.5, -0.5, -0.5), Vec3.new(0.5, 0.5, 0.5));
    var opaque_mesh = Mesh{
        .name = "skinned_opaque",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(10, 0, 0),
        .local_bounding_box = unit_box,
        .culling_strategy = .always_render,
        .material = mat,
        .skeleton = skel,
    };
    var trans_mesh = Mesh{
        .name = "skinned_trans",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(0, 0, 10),
        .local_bounding_box = unit_box,
        .culling_strategy = .always_render,
        .material = blend,
        .skeleton = skel,
    };

    var queues = RenderQueues{};
    defer queues.deinit(ally);
    var stats = SceneStats{};
    var culler = visibility.OcclusionCuller.init();
    const meshes = [_]*Mesh{ &opaque_mesh, &trans_mesh };
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 7,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });

    try std.testing.expectEqual(@as(usize, 1), queues.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), queues.transparent.items.len);
    try std.testing.expectEqual(@as(usize, 2), queues.skin_storage.items.len);

    const oq = queues.items.items[0];
    const tq = queues.transparent.items[0];
    try std.testing.expect(oq.skin_index != null and tq.skin_index != null);

    // Мутация источников: TRS, материал, две публикации скелета (вторая
    // перезаписывает исходный слот новым значением x=5).
    opaque_mesh.position = Vec3.new(99, 99, 99);
    trans_mesh.position = Vec3.new(99, 99, 99);
    std_mat.diffuse_color = math.Color3.new(9, 9, 9);
    blend_mat.diffuse_color = math.Color3.new(9, 9, 9);
    skel.bones[0].local_position = Vec3.new(5, 0, 0);
    skel.update();
    skel.update();
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), skel.getRenderSkinMatrices()[0].m[12], 1e-4);

    // Снимки неизменны: модель, draw_record и копии скинов.
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), queues.items.items[0].model.m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), queues.items.items[0].draw_record.base_color[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), queues.transparent.items[0].draw_record.base_color[2], 1e-6);
    for ([_]RenderMeshItem{ queues.items.items[0], queues.transparent.items[0] }) |it| {
        const bones = queues.skin_storage.items[it.skin_index.?];
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), bones[0].m[12], 1e-4);
    }
}

// Instanced-запись тоже snapshot: мутация материала/TRS после prepare
// не меняет draw_record батча.
test "P4: instanced batch record survives source mutation" {
    const ally = std.testing.allocator;

    var inst_mat = material_mod.StandardMaterial.init("inst_snap");
    inst_mat.diffuse_color = math.Color3.new(0.5, 0.25, 0.125);
    const unit_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1));
    var src = Mesh{
        .name = "inst_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = unit_box,
    };
    var mesh = Mesh{
        .name = "inst_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 36,
        .index_type = .UINT16,
        .material = .{ .standard = &inst_mat },
    };
    var inst0 = InstancedMesh{ .name = "i0", .source_mesh = &src, .position = Vec3.new(0, 0, 10) };
    var ptrs = [_]*InstancedMesh{&inst0};
    mesh.instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &ptrs, .capacity = 1 };

    var queues = RenderQueues{};
    defer queues.deinit(ally);
    var stats = SceneStats{};
    var culler = visibility.OcclusionCuller.init();
    const meshes = [_]*Mesh{&mesh};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 3,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });
    try std.testing.expectEqual(@as(usize, 1), queues.opaque_instanced.items.len);

    inst_mat.diffuse_color = math.Color3.new(9, 9, 9);
    mesh.position = Vec3.new(99, 0, 0);
    const batch = queues.opaque_instanced.items[0];
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), batch.draw_record.base_color[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.125), batch.draw_record.base_color[2], 1e-6);
}

// Копии скинов масштабируются числом skinned-draws (а не MAX_BONES на item),
// индексы стабильны при росте хранилища и независимы между видами.
test "P4: skin storage scales with skinned draws and stays stable across growth" {
    const ally = std.testing.allocator;
    const Skeleton = skeleton_mod.Skeleton;
    const count = 160;

    const skels = try ally.alloc(*Skeleton, count);
    defer ally.free(skels);
    const meshes_arr = try ally.alloc(Mesh, count);
    defer ally.free(meshes_arr);
    const ptrs = try ally.alloc(*Mesh, count);
    defer ally.free(ptrs);
    for (0..count) |i| {
        const sk = try Skeleton.init(ally, 1);
        skels[i] = sk;
        sk.bones[0].local_position = Vec3.new(@floatFromInt(i), 0, 0);
        sk.update();
        meshes_arr[i] = .{
            .name = "sk",
            .vertex_buffer = .{},
            .index_buffer = .{},
            .index_count = 3,
            .culling_strategy = .always_render,
            .skeleton = sk,
        };
        ptrs[i] = &meshes_arr[i];
    }
    defer for (skels) |sk| sk.deinit();

    var culler = visibility.OcclusionCuller.init();
    var qa = RenderQueues{};
    defer qa.deinit(ally);
    var qb = RenderQueues{};
    defer qb.deinit(ally);

    // Два вида (multi-camera): у каждой очереди своё хранилище.
    var sa = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = ptrs,
        .cache_key = 11,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &sa,
        .queues = &qa,
        .default_white_id = 1,
    });
    var sb = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = ptrs,
        .cache_key = 11,
        .view_proj = Mat4.mul(Mat4.identity, Mat4.translation(Vec3.new(0, 0, 5))),
        .eye = Vec3.new(0, 0, 5),
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &sb,
        .queues = &qb,
        .default_white_id = 1,
    });

    try std.testing.expectEqual(@as(usize, count), qa.items.items.len);
    try std.testing.expectEqual(@as(usize, count), qa.skin_storage.items.len);
    try std.testing.expectEqual(@as(usize, count), qb.skin_storage.items.len);
    // Хранилище пережило несколько реаллокаций: каждый индекс резолвится
    // в копию своего скелета (x == номер меша).
    for (qa.items.items, qb.items.items) |a, b| {
        const ea: f32 = @floatFromInt(a.mesh_index);
        const eb: f32 = @floatFromInt(b.mesh_index);
        try std.testing.expectApproxEqAbs(ea, qa.skin_storage.items[a.skin_index.?][0].m[12], 1e-4);
        try std.testing.expectApproxEqAbs(eb, qb.skin_storage.items[b.skin_index.?][0].m[12], 1e-4);
    }

    // Мутация всех скелетов (x=1000+i, две публикации) — снимки целы.
    for (skels, 0..) |sk, i| {
        sk.bones[0].local_position = Vec3.new(1000.0 + @as(f32, @floatFromInt(i)), 0, 0);
        sk.update();
        sk.update();
    }
    for (qa.items.items) |a| {
        const ea: f32 = @floatFromInt(a.mesh_index);
        try std.testing.expectApproxEqAbs(ea, qa.skin_storage.items[a.skin_index.?][0].m[12], 1e-4);
    }

    // Reset/reuse: ёмкости retained, содержимое пересобрано корректно.
    const cap = qa.skin_storage.capacity;
    try std.testing.expect(cap >= count);
    qa.reset();
    try std.testing.expectEqual(@as(usize, 0), qa.items.items.len);
    try std.testing.expectEqual(@as(usize, 0), qa.skin_storage.items.len);
    try std.testing.expectEqual(cap, qa.skin_storage.capacity);
    var sa2 = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = ptrs,
        .cache_key = 12,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &sa2,
        .queues = &qa,
        .default_white_id = 1,
    });
    try std.testing.expectEqual(@as(usize, count), qa.items.items.len);
    for (qa.items.items) |a| {
        const ea: f32 = 1000.0 + @as(f32, @floatFromInt(a.mesh_index));
        try std.testing.expectApproxEqAbs(ea, qa.skin_storage.items[a.skin_index.?][0].m[12], 1e-4);
    }
}

// Фиксированная цена item не содержит MAX_BONES-матриц; пустые хранилища
// ничего не стоят, когда skinned/shader-draws отсутствуют.
test "P4: no fixed huge per-item skin cost" {
    try std.testing.expect(@sizeOf(RenderMeshItem) < 1024);
    try std.testing.expect(@sizeOf(RenderInstancedBatch) < 1024);
    // Один MAX_BONES-слот — 4 КиБ: item обязан быть кратно меньше.
    try std.testing.expect(@sizeOf(RenderMeshItem) * 8 < @sizeOf([MAX_BONES]Mat4));

    const ally = std.testing.allocator;
    var mesh = Mesh{
        .name = "plain",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .culling_strategy = .always_render,
    };
    var queues = RenderQueues{};
    defer queues.deinit(ally);
    var stats = SceneStats{};
    var culler = visibility.OcclusionCuller.init();
    const meshes = [_]*Mesh{&mesh};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 1,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });
    try std.testing.expectEqual(@as(usize, 1), queues.items.items.len);
    try std.testing.expectEqual(@as(usize, 0), queues.skin_storage.items.len);
    try std.testing.expectEqual(@as(usize, 0), queues.shader_storage.items.len);
    try std.testing.expect(queues.items.items[0].skin_index == null);
    try std.testing.expect(queues.items.items[0].shader_index == null);
}

fn expectP4QueueRefsValid(queues: *const RenderQueues) !void {
    for (queues.items.items) |it| {
        if (it.is_skinned) {
            try std.testing.expect(it.skin_index != null);
            try std.testing.expect(it.skin_index.? < queues.skin_storage.items.len);
        } else {
            try std.testing.expect(it.skin_index == null);
        }
        if (it.shader_index) |s| try std.testing.expect(s < queues.shader_storage.items.len);
    }
    for (queues.transparent.items) |it| {
        if (it.is_skinned) {
            try std.testing.expect(it.skin_index != null);
            try std.testing.expect(it.skin_index.? < queues.skin_storage.items.len);
        } else {
            try std.testing.expect(it.skin_index == null);
        }
        if (it.shader_index) |s| try std.testing.expect(s < queues.shader_storage.items.len);
    }
    // Нет orphan-ссылок порядка: каждая запись указывает в существующий слот.
    for (queues.transparent_order.items) |e| {
        if (e.kind == .regular) {
            try std.testing.expect(e.index < queues.transparent.items.len);
        } else {
            try std.testing.expect(e.index < queues.transparent_instanced.items.len);
        }
    }
}

// OOM в любой точке prepare-фазы: item либо целиком в очереди с валидными
// индексами, либо отсутствует. Тихого отката к живым матрицам нет.
// std.testing.FailingAllocator роняет ровно n-ю аллокацию при прочих успешных —
// так достигаются и поздние отказы (transparent/skin/shader) после ранних успехов.
test "P4: OOM never leaves items with dangling or live skin refs" {
    const ally = std.testing.allocator;
    const Skeleton = skeleton_mod.Skeleton;

    var std_mat = material_mod.StandardMaterial.init("oom");
    const mat: Material = .{ .standard = &std_mat };
    var blend_mat = material_mod.StandardMaterial.init("oom_blend");
    blend_mat.alpha_mode = .blend;
    const blend: Material = .{ .standard = &blend_mat };
    var hook_mat = material_mod.ShaderMaterial.init("oom_hook");
    const hook: Material = .{ .shader_material = &hook_mat };
    var hook_blend_mat = material_mod.ShaderMaterial.init("oom_hook_blend");
    hook_blend_mat.alpha_mode = .blend;
    const hook_blend: Material = .{ .shader_material = &hook_blend_mat };

    const skel = try Skeleton.init(ally, 1);
    defer skel.deinit();
    skel.bones[0].local_position = Vec3.new(2, 0, 0);
    skel.update();

    var skinned = Mesh{
        .name = "oom_skinned",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .culling_strategy = .always_render,
        .material = mat,
        .skeleton = skel,
    };
    var skinned_trans = Mesh{
        .name = "oom_skinned_trans",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .culling_strategy = .always_render,
        .material = blend,
        .skeleton = skel,
    };
    var plain = Mesh{
        .name = "oom_plain",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .culling_strategy = .always_render,
        .material = mat,
    };
    var hooked = Mesh{
        .name = "oom_hook",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .culling_strategy = .always_render,
        .material = hook,
    };
    var hooked_trans = Mesh{
        .name = "oom_hook_trans",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .culling_strategy = .always_render,
        .material = hook_blend,
    };
    const meshes = [_]*Mesh{ &skinned, &skinned_trans, &plain, &hooked, &hooked_trans };
    var culler = visibility.OcclusionCuller.init();

    var full = RenderQueues{};
    defer full.deinit(ally);
    var stats_full = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 1,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_full,
        .queues = &full,
        .default_white_id = 1,
    });
    try std.testing.expectEqual(@as(usize, 3), full.items.items.len);
    try std.testing.expectEqual(@as(usize, 2), full.transparent.items.len);
    try std.testing.expectEqual(@as(usize, 2), full.skin_storage.items.len);
    try std.testing.expectEqual(@as(usize, 2), full.shader_storage.items.len);
    try expectP4QueueRefsValid(&full);

    // Прогон по каждому n-му отказу отдельно: ранние успехи + поздний отказ.
    var saw_induced = false;
    var saw_partial = false;
    var n: usize = 0;
    while (n <= 40) : (n += 1) {
        var failing = std.testing.FailingAllocator.init(ally, .{ .fail_index = n });
        var q = RenderQueues{};
        defer q.deinit(failing.allocator());
        var st = SceneStats{};
        buildFrameQueues(.{
            .allocator = failing.allocator(),
            .meshes = &meshes,
            .cache_key = 2,
            .view_proj = Mat4.identity,
            .eye = Vec3.zero,
            .cull_frustum = false,
            .cull_occlusion = false,
            .occlusion_culler = &culler,
            .stats = &st,
            .queues = &q,
            .default_white_id = 1,
        });
        try std.testing.expect(q.items.items.len <= full.items.items.len);
        try std.testing.expect(q.transparent.items.len <= full.transparent.items.len);
        try expectP4QueueRefsValid(&q);
        if (failing.has_induced_failure) saw_induced = true;
        if (q.items.items.len < full.items.items.len or q.transparent.items.len < full.transparent.items.len) saw_partial = true;
    }
    // Хотя бы один отказ реально сработал и хотя бы один уронил item.
    try std.testing.expect(saw_induced);
    try std.testing.expect(saw_partial);
}

// Hook-снимки на уровне очередей: opaque/transparent записи несут точные копии
// tint/uniforms/texture/entry, мутация источника их не меняет; parallel-merge
// копирует shader-значения эквивалентно серийному пути.
test "P4: queue shader snapshots are exact and merge-equivalent" {
    const ally = std.testing.allocator;

    var hook_opaque = material_mod.ShaderMaterial.init("hook_opaque");
    hook_opaque.entry_index = 11;
    hook_opaque.tint_color = math.Color3.new(0.1, 0.2, 0.3);
    hook_opaque.alpha = 0.8;
    hook_opaque.texture = Texture{ .image = .{}, .view = .{ .id = 51 }, .sampler = .{ .id = 52 }, .width = 4, .height = 4 };
    hook_opaque.uniforms[0] = .{ 1, 2, 3, 4 };
    hook_opaque.uniforms[7] = .{ 5, 6, 7, 8 };

    var hook_blend = material_mod.ShaderMaterial.init("hook_blend");
    hook_blend.entry_index = 13;
    hook_blend.alpha_mode = .blend;
    hook_blend.tint_color = math.Color3.new(0.4, 0.5, 0.6);
    hook_blend.alpha = 0.7;
    hook_blend.texture = Texture{ .image = .{}, .view = .{ .id = 53 }, .sampler = .{ .id = 54 }, .width = 4, .height = 4 };
    hook_blend.uniforms[0] = .{ 9, 9, 9, 9 };

    var mo = Mesh{
        .name = "hook_opaque_mesh",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .culling_strategy = .always_render,
        .material = .{ .shader_material = &hook_opaque },
    };
    var mt = Mesh{
        .name = "hook_blend_mesh",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .culling_strategy = .always_render,
        .material = .{ .shader_material = &hook_blend },
    };
    const meshes = [_]*Mesh{ &mo, &mt };
    var culler = visibility.OcclusionCuller.init();

    var qs = RenderQueues{};
    defer qs.deinit(ally);
    var ss = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 41,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &ss,
        .queues = &qs,
        .default_white_id = 1,
    });
    try std.testing.expectEqual(@as(usize, 1), qs.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), qs.transparent.items.len);
    try std.testing.expectEqual(@as(usize, 2), qs.shader_storage.items.len);

    // Parallel-merge копирует shader-значения эквивалентно серийному пути
    // (те же снимки в том же порядке).
    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();
    var qp = RenderQueues{};
    defer qp.deinit(ally);
    var sp = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 42,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &sp,
        .queues = &qp,
        .default_white_id = 1,
        .thread_pool = pool,
        .parallel_min_meshes = 1,
    });
    try std.testing.expectEqual(qs.shader_storage.items.len, qp.shader_storage.items.len);
    for (qs.shader_storage.items, qp.shader_storage.items) |a, b| {
        try std.testing.expectEqual(a.entry_index, b.entry_index);
        try std.testing.expectEqual(a.tint, b.tint);
        try std.testing.expectEqual(a.tex_view.id, b.tex_view.id);
        try std.testing.expectEqual(a.tex_sampler.id, b.tex_sampler.id);
        try std.testing.expectEqual(a.uniforms, b.uniforms);
    }
    try std.testing.expectEqual(
        qs.items.items[0].shader_index,
        qp.items.items[0].shader_index,
    );
    try std.testing.expectEqual(
        qs.transparent.items[0].shader_index,
        qp.transparent.items[0].shader_index,
    );

    // Мутация источников после prepare: обе очереди хранят точные копии.
    hook_opaque.tint_color = math.Color3.new(9, 9, 9);
    hook_opaque.alpha = 0.0;
    hook_opaque.texture = null;
    hook_opaque.uniforms[0] = .{ 9, 9, 9, 9 };
    hook_opaque.entry_index = 99;
    hook_blend.tint_color = math.Color3.new(8, 8, 8);
    hook_blend.uniforms[0] = .{ 8, 8, 8, 8 };
    hook_blend.entry_index = 98;

    for ([2]*const RenderQueues{ &qs, &qp }) |qq| {
        const so = qq.shader_storage.items[qq.items.items[0].shader_index.?];
        try std.testing.expectEqual(@as(u32, 11), so.entry_index);
        try std.testing.expectEqual([4]f32{ 0.1, 0.2, 0.3, 0.8 }, so.tint);
        try std.testing.expectEqual(@as(u32, 51), so.tex_view.id);
        try std.testing.expectEqual(@as(u32, 52), so.tex_sampler.id);
        try std.testing.expectEqual([4]f32{ 1, 2, 3, 4 }, so.uniforms[0]);
        try std.testing.expectEqual([4]f32{ 5, 6, 7, 8 }, so.uniforms[7]);
        const s_t = qq.shader_storage.items[qq.transparent.items[0].shader_index.?];
        try std.testing.expectEqual(@as(u32, 13), s_t.entry_index);
        try std.testing.expectEqual([4]f32{ 0.4, 0.5, 0.6, 0.7 }, s_t.tint);
        try std.testing.expectEqual(@as(u32, 53), s_t.tex_view.id);
        try std.testing.expectEqual([4]f32{ 9, 9, 9, 9 }, s_t.uniforms[0]);
    }
}

// Hook-sidedness как до P4: draw-путь hook-материалов использует собственный
// double_sided снимка, а не item.double_sided (куда decal- meshes форсят true
// для regular-пути). Decal с single-sided hook-материалом: item — double-sided
// (regular-контракт), снимок — single-sided (hook-контракт).
test "P4: hook shader snapshot preserves material sidedness without decal forcing" {
    const ally = std.testing.allocator;

    var hook_single = material_mod.ShaderMaterial.init("hook_single");
    hook_single.double_sided = false;

    var decal_hook = Mesh{
        .name = "decal_hook",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .culling_strategy = .always_render,
        .material = .{ .shader_material = &hook_single },
        .is_decal = true,
    };
    const meshes = [_]*Mesh{&decal_hook};
    var culler = visibility.OcclusionCuller.init();
    var queues = RenderQueues{};
    defer queues.deinit(ally);
    var stats = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 43,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });

    // Decal едет в transparent-очередь с форсированным double_sided (regular).
    try std.testing.expectEqual(@as(usize, 1), queues.transparent.items.len);
    const it = queues.transparent.items[0];
    try std.testing.expect(it.transparent and it.is_decal and it.double_sided);
    // А hook-снимок — single-sided, как sm.double_sided материала.
    const snap = queues.shader_storage.items[it.shader_index.?];
    try std.testing.expect(!snap.double_sided);

    // Мутация материала после prepare снимок не меняет.
    hook_single.double_sided = true;
    try std.testing.expect(!queues.shader_storage.items[it.shader_index.?].double_sided);
}

// Серийный и параллельный пути дают идентичные очереди включая копии скинов:
// воркеры только заимствуют слоты в скретч, копии делаются серийно в merge.
test "P4: parallel cull matches serial on skinned snapshots" {
    const ally = std.testing.allocator;
    const Skeleton = skeleton_mod.Skeleton;
    const count = 40;

    const skels = try ally.alloc(*Skeleton, count);
    defer ally.free(skels);
    const meshes_arr = try ally.alloc(Mesh, count);
    defer ally.free(meshes_arr);
    const ptrs = try ally.alloc(*Mesh, count);
    defer ally.free(ptrs);
    for (0..count) |i| {
        const sk = try Skeleton.init(ally, 2);
        skels[i] = sk;
        sk.bones[0].local_position = Vec3.new(@floatFromInt(i), 0, 0);
        sk.bones[1].local_position = Vec3.new(0, @floatFromInt(i), 0);
        sk.update();
        meshes_arr[i] = .{
            .name = "psk",
            .vertex_buffer = .{},
            .index_buffer = .{},
            .index_count = 3,
            .position = Vec3.new(@as(f32, @floatFromInt(i)) * 0.05, 0, 0),
            .culling_strategy = .always_render,
            .skeleton = sk,
        };
        ptrs[i] = &meshes_arr[i];
    }
    defer for (skels) |sk| sk.deinit();

    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();

    var culler = visibility.OcclusionCuller.init();
    var qs = RenderQueues{};
    defer qs.deinit(ally);
    var qp = RenderQueues{};
    defer qp.deinit(ally);
    var ss = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = ptrs,
        .cache_key = 21,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &ss,
        .queues = &qs,
        .default_white_id = 1,
    });
    var sp = SceneStats{};
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = ptrs,
        .cache_key = 21,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &sp,
        .queues = &qp,
        .default_white_id = 1,
        .thread_pool = pool,
        .parallel_min_meshes = 1,
    });

    try std.testing.expectEqual(qs.items.items.len, qp.items.items.len);
    try std.testing.expectEqual(@as(usize, count), qs.items.items.len);
    for (qs.items.items, qp.items.items) |a, b| {
        try std.testing.expectEqual(a.mesh_index, b.mesh_index);
        try std.testing.expectEqual(a.model, b.model);
        try std.testing.expect(a.skin_index != null and b.skin_index != null);
        const sa = qs.skin_storage.items[a.skin_index.?];
        const sb = qp.skin_storage.items[b.skin_index.?];
        try std.testing.expectEqual(sa, sb);
        const ea: f32 = @floatFromInt(a.mesh_index);
        try std.testing.expectApproxEqAbs(ea, sa[0].m[12], 1e-4);
        try std.testing.expectApproxEqAbs(ea, sa[1].m[13], 1e-4);
    }
}

test "stage-2A: instanced batch carries source uid and list index" {
    const ally = std.testing.allocator;
    var src = Mesh{
        .name = "id_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var mem: [2]InstancedMesh = .{
        .{ .name = "i0", .source_mesh = &src, .position = Vec3.zero },
        .{ .name = "i1", .source_mesh = &src, .position = Vec3.new(2, 0, 0) },
    };
    var ptrs = [_]*InstancedMesh{ &mem[0], &mem[1] };
    var plain = Mesh{
        .name = "plain0",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
    };
    var parent = Mesh{
        .name = "inst1",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = &ptrs, .capacity = 2 },
    };
    const meshes = [_]*Mesh{ &plain, &parent };

    var queues = RenderQueues{};
    defer queues.deinit(ally);
    var stats = SceneStats{};
    var culler = visibility.OcclusionCuller.init();
    buildFrameQueues(.{
        .allocator = ally,
        .meshes = &meshes,
        .cache_key = 99,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats,
        .queues = &queues,
        .default_white_id = 1,
    });

    try std.testing.expectEqual(@as(usize, 1), queues.opaque_instanced.items.len);
    const b = queues.opaque_instanced.items[0];
    // source_mesh equals the mesh-list index at build time (parent is index 1).
    try std.testing.expectEqual(@as(u32, 1), b.source_mesh);
    try std.testing.expect(b.source_uid != 0);
    try std.testing.expectEqual(parent.uid, b.source_uid);
    try std.testing.expect(parent.uid != 0);
    try std.testing.expect(plain.uid != 0);
}
