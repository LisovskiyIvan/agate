//! Serial/parallel equivalence integration tests: the parallel pass must
//! reproduce the serial loop's queues and stats exactly (deterministic
//! chunk-order merge, scratch reuse across frames, stale-AABB behavior,
//! mixed scenes, skinned snapshots) and parallel setup OOM must fall back
//! to the serial loop cleanly. Test-only leaf: owns `FailFirstN` plus the
//! equivalence test blocks, exercising `frame.buildFrameQueues`
//! (one-directional edge `equivalence` → `frame`; never the `build.zig`
//! facade). No production logic lives here, so the facade does not
//! re-export anything from it.
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
const frame = @import("frame.zig");
const buildFrameQueues = frame.buildFrameQueues;

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

test "parallel cull mixed scene matches serial on all queues" {
    const ally = std.testing.allocator;
    const material = @import("../../../material.zig");

    var blend_mat = material.PBRMaterial.init("mixed_blend");
    blend_mat.alpha_mode = .blend;
    const blend: Material = .{ .pbr = &blend_mat };
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
        // Batch sentinel handles: ownership verified through parent geometry snapshots.
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
    // Membership and ordering verified through sentinel geometry snapshots.
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
    // the entries are bit-identical distances and mesh-index order wins on both paths.
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
    const material = @import("../../../material.zig");

    var blend_mat = material.PBRMaterial.init("fallback_blend");
    blend_mat.alpha_mode = .blend;
    const blend: Material = .{ .pbr = &blend_mat };
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
