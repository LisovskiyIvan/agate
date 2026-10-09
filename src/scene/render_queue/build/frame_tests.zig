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
const PBRMaterial = material_mod.PBRMaterial;
const jobs = @import("../../../jobs.zig");
const visibility = @import("../../../visibility/mod.zig");
const stats_mod = @import("../../stats.zig");
const SceneStats = stats_mod.SceneStats;
const instance_staging = @import("../../instance_staging.zig");
const items = @import("../items.zig");
const RenderQueues = items.RenderQueues;
const RenderInstancedBatch = items.RenderInstancedBatch;
const TransparentKind = items.TransparentKind;
const TransparentDrawEntry = items.TransparentDrawEntry;
const sortTransparentDrawOrder = items.sortTransparentDrawOrder;
const frame = @import("frame.zig");
const buildFrameQueues = frame.buildFrameQueues;

test "shared LOD mesh preserves entity transforms without mutation" {
    const ally = std.testing.allocator;

    // Fully initialized shared LOD child: sentinel handles distinguish it from parents.
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

    // Sentinel handles confirm selection of shared LOD child geometry across distinct world matrices.
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

test "stale AABB does not drive LOD selection" {
    const ally = std.testing.allocator;
    const unit_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1));

    var lod_far = Mesh{
        .name = "lod_far",
        .vertex_buffer = .{},
        .index_buffer = .{},
        // Distinct index count to identify the far LOD child.
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

    // Seed a stale AABB tagged with an older frame. Fresh AABB must pick the far child.
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

test "transparent regular+instanced groups share one back-to-front order" {
    const ally = std.testing.allocator;
    const material = @import("../../../material.zig");

    var blend_mat = material.PBRMaterial.init("blend");
    blend_mat.alpha_mode = .blend;
    const blend: Material = .{ .pbr = &blend_mat };
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
    try std.testing.expectEqual(@as(u32, 5), stats.rendered_meshes);
    // Material snapshot survives the queue.
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
    // Batches verified by seq: far_parent is meshes[1], near_parent is meshes[2].
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

    // Strict weak ordering: near-equal but distinct distances sort by exact distance.
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

    var trans_mat = PBRMaterial.init("trans_mat");
    trans_mat.alpha_mode = .blend;
    mesh.material = .{ .pbr = &trans_mat };

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

    try std.testing.expectEqual(@as(usize, 4), queues.instance_matrices.items.len);
    // Back-to-front: z=30 (farthest), then z=20, z=10, and the source mesh
    // entry (its own transform, z=0) sorted in with them as the nearest.
    try std.testing.expectEqual(@as(f32, 30.0), queues.instance_matrices.items[0].m[14]);
    try std.testing.expectEqual(@as(f32, 20.0), queues.instance_matrices.items[1].m[14]);
    try std.testing.expectEqual(@as(f32, 10.0), queues.instance_matrices.items[2].m[14]);
    try std.testing.expectEqual(@as(f32, 0.0), queues.instance_matrices.items[3].m[14]);

    // Opaque instanced mesh preserves original instance creation order
    var opaque_mat = PBRMaterial.init("opaque_mat");
    opaque_mat.alpha_mode = .@"opaque";
    mesh.material = .{ .pbr = &opaque_mat };

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

    try std.testing.expectEqual(@as(usize, 4), queues.instance_matrices.items.len);
    // Original insertion order: the source mesh entry first (own transform,
    // z=0), then z=10, z=30, z=20.
    try std.testing.expectEqual(@as(f32, 0.0), queues.instance_matrices.items[0].m[14]);
    try std.testing.expectEqual(@as(f32, 10.0), queues.instance_matrices.items[1].m[14]);
    try std.testing.expectEqual(@as(f32, 30.0), queues.instance_matrices.items[2].m[14]);
    try std.testing.expectEqual(@as(f32, 20.0), queues.instance_matrices.items[3].m[14]);
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
    try std.testing.expectEqual(@as(u32, 3), queues.opaque_instanced.items[0].visible_instance_count);
    try std.testing.expectEqual(@as(u32, 3), queues.opaque_instanced.items[0].index_count);
}

test "warmed instance parents cull on staged bounds, serial and parallel" {
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
        .is_visible = false,
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
    near_parent.instance_render.buffer = .{ .id = 77 };
    const meshes = [_]*Mesh{ &far_parent, &near_parent };

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
    try std.testing.expectEqual(@as(u32, 2), near_parent.instance_render.count);
    try std.testing.expectApproxEqAbs(@as(f32, 49.5), far_parent.instance_render.bounds.min.x, 1e-4);

    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();

    var serial_stats: ?SceneStats = null;
    var serial_batch: ?RenderInstancedBatch = null;
    var parallel_stats: ?SceneStats = null;
    var parallel_batch: ?RenderInstancedBatch = null;

    // Serial build
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

    // Parallel build
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

    const sb = serial_batch orelse return error.TestUnexpectedResult;
    const pb = parallel_batch orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u32, 2), sb.visible_instance_count);
    try std.testing.expectEqual(@as(u32, 77), sb.instance_buffer.id);
    try std.testing.expectEqual(sb.visible_instance_count, pb.visible_instance_count);
    try std.testing.expectEqual(sb.instance_buffer.id, pb.instance_buffer.id);
    try std.testing.expectEqual(sb.index_count, pb.index_count);

    const ss = serial_stats orelse return error.TestUnexpectedResult;
    const ps = parallel_stats orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u32, 1), ss.culled_meshes);
    try std.testing.expectEqual(@as(u32, 1), ss.total_meshes);
    try std.testing.expectEqual(@as(u32, 2), ss.rendered_meshes);
    try std.testing.expectEqual(ss.culled_meshes, ps.culled_meshes);
    try std.testing.expectEqual(ss.total_meshes, ps.total_meshes);
    try std.testing.expectEqual(ss.rendered_meshes, ps.rendered_meshes);

    try std.testing.expectApproxEqAbs(@as(f32, -0.5), far_parent.cached_aabb.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), far_parent.cached_aabb.max.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 49.5), far_parent.instance_render.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 50.5), far_parent.instance_render.bounds.max.x, 1e-4);
    try std.testing.expectEqual(@as(u32, 1), far_parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 81), far_parent.instance_render.staged_frame);
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
    try std.testing.expectEqual(@as(u32, 1), b.source_mesh);
    try std.testing.expect(b.source_uid != 0);
    try std.testing.expectEqual(parent.uid, b.source_uid);
    try std.testing.expect(parent.uid != 0);
    try std.testing.expect(plain.uid != 0);
}
