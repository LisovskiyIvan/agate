const std = @import("std");

const math = @import("math");
const Mat4 = math.Mat4;
const Frustum = math.Frustum;
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const Mesh = @import("../../mesh.zig").Mesh;
const InstancedMesh = @import("../../mesh.zig").InstancedMesh;
const jobs = @import("../../jobs.zig");
const visibility = @import("../../visibility/mod.zig");
const stats_mod = @import("../stats.zig");
const SceneStats = stats_mod.SceneStats;
const instance_staging = @import("../instance_staging.zig");
const items = @import("items.zig");
const RenderQueues = items.RenderQueues;
const cull = @import("cull.zig");
const FrameCullContext = cull.FrameCullContext;
const instances = @import("instances.zig");
const submitInstancedMesh = instances.submitInstancedMesh;

test "parallel instanced staging produces serial-identical instance matrices" {
    const ally = std.testing.allocator;
    var source_mesh = Mesh{
        .name = "source",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
        .base_matrix = Mat4.identity,
    };

    const count = 300;
    const inst_mem = try ally.alloc(InstancedMesh, count);
    defer ally.free(inst_mem);
    const inst_ptrs = try ally.alloc(*InstancedMesh, count);
    defer ally.free(inst_ptrs);

    for (0..count) |i| {
        const fi: f32 = @floatFromInt(i);
        inst_mem[i] = InstancedMesh{
            .name = "inst",
            .source_mesh = &source_mesh,
            .position = Vec3.new(fi * 0.1, fi * 0.2, fi * 0.3),
            .rotation = Vec3.new(fi * 0.01, fi * 0.02, 0),
            .scaling = Vec3.new(1, 1, 1),
            .is_visible = (i % 7 != 0),
        };
        inst_ptrs[i] = &inst_mem[i];
    }

    var mesh_serial = Mesh{
        .name = "inst_parent_s",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = inst_ptrs, .capacity = count },
    };

    // Serial staging
    var queues_s = RenderQueues{};
    defer queues_s.deinit(ally);
    var stats_s = SceneStats{};
    var culler = visibility.OcclusionCuller.init();
    const ctx_s = FrameCullContext{
        .allocator = ally,
        .thread_pool = null,
        .default_white_id = 1,
        .meshes = &.{},
        .cache_key = 1,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_s,
        .queues = &queues_s,
    };
    submitInstancedMesh(ctx_s, Frustum.fromViewProjection(Mat4.identity), &mesh_serial, 0);

    // Parallel staging
    const pool = try jobs.Pool.init(ally, 2);
    defer pool.deinit();

    var mesh_parallel = Mesh{
        .name = "inst_parent_p",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = inst_ptrs, .capacity = count },
    };
    var queues_p = RenderQueues{};
    defer queues_p.deinit(ally);
    var stats_p = SceneStats{};
    const ctx_p = FrameCullContext{
        .allocator = ally,
        .thread_pool = pool,
        .default_white_id = 1,
        .meshes = &.{},
        .cache_key = 2,
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .cull_frustum = false,
        .cull_occlusion = false,
        .occlusion_culler = &culler,
        .stats = &stats_p,
        .queues = &queues_p,
    };
    submitInstancedMesh(ctx_p, Frustum.fromViewProjection(Mat4.identity), &mesh_parallel, 0);

    try std.testing.expectEqual(mesh_serial.instance_render.bounds, mesh_parallel.instance_render.bounds);
    try std.testing.expectEqual(@as(u32, 258), mesh_serial.instance_render.count);
    try std.testing.expectEqual(@as(u32, 258), mesh_parallel.instance_render.count);
    try std.testing.expectEqual(@as(u64, 1), mesh_serial.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u64, 2), mesh_parallel.instance_render.staged_frame);
    try std.testing.expectEqual(queues_s.instance_matrices.items.len, queues_p.instance_matrices.items.len);
    try std.testing.expect(queues_s.instance_matrices.items.len > 0);
    for (queues_s.instance_matrices.items, queues_p.instance_matrices.items) |m_s, m_p| {
        try std.testing.expectEqual(m_s, m_p);
    }
}

test "stageInstances stages visible count and combined AABB" {
    const ally = std.testing.allocator;
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var src = Mesh{
        .name = "stage_src",
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
        .name = "stage_parent",
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
        .frame_id = 7,
        .eye = Vec3.zero,
    }, &meshes);

    try std.testing.expectEqual(@as(u32, 3), parent.instance_render.count);
    try std.testing.expect(parent.instance_render.bounds.isValid());
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), parent.instance_render.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), parent.instance_render.bounds.max.x, 1e-4);
}

test "stageInstances is idempotent within a frame" {
    const ally = std.testing.allocator;
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var src = Mesh{
        .name = "idem_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "inst0", .source_mesh = &src, .position = Vec3.new(0, 0, 0) };
    var inst1 = InstancedMesh{ .name = "inst1", .source_mesh = &src, .position = Vec3.new(5, 0, 0) };
    var ptrs = [_]*InstancedMesh{ &inst0, &inst1 };
    var parent = Mesh{
        .name = "idem_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &ptrs, .capacity = 2 },
    };
    const meshes = [_]*Mesh{&parent};
    const sc = instance_staging.InstanceStageContext{
        .allocator = ally,
        .instance_matrices = &queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 7,
        .eye = Vec3.zero,
    };
    instance_staging.stageInstances(sc, &meshes);
    const first_bounds = parent.instance_render.bounds;
    const first_count = parent.instance_render.count;
    const first_frame = parent.instance_render.staged_frame;

    inst0.position = Vec3.new(100, 0, 0);
    instance_staging.stageInstances(sc, &meshes);
    try std.testing.expectEqual(first_count, parent.instance_render.count);
    try std.testing.expectEqual(first_bounds, parent.instance_render.bounds);
    try std.testing.expectEqual(first_frame, parent.instance_render.staged_frame);
}

test "stageInstances skips gpu_pending meshes" {
    const ally = std.testing.allocator;
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var src = Mesh{
        .name = "pend_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "inst0", .source_mesh = &src, .position = Vec3.zero };
    var ptrs = [_]*InstancedMesh{&inst0};
    var parent = Mesh{
        .name = "pend_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .gpu_pending = true,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &ptrs, .capacity = 1 },
    };
    const meshes = [_]*Mesh{&parent};
    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 7,
        .eye = Vec3.zero,
    }, &meshes);

    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, std.math.maxInt(u64)), parent.instance_render.staged_frame);
}

test "P5: staging leaves source TRS and game caches untouched" {
    const ally = std.testing.allocator;
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var src = Mesh{
        .name = "pure_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .base_matrix = Mat4.translation(Vec3.new(10, 0, 0)),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
        .cached_matrix = Mat4.translation(Vec3.new(777, 0, 0)),
        .cached_aabb = BoundingBox.init(Vec3.new(777, 777, 777), Vec3.new(778, 778, 778)),
        .cached_frame = 4242,
    };
    var inst = InstancedMesh{
        .name = "pure_inst",
        .source_mesh = &src,
        .position = Vec3.new(3, 4, 5),
        .cached_world_matrix = Mat4.translation(Vec3.new(999, 0, 0)),
        .cached_bounding_box = BoundingBox.init(Vec3.new(999, 999, 999), Vec3.new(1000, 1000, 1000)),
        .last_position = Vec3.new(111, 0, 0),
        .last_rotation = Vec3.new(0, 222, 0),
        .last_scaling = Vec3.new(0, 0, 333),
        .dirty = true,
    };
    var ptrs = [_]*InstancedMesh{&inst};
    var parent = Mesh{
        .name = "pure_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .cached_matrix = Mat4.translation(Vec3.new(555, 0, 0)),
        .cached_aabb = BoundingBox.init(Vec3.new(555, 555, 555), Vec3.new(556, 556, 556)),
        .cached_frame = 8484,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &ptrs, .capacity = 1 },
    };
    const meshes = [_]*Mesh{&parent};
    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 11,
        .eye = Vec3.zero,
    }, &meshes);

    try std.testing.expectEqual(@as(u32, 2), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 11), parent.instance_render.staged_frame);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), parent.instance_render.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 14.0), parent.instance_render.bounds.max.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), parent.instance_render.bounds.min.y, 1e-4);
    try std.testing.expectEqual(@as(usize, 2), queues.instance_matrices.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), queues.instance_matrices.items[0].m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 13.0), queues.instance_matrices.items[1].m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), queues.instance_matrices.items[1].m[13], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), queues.instance_matrices.items[1].m[14], 1e-4);

    try std.testing.expectApproxEqAbs(@as(f32, 999.0), inst.cached_world_matrix.m[12], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 999.0), inst.cached_bounding_box.min.x, 1e-6);
    try std.testing.expectEqual(Vec3.new(111, 0, 0), inst.last_position);
    try std.testing.expectEqual(Vec3.new(0, 222, 0), inst.last_rotation);
    try std.testing.expectEqual(Vec3.new(0, 0, 333), inst.last_scaling);
    try std.testing.expect(inst.dirty);
    try std.testing.expectApproxEqAbs(@as(f32, 777.0), src.cached_matrix.m[12], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 777.0), src.cached_aabb.min.x, 1e-6);
    try std.testing.expectEqual(@as(u64, 4242), src.cached_frame);
    try std.testing.expectApproxEqAbs(@as(f32, 555.0), parent.cached_matrix.m[12], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 555.0), parent.cached_aabb.min.x, 1e-6);
    try std.testing.expectEqual(@as(u64, 8484), parent.cached_frame);
}

test "P5: staged frame survives a second eye; next frame restages" {
    const ally = std.testing.allocator;
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var src = Mesh{
        .name = "eye_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "e0", .source_mesh = &src, .position = Vec3.new(0, 0, 0) };
    var inst1 = InstancedMesh{ .name = "e1", .source_mesh = &src, .position = Vec3.new(8, 0, 0) };
    var ptrs = [_]*InstancedMesh{ &inst0, &inst1 };
    var parent = Mesh{
        .name = "eye_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &ptrs, .capacity = 2 },
    };
    const meshes = [_]*Mesh{&parent};

    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 21,
        .eye = Vec3.new(-50, 0, 0),
    }, &meshes);
    const first_scratch = try ally.dupe(Mat4, queues.instance_matrices.items);
    defer ally.free(first_scratch);
    const first_bounds = parent.instance_render.bounds;

    inst0.position = Vec3.new(100, 0, 0);
    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 21,
        .eye = Vec3.new(50, 0, 0),
    }, &meshes);
    try std.testing.expectEqual(@as(u64, 21), parent.instance_render.staged_frame);
    try std.testing.expectEqual(first_bounds, parent.instance_render.bounds);
    try std.testing.expectEqual(@as(u32, 3), parent.instance_render.count);
    try std.testing.expectEqual(first_scratch.len, queues.instance_matrices.items.len);
    for (first_scratch, queues.instance_matrices.items) |a, b| {
        try std.testing.expectEqual(a, b);
    }

    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 22,
        .eye = Vec3.zero,
    }, &meshes);
    try std.testing.expectEqual(@as(u64, 22), parent.instance_render.staged_frame);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), parent.instance_render.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 101.0), parent.instance_render.bounds.max.x, 1e-4);
}

test "P5: visible transitions and the empty path stay coherent" {
    const ally = std.testing.allocator;
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var src = Mesh{
        .name = "trans_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "t0", .source_mesh = &src, .position = Vec3.new(0, 0, 0) };
    var inst1 = InstancedMesh{ .name = "t1", .source_mesh = &src, .position = Vec3.new(20, 0, 0) };
    var ptrs = [_]*InstancedMesh{ &inst0, &inst1 };
    var parent = Mesh{
        .name = "trans_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &ptrs, .capacity = 2 },
    };
    const meshes = [_]*Mesh{&parent};
    const stage = struct {
        fn run(
            allocator: std.mem.Allocator,
            matrices: *std.ArrayListUnmanaged(Mat4),
            frame: u64,
            list: []const *Mesh,
        ) void {
            instance_staging.stageInstances(.{
                .allocator = allocator,
                .instance_matrices = matrices,
                .thread_pool = null,
                .frame_id = frame,
                .eye = Vec3.zero,
            }, list);
        }
    }.run;

    stage(ally, &queues.instance_matrices, 31, &meshes);
    try std.testing.expectEqual(@as(u32, 3), parent.instance_render.count);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), parent.instance_render.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 21.0), parent.instance_render.bounds.max.x, 1e-4);

    inst1.is_visible = false;
    stage(ally, &queues.instance_matrices, 32, &meshes);
    try std.testing.expectEqual(@as(u32, 2), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 32), parent.instance_render.staged_frame);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), parent.instance_render.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), parent.instance_render.bounds.max.x, 1e-4);

    inst0.is_visible = false;
    parent.is_visible = false;
    parent.instance_render.buffer = .{ .id = 123 };
    parent.instance_render.capacity = 7;
    parent.instance_render.hash = 0xABCD;
    parent.instance_render.uploaded_count = 5;
    stage(ally, &queues.instance_matrices, 33, &meshes);
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 33), parent.instance_render.staged_frame);
    try std.testing.expect(!parent.instance_render.bounds.isValid());
    try std.testing.expectEqual(@as(u32, 123), parent.instance_render.buffer.id);
    try std.testing.expectEqual(@as(usize, 7), parent.instance_render.capacity);
    try std.testing.expectEqual(@as(u64, 0xABCD), parent.instance_render.hash);
    try std.testing.expectEqual(@as(usize, 5), parent.instance_render.uploaded_count);

    inst0.is_visible = true;
    inst1.is_visible = true;
    parent.is_visible = true;
    stage(ally, &queues.instance_matrices, 34, &meshes);
    try std.testing.expectEqual(@as(u32, 3), parent.instance_render.count);
    try std.testing.expect(parent.instance_render.bounds.isValid());
    try std.testing.expectApproxEqAbs(@as(f32, 21.0), parent.instance_render.bounds.max.x, 1e-4);
}

test "P5: scratch OOM keeps the previous publish and allows retry" {
    const ally = std.testing.allocator;
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var src = Mesh{
        .name = "oom_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "o0", .source_mesh = &src, .position = Vec3.new(0, 0, 0) };
    var ptrs = [_]*InstancedMesh{&inst0};
    var parent = Mesh{
        .name = "oom_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &ptrs, .capacity = 1 },
    };
    const meshes = [_]*Mesh{&parent};

    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 41,
        .eye = Vec3.zero,
    }, &meshes);
    try std.testing.expectEqual(@as(u32, 2), parent.instance_render.count);

    var bare = RenderQueues{};
    defer bare.deinit(ally);
    inst0.position = Vec3.new(30, 0, 0);
    var failing = std.testing.FailingAllocator.init(ally, .{ .fail_index = 0 });
    instance_staging.stageInstances(.{
        .allocator = failing.allocator(),
        .instance_matrices = &bare.instance_matrices,
        .thread_pool = null,
        .frame_id = 42,
        .eye = Vec3.zero,
    }, &meshes);
    try std.testing.expectEqual(@as(u32, 2), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 41), parent.instance_render.staged_frame);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), parent.instance_render.bounds.min.x, 1e-4);

    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &bare.instance_matrices,
        .thread_pool = null,
        .frame_id = 42,
        .eye = Vec3.zero,
    }, &meshes);
    try std.testing.expectEqual(@as(u64, 42), parent.instance_render.staged_frame);
    try std.testing.expectApproxEqAbs(@as(f32, 29.0), parent.instance_render.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 31.0), parent.instance_render.bounds.max.x, 1e-4);
}

test "P5: CPU-only staging publishes no upload identity" {
    const ally = std.testing.allocator;
    var queues = RenderQueues{};
    defer queues.deinit(ally);

    var src = Mesh{
        .name = "noupload_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "n0", .source_mesh = &src, .position = Vec3.zero };
    var ptrs = [_]*InstancedMesh{&inst0};
    var parent = Mesh{
        .name = "noupload_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = std.ArrayListUnmanaged(*InstancedMesh){ .items = &ptrs, .capacity = 1 },
    };
    const meshes = [_]*Mesh{&parent};
    const ctx = instance_staging.InstanceStageContext{
        .allocator = ally,
        .instance_matrices = &queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 51,
        .eye = Vec3.zero,
    };
    instance_staging.stageInstances(ctx, &meshes);
    try std.testing.expectEqual(@as(u32, 2), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 51), parent.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u64, 0), parent.instance_render.hash);
    try std.testing.expectEqual(@as(usize, 0), parent.instance_render.uploaded_count);

    instance_staging.stageInstances(.{
        .allocator = ally,
        .instance_matrices = &queues.instance_matrices,
        .thread_pool = null,
        .frame_id = 52,
        .eye = Vec3.zero,
    }, &meshes);
    try std.testing.expectEqual(@as(u64, 0), parent.instance_render.hash);
    try std.testing.expectEqual(@as(usize, 0), parent.instance_render.uploaded_count);
}
