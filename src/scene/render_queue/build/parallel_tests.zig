const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;
const Frustum = math.Frustum;
const Vec3 = math.Vec3;
const Mesh = @import("../../../mesh.zig").Mesh;
const jobs = @import("../../../jobs.zig");
const SceneStats = @import("../../stats.zig").SceneStats;
const RenderQueues = @import("../items.zig").RenderQueues;
const visibility = @import("../../../visibility/mod.zig");
const parallel = @import("parallel.zig");
const buildFrameQueuesParallel = parallel.buildFrameQueuesParallel;

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
