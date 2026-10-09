const std = @import("std");
const gpu_timing = @import("../gpu_timing.zig");
const stats_mod = @import("stats.zig");
const SceneStats = stats_mod.SceneStats;

test "SceneStats mergeFrom never touches context-owned GPU metadata" {
    // Game-side build stats must not observe or disturb the render-owned
    // GPU samples: durations, submission ids, and scope stay intact.
    var latch: SceneStats = .{
        .gpu_frame_ms = 2.5,
        .gpu_frame_submit = 11,
        .gpu_frame_scope = .command_buffer,
        .gpu_shadow_ms = 0.5,
        .gpu_shadow_submit = 11,
        .gpu_main_ms = 1.5,
        .gpu_main_submit = 12,
        .gpu_post_ms = 0,
        .gpu_post_submit = 13, // valid quantized zero: present, zero
        .total_meshes = 4,
    };
    const build_side: SceneStats = .{
        .total_meshes = 10,
        .rendered_meshes = 6,
        .culled_meshes = 2,
        .occluded_meshes = 1,
        .occluders_count = 3,
        .occluder_triangles = 99,
        .build_oom_drops = 2,
        // A hostile/stale game build carrying GPU-looking values must not
        // leak them into the latch.
        .gpu_frame_ms = 99.0,
        .gpu_frame_submit = 99,
        .gpu_frame_scope = .pass_sum,
        .gpu_shadow_ms = 99.0,
        .gpu_shadow_submit = 99,
        .gpu_main_ms = 99.0,
        .gpu_main_submit = 99,
        .gpu_post_ms = 99.0,
        .gpu_post_submit = 99,
        .shadow_ms = 99.0,
        .update_ms = 99.0,
    };
    latch.mergeFrom(&build_side);
    try std.testing.expectEqual(@as(u32, 14), latch.total_meshes);
    try std.testing.expectEqual(@as(u32, 2), latch.build_oom_drops);
    try std.testing.expectEqual(@as(f32, 2.5), latch.gpu_frame_ms);
    try std.testing.expectEqual(@as(u32, 11), latch.gpu_frame_submit);
    try std.testing.expectEqual(gpu_timing.FrameScope.command_buffer, latch.gpu_frame_scope);
    try std.testing.expectEqual(@as(f32, 0.5), latch.gpu_shadow_ms);
    try std.testing.expectEqual(@as(u32, 11), latch.gpu_shadow_submit);
    try std.testing.expectEqual(@as(f32, 1.5), latch.gpu_main_ms);
    try std.testing.expectEqual(@as(u32, 12), latch.gpu_main_submit);
    try std.testing.expectEqual(@as(f32, 0), latch.gpu_post_ms);
    try std.testing.expectEqual(@as(u32, 13), latch.gpu_post_submit);
    try std.testing.expectEqual(@as(f32, 0), latch.shadow_ms);
    try std.testing.expectEqual(@as(f32, 0), latch.update_ms);
}

test "SceneStats default reset carries no GPU sample" {
    const empty: SceneStats = .{};
    try std.testing.expectEqual(@as(u32, 0), empty.gpu_frame_submit);
    try std.testing.expectEqual(@as(u32, 0), empty.gpu_shadow_submit);
    try std.testing.expectEqual(@as(u32, 0), empty.gpu_main_submit);
    try std.testing.expectEqual(@as(u32, 0), empty.gpu_post_submit);
    try std.testing.expectEqual(gpu_timing.FrameScope.none, empty.gpu_frame_scope);
}
