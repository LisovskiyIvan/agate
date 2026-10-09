const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const occlusion_mod = @import("occlusion.zig");
const AudioOcclusionConfig = occlusion_mod.AudioOcclusionConfig;
const AudioOcclusionTracker = occlusion_mod.AudioOcclusionTracker;
const evaluateRaycastOcclusion = occlusion_mod.evaluateRaycastOcclusion;

test "AudioOcclusionTracker smooth transition" {
    var tracker = AudioOcclusionTracker.init(0.0);
    try std.testing.expectEqual(@as(f32, 0.0), tracker.current);

    // After 0.15s with smooth_time = 0.15s: 1 - 1/e ≈ 0.632
    const v = tracker.update(1.0, 0.15, 0.15);
    try std.testing.expect(v > 0.55 and v < 0.70);

    // After 5 more seconds: reaches target ~1.0
    _ = tracker.update(1.0, 5.0, 0.15);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), tracker.current, 1e-3);

    tracker.reset(0.0);
    try std.testing.expectEqual(@as(f32, 0.0), tracker.current);
}

test "evaluateRaycastOcclusion single and multi-tap" {
    const MockWall = struct {
        wall_x: f32 = 5.0,

        fn raycast(origin: Vec3, direction: Vec3, max_dist: f32, user_data: ?*anyopaque) bool {
            const self: *const @This() = @ptrCast(@alignCast(user_data orelse return false));
            // Ray: origin + t * direction. Check if ray crosses plane x = wall_x within max_dist
            if (@abs(direction.x) < 1e-6) return false;
            const t = (self.wall_x - origin.x) / direction.x;
            return t >= 0.0 and t <= max_dist;
        }
    };

    var wall = MockWall{ .wall_x = 5.0 };
    const listener = Vec3.new(0.0, 0.0, 0.0);
    const emitter_behind = Vec3.new(10.0, 0.0, 0.0);
    const emitter_in_front = Vec3.new(2.0, 0.0, 0.0);

    // Single ray behind wall -> 1.0 (occluded)
    const occ_blocked = evaluateRaycastOcclusion(listener, emitter_behind, .{}, MockWall.raycast, &wall);
    try std.testing.expectEqual(@as(f32, 1.0), occ_blocked);

    // Single ray in front of wall -> 0.0 (clear)
    const occ_clear = evaluateRaycastOcclusion(listener, emitter_in_front, .{}, MockWall.raycast, &wall);
    try std.testing.expectEqual(@as(f32, 0.0), occ_clear);

    // Multi-tap raycast
    const multi_cfg = AudioOcclusionConfig{ .num_rays = 5, .spread_radius = 1.0 };
    const occ_multi = evaluateRaycastOcclusion(listener, emitter_behind, multi_cfg, MockWall.raycast, &wall);
    try std.testing.expectEqual(@as(f32, 1.0), occ_multi);
}
