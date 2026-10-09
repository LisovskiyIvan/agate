const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const follow_mod = @import("follow.zig");
const FollowCamera = follow_mod.FollowCamera;

test "FollowCamera snaps behind the target with lerp_speed 0" {
    var cam = FollowCamera.init("test", .{
        .target_position = Vec3.new(10.0, 0.0, 0.0),
        .radius = 5.0,
        .height_offset = 2.0,
        .lerp_speed = 0.0,
    });
    cam.position = Vec3.zero;
    cam.update(0.016);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), cam.position.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), cam.position.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), cam.position.z, 1e-5);
    const fwd = cam.getForward();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.x, 1e-5);
    try std.testing.expect(fwd.y < 0.0 and fwd.z < 0.0);
}
