const std = @import("std");
const sokol = @import("sokol");
const math = @import("math");
const Vec3 = math.Vec3;
const FreeCamera = @import("free.zig").FreeCamera;

test "FreeCamera faces -Z at zero yaw and pitch" {
    const cam = FreeCamera.init("test", .{});
    const fwd = cam.getForward();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), fwd.z, 1e-5);
}

test "FreeCamera view matrix maps its position to the origin" {
    const cam = FreeCamera.init("test", .{ .position = Vec3.new(1.0, 2.0, 5.0) });
    const p = cam.getViewMatrix().transformPoint(cam.getPosition());
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), p.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), p.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), p.z, 1e-4);
}

test "FreeCamera projects the look-at point to the screen center" {
    const cam = FreeCamera.init("test", .{ .position = Vec3.new(0.0, 0.0, 5.0) });
    const vp = cam.getViewProjection(800.0 / 600.0);
    const p = vp.projectPoint(Vec3.zero, 800.0, 600.0) orelse return error.PointBehindCamera;
    try std.testing.expectApproxEqAbs(@as(f32, 400.0), p.x, 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 300.0), p.y, 1e-2);
}

test "FreeCamera moves forward relative to yaw" {
    var cam = FreeCamera.init("test", .{});
    cam.move_forward = true;
    cam.update(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cam.position.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cam.position.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -6.0), cam.position.z, 1e-5);

    var strafe = FreeCamera.init("strafe", .{ .rotation = Vec3.new(0.0, 90.0, 0.0) });
    strafe.move_right = true;
    strafe.update(1.0);
    // Yaw 90 faces -X, so its right hand points toward -Z.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), strafe.position.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), strafe.position.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -6.0), strafe.position.z, 1e-4);
}

test "FreeCamera inertia enables smooth rotation damping" {
    var cam = FreeCamera.init("free", .{ .inertia = 0.85 });
    var down_ev = sokol.app.Event{
        .type = .MOUSE_DOWN,
        .mouse_button = .LEFT,
        .mouse_x = 200.0,
        .mouse_y = 200.0,
    };
    cam.handleEvent(&down_ev);

    var move_ev = sokol.app.Event{
        .type = .MOUSE_MOVE,
        .mouse_x = 220.0,
        .mouse_y = 210.0,
    };
    cam.handleEvent(&move_ev);

    try std.testing.expectEqual(@as(f32, 0.0), cam.rotation.x);
    try std.testing.expectEqual(@as(f32, 0.0), cam.rotation.y);
    try std.testing.expect(cam.inertial_rotation_y != 0.0);

    cam.update(1.0 / 60.0);
    try std.testing.expect(cam.rotation.y != 0.0);
    const rot1 = cam.rotation.y;
    cam.update(1.0 / 60.0);
    try std.testing.expect(cam.rotation.y < rot1);
}
