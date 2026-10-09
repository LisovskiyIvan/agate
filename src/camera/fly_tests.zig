const std = @import("std");
const sokol = @import("sokol");
const math = @import("math");
const Vec3 = math.Vec3;
const FlyCamera = @import("fly.zig").FlyCamera;

test "FlyCamera forward is roll-invariant, pitch 90 looks straight up" {
    const cam = FlyCamera.init("fly", .{ .rotation = Vec3.new(90.0, 90.0, 90.0) });
    const fwd = cam.getForward();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), fwd.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.z, 1e-4);
    // Yaw 90 alone faces -X, whatever the roll is.
    const yawed = FlyCamera.init("yawed", .{ .rotation = Vec3.new(0.0, 90.0, 45.0) });
    const f2 = yawed.getForward();
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), f2.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), f2.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), f2.z, 1e-5);
}

test "FlyCamera roll of 90 degrees swaps the up/right basis" {
    const cam = FlyCamera.init("fly", .{ .rotation = Vec3.new(0.0, 0.0, 90.0) });
    const fwd = cam.getForward();
    const up = cam.getUp();
    const right = cam.getRight();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), fwd.z, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), up.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), up.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), up.z, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), right.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), right.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), right.z, 1e-5);
    // Orthonormal right-handed basis is preserved under roll.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), right.dot(up), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), right.length(), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), up.length(), 1e-5);
    const handed = right.cross(up);
    try std.testing.expectApproxEqAbs(-fwd.x, handed.x, 1e-5);
    try std.testing.expectApproxEqAbs(-fwd.y, handed.y, 1e-5);
    try std.testing.expectApproxEqAbs(-fwd.z, handed.z, 1e-5);
}

test "FlyCamera WASD flight follows forward and rolled right" {
    var cam = FlyCamera.init("fly", .{});
    cam.move_forward = true;
    cam.update(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cam.position.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cam.position.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -6.0), cam.position.z, 1e-5);

    // With 90 degrees of roll the strafe axis points along +Y.
    var rolled = FlyCamera.init("rolled", .{ .rotation = Vec3.new(0.0, 0.0, 90.0) });
    rolled.move_right = true;
    rolled.update(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), rolled.position.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), rolled.position.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), rolled.position.z, 1e-4);

    // R/F climbs along the (rolled) up axis; boost scales the speed.
    var climb = FlyCamera.init("climb", .{});
    climb.move_up = true;
    climb.boost_held = true;
    climb.update(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 24.0), climb.position.y, 1e-4);
}

test "FlyCamera inertia enables smooth rotation damping" {
    var cam = FlyCamera.init("fly", .{ .inertia = 0.85 });
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
