const std = @import("std");
const sokol = @import("sokol");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const viewport_mod = @import("viewport.zig");
const Viewport = viewport_mod.Viewport;
const arc_rotate_mod = @import("arc_rotate.zig");
const ArcRotateCamera = arc_rotate_mod.ArcRotateCamera;
const free_mod = @import("free.zig");
const FreeCamera = free_mod.FreeCamera;
const follow_mod = @import("follow.zig");
const FollowCamera = follow_mod.FollowCamera;
const target_mod = @import("target.zig");
const TargetCamera = target_mod.TargetCamera;
const fly_mod = @import("fly.zig");
const FlyCamera = fly_mod.FlyCamera;
const union_mod = @import("union.zig");
const Camera = union_mod.Camera;

test "Camera union dispatches target and fly views and lens params" {
    const aspect = 800.0 / 600.0;
    const arc: Camera = .{ .arc_rotate = ArcRotateCamera.init("orbit", .{}) };
    const free: Camera = .{ .free = FreeCamera.init("free", .{ .position = Vec3.new(0.0, 0.0, 5.0) }) };
    const follow: Camera = .{ .follow = FollowCamera.init("follow", .{ .target_position = Vec3.zero }) };
    const target: Camera = .{ .target = TargetCamera.init("watcher", .{
        .position = Vec3.new(0.0, 0.0, 5.0),
        .target = Vec3.zero,
        .fov_deg = 50.0,
        .near = 0.5,
        .far = 200.0,
    }) };
    const fly: Camera = .{ .fly = FlyCamera.init("fly", .{
        .position = Vec3.new(0.0, 0.0, 5.0),
        .fov_deg = 70.0,
        .near = 0.2,
        .far = 300.0,
    }) };
    // Every variant produces a usable view-projection: the point in front
    // of each camera projects inside the viewport.
    for ([_]Camera{ arc, free, follow, target, fly }) |cam| {
        const eye = cam.getPosition();
        const ahead = eye.add(cam.getForward().scale(5.0));
        const vp = cam.getViewProjection(aspect);
        const p = vp.projectPoint(ahead, 800.0, 600.0) orelse return error.PointBehindCamera;
        try std.testing.expect(p.x >= 0.0 and p.x <= 800.0);
        try std.testing.expect(p.y >= 0.0 and p.y <= 600.0);
    }
}

test "Camera union dispatches update to arc_rotate with inertia" {
    var cam: Camera = .{ .arc_rotate = ArcRotateCamera.init("orbit", .{ .inertia = 0.9 }) };
    var down_ev = sokol.app.Event{
        .type = .MOUSE_DOWN,
        .mouse_button = .LEFT,
        .mouse_x = 50.0,
        .mouse_y = 50.0,
    };
    cam.handleEvent(&down_ev);

    var move_ev = sokol.app.Event{
        .type = .MOUSE_MOVE,
        .mouse_x = 70.0,
        .mouse_y = 60.0,
    };
    cam.handleEvent(&move_ev);

    const pos0 = cam.getPosition();
    cam.update(1.0 / 60.0);
    const pos1 = cam.getPosition();
    try std.testing.expect(pos0.x != pos1.x or pos0.z != pos1.z);
}

test "Camera union getRight and getUp" {
    const free: Camera = .{ .free = FreeCamera.init("free", .{ .position = Vec3.new(0, 0, 5), .rotation = Vec3.zero }) };
    const r = free.getRight();
    const u = free.getUp();
    const f = free.getForward();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), r.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), r.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), r.z, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), u.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), u.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), u.z, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), f.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), f.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), f.z, 1e-4);
}

test "Camera union setLookAt updates forward" {
    var target_cam: Camera = .{ .target = TargetCamera.init("target", .{
        .position = Vec3.new(0, 0, 10),
        .target = Vec3.zero,
        .inertia = 0.0,
    }) };
    target_cam.setLookAt(Vec3.new(0, 0, 5), Vec3.new(0, 0, 0), Vec3.up);
    const fwd = target_cam.getForward();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), fwd.z, 1e-5);
}
