const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const target_mod = @import("target.zig");
const TargetCamera = target_mod.TargetCamera;

test "TargetCamera view looks at the target" {
    const cam = TargetCamera.init("watcher", .{
        .position = Vec3.new(0.0, 0.0, 5.0),
        .target = Vec3.zero,
    });
    const fwd = cam.getForward();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), fwd.z, 1e-5);
    // The look-at point projects to the screen center.
    const vp = cam.getViewProjection(800.0 / 600.0);
    const p = vp.projectPoint(Vec3.zero, 800.0, 600.0) orelse return error.PointBehindCamera;
    try std.testing.expectApproxEqAbs(@as(f32, 400.0), p.x, 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 300.0), p.y, 1e-2);
}

test "TargetCamera top-down view does not degenerate" {
    const cam = TargetCamera.init("top", .{
        .position = Vec3.new(0.0, 5.0, 0.0),
        .target = Vec3.zero,
        .up = Vec3.up, // Parallel to the view direction: must be handled.
    });
    const fwd = cam.getForward();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), fwd.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.z, 1e-5);
    const view = cam.getViewMatrix();
    for (view.m) |v| try std.testing.expect(std.math.isFinite(v));
    // Target still projects to the screen center despite the up fallback.
    const vp = cam.getViewProjection(800.0 / 600.0);
    const p = vp.projectPoint(Vec3.zero, 800.0, 600.0) orelse return error.PointBehindCamera;
    try std.testing.expectApproxEqAbs(@as(f32, 400.0), p.x, 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 300.0), p.y, 1e-2);
}

test "TargetCamera smoothing converges to the goals" {
    var cam = TargetCamera.init("watcher", .{
        .position = Vec3.new(0.0, 0.0, 10.0),
        .target = Vec3.new(5.0, 0.0, 0.0),
        .smoothing = 8.0,
    });
    cam.setDesiredPosition(Vec3.new(0.0, 0.0, 5.0));
    cam.setTarget(Vec3.zero);
    var i: usize = 0;
    while (i < 600) : (i += 1) cam.update(1.0 / 60.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cam.position.x, 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cam.position.y, 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), cam.position.z, 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cam.target.x, 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cam.target.y, 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cam.target.z, 1e-3);

    // Zero smoothing snaps instantly.
    var snap = TargetCamera.init("snap", .{ .smoothing = 0.0 });
    snap.setDesiredPosition(Vec3.new(1.0, 2.0, 3.0));
    snap.setTarget(Vec3.new(4.0, 5.0, 6.0));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), snap.position.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), snap.target.z, 1e-6);
}
