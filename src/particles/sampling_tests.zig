const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const sampling = @import("sampling.zig");
const normalizeAngleDeg = sampling.normalizeAngleDeg;
const rotationToRadians = sampling.rotationToRadians;
const spritesheetFrameCount = sampling.spritesheetFrameCount;
const spritesheetFrameForAge = sampling.spritesheetFrameForAge;
const spritesheetUvRect = sampling.spritesheetUvRect;
const localToWorld = sampling.localToWorld;
const worldScaleFactor = sampling.worldScaleFactor;
const sys = @import("system.zig");

test "spritesheet frames across ages and loops" {
    // 2x2 sheet, lifetime 4s, single loop: one frame per second.
    try std.testing.expectEqual(@as(u32, 0), spritesheetFrameForAge(0.0, 4.0, 2, 2, 1.0));
    try std.testing.expectEqual(@as(u32, 1), spritesheetFrameForAge(1.0, 4.0, 2, 2, 1.0));
    try std.testing.expectEqual(@as(u32, 2), spritesheetFrameForAge(2.0, 4.0, 2, 2, 1.0));
    try std.testing.expectEqual(@as(u32, 3), spritesheetFrameForAge(3.0, 4.0, 2, 2, 1.0));
    // Double loop: age 1s (norm 0.25) -> floor(0.25*2*4) = 2.
    try std.testing.expectEqual(@as(u32, 2), spritesheetFrameForAge(1.0, 4.0, 2, 2, 2.0));
    try std.testing.expectEqual(@as(u32, 1), spritesheetFrameForAge(0.5, 4.0, 2, 2, 2.0));
    // UV rects run left-to-right, bottom-to-top.
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.5, 0.5 }, spritesheetUvRect(0, 2, 2));
    try std.testing.expectEqual([4]f32{ 0.5, 0.0, 0.5, 0.5 }, spritesheetUvRect(1, 2, 2));
    try std.testing.expectEqual([4]f32{ 0.0, 0.5, 0.5, 0.5 }, spritesheetUvRect(2, 2, 2));
    try std.testing.expectEqual([4]f32{ 0.5, 0.5, 0.5, 0.5 }, spritesheetUvRect(3, 2, 2));
}

test "spritesheet boundaries and 1x1 default" {
    // 1x1 (default) always yields frame 0 / full-texture UV.
    try std.testing.expectEqual(@as(u32, 0), spritesheetFrameForAge(0.0, 1.0, 1, 1, 1.0));
    try std.testing.expectEqual(@as(u32, 0), spritesheetFrameForAge(0.99, 1.0, 1, 1, 5.0));
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 1.0, 1.0 }, spritesheetUvRect(0, 1, 1));
    // Boundary: exactly age == lifetime wraps per spec (floor(loops*frames) % frames).
    try std.testing.expectEqual(@as(u32, 0), spritesheetFrameForAge(4.0, 4.0, 2, 2, 1.0));
    try std.testing.expectEqual(@as(u32, 3), spritesheetFrameForAge(3.999, 4.0, 2, 2, 1.0));
    // Zero grid dimensions are guarded to 1 (no div-by-zero).
    try std.testing.expectEqual(@as(u32, 0), spritesheetFrameForAge(0.5, 1.0, 0, 0, 1.0));
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 1.0, 1.0 }, spritesheetUvRect(7, 0, 0));
    try std.testing.expectEqual(@as(u32, 4), spritesheetFrameCount(2, 2));
    try std.testing.expectEqual(@as(u32, 1), spritesheetFrameCount(0, 0));
}

test "rotation integrates angular velocity" {
    var ps = try sys.makeTestSystem(std.testing.allocator, 4);
    defer sys.freeTestSystem(&ps);
    ps.direction_min = Vec3.zero;
    ps.direction_max = Vec3.zero;
    ps.speed_min = 0.0;
    ps.speed_max = 0.0;
    ps.gravity = Vec3.zero;
    ps.lifetime_min = 10.0;
    ps.lifetime_max = 10.0;
    ps.rotation_min = 0.0;
    ps.rotation_max = 0.0;
    ps.angular_velocity_min = 90.0;
    ps.angular_velocity_max = 90.0;
    ps.emitOne();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ps.particles[0].rotation, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 90.0), ps.particles[0].angular_velocity, 1e-5);
    ps.updateCpu(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 90.0), ps.particles[0].rotation, 1e-4);
    // Instance carries radians: 90deg = pi/2.
    try std.testing.expectApproxEqAbs(
        std.math.pi / 2.0,
        ps.instances[0].rotation_misc[0],
        1e-5,
    );
    // Wrap-around: 350deg + 20deg/s * 1s = 10deg.
    var ps2 = try sys.makeTestSystem(std.testing.allocator, 4);
    defer sys.freeTestSystem(&ps2);
    ps2.direction_min = Vec3.zero;
    ps2.direction_max = Vec3.zero;
    ps2.speed_min = 0.0;
    ps2.speed_max = 0.0;
    ps2.gravity = Vec3.zero;
    ps2.lifetime_min = 10.0;
    ps2.lifetime_max = 10.0;
    ps2.rotation_min = 350.0;
    ps2.rotation_max = 350.0;
    ps2.angular_velocity_min = 20.0;
    ps2.angular_velocity_max = 20.0;
    ps2.emitOne();
    ps2.updateCpu(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), ps2.particles[0].rotation, 1e-4);
}

test "angle normalization" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), normalizeAngleDeg(0.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), normalizeAngleDeg(360.0), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), normalizeAngleDeg(-360.0), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), normalizeAngleDeg(370.0), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 350.0), normalizeAngleDeg(-10.0), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 180.0), normalizeAngleDeg(540.0), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), rotationToRadians(0.0), 1e-6);
    try std.testing.expectApproxEqAbs(std.math.pi, rotationToRadians(180.0), 1e-5);
}

test "localToWorld helper with rotation and scale" {
    const m = Mat4.fromRotationTranslationScale(
        Vec3.new(10.0, 0.0, 0.0),
        Vec3.new(0.0, 0.0, 90.0),
        Vec3.new(2.0, 2.0, 2.0),
    );
    // (1,0,0) -> scaled (2,0,0) -> rotZ90 -> (0,2,0) -> translated (10,2,0).
    const w = localToWorld(m, Vec3.new(1.0, 0.0, 0.0));
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), w.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), w.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), w.z, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), worldScaleFactor(m), 1e-5);
    // Identity is exact.
    const id = Mat4.identity;
    const p = localToWorld(id, Vec3.new(1.0, 2.0, 3.0));
    try std.testing.expectEqual(Vec3.new(1.0, 2.0, 3.0), p);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), worldScaleFactor(id), 1e-6);
}
