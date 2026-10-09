const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const types = @import("types.zig");
const slotAge = types.slotAge;
const analyticPosition = types.analyticPosition;

test "slot age gates the alive window" {
    // Unborn (spawn in the future): also culls stale slots after a reset.
    const unborn = slotAge(1.0, 2.0, 1.5);
    try std.testing.expectEqual(false, unborn.alive);
    try std.testing.expectEqual(@as(f32, 0.0), unborn.t);
    // Mid-life.
    const mid = slotAge(2.75, 2.0, 1.5);
    try std.testing.expectEqual(true, mid.alive);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mid.t, 1e-6);
    // Death at exactly t == 1.
    const dead = slotAge(3.5, 2.0, 1.5);
    try std.testing.expectEqual(false, dead.alive);
    try std.testing.expectEqual(@as(f32, 1.0), dead.t);
    // Degenerate lifetimes clamp so the shader division stays finite; a
    // zeroed slot (never written) is dead at any clock >= 1e-4 and unborn
    // exactly at the epoch.
    try std.testing.expectEqual(false, slotAge(0.5, 0.0, 0.0).alive);
    try std.testing.expectEqual(true, slotAge(0.0, 0.0, 0.0).alive);
}

test "analytic trajectory matches golden values" {
    const p0 = Vec3.new(1.0, 2.0, 3.0);
    const v0 = Vec3.new(1.0, 0.0, -1.0);
    const g = Vec3.new(0.0, -9.8, 0.0);

    // No drag: p = p0 + v0*t + 0.5*g*t^2 at t = 0.5 ->
    // (1.5, 2 - 1.225, 2.5).
    const free = analyticPosition(p0, v0, g, 0.0, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), free.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.775), free.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), free.z, 1e-6);

    // Zero elapsed time returns the spawn point exactly.
    try std.testing.expectEqual(p0, analyticPosition(p0, v0, g, 3.0, 0.0));

    // Drag k = 2 at t = 0.5: s = (1 - e^-1)/2 = 0.3160602794,
    // s2 = (0.5 - s)/2 = 0.0919698603:
    //   x = 1 + s         = 1.3160602794
    //   y = 2 - 9.8 * s2  = 1.0986953694
    //   z = 3 - s         = 2.6839397206
    // Tolerance 1e-4 covers the float32 rounding of the hand-computed
    // doubles; the GLSL duplicate may deviate by ~1e-6 relative on top
    // (documented on analyticPosition).
    const dragged = analyticPosition(p0, v0, g, 2.0, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.3160603), dragged.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0986954), dragged.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.6839397), dragged.z, 1e-4);

    // Drag below the cancellation threshold (1e-6) degrades to the free
    // trajectory without catastrophic cancellation.
    const tiny = analyticPosition(p0, v0, g, 1.0e-7, 0.5);
    try std.testing.expectApproxEqAbs(free.x, tiny.x, 1e-5);
    try std.testing.expectApproxEqAbs(free.y, tiny.y, 1e-5);
    try std.testing.expectApproxEqAbs(free.z, tiny.z, 1e-5);
}
