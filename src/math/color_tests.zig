const std = @import("std");
const color_mod = @import("color.zig");
const Color3 = color_mod.Color3;
const Color4 = color_mod.Color4;

test "Color3 scaling and Color4 conversion" {
    const c = Color3.new(0.2, 0.4, 0.8);
    const scaled = c.scale(2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), scaled.r, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), scaled.g, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.6), scaled.b, 1e-6);

    const c4 = c.toColor4(0.5);
    try std.testing.expectEqual(@as(f32, 0.2), c4.r);
    try std.testing.expectEqual(@as(f32, 0.4), c4.g);
    try std.testing.expectEqual(@as(f32, 0.8), c4.b);
    try std.testing.expectEqual(@as(f32, 0.5), c4.a);

    // Predefined constants
    try std.testing.expectEqual(Color3{ .r = 1.0, .g = 1.0, .b = 1.0 }, Color3.white);
    try std.testing.expectEqual(Color3{ .r = 0.0, .g = 0.0, .b = 0.0 }, Color3.black);
    try std.testing.expectEqual(Color3{ .r = 1.0, .g = 0.0, .b = 0.0 }, Color3.red);
}

test "Color4 presets, lerp, scaling and SIMD operations" {
    try std.testing.expectEqual(Color4{ .r = 1, .g = 1, .b = 1, .a = 1 }, Color4.white);
    try std.testing.expectEqual(Color4{ .r = 0, .g = 0, .b = 0, .a = 0 }, Color4.transparent);

    const c1 = Color4.new(0.0, 0.0, 0.0, 0.0);
    const c2 = Color4.new(1.0, 1.0, 1.0, 1.0);
    const mid = Color4.lerp(c1, c2, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mid.r, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mid.g, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mid.b, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mid.a, 1e-6);

    const scaled = mid.scale(2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), scaled.r, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), scaled.g, 1e-6);

    const arr = mid.toArray();
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 0.5, 0.5, 0.5 }, &arr);
}
