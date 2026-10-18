const std = @import("std");
const vec = @import("vec.zig");
const Vec2 = vec.Vec2;
const Vec3 = vec.Vec3;
const Vec4 = vec.Vec4;

test "Vec3 cross follows the right-handed convention" {
    // X x Y = Z
    const x = Vec3.right; // (1, 0, 0)
    const y = Vec3.up; // (0, 1, 0)
    try std.testing.expectEqual(Vec3.new(0.0, 0.0, 1.0), x.cross(y));

    // Y x X = -Z (anti-commutative)
    try std.testing.expectEqual(Vec3.new(0.0, 0.0, -1.0), y.cross(x));
}

test "Vec3 normalize handles zero without NaN and scales correctly" {
    // Zero vector normalization returns zero
    const norm_zero = Vec3.zero.normalize();
    try std.testing.expectEqual(Vec3.zero, norm_zero);
    try std.testing.expect(!std.math.isNan(norm_zero.x));
    try std.testing.expect(!std.math.isNan(norm_zero.y));
    try std.testing.expect(!std.math.isNan(norm_zero.z));

    // Normalizing non-zero vector produces unit vector
    const v = Vec3.new(10.0, -20.0, 5.0);
    const n = v.normalize();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), n.length(), 1e-6);
}
