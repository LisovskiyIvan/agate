const std = @import("std");
const vec = @import("vec.zig");
const Vec2 = vec.Vec2;
const Vec3 = vec.Vec3;
const Vec4 = vec.Vec4;

test "Vec2 basic operations and constants" {
    const v0 = Vec2.zero;
    try std.testing.expectEqual(@as(f32, 0.0), v0.x);
    try std.testing.expectEqual(@as(f32, 0.0), v0.y);

    const v1 = Vec2.one;
    try std.testing.expectEqual(@as(f32, 1.0), v1.x);
    try std.testing.expectEqual(@as(f32, 1.0), v1.y);

    const custom = Vec2.new(3.5, -2.25);
    try std.testing.expectEqual(@as(f32, 3.5), custom.x);
    try std.testing.expectEqual(@as(f32, -2.25), custom.y);
}

test "Vec3 arithmetic, cross, dot and length" {
    const a = Vec3.new(1.0, 2.0, 3.0);
    const b = Vec3.new(4.0, 5.0, 6.0);

    const added = a.add(b);
    try std.testing.expectEqual(Vec3.new(5.0, 7.0, 9.0), added);

    const subbed = b.sub(a);
    try std.testing.expectEqual(Vec3.new(3.0, 3.0, 3.0), subbed);

    const scaled = a.scale(2.5);
    try std.testing.expectEqual(Vec3.new(2.5, 5.0, 7.5), scaled);

    // Dot product: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), a.dot(b), 1e-6);

    // Cross product: X x Y = Z
    const x = Vec3.right; // (1, 0, 0)
    const y = Vec3.up; // (0, 1, 0)
    const z = x.cross(y);
    try std.testing.expectEqual(Vec3.new(0.0, 0.0, 1.0), z);

    // Y x X = -Z
    try std.testing.expectEqual(Vec3.new(0.0, 0.0, -1.0), y.cross(x));

    // Length and LengthSq
    const v = Vec3.new(3.0, 4.0, 0.0);
    try std.testing.expectEqual(@as(f32, 25.0), v.lengthSq());
    try std.testing.expectEqual(@as(f32, 5.0), v.length());
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

test "Vec3 lerp, distance and SIMD conversions" {
    const a = Vec3.new(0.0, 10.0, 20.0);
    const b = Vec3.new(10.0, 20.0, 30.0);

    const mid = Vec3.lerp(a, b, 0.5);
    try std.testing.expectEqual(Vec3.new(5.0, 15.0, 25.0), mid);

    try std.testing.expectEqual(@as(f32, 300.0), Vec3.distanceSq(a, b));
    try std.testing.expectApproxEqAbs(@sqrt(@as(f32, 300.0)), Vec3.distance(a, b), 1e-5);

    // SIMD conversions
    const simd = a.toSimd();
    const from_simd = Vec3.fromSimd(simd);
    try std.testing.expect(a.eql(from_simd));

    const arr = a.toArray();
    try std.testing.expectEqualSlices(f32, &.{ 0.0, 10.0, 20.0 }, &arr);
}

test "Vec4 arithmetic, scaling, lerp and SIMD conversions" {
    const v0 = Vec4.zero;
    try std.testing.expectEqual(Vec4.new(0.0, 0.0, 0.0, 0.0), v0);

    const a = Vec4.new(1.0, 2.0, 3.0, 4.0);
    const b = Vec4.new(5.0, 6.0, 7.0, 8.0);

    const added = a.add(b);
    try std.testing.expectEqual(Vec4.new(6.0, 8.0, 10.0, 12.0), added);

    const subbed = b.sub(a);
    try std.testing.expectEqual(Vec4.new(4.0, 4.0, 4.0, 4.0), subbed);

    const scaled = a.scale(2.0);
    try std.testing.expectEqual(Vec4.new(2.0, 4.0, 6.0, 8.0), scaled);

    const lerped = Vec4.lerp(a, b, 0.25);
    try std.testing.expectEqual(Vec4.new(2.0, 3.0, 4.0, 5.0), lerped);

    const arr = a.toArray();
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 2.0, 3.0, 4.0 }, &arr);
}
