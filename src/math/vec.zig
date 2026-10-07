const std = @import("std");

pub const Vec2 = extern struct {
    x: f32 = 0.0,
    y: f32 = 0.0,

    pub const zero = Vec2{ .x = 0, .y = 0 };
    pub const one = Vec2{ .x = 1, .y = 1 };

    pub fn new(x: f32, y: f32) Vec2 {
        return .{ .x = x, .y = y };
    }
};

pub const Vec3 = extern struct {
    x: f32 = 0.0,
    y: f32 = 0.0,
    z: f32 = 0.0,

    pub const zero = Vec3{ .x = 0, .y = 0, .z = 0 };
    pub const one = Vec3{ .x = 1, .y = 1, .z = 1 };
    pub const up = Vec3{ .x = 0, .y = 1, .z = 0 };
    pub const down = Vec3{ .x = 0, .y = -1, .z = 0 };
    pub const left = Vec3{ .x = -1, .y = 0, .z = 0 };
    pub const right = Vec3{ .x = 1, .y = 0, .z = 0 };
    pub const forward = Vec3{ .x = 0, .y = 0, .z = 1 };
    pub const backward = Vec3{ .x = 0, .y = 0, .z = -1 };

    pub fn new(x: f32, y: f32, z: f32) Vec3 {
        return .{ .x = x, .y = y, .z = z };
    }

    pub fn add(a: Vec3, b: Vec3) Vec3 {
        return .{ .x = a.x + b.x, .y = a.y + b.y, .z = a.z + b.z };
    }

    pub fn sub(a: Vec3, b: Vec3) Vec3 {
        return .{ .x = a.x - b.x, .y = a.y - b.y, .z = a.z - b.z };
    }

    pub fn scale(v: Vec3, s: f32) Vec3 {
        return .{ .x = v.x * s, .y = v.y * s, .z = v.z * s };
    }

    pub fn dot(a: Vec3, b: Vec3) f32 {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    pub fn cross(a: Vec3, b: Vec3) Vec3 {
        return .{
            .x = a.y * b.z - a.z * b.y,
            .y = a.z * b.x - a.x * b.z,
            .z = a.x * b.y - a.y * b.x,
        };
    }

    pub fn lengthSq(v: Vec3) f32 {
        return v.dot(v);
    }

    pub fn length(v: Vec3) f32 {
        return @sqrt(v.lengthSq());
    }

    pub fn normalize(v: Vec3) Vec3 {
        const len = v.length();
        if (len == 0.0) return Vec3.zero;
        return v.scale(1.0 / len);
    }

    pub fn lerp(a: Vec3, b: Vec3, t: f32) Vec3 {
        return .{
            .x = a.x + (b.x - a.x) * t,
            .y = a.y + (b.y - a.y) * t,
            .z = a.z + (b.z - a.z) * t,
        };
    }

    pub fn distance(a: Vec3, b: Vec3) f32 {
        return a.sub(b).length();
    }

    pub fn distanceSq(a: Vec3, b: Vec3) f32 {
        return a.sub(b).lengthSq();
    }

    pub inline fn toSimd(v: Vec3) @Vector(4, f32) {
        return .{ v.x, v.y, v.z, 0.0 };
    }

    pub inline fn fromSimd(v: @Vector(4, f32)) Vec3 {
        return .{ .x = v[0], .y = v[1], .z = v[2] };
    }

    pub inline fn eql(a: Vec3, b: Vec3) bool {
        return a.x == b.x and a.y == b.y and a.z == b.z;
    }

    pub inline fn toArray(v: Vec3) [3]f32 {
        return .{ v.x, v.y, v.z };
    }
};

pub const Vec4 = extern struct {
    x: f32 = 0.0,
    y: f32 = 0.0,
    z: f32 = 0.0,
    w: f32 = 0.0,

    pub const zero = Vec4{ .x = 0, .y = 0, .z = 0, .w = 0 };
    pub const one = Vec4{ .x = 1, .y = 1, .z = 1, .w = 1 };

    pub fn new(x: f32, y: f32, z: f32, w: f32) Vec4 {
        return .{ .x = x, .y = y, .z = z, .w = w };
    }

    pub inline fn toSimd(v: Vec4) @Vector(4, f32) {
        return .{ v.x, v.y, v.z, v.w };
    }

    pub inline fn fromSimd(v: @Vector(4, f32)) Vec4 {
        return .{ .x = v[0], .y = v[1], .z = v[2], .w = v[3] };
    }

    pub inline fn toArray(v: Vec4) [4]f32 {
        return .{ v.x, v.y, v.z, v.w };
    }

    pub inline fn add(a: Vec4, b: Vec4) Vec4 {
        return fromSimd(a.toSimd() + b.toSimd());
    }

    pub inline fn sub(a: Vec4, b: Vec4) Vec4 {
        return fromSimd(a.toSimd() - b.toSimd());
    }

    pub inline fn scale(v: Vec4, s: f32) Vec4 {
        const vs: @Vector(4, f32) = @splat(s);
        return fromSimd(v.toSimd() * vs);
    }

    pub inline fn lerp(a: Vec4, b: Vec4, t: f32) Vec4 {
        const va = a.toSimd();
        const vb = b.toSimd();
        const vt: @Vector(4, f32) = @splat(t);
        return fromSimd(va + (vb - va) * vt);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
