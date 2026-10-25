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

    pub inline fn add(a: Vec3, b: Vec3) Vec3 {
        return fromSimd(a.toSimd() + b.toSimd());
    }

    pub inline fn sub(a: Vec3, b: Vec3) Vec3 {
        return fromSimd(a.toSimd() - b.toSimd());
    }

    pub inline fn scale(v: Vec3, s: f32) Vec3 {
        return fromSimd(v.toSimd() * @as(@Vector(4, f32), @splat(s)));
    }

    pub inline fn mulAdd(a: Vec3, s: f32, b: Vec3) Vec3 {
        return fromSimd(@mulAdd(@Vector(4, f32), a.toSimd(), @as(@Vector(4, f32), @splat(s)), b.toSimd()));
    }

    pub inline fn dot(a: Vec3, b: Vec3) f32 {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    pub inline fn cross(a: Vec3, b: Vec3) Vec3 {
        return .{
            .x = a.y * b.z - a.z * b.y,
            .y = a.z * b.x - a.x * b.z,
            .z = a.x * b.y - a.y * b.x,
        };
    }

    pub inline fn lengthSq(v: Vec3) f32 {
        return v.dot(v);
    }

    pub inline fn length(v: Vec3) f32 {
        return @sqrt(v.lengthSq());
    }

    pub inline fn normalize(v: Vec3) Vec3 {
        const len = v.length();
        if (len == 0.0) return Vec3.zero;
        return v.scale(1.0 / len);
    }

    pub fn lerp(a: Vec3, b: Vec3, t: f32) Vec3 {
        const va: @Vector(4, f32) = .{ a.x, a.y, a.z, 0.0 };
        const vb: @Vector(4, f32) = .{ b.x, b.y, b.z, 0.0 };
        const vt: @Vector(4, f32) = @splat(t);
        const vr = va + (vb - va) * vt;
        return .{ .x = vr[0], .y = vr[1], .z = vr[2] };
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
