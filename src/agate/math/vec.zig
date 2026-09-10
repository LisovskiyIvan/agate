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
