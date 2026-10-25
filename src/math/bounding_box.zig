const std = @import("std");
const Vec3 = @import("vec.zig").Vec3;
const Mat4 = @import("mat4.zig").Mat4;

pub const BoundingBox = struct {
    min: Vec3,
    max: Vec3,

    pub const zero = BoundingBox{
        .min = Vec3.zero,
        .max = Vec3.zero,
    };

    pub fn init(min_v: Vec3, max_v: Vec3) BoundingBox {
        return .{ .min = min_v, .max = max_v };
    }

    pub fn isValid(self: BoundingBox) bool {
        const dx = self.max.x - self.min.x;
        const dy = self.max.y - self.min.y;
        const dz = self.max.z - self.min.z;
        return dx >= 0.0 and dy >= 0.0 and dz >= 0.0 and (dx > 0.0 or dy > 0.0 or dz > 0.0);
    }

    pub fn center(self: BoundingBox) Vec3 {
        return Vec3.new(
            (self.min.x + self.max.x) * 0.5,
            (self.min.y + self.max.y) * 0.5,
            (self.min.z + self.max.z) * 0.5,
        );
    }

    pub fn extents(self: BoundingBox) Vec3 {
        return Vec3.new(
            (self.max.x - self.min.x) * 0.5,
            (self.max.y - self.min.y) * 0.5,
            (self.max.z - self.min.z) * 0.5,
        );
    }

    pub fn corners(self: BoundingBox) [8]Vec3 {
        return .{
            Vec3.new(self.min.x, self.min.y, self.min.z),
            Vec3.new(self.max.x, self.min.y, self.min.z),
            Vec3.new(self.max.x, self.max.y, self.min.z),
            Vec3.new(self.min.x, self.max.y, self.min.z),
            Vec3.new(self.min.x, self.min.y, self.max.z),
            Vec3.new(self.max.x, self.min.y, self.max.z),
            Vec3.new(self.max.x, self.max.y, self.max.z),
            Vec3.new(self.min.x, self.max.y, self.max.z),
        };
    }

    /// Transforms an AABB by a 4x4 matrix and computes the tightest new AABB using SIMD.
    pub fn transform(self: BoundingBox, m: Mat4) BoundingBox {
        const min_in: @Vector(4, f32) = .{ self.min.x, self.min.y, self.min.z, 0.0 };
        const max_in: @Vector(4, f32) = .{ self.max.x, self.max.y, self.max.z, 0.0 };
        const half: @Vector(4, f32) = @splat(0.5);
        const c_v = (min_in + max_in) * half;
        const e_v = (max_in - min_in) * half;

        const col0: @Vector(4, f32) = m.m[0..4].*;
        const col1: @Vector(4, f32) = m.m[4..8].*;
        const col2: @Vector(4, f32) = m.m[8..12].*;
        const col3: @Vector(4, f32) = m.m[12..16].*;

        const tc_v = @mulAdd(
            @Vector(4, f32),
            col0,
            @splat(c_v[0]),
            @mulAdd(
                @Vector(4, f32),
                col1,
                @splat(c_v[1]),
                @mulAdd(@Vector(4, f32), col2, @splat(c_v[2]), col3),
            ),
        );

        const te_v = @mulAdd(
            @Vector(4, f32),
            @abs(col0),
            @splat(e_v[0]),
            @mulAdd(
                @Vector(4, f32),
                @abs(col1),
                @splat(e_v[1]),
                @abs(col2) * @as(@Vector(4, f32), @splat(e_v[2])),
            ),
        );

        const min_v = tc_v - te_v;
        const max_v = tc_v + te_v;

        return .{
            .min = Vec3.new(min_v[0], min_v[1], min_v[2]),
            .max = Vec3.new(max_v[0], max_v[1], max_v[2]),
        };
    }

    pub fn intersects(self: BoundingBox, other: BoundingBox) bool {
        return (self.min.x <= other.max.x and self.max.x >= other.min.x) and
            (self.min.y <= other.max.y and self.max.y >= other.min.y) and
            (self.min.z <= other.max.z and self.max.z >= other.min.z);
    }

    pub fn merge(self: BoundingBox, other: BoundingBox) BoundingBox {
        return .{
            .min = Vec3.new(
                @min(self.min.x, other.min.x),
                @min(self.min.y, other.min.y),
                @min(self.min.z, other.min.z),
            ),
            .max = Vec3.new(
                @max(self.max.x, other.max.x),
                @max(self.max.y, other.max.y),
                @max(self.max.z, other.max.z),
            ),
        };
    }

    pub fn containsPoint(self: BoundingBox, pt: Vec3) bool {
        return (pt.x >= self.min.x and pt.x <= self.max.x) and
            (pt.y >= self.min.y and pt.y <= self.max.y) and
            (pt.z >= self.min.z and pt.z <= self.max.z);
    }

    pub fn closestPoint(self: BoundingBox, pt: Vec3) Vec3 {
        return Vec3.new(
            std.math.clamp(pt.x, self.min.x, self.max.x),
            std.math.clamp(pt.y, self.min.y, self.max.y),
            std.math.clamp(pt.z, self.min.z, self.max.z),
        );
    }
};
