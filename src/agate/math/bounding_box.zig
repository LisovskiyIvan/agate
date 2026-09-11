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

    /// Transforms an AABB by a 4x4 matrix and computes the tightest new AABB.
    pub fn transform(self: BoundingBox, m: Mat4) BoundingBox {
        const c = self.center();
        const e = self.extents();

        // Transform center: M * vec4(c, 1.0)
        // m is column-major: m[col * 4 + row]
        const cx = m.m[0] * c.x + m.m[4] * c.y + m.m[8] * c.z + m.m[12];
        const cy = m.m[1] * c.x + m.m[5] * c.y + m.m[9] * c.z + m.m[13];
        const cz = m.m[2] * c.x + m.m[6] * c.y + m.m[10] * c.z + m.m[14];
        const tc = Vec3.new(cx, cy, cz);

        // Transform extents: sum(|M_ij| * e_j)
        const ex = @abs(m.m[0]) * e.x + @abs(m.m[4]) * e.y + @abs(m.m[8]) * e.z;
        const ey = @abs(m.m[1]) * e.x + @abs(m.m[5]) * e.y + @abs(m.m[9]) * e.z;
        const ez = @abs(m.m[2]) * e.x + @abs(m.m[6]) * e.y + @abs(m.m[10]) * e.z;

        return .{
            .min = Vec3.new(tc.x - ex, tc.y - ey, tc.z - ez),
            .max = Vec3.new(tc.x + ex, tc.y + ey, tc.z + ez),
        };
    }

    pub fn intersects(self: BoundingBox, other: BoundingBox) bool {
        return (self.min.x <= other.max.x and self.max.x >= other.min.x) and
            (self.min.y <= other.max.y and self.max.y >= other.min.y) and
            (self.min.z <= other.max.z and self.max.z >= other.min.z);
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
