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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "BoundingBox init, validity, center, extents and corners" {
    const min_p = Vec3.new(-2.0, -4.0, -6.0);
    const max_p = Vec3.new(2.0, 4.0, 6.0);
    const box = BoundingBox.init(min_p, max_p);

    try std.testing.expect(box.isValid());
    try std.testing.expect(!BoundingBox.zero.isValid());

    // Inverted box is invalid
    const inv = BoundingBox.init(max_p, min_p);
    try std.testing.expect(!inv.isValid());

    // Center and extents
    try std.testing.expectEqual(Vec3.zero, box.center());
    try std.testing.expectEqual(Vec3.new(2.0, 4.0, 6.0), box.extents());

    // Corners check
    const corn = box.corners();
    try std.testing.expectEqual(8, corn.len);
    try std.testing.expectEqual(min_p, corn[0]);
    try std.testing.expectEqual(max_p, corn[6]);
}

test "BoundingBox containsPoint, closestPoint and intersects" {
    const box = BoundingBox.init(Vec3.new(-1.0, -1.0, -1.0), Vec3.new(1.0, 1.0, 1.0));

    // Points
    try std.testing.expect(box.containsPoint(Vec3.zero));
    try std.testing.expect(box.containsPoint(Vec3.new(1.0, 0.0, -0.5)));
    try std.testing.expect(!box.containsPoint(Vec3.new(1.1, 0.0, 0.0)));
    try std.testing.expect(!box.containsPoint(Vec3.new(0.0, -1.5, 0.0)));

    // Closest point
    const clamped = box.closestPoint(Vec3.new(5.0, -3.0, 0.5));
    try std.testing.expectEqual(Vec3.new(1.0, -1.0, 0.5), clamped);

    // Intersection
    const overlapping = BoundingBox.init(Vec3.new(0.5, 0.5, 0.5), Vec3.new(2.0, 2.0, 2.0));
    try std.testing.expect(box.intersects(overlapping));
    try std.testing.expect(overlapping.intersects(box));

    const disjoint = BoundingBox.init(Vec3.new(2.0, 2.0, 2.0), Vec3.new(3.0, 3.0, 3.0));
    try std.testing.expect(!box.intersects(disjoint));
    try std.testing.expect(!disjoint.intersects(box));
}

test "BoundingBox merge and transform by matrix" {
    const b1 = BoundingBox.init(Vec3.new(-1.0, 0.0, 0.0), Vec3.new(1.0, 2.0, 1.0));
    const b2 = BoundingBox.init(Vec3.new(0.0, -1.0, -2.0), Vec3.new(2.0, 1.0, 0.0));

    const merged = b1.merge(b2);
    try std.testing.expectEqual(Vec3.new(-1.0, -1.0, -2.0), merged.min);
    try std.testing.expectEqual(Vec3.new(2.0, 2.0, 1.0), merged.max);

    // Transform by translation matrix
    const trans = Mat4.translation(Vec3.new(10.0, 20.0, 30.0));
    const box = BoundingBox.init(Vec3.new(-1.0, -1.0, -1.0), Vec3.new(1.0, 1.0, 1.0));
    const transformed = box.transform(trans);

    try std.testing.expectApproxEqAbs(@as(f32, 9.0), transformed.min.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 19.0), transformed.min.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 29.0), transformed.min.z, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), transformed.max.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 21.0), transformed.max.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 31.0), transformed.max.z, 1e-5);
}
