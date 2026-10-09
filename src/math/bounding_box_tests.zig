const std = @import("std");
const Vec3 = @import("vec.zig").Vec3;
const Mat4 = @import("mat4.zig").Mat4;
const BoundingBox = @import("bounding_box.zig").BoundingBox;

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
