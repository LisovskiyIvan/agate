const std = @import("std");
const vec = @import("vec.zig");
const Vec3 = vec.Vec3;
const bounding_box = @import("bounding_box.zig");
const BoundingBox = bounding_box.BoundingBox;
const ray = @import("ray.zig");
const Ray = ray.Ray;

test "Ray basic operations" {
    const r = Ray.new(Vec3.new(0, 0, -5), Vec3.new(0, 0, 1));
    const pt = r.getPoint(5.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), pt.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), pt.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), pt.z, 1e-5);
}

test "Ray AABB intersection" {
    const box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1));
    const r_hit = Ray.new(Vec3.new(0, 0, -5), Vec3.new(0, 0, 1));
    const hit = r_hit.intersectsAABBNormal(box);
    try std.testing.expect(hit != null);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), hit.?.distance, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), hit.?.point.z, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), hit.?.normal.z, 1e-5);

    const r_miss = Ray.new(Vec3.new(5, 5, -5), Vec3.new(0, 0, 1));
    try std.testing.expect(r_miss.intersectsAABBNormal(box) == null);
}

test "Ray Sphere intersection" {
    const center = Vec3.new(0, 0, 0);
    const radius: f32 = 1.0;
    const r_hit = Ray.new(Vec3.new(0, 0, -3), Vec3.new(0, 0, 1));
    const hit = r_hit.intersectsSphereNormal(center, radius);
    try std.testing.expect(hit != null);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), hit.?.distance, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), hit.?.point.z, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), hit.?.normal.z, 1e-5);

    const r_miss = Ray.new(Vec3.new(2, 2, -3), Vec3.new(0, 0, 1));
    try std.testing.expect(r_miss.intersectsSphereNormal(center, radius) == null);
}

test "Ray Triangle intersection" {
    const v0 = Vec3.new(-1, -1, 0);
    const v1 = Vec3.new(1, -1, 0);
    const v2 = Vec3.new(0, 1, 0);
    const r_hit = Ray.new(Vec3.new(0, 0, -2), Vec3.new(0, 0, 1));
    const hit = r_hit.intersectsTriangle(v0, v1, v2);
    try std.testing.expect(hit != null);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), hit.?.distance, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), hit.?.point.z, 1e-5);

    const r_miss = Vec3.new(5, 5, -2);
    const r_miss_ray = Ray.new(r_miss, Vec3.new(0, 0, 1));
    try std.testing.expect(r_miss_ray.intersectsTriangle(v0, v1, v2) == null);
}

test "Ray Plane intersection" {
    const r = Ray.new(Vec3.new(0, 10, 0), Vec3.new(0, -1, 0));
    const dist = r.intersectsPlane(Vec3.new(0, 0, 0), Vec3.new(0, 1, 0));
    try std.testing.expect(dist != null);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), dist.?, 1e-5);
}
