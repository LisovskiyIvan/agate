const std = @import("std");
const vec = @import("vec.zig");
const Vec3 = vec.Vec3;
const Mat4 = @import("mat4.zig").Mat4;
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

test "Ray Sphere intersection with non-unit direction" {
    const center = Vec3.new(0, 0, 0);

    // Uniformly scaled ray (factor 2): entry on the radius-2 sphere at
    // z=-2, ray parameter t=2.
    const r_scaled = Ray{ .origin = Vec3.new(0, 0, -6), .direction = Vec3.new(0, 0, 2) };
    const hit_scaled = r_scaled.intersectsSphereNormal(center, 2.0);
    try std.testing.expect(hit_scaled != null);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), hit_scaled.?.distance, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), hit_scaled.?.point.z, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), hit_scaled.?.normal.z, 1e-5);

    // Non-uniform scale factor 3 along z: t=4/3 still lands on z=-2.
    const r_nu = Ray{ .origin = Vec3.new(0, 0, -6), .direction = Vec3.new(0, 0, 3) };
    const hit_nu = r_nu.intersectsSphereNormal(center, 2.0);
    try std.testing.expect(hit_nu != null);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0 / 3.0), hit_nu.?.distance, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), hit_nu.?.point.z, 1e-5);

    // Inside origin with scaled direction: exit at z=1, t=0.5.
    const r_inside = Ray{ .origin = Vec3.zero, .direction = Vec3.new(0, 0, 2) };
    const hit_inside = r_inside.intersectsSphereNormal(center, 1.0);
    try std.testing.expect(hit_inside != null);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), hit_inside.?.distance, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), hit_inside.?.point.z, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), hit_inside.?.normal.z, 1e-5);

    // Tangent with scaled direction: single root t=1.5 at (1,0,0).
    const r_tan = Ray{ .origin = Vec3.new(1, 0, -3), .direction = Vec3.new(0, 0, 2) };
    const hit_tan = r_tan.intersectsSphereNormal(center, 1.0);
    try std.testing.expect(hit_tan != null);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), hit_tan.?.distance, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), hit_tan.?.point.z, 1e-5);

    // Miss and backwards rays stay misses under scale.
    try std.testing.expect((Ray{ .origin = Vec3.new(2, 2, -3), .direction = Vec3.new(0, 0, 2) }).intersectsSphere(center, 1.0) == null);
    try std.testing.expect((Ray{ .origin = Vec3.new(0, 0, -3), .direction = Vec3.new(0, 0, -2) }).intersectsSphere(center, 1.0) == null);

    // Zero-length direction: no hit, never divide by zero.
    try std.testing.expect((Ray{ .origin = Vec3.new(0, 0, -3), .direction = Vec3.zero }).intersectsSphere(center, 1.0) == null);

    // Tiny but nonzero direction: t=1e7 lands on z=-1 (no magnitude cutoff).
    const r_tiny = Ray{ .origin = Vec3.new(0, 0, -2), .direction = Vec3.new(0, 0, 1e-7) };
    const hit_tiny = r_tiny.intersectsSphereNormal(center, 1.0);
    try std.testing.expect(hit_tiny != null);
    try std.testing.expectApproxEqRel(@as(f32, 1e7), hit_tiny.?.distance, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), hit_tiny.?.point.z, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), hit_tiny.?.normal.z, 1e-5);
}

test "Ray Sphere intersection after Ray.transform with scale" {
    const center = Vec3.new(0, 0, 0);
    // Uniform scale by 2: origin z=-6, entry on radius-2 sphere at z=-2.
    const r2 = Ray.new(Vec3.new(0, 0, -3), Vec3.new(0, 0, 1)).transform(Mat4.scaling(Vec3.new(2, 2, 2)));
    const hit2 = r2.intersectsSphereNormal(center, 2.0);
    try std.testing.expect(hit2 != null);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), hit2.?.distance, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), hit2.?.point.z, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), hit2.?.normal.z, 1e-5);

    // Non-uniform scale (1,1,4): origin z=-12, entry on radius-4 sphere at z=-4.
    const r4 = Ray.new(Vec3.new(0, 0, -3), Vec3.new(0, 0, 1)).transform(Mat4.scaling(Vec3.new(1, 1, 4)));
    const hit4 = r4.intersectsSphereNormal(center, 4.0);
    try std.testing.expect(hit4 != null);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), hit4.?.distance, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -4.0), hit4.?.point.z, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), hit4.?.normal.z, 1e-5);
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
