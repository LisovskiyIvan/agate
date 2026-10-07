const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const csg_mod = @import("csg.zig");
const CSG = csg_mod.CSG;
const CSGPlane = csg_mod.CSGPlane;
const CSGVertex = csg_mod.CSGVertex;
const CSGPolygon = csg_mod.CSGPolygon;

test "CSGPlane: fromPoints and splitPolygon" {
    const a = Vec3.new(0.0, 0.0, 0.0);
    const b = Vec3.new(1.0, 0.0, 0.0);
    const c = Vec3.new(0.0, 1.0, 0.0);
    var plane = CSGPlane.fromPoints(a, b, c);

    // Plane normal should be (0, 0, 1) and w = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), plane.normal.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), plane.normal.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), plane.normal.z, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), plane.w, 1e-4);

    plane.flip();
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), plane.normal.z, 1e-4);
}

test "CSG: Box creation and conversion to GeometryData" {
    const allocator = std.testing.allocator;

    var box = try CSG.fromBox(allocator, .{
        .width = 2.0,
        .height = 2.0,
        .depth = 2.0,
    }, null);
    defer box.deinit();

    try std.testing.expect(box.polygons.items.len > 0);

    var geom = try box.toGeometryData(allocator);
    defer geom.deinit(allocator);

    // Box has 6 faces * 2 triangles = 12 triangles = 36 vertices
    try std.testing.expectEqual(@as(usize, 36), geom.vertices.len);
    try std.testing.expectEqual(@as(usize, 36), geom.indices.len);

    // Verify bounds are approximately [-1, 1]
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), geom.bounds.min.x, 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), geom.bounds.max.x, 1e-3);
}

test "CSG: Union of non-overlapping boxes" {
    const allocator = std.testing.allocator;

    var b1 = try CSG.fromBox(allocator, .{ .width = 1.0, .height = 1.0, .depth = 1.0 }, Mat4.translation(Vec3.new(-2.0, 0.0, 0.0)));
    defer b1.deinit();

    var b2 = try CSG.fromBox(allocator, .{ .width = 1.0, .height = 1.0, .depth = 1.0 }, Mat4.translation(Vec3.new(2.0, 0.0, 0.0)));
    defer b2.deinit();

    var u = try b1.unionWith(&b2);
    defer u.deinit();

    var geom = try u.toGeometryData(allocator);
    defer geom.deinit(allocator);

    // Two non-overlapping cubes: total 24 triangles = 72 vertices
    try std.testing.expectEqual(@as(usize, 72), geom.vertices.len);
}

test "CSG: Subtraction carving notch" {
    const allocator = std.testing.allocator;

    // Main box at origin size 2x2x2
    var main_box = try CSG.fromBox(allocator, .{ .width = 2.0, .height = 2.0, .depth = 2.0 }, null);
    defer main_box.deinit();

    // Cutter box shifted so it overlaps a corner of the main box
    var cutter = try CSG.fromBox(allocator, .{ .width = 1.2, .height = 1.2, .depth = 1.2 }, Mat4.translation(Vec3.new(0.6, 0.6, 0.6)));
    defer cutter.deinit();

    var carved = try main_box.subtract(&cutter);
    defer carved.deinit();

    var geom = try carved.toGeometryData(allocator);
    defer geom.deinit(allocator);

    // Subtraction produces new faces along the cut boundary
    try std.testing.expect(geom.vertices.len > 36);
    try std.testing.expect(geom.indices.len > 36);
}

test "CSG: Intersection of disjoint solids is empty" {
    const allocator = std.testing.allocator;

    var b1 = try CSG.fromBox(allocator, .{ .width = 1.0, .height = 1.0, .depth = 1.0 }, Mat4.translation(Vec3.new(-5.0, 0.0, 0.0)));
    defer b1.deinit();

    var b2 = try CSG.fromBox(allocator, .{ .width = 1.0, .height = 1.0, .depth = 1.0 }, Mat4.translation(Vec3.new(5.0, 0.0, 0.0)));
    defer b2.deinit();

    var inter = try b1.intersect(&b2);
    defer inter.deinit();

    var geom = try inter.toGeometryData(allocator);
    defer geom.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), geom.vertices.len);
}

test "CSG: Intersection of overlapping boxes produces solid" {
    const allocator = std.testing.allocator;

    var b1 = try CSG.fromBox(allocator, .{ .width = 2.0, .height = 2.0, .depth = 2.0 }, null);
    defer b1.deinit();

    var b2 = try CSG.fromBox(allocator, .{ .width = 2.0, .height = 2.0, .depth = 2.0 }, Mat4.translation(Vec3.new(0.5, 0.5, 0.5)));
    defer b2.deinit();

    var inter = try b1.intersect(&b2);
    defer inter.deinit();

    var geom = try inter.toGeometryData(allocator);
    defer geom.deinit(allocator);

    try std.testing.expect(geom.vertices.len > 0);
    // Intersection bounds must be strictly within both original bounds
    try std.testing.expect(geom.bounds.min.x >= -1.001 and geom.bounds.min.x <= 0.501);
    try std.testing.expect(geom.bounds.max.x <= 1.001);
}

test "CSG: Cylinder and Sphere constructors" {
    const allocator = std.testing.allocator;

    var cyl = try CSG.fromCylinder(allocator, .{
        .height = 2.0,
        .diameter = 1.0,
        .tessellation = 8,
    }, null);
    defer cyl.deinit();

    try std.testing.expect(cyl.polygons.items.len > 0);

    var sph = try CSG.fromSphere(allocator, .{
        .diameter = 1.5,
        .segments = 8,
    }, null);
    defer sph.deinit();

    try std.testing.expect(sph.polygons.items.len > 0);

    // Carve cylinder through sphere
    var drilled = try sph.subtract(&cyl);
    defer drilled.deinit();

    var geom = try drilled.toGeometryData(allocator);
    defer geom.deinit(allocator);

    try std.testing.expect(geom.vertices.len > 0);
}

test "CSG: Multi-operation chain (Cube intersect Sphere then subtract Cylinder)" {
    const allocator = std.testing.allocator;

    var cube = try CSG.fromBox(allocator, .{ .width = 1.4, .height = 1.4, .depth = 1.4 }, null);
    defer cube.deinit();

    var sphere = try CSG.fromSphere(allocator, .{ .diameter = 1.8, .segments = 8 }, null);
    defer sphere.deinit();

    var rounded = try cube.intersect(&sphere);
    defer rounded.deinit();

    var cyl = try CSG.fromCylinder(allocator, .{ .height = 2.0, .diameter = 0.6, .tessellation = 8 }, null);
    defer cyl.deinit();

    var sculpt = try rounded.subtract(&cyl);
    defer sculpt.deinit();

    var geom = try sculpt.toGeometryData(allocator);
    defer geom.deinit(allocator);

    try std.testing.expect(geom.vertices.len > 0);
    try std.testing.expect(geom.indices.len > 0);
}
