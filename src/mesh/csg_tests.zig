const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Vec2 = math.Vec2;
const Mat4 = math.Mat4;
const BoundingBox = math.BoundingBox;
const csg_mod = @import("csg.zig");
const CSG = csg_mod.CSG;
const CSGVertex = csg_mod.CSGVertex;
const types = @import("types.zig");
const Vertex = types.Vertex;
const GeometryData = types.GeometryData;
const builders = @import("builders.zig");

test "CSG: interpolate splits uv and uv1 independently" {
    const a = CSGVertex{
        .pos = Vec3.new(0, 0, 0),
        .normal = Vec3.new(0, 0, 1),
        .uv = Vec2.new(0, 0),
        .uv1 = Vec2.new(5, 7),
    };
    const b = CSGVertex{
        .pos = Vec3.new(1, 0, 0),
        .normal = Vec3.new(0, 0, 1),
        .uv = Vec2.new(1, 1),
        .uv1 = Vec2.new(15, 17),
    };
    const m = a.interpolate(b, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), m.uv.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), m.uv.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), m.uv1.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), m.uv1.y, 1e-6);
}

test "CSG: multiUV uv1 survives fromGeometryData/toGeometryData roundtrip" {
    const ally = std.testing.allocator;
    var quad_verts = [_]Vertex{
        .{ .position = .{ -1, -1, 0 }, .normal = .{ 0, 0, 1 }, .color = .{ 1, 1, 1, 1 }, .uv = .{ 0, 0 }, .uv1 = .{ 5, 7 } },
        .{ .position = .{ 1, -1, 0 }, .normal = .{ 0, 0, 1 }, .color = .{ 1, 1, 1, 1 }, .uv = .{ 1, 0 }, .uv1 = .{ 15, 7 } },
        .{ .position = .{ 1, 1, 0 }, .normal = .{ 0, 0, 1 }, .color = .{ 1, 1, 1, 1 }, .uv = .{ 1, 1 }, .uv1 = .{ 15, 17 } },
        .{ .position = .{ -1, 1, 0 }, .normal = .{ 0, 0, 1 }, .color = .{ 1, 1, 1, 1 }, .uv = .{ 0, 1 }, .uv1 = .{ 5, 17 } },
    };
    var quad_idx = [_]u32{ 0, 1, 2, 0, 2, 3 };
    const quad = GeometryData{
        .vertices = &quad_verts,
        .indices = &quad_idx,
        .bounds = BoundingBox.init(Vec3.new(-1, -1, 0), Vec3.new(1, 1, 0)),
    };
    var csg = try CSG.fromGeometryData(ally, quad, null);
    defer csg.deinit();
    var out = try csg.toGeometryData(ally);
    defer out.deinit(ally);

    try std.testing.expectEqual(@as(usize, 6), out.vertices.len);
    // Fan triangulation preserves the input triangles exactly; each output
    // uv1 must match one of the distinct input uv1 corners.
    const corners = [4][2]f32{ .{ 5, 7 }, .{ 15, 7 }, .{ 15, 17 }, .{ 5, 17 } };
    for (out.vertices) |v| {
        var hit = false;
        for (corners) |c| {
            if (@abs(v.uv1[0] - c[0]) < 1e-5 and @abs(v.uv1[1] - c[1]) < 1e-5) hit = true;
        }
        try std.testing.expect(hit);
        // Distinct from uv0 (uv in [0,1], uv1 in [5,15]x[7,17]).
        try std.testing.expect(v.uv1[0] > 1.0 and v.uv1[1] > 1.0);
    }
}

test "CSG: multiUV uv1 survives real boolean union" {
    const ally = std.testing.allocator;
    var box_a = try builders.buildBoxData(ally, .{ .width = 2.0, .height = 2.0, .depth = 2.0 });
    defer box_a.deinit(ally);
    var box_b = try builders.buildBoxData(ally, .{ .width = 2.0, .height = 2.0, .depth = 2.0 });
    defer box_b.deinit(ally);
    // Paint distinct channels: uv stays builder-provided, uv1 is offset far away.
    for (box_a.vertices) |*v| v.uv1 = .{ v.uv[0] * 10.0 + 5.0, v.uv[1] * 10.0 + 7.0 };
    for (box_b.vertices) |*v| v.uv1 = .{ v.uv[0] * 10.0 + 5.0, v.uv[1] * 10.0 + 7.0 };

    var csg_a = try CSG.fromGeometryData(ally, box_a, null);
    defer csg_a.deinit();
    var csg_b = try CSG.fromGeometryData(ally, box_b, Mat4.translation(Vec3.new(1, 0, 0)));
    defer csg_b.deinit();

    var joined = try csg_a.unionWith(&csg_b);
    defer joined.deinit();
    var out = try joined.toGeometryData(ally);
    defer out.deinit(ally);

    try std.testing.expect(out.vertices.len > 0);
    // Split planes interpolate new verts: every output uv1 must lie within
    // the painted input range and keep the 10x+offset split from uv0.
    var saw_nonzero_uv1 = false;
    for (out.vertices) |v| {
        try std.testing.expect(v.uv1[0] >= 5.0 - 1e-3 and v.uv1[0] <= 15.0 + 1e-3);
        try std.testing.expect(v.uv1[1] >= 7.0 - 1e-3 and v.uv1[1] <= 17.0 + 1e-3);
        try std.testing.expectApproxEqAbs(10.0 * v.uv[0] + 5.0, v.uv1[0], 1e-3);
        try std.testing.expectApproxEqAbs(10.0 * v.uv[1] + 7.0, v.uv1[1], 1e-3);
        if (v.uv1[0] != 0.0 or v.uv1[1] != 0.0) saw_nonzero_uv1 = true;
    }
    try std.testing.expect(saw_nonzero_uv1);
}
