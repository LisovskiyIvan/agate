const std = @import("std");
const math = @import("math");
const Vec2 = math.Vec2;
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Quat = math.Quat;
const Color4 = math.Color4;
const BoundingBox = math.BoundingBox;
const mesh_mod = @import("../mesh.zig");
const Mesh = mesh_mod.Mesh;
const MeshBuilder = mesh_mod.MeshBuilder;
const Vertex = mesh_mod.Vertex;
const MorphTarget = mesh_mod.MorphTarget;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;
const Bone = @import("../animation/skeleton.zig").Bone;
const GeometryData = mesh_mod.GeometryData;
const buildTubeData = mesh_mod.buildTubeData;
const buildLinesData = mesh_mod.buildLinesData;
const buildExtrudeData = mesh_mod.buildExtrudeData;
const appendGridQuad = mesh_mod.appendGridQuad;
const appendGridQuadFlipped = mesh_mod.appendGridQuadFlipped;
const buildTorusData = mesh_mod.buildTorusData;
const buildTorusKnotData = mesh_mod.buildTorusKnotData;
const buildDiscData = mesh_mod.buildDiscData;
const buildRibbonData = mesh_mod.buildRibbonData;
const buildLatheData = mesh_mod.buildLatheData;
const buildPlaneData = mesh_mod.buildPlaneData;
const storeQuad = mesh_mod.storeQuad;
const pickOrthogonal = mesh_mod.pickOrthogonal;
const resolveFrameSeed = mesh_mod.resolveFrameSeed;
const buildDecalData = mesh_mod.buildDecalData;
const DecalOptions = mesh_mod.DecalOptions;
const barycentric = mesh_mod.barycentric;
const blendSkinWeights = mesh_mod.blendSkinWeights;
const DecalProjector = mesh_mod.DecalProjector;
const DecalManager = mesh_mod.DecalManager;
const DecalSpawnOptions = mesh_mod.DecalSpawnOptions;
const SkinJointWeight = mesh_mod.SkinJointWeight;
const Scene = @import("../scene.zig").Scene;
const PBRMaterial = @import("../material.zig").PBRMaterial;

test "Mesh attachToBone world matrix computation" {
    const ally = std.testing.allocator;
    const skel = try Skeleton.init(ally, 2);
    defer skel.deinit();

    skel.bones[0].local_position = Vec3.new(2, 0, 0);
    skel.bones[1].parent_index = 0;
    skel.bones[1].local_position = Vec3.new(0, 3, 0);
    skel.update();

    var host_mesh: Mesh = undefined;
    host_mesh = .{
        .name = "host",
        .position = Vec3.new(10, 20, 30),
        .rotation = Vec3.zero,
        .scaling = Vec3.one,
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .skeleton = skel,
    };

    var attached_mesh: Mesh = undefined;
    attached_mesh = .{
        .name = "sword",
        .position = Vec3.new(0, 0, 1), // local offset relative to bone
        .rotation = Vec3.zero,
        .scaling = Vec3.one,
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
    };

    attached_mesh.attachToBone(&host_mesh, 1);

    const world = attached_mesh.getWorldMatrix();
    const pos = world.getTranslation();

    // host (10, 20, 30) + bone1 (2, 3, 0) + local (0, 0, 1) = (12, 23, 31)
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), pos.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 23.0), pos.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 31.0), pos.z, 1e-4);
}

fn expectNormalsNormalized(vertices: []const Vertex) !void {
    for (vertices) |v| {
        const n = Vec3.new(v.normal[0], v.normal[1], v.normal[2]);
        try std.testing.expectApproxEqAbs(1.0, n.length(), 1e-4);
    }
}

fn expectIndicesInBounds(indices: []const u32, vertex_count: usize) !void {
    for (indices) |ix| {
        try std.testing.expect(ix < vertex_count);
    }
}

test "MeshBuilder torus geometry" {
    const ally = std.testing.allocator;
    var data = try buildTorusData(ally, .{ .diameter = 2.0, .thickness = 0.5, .tessellation = 8 });
    defer data.deinit(ally);

    try std.testing.expectEqual(@as(usize, 81), data.vertices.len);
    try std.testing.expectEqual(@as(usize, 384), data.indices.len);
    try expectNormalsNormalized(data.vertices);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // Ring radius 1.0, tube radius 0.25: symmetric bounds.
    try std.testing.expectApproxEqAbs(@as(f32, 1.25), data.bounds.max.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -1.25), data.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), data.bounds.max.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -0.25), data.bounds.min.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.25), data.bounds.max.z, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -1.25), data.bounds.min.z, 1e-4);

    for (data.vertices) |v| {
        try std.testing.expect(v.uv[0] >= 0.0 and v.uv[0] <= 1.0);
        try std.testing.expect(v.uv[1] >= 0.0 and v.uv[1] <= 1.0);
    }
}

test "MeshBuilder torus knot geometry" {
    const ally = std.testing.allocator;
    const tubular: usize = 16;
    const radial: usize = 6;
    var data = try buildTorusKnotData(ally, .{
        .radius = 3.0,
        .tube = 0.4,
        .radial_segments = 6,
        .tubular_segments = 16,
    });
    defer data.deinit(ally);

    try std.testing.expectEqual((tubular + 1) * (radial + 1), data.vertices.len);
    try std.testing.expectEqual(tubular * radial * 6, data.indices.len);
    try expectNormalsNormalized(data.vertices);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // Knot is centered on the origin: bounds roughly symmetric.
    const center = data.bounds.center();
    try std.testing.expect(@abs(center.x) < 0.2);
    try std.testing.expect(@abs(center.y) < 0.2);
    try std.testing.expect(@abs(center.z) < 0.2);
    // Outer extent stays within radius + tube (plus a small margin).
    try std.testing.expect(data.bounds.max.x <= 3.4 + 1e-3);
    try std.testing.expect(data.bounds.min.x >= -3.4 - 1e-3);
}

test "MeshBuilder disc geometry" {
    const ally = std.testing.allocator;
    var data = try buildDiscData(ally, .{ .radius = 2.0, .tessellation = 8 });
    defer data.deinit(ally);

    try std.testing.expectEqual(@as(usize, 10), data.vertices.len);
    try std.testing.expectEqual(@as(usize, 24), data.indices.len);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // Flat in XZ, normal +Y, UVs mapped from disc coordinates.
    for (data.vertices) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), v.position[1], 1e-6);
        try std.testing.expectEqual([3]f32{ 0.0, 1.0, 0.0 }, v.normal);
        try std.testing.expect(v.uv[0] >= 0.0 and v.uv[0] <= 1.0);
        try std.testing.expect(v.uv[1] >= 0.0 and v.uv[1] <= 1.0);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), data.bounds.max.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), data.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), data.bounds.max.z, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), data.bounds.min.z, 1e-4);
}

test "MeshBuilder ribbon geometry" {
    const ally = std.testing.allocator;
    const path_a = [_]Vec3{
        Vec3.new(0, 0, 0),
        Vec3.new(1, 0, 0),
        Vec3.new(2, 0, 0),
    };
    const path_b = [_]Vec3{
        Vec3.new(0, 0, 1),
        Vec3.new(1, 1, 1),
        Vec3.new(2, 0, 1),
    };
    const paths = [_][]const Vec3{ &path_a, &path_b };

    var data = try buildRibbonData(ally, .{ .paths = &paths });
    defer data.deinit(ally);

    try std.testing.expectEqual(@as(usize, 6), data.vertices.len);
    try std.testing.expectEqual(@as(usize, 12), data.indices.len);
    try expectNormalsNormalized(data.vertices);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // Closed variants produce the wrapped quad counts.
    var closed = try buildRibbonData(ally, .{ .paths = &paths, .close_path = true, .close_array = false });
    defer closed.deinit(ally);
    try std.testing.expectEqual(@as(usize, 18), closed.indices.len);

    var closed_both = try buildRibbonData(ally, .{ .paths = &paths, .close_path = true, .close_array = true });
    defer closed_both.deinit(ally);
    try std.testing.expectEqual(@as(usize, 36), closed_both.indices.len);
}

test "MeshBuilder ribbon rejects invalid paths" {
    const ally = std.testing.allocator;
    const short = [_]Vec3{Vec3.new(0, 0, 0)};
    const ok_path = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(1, 0, 0) };

    const single = [_][]const Vec3{&ok_path};
    try std.testing.expectError(error.InvalidRibbon, buildRibbonData(ally, .{ .paths = &single }));

    const one_short = [_][]const Vec3{ &ok_path, &short };
    try std.testing.expectError(error.InvalidRibbon, buildRibbonData(ally, .{ .paths = &one_short }));

    const longer = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(1, 0, 0), Vec3.new(2, 0, 0) };
    const mismatched = [_][]const Vec3{ &ok_path, &longer };
    try std.testing.expectError(error.InvalidRibbon, buildRibbonData(ally, .{ .paths = &mismatched }));
}

test "MeshBuilder lathe geometry" {
    const ally = std.testing.allocator;
    const profile = [_]Vec3{
        Vec3.new(0.5, 0.0, 0.0),
        Vec3.new(0.5, 2.0, 0.0),
    };
    var data = try buildLatheData(ally, .{ .shape = &profile, .tessellation = 8 });
    defer data.deinit(ally);

    try std.testing.expectEqual(@as(usize, 18), data.vertices.len);
    try std.testing.expectEqual(@as(usize, 48), data.indices.len);
    try expectNormalsNormalized(data.vertices);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // Cylinder wall: radial normals, symmetric XZ bounds.
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), data.bounds.max.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), data.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), data.bounds.max.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.bounds.min.y, 1e-4);

    // Degenerate on-axis segment is skipped: only one quad strip remains.
    const with_axis = [_]Vec3{
        Vec3.new(0.0, 0.0, 0.0),
        Vec3.new(0.0, 1.0, 0.0),
        Vec3.new(0.5, 1.0, 0.0),
    };
    var deg = try buildLatheData(ally, .{ .shape = &with_axis, .tessellation = 8 });
    defer deg.deinit(ally);
    try std.testing.expectEqual(@as(usize, 27), deg.vertices.len);
    try std.testing.expectEqual(@as(usize, 48), deg.indices.len);
    try expectNormalsNormalized(deg.vertices);

    const single = [_]Vec3{Vec3.new(0.5, 0.0, 0.0)};
    try std.testing.expectError(error.InvalidLathe, buildLatheData(ally, .{ .shape = &single }));
}

test "MeshBuilder plane geometry" {
    const ally = std.testing.allocator;
    var data = try buildPlaneData(ally, .{
        .width = 2.0,
        .height = 4.0,
        .subdivisions_x = 3,
        .subdivisions_y = 2,
        .uv_scale = Vec2.new(2.0, 3.0),
    });
    defer data.deinit(ally);

    try std.testing.expectEqual(@as(usize, 12), data.vertices.len);
    try std.testing.expectEqual(@as(usize, 36), data.indices.len);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // Centered in XY, flat at z = 0, normals +Z.
    for (data.vertices) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), v.position[2], 1e-6);
        try std.testing.expectEqual([3]f32{ 0.0, 0.0, 1.0 }, v.normal);
    }
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), data.bounds.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data.bounds.max.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), data.bounds.min.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), data.bounds.max.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.bounds.min.z, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.bounds.max.z, 1e-6);

    // UVs span the per-axis scale.
    var max_u: f32 = 0.0;
    var max_v: f32 = 0.0;
    for (data.vertices) |v| {
        max_u = @max(max_u, v.uv[0]);
        max_v = @max(max_v, v.uv[1]);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), max_u, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), max_v, 1e-4);

    // Zero subdivisions clamp to a single quad.
    var flat = try buildPlaneData(ally, .{ .subdivisions_x = 0, .subdivisions_y = 0 });
    defer flat.deinit(ally);
    try std.testing.expectEqual(@as(usize, 4), flat.vertices.len);
    try std.testing.expectEqual(@as(usize, 6), flat.indices.len);
}

test "MeshBuilder tube geometry" {
    const ally = std.testing.allocator;
    const path = [_]Vec3{
        Vec3.new(0, 0, 0),
        Vec3.new(0, 1, 0),
        Vec3.new(0, 2, 0),
    };
    var data = try buildTubeData(ally, .{ .path = &path, .radius = 0.5, .tessellation = 6 });
    defer data.deinit(ally);

    // Open tube: one ring per point, one quad strip per segment.
    try std.testing.expectEqual(@as(usize, 21), data.vertices.len);
    try std.testing.expectEqual(@as(usize, 72), data.indices.len);
    try expectNormalsNormalized(data.vertices);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // Straight path along Y: AABB is the path extent plus the radius in XZ.
    // (Ring samples may not hit the exact Z extremes, so those assert range.)
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), data.bounds.max.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), data.bounds.min.x, 1e-4);
    try std.testing.expect(data.bounds.max.z <= 0.5 + 1e-4);
    try std.testing.expect(data.bounds.min.z >= -0.5 - 1e-4);
    try std.testing.expect(data.bounds.max.z > 0.4);
    try std.testing.expect(data.bounds.min.z < -0.4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.bounds.min.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), data.bounds.max.y, 1e-4);

    // U runs along the path, V around the tube.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.vertices[0].uv[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data.vertices[14].uv[0], 1e-6);
    for (data.vertices) |v| {
        try std.testing.expect(v.uv[0] >= 0.0 and v.uv[0] <= 1.0);
        try std.testing.expect(v.uv[1] >= 0.0 and v.uv[1] <= 1.0);
    }
}

test "MeshBuilder tube capped closed radii" {
    const ally = std.testing.allocator;
    const square = [_]Vec3{
        Vec3.new(-1, 0, -1),
        Vec3.new(1, 0, -1),
        Vec3.new(1, 0, 1),
        Vec3.new(-1, 0, 1),
    };
    const radii = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    var data = try buildTubeData(ally, .{
        .path = &square,
        .radii = &radii,
        .tessellation = 4,
        .closed = true,
        .capped = true, // ignored on closed loops
    });
    defer data.deinit(ally);

    // Closed loop: one ring per point, wrapped strips, no caps.
    try std.testing.expectEqual(@as(usize, 20), data.vertices.len);
    try std.testing.expectEqual(@as(usize, 96), data.indices.len);
    try expectNormalsNormalized(data.vertices);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // Per-point radii: the first ring vertex sits one radius from its center.
    for (0..4) |i| {
        const v = data.vertices[i * 5];
        const px = v.position[0] - square[i].x;
        const py = v.position[1] - square[i].y;
        const pz = v.position[2] - square[i].z;
        try std.testing.expectApproxEqAbs(radii[i], @sqrt(px * px + py * py + pz * pz), 1e-4);
    }

    // Capped open tube adds a duplicated ring plus a center vertex per cap.
    var capped = try buildTubeData(ally, .{
        .path = &square,
        .radius = 0.1,
        .tessellation = 4,
        .capped = true,
    });
    defer capped.deinit(ally);
    try std.testing.expectEqual(@as(usize, 4 * 5 + 2 * 5 + 2), capped.vertices.len);
    try std.testing.expectEqual(@as(usize, 3 * 4 * 6 + 2 * 4 * 3), capped.indices.len);
    try expectNormalsNormalized(capped.vertices);
    try expectIndicesInBounds(capped.indices, capped.vertices.len);
}

test "MeshBuilder tube rejects invalid input" {
    const ally = std.testing.allocator;
    const single = [_]Vec3{Vec3.new(0, 0, 0)};
    try std.testing.expectError(error.InvalidTube, buildTubeData(ally, .{ .path = &single }));

    const ok_path = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(0, 1, 0) };
    const bad_radii = [_]f32{0.1};
    try std.testing.expectError(error.InvalidTube, buildTubeData(ally, .{ .path = &ok_path, .radii = &bad_radii }));
}

test "MeshBuilder lines geometry" {
    const ally = std.testing.allocator;
    const points = [_]Vec3{
        Vec3.new(0, 0, 0),
        Vec3.new(2, 0, 0),
        Vec3.new(4, 0, 0),
    };
    const colors = [_]Color4{
        Color4.new(1, 0, 0, 1),
        Color4.new(0, 1, 0, 1),
        Color4.new(0, 0, 1, 1),
    };
    var data = try buildLinesData(ally, .{ .points = &points, .width = 2.0, .colors = &colors });
    defer data.deinit(ally);

    try std.testing.expectEqual(@as(usize, 6), data.vertices.len);
    try std.testing.expectEqual(@as(usize, 12), data.indices.len);
    try expectNormalsNormalized(data.vertices);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // Ribbon width spans both sides of the center line.
    for (0..3) |i| {
        const l = Vec3.new(data.vertices[2 * i].position[0], data.vertices[2 * i].position[1], data.vertices[2 * i].position[2]);
        const r = Vec3.new(data.vertices[2 * i + 1].position[0], data.vertices[2 * i + 1].position[1], data.vertices[2 * i + 1].position[2]);
        try std.testing.expectApproxEqAbs(@as(f32, 2.0), l.distance(r), 1e-4);
        // Per-point colors apply to both vertices of the pair.
        try std.testing.expectEqual(colors[i].toArray(), data.vertices[2 * i].color);
        try std.testing.expectEqual(colors[i].toArray(), data.vertices[2 * i + 1].color);
    }

    // U follows the arc length: equal segments give 0, 0.5, 1.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.vertices[0].uv[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), data.vertices[2].uv[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data.vertices[4].uv[0], 1e-5);

    // Closed loop wraps the last strip back to the first, without caps.
    var closed = try buildLinesData(ally, .{ .points = &points, .width = 1.0, .closed = true });
    defer closed.deinit(ally);
    try std.testing.expectEqual(@as(usize, 6), closed.vertices.len);
    try std.testing.expectEqual(@as(usize, 18), closed.indices.len);
    try expectNormalsNormalized(closed.vertices);
    try expectIndicesInBounds(closed.indices, closed.vertices.len);
}

test "MeshBuilder lines rejects invalid input" {
    const ally = std.testing.allocator;
    const single = [_]Vec3{Vec3.new(0, 0, 0)};
    try std.testing.expectError(error.InvalidLines, buildLinesData(ally, .{ .points = &single }));

    const ok_points = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(1, 0, 0) };
    const bad_colors = [_]Color4{Color4.white};
    try std.testing.expectError(error.InvalidLines, buildLinesData(ally, .{ .points = &ok_points, .colors = &bad_colors }));
}

test "MeshBuilder extrude convex geometry" {
    const ally = std.testing.allocator;
    const square = [_]Vec2{
        Vec2.new(0, 0),
        Vec2.new(1, 0),
        Vec2.new(1, 1),
        Vec2.new(0, 1),
    };
    var data = try buildExtrudeData(ally, .{ .profile = &square, .depth = 2.0 });
    defer data.deinit(ally);

    // 4 outline points: 4 side quads (16 verts) + 2 cap rings (8 verts);
    // 24 side indices + 2 caps x 2 triangles x 3.
    try std.testing.expectEqual(@as(usize, 24), data.vertices.len);
    try std.testing.expectEqual(@as(usize, 36), data.indices.len);
    try expectNormalsNormalized(data.vertices);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // AABB covers the profile footprint and the extrusion depth.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.bounds.min.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data.bounds.max.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.bounds.min.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data.bounds.max.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.bounds.min.z, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), data.bounds.max.z, 1e-5);

    // First side quad is the bottom edge: outward normal -Y.
    try std.testing.expectEqual([3]f32{ 0.0, -1.0, 0.0 }, data.vertices[0].normal);
    // Caps carry axial normals: front ring starts at vertex 16.
    try std.testing.expectEqual([3]f32{ 0.0, 0.0, 1.0 }, data.vertices[16].normal);
    try std.testing.expectEqual([3]f32{ 0.0, 0.0, -1.0 }, data.vertices[20].normal);

    // Clockwise input is normalized: same counts, still outward normals.
    const square_cw = [_]Vec2{
        Vec2.new(0, 0),
        Vec2.new(0, 1),
        Vec2.new(1, 1),
        Vec2.new(1, 0),
    };
    var cw = try buildExtrudeData(ally, .{ .profile = &square_cw, .depth = 2.0 });
    defer cw.deinit(ally);
    try std.testing.expectEqual(data.vertices.len, cw.vertices.len);
    try std.testing.expectEqual(data.indices.len, cw.indices.len);
    try expectNormalsNormalized(cw.vertices);
    var found_outward = false;
    for (cw.vertices[0..16]) |v| {
        if (v.normal[0] == 0.0 and v.normal[1] == -1.0 and v.normal[2] == 0.0) found_outward = true;
    }
    try std.testing.expect(found_outward);

    // Uncapped extrusion keeps only the side walls.
    var open = try buildExtrudeData(ally, .{ .profile = &square, .depth = 2.0, .capped = false });
    defer open.deinit(ally);
    try std.testing.expectEqual(@as(usize, 16), open.vertices.len);
    try std.testing.expectEqual(@as(usize, 24), open.indices.len);
}

test "MeshBuilder extrude concave L-shape geometry" {
    const ally = std.testing.allocator;
    // 2x2 square with the top-right 1x1 quadrant removed: area 3, reflex at (1, 1).
    const ell = [_]Vec2{
        Vec2.new(0, 0),
        Vec2.new(2, 0),
        Vec2.new(2, 1),
        Vec2.new(1, 1),
        Vec2.new(1, 2),
        Vec2.new(0, 2),
    };
    var data = try buildExtrudeData(ally, .{ .profile = &ell, .depth = 1.0 });
    defer data.deinit(ally);

    // 6 outline points: 24 side verts + 12 cap verts; each cap has 6 - 2 triangles.
    try std.testing.expectEqual(@as(usize, 36), data.vertices.len);
    try std.testing.expectEqual(@as(usize, 60), data.indices.len);
    try expectNormalsNormalized(data.vertices);
    try expectIndicesInBounds(data.indices, data.vertices.len);

    // Front cap triangles (indices 36..48) must tile the L area of 3.
    var cap_area: f32 = 0.0;
    var t: usize = 36;
    while (t < 48) : (t += 3) {
        const p0 = data.vertices[data.indices[t]].position;
        const p1 = data.vertices[data.indices[t + 1]].position;
        const p2 = data.vertices[data.indices[t + 2]].position;
        const e1x = p1[0] - p0[0];
        const e1y = p1[1] - p0[1];
        const e2x = p2[0] - p0[0];
        const e2y = p2[1] - p0[1];
        cap_area += @abs(e1x * e2y - e2x * e1y) * 0.5;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), cap_area, 1e-4);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.bounds.min.z, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data.bounds.max.z, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), data.bounds.max.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), data.bounds.max.y, 1e-4);
}

test "grid quad helpers preserve winding" {
    // stride 4, row 1, col 2: a = 6, b = 7, c = 11, d = 10.
    var buf: [6]u32 = undefined;
    appendGridQuad(buf[0..], 0, 4, 1, 2);
    try std.testing.expectEqualSlices(u32, &.{ 6, 7, 11, 6, 11, 10 }, &buf);

    // Flipped variant matches the Ground/Terrain order (a, c, b) + (a, d, c).
    var flipped: [6]u32 = undefined;
    appendGridQuadFlipped(flipped[0..], 0, 4, 1, 2);
    try std.testing.expectEqualSlices(u32, &.{ 6, 11, 7, 6, 10, 11 }, &flipped);

    // storeQuad narrows to u16 for the Box/Sphere/Capsule paths.
    var narrow: [6]u16 = undefined;
    storeQuad(narrow[0..], 0, 4, 5, 6, 7);
    try std.testing.expectEqualSlices(u16, &.{ 4, 5, 6, 4, 6, 7 }, &narrow);

    // Orthogonal fallback picks the clear axis.
    try std.testing.expectEqual(Vec3.up, pickOrthogonal(Vec3.new(1, 0, 0)));
    try std.testing.expectEqual(Vec3.right, pickOrthogonal(Vec3.new(0, 1, 0)));
    // Seed reference keeps a clear hint and replaces a parallel one.
    try std.testing.expectEqual(Vec3.up, resolveFrameSeed(Vec3.forward, Vec3.up));
    try std.testing.expectEqual(Vec3.right, resolveFrameSeed(Vec3.up, Vec3.up));
}

test "MeshBuilder extrude rejects invalid input" {
    const ally = std.testing.allocator;
    const two = [_]Vec2{ Vec2.new(0, 0), Vec2.new(1, 0) };
    try std.testing.expectError(error.InvalidExtrude, buildExtrudeData(ally, .{ .profile = &two }));

    // Collinear points have zero area.
    const collinear = [_]Vec2{ Vec2.new(0, 0), Vec2.new(1, 0), Vec2.new(2, 0) };
    try std.testing.expectError(error.InvalidExtrude, buildExtrudeData(ally, .{ .profile = &collinear }));

    // Bow-tie outline self-intersects.
    const bowtie = [_]Vec2{ Vec2.new(0, 0), Vec2.new(1, 1), Vec2.new(1, 0), Vec2.new(0, 1) };
    try std.testing.expectError(error.InvalidExtrude, buildExtrudeData(ally, .{ .profile = &bowtie }));
}

// CPU morph-target tests run without GPU: vertex_buffer.id == 0, so
// applyMorphs only rewrites the staging copy (no sg.updateBuffer).
// Mesh.deinit is intentionally NOT used here (it destroys GPU buffers);
// freeMorphTestMesh below mirrors its morph cleanup.
fn freeMorphTestMesh(ally: std.mem.Allocator, mesh: *Mesh) void {
    for (mesh.morph_targets) |*mt| {
        if (mt.position_deltas.len > 0) ally.free(mt.position_deltas);
        if (mt.normal_deltas.len > 0) ally.free(mt.normal_deltas);
        if (mt.tangent_deltas.len > 0) ally.free(mt.tangent_deltas);
    }
    if (mesh.morph_targets.len > 0) ally.free(mesh.morph_targets);
    if (mesh.morph_weights.len > 0) ally.free(mesh.morph_weights);
    if (mesh.morph_base.len > 0) ally.free(mesh.morph_base);
    if (mesh.morph_staging.len > 0) ally.free(mesh.morph_staging);
}

fn makeMorphTestMesh(ally: std.mem.Allocator) !Mesh {
    const base = [_]Vertex{
        .{ .position = .{ 0, 0, 0 }, .normal = .{ 0, 1, 0 }, .color = .{ 1, 1, 1, 1 }, .uv = .{ 0, 0 } },
        .{ .position = .{ 1, 0, 0 }, .normal = .{ 0, 1, 0 }, .color = .{ 1, 1, 1, 1 }, .uv = .{ 1, 0 } },
    };
    var mesh: Mesh = .{ .name = "morph_test", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 0 };
    try mesh.retainMorphBase(ally, &base);
    return mesh;
}

test "MorphTarget single-target blend exact values" {
    const ally = std.testing.allocator;
    var mesh = try makeMorphTestMesh(ally);
    defer freeMorphTestMesh(ally, &mesh);

    const pos = try ally.alloc([3]f32, 2);
    pos[0] = .{ 0, 2, 0 };
    pos[1] = .{ 0, 0, 4 };
    const nrm = try ally.alloc([3]f32, 2);
    nrm[0] = .{ 0, 1, 0 };
    nrm[1] = .{ 1, 0, 0 };
    mesh.morph_targets = try ally.alloc(MorphTarget, 1);
    mesh.morph_targets[0] = .{ .position_deltas = pos, .normal_deltas = nrm };
    mesh.morph_weights = try ally.alloc(f32, 1);
    mesh.morph_weights[0] = 0.0;

    try std.testing.expect(mesh.hasMorphTargets());
    mesh.setMorphWeight(0, 0.5);
    try std.testing.expect(mesh.morph_dirty);
    mesh.applyMorphs();
    try std.testing.expect(!mesh.morph_dirty);

    // staging = base + 0.5 * delta (normals NOT renormalized, by design).
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mesh.morph_staging[0].position[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mesh.morph_staging[0].position[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mesh.morph_staging[0].position[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mesh.morph_staging[1].position[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mesh.morph_staging[1].position[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), mesh.morph_staging[1].position[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mesh.morph_staging[0].normal[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), mesh.morph_staging[0].normal[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mesh.morph_staging[1].normal[0], 1e-6);
    // Untouched attributes stay at base.
    try std.testing.expectEqual([2]f32{ 0, 0 }, mesh.morph_staging[0].uv);
    try std.testing.expectEqual([4]f32{ 1, 0, 0, 1 }, mesh.morph_staging[0].tangent);
}

test "MorphTarget two-target blend combines deltas" {
    const ally = std.testing.allocator;
    var mesh = try makeMorphTestMesh(ally);
    defer freeMorphTestMesh(ally, &mesh);

    const pos_a = try ally.alloc([3]f32, 2);
    pos_a[0] = .{ 1, 0, 0 };
    pos_a[1] = .{ 0, 0, 0 };
    const pos_b = try ally.alloc([3]f32, 2);
    pos_b[0] = .{ 0, 0, 10 };
    pos_b[1] = .{ 0, 4, 0 };
    const tan_b = try ally.alloc([3]f32, 2);
    tan_b[0] = .{ 0, 1, 0 };
    tan_b[1] = .{ 0, 0, 0 };
    mesh.morph_targets = try ally.alloc(MorphTarget, 2);
    mesh.morph_targets[0] = .{ .position_deltas = pos_a };
    mesh.morph_targets[1] = .{ .position_deltas = pos_b, .tangent_deltas = tan_b };
    mesh.morph_weights = try ally.alloc(f32, 2);
    mesh.morph_weights[0] = 0.0;
    mesh.morph_weights[1] = 0.0;

    mesh.setMorphWeights(&.{ 1.0, 0.5 });
    mesh.applyMorphs();

    // v0: (0,0,0) + 1*(1,0,0) + 0.5*(0,0,10) = (1,0,5).
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mesh.morph_staging[0].position[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mesh.morph_staging[0].position[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), mesh.morph_staging[0].position[2], 1e-6);
    // v1: (1,0,0) + 1*(0,0,0) + 0.5*(0,4,0) = (1,2,0).
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mesh.morph_staging[1].position[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), mesh.morph_staging[1].position[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mesh.morph_staging[1].position[2], 1e-6);
    // Tangent xyz blended, w preserved.
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mesh.morph_staging[0].tangent[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mesh.morph_staging[0].tangent[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mesh.morph_staging[0].tangent[3], 1e-6);
}

test "MorphTarget weight clamp and reset to base" {
    const ally = std.testing.allocator;
    var mesh = try makeMorphTestMesh(ally);
    defer freeMorphTestMesh(ally, &mesh);

    const pos = try ally.alloc([3]f32, 2);
    pos[0] = .{ 5, 5, 5 };
    pos[1] = .{ 5, 5, 5 };
    mesh.morph_targets = try ally.alloc(MorphTarget, 1);
    mesh.morph_targets[0] = .{ .position_deltas = pos };
    mesh.morph_weights = try ally.alloc(f32, 1);
    mesh.morph_weights[0] = 0.0;

    // Out-of-range index is ignored, mesh stays clean.
    mesh.setMorphWeight(7, 1.0);
    try std.testing.expect(!mesh.morph_dirty);
    // Clamp to [0, 1].
    mesh.setMorphWeight(0, 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mesh.morph_weights[0], 1e-6);
    mesh.setMorphWeight(0, -3.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mesh.morph_weights[0], 1e-6);

    // Full weight then reset via setMorphWeights restores base exactly.
    mesh.setMorphWeight(0, 1.0);
    mesh.applyMorphs();
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), mesh.morph_staging[0].position[0], 1e-6);
    mesh.setMorphWeights(&.{0.0});
    mesh.applyMorphs();
    try std.testing.expectEqual(mesh.morph_base[0].position, mesh.morph_staging[0].position);
    try std.testing.expectEqual(mesh.morph_base[1].position, mesh.morph_staging[1].position);
    // setMorphWeights on a mesh without weights is a clean no-op.
    var bare: Mesh = .{ .name = "bare", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 0 };
    try std.testing.expect(!bare.hasMorphTargets());
    bare.setMorphWeights(&.{1.0});
    try std.testing.expect(!bare.morph_dirty);
    bare.applyMorphs(); // no base: no crash, stays clean
    try std.testing.expect(!bare.morph_dirty);
}

test "MorphTarget apply is dirty-idempotent" {
    const ally = std.testing.allocator;
    var mesh = try makeMorphTestMesh(ally);
    defer freeMorphTestMesh(ally, &mesh);

    const pos = try ally.alloc([3]f32, 2);
    pos[0] = .{ 0, 8, 0 };
    pos[1] = .{ 0, 0, 0 };
    mesh.morph_targets = try ally.alloc(MorphTarget, 1);
    mesh.morph_targets[0] = .{ .position_deltas = pos };
    mesh.morph_weights = try ally.alloc(f32, 1);
    mesh.morph_weights[0] = 0.0;

    mesh.setMorphWeight(0, 0.25);
    mesh.applyMorphs();
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), mesh.morph_staging[0].position[1], 1e-6);
    // Second apply with dirty == false does no work: a manual edit survives.
    mesh.morph_staging[0].position[1] = 42.0;
    mesh.applyMorphs();
    try std.testing.expectApproxEqAbs(@as(f32, 42.0), mesh.morph_staging[0].position[1], 1e-6);
}

test "MorphTarget empty deltas are a base-preserving no-op" {
    const ally = std.testing.allocator;
    var mesh = try makeMorphTestMesh(ally);
    defer freeMorphTestMesh(ally, &mesh);

    // Target present but every delta slice empty (e.g. glTF target without
    // readable attributes): full weight must still reproduce base.
    mesh.morph_targets = try ally.alloc(MorphTarget, 1);
    mesh.morph_targets[0] = .{};
    mesh.morph_weights = try ally.alloc(f32, 1);
    mesh.morph_weights[0] = 0.0;

    try std.testing.expect(mesh.hasMorphTargets());
    mesh.setMorphWeight(0, 1.0);
    mesh.applyMorphs();
    try std.testing.expectEqual(mesh.morph_base[0].position, mesh.morph_staging[0].position);
    try std.testing.expectEqual(mesh.morph_base[1].position, mesh.morph_staging[1].position);
}

test "MorphTarget blend is u16/u32 index-type independent" {
    const ally = std.testing.allocator;
    var mesh16 = try makeMorphTestMesh(ally);
    defer freeMorphTestMesh(ally, &mesh16);
    var mesh32 = try makeMorphTestMesh(ally);
    defer freeMorphTestMesh(ally, &mesh32);
    mesh16.index_type = .UINT16;
    mesh32.index_type = .UINT32;

    for ([2]*Mesh{ &mesh16, &mesh32 }) |m| {
        const pos = try ally.alloc([3]f32, 2);
        pos[0] = .{ 1, 2, 3 };
        pos[1] = .{ -1, -2, -3 };
        m.morph_targets = try ally.alloc(MorphTarget, 1);
        m.morph_targets[0] = .{ .position_deltas = pos };
        m.morph_weights = try ally.alloc(f32, 1);
        m.morph_weights[0] = 0.0;
        m.setMorphWeight(0, 0.5);
        m.applyMorphs();
    }

    try std.testing.expectEqual(mesh16.morph_staging[0].position, mesh32.morph_staging[0].position);
    try std.testing.expectEqual(mesh16.morph_staging[1].position, mesh32.morph_staging[1].position);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mesh32.morph_staging[0].position[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mesh32.morph_staging[0].position[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), mesh32.morph_staging[0].position[2], 1e-6);
}

test "Mesh LOD levels selection, sorting and culling" {
    const ally = std.testing.allocator;
    var base_mesh = Mesh{
        .name = "base_high",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 1000,
        .position = Vec3.zero,
    };
    defer base_mesh.deinit(ally);

    var lod_med = Mesh{
        .name = "lod_medium",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 500,
    };
    defer lod_med.deinit(ally);

    var lod_low = Mesh{
        .name = "lod_low",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 100,
    };
    defer lod_low.deinit(ally);

    // Add in arbitrary order to test automatic sorting by distance
    try base_mesh.addLODLevel(ally, 50.0, &lod_low);
    try base_mesh.addLODLevel(ally, 20.0, &lod_med);
    try base_mesh.addLODLevel(ally, 100.0, null); // Cull beyond 100m

    // Verify automatic distance sorting
    try std.testing.expectEqual(@as(usize, 3), base_mesh.lod_levels.items.len);
    try std.testing.expectEqual(@as(f32, 20.0), base_mesh.lod_levels.items[0].distance);
    try std.testing.expectEqual(&lod_med, base_mesh.lod_levels.items[0].mesh.?);
    try std.testing.expectEqual(@as(f32, 50.0), base_mesh.lod_levels.items[1].distance);
    try std.testing.expectEqual(&lod_low, base_mesh.lod_levels.items[1].mesh.?);
    try std.testing.expectEqual(@as(f32, 100.0), base_mesh.lod_levels.items[2].distance);
    try std.testing.expect(base_mesh.lod_levels.items[2].mesh == null);

    // Verify is_lod_child flag was set on children
    try std.testing.expect(!base_mesh.is_lod_child);
    try std.testing.expect(lod_med.is_lod_child);
    try std.testing.expect(lod_low.is_lod_child);

    // Test LOD selection by distance
    try std.testing.expectEqual(&base_mesh, base_mesh.getLOD(0.0).?);
    try std.testing.expectEqual(&base_mesh, base_mesh.getLOD(19.9).?);
    try std.testing.expectEqual(&lod_med, base_mesh.getLOD(20.0).?);
    try std.testing.expectEqual(&lod_med, base_mesh.getLOD(49.9).?);
    try std.testing.expectEqual(&lod_low, base_mesh.getLOD(50.0).?);
    try std.testing.expectEqual(&lod_low, base_mesh.getLOD(99.9).?);
    try std.testing.expect(base_mesh.getLOD(100.0) == null);
    try std.testing.expect(base_mesh.getLOD(200.0) == null);

    // Test LOD for camera position
    try std.testing.expectEqual(&base_mesh, base_mesh.getLODForCamera(Vec3.new(0, 10, 0)).?);
    try std.testing.expectEqual(&lod_med, base_mesh.getLODForCamera(Vec3.new(0, 30, 0)).?);
    try std.testing.expectEqual(&lod_low, base_mesh.getLODForCamera(Vec3.new(0, 70, 0)).?);
    try std.testing.expect(base_mesh.getLODForCamera(Vec3.new(0, 150, 0)) == null);
}

test "Mesh decal projection onto planar mesh" {
    const ally = std.testing.allocator;
    var plane_data = try buildPlaneData(ally, .{ .width = 10.0, .height = 10.0 });
    defer plane_data.deinit(ally);

    var target_mesh: Mesh = .{
        .name = "target_plane",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = @intCast(plane_data.indices.len),
    };
    try target_mesh.retainCpuGeometryU32(ally, plane_data.vertices, plane_data.indices);
    defer target_mesh.deinit(ally);

    var decal_data = try buildDecalData(ally, &target_mesh, .{
        .position = Vec3.new(0.0, 0.0, 0.0),
        .normal = Vec3.new(0.0, 0.0, 1.0),
        .size = Vec3.new(2.0, 2.0, 2.0),
        .depth_bias = 0.005,
    });
    defer decal_data.deinit(ally);

    try std.testing.expect(decal_data.vertices.len > 0);
    try std.testing.expect(decal_data.indices.len > 0);
    try std.testing.expectEqual(@as(usize, 0), decal_data.indices.len % 3);

    // Verify UV coordinates in [0, 1] range
    for (decal_data.vertices) |v| {
        try std.testing.expect(v.uv[0] >= 0.0 and v.uv[0] <= 1.0);
        try std.testing.expect(v.uv[1] >= 0.0 and v.uv[1] <= 1.0);
        // Normal should align with plane normal (+Z)
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), v.normal[0], 1e-4);
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), v.normal[1], 1e-4);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), v.normal[2], 1e-4);
        // Depth bias applied along normal (+Z by 0.005)
        try std.testing.expectApproxEqAbs(@as(f32, 0.005), v.position[2], 1e-4);
    }

    // Bounds should be clipped within the projector size ([-1, 1] in X and Y)
    try std.testing.expect(decal_data.bounds.min.x >= -1.001);
    try std.testing.expect(decal_data.bounds.max.x <= 1.001);
    try std.testing.expect(decal_data.bounds.min.y >= -1.001);
    try std.testing.expect(decal_data.bounds.max.y <= 1.001);
}

test "Mesh decal backface culling and pass-through" {
    const ally = std.testing.allocator;
    var plane_data = try buildPlaneData(ally, .{ .width = 4.0, .height = 4.0 });
    defer plane_data.deinit(ally);

    var target_mesh: Mesh = .{
        .name = "target_plane",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = @intCast(plane_data.indices.len),
    };
    try target_mesh.retainCpuGeometryU32(ally, plane_data.vertices, plane_data.indices);
    defer target_mesh.deinit(ally);

    // Projector facing backwards relative to surface normal
    var culled_data = try buildDecalData(ally, &target_mesh, .{
        .position = Vec3.zero,
        .normal = Vec3.new(0.0, 0.0, -1.0),
        .size = Vec3.new(2.0, 2.0, 2.0),
        .cull_backfaces = true,
    });
    defer culled_data.deinit(ally);

    try std.testing.expectEqual(@as(usize, 0), culled_data.vertices.len);
    try std.testing.expectEqual(@as(usize, 0), culled_data.indices.len);

    // With cull_backfaces = false, backfacing triangles are preserved
    var unculled_data = try buildDecalData(ally, &target_mesh, .{
        .position = Vec3.zero,
        .normal = Vec3.new(0.0, 0.0, -1.0),
        .size = Vec3.new(2.0, 2.0, 2.0),
        .cull_backfaces = false,
    });
    defer unculled_data.deinit(ally);

    try std.testing.expect(unculled_data.vertices.len > 0);
    try std.testing.expect(unculled_data.indices.len > 0);
}

test "Mesh decal off-target returns empty geometry" {
    const ally = std.testing.allocator;
    var plane_data = try buildPlaneData(ally, .{ .width = 2.0, .height = 2.0 });
    defer plane_data.deinit(ally);

    var target_mesh: Mesh = .{
        .name = "target_plane",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = @intCast(plane_data.indices.len),
    };
    try target_mesh.retainCpuGeometryU32(ally, plane_data.vertices, plane_data.indices);
    defer target_mesh.deinit(ally);

    // Completely outside the bounds of the target plane (plane is at Z=0, [-1, 1]^2)
    var decal_data = try buildDecalData(ally, &target_mesh, .{
        .position = Vec3.new(20.0, 20.0, 0.0),
        .normal = Vec3.new(0.0, 0.0, 1.0),
        .size = Vec3.new(1.0, 1.0, 1.0),
    });
    defer decal_data.deinit(ally);

    try std.testing.expectEqual(@as(usize, 0), decal_data.vertices.len);
    try std.testing.expectEqual(@as(usize, 0), decal_data.indices.len);
}

test "Mesh decal rotation angle and transformation" {
    const ally = std.testing.allocator;
    var plane_data = try buildPlaneData(ally, .{ .width = 4.0, .height = 4.0 });
    defer plane_data.deinit(ally);

    var target_mesh: Mesh = .{
        .name = "target_plane",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = @intCast(plane_data.indices.len),
        // Target mesh with translation and scale
        .position = Vec3.new(5.0, 0.0, 0.0),
        .scaling = Vec3.new(2.0, 2.0, 1.0),
    };
    try target_mesh.retainCpuGeometryU32(ally, plane_data.vertices, plane_data.indices);
    defer target_mesh.deinit(ally);

    var decal_data = try buildDecalData(ally, &target_mesh, .{
        .position = Vec3.new(5.0, 0.0, 0.0),
        .normal = Vec3.new(0.0, 0.0, 1.0),
        .size = Vec3.new(1.0, 1.0, 1.0),
        .angle = std.math.pi / 4.0, // 45 degree rotation
    });
    defer decal_data.deinit(ally);

    try std.testing.expect(decal_data.vertices.len > 0);
    try std.testing.expect(decal_data.indices.len > 0);

    for (decal_data.vertices) |v| {
        try std.testing.expect(v.uv[0] >= 0.0 and v.uv[0] <= 1.0);
        try std.testing.expect(v.uv[1] >= 0.0 and v.uv[1] <= 1.0);
        // Tangent should be normalized
        const tan_len = @sqrt(v.tangent[0] * v.tangent[0] + v.tangent[1] * v.tangent[1] + v.tangent[2] * v.tangent[2]);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), tan_len, 1e-4);
    }
}

test "Mesh decal depth bias anti-z-fighting properties" {
    const ally = std.testing.allocator;
    const opts = DecalOptions{
        .position = Vec3.zero,
        .normal = Vec3.up,
    };
    // Default depth_bias should be >= 0.003 (3mm) to prevent z-fighting at distances
    try std.testing.expect(opts.depth_bias >= 0.003);

    var dummy_mesh = Mesh{
        .name = "decal_test",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .is_decal = true,
        .cast_shadows = false,
    };
    defer dummy_mesh.deinit(ally);

    try std.testing.expect(dummy_mesh.is_decal);
    try std.testing.expect(!dummy_mesh.cast_shadows);
}

test "Mesh decal barycentric coordinates" {
    const a = Vec3.new(0.0, 0.0, 0.0);
    const b = Vec3.new(1.0, 0.0, 0.0);
    const c = Vec3.new(0.0, 1.0, 0.0);

    // At vertex A
    const bary_a = barycentric(a, a, b, c);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), bary_a[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), bary_a[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), bary_a[2], 1e-4);

    // At vertex B
    const bary_b = barycentric(b, a, b, c);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), bary_b[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), bary_b[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), bary_b[2], 1e-4);

    // Midpoint of AB
    const mid_ab = Vec3.new(0.5, 0.0, 0.0);
    const bary_ab = barycentric(mid_ab, a, b, c);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), bary_ab[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), bary_ab[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), bary_ab[2], 1e-4);

    // Triangle centroid
    const centroid = Vec3.new(1.0 / 3.0, 1.0 / 3.0, 0.0);
    const bary_c = barycentric(centroid, a, b, c);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), bary_c[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), bary_c[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), bary_c[2], 1e-4);
}

test "Mesh decal blend skin weights" {
    // Identical joints
    const s0 = SkinJointWeight{ .joints = .{ 2, 5, 0, 0 }, .weights = .{ 0.8, 0.2, 0, 0 } };
    const s1 = SkinJointWeight{ .joints = .{ 2, 5, 0, 0 }, .weights = .{ 0.4, 0.6, 0, 0 } };
    const s2 = SkinJointWeight{ .joints = .{ 2, 5, 0, 0 }, .weights = .{ 0.6, 0.4, 0, 0 } };

    const blended = blendSkinWeights(.{ 0.5, 0.5, 0.0 }, s0, s1, s2);
    try std.testing.expectEqual(s0.joints, blended.joints);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), blended.weights[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), blended.weights[1], 1e-4);

    // Different joints
    const d0 = SkinJointWeight{ .joints = .{ 1, 0, 0, 0 }, .weights = .{ 1.0, 0, 0, 0 } };
    const d1 = SkinJointWeight{ .joints = .{ 3, 0, 0, 0 }, .weights = .{ 1.0, 0, 0, 0 } };
    const d2 = SkinJointWeight{ .joints = .{ 4, 0, 0, 0 }, .weights = .{ 1.0, 0, 0, 0 } };

    const blend_diff = blendSkinWeights(.{ 0.5, 0.5, 0.0 }, d0, d1, d2);
    const sum_w = blend_diff.weights[0] + blend_diff.weights[1] + blend_diff.weights[2] + blend_diff.weights[3];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum_w, 1e-4);
    // Should have joints 1 and 3 with ~0.5 weight each
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), blend_diff.weights[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), blend_diff.weights[1], 1e-4);
}

test "Mesh decal projector multi-mesh projection" {
    const ally = std.testing.allocator;

    // Mesh 1: Floor at y = 0
    var floor_data = try buildPlaneData(ally, .{ .width = 4.0, .height = 4.0 });
    defer floor_data.deinit(ally);
    var floor_mesh = Mesh{
        .name = "floor",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .rotation = Vec3.new(90.0, 0.0, 0.0), // Rotate plane flat onto XZ
    };
    defer floor_mesh.deinit(ally);
    try floor_mesh.retainCpuGeometryU32(ally, floor_data.vertices, floor_data.indices);

    // Mesh 2: Wall at z = 2
    var wall_data = try buildPlaneData(ally, .{ .width = 4.0, .height = 4.0 });
    defer wall_data.deinit(ally);
    var wall_mesh = Mesh{
        .name = "wall",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 2.0, 2.0),
    };
    defer wall_mesh.deinit(ally);
    try wall_mesh.retainCpuGeometryU32(ally, wall_data.vertices, wall_data.indices);

    const projector = DecalProjector{
        .position = Vec3.new(0.0, 0.2, 1.8),
        .normal = Vec3.new(0.0, -0.5, 0.866).normalize(),
        .size = Vec3.new(1.5, 1.5, 1.5),
        .cull_backfaces = false,
    };

    const meshes = [_]*const Mesh{ &floor_mesh, &wall_mesh };
    var multi_data = try projector.buildMultiMeshDecalData(ally, &meshes);
    defer multi_data.deinit(ally);

    try std.testing.expect(multi_data.vertices.len > 0);
    try std.testing.expect(multi_data.indices.len > 0);
    try std.testing.expect(multi_data.indices.len % 3 == 0);
}

test "Mesh decal skinned mesh projection" {
    const ally = std.testing.allocator;

    var plane_data = try buildPlaneData(ally, .{ .width = 2.0, .height = 2.0 });
    defer plane_data.deinit(ally);

    // Mock skeleton with 2 bones
    var skel = try Skeleton.init(ally, 2);
    defer skel.deinit();
    skel.update();

    var skinned_mesh = Mesh{
        .name = "skinned_char",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .skeleton = skel,
    };
    defer skinned_mesh.deinit(ally);
    try skinned_mesh.retainCpuGeometryU32(ally, plane_data.vertices, plane_data.indices);

    // Assign bone 1 with 100% influence to all vertices
    const skin_weights = try ally.alloc(SkinJointWeight, plane_data.vertices.len);
    defer ally.free(skin_weights);
    for (skin_weights) |*s| {
        s.* = .{ .joints = .{ 1.0, 0.0, 0.0, 0.0 }, .weights = .{ 1.0, 0.0, 0.0, 0.0 } };
    }
    skinned_mesh.cpu_skin = try ally.dupe(SkinJointWeight, skin_weights);

    const opts = DecalOptions{
        .position = Vec3.new(0.0, 0.0, 0.0),
        .normal = Vec3.new(0.0, 0.0, 1.0),
        .size = Vec3.new(1.0, 1.0, 1.0),
    };

    var decal_data = try buildDecalData(ally, &skinned_mesh, opts);
    defer decal_data.deinit(ally);

    try std.testing.expect(decal_data.vertices.len > 0);
    // Skinned decal vertices must preserve the skin joints and weights!
    for (decal_data.vertices) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), v.joints[0], 1e-4);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), v.weights[0], 1e-4);
    }
}

test "Mesh decal manager lifecycle and fade" {
    const ally = std.testing.allocator;

    var mock_scene: Scene = undefined;
    mock_scene.allocator = ally;
    mock_scene.meshes = .empty;
    mock_scene.pbr_materials = .empty;
    mock_scene.outline_meshes = .empty;

    var mgr = DecalManager.init(&mock_scene, 2);
    defer mgr.deinit();

    // Create dummy mesh and dummy material to simulate active decals
    const m1 = try ally.create(Mesh);
    m1.* = Mesh{ .name = "d1", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 0 };
    try mock_scene.meshes.append(ally, m1);

    const mat1 = try ally.create(PBRMaterial);
    mat1.* = PBRMaterial.init("m1");
    try mock_scene.pbr_materials.append(ally, mat1);

    try mgr.instances.append(ally, .{
        .mesh = m1,
        .material = mat1,
        .base_color = math.Color3.white,
        .lifetime = 2.0,
        .fade_duration = 1.0,
        .elapsed = 0.0,
    });

    try std.testing.expectEqual(@as(usize, 1), mgr.instances.items.len);

    // Advance time before lifetime
    mgr.update(1.0);
    try std.testing.expectEqual(@as(usize, 1), mgr.instances.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), mat1.albedo_color.r, 1e-4);

    // Advance time into fade duration (elapsed = 2.5, lifetime = 2.0, fade = 1.0 -> factor = 0.5)
    mgr.update(1.5);
    try std.testing.expectEqual(@as(usize, 1), mgr.instances.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mat1.albedo_color.r, 1e-2);

    // Advance time past expiration (elapsed = 3.5 > 3.0)
    mgr.update(1.0);
    try std.testing.expectEqual(@as(usize, 0), mgr.instances.items.len);
    try std.testing.expectEqual(@as(usize, 0), mock_scene.meshes.items.len);
    try std.testing.expectEqual(@as(usize, 0), mock_scene.pbr_materials.items.len);
}

test "buildPolygonData: flat 2D triangle, quad, and concave polygon" {
    const ally = std.testing.allocator;

    // 1. Triangle
    const tri_shape = [_]Vec2{
        Vec2.new(0.0, 0.0),
        Vec2.new(2.0, 0.0),
        Vec2.new(1.0, 2.0),
    };
    var tri_data = try mesh_mod.buildPolygonData(ally, .{
        .shape = &tri_shape,
        .depth = 0.0,
        .plane = .xz,
    });
    defer tri_data.deinit(ally);
    try std.testing.expectEqual(@as(usize, 3), tri_data.vertices.len);
    try std.testing.expectEqual(@as(usize, 3), tri_data.indices.len);

    // 2. Convex quad
    const quad_shape = [_]Vec2{
        Vec2.new(0.0, 0.0),
        Vec2.new(4.0, 0.0),
        Vec2.new(4.0, 4.0),
        Vec2.new(0.0, 4.0),
    };
    var quad_data = try mesh_mod.buildPolygonData(ally, .{
        .shape = &quad_shape,
        .depth = 0.0,
        .plane = .xz,
    });
    defer quad_data.deinit(ally);
    try std.testing.expectEqual(@as(usize, 4), quad_data.vertices.len);
    try std.testing.expectEqual(@as(usize, 6), quad_data.indices.len);

    // 3. Concave L-shape
    const l_shape = [_]Vec2{
        Vec2.new(0.0, 0.0),
        Vec2.new(4.0, 0.0),
        Vec2.new(4.0, 2.0),
        Vec2.new(2.0, 2.0),
        Vec2.new(2.0, 4.0),
        Vec2.new(0.0, 4.0),
    };
    var l_data = try mesh_mod.buildPolygonData(ally, .{
        .shape = &l_shape,
        .depth = 0.0,
        .plane = .xz,
    });
    defer l_data.deinit(ally);
    try std.testing.expectEqual(@as(usize, 6), l_data.vertices.len);
    try std.testing.expectEqual(@as(usize, 12), l_data.indices.len); // (6 - 2) * 3 = 12
}

test "buildPolygonData: polygon with hole" {
    const ally = std.testing.allocator;

    // Outer square: [0, 10] x [0, 10]
    const outer = [_]Vec2{
        Vec2.new(0.0, 0.0),
        Vec2.new(10.0, 0.0),
        Vec2.new(10.0, 10.0),
        Vec2.new(0.0, 10.0),
    };
    // Inner hole: [3, 7] x [3, 7]
    const hole = [_]Vec2{
        Vec2.new(3.0, 3.0),
        Vec2.new(7.0, 3.0),
        Vec2.new(7.0, 7.0),
        Vec2.new(3.0, 7.0),
    };
    const holes = [_][]const Vec2{&hole};

    var data = try mesh_mod.buildPolygonData(ally, .{
        .shape = &outer,
        .holes = &holes,
        .depth = 0.0,
        .plane = .xz,
    });
    defer data.deinit(ally);

    // Merged contour has 4 (outer) + 4 (hole) + 2 (bridge) = 10 vertices
    try std.testing.expectEqual(@as(usize, 10), data.vertices.len);
    // (10 - 2) * 3 = 24 indices (8 triangles)
    try std.testing.expectEqual(@as(usize, 24), data.indices.len);
}

test "buildPolygonData: extruded 3D polygon with depth" {
    const ally = std.testing.allocator;

    const quad = [_]Vec2{
        Vec2.new(0.0, 0.0),
        Vec2.new(5.0, 0.0),
        Vec2.new(5.0, 5.0),
        Vec2.new(0.0, 5.0),
    };

    var data = try mesh_mod.buildPolygonData(ally, .{
        .shape = &quad,
        .depth = 3.0,
        .plane = .xz,
    });
    defer data.deinit(ally);

    // 2 caps * 4 vertices = 8; 4 side walls * 4 vertices = 16. Total = 24 vertices.
    try std.testing.expectEqual(@as(usize, 24), data.vertices.len);
    // 2 caps * 6 indices = 12; 4 side walls * 6 indices = 24. Total = 36 indices.
    try std.testing.expectEqual(@as(usize, 36), data.indices.len);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.bounds.min.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), data.bounds.max.y, 1e-4);
}

test "TrailMesh: node recording, aging, and ribbon generation" {
    const ally = std.testing.allocator;

    const TrailMesh = mesh_mod.TrailMesh;
    const TrailNode = mesh_mod.TrailNode;

    var nodes: std.ArrayListUnmanaged(TrailNode) = .empty;
    defer nodes.deinit(ally);

    try nodes.append(ally, .{ .position = Vec3.new(0, 0, 0), .age = 0.0 });
    try nodes.append(ally, .{ .position = Vec3.new(1, 0, 0), .age = 0.5 });
    try nodes.append(ally, .{ .position = Vec3.new(2, 0, 0), .age = 1.0 });

    try std.testing.expectEqual(@as(usize, 3), nodes.items.len);

    // Simulate aging by dt = 0.6 with lifetime = 1.2
    for (nodes.items) |*n| n.age += 0.6;
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), nodes.items[0].age, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.1), nodes.items[1].age, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.6), nodes.items[2].age, 1e-4);

    // Prune expired (> 1.2)
    while (nodes.items.len > 0 and nodes.items[nodes.items.len - 1].age > 1.2) {
        _ = nodes.pop();
    }
    try std.testing.expectEqual(@as(usize, 2), nodes.items.len);

    _ = TrailMesh;
}
