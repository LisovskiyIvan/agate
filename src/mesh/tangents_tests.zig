const std = @import("std");
const tangents = @import("tangents.zig");
const computeTangents = tangents.computeTangents;
const computeTangentsForUv = tangents.computeTangentsForUv;
const computeNormals = tangents.computeNormals;
const Vertex = @import("types.zig").Vertex;

test "computeTangentsForUv uses UV1 without changing the UV0 input" {
    var vertices = [_]Vertex{
        .{ .position = .{ 0, 0, 0 }, .normal = .{ 0, 0, 1 }, .color = .{ 1, 1, 1, 1 }, .uv = .{ 0, 0 }, .uv1 = .{ 0, 0 } },
        .{ .position = .{ 1, 0, 0 }, .normal = .{ 0, 0, 1 }, .color = .{ 1, 1, 1, 1 }, .uv = .{ 1, 0 }, .uv1 = .{ 0, 1 } },
        .{ .position = .{ 0, 1, 0 }, .normal = .{ 0, 0, 1 }, .color = .{ 1, 1, 1, 1 }, .uv = .{ 0, 1 }, .uv1 = .{ 1, 0 } },
    };
    computeTangents(&vertices, null, null);
    try std.testing.expectApproxEqAbs(@as(f32, 1), vertices[0].tangent[0], 0.00001);
    computeTangentsForUv(&vertices, null, null, 1);
    try std.testing.expectApproxEqAbs(@as(f32, 1), vertices[0].tangent[1], 0.00001);
    try std.testing.expectEqual(@as(f32, -1), vertices[0].tangent[3]);
    try std.testing.expectEqualSlices(f32, &.{ 1, 0 }, &vertices[1].uv);
}

test "computeTangents right-handed vs mirrored UV handedness" {
    // Triangle 1: Standard right-handed UV coordinates
    // Pos: (0,0,0), (1,0,0), (0,1,0)
    // UV:  (0,0),   (1,0),   (0,1)
    // Normal: (0,0,1) -> T should be (1,0,0), B should be (0,1,0), cross(N, T) = (0,1,0) -> handedness = +1.0
    var rh_verts = [_]Vertex{
        .{ .position = .{ 0, 0, 0 }, .normal = .{ 0, 0, 1 }, .uv = .{ 0, 0 }, .color = .{ 1, 1, 1, 1 } },
        .{ .position = .{ 1, 0, 0 }, .normal = .{ 0, 0, 1 }, .uv = .{ 1, 0 }, .color = .{ 1, 1, 1, 1 } },
        .{ .position = .{ 0, 1, 0 }, .normal = .{ 0, 0, 1 }, .uv = .{ 0, 1 }, .color = .{ 1, 1, 1, 1 } },
    };
    computeTangents(&rh_verts, null, null);
    try std.testing.expectEqual(@as(f32, 1.0), rh_verts[0].tangent[3]);
    try std.testing.expectEqual(@as(f32, 1.0), rh_verts[1].tangent[3]);
    try std.testing.expectEqual(@as(f32, 1.0), rh_verts[2].tangent[3]);

    // Triangle 2: Mirrored left-handed UV coordinates (flipped U)
    // Pos: (0,0,0), (1,0,0), (0,1,0)
    // UV:  (1,0),   (0,0),   (1,1)
    // Handedness must be -1.0!
    var lh_verts = [_]Vertex{
        .{ .position = .{ 0, 0, 0 }, .normal = .{ 0, 0, 1 }, .uv = .{ 1, 0 }, .color = .{ 1, 1, 1, 1 } },
        .{ .position = .{ 1, 0, 0 }, .normal = .{ 0, 0, 1 }, .uv = .{ 0, 0 }, .color = .{ 1, 1, 1, 1 } },
        .{ .position = .{ 0, 1, 0 }, .normal = .{ 0, 0, 1 }, .uv = .{ 1, 1 }, .color = .{ 1, 1, 1, 1 } },
    };
    computeTangents(&lh_verts, null, null);
    try std.testing.expectEqual(@as(f32, -1.0), lh_verts[0].tangent[3]);
    try std.testing.expectEqual(@as(f32, -1.0), lh_verts[1].tangent[3]);
    try std.testing.expectEqual(@as(f32, -1.0), lh_verts[2].tangent[3]);
}

test "computeNormals generates correct triangle surface normals" {
    // Triangle in XY plane CCW: (0,0,0), (1,0,0), (0,1,0)
    // Edge1 = (1,0,0), Edge2 = (0,1,0) -> Cross = (0,0,1)
    var verts = [_]Vertex{
        .{ .position = .{ 0, 0, 0 }, .normal = .{ 0, 0, 0 }, .uv = .{ 0, 0 }, .color = .{ 1, 1, 1, 1 } },
        .{ .position = .{ 1, 0, 0 }, .normal = .{ 0, 0, 0 }, .uv = .{ 1, 0 }, .color = .{ 1, 1, 1, 1 } },
        .{ .position = .{ 0, 1, 0 }, .normal = .{ 0, 0, 0 }, .uv = .{ 0, 1 }, .color = .{ 1, 1, 1, 1 } },
    };
    computeNormals(&verts, null, null);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), verts[0].normal[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), verts[0].normal[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), verts[0].normal[2], 1e-5);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), verts[1].normal[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), verts[1].normal[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), verts[1].normal[2], 1e-5);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), verts[2].normal[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), verts[2].normal[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), verts[2].normal[2], 1e-5);
}
