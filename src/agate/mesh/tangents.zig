const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Vertex = @import("types.zig").Vertex;

/// Dot product above which two axes count as parallel in pickOrthogonal.
pub const orthogonal_dot_threshold: f32 = 0.9;

/// Fallback axis when a normal is degenerate: up when the normal is near ±X,
/// right otherwise.
pub inline fn pickOrthogonal(n: Vec3) Vec3 {
    return if (@abs(n.x) > orthogonal_dot_threshold) Vec3.up else Vec3.right;
}

inline fn accumulateTriangleTangent(
    vertices: []Vertex,
    bitangents: ?[]Vec3,
    idx0: usize,
    idx1: usize,
    idx2: usize,
) void {
    if (idx0 >= vertices.len or idx1 >= vertices.len or idx2 >= vertices.len) return;

    const v0 = vertices[idx0];
    const v1 = vertices[idx1];
    const v2 = vertices[idx2];

    const edge1 = Vec3.new(v1.position[0] - v0.position[0], v1.position[1] - v0.position[1], v1.position[2] - v0.position[2]);
    const edge2 = Vec3.new(v2.position[0] - v0.position[0], v2.position[1] - v0.position[1], v2.position[2] - v0.position[2]);

    const delta_u1 = v1.uv[0] - v0.uv[0];
    const delta_v1 = v1.uv[1] - v0.uv[1];
    const delta_u2 = v2.uv[0] - v0.uv[0];
    const delta_v2 = v2.uv[1] - v0.uv[1];

    const det = delta_u1 * delta_v2 - delta_u2 * delta_v1;
    if (@abs(det) > 1e-6) {
        const r = 1.0 / det;
        const tangent = Vec3.new(
            (edge1.x * delta_v2 - edge2.x * delta_v1) * r,
            (edge1.y * delta_v2 - edge2.y * delta_v1) * r,
            (edge1.z * delta_v2 - edge2.z * delta_v1) * r,
        );
        const bitangent = Vec3.new(
            (edge2.x * delta_u1 - edge1.x * delta_u2) * r,
            (edge2.y * delta_u1 - edge1.y * delta_u2) * r,
            (edge2.z * delta_u1 - edge1.z * delta_u2) * r,
        );

        vertices[idx0].tangent[0] += tangent.x;
        vertices[idx0].tangent[1] += tangent.y;
        vertices[idx0].tangent[2] += tangent.z;

        vertices[idx1].tangent[0] += tangent.x;
        vertices[idx1].tangent[1] += tangent.y;
        vertices[idx1].tangent[2] += tangent.z;

        vertices[idx2].tangent[0] += tangent.x;
        vertices[idx2].tangent[1] += tangent.y;
        vertices[idx2].tangent[2] += tangent.z;

        if (bitangents) |bts| {
            bts[idx0] = bts[idx0].add(bitangent);
            bts[idx1] = bts[idx1].add(bitangent);
            bts[idx2] = bts[idx2].add(bitangent);
        }
    }
}

inline fn accumulateTriangleNormal(
    vertices: []Vertex,
    idx0: usize,
    idx1: usize,
    idx2: usize,
) void {
    if (idx0 >= vertices.len or idx1 >= vertices.len or idx2 >= vertices.len) return;

    const p0 = Vec3.new(vertices[idx0].position[0], vertices[idx0].position[1], vertices[idx0].position[2]);
    const p1 = Vec3.new(vertices[idx1].position[0], vertices[idx1].position[1], vertices[idx1].position[2]);
    const p2 = Vec3.new(vertices[idx2].position[0], vertices[idx2].position[1], vertices[idx2].position[2]);

    const edge1 = p1.sub(p0);
    const edge2 = p2.sub(p0);
    const fnorm = Vec3.cross(edge1, edge2);

    vertices[idx0].normal[0] += fnorm.x;
    vertices[idx0].normal[1] += fnorm.y;
    vertices[idx0].normal[2] += fnorm.z;

    vertices[idx1].normal[0] += fnorm.x;
    vertices[idx1].normal[1] += fnorm.y;
    vertices[idx1].normal[2] += fnorm.z;

    vertices[idx2].normal[0] += fnorm.x;
    vertices[idx2].normal[1] += fnorm.y;
    vertices[idx2].normal[2] += fnorm.z;
}

/// Generates smooth area-weighted vertex normals for meshes lacking normal attributes.
pub fn computeNormals(vertices: []Vertex, indices: ?[]const u32, indices16: ?[]const u16) void {
    for (vertices) |*v| {
        v.normal = .{ 0, 0, 0 };
    }

    if (indices) |idx| {
        var tri_i: usize = 0;
        while (tri_i + 2 < idx.len) : (tri_i += 3) {
            accumulateTriangleNormal(vertices, idx[tri_i], idx[tri_i + 1], idx[tri_i + 2]);
        }
    } else if (indices16) |idx16| {
        var tri_i: usize = 0;
        while (tri_i + 2 < idx16.len) : (tri_i += 3) {
            accumulateTriangleNormal(vertices, idx16[tri_i], idx16[tri_i + 1], idx16[tri_i + 2]);
        }
    } else {
        var tri_i: usize = 0;
        while (tri_i + 2 < vertices.len) : (tri_i += 3) {
            accumulateTriangleNormal(vertices, tri_i, tri_i + 1, tri_i + 2);
        }
    }

    for (vertices) |*v| {
        const n = Vec3.new(v.normal[0], v.normal[1], v.normal[2]);
        const len = n.length();
        if (len > 1e-6) {
            const norm = n.scale(1.0 / len);
            v.normal = .{ norm.x, norm.y, norm.z };
        } else {
            v.normal = .{ 0, 1, 0 };
        }
    }
}

pub fn computeTangents(vertices: []Vertex, indices: ?[]const u32, indices16: ?[]const u16) void {
    for (vertices) |*v| {
        v.tangent = .{ 0, 0, 0, 1 };
    }

    var stack_bitangents: [512]Vec3 = undefined;
    const bitangents: ?[]Vec3 = if (vertices.len <= stack_bitangents.len)
        stack_bitangents[0..vertices.len]
    else
        std.heap.page_allocator.alloc(Vec3, vertices.len) catch null;
    defer if (vertices.len > stack_bitangents.len) {
        if (bitangents) |b| std.heap.page_allocator.free(b);
    };

    if (bitangents) |bts| {
        @memset(bts, Vec3.zero);
    }

    // One loop per index source so the per-triangle source selection branches
    // disappear; accumulation order (and hence shared-vertex sums) is unchanged.
    if (indices) |idx| {
        var tri_i: usize = 0;
        while (tri_i + 2 < idx.len) : (tri_i += 3) {
            accumulateTriangleTangent(vertices, bitangents, idx[tri_i], idx[tri_i + 1], idx[tri_i + 2]);
        }
    } else if (indices16) |idx16| {
        var tri_i: usize = 0;
        while (tri_i + 2 < idx16.len) : (tri_i += 3) {
            accumulateTriangleTangent(vertices, bitangents, idx16[tri_i], idx16[tri_i + 1], idx16[tri_i + 2]);
        }
    } else {
        var tri_i: usize = 0;
        while (tri_i + 2 < vertices.len) : (tri_i += 3) {
            accumulateTriangleTangent(vertices, bitangents, tri_i, tri_i + 1, tri_i + 2);
        }
    }

    for (vertices, 0..) |*v, i| {
        const n = Vec3.new(v.normal[0], v.normal[1], v.normal[2]);
        var t = Vec3.new(v.tangent[0], v.tangent[1], v.tangent[2]);

        if (t.lengthSq() < 1e-6) {
            t = pickOrthogonal(n);
        }

        const t_proj = t.sub(n.scale(n.dot(t)));
        if (t_proj.lengthSq() > 1e-6) {
            const t_norm = t_proj.normalize();
            v.tangent[0] = t_norm.x;
            v.tangent[1] = t_norm.y;
            v.tangent[2] = t_norm.z;

            // Handedness calculation:
            // Shaders calculate: B = cross(N, T) * tangent.w
            // If cross(N, T) is anti-parallel to accumulated bitangent B, handedness must be -1.0.
            if (bitangents) |bts| {
                const b = bts[i];
                if (b.lengthSq() > 1e-6) {
                    const handedness: f32 = if (n.cross(t_norm).dot(b) < 0.0) -1.0 else 1.0;
                    v.tangent[3] = handedness;
                } else {
                    v.tangent[3] = 1.0;
                }
            } else {
                v.tangent[3] = 1.0;
            }
        } else {
            v.tangent = .{ 1.0, 0.0, 0.0, 1.0 };
        }
    }
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
