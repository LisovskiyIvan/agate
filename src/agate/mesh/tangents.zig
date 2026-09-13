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

inline fn accumulateTriangleTangent(vertices: []Vertex, idx0: usize, idx1: usize, idx2: usize) void {
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

        vertices[idx0].tangent[0] += tangent.x;
        vertices[idx0].tangent[1] += tangent.y;
        vertices[idx0].tangent[2] += tangent.z;

        vertices[idx1].tangent[0] += tangent.x;
        vertices[idx1].tangent[1] += tangent.y;
        vertices[idx1].tangent[2] += tangent.z;

        vertices[idx2].tangent[0] += tangent.x;
        vertices[idx2].tangent[1] += tangent.y;
        vertices[idx2].tangent[2] += tangent.z;
    }
}

pub fn computeTangents(vertices: []Vertex, indices: ?[]const u32, indices16: ?[]const u16) void {
    for (vertices) |*v| {
        v.tangent = .{ 0, 0, 0, 1 };
    }

    // One loop per index source so the per-triangle source selection branches
    // disappear; accumulation order (and hence shared-vertex sums) is unchanged.
    if (indices) |idx| {
        var tri_i: usize = 0;
        while (tri_i + 2 < idx.len) : (tri_i += 3) {
            accumulateTriangleTangent(vertices, idx[tri_i], idx[tri_i + 1], idx[tri_i + 2]);
        }
    } else if (indices16) |idx16| {
        var tri_i: usize = 0;
        while (tri_i + 2 < idx16.len) : (tri_i += 3) {
            accumulateTriangleTangent(vertices, idx16[tri_i], idx16[tri_i + 1], idx16[tri_i + 2]);
        }
    } else {
        var tri_i: usize = 0;
        while (tri_i + 2 < vertices.len) : (tri_i += 3) {
            accumulateTriangleTangent(vertices, tri_i, tri_i + 1, tri_i + 2);
        }
    }

    for (vertices) |*v| {
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
            v.tangent[3] = 1.0;
        } else {
            v.tangent = .{ 1.0, 0.0, 0.0, 1.0 };
        }
    }
}
