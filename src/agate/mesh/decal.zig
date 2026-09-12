const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;

const types = @import("types.zig");
const Vertex = types.Vertex;
const GeometryData = types.GeometryData;
const tangents = @import("tangents.zig");

const mesh_mod = @import("mesh.zig");
const Mesh = mesh_mod.Mesh;
const Scene = @import("../scene.zig").Scene;

pub const DecalOptions = struct {
    position: Vec3,
    normal: Vec3,
    size: Vec3 = Vec3.new(1.0, 1.0, 1.0),
    angle: f32 = 0.0,
    cull_backfaces: bool = true,
    depth_bias: f32 = 0.002,
};

const DecalVertex = struct {
    world_pos: Vec3,
    local_pos: Vec3,
    normal: Vec3,
};

inline fn lerpVertex(a: DecalVertex, b: DecalVertex, t: f32) DecalVertex {
    const clamped_t = std.math.clamp(t, 0.0, 1.0);
    return .{
        .world_pos = a.world_pos.lerp(b.world_pos, clamped_t),
        .local_pos = a.local_pos.lerp(b.local_pos, clamped_t),
        .normal = a.normal.lerp(b.normal, clamped_t).normalize(),
    };
}

fn clipPolygonAgainstPlane(
    in_verts: []const DecalVertex,
    out_verts: []DecalVertex,
    axis: usize,
    is_min: bool,
    limit: f32,
) usize {
    if (in_verts.len == 0) return 0;
    var out_count: usize = 0;
    var i: usize = 0;
    while (i < in_verts.len) : (i += 1) {
        const a = in_verts[i];
        const next_idx = if (i + 1 < in_verts.len) i + 1 else 0;
        const b = in_verts[next_idx];

        const val_a = switch (axis) {
            0 => a.local_pos.x,
            1 => a.local_pos.y,
            else => a.local_pos.z,
        };
        const val_b = switch (axis) {
            0 => b.local_pos.x,
            1 => b.local_pos.y,
            else => b.local_pos.z,
        };

        const in_a = if (is_min) val_a >= limit else val_a <= limit;
        const in_b = if (is_min) val_b >= limit else val_b <= limit;

        if (in_a and in_b) {
            if (out_count < out_verts.len) {
                out_verts[out_count] = b;
                out_count += 1;
            }
        } else if (in_a and !in_b) {
            const denom = val_b - val_a;
            if (@abs(denom) > 1e-7) {
                const t = (limit - val_a) / denom;
                if (out_count < out_verts.len) {
                    out_verts[out_count] = lerpVertex(a, b, t);
                    out_count += 1;
                }
            }
        } else if (!in_a and in_b) {
            const denom = val_b - val_a;
            if (@abs(denom) > 1e-7) {
                const t = (limit - val_a) / denom;
                if (out_count < out_verts.len) {
                    out_verts[out_count] = lerpVertex(a, b, t);
                    out_count += 1;
                }
            }
            if (out_count < out_verts.len) {
                out_verts[out_count] = b;
                out_count += 1;
            }
        }
    }
    return out_count;
}

inline fn projectPointToLocal(p: Vec3, center: Vec3, u: Vec3, v: Vec3, w: Vec3, size: Vec3) Vec3 {
    const d = p.sub(center);
    const sx = if (size.x > 1e-6) size.x else 1.0;
    const sy = if (size.y > 1e-6) size.y else 1.0;
    const sz = if (size.z > 1e-6) size.z else 1.0;
    return Vec3.new(
        d.dot(u) / sx,
        d.dot(v) / sy,
        d.dot(w) / sz,
    );
}

/// Computes decal geometry by projecting an oriented bounding box onto the
/// target mesh and clipping all intersecting triangles with Sutherland-Hodgman.
pub fn buildDecalData(
    allocator: std.mem.Allocator,
    target_mesh: *const Mesh,
    options: DecalOptions,
) !GeometryData {
    if (target_mesh.cpu_positions.len < 3 or target_mesh.cpu_indices.len < 3) {
        return GeometryData{
            .vertices = try allocator.alloc(Vertex, 0),
            .indices = try allocator.alloc(u32, 0),
            .bounds = BoundingBox.zero,
        };
    }

    const norm = if (options.normal.lengthSq() > 1e-6) options.normal.normalize() else Vec3.up;
    const helper = tangents.pickOrthogonal(norm);
    var u_axis = norm.cross(helper).normalize();
    var v_axis = norm.cross(u_axis).normalize();

    if (@abs(options.angle) > 1e-6) {
        const cos_a = @cos(options.angle);
        const sin_a = @sin(options.angle);
        const new_u = u_axis.scale(cos_a).add(v_axis.scale(sin_a));
        const new_v = u_axis.scale(-sin_a).add(v_axis.scale(cos_a));
        u_axis = new_u.normalize();
        v_axis = new_v.normalize();
    }
    const w_axis = norm;

    var out_vertices = std.ArrayList(Vertex).empty;
    defer out_vertices.deinit(allocator);
    var out_indices = std.ArrayList(u32).empty;
    defer out_indices.deinit(allocator);

    const target_mat = @constCast(target_mesh).getWorldMatrix();

    var tri_i: usize = 0;
    while (tri_i + 2 < target_mesh.cpu_indices.len) : (tri_i += 3) {
        const idx0 = target_mesh.cpu_indices[tri_i + 0];
        const idx1 = target_mesh.cpu_indices[tri_i + 1];
        const idx2 = target_mesh.cpu_indices[tri_i + 2];
        if (idx0 >= target_mesh.cpu_positions.len or idx1 >= target_mesh.cpu_positions.len or idx2 >= target_mesh.cpu_positions.len) {
            continue;
        }

        const w0 = target_mat.transformPoint(target_mesh.cpu_positions[idx0]);
        const w1 = target_mat.transformPoint(target_mesh.cpu_positions[idx1]);
        const w2 = target_mat.transformPoint(target_mesh.cpu_positions[idx2]);

        const cross_prod = (w1.sub(w0)).cross(w2.sub(w0));
        const cross_len_sq = cross_prod.lengthSq();
        if (cross_len_sq < 1e-8) continue;
        const tri_norm = cross_prod.scale(1.0 / @sqrt(cross_len_sq));

        if (options.cull_backfaces and tri_norm.dot(w_axis) <= 0.0) {
            continue;
        }

        const l0 = projectPointToLocal(w0, options.position, u_axis, v_axis, w_axis, options.size);
        const l1 = projectPointToLocal(w1, options.position, u_axis, v_axis, w_axis, options.size);
        const l2 = projectPointToLocal(w2, options.position, u_axis, v_axis, w_axis, options.size);

        // Fast bounding box rejection against [-0.5, 0.5]^3
        if ((l0.x < -0.5 and l1.x < -0.5 and l2.x < -0.5) or
            (l0.x > 0.5 and l1.x > 0.5 and l2.x > 0.5) or
            (l0.y < -0.5 and l1.y < -0.5 and l2.y < -0.5) or
            (l0.y > 0.5 and l1.y > 0.5 and l2.y > 0.5) or
            (l0.z < -0.5 and l1.z < -0.5 and l2.z < -0.5) or
            (l0.z > 0.5 and l1.z > 0.5 and l2.z > 0.5))
        {
            continue;
        }

        // Polygon clipping against the 6 planes of the projector box
        var buf_a: [32]DecalVertex = undefined;
        var buf_b: [32]DecalVertex = undefined;

        buf_a[0] = .{ .world_pos = w0, .local_pos = l0, .normal = tri_norm };
        buf_a[1] = .{ .world_pos = w1, .local_pos = l1, .normal = tri_norm };
        buf_a[2] = .{ .world_pos = w2, .local_pos = l2, .normal = tri_norm };
        var count_a: usize = 3;

        // Plane 0: X >= -0.5
        var count_b = clipPolygonAgainstPlane(buf_a[0..count_a], &buf_b, 0, true, -0.5);
        if (count_b < 3) continue;

        // Plane 1: X <= 0.5
        count_a = clipPolygonAgainstPlane(buf_b[0..count_b], &buf_a, 0, false, 0.5);
        if (count_a < 3) continue;

        // Plane 2: Y >= -0.5
        count_b = clipPolygonAgainstPlane(buf_a[0..count_a], &buf_b, 1, true, -0.5);
        if (count_b < 3) continue;

        // Plane 3: Y <= 0.5
        count_a = clipPolygonAgainstPlane(buf_b[0..count_b], &buf_a, 1, false, 0.5);
        if (count_a < 3) continue;

        // Plane 4: Z >= -0.5
        count_b = clipPolygonAgainstPlane(buf_a[0..count_a], &buf_b, 2, true, -0.5);
        if (count_b < 3) continue;

        // Plane 5: Z <= 0.5
        count_a = clipPolygonAgainstPlane(buf_b[0..count_b], &buf_a, 2, false, 0.5);
        if (count_a < 3) continue;

        // Triangulate clipped convex polygon into fan
        const base_idx: u32 = @intCast(out_vertices.items.len);
        for (buf_a[0..count_a]) |v| {
            const pos = v.world_pos.add(v.normal.scale(options.depth_bias));
            const uv_x = std.math.clamp(v.local_pos.x + 0.5, 0.0, 1.0);
            const uv_y = std.math.clamp(1.0 - (v.local_pos.y + 0.5), 0.0, 1.0);
            try out_vertices.append(allocator, .{
                .position = .{ pos.x, pos.y, pos.z },
                .normal = .{ v.normal.x, v.normal.y, v.normal.z },
                .color = .{ 1.0, 1.0, 1.0, 1.0 },
                .uv = .{ uv_x, uv_y },
                .tangent = .{ u_axis.x, u_axis.y, u_axis.z, 1.0 },
                .joints = .{ 0.0, 0.0, 0.0, 0.0 },
                .weights = .{ 1.0, 0.0, 0.0, 0.0 },
            });
        }

        var j: u32 = 1;
        while (j + 1 < count_a) : (j += 1) {
            try out_indices.append(allocator, base_idx);
            try out_indices.append(allocator, base_idx + j);
            try out_indices.append(allocator, base_idx + j + 1);
        }
    }

    if (out_vertices.items.len == 0) {
        return GeometryData{
            .vertices = try allocator.alloc(Vertex, 0),
            .indices = try allocator.alloc(u32, 0),
            .bounds = BoundingBox.zero,
        };
    }

    var min_p = Vec3.new(std.math.inf(f32), std.math.inf(f32), std.math.inf(f32));
    var max_p = Vec3.new(-std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32));
    for (out_vertices.items) |v| {
        min_p.x = @min(min_p.x, v.position[0]);
        min_p.y = @min(min_p.y, v.position[1]);
        min_p.z = @min(min_p.z, v.position[2]);
        max_p.x = @max(max_p.x, v.position[0]);
        max_p.y = @max(max_p.y, v.position[1]);
        max_p.z = @max(max_p.z, v.position[2]);
    }

    return GeometryData{
        .vertices = try out_vertices.toOwnedSlice(allocator),
        .indices = try out_indices.toOwnedSlice(allocator),
        .bounds = BoundingBox.init(min_p, max_p),
    };
}

/// Creates a decal mesh projecting onto target_mesh with the specified options.
pub fn createDecal(
    scene: *Scene,
    name: []const u8,
    target_mesh: *const Mesh,
    options: DecalOptions,
) !*Mesh {
    var data = try buildDecalData(scene.allocator, target_mesh, options);
    if (data.vertices.len == 0) {
        data.deinit(scene.allocator);
        return error.DecalNoIntersection;
    }
    defer data.deinit(scene.allocator);
    const decal_mesh = try mesh_mod.uploadGeometry(scene, name, data);
    decal_mesh.culling_strategy = .frustum;
    return decal_mesh;
}
