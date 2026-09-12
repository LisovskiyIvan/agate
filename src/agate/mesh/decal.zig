const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const BoundingBox = math.BoundingBox;
const Color3 = math.Color3;

const types = @import("types.zig");
const Vertex = types.Vertex;
const GeometryData = types.GeometryData;
const SkinJointWeight = types.SkinJointWeight;
const tangents = @import("tangents.zig");

const mesh_mod = @import("mesh.zig");
const Mesh = mesh_mod.Mesh;
const Scene = @import("../scene.zig").Scene;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;
const MAX_BONES = @import("../animation/skeleton.zig").MAX_BONES;
const PBRMaterial = @import("../material.zig").PBRMaterial;
const Material = @import("../material.zig").Material;
const AlphaMode = @import("../material.zig").AlphaMode;
const Texture = @import("../texture.zig").Texture;

pub const DecalOptions = struct {
    position: Vec3,
    normal: Vec3,
    size: Vec3 = Vec3.new(1.0, 1.0, 1.0),
    angle: f32 = 0.0,
    cull_backfaces: bool = true,
    depth_bias: f32 = 0.004,
    parent_to_target: bool = false,
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

/// Computes barycentric coordinates (u, v, w) of point p relative to triangle (a, b, c).
/// u + v + w = 1.0. Clamped to non-negative coordinates.
pub fn barycentric(p: Vec3, a: Vec3, b: Vec3, c: Vec3) [3]f32 {
    const v0 = b.sub(a);
    const v1 = c.sub(a);
    const v2 = p.sub(a);
    const d00 = v0.dot(v0);
    const d01 = v0.dot(v1);
    const d11 = v1.dot(v1);
    const d20 = v2.dot(v0);
    const d21 = v2.dot(v1);
    const denom = d00 * d11 - d01 * d01;
    if (@abs(denom) < 1e-8) {
        return .{ 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 };
    }
    const inv = 1.0 / denom;
    var v = (d11 * d20 - d01 * d21) * inv;
    var w = (d00 * d21 - d01 * d20) * inv;
    var u = 1.0 - v - w;
    u = std.math.clamp(u, 0.0, 1.0);
    v = std.math.clamp(v, 0.0, 1.0);
    w = std.math.clamp(w, 0.0, 1.0);
    const sum = u + v + w;
    if (sum > 1e-6) {
        const inv_sum = 1.0 / sum;
        return .{ u * inv_sum, v * inv_sum, w * inv_sum };
    }
    return .{ 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 };
}

/// Blends skin joints and weights from triangle vertices according to barycentric weights.
pub fn blendSkinWeights(
    bary: [3]f32,
    s0: SkinJointWeight,
    s1: SkinJointWeight,
    s2: SkinJointWeight,
) struct { joints: [4]f32, weights: [4]f32 } {
    // Fast path: all 3 vertices have identical joint assignments
    if (std.mem.eql(f32, &s0.joints, &s1.joints) and std.mem.eql(f32, &s0.joints, &s2.joints)) {
        var w: [4]f32 = undefined;
        var sum: f32 = 0.0;
        inline for (0..4) |k| {
            w[k] = bary[0] * s0.weights[k] + bary[1] * s1.weights[k] + bary[2] * s2.weights[k];
            sum += w[k];
        }
        if (sum > 1e-6) {
            const inv = 1.0 / sum;
            inline for (0..4) |k| w[k] *= inv;
        } else {
            w = .{ 1.0, 0.0, 0.0, 0.0 };
        }
        return .{ .joints = s0.joints, .weights = w };
    }

    // General path: collect unique bones and sum weights
    var bones: [12]f32 = undefined;
    var weights: [12]f32 = undefined;
    var count: usize = 0;

    const inputs = [3]struct { skin: SkinJointWeight, lambda: f32 }{
        .{ .skin = s0, .lambda = bary[0] },
        .{ .skin = s1, .lambda = bary[1] },
        .{ .skin = s2, .lambda = bary[2] },
    };

    for (inputs) |inp| {
        if (inp.lambda <= 1e-6) continue;
        inline for (0..4) |k| {
            const j = inp.skin.joints[k];
            const w = inp.skin.weights[k] * inp.lambda;
            if (w > 1e-5) {
                var found: bool = false;
                for (0..count) |bi| {
                    if (@abs(bones[bi] - j) < 0.1) {
                        weights[bi] += w;
                        found = true;
                        break;
                    }
                }
                if (!found and count < 12) {
                    bones[count] = j;
                    weights[count] = w;
                    count += 1;
                }
            }
        }
    }

    if (count == 0) {
        return .{ .joints = .{ 0, 0, 0, 0 }, .weights = .{ 1, 0, 0, 0 } };
    }

    // Sort top weights descending
    var i: usize = 1;
    while (i < count) : (i += 1) {
        var j = i;
        while (j > 0 and weights[j] > weights[j - 1]) : (j -= 1) {
            std.mem.swap(f32, &weights[j], &weights[j - 1]);
            std.mem.swap(f32, &bones[j], &bones[j - 1]);
        }
    }

    var out_j = [4]f32{ 0, 0, 0, 0 };
    var out_w = [4]f32{ 0, 0, 0, 0 };
    var sum_w: f32 = 0.0;
    const take = @min(count, 4);
    for (0..take) |k| {
        out_j[k] = bones[k];
        out_w[k] = weights[k];
        sum_w += weights[k];
    }
    if (sum_w > 1e-6) {
        const inv = 1.0 / sum_w;
        for (0..take) |k| out_w[k] *= inv;
    } else {
        out_w[0] = 1.0;
    }
    return .{ .joints = out_j, .weights = out_w };
}

/// Evaluates the posed world position of a vertex on a skinned mesh.
pub fn getSkinnedWorldPosition(
    world_mat: Mat4,
    skel: *const Skeleton,
    pos: Vec3,
    skin: SkinJointWeight,
) Vec3 {
    var p_posed = Vec3.zero;
    var total_w: f32 = 0.0;
    inline for (0..4) |k| {
        const j_idx: usize = @intFromFloat(skin.joints[k]);
        const w = skin.weights[k];
        if (w > 0.0001 and j_idx < MAX_BONES and j_idx < skel.bones.len) {
            const transformed = skel.skin_matrices[j_idx].transformPoint(pos);
            p_posed = p_posed.add(transformed.scale(w));
            total_w += w;
        }
    }
    if (total_w < 1e-5) {
        p_posed = pos;
    }
    return world_mat.transformPoint(p_posed);
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
/// Automatically handles both static meshes and animated/skinned meshes.
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
    const inv_target_mat = if (options.parent_to_target) (target_mat.invert() orelse Mat4.identity) else Mat4.identity;

    const is_skinned = target_mesh.skeleton != null and
        target_mesh.cpu_skin.len == target_mesh.cpu_positions.len;

    var tri_i: usize = 0;
    while (tri_i + 2 < target_mesh.cpu_indices.len) : (tri_i += 3) {
        const idx0 = target_mesh.cpu_indices[tri_i + 0];
        const idx1 = target_mesh.cpu_indices[tri_i + 1];
        const idx2 = target_mesh.cpu_indices[tri_i + 2];
        if (idx0 >= target_mesh.cpu_positions.len or idx1 >= target_mesh.cpu_positions.len or idx2 >= target_mesh.cpu_positions.len) {
            continue;
        }

        const p0_bind = target_mesh.cpu_positions[idx0];
        const p1_bind = target_mesh.cpu_positions[idx1];
        const p2_bind = target_mesh.cpu_positions[idx2];

        var w0: Vec3 = undefined;
        var w1: Vec3 = undefined;
        var w2: Vec3 = undefined;

        if (is_skinned) {
            const skel = target_mesh.skeleton.?;
            w0 = getSkinnedWorldPosition(target_mat, skel, p0_bind, target_mesh.cpu_skin[idx0]);
            w1 = getSkinnedWorldPosition(target_mat, skel, p1_bind, target_mesh.cpu_skin[idx1]);
            w2 = getSkinnedWorldPosition(target_mat, skel, p2_bind, target_mesh.cpu_skin[idx2]);
        } else {
            w0 = target_mat.transformPoint(p0_bind);
            w1 = target_mat.transformPoint(p1_bind);
            w2 = target_mat.transformPoint(p2_bind);
        }

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
            const uv_x = std.math.clamp(v.local_pos.x + 0.5, 0.0, 1.0);
            const uv_y = std.math.clamp(1.0 - (v.local_pos.y + 0.5), 0.0, 1.0);

            if (is_skinned) {
                // Skinned mesh: reconstruct bind-pose position and interpolate joints/weights
                const bary = barycentric(v.world_pos, w0, w1, w2);
                const bind_cross = (p1_bind.sub(p0_bind)).cross(p2_bind.sub(p0_bind));
                const bind_norm = if (bind_cross.lengthSq() > 1e-8) bind_cross.normalize() else Vec3.up;

                const pos_bind = p0_bind.scale(bary[0]).add(p1_bind.scale(bary[1])).add(p2_bind.scale(bary[2]));
                const pos_biased = pos_bind.add(bind_norm.scale(options.depth_bias));

                const skin_interp = blendSkinWeights(
                    bary,
                    target_mesh.cpu_skin[idx0],
                    target_mesh.cpu_skin[idx1],
                    target_mesh.cpu_skin[idx2],
                );

                try out_vertices.append(allocator, .{
                    .position = .{ pos_biased.x, pos_biased.y, pos_biased.z },
                    .normal = .{ bind_norm.x, bind_norm.y, bind_norm.z },
                    .color = .{ 1.0, 1.0, 1.0, 1.0 },
                    .uv = .{ uv_x, uv_y },
                    .tangent = .{ u_axis.x, u_axis.y, u_axis.z, 1.0 },
                    .joints = skin_interp.joints,
                    .weights = skin_interp.weights,
                });
            } else if (options.parent_to_target) {
                // Target parented: transform biased world pos into target mesh local coords
                const pos = v.world_pos.add(v.normal.scale(options.depth_bias));
                const local_p = inv_target_mat.transformPoint(pos);
                const local_n = inv_target_mat.transformDirection(v.normal).normalize();

                try out_vertices.append(allocator, .{
                    .position = .{ local_p.x, local_p.y, local_p.z },
                    .normal = .{ local_n.x, local_n.y, local_n.z },
                    .color = .{ 1.0, 1.0, 1.0, 1.0 },
                    .uv = .{ uv_x, uv_y },
                    .tangent = .{ u_axis.x, u_axis.y, u_axis.z, 1.0 },
                    .joints = .{ 0.0, 0.0, 0.0, 0.0 },
                    .weights = .{ 1.0, 0.0, 0.0, 0.0 },
                });
            } else {
                // Static world-space decal
                const pos = v.world_pos.add(v.normal.scale(options.depth_bias));
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
    decal_mesh.is_decal = true;
    decal_mesh.cast_shadows = false;

    if (target_mesh.skeleton != null and target_mesh.cpu_skin.len == target_mesh.cpu_positions.len) {
        decal_mesh.skeleton = target_mesh.skeleton;
        decal_mesh.base_matrix = target_mesh.base_matrix;
        decal_mesh.position = target_mesh.position;
        decal_mesh.rotation = target_mesh.rotation;
        decal_mesh.scaling = target_mesh.scaling;
        decal_mesh.parent = target_mesh.parent;
    } else if (options.parent_to_target) {
        decal_mesh.parent = @constCast(target_mesh);
        decal_mesh.position = Vec3.zero;
        decal_mesh.rotation = Vec3.zero;
        decal_mesh.scaling = Vec3.one;
    }
    return decal_mesh;
}

/// Dynamic Decal Projector component that projects oriented decal volumes
/// onto single meshes or across multiple intersecting scene meshes.
pub const DecalProjector = struct {
    position: Vec3,
    normal: Vec3,
    size: Vec3 = Vec3.new(1.0, 1.0, 1.0),
    angle: f32 = 0.0,
    cull_backfaces: bool = true,
    depth_bias: f32 = 0.004,
    parent_to_target: bool = false,

    pub fn toDecalOptions(self: DecalProjector) DecalOptions {
        return .{
            .position = self.position,
            .normal = self.normal,
            .size = self.size,
            .angle = self.angle,
            .cull_backfaces = self.cull_backfaces,
            .depth_bias = self.depth_bias,
            .parent_to_target = self.parent_to_target,
        };
    }

    /// Project onto a specific target mesh (static or skinned).
    pub fn projectMesh(self: DecalProjector, scene: *Scene, name: []const u8, target_mesh: *const Mesh) !*Mesh {
        return createDecal(scene, name, target_mesh, self.toDecalOptions());
    }

    /// Builds unified decal geometry projecting across multiple candidate meshes (e.g. adjacent wall & floor).
    pub fn buildMultiMeshDecalData(
        self: DecalProjector,
        allocator: std.mem.Allocator,
        target_meshes: []const *const Mesh,
    ) !GeometryData {
        var combined_verts = std.ArrayList(Vertex).empty;
        defer combined_verts.deinit(allocator);
        var combined_indices = std.ArrayList(u32).empty;
        defer combined_indices.deinit(allocator);

        var opts = self.toDecalOptions();
        opts.parent_to_target = false; // Multi-mesh projection is always unified in world space

        for (target_meshes) |mesh| {
            var data = try buildDecalData(allocator, mesh, opts);
            defer data.deinit(allocator);
            if (data.vertices.len == 0) continue;

            const base_idx: u32 = @intCast(combined_verts.items.len);
            try combined_verts.appendSlice(allocator, data.vertices);
            for (data.indices) |idx| {
                try combined_indices.append(allocator, base_idx + idx);
            }
        }

        if (combined_verts.items.len == 0) {
            return GeometryData{
                .vertices = try allocator.alloc(Vertex, 0),
                .indices = try allocator.alloc(u32, 0),
                .bounds = BoundingBox.zero,
            };
        }

        var min_p = Vec3.new(std.math.inf(f32), std.math.inf(f32), std.math.inf(f32));
        var max_p = Vec3.new(-std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32));
        for (combined_verts.items) |v| {
            min_p.x = @min(min_p.x, v.position[0]);
            min_p.y = @min(min_p.y, v.position[1]);
            min_p.z = @min(min_p.z, v.position[2]);
            max_p.x = @max(max_p.x, v.position[0]);
            max_p.y = @max(max_p.y, v.position[1]);
            max_p.z = @max(max_p.z, v.position[2]);
        }

        return GeometryData{
            .vertices = try combined_verts.toOwnedSlice(allocator),
            .indices = try combined_indices.toOwnedSlice(allocator),
            .bounds = BoundingBox.init(min_p, max_p),
        };
    }

    /// Spatial query that finds all scene meshes intersecting the projector volume and
    /// stamps a unified multi-mesh decal. Returns null if no intersection occurs.
    pub fn projectScene(self: DecalProjector, scene: *Scene, name: []const u8) !?*Mesh {
        var candidates = std.ArrayList(*const Mesh).empty;
        defer candidates.deinit(scene.allocator);

        const radius = 0.5 * self.size.length();
        const min_p = self.position.sub(Vec3.splat(radius));
        const max_p = self.position.add(Vec3.splat(radius));
        const proj_aabb = BoundingBox.init(min_p, max_p);

        for (scene.meshes.items) |mesh| {
            if (!mesh.is_visible or mesh.is_lod_child or mesh.is_decal) continue;
            if (mesh.cpu_positions.len < 3 or mesh.cpu_indices.len < 3) continue;
            if (!mesh.getWorldBoundingBox().intersects(proj_aabb)) continue;
            try candidates.append(scene.allocator, mesh);
        }

        if (candidates.items.len == 0) return null;

        var data = try self.buildMultiMeshDecalData(scene.allocator, candidates.items);
        if (data.vertices.len == 0) {
            data.deinit(scene.allocator);
            return null;
        }
        defer data.deinit(scene.allocator);

        const decal_mesh = try mesh_mod.uploadGeometry(scene, name, data);
        decal_mesh.culling_strategy = .frustum;
        decal_mesh.is_decal = true;
        decal_mesh.cast_shadows = false;
        return decal_mesh;
    }
};

pub const DecalSpawnOptions = struct {
    lifetime: f32 = 0.0, // <= 0.0 means permanent until evicted by max_decals
    fade_duration: f32 = 2.0,
    albedo_color: Color3 = Color3.white,
    alpha_mode: AlphaMode = .blend,
};

pub const DecalInstance = struct {
    mesh: *Mesh,
    material: *PBRMaterial,
    base_color: Color3,
    lifetime: f32,
    fade_duration: f32,
    elapsed: f32 = 0.0,
};

/// Dynamic Decal Manager that handles a ring-buffer pool of active decals,
/// automatic lifetime tracking, smooth alpha fade-out, and memory recycling.
pub const DecalManager = struct {
    allocator: std.mem.Allocator,
    scene: *Scene,
    max_decals: usize = 128,
    instances: std.ArrayListUnmanaged(DecalInstance) = .empty,

    pub fn init(scene: *Scene, max_decals: usize) DecalManager {
        return .{
            .allocator = scene.allocator,
            .scene = scene,
            .max_decals = max_decals,
        };
    }

    pub fn deinit(self: *DecalManager) void {
        self.clear();
        self.instances.deinit(self.allocator);
    }

    pub fn clear(self: *DecalManager) void {
        while (self.instances.items.len > 0) {
            self.destroyOldest();
        }
    }

    pub fn destroyOldest(self: *DecalManager) void {
        if (self.instances.items.len == 0) return;
        const old = self.instances.orderedRemove(0);
        self.scene.destroyMesh(old.mesh);
        self.scene.destroyPBRMaterial(old.material);
    }

    pub fn spawnDecal(
        self: *DecalManager,
        projector: DecalProjector,
        target_mesh: *const Mesh,
        texture: ?Texture,
        options: DecalSpawnOptions,
    ) !*Mesh {
        if (self.instances.items.len >= self.max_decals) {
            self.destroyOldest();
        }

        const mesh = try projector.projectMesh(self.scene, "dynamic_decal", target_mesh);
        errdefer self.scene.destroyMesh(mesh);

        const mat = try self.scene.createPBRMaterial("dynamic_decal_mat");
        errdefer self.scene.destroyPBRMaterial(mat);

        mat.albedo_texture = texture;
        mat.albedo_color = options.albedo_color;
        mat.alpha_mode = options.alpha_mode;
        mat.double_sided = true;
        mesh.setPBRMaterial(mat);

        try self.instances.append(self.allocator, .{
            .mesh = mesh,
            .material = mat,
            .base_color = options.albedo_color,
            .lifetime = options.lifetime,
            .fade_duration = options.fade_duration,
        });

        return mesh;
    }

    pub fn spawnSceneDecal(
        self: *DecalManager,
        projector: DecalProjector,
        texture: ?Texture,
        options: DecalSpawnOptions,
    ) !?*Mesh {
        if (self.instances.items.len >= self.max_decals) {
            self.destroyOldest();
        }

        const mesh = (try projector.projectScene(self.scene, "dynamic_decal")) orelse return null;
        errdefer self.scene.destroyMesh(mesh);

        const mat = try self.scene.createPBRMaterial("dynamic_decal_mat");
        errdefer self.scene.destroyPBRMaterial(mat);

        mat.albedo_texture = texture;
        mat.albedo_color = options.albedo_color;
        mat.alpha_mode = options.alpha_mode;
        mat.double_sided = true;
        mesh.setPBRMaterial(mat);

        try self.instances.append(self.allocator, .{
            .mesh = mesh,
            .material = mat,
            .base_color = options.albedo_color,
            .lifetime = options.lifetime,
            .fade_duration = options.fade_duration,
        });

        return mesh;
    }

    pub fn update(self: *DecalManager, dt: f32) void {
        var i: usize = 0;
        while (i < self.instances.items.len) {
            var inst = &self.instances.items[i];
            if (inst.lifetime <= 0.0) {
                i += 1;
                continue;
            }

            inst.elapsed += dt;
            if (inst.elapsed >= inst.lifetime) {
                const fade_time = inst.elapsed - inst.lifetime;
                if (fade_time >= inst.fade_duration or inst.fade_duration <= 1e-4) {
                    const removed = self.instances.orderedRemove(i);
                    self.scene.destroyMesh(removed.mesh);
                    self.scene.destroyPBRMaterial(removed.material);
                    continue;
                } else {
                    const factor = 1.0 - (fade_time / inst.fade_duration);
                    inst.material.albedo_color = inst.base_color.scale(factor);
                }
            }
            i += 1;
        }
    }
};
