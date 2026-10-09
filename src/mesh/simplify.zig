//! Agate Mesh Simplification Subsystem.
//!
//! High-quality polygonal mesh decimation based on Garland & Heckbert
//! Quadric Error Metrics (QEM). Supports customizable decimation ratios,
//! target triangle counts, boundary protection, attribute interpolation
//! (UVs, colors, tangents), and automatic LOD level generation.

const std = @import("std");
const math = @import("math");
const Vec2 = math.Vec2;
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color4 = math.Color4;
const BoundingBox = math.BoundingBox;

const Mesh = @import("mesh.zig").Mesh;
const Vertex = @import("types.zig").Vertex;
const GeometryData = @import("types.zig").GeometryData;
const Scene = @import("../scene.zig").Scene;
const uploadGeometry = @import("mesh.zig").uploadGeometry;
const tangents = @import("tangents.zig");

/// Symmetric 4x4 Quadric error matrix for 3D points.
pub const Quadric3D = struct {
    a00: f32 = 0.0,
    a01: f32 = 0.0,
    a02: f32 = 0.0,
    a11: f32 = 0.0,
    a12: f32 = 0.0,
    a22: f32 = 0.0,
    b0: f32 = 0.0,
    b1: f32 = 0.0,
    b2: f32 = 0.0,
    c: f32 = 0.0,

    pub const zero = Quadric3D{};

    /// Creates a fundamental quadric matrix from a plane equation n . x + d = 0.
    pub fn fromPlane(n: Vec3, d: f32, weight: f32) Quadric3D {
        const w = @max(weight, 0.0);
        return .{
            .a00 = n.x * n.x * w,
            .a01 = n.x * n.y * w,
            .a02 = n.x * n.z * w,
            .a11 = n.y * n.y * w,
            .a12 = n.y * n.z * w,
            .a22 = n.z * n.z * w,
            .b0 = n.x * d * w,
            .b1 = n.y * d * w,
            .b2 = n.z * d * w,
            .c = d * d * w,
        };
    }

    pub fn add(self: Quadric3D, other: Quadric3D) Quadric3D {
        return .{
            .a00 = self.a00 + other.a00,
            .a01 = self.a01 + other.a01,
            .a02 = self.a02 + other.a02,
            .a11 = self.a11 + other.a11,
            .a12 = self.a12 + other.a12,
            .a22 = self.a22 + other.a22,
            .b0 = self.b0 + other.b0,
            .b1 = self.b1 + other.b1,
            .b2 = self.b2 + other.b2,
            .c = self.c + other.c,
        };
    }

    /// Evaluates quadric error v^T Q v at point p.
    pub fn evaluate(self: Quadric3D, p: Vec3) f32 {
        const ax = self.a00 * p.x + self.a01 * p.y + self.a02 * p.z;
        const ay = self.a01 * p.x + self.a11 * p.y + self.a12 * p.z;
        const az = self.a02 * p.x + self.a12 * p.y + self.a22 * p.z;
        const quad = p.x * ax + p.y * ay + p.z * az;
        const lin = 2.0 * (self.b0 * p.x + self.b1 * p.y + self.b2 * p.z);
        return @max(quad + lin + self.c, 0.0);
    }

    /// Solves for the optimal point that minimizes the quadric error,
    /// falling back to evaluating endpoints and midpoint if A is singular or ill-conditioned.
    pub fn solveOptimal(self: Quadric3D, p0: Vec3, p1: Vec3) Vec3 {
        // Compute determinant of A (3x3 top-left)
        const d0 = self.a11 * self.a22 - self.a12 * self.a12;
        const d1 = self.a02 * self.a12 - self.a01 * self.a22;
        const d2 = self.a01 * self.a12 - self.a02 * self.a11;
        const det = self.a00 * d0 + self.a01 * d1 + self.a02 * d2;

        if (@abs(det) > 1e-6) {
            const inv_det = 1.0 / det;
            const inv00 = d0 * inv_det;
            const inv01 = d1 * inv_det;
            const inv02 = d2 * inv_det;
            const inv11 = (self.a00 * self.a22 - self.a02 * self.a02) * inv_det;
            const inv12 = (self.a02 * self.a01 - self.a00 * self.a12) * inv_det;
            const inv22 = (self.a00 * self.a11 - self.a01 * self.a01) * inv_det;

            // v = - A^-1 * b
            const opt_x = -(inv00 * self.b0 + inv01 * self.b1 + inv02 * self.b2);
            const opt_y = -(inv01 * self.b0 + inv11 * self.b1 + inv12 * self.b2);
            const opt_z = -(inv02 * self.b0 + inv12 * self.b1 + inv22 * self.b2);
            const opt = Vec3.new(opt_x, opt_y, opt_z);

            if (!std.math.isNan(opt.x) and !std.math.isNan(opt.y) and !std.math.isNan(opt.z)) {
                // Verify candidate lies within a reasonable bounding zone around the edge
                const margin = 0.5 * p0.distance(p1) + 0.1;
                const min_b = Vec3.new(@min(p0.x, p1.x) - margin, @min(p0.y, p1.y) - margin, @min(p0.z, p1.z) - margin);
                const max_b = Vec3.new(@max(p0.x, p1.x) + margin, @max(p0.y, p1.y) + margin, @max(p0.z, p1.z) + margin);
                if (opt.x >= min_b.x and opt.x <= max_b.x and
                    opt.y >= min_b.y and opt.y <= max_b.y and
                    opt.z >= min_b.z and opt.z <= max_b.z)
                {
                    return opt;
                }
            }
        }

        // Fallback: evaluate endpoints and midpoint
        const mid = p0.add(p1).scale(0.5);
        const e0 = self.evaluate(p0);
        const e1 = self.evaluate(p1);
        const em = self.evaluate(mid);

        if (em <= e0 and em <= e1) return mid;
        if (e0 <= e1) return p0;
        return p1;
    }
};

pub const SimplifyOptions = struct {
    /// Target fraction of triangles to keep (e.g. 0.5 = 50% of faces, 0.25 = 25%).
    target_ratio: f32 = 0.5,
    /// Explicit target triangle count. If set, overrides `target_ratio`.
    target_triangles: ?usize = null,
    /// Maximum allowed quadric error before stopping simplification early.
    max_error: f32 = 1.0,
    /// Whether to strictly preserve boundary/silhouette edges on open surfaces.
    preserve_border: bool = true,
    /// Whether to linearly interpolate UV coordinates, vertex colors, and tangents.
    preserve_attributes: bool = true,
    /// Whether to reject edge collapses that invert or flip face normals.
    prevent_normal_flips: bool = true,
    /// Penalty weight added to boundary edge quadrics.
    border_penalty: f32 = 500.0,
};

const SimpVertex = struct {
    pos: Vec3,
    normal: Vec3,
    uv: Vec2,
    uv1: Vec2,
    color: Color4,
    tangent: [4]f32,
    quadric: Quadric3D,
    triangles: std.ArrayListUnmanaged(usize) = .empty,
    is_border: bool = false,
    deleted: bool = false,
};

const SimpTriangle = struct {
    v: [3]usize,
    normal: Vec3,
    area: f32,
    deleted: bool = false,
};

const SimpEdge = struct {
    u: usize,
    v: usize,
    target_pos: Vec3,
    cost: f32,
    valid: bool = true,
};

/// Simplifies a GeometryData mesh using Quadric Error Metrics edge collapse decimation.
pub fn simplifyGeometry(allocator: std.mem.Allocator, data: *const GeometryData, options: SimplifyOptions) !GeometryData {
    const num_indices = data.indices.len;
    if (num_indices < 3) return data.*;
    const num_triangles = num_indices / 3;
    const num_vertices = data.vertices.len;
    if (num_vertices < 3) return data.*;

    const target_tris: usize = if (options.target_triangles) |tt|
        @min(tt, num_triangles)
    else
        @max(@as(usize, @intFromFloat(@as(f32, @floatFromInt(num_triangles)) * std.math.clamp(options.target_ratio, 0.0, 1.0))), 1);

    if (target_tris >= num_triangles) {
        // No decimation requested: return identical copy
        const copy_verts = try allocator.dupe(Vertex, data.vertices);
        errdefer allocator.free(copy_verts);
        const copy_indices = try allocator.dupe(u32, data.indices);
        errdefer allocator.free(copy_indices);
        return .{
            .vertices = copy_verts,
            .indices = copy_indices,
            .bounds = data.bounds,
        };
    }

    // 1. Build vertex and triangle structures
    var vertices = try allocator.alloc(SimpVertex, num_vertices);
    defer {
        for (vertices) |*v| v.triangles.deinit(allocator);
        allocator.free(vertices);
    }

    for (data.vertices, 0..) |v, i| {
        vertices[i] = .{
            .pos = Vec3.new(v.position[0], v.position[1], v.position[2]),
            .normal = Vec3.new(v.normal[0], v.normal[1], v.normal[2]),
            .uv = Vec2.new(v.uv[0], v.uv[1]),
            .uv1 = Vec2.new(v.uv1[0], v.uv1[1]),
            .color = Color4.new(v.color[0], v.color[1], v.color[2], v.color[3]),
            .tangent = v.tangent,
            .quadric = Quadric3D.zero,
        };
    }

    var triangles = try allocator.alloc(SimpTriangle, num_triangles);
    defer allocator.free(triangles);

    for (0..num_triangles) |i| {
        const idx0 = data.indices[i * 3 + 0];
        const idx1 = data.indices[i * 3 + 1];
        const idx2 = data.indices[i * 3 + 2];

        const p0 = vertices[idx0].pos;
        const p1 = vertices[idx1].pos;
        const p2 = vertices[idx2].pos;

        const e1 = p1.sub(p0);
        const e2 = p2.sub(p0);
        const n_unnorm = e1.cross(e2);
        const len = n_unnorm.length();
        const area = 0.5 * len;
        const norm = if (len > 1e-7) n_unnorm.scale(1.0 / len) else Vec3.up;

        triangles[i] = .{
            .v = .{ idx0, idx1, idx2 },
            .normal = norm,
            .area = area,
        };

        try vertices[idx0].triangles.append(allocator, i);
        try vertices[idx1].triangles.append(allocator, i);
        try vertices[idx2].triangles.append(allocator, i);

        // Plane quadric weighted by triangle area
        const d = -norm.dot(p0);
        const q_face = Quadric3D.fromPlane(norm, d, area);
        vertices[idx0].quadric = vertices[idx0].quadric.add(q_face);
        vertices[idx1].quadric = vertices[idx1].quadric.add(q_face);
        vertices[idx2].quadric = vertices[idx2].quadric.add(q_face);
    }

    // 2. Identify boundary edges
    const EdgeKey = struct {
        min_v: usize,
        max_v: usize,

        fn init(a: usize, b: usize) @This() {
            return if (a < b) .{ .min_v = a, .max_v = b } else .{ .min_v = b, .max_v = a };
        }
    };

    var edge_counts = std.AutoHashMap(EdgeKey, u32).init(allocator);
    defer edge_counts.deinit();

    for (triangles) |tri| {
        for (0..3) |j| {
            const a = tri.v[j];
            const b = tri.v[(j + 1) % 3];
            const key = EdgeKey.init(a, b);
            const entry = try edge_counts.getOrPut(key);
            if (!entry.found_existing) entry.value_ptr.* = 0;
            entry.value_ptr.* += 1;
        }
    }

    var edge_it = edge_counts.iterator();
    while (edge_it.next()) |entry| {
        if (entry.value_ptr.* == 1) {
            // Topological boundary
            const u = entry.key_ptr.min_v;
            const v = entry.key_ptr.max_v;
            vertices[u].is_border = true;
            vertices[v].is_border = true;

            if (options.preserve_border) {
                // Add penalty quadric to lock the boundary plane
                const edge_dir = vertices[v].pos.sub(vertices[u].pos).normalize();
                const bnd_norm = vertices[u].normal.cross(edge_dir).normalize();
                const d = -bnd_norm.dot(vertices[u].pos);
                const q_bnd = Quadric3D.fromPlane(bnd_norm, d, options.border_penalty);
                vertices[u].quadric = vertices[u].quadric.add(q_bnd);
                vertices[v].quadric = vertices[v].quadric.add(q_bnd);
            }
        }
    }

    // 3. Build candidate edges list
    var edges: std.ArrayListUnmanaged(SimpEdge) = .empty;
    defer edges.deinit(allocator);

    var edge_it2 = edge_counts.keyIterator();
    while (edge_it2.next()) |k| {
        const u = k.min_v;
        const v = k.max_v;
        const cost_info = evaluateEdgeCollapse(vertices, triangles, u, v, options);
        try edges.append(allocator, .{
            .u = u,
            .v = v,
            .target_pos = cost_info.pos,
            .cost = cost_info.cost,
            .valid = cost_info.valid,
        });
    }

    // 4. Decimation loop
    var surviving_triangles = num_triangles;

    while (surviving_triangles > target_tris) {
        // Find minimum cost valid edge
        var best_idx: ?usize = null;
        var min_cost: f32 = std.math.inf(f32);

        for (edges.items, 0..) |*e, idx| {
            if (!e.valid) continue;
            if (vertices[e.u].deleted or vertices[e.v].deleted) {
                e.valid = false;
                continue;
            }
            if (e.cost < min_cost) {
                min_cost = e.cost;
                best_idx = idx;
            }
        }

        const idx = best_idx orelse break;
        if (min_cost > options.max_error) break;

        const best_edge = &edges.items[idx];
        best_edge.valid = false;

        const u = best_edge.u;
        const v = best_edge.v;
        const target_p = best_edge.target_pos;

        // Verify normal inversion before executing collapse
        if (options.prevent_normal_flips) {
            if (hasNormalFlip(vertices, triangles, u, v, target_p)) {
                continue;
            }
        }

        // Execute edge collapse u <- v
        const p_u = vertices[u].pos;
        const p_v = vertices[v].pos;
        const total_d = p_u.distance(p_v);
        const t: f32 = if (total_d > 1e-6) std.math.clamp(target_p.distance(p_u) / total_d, 0.0, 1.0) else 0.5;

        vertices[u].pos = target_p;
        if (options.preserve_attributes) {
            vertices[u].uv = Vec2.new(
                (1.0 - t) * vertices[u].uv.x + t * vertices[v].uv.x,
                (1.0 - t) * vertices[u].uv.y + t * vertices[v].uv.y,
            );
            vertices[u].uv1 = Vec2.new(
                (1.0 - t) * vertices[u].uv1.x + t * vertices[v].uv1.x,
                (1.0 - t) * vertices[u].uv1.y + t * vertices[v].uv1.y,
            );
            vertices[u].color = Color4.lerp(vertices[u].color, vertices[v].color, t);
        }
        vertices[u].quadric = vertices[u].quadric.add(vertices[v].quadric);
        if (vertices[v].is_border) vertices[u].is_border = true;

        vertices[v].deleted = true;

        // Update triangles incident to v: replace v with u
        for (vertices[v].triangles.items) |tri_idx| {
            var tri = &triangles[tri_idx];
            if (tri.deleted) continue;

            for (0..3) |j| {
                if (tri.v[j] == v) tri.v[j] = u;
            }

            // Check if triangle became degenerate
            if (tri.v[0] == tri.v[1] or tri.v[1] == tri.v[2] or tri.v[0] == tri.v[2]) {
                tri.deleted = true;
                surviving_triangles -= 1;
            } else {
                // Update face normal and area
                const q0 = vertices[tri.v[0]].pos;
                const q1 = vertices[tri.v[1]].pos;
                const q2 = vertices[tri.v[2]].pos;
                const norm_unnorm = q1.sub(q0).cross(q2.sub(q0));
                const len = norm_unnorm.length();
                tri.normal = if (len > 1e-7) norm_unnorm.scale(1.0 / len) else Vec3.up;
                tri.area = 0.5 * len;
                try vertices[u].triangles.append(allocator, tri_idx);
            }
        }

        // Re-evaluate edges incident to u
        for (edges.items) |*e| {
            if (!e.valid) continue;
            if (e.u == u or e.v == u or e.u == v or e.v == v) {
                if (e.u == v) e.u = u;
                if (e.v == v) e.v = u;
                if (e.u == e.v) {
                    e.valid = false;
                    continue;
                }
                // Recalculate cost
                const new_cost = evaluateEdgeCollapse(vertices, triangles, e.u, e.v, options);
                e.target_pos = new_cost.pos;
                e.cost = new_cost.cost;
                e.valid = new_cost.valid;
            }
        }
    }

    // 5. Compact surviving vertices and indices
    var vertex_remap = try allocator.alloc(i32, num_vertices);
    defer allocator.free(vertex_remap);
    @memset(vertex_remap, -1);

    var final_vertices: std.ArrayListUnmanaged(Vertex) = .empty;
    errdefer final_vertices.deinit(allocator);

    var final_indices: std.ArrayListUnmanaged(u32) = .empty;
    errdefer final_indices.deinit(allocator);

    var min_pt = Vec3.new(std.math.inf(f32), std.math.inf(f32), std.math.inf(f32));
    var max_pt = Vec3.new(-std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32));

    for (triangles) |tri| {
        if (tri.deleted) continue;

        for (0..3) |j| {
            const v_idx = tri.v[j];
            if (vertex_remap[v_idx] == -1) {
                const new_idx: u32 = @intCast(final_vertices.items.len);
                vertex_remap[v_idx] = @intCast(new_idx);

                const sv = &vertices[v_idx];
                try final_vertices.append(allocator, .{
                    .position = sv.pos.toArray(),
                    .normal = sv.normal.toArray(),
                    .uv = .{ sv.uv.x, sv.uv.y },
                    .uv1 = .{ sv.uv1.x, sv.uv1.y },
                    .color = sv.color.toArray(),
                    .tangent = sv.tangent,
                });

                min_pt = Vec3.new(
                    @min(min_pt.x, sv.pos.x),
                    @min(min_pt.y, sv.pos.y),
                    @min(min_pt.z, sv.pos.z),
                );
                max_pt = Vec3.new(
                    @max(max_pt.x, sv.pos.x),
                    @max(max_pt.y, sv.pos.y),
                    @max(max_pt.z, sv.pos.z),
                );
            }
            try final_indices.append(allocator, @intCast(vertex_remap[v_idx]));
        }
    }

    // 6. Recalculate smooth normals from surviving triangles
    const final_verts_slice = try final_vertices.toOwnedSlice(allocator);
    errdefer allocator.free(final_verts_slice);
    const final_indices_slice = try final_indices.toOwnedSlice(allocator);
    errdefer allocator.free(final_indices_slice);

    tangents.computeNormals(final_verts_slice, final_indices_slice, null);

    return .{
        .vertices = final_verts_slice,
        .indices = final_indices_slice,
        .bounds = BoundingBox.init(min_pt, max_pt),
    };
}

const CollapseEval = struct {
    pos: Vec3,
    cost: f32,
    valid: bool,
};

fn evaluateEdgeCollapse(
    vertices: []const SimpVertex,
    triangles: []const SimpTriangle,
    u: usize,
    v: usize,
    options: SimplifyOptions,
) CollapseEval {
    _ = triangles;
    const v_u = &vertices[u];
    const v_v = &vertices[v];
    if (v_u.deleted or v_v.deleted) return .{ .pos = Vec3.zero, .cost = std.math.inf(f32), .valid = false };

    // Border protection: only collapse along border or inward
    if (options.preserve_border) {
        if (v_u.is_border and !v_v.is_border) {
            // Keep border vertex position
            const q = v_u.quadric.add(v_v.quadric);
            return .{ .pos = v_u.pos, .cost = q.evaluate(v_u.pos), .valid = true };
        } else if (!v_u.is_border and v_v.is_border) {
            const q = v_u.quadric.add(v_v.quadric);
            return .{ .pos = v_v.pos, .cost = q.evaluate(v_v.pos), .valid = true };
        }
    }

    const q = v_u.quadric.add(v_v.quadric);
    const opt_pos = q.solveOptimal(v_u.pos, v_v.pos);
    const cost = q.evaluate(opt_pos);

    return .{
        .pos = opt_pos,
        .cost = cost,
        .valid = true,
    };
}

fn hasNormalFlip(
    vertices: []const SimpVertex,
    triangles: []const SimpTriangle,
    u: usize,
    v: usize,
    target_pos: Vec3,
) bool {
    // Check triangles incident to u
    for (vertices[u].triangles.items) |tri_idx| {
        const tri = &triangles[tri_idx];
        if (tri.deleted) continue;
        if (tri.v[0] == v or tri.v[1] == v or tri.v[2] == v) continue; // Will be deleted

        const p0 = if (tri.v[0] == u) target_pos else vertices[tri.v[0]].pos;
        const p1 = if (tri.v[1] == u) target_pos else vertices[tri.v[1]].pos;
        const p2 = if (tri.v[2] == u) target_pos else vertices[tri.v[2]].pos;

        const new_n = p1.sub(p0).cross(p2.sub(p0));
        const len = new_n.length();
        if (len < 1e-7) continue;
        const unit_n = new_n.scale(1.0 / len);

        if (unit_n.dot(tri.normal) < 0.2) return true;
    }

    // Check triangles incident to v
    for (vertices[v].triangles.items) |tri_idx| {
        const tri = &triangles[tri_idx];
        if (tri.deleted) continue;
        if (tri.v[0] == u or tri.v[1] == u or tri.v[2] == u) continue;

        const p0 = if (tri.v[0] == v) target_pos else vertices[tri.v[0]].pos;
        const p1 = if (tri.v[1] == v) target_pos else vertices[tri.v[1]].pos;
        const p2 = if (tri.v[2] == v) target_pos else vertices[tri.v[2]].pos;

        const new_n = p1.sub(p0).cross(p2.sub(p0));
        const len = new_n.length();
        if (len < 1e-7) continue;
        const unit_n = new_n.scale(1.0 / len);

        if (unit_n.dot(tri.normal) < 0.2) return true;
    }

    return false;
}

/// Simplifies a scene mesh and returns a newly created simplified Mesh.
pub fn simplifyMesh(
    allocator: std.mem.Allocator,
    scene: *Scene,
    name: []const u8,
    source_mesh: *Mesh,
    options: SimplifyOptions,
) !*Mesh {
    var source_geom = try source_mesh.toGeometryData(allocator);
    defer source_geom.deinit(allocator);

    var simplified_geom = try simplifyGeometry(allocator, &source_geom, options);
    defer simplified_geom.deinit(allocator);

    const new_mesh = try uploadGeometry(scene, name, simplified_geom);
    new_mesh.material = source_mesh.material;
    new_mesh.position = source_mesh.position;
    new_mesh.rotation = source_mesh.rotation;
    new_mesh.scaling = source_mesh.scaling;
    return new_mesh;
}

pub const LODLevelSpec = struct {
    distance: f32,
    ratio: f32,
    options: ?SimplifyOptions = null,
};

/// Automatically simplifies `source_mesh` at each distance bracket
/// and links the generated child meshes via `source_mesh.addLODLevel(...)`.
pub fn generateLODLevels(
    allocator: std.mem.Allocator,
    scene: *Scene,
    source_mesh: *Mesh,
    specs: []const LODLevelSpec,
) !void {
    var source_geom = try source_mesh.toGeometryData(allocator);
    defer source_geom.deinit(allocator);

    for (specs, 0..) |spec, idx| {
        var opts = spec.options orelse SimplifyOptions{};
        opts.target_ratio = spec.ratio;

        var lod_geom = try simplifyGeometry(allocator, &source_geom, opts);
        defer lod_geom.deinit(allocator);

        var name_buf: [64]u8 = undefined;
        const lod_name = std.fmt.bufPrint(&name_buf, "{s}_lod{d}", .{ source_mesh.name, idx + 1 }) catch "lod_mesh";

        const lod_mesh = try uploadGeometry(scene, lod_name, lod_geom);
        lod_mesh.material = source_mesh.material;
        lod_mesh.position = source_mesh.position;
        lod_mesh.rotation = source_mesh.rotation;
        lod_mesh.scaling = source_mesh.scaling;

        try source_mesh.addLODLevel(allocator, spec.distance, lod_mesh);
    }
}
