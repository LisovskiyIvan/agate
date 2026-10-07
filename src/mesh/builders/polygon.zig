//! 2D shape triangulation (see `mesh/builders.zig`): flat polygons with
//! holes on the XZ/XY planes, or shallow prisms when depth is set, plus the
//! option structs. Imports the `common` sibling only; never the
//! `builders.zig` facade.
const std = @import("std");
const math = @import("math");
const Vec2 = math.Vec2;
const Vec3 = math.Vec3;
const Color4 = math.Color4;
const BoundingBox = math.BoundingBox;
const Vertex = @import("../types.zig").Vertex;
const GeometryData = @import("../types.zig").GeometryData;
const common = @import("common.zig");
const pointInTriangle2d = common.pointInTriangle2d;
const isEarTip = common.isEarTip;

pub const PolygonSideOrientation = enum {
    front,
    back,
    double_sided,
};

pub const PolygonPlane = enum {
    xz, // Ground plane, depth along +Y (default)
    xy, // Upright plane, depth along +Z
};

pub const PolygonOptions = struct {
    /// 2D outer perimeter contour in CCW order (CW is automatically normalized).
    shape: []const Vec2,
    /// Optional inner hole contours in CW order (CCW is automatically normalized).
    holes: []const []const Vec2 = &.{},
    /// Extrusion depth. 0.0 creates a flat 2D planar polygon; > 0.0 extrudes into a 3D prism.
    depth: f32 = 0.0,
    /// Reference plane for the polygon: .xz (ground) or .xy (upright).
    plane: PolygonPlane = .xz,
    /// Side orientation for flat polygon faces.
    side_orientation: PolygonSideOrientation = .front,
    /// UV multiplier.
    uv_scale: Vec2 = Vec2.one,
    color: Color4 = Color4.white,
};

fn cleanPolygonContour(allocator: std.mem.Allocator, src: []const Vec2) ![]Vec2 {
    if (src.len < 3) return error.InvalidPolygon;
    const clean = try allocator.alloc(Vec2, src.len);
    errdefer allocator.free(clean);
    var m: usize = 0;
    for (src) |p| {
        if (m > 0) {
            const dx = p.x - clean[m - 1].x;
            const dy = p.y - clean[m - 1].y;
            if (dx * dx + dy * dy < 1e-12) continue;
        }
        clean[m] = p;
        m += 1;
    }
    if (m > 1) {
        const dx = clean[m - 1].x - clean[0].x;
        const dy = clean[m - 1].y - clean[0].y;
        if (dx * dx + dy * dy < 1e-12) m -= 1;
    }
    if (m < 3) return error.InvalidPolygon;
    return clean[0..m];
}

fn polygonSignedArea2(poly: []const Vec2) f32 {
    var area2: f32 = 0.0;
    const m = poly.len;
    for (0..m) |i| {
        const a = poly[i];
        const b = poly[(i + 1) % m];
        area2 += a.x * b.y - b.x * a.y;
    }
    return area2;
}

fn reversePolygonContour(poly: []Vec2) void {
    if (poly.len < 2) return;
    var lo: usize = 0;
    var hi: usize = poly.len - 1;
    while (lo < hi) {
        const tmp = poly[lo];
        poly[lo] = poly[hi];
        poly[hi] = tmp;
        lo += 1;
        hi -= 1;
    }
}

pub fn buildPolygonData(allocator: std.mem.Allocator, options: PolygonOptions) !GeometryData {
    // 1. Clean and normalize outer boundary to CCW
    const outer_clean = try cleanPolygonContour(allocator, options.shape);
    defer allocator.free(outer_clean);
    const outer_area2 = polygonSignedArea2(outer_clean);
    if (@abs(outer_area2) < 1e-9) return error.InvalidPolygon;
    if (outer_area2 < 0.0) reversePolygonContour(outer_clean);

    // 2. Clean and normalize holes to CW
    var holes_clean: std.ArrayListUnmanaged([]Vec2) = .empty;
    defer {
        for (holes_clean.items) |h| allocator.free(h);
        holes_clean.deinit(allocator);
    }
    for (options.holes) |h_src| {
        const h = cleanPolygonContour(allocator, h_src) catch continue;
        const h_area2 = polygonSignedArea2(h);
        if (@abs(h_area2) < 1e-9) {
            allocator.free(h);
            continue;
        }
        if (h_area2 > 0.0) reversePolygonContour(h); // Ensure CW for holes
        try holes_clean.append(allocator, h);
    }

    // 3. Merge holes into outer contour via bridge edges
    var merged: std.ArrayListUnmanaged(Vec2) = .empty;
    defer merged.deinit(allocator);
    try merged.appendSlice(allocator, outer_clean);

    for (holes_clean.items) |hole| {
        // Find vertex in hole with maximum X
        var h_max_idx: usize = 0;
        var max_hx = hole[0].x;
        for (hole[1..], 1..) |p, i| {
            if (p.x > max_hx) {
                max_hx = p.x;
                h_max_idx = i;
            }
        }
        const h_pt = hole[h_max_idx];

        // Shoot horizontal ray from h_pt to the right (+X direction)
        var best_edge_idx: ?usize = null;
        var min_intersect_x: f32 = std.math.inf(f32);
        const m_len = merged.items.len;
        for (0..m_len) |ei| {
            const a = merged.items[ei];
            const b = merged.items[(ei + 1) % m_len];
            if ((a.y <= h_pt.y and b.y > h_pt.y) or (b.y <= h_pt.y and a.y > h_pt.y)) {
                const dy = b.y - a.y;
                if (@abs(dy) > 1e-7) {
                    const t_param = (h_pt.y - a.y) / dy;
                    const ix = a.x + t_param * (b.x - a.x);
                    if (ix >= h_pt.x and ix < min_intersect_x) {
                        min_intersect_x = ix;
                        best_edge_idx = ei;
                    }
                }
            }
        }

        if (best_edge_idx) |ei| {
            const a = merged.items[ei];
            const b = merged.items[(ei + 1) % m_len];
            var v_mut_idx: usize = if (a.x >= b.x) ei else (ei + 1) % m_len;

            const inter_pt = Vec2.new(min_intersect_x, h_pt.y);
            const cand_pt = merged.items[v_mut_idx];
            var min_slope: f32 = std.math.inf(f32);
            for (merged.items, 0..) |v, vi| {
                if (vi == v_mut_idx) continue;
                if (pointInTriangle2d(v, h_pt, inter_pt, cand_pt)) {
                    const dx = v.x - h_pt.x;
                    const dy = @abs(v.y - h_pt.y);
                    const slope = dy / @max(dx, 1e-6);
                    if (slope < min_slope) {
                        min_slope = slope;
                        v_mut_idx = vi;
                    }
                }
            }

            // Splice hole into merged polygon at v_mut_idx
            var splice: std.ArrayListUnmanaged(Vec2) = .empty;
            defer splice.deinit(allocator);
            for (h_max_idx..hole.len) |hi| try splice.append(allocator, hole[hi]);
            for (0..h_max_idx + 1) |hi| try splice.append(allocator, hole[hi]);
            try splice.append(allocator, merged.items[v_mut_idx]);
            try merged.insertSlice(allocator, v_mut_idx + 1, splice.items);
        }
    }

    // 4. Triangulate the merged contour via ear clipping
    const poly = merged.items;
    const m = poly.len;
    if (m < 3) return error.InvalidPolygon;

    const order = try allocator.alloc(usize, m);
    defer allocator.free(order);
    for (0..m) |k| order[k] = k;

    const cap_tris = try allocator.alloc([3]u32, m - 2);
    defer allocator.free(cap_tris);

    var live_count = m;
    var tri_count: usize = 0;
    var scan_idx: usize = 0;
    var loops_without_clip: usize = 0;
    while (live_count > 3) {
        const ip = order[(scan_idx + live_count - 1) % live_count];
        const ic = order[scan_idx % live_count];
        const inext = order[(scan_idx + 1) % live_count];
        if (isEarTip(poly, ip, ic, inext, order[0..live_count])) {
            cap_tris[tri_count] = .{ @intCast(ip), @intCast(ic), @intCast(inext) };
            tri_count += 1;
            const remove_at = scan_idx % live_count;
            var shift = remove_at;
            while (shift + 1 < live_count) : (shift += 1) {
                order[shift] = order[shift + 1];
            }
            live_count -= 1;
            loops_without_clip = 0;
        } else {
            scan_idx += 1;
            loops_without_clip += 1;
            if (loops_without_clip > live_count * 2) {
                cap_tris[tri_count] = .{ @intCast(ip), @intCast(ic), @intCast(inext) };
                tri_count += 1;
                const remove_at = scan_idx % live_count;
                var shift = remove_at;
                while (shift + 1 < live_count) : (shift += 1) {
                    order[shift] = order[shift + 1];
                }
                live_count -= 1;
                loops_without_clip = 0;
            }
        }
    }
    if (live_count == 3) {
        cap_tris[tri_count] = .{ @intCast(order[0]), @intCast(order[1]), @intCast(order[2]) };
        tri_count += 1;
    }

    // 5. Compute 2D bounds
    var min_x: f32 = outer_clean[0].x;
    var max_x: f32 = outer_clean[0].x;
    var min_y: f32 = outer_clean[0].y;
    var max_y: f32 = outer_clean[0].y;
    for (outer_clean[1..]) |p| {
        min_x = @min(min_x, p.x);
        max_x = @max(max_x, p.x);
        min_y = @min(min_y, p.y);
        max_y = @max(max_y, p.y);
    }
    const span_x = @max(max_x - min_x, 1e-4);
    const span_y = @max(max_y - min_y, 1e-4);

    const is_3d = options.depth > 1e-5;
    const depth = options.depth;
    const is_xz = (options.plane == .xz);

    if (!is_3d) {
        // Flat 2D polygon
        const double_sided = (options.side_orientation == .double_sided);
        const total_indices = tri_count * 3 * (if (double_sided) @as(usize, 2) else 1);
        const vertices = try allocator.alloc(Vertex, m);
        const indices = try allocator.alloc(u32, total_indices);

        const color_arr = options.color.toArray();
        for (poly, 0..) |p, i| {
            const u = (p.x - min_x) / span_x * options.uv_scale.x;
            const v = (p.y - min_y) / span_y * options.uv_scale.y;
            if (is_xz) {
                vertices[i] = .{
                    .position = .{ p.x, 0.0, p.y },
                    .normal = .{ 0.0, 1.0, 0.0 },
                    .color = color_arr,
                    .uv = .{ u, v },
                    .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
                };
            } else {
                vertices[i] = .{
                    .position = .{ p.x, p.y, 0.0 },
                    .normal = .{ 0.0, 0.0, 1.0 },
                    .color = color_arr,
                    .uv = .{ u, v },
                    .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
                };
            }
        }

        var ii: usize = 0;
        if (is_xz) {
            for (cap_tris[0..tri_count]) |tri| {
                indices[ii + 0] = tri[0];
                indices[ii + 1] = tri[2];
                indices[ii + 2] = tri[1];
                ii += 3;
            }
            if (double_sided) {
                for (cap_tris[0..tri_count]) |tri| {
                    indices[ii + 0] = tri[0];
                    indices[ii + 1] = tri[1];
                    indices[ii + 2] = tri[2];
                    ii += 3;
                }
            }
        } else {
            for (cap_tris[0..tri_count]) |tri| {
                indices[ii + 0] = tri[0];
                indices[ii + 1] = tri[1];
                indices[ii + 2] = tri[2];
                ii += 3;
            }
            if (double_sided) {
                for (cap_tris[0..tri_count]) |tri| {
                    indices[ii + 0] = tri[0];
                    indices[ii + 1] = tri[2];
                    indices[ii + 2] = tri[1];
                    ii += 3;
                }
            }
        }

        const b_min = if (is_xz) Vec3.new(min_x, -0.01, min_y) else Vec3.new(min_x, min_y, -0.01);
        const b_max = if (is_xz) Vec3.new(max_x, 0.01, max_y) else Vec3.new(max_x, max_y, 0.01);
        return .{
            .vertices = vertices,
            .indices = indices,
            .bounds = BoundingBox.init(b_min, b_max),
        };
    } else {
        // Extruded 3D prism: top cap, bottom cap, and side walls for outer + holes
        var total_side_edges: usize = outer_clean.len;
        for (holes_clean.items) |h| total_side_edges += h.len;

        const cap_vert_count = 2 * m;
        const side_vert_count = total_side_edges * 4;
        const total_verts = cap_vert_count + side_vert_count;

        const cap_index_count = 2 * tri_count * 3;
        const side_index_count = total_side_edges * 6;
        const total_indices = cap_index_count + side_index_count;

        const vertices = try allocator.alloc(Vertex, total_verts);
        const indices = try allocator.alloc(u32, total_indices);

        var vi: usize = 0;
        var ii: usize = 0;
        const color_arr = options.color.toArray();

        // Top Cap
        const top_base: u32 = @intCast(vi);
        for (poly) |p| {
            const u = (p.x - min_x) / span_x * options.uv_scale.x;
            const v = (p.y - min_y) / span_y * options.uv_scale.y;
            if (is_xz) {
                vertices[vi] = .{
                    .position = .{ p.x, depth, p.y },
                    .normal = .{ 0.0, 1.0, 0.0 },
                    .color = color_arr,
                    .uv = .{ u, v },
                    .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
                };
            } else {
                vertices[vi] = .{
                    .position = .{ p.x, p.y, depth },
                    .normal = .{ 0.0, 0.0, 1.0 },
                    .color = color_arr,
                    .uv = .{ u, v },
                    .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
                };
            }
            vi += 1;
        }

        // Bottom Cap
        const bot_base: u32 = @intCast(vi);
        for (poly) |p| {
            const u = (p.x - min_x) / span_x * options.uv_scale.x;
            const v = (p.y - min_y) / span_y * options.uv_scale.y;
            if (is_xz) {
                vertices[vi] = .{
                    .position = .{ p.x, 0.0, p.y },
                    .normal = .{ 0.0, -1.0, 0.0 },
                    .color = color_arr,
                    .uv = .{ u, v },
                    .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
                };
            } else {
                vertices[vi] = .{
                    .position = .{ p.x, p.y, 0.0 },
                    .normal = .{ 0.0, 0.0, -1.0 },
                    .color = color_arr,
                    .uv = .{ u, v },
                    .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
                };
            }
            vi += 1;
        }

        if (is_xz) {
            // Top cap indices (facing +Y)
            for (cap_tris[0..tri_count]) |tri| {
                indices[ii + 0] = top_base + tri[0];
                indices[ii + 1] = top_base + tri[2];
                indices[ii + 2] = top_base + tri[1];
                ii += 3;
            }

            // Bottom cap indices (facing -Y)
            for (cap_tris[0..tri_count]) |tri| {
                indices[ii + 0] = bot_base + tri[0];
                indices[ii + 1] = bot_base + tri[1];
                indices[ii + 2] = bot_base + tri[2];
                ii += 3;
            }
        } else {
            // Top cap indices (facing +Z, CCW)
            for (cap_tris[0..tri_count]) |tri| {
                indices[ii + 0] = top_base + tri[0];
                indices[ii + 1] = top_base + tri[1];
                indices[ii + 2] = top_base + tri[2];
                ii += 3;
            }

            // Bottom cap indices (facing -Z, reversed winding)
            for (cap_tris[0..tri_count]) |tri| {
                indices[ii + 0] = bot_base + tri[0];
                indices[ii + 1] = bot_base + tri[2];
                indices[ii + 2] = bot_base + tri[1];
                ii += 3;
            }
        }

        // Side walls helper
        const emitSideWalls = struct {
            fn emit(
                ring: []const Vec2,
                verts: []Vertex,
                inds: []u32,
                cur_vi: *usize,
                cur_ii: *usize,
                d: f32,
                xz_mode: bool,
                col: [4]f32,
                uv_s: Vec2,
            ) void {
                const r_len = ring.len;
                var dist_accum: f32 = 0.0;
                for (0..r_len) |edge_i| {
                    const p0 = ring[edge_i];
                    const p1 = ring[(edge_i + 1) % r_len];
                    const dx = p1.x - p0.x;
                    const dy = p1.y - p0.y;
                    const edge_len = @max(@sqrt(dx * dx + dy * dy), 1e-6);

                    const nx = dy / edge_len;
                    const ny = -dx / edge_len;
                    const tx = dx / edge_len;
                    const ty = dy / edge_len;

                    const uv_u0 = dist_accum * uv_s.x;
                    const uv_u1 = (dist_accum + edge_len) * uv_s.x;
                    dist_accum += edge_len;

                    const base_v: u32 = @intCast(cur_vi.*);

                    if (xz_mode) {
                        verts[cur_vi.* + 0] = .{
                            .position = .{ p0.x, 0.0, p0.y },
                            .normal = .{ nx, 0.0, ny },
                            .color = col,
                            .uv = .{ uv_u0, 0.0 },
                            .tangent = .{ tx, 0.0, ty, 1.0 },
                        };
                        verts[cur_vi.* + 1] = .{
                            .position = .{ p1.x, 0.0, p1.y },
                            .normal = .{ nx, 0.0, ny },
                            .color = col,
                            .uv = .{ uv_u1, 0.0 },
                            .tangent = .{ tx, 0.0, ty, 1.0 },
                        };
                        verts[cur_vi.* + 2] = .{
                            .position = .{ p0.x, d, p0.y },
                            .normal = .{ nx, 0.0, ny },
                            .color = col,
                            .uv = .{ uv_u0, d * uv_s.y },
                            .tangent = .{ tx, 0.0, ty, 1.0 },
                        };
                        verts[cur_vi.* + 3] = .{
                            .position = .{ p1.x, d, p1.y },
                            .normal = .{ nx, 0.0, ny },
                            .color = col,
                            .uv = .{ uv_u1, d * uv_s.y },
                            .tangent = .{ tx, 0.0, ty, 1.0 },
                        };
                    } else {
                        verts[cur_vi.* + 0] = .{
                            .position = .{ p0.x, p0.y, 0.0 },
                            .normal = .{ nx, ny, 0.0 },
                            .color = col,
                            .uv = .{ uv_u0, 0.0 },
                            .tangent = .{ tx, ty, 0.0, 1.0 },
                        };
                        verts[cur_vi.* + 1] = .{
                            .position = .{ p1.x, p1.y, 0.0 },
                            .normal = .{ nx, ny, 0.0 },
                            .color = col,
                            .uv = .{ uv_u1, 0.0 },
                            .tangent = .{ tx, ty, 0.0, 1.0 },
                        };
                        verts[cur_vi.* + 2] = .{
                            .position = .{ p0.x, p0.y, d },
                            .normal = .{ nx, ny, 0.0 },
                            .color = col,
                            .uv = .{ uv_u0, d * uv_s.y },
                            .tangent = .{ tx, ty, 0.0, 1.0 },
                        };
                        verts[cur_vi.* + 3] = .{
                            .position = .{ p1.x, p1.y, d },
                            .normal = .{ nx, ny, 0.0 },
                            .color = col,
                            .uv = .{ uv_u1, d * uv_s.y },
                            .tangent = .{ tx, ty, 0.0, 1.0 },
                        };
                    }
                    cur_vi.* += 4;

                    if (xz_mode) {
                        inds[cur_ii.* + 0] = base_v + 0;
                        inds[cur_ii.* + 1] = base_v + 2;
                        inds[cur_ii.* + 2] = base_v + 1;

                        inds[cur_ii.* + 3] = base_v + 1;
                        inds[cur_ii.* + 4] = base_v + 2;
                        inds[cur_ii.* + 5] = base_v + 3;
                    } else {
                        inds[cur_ii.* + 0] = base_v + 0;
                        inds[cur_ii.* + 1] = base_v + 1;
                        inds[cur_ii.* + 2] = base_v + 2;

                        inds[cur_ii.* + 3] = base_v + 1;
                        inds[cur_ii.* + 4] = base_v + 3;
                        inds[cur_ii.* + 5] = base_v + 2;
                    }
                    cur_ii.* += 6;
                }
            }
        }.emit;

        // Outer contour side walls
        emitSideWalls(outer_clean, vertices, indices, &vi, &ii, depth, is_xz, color_arr, options.uv_scale);

        // Hole contours side walls
        for (holes_clean.items) |hole| {
            emitSideWalls(hole, vertices, indices, &vi, &ii, depth, is_xz, color_arr, options.uv_scale);
        }

        const b_min = if (is_xz) Vec3.new(min_x, 0.0, min_y) else Vec3.new(min_x, min_y, 0.0);
        const b_max = if (is_xz) Vec3.new(max_x, depth, max_y) else Vec3.new(max_x, max_y, depth);
        return .{
            .vertices = vertices,
            .indices = indices,
            .bounds = BoundingBox.init(b_min, b_max),
        };
    }
}
