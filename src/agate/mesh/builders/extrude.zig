//! 2D profile to 3D prism builder (see `mesh/builders.zig`) plus its option
//! struct. Imports the `common` sibling only; never the `builders.zig`
//! facade.
const std = @import("std");
const math = @import("math");
const Vec2 = math.Vec2;
const Vec3 = math.Vec3;
const Color4 = math.Color4;
const BoundingBox = math.BoundingBox;
const Vertex = @import("../types.zig").Vertex;
const GeometryData = @import("../types.zig").GeometryData;
const common = @import("common.zig");
const storeQuad = common.storeQuad;
const isEarTip = common.isEarTip;
const segmentsCross2d = common.segmentsCross2d;

pub const ExtrudeOptions = struct {
    profile: []const Vec2, // closed 2D outline in XY (CCW preferred, CW is normalized)
    depth: f32 = 1.0, // extrusion distance along +Z, from z = 0 to z = depth
    capped: bool = true, // front/back caps triangulated with ear clipping
    uv_scale: Vec2 = Vec2.one, // UV multiplier (arclength/depth on sides, XY on caps)
    color: Color4 = Color4.white,
};

pub fn buildExtrudeData(allocator: std.mem.Allocator, options: ExtrudeOptions) !GeometryData {
    const src = options.profile;
    if (src.len < 3) return error.InvalidExtrude;

    const clean = try allocator.alloc(Vec2, src.len);
    defer allocator.free(clean);
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
    if (m < 3) return error.InvalidExtrude;
    const poly = clean[0..m];

    var area2: f32 = 0.0;
    for (0..m) |i| {
        const a = poly[i];
        const b = poly[(i + 1) % m];
        area2 += a.x * b.y - b.x * a.y;
    }
    if (@abs(area2) < 1e-9) return error.InvalidExtrude;
    if (area2 < 0.0) {
        var lo: usize = 0;
        var hi: usize = m - 1;
        while (lo < hi) {
            const tmp = poly[lo];
            poly[lo] = poly[hi];
            poly[hi] = tmp;
            lo += 1;
            hi -= 1;
        }
        area2 = -area2;
    }

    for (0..m) |i| {
        const a0 = poly[i];
        const a1 = poly[(i + 1) % m];
        for (i + 1..m) |j| {
            if (j == i + 1) continue;
            if (i == 0 and j == m - 1) continue;
            if (segmentsCross2d(a0, a1, poly[j], poly[(j + 1) % m])) {
                return error.InvalidExtrude;
            }
        }
    }

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
            scan_idx = (scan_idx + 1) % live_count;
            loops_without_clip += 1;
            if (loops_without_clip > live_count) return error.InvalidExtrude;
        }
    }
    cap_tris[tri_count] = .{ @intCast(order[0]), @intCast(order[1]), @intCast(order[2]) };
    tri_count += 1;

    const depth = options.depth;
    const color = options.color.toArray();
    const capped = options.capped;

    const side_verts = m * 4;
    const cap_verts = if (capped) m * 2 else 0;
    const total_verts = side_verts + cap_verts;

    const side_indices = m * 6;
    const cap_indices = if (capped) (m - 2) * 3 * 2 else 0;
    const total_indices = side_indices + cap_indices;

    const vertices = try allocator.alloc(Vertex, total_verts);
    errdefer allocator.free(vertices);
    const indices = try allocator.alloc(u32, total_indices);
    errdefer allocator.free(indices);

    var min_x = poly[0].x;
    var max_x = poly[0].x;
    var min_y = poly[0].y;
    var max_y = poly[0].y;
    for (poly[1..]) |p| {
        min_x = @min(min_x, p.x);
        max_x = @max(max_x, p.x);
        min_y = @min(min_y, p.y);
        max_y = @max(max_y, p.y);
    }

    var arclen: f32 = 0.0;
    var vi: usize = 0;
    var ii: usize = 0;
    for (0..m) |i| {
        const a = poly[i];
        const b = poly[(i + 1) % m];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const len = @sqrt(dx * dx + dy * dy);
        const inv = 1.0 / @max(len, 1e-9);
        const nx = dy * inv;
        const ny = -dx * inv;
        const us = arclen * options.uv_scale.x;
        arclen += len;
        const ue = arclen * options.uv_scale.x;
        const vs: f32 = 0.0;
        const ve = depth * options.uv_scale.y;
        const normal = [3]f32{ nx, ny, 0.0 };
        const tangent = [4]f32{ 0.0, 0.0, 1.0, 1.0 };
        const base: u32 = @intCast(vi);
        vertices[vi + 0] = .{ .position = .{ a.x, a.y, 0.0 }, .normal = normal, .color = color, .uv = .{ us, vs }, .tangent = tangent };
        vertices[vi + 1] = .{ .position = .{ b.x, b.y, 0.0 }, .normal = normal, .color = color, .uv = .{ ue, vs }, .tangent = tangent };
        vertices[vi + 2] = .{ .position = .{ b.x, b.y, depth }, .normal = normal, .color = color, .uv = .{ ue, ve }, .tangent = tangent };
        vertices[vi + 3] = .{ .position = .{ a.x, a.y, depth }, .normal = normal, .color = color, .uv = .{ us, ve }, .tangent = tangent };
        vi += 4;
        storeQuad(indices, ii, base, base + 1, base + 2, base + 3);
        ii += 6;
    }

    if (capped) {
        const front_base: u32 = @intCast(vi);
        for (poly) |p| {
            vertices[vi] = .{
                .position = .{ p.x, p.y, depth },
                .normal = .{ 0.0, 0.0, 1.0 },
                .color = color,
                .uv = .{ p.x * options.uv_scale.x, p.y * options.uv_scale.y },
                .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
            };
            vi += 1;
        }
        const back_base: u32 = @intCast(vi);
        for (poly) |p| {
            vertices[vi] = .{
                .position = .{ p.x, p.y, 0.0 },
                .normal = .{ 0.0, 0.0, -1.0 },
                .color = color,
                .uv = .{ p.x * options.uv_scale.x, p.y * options.uv_scale.y },
                .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
            };
            vi += 1;
        }
        for (cap_tris[0..tri_count]) |tri| {
            indices[ii + 0] = front_base + tri[0];
            indices[ii + 1] = front_base + tri[1];
            indices[ii + 2] = front_base + tri[2];
            ii += 3;
        }
        for (cap_tris[0..tri_count]) |tri| {
            indices[ii + 0] = back_base + tri[0];
            indices[ii + 1] = back_base + tri[2];
            indices[ii + 2] = back_base + tri[1];
            ii += 3;
        }
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(
            Vec3.new(min_x, min_y, @min(0.0, depth)),
            Vec3.new(max_x, max_y, @max(0.0, depth)),
        ),
    };
}
