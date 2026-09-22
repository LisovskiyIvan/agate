//! Flattening + supersampled rasterization (split out of `ttf.zig`, facade).
//!
//! Owns `Segment`, `flattenContours` (quadratic subdivision in y-down
//! device space) and `rasterizeSegments` (4x4 non-zero-winding fill).
//! Imports `types` + the `outline` sibling for the contour types only.

const std = @import("std");
const types = @import("types.zig");
const outline_mod = @import("outline.zig");

const TtfError = types.TtfError;
const max_glyph_bitmap: u32 = types.max_glyph_bitmap;
const OutlinePoint = outline_mod.OutlinePoint;
const Contour = outline_mod.Contour;

pub const Segment = struct {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
};

/// Flattens quadratic contours into line segments in a y-DOWN device
/// space: device = (ux * scale + tx, -(uy * scale) + ty). `tol` is the
/// maximum quadratic deviation in device pixels (0.25 is a good default).
/// Implied on-curve midpoints are inserted between consecutive off-curve
/// points per the TrueType spec.
pub fn flattenContours(
    allocator: std.mem.Allocator,
    contours: []const Contour,
    scale: f32,
    tx: f32,
    ty: f32,
    tol: f32,
) TtfError![]Segment {
    var segs: std.ArrayListUnmanaged(Segment) = .empty;
    errdefer segs.deinit(allocator);
    for (contours) |c| {
        const n = c.points.len;
        if (n == 0) continue;
        if (n == 1) {
            // Degenerate single-point contour: nothing to fill.
            continue;
        }
        // Expand: insert implied on-curve midpoints between consecutive
        // off-curve points (wrapping).
        var exp: std.ArrayListUnmanaged(OutlinePoint) = .empty;
        defer exp.deinit(allocator);
        for (c.points, 0..) |p, k| {
            try exp.append(allocator, p);
            const q = c.points[(k + 1) % n];
            if (!p.on_curve and !q.on_curve) {
                try exp.append(allocator, .{
                    .x = (p.x + q.x) * 0.5,
                    .y = (p.y + q.y) * 0.5,
                    .on_curve = true,
                });
            }
        }
        const m = exp.items.len;
        // Start at an on-curve point (one always exists after expansion:
        // a contour of all off-curve points gains midpoints everywhere).
        var start: usize = 0;
        while (start < m and !exp.items[start].on_curve) : (start += 1) {}
        if (start >= m) continue;
        var j = start;
        while (true) {
            const cur = exp.items[j % m];
            const nxt = exp.items[(j + 1) % m];
            if (nxt.on_curve) {
                try segs.append(allocator, .{
                    .x0 = cur.x * scale + tx,
                    .y0 = -(cur.y * scale) + ty,
                    .x1 = nxt.x * scale + tx,
                    .y1 = -(nxt.y * scale) + ty,
                });
                j += 1;
            } else {
                const nn = exp.items[(j + 2) % m];
                try flattenQuad(allocator, &segs, cur, nxt, nn, scale, tx, ty, tol, 0);
                j += 2;
            }
            if (j % m == start % m and j > start) break;
            if (j - start > m + 2) break; // paranoia: never spin
        }
    }
    return segs.toOwnedSlice(allocator) catch return error.OutOfMemory;
}

fn flattenQuad(
    allocator: std.mem.Allocator,
    segs: *std.ArrayListUnmanaged(Segment),
    p0: OutlinePoint,
    c: OutlinePoint,
    p1: OutlinePoint,
    scale: f32,
    tx: f32,
    ty: f32,
    tol: f32,
    level: u8,
) TtfError!void {
    // Deviation of the curve midpoint from the chord midpoint, in font
    // units; comparing against tol/scale keeps the recursion bounded and
    // resolution-independent (level cap is the backstop, never the path).
    const mx = (p0.x + 2.0 * c.x + p1.x) * 0.25;
    const my = (p0.y + 2.0 * c.y + p1.y) * 0.25;
    const cx = (p0.x + p1.x) * 0.5;
    const cy = (p0.y + p1.y) * 0.5;
    const dev = @max(@abs(mx - cx), @abs(my - cy)) * scale;
    if (dev <= tol or level >= 12) {
        try segs.append(allocator, .{
            .x0 = p0.x * scale + tx,
            .y0 = -(p0.y * scale) + ty,
            .x1 = p1.x * scale + tx,
            .y1 = -(p1.y * scale) + ty,
        });
        return;
    }
    const p01 = OutlinePoint{ .x = (p0.x + c.x) * 0.5, .y = (p0.y + c.y) * 0.5, .on_curve = true };
    const p12 = OutlinePoint{ .x = (c.x + p1.x) * 0.5, .y = (c.y + p1.y) * 0.5, .on_curve = true };
    const mid = OutlinePoint{ .x = (p01.x + p12.x) * 0.5, .y = (p01.y + p12.y) * 0.5, .on_curve = true };
    try flattenQuad(allocator, segs, p0, p01, mid, scale, tx, ty, tol, level + 1);
    try flattenQuad(allocator, segs, mid, p12, p1, scale, tx, ty, tol, level + 1);
}

/// 4x4-supersampled scanline fill with non-zero winding into 8-bit alpha.
/// `segs` are y-down device pixels; the bitmap covers
/// [ox, ox+w) x [oy, oy+h). Returns owned w*h bytes (0 = transparent).
pub fn rasterizeSegments(
    allocator: std.mem.Allocator,
    segs: []const Segment,
    ox: i32,
    oy: i32,
    w: u32,
    h: u32,
) TtfError![]u8 {
    if (w == 0 or h == 0) return allocator.alloc(u8, 0) catch return error.OutOfMemory;
    if (w > max_glyph_bitmap or h > max_glyph_bitmap) return error.GlyphTooLarge;
    const out = allocator.alloc(u8, @as(usize, w) * h) catch return error.OutOfMemory;
    @memset(out, 0);
    const ss: u32 = 4;
    const fx = @as(f32, @floatFromInt(ox));
    const fy = @as(f32, @floatFromInt(oy));
    var row: u32 = 0;
    while (row < h) : (row += 1) {
        var col: u32 = 0;
        while (col < w) : (col += 1) {
            var covered: u32 = 0;
            var sy: u32 = 0;
            while (sy < ss) : (sy += 1) {
                var sx: u32 = 0;
                while (sx < ss) : (sx += 1) {
                    const px = fx + @as(f32, @floatFromInt(col)) +
                        (@as(f32, @floatFromInt(sx)) + 0.5) / @as(f32, ss);
                    const py = fy + @as(f32, @floatFromInt(row)) +
                        (@as(f32, @floatFromInt(sy)) + 0.5) / @as(f32, ss);
                    if (windingAt(segs, px, py) != 0) covered += 1;
                }
            }
            out[@as(usize, row) * w + col] = @intCast(covered * 255 / (ss * ss));
        }
    }
    return out;
}

fn windingAt(segs: []const Segment, px: f32, py: f32) i32 {
    var winding: i32 = 0;
    for (segs) |s| {
        const above0 = s.y0 > py;
        const above1 = s.y1 > py;
        if (above0 == above1) continue;
        const t = (py - s.y0) / (s.y1 - s.y0);
        const xi = s.x0 + t * (s.x1 - s.x0);
        if (xi > px) winding += if (s.y1 > s.y0) 1 else -1;
    }
    return winding;
}
