//! Glyph outline extraction (split out of `ttf.zig`, facade).
//!
//! Owns `OutlinePoint`/`Contour`, `extractOutline` (simple + composite,
//! bounded recursion) and `freeContours`. Imports `types` + the `font`
//! sibling for the `Font` type only (`font: *const Font` params — same
//! leaf-to-leaf discipline as `audio/*`). `freeContours` is `pub` for the
//! `atlas` sibling and the facade tests but is NOT re-exported by the
//! facade, so the public surface is unchanged.

const std = @import("std");
const types = @import("types.zig");
const font_mod = @import("font.zig");

const TtfError = types.TtfError;
const readU16At = types.readU16At;
const max_contours: usize = types.max_contours;
const max_points: usize = types.max_points;
const max_components: usize = types.max_components;
const max_composite_depth: u8 = types.max_composite_depth;
const Font = font_mod.Font;

pub const OutlinePoint = struct {
    x: f32, // font units (fractional after composite transforms)
    y: f32,
    on_curve: bool,
};

pub const Contour = struct {
    points: []OutlinePoint,
};

pub fn freeContours(allocator: std.mem.Allocator, contours: []Contour) void {
    for (contours) |c| allocator.free(c.points);
    allocator.free(contours);
}

/// Extracts the outline of `gid` (simple or composite) in font units.
/// Out-of-range ids and empty glyphs yield zero contours. Caller frees
/// with the matching free (see `TtfFont`, which owns this transiently).
pub fn extractOutline(
    allocator: std.mem.Allocator,
    font: *const Font,
    gid: u16,
) TtfError![]Contour {
    var ancestors: [max_composite_depth]u16 = undefined;
    return extractOutlineDepth(allocator, font, gid, 0, ancestors[0..0]);
}

fn extractOutlineDepth(
    allocator: std.mem.Allocator,
    font: *const Font,
    gid: u16,
    depth: u8,
    ancestors: []const u16,
) TtfError![]Contour {
    var g = gid;
    if (g >= font.num_glyphs) g = 0;
    for (ancestors) |a| {
        if (a == g) return error.CyclicComposite;
    }
    const rec = try font.glyphRange(g);
    if (rec.len == 0) return allocator.alloc(Contour, 0) catch return error.OutOfMemory;
    if (rec.len < 10) return error.BadGlyf;
    const num_contours = std.mem.readInt(i16, rec[0..][0..2], .big);
    if (num_contours == 0) return allocator.alloc(Contour, 0) catch return error.OutOfMemory;
    if (num_contours > 0) return extractSimple(allocator, rec);
    return extractComposite(allocator, font, rec, depth, ancestors, g);
}

fn extractSimple(allocator: std.mem.Allocator, rec: []const u8) TtfError![]Contour {
    const num_contours = @as(usize, @intCast(std.mem.readInt(i16, rec[0..][0..2], .big)));
    if (num_contours > max_contours) return error.BadGlyf;
    if (rec.len < 10 + num_contours * 2 + 2) return error.Truncated;
    var num_points: usize = 0;
    var ci: usize = 0;
    var prev_end: i32 = -1;
    while (ci < num_contours) : (ci += 1) {
        const end = std.mem.readInt(u16, rec[10 + ci * 2 ..][0..2], .big);
        if (@as(i32, end) <= prev_end) return error.BadGlyf;
        prev_end = end;
        num_points = @as(usize, end) + 1;
    }
    if (num_points == 0 or num_points > max_points) return error.BadGlyf;

    var pos: usize = 10 + num_contours * 2;
    const instr_len = try readU16At(rec, pos);
    pos += 2;
    if (pos > rec.len or instr_len > rec.len - pos) return error.Truncated;
    pos += instr_len; // hinting instructions are skipped (no hinting support)

    // Flags (with repeat runs) — exactly numPoints flag values.
    var flags = allocator.alloc(u8, num_points) catch return error.OutOfMemory;
    defer allocator.free(flags);
    var fi: usize = 0;
    while (fi < num_points) {
        if (pos >= rec.len) return error.Truncated;
        const f = rec[pos];
        pos += 1;
        var repeat: usize = 1;
        if (f & 0x08 != 0) {
            if (pos >= rec.len) return error.Truncated;
            repeat = @as(usize, rec[pos]) + 1;
            pos += 1;
        }
        if (fi + repeat > num_points) return error.BadGlyf;
        @memset(flags[fi..][0..repeat], f);
        fi += repeat;
    }

    var xs = allocator.alloc(i32, num_points) catch return error.OutOfMemory;
    defer allocator.free(xs);
    var ys = allocator.alloc(i32, num_points) catch return error.OutOfMemory;
    defer allocator.free(ys);
    var x: i32 = 0;
    var y: i32 = 0;
    for (flags, 0..) |f, k| {
        const dx: i32 = if (f & 0x02 != 0) blk: {
            if (pos >= rec.len) return error.Truncated;
            const v: i32 = rec[pos];
            pos += 1;
            break :blk if (f & 0x10 != 0) v else -v;
        } else if (f & 0x10 != 0) 0 else blk: {
            if (pos + 2 > rec.len) return error.Truncated;
            const v = std.mem.readInt(i16, rec[pos..][0..2], .big);
            pos += 2;
            break :blk v;
        };
        x += dx;
        xs[k] = x;
    }
    for (flags, 0..) |f, k| {
        const dy: i32 = if (f & 0x04 != 0) blk: {
            if (pos >= rec.len) return error.Truncated;
            const v: i32 = rec[pos];
            pos += 1;
            break :blk if (f & 0x20 != 0) v else -v;
        } else if (f & 0x20 != 0) 0 else blk: {
            if (pos + 2 > rec.len) return error.Truncated;
            const v = std.mem.readInt(i16, rec[pos..][0..2], .big);
            pos += 2;
            break :blk v;
        };
        y += dy;
        ys[k] = y;
    }

    var contours = allocator.alloc(Contour, num_contours) catch return error.OutOfMemory;
    errdefer {
        for (contours) |*c| {
            if (c.points.len > 0) allocator.free(c.points);
        }
        allocator.free(contours);
    }
    // NUL the slices so errdefer never frees garbage on partial fill.
    for (contours) |*c| c.points = &.{};
    var start: usize = 0;
    ci = 0;
    while (ci < num_contours) : (ci += 1) {
        const end = @as(usize, std.mem.readInt(u16, rec[10 + ci * 2 ..][0..2], .big));
        const n = end + 1 - start;
        const pts = allocator.alloc(OutlinePoint, n) catch return error.OutOfMemory;
        for (pts, 0..) |*p, k| {
            p.* = .{
                .x = @floatFromInt(xs[start + k]),
                .y = @floatFromInt(ys[start + k]),
                .on_curve = flags[start + k] & 0x01 != 0,
            };
        }
        contours[ci].points = pts;
        start = end + 1;
    }
    return contours;
}

// Composite component flags (TrueType spec).
const comp_args_words: u16 = 0x0001;
const comp_args_xy: u16 = 0x0002;
const comp_have_scale: u16 = 0x0008;
const comp_more: u16 = 0x0020;
const comp_have_xy_scale: u16 = 0x0040;
const comp_have_2x2: u16 = 0x0080;
const comp_have_instr: u16 = 0x0100;
const comp_unscaled_offset: u16 = 0x1000;

fn f2dot14(v: u16) f32 {
    return @as(f32, @floatFromInt(@as(i16, @bitCast(v)))) / 16384.0;
}

fn extractComposite(
    allocator: std.mem.Allocator,
    font: *const Font,
    rec: []const u8,
    depth: u8,
    ancestors: []const u16,
    self_gid: u16,
) TtfError![]Contour {
    if (depth >= max_composite_depth) return error.CompositeTooDeep;
    var stack: [max_composite_depth]u16 = undefined;
    @memcpy(stack[0..ancestors.len], ancestors);
    stack[ancestors.len] = self_gid;
    const chain = stack[0 .. ancestors.len + 1];

    var out: std.ArrayListUnmanaged(Contour) = .empty;
    errdefer {
        for (out.items) |c| allocator.free(c.points);
        out.deinit(allocator);
    }
    var pos: usize = 10;
    var n_comp: usize = 0;
    while (true) {
        if (n_comp >= max_components) return error.BadGlyf;
        n_comp += 1;
        if (pos + 4 > rec.len) return error.Truncated;
        const flags = std.mem.readInt(u16, rec[pos..][0..2], .big);
        const sub_gid = std.mem.readInt(u16, rec[pos + 2 ..][0..2], .big);
        pos += 4;
        var dx: f32 = 0;
        var dy: f32 = 0;
        if (flags & comp_args_xy != 0) {
            if (flags & comp_args_words != 0) {
                if (pos + 4 > rec.len) return error.Truncated;
                dx = @floatFromInt(std.mem.readInt(i16, rec[pos..][0..2], .big));
                dy = @floatFromInt(std.mem.readInt(i16, rec[pos + 2 ..][0..2], .big));
                pos += 4;
            } else {
                if (pos + 2 > rec.len) return error.Truncated;
                dx = @floatFromInt(@as(i8, @bitCast(rec[pos])));
                dy = @floatFromInt(@as(i8, @bitCast(rec[pos + 1])));
                pos += 2;
            }
        } else {
            // Anchor-point positioning is NOT supported: matching anchor
            // points across glyphs needs full point indexing. The args are
            // consumed and treated as a zero offset (documented).
            if (flags & comp_args_words != 0) {
                if (pos + 4 > rec.len) return error.Truncated;
                pos += 4;
            } else {
                if (pos + 2 > rec.len) return error.Truncated;
                pos += 2;
            }
        }
        var a: f32 = 1;
        var b: f32 = 0;
        var cc: f32 = 0;
        var d: f32 = 1;
        if (flags & comp_have_scale != 0) {
            if (pos + 2 > rec.len) return error.Truncated;
            a = f2dot14(std.mem.readInt(u16, rec[pos..][0..2], .big));
            d = a;
            pos += 2;
        } else if (flags & comp_have_xy_scale != 0) {
            if (pos + 4 > rec.len) return error.Truncated;
            a = f2dot14(std.mem.readInt(u16, rec[pos..][0..2], .big));
            d = f2dot14(std.mem.readInt(u16, rec[pos + 2 ..][0..2], .big));
            pos += 4;
        } else if (flags & comp_have_2x2 != 0) {
            if (pos + 8 > rec.len) return error.Truncated;
            a = f2dot14(std.mem.readInt(u16, rec[pos..][0..2], .big));
            b = f2dot14(std.mem.readInt(u16, rec[pos + 2 ..][0..2], .big));
            cc = f2dot14(std.mem.readInt(u16, rec[pos + 4 ..][0..2], .big));
            d = f2dot14(std.mem.readInt(u16, rec[pos + 6 ..][0..2], .big));
            pos += 8;
        }
        // Offset scaling follows the Microsoft convention: the component
        // offset is transformed by the linear part unless the font asks
        // for UNSCALED_COMPONENT_OFFSET (Apple parity would need the
        // opposite default; documented as UNSCALED-only divergence).
        var ox = dx;
        var oy = dy;
        if (flags & comp_unscaled_offset == 0) {
            ox = a * dx + b * dy;
            oy = cc * dx + d * dy;
        }
        const child = try extractOutlineDepth(allocator, font, sub_gid, depth + 1, chain);
        defer allocator.free(child); // contour structs only; point arrays move out
        for (child) |*c| {
            for (c.points) |*p| {
                const nx = a * p.x + b * p.y + ox;
                const ny = cc * p.x + d * p.y + oy;
                p.x = nx;
                p.y = ny;
            }
            try out.append(allocator, c.*);
        }
        if (flags & comp_more == 0) break;
    }
    // WE_HAVE_INSTRUCTIONS bytes (hinting programs) are skipped.
    return out.toOwnedSlice(allocator) catch return error.OutOfMemory;
}
