//! Tests for `ttf.zig` (moved from `ttf.zig` inline blocks).
const std = @import("std");
const types = @import("ttf/types.zig");
const font_mod = @import("ttf/font.zig");
const outline = @import("ttf/outline.zig");
const raster = @import("ttf/raster.zig");
const atlas = @import("ttf/atlas.zig");
const fixture = @import("ttf/fixture.zig");
const testing = std.testing;
const ttf_mod = @import("ttf.zig");
const atlas_size = ttf_mod.atlas_size;
const sniff = ttf_mod.sniff;
const Font = ttf_mod.Font;
const extractOutline = ttf_mod.extractOutline;
const flattenContours = ttf_mod.flattenContours;
const rasterizeSegments = ttf_mod.rasterizeSegments;
const TtfFont = ttf_mod.TtfFont;
const buildFixture = ttf_mod.buildFixture;

const freeContours = outline.freeContours;
const ascii_set = [_]u21{ 'A', 'B', 'C' };

test "sniff accepts glyf sfnts and rejects CFF/foreign magics" {
    try testing.expect(sniff(&.{ 0, 1, 0, 0 }));
    try testing.expect(sniff(&.{ 't', 'r', 'u', 'e' }));
    try testing.expect(sniff(&.{ 't', 'y', 'p', '1' }));
    try testing.expect(!sniff(&.{ 'O', 'T', 'T', 'O' }));
    try testing.expect(!sniff(&.{ 'w', 'O', 'F', 'F' }));
    try testing.expect(!sniff(&.{ 0, 1, 0 }));
    try testing.expect(!sniff(&.{}));
}

test "parse reads the header and required tables" {
    const file = try buildFixture(testing.allocator, .{});
    defer testing.allocator.free(file);
    const font = try Font.parse(file);
    try testing.expectEqual(@as(u16, 1000), font.units_per_em);
    try testing.expectEqual(@as(u16, 4), font.num_glyphs);
    try testing.expectEqual(@as(i16, 800), font.ascender);
    try testing.expectEqual(@as(i16, -200), font.descender);
    try testing.expectEqual(@as(i16, 100), font.line_gap);
    try testing.expect(!font.loca_short);
    try testing.expect(font.cmap_fmt4 != null);
    try testing.expect(font.cmap_fmt12 == null);
    try testing.expect(font.kern != null);
}

test "cmap format 4 maps ASCII, unknown maps to .notdef" {
    const file = try buildFixture(testing.allocator, .{});
    defer testing.allocator.free(file);
    const font = try Font.parse(file);
    try testing.expectEqual(@as(u16, 1), try font.glyphIndex('A'));
    try testing.expectEqual(@as(u16, 2), try font.glyphIndex('B'));
    try testing.expectEqual(@as(u16, 3), try font.glyphIndex('C'));
    try testing.expectEqual(@as(u16, 0), try font.glyphIndex('Z'));
    try testing.expectEqual(@as(u16, 0), try font.glyphIndex(0x4E2D));
}

test "cmap format 4 with idRangeOffset and glyphIdArray" {
    const file = try buildFixture(testing.allocator, .{ .cmap = .fmt4_range_offset });
    defer testing.allocator.free(file);
    const font = try Font.parse(file);
    try testing.expect(font.cmap_fmt4 != null);
    try testing.expectEqual(@as(u16, 1), try font.glyphIndex('A'));
    try testing.expectEqual(@as(u16, 2), try font.glyphIndex('B'));
    try testing.expectEqual(@as(u16, 3), try font.glyphIndex('C'));
    try testing.expectEqual(@as(u16, 0), try font.glyphIndex('Z'));
}

test "cmap format 12 maps BMP and supplementary codepoints" {
    const file = try buildFixture(testing.allocator, .{ .cmap = .fmt12 });
    defer testing.allocator.free(file);
    const font = try Font.parse(file);
    try testing.expect(font.cmap_fmt4 == null);
    try testing.expect(font.cmap_fmt12 != null);
    try testing.expectEqual(@as(u16, 1), try font.glyphIndex(0x4E2D));
    try testing.expectEqual(@as(u16, 2), try font.glyphIndex(0x4E2E));
    try testing.expectEqual(@as(u16, 3), try font.glyphIndex(0x10437));
    try testing.expectEqual(@as(u16, 0), try font.glyphIndex('A'));
}

test "cmap both prefers format 12, falls back to format 4" {
    const file = try buildFixture(testing.allocator, .{ .cmap = .both });
    defer testing.allocator.free(file);
    const font = try Font.parse(file);
    try testing.expectEqual(@as(u16, 1), try font.glyphIndex(0x4E2D));
    try testing.expectEqual(@as(u16, 1), try font.glyphIndex('A'));
}

test "short loca format resolves the same glyph ranges" {
    const file = try buildFixture(testing.allocator, .{ .index_to_loc_format = 0 });
    defer testing.allocator.free(file);
    const font = try Font.parse(file);
    try testing.expect(font.loca_short);
    try testing.expectEqual(@as(u16, 2), try font.glyphIndex('B'));
    const r1 = try font.glyphRange(1);
    try testing.expect(r1.len > 10);
    // gid0 is an empty .notdef: header-only record (10 bytes), and it
    // extracts to zero contours.
    const r0 = try font.glyphRange(0);
    try testing.expectEqual(@as(usize, 10), r0.len);
    const c0 = try extractOutline(testing.allocator, &font, 0);
    defer freeContours(testing.allocator, c0);
    try testing.expectEqual(@as(usize, 0), c0.len);
    // Out-of-range gid resolves to .notdef instead of erroring.
    const rbig = try font.glyphRange(99);
    try testing.expectEqual(@as(usize, 10), rbig.len);
}

test "simple glyph outline extraction: rectangle" {
    const file = try buildFixture(testing.allocator, .{});
    defer testing.allocator.free(file);
    const font = try Font.parse(file);
    const contours = try extractOutline(testing.allocator, &font, 1);
    defer freeContours(testing.allocator, contours);
    try testing.expectEqual(@as(usize, 1), contours.len);
    try testing.expectEqual(@as(usize, 4), contours[0].points.len);
    for (contours[0].points) |p| try testing.expect(p.on_curve);
    try testing.expectEqual(@as(f32, 0), contours[0].points[0].x);
    try testing.expectEqual(@as(f32, 600), contours[0].points[1].x);
    try testing.expectEqual(@as(f32, 800), contours[0].points[2].y);
}

test "simple glyph with a quadratic curve flattens inside its bbox" {
    const file = try buildFixture(testing.allocator, .{});
    defer testing.allocator.free(file);
    const font = try Font.parse(file);
    const contours = try extractOutline(testing.allocator, &font, 2);
    defer freeContours(testing.allocator, contours);
    try testing.expectEqual(@as(usize, 1), contours.len);
    try testing.expectEqual(@as(usize, 3), contours[0].points.len);
    try testing.expect(!contours[0].points[1].on_curve);
    const segs = try flattenContours(testing.allocator, contours, 0.02, 0, 0, 0.25);
    defer testing.allocator.free(segs);
    // One quadratic must subdivide into several segments (not just 2).
    try testing.expect(segs.len > 4);
    for (segs) |s| {
        for ([_]f32{ s.x0, s.x1 }) |x| {
            try testing.expect(x >= -0.01 and x <= 12.01);
        }
        for ([_]f32{ s.y0, s.y1 }) |y| {
            // y-down device space: triangle apex (800 units) maps to -16.
            try testing.expect(y >= -16.01 and y <= 0.01);
        }
    }
}

test "composite glyph applies the component offset" {
    const file = try buildFixture(testing.allocator, .{});
    defer testing.allocator.free(file);
    const font = try Font.parse(file);
    const contours = try extractOutline(testing.allocator, &font, 3);
    defer freeContours(testing.allocator, contours);
    try testing.expectEqual(@as(usize, 1), contours.len);
    try testing.expectEqual(@as(usize, 4), contours[0].points.len);
    // Rectangle (0..600, 0..800) shifted by (100, 50).
    try testing.expectEqual(@as(f32, 100), contours[0].points[0].x);
    try testing.expectEqual(@as(f32, 50), contours[0].points[0].y);
    try testing.expectEqual(@as(f32, 700), contours[0].points[1].x);
    try testing.expectEqual(@as(f32, 850), contours[0].points[2].y);
}

test "rasterization covers the rectangle center, not the corners" {
    const file = try buildFixture(testing.allocator, .{});
    defer testing.allocator.free(file);
    const font = try Font.parse(file);
    const contours = try extractOutline(testing.allocator, &font, 1);
    defer freeContours(testing.allocator, contours);
    const scale = 20.0 / 1000.0; // 20px em
    const segs = try flattenContours(testing.allocator, contours, scale, 0, 0, 0.25);
    defer testing.allocator.free(segs);
    // Rectangle (0..600, 0..800) units -> device (0..12, -16..0).
    const alpha = try rasterizeSegments(testing.allocator, segs, 0, -16, 12, 16);
    defer testing.allocator.free(alpha);
    try testing.expectEqual(@as(u8, 255), alpha[8 * 12 + 6]); // center
    try testing.expectEqual(@as(u8, 255), alpha[1 * 12 + 1]);
    // A crop fully past the rect's right edge (x = 12) stays transparent.
    const cut = try rasterizeSegments(testing.allocator, segs, 12, -16, 4, 4);
    defer testing.allocator.free(cut);
    for (cut) |a| try testing.expectEqual(@as(u8, 0), a);
    // A crop straddling the edge is covered inside, empty outside.
    const edge = try rasterizeSegments(testing.allocator, segs, 11, -16, 2, 2);
    defer testing.allocator.free(edge);
    try testing.expectEqual(@as(u8, 255), edge[0]); // device x 11..12: inside
    try testing.expectEqual(@as(u8, 0), edge[1]); // device x 12..13: outside
}

test "hmtx advances match the em square" {
    const file = try buildFixture(testing.allocator, .{});
    defer testing.allocator.free(file);
    const font = try Font.parse(file);
    try testing.expectEqual(@as(u16, 700), font.advanceUnits(1));
    try testing.expectEqual(@as(u16, 650), font.advanceUnits(2));
    try testing.expectEqual(@as(u16, 500), font.advanceUnits(0));
    // 700/1000 em at 20px = 14px.
    try testing.expectApproxEqAbs(@as(f32, 14.0), @as(f32, @floatFromInt(font.advanceUnits(1))) * 0.02, 1e-5);
}

test "kern format 0 applies to the authored pair only" {
    const file = try buildFixture(testing.allocator, .{});
    defer testing.allocator.free(file);
    const font = try Font.parse(file);
    try testing.expectEqual(@as(i16, -80), try font.kernUnits(1, 2));
    try testing.expectEqual(@as(i16, 0), try font.kernUnits(2, 1));
    try testing.expectEqual(@as(i16, 0), try font.kernUnits(1, 3));
}

test "TtfFont bakes an atlas: no overlap, in bounds, sane metrics" {
    const file = try buildFixture(testing.allocator, .{});
    defer testing.allocator.free(file);
    var ttf = try TtfFont.init(testing.allocator, file, 20.0, &ascii_set);
    defer ttf.deinit();
    try testing.expectEqual(@as(usize, 3), ttf.glyphs.len);
    try testing.expectApproxEqAbs(@as(f32, 16.0), ttf.ascent, 1e-4); // 800 * 0.02
    try testing.expectApproxEqAbs(@as(f32, 22.0), ttf.line_height, 1e-4); // (800+200+100) * 0.02
    // Cells do not overlap and stay inside the page.
    for (ttf.glyphs, 0..) |a, k| {
        try testing.expect(a.ax + a.w <= atlas_size);
        try testing.expect(a.ay + a.h <= atlas_size);
        for (ttf.glyphs, 0..) |b, j| {
            if (k == j or a.w == 0 or b.w == 0) continue;
            const overlap = a.ax < b.ax + b.w and b.ax < a.ax + a.w and
                a.ay < b.ay + b.h and b.ay < a.ay + a.h;
            try testing.expect(!overlap);
        }
    }
    // Atlas texels: white RGB, coverage alpha; glyph A (rect) center opaque.
    const ga = ttf.lookup('A').?;
    const cx = ga.ax + ga.w / 2;
    const cy = ga.ay + ga.h / 2;
    const px = (@as(usize, cy) * atlas_size + cx) * 4;
    try testing.expectEqual(@as(u8, 255), ttf.atlas_pixels[px]);
    try testing.expectEqual(@as(u8, 255), ttf.atlas_pixels[px + 3]);
    // Advance + kern: "AB" = 14 + (-1.6) + 13 at 20px.
    const gb = ttf.lookup('B').?;
    const adv = ga.advance + ttf.kernPx(ga, gb) + gb.advance;
    try testing.expectApproxEqAbs(@as(f32, 14.0 - 1.6 + 13.0), adv, 1e-4);
    // Unknown codepoint falls back to .notdef (empty here: zero size).
    const missing = ttf.lookup('Z').?;
    try testing.expectEqual(@as(u16, 0), missing.gid);
}

test "TtfFont rejects an invalid pixel size" {
    const file = try buildFixture(testing.allocator, .{});
    defer testing.allocator.free(file);
    try testing.expectError(error.InvalidPixelSize, TtfFont.init(testing.allocator, file, 2.0, &ascii_set));
    try testing.expectError(error.InvalidPixelSize, TtfFont.init(testing.allocator, file, 512.0, &ascii_set));
}

test "bad magic is NotTtf, truncation is Truncated" {
    const file = try buildFixture(testing.allocator, .{});
    defer testing.allocator.free(file);
    var bad = try testing.allocator.dupe(u8, file);
    defer testing.allocator.free(bad);
    @memcpy(bad[0..4], "wOFF");
    try testing.expectError(error.NotTtf, Font.parse(bad));
    try testing.expectError(error.Truncated, Font.parse(file[0..40]));
    const cut = try buildFixture(testing.allocator, .{ .truncate_bytes = 64 });
    defer testing.allocator.free(cut);
    const err = Font.parse(cut) catch |e| e;
    try testing.expect(err == error.Truncated or err == error.BadLoca or err == error.BadTable);
}

test "OTF/CFF, variable and CFF-tabled fonts are rejected explicitly" {
    const file = try buildFixture(testing.allocator, .{});
    defer testing.allocator.free(file);
    var otto = try testing.allocator.dupe(u8, file);
    defer testing.allocator.free(otto);
    @memcpy(otto[0..4], "OTTO");
    try testing.expectError(error.UnsupportedCff, Font.parse(otto));

    const cff = try buildFixture(testing.allocator, .{ .include_cff_table = true });
    defer testing.allocator.free(cff);
    try testing.expectError(error.UnsupportedCff, Font.parse(cff));

    const fvar = try buildFixture(testing.allocator, .{ .include_fvar = true });
    defer testing.allocator.free(fvar);
    try testing.expectError(error.UnsupportedVariableFont, Font.parse(fvar));
}

test "cmap without format 4/12 is UnsupportedCmap" {
    const file = try buildFixture(testing.allocator, .{ .cmap = .unsupported });
    defer testing.allocator.free(file);
    try testing.expectError(error.UnsupportedCmap, Font.parse(file));
}

test "bad loca offsets are BadLoca, not hangs" {
    const file = try buildFixture(testing.allocator, .{ .loca = .out_of_range });
    defer testing.allocator.free(file);
    const font = try Font.parse(file);
    try testing.expectError(error.BadLoca, font.glyphRange(1));
}

test "cyclic composites error instead of hanging" {
    const self_ref = try buildFixture(testing.allocator, .{ .composite = .self_cyclic });
    defer testing.allocator.free(self_ref);
    const font = try Font.parse(self_ref);
    try testing.expectError(error.CyclicComposite, extractOutline(testing.allocator, &font, 3));

    const pair = try buildFixture(testing.allocator, .{ .composite = .pair_cyclic });
    defer testing.allocator.free(pair);
    const font2 = try Font.parse(pair);
    try testing.expectError(error.CyclicComposite, extractOutline(testing.allocator, &font2, 3));
}

test "deeply nested composites hit CompositeTooDeep" {
    const deep = try buildFixture(testing.allocator, .{ .composite = .deep });
    defer testing.allocator.free(deep);
    const font = try Font.parse(deep);
    try testing.expectError(error.CompositeTooDeep, extractOutline(testing.allocator, &font, 3));
}

test "lookup resolves every baked codepoint, misses fall back to .notdef" {
    const file = try buildFixture(testing.allocator, .{});
    defer testing.allocator.free(file);
    const font = try Font.parse(file);
    var ttf = try TtfFont.init(testing.allocator, file, 20.0, &ascii_set);
    defer ttf.deinit();
    for (ascii_set) |cp| {
        const g = ttf.lookup(cp).?;
        try testing.expectEqual(cp, g.codepoint);
        try testing.expectEqual(try font.glyphIndex(cp), g.gid);
    }
    // Unrequested codepoints (mapped or not) fall back to .notdef.
    const missing = ttf.lookup('Z').?;
    try testing.expectEqual(@as(u16, 0), missing.gid);
    try testing.expectEqual(@as(u21, 0xFFFD), missing.codepoint);
    const missing_uni = ttf.lookup(0x4E2D).?;
    try testing.expectEqual(@as(u16, 0), missing_uni.gid);
}

test "lookup with duplicate and empty charsets keeps first-match entries" {
    const file = try buildFixture(testing.allocator, .{});
    defer testing.allocator.free(file);
    const dupes = [_]u21{ 'A', 'B', 'A', 'C', 'B' };
    var ttf = try TtfFont.init(testing.allocator, file, 20.0, &dupes);
    defer ttf.deinit();
    try testing.expectEqual(dupes.len, ttf.glyphs.len);
    // Duplicate codepoints bake identical entries, so lookup must match
    // every same-codepoint slot (positions are normalized by the sort).
    for ([_]u21{ 'A', 'B' }) |cp| {
        const got = ttf.lookup(cp).?;
        try testing.expectEqual(cp, got.codepoint);
        for (ttf.glyphs) |g| {
            if (g.codepoint == cp) try testing.expectEqual(g, got);
        }
    }

    var empty = try TtfFont.init(testing.allocator, file, 20.0, &.{});
    defer empty.deinit();
    try testing.expectEqual(@as(usize, 0), empty.glyphs.len);
    const fallback = empty.lookup('A').?;
    try testing.expectEqual(@as(u16, 0), fallback.gid);
}

test "lookup of a requested-but-unmapped codepoint keeps its codepoint" {
    const file = try buildFixture(testing.allocator, .{});
    defer testing.allocator.free(file);
    const set = [_]u21{ 'A', 'Z' };
    var ttf = try TtfFont.init(testing.allocator, file, 20.0, &set);
    defer ttf.deinit();
    const z = ttf.lookup('Z').?;
    try testing.expectEqual(@as(u16, 0), z.gid);
    try testing.expectEqual(@as(u21, 'Z'), z.codepoint);
}

test "lookup over a fmt12 mixed-unicode set resolves BMP and supplementary" {
    const file = try buildFixture(testing.allocator, .{ .cmap = .fmt12 });
    defer testing.allocator.free(file);
    const set = [_]u21{ 0x4E2D, 0x10437 };
    var ttf = try TtfFont.init(testing.allocator, file, 20.0, &set);
    defer ttf.deinit();
    try testing.expectEqual(@as(u16, 1), ttf.lookup(0x4E2D).?.gid);
    try testing.expectEqual(@as(u16, 3), ttf.lookup(0x10437).?.gid);
    try testing.expectEqual(@as(u16, 0), ttf.lookup('A').?.gid);
}

test "kernPx matches kernUnits on known, reversed and missing pairs" {
    const file = try buildFixture(testing.allocator, .{});
    defer testing.allocator.free(file);
    var ttf = try TtfFont.init(testing.allocator, file, 20.0, &ascii_set);
    defer ttf.deinit();
    const scale = ttf.scale;
    const ga = ttf.lookup('A').?;
    const gb = ttf.lookup('B').?;
    const gc = ttf.lookup('C').?;
    const gz = ttf.lookup('Z').?; // .notdef fallback, gid 0
    try testing.expectApproxEqAbs(@as(f32, -80) * scale, ttf.kernPx(ga, gb), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), ttf.kernPx(gb, ga), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), ttf.kernPx(ga, gc), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), ttf.kernPx(gz, ga), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), ttf.kernPx(ga, gz), 1e-6);
}

const KernPairSpec = struct { left: u16, right: u16, value: i16 };

fn kernFmt0Subtable(
    allocator: std.mem.Allocator,
    format: u8,
    pairs: []const KernPairSpec,
    declared_pairs: ?u16,
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const w = &out.writer;
    try w.writeInt(u16, 0, .big); // subtable version
    const len: u16 = @intCast(14 + pairs.len * 6);
    try w.writeInt(u16, len, .big);
    try w.writeByte(format);
    try w.writeByte(0); // coverage
    try w.writeInt(u16, declared_pairs orelse @intCast(pairs.len), .big);
    try w.writeInt(u16, 2, .big); // searchRange (unused by the parser)
    try w.writeInt(u16, 0, .big); // entrySelector
    try w.writeInt(u16, 2, .big); // rangeShift
    for (pairs) |p| {
        try w.writeInt(u16, p.left, .big);
        try w.writeInt(u16, p.right, .big);
        try w.writeInt(i16, p.value, .big);
    }
    return out.toOwnedSlice();
}

fn kernTable(allocator: std.mem.Allocator, subtables: []const []const u8) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const w = &out.writer;
    try w.writeInt(u16, 0, .big); // version
    try w.writeInt(u16, @intCast(subtables.len), .big);
    for (subtables) |s| try w.writeAll(s);
    return out.toOwnedSlice();
}

fn kernFontWith(kern: ?[]const u8) Font {
    var f: Font = undefined;
    f.kern = kern;
    return f;
}

test "kernUnits finds unsorted pairs and lets the first match win" {
    // Unsorted pairs are all found regardless of table order.
    const s0 = try kernFmt0Subtable(testing.allocator, 0, &.{
        .{ .left = 3, .right = 4, .value = 10 },
        .{ .left = 1, .right = 2, .value = -80 },
        .{ .left = 2, .right = 1, .value = 5 },
    }, null);
    defer testing.allocator.free(s0);
    const t0 = try kernTable(testing.allocator, &.{s0});
    defer testing.allocator.free(t0);
    try testing.expectEqual(@as(i16, 10), try kernFontWith(t0).kernUnits(3, 4));
    try testing.expectEqual(@as(i16, -80), try kernFontWith(t0).kernUnits(1, 2));
    try testing.expectEqual(@as(i16, 5), try kernFontWith(t0).kernUnits(2, 1));
    try testing.expectEqual(@as(i16, 0), try kernFontWith(t0).kernUnits(9, 9));

    // Duplicate pairs within one subtable: the first entry wins.
    const s1 = try kernFmt0Subtable(testing.allocator, 0, &.{
        .{ .left = 1, .right = 2, .value = -80 },
        .{ .left = 1, .right = 2, .value = 99 },
    }, null);
    defer testing.allocator.free(s1);
    const t1 = try kernTable(testing.allocator, &.{s1});
    defer testing.allocator.free(t1);
    try testing.expectEqual(@as(i16, -80), try kernFontWith(t1).kernUnits(1, 2));

    // Across subtables the earlier subtable wins; later pairs still apply.
    const s2a = try kernFmt0Subtable(testing.allocator, 0, &.{
        .{ .left = 1, .right = 2, .value = -80 },
    }, null);
    defer testing.allocator.free(s2a);
    const s2b = try kernFmt0Subtable(testing.allocator, 0, &.{
        .{ .left = 1, .right = 2, .value = 7 },
        .{ .left = 5, .right = 6, .value = 3 },
    }, null);
    defer testing.allocator.free(s2b);
    const t2 = try kernTable(testing.allocator, &.{ s2a, s2b });
    defer testing.allocator.free(t2);
    try testing.expectEqual(@as(i16, -80), try kernFontWith(t2).kernUnits(1, 2));
    try testing.expectEqual(@as(i16, 3), try kernFontWith(t2).kernUnits(5, 6));

    // Non-zero-format subtables are skipped by length, never misread.
    const s3a = try kernFmt0Subtable(testing.allocator, 1, &.{
        .{ .left = 1, .right = 2, .value = 42 },
    }, null);
    defer testing.allocator.free(s3a);
    const s3b = try kernFmt0Subtable(testing.allocator, 0, &.{
        .{ .left = 1, .right = 2, .value = -80 },
    }, null);
    defer testing.allocator.free(s3b);
    const t3 = try kernTable(testing.allocator, &.{ s3a, s3b });
    defer testing.allocator.free(t3);
    try testing.expectEqual(@as(i16, -80), try kernFontWith(t3).kernUnits(1, 2));

    // A truncated pair array is BadTable, and a missing table is 0.
    const s4 = try kernFmt0Subtable(testing.allocator, 0, &.{
        .{ .left = 1, .right = 2, .value = -80 },
    }, 2);
    defer testing.allocator.free(s4);
    const t4 = try kernTable(testing.allocator, &.{s4});
    defer testing.allocator.free(t4);
    try testing.expectError(error.BadTable, kernFontWith(t4).kernUnits(1, 2));
    try testing.expectEqual(@as(i16, 0), try kernFontWith(null).kernUnits(1, 2));
}

test "TtfFont init propagates allocator OOM" {
    const file = try buildFixture(testing.allocator, .{});
    defer testing.allocator.free(file);
    var fa = std.testing.FailingAllocator.init(testing.allocator, .{ .fail_index = 0 });
    try testing.expectError(error.OutOfMemory, TtfFont.init(fa.allocator(), file, 20.0, &ascii_set));
}

fn kernIndexValue(index: []const font_mod.KernEntry, left: u16, right: u16) ?i16 {
    const key = (@as(u32, left) << 16) | right;
    const S = struct {
        fn order(key_: u32, e: font_mod.KernEntry) std.math.Order {
            return std.math.order(key_, e.key);
        }
    };
    if (std.sort.binarySearch(font_mod.KernEntry, index, key, S.order)) |i| return index[i].value;
    return null;
}

test "kern index agrees with kernUnits on synthetic tables" {
    const queries = [_][2]u16{ .{ 1, 2 }, .{ 2, 1 }, .{ 3, 4 }, .{ 5, 6 }, .{ 9, 9 } };
    // Same shapes as the kernUnits first-match test: unsorted, duplicate,
    // multi-subtable and format-skipped tables must read back identically.
    const s_unsorted = try kernFmt0Subtable(testing.allocator, 0, &.{
        .{ .left = 3, .right = 4, .value = 10 },
        .{ .left = 1, .right = 2, .value = -80 },
        .{ .left = 2, .right = 1, .value = 5 },
    }, null);
    defer testing.allocator.free(s_unsorted);
    const t_unsorted = try kernTable(testing.allocator, &.{s_unsorted});
    defer testing.allocator.free(t_unsorted);

    const s_dup = try kernFmt0Subtable(testing.allocator, 0, &.{
        .{ .left = 1, .right = 2, .value = -80 },
        .{ .left = 1, .right = 2, .value = 99 },
    }, null);
    defer testing.allocator.free(s_dup);
    const t_dup = try kernTable(testing.allocator, &.{s_dup});
    defer testing.allocator.free(t_dup);

    const s_m0 = try kernFmt0Subtable(testing.allocator, 0, &.{
        .{ .left = 1, .right = 2, .value = -80 },
    }, null);
    defer testing.allocator.free(s_m0);
    const s_m1 = try kernFmt0Subtable(testing.allocator, 0, &.{
        .{ .left = 1, .right = 2, .value = 7 },
        .{ .left = 5, .right = 6, .value = 3 },
    }, null);
    defer testing.allocator.free(s_m1);
    const t_multi = try kernTable(testing.allocator, &.{ s_m0, s_m1 });
    defer testing.allocator.free(t_multi);

    const s_skip = try kernFmt0Subtable(testing.allocator, 1, &.{
        .{ .left = 1, .right = 2, .value = 42 },
    }, null);
    defer testing.allocator.free(s_skip);
    const s_after = try kernFmt0Subtable(testing.allocator, 0, &.{
        .{ .left = 1, .right = 2, .value = -80 },
    }, null);
    defer testing.allocator.free(s_after);
    const t_skip = try kernTable(testing.allocator, &.{ s_skip, s_after });
    defer testing.allocator.free(t_skip);

    for ([_][]const u8{ t_unsorted, t_dup, t_multi, t_skip }) |blob| {
        const index = try font_mod.buildKernIndex(testing.allocator, blob);
        defer testing.allocator.free(index);
        // Sorted, unique keys.
        for (index, 0..) |e, k| {
            if (k > 0) try testing.expect(index[k - 1].key < e.key);
        }
        for (queries) |q| {
            const want = try kernFontWith(blob).kernUnits(q[0], q[1]);
            const got = kernIndexValue(index, q[0], q[1]) orelse 0;
            try testing.expectEqual(want, got);
        }
    }

    // Truncated tables fail identically; a missing table indexes empty.
    const s_trunc = try kernFmt0Subtable(testing.allocator, 0, &.{
        .{ .left = 1, .right = 2, .value = -80 },
    }, 2);
    defer testing.allocator.free(s_trunc);
    const t_trunc = try kernTable(testing.allocator, &.{s_trunc});
    defer testing.allocator.free(t_trunc);
    try testing.expectError(error.BadTable, font_mod.buildKernIndex(testing.allocator, t_trunc));
    // Short header is BadTable in both paths.
    try testing.expectError(error.BadTable, font_mod.buildKernIndex(testing.allocator, &.{ 0, 0 }));
    const empty_index = try font_mod.buildKernIndex(testing.allocator, null);
    try testing.expectEqual(@as(usize, 0), empty_index.len);
}

test "lookup is order-independent over shuffled charsets" {
    const file = try buildFixture(testing.allocator, .{});
    defer testing.allocator.free(file);
    const font = try Font.parse(file);
    const shuffled = [_]u21{ 'C', 'A', 'B' };
    var ttf = try TtfFont.init(testing.allocator, file, 20.0, &shuffled);
    defer ttf.deinit();
    // Baked set is sorted ascending by codepoint.
    for (ttf.glyphs, 0..) |g, k| {
        if (k > 0) try testing.expect(ttf.glyphs[k - 1].codepoint < g.codepoint);
    }
    for (shuffled) |cp| {
        const g = ttf.lookup(cp).?;
        try testing.expectEqual(cp, g.codepoint);
        try testing.expectEqual(try font.glyphIndex(cp), g.gid);
    }
}

test "kern index keeps first-match over many duplicate keys and subtables" {
    // 3 subtables x 24 pairs over 4 shared keys (72 entries, far above
    // any small-sort fast path). Every occurrence carries a distinct
    // value encoding its global file order, so any stability slip in the
    // bake-time sort surfaces as a wrong first-match value.
    const shared = [_][2]u16{ .{ 1, 2 }, .{ 3, 4 }, .{ 5, 6 }, .{ 7, 8 } };
    var subs: [3][]u8 = undefined;
    var s: usize = 0;
    while (s < 3) : (s += 1) {
        var pairs: [24]KernPairSpec = undefined;
        var j: usize = 0;
        while (j < 24) : (j += 1) {
            const k = shared[(j + s) % shared.len];
            pairs[j] = .{
                .left = k[0],
                .right = k[1],
                .value = @intCast(@as(i32, @intCast(s * 100 + j)) - 200),
            };
        }
        subs[s] = try kernFmt0Subtable(testing.allocator, 0, &pairs, null);
    }
    defer for (subs) |sub| testing.allocator.free(sub);
    const blob = try kernTable(testing.allocator, &subs);
    defer testing.allocator.free(blob);

    // Hardcoded anchors: key (2s+1, 2s+2) first occurs at subtable 0,
    // position s, i.e. value s - 200.
    const anchors = [_]i16{ -200, -199, -198, -197 };
    for (shared, 0..) |k, i| {
        try testing.expectEqual(anchors[i], try kernFontWith(blob).kernUnits(k[0], k[1]));
    }

    const index = try font_mod.buildKernIndex(testing.allocator, blob);
    defer testing.allocator.free(index);
    try testing.expectEqual(shared.len, index.len);
    for (shared, 0..) |k, i| {
        try testing.expectEqual(anchors[i], kernIndexValue(index, k[0], k[1]).?);
    }
    // Every indexed lookup (hits and misses) agrees with legacy kernUnits.
    for (shared) |k| {
        const want = try kernFontWith(blob).kernUnits(k[0], k[1]);
        try testing.expectEqual(want, kernIndexValue(index, k[0], k[1]).?);
    }
    for ([_][2]u16{ .{ 9, 9 }, .{ 1, 3 }, .{ 8, 7 } }) |k| {
        try testing.expectEqual(@as(i16, 0), try kernFontWith(blob).kernUnits(k[0], k[1]));
        try testing.expect(kernIndexValue(index, k[0], k[1]) == null);
    }
}
