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
