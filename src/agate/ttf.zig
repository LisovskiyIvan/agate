// ---------------------------------------------------------------------------
// TrueType (glyf-based) font parser, rasterizer and CPU glyph atlas.
//
// Parses a subset of the TrueType container sufficient for crisp UI text:
// offset table + required tables (head, maxp, cmap, loca, glyf, hhea,
// hmtx) with optional kern (format 0) and OS/2 (parsed for nothing yet —
// reserved for future use). Simple (quadratic) and composite glyphs are
// extracted, flattened adaptively, and rasterized with a supersampled
// scanline fill (non-zero winding) into 8-bit alpha; glyphs are packed
// with a shelf packer into RGBA8 atlas pages (white + alpha) at a caller
// requested pixel size. `TtfFont` is the UI-facing handle: per-codepoint
// metrics (bearing, advance), kerning, and UVs feed `ui/text.zig`, which
// draws the SAME UIVertex quads as the bitmap/SDF path (atlas swap +
// UV/advance computation, mode 3 coverage sampling).
//
// Supported:
//   - sfnt versions 0x00010000, 'true', 'typ1' with glyf outlines.
//   - head unitsPerEm + both loca formats; maxp short (0x00005000) and
//     full (1.0) layouts; hhea metrics + hmtx advances.
//   - cmap format 4 (BMP, with idDelta/idRangeOffset) and format 12
//     (full Unicode incl. supplementary planes). When both are present,
//     format 12 is consulted first, then format 4.
//   - kern format 0 pairs (other kern formats are skipped by length).
//   - Simple glyphs: on/off-curve points, implied on-curve midpoints,
//     repeat flags, short/long delta encoding.
//   - Composite glyphs: 1- and 2-byte args, XY-value and anchor (anchor
//     matching is NOT performed — anchor args are treated as a zero
//     offset, documented below) positioning, uniform / x-and-y / 2x2
//     transforms, nested components up to 8 deep.
//
// Explicitly NOT supported (explicit error, never a silent fallback):
//   - OTF/CFF outlines: sfnt 'OTTO' or a present CFF/CFF2 table
//     (error.UnsupportedCff).
//   - Variable fonts: a present fvar table (error.UnsupportedVariableFont).
//     The static default outlines could be read, but interpolating
//     instances is out of scope, so the file is rejected outright.
//   - cmap files with neither format 4 nor 12 (error.UnsupportedCmap).
//   - Truncated reads (error.Truncated) vs structurally inconsistent
//     tables (error.BadTable/BadHead/BadCmap/BadLoca/BadGlyf).
//   - Composite cycles (error.CyclicComposite) and nesting deeper than 8
//     (error.CompositeTooDeep) — recursion is bounded, never hangs.
//   - Missing glyf data for a mapped codepoint: the glyph id resolves to
//     .notdef (gid 0) instead — an empty .notdef renders nothing but keeps
//     its advance, so text layout never breaks. A cmap id that exceeds
//     numGlyphs likewise resolves to 0.
//
// Deliberately out of scope (documented honestly, not attempted):
//   - Hinting (glyf instructions are skipped), ligatures, complex text
//     shaping, RTL/bidi reordering, subpixel positioning (advances snap
//     to whole baked pixels, scaled at draw), CFF/OTF, color fonts
//     (CBDT/CBLC/COLR/sbix/SVG), vertical metrics, anchor-based composite
//     positioning (treated as zero offset), checksums (table checksums
//     are NOT verified).
//
// Intended use: crisp UI text at any DPI for simple scripts where the
// font provides codepoint-mapped glyphs (Latin/Cyrillic/CJK codepoints
// via cmap 4/12). No external dependencies, no font files bundled: tests
// synthesize minimal TTF bytes programmatically (see `Fixture`).
//
// Memory: `Font` borrows the caller's `data` slice (no copy — the caller
// must keep it alive longer than the font). `TtfFont` owns its atlas
// pixels and glyph list; `deinit` frees them. Outline extraction
// allocates transiently per glyph (freed before return except on the
// rasterize path, which frees internally).
// ---------------------------------------------------------------------------

//
// Facade for the TTF modules. `ttf.zig` was split into focused leaves
// under `ttf/` following the wave-33 repo pattern (same-named re-exports;
// Zig 0.16 has no usingnamespace; see `mesh/builders.zig`, `audio.zig`):
//
// - `ttf/types.zig` — shared vocabulary: `TtfError`, bounds-checked
//   big-endian readers, sfnt table tags, parse limits, atlas sizing.
//   Imports `std` only (anti-cycle root).
// - `ttf/font.zig` — `sniff` + the borrowing `Font` handle (parse, glyph
//   ranges, cmap 4/12, advances, kern) with its private validators.
//   Imports `types` only.
// - `ttf/outline.zig` — `OutlinePoint`/`Contour`, `extractOutline`
//   (simple + composite) and `freeContours`. Imports `types` + `font`.
// - `ttf/raster.zig` — `Segment`, `flattenContours`, `rasterizeSegments`.
//   Imports `types` + `outline`.
// - `ttf/atlas.zig` — `GlyphInfo` + the UI-facing `TtfFont` (bake, shelf
//   packer, lookup, kern, UVs). Imports `types` + `font` + `outline` +
//   `raster`.
// - `ttf/fixture.zig` — `FixtureOptions` + `buildFixture` (in-memory test
//   fonts, no bundled files). Imports `types` only.
//
// Everything that was public before the split is re-exported here
// unchanged; consumers (`root.zig`, `ui/text.zig`, `ui/font.zig`,
// `scene/ui_frame.zig`) see the same API as when everything lived in
// this file, with no call-site changes.
//
// Documented anti-cycle rule: leaves must never import this facade —
// importing it back would make the re-exports depend on their own
// consumers. `Font`/`TtfFont` move whole with their methods (no `anytype`
// needed: unlike `audio/playback.zig` no owner struct stays behind, so no
// method had to become a free function); leaf-to-leaf calls use direct
// sibling imports (same discipline as `audio/*`). `freeContours` is `pub`
// in `outline` for the `atlas` sibling and the tests below but is
// deliberately NOT re-exported here, so the public surface is identical
// to the pre-split file.
//
// Honest structural notes:
// - Leaves alias shared constants/types (`const tag_head: u32 =
//   types.tag_head;`, `const Font = font_mod.Font;`) so moved function
//   bodies stay byte-identical; only the four cross-leaf calls in
//   `atlas.zig` gained a module prefix (`outline.`/`raster.`).
// - The inline tests stay in this facade (they exercise the public API);
//   `freeContours` resolves through a private alias below.
// ---------------------------------------------------------------------------

const std = @import("std");

const types = @import("ttf/types.zig");
const font_mod = @import("ttf/font.zig");
const outline = @import("ttf/outline.zig");
const raster = @import("ttf/raster.zig");
const atlas = @import("ttf/atlas.zig");
const fixture = @import("ttf/fixture.zig");

// Shared vocabulary (lives in ttf/types.zig).
pub const TtfError = types.TtfError;
pub const atlas_size: u32 = types.atlas_size;

// sfnt container (lives in ttf/font.zig).
pub const sniff = font_mod.sniff;
pub const Font = font_mod.Font;

// Outlines (live in ttf/outline.zig).
pub const OutlinePoint = outline.OutlinePoint;
pub const Contour = outline.Contour;
pub const extractOutline = outline.extractOutline;

// Private for the tests below (NOT part of the public surface).
const freeContours = outline.freeContours;

// Flattening + rasterization (live in ttf/raster.zig).
pub const Segment = raster.Segment;
pub const flattenContours = raster.flattenContours;
pub const rasterizeSegments = raster.rasterizeSegments;

// CPU atlas + UI handle (live in ttf/atlas.zig).
pub const GlyphInfo = atlas.GlyphInfo;
pub const TtfFont = atlas.TtfFont;

// Fixture writer (lives in ttf/fixture.zig).
pub const FixtureCmap = fixture.FixtureCmap;
pub const FixtureComposite = fixture.FixtureComposite;
pub const FixtureLoca = fixture.FixtureLoca;
pub const FixtureOptions = fixture.FixtureOptions;
pub const buildFixture = fixture.buildFixture;

// ---------------------------------------------------------------------------
// Tests — fixtures are synthesized in-memory by buildFixture (no external
// font files): header + tables with a rectangle outline, a quadratic
// triangle, a composite, cmap 4/12, hmtx advances and a kern pair.
// ---------------------------------------------------------------------------

const testing = std.testing;

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
