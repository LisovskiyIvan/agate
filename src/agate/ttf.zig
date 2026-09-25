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
