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

const std = @import("std");

pub const TtfError = error{
    NotTtf,
    Truncated,
    MissingTable,
    BadTable,
    BadHead,
    BadCmap,
    BadLoca,
    BadGlyf,
    UnsupportedCff,
    UnsupportedVariableFont,
    UnsupportedCmap,
    CompositeTooDeep,
    CyclicComposite,
    AtlasFull,
    GlyphTooLarge,
    InvalidPixelSize,
    OutOfMemory,
};

/// True when `bytes` looks like a glyf-based sfnt (version only — full
/// validation happens in `Font.parse`). 'OTTO' (CFF) is deliberately NOT
/// sniffed: it parses to error.UnsupportedCff, never routes here.
pub fn sniff(bytes: []const u8) bool {
    if (bytes.len < 4) return false;
    const v = std.mem.readInt(u32, bytes[0..4], .big);
    return v == 0x00010000 or v == 0x74727565 or v == 0x74797031; // 1.0, 'true', 'typ1'
}

/// Bounds-checked big-endian reader. Every advance goes through `take`,
/// so short files surface as Truncated instead of panicking.
const Cursor = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn take(self: *Cursor, n: usize) TtfError![]const u8 {
        if (self.pos > self.bytes.len or n > self.bytes.len - self.pos) return error.Truncated;
        const s = self.bytes[self.pos..][0..n];
        self.pos += n;
        return s;
    }

    fn readU16(self: *Cursor) TtfError!u16 {
        return std.mem.readInt(u16, (try self.take(2))[0..2], .big);
    }

    fn readU32(self: *Cursor) TtfError!u32 {
        return std.mem.readInt(u32, (try self.take(4))[0..4], .big);
    }
};

fn readU16At(bytes: []const u8, off: usize) TtfError!u16 {
    if (off > bytes.len or 2 > bytes.len - off) return error.Truncated;
    return std.mem.readInt(u16, bytes[off..][0..2], .big);
}

fn readI16At(bytes: []const u8, off: usize) TtfError!i16 {
    return @bitCast(try readU16At(bytes, off));
}

fn readU32At(bytes: []const u8, off: usize) TtfError!u32 {
    if (off > bytes.len or 4 > bytes.len - off) return error.Truncated;
    return std.mem.readInt(u32, bytes[off..][0..4], .big);
}

const tag_head: u32 = 0x68656164; // 'head'
const tag_maxp: u32 = 0x6D617870; // 'maxp'
const tag_cmap: u32 = 0x636D6170; // 'cmap'
const tag_loca: u32 = 0x6C6F6361; // 'loca'
const tag_glyf: u32 = 0x676C7966; // 'glyf'
const tag_hhea: u32 = 0x68686561; // 'hhea'
const tag_hmtx: u32 = 0x686D7478; // 'hmtx'
const tag_kern: u32 = 0x6B65726E; // 'kern'
const tag_cff: u32 = 0x43464620; // 'CFF '
const tag_cff2: u32 = 0x43464632; // 'CFF2'
const tag_fvar: u32 = 0x66766172; // 'fvar'

const max_tables: usize = 64;
const max_cmap_records: usize = 64;
const max_contours: usize = 256;
const max_points: usize = 8192;
const max_components: usize = 64;
const max_composite_depth: u8 = 8;
const max_glyph_bitmap: u32 = 512;

/// Parsed font: borrows `data` (no copy). Obtain via `Font.parse`.
pub const Font = struct {
    data: []const u8,
    units_per_em: u16,
    loca_short: bool,
    num_glyphs: u16,
    ascender: i16,
    descender: i16,
    line_gap: i16,
    num_h_metrics: u16,
    cmap: []const u8,
    loca: []const u8,
    glyf: []const u8,
    hmtx: []const u8,
    kern: ?[]const u8,
    cmap_fmt4: ?[]const u8 = null,
    cmap_fmt12: ?[]const u8 = null,

    fn tableSlice(data: []const u8, off: u32, len: u32) TtfError![]const u8 {
        const o: usize = off;
        const l: usize = len;
        if (o > data.len or l > data.len - o) return error.Truncated;
        return data[o..][0..l];
    }

    /// Full parse of the sfnt container. Rejects CFF/OTF, variable fonts,
    /// missing tables and malformed headers with explicit errors.
    pub fn parse(data: []const u8) TtfError!Font {
        var cur = Cursor{ .bytes = data };
        const sfnt = try cur.readU32();
        if (sfnt == 0x4F54544F) return error.UnsupportedCff; // 'OTTO'
        if (sfnt != 0x00010000 and sfnt != 0x74727565 and sfnt != 0x74797031) return error.NotTtf;
        const num_tables = try cur.readU16();
        _ = try cur.readU16(); // searchRange
        _ = try cur.readU16(); // entrySelector
        _ = try cur.readU16(); // rangeShift
        if (num_tables == 0 or num_tables > max_tables) return error.BadTable;

        var off_head: ?[2]u32 = null;
        var off_maxp: ?[2]u32 = null;
        var off_cmap: ?[2]u32 = null;
        var off_loca: ?[2]u32 = null;
        var off_glyf: ?[2]u32 = null;
        var off_hhea: ?[2]u32 = null;
        var off_hmtx: ?[2]u32 = null;
        var off_kern: ?[2]u32 = null;
        var i: usize = 0;
        while (i < num_tables) : (i += 1) {
            const tag = try cur.readU32();
            _ = try cur.readU32(); // checksum (NOT verified — documented)
            const off = try cur.readU32();
            const len = try cur.readU32();
            const slot: ?*?[2]u32 = switch (tag) {
                tag_head => &off_head,
                tag_maxp => &off_maxp,
                tag_cmap => &off_cmap,
                tag_loca => &off_loca,
                tag_glyf => &off_glyf,
                tag_hhea => &off_hhea,
                tag_hmtx => &off_hmtx,
                tag_kern => &off_kern,
                tag_cff, tag_cff2 => return error.UnsupportedCff,
                tag_fvar => return error.UnsupportedVariableFont,
                else => null,
            };
            // Duplicate table tags are malformed.
            if (slot) |s| {
                if (s.* != null) return error.BadTable;
                s.* = .{ off, len };
            }
        }

        const head = if (off_head) |o| try tableSlice(data, o[0], o[1]) else return error.MissingTable;
        const maxp = if (off_maxp) |o| try tableSlice(data, o[0], o[1]) else return error.MissingTable;
        const cmap = if (off_cmap) |o| try tableSlice(data, o[0], o[1]) else return error.MissingTable;
        const loca = if (off_loca) |o| try tableSlice(data, o[0], o[1]) else return error.MissingTable;
        const glyf = if (off_glyf) |o| try tableSlice(data, o[0], o[1]) else return error.MissingTable;
        const hhea = if (off_hhea) |o| try tableSlice(data, o[0], o[1]) else return error.MissingTable;
        const hmtx = if (off_hmtx) |o| try tableSlice(data, o[0], o[1]) else return error.MissingTable;
        const kern = if (off_kern) |o| try tableSlice(data, o[0], o[1]) else null;

        // head (54 bytes).
        if (head.len < 54) return error.BadHead;
        if (std.mem.readInt(u32, head[12..][0..4], .big) != 0x5F0F3CF5) return error.BadHead;
        const units = std.mem.readInt(u16, head[18..][0..2], .big);
        if (units < 16 or units > 16384) return error.BadHead;
        const loca_fmt = std.mem.readInt(i16, head[50..][0..2], .big);
        if (loca_fmt != 0 and loca_fmt != 1) return error.BadHead;

        // maxp: short (6) or full (32) layout; numGlyphs at offset 4.
        if (maxp.len != 6 and maxp.len < 32) return error.BadTable;
        const maxp_ver = std.mem.readInt(u32, maxp[0..][0..4], .big);
        if (maxp_ver != 0x00005000 and maxp_ver != 0x00010000) return error.BadTable;
        const num_glyphs = std.mem.readInt(u16, maxp[4..][0..2], .big);
        if (num_glyphs == 0) return error.BadTable;

        // hhea (36 bytes).
        if (hhea.len < 36) return error.BadTable;
        const num_h_metrics = std.mem.readInt(u16, hhea[34..][0..2], .big);
        if (num_h_metrics == 0 or num_h_metrics > num_glyphs) return error.BadTable;

        // hmtx must cover the long metrics plus the trailing left bearings.
        const hmtx_need: usize =
            @as(usize, num_h_metrics) * 4 + @as(usize, num_glyphs - num_h_metrics) * 2;
        if (hmtx.len < hmtx_need) return error.BadTable;

        // loca must hold numGlyphs+1 offsets in the selected format.
        const loca_entry: usize = if (loca_fmt == 0) 2 else 4;
        if (loca.len < (@as(usize, num_glyphs) + 1) * loca_entry) return error.BadLoca;

        var font = Font{
            .data = data,
            .units_per_em = units,
            .loca_short = loca_fmt == 0,
            .num_glyphs = num_glyphs,
            .ascender = std.mem.readInt(i16, hhea[4..][0..2], .big),
            .descender = std.mem.readInt(i16, hhea[6..][0..2], .big),
            .line_gap = std.mem.readInt(i16, hhea[8..][0..2], .big),
            .num_h_metrics = num_h_metrics,
            .cmap = cmap,
            .loca = loca,
            .glyf = glyf,
            .hmtx = hmtx,
            .kern = kern,
        };
        const cmaps = try parseCmap(cmap);
        font.cmap_fmt4 = cmaps.fmt4;
        font.cmap_fmt12 = cmaps.fmt12;
        if (kern) |k| try validateKern(k);
        return font;
    }

    /// Byte range of glyph `gid` inside the glyf table. Out-of-range ids
    /// resolve to .notdef (gid 0); an empty range is a valid empty glyph.
    pub fn glyphRange(self: *const Font, gid: u16) TtfError![]const u8 {
        var g = gid;
        if (g >= self.num_glyphs) g = 0;
        const entry: usize = if (self.loca_short) 2 else 4;
        const o0: usize = if (self.loca_short)
            @as(usize, try readU16At(self.loca, @as(usize, g) * entry)) * 2
        else
            try readU32At(self.loca, @as(usize, g) * entry);
        const o1: usize = if (self.loca_short)
            @as(usize, try readU16At(self.loca, (@as(usize, g) + 1) * entry)) * 2
        else
            try readU32At(self.loca, (@as(usize, g) + 1) * entry);
        if (o0 > self.glyf.len or o1 > self.glyf.len or o1 < o0) return error.BadLoca;
        return self.glyf[o0..o1];
    }

    /// cmap lookup: format 12 first, then format 4. Returns 0 (.notdef)
    /// when unmapped; ids past numGlyphs are clamped by `glyphRange`.
    pub fn glyphIndex(self: *const Font, codepoint: u21) TtfError!u16 {
        if (self.cmap_fmt12) |sub| {
            const g = try cmap12Lookup(sub, codepoint);
            if (g != 0) return g;
        }
        if (self.cmap_fmt4) |sub| {
            if (codepoint <= 0xFFFF) {
                const g = try cmap4Lookup(sub, @intCast(codepoint));
                if (g != 0) return g;
            }
        }
        return 0;
    }

    /// Horizontal advance in font units (hmtx, trailing LSBs reuse the
    /// last advance). Out-of-range ids resolve to .notdef.
    pub fn advanceUnits(self: *const Font, gid: u16) u16 {
        const g = if (gid >= self.num_glyphs) 0 else gid;
        const idx = @min(g, self.num_h_metrics - 1);
        return std.mem.readInt(u16, self.hmtx[@as(usize, idx) * 4 ..][0..2], .big);
    }

    /// kern format 0 adjustment in font units (0 when no kern table or no
    /// pair). Other kern formats are skipped by length, never misread.
    pub fn kernUnits(self: *const Font, left_gid: u16, right_gid: u16) TtfError!i16 {
        const k = self.kern orelse return 0;
        if (k.len < 4) return error.BadTable;
        const n_tables = std.mem.readInt(u16, k[2..][0..2], .big);
        var pos: usize = 4;
        var t: usize = 0;
        while (t < n_tables) : (t += 1) {
            if (pos + 6 > k.len) return error.BadTable;
            const version = std.mem.readInt(u16, k[pos..][0..2], .big);
            const length = std.mem.readInt(u16, k[pos + 2 ..][0..2], .big);
            const format = k[pos + 4];
            if (length < 6 or pos + length > k.len) return error.BadTable;
            if (version == 0 and format == 0) {
                const sub = k[pos..][0..length];
                if (sub.len < 14) return error.BadTable;
                const n_pairs = std.mem.readInt(u16, sub[6..][0..2], .big);
                if (sub.len < 14 + @as(usize, n_pairs) * 6) return error.BadTable;
                var p: usize = 0;
                while (p < n_pairs) : (p += 1) {
                    const l = std.mem.readInt(u16, sub[14 + p * 6 ..][0..2], .big);
                    const r = std.mem.readInt(u16, sub[14 + p * 6 + 2 ..][0..2], .big);
                    if (l == left_gid and r == right_gid) {
                        return std.mem.readInt(i16, sub[14 + p * 6 + 4 ..][0..2], .big);
                    }
                }
            }
            pos += length;
        }
        return 0;
    }
};

const CmapPair = struct { fmt4: ?[]const u8, fmt12: ?[]const u8 };

/// Validates the kern table directory at parse time (lengths fit, format
/// 0 pair arrays fit) so later `kernUnits` lookups are bounds-safe.
fn validateKern(k: []const u8) TtfError!void {
    if (k.len < 4) return error.BadTable;
    const n_tables = std.mem.readInt(u16, k[2..][0..2], .big);
    if (n_tables > 32) return error.BadTable;
    var pos: usize = 4;
    var t: usize = 0;
    while (t < n_tables) : (t += 1) {
        if (pos + 6 > k.len) return error.BadTable;
        const version = std.mem.readInt(u16, k[pos..][0..2], .big);
        const length = std.mem.readInt(u16, k[pos + 2 ..][0..2], .big);
        const format = k[pos + 4];
        if (length < 6 or pos + length > k.len) return error.BadTable;
        if (version == 0 and format == 0) {
            if (length < 14) return error.BadTable;
            const n_pairs = std.mem.readInt(u16, k[pos + 6 ..][0..2], .big);
            if (n_pairs > 4096) return error.BadTable;
            if (length < 14 + @as(usize, n_pairs) * 6) return error.BadTable;
        }
        pos += length;
    }
}

fn parseCmap(cmap: []const u8) TtfError!CmapPair {
    if (cmap.len < 4) return error.BadCmap;
    const n = std.mem.readInt(u16, cmap[2..][0..2], .big);
    if (n > max_cmap_records) return error.BadCmap;
    if (cmap.len < 4 + @as(usize, n) * 8) return error.BadCmap;
    var f4: ?[]const u8 = null;
    var f12: ?[]const u8 = null;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const rec = cmap[4 + i * 8 ..][0..8];
        const off = std.mem.readInt(u32, rec[4..][0..4], .big);
        if (off >= cmap.len) return error.BadCmap;
        const fmt16 = try readU16At(cmap, off);
        if (fmt16 == 4 and f4 == null) {
            if (off + 16 > cmap.len) return error.BadCmap;
            const seg_count = @as(usize, try readU16At(cmap, off + 6)) / 2;
            if (seg_count == 0 or seg_count > 4096) return error.BadCmap;
            const need = 16 + seg_count * 8;
            if (off + need > cmap.len) return error.BadCmap;
            const sub = cmap[off..][0..need];
            // The spec requires the final segment to end at 0xFFFF.
            if (try readU16At(sub, 14 + (seg_count - 1) * 2) != 0xFFFF) return error.BadCmap;
            f4 = sub;
        } else if (fmt16 == 12 and f12 == null) {
            // Format 12 header is 16 bytes (format u16 + reserved u16 +
            // length u32 + language u32 + nGroups u32).
            if (off + 16 > cmap.len) return error.BadCmap;
            const length = try readU32At(cmap, off + 4);
            if (length < 16 or off + length > cmap.len) return error.BadCmap;
            const n_groups = try readU32At(cmap, off + 12);
            if (n_groups > 65536) return error.BadCmap;
            if (off + 16 + @as(usize, n_groups) * 12 > cmap.len) return error.BadCmap;
            f12 = cmap[off..][0..(off + length - off)];
        }
        // All other formats (0, 6, ...) are ignored: they only matter
        // when no 4/12 exists, which is UnsupportedCmap below.
    }
    if (f4 == null and f12 == null) return error.UnsupportedCmap;
    return .{ .fmt4 = f4, .fmt12 = f12 };
}

fn cmap4Lookup(sub: []const u8, cp: u16) TtfError!u16 {
    const seg_count = @as(usize, try readU16At(sub, 6)) / 2;
    const end_base: usize = 14;
    const start_base: usize = 16 + seg_count * 2;
    const delta_base: usize = 16 + seg_count * 4;
    const range_base: usize = 16 + seg_count * 6;
    var i: usize = 0;
    while (i < seg_count) : (i += 1) {
        const end = try readU16At(sub, end_base + i * 2);
        const start = try readU16At(sub, start_base + i * 2);
        if (start > end) return error.BadCmap;
        if (cp < start or cp > end) continue;
        const delta = try readU16At(sub, delta_base + i * 2);
        const ro = try readU16At(sub, range_base + i * 2);
        if (ro == 0) return cp +% delta;
        const addr = range_base + i * 2 + ro + @as(usize, cp - start) * 2;
        const g = try readU16At(sub, addr);
        if (g == 0) return 0;
        return g +% delta;
    }
    return 0;
}

fn cmap12Lookup(sub: []const u8, cp: u21) TtfError!u16 {
    const n_groups = try readU32At(sub, 12);
    var i: usize = 0;
    while (i < n_groups) : (i += 1) {
        const base = 16 + i * 12;
        const start = try readU32At(sub, base);
        const end = try readU32At(sub, base + 4);
        const first = try readU32At(sub, base + 8);
        if (start > end) return error.BadCmap;
        if (cp >= start and cp <= end) {
            const g = first + (cp - start);
            if (g > 0xFFFF) return 0;
            return @intCast(g);
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Outlines
// ---------------------------------------------------------------------------

pub const OutlinePoint = struct {
    x: f32, // font units (fractional after composite transforms)
    y: f32,
    on_curve: bool,
};

pub const Contour = struct {
    points: []OutlinePoint,
};

fn freeContours(allocator: std.mem.Allocator, contours: []Contour) void {
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

// ---------------------------------------------------------------------------
// Flattening + rasterization
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// CPU atlas + UI-facing font handle
// ---------------------------------------------------------------------------

pub const atlas_size: u32 = 512;
const atlas_pad: u32 = 1; // 1px gutter stops LINEAR bleed between cells

/// One baked glyph: atlas rect + font-pixel metrics (bake scale applied).
pub const GlyphInfo = struct {
    codepoint: u21,
    gid: u16,
    ax: u32, // atlas origin (content, without gutter)
    ay: u32,
    w: u32,
    h: u32,
    bearing_x: f32, // bitmap left relative to pen, bake px
    bearing_y: f32, // bitmap top BELOW... (see below), bake px
    advance: f32, // horizontal advance, bake px
};

/// UI-facing TrueType font: parsed `Font` (borrowing caller `data`) plus
/// an RGBA8 shelf atlas (white + alpha) baked at `pixel_size`. Obtain via
/// `init`; free with `deinit`. The atlas pixels (`atlas_pixels`,
/// `atlas_size` x `atlas_size`) upload 1:1 through Texture.initRaw.
pub const TtfFont = struct {
    allocator: std.mem.Allocator,
    font: Font,
    pixel_size: f32,
    scale: f32, // pixel_size / unitsPerEm
    ascent: f32, // bake px
    descent: f32, // bake px (negative or zero)
    line_gap: f32, // bake px
    line_height: f32, // bake px
    glyphs: []GlyphInfo,
    notdef: ?GlyphInfo,
    atlas_pixels: []u8, // RGBA8, atlas_size^2 * 4
    shelf_x: u32 = 0,
    shelf_y: u32 = 0,
    shelf_h: u32 = 0,

    /// Bakes `codepoints` (plus .notdef) at `pixel_size` px em. Duplicate
    /// codepoints mapping to one gid share the atlas cell. Errors:
    /// parse failures, error.AtlasFull (glyphs do not fit 512x512 —
    /// bake fewer/smaller), error.InvalidPixelSize for px outside 4..256.
    pub fn init(
        allocator: std.mem.Allocator,
        data: []const u8,
        pixel_size: f32,
        codepoints: []const u21,
    ) TtfError!TtfFont {
        if (!(pixel_size >= 4.0 and pixel_size <= 256.0)) return error.InvalidPixelSize;
        const font = try Font.parse(data);
        const scale = pixel_size / @as(f32, @floatFromInt(font.units_per_em));
        var self = TtfFont{
            .allocator = allocator,
            .font = font,
            .pixel_size = pixel_size,
            .scale = scale,
            .ascent = @as(f32, @floatFromInt(font.ascender)) * scale,
            .descent = @as(f32, @floatFromInt(font.descender)) * scale,
            .line_gap = @as(f32, @floatFromInt(font.line_gap)) * scale,
            .line_height = (@as(f32, @floatFromInt(font.ascender)) -
                @as(f32, @floatFromInt(font.descender)) +
                @as(f32, @floatFromInt(font.line_gap))) * scale,
            .glyphs = &.{},
            .notdef = null,
            .atlas_pixels = allocator.alloc(u8, @as(usize, atlas_size) * atlas_size * 4) catch return error.OutOfMemory,
        };
        errdefer self.deinit();
        @memset(self.atlas_pixels, 0);

        // Unique gid set: requested codepoints + gid 0 (.notdef).
        var ugids: [512]u16 = undefined;
        var n_ugids: usize = 0;
        ugids[0] = 0;
        n_ugids = 1;
        for (codepoints) |cp| {
            const g = try font.glyphIndex(cp);
            var found = false;
            for (ugids[0..n_ugids]) |u| {
                if (u == g) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                if (n_ugids >= ugids.len) return error.AtlasFull;
                ugids[n_ugids] = g;
                n_ugids += 1;
            }
        }

        // Bake each unique gid once; remember its cell + metrics.
        var baked = allocator.alloc(BakedGlyph, n_ugids) catch return error.OutOfMemory;
        defer allocator.free(baked);
        for (ugids[0..n_ugids], 0..) |gid, k| {
            baked[k] = try self.bakeGlyph(gid);
        }

        // Per-codepoint entries (shared cells for shared gids).
        var list: std.ArrayListUnmanaged(GlyphInfo) = .empty;
        errdefer list.deinit(allocator);
        for (codepoints) |cp| {
            const g = try font.glyphIndex(cp);
            for (baked) |b| {
                if (b.gid == g) {
                    try list.append(allocator, .{
                        .codepoint = cp,
                        .gid = g,
                        .ax = b.ax,
                        .ay = b.ay,
                        .w = b.w,
                        .h = b.h,
                        .bearing_x = b.bearing_x,
                        .bearing_y = b.bearing_y,
                        .advance = b.advance,
                    });
                    break;
                }
            }
        }
        self.glyphs = try list.toOwnedSlice(allocator);
        for (baked) |b| {
            if (b.gid == 0) {
                self.notdef = GlyphInfo{
                    .codepoint = 0xFFFD,
                    .gid = 0,
                    .ax = b.ax,
                    .ay = b.ay,
                    .w = b.w,
                    .h = b.h,
                    .bearing_x = b.bearing_x,
                    .bearing_y = b.bearing_y,
                    .advance = b.advance,
                };
                break;
            }
        }
        return self;
    }

    pub fn deinit(self: *TtfFont) void {
        if (self.atlas_pixels.len > 0) self.allocator.free(self.atlas_pixels);
        if (self.glyphs.len > 0) self.allocator.free(self.glyphs);
        self.* = undefined;
    }

    /// Codepoint lookup within the baked set; unmapped codepoints fall
    /// back to .notdef, unknown-to-the-set returns null (caller skips).
    pub fn lookup(self: *const TtfFont, codepoint: u21) ?GlyphInfo {
        for (self.glyphs) |g| {
            if (g.codepoint == codepoint) return g;
        }
        return self.notdef;
    }

    /// Kerning between two baked glyphs in bake px (0 without kern data).
    pub fn kernPx(self: *const TtfFont, left: GlyphInfo, right: GlyphInfo) f32 {
        const k = self.font.kernUnits(left.gid, right.gid) catch 0;
        return @as(f32, @floatFromInt(k)) * self.scale;
    }

    /// Normalized atlas UVs for a baked glyph (content rect, no gutter).
    pub fn glyphUv(self: *const TtfFont, g: GlyphInfo) [4]f32 {
        _ = self;
        const s: f32 = @floatFromInt(atlas_size);
        return .{
            @as(f32, @floatFromInt(g.ax)) / s,
            @as(f32, @floatFromInt(g.ay)) / s,
            @as(f32, @floatFromInt(g.ax + g.w)) / s,
            @as(f32, @floatFromInt(g.ay + g.h)) / s,
        };
    }

    const BakedGlyph = struct {
        gid: u16,
        ax: u32,
        ay: u32,
        w: u32,
        h: u32,
        bearing_x: f32,
        bearing_y: f32,
        advance: f32,
    };

    fn bakeGlyph(self: *TtfFont, gid: u16) TtfError!BakedGlyph {
        const advance = @as(f32, @floatFromInt(self.font.advanceUnits(gid))) * self.scale;
        const contours = try extractOutline(self.allocator, &self.font, gid);
        defer freeContours(self.allocator, contours);
        if (contours.len == 0) {
            // Empty glyph (space, empty .notdef): valid entry, no cell.
            return .{ .gid = gid, .ax = 0, .ay = 0, .w = 0, .h = 0, .bearing_x = 0, .bearing_y = 0, .advance = advance };
        }
        const segs = try flattenContours(self.allocator, contours, self.scale, 0, 0, 0.25);
        defer self.allocator.free(segs);
        if (segs.len == 0) {
            return .{ .gid = gid, .ax = 0, .ay = 0, .w = 0, .h = 0, .bearing_x = 0, .bearing_y = 0, .advance = advance };
        }
        var min_x: f32 = segs[0].x0;
        var max_x: f32 = segs[0].x0;
        var min_y: f32 = segs[0].y0;
        var max_y: f32 = segs[0].y0;
        for (segs) |s| {
            min_x = @min(min_x, @min(s.x0, s.x1));
            max_x = @max(max_x, @max(s.x0, s.x1));
            min_y = @min(min_y, @min(s.y0, s.y1));
            max_y = @max(max_y, @max(s.y0, s.y1));
        }
        // y-down device space with baseline at 0: bearing_y is the bitmap
        // top's distance BELOW the baseline's upward extent... concretely:
        // bitmap rows cover [floor(min_y), ceil(max_y)); drawing places
        // the bitmap top at baseline - bearing_y where
        // bearing_y = -floor(min_y) (positive when the glyph rises above
        // the baseline, as usual for Latin ascenders).
        const ox = @as(i32, @intFromFloat(@floor(min_x)));
        const oy = @as(i32, @intFromFloat(@floor(min_y)));
        const w: u32 = @intCast(@as(i32, @intFromFloat(@ceil(max_x))) - ox);
        const h: u32 = @intCast(@as(i32, @intFromFloat(@ceil(max_y))) - oy);
        if (w == 0 or h == 0) {
            return .{ .gid = gid, .ax = 0, .ay = 0, .w = 0, .h = 0, .bearing_x = 0, .bearing_y = 0, .advance = advance };
        }
        if (w > max_glyph_bitmap or h > max_glyph_bitmap) return error.GlyphTooLarge;
        const alpha = try rasterizeSegments(self.allocator, segs, ox, oy, w, h);
        defer self.allocator.free(alpha);
        const cell = try self.placeCell(w, h);
        // Blit white + alpha (mode 3 samples .a for coverage).
        var row: u32 = 0;
        while (row < h) : (row += 1) {
            var col: u32 = 0;
            while (col < w) : (col += 1) {
                const dst = (@as(usize, cell.y + row) * atlas_size + cell.x + col) * 4;
                self.atlas_pixels[dst + 0] = 255;
                self.atlas_pixels[dst + 1] = 255;
                self.atlas_pixels[dst + 2] = 255;
                self.atlas_pixels[dst + 3] = alpha[@as(usize, row) * w + col];
            }
        }
        return .{
            .gid = gid,
            .ax = cell.x,
            .ay = cell.y,
            .w = w,
            .h = h,
            .bearing_x = @as(f32, @floatFromInt(ox)),
            .bearing_y = -@as(f32, @floatFromInt(oy)),
            .advance = advance,
        };
    }

    /// Shelf packer: rows left to right, 1px gutters. error.AtlasFull
    /// when the cell fits neither the current row nor a new row.
    fn placeCell(self: *TtfFont, w: u32, h: u32) TtfError!struct { x: u32, y: u32 } {
        const cw = w + atlas_pad * 2;
        const ch = h + atlas_pad * 2;
        if (cw > atlas_size or ch > atlas_size) return error.AtlasFull;
        if (self.shelf_x + cw > atlas_size) {
            self.shelf_x = 0;
            self.shelf_y += self.shelf_h;
            self.shelf_h = 0;
        }
        if (self.shelf_y + ch > atlas_size) return error.AtlasFull;
        const x = self.shelf_x + atlas_pad;
        const y = self.shelf_y + atlas_pad;
        self.shelf_x += cw;
        self.shelf_h = @max(self.shelf_h, ch);
        return .{ .x = x, .y = y };
    }
};

// ---------------------------------------------------------------------------
// Fixture writer (tests only, programmatic — no font files bundled).
// Public so the UI text-path tests (ui/text.zig) reuse the same minimal
// fonts without duplicating the sfnt writer.
// ---------------------------------------------------------------------------

pub const FixtureCmap = enum { fmt4, fmt12, both, unsupported };
pub const FixtureComposite = enum { normal, self_cyclic, pair_cyclic, deep };
pub const FixtureLoca = enum { ok, out_of_range };

pub const FixtureOptions = struct {
    units_per_em: u16 = 1000,
    index_to_loc_format: u16 = 1,
    cmap: FixtureCmap = .fmt4,
    include_kern: bool = true,
    include_fvar: bool = false,
    include_cff_table: bool = false,
    sfnt_version: u32 = 0x00010000,
    composite: FixtureComposite = .normal,
    loca: FixtureLoca = .ok,
    truncate_bytes: usize = 0,
};

const FixturePoint = struct {
    x: i32,
    y: i32,
    on: bool,
};

fn fixtureWriteU16(w: *std.Io.Writer, v: u16) !void {
    try w.writeInt(u16, v, .big);
}

fn fixtureWriteU32(w: *std.Io.Writer, v: u32) !void {
    try w.writeInt(u32, v, .big);
}

fn fixtureWriteI16(w: *std.Io.Writer, v: i16) !void {
    try w.writeInt(i16, v, .big);
}

/// Encodes one simple glyph from per-contour point lists (delta-encoded
/// long coordinates except exact zeros; repeat-compressed flags).
fn fixtureSimpleGlyph(
    allocator: std.mem.Allocator,
    contours: []const []const FixturePoint,
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const w = &out.writer;
    var n_pts: usize = 0;
    var x0: i32 = std.math.maxInt(i32);
    var y0: i32 = std.math.maxInt(i32);
    var x1: i32 = std.math.minInt(i32);
    var y1: i32 = std.math.minInt(i32);
    for (contours) |c| {
        for (c) |p| {
            n_pts += 1;
            x0 = @min(x0, p.x);
            y0 = @min(y0, p.y);
            x1 = @max(x1, p.x);
            y1 = @max(y1, p.y);
        }
    }
    try fixtureWriteI16(w, @intCast(contours.len));
    try fixtureWriteI16(w, if (n_pts == 0) 0 else @intCast(x0));
    try fixtureWriteI16(w, if (n_pts == 0) 0 else @intCast(y0));
    try fixtureWriteI16(w, if (n_pts == 0) 0 else @intCast(x1));
    try fixtureWriteI16(w, if (n_pts == 0) 0 else @intCast(y1));
    var acc: usize = 0;
    for (contours) |c| {
        acc += c.len;
        try fixtureWriteU16(w, @intCast(acc - 1));
    }
    try fixtureWriteU16(w, 0); // no hinting instructions
    // Flags with repeat runs.
    var flags: [8192]u8 = undefined;
    var nf: usize = 0;
    var px: i32 = 0;
    var py: i32 = 0;
    for (contours) |c| {
        for (c) |p| {
            var f: u8 = if (p.on) 0x01 else 0x00;
            const dx = p.x - px;
            const dy = p.y - py;
            px = p.x;
            py = p.y;
            if (dx == 0) {
                f |= 0x10;
            } else if (dx >= -255 and dx <= 255) {
                f |= 0x02;
                if (dx > 0) f |= 0x10;
            }
            if (dy == 0) {
                f |= 0x20;
            } else if (dy >= -255 and dy <= 255) {
                f |= 0x04;
                if (dy > 0) f |= 0x20;
            }
            flags[nf] = f;
            nf += 1;
        }
    }
    var k: usize = 0;
    while (k < nf) {
        var run: usize = 1;
        while (k + run < nf and flags[k + run] == flags[k] and run < 256) : (run += 1) {}
        if (run > 1) {
            try w.writeByte(flags[k] | 0x08);
            try w.writeByte(@intCast(run - 1));
        } else {
            try w.writeByte(flags[k]);
        }
        k += run;
    }
    // X deltas then Y deltas.
    px = 0;
    py = 0;
    for (contours) |c| {
        for (c) |p| {
            const dx = p.x - px;
            px = p.x;
            if (dx == 0) {} else if (dx >= -255 and dx <= 255) {
                try w.writeByte(@intCast(@abs(dx)));
            } else {
                try fixtureWriteI16(w, @intCast(dx));
            }
        }
    }
    for (contours) |c| {
        for (c) |p| {
            const dy = p.y - py;
            py = p.y;
            if (dy == 0) {} else if (dy >= -255 and dy <= 255) {
                try w.writeByte(@intCast(@abs(dy)));
            } else {
                try fixtureWriteI16(w, @intCast(dy));
            }
        }
    }
    return out.toOwnedSlice();
}

/// Encodes one composite glyph: components of (gid, dx, dy, scale?) with
/// ARGS_ARE_XY_VALUES + byte args; `two_by_two` adds a 2x2 swap matrix.
fn fixtureCompositeGlyph(
    allocator: std.mem.Allocator,
    comps: []const struct { gid: u16, dx: i16, dy: i16, scale: ?u16, swap_xy: bool = false },
) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const w = &out.writer;
    try fixtureWriteI16(w, -1);
    try fixtureWriteI16(w, 0);
    try fixtureWriteI16(w, 0);
    try fixtureWriteI16(w, 1000);
    try fixtureWriteI16(w, 1000);
    for (comps, 0..) |c, k| {
        var flags: u16 = 0x0002; // ARGS_ARE_XY_VALUES
        if (c.scale) |_| flags |= 0x0008; // WE_HAVE_A_SCALE
        if (c.swap_xy) flags |= 0x0080; // WE_HAVE_A_TWO_BY_TWO
        if (k + 1 < comps.len) flags |= 0x0020; // MORE_COMPONENTS
        try fixtureWriteU16(w, flags);
        try fixtureWriteU16(w, c.gid);
        // Byte args (fits the fixture offsets).
        try w.writeByte(@intCast(@as(i8, @intCast(c.dx))));
        try w.writeByte(@intCast(@as(i8, @intCast(c.dy))));
        if (c.scale) |s| try fixtureWriteU16(w, s);
        if (c.swap_xy) {
            // Swap matrix [[0,1],[1,0]] in F2DOT14.
            try fixtureWriteU16(w, 0);
            try fixtureWriteU16(w, 0x4000);
            try fixtureWriteU16(w, 0x4000);
            try fixtureWriteU16(w, 0);
        }
    }
    return out.toOwnedSlice();
}

const FixtureTable = struct { tag: u32, data: []u8 };

/// Builds a minimal valid TTF in memory: head/maxp/hhea/hmtx/cmap/loca/
/// glyf (+kern) with glyphs: gid0 empty .notdef, gid1 rectangle outline,
/// gid2 triangle with a quadratic curve, gid3 composite (see
/// FixtureComposite modes). cmap format 4 maps A/B/C; format 12 maps
/// U+4E2D/U+4E2E (+U+10437 supplementary); kern pairs A-B = -80.
pub fn buildFixture(allocator: std.mem.Allocator, opts: FixtureOptions) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const tmp = arena.allocator();

    const rect = [_]FixturePoint{
        .{ .x = 0, .y = 0, .on = true },
        .{ .x = 600, .y = 0, .on = true },
        .{ .x = 600, .y = 800, .on = true },
        .{ .x = 0, .y = 800, .on = true },
    };
    const tri = [_]FixturePoint{
        .{ .x = 0, .y = 0, .on = true },
        .{ .x = 300, .y = 800, .on = false },
        .{ .x = 600, .y = 0, .on = true },
    };

    // Glyph list depends on the composite mode.
    var glyphs: std.ArrayListUnmanaged([]u8) = .empty;
    defer {
        for (glyphs.items) |g| allocator.free(g);
        glyphs.deinit(allocator);
    }
    const empty = try fixtureSimpleGlyph(tmp, &.{});
    _ = empty;
    // gid0: empty .notdef = 10 zero bytes (0 contours + bbox).
    {
        const z = try allocator.alloc(u8, 10);
        @memset(z, 0);
        try glyphs.append(allocator, z);
    }
    try glyphs.append(allocator, try fixtureSimpleGlyph(allocator, &.{&rect}));
    try glyphs.append(allocator, try fixtureSimpleGlyph(allocator, &.{&tri}));
    switch (opts.composite) {
        .normal => {
            try glyphs.append(allocator, try fixtureCompositeGlyph(allocator, &.{
                .{ .gid = 1, .dx = 100, .dy = 50, .scale = null },
            }));
        },
        .self_cyclic => {
            try glyphs.append(allocator, try fixtureCompositeGlyph(allocator, &.{
                .{ .gid = 3, .dx = 0, .dy = 0, .scale = null },
            }));
        },
        .pair_cyclic => {
            try glyphs.append(allocator, try fixtureCompositeGlyph(allocator, &.{
                .{ .gid = 4, .dx = 0, .dy = 0, .scale = null },
            }));
            try glyphs.append(allocator, try fixtureCompositeGlyph(allocator, &.{
                .{ .gid = 3, .dx = 0, .dy = 0, .scale = null },
            }));
        },
        .deep => {
            // Chain gid3 -> gid4 -> ... -> gid12 (10 nested levels, over
            // the depth-8 limit), gid12 a plain rectangle.
            var g: u16 = 12;
            const last = try fixtureSimpleGlyph(allocator, &.{&rect});
            // Build backwards: index = gid; glyphs[3..12] composites.
            var chain: [10][]u8 = undefined;
            chain[9] = last;
            var k: usize = 0;
            while (k < 9) : (k += 1) {
                chain[8 - k] = try fixtureCompositeGlyph(allocator, &.{
                    .{ .gid = g, .dx = 0, .dy = 0, .scale = null },
                });
                g -= 1;
            }
            for (chain[0..9]) |c| try glyphs.append(allocator, c);
            try glyphs.append(allocator, chain[9]);
        },
    }
    const num_glyphs: u16 = @intCast(glyphs.items.len);

    var tables: std.ArrayListUnmanaged(FixtureTable) = .empty;
    defer {
        for (tables.items) |t| allocator.free(t.data);
        tables.deinit(allocator);
    }

    // head (54 bytes).
    {
        var o: std.Io.Writer.Allocating = .init(tmp);
        const w = &o.writer;
        try fixtureWriteU32(w, 0x00010000);
        try fixtureWriteU32(w, 0);
        try fixtureWriteU32(w, 0); // checkSumAdjustment (unverified)
        try fixtureWriteU32(w, 0x5F0F3CF5);
        try fixtureWriteU16(w, 0);
        try fixtureWriteU16(w, opts.units_per_em);
        try w.writeAll(&.{ 0, 0, 0, 0, 0, 0, 0, 0 }); // created
        try w.writeAll(&.{ 0, 0, 0, 0, 0, 0, 0, 0 }); // modified
        try fixtureWriteI16(w, 0);
        try fixtureWriteI16(w, 0);
        try fixtureWriteI16(w, 1000);
        try fixtureWriteI16(w, 1000);
        try fixtureWriteU16(w, 0);
        try fixtureWriteU16(w, 8);
        try fixtureWriteI16(w, 2);
        try fixtureWriteI16(w, @intCast(opts.index_to_loc_format));
        try fixtureWriteI16(w, 0);
        try tables.append(allocator, .{ .tag = tag_head, .data = try allocator.dupe(u8, o.written()) });
    }
    // maxp (short layout).
    {
        const d = try allocator.alloc(u8, 6);
        std.mem.writeInt(u32, d[0..4], 0x00005000, .big);
        std.mem.writeInt(u16, d[4..6], num_glyphs, .big);
        try tables.append(allocator, .{ .tag = tag_maxp, .data = d });
    }
    // hhea (36 bytes): ascender 800, descender -200, lineGap 100.
    {
        const d = try allocator.alloc(u8, 36);
        @memset(d, 0);
        std.mem.writeInt(u32, d[0..4], 0x00010000, .big);
        std.mem.writeInt(i16, d[4..6], 800, .big);
        std.mem.writeInt(i16, d[6..8], -200, .big);
        std.mem.writeInt(i16, d[8..10], 100, .big);
        std.mem.writeInt(u16, d[34..36], num_glyphs, .big);
        try tables.append(allocator, .{ .tag = tag_hhea, .data = d });
    }
    // hmtx: advances {500,700,650,750,600,...}, lsb 50.
    {
        var o: std.Io.Writer.Allocating = .init(tmp);
        const w = &o.writer;
        const advs = [_]u16{ 500, 700, 650, 750, 600, 600, 600, 600, 600, 600, 600, 600 };
        for (0..num_glyphs) |k| {
            try fixtureWriteU16(w, advs[@min(k, advs.len - 1)]);
            try fixtureWriteI16(w, 50);
        }
        try tables.append(allocator, .{ .tag = tag_hmtx, .data = try allocator.dupe(u8, o.written()) });
    }
    // cmap.
    {
        var o: std.Io.Writer.Allocating = .init(tmp);
        const w = &o.writer;
        const want4 = opts.cmap == .fmt4 or opts.cmap == .both;
        const want12 = opts.cmap == .fmt12 or opts.cmap == .both;
        const n_sub: u16 = if (opts.cmap == .unsupported) 1 else (if (want4 and want12) 2 else 1);
        try fixtureWriteU16(w, 0);
        try fixtureWriteU16(w, n_sub);
        // Offsets patched below; subtables appended after the header.
        var off_pos: [2]usize = undefined;
        var n_rec: usize = 0;
        if (want4) {
            try fixtureWriteU16(w, 3);
            try fixtureWriteU16(w, 1);
            off_pos[n_rec] = o.written().len;
            n_rec += 1;
            try fixtureWriteU32(w, 0);
        }
        if (want12) {
            try fixtureWriteU16(w, 3);
            try fixtureWriteU16(w, 10);
            off_pos[n_rec] = o.written().len;
            n_rec += 1;
            try fixtureWriteU32(w, 0);
        }
        if (opts.cmap == .unsupported) {
            // Format 0 (256-byte identity-less table): valid but useless —
            // the parser must reject the file with UnsupportedCmap.
            try fixtureWriteU16(w, 3);
            try fixtureWriteU16(w, 1);
            off_pos[0] = o.written().len;
            n_rec = 1;
            try fixtureWriteU32(w, 0);
        }
        var sub_off: [2]u32 = undefined;
        if (want4) {
            sub_off[0] = @intCast(o.written().len);
            // Format 4: one segment A..C + terminator. idDelta maps
            // 0x41->1: delta = (1 - 0x41) mod 65536 = 65472.
            const seg_count: u16 = 2;
            try fixtureWriteU16(w, 4);
            try fixtureWriteU16(w, 16 + seg_count * 8);
            try fixtureWriteU16(w, 0);
            try fixtureWriteU16(w, seg_count * 2);
            try fixtureWriteU16(w, 4); // searchRange 2*2^1
            try fixtureWriteU16(w, 1); // entrySelector
            try fixtureWriteU16(w, 0); // rangeShift
            try fixtureWriteU16(w, 0x43);
            try fixtureWriteU16(w, 0xFFFF);
            try fixtureWriteU16(w, 0); // reservedPad
            try fixtureWriteU16(w, 0x41);
            try fixtureWriteU16(w, 0xFFFF);
            try fixtureWriteU16(w, 65472); // (1-0x41) idDelta
            try fixtureWriteU16(w, 1);
            try fixtureWriteU16(w, 0);
            try fixtureWriteU16(w, 0);
        }
        if (want12) {
            sub_off[if (want4) 1 else 0] = @intCast(o.written().len);
            // Format 12: U+4E2D->1, U+4E2E->2, U+10437->3.
            const groups: [3][3]u32 = .{
                .{ 0x4E2D, 0x4E2D, 1 },
                .{ 0x4E2E, 0x4E2E, 2 },
                .{ 0x10437, 0x10437, 3 },
            };
            try fixtureWriteU16(w, 12);
            try fixtureWriteU16(w, 0);
            try fixtureWriteU32(w, 16 + 12 * 3);
            try fixtureWriteU32(w, 0);
            try fixtureWriteU32(w, 3);
            for (groups) |gr| {
                try fixtureWriteU32(w, gr[0]);
                try fixtureWriteU32(w, gr[1]);
                try fixtureWriteU32(w, gr[2]);
            }
        }
        if (opts.cmap == .unsupported) {
            sub_off[0] = @intCast(o.written().len);
            try fixtureWriteU16(w, 0);
            try fixtureWriteU16(w, 262);
            try fixtureWriteU16(w, 0);
            var z: usize = 0;
            while (z < 256) : (z += 1) try w.writeByte(0);
        }
        const bytes = o.written();
        for (0..n_rec) |k| {
            std.mem.writeInt(u32, @constCast(bytes[off_pos[k]..][0..4]), sub_off[k], .big);
        }
        try tables.append(allocator, .{ .tag = tag_cmap, .data = try allocator.dupe(u8, bytes) });
    }
    // loca + glyf.
    {
        var lo: std.Io.Writer.Allocating = .init(tmp);
        var go: std.Io.Writer.Allocating = .init(tmp);
        const lw = &lo.writer;
        const gw = &go.writer;
        var cur_off: u32 = 0;
        for (glyphs.items) |g| {
            const use: u32 = if (opts.loca == .out_of_range) cur_off + 100000 else cur_off;
            if (opts.index_to_loc_format == 0) {
                try fixtureWriteU16(lw, @intCast(use / 2));
            } else {
                try fixtureWriteU32(lw, use);
            }
            try gw.writeAll(g);
            cur_off += @intCast(g.len);
        }
        if (opts.index_to_loc_format == 0) {
            try fixtureWriteU16(lw, @intCast(cur_off / 2));
        } else {
            try fixtureWriteU32(lw, cur_off);
        }
        // loca entries must be even for short format: glyph blobs are
        // already even-length (all encoders emit multiples of 2).
        try tables.append(allocator, .{ .tag = tag_loca, .data = try allocator.dupe(u8, lo.written()) });
        try tables.append(allocator, .{ .tag = tag_glyf, .data = try allocator.dupe(u8, go.written()) });
    }
    // kern format 0: pair (A-gid 1, B-gid 2) = -80.
    if (opts.include_kern) {
        var o: std.Io.Writer.Allocating = .init(tmp);
        const w = &o.writer;
        try fixtureWriteU16(w, 0);
        try fixtureWriteU16(w, 1);
        try fixtureWriteU16(w, 0); // subtable version
        try fixtureWriteU16(w, 20); // length
        try w.writeByte(0); // format 0
        try w.writeByte(0); // coverage
        try fixtureWriteU16(w, 1); // nPairs
        try fixtureWriteU16(w, 2); // searchRange
        try fixtureWriteU16(w, 0); // entrySelector
        try fixtureWriteU16(w, 2); // rangeShift
        try fixtureWriteU16(w, 1); // left
        try fixtureWriteU16(w, 2); // right
        try fixtureWriteI16(w, -80); // value
        try tables.append(allocator, .{ .tag = tag_kern, .data = try allocator.dupe(u8, o.written()) });
    }
    if (opts.include_fvar) {
        const d = try allocator.alloc(u8, 16);
        @memset(d, 0);
        try tables.append(allocator, .{ .tag = tag_fvar, .data = d });
    }
    if (opts.include_cff_table) {
        const d = try allocator.alloc(u8, 8);
        @memset(d, 0);
        try tables.append(allocator, .{ .tag = tag_cff, .data = d });
    }

    // Offset table + records + 4-aligned table data.
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const w = &out.writer;
    const n_tables: u16 = @intCast(tables.items.len);
    try fixtureWriteU32(w, opts.sfnt_version);
    try fixtureWriteU16(w, n_tables);
    var pow: u16 = 1;
    while (pow * 2 <= n_tables) : (pow *= 2) {}
    try fixtureWriteU16(w, pow * 16);
    var elog: u16 = 0;
    var p2: u16 = pow;
    while (p2 > 1) : (p2 /= 2) {
        elog += 1;
    }
    try fixtureWriteU16(w, elog);
    try fixtureWriteU16(w, n_tables * 16 - pow * 16);
    var data_off: u32 = 12 + @as(u32, n_tables) * 16;
    for (tables.items) |t| {
        try fixtureWriteU32(w, t.tag);
        try fixtureWriteU32(w, 0); // checksum (unverified by design)
        try fixtureWriteU32(w, data_off);
        try fixtureWriteU32(w, @intCast(t.data.len));
        data_off += @intCast((t.data.len + 3) & ~@as(usize, 3));
    }
    for (tables.items) |t| {
        try w.writeAll(t.data);
        const pad = (@as(usize, 4) - (t.data.len & 3)) & 3;
        const zeros = [_]u8{ 0, 0, 0 };
        if (pad > 0) try w.writeAll(zeros[0..pad]);
    }
    var file = try out.toOwnedSlice();
    if (opts.truncate_bytes > 0) {
        // Shrink with realloc (not a reslice: the freed size must match
        // the allocation the writer owned).
        if (opts.truncate_bytes >= file.len) {
            allocator.free(file);
            return error.OutOfMemory; // test bug, not a font error
        }
        file = try allocator.realloc(file, file.len - opts.truncate_bytes);
    }
    return file;
}

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
