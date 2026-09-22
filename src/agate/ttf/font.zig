//! sfnt container parse + table queries (split out of `ttf.zig`, facade).
//!
//! Owns `sniff`, the borrowing `Font` handle (parse, glyph ranges, cmap,
//! advances, kern) and the private cmap/kern validators. Imports `types`
//! only. Constant/type aliases above keep the moved bodies byte-identical;
//! see the facade for the anti-cycle rule.

const std = @import("std");
const types = @import("types.zig");

const TtfError = types.TtfError;
const readU16At = types.readU16At;
const readU32At = types.readU32At;
const tag_head: u32 = types.tag_head;
const tag_maxp: u32 = types.tag_maxp;
const tag_cmap: u32 = types.tag_cmap;
const tag_loca: u32 = types.tag_loca;
const tag_glyf: u32 = types.tag_glyf;
const tag_hhea: u32 = types.tag_hhea;
const tag_hmtx: u32 = types.tag_hmtx;
const tag_kern: u32 = types.tag_kern;
const tag_cff: u32 = types.tag_cff;
const tag_cff2: u32 = types.tag_cff2;
const tag_fvar: u32 = types.tag_fvar;
const max_tables: usize = types.max_tables;
const max_cmap_records: usize = types.max_cmap_records;

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
