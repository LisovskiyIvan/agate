//! In-memory fixture font writer (split out of `ttf.zig`, facade).
//!
//! Owns `FixtureOptions` and `buildFixture`: the programmatic minimal TTF
//! the TTF tests and the UI text-path tests (`ui/text.zig`) share, so no
//! font files are bundled. Imports `types` for the table tags only.

const std = @import("std");
const types = @import("types.zig");

const tag_head: u32 = types.tag_head;
const tag_maxp: u32 = types.tag_maxp;
const tag_hhea: u32 = types.tag_hhea;
const tag_hmtx: u32 = types.tag_hmtx;
const tag_cmap: u32 = types.tag_cmap;
const tag_loca: u32 = types.tag_loca;
const tag_glyf: u32 = types.tag_glyf;
const tag_kern: u32 = types.tag_kern;
const tag_fvar: u32 = types.tag_fvar;
const tag_cff: u32 = types.tag_cff;

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
