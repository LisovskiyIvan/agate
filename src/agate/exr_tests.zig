const std = @import("std");

const exr = @import("exr.zig");
const magic = exr.magic;
const sniff = exr.sniff;
const decode = exr.decode;
const DecodeError = exr.DecodeError;
const DecodeOptions = exr.DecodeOptions;
const RawHDR = exr.RawHDR;
const compression_none = exr.compression_none;
const compression_rle = exr.compression_rle;
const compression_zips = exr.compression_zips;
const compression_zip = exr.compression_zip;
const pixel_uint = exr.pixel_uint;
const pixel_half = exr.pixel_half;
const pixel_float = exr.pixel_float;
const zlibDecompress = exr.zlibDecompress;
const rleDecompress = exr.rleDecompress;
const unpredictAndReorder = exr.unpredictAndReorder;
const floatToHalfBits = exr.floatToHalfBits;
const decodeRleBlock = exr.decodeRleBlock;

// ---------------------------------------------------------------------------
// Tests — fixtures are synthesized in-memory by TestExr (magic + version +
// attributes + offset table + scanline blocks): the container is simple
// enough that no external .exr files are needed.
// ---------------------------------------------------------------------------

const testing = std.testing;

const TestChannel = struct {
    name: []const u8,
    pixel_type: u32 = pixel_half,
    x_sampling: u32 = 1,
    y_sampling: u32 = 1,
};

/// Minimal EXR builder for tests: header attributes in the canonical order,
/// an exact offset table, then one scanline block per row (NONE). Raw sample
/// bytes are passed per channel (`channel_data[i]` holds width*height
/// samples, little-endian) so bit-exact half patterns are expressible.
const TestExr = struct {
    width: u32 = 2,
    height: u32 = 2,
    xmin: i32 = 0,
    ymin: i32 = 0,
    channels: []const TestChannel = &.{ .{ .name = "R" }, .{ .name = "G" }, .{ .name = "B" }, .{ .name = "A" } },
    channel_data: []const []const u8 = &.{},
    compression: u8 = compression_none,
    line_order: u8 = 0,
    aspect: f32 = 1.0,
    version: u32 = 2,
    corrupt_magic: bool = false,
    /// Omits one required attribute by name (MissingAttribute fixtures).
    drop_attr: ?[]const u8 = null,
    /// Unknown attribute carried through the header (skip-path fixture).
    extra_attr_name: ?[]const u8 = null,
    extra_attr_type: []const u8 = "string",
    extra_attr_value: []const u8 = "hello",
    /// Display window override (null = same as the data window).
    disp: ?[4]i32 = null,
    /// Corrupts one offset-table entry (index) to point past EOF.
    corrupt_offset: ?usize = null,
    /// Declares one block's size one byte short (payload stays full).
    shrink_block: ?usize = null,
    /// Writes the second block with the first block's y (duplicate scanline).
    dupe_y: bool = false,
    /// Writes the first block with y = ymax + 1 (out-of-range scanline).
    bad_y: bool = false,
    /// Writes the first block one row past its aligned start.
    misalign_y: bool = false,
    /// Flips the first payload byte of one block (file-order index).
    corrupt_payload: ?usize = null,
    trailing_garbage: bool = false,

    fn build(self: TestExr, allocator: std.mem.Allocator) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(allocator);
        errdefer out.deinit();
        const w = &out.writer;

        if (self.corrupt_magic) {
            try w.writeAll("NOPE");
        } else {
            try w.writeAll(&magic);
        }
        try w.writeInt(u32, self.version, .little);

        const xmax = self.xmin + @as(i64, self.width) - 1;
        const ymax = self.ymin + @as(i64, self.height) - 1;
        const dw: [4]i32 = .{
            self.xmin,
            self.ymin,
            @intCast(xmax),
            @intCast(ymax),
        };
        const disp = self.disp orelse dw;

        // Channels attribute value (built first: size-prefixed).
        var ch_value: std.Io.Writer.Allocating = .init(allocator);
        defer ch_value.deinit();
        for (self.channels) |ch| {
            try ch_value.writer.writeAll(ch.name);
            try ch_value.writer.writeByte(0);
            try ch_value.writer.writeInt(u32, ch.pixel_type, .little);
            try ch_value.writer.writeByte(0); // pLinear
            try ch_value.writer.writeAll(&.{ 0, 0, 0 }); // reserved
            try ch_value.writer.writeInt(u32, ch.x_sampling, .little);
            try ch_value.writer.writeInt(u32, ch.y_sampling, .little);
        }
        try ch_value.writer.writeByte(0); // chlist terminator

        try writeAttr(w, "channels", "chlist", ch_value.written(), self.drop_attr);
        try writeAttrByte(w, "compression", "compression", self.compression, self.drop_attr);
        try writeAttrBox(w, "dataWindow", dw, self.drop_attr);
        try writeAttrBox(w, "displayWindow", disp, self.drop_attr);
        try writeAttrByte(w, "lineOrder", "lineOrder", self.line_order, self.drop_attr);
        var aspect_bits: [4]u8 = undefined;
        std.mem.writeInt(u32, &aspect_bits, @bitCast(self.aspect), .little);
        try writeAttr(w, "pixelAspectRatio", "float", &aspect_bits, self.drop_attr);
        if (self.extra_attr_name) |ename| {
            try writeAttr(w, ename, self.extra_attr_type, self.extra_attr_value, null);
        }
        try w.writeByte(0); // header terminator

        // Blocks: ceil(h / lines-per-block) chunks in file order. The order
        // follows the line order so valid fixtures exercise the
        // placement-by-y logic (decreasing/random are still legal files,
        // not just decoder corner cases). Payloads are encoded first so the
        // offset table (one entry per block) can be written up front.
        const h: usize = self.height;
        const lpb: usize = switch (self.compression) {
            compression_none, compression_rle, compression_zips => 1,
            compression_zip => 16,
            else => 1, // unreadable codes still need a well-formed layout
        };
        const num_blocks = (h + lpb - 1) / lpb;
        var border_buf: [16384]usize = undefined;
        const border = border_buf[0..num_blocks];
        for (0..num_blocks) |i| border[i] = i;
        if (self.line_order == 1) std.mem.reverse(usize, border);
        if (self.line_order == 2) {
            // Deterministic pairwise swap (a non-monotonic but valid order).
            var i: usize = 0;
            while (i + 1 < num_blocks) : (i += 2) std.mem.swap(usize, &border[i], &border[i + 1]);
        }

        // Bytes per scanline across channels.
        var scan_bytes: usize = 0;
        for (self.channels) |ch| {
            scan_bytes += @as(usize, self.width) * (if (ch.pixel_type == pixel_half) @as(usize, 2) else 4);
        }

        // Encode every payload (file order) before writing the table.
        var payloads: [16384][]u8 = undefined;
        var encoded: usize = 0;
        // Single defer: frees stored payloads on success and on error.
        defer {
            for (payloads[0..encoded]) |p| allocator.free(p);
        }
        for (0..num_blocks) |f| {
            const bi = border[f];
            const first_row = bi * lpb;
            const rows = @min(lpb, h - first_row);
            // Raw rows, addressed by true row so shuffled orders still carry
            // the right samples per y. Short fixtures zero-pad so
            // degenerate-dimension cases still build.
            const raw = try allocator.alloc(u8, rows * scan_bytes);
            defer allocator.free(raw);
            for (0..rows) |li| {
                const row = first_row + li;
                for (self.channels, 0..) |ch, ci| {
                    const bpp: usize = if (ch.pixel_type == pixel_half) 2 else 4;
                    const need: usize = @as(usize, self.width) * bpp;
                    const src = self.channel_data[ci];
                    const off: usize = row * need;
                    const dst = (li * scan_bytes) + prefixLen(self.channels, self.width, ci);
                    const avail: usize = if (off < src.len) @min(need, src.len - off) else 0;
                    if (avail > 0) @memcpy(raw[dst .. dst + avail], src[off .. off + avail]);
                    if (avail < need) @memset(raw[dst + avail .. dst + need], 0);
                }
            }
            payloads[f] = if (self.compression == compression_rle)
                try rlePredictEncode(allocator, raw)
            else if (self.compression == compression_zips or self.compression == compression_zip)
                try zipDeflateEncode(allocator, raw)
            else
                try allocator.dupe(u8, raw);
            encoded = f + 1;
        }

        const table_pos: u64 = @intCast(out.written().len);
        var off: u64 = table_pos + @as(u64, num_blocks) * 8;
        for (0..num_blocks) |f| {
            var entry = off;
            if (self.corrupt_offset) |ci| {
                if (ci == f) entry = @as(u64, @intCast(out.written().len)) + 0x10000;
            }
            try w.writeInt(u64, entry, .little);
            off += 8 + @as(u64, payloads[f].len);
        }
        for (0..num_blocks) |f| {
            const bi = border[f];
            var y = self.ymin + @as(i32, @intCast(bi * lpb));
            if (self.dupe_y and f == 1) y = self.ymin + @as(i32, @intCast(border[0] * lpb));
            if (self.bad_y and f == 0) y = @as(i32, @intCast(ymax)) + 1;
            // Misaligned block start (only meaningful for single-block
            // files: the row is otherwise valid and unseen).
            if (self.misalign_y and f == 0) y += 1;
            // Payload mutation happens after the table is written (lengths
            // are unchanged, so offsets stay exact).
            if (self.corrupt_payload) |ci| {
                if (ci == f and payloads[f].len > 0) payloads[f][0] ^= 0xFF;
            }
            try w.writeInt(i32, y, .little);
            const full_len = payloads[f].len;
            const declared: u32 = @intCast(if (self.shrink_block) |si| (if (si == f and full_len > 0) full_len - 1 else full_len) else full_len);
            try w.writeInt(u32, declared, .little);
            try w.writeAll(payloads[f]);
        }
        if (self.trailing_garbage) try w.writeAll(&.{ 0xAA, 0xBB });
        return out.toOwnedSlice();
    }

    /// Scanline bytes occupied by the channels before index `before`.
    fn prefixLen(channels: []const TestChannel, width: u32, before: usize) usize {
        var len: usize = 0;
        for (channels[0..before]) |ch| {
            len += @as(usize, width) * (if (ch.pixel_type == pixel_half) @as(usize, 2) else 4);
        }
        return len;
    }

    fn writeAttr(w: *std.Io.Writer, name: []const u8, typ: []const u8, value: []const u8, drop: ?[]const u8) !void {
        if (drop) |d| {
            if (std.mem.eql(u8, d, name)) return;
        }
        try w.writeAll(name);
        try w.writeByte(0);
        try w.writeAll(typ);
        try w.writeByte(0);
        try w.writeInt(u32, @intCast(value.len), .little);
        try w.writeAll(value);
    }

    fn writeAttrByte(w: *std.Io.Writer, name: []const u8, typ: []const u8, v: u8, drop: ?[]const u8) !void {
        return writeAttr(w, name, typ, &.{v}, drop);
    }

    fn writeAttrBox(w: *std.Io.Writer, name: []const u8, box: [4]i32, drop: ?[]const u8) !void {
        var buf: [16]u8 = undefined;
        for (box, 0..) |c, i| std.mem.writeInt(i32, buf[i * 4 ..][0..4], c, .little);
        return writeAttr(w, name, "box2i", &buf, drop);
    }
};

/// Packs u16 half patterns as little-endian sample bytes for TestExr.
fn halfSamples(allocator: std.mem.Allocator, patterns: []const u16) ![]u8 {
    const buf = try allocator.alloc(u8, patterns.len * 2);
    for (patterns, 0..) |p, i| std.mem.writeInt(u16, buf[i * 2 ..][0..2], p, .little);
    return buf;
}

/// Packs f32 values as little-endian sample bytes for TestExr.
fn floatSamples(allocator: std.mem.Allocator, values: []const f32) ![]u8 {
    const buf = try allocator.alloc(u8, values.len * 4);
    for (values, 0..) |v, i| std.mem.writeInt(u32, buf[i * 4 ..][0..4], @as(u32, @bitCast(v)), .little);
    return buf;
}

/// Applies the OpenEXR predictor + interleave to `raw`, writing the
/// transformed bytes into `out` (exact port of openexr's
/// reorder_and_predict): even bytes pack into the first half, odd bytes
/// into the second, then each byte becomes its difference from the
/// predecessor plus 128.
fn reorderAndPredict(out: []u8, raw: []const u8) void {
    std.debug.assert(out.len == raw.len);
    if (raw.len == 0) return;
    const half = (raw.len + 1) / 2;
    for (raw, 0..) |b, i| {
        if (i % 2 == 0) {
            out[i / 2] = b;
        } else {
            out[half + i / 2] = b;
        }
    }
    var prev = out[0];
    for (out[1..]) |*t| {
        const cur = t.*;
        t.* = cur -% prev +% 128;
        prev = cur;
    }
}

/// Run-length-encodes `src` (exact port of openexr's internal_rle_compress):
/// runs of 3+ equal bytes become (length-1, value), everything else packs
/// into literal runs of up to 127 as (-length, bytes...).
fn rleCompress(allocator: std.mem.Allocator, src: []const u8) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    const w = &out.writer;
    var runs: usize = 0;
    var rune: usize = 1;
    while (runs < src.len) {
        var curcount: usize = 0;
        while (rune < src.len and src[runs] == src[rune] and curcount < 127) {
            rune += 1;
            curcount += 1;
        }
        if (curcount >= 2) {
            try w.writeByte(@intCast(curcount));
            try w.writeByte(src[runs]);
            runs = rune;
        } else {
            curcount += 1;
            while (rune < src.len and
                ((rune + 1 >= src.len or src[rune] != src[rune + 1]) or
                    (rune + 2 >= src.len or src[rune + 1] != src[rune + 2])) and
                curcount < 127)
            {
                curcount += 1;
                rune += 1;
            }
            try w.writeByte(0 -% @as(u8, @intCast(curcount)));
            try w.writeAll(src[runs..rune]);
            runs = rune;
        }
        rune += 1;
    }
    return out.toOwnedSlice();
}

/// Predictor + RLE as stored in RLE scanline blocks.
fn rlePredictEncode(allocator: std.mem.Allocator, raw: []const u8) ![]u8 {
    if (raw.len == 0) return allocator.dupe(u8, raw);
    const tmp = try allocator.alloc(u8, raw.len);
    defer allocator.free(tmp);
    reorderAndPredict(tmp, raw);
    return rleCompress(allocator, tmp);
}

/// Predictor + zlib/deflate as stored in ZIP/ZIPS scanline blocks.
fn zipDeflateEncode(allocator: std.mem.Allocator, raw: []const u8) ![]u8 {
    const tmp = try allocator.alloc(u8, raw.len);
    defer allocator.free(tmp);
    reorderAndPredict(tmp, raw);
    const history = try allocator.alloc(u8, std.compress.flate.max_window_len);
    defer allocator.free(history);
    // Test payloads are small (tens of scanlines); the fixed buffer only
    // needs to hold the largest single-block stream.
    var buf: [16384]u8 = undefined;
    var fixed: std.Io.Writer = .fixed(&buf);
    var comp = try std.compress.flate.Compress.init(&fixed, history, .zlib, std.compress.flate.Compress.Options.fastest);
    try comp.writer.writeAll(tmp);
    try comp.finish();
    return allocator.dupe(u8, buf[0..fixed.end]);
}

test "sniff matches the EXR magic and nothing else" {
    try testing.expect(sniff(&magic));
    try testing.expect(!sniff("not an exr file!!"));
    try testing.expect(!sniff(magic[0..3])); // truncated magic
    try testing.expect(!sniff(&.{}));
    // Neighboring container magics must not route here.
    try testing.expect(!sniff(&[4]u8{ 'D', 'D', 'S', ' ' }));
    try testing.expect(!sniff(&[12]u8{ 0xAB, 'K', 'T', 'X', ' ', '2', '0', 0xBB, 0x0D, 0x0A, 0x1A, 0x0A }));
}

test "decode reads uncompressed RGBA half exactly as authored" {
    const allocator = testing.allocator;
    // 2x2 texels with distinct half patterns (incl. subnormal + negative).
    const r = try halfSamples(allocator, &.{ 0x3C00, 0x4000, 0x0001, 0xBC00 });
    defer allocator.free(r);
    const g = try halfSamples(allocator, &.{ 0x0000, 0x3800, 0x3C00, 0x3C00 });
    defer allocator.free(g);
    const b = try halfSamples(allocator, &.{ 0x7C00, 0x0000, 0x3C00, 0x3555 });
    defer allocator.free(b);
    const a = try halfSamples(allocator, &.{ 0x3C00, 0x3C00, 0x3C00, 0x4000 });
    defer allocator.free(a);
    const file = try (TestExr{ .channel_data = &.{ r, g, b, a } }).build(allocator);
    defer allocator.free(file);

    var dec = try decode(allocator, file);
    defer dec.deinit(allocator);
    try testing.expectEqual(@as(u32, 2), dec.width);
    try testing.expectEqual(@as(u32, 2), dec.height);
    const want = .{ 0x3C00, 0x0000, 0x7C00, 0x3C00, 0x4000, 0x3800, 0x0000, 0x3C00, 0x0001, 0x3C00, 0x3C00, 0x3C00, 0xBC00, 0x3C00, 0x3555, 0x4000 };
    try testing.expectEqualSlices(u16, &want, dec.pixels);
    // Owned copy (the asset queue frees the source right after decode).
    @memset(file, 0xFF);
    try testing.expectEqual(@as(u16, 0x3C00), dec.pixels[0]);
}

test "decode reads uncompressed RGB float and defaults alpha to 1.0" {
    const allocator = testing.allocator;
    const r = try floatSamples(allocator, &.{1.5});
    defer allocator.free(r);
    const g = try floatSamples(allocator, &.{-0.25});
    defer allocator.free(g);
    const b = try floatSamples(allocator, &.{100000.0});
    defer allocator.free(b);
    const file = try (TestExr{
        .width = 1,
        .height = 1,
        .channels = &.{ .{ .name = "R", .pixel_type = pixel_float }, .{ .name = "G", .pixel_type = pixel_float }, .{ .name = "B", .pixel_type = pixel_float } },
        .channel_data = &.{ r, g, b },
    }).build(allocator);
    defer allocator.free(file);

    var dec = try decode(allocator, file);
    defer dec.deinit(allocator);
    try testing.expectEqual(@as(u32, 1), dec.width);
    try testing.expectEqual([4]u16{
        floatToHalfBits(1.5),
        floatToHalfBits(-0.25),
        floatToHalfBits(100000.0),
        0x3C00, // missing alpha defaults to 1.0
    }, dec.pixels[0..4].*);
    try testing.expectEqual(floatToHalfBits(1.5), @as(u16, 0x3E00));
}

test "decode expands grayscale Y with a non-zero data-window origin" {
    const allocator = testing.allocator;
    const y = try halfSamples(allocator, &.{ 0x4000, 0xC000 });
    defer allocator.free(y);
    const file = try (TestExr{
        .width = 2,
        .height = 1,
        .xmin = 5,
        .ymin = 7,
        .channels = &.{.{ .name = "Y" }},
        .channel_data = &.{y},
    }).build(allocator);
    defer allocator.free(file);

    var dec = try decode(allocator, file);
    defer dec.deinit(allocator);
    try testing.expectEqual(@as(u32, 2), dec.width);
    try testing.expectEqual(@as(u32, 1), dec.height);
    try testing.expectEqualSlices(u16, &.{ 0x4000, 0x4000, 0x4000, 0x3C00, 0xC000, 0xC000, 0xC000, 0x3C00 }, dec.pixels);
}

test "decode places decreasing-line-order rows by their y headers" {
    const allocator = testing.allocator;
    const r = try halfSamples(allocator, &.{ 0x3C00, 0x4000, 0x4200 });
    defer allocator.free(r);
    const file = try (TestExr{
        .width = 1,
        .height = 3,
        .channels = &.{.{ .name = "R" }},
        .channel_data = &.{r},
        .line_order = 1,
    }).build(allocator);
    defer allocator.free(file);

    var dec = try decode(allocator, file);
    defer dec.deinit(allocator);
    // Row 0 carries 1.0, row 1 carries 2.0, row 2 carries 3.0 regardless of
    // the file's bottom-up block order.
    try testing.expectEqualSlices(u16, &.{
        0x3C00, 0x0000, 0x0000, 0x3C00,
        0x4000, 0x0000, 0x0000, 0x3C00,
        0x4200, 0x0000, 0x0000, 0x3C00,
    }, dec.pixels);
}

test "decode reads random-line-order blocks by their y headers" {
    const allocator = testing.allocator;
    const r = try halfSamples(allocator, &.{ 0x3C00, 0x4000, 0x4200, 0x4400 });
    defer allocator.free(r);
    const file = try (TestExr{
        .width = 1,
        .height = 4,
        .channels = &.{.{ .name = "R" }},
        .channel_data = &.{r},
        .line_order = 2,
    }).build(allocator);
    defer allocator.free(file);

    var dec = try decode(allocator, file);
    defer dec.deinit(allocator);
    // Pairwise-swapped file order lands on the same pixels as increasing.
    try testing.expectEqual(@as(u16, 0x3C00), dec.pixels[0 * 4]);
    try testing.expectEqual(@as(u16, 0x4000), dec.pixels[1 * 4]);
    try testing.expectEqual(@as(u16, 0x4200), dec.pixels[2 * 4]);
    try testing.expectEqual(@as(u16, 0x4400), dec.pixels[3 * 4]);
}

test "decode skips unknown attributes and channels" {
    const allocator = testing.allocator;
    const r = try halfSamples(allocator, &.{0x3C00});
    defer allocator.free(r);
    const z = try floatSamples(allocator, &.{42.0});
    defer allocator.free(z);
    const file = try (TestExr{
        .width = 1,
        .height = 1,
        .channels = &.{ .{ .name = "R" }, .{ .name = "Z", .pixel_type = pixel_float } },
        .channel_data = &.{ r, z },
        .extra_attr_name = "comments",
    }).build(allocator);
    defer allocator.free(file);

    var dec = try decode(allocator, file);
    defer dec.deinit(allocator);
    try testing.expectEqual([4]u16{ 0x3C00, 0x0000, 0x0000, 0x3C00 }, dec.pixels[0..4].*);
}

test "decode reads RLE RGBA half across multiple blocks" {
    const allocator = testing.allocator;
    // 4x3: row 0 distinct, row 1 constant (repeat-run heavy after the
    // predictor), row 2 distinct — every block exercises a different shape.
    const r = try halfSamples(allocator, &.{ 0x3C00, 0x4000, 0x4200, 0x4400, 0x3C00, 0x3C00, 0x3C00, 0x3C00, 0xBC00, 0xBC00, 0x3800, 0x3800 });
    defer allocator.free(r);
    const g = try halfSamples(allocator, &.{ 0x0000, 0x0000, 0x0000, 0x0000, 0x4000, 0x4000, 0x4000, 0x4000, 0x0001, 0x0001, 0x0001, 0x0001 });
    defer allocator.free(g);
    const b = try halfSamples(allocator, &.{ 0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x0000, 0x0000, 0x0000, 0x0000, 0x7C00, 0x0000, 0x7C00, 0x0000 });
    defer allocator.free(b);
    const a = try halfSamples(allocator, &.{ 0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00, 0x3C00 });
    defer allocator.free(a);
    const file = try (TestExr{
        .width = 4,
        .height = 3,
        .channel_data = &.{ r, g, b, a },
        .compression = compression_rle,
    }).build(allocator);
    defer allocator.free(file);

    var dec = try decode(allocator, file);
    defer dec.deinit(allocator);
    try testing.expectEqual(@as(u32, 4), dec.width);
    try testing.expectEqual(@as(u32, 3), dec.height);
    // Spot-check each row's R lane (full-row equality would just re-spell
    // the fixture; the decoder must place every block's rows exactly).
    try testing.expectEqual(@as(u16, 0x3C00), dec.pixels[0 * 4]);
    try testing.expectEqual(@as(u16, 0x4400), dec.pixels[3 * 4]);
    try testing.expectEqual(@as(u16, 0x3C00), dec.pixels[4 * 4]);
    try testing.expectEqual(@as(u16, 0xBC00), dec.pixels[8 * 4]);
    try testing.expectEqual(@as(u16, 0x3800), dec.pixels[10 * 4]);
    try testing.expectEqual(@as(u16, 0x0001), dec.pixels[8 * 4 + 1]);
    try testing.expectEqual(@as(u16, 0x7C00), dec.pixels[8 * 4 + 2]);
    try testing.expectEqual(@as(u16, 0x0000), dec.pixels[9 * 4 + 2]);
    for (dec.pixels, 0..) |lane, i| {
        if (i % 4 == 3) try testing.expectEqual(@as(u16, 0x3C00), lane);
    }
}

test "decode reads RLE float Y gray with decreasing line order" {
    const allocator = testing.allocator;
    const y = try floatSamples(allocator, &.{ 0.5, 2.0 });
    defer allocator.free(y);
    const file = try (TestExr{
        .width = 1,
        .height = 2,
        .channels = &.{.{ .name = "Y", .pixel_type = pixel_float }},
        .channel_data = &.{y},
        .compression = compression_rle,
        .line_order = 1,
    }).build(allocator);
    defer allocator.free(file);

    var dec = try decode(allocator, file);
    defer dec.deinit(allocator);
    try testing.expectEqualSlices(u16, &.{
        floatToHalfBits(0.5), floatToHalfBits(0.5), floatToHalfBits(0.5), 0x3C00,
        floatToHalfBits(2.0), floatToHalfBits(2.0), floatToHalfBits(2.0), 0x3C00,
    }, dec.pixels);
    try testing.expectEqual(@as(u16, 0x3800), floatToHalfBits(0.5));
    try testing.expectEqual(@as(u16, 0x4000), floatToHalfBits(2.0));
}

test "predictor pair matches the spec vector and inverts" {
    // Hand-computed from openexr's reorder_and_predict/unpredict_and_reorder:
    // split [10,20,30,40] -> [10,30,20,40], differences +128 -> the vector.
    var predicted: [4]u8 = undefined;
    reorderAndPredict(&predicted, &.{ 0x10, 0x20, 0x30, 0x40 });
    try testing.expectEqualSlices(u8, &.{ 0x10, 0xA0, 0x70, 0xA0 }, &predicted);

    var back: [4]u8 = undefined;
    var scratch = predicted;
    unpredictAndReorder(&back, &scratch);
    try testing.expectEqualSlices(u8, &.{ 0x10, 0x20, 0x30, 0x40 }, &back);

    // Odd lengths route the trailing byte through the first half.
    var odd: [5]u8 = undefined;
    reorderAndPredict(&odd, &.{ 1, 2, 3, 4, 5 });
    var odd_back: [5]u8 = undefined;
    var odd_scratch = odd;
    unpredictAndReorder(&odd_back, &odd_scratch);
    try testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 5 }, &odd_back);
}

test "rleDecompress follows the wire spec and fails cleanly" {
    var out: [8]u8 = undefined;

    // Repeat run: control 2 -> 3 copies of the next byte.
    try testing.expectEqual(@as(usize, 3), try rleDecompress(out[0..3], &.{ 0x02, 0xAB }));
    try testing.expectEqualSlices(u8, &.{ 0xAB, 0xAB, 0xAB }, out[0..3]);

    // Literal run: control -2 -> the next 2 bytes verbatim.
    try testing.expectEqual(@as(usize, 2), try rleDecompress(out[0..2], &.{ 0xFE, 0x11, 0x22 }));
    try testing.expectEqualSlices(u8, &.{ 0x11, 0x22 }, out[0..2]);

    // Mixed: 1 repeat + 2 literals.
    try testing.expectEqual(@as(usize, 3), try rleDecompress(out[0..3], &.{ 0x00, 0x77, 0xFE, 0x11, 0x22 }));
    try testing.expectEqualSlices(u8, &.{ 0x77, 0x11, 0x22 }, out[0..3]);

    // Overruns and truncations never panic.
    try testing.expectError(error.DecompressionFailed, rleDecompress(out[0..2], &.{ 0x05, 0xAB })); // 6 into 2
    try testing.expectError(error.DecompressionFailed, rleDecompress(&out, &.{ 0xFE, 0x11 })); // literal cut short
    try testing.expectError(error.DecompressionFailed, rleDecompress(&out, &.{0x00})); // repeat without a value
    try testing.expectError(error.DecompressionFailed, rleDecompress(out[0..0], &.{ 0x00, 0xAB })); // 1 into 0

    // The block wrapper rejects size mismatches even for well-formed runs.
    var tmp: [4]u8 = undefined;
    var raw: [4]u8 = undefined;
    try testing.expectError(error.DecompressionFailed, decodeRleBlock(tmp[0..2], raw[0..2], &.{ 0x00, 0x77 }));
}

test "decode reads ZIPS float RGB line by line" {
    const allocator = testing.allocator;
    const r = try floatSamples(allocator, &.{ 1.0, -1.0, 0.5 });
    defer allocator.free(r);
    const g = try floatSamples(allocator, &.{ 0.0, 100.0, -0.0 });
    defer allocator.free(g);
    const b = try floatSamples(allocator, &.{ 3.25, 3.25, 3.25 });
    defer allocator.free(b);
    const file = try (TestExr{
        .width = 1,
        .height = 3,
        .channels = &.{ .{ .name = "R", .pixel_type = pixel_float }, .{ .name = "G", .pixel_type = pixel_float }, .{ .name = "B", .pixel_type = pixel_float } },
        .channel_data = &.{ r, g, b },
        .compression = compression_zips,
    }).build(allocator);
    defer allocator.free(file);

    var dec = try decode(allocator, file);
    defer dec.deinit(allocator);
    try testing.expectEqual(@as(u32, 1), dec.width);
    try testing.expectEqual(@as(u32, 3), dec.height);
    const h = floatToHalfBits;
    try testing.expectEqualSlices(u16, &.{
        h(1.0),  h(0.0),   h(3.25), 0x3C00,
        h(-1.0), h(100.0), h(3.25), 0x3C00,
        h(0.5),  h(-0.0),  h(3.25), 0x3C00,
    }, dec.pixels);
}

test "decode reads ZIP across 16-line blocks with a short tail" {
    const allocator = testing.allocator;
    // 4x20: two blocks (16 + 4 rows). R carries the row index as a half,
    // G/B/A are constant so the deflate stream has real back-references.
    const w = 4;
    const h = 20;
    var r_pat: [w * h]u16 = undefined;
    for (0..h) |y| {
        for (0..w) |x| r_pat[y * w + x] = floatToHalfBits(@as(f32, @floatFromInt(y)) + 0.25 * @as(f32, @floatFromInt(x)));
    }
    const r = try halfSamples(allocator, &r_pat);
    defer allocator.free(r);
    var gba_pat: [w * h]u16 = undefined;
    @memset(&gba_pat, 0);
    for (0..w * h) |i| gba_pat[i] = 0x3C00;
    const g = try halfSamples(allocator, &gba_pat);
    defer allocator.free(g);
    const b = try halfSamples(allocator, &gba_pat);
    defer allocator.free(b);
    const a = try halfSamples(allocator, &gba_pat);
    defer allocator.free(a);
    const file = try (TestExr{
        .width = w,
        .height = h,
        .channel_data = &.{ r, g, b, a },
        .compression = compression_zip,
    }).build(allocator);
    defer allocator.free(file);

    var dec = try decode(allocator, file);
    defer dec.deinit(allocator);
    try testing.expectEqual(@as(u32, w), dec.width);
    try testing.expectEqual(@as(u32, h), dec.height);
    for (0..h) |y| {
        for (0..w) |x| {
            const px = dec.pixels[(y * w + x) * 4 ..][0..4];
            try testing.expectEqual(r_pat[y * w + x], px[0]);
            try testing.expectEqual(@as(u16, 0x3C00), px[1]);
            try testing.expectEqual(@as(u16, 0x3C00), px[2]);
            try testing.expectEqual(@as(u16, 0x3C00), px[3]);
        }
    }
}

test "zlibDecompress inflates exactly and fails cleanly" {
    const allocator = testing.allocator;
    var window: [std.compress.flate.max_window_len]u8 = undefined;
    const raw = "the quick brown fox jumps over the lazy dog, twice! " ** 4;
    // zipDeflateEncode includes the EXR predictor, so inflation yields the
    // predicted bytes (the unpredict step is covered separately + end to end).
    const predicted = try allocator.alloc(u8, raw.len);
    defer allocator.free(predicted);
    reorderAndPredict(predicted, raw);
    const enc = try zipDeflateEncode(allocator, raw);
    defer allocator.free(enc);

    var out: [raw.len]u8 = undefined;
    try zlibDecompress(&out, enc, &window);
    try testing.expectEqualSlices(u8, predicted, &out);

    // Truncated stream, bad header, bad checksum, short/long outputs.
    try testing.expectError(error.DecompressionFailed, blk: {
        var o: [raw.len]u8 = undefined;
        break :blk zlibDecompress(&o, enc[0 .. enc.len - 1], &window);
    });
    var bad_header = try allocator.dupe(u8, enc);
    defer allocator.free(bad_header);
    bad_header[0] ^= 0xFF;
    try testing.expectError(error.DecompressionFailed, blk: {
        var o: [raw.len]u8 = undefined;
        break :blk zlibDecompress(&o, bad_header, &window);
    });
    var bad_adler = try allocator.dupe(u8, enc);
    defer allocator.free(bad_adler);
    bad_adler[bad_adler.len - 1] ^= 0xFF;
    try testing.expectError(error.DecompressionFailed, blk: {
        var o: [raw.len]u8 = undefined;
        break :blk zlibDecompress(&o, bad_adler, &window);
    });
    try testing.expectError(error.DecompressionFailed, blk: {
        var o: [raw.len / 2]u8 = undefined;
        break :blk zlibDecompress(&o, enc, &window);
    });
    try testing.expectError(error.DecompressionFailed, blk: {
        var o: [raw.len * 2]u8 = undefined;
        break :blk zlibDecompress(&o, enc, &window);
    });
}

test "decode rejects corrupt ZIP payloads and misaligned blocks" {
    const allocator = testing.allocator;
    const r = try halfSamples(allocator, &.{ 0x3C00, 0x4000, 0x4200, 0x4400 });
    defer allocator.free(r);
    const corrupt = try (TestExr{ .corrupt_payload = 0, .width = 2, .height = 2, .channels = &.{.{ .name = "R" }}, .channel_data = &.{r}, .compression = compression_zip }).build(allocator);
    defer allocator.free(corrupt);
    try testing.expectError(error.DecompressionFailed, decode(allocator, corrupt));

    const misaligned = try (TestExr{ .misalign_y = true, .width = 2, .height = 2, .channels = &.{.{ .name = "R" }}, .channel_data = &.{r}, .compression = compression_zip }).build(allocator);
    defer allocator.free(misaligned);
    try testing.expectError(error.InvalidScanline, decode(allocator, misaligned));
}

test "Texture.decodeHDRMemory routes EXR payloads to the EXR reader" {
    // Function-local import: production code depends on nothing; only this
    // routing test needs the Texture facade (same pattern as dds.zig).
    const Texture = @import("texture.zig").Texture;
    const allocator = testing.allocator;

    const r = try halfSamples(allocator, &.{ 0x4200, 0xC000 });
    defer allocator.free(r);
    const file = try (TestExr{
        .width = 2,
        .height = 1,
        .channels = &.{.{ .name = "R" }},
        .channel_data = &.{r},
        .compression = compression_zips,
    }).build(allocator);
    defer allocator.free(file);

    // The sniffed route decodes EXR and skips stb entirely: R expands with
    // G/B at zero and alpha at 1.0, ready for the RGBA16F HDR upload.
    var hdr = try Texture.decodeHDRMemory(allocator, file);
    defer hdr.deinit(allocator);
    try testing.expectEqual(@as(u32, 2), hdr.width);
    try testing.expectEqual(@as(u32, 1), hdr.height);
    try testing.expectEqualSlices(u16, &.{ 0x4200, 0x0000, 0x0000, 0x3C00, 0xC000, 0x0000, 0x0000, 0x3C00 }, hdr.pixels);

    // Foreign data still falls through to stb (unchanged behavior)...
    try testing.expectError(error.ImageDecodeFailed, Texture.decodeHDRMemory(allocator, "png data pretending"));
    // ...a corrupt EXR surfaces the reader's own validation error...
    const short = try (TestExr{
        .width = 2,
        .height = 1,
        .channels = &.{.{ .name = "R" }},
        .channel_data = &.{r},
        .trailing_garbage = true,
    }).build(allocator);
    defer allocator.free(short);
    try testing.expectError(error.InvalidScanline, Texture.decodeHDRMemory(allocator, short));
    // ...and the strict entry point names NotExr for foreign data.
    try testing.expectError(error.NotExr, Texture.fromExrMemory(allocator, "png data pretending", .{}));
    // Missing files surface without leaking (GPU upload never starts).
    try testing.expectError(error.FileNotFound, Texture.fromExrFile(allocator, "definitely/missing/file.exr", .{}));
}

test "decode rejects malformed files with explicit errors" {
    const allocator = testing.allocator;
    const r = try halfSamples(allocator, &.{ 0x3C00, 0x3C00 });
    defer allocator.free(r);
    const g = try halfSamples(allocator, &.{ 0x3C00, 0x3C00 });
    defer allocator.free(g);
    const base = TestExr{
        .width = 2,
        .height = 1,
        .channels = &.{ .{ .name = "R" }, .{ .name = "G" } },
        .channel_data = &.{ r, g },
    };
    const full = try base.build(allocator);
    defer allocator.free(full);

    // Foreign data and corrupted magic.
    try testing.expectError(error.NotExr, decode(allocator, "png data pretending to be exr...."));
    const bad_magic = try (TestExr{
        .width = 2,
        .height = 1,
        .channels = &.{ .{ .name = "R" }, .{ .name = "G" } },
        .channel_data = &.{ r, g },
        .corrupt_magic = true,
    }).build(allocator);
    defer allocator.free(bad_magic);
    try testing.expectError(error.NotExr, decode(allocator, bad_magic));

    // Truncations: header, offset table, block headers, pixel data.
    try testing.expectError(error.Truncated, decode(allocator, full[0..20]));
    try testing.expectError(error.Truncated, decode(allocator, full[0 .. full.len - 1]));
    // ...but trailing garbage past the last block is rejected too.
    const garbage = try (TestExr{
        .width = 2,
        .height = 1,
        .channels = &.{ .{ .name = "R" }, .{ .name = "G" } },
        .channel_data = &.{ r, g },
        .trailing_garbage = true,
    }).build(allocator);
    defer allocator.free(garbage);
    try testing.expectError(error.InvalidScanline, decode(allocator, garbage));

    const BadCase = struct {
        name: []const u8,
        spec: TestExr,
        expected: DecodeError,
    };
    const cases = [_]BadCase{
        .{ .name = "bad version", .spec = .{ .width = 2, .height = 1, .channels = &.{ .{ .name = "R" }, .{ .name = "G" } }, .channel_data = &.{ r, g }, .version = 1 }, .expected = error.UnsupportedVersion },
        .{ .name = "tiled flag", .spec = .{ .width = 2, .height = 1, .channels = &.{ .{ .name = "R" }, .{ .name = "G" } }, .channel_data = &.{ r, g }, .version = 2 | 0x200 }, .expected = error.UnsupportedTiled },
        .{ .name = "deep flag", .spec = .{ .width = 2, .height = 1, .channels = &.{ .{ .name = "R" }, .{ .name = "G" } }, .channel_data = &.{ r, g }, .version = 2 | 0x800 }, .expected = error.UnsupportedDeep },
        .{ .name = "multipart flag", .spec = .{ .width = 2, .height = 1, .channels = &.{ .{ .name = "R" }, .{ .name = "G" } }, .channel_data = &.{ r, g }, .version = 2 | 0x1000 }, .expected = error.UnsupportedMultipart },
        .{ .name = "PXR24 compression", .spec = .{ .width = 2, .height = 1, .channels = &.{ .{ .name = "R" }, .{ .name = "G" } }, .channel_data = &.{ r, g }, .compression = 4 }, .expected = error.UnsupportedCompression },
        .{ .name = "unknown compression", .spec = .{ .width = 2, .height = 1, .channels = &.{ .{ .name = "R" }, .{ .name = "G" } }, .channel_data = &.{ r, g }, .compression = 99 }, .expected = error.UnsupportedCompression },
        .{ .name = "UINT channel", .spec = .{ .width = 2, .height = 1, .channels = &.{ .{ .name = "R", .pixel_type = 0 }, .{ .name = "G" } }, .channel_data = &.{ r, g } }, .expected = error.UnsupportedPixelType },
        .{ .name = "subsampled channel", .spec = .{ .width = 2, .height = 1, .channels = &.{ .{ .name = "R", .x_sampling = 2 }, .{ .name = "G" } }, .channel_data = &.{ r, g } }, .expected = error.UnsupportedSubsampling },
        .{ .name = "zero width", .spec = .{ .width = 0, .height = 1, .channels = &.{.{ .name = "R" }}, .channel_data = &.{&.{}} }, .expected = error.UnsupportedDimensions },
        .{ .name = "negative origin", .spec = .{ .width = 2, .height = 1, .xmin = -1, .channels = &.{ .{ .name = "R" }, .{ .name = "G" } }, .channel_data = &.{ r, g } }, .expected = error.UnsupportedDimensions },
        .{ .name = "oversized", .spec = .{ .width = 20000, .height = 1, .channels = &.{.{ .name = "R" }}, .channel_data = &.{&.{}} }, .expected = error.UnsupportedDimensions },
        .{ .name = "missing compression attr", .spec = .{ .width = 2, .height = 1, .channels = &.{ .{ .name = "R" }, .{ .name = "G" } }, .channel_data = &.{ r, g }, .drop_attr = "compression" }, .expected = error.MissingAttribute },
        .{ .name = "duplicate channel", .spec = .{ .width = 2, .height = 1, .channels = &.{ .{ .name = "R" }, .{ .name = "R" } }, .channel_data = &.{ r, r } }, .expected = error.UnsupportedChannels },
        .{ .name = "no color channels", .spec = .{ .width = 2, .height = 1, .channels = &.{.{ .name = "Z", .pixel_type = pixel_float }}, .channel_data = &.{r} }, .expected = error.UnsupportedChannels },
        .{ .name = "bad offset entry", .spec = .{ .width = 2, .height = 1, .channels = &.{ .{ .name = "R" }, .{ .name = "G" } }, .channel_data = &.{ r, g }, .corrupt_offset = 0 }, .expected = error.InvalidScanline },
        .{ .name = "short block size", .spec = .{ .width = 2, .height = 1, .channels = &.{ .{ .name = "R" }, .{ .name = "G" } }, .channel_data = &.{ r, g }, .shrink_block = 0 }, .expected = error.InvalidScanline },
        .{ .name = "duplicate scanline", .spec = .{ .width = 2, .height = 2, .channels = &.{.{ .name = "R" }}, .channel_data = &.{r}, .dupe_y = true }, .expected = error.InvalidScanline },
        .{ .name = "scanline out of range", .spec = .{ .width = 2, .height = 1, .channels = &.{ .{ .name = "R" }, .{ .name = "G" } }, .channel_data = &.{ r, g }, .bad_y = true }, .expected = error.InvalidScanline },
        .{ .name = "bad line order", .spec = .{ .width = 2, .height = 1, .channels = &.{ .{ .name = "R" }, .{ .name = "G" } }, .channel_data = &.{ r, g }, .line_order = 5 }, .expected = error.InvalidHeader },
        .{ .name = "bad aspect", .spec = .{ .width = 2, .height = 1, .channels = &.{ .{ .name = "R" }, .{ .name = "G" } }, .channel_data = &.{ r, g }, .aspect = 0.0 }, .expected = error.InvalidHeader },
    };
    for (cases) |case| {
        const file = try case.spec.build(allocator);
        defer allocator.free(file);
        testing.expectError(case.expected, decode(allocator, file)) catch |e| {
            std.debug.print("case '{s}' failed: {}\n", .{ case.name, e });
            return e;
        };
    }
}
