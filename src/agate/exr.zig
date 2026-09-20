const std = @import("std");

// ---------------------------------------------------------------------------
// OpenEXR scanline reader — the single-part scanline subset, decoded to
// half-float RGBA for the engine's HDR upload path (Texture.RawHdrTexture).
//
// Supports (OpenEXR file layout, all little-endian):
//   - magic 0x76 0x2f 0x31 0x01, format version 2, scanline (non-tiled)
//     single-part files.
//   - header attributes: channels (HALF/FLOAT, sampling 1x1), compression,
//     dataWindow, displayWindow, lineOrder, pixelAspectRatio. Unknown
//     attributes are skipped by size; the header ends at the empty-name
//     terminator. Required attributes must be present with the exact
//     type/size or the file is rejected (MissingAttribute/InvalidHeader).
//   - channels HALF (f16 bits copied verbatim) and FLOAT (f32 converted via
//     a local f32->f16 cast — this module must not import texture.zig, which
//     imports this module for the upload path, same as dds.zig/ktx2.zig).
//     Output is always RGBA: R/G/B/A map to their slots, a lone `Y`
//     luminance channel expands to gray, missing alpha defaults to 1.0.
//     Unknown channel names are decoded and skipped (their bytes still
//     occupy scanline space); UINT pixel types are rejected.
//   - compression NONE, RLE, ZIPS (1 line/block) and ZIP (16 lines/block,
//     zlib/deflate via std.compress.flate). Other codes reject explicitly.
//   - line orders INCREASING_Y/DECREASING_Y/RANDOM_Y: blocks are read
//     sequentially and placed by each block's own y coordinate, so any file
//     order decodes identically. Data-window origins may be non-zero
//     (negative origins are rejected).
//
// Deliberately rejected with explicit errors (never a silent fallback):
//   - tiled files, deep (non-image) data, multipart files,
//   - UINT/unknown pixel types, channel sampling != 1,
//   - PXR24/B44(A)/DWAA/DWAB/unknown compression codes,
//   - zero/negative/oversized (> 16384 per axis) or inconsistent windows,
//   - short reads anywhere (Truncated), duplicate/missing/out-of-range
//     scanlines and trailing garbage (InvalidScanline).
//
// Produces `Decoded` (owned RGBA half-float bits); texture.zig adopts the
// pixels into Texture.RawHdrTexture for fromRawHdr upload. EXR is linear,
// so no sRGB handling exists anywhere on this path.
// ---------------------------------------------------------------------------

/// The 4-byte OpenEXR magic: 0x76 0x2f 0x31 0x01.
pub const magic = [4]u8{ 0x76, 0x2f, 0x31, 0x01 };

/// Largest accepted dimension (matches dds.zig); bigger means corrupt.
const max_dimension: u32 = 16384;
/// Most channels a header may declare; bounds the per-block scratch.
const max_channels: usize = 64;
/// Largest accepted decoded pixel payload (width*height*4 u16s).
const max_output_bytes: usize = 512 * 1024 * 1024;

// Version-field flag bits (OpenEXR ImfVersion.h).
const flag_tiled: u32 = 0x200;
const flag_deep: u32 = 0x800;
const flag_multipart: u32 = 0x1000;

// Compression codes (OpenEXR ImfCompression.h).
const compression_none: u8 = 0;
const compression_rle: u8 = 1;
const compression_zips: u8 = 2;
const compression_zip: u8 = 3;
const compression_last_known: u8 = 8; // DWAB; above this the code is unknown

// Pixel type codes (OpenEXR ImfPixelType.h).
const pixel_uint: u32 = 0;
const pixel_half: u32 = 1;
const pixel_float: u32 = 2;

pub const DecodeError = error{
    NotExr,
    Truncated,
    UnsupportedVersion,
    UnsupportedTiled,
    UnsupportedDeep,
    UnsupportedMultipart,
    UnsupportedCompression,
    UnsupportedPixelType,
    UnsupportedSubsampling,
    UnsupportedChannels,
    MissingAttribute,
    InvalidHeader,
    InvalidScanline,
    DecompressionFailed,
    UnsupportedDimensions,
    ImageTooLarge,
    OutOfMemory,
};

/// True when `bytes` starts with the OpenEXR magic. Cheap guard used to
/// route HDR assets between the EXR reader and stb_image.
pub fn sniff(bytes: []const u8) bool {
    return bytes.len >= magic.len and std.mem.eql(u8, bytes[0..magic.len], &magic);
}

/// Converts one f32 channel to an IEEE-754 half-precision bit pattern.
/// Local copy (same two lines as Texture.floatToHalfBits): this module must
/// not import texture.zig (which imports this module for the upload path).
fn floatToHalfBits(value: f32) u16 {
    return @bitCast(@as(f16, @floatCast(value)));
}

/// Bounds-checked forward reader over the input. Every advance goes through
/// `take`, so short files surface as Truncated instead of panicking.
const Cursor = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn take(self: *Cursor, n: usize) DecodeError![]const u8 {
        if (self.pos > self.bytes.len or n > self.bytes.len - self.pos) return error.Truncated;
        const s = self.bytes[self.pos..][0..n];
        self.pos += n;
        return s;
    }

    fn takeByte(self: *Cursor) DecodeError!u8 {
        return (try self.take(1))[0];
    }

    fn takeU32(self: *Cursor) DecodeError!u32 {
        return std.mem.readInt(u32, (try self.take(4))[0..4], .little);
    }

    fn takeI32(self: *Cursor) DecodeError!i32 {
        return std.mem.readInt(i32, (try self.take(4))[0..4], .little);
    }

    fn takeU64(self: *Cursor) DecodeError!u64 {
        return std.mem.readInt(u64, (try self.take(8))[0..8], .little);
    }

    fn takeF32(self: *Cursor) DecodeError!f32 {
        return @bitCast(try self.takeU32());
    }

    /// Reads a null-terminated name. An empty name (immediate terminator) is
    /// a valid result — the caller decides whether it ends a list. A name
    /// running past `max_len` without a terminator is corrupt; running past
    /// EOF is truncated.
    fn takeName(self: *Cursor, max_len: usize) DecodeError![]const u8 {
        const start = self.pos;
        var end = start;
        while (true) {
            if (end >= self.bytes.len) return error.Truncated;
            if (end - start > max_len) return error.InvalidHeader;
            if (self.bytes[end] == 0) break;
            end += 1;
        }
        self.pos = end + 1;
        return self.bytes[start..end];
    }
};

/// One parsed channel: where its samples land in the RGBA output.
/// `slot` is the RGBA lane (0..3); `gray` fans a Y sample out to R+G+B.
/// Neither set means "decode and skip" (unknown channel name).
const Channel = struct {
    pixel_type: u32, // pixel_half or pixel_float (validated at parse)
    bytes_per_sample: usize, // 2 or 4
    slot: ?u8 = null,
    gray: bool = false,
};

const Header = struct {
    channels: [max_channels]Channel = undefined,
    num_channels: usize = 0,
    compression: u8 = compression_none,
    xmin: i32 = 0,
    ymin: i32 = 0,
    xmax: i32 = -1,
    ymax: i32 = -1,
};

/// Parses the channel list attribute value. Rejects UINT/unknown pixel types
/// and non-1 sampling immediately so the scanline loop only ever sees
/// HALF/FLOAT 1x1 channels. Channel names borrow from `value` and are
/// returned in `names` for the slot-assignment pass.
fn parseChannels(value: []const u8, header: *Header, names: *[max_channels][]const u8) DecodeError!void {
    var cur = Cursor{ .bytes = value };
    var n: usize = 0;
    while (true) {
        const name = try cur.takeName(255);
        if (name.len == 0) break;
        if (n >= max_channels) return error.InvalidHeader;
        const pixel_type = try cur.takeU32();
        _ = try cur.takeByte(); // pLinear: predictor hint, no decode effect
        _ = try cur.take(3); // reserved
        const x_sampling = try cur.takeU32();
        const y_sampling = try cur.takeU32();
        // pixel_uint (0) and unknown codes share one explicit error.
        if (pixel_type != pixel_half and pixel_type != pixel_float) return error.UnsupportedPixelType;
        if (x_sampling != 1 or y_sampling != 1) return error.UnsupportedSubsampling;
        header.channels[n] = .{
            .pixel_type = pixel_type,
            .bytes_per_sample = if (pixel_type == pixel_half) 2 else 4,
        };
        names[n] = name;
        n += 1;
    }
    if (cur.pos != value.len) return error.InvalidHeader;
    if (n == 0) return error.UnsupportedChannels;
    header.num_channels = n;
}

/// Owned RGBA half-float pixels, width*height*4 entries. Pair with
/// Texture.fromRawHdr (texture.zig adopts `.pixels` into RawHdrTexture).
pub const Decoded = struct {
    width: u32 = 0,
    height: u32 = 0,
    pixels: []u16 = &.{},

    pub fn deinit(self: *Decoded, allocator: std.mem.Allocator) void {
        if (self.pixels.len > 0) allocator.free(self.pixels);
        self.* = .{};
    }
};

/// Scanline rows per compressed block. NONE/RLE/ZIPS pack one line per
/// block; ZIP packs 16 (last block shorter).
fn linesPerBlock(compression: u8) DecodeError!u32 {
    return switch (compression) {
        compression_none, compression_rle, compression_zips => 1,
        compression_zip => 16,
        else => error.UnsupportedCompression,
    };
}

/// Full decode of a single-part scanline EXR file into half-float RGBA.
/// GPU-free and thread-safe.
pub fn decode(allocator: std.mem.Allocator, bytes: []const u8) DecodeError!Decoded {
    if (!sniff(bytes)) return error.NotExr;
    var cur = Cursor{ .bytes = bytes, .pos = magic.len };

    const version = try cur.takeU32();
    if (version & 0xff != 2) return error.UnsupportedVersion;
    if (version & flag_tiled != 0) return error.UnsupportedTiled;
    if (version & flag_deep != 0) return error.UnsupportedDeep;
    if (version & flag_multipart != 0) return error.UnsupportedMultipart;

    var header = Header{};
    // Attribute presence tracking: every required attribute must appear
    // exactly once with its exact type/size.
    var seen_channels = false;
    var seen_compression = false;
    var seen_data_window = false;
    var seen_display_window = false;
    var seen_line_order = false;
    var seen_aspect = false;
    // Channel names borrow from the input; collected here for the slot pass.
    var channel_names: [max_channels][]const u8 = undefined;

    while (true) {
        const name = try cur.takeName(255);
        if (name.len == 0) break;
        const typ = try cur.takeName(255);
        const size = try cur.takeU32();
        const value = try cur.take(size);

        if (std.mem.eql(u8, name, "channels")) {
            if (seen_channels or !std.mem.eql(u8, typ, "chlist")) return error.InvalidHeader;
            seen_channels = true;
            try parseChannels(value, &header, &channel_names);
        } else if (std.mem.eql(u8, name, "compression")) {
            if (seen_compression or !std.mem.eql(u8, typ, "compression") or size != 1) return error.InvalidHeader;
            seen_compression = true;
            const code = value[0];
            if (code > compression_last_known) return error.UnsupportedCompression;
            header.compression = code;
        } else if (std.mem.eql(u8, name, "dataWindow")) {
            if (seen_data_window or !std.mem.eql(u8, typ, "box2i") or size != 16) return error.InvalidHeader;
            seen_data_window = true;
            header.xmin = std.mem.readInt(i32, value[0..4], .little);
            header.ymin = std.mem.readInt(i32, value[4..8], .little);
            header.xmax = std.mem.readInt(i32, value[8..12], .little);
            header.ymax = std.mem.readInt(i32, value[12..16], .little);
        } else if (std.mem.eql(u8, name, "displayWindow")) {
            if (seen_display_window or !std.mem.eql(u8, typ, "box2i") or size != 16) return error.InvalidHeader;
            seen_display_window = true;
        } else if (std.mem.eql(u8, name, "lineOrder")) {
            if (seen_line_order or !std.mem.eql(u8, typ, "lineOrder") or size != 1) return error.InvalidHeader;
            seen_line_order = true;
            if (value[0] > 2) return error.InvalidHeader;
        } else if (std.mem.eql(u8, name, "pixelAspectRatio")) {
            if (seen_aspect or !std.mem.eql(u8, typ, "float") or size != 4) return error.InvalidHeader;
            seen_aspect = true;
            const aspect: f32 = @bitCast(std.mem.readInt(u32, value[0..4], .little));
            if (!(aspect > 0) or std.math.isInf(aspect)) return error.InvalidHeader;
        }
        // Unknown attributes (screenWindowCenter, comments, ...) are skipped
        // by size — already bounds-checked by `take`.
    }
    if (!seen_channels or !seen_compression or !seen_data_window or
        !seen_display_window or !seen_line_order or !seen_aspect)
    {
        return error.MissingAttribute;
    }

    // Slot assignment: RGB(A) wins when present; otherwise a lone Y expands
    // to gray. Duplicate mappings (two Rs, R+Y with R present claiming...)
    // are ambiguous and rejected. Unknown names stay slot-less (skipped).
    var has_rgb = false;
    for (channel_names[0..header.num_channels]) |cname| {
        if (std.mem.eql(u8, cname, "R") or std.mem.eql(u8, cname, "G") or std.mem.eql(u8, cname, "B")) {
            has_rgb = true;
        }
    }
    var has_color = false;
    var used_slots: [4]bool = .{ false, false, false, false };
    for (channel_names[0..header.num_channels], 0..) |cname, i| {
        const slot: ?u8 = if (std.mem.eql(u8, cname, "R"))
            0
        else if (std.mem.eql(u8, cname, "G"))
            1
        else if (std.mem.eql(u8, cname, "B"))
            2
        else if (std.mem.eql(u8, cname, "A"))
            3
        else
            null;
        if (slot) |s| {
            if (used_slots[s]) return error.UnsupportedChannels;
            used_slots[s] = true;
            header.channels[i].slot = s;
            has_color = true;
        } else if (!has_rgb and std.mem.eql(u8, cname, "Y")) {
            if (has_color) return error.UnsupportedChannels; // two Y channels
            header.channels[i].gray = true;
            has_color = true;
        }
    }
    if (!has_color) return error.UnsupportedChannels;

    // Data-window dimensions: non-zero origins allowed, negative rejected.
    if (header.xmin < 0 or header.ymin < 0) return error.UnsupportedDimensions;
    const w64: i64 = @as(i64, header.xmax) - header.xmin + 1;
    const h64: i64 = @as(i64, header.ymax) - header.ymin + 1;
    if (w64 < 1 or h64 < 1) return error.UnsupportedDimensions;
    if (w64 > max_dimension or h64 > max_dimension) return error.UnsupportedDimensions;
    const width: u32 = @intCast(w64);
    const height: u32 = @intCast(h64);

    var scanline_bytes: usize = 0;
    for (header.channels[0..header.num_channels]) |ch| {
        scanline_bytes = std.math.add(usize, scanline_bytes, std.math.mul(usize, width, ch.bytes_per_sample) catch return error.ImageTooLarge) catch return error.ImageTooLarge;
    }
    if (scanline_bytes == 0) return error.InvalidHeader;

    const pixel_count: usize = @as(usize, width) * height;
    if (pixel_count * 8 > max_output_bytes) return error.ImageTooLarge;

    const block_lines = try linesPerBlock(header.compression);
    const num_blocks: usize = (height + block_lines - 1) / block_lines;

    // Chunk offset table: one u64 per scanline block. Entries are
    // sanity-checked (inside the file, past the table); blocks themselves
    // are read sequentially and placed by their own y headers.
    const table_start = cur.pos;
    _ = try cur.take(std.math.mul(usize, num_blocks, 8) catch return error.ImageTooLarge);
    const table_end = cur.pos;
    for (0..num_blocks) |i| {
        const off = std.mem.readInt(u64, bytes[table_start + i * 8 ..][0..8], .little);
        if (off < table_end or off >= bytes.len) return error.InvalidScanline;
        if (bytes.len - @as(usize, @intCast(off)) < 8) return error.InvalidScanline;
    }

    var out = Decoded{ .width = width, .height = height };
    out.pixels = try allocator.alloc(u16, pixel_count * 4);
    errdefer allocator.free(out.pixels);
    @memset(out.pixels, 0);
    // Missing color lanes stay 0; missing alpha defaults to 1.0.
    var p: usize = 3;
    while (p < out.pixels.len) : (p += 4) out.pixels[p] = 0x3C00;

    const seen = try allocator.alloc(bool, height);
    defer allocator.free(seen);
    @memset(seen, false);
    var remaining: usize = height;

    // Compressed blocks decode through reusable scratch buffers sized for
    // the largest block of this file (channels are capped at 64 and each
    // axis at 16384, so this stays bounded; allocation failure is a clean
    // OutOfMemory, never a panic). The zlib window is needed only for the
    // deflate compressions.
    const max_block_bytes = std.math.mul(usize, block_lines, scanline_bytes) catch return error.ImageTooLarge;
    var tmp: []u8 = &.{};
    var raw: []u8 = &.{};
    var window: []u8 = &.{};
    if (header.compression != compression_none) {
        tmp = try allocator.alloc(u8, max_block_bytes);
        errdefer allocator.free(tmp);
        raw = try allocator.alloc(u8, max_block_bytes);
        errdefer allocator.free(raw);
        if (header.compression == compression_zips or header.compression == compression_zip) {
            window = try allocator.alloc(u8, std.compress.flate.max_window_len);
            errdefer allocator.free(window);
        }
    }
    defer {
        if (tmp.len > 0) allocator.free(tmp);
        if (raw.len > 0) allocator.free(raw);
        if (window.len > 0) allocator.free(window);
    }

    for (0..num_blocks) |_| {
        const y = try cur.takeI32();
        const size = try cur.takeU32();
        const data = try cur.take(size);
        if (y < header.ymin or y > header.ymax) return error.InvalidScanline;
        const row: usize = @intCast(y - header.ymin);
        if (seen[row]) return error.InvalidScanline;
        if (row % block_lines != 0) return error.InvalidScanline;
        const rows_in_block: usize = @min(block_lines, height - row);
        seen[row] = true;
        // Multi-line blocks mark every covered row (a repeated block start
        // is still caught by the `seen` check above).
        for (1..rows_in_block) |li| {
            if (seen[row + li]) return error.InvalidScanline;
            seen[row + li] = true;
        }
        remaining -= rows_in_block;

        switch (header.compression) {
            compression_none => {
                if (data.len != scanline_bytes) return error.InvalidScanline;
                decodeScanline(&header, data, width, row, out.pixels);
            },
            compression_rle => {
                const unpacked_len = rows_in_block * scanline_bytes;
                try decodeRleBlock(tmp[0..unpacked_len], raw[0..unpacked_len], data);
                for (0..rows_in_block) |li| {
                    decodeScanline(&header, raw[li * scanline_bytes ..][0..scanline_bytes], width, row + li, out.pixels);
                }
            },
            compression_zips, compression_zip => {
                const unpacked_len = rows_in_block * scanline_bytes;
                try decodeZipBlock(tmp[0..unpacked_len], raw[0..unpacked_len], data, window);
                for (0..rows_in_block) |li| {
                    decodeScanline(&header, raw[li * scanline_bytes ..][0..scanline_bytes], width, row + li, out.pixels);
                }
            },
            else => return error.UnsupportedCompression,
        }
    }
    if (remaining != 0) return error.InvalidScanline;
    if (cur.pos != bytes.len) return error.InvalidScanline; // trailing garbage
    return out;
}

/// Decodes one RLE block: run-decode `packed` into `tmp`, undo the
/// predictor/interleave into `raw` (both sized `unpacked_len` by the caller).
fn decodeRleBlock(tmp: []u8, raw: []u8, compressed: []const u8) DecodeError!void {
    const got = try rleDecompress(tmp, compressed);
    if (got != tmp.len) return error.DecompressionFailed;
    unpredictAndReorder(raw, tmp);
}

/// Decodes one ZIP/ZIPS block: zlib-inflate `compressed` into `tmp`, undo
/// the predictor/interleave into `raw` (both sized `unpacked_len` by the
/// caller; `window` is the 64 KiB inflate history).
fn decodeZipBlock(tmp: []u8, raw: []u8, compressed: []const u8, window: []u8) DecodeError!void {
    try zlibDecompress(tmp, compressed, window);
    unpredictAndReorder(raw, tmp);
}

/// Inflates exactly `out.len` bytes of zlib data into `out`. Short streams,
/// bad headers/checksums and trailing bytes are all DecompressionFailed —
/// the caller sizes `out` to the block's exact unpacked length, so anything
/// else means corruption.
fn zlibDecompress(out: []u8, compressed: []const u8, window: []u8) DecodeError!void {
    std.debug.assert(window.len >= std.compress.flate.max_window_len);
    var in: std.Io.Reader = .fixed(compressed);
    var decomp = std.compress.flate.Decompress.init(&in, .zlib, window);
    decomp.reader.readSliceAll(out) catch return error.DecompressionFailed;
    // The stream must end exactly here: a further byte means trailing data,
    // while EndOfStream confirms the footer was reached. Any protocol
    // failure (bad header, truncated footer, ...) surfaces as a read error.
    if (decomp.reader.takeByte()) |_| {
        return error.DecompressionFailed;
    } else |e| {
        if (e != error.EndOfStream) return error.DecompressionFailed;
    }
    // The inflate reader reports the zlib adler32 without checking it (like
    // std's own harness, the caller owns verification): a mismatch means
    // the block is corrupt even though its length was exact.
    var hasher: std.hash.Adler32 = .{};
    hasher.update(out);
    if (hasher.adler != decomp.container_metadata.zlib.adler) return error.DecompressionFailed;
}
/// Run-decodes OpenEXR RLE data (exact port of openexr's
/// internal_rle_decompress): a signed control byte starts each run —
/// negative = that many literal bytes follow, non-negative = repeat the
/// next byte (control + 1) times. Returns the unpacked length; overruns of
/// either side are DecompressionFailed, never panics.
fn rleDecompress(out: []u8, compressed: []const u8) DecodeError!usize {
    var inp: usize = 0;
    var outp: usize = 0;
    while (inp < compressed.len) {
        const control: i8 = @bitCast(compressed[inp]);
        inp += 1;
        if (control < 0) {
            const count: usize = @intCast(-@as(i16, control));
            if (inp + count > compressed.len) return error.DecompressionFailed;
            if (outp + count > out.len) return error.DecompressionFailed;
            @memcpy(out[outp..][0..count], compressed[inp..][0..count]);
            inp += count;
            outp += count;
        } else {
            if (inp + 1 > compressed.len) return error.DecompressionFailed;
            const count: usize = @as(usize, @intCast(control)) + 1;
            if (outp + count > out.len) return error.DecompressionFailed;
            @memset(out[outp..][0..count], compressed[inp]);
            inp += 1;
            outp += count;
        }
    }
    return outp;
}

/// Undoes the OpenEXR predictor + interleave (exact port of openexr's
/// unpredict_and_reorder): first the byte differences are reintegrated over
/// the whole buffer (`t[i] += t[i-1] - 128`), then even/odd halves are
/// de-interleaved. `out` and `scratch` must be distinct equal-length buffers.
fn unpredictAndReorder(out: []u8, scratch: []u8) void {
    std.debug.assert(out.len == scratch.len);
    for (1..scratch.len) |i| scratch[i] = scratch[i - 1] +% scratch[i] -% 128;
    const half = (scratch.len + 1) / 2;
    var t1: usize = 0;
    var t2: usize = half;
    var s: usize = 0;
    while (s < out.len) {
        out[s] = scratch[t1];
        t1 += 1;
        s += 1;
        if (s < out.len) {
            out[s] = scratch[t2];
            t2 += 1;
            s += 1;
        }
    }
}

/// Decodes one scanline: channels are stored contiguously
/// (channel-major), each `width` samples of its own sample size.
fn decodeScanline(header: *const Header, data: []const u8, width: u32, row: usize, pixels: []u16) void {
    var cursor: usize = 0;
    for (header.channels[0..header.num_channels]) |ch| {
        for (0..width) |x| {
            const dst = (row * @as(usize, width) + x) * 4;
            if (ch.pixel_type == pixel_half) {
                const bits = std.mem.readInt(u16, data[cursor..][0..2], .little);
                cursor += 2;
                if (ch.slot) |s| {
                    pixels[dst + s] = bits;
                } else if (ch.gray) {
                    pixels[dst + 0] = bits;
                    pixels[dst + 1] = bits;
                    pixels[dst + 2] = bits;
                }
            } else {
                const f: f32 = @bitCast(std.mem.readInt(u32, data[cursor..][0..4], .little));
                cursor += 4;
                const bits = floatToHalfBits(f);
                if (ch.slot) |s| {
                    pixels[dst + s] = bits;
                } else if (ch.gray) {
                    pixels[dst + 0] = bits;
                    pixels[dst + 1] = bits;
                    pixels[dst + 2] = bits;
                }
            }
        }
    }
}

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
