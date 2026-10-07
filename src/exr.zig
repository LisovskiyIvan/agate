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
pub const compression_none: u8 = 0;
pub const compression_rle: u8 = 1;
pub const compression_zips: u8 = 2;
pub const compression_zip: u8 = 3;
const compression_last_known: u8 = 8; // DWAB; above this the code is unknown

// Pixel type codes (OpenEXR ImfPixelType.h).
pub const pixel_uint: u32 = 0;
pub const pixel_half: u32 = 1;
pub const pixel_float: u32 = 2;

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
pub fn floatToHalfBits(value: f32) u16 {
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
pub fn decodeRleBlock(tmp: []u8, raw: []u8, compressed: []const u8) DecodeError!void {
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
pub fn zlibDecompress(out: []u8, compressed: []const u8, window: []u8) DecodeError!void {
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
pub fn rleDecompress(out: []u8, compressed: []const u8) DecodeError!usize {
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
pub fn unpredictAndReorder(out: []u8, scratch: []u8) void {
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
