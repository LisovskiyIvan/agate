//! HDR CPU decode (Radiance .hdr / OpenEXR to half-float). Split out of `core.zig` (facade).
//!
//! `RawHdrTexture` plus the f32->f16 builder, the sniff-routed decode entries
//! (`decodeHDRMemory`/`decodeHDRFile`), the half-float bit forwarders, and
//! their file-private workers. Thread-safe; pair with `fromRawHdr` on the
//! main thread. Leaf: `std` + `c` + `exr` + `color` only.

const std = @import("std");
const c = @import("../c.zig").c;
const exr = @import("../exr.zig");
const color = @import("color.zig");

/// Converts one f32 channel to an IEEE-754 half-precision bit pattern.
/// Out-of-range magnitudes become half infinity, NaN stays NaN.
/// Content above 65504 loses detail: tone-map before upload if it matters.
pub fn floatToHalfBits(value: f32) u16 {
    return color.floatToHalfBits(value);
}

/// Converts an IEEE-754 half-precision bit pattern back to f32.
/// Used by tests and debugging; the GPU upload path never needs it.
pub fn halfBitsToFloat(bits: u16) f32 {
    return color.halfBitsToFloat(bits);
}
/// CPU-decoded HDR pixels in half-float RGBA. Decoding and the f32->f16
/// conversion are GPU-free, so they can run on worker threads; the
/// sg.Image must be created later with `fromRawHdr` on the main thread.
/// Single mip level only: HDR images upload without mipmaps (linear
/// min/mag, clamp wrap, no gamma correction).
pub const RawHdrTexture = struct {
    width: u32 = 0,
    height: u32 = 0,
    /// RGBA half-float bit patterns, width*height*4 entries.
    pixels: []u16 = &.{},

    pub fn deinit(self: *RawHdrTexture, allocator: std.mem.Allocator) void {
        if (self.pixels.len > 0) allocator.free(self.pixels);
        self.* = .{};
    }
};
/// Copies f32 RGBA samples and converts them to half-float bits.
/// Rejects empty dimensions and short sample buffers.
fn buildRawHdr(allocator: std.mem.Allocator, width: u32, height: u32, samples: []const f32) !RawHdrTexture {
    const pixel_count = std.math.mul(usize, width, height) catch return error.ImageTooLarge;
    const channel_count = std.math.mul(usize, pixel_count, 4) catch return error.ImageTooLarge;
    if (channel_count == 0 or samples.len < channel_count) return error.InvalidDimensions;

    const pixels = try allocator.alloc(u16, channel_count);
    errdefer allocator.free(pixels);
    for (samples[0..channel_count], 0..) |sample, i| {
        pixels[i] = floatToHalfBits(sample);
    }
    return .{ .width = width, .height = height, .pixels = pixels };
}
/// Decodes an image to half-float RGBA: OpenEXR (magic sniff) via the
/// engine's scanline reader, Radiance .hdr (and LDR formats upconverted
/// to float) via stbi_loadf. Thread-safe; pair with `fromRawHdr`.
/// Foreign or corrupt data yields error.ImageDecodeFailed, empty sizes
/// error.InvalidDimensions. EXR validation errors surface as their own
/// failures (never a silent stb fallback).
pub fn decodeHDRMemory(allocator: std.mem.Allocator, bytes: []const u8) !RawHdrTexture {
    if (exr.sniff(bytes)) {
        const dec = try exr.decode(allocator, bytes);
        // Same layout (RGBA half-float bits); adopt without copying.
        return .{ .width = dec.width, .height = dec.height, .pixels = dec.pixels };
    }
    if (bytes.len == 0) return error.ImageDecodeFailed;
    if (bytes.len > std.math.maxInt(c_int)) return error.ImageTooLarge;

    var w: c_int = 0;
    var h: c_int = 0;
    var channels_in_file: c_int = 0;

    const data = c.stbi_loadf_from_memory(
        bytes.ptr,
        @intCast(bytes.len),
        &w,
        &h,
        &channels_in_file,
        4,
    );
    if (data == null) return error.ImageDecodeFailed;
    defer c.stbi_image_free(data);
    if (w <= 0 or h <= 0) return error.InvalidDimensions;

    const width: u32 = @intCast(w);
    const height: u32 = @intCast(h);
    const channel_count = std.math.mul(usize, std.math.mul(usize, width, height) catch return error.ImageTooLarge, 4) catch return error.ImageTooLarge;
    // stb_image's flat Radiance path ignores stbi__getn's result and
    // zero-fills truncated pixel data, so short payloads are rejected
    // here: after the resolution line a flat file needs exactly
    // width*height RGBE pixels. RLE payloads are variable-length, but
    // stb errors on bad RLE runs, so they need no extra check.
    if (width < 8 or width >= 32768) {
        if (radiancePayload(bytes)) |payload| {
            if (payload.len < channel_count) return error.ImageDecodeFailed;
        }
    }
    return buildRawHdr(allocator, width, height, data[0..channel_count]);
}
/// Returns the pixel-data slice after the Radiance resolution line
/// ("-Y h +X w"), or null when the header has no such line. The
/// resolution line is the first line after the header's empty separator
/// line; scanning beyond it could hit binary payload bytes.
fn radiancePayload(bytes: []const u8) ?[]const u8 {
    var pos: usize = 0;
    while (pos < bytes.len) {
        const nl = std.mem.indexOfScalarPos(u8, bytes, pos, '\n') orelse return null;
        if (nl == pos) {
            pos = nl + 1;
            break;
        }
        pos = nl + 1;
    } else return null;
    const nl = std.mem.indexOfScalarPos(u8, bytes, pos, '\n') orelse return null;
    if (!std.mem.startsWith(u8, bytes[pos..nl], "-Y ")) return null;
    return bytes[nl + 1 ..];
}
/// File variant of `decodeHDRMemory`. Thread-safe; pair with `fromRawHdr`.
/// Buffered read so EXR files take the same sniff-routed reader as
/// in-memory payloads (stb decodes the same bytes from memory).
pub fn decodeHDRFile(allocator: std.mem.Allocator, file_path: []const u8) !RawHdrTexture {
    const io = std.Io.Threaded.global_single_threaded.io();
    // Missing/unreadable files keep the historical ImageDecodeFailed
    // (same observable contract as the old stbi_loadf path).
    const file = std.Io.Dir.cwd().openFile(io, file_path, .{}) catch return error.ImageDecodeFailed;
    defer file.close(io);

    const file_size = try file.length(io);
    const bytes = try allocator.alloc(u8, std.math.cast(usize, file_size) orelse return error.ImageTooLarge);
    defer allocator.free(bytes);

    const read = try file.readPositionalAll(io, bytes, 0);
    if (read < bytes.len) return error.ImageDecodeFailed;
    return decodeHDRMemory(allocator, bytes);
}
// Minimal 1x1 Radiance .hdr with flat (non-RLE) RGBE data. Width < 8 takes
// the flat decode path in stb_image. RGBE {128,128,128,128} decodes to
// 128 * 2^(128-136) = 0.5 per channel, alpha forced to 1.0.
const hdr_1x1_flat: []const u8 = "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n\x80\x80\x80\x80";
// Minimal 8x1 Radiance .hdr exercising the RLE scanline path: scanline
// header {2, 2, 0, 8} (width 8), then each of the 4 channels as one run of
// 8 pixels with value 128 (count byte 128+8 = 0x88). Same 0.5 gray texels.
const hdr_8x1_rle: []const u8 = "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 8\n\x02\x02\x00\x08\x88\x80\x88\x80\x88\x80\x88\x80";
