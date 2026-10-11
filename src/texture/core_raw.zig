//! CPU-side RGBA8 pixels and mip-chain builders. Split out of `core.zig` (facade).
//!
//! `RawTexture` plus the copy/convert entry points (`buildRaw`,
//! `buildRawSrgbFused`), the single-level downsample/mip-count helpers, and
//! their file-private workers. Decoding and mip generation stay GPU-free, so
//! worker threads can build here and upload later via `fromRaw`. Leaf:
//! `std` + `color` + `mip` only.

const std = @import("std");
const color = @import("color.zig");
const mip = @import("mip.zig");

/// Box-filter downsample of one RGBA8 level. Dims floor at 1, source coords
/// clamp at edges (handles NPOT). Thin wrapper over the shared
/// file-private `boxDownsampleU8` so 2D and cube paths stay in sync.
pub fn downsampleLevel(src: []const u8, src_w: u32, src_h: u32, dst: []u8, dst_w: u32, dst_h: u32) void {
    mip.boxDownsampleU8(src, src_w, src_h, dst, dst_w, dst_h);
}
pub fn mipLevelCount(width: u32, height: u32) u32 {
    var levels: u32 = 1;
    var w = width;
    var h = height;
    while ((w > 1 or h > 1) and levels < 16) {
        w = @max(1, w / 2);
        h = @max(1, h / 2);
        levels += 1;
    }
    return levels;
}
/// CPU-decoded RGBA pixels plus an optional mip chain. Decoding and mip
/// generation are GPU-free, so they can run on worker threads; the
/// sg.Image must be created later with `fromRaw` on the main thread.
pub const RawTexture = struct {
    width: u32 = 0,
    height: u32 = 0,
    num_levels: u32 = 0,
    is_srgb: bool = false,
    levels: [16]?[]u8 = @splat(null),

    pub fn deinit(self: *RawTexture, allocator: std.mem.Allocator) void {
        for (self.levels[0..self.num_levels]) |level| {
            if (level) |buf| allocator.free(buf);
        }
        self.* = .{};
    }
};
/// Copies `rgba_pixels` and optionally builds the box-filtered mip chain.
/// Public so the KTX2 reader (ktx2.zig) can reuse the generator for
/// single-level files; the decode paths above use it internally.
/// L0 fast copy: plain `@memcpy` (no per-byte modulo/divide).
pub fn buildRaw(allocator: std.mem.Allocator, width: u32, height: u32, rgba_pixels: []const u8, gen_mipmaps: bool) !RawTexture {
    return buildRawImpl(allocator, width, height, rgba_pixels, gen_mipmaps, false);
}
fn buildRawImpl(allocator: std.mem.Allocator, width: u32, height: u32, rgba_pixels: []const u8, gen_mipmaps: bool, comptime srgb: bool) !RawTexture {
    // Guard the public entry point: the mip chain treats the buffer as
    // width*height*4, and downsampleLevel runs unsafely, so a short
    // buffer would cause out-of-bounds reads. Checked arithmetic keeps
    // huge dimensions from wrapping to a small expected size.
    if (width == 0 or height == 0) return error.InvalidDimensions;
    const pixel_count = std.math.mul(u32, width, height) catch return error.ImageTooLarge;
    const expected_bytes = std.math.mul(u32, pixel_count, 4) catch return error.ImageTooLarge;
    if (rgba_pixels.len != expected_bytes) return error.InvalidDimensions;

    var raw = RawTexture{ .width = width, .height = height, .num_levels = 1 };
    errdefer raw.deinit(allocator);

    const level0 = try allocator.alloc(u8, rgba_pixels.len);
    if (srgb) copySrgbToLinearStride(level0, rgba_pixels) else @memcpy(level0, rgba_pixels);
    raw.levels[0] = level0;

    if (gen_mipmaps) {
        var prev_w = width;
        var prev_h = height;
        var prev: []const u8 = level0;
        const levels = mipLevelCount(width, height);
        for (1..levels) |m| {
            const cur_w: u32 = @max(1, prev_w / 2);
            const cur_h: u32 = @max(1, prev_h / 2);
            const cur = try allocator.alloc(u8, @as(usize, cur_w) * cur_h * 4);
            raw.levels[m] = cur;
            downsampleLevel(prev, prev_w, prev_h, cur, cur_w, cur_h);
            raw.num_levels = @intCast(m + 1);
            prev_w = cur_w;
            prev_h = cur_h;
            prev = cur;
        }
    }
    return raw;
}
/// Fused L0 copy + sRGB->linear for the stb decode path: one pass over
/// level 0 instead of convert-in-place + memcpy (two passes). Output is
/// bit-identical to `convertSrgbToLinearInPlace` + `buildRaw`: same
/// `color.srgbToLinearU8` LUT on RGB lanes, alpha untouched, same
/// box-filter chain afterwards. Stride loop (no `i % 4` per byte).
pub fn buildRawSrgbFused(allocator: std.mem.Allocator, width: u32, height: u32, rgba_pixels: []const u8, gen_mipmaps: bool) !RawTexture {
    return buildRawImpl(allocator, width, height, rgba_pixels, gen_mipmaps, true);
}
/// Stride-4 sRGB->linear copy: 3 LUT lookups per pixel, alpha copied
/// verbatim. Bit-identical to `color.convertSrgbToLinearInPlace` followed
/// by `@memcpy`; avoids the old per-byte `i % 4` branch.
fn copySrgbToLinearStride(dst: []u8, src: []const u8) void {
    std.debug.assert(dst.len == src.len);
    const pixels = dst.len / 4;
    var i: usize = 0;
    while (i < pixels) : (i += 1) {
        const o = i * 4;
        dst[o + 0] = color.srgbToLinearU8(src[o + 0]);
        dst[o + 1] = color.srgbToLinearU8(src[o + 1]);
        dst[o + 2] = color.srgbToLinearU8(src[o + 2]);
        dst[o + 3] = src[o + 3];
    }
}
