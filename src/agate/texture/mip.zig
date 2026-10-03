//! Mip/downsample/size helpers. Split out of `texture.zig` (facade).
//!
//! `boxDownsampleF16` moved here from `CubeTexture` (same body, dedented to
//! a free function; the half-float conversions now come from `color.zig`).
//! Everything is `pub` for sibling leaves but deliberately NOT re-exported
//! by the facade, except `pixelFormatBytes`, which was always public —
//! except that one, the public surface matches the pre-split file.
const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const color = @import("color.zig");

/// Shared RGBA8 box-filter downsample of one mip level. Dims floor at 1,
/// source coords clamp at edges (handles NPOT). Fast path for the exact
/// 2:1 case keeps power-of-two results bit-identical to the old inline
/// loops; the generic path covers odd/NPOT sizes.
/// Precondition: src.len >= src_w*src_h*4, dst.len >= dst_w*dst_h*4.
pub fn boxDownsampleU8(src: []const u8, src_w: u32, src_h: u32, dst: []u8, dst_w: u32, dst_h: u32) void {
    @setRuntimeSafety(false);
    if (src_w == dst_w * 2 and src_h == dst_h * 2) {
        const src_stride: usize = @as(usize, src_w) * 4;
        const dst_stride: usize = @as(usize, dst_w) * 4;
        var y: usize = 0;
        while (y < dst_h) : (y += 1) {
            const r0 = src[(y * 2) * src_stride ..];
            const r1 = src[(y * 2 + 1) * src_stride ..];
            const out_row = dst[y * dst_stride ..];
            var x: usize = 0;
            while (x < dst_w) : (x += 1) {
                const si = x * 8;
                const di = x * 4;
                inline for (0..4) |ch| {
                    const sum: u32 = @as(u32, r0[si + ch]) +
                        @as(u32, r0[si + 4 + ch]) +
                        @as(u32, r1[si + ch]) +
                        @as(u32, r1[si + 4 + ch]);
                    out_row[di + ch] = @intCast((sum + 2) >> 2);
                }
            }
        }
        return;
    }

    var y: u32 = 0;
    while (y < dst_h) : (y += 1) {
        const sy0 = @min(y * 2, src_h - 1);
        const sy1 = @min(y * 2 + 1, src_h - 1);
        // Hoisted row bases: identical taps/offsets as before, fewer
        // per-pixel multiplies on the NPOT tail path.
        const row0: usize = @as(usize, sy0) * @as(usize, src_w) * 4;
        const row1: usize = @as(usize, sy1) * @as(usize, src_w) * 4;
        const dst_row: usize = @as(usize, y) * @as(usize, dst_w) * 4;
        var x: u32 = 0;
        while (x < dst_w) : (x += 1) {
            const sx0 = @min(x * 2, src_w - 1);
            const sx1 = @min(x * 2 + 1, src_w - 1);
            const q00 = row0 + @as(usize, sx0) * 4;
            const q10 = row0 + @as(usize, sx1) * 4;
            const q01 = row1 + @as(usize, sx0) * 4;
            const q11 = row1 + @as(usize, sx1) * 4;
            const o = dst_row + @as(usize, x) * 4;
            inline for (0..4) |ch| {
                const sum: u32 = @as(u32, src[q00 + ch]) + @as(u32, src[q10 + ch]) + @as(u32, src[q01 + ch]) + @as(u32, src[q11 + ch]);
                dst[o + ch] = @intCast((sum + 2) >> 2);
            }
        }
    }
}

/// One RGBA8-style box-filter mip step over RGBA16F (half-float) texels.
/// Averages in f32 (decode via halfBitsToFloat, encode via
/// floatToHalfBits); dims floor at 1, source coords clamp at edges —
/// same contract as `boxDownsampleU8`.
pub fn boxDownsampleF16(src: []const u16, src_w: u32, src_h: u32, dst: []u16, dst_w: u32, dst_h: u32) void {
    var y: u32 = 0;
    while (y < dst_h) : (y += 1) {
        const sy0 = @min(y * 2, src_h - 1);
        const sy1 = @min(y * 2 + 1, src_h - 1);
        var x: u32 = 0;
        while (x < dst_w) : (x += 1) {
            const sx0 = @min(x * 2, src_w - 1);
            const sx1 = @min(x * 2 + 1, src_w - 1);
            const q00 = (sy0 * src_w + sx0) * 4;
            const q10 = (sy0 * src_w + sx1) * 4;
            const q01 = (sy1 * src_w + sx0) * 4;
            const q11 = (sy1 * src_w + sx1) * 4;
            const o = (y * dst_w + x) * 4;
            inline for (0..4) |ch| {
                const sum = color.halfBitsToFloat(src[q00 + ch]) +
                    color.halfBitsToFloat(src[q10 + ch]) +
                    color.halfBitsToFloat(src[q01 + ch]) +
                    color.halfBitsToFloat(src[q11 + ch]);
                dst[o + ch] = color.floatToHalfBits(sum * 0.25);
            }
        }
    }
}

/// Checked RGBA8 face size (size*size*4). The old u32 `size * size * 4`
/// wrapped to a small value for large sizes, causing undersized allocations
/// and OOB writes. Uses the file's ImageTooLarge/InvalidDimensions
/// conventions; returns usize for direct use as an alloc length.
pub fn checkedFaceBytes(size: u32) !usize {
    if (size == 0) return error.InvalidDimensions;
    const pixels = std.math.mul(u32, size, size) catch return error.ImageTooLarge;
    const bytes = std.math.mul(u32, pixels, 4) catch return error.ImageTooLarge;
    return @as(usize, bytes);
}

/// Approximate bytes per pixel for a Sokol pixel format.
pub fn pixelFormatBytes(format: sg.PixelFormat) usize {
    return switch (format) {
        .R8, .R8UI, .R8SI, .R8SN => 1,
        .R16F, .R16UI, .R16SI, .R16, .R16SN, .RG8, .RG8UI, .RG8SI, .RG8SN => 2,
        .RGBA8, .BGRA8, .RGBA8UI, .RGBA8SI, .RGBA8SN, .RG16F, .RG16UI, .RG16SI, .RG16, .RG16SN, .R32F, .R32UI, .R32SI, .DEPTH, .DEPTH_STENCIL => 4,
        .RGBA16F, .RGBA16UI, .RGBA16SI, .RGBA16, .RGBA16SN, .RG32F, .RG32UI, .RG32SI => 8,
        .RGBA32F, .RGBA32UI, .RGBA32SI => 16,
        .BC1_RGBA, .BC4_R, .BC4_RSN, .ETC2_RGB8, .ETC2_RGB8A1 => 1,
        .BC2_RGBA, .BC3_RGBA, .BC3_SRGBA, .BC5_RG, .BC5_RGSN, .BC6H_RGBF, .BC6H_RGBUF, .BC7_RGBA, .ETC2_RGBA8 => 1,
        else => 4,
    };
}

test "checkedFaceBytes rejects empty and overflowing sizes" {
    try std.testing.expectError(error.InvalidDimensions, checkedFaceBytes(0));
    // 100000^2 overflows u32: old `size * size * 4` wrapped to a small
    // alloc size; now ImageTooLarge before any allocation or GPU upload.
    try std.testing.expectError(error.ImageTooLarge, checkedFaceBytes(100000));
    try std.testing.expectEqual(@as(usize, 2 * 2 * 4), try checkedFaceBytes(2));
}

test {
    // sRGB/linear mip-contract suites (decode -> convert -> box filter)
    // live in a sibling file; this file is the test-root import target.
    _ = @import("mip_tests.zig");
}
