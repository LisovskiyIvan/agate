//! sRGB-aware mip-generation tests: pin that the RGBA8 decode path builds
//! mip levels in LINEAR space for sRGB color textures (decode -> box average
//! -> store linear), matching what a GLES3/WebGL2 GPU does for
//! `generateMipmap` on an sRGB internal format (filter after decode).
//!
//! agate's contract (core.zig decodeMemory): `srgb_to_linear = true` runs
//! `color.convertSrgbToLinearInPlace` on level 0 BEFORE `buildRaw`, so
//! `mip.boxDownsampleU8` averages already-linear bytes and the chain is
//! uploaded as plain RGBA8 (linear). A shader then reads level-N texels as
//! `byte/255` — the same linear value Chrome's GPU produces by decoding the
//! re-encoded sRGB bytes of its own mip chain.
//!
//! These tests would FAIL if the ordering regressed to filter-then-convert
//! (gamma-space filtering): every "linear average" expectation below has the
//! gamma-space value it must NOT be, in the comment next to it.
const std = @import("std");
const sokol = @import("sokol");
const color = @import("color.zig");
const core = @import("core.zig");
const mip = @import("mip.zig");

const Texture = core.Texture;
const TestPng = core.TestPng;

/// Wraps RGBA8 pixels in a PNG. `with_alpha` picks truecolor (2) vs
/// truecolor+alpha (6): truecolor drops the alpha lane (stb fills 255), so
/// alpha tests must use 6.
fn pngFromPixels(allocator: std.mem.Allocator, width: u32, height: u32, rgba: []const u8, with_alpha: bool) ![]u8 {
    const px_bytes: usize = if (with_alpha) 4 else 3;
    const color_type: u8 = if (with_alpha) 6 else 2;
    const row_len = 1 + @as(usize, width) * px_bytes;
    const scanlines = try allocator.alloc(u8, row_len * height);
    defer allocator.free(scanlines);
    @memset(scanlines, 0);
    for (0..height) |y| {
        for (0..width) |x| {
            const si = (y * @as(usize, width) + x) * 4;
            const di = y * row_len + 1 + x * px_bytes;
            scanlines[di + 0] = rgba[si + 0];
            scanlines[di + 1] = rgba[si + 1];
            scanlines[di + 2] = rgba[si + 2];
            if (with_alpha) scanlines[di + 3] = rgba[si + 3];
        }
    }
    return TestPng.build(allocator, width, height, 8, color_type, null, scanlines);
}

/// IEC 61966-2-1 sRGB decode to linear f32 (the GPU's EOTF, linear toe
/// below 0.04045). Used to state expectations in the shader's value domain.
fn srgbDecodeF32(byte: u8) f32 {
    const srgb = @as(f32, @floatFromInt(byte)) / 255.0;
    if (srgb <= 0.04045) return srgb / 12.92;
    return std.math.pow(f32, (srgb + 0.055) / 1.055, 2.4);
}

test "sRGB texture mips average decoded linear values, not gamma bytes (2x1)" {
    const allocator = std.testing.allocator;
    // Mid-tone gray pair, chosen because decode(v) != v: the pipeline order
    // is observable. sRGB 255 -> linear 255, sRGB 96 -> linear 30.
    const rgba = [_]u8{
        255, 255, 255, 255,
        96,  96,  96,  255,
    };
    const png = try pngFromPixels(allocator, 2, 1, &rgba, false);
    defer allocator.free(png);
    sokol.time.setup();
    var raw = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = true, .srgb_to_linear = true });
    defer raw.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 2), raw.num_levels);
    try std.testing.expect(raw.is_srgb);

    // Level 0 is stored linearized (convertSrgbToLinearInPlace before
    // buildRaw): 255 stays 255, 96 becomes 30.
    const level0 = raw.levels[0].?;
    try std.testing.expectEqual(@as(u8, 255), level0[0]);
    try std.testing.expectEqual(@as(u8, 30), level0[4]);
    try std.testing.expectEqual(@as(u8, color.srgbToLinearU8(96)), level0[4]);

    // Level 1 = box average of the LINEAR bytes: (255+30)/2 = 142.5 -> 143
    // (generic path duplicates the single source row: (2*(255+30)+2)>>2).
    // Gamma-space filtering would instead average the sRGB bytes
    // ((255+96)/2 = 176) and every mip past level 0 would be too light...
    // which after the shader's sRGB decode reads FAR too dark (0.097 linear
    // instead of 0.558): the classic minified-sRGB artifact.
    const level1 = raw.levels[1].?;
    try std.testing.expectEqual(@as(u8, 143), level1[0]);
    try std.testing.expectEqual(@as(u8, 255), level1[3]);

    // Shader-domain equivalence with Chrome's GPU chain: agate stores the
    // linear average as a linear byte (143/255 = 0.5608); Chrome's GPU
    // averages decoded linear values and re-encodes to sRGB for storage
    // (byte 198 -> decodes back to ~0.558). Both shaders sample ~0.558
    // linear; the two pipelines agree to quantization.
    const agate_linear = @as(f32, @floatFromInt(level1[0])) / 255.0;
    const chrome_linear = (srgbDecodeF32(255) + srgbDecodeF32(96)) * 0.5;
    try std.testing.expectApproxEqAbs(chrome_linear, agate_linear, 0.01);
}

test "sRGB texture mips average linear values through the 2x2 fast path" {
    const allocator = std.testing.allocator;
    // Red / green / blue / mid gray in sRGB. Linear: 255->255, 96->30, so
    // the 2x2 -> 1x1 average per channel is (255+30+2)>>2 = 71 exactly.
    const rgba = [_]u8{
        255, 0,   0,   255,
        0,   255, 0,   255,
        0,   0,   255, 255,
        96,  96,  96,  255,
    };
    const png = try pngFromPixels(allocator, 2, 2, &rgba, false);
    defer allocator.free(png);
    sokol.time.setup();
    var raw = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = true, .srgb_to_linear = true });
    defer raw.deinit(allocator);

    const level1 = raw.levels[1].?;
    // Correct (linear): (71,71,71). Gamma-space bug would give
    // ((255+0+0+96)+2)>>2 = 88 per channel.
    try std.testing.expectEqual(@as(u8, 71), level1[0]);
    try std.testing.expectEqual(@as(u8, 71), level1[1]);
    try std.testing.expectEqual(@as(u8, 71), level1[2]);
    try std.testing.expectEqual(@as(u8, 255), level1[3]);
}

test "sRGB red|green 2x1 level 1 matches Chrome's ~188 sRGB re-encode in linear" {
    const allocator = std.testing.allocator;
    // The canonical pair: pure red + pure green. Linear average is exactly
    // 0.5; Chrome's GPU writes encode(0.5) = sRGB byte 188 into the chain,
    // which decodes back to ~0.503 at sample time. agate stores the linear
    // average directly (byte 128 = 0.502 linear). Endpoint colors map to
    // themselves under sRGB decode, so the STORED byte here is 128 either
    // way — the mid-tone tests above are the ones that discriminate order.
    const rgba = [_]u8{
        255, 0,   0, 255,
        0,   255, 0, 255,
    };
    const png = try pngFromPixels(allocator, 2, 1, &rgba, false);
    defer allocator.free(png);
    sokol.time.setup();
    var raw = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = true, .srgb_to_linear = true });
    defer raw.deinit(allocator);

    const level1 = raw.levels[1].?;
    try std.testing.expectEqual(@as(u8, 128), level1[0]);
    try std.testing.expectEqual(@as(u8, 128), level1[1]);
    try std.testing.expectEqual(@as(u8, 0), level1[2]);
    // Shader-domain check: linear value ~0.5, i.e. what a GLES3 GPU leaves
    // after filtering an sRGB texture (its stored byte would be 188).
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), @as(f32, @floatFromInt(level1[0])) / 255.0, 0.005);
}

test "sRGB mip generation never premultiplies and never converts alpha" {
    const allocator = std.testing.allocator;
    // Alpha lanes are averaged as plain bytes and are NOT run through the
    // sRGB LUT (color.convertSrgbToLinearInPlace skips lane 3). RGB
    // averaging is independent of alpha — no premultiply anywhere.
    const rgba = [_]u8{
        255, 0,   0, 255,
        0,   255, 0, 0,
    };
    const png = try pngFromPixels(allocator, 2, 1, &rgba, true);
    defer allocator.free(png);
    sokol.time.setup();
    var raw = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = true, .srgb_to_linear = true });
    defer raw.deinit(allocator);

    // Level 0: alpha untouched (255, 0), RGB linearized (endpoints stay).
    const level0 = raw.levels[0].?;
    try std.testing.expectEqual(@as(u8, 255), level0[3]);
    try std.testing.expectEqual(@as(u8, 0), level0[7]);

    // Level 1: straight byte average of the ORIGINAL alphas — (255+0)/2
    // -> 128, not a linear average of decoded alphas, and RGB is not
    // weighted by coverage.
    const level1 = raw.levels[1].?;
    try std.testing.expectEqual(@as(u8, 128), level1[0]);
    try std.testing.expectEqual(@as(u8, 128), level1[1]);
    try std.testing.expectEqual(@as(u8, 0), level1[2]);
    try std.testing.expectEqual(@as(u8, 128), level1[3]);
}

test "odd 3x1 sRGB texture: documented tail behavior of the box filter" {
    const allocator = std.testing.allocator;
    // Odd sizes: each level halves with floor(w/2) (GL mip semantics:
    // 3 -> 1, not 3 -> 2). agate's generic box taps {2x, 2x+1} clamped at
    // the edge, so a 3 -> 1 step averages texels 0 and 1 and the tail
    // texel 2 is not sampled. GLES3 leaves odd-dimension generateMipmap
    // behavior implementation-defined; desktop GL drivers commonly use
    // this same clamped 2-tap box (some mobile GPUs weight all covered
    // texels instead). This test PINS agate's choice; it is not a claim
    // that every GPU matches it.
    const rgba = [_]u8{
        255, 255, 255, 255,
        96,  96,  96,  255,
        200, 200, 200, 255,
    };
    const png = try pngFromPixels(allocator, 3, 1, &rgba, false);
    defer allocator.free(png);
    sokol.time.setup();
    var raw = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = true, .srgb_to_linear = true });
    defer raw.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 2), raw.num_levels);
    try std.testing.expectEqual(@as(u32, 3), raw.width);
    // Level 1 floors to 1x1 (GL mip semantics: 3 -> 1, not 3 -> 2).
    try std.testing.expectEqual(@as(usize, 4), raw.levels[1].?.len);
    // Linear level 0: 255, 30 (96), 147 (200).
    const level0 = raw.levels[0].?;
    try std.testing.expectEqual(@as(u8, 255), level0[0]);
    try std.testing.expectEqual(@as(u8, 30), level0[4]);
    try std.testing.expectEqual(@as(u8, color.srgbToLinearU8(200)), level0[8]);
    // Level 1 (1x1): average of texels 0,1 in linear space = 143; texel 2
    // (147) does not contribute. Gamma-space filtering would give 176.
    const level1 = raw.levels[1].?;
    try std.testing.expectEqual(@as(u8, 143), level1[0]);
}

test "linear data texture mips average raw bytes (srgb_to_linear = false)" {
    const allocator = std.testing.allocator;
    // Data textures (normal / metallic-roughness / occlusion maps) are
    // authored linear: srgb_to_linear stays false and the box filter
    // averages the raw bytes. Same 255|96 pair as the sRGB test: level 1
    // is 176 here (byte average) vs 143 there (linear average) — pinning
    // that the two paths really differ.
    const rgba = [_]u8{
        255, 255, 255, 255,
        96,  96,  96,  255,
    };
    const png = try pngFromPixels(allocator, 2, 1, &rgba, false);
    defer allocator.free(png);
    sokol.time.setup();
    var raw = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = true, .srgb_to_linear = false });
    defer raw.deinit(allocator);

    try std.testing.expect(!raw.is_srgb);
    const level0 = raw.levels[0].?;
    try std.testing.expectEqual(@as(u8, 96), level0[4]); // untouched
    const level1 = raw.levels[1].?;
    try std.testing.expectEqual(@as(u8, 176), level1[0]); // (2*(255+96)+2)>>2
}

test "sRGB mip chain shape: levels halve to 1x1 and stop" {
    const allocator = std.testing.allocator;
    var pixels: [4 * 4 * 4]u8 = undefined;
    for (&pixels) |*b| b.* = 200;
    const png = try pngFromPixels(allocator, 4, 4, &pixels, false);
    defer allocator.free(png);
    sokol.time.setup();
    var raw = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = true, .srgb_to_linear = true });
    defer raw.deinit(allocator);

    // 4x4 -> 2x2 -> 1x1: three levels, last is exactly one RGBA texel and
    // is the linear average of a uniform sRGB 200 field (147).
    try std.testing.expectEqual(Texture.mipLevelCount(4, 4), raw.num_levels);
    try std.testing.expectEqual(@as(u32, 3), raw.num_levels);
    try std.testing.expectEqual(@as(usize, 4), raw.levels[2].?.len);
    const last = raw.levels[2].?;
    try std.testing.expectEqual(@as(u8, color.srgbToLinearU8(200)), last[0]);
    try std.testing.expectEqual(@as(u8, 255), last[3]);
}

test "boxDownsampleU8 generic path averages the clamped 2x2 taps" {
    // A 3x2 -> 1x1 step misses the exact-2:1 fast path (src_w != 2*dst_w)
    // and runs the generic loop: taps {2x, 2x+1} clamped per axis, so dst
    // averages the 2D neighbourhood (0,0),(1,0),(0,1),(1,1) of the source
    // grid. Distinct bytes make every channel's contribution visible.
    var src: [3 * 2 * 4]u8 = undefined;
    for (0..6) |i| {
        src[i * 4 + 0] = @intCast(10 * i);
        src[i * 4 + 1] = 255 - @as(u8, @intCast(10 * i));
        src[i * 4 + 2] = 0;
        src[i * 4 + 3] = 255;
    }
    var dst: [1 * 1 * 4]u8 = undefined;
    mip.boxDownsampleU8(&src, 3, 2, &dst, 1, 1);
    // R taps: (0,0)=0, (1,0)=10, (0,1)=30, (1,1)=40 -> sum 80 -> 20.
    try std.testing.expectEqual(@as(u8, 20), dst[0]);
    try std.testing.expectEqual(@as(u8, 235), dst[1]); // 255+245+225+215=940 -> (940+2)>>2
    try std.testing.expectEqual(@as(u8, 0), dst[2]);
    try std.testing.expectEqual(@as(u8, 255), dst[3]);
}

test "boxDownsampleU8 rounds the 4-tap sum half-up" {
    // Document the rounding: byte = (sum + 2) >> 2, i.e. round-half-up on
    // the .5 boundary (sum 2 -> 1) and round-down below it (sum 1 -> 0).
    var src: [2 * 2 * 4]u8 = .{
        0, 0, 0, 255, 0, 0, 1, 255,
        0, 0, 1, 255, 1, 0, 1, 255,
    };
    var dst: [1 * 1 * 4]u8 = undefined;
    mip.boxDownsampleU8(&src, 2, 2, &dst, 1, 1);
    // R taps 0,0,0,1 -> sum 1 (0.25 avg) -> down.
    try std.testing.expectEqual(@as(u8, 0), dst[0]);
    try std.testing.expectEqual(@as(u8, 0), dst[1]);
    // B taps 0,1,1,1 -> sum 3 (0.75 avg) -> up.
    try std.testing.expectEqual(@as(u8, 1), dst[2]);
    try std.testing.expectEqual(@as(u8, 255), dst[3]);

    // Half exactly (sum 2) rounds UP: R taps 0,0,1,1.
    var half: [2 * 2 * 4]u8 = .{
        0, 0, 0, 255, 0, 0, 0, 255,
        1, 0, 0, 255, 1, 0, 0, 255,
    };
    mip.boxDownsampleU8(&half, 2, 2, &dst, 1, 1);
    try std.testing.expectEqual(@as(u8, 1), dst[0]);
}
