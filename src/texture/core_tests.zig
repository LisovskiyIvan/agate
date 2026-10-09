//! Tests for `core.zig` (moved from `core.zig` inline blocks).
const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const ktx2 = @import("../ktx2.zig");
const dds = @import("../dds.zig");
const exr = @import("../exr.zig");
const color = @import("color.zig");
const mip = @import("mip.zig");
const core = @import("core.zig");
const Texture = core.Texture;
const TestPng = core.TestPng;

// HDR fixtures (copies of the byte strings formerly defined beside the
// moved decode tests in core.zig).
const hdr_1x1_flat: []const u8 = "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n\x80\x80\x80\x80";
const hdr_8x1_rle: []const u8 = "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 8\n\x02\x02\x00\x08\x88\x80\x88\x80\x88\x80\x88\x80";

test "mipLevelCount covers powers of two and minimums" {
    try std.testing.expectEqual(@as(u32, 1), Texture.mipLevelCount(1, 1));
    try std.testing.expectEqual(@as(u32, 2), Texture.mipLevelCount(2, 2));
    try std.testing.expectEqual(@as(u32, 10), Texture.mipLevelCount(512, 512));
    try std.testing.expectEqual(@as(u32, 3), Texture.mipLevelCount(5, 3));
}

test "sgPixelFormatForBlock maps UNORM/SRGB variants exactly" {
    try std.testing.expectEqual(sg.PixelFormat.BC1_RGBA, Texture.sgPixelFormatForBlock(.bc1_unorm));
    try std.testing.expectEqual(sg.PixelFormat.BC2_RGBA, Texture.sgPixelFormatForBlock(.bc2_unorm));
    try std.testing.expectEqual(sg.PixelFormat.BC3_RGBA, Texture.sgPixelFormatForBlock(.bc3_unorm));
    try std.testing.expectEqual(sg.PixelFormat.BC3_SRGBA, Texture.sgPixelFormatForBlock(.bc3_srgb));
    try std.testing.expectEqual(sg.PixelFormat.BC7_RGBA, Texture.sgPixelFormatForBlock(.bc7_unorm));
    try std.testing.expectEqual(sg.PixelFormat.BC7_SRGBA, Texture.sgPixelFormatForBlock(.bc7_srgb));
    try std.testing.expectEqual(sg.PixelFormat.ETC2_RGBA8, Texture.sgPixelFormatForBlock(.etc2_rgba8_unorm));
    try std.testing.expectEqual(sg.PixelFormat.ETC2_SRGB8A8, Texture.sgPixelFormatForBlock(.etc2_rgba8_srgb));
    try std.testing.expectEqual(sg.PixelFormat.ASTC_4x4_RGBA, Texture.sgPixelFormatForBlock(.astc_4x4_unorm));
    try std.testing.expectEqual(sg.PixelFormat.ASTC_4x4_SRGBA, Texture.sgPixelFormatForBlock(.astc_4x4_srgb));
}

test "BlockSupport gates exact variants and prefers BC7 over ASTC and ETC2" {
    const full: Texture.BlockSupport = .{ .bc7_sample = true, .bc7_filter = true, .etc2_sample = true, .etc2_filter = true, .astc_sample = true, .astc_filter = true };
    try std.testing.expect(full.supportsFormat(.BC7_RGBA));
    try std.testing.expect(full.supportsFormat(.BC7_SRGBA));
    try std.testing.expect(full.supportsFormat(.ASTC_4x4_RGBA));
    try std.testing.expect(full.supportsFormat(.ASTC_4x4_SRGBA));
    try std.testing.expect(full.supportsFormat(.ETC2_RGBA8));
    try std.testing.expect(full.supportsFormat(.ETC2_SRGB8A8));
    try std.testing.expect(!full.supportsFormat(.RGBA8));
    try std.testing.expect(!full.supportsFormat(.BC3_RGBA));
    // Preference order BC7 -> ASTC 4x4.
    try std.testing.expectEqual(sg.PixelFormat.BC7_RGBA, full.preferred().?);

    const astc_only: Texture.BlockSupport = .{ .astc_sample = true, .astc_filter = true };
    try std.testing.expect(!astc_only.supportsFormat(.BC7_RGBA));
    try std.testing.expect(!astc_only.supportsFormat(.BC7_SRGBA));
    try std.testing.expect(astc_only.supportsFormat(.ASTC_4x4_RGBA));
    try std.testing.expectEqual(sg.PixelFormat.ASTC_4x4_RGBA, astc_only.preferred().?);

    const etc2_only: Texture.BlockSupport = .{ .etc2_sample = true, .etc2_filter = true };
    try std.testing.expect(!etc2_only.supportsFormat(.ASTC_4x4_RGBA));
    try std.testing.expect(etc2_only.supportsFormat(.ETC2_SRGB8A8));
    try std.testing.expectEqual(sg.PixelFormat.ETC2_RGBA8, etc2_only.preferred().?);

    // Sample-without-filter still gates as supported (upload forces
    // NEAREST); SRGB and UNORM share the family bit.
    const bc7_no_filter: Texture.BlockSupport = .{ .bc7_sample = true };
    try std.testing.expect(bc7_no_filter.supportsFormat(.BC7_SRGBA));
    try std.testing.expect(!bc7_no_filter.supportsFormat(.ASTC_4x4_SRGBA));

    const none: Texture.BlockSupport = .{};
    try std.testing.expect(none.preferred() == null);
    try std.testing.expect(!none.supportsFormat(.BC7_RGBA));
    try std.testing.expect(!none.supportsFormat(.ETC2_RGBA8));
}

test "BlockSupport gates the BC1/BC2/BC3 families for the DDS path" {
    // BC2 rides the BC3 (S3TC) feature bit; BC3_SRGBA needs no extra bit.
    const s3tc: Texture.BlockSupport = .{ .bc3_sample = true };
    try std.testing.expect(s3tc.supportsFormat(.BC2_RGBA));
    try std.testing.expect(s3tc.supportsFormat(.BC3_RGBA));
    try std.testing.expect(s3tc.supportsFormat(.BC3_SRGBA));
    try std.testing.expect(!s3tc.supportsFormat(.BC1_RGBA));
    try std.testing.expect(!s3tc.supportsFormat(.BC7_RGBA));

    const bc1_only: Texture.BlockSupport = .{ .bc1_sample = true };
    try std.testing.expect(bc1_only.supportsFormat(.BC1_RGBA));
    try std.testing.expect(!bc1_only.supportsFormat(.BC2_RGBA));
    try std.testing.expectEqual(sg.PixelFormat.BC1_RGBA, bc1_only.preferred().?);

    // Preference order BC7 -> BC3 -> BC1 -> ASTC 4x4.
    const bc3_astc: Texture.BlockSupport = .{ .bc3_sample = true, .astc_sample = true };
    try std.testing.expectEqual(sg.PixelFormat.BC3_RGBA, bc3_astc.preferred().?);
}

test "decodeMemory returns RGBA levels and owns its mip chain" {
    const png = @embedFile("../assets/font_sdf.png");
    const allocator = std.testing.allocator;

    // decodeMemory logs decode timings through sokol.time; sokol_time is
    // CPU-only and needs its one-time setup (normally done at app startup).
    sokol.time.setup();

    var single = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = false });
    defer single.deinit(allocator);
    try std.testing.expectEqual(@as(u32, 512), single.width);
    try std.testing.expectEqual(@as(u32, 512), single.height);
    try std.testing.expectEqual(@as(u32, 1), single.num_levels);
    try std.testing.expectEqual(@as(usize, 512 * 512 * 4), single.levels[0].?.len);

    var mipped = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = true });
    defer mipped.deinit(allocator);
    try std.testing.expectEqual(Texture.mipLevelCount(512, 512), mipped.num_levels);
    try std.testing.expectEqual(@as(usize, 4), mipped.levels[mipped.num_levels - 1].?.len);
}

test "decodeHDRMemory decodes minimal flat Radiance .hdr" {
    const allocator = std.testing.allocator;

    var raw = try Texture.decodeHDRMemory(allocator, hdr_1x1_flat);
    defer raw.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 1), raw.width);
    try std.testing.expectEqual(@as(u32, 1), raw.height);
    try std.testing.expectEqual(@as(usize, 4), raw.pixels.len);
    // Exact half patterns: 0.5 -> 0x3800, 1.0 -> 0x3C00.
    try std.testing.expectEqual(@as(u16, 0x3800), raw.pixels[0]);
    try std.testing.expectEqual(@as(u16, 0x3800), raw.pixels[1]);
    try std.testing.expectEqual(@as(u16, 0x3800), raw.pixels[2]);
    try std.testing.expectEqual(@as(u16, 0x3C00), raw.pixels[3]);
}

test "decodeHDRMemory decodes RLE Radiance scanlines" {
    const allocator = std.testing.allocator;

    var raw = try Texture.decodeHDRMemory(allocator, hdr_8x1_rle);
    defer raw.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 8), raw.width);
    try std.testing.expectEqual(@as(u32, 1), raw.height);
    try std.testing.expectEqual(@as(usize, 8 * 1 * 4), raw.pixels.len);
    for (0..8) |i| {
        try std.testing.expectEqual(@as(u16, 0x3800), raw.pixels[i * 4 + 0]);
        try std.testing.expectEqual(@as(u16, 0x3800), raw.pixels[i * 4 + 1]);
        try std.testing.expectEqual(@as(u16, 0x3800), raw.pixels[i * 4 + 2]);
        try std.testing.expectEqual(@as(u16, 0x3C00), raw.pixels[i * 4 + 3]);
    }
}

test "decodeHDRMemory rejects foreign, truncated and empty data" {
    const allocator = std.testing.allocator;

    try std.testing.expectError(
        error.ImageDecodeFailed,
        Texture.decodeHDRMemory(allocator, "hello world, this is not an image"),
    );
    try std.testing.expectError(
        error.ImageDecodeFailed,
        Texture.decodeHDRMemory(allocator, "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n"),
    );
    try std.testing.expectError(
        error.ImageDecodeFailed,
        Texture.decodeHDRMemory(allocator, hdr_1x1_flat[0 .. hdr_1x1_flat.len - 2]),
    );
    const empty: []const u8 = &[_]u8{};
    try std.testing.expectError(
        error.ImageDecodeFailed,
        Texture.decodeHDRMemory(allocator, empty),
    );
}

test "floatToHalfBits covers exact values, overflow and NaN" {
    try std.testing.expectEqual(@as(u16, 0x3C00), Texture.floatToHalfBits(1.0));
    try std.testing.expectEqual(@as(u16, 0x3800), Texture.floatToHalfBits(0.5));
    try std.testing.expectEqual(@as(u16, 0x0000), Texture.floatToHalfBits(0.0));
    try std.testing.expectEqual(@as(f32, 1.0), Texture.halfBitsToFloat(0x3C00));

    // Beyond f16 max (65504): documents the half-float precision ceiling.
    try std.testing.expect(std.math.isInf(Texture.halfBitsToFloat(Texture.floatToHalfBits(1.0e10))));
    try std.testing.expect(std.math.isNan(Texture.halfBitsToFloat(Texture.floatToHalfBits(std.math.nan(f32)))));
}

test "decodeHDRFile reports missing files without leaking" {
    const allocator = std.testing.allocator;

    try std.testing.expectError(
        error.ImageDecodeFailed,
        Texture.decodeHDRFile(allocator, "definitely/missing/file.hdr"),
    );
}

test "createCheckerboard overflow returns error instead of panicking" {
    const allocator = std.testing.allocator;
    const c1 = [4]u8{ 0, 0, 0, 255 };
    const c2 = [4]u8{ 255, 255, 255, 255 };
    // 100000^2 overflows u32: old `width * height` panicked in debug;
    // now returns ImageTooLarge before any allocation or GPU upload.
    try std.testing.expectError(
        error.ImageTooLarge,
        Texture.createCheckerboard(allocator, 100000, 100000, 8, c1, c2),
    );
}

test "decodeMemory handles synthesized 8-bit grayscale PNG (gray replicated to RGB)" {
    const allocator = std.testing.allocator;
    sokol.time.setup();

    // 1x2 grayscale, values 0x00 and 0x80, filter byte 0 per row.
    const scanlines = [_]u8{ 0, 0x00, 0, 0x80 };
    const png = try TestPng.build(allocator, 1, 2, 8, 0, null, &scanlines);
    defer allocator.free(png);

    var raw = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = false });
    defer raw.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 1), raw.width);
    try std.testing.expectEqual(@as(u32, 2), raw.height);
    const px = raw.levels[0].?;
    // stb replicates the single gray channel into RGB; alpha becomes 255.
    try std.testing.expectEqual([4]u8{ 0, 0, 0, 255 }, px[0..4].*);
    try std.testing.expectEqual([4]u8{ 128, 128, 128, 255 }, px[4..8].*);
}

test "decodeMemory keeps the HIGH byte of 16-bit PNG channels" {
    const allocator = std.testing.allocator;
    sokol.time.setup();

    // 1x1 RGB 16-bit: R=0x1234, G=0xABCD, B=0x0001. stb's LDR output keeps
    // the top byte of each channel (stbi__convert_16_to_8: orig >> 8).
    var scanlines: [7]u8 = undefined;
    scanlines[0] = 0; // filter
    std.mem.writeInt(u16, scanlines[1..3], 0x1234, .big);
    std.mem.writeInt(u16, scanlines[3..5], 0xABCD, .big);
    std.mem.writeInt(u16, scanlines[5..7], 0x0001, .big);
    const png = try TestPng.build(allocator, 1, 1, 16, 2, null, &scanlines);
    defer allocator.free(png);

    var raw = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = false });
    defer raw.deinit(allocator);

    try std.testing.expectEqual([4]u8{ 0x12, 0xAB, 0x00, 255 }, raw.levels[0].?[0..4].*);
}

test "decodeMemory keeps the HIGH byte of 16-bit grayscale PNG" {
    const allocator = std.testing.allocator;
    sokol.time.setup();

    // 1x1 gray 16-bit value 0xABCD -> high byte 0xAB replicated to RGB.
    var scanlines: [3]u8 = undefined;
    scanlines[0] = 0;
    std.mem.writeInt(u16, scanlines[1..3], 0xABCD, .big);
    const png = try TestPng.build(allocator, 1, 1, 16, 0, null, &scanlines);
    defer allocator.free(png);

    var raw = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = false });
    defer raw.deinit(allocator);

    try std.testing.expectEqual([4]u8{ 0xAB, 0xAB, 0xAB, 255 }, raw.levels[0].?[0..4].*);
}

test "decodeMemory decodes synthesized palette PNG through PLTE" {
    const allocator = std.testing.allocator;
    sokol.time.setup();

    // 2x1 indexed (color type 3), palette {red, blue}; indices 0 and 1.
    const scanlines = [_]u8{ 0, 0, 1 };
    const plte = [_]u8{ 255, 0, 0, 0, 0, 255 };
    const png = try TestPng.build(allocator, 2, 1, 8, 3, &plte, &scanlines);
    defer allocator.free(png);

    var raw = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = false });
    defer raw.deinit(allocator);

    const px = raw.levels[0].?;
    try std.testing.expectEqual([4]u8{ 255, 0, 0, 255 }, px[0..4].*);
    try std.testing.expectEqual([4]u8{ 0, 0, 255, 255 }, px[4..8].*);
}

test "decodeMemory srgb_to_linear converts decoded pixels before upload" {
    const allocator = std.testing.allocator;
    sokol.time.setup();

    // 1x1 RGBA PNG with sRGB value 200 in R and a distinctive alpha 42.
    const scanlines = [_]u8{ 0, 200, 128, 25, 42 };
    const png = try TestPng.build(allocator, 1, 1, 8, 6, null, &scanlines);
    defer allocator.free(png);

    var raw = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = false, .srgb_to_linear = true });
    defer raw.deinit(allocator);

    const px = raw.levels[0].?;
    try std.testing.expectEqual(@as(u8, 147), px[0]); // 200 -> 147 golden
    try std.testing.expectEqual(color.srgbToLinearU8(128), px[1]);
    try std.testing.expectEqual(color.srgbToLinearU8(25), px[2]);
    try std.testing.expectEqual(@as(u8, 42), px[3]); // alpha never converted
}

test "decodeMemory TextureSlot and TextureColorSpace contract" {
    const allocator = std.testing.allocator;
    sokol.time.setup();

    const scanlines = [_]u8{ 0, 200, 128, 25, 42 };
    const png = try TestPng.build(allocator, 1, 1, 8, 6, null, &scanlines);
    defer allocator.free(png);

    // .slot = .color -> converts sRGB to linear, is_srgb = true
    {
        var raw = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = false, .slot = .color });
        defer raw.deinit(allocator);
        try std.testing.expect(raw.is_srgb);
        const px = raw.levels[0].?;
        try std.testing.expectEqual(@as(u8, 147), px[0]);
    }

    // .slot = .data -> keeps linear raw bytes, is_srgb = false
    {
        var raw = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = false, .slot = .data });
        defer raw.deinit(allocator);
        try std.testing.expect(!raw.is_srgb);
        const px = raw.levels[0].?;
        try std.testing.expectEqual(@as(u8, 200), px[0]);
    }

    // .color_space = .srgb -> converts
    {
        var raw = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = false, .color_space = .srgb });
        defer raw.deinit(allocator);
        try std.testing.expect(raw.is_srgb);
        const px = raw.levels[0].?;
        try std.testing.expectEqual(@as(u8, 147), px[0]);
    }

    // .color_space = .linear -> does not convert
    {
        var raw = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = false, .color_space = .linear });
        defer raw.deinit(allocator);
        try std.testing.expect(!raw.is_srgb);
        const px = raw.levels[0].?;
        try std.testing.expectEqual(@as(u8, 200), px[0]);
    }

    // Precedence: color_space overrides slot and srgb_to_linear
    {
        var raw = try Texture.decodeMemory(allocator, png, .{
            .gen_mipmaps = false,
            .slot = .color,
            .srgb_to_linear = true,
            .color_space = .linear,
        });
        defer raw.deinit(allocator);
        try std.testing.expect(!raw.is_srgb);
        const px = raw.levels[0].?;
        try std.testing.expectEqual(@as(u8, 200), px[0]);
    }
}

test "buildRaw mip chain has correct levels, sizes and box-filter colors" {
    const allocator = std.testing.allocator;

    // 4x4: top half pure red, bottom half pure blue, alpha 255 everywhere.
    var pixels: [4 * 4 * 4]u8 = undefined;
    for (0..4) |y| {
        for (0..4) |x| {
            const o = (y * 4 + x) * 4;
            const red = y < 2;
            pixels[o + 0] = if (red) 255 else 0;
            pixels[o + 1] = 0;
            pixels[o + 2] = if (red) 0 else 255;
            pixels[o + 3] = 255;
        }
    }

    var raw = try Texture.buildRaw(allocator, 4, 4, &pixels, true);
    defer raw.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 3), raw.num_levels);
    try std.testing.expectEqual(@as(u32, 4), raw.width);
    try std.testing.expectEqual(@as(u32, 4), raw.height);

    // Level 1 (2x2): top row red, bottom row blue, untouched by the filter.
    const l1 = raw.levels[1].?;
    try std.testing.expectEqual(@as(usize, 2 * 2 * 4), l1.len);
    try std.testing.expectEqual([4]u8{ 255, 0, 0, 255 }, l1[0..4].*);
    try std.testing.expectEqual([4]u8{ 0, 0, 255, 255 }, l1[8..12].*);

    // Level 2 (1x1): equal red/blue mix rounds to 128 via (sum+2)>>2.
    const l2 = raw.levels[2].?;
    try std.testing.expectEqual(@as(usize, 4), l2.len);
    try std.testing.expectEqual([4]u8{ 128, 0, 128, 255 }, l2[0..4].*);
}

test "buildRaw without mipmaps uploads exactly one level" {
    const allocator = std.testing.allocator;
    var pixels: [2 * 2 * 4]u8 = @splat(200);
    var raw = try Texture.buildRaw(allocator, 2, 2, &pixels, false);
    defer raw.deinit(allocator);
    try std.testing.expectEqual(@as(u32, 1), raw.num_levels);
    try std.testing.expectEqual(@as(u32, 200), raw.levels[0].?[0]);
}

test "buildRaw rejects short, oversized and empty pixel buffers" {
    const allocator = std.testing.allocator;
    var pixels: [2 * 2 * 4]u8 = @splat(128);
    // Short by one byte (would over-read in the unsafely-downsampled chain).
    try std.testing.expectError(
        error.InvalidDimensions,
        Texture.buildRaw(allocator, 2, 2, pixels[0 .. pixels.len - 1], true),
    );
    // One byte too many.
    var over: [2 * 2 * 4 + 1]u8 = @splat(128);
    try std.testing.expectError(
        error.InvalidDimensions,
        Texture.buildRaw(allocator, 2, 2, &over, false),
    );
    // Zero dimensions rejected even with an empty buffer.
    try std.testing.expectError(
        error.InvalidDimensions,
        Texture.buildRaw(allocator, 0, 2, &.{}, false),
    );
    // Huge dimensions fail on checked arithmetic before any allocation.
    try std.testing.expectError(
        error.ImageTooLarge,
        Texture.buildRaw(allocator, 100000, 100000, &.{}, false),
    );
}

test "buildRaw accepts an exact-size tiny raw texture" {
    const allocator = std.testing.allocator;
    var pixels: [2 * 2 * 4]u8 = @splat(200);
    var raw = try Texture.buildRaw(allocator, 2, 2, &pixels, true);
    defer raw.deinit(allocator);
    try std.testing.expectEqual(@as(u32, 2), raw.num_levels);
    try std.testing.expectEqual(@as(usize, 2 * 2 * 4), raw.levels[0].?.len);
    try std.testing.expectEqual(@as(usize, 1 * 1 * 4), raw.levels[1].?.len);
}

test "createParticleDot validates size without GPU upload" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(
        error.InvalidDimensions,
        Texture.createParticleDot(allocator, 0),
    );
    try std.testing.expectError(
        error.ImageTooLarge,
        Texture.createParticleDot(allocator, 100000),
    );
}

test "Options.max_anisotropy defaults to Babylon's 4" {
    // babylon.js: e.DEFAULT_ANISOTROPIC_FILTERING_LEVEL=4, assigned to
    // every Texture.anisotropicFilteringLevel; the glTF loader never
    // overrides it. A default-constructed Options must ask for that, and a
    // mipmapped LINEAR texture must actually get it (the bench's
    // DamagedHelmet/Fox path).
    const defaults: Texture.Options = .{};
    try std.testing.expectEqual(@as(u32, 4), defaults.max_anisotropy);
    try std.testing.expectEqual(@as(u32, 4), Texture.effectiveAnisotropy(defaults, 10));
}

test "effectiveAnisotropy clamps to 1 without LINEAR min/mag/mip" {
    // sokol requires LINEAR min AND mag AND mipmap for anisotropy > 1
    // (VALIDATE_SAMPLERDESC_ANISTROPIC_REQUIRES_LINEAR_FILTERING); Babylon
    // clamps identically in _setAnisotropicLevel.
    const nearest_min: Texture.Options = .{ .min_filter = .NEAREST, .max_anisotropy = 8 };
    const nearest_mag: Texture.Options = .{ .mag_filter = .NEAREST, .max_anisotropy = 8 };
    const nearest_mip: Texture.Options = .{ .mip_filter = .NEAREST, .max_anisotropy = 8 };
    try std.testing.expectEqual(@as(u32, 1), Texture.effectiveAnisotropy(nearest_min, 10));
    try std.testing.expectEqual(@as(u32, 1), Texture.effectiveAnisotropy(nearest_mag, 10));
    try std.testing.expectEqual(@as(u32, 1), Texture.effectiveAnisotropy(nearest_mip, 10));

    // Same options WITH a single level: the sampler mip filter becomes
    // NEAREST regardless of the authored one, so anisotropy clamps too.
    try std.testing.expectEqual(@as(u32, 1), Texture.effectiveAnisotropy(.{ .max_anisotropy = 4 }, 1));
    try std.testing.expectEqual(@as(u32, 4), Texture.effectiveAnisotropy(.{ .max_anisotropy = 4 }, 2));

    // An explicit 1 is the escape hatch back to agate's old sampling.
    try std.testing.expectEqual(@as(u32, 1), Texture.effectiveAnisotropy(.{ .max_anisotropy = 1 }, 10));

    // sokol's documented range is 1..16.
    try std.testing.expectEqual(@as(u32, 16), Texture.effectiveAnisotropy(.{ .max_anisotropy = 64 }, 10));
    try std.testing.expectEqual(@as(u32, 0), Texture.effectiveAnisotropy(.{ .max_anisotropy = 0 }, 10));
}

test "brdf lut sampler pins CLAMP wrap (REPEAT regression)" {
    // v = 1.0 is roughness exactly 1.0: under REPEAT a LINEAR sample blends the
    // LUT's last row with its first, which silently under-reported
    // coloredEnergyConservationFactor (PROBE.md 10.9). The engine default stays
    // REPEAT for ordinary textures, so the LUT must pin CLAMP explicitly and
    // this constant is the only place the sampler is described.
    try std.testing.expectEqual(sg.Wrap.CLAMP_TO_EDGE, Texture.brdf_lut_options.wrap_u);
    try std.testing.expectEqual(sg.Wrap.CLAMP_TO_EDGE, Texture.brdf_lut_options.wrap_v);
    try std.testing.expectEqual(sg.Filter.LINEAR, Texture.brdf_lut_options.min_filter);
    try std.testing.expectEqual(sg.Filter.LINEAR, Texture.brdf_lut_options.mag_filter);
}

test "downsampleLevel handles odd dimensions and averages correctly" {
    var src: [3 * 5 * 4]u8 = undefined;
    for (&src, 0..) |*b, i| b.* = @intCast(i % 251);
    var dst: [1 * 2 * 4]u8 = undefined;
    Texture.downsampleLevel(&src, 3, 5, &dst, 1, 2);
    // Top-left texel averages src bytes {0,4,12,16}: (0+4+12+16+2)/4 = 8.
    try std.testing.expectEqual(@as(u8, 8), dst[0]);
}

test "boxDownsampleU8 handles NPOT cube-face steps without OOB" {
    // 3x3 -> 1x1 is the exact mip step of a size-3 NPOT cube face
    // (cur = max(1, prev / 2)). R channel holds the texel index.
    var src: [3 * 3 * 4]u8 = undefined;
    for (0..9) |i| {
        src[i * 4 + 0] = @intCast(i);
        src[i * 4 + 1] = 0;
        src[i * 4 + 2] = 0;
        src[i * 4 + 3] = 255;
    }
    var dst: [1 * 1 * 4]u8 = undefined;
    mip.boxDownsampleU8(&src, 3, 3, &dst, 1, 1);
    // Averages top-left quad {0,1,3,4}: (0+1+3+4+2)>>2 = 2.
    try std.testing.expectEqual(@as(u8, 2), dst[0]);
    try std.testing.expectEqual(@as(u8, 255), dst[3]);
    // Same dims through the 2D wrapper must agree (shared helper).
    var dst2: [1 * 1 * 4]u8 = undefined;
    Texture.downsampleLevel(&src, 3, 3, &dst2, 1, 1);
    try std.testing.expectEqualSlices(u8, &dst, &dst2);

    // 2x1 -> 1x1 exercises edge clamping (sy1 clamps to 0): the single
    // source row is sampled twice, i.e. a plain average, no OOB read.
    var edge_src: [2 * 1 * 4]u8 = .{ 10, 0, 0, 255, 20, 0, 0, 255 };
    var edge_dst: [1 * 1 * 4]u8 = undefined;
    mip.boxDownsampleU8(&edge_src, 2, 1, &edge_dst, 1, 1);
    // (10+20+10+20+2)>>2 = 15.
    try std.testing.expectEqual(@as(u8, 15), edge_dst[0]);
}
