//! sRGB / procedural-dot / half-float helpers. Split out of `texture.zig`
//! (facade).
//!
//! Leaf: imports `std` only. `srgbToLinearU8` is re-exported by the facade
//! (used by `ktx2.zig`); the rest is `pub` for sibling leaves but NOT
//! re-exported, so the public surface matches the pre-split file.
const std = @import("std");

/// Comptime sRGB -> linear LUT for one u8 channel (IEC 61966-2-1). Round to
/// nearest: linear byte = trunc(linear_f32 * 255 + 0.5).
const srgb_to_linear_lut: [256]u8 = blk: {
    @setEvalBranchQuota(200000);
    var table: [256]u8 = undefined;
    for (0..256) |i| {
        const srgb: f32 = @as(f32, @floatFromInt(i)) / 255.0;
        const lin = if (srgb <= 0.04045)
            srgb / 12.92
        else
            std.math.pow(f32, (srgb + 0.055) / 1.055, 2.4);
        table[i] = @intFromFloat(std.math.clamp(lin * 255.0 + 0.5, 0.0, 255.0));
    }
    break :blk table;
};

/// Exact per-byte sRGB -> linear conversion used for LDR color textures.
pub fn srgbToLinearU8(value: u8) u8 {
    return srgb_to_linear_lut[value];
}

/// In-place sRGB -> linear on RGB lanes of an RGBA8 buffer; alpha lanes are
/// never touched. Runs BEFORE mip generation so the box filter averages in
/// linear space.
pub fn convertSrgbToLinearInPlace(pixels: []u8) void {
    for (pixels, 0..) |*byte, i| {
        if (i % 4 != 3) byte.* = srgb_to_linear_lut[byte.*];
    }
}

/// Radial particle-dot falloff shared by createDefaultParticleDot32 and
/// createParticleDot. Takes already-computed center/radius so both callers
/// keep their exact historic parameters and output is unchanged.
pub fn particleDotAlpha(x: u32, y: u32, center: f32, radius: f32) u8 {
    const dx = @as(f32, @floatFromInt(x)) - center;
    const dy = @as(f32, @floatFromInt(y)) - center;
    const dist = @sqrt(dx * dx + dy * dy);
    const norm_dist = @min(1.0, dist / radius);
    const alpha_f = (1.0 - norm_dist) * (1.0 - norm_dist);
    return @intFromFloat(std.math.clamp(alpha_f * 255.0, 0.0, 255.0));
}

/// Converts one f32 channel to an IEEE-754 half-precision bit pattern.
/// Out-of-range magnitudes become half infinity, NaN stays NaN.
/// Content above 65504 loses detail: tone-map before upload if it matters.
pub fn floatToHalfBits(value: f32) u16 {
    return @bitCast(@as(f16, @floatCast(value)));
}

/// Converts an IEEE-754 half-precision bit pattern back to f32.
/// Used by tests and debugging; the GPU upload path never needs it.
pub fn halfBitsToFloat(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

// ---------------------------------------------------------------------------
// Golden tests: sRGB conversion
// ---------------------------------------------------------------------------
test "srgbToLinearU8 hits exact golden values and the endpoints" {
    // Computed with the IEC 61966-2-1 formula, rounded to nearest.
    try std.testing.expectEqual(@as(u8, 0), srgbToLinearU8(0));
    try std.testing.expectEqual(@as(u8, 255), srgbToLinearU8(255));
    try std.testing.expectEqual(@as(u8, 55), srgbToLinearU8(128));
    try std.testing.expectEqual(@as(u8, 13), srgbToLinearU8(64));
    try std.testing.expectEqual(@as(u8, 2), srgbToLinearU8(25));
    try std.testing.expectEqual(@as(u8, 147), srgbToLinearU8(200));
    // Monotonic non-decreasing over the whole table.
    var v: usize = 1;
    while (v < 256) : (v += 1) {
        try std.testing.expect(srgbToLinearU8(@intCast(v)) >= srgbToLinearU8(@intCast(v - 1)));
    }
}

test "convertSrgbToLinearInPlace converts RGB lanes and preserves alpha" {
    var pixels = [_]u8{ 200, 128, 0, 42, 255, 0, 25, 7 };
    convertSrgbToLinearInPlace(&pixels);
    try std.testing.expectEqual(@as(u8, srgbToLinearU8(200)), pixels[0]);
    try std.testing.expectEqual(@as(u8, srgbToLinearU8(128)), pixels[1]);
    try std.testing.expectEqual(@as(u8, 0), pixels[2]);
    try std.testing.expectEqual(@as(u8, 42), pixels[3]); // alpha untouched
    try std.testing.expectEqual(@as(u8, 255), pixels[4]);
    try std.testing.expectEqual(@as(u8, 0), pixels[5]);
    try std.testing.expectEqual(@as(u8, srgbToLinearU8(25)), pixels[6]);
    try std.testing.expectEqual(@as(u8, 7), pixels[7]); // alpha untouched
}
