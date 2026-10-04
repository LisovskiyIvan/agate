const std = @import("std");
const types = @import("types.zig");
const color_curves = @import("color_curves.zig");
const bloom = @import("bloom.zig");
const options = @import("options.zig");

// Glow layer v1 (global postfx halo; Babylon.js "glow layer" parity item,
// currently the highlight layer stays per-mesh and out of scope): threshold
// extract from the scene color + separable H/V Gaussian blur + additive
// composite after bloom. A distinct, user-controllable effect — not a bloom
// duplicate: own toggle, own RTs (GlowPass), own composite block, and valid
// while bloom is on or off without changing bloom's output.
pub const GLOW_BLUR_TAPS: u32 = 9;
// Half-width of the blur kernel: tap offsets run -GLOW_HALF_TAPS..GLOW_HALF_TAPS.
pub const GLOW_HALF_TAPS: i32 = 4;
// Fullscreen draws one glow frame issues: extract + horizontal + vertical.
pub const GLOW_PASS_DRAWS: u32 = 3;
// Defaults (see PostProcessOptions): threshold above bloom's 0.8 so only
// true brights bleed by default, radius wider than bloom's 2.0.
pub const GLOW_THRESHOLD_DEFAULT: f32 = 1.0;
pub const GLOW_INTENSITY_DEFAULT: f32 = 0.5;
pub const GLOW_RADIUS_DEFAULT: f32 = 4.0;

/// Clamp one glow tint triplet into [0, 1] per channel. A tint is a color
/// multiplier (unlike a grade lift it can never go negative); values above 1
/// would double as intensity, which already has its own knob.
pub fn clampTint(v: [3]f32) [3]f32 {
    return .{
        std.math.clamp(v[0], 0.0, 1.0),
        std.math.clamp(v[1], 0.0, 1.0),
        std.math.clamp(v[2], 0.0, 1.0),
    };
}

/// Strict range check for the glow knobs. Finite out-of-range values are the
/// clamped() domain (bloom precedent: silent sanitize on load); non-finite
/// values (NaN/Inf) can never sanitize meaningfully, so they are a hard
/// InvalidGlowOptions error here instead of a silent clamp.
pub fn validateGlow(cfg: options.PostProcessOptions) !void {
    if (!std.math.isFinite(cfg.glow_threshold)) return error.InvalidGlowOptions;
    if (!std.math.isFinite(cfg.glow_intensity)) return error.InvalidGlowOptions;
    if (!std.math.isFinite(cfg.glow_radius)) return error.InvalidGlowOptions;
    for (cfg.glow_tint) |c| {
        if (!std.math.isFinite(c)) return error.InvalidGlowOptions;
    }
    if (cfg.glow_threshold < 0.0) return error.InvalidGlowOptions;
    if (cfg.glow_intensity < 0.0) return error.InvalidGlowOptions;
    if (cfg.glow_radius < 0.0) return error.InvalidGlowOptions;
    for (cfg.glow_tint) |c| {
        if (c < 0.0 or c > 1.0) return error.InvalidGlowOptions;
    }
}

/// Pass-construction decisions (pure; PostFXStack.renderChain gates the GPU
/// passes on these).
pub fn glowActive(post_enabled: bool, cfg: options.PostProcessOptions) bool {
    return post_enabled and cfg.glow_enabled;
}

/// Bright-pass extract. Mirrors extractBright in postprocess.glsl and the
/// glow_extract.glsl stage: below-threshold luminance maps to black, above
/// scales the color by the over-threshold fraction.
pub fn glowExtract(color: [3]f32, threshold: f32) [3]f32 {
    const l = color_curves.rgbLuma(color);
    const factor = @max(0.0, l - threshold) / @max(l, 0.0001);
    return .{ color[0] * factor, color[1] * factor, color[2] * factor };
}

/// 1D Gaussian tap weight for integer `offset` at `sigma`. Mirrors
/// glow_blur.glsl (sigma is glow_radius in glow-target texels, floored at
/// 0.5 exactly like the shader). Raw (unnormalized): both sides divide by
/// the kernel sum over -GLOW_HALF_TAPS..GLOW_HALF_TAPS.
pub fn glowGaussianWeight(offset: f32, sigma: f32) f32 {
    const s = @max(sigma, 0.5);
    const t = offset / s;
    return @exp(-0.5 * t * t);
}

/// Kernel normalization both sides apply: sum of glowGaussianWeight over the
/// 9 taps at `sigma`.
pub fn glowKernelSum(sigma: f32) f32 {
    var sum: f32 = 0.0;
    var i: i32 = -GLOW_HALF_TAPS;
    while (i <= GLOW_HALF_TAPS) : (i += 1) {
        sum += glowGaussianWeight(@floatFromInt(i), sigma);
    }
    return sum;
}

/// Pack the composite glow_params vec4: (enabled 1/0, intensity, 0, 0).
/// Disabled packs all zeros, which keeps the composite bit-identical to the
/// pre-glow path (the shader returns before sampling glow_tex).
pub fn glowParams(cfg: options.PostProcessOptions) [4]f32 {
    if (!cfg.glow_enabled) return .{ 0.0, 0.0, 0.0, 0.0 };
    const c = cfg.clamped();
    return .{ 1.0, c.glow_intensity, 0.0, 0.0 };
}

/// Pack the composite glow_tint vec4: (tint rgb, 0). Disabled packs zeros
/// (harmless: the shader never reads tint while glow_params.x is 0).
pub fn glowTintParams(cfg: options.PostProcessOptions) [4]f32 {
    if (!cfg.glow_enabled) return .{ 0.0, 0.0, 0.0, 0.0 };
    const c = cfg.clamped();
    return .{ c.glow_tint[0], c.glow_tint[1], c.glow_tint[2], 0.0 };
}

test "glow defaults are off and neutral" {
    const cfg = options.PostProcessOptions{};
    // Default OFF: disabled glow runs zero passes and packs zero uniforms,
    // so the composite stays bit-identical to the pre-glow path.
    try std.testing.expect(!cfg.glow_enabled);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cfg.glow_threshold, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), cfg.glow_intensity, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), cfg.glow_radius, 1e-6);
    try std.testing.expectEqual([3]f32{ 1.0, 1.0, 1.0 }, cfg.glow_tint);
    // Distinct from bloom: threshold above bloom's 0.8, radius wider than
    // bloom's 2.0.
    try std.testing.expect(cfg.glow_threshold > 0.8);
    try std.testing.expect(cfg.glow_radius > 2.0);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, glowParams(cfg));
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, glowTintParams(cfg));
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, glowParams(cfg.clamped()));
    try std.testing.expect(!glowActive(true, cfg));
    try std.testing.expect(!glowActive(false, cfg));
}

test "glow clamped sanitizes ranges like bloom" {
    var cfg = options.PostProcessOptions{
        .glow_threshold = -1.0,
        .glow_intensity = -0.5,
        .glow_radius = -4.0,
        .glow_tint = .{ 2.0, -1.0, 0.5 },
    };
    const out = cfg.clamped();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.glow_threshold, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.glow_intensity, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.glow_radius, 1e-6);
    try std.testing.expectEqual([3]f32{ 1.0, 0.0, 0.5 }, out.glow_tint);
    // Bloom's own sanitization is untouched by the glow fields.
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), out.bloom_threshold, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out.bloom_intensity, 1e-6);
}

test "glow validate rejects non-finite and out-of-range" {
    // Defaults validate clean.
    try validateGlow(options.PostProcessOptions{});
    try validateGlow(options.PostProcessOptions{ .glow_enabled = true });

    // NaN/Inf can never sanitize: hard errors, never silent clamps.
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    try std.testing.expectError(error.InvalidGlowOptions, validateGlow(.{ .glow_threshold = nan }));
    try std.testing.expectError(error.InvalidGlowOptions, validateGlow(.{ .glow_intensity = inf }));
    try std.testing.expectError(error.InvalidGlowOptions, validateGlow(.{ .glow_radius = nan }));
    try std.testing.expectError(error.InvalidGlowOptions, validateGlow(.{ .glow_tint = .{ 1.0, nan, 1.0 } }));

    // Finite out-of-range values are rejected here (clamped() is the
    // sanitizing path for those, mirroring bloom).
    try std.testing.expectError(error.InvalidGlowOptions, validateGlow(.{ .glow_threshold = -0.1 }));
    try std.testing.expectError(error.InvalidGlowOptions, validateGlow(.{ .glow_intensity = -1.0 }));
    try std.testing.expectError(error.InvalidGlowOptions, validateGlow(.{ .glow_radius = -2.0 }));
    try std.testing.expectError(error.InvalidGlowOptions, validateGlow(.{ .glow_tint = .{ 1.0, 1.0, 1.5 } }));
    try std.testing.expectError(error.InvalidGlowOptions, validateGlow(.{ .glow_tint = .{ -0.1, 0.0, 0.0 } }));
}

test "glow extract and gaussian kernel math" {
    // Below threshold extracts black (soft bright-pass).
    try std.testing.expectEqual([3]f32{ 0.0, 0.0, 0.0 }, glowExtract(.{ 0.2, 0.3, 0.4 }, 1.0));
    // At double the threshold the over-threshold fraction is 1/2.
    const hot = glowExtract(.{ 2.0, 2.0, 2.0 }, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), hot[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), hot[1], 1e-5);
    // Black stays black for any threshold (no divide-by-zero).
    try std.testing.expectEqual([3]f32{ 0.0, 0.0, 0.0 }, glowExtract(.{ 0.0, 0.0, 0.0 }, 0.0));

    // Gaussian kernel: center is the peak, symmetric, normalized sum is 1.
    const sigma: f32 = GLOW_RADIUS_DEFAULT;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), glowGaussianWeight(0.0, sigma), 1e-6);
    try std.testing.expect(glowGaussianWeight(1.0, sigma) < 1.0);
    try std.testing.expectApproxEqAbs(
        glowGaussianWeight(2.0, sigma),
        glowGaussianWeight(-2.0, sigma),
        1e-6,
    );
    try std.testing.expect(glowGaussianWeight(4.0, sigma) > 0.0);
    var norm: f32 = 0.0;
    var i: i32 = -GLOW_HALF_TAPS;
    while (i <= GLOW_HALF_TAPS) : (i += 1) {
        norm += glowGaussianWeight(@floatFromInt(i), sigma) / glowKernelSum(sigma);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, 1e-6);
    // Wider radius spreads the weight outward (edge taps gain share).
    const tight = glowGaussianWeight(4.0, 1.0) / glowKernelSum(1.0);
    const wide = glowGaussianWeight(4.0, 8.0) / glowKernelSum(8.0);
    try std.testing.expect(wide > tight);
}

test "glow params and tint packing, independent of bloom" {
    // Enabled packs the clamped intensity and tint.
    const on = options.PostProcessOptions{ .glow_enabled = true };
    try std.testing.expectEqual([4]f32{ 1.0, 0.5, 0.0, 0.0 }, glowParams(on));
    try std.testing.expectEqual([4]f32{ 1.0, 1.0, 1.0, 0.0 }, glowTintParams(on));

    // Out-of-range packs clamped (the shader additionally gates on
    // intensity > 0.001, so zero intensity is a no-op even when enabled).
    var hot = options.PostProcessOptions{ .glow_enabled = true, .glow_intensity = 9.0, .glow_tint = .{ 3.0, -2.0, 0.25 } };
    const hp = glowParams(hot);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), hp[1], 1e-6);
    const ht = glowTintParams(hot);
    try std.testing.expectEqual([4]f32{ 1.0, 0.0, 0.25, 0.0 }, ht);
    hot.glow_intensity = -4.0;
    try std.testing.expectEqual(@as(f32, 0.0), glowParams(hot)[1]);

    // Bloom knobs never leak into the glow uniforms and vice versa: the two
    // effects stay independently toggleable.
    const bloom_only = options.PostProcessOptions{ .bloom_enabled = true, .bloom_intensity = 0.9 };
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, glowParams(bloom_only));
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, glowTintParams(bloom_only));
    try std.testing.expect(bloom.bloomPyramidActive(true, bloom_only));
    try std.testing.expect(!glowActive(true, bloom_only));
}

test "glow and bloom pass decisions are independent" {
    const def = options.PostProcessOptions{};
    // Bloom uses one pyramid implementation; glow remains a separate effect.
    try std.testing.expect(bloom.bloomPyramidActive(true, def));
    try std.testing.expect(!bloom.bloomPyramidActive(false, def));
    try std.testing.expect(!glowActive(true, def));
    // Post off disables both regardless of their own toggles.
    const both_on = options.PostProcessOptions{ .glow_enabled = true };
    try std.testing.expect(!bloom.bloomPyramidActive(false, both_on));
    try std.testing.expect(!glowActive(false, both_on));
    try std.testing.expect(bloom.bloomPyramidActive(true, both_on));
    try std.testing.expect(glowActive(true, both_on));

    // Toggling glow never changes the bloom decision (and vice versa).
    var cfg = options.PostProcessOptions{};
    const bloom_before = bloom.bloomPyramidActive(true, cfg);
    cfg.glow_enabled = true;
    try std.testing.expectEqual(bloom_before, bloom.bloomPyramidActive(true, cfg));
    try std.testing.expect(glowActive(true, cfg));
    cfg.glow_enabled = false;
    try std.testing.expect(!glowActive(true, cfg));
    try std.testing.expectEqual(bloom_before, bloom.bloomPyramidActive(true, cfg));
    cfg.bloom_enabled = false;
    try std.testing.expect(!bloom.bloomPyramidActive(true, cfg));
    try std.testing.expect(!glowActive(true, cfg));

    // One glow frame is exactly extract + H + V draws (ordering constant
    // the renderChain stats pin); the glow target reuses bloom's half-res
    // level-0 sizing.
    try std.testing.expectEqual(@as(u32, 3), GLOW_PASS_DRAWS);
    try std.testing.expectEqual(types.BloomMipSize{ .w = 640, .h = 360 }, bloom.bloomMipSize(1280, 720, 0));
}
