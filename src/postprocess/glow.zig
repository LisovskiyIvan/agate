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
