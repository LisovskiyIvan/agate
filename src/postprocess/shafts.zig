const std = @import("std");
const types = @import("types.zig");
const bloom = @import("bloom.zig");
const glow = @import("glow.zig");
const options = @import("options.zig");

pub const ShaftResolution = types.ShaftResolution;
pub const ShaftTargetSize = types.ShaftTargetSize;

/// Raymarch step clamp range (the shader loop bound is SHAFT_MARCH_MAX).
pub const SHAFT_STEPS_MIN: u32 = 4;
pub const SHAFT_STEPS_MAX: u32 = 32;
/// Hard loop bound mirrored by the raymarch shader (steps break out early).
pub const SHAFT_MARCH_MAX: u32 = 32;
/// Fullscreen draws one shaft frame issues: raymarch + bilateral H + V.
pub const SHAFT_PASS_DRAWS: u32 = 3;
/// Anisotropy clamp (|g| < 1 keeps the HG denominator nonzero).
pub const SHAFT_ANISOTROPY_MAX: f32 = 0.9;
/// Bilateral blur taps per axis (mirrors volumetric_blur.glsl, same 9-tap
/// shape as the glow blur so the kernel sum helper stays shared).
pub const SHAFT_BLUR_TAPS: u32 = 9;
pub const SHAFT_BLUR_HALF_TAPS: i32 = 4;

/// Shaft target size for a base framebuffer: half is bloomMipSize level 0,
/// quarter halves once more, clamped to 1x1. Mirrors VolumetricPass.resize.
pub fn shaftTargetSize(base_w: i32, base_h: i32, res: ShaftResolution) ShaftTargetSize {
    const half = bloom.bloomMipSize(base_w, base_h, 0);
    switch (res) {
        .half => return .{ .w = half.w, .h = half.h },
        .quarter => return .{
            .w = @max(1, @divTrunc(half.w, 2)),
            .h = @max(1, @divTrunc(half.h, 2)),
        },
    }
}

/// Pass-construction decision (pure; PostFXStack.renderChain gates PASS 2.9
/// on this). Post off, shaft off, or no rendered CSM atlas (shadows off)
/// runs zero passes and binds the placeholder (bit-identical composite).
pub fn shaftActive(post_enabled: bool, cfg: options.PostProcessOptions, shadows_enabled: bool) bool {
    return post_enabled and cfg.shaft_enabled and shadows_enabled;
}

/// Pack the composite shaft_params vec4: (enabled 1/0, intensity, 0, 0).
/// No valid raymarch result (including shadows-off/fail-closed) or a disabled
/// config packs zeros; the scene-texture binding used as a placeholder must
/// never be mistaken for shaft radiance.
pub fn shaftParams(cfg: options.PostProcessOptions, result_available: bool) [4]f32 {
    if (!cfg.shaft_enabled or !result_available) return .{ 0.0, 0.0, 0.0, 0.0 };
    const c = cfg.clamped();
    return .{ 1.0, c.shaft_intensity, 0.0, 0.0 };
}

/// Strict range check for the shaft knobs. Finite out-of-range values are
/// the clamped() domain (bloom/glow precedent: silent sanitize on load);
/// non-finite values (NaN/Inf) can never sanitize meaningfully, so they
/// are a hard InvalidShaftOptions error here instead of a silent clamp.
pub fn validateShaft(cfg: options.PostProcessOptions) !void {
    if (!std.math.isFinite(cfg.shaft_intensity)) return error.InvalidShaftOptions;
    if (!std.math.isFinite(cfg.shaft_density)) return error.InvalidShaftOptions;
    if (!std.math.isFinite(cfg.shaft_anisotropy)) return error.InvalidShaftOptions;
    if (!std.math.isFinite(cfg.shaft_max_distance)) return error.InvalidShaftOptions;
    if (!std.math.isFinite(cfg.shaft_blur_sigma)) return error.InvalidShaftOptions;
    if (!std.math.isFinite(cfg.shaft_edge_sigma)) return error.InvalidShaftOptions;
    if (cfg.shaft_intensity < 0.0) return error.InvalidShaftOptions;
    if (cfg.shaft_density < 0.0) return error.InvalidShaftOptions;
    if (@abs(cfg.shaft_anisotropy) > SHAFT_ANISOTROPY_MAX) return error.InvalidShaftOptions;
    if (cfg.shaft_max_distance < 0.0) return error.InvalidShaftOptions;
    if (cfg.shaft_blur_sigma < 0.0) return error.InvalidShaftOptions;
    if (cfg.shaft_edge_sigma < 0.0) return error.InvalidShaftOptions;
    if (cfg.shaft_steps < SHAFT_STEPS_MIN or cfg.shaft_steps > SHAFT_MARCH_MAX) return error.InvalidShaftOptions;
}

/// Henyey-Greenstein phase term for cos_theta = dot(view_ray, sun_dir)
/// (both toward the scene: view ray from the camera, sun dir toward the
/// sun) and anisotropy g. Mirrors hgPhase in volumetric_raymarch.glsl so
/// CPU tests pin the exact shader formula.
pub fn hgPhase(cos_theta: f32, g: f32) f32 {
    const gg = g * g;
    const denom = 1.0 + gg - 2.0 * g * cos_theta;
    return (1.0 - gg) / (4.0 * std.math.pi * denom * @sqrt(@max(denom, 1e-6)));
}

/// CSM cascade index for a camera-distance sample. Mirrors
/// calculateShadow in the forward shaders (and shaftCascade in
/// volumetric_raymarch.glsl): distance-ordered splits, cascade 3 past
/// split 2, so the march taps the same tile the surface shader uses.
pub fn shaftCascadeIndex(view_dist: f32, splits: [4]f32) usize {
    if (view_dist < splits[0]) return 0;
    if (view_dist < splits[1]) return 1;
    if (view_dist < splits[2]) return 2;
    return 3;
}

/// One bilateral blur tap weight: spatial Gaussian (offset in shaft texels,
/// sigma floored at 0.5 exactly like the shader) times a raw-depth gate
/// (edge_sigma <= 0 disables the gate: plain Gaussian). Mirrors
/// volumetric_blur.glsl; both sides divide by the kernel sum.
pub fn shaftBilateralWeight(offset: f32, sigma_spatial: f32, depth_diff: f32, sigma_depth: f32) f32 {
    const s = @max(sigma_spatial, 0.5);
    const t = offset / s;
    const spatial = @exp(-0.5 * t * t);
    if (sigma_depth <= 0.0) return spatial;
    const d = depth_diff / sigma_depth;
    return spatial * @exp(-0.5 * d * d);
}

/// Kernel normalization both blur stages apply: sum of
/// shaftBilateralWeight over the taps at zero depth difference (the
/// center-pixel normalization; per-tap depth gates only shrink weights).
pub fn shaftKernelSum(sigma_spatial: f32) f32 {
    var sum: f32 = 0.0;
    var i: i32 = -SHAFT_BLUR_HALF_TAPS;
    while (i <= SHAFT_BLUR_HALF_TAPS) : (i += 1) {
        sum += shaftBilateralWeight(@floatFromInt(i), sigma_spatial, 0.0, 0.0);
    }
    return sum;
}
