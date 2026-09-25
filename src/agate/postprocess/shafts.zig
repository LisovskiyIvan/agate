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

test "shaft defaults are off and bit-identical" {
    const cfg = options.PostProcessOptions{};
    try std.testing.expect(!cfg.shaft_enabled);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cfg.shaft_intensity, 1e-6);
    try std.testing.expectEqual(@as(u32, 12), cfg.shaft_steps);
    try std.testing.expectApproxEqAbs(@as(f32, 0.05), cfg.shaft_density, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), cfg.shaft_anisotropy, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 60.0), cfg.shaft_max_distance, 1e-6);
    try std.testing.expectEqual(ShaftResolution.quarter, cfg.shaft_resolution);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), cfg.shaft_blur_sigma, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.02), cfg.shaft_edge_sigma, 1e-6);
    // Off packs zeros: the composite never samples the shaft texture.
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, shaftParams(cfg, true));
    try std.testing.expect(!shaftActive(true, cfg, true));
    try std.testing.expect(!shaftActive(false, options.PostProcessOptions{ .shaft_enabled = true }, true));
    try std.testing.expect(!shaftActive(true, options.PostProcessOptions{ .shaft_enabled = true }, false));
    try std.testing.expect(shaftActive(true, options.PostProcessOptions{ .shaft_enabled = true }, true));
    // A configured effect without a successful CSM-backed result must not
    // sample the scene-view placeholder as if it were shaft radiance.
    try std.testing.expectEqual(
        [4]f32{ 0.0, 0.0, 0.0, 0.0 },
        shaftParams(options.PostProcessOptions{ .shaft_enabled = true }, false),
    );
}

test "shaft config clamped sanitizes ranges" {
    var cfg = options.PostProcessOptions{
        .shaft_intensity = -1.0,
        .shaft_steps = 100,
        .shaft_density = -0.5,
        .shaft_anisotropy = 2.0,
        .shaft_max_distance = -10.0,
        .shaft_blur_sigma = -3.0,
        .shaft_edge_sigma = -0.1,
    };
    const out = cfg.clamped();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.shaft_intensity, 1e-6);
    try std.testing.expectEqual(SHAFT_STEPS_MAX, out.shaft_steps);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.shaft_density, 1e-6);
    try std.testing.expectApproxEqAbs(SHAFT_ANISOTROPY_MAX, out.shaft_anisotropy, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.shaft_max_distance, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.shaft_blur_sigma, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.shaft_edge_sigma, 1e-6);
    cfg.shaft_steps = 0;
    try std.testing.expectEqual(SHAFT_STEPS_MIN, cfg.clamped().shaft_steps);
    cfg.shaft_anisotropy = -2.0;
    try std.testing.expectApproxEqAbs(-SHAFT_ANISOTROPY_MAX, cfg.clamped().shaft_anisotropy, 1e-6);
    // Enabled packs the clamped intensity.
    const on = options.PostProcessOptions{ .shaft_enabled = true, .shaft_intensity = 2.5 };
    try std.testing.expectEqual([4]f32{ 1.0, 2.5, 0.0, 0.0 }, shaftParams(on, true));
}

test "shaft validation rejects non-finite and out-of-range" {
    const ok = options.PostProcessOptions{ .shaft_enabled = true };
    try validateShaft(ok);
    var bad = ok;
    bad.shaft_intensity = std.math.nan(f32);
    try std.testing.expectError(error.InvalidShaftOptions, validateShaft(bad));
    bad = ok;
    bad.shaft_anisotropy = 1.0;
    try std.testing.expectError(error.InvalidShaftOptions, validateShaft(bad));
    bad = ok;
    bad.shaft_steps = 3;
    try std.testing.expectError(error.InvalidShaftOptions, validateShaft(bad));
    bad = ok;
    bad.shaft_steps = 33;
    try std.testing.expectError(error.InvalidShaftOptions, validateShaft(bad));
    bad = ok;
    bad.shaft_density = -1.0;
    try std.testing.expectError(error.InvalidShaftOptions, validateShaft(bad));
    bad = ok;
    bad.shaft_edge_sigma = std.math.inf(f32);
    try std.testing.expectError(error.InvalidShaftOptions, validateShaft(bad));
}

test "hg phase golden values" {
    // Isotropic (g = 0): uniform 1/(4π) whatever the angle.
    const iso = 1.0 / (4.0 * std.math.pi);
    try std.testing.expectApproxEqAbs(iso, hgPhase(1.0, 0.0), 1e-6);
    try std.testing.expectApproxEqAbs(iso, hgPhase(-1.0, 0.0), 1e-6);
    try std.testing.expectApproxEqAbs(iso, hgPhase(0.0, 0.0), 1e-6);
    // Forward scattering peaks looking toward the sun.
    const fwd = hgPhase(1.0, 0.4);
    try std.testing.expect(fwd > hgPhase(0.0, 0.4));
    try std.testing.expect(hgPhase(0.0, 0.4) > hgPhase(-1.0, 0.4));
    // Negative g mirrors: back-scatter peak.
    try std.testing.expectApproxEqAbs(fwd, hgPhase(-1.0, -0.4), 1e-5);
    // Spot value: g = 0.4 straight into the sun: (1-0.16)/(4π·0.36^1.5).
    try std.testing.expectApproxEqAbs(@as(f32, 0.3095), fwd, 1e-3);
    // Always finite and positive inside the clamp range.
    var g: f32 = -0.9;
    while (g <= 0.9) : (g += 0.1) {
        const p = hgPhase(0.7, g);
        try std.testing.expect(std.math.isFinite(p) and p > 0.0);
    }
}

test "shaft cascade index mirrors the forward splits" {
    const splits = [4]f32{ 10.0, 26.0, 65.0, 150.0 };
    try std.testing.expectEqual(@as(usize, 0), shaftCascadeIndex(5.0, splits));
    try std.testing.expectEqual(@as(usize, 0), shaftCascadeIndex(9.99, splits));
    try std.testing.expectEqual(@as(usize, 1), shaftCascadeIndex(10.0, splits));
    try std.testing.expectEqual(@as(usize, 1), shaftCascadeIndex(25.99, splits));
    try std.testing.expectEqual(@as(usize, 2), shaftCascadeIndex(26.0, splits));
    try std.testing.expectEqual(@as(usize, 3), shaftCascadeIndex(65.0, splits));
    try std.testing.expectEqual(@as(usize, 3), shaftCascadeIndex(1000.0, splits));
}

test "shaft target sizes halve and quarter" {
    try std.testing.expectEqual(ShaftTargetSize{ .w = 640, .h = 360 }, shaftTargetSize(1280, 720, .half));
    try std.testing.expectEqual(ShaftTargetSize{ .w = 320, .h = 180 }, shaftTargetSize(1280, 720, .quarter));
    // Quarter of half-res level 0, never below 1x1.
    try std.testing.expectEqual(ShaftTargetSize{ .w = 1, .h = 1 }, shaftTargetSize(3, 3, .quarter));
    const q = shaftTargetSize(1920, 1080, .quarter);
    try std.testing.expect(q.w >= 1 and q.h >= 1);
    // Doubling both dims quadruples the census area either way.
    const h1 = shaftTargetSize(640, 360, .half);
    const h2 = shaftTargetSize(1280, 720, .half);
    try std.testing.expectEqual(h1.w * 2, h2.w);
    try std.testing.expectEqual(h1.h * 2, h2.h);
}

test "shaft bilateral weights gate on depth edges" {
    // Center tap with no depth step is exactly 1.
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), shaftBilateralWeight(0.0, 2.0, 0.0, 0.02), 1e-6);
    // Spatial falloff matches the glow Gaussian at zero depth difference.
    try std.testing.expectApproxEqAbs(glow.glowGaussianWeight(2.0, 2.0), shaftBilateralWeight(2.0, 2.0, 0.0, 0.02), 1e-6);
    // A depth step far beyond the edge sigma kills the tap.
    try std.testing.expect(shaftBilateralWeight(0.0, 2.0, 1.0, 0.02) < 1e-6);
    // Zero edge sigma disables the gate: plain Gaussian everywhere.
    try std.testing.expectApproxEqAbs(glow.glowGaussianWeight(1.0, 2.0), shaftBilateralWeight(1.0, 2.0, 1.0, 0.0), 1e-6);
    // Kernel is normalized the same way (center normalization).
    var sum: f32 = 0.0;
    var i: i32 = -SHAFT_BLUR_HALF_TAPS;
    while (i <= SHAFT_BLUR_HALF_TAPS) : (i += 1) {
        sum += glow.glowGaussianWeight(@floatFromInt(i), 2.0);
    }
    try std.testing.expectApproxEqAbs(sum, shaftKernelSum(2.0), 1e-6);
}
