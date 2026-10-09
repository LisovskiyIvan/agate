//! Screen-space one-bounce diffuse GI (v1 color bleed) — CPU mirrors of
//! the composite gather (postprocess.glsl `applySSGI`). Pure math only:
//! the shader runs the same weights; these functions pin them in tests and
//! feed the packed uniform lane. No GPU imports, no allocations.

const std = @import("std");
const PostProcessOptions = @import("options.zig").PostProcessOptions;

pub const SSGI_STEPS_MIN: u32 = 4;
pub const SSGI_STEPS_MAX: u32 = 32;
pub const SSGI_RADIUS_MIN: f32 = 0.1;
pub const SSGI_RADIUS_MAX: f32 = 8.0;
/// Per-sample HDR firefly cap (GLSL mirror: SSGI_LUMA_CAP in
/// postprocess.glsl). Bleed colors above this clamp per channel — an
/// emissive panel must not paint the whole room through one tap.
pub const SSGI_LUMA_CAP: f32 = 8.0;

/// Activity gate: enabled AND a non-zero intensity (the composite early-outs
/// on the same test through the packed lane).
pub fn ssgiActive(config: PostProcessOptions) bool {
    return config.ssgi_enabled and config.ssgi_intensity > 0.001;
}

/// Packed uniform lane: (enabled, intensity, radius, steps). Zeros when
/// inactive, which keeps the composite bit-identical to the pre-SSGI path.
pub fn ssgiParams(config: PostProcessOptions) [4]f32 {
    if (!ssgiActive(config)) return .{ 0.0, 0.0, 0.0, 0.0 };
    return .{
        1.0,
        config.ssgi_intensity,
        config.ssgi_radius,
        @floatFromInt(config.ssgi_steps),
    };
}

/// One gather sample weight: hemisphere cosine times the SQUARED linear
/// falloff. Mirrors `w = cos_nd * falloff * falloff` in applySSGI:
/// behind-hemisphere samples, samples at/past the radius, and degenerate
/// radii weigh nothing.
pub fn ssgiWeight(cos_nd: f32, dist: f32, radius: f32) f32 {
    if (cos_nd <= 0.0 or radius <= 0.0 or dist >= radius) return 0.0;
    const falloff = 1.0 - dist / radius;
    return cos_nd * falloff * falloff;
}

/// Gather coverage in [0, 1]: how much of the disk actually contributed.
/// Mirrors `clamp(total_w / steps * 3.0, 0.0, 1.0)` — an open scene (sky
/// pixels around) gathers almost nothing and adds no bleed; a corner
/// saturates quickly.
pub fn ssgiCoverage(total_w: f32, steps: u32) f32 {
    if (steps == 0) return 0.0;
    return std.math.clamp(total_w / @as(f32, @floatFromInt(steps)) * 3.0, 0.0, 1.0);
}

/// Final one-bounce additive term: the normalized bleed color (per-channel
/// luma-capped) scaled by coverage and intensity. Mirrors the composite
/// tail; a zero/degenerate gather adds exactly nothing.
pub fn ssgiBleedAdd(bleed_sum: [3]f32, total_w: f32, steps: u32, intensity: f32) [3]f32 {
    if (total_w <= 0.0001 or intensity <= 0.0) return .{ 0.0, 0.0, 0.0 };
    const coverage = ssgiCoverage(total_w, steps);
    var out: [3]f32 = undefined;
    for (0..3) |c| {
        const capped = @min(bleed_sum[c] / total_w, SSGI_LUMA_CAP);
        out[c] = capped * coverage * intensity;
    }
    return out;
}
