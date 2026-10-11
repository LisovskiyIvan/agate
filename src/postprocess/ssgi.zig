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
/// Compile-time tap budget of the v2 half-res gather shader (ssgi.glsl):
/// the loop runs `for (i < 32) break at steps`, so a clamped steps count
/// can never overrun the loop. Equals SSGI_STEPS_MAX by construction.
pub const SSGI_GATHER_TAPS_MAX: u32 = 32;

/// SSGI execution mode. `inline_gather` runs the v1 gather inside the
/// composite (default, bit-identical); `half_res_pass` runs the v2
/// dedicated half-res gather + bilateral denoise (SsgiPass) and samples
/// the denoised texture in the composite. Temporal accumulation is a
/// documented v2.1 item, not part of either mode.
pub const SsgiMode = enum(u8) {
    inline_gather = 0,
    half_res_pass = 1,
};
/// Per-sample HDR firefly cap (GLSL mirror: SSGI_LUMA_CAP in
/// postprocess.glsl). Bleed colors above this clamp per channel — an
/// emissive panel must not paint the whole room through one tap.
pub const SSGI_LUMA_CAP: f32 = 8.0;

/// Activity gate: enabled AND a non-zero intensity (the composite early-outs
/// on the same test through the packed lane).
pub fn ssgiActive(config: PostProcessOptions) bool {
    return config.ssgi_enabled and config.ssgi_intensity > 0.001;
}

/// Inline (v1) activity: the master gate plus inline mode. The composite
/// gather branch keys off this lane, so pass mode never double-adds.
pub fn ssgiInlineActive(config: PostProcessOptions) bool {
    return ssgiActive(config) and config.ssgi_mode == .inline_gather;
}

/// Pass (v2) activity: the master gate plus pass mode. The chain keys the
/// SsgiPass render off this; MSAA suppression clears ssgi_enabled, so
/// pass mode joins the existing suppression with no extra plumbing.
pub fn ssgiPassActive(config: PostProcessOptions) bool {
    return ssgiActive(config) and config.ssgi_mode == .half_res_pass;
}

/// Packed INLINE uniform lane: (enabled, intensity, radius, steps). Zeros
/// when inactive OR when pass mode owns the effect, which keeps the
/// composite bit-identical to the pre-SSGI path in both cases.
pub fn ssgiParams(config: PostProcessOptions) [4]f32 {
    if (!ssgiInlineActive(config)) return .{ 0.0, 0.0, 0.0, 0.0 };
    return .{
        1.0,
        config.ssgi_intensity,
        config.ssgi_radius,
        @floatFromInt(config.ssgi_steps),
    };
}

/// Packed PASS-SAMPLE lane: (active 1/0, intensity, 0, 0). Zeros unless
/// the pass ran this frame (tex_valid: the parent feeds the denoised
/// view only then), so the off path is bit-identical. Intensity rides
/// raw — the shader gates on x before reading y, and the pass output
/// carries normalized bleed * coverage with no intensity baked in.
pub fn ssgiPassParams(config: PostProcessOptions, tex_valid: bool) [4]f32 {
    if (!ssgiPassActive(config) or !tex_valid) return .{ 0.0, 0.0, 0.0, 0.0 };
    return .{ 1.0, config.ssgi_intensity, 0.0, 0.0 };
}

/// Half-res pass target size for a full EFFECTIVE (render-scale) base
/// size: ceil-half per axis, mirroring the SSAO raw target. Pure (no GPU),
/// so the pass and the census share one sizing rule.
pub const SsgiTargetSize = struct { w: i32, h: i32 };

pub fn ssgiTargetSize(base_w: i32, base_h: i32) SsgiTargetSize {
    if (base_w <= 0 or base_h <= 0) return .{ .w = base_w, .h = base_h };
    return .{
        .w = @divTrunc(base_w, 2) + @mod(base_w, 2),
        .h = @divTrunc(base_h, 2) + @mod(base_h, 2),
    };
}

/// VRAM census helper (pure byte math, no GPU calls, headless-safe): two
/// half-res targets (gather + denoised) at `bytes_per_pixel`. Single
/// source of truth for profiler/snapshot.zig.
pub fn ssgiTargetBytes(base_w: i32, base_h: i32, bytes_per_pixel: usize) usize {
    const size = ssgiTargetSize(base_w, base_h);
    if (size.w <= 0 or size.h <= 0) return 0;
    const w: usize = @intCast(size.w);
    const h: usize = @intCast(size.h);
    return 2 * w * h * bytes_per_pixel;
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
