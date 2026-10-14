//! Screen-space reflections (SSR) — CPU mirrors of the composite raymarch
//! (postprocess.glsl `applySSR`) plus the opt-in half-res pass gates.
//! Pure math only: no GPU imports, no allocations.

const std = @import("std");
const PostProcessOptions = @import("options.zig").PostProcessOptions;

pub const SSR_STEPS_MIN: u32 = 4;
pub const SSR_STEPS_MAX: u32 = 64;
/// Historical inline defaults (were dropped knobs, now constants).
pub const SSR_THICKNESS_DEFAULT: f32 = 0.4;
pub const SSR_MAX_DISTANCE_DEFAULT: f32 = 25.0;
/// Floor-only surfaces reflect (world normal Y gate in applySSR).
pub const SSR_NORMAL_Y_MIN: f32 = 0.45;
/// Reflected ray must leave the surface (R.y gate in applySSR).
pub const SSR_REFLECT_Y_MIN: f32 = 0.02;
/// Bilateral upscale depth gate in meters (linearized depth).
pub const SSR_BLUR_DEPTH_GATE: f32 = 0.05;
/// Trace tap budget (mirrors SSR_STEPS_MAX so clamped config never overruns).
pub const SSR_TAPS_MAX: u32 = 64;

/// SSR execution mode. `inline_` runs the v1 raymarch inside the composite
/// (default, bit-identical); `half_res_pass` runs the dedicated half-res
/// trace + bilateral denoise/upscale (SsrPass) and mixes the denoised
/// reflection in the composite instead.
pub const SsrMode = enum(u8) {
    inline_ = 0,
    half_res_pass = 1,
};

/// Master gate: enabled AND a non-zero intensity (the composite early-outs
/// on the same test through the packed lane).
pub fn ssrActive(config: PostProcessOptions) bool {
    return config.ssr.enabled and config.ssr.intensity > 0.001;
}

/// Inline (v1) activity: master gate plus inline mode.
pub fn ssrInlineActive(config: PostProcessOptions) bool {
    return ssrActive(config) and config.ssr.mode == .inline_;
}

/// Pass activity: master gate plus pass mode. The chain keys the SsrPass
/// render off this; MSAA suppression clears ssr.enabled, so pass mode
/// joins the existing suppression with no extra plumbing.
pub fn ssrPassActive(config: PostProcessOptions) bool {
    return ssrActive(config) and config.ssr.mode == .half_res_pass;
}

/// Packed INLINE uniform lane: (enabled, intensity, thickness, max_dist).
/// Zeros when inactive OR when pass mode owns the effect, which keeps the
/// composite bit-identical to the pre-SSR path in both cases.
pub fn ssrParams(config: PostProcessOptions) [4]f32 {
    if (!ssrInlineActive(config)) return .{ 0.0, 0.0, 0.0, 0.0 };
    return .{ 1.0, config.ssr.intensity, SSR_THICKNESS_DEFAULT, SSR_MAX_DISTANCE_DEFAULT };
}

/// Packed INLINE steps lane: (steps, 0, 0, 0). Zeros when the inline lane
/// is inactive (off or pass mode); the shader early-outs on ssr_params.x
/// before reading this lane.
pub fn ssrStepsParams(config: PostProcessOptions) [4]f32 {
    if (!ssrInlineActive(config)) return .{ 0.0, 0.0, 0.0, 0.0 };
    return .{ @floatFromInt(config.ssr.steps), 0.0, 0.0, 0.0 };
}

/// Packed PASS-SAMPLE lane: (active 1/0, intensity, 0, 0). Zeros unless
/// the pass ran this frame (tex_valid: the parent feeds the denoised
/// view only then), so the off path is bit-identical. Intensity rides
/// raw — the shader gates on x before reading y, and the pass output
/// carries (reflection rgb, blend weight) with no intensity baked in.
pub fn ssrPassParams(config: PostProcessOptions, tex_valid: bool) [4]f32 {
    if (!ssrPassActive(config) or !tex_valid) return .{ 0.0, 0.0, 0.0, 0.0 };
    return .{ 1.0, config.ssr.intensity, 0.0, 0.0 };
}

/// Half-res pass target size for a full EFFECTIVE (render-scale) base
/// size: ceil-half per axis, mirroring the SSAO/SSGI raw target. Pure.
pub const SsrTargetSize = struct { w: i32, h: i32 };

pub fn ssrTargetSize(base_w: i32, base_h: i32) SsrTargetSize {
    if (base_w <= 0 or base_h <= 0) return .{ .w = base_w, .h = base_h };
    return .{
        .w = @divTrunc(base_w, 2) + @mod(base_w, 2),
        .h = @divTrunc(base_h, 2) + @mod(base_h, 2),
    };
}

/// VRAM census helper (pure byte math): two half-res targets
/// (trace + denoised) at `bytes_per_pixel`.
pub fn ssrTargetBytes(base_w: i32, base_h: i32, bytes_per_pixel: usize) usize {
    const size = ssrTargetSize(base_w, base_h);
    if (size.w <= 0 or size.h <= 0) return 0;
    const w: usize = @intCast(size.w);
    const h: usize = @intCast(size.h);
    return 2 * w * h * bytes_per_pixel;
}

/// Fresnel term mirror of applySSR:
/// `clamp((0.04 + 0.96 * (1 - cos_vn)^5) * 1.8, 0.15, 0.9)`.
pub fn ssrFresnel(cos_vn: f32) f32 {
    if (!std.math.isFinite(cos_vn)) return 0.15;
    const c = std.math.clamp(cos_vn, 0.0, 1.0);
    const f = 0.04 + 0.96 * std.math.pow(f32, 1.0 - c, 5.0);
    return std.math.clamp(f * 1.8, 0.15, 0.9);
}

/// Screen-edge fade mirror: `clamp(min(edge_x, edge_y) * 10, 0, 1)`.
pub fn ssrEdgeFade(march_uv: [2]f32) f32 {
    const ex = @min(march_uv[0], 1.0 - march_uv[0]);
    const ey = @min(march_uv[1], 1.0 - march_uv[1]);
    const m = @min(ex, ey);
    if (!std.math.isFinite(m)) return 0.0;
    return std.math.clamp(m * 10.0, 0.0, 1.0);
}

/// Distance fade mirror: `(1 - (i-1)/steps)^2`, 1 step behind the hit.
pub fn ssrDistFade(hit_step_1based: u32, steps: u32) f32 {
    if (steps == 0 or hit_step_1based == 0) return 0.0;
    const i: f32 = @floatFromInt(hit_step_1based);
    const n: f32 = @floatFromInt(steps);
    const f = 1.0 - (i - 1.0) / n;
    return f * f;
}

/// Full blend weight mirror: `edge * dist * fresnel` (intensity stays in
/// the composite). Non-finite inputs fade to zero (fail closed).
pub fn ssrBlendWeight(edge_fade: f32, dist_fade: f32, fresnel: f32) f32 {
    if (!std.math.isFinite(edge_fade) or !std.math.isFinite(dist_fade) or !std.math.isFinite(fresnel)) return 0.0;
    return edge_fade * dist_fade * fresnel;
}

/// Composite mix mirror: `mix(scene, refl, intensity * weight)`.
pub fn ssrMix(scene: [3]f32, refl: [3]f32, intensity: f32, weight: f32) [3]f32 {
    const k = std.math.clamp(intensity * weight, 0.0, 1.0);
    return .{
        scene[0] + (refl[0] - scene[0]) * k,
        scene[1] + (refl[1] - scene[1]) * k,
        scene[2] + (refl[2] - scene[2]) * k,
    };
}

/// Bilateral upscale weight mirror (ssr_blur.glsl): spatial Gaussian
/// (sigma ~1.5) times the linear depth tent. Zero outside the gate.
pub fn ssrBilateralWeight(delta_texels: [2]f32, depth_diff: f32) f32 {
    if (!std.math.isFinite(depth_diff) or !std.math.isFinite(delta_texels[0]) or !std.math.isFinite(delta_texels[1])) return 0.0;
    if (depth_diff < 0.0 or depth_diff >= SSR_BLUR_DEPTH_GATE) return 0.0;
    const d2 = delta_texels[0] * delta_texels[0] + delta_texels[1] * delta_texels[1];
    return @exp(-d2 * 0.5) * (1.0 - depth_diff / SSR_BLUR_DEPTH_GATE);
}

/// One tap of a joint-bilateral SSR resolve (denoise 3x3 or composite
/// 2x2): the reflection + weight at a half-res texel center, its
/// texel-space offset from the center pixel, and its linearized-depth
/// difference from the center pixel. Shared by both stages so the blur
/// output matches the upscale input representation exactly.
pub const SsrUpscaleTap = struct {
    rgb: [3]f32,
    weight_a: f32,
    delta_texels: [2]f32,
    depth_diff: f32,
};

/// Joint-bilateral SSR resolve value: confidence-normalized reflection
/// plus coverage weight. Named once so the denoise and upscale mirrors
/// share one representation (anonymous structs are nominal in Zig).
pub const SsrResolve = struct { rgb: [3]f32, a: f32 };

/// Shared joint-bilateral normalization: rgb normalizes by the
/// reflection-confidence accumulator (sum rgb*a*w / sum a*w), alpha by
/// the spatial/depth weight (sum a*w / sum w). Zero rgb with no
/// confidence; zeros when nothing accumulates. Pure.
fn ssrJointBilateral(taps: []const SsrUpscaleTap) SsrResolve {
    var acc = [3]f32{ 0.0, 0.0, 0.0 };
    var acc_a: f32 = 0.0;
    var wsum: f32 = 0.0;
    for (taps) |t| {
        const w = ssrBilateralWeight(t.delta_texels, t.depth_diff);
        for (0..3) |c| acc[c] += t.rgb[c] * t.weight_a * w;
        acc_a += t.weight_a * w;
        wsum += w;
    }
    if (wsum <= 0.0001) return .{ .rgb = .{ 0.0, 0.0, 0.0 }, .a = 0.0 };
    const a = acc_a / wsum;
    if (acc_a <= 0.0001) return .{ .rgb = .{ 0.0, 0.0, 0.0 }, .a = a };
    return .{
        .rgb = .{ acc[0] / acc_a, acc[1] / acc_a, acc[2] / acc_a },
        .a = a,
    };
}

/// Full-res reconstruction mirror of sampleSSRPass: joint bilateral over
/// the 2x2 footprint. The caller mixes by intensity * a exactly like the
/// inline path mixes by intensity * edge * dist * fresnel.
/// All-rejected footprints resolve to zeros (the caller mixes nothing).
/// Pure.
pub fn ssrUpscaleResolve(taps: [4]SsrUpscaleTap) SsrResolve {
    return ssrJointBilateral(taps[0..]);
}

/// Denoise mirror of ssr_blur.glsl: joint bilateral over the 3x3 trace
/// footprint, same normalization as the upscale stage (a miss tap dilutes
/// coverage but never darkens a valid reflection). Pure.
pub fn ssrDenoiseResolve(taps: [9]SsrUpscaleTap) SsrResolve {
    return ssrJointBilateral(taps[0..]);
}

/// Half-res geometry as the composite derives it from the full EFFECTIVE
/// size: floor((w + 1) / 2) per axis — the float form of the ceil-half
/// ssrTargetSize rule, so the 2x2 footprint lands on real texel centers
/// with no extra uniform. Pure.
pub const SsrHalfResSize = struct { w: f32, h: f32 };

pub fn ssrHalfResSize(full_w: f32, full_h: f32) SsrHalfResSize {
    return .{
        .w = @floor((full_w + 1.0) * 0.5),
        .h = @floor((full_h + 1.0) * 0.5),
    };
}
