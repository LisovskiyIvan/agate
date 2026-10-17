//! Screen-space one-bounce diffuse GI (v1 color bleed) — CPU mirrors of
//! the composite gather (postprocess.glsl `applySSGI`). Pure math only:
//! the shader runs the same weights; these functions pin them in tests and
//! feed the packed uniform lane. No GPU imports, no allocations.

const std = @import("std");
const PostProcessOptions = @import("options.zig").PostProcessOptions;
const TaaReset = @import("types.zig").TaaReset;
const TaaBounds = @import("types.zig").TaaBounds;

pub const SSGI_STEPS_MIN: u32 = 4;
pub const SSGI_STEPS_MAX: u32 = 32;
pub const SSGI_RADIUS_MIN: f32 = 0.1;
pub const SSGI_RADIUS_MAX: f32 = 8.0;
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
    return config.ssgi.enabled and config.ssgi.intensity > 0.001;
}

/// Inline (v1) activity: the master gate plus inline mode. The composite
/// gather branch keys off this lane, so pass mode never double-adds.
pub fn ssgiInlineActive(config: PostProcessOptions) bool {
    return ssgiActive(config) and config.ssgi.mode == .inline_gather;
}

/// Pass (v2) activity: the master gate plus pass mode. The chain keys the
/// SsgiPass render off this; MSAA suppression clears ssgi.enabled, so
/// pass mode joins the existing suppression with no extra plumbing.
pub fn ssgiPassActive(config: PostProcessOptions) bool {
    return ssgiActive(config) and config.ssgi.mode == .half_res_pass;
}

/// Packed INLINE uniform lane: (enabled, intensity, radius, steps). Zeros
/// when inactive OR when pass mode owns the effect, which keeps the
/// composite bit-identical to the pre-SSGI path in both cases.
pub fn ssgiParams(config: PostProcessOptions) [4]f32 {
    if (!ssgiInlineActive(config)) return .{ 0.0, 0.0, 0.0, 0.0 };
    return .{
        1.0,
        config.ssgi.intensity,
        config.ssgi.radius,
        @floatFromInt(config.ssgi.steps),
    };
}

/// Packed PASS-SAMPLE lane: (active 1/0, intensity, 0, 0). Zeros unless
/// the pass ran this frame (tex_valid: the parent feeds the denoised
/// view only then), so the off path is bit-identical. Intensity rides
/// raw — the shader gates on x before reading y, and the pass output
/// carries normalized bleed * coverage with no intensity baked in.
pub fn ssgiPassParams(config: PostProcessOptions, tex_valid: bool) [4]f32 {
    if (!ssgiPassActive(config) or !tex_valid) return .{ 0.0, 0.0, 0.0, 0.0 };
    return .{ 1.0, config.ssgi.intensity, 0.0, 0.0 };
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

/// Receiver-color proxy used by the composite after the normalized gather:
/// normalize scene color to its dominant channel (CPU mirror of
/// `ssgiReceiverTint` in postprocess.glsl). A dark/NaN receiver falls back
/// to white, i.e. the un-tinted legacy result, never to an invalid color.
pub fn ssgiReceiverTint(color: [3]f32) [3]f32 {
    var peak: f32 = 0.0;
    for (color) |c| {
        if (std.math.isFinite(c)) peak = @max(peak, @max(c, 0.0));
    }
    if (peak < 1e-4) return .{ 1.0, 1.0, 1.0 };
    var out: [3]f32 = undefined;
    for (color, 0..) |c, i| {
        out[i] = @max(if (std.math.isFinite(c)) c else 0.0, 0.0) / peak;
    }
    return out;
}

/// Relative halo cap for the additive SSGI term: the one-bounce add cannot
/// exceed 1.5x the local direct radiance, with a 0.12 floor so shadowed
/// surfaces still receive fill. CPU mirror of the composite clamp.
pub fn ssgiHaloCap(scene_color: [3]f32) [3]f32 {
    var out: [3]f32 = undefined;
    for (scene_color, 0..) |c, i| {
        const v = if (std.math.isFinite(c)) @max(c, 0.12) else 0.12;
        out[i] = v * 1.5;
    }
    return out;
}

/// Full composite add: core `ssgiBleedAdd` (firefly cap + coverage *
/// intensity), then receiver tint and halo cap. This is the shader's
/// `scene_color + min(indirect, cap)` tail.
pub fn ssgiBleedAddTinted(
    bleed_sum: [3]f32,
    total_w: f32,
    steps: u32,
    intensity: f32,
    receiver: [3]f32,
) [3]f32 {
    const base = ssgiBleedAdd(bleed_sum, total_w, steps, intensity);
    const tint = ssgiReceiverTint(receiver);
    const cap = ssgiHaloCap(receiver);
    var out: [3]f32 = undefined;
    for (0..3) |c| out[c] = @min(base[c] * tint[c], cap[c]);
    return out;
}

// --- SSGI v2.1 temporal accumulation (opt-in, half_res_pass only) ---
//
// The temporal stage sits between the gather and the bilateral denoise:
// gather -> temporal -> denoise -> composite. History holds the RAW
// accumulated bleed signal (normalized bleed * coverage, no intensity),
// exactly the gather lane; the bilateral denoise stays the final cleaner
// before the composite, so the composite still scales one denoised signal
// by intensity after all temporal work.

/// Recommended history-weight window. The option clamps to [0, 1]; below
/// the window the accumulation converges too slowly to matter, above it
/// disocclusion trails survive the rejection gates.
pub const SSGI_TEMPORAL_BLEND_MIN: f32 = 0.85;
pub const SSGI_TEMPORAL_BLEND_MAX: f32 = 0.95;
/// Default history weight (same compromise class as taa_blend 0.9).
pub const SSGI_TEMPORAL_BLEND_DEFAULT: f32 = 0.9;

/// Variance-clip tightness: history must survive mean +/- gamma * stddev
/// of the current 3x3 gather neighborhood (TAA uses 1.25 on radiance; the
/// low-frequency bleed signal tolerates a tighter box).
pub const SSGI_TEMPORAL_GAMMA: f32 = 1.0;

/// Normal-consistency gate: history survives only when the current surface
/// normal and the normal reconstructed at the reprojected UV agree to this
/// cosine (about 25 degrees). Below it the pixel reprojected across a
/// geometric edge and the fresh gather wins.
pub const SSGI_TEMPORAL_NORMAL_MIN_DOT: f32 = 0.9;

/// Reactive-marker band over the velocity alpha: particle/glass coverage
/// the velocity pass stamps with exactly 0.25 (see velocity.glsl
/// fs_marker). Mirrors TAA_REACTIVE_MARKER_LO/HI in taa.zig (kept as local
/// copies: taa.zig depends on options.zig which depends on this module, so
/// a direct import would cycle; ssgi_tests.zig cross-pins the values).
pub const SSGI_TEMPORAL_MARKER_LO: f32 = 0.15;
pub const SSGI_TEMPORAL_MARKER_HI: f32 = 0.35;

/// Temporal activity: the pass gate plus the opt-in flag. Inline mode with
/// the flag set warns (orchestrator, WarnOnce) and stays bit-identical to
/// the untemporal pass path — this gate is what enforces it.
pub fn ssgiTemporalActive(config: PostProcessOptions) bool {
    return ssgiPassActive(config) and config.ssgi.temporal.enabled;
}

/// History blend weight in [0, 1] after clamped(). Raw config outside the
/// range clamps; the documented sweet spot is [0.85, 0.95].
pub fn ssgiTemporalBlend(config: PostProcessOptions) f32 {
    return std.math.clamp(config.ssgi.temporal.blend, 0.0, 1.0);
}

/// Reset conditions mirror taaShouldReset (same TaaReset shape): first
/// frame, toggle-on, resize/recreate, camera cut, explicit reset. The
/// camera-cut bit rides the shared taa_camera_cut snapshot flag.
pub fn ssgiTemporalShouldReset(r: TaaReset) bool {
    return r.first_frame or r.toggled_on or r.resized or r.camera_cut or r.explicit_reset;
}

/// History ping-pong slots driven by SsgiPass.temp_frame (same parity
/// shape as the TAA helpers so tests pin the alternation).
pub fn ssgiTemporalReadIndex(frame: u64) u8 {
    return @intCast(frame & 1);
}

pub fn ssgiTemporalWriteIndex(frame: u64) u8 {
    return 1 - ssgiTemporalReadIndex(frame);
}

/// History UV from a velocity tap: the velocity buffer stores
/// uv_curr - uv_prev (see velocity.glsl), so prev = uv - vel. Pure.
pub fn ssgiTemporalHistoryUv(uv: [2]f32, velocity_xy: [2]f32) [2]f32 {
    return .{ uv[0] - velocity_xy[0], uv[1] - velocity_xy[1] };
}

/// Reactive-marker classification over the velocity alpha (same band the
/// TAA resolve tests; non-finite input is never a marker). Pure.
pub fn ssgiTemporalIsMarker(alpha: f32) bool {
    if (!std.math.isFinite(alpha)) return false;
    return alpha >= SSGI_TEMPORAL_MARKER_LO and alpha <= SSGI_TEMPORAL_MARKER_HI;
}

/// Depth rejection gate (linearized depths, meters): history survives only
/// when the stored surface is within the same bilateral gate the denoise
/// uses (max(0.001, radius * 0.2)), so silhouette-crossing history never
/// blends. Pure.
pub fn ssgiTemporalDepthOk(cur_lin: f32, hist_lin: f32, radius: f32) bool {
    if (!std.math.isFinite(cur_lin) or !std.math.isFinite(hist_lin)) return false;
    return @abs(cur_lin - hist_lin) < @max(0.001, radius * 0.2);
}

/// Normal rejection gate: history survives only when the current and the
/// reprojected-location normals agree within SSGI_TEMPORAL_NORMAL_MIN_DOT
/// (degenerate zero normals reject: there is no surface to agree on).
/// Pure.
pub fn ssgiTemporalNormalOk(cur_n: [3]f32, hist_n: [3]f32) bool {
    const dot = cur_n[0] * hist_n[0] + cur_n[1] * hist_n[1] + cur_n[2] * hist_n[2];
    if (!std.math.isFinite(dot)) return false;
    return dot >= SSGI_TEMPORAL_NORMAL_MIN_DOT;
}

/// 3x3 variance bounds over center + 8 gather neighbors (same estimator as
/// taaVarianceBounds, local copy for the import-cycle reason above):
/// [mu - gamma*sigma, mu + gamma*sigma] intersected with the min/max box.
/// Pure.
pub fn ssgiTemporalVarianceBounds(center: [3]f32, neighbors: [8][3]f32, gamma: f32) TaaBounds {
    var mn = center;
    var mx = center;
    var sum = center;
    var sum_sq = [3]f32{ center[0] * center[0], center[1] * center[1], center[2] * center[2] };
    for (neighbors) |n| {
        for (0..3) |c| {
            mn[c] = @min(mn[c], n[c]);
            mx[c] = @max(mx[c], n[c]);
            sum[c] += n[c];
            sum_sq[c] += n[c] * n[c];
        }
    }
    const g = @max(gamma, 0.0);
    var v_min: [3]f32 = undefined;
    var v_max: [3]f32 = undefined;
    for (0..3) |c| {
        const mu = sum[c] / 9.0;
        const variance = @max(sum_sq[c] / 9.0 - mu * mu, 0.0);
        const sigma = @sqrt(variance);
        var lo = @max(mn[c], mu - g * sigma);
        var hi = @min(mx[c], mu + g * sigma);
        if (lo > hi) {
            lo = mn[c];
            hi = mx[c];
        }
        v_min[c] = lo;
        v_max[c] = hi;
    }
    return .{ .min = v_min, .max = v_max };
}

/// History acceptance weight in [0, 1] from the variance-box divergence
/// (same 1/(1+4d^2) falloff as taaRejectionWeight). Pure.
pub fn ssgiTemporalRejectionWeight(history: [3]f32, box_min: [3]f32, box_max: [3]f32) f32 {
    var max_dist: f32 = 0.0;
    for (0..3) |c| {
        const lo = @min(box_min[c], box_max[c]);
        const hi = @max(box_min[c], box_max[c]);
        const span = @max(hi - lo, 1e-4);
        if (history[c] < lo) {
            max_dist = @max(max_dist, (lo - history[c]) / span);
        } else if (history[c] > hi) {
            max_dist = @max(max_dist, (history[c] - hi) / span);
        }
    }
    return 1.0 / (1.0 + max_dist * max_dist * 4.0);
}

/// Full temporal resolve mirror of the ssgi_temporal.glsl main: variance
/// clip (full strength) then mix(current, clamped, blend * accept) where
/// accept folds the variance divergence, the depth/normal gates, and the
/// reactive-marker shed (markers take exactly zero history). Reset frames
/// (history_valid=false) return current. Pure.
pub fn ssgiTemporalResolve(
    current: [3]f32,
    neighbors: [8][3]f32,
    history: [3]f32,
    history_valid: bool,
    blend: f32,
    gamma: f32,
    depth_ok: bool,
    normal_ok: bool,
    is_marker: bool,
) [3]f32 {
    if (!history_valid) return current;
    const box = ssgiTemporalVarianceBounds(current, neighbors, gamma);
    const reject = ssgiTemporalRejectionWeight(history, box.min, box.max);
    var accept = reject;
    if (!depth_ok or !normal_ok) accept = 0.0;
    if (is_marker) accept = 0.0;
    const b = std.math.clamp(blend, 0.0, 1.0) * accept;
    var hc: [3]f32 = undefined;
    for (0..3) |c| {
        const lo = @min(box.min[c], box.max[c]);
        const hi = @max(box.min[c], box.max[c]);
        hc[c] = std.math.clamp(history[c], lo, hi);
    }
    return .{
        current[0] + (hc[0] - current[0]) * b,
        current[1] + (hc[1] - current[1]) * b,
        current[2] + (hc[2] - current[2]) * b,
    };
}

/// VRAM census helper (pure byte math): the history ping-pong pair at the
/// half-res size, same pixel format as the gather/denoise targets. Single
/// source of truth alongside SsgiPass.temporalTargetBytes.
pub fn ssgiTemporalTargetBytes(base_w: i32, base_h: i32, bytes_per_pixel: usize) usize {
    const size = ssgiTargetSize(base_w, base_h);
    if (size.w <= 0 or size.h <= 0) return 0;
    const w: usize = @intCast(size.w);
    const h: usize = @intCast(size.h);
    return 2 * w * h * bytes_per_pixel;
}
