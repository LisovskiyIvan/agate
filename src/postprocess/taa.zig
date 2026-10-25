const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const hdr = @import("hdr.zig");
const types = @import("types.zig");
const options = @import("options.zig");

pub const TaaReset = types.TaaReset;
pub const TaaBounds = types.TaaBounds;

/// Halton jitter cycle length (frames). The sequence itself is infinite;
/// cycling every 8 keeps the history correlation stable.
pub const TAA_JITTER_PERIOD: u64 = 8;

/// Radical-inverse Halton term for `index` in `base`. Pure; CPU tests pin
/// the golden values the jitter sequence is built from.
pub fn halton(index: u32, base: u32) f32 {
    if (base < 2) return 0.0;
    var f: f32 = 1.0;
    var r: f32 = 0.0;
    var i = index;
    const b: f32 = @floatFromInt(base);
    while (i > 0) {
        f /= b;
        r += f * @as(f32, @floatFromInt(i % base));
        i /= base;
    }
    return r;
}

/// Sub-pixel jitter in pixels, centered on [-0.5, 0.5] * scale. Halton(2,3)
/// cycled every TAA_JITTER_PERIOD frames; the Halton index starts at 1 so
/// frame 0 is not the degenerate (0, 0) sample. Negative scales floor to 0.
pub fn taaJitter(frame_index: u64, scale: f32) [2]f32 {
    const i: u32 = @intCast((frame_index % TAA_JITTER_PERIOD) + 1);
    const s = @max(scale, 0.0);
    return .{ (halton(i, 2) - 0.5) * s, (halton(i, 3) - 0.5) * s };
}

/// Fold a pixel-space jitter into a view-projection matrix: NDC offsets
/// tx = 2*jx/w, ty = -2*jy/h (screen y grows down, NDC y grows up) applied as
/// a post-translation T * vp (row0 += tx*row3, row1 += ty*row3). Degenerate
/// sizes return vp unchanged. The same matrix feeds inv/prev in the
/// composite, so reprojection matches the rasterized (jittered) depth.
pub fn applyTaaJitterToViewProj(vp: Mat4, jitter_px: [2]f32, width: i32, height: i32) Mat4 {
    if (width <= 0 or height <= 0) return vp;
    const w: f32 = @floatFromInt(width);
    const h: f32 = @floatFromInt(height);
    const tx = 2.0 * jitter_px[0] / w;
    const ty = -2.0 * jitter_px[1] / h;
    var r = vp;
    r.m[0] += tx * vp.m[3];
    r.m[4] += tx * vp.m[7];
    r.m[8] += tx * vp.m[11];
    r.m[12] += tx * vp.m[15];
    r.m[1] += ty * vp.m[3];
    r.m[5] += ty * vp.m[7];
    r.m[9] += ty * vp.m[11];
    r.m[13] += ty * vp.m[15];
    return r;
}

/// History ping-pong slots driven by PostFXStack.taa_frame (internal,
/// incremented per TAA-active composite so skipped/reused snapshot ids can
/// never alias read/write). Pure parity helpers so tests pin the alternation.
pub fn taaReadIndex(frame: u64) u8 {
    return @intCast(frame & 1);
}

pub fn taaWriteIndex(frame: u64) u8 {
    return 1 - taaReadIndex(frame);
}

pub fn taaShouldReset(r: TaaReset) bool {
    return r.first_frame or r.toggled_on or r.resized or r.camera_cut or r.explicit_reset;
}

/// 3x3 neighborhood bounds over center + 8 neighbors. Mirrors the GLSL
/// taaNeighborhood box (component-wise min/max); the shader feeds the
/// finite-bound radiance center + 8 finite-bound fast taps.
pub fn taaNeighborhoodBounds(center: [3]f32, neighbors: [8][3]f32) TaaBounds {
    var mn = center;
    var mx = center;
    for (neighbors) |n| {
        for (0..3) |c| {
            mn[c] = @min(mn[c], n[c]);
            mx[c] = @max(mx[c], n[c]);
        }
    }
    return .{ .min = mn, .max = mx };
}

pub fn taaNeighborhoodAvg(center: [3]f32, neighbors: [8][3]f32) [3]f32 {
    var sum = center;
    for (neighbors) |n| {
        for (0..3) |c| sum[c] += n[c];
    }
    return .{ sum[0] / 9.0, sum[1] / 9.0, sum[2] / 9.0 };
}

/// Ghosting clamp: mix raw history toward the box-clamped history by
/// strength in [0, 1] (1 = full clamp). Mirrors the GLSL resolve.
pub fn taaClampHistory(history: [3]f32, mn: [3]f32, mx: [3]f32, strength: f32) [3]f32 {
    const s = std.math.clamp(strength, 0.0, 1.0);
    var out: [3]f32 = undefined;
    for (0..3) |c| {
        const lo = @min(mn[c], mx[c]);
        const hi = @max(mn[c], mx[c]);
        const clamped = std.math.clamp(history[c], lo, hi);
        out[c] = history[c] + (clamped - history[c]) * s;
    }
    return out;
}

/// Temporal blend: mix current toward clamped history by blend in [0, 1]
/// (blend = history weight; 0 returns current for reset frames).
pub fn taaResolve(current: [3]f32, history_clamped: [3]f32, blend: f32) [3]f32 {
    const b = std.math.clamp(blend, 0.0, 1.0);
    return .{
        current[0] + (history_clamped[0] - current[0]) * b,
        current[1] + (history_clamped[1] - current[1]) * b,
        current[2] + (history_clamped[2] - current[2]) * b,
    };
}

/// Optional unsharp after the blend, re-clamped to the neighborhood box so
/// low amounts cannot ring. Mirrors the GLSL tail (amount <= ~0 is a no-op).
pub fn taaApplySharpen(resolved: [3]f32, current: [3]f32, avg: [3]f32, amount: f32, mn: [3]f32, mx: [3]f32) [3]f32 {
    const a = std.math.clamp(amount, 0.0, 1.0);
    if (a <= 0.0001) return resolved;
    var out: [3]f32 = undefined;
    for (0..3) |c| {
        const lo = @min(mn[c], mx[c]);
        const hi = @max(mn[c], mx[c]);
        out[c] = std.math.clamp(resolved[c] + (current[c] - avg[c]) * a, lo, hi);
    }
    return out;
}

/// Statistical variance bounding box over center + 8 neighbors with tightness `gamma`.
/// Computes sample mean and standard deviation per channel, clamping the box to
/// [mu - gamma * sigma, mu + gamma * sigma] intersected with the min/max box.
pub fn taaVarianceBounds(center: [3]f32, neighbors: [8][3]f32, gamma: f32) TaaBounds {
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
        const mu2 = sum_sq[c] / 9.0;
        const variance = @max(mu2 - mu * mu, 0.0);
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

/// Computes temporal history acceptance weight in [0, 1].
/// When history diverges from the neighborhood bounding box (e.g. occlusion/disocclusion),
/// the acceptance factor drops rapidly towards 0, rejecting ghost history and taking fresh current samples.
pub fn taaRejectionWeight(history: [3]f32, box_min: [3]f32, box_max: [3]f32) f32 {
    var max_dist: f32 = 0.0;
    for (0..3) |c| {
        const lo = @min(box_min[c], box_max[c]);
        const hi = @max(box_min[c], box_max[c]);
        const span = @max(hi - lo, 1e-4);
        if (history[c] < lo) {
            const d = (lo - history[c]) / span;
            max_dist = @max(max_dist, d);
        } else if (history[c] > hi) {
            const d = (history[c] - hi) / span;
            max_dist = @max(max_dist, d);
        }
    }
    return 1.0 / (1.0 + max_dist * max_dist * 4.0);
}

/// Finds the pixel offset [-texel, +texel] of the closest depth sample in a 3x3 cross
/// (minimum depth in standard Z). Used to sample velocity at silhouette edges without background smearing.
pub fn taaClosestDepthOffset(center_depth: f32, cross_depths: [4]f32, texel: [2]f32) [2]f32 {
    var best_depth = center_depth;
    var best_offset = [2]f32{ 0.0, 0.0 };
    const offsets = [4][2]f32{
        .{ -texel[0], 0.0 },
        .{ texel[0], 0.0 },
        .{ 0.0, -texel[1] },
        .{ 0.0, texel[1] },
    };
    for (cross_depths, offsets) |d, off| {
        if (d < best_depth) {
            best_depth = d;
            best_offset = off;
        }
    }
    return best_offset;
}

/// Full pixel resolve with variance clipping and history rejection.
pub fn taaResolvePixelWithRejection(
    current: [3]f32,
    neighbors: [8][3]f32,
    history: [3]f32,
    blend: f32,
    clamp_strength: f32,
    sharpness: f32,
    gamma: f32,
) [3]f32 {
    const c = hdr.boundHdr3(current);
    var ns: [8][3]f32 = undefined;
    for (neighbors, 0..) |n, i| ns[i] = hdr.boundHdr3(n);
    const h = hdr.boundHdr3(history);
    const box = taaVarianceBounds(c, ns, gamma);
    const avg = taaNeighborhoodAvg(c, ns);
    const reject = taaRejectionWeight(h, box.min, box.max);
    const hc = taaClampHistory(h, box.min, box.max, clamp_strength);
    const r = taaResolve(c, hc, blend * reject);
    return hdr.boundHdr3(taaApplySharpen(r, c, avg, sharpness, box.min, box.max));
}

/// Full pixel resolve (bounds + clamp + blend + sharpen): the exact GLSL
/// applyTAA tail after reprojection, plus the finite-HDR bound counterpart:
/// inputs are finite-bound on entry (history tap, center, neighbors) and the
/// result is finite-bound on exit, mirroring the shader's boundRadiance at
/// the history read, the neighborhood taps, and the resolve return.
/// Headless golden tests pin it.
pub fn taaResolvePixel(
    current: [3]f32,
    neighbors: [8][3]f32,
    history: [3]f32,
    blend: f32,
    clamp_strength: f32,
    sharpness: f32,
) [3]f32 {
    const c = hdr.boundHdr3(current);
    var ns: [8][3]f32 = undefined;
    for (neighbors, 0..) |n, i| ns[i] = hdr.boundHdr3(n);
    const h = hdr.boundHdr3(history);
    const box = taaNeighborhoodBounds(c, ns);
    const avg = taaNeighborhoodAvg(c, ns);
    const hc = taaClampHistory(h, box.min, box.max, clamp_strength);
    const r = taaResolve(c, hc, blend);
    return hdr.boundHdr3(taaApplySharpen(r, c, avg, sharpness, box.min, box.max));
}

/// Reactive-treatment depth knees (raw-depth units over the 3x3 cross
/// used by the closest-depth search): below LO the neighborhood counts as
/// flat (exactly zero boost), above HI as a hard edge (full strength).
/// Raw depths on flat surfaces agree to ~1e-6; real silhouette edges
/// differ by 0.01+, so the ramp sits two decades above quantization noise
/// and an order below real edges. Mirrors the GLSL edge ramp in applyTAA.
pub const TAA_REACTIVE_DEPTH_LO: f32 = 0.0001;
pub const TAA_REACTIVE_DEPTH_HI: f32 = 0.002;

/// Reactive history-shed weight in [0, strength] for velocity-less content
/// (particles, glass, disocclusion over background): those pixels carry the
/// background depth AND the background velocity, so their true motion is
/// invisible to the resolve. The structurally unsafe class is the
/// camera-fallback path (velocity mask < 0.5) over an inconsistent depth
/// neighborhood. Static flat pixels (exact reprojection match, zero depth
/// spread) and per-object-velocity pixels get exactly zero boost.
pub fn taaReactiveBoost(is_fallback: bool, depth_min: f32, depth_max: f32, strength: f32) f32 {
    if (!is_fallback) return 0.0;
    const s = std.math.clamp(strength, 0.0, 1.0);
    if (s <= 0.0) return 0.0;
    if (!std.math.isFinite(depth_min) or !std.math.isFinite(depth_max)) return 0.0;
    const spread = @max(depth_min, depth_max) - @min(depth_min, depth_max);
    const edge = std.math.clamp(
        (spread - TAA_REACTIVE_DEPTH_LO) / (TAA_REACTIVE_DEPTH_HI - TAA_REACTIVE_DEPTH_LO),
        0.0,
        1.0,
    );
    return s * edge;
}

/// Reactive activity gate: enabled AND a non-zero strength (the composite
/// early-outs on the same test through the packed lane).
pub fn taaReactiveActive(cfg: options.PostProcessOptions) bool {
    return cfg.taa.reactive.enabled and cfg.taa.reactive.intensity > 0.001;
}

/// True-reactive marker band (C.3 v2): the velocity pass stamps particle
/// coverage with exactly TAA_REACTIVE_MARKER_ALPHA (see velocity.glsl
/// fs_marker); content that cannot carry velocity is therefore identifiable
/// per pixel, unlike the v1 depth-spread heuristic above. The band sits
/// strictly between the clear value (0.0) and the per-object gate (0.5) so
/// the existing fallback/per-object branches never misclassify it. Nearest
/// sampling (velocity_smp) keeps the stamped value exact — no filtering
/// smear across the band edges.
pub const TAA_REACTIVE_MARKER_ALPHA: f32 = 0.25;
pub const TAA_REACTIVE_MARKER_LO: f32 = 0.15;
pub const TAA_REACTIVE_MARKER_HI: f32 = 0.35;

/// Velocity-alpha classification mirror of the applyTAA branch (the gap
/// band (HI, 0.5] has no writer and fails closed to fallback, keeping the
/// v1 heuristic as its treatment; non-finite input is fallback, never a
/// marker).
pub const TaaVelocityKind = enum { fallback, reactive, per_object };

pub fn taaVelocityKind(alpha: f32) TaaVelocityKind {
    if (!std.math.isFinite(alpha)) return .fallback;
    if (alpha > 0.5) return .per_object;
    if (alpha >= TAA_REACTIVE_MARKER_LO and alpha <= TAA_REACTIVE_MARKER_HI) return .reactive;
    return .fallback;
}

/// True-marker history shed in [0, 1]: the stamped coverage IS the signal,
/// so there is no depth edge ramp — at marker strength 1 the history weight
/// goes to exactly zero (near-zero blend toward history). Replaces the v1
/// heuristic for marked pixels (the resolve multiplies one term or the
/// other, never both).
pub fn taaReactiveMarkerBoost(marker_strength: f32) f32 {
    return std.math.clamp(marker_strength, 0.0, 1.0);
}

/// Marker-draw gate for the velocity pass (pure): markers are a TAA signal,
/// so they render only when the TAA resolve is active (taa_on already
/// implies samples == 1 in frame_render) AND the reactive family is on.
/// Blur-only mode keeps its bit-identical zero-mask path; under MSAA the
/// whole velocity pass is suppressed upstream, so markers are 1x-only and
/// the composite degrades to the v1 heuristic (and TAA itself is off).
pub fn taaReactiveMarkersActive(taa_on: bool, cfg: options.PostProcessOptions) bool {
    return taa_on and taaReactiveActive(cfg);
}

/// Pack the shader taa_reactive_params vec4: (enabled 1/0, heuristic
/// strength, marker strength, 0). Zeros when inactive, which keeps the
/// composite bit-identical to the pre-reactive path (the shader multiplies
/// a zero boost into the blend on both the marker and the heuristic lane).
pub fn taaReactiveParams(cfg: options.PostProcessOptions) [4]f32 {
    if (!taaReactiveActive(cfg)) return .{ 0.0, 0.0, 0.0, 0.0 };
    const c = cfg.clamped();
    return .{ 1.0, c.taa.reactive.intensity, c.taa.reactive.marker_intensity, 0.0 };
}

/// Full pixel resolve with variance clipping, history rejection, AND the
/// reactive treatment: on fallback-path pixels over inconsistent depth the
/// history weight sheds by the reactive boost (multiplicative with the
/// rejection weight, so already-rejected history stays rejected).
/// Headless golden tests pin it.
pub fn taaResolvePixelReactive(
    current: [3]f32,
    neighbors: [8][3]f32,
    history: [3]f32,
    blend: f32,
    clamp_strength: f32,
    sharpness: f32,
    gamma: f32,
    is_fallback: bool,
    depth_min: f32,
    depth_max: f32,
    reactive_strength: f32,
) [3]f32 {
    const c = hdr.boundHdr3(current);
    var ns: [8][3]f32 = undefined;
    for (neighbors, 0..) |n, i| ns[i] = hdr.boundHdr3(n);
    const h = hdr.boundHdr3(history);
    const box = taaVarianceBounds(c, ns, gamma);
    const avg = taaNeighborhoodAvg(c, ns);
    const reject = taaRejectionWeight(h, box.min, box.max);
    const rb = taaReactiveBoost(is_fallback, depth_min, depth_max, reactive_strength);
    const hc = taaClampHistory(h, box.min, box.max, clamp_strength);
    const r = taaResolve(c, hc, blend * reject * (1.0 - rb));
    return hdr.boundHdr3(taaApplySharpen(r, c, avg, sharpness, box.min, box.max));
}

/// Full pixel resolve with variance clipping, history rejection, AND the
/// true-reactive marker treatment: on stamped (marker-band) pixels the
/// history weight sheds by the marker boost (multiplicative with the
/// rejection weight, so already-rejected history stays rejected), REPLACING
/// the v1 depth-spread heuristic (which cannot see velocity-less content and
/// stays exactly zero here). With is_marker=false this is bit-identical to
/// taaResolvePixelWithRejection (pinned by test). Headless golden tests pin it.
pub fn taaResolvePixelMarker(
    current: [3]f32,
    neighbors: [8][3]f32,
    history: [3]f32,
    blend: f32,
    clamp_strength: f32,
    sharpness: f32,
    gamma: f32,
    is_marker: bool,
    marker_strength: f32,
) [3]f32 {
    const c = hdr.boundHdr3(current);
    var ns: [8][3]f32 = undefined;
    for (neighbors, 0..) |n, i| ns[i] = hdr.boundHdr3(n);
    const h = hdr.boundHdr3(history);
    const box = taaVarianceBounds(c, ns, gamma);
    const avg = taaNeighborhoodAvg(c, ns);
    const reject = taaRejectionWeight(h, box.min, box.max);
    const mb = if (is_marker) taaReactiveMarkerBoost(marker_strength) else 0.0;
    const hc = taaClampHistory(h, box.min, box.max, clamp_strength);
    const r = taaResolve(c, hc, blend * reject * (1.0 - mb));
    return hdr.boundHdr3(taaApplySharpen(r, c, avg, sharpness, box.min, box.max));
}

/// Pack the shader taa_params vec4: (enabled 1/0, blend, clamp, sharpness).
/// Disabled packs all zeros, which keeps the composite bit-identical to the
/// pre-TAA path (the shader early-outs before any history/depth sampling).
pub fn taaParams(cfg: options.PostProcessOptions) [4]f32 {
    if (!cfg.taa.enabled) return .{ 0.0, 0.0, 0.0, 0.0 };
    const c = cfg.clamped();
    return .{ 1.0, c.taa.blend, c.taa.clamp_intensity, c.taa.sharpness };
}

/// Pack the shader taa_state vec4: (history_valid 1/0, capture_only 1/0,
/// scene_upscale 1/0, scene_downscale 1/0). `scene_upscale` gates the Mitchell
/// reconstruction of the scene base fetch in the composite while the output
/// size exceeds the scene target (render_scale < 1). `scene_downscale` gates
/// the 4-tap quadrant box downsampler while the scene target size exceeds
/// the output size (SSAA > 1.0). When both are off — including every native
/// 1:1 path — keeps the single bilinear fetch and stays bit-identical.
pub fn taaState(history_valid: bool, capture_only: bool, scene_upscale: bool, scene_downscale: bool) [4]f32 {
    return .{
        if (history_valid) 1.0 else 0.0,
        if (capture_only) 1.0 else 0.0,
        if (scene_upscale) 1.0 else 0.0,
        if (scene_downscale) 1.0 else 0.0,
    };
}

/// Whether the composite's scene base fetch reconstructs (Catmull-Rom) or
/// reads through (bilinear): on iff the output/history size exceeds the
/// scene target size, i.e. the composite is upsampling (render_scale < 1).
/// `taa_size <= 0` means "no history override" and falls back to the scene
/// size (mirrors the parent's taa_resolution fallback). Equal sizes keep the
/// bilinear fetch — the reconstruction must be a strict no-op at native.
pub fn sceneUpscaleEnabled(taa_w: i32, taa_h: i32, scene_w: i32, scene_h: i32) bool {
    const tw = if (taa_w > 0) taa_w else scene_w;
    const th = if (taa_h > 0) taa_h else scene_h;
    return tw > scene_w or th > scene_h;
}

/// CPU mirror of the shader Mitchell–Netravali (B=C=1/3) weights
/// (postprocess.glsl): the 4 texel weights around a fractional position in
/// [0, 1). Sum == 1 for every t; t == 0 is exactly (1/18, 8/9, 1/18, 0) and
/// t == 1 is (0, 1/18, 8/9, 1/18) — the filter blends a 1/18 of the outer
/// neighbors even at texel centers (no pass-through, unlike Catmull-Rom).
/// Chosen over Catmull-Rom after gate calibration: CR's negative lobes
/// (-0.094 peak) spilled edge ringing into the flat regions of the rscale
/// ROI (taa067 flatP95 0.0518 vs the 0.04 cap; bilinear baseline 0.0277);
/// Mitchell's lobes peak at ~-0.035 and keep most of the sharpening.
pub fn mitchellWeights(t: f32) [4]f32 {
    const t2 = t * t;
    const t3 = t2 * t;
    return .{
        (-7.0 / 18.0) * t3 + (5.0 / 6.0) * t2 - 0.5 * t + (1.0 / 18.0),
        (7.0 / 6.0) * t3 - 2.0 * t2 + (8.0 / 9.0),
        -(7.0 / 6.0) * t3 + 1.5 * t2 + 0.5 * t + (1.0 / 18.0),
        (7.0 / 18.0) * t3 - (1.0 / 3.0) * t2,
    };
}
