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

/// Pack the shader taa_params vec4: (enabled 1/0, blend, clamp, sharpness).
/// Disabled packs all zeros, which keeps the composite bit-identical to the
/// pre-TAA path (the shader early-outs before any history/depth sampling).
pub fn taaParams(cfg: options.PostProcessOptions) [4]f32 {
    if (!cfg.taa_enabled) return .{ 0.0, 0.0, 0.0, 0.0 };
    const c = cfg.clamped();
    return .{ 1.0, c.taa_blend, c.taa_clamp_strength, c.taa_sharpness };
}

/// Pack the shader taa_state vec4: (history_valid 1/0, capture_only 1/0).
pub fn taaState(history_valid: bool, capture_only: bool) [4]f32 {
    return .{
        if (history_valid) 1.0 else 0.0,
        if (capture_only) 1.0 else 0.0,
        0.0,
        0.0,
    };
}
