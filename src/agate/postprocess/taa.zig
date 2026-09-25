const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
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
/// tonemapped-LDR center + 8 fast-LDR taps.
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
        const clamped = std.math.clamp(history[c], mn[c], mx[c]);
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
        out[c] = std.math.clamp(resolved[c] + (current[c] - avg[c]) * a, mn[c], mx[c]);
    }
    return out;
}

/// Full pixel resolve (bounds + clamp + blend + sharpen): the exact GLSL
/// applyTAA tail after reprojection. Headless golden tests pin it.
pub fn taaResolvePixel(
    current: [3]f32,
    neighbors: [8][3]f32,
    history: [3]f32,
    blend: f32,
    clamp_strength: f32,
    sharpness: f32,
) [3]f32 {
    const box = taaNeighborhoodBounds(current, neighbors);
    const avg = taaNeighborhoodAvg(current, neighbors);
    const hc = taaClampHistory(history, box.min, box.max, clamp_strength);
    const r = taaResolve(current, hc, blend);
    return taaApplySharpen(r, current, avg, sharpness, box.min, box.max);
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

test "taa defaults and clamps" {
    const cfg = options.PostProcessOptions{};
    try std.testing.expect(!cfg.taa_enabled);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), cfg.taa_blend, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cfg.taa_jitter_scale, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cfg.taa_sharpness, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cfg.taa_clamp_strength, 1e-6);
    try std.testing.expect(!cfg.taa_camera_cut);

    var bad = options.PostProcessOptions{
        .taa_blend = 2.0,
        .taa_jitter_scale = -1.0,
        .taa_sharpness = 5.0,
        .taa_clamp_strength = -2.0,
    };
    const out = bad.clamped();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out.taa_blend, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.taa_jitter_scale, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out.taa_sharpness, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.taa_clamp_strength, 1e-6);

    bad.taa_blend = -0.5;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), bad.clamped().taa_blend, 1e-6);
}

test "taa halton jitter sequence, wrap, and scale" {
    // Radical-inverse golden values.
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), halton(1, 2), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), halton(2, 2), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), halton(3, 2), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.125), halton(4, 2), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), halton(1, 3), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0), halton(2, 3), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 9.0), halton(3, 3), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0 / 9.0), halton(8, 3), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), halton(0, 2), 1e-6);

    // Frame 0 uses Halton index 1 (never the degenerate origin).
    const j0 = taaJitter(0, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), j0[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0 - 0.5), j0[1], 1e-6);
    const j1 = taaJitter(1, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, -0.25), j1[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 3.0 - 0.5), j1[1], 1e-6);

    // Cycle wraps every TAA_JITTER_PERIOD frames.
    const jw = taaJitter(TAA_JITTER_PERIOD, 1.0);
    try std.testing.expectApproxEqAbs(j0[0], jw[0], 1e-6);
    try std.testing.expectApproxEqAbs(j0[1], jw[1], 1e-6);
    const jw2 = taaJitter(TAA_JITTER_PERIOD + 1, 1.0);
    try std.testing.expectApproxEqAbs(j1[0], jw2[0], 1e-6);

    // Scale multiplies, zero/negative scale collapses to the origin.
    const js = taaJitter(1, 2.0);
    try std.testing.expectApproxEqAbs(j1[0] * 2.0, js[0], 1e-6);
    try std.testing.expectApproxEqAbs(j1[1] * 2.0, js[1], 1e-6);
    const jz = taaJitter(3, 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), jz[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), jz[1], 1e-6);
    const jn = taaJitter(3, -2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), jn[0], 1e-6);

    // Every sample stays inside the half-pixel box at scale 1.
    for (0..TAA_JITTER_PERIOD) |f| {
        const j = taaJitter(f, 1.0);
        try std.testing.expect(@abs(j[0]) <= 0.5 and @abs(j[1]) <= 0.5);
    }
}

test "taa jitter folds into view_proj as an NDC post-translation" {
    // Identity VP: row3 is (0,0,0,1), so the jitter lands exactly in the
    // translation column with tx = 2*jx/w, ty = -2*jy/h.
    const j = [2]f32{ 0.5, -0.25 };
    const r = applyTaaJitterToViewProj(Mat4.identity, j, 100, 200);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), r.m[12], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0025), r.m[13], 1e-6);
    for ([_]usize{ 0, 1, 4, 5, 8, 9, 10, 15 }) |k| {
        const want: f32 = switch (k) {
            0, 5, 10, 15 => 1.0,
            else => 0.0,
        };
        if (k != 12 and k != 13) try std.testing.expectApproxEqAbs(want, r.m[k], 1e-6);
    }
    // Zero jitter is the identity transform.
    const id = applyTaaJitterToViewProj(Mat4.identity, .{ 0.0, 0.0 }, 1280, 720);
    for (0..16) |k| try std.testing.expectApproxEqAbs(Mat4.identity.m[k], id.m[k], 1e-6);
    // Degenerate sizes return the input unchanged.
    const vp = Mat4.translation(Vec3.new(1.0, 2.0, 3.0));
    const d0 = applyTaaJitterToViewProj(vp, j, 0, 720);
    const d1 = applyTaaJitterToViewProj(vp, j, 1280, -4);
    for (0..16) |k| {
        try std.testing.expectApproxEqAbs(vp.m[k], d0.m[k], 1e-6);
        try std.testing.expectApproxEqAbs(vp.m[k], d1.m[k], 1e-6);
    }
}

test "taa history ping-pong alternates every frame" {
    var f: u64 = 0;
    while (f < 6) : (f += 1) {
        try std.testing.expectEqual(@as(u8, @intCast(f & 1)), taaReadIndex(f));
        try std.testing.expectEqual(@as(u8, 1 - @as(u8, @intCast(f & 1))), taaWriteIndex(f));
        // Read and write never alias.
        try std.testing.expect(taaReadIndex(f) != taaWriteIndex(f));
    }
}

test "taa reset triggers" {
    try std.testing.expect(!taaShouldReset(.{}));
    try std.testing.expect(taaShouldReset(.{ .first_frame = true }));
    try std.testing.expect(taaShouldReset(.{ .toggled_on = true }));
    try std.testing.expect(taaShouldReset(.{ .resized = true }));
    try std.testing.expect(taaShouldReset(.{ .camera_cut = true }));
    try std.testing.expect(taaShouldReset(.{ .explicit_reset = true }));
    try std.testing.expect(taaShouldReset(.{
        .first_frame = true,
        .toggled_on = true,
        .resized = true,
        .camera_cut = true,
        .explicit_reset = true,
    }));
}

test "taa neighborhood clamp and resolve math" {
    const center = [3]f32{ 0.5, 0.5, 0.5 };
    const neighbors = [8][3]f32{
        .{ 0.4, 0.5, 0.6 },
        .{ 0.6, 0.5, 0.4 },
        .{ 0.5, 0.3, 0.5 },
        .{ 0.5, 0.7, 0.5 },
        .{ 0.0, 1.0, 0.5 },
        .{ 1.0, 0.0, 0.5 },
        .{ 0.5, 0.5, 0.5 },
        .{ 0.5, 0.5, 0.5 },
    };
    const box = taaNeighborhoodBounds(center, neighbors);
    try std.testing.expectEqual([3]f32{ 0.0, 0.0, 0.4 }, box.min);
    try std.testing.expectEqual([3]f32{ 1.0, 1.0, 0.6 }, box.max);

    // Full clamp pulls outliers into the box; strength 0 keeps history raw.
    const hist = [3]f32{ 2.0, -1.0, 0.55 };
    const full = taaClampHistory(hist, box.min, box.max, 1.0);
    try std.testing.expectEqual([3]f32{ 1.0, 0.0, 0.55 }, full);
    const raw = taaClampHistory(hist, box.min, box.max, 0.0);
    try std.testing.expectEqual(hist, raw);
    const half = taaClampHistory(hist, box.min, box.max, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), half[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), half[1], 1e-6);

    // Blend 0 returns current (reset frames), 1 returns history.
    const cur = [3]f32{ 0.2, 0.4, 0.6 };
    const hcl = [3]f32{ 0.8, 0.8, 0.8 };
    try std.testing.expectEqual(cur, taaResolve(cur, hcl, 0.0));
    try std.testing.expectEqual(hcl, taaResolve(cur, hcl, 1.0));
    const b9 = taaResolve(cur, hcl, 0.9);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2 + (0.8 - 0.2) * 0.9), b9[0], 1e-6);

    // Sharpen 0 is a no-op; a flat neighborhood sharpens to itself.
    const flat_n = [_][3]f32{cur} ** 8;
    const flat_box = taaNeighborhoodBounds(cur, flat_n);
    const flat_avg = taaNeighborhoodAvg(cur, flat_n);
    const nosharp = taaApplySharpen(b9, cur, flat_avg, 0.0, flat_box.min, flat_box.max);
    try std.testing.expectEqual(b9, nosharp);
    const self_sharp = taaApplySharpen(cur, cur, flat_avg, 1.0, flat_box.min, flat_box.max);
    try std.testing.expectEqual(cur, self_sharp);

    // End-to-end golden: point box collapses history onto center, so any
    // blend returns center.
    const px = taaResolvePixel(center, [_][3]f32{center} ** 8, .{ 0.8, 0.8, 0.8 }, 0.5, 1.0, 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), px[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), px[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), px[2], 1e-6);
    // In-box history blends toward history: x/y 0.8 sit inside the box.
    const px2 = taaResolvePixel(center, neighbors, .{ 0.8, 0.8, 0.8 }, 0.5, 1.0, 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.65), px2[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.65), px2[1], 1e-6);
}

test "taa params and state packing" {
    // Disabled packs all zeros: the pre-TAA composite path.
    const def = options.PostProcessOptions{};
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, taaParams(def));
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, taaParams(def.clamped()));

    var on = options.PostProcessOptions{ .taa_enabled = true };
    try std.testing.expectEqual([4]f32{ 1.0, 0.9, 1.0, 0.0 }, taaParams(on));
    // Out-of-range values pack clamped.
    on.taa_blend = 3.0;
    on.taa_sharpness = -1.0;
    const got = taaParams(on);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), got[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), got[3], 1e-6);

    try std.testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 0.0 }, taaState(true, false));
    try std.testing.expectEqual([4]f32{ 0.0, 1.0, 0.0, 0.0 }, taaState(false, true));
    try std.testing.expectEqual([4]f32{ 1.0, 1.0, 0.0, 0.0 }, taaState(true, true));
}
