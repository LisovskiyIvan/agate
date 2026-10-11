const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const options = @import("options.zig");
const PostProcessOptions = options.PostProcessOptions;

const taa = @import("taa.zig");
const halton = taa.halton;
const taaJitter = taa.taaJitter;
const applyTaaJitterToViewProj = taa.applyTaaJitterToViewProj;
const taaReadIndex = taa.taaReadIndex;
const taaWriteIndex = taa.taaWriteIndex;
const taaShouldReset = taa.taaShouldReset;
const taaNeighborhoodBounds = taa.taaNeighborhoodBounds;
const taaNeighborhoodAvg = taa.taaNeighborhoodAvg;
const taaClampHistory = taa.taaClampHistory;
const taaResolve = taa.taaResolve;
const taaApplySharpen = taa.taaApplySharpen;
const taaVarianceBounds = taa.taaVarianceBounds;
const taaRejectionWeight = taa.taaRejectionWeight;
const taaClosestDepthOffset = taa.taaClosestDepthOffset;
const taaResolvePixelWithRejection = taa.taaResolvePixelWithRejection;
const taaResolvePixelReactive = taa.taaResolvePixelReactive;
const taaResolvePixelMarker = taa.taaResolvePixelMarker;
const taaReactiveBoost = taa.taaReactiveBoost;
const taaReactiveMarkerBoost = taa.taaReactiveMarkerBoost;
const taaVelocityKind = taa.taaVelocityKind;
const TaaVelocityKind = taa.TaaVelocityKind;
const taaReactiveMarkersActive = taa.taaReactiveMarkersActive;
const taaReactiveParams = taa.taaReactiveParams;
const taaReactiveActive = taa.taaReactiveActive;
const taaResolvePixel = taa.taaResolvePixel;
const taaParams = taa.taaParams;
const taaState = taa.taaState;
const TaaReset = taa.TaaReset;
const TaaBounds = taa.TaaBounds;
const TAA_JITTER_PERIOD = taa.TAA_JITTER_PERIOD;

test "taa clamps" {
    var bad = options.PostProcessOptions{
        .taa = .{
            .blend = 2.0,
            .jitter_scale = -1.0,
            .sharpness = 5.0,
            .clamp_intensity = -2.0,
        },
    };
    const out = bad.clamped();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out.taa.blend, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.taa.jitter_scale, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out.taa.sharpness, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.taa.clamp_intensity, 1e-6);

    bad.taa.blend = -0.5;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), bad.clamped().taa.blend, 1e-6);

    // Reactive strength clamps to [0, 1] like the other TAA knobs.
    bad.taa.reactive.intensity = 2.0;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), bad.clamped().taa.reactive.intensity, 1e-6);
    bad.taa.reactive.intensity = -1.0;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), bad.clamped().taa.reactive.intensity, 1e-6);
    // Marker strength clamps to [0, 1] on the same raw lane.
    bad.taa.reactive.marker_intensity = 2.0;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), bad.clamped().taa.reactive.marker_intensity, 1e-6);
    bad.taa.reactive.marker_intensity = -1.0;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), bad.clamped().taa.reactive.marker_intensity, 1e-6);
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

test "taa resolve pixel bounds HDR radiance and rejects poisoned history" {
    // HDR neighborhood 1/4/16: values above 1 survive the resolve and stay
    // distinct when history sits inside the box.
    const n16 = [_][3]f32{.{ 16.0, 16.0, 16.0 }} ** 8;
    const r1 = taaResolvePixel(.{ 1.0, 1.0, 1.0 }, [_][3]f32{.{ 1.0, 1.0, 1.0 }} ** 8, .{ 1.0, 1.0, 1.0 }, 0.5, 1.0, 0.0);
    const r4 = taaResolvePixel(.{ 4.0, 4.0, 4.0 }, n16, .{ 4.0, 4.0, 4.0 }, 0.5, 1.0, 0.0);
    const r16 = taaResolvePixel(.{ 16.0, 16.0, 16.0 }, n16, .{ 16.0, 16.0, 16.0 }, 0.5, 1.0, 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), r1[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), r4[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), r16[0], 1e-6);

    // Poisoned history (NaN/Inf) finite-bounds to 0/65504 first, then the
    // box clamp pulls it inside: output stays finite and inside the box.
    const center = [3]f32{ 4.0, 4.0, 4.0 };
    const poisoned = taaResolvePixel(center, n16, .{ std.math.nan(f32), std.math.inf(f32), std.math.inf(f32) }, 0.5, 1.0, 0.5);
    for (poisoned) |v| {
        try std.testing.expect(std.math.isFinite(v));
        try std.testing.expect(v >= 4.0 and v <= 16.0);
    }
    // Poisoned current and neighbors cannot widen the box to non-finite:
    // NaN center bounds to 0, Inf neighbors bound to half max.
    const bad_box = taaResolvePixel(
        .{ std.math.nan(f32), 4.0, 4.0 },
        .{ .{ std.math.inf(f32), 4.0, 4.0 }, .{ 4.0, 4.0, 4.0 }, .{ 4.0, 4.0, 4.0 }, .{ 4.0, 4.0, 4.0 }, .{ 4.0, 4.0, 4.0 }, .{ 4.0, 4.0, 4.0 }, .{ 4.0, 4.0, 4.0 }, .{ 4.0, 4.0, 4.0 } },
        .{ 4.0, 4.0, 4.0 },
        0.5,
        1.0,
        0.0,
    );
    for (bad_box) |v| try std.testing.expect(std.math.isFinite(v));
    // NaN center bounds to 0 on entry, then blends halfway toward history.
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), bad_box[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), bad_box[1], 1e-6);
}

test "taa params and state packing" {
    // Disabled packs all zeros: the pre-TAA composite path.
    const def = options.PostProcessOptions{};
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, taaParams(def));
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, taaParams(def.clamped()));

    var on = options.PostProcessOptions{
        .taa = .{
            .enabled = true,
        },
    };
    try std.testing.expectEqual([4]f32{ 1.0, 0.9, 1.0, 0.0 }, taaParams(on));
    // Out-of-range values pack clamped.
    on.taa.blend = 3.0;
    on.taa.sharpness = -1.0;
    const got = taaParams(on);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), got[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), got[3], 1e-6);

    try std.testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 0.0 }, taaState(true, false));
    try std.testing.expectEqual([4]f32{ 0.0, 1.0, 0.0, 0.0 }, taaState(false, true));
    try std.testing.expectEqual([4]f32{ 1.0, 1.0, 0.0, 0.0 }, taaState(true, true));
}

test "taa variance bounds tighten around distribution" {
    const center = [3]f32{ 0.5, 0.5, 0.5 };
    // 7 neighbors close to 0.5, 1 outlier at 1.0
    const neighbors = [8][3]f32{
        .{ 0.5, 0.5, 0.5 },
        .{ 0.5, 0.5, 0.5 },
        .{ 0.5, 0.5, 0.5 },
        .{ 0.5, 0.5, 0.5 },
        .{ 0.5, 0.5, 0.5 },
        .{ 0.5, 0.5, 0.5 },
        .{ 0.5, 0.5, 0.5 },
        .{ 1.0, 1.0, 1.0 },
    };
    const minmax = taaNeighborhoodBounds(center, neighbors);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), minmax.max[0], 1e-6);

    // Variance bounding with gamma=1.0 tightens the upper bound significantly below 1.0
    const var_box = taaVarianceBounds(center, neighbors, 1.0);
    try std.testing.expect(var_box.max[0] < 0.8);
    try std.testing.expect(var_box.min[0] >= minmax.min[0]);
    try std.testing.expect(var_box.max[0] <= minmax.max[0]);
}

test "taa rejection weight falls rapidly on disoccluded history" {
    const bmin = [3]f32{ 0.2, 0.2, 0.2 };
    const bmax = [3]f32{ 0.4, 0.4, 0.4 };

    // In-box history receives full weight (1.0)
    const in_hist = [3]f32{ 0.3, 0.3, 0.3 };
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), taaRejectionWeight(in_hist, bmin, bmax), 1e-6);

    // History 1 span away (e.g. 0.6 where span is 0.2) drops to 0.2
    const near_hist = [3]f32{ 0.6, 0.3, 0.3 };
    const w_near = taaRejectionWeight(near_hist, bmin, bmax);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), w_near, 1e-5);

    // History far away (disocclusion of bright background / another object) drops near zero
    const far_hist = [3]f32{ 1.0, 1.0, 1.0 };
    const w_far = taaRejectionWeight(far_hist, bmin, bmax);
    try std.testing.expect(w_far < 0.03);
}

test "taa closest depth selects silhouette edge velocity offset" {
    const center_d: f32 = 0.85; // background depth
    // Neighbor 1 (+X) is foreground object edge at depth 0.2
    const cross_d = [4]f32{ 0.85, 0.20, 0.85, 0.85 };
    const texel = [2]f32{ 1.0 / 1920.0, 1.0 / 1080.0 };
    const off = taaClosestDepthOffset(center_d, cross_d, texel);
    try std.testing.expectApproxEqAbs(texel[0], off[0], 1e-8);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), off[1], 1e-8);
}

test "taa disocclusion quality gate: rejection reduces ghosting error by >85%" {
    // Current frame pixel has transitioned from dark (0.1) to bright (0.9) due to object motion.
    const current = [3]f32{ 0.9, 0.9, 0.9 };
    const neighbors = [_][3]f32{current} ** 8;
    const old_history = [3]f32{ 0.1, 0.1, 0.1 }; // stale history from previous object
    const blend: f32 = 0.9; // standard 90% history accumulation

    // Un-rejected resolve retains 90% of old history (massive ghosting)
    const hc_raw = taaClampHistory(old_history, current, current, 0.0);
    const unrejected = taaResolve(current, hc_raw, blend);
    const err_unrejected = @abs(unrejected[0] - current[0]);
    // Without rejection, error is (0.9 - 0.1) * 0.9 = 0.72
    try std.testing.expect(err_unrejected > 0.7);

    // With variance clipping and rejection weight:
    const resolved = taaResolvePixelWithRejection(current, neighbors, old_history, blend, 1.0, 0.0, 1.25);
    const err_rejected = @abs(resolved[0] - current[0]);
    // Ghosting error drops from 0.72 to 0.0 (error reduction > 85%)
    try std.testing.expect(err_rejected < 0.1);
    const reduction = (err_unrejected - err_rejected) / err_unrejected;
    try std.testing.expect(reduction >= 0.85);
}

test "taa edge contrast quality gate: unsharp preserves high-frequency detail" {
    // High-frequency 1D edge pattern: alternating 1.0 and 0.0, with blurred temporal history (0.55)
    const edge_center = [3]f32{ 1.0, 1.0, 1.0 };
    const edge_neighbors = [8][3]f32{
        .{ 0.0, 0.0, 0.0 }, .{ 1.0, 1.0, 1.0 }, .{ 0.0, 0.0, 0.0 }, .{ 1.0, 1.0, 1.0 },
        .{ 0.0, 0.0, 0.0 }, .{ 1.0, 1.0, 1.0 }, .{ 0.0, 0.0, 0.0 }, .{ 1.0, 1.0, 1.0 },
    };
    const blurred_hist = [3]f32{ 0.55, 0.55, 0.55 };
    // Without sharpening: temporal blend attenuates high-frequency contrast
    const r_nosharp = taaResolvePixelWithRejection(edge_center, edge_neighbors, blurred_hist, 0.5, 1.0, 0.0, 2.0);
    // With sharpening: high-frequency edge contrast is boosted back toward 1.0
    const r_sharp = taaResolvePixelWithRejection(edge_center, edge_neighbors, blurred_hist, 0.5, 1.0, 0.5, 2.0);
    try std.testing.expect(r_sharp[0] > r_nosharp[0]);
    try std.testing.expect(r_sharp[0] <= 1.0);
}

test "taa reactive boost is zero for static and velocity-carrying pixels" {
    // Per-object-velocity pixels never boost, whatever the depth spread.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), taaReactiveBoost(false, 0.1, 0.9, 1.0), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), taaReactiveBoost(false, 0.5, 0.5, 1.0), 1e-7);
    // Fallback path over flat depth (static reprojection-exact pixels).
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), taaReactiveBoost(true, 0.5, 0.5, 1.0), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), taaReactiveBoost(true, 0.5, 0.50005, 1.0), 1e-7);
    // Zero strength never boosts, even on a hard edge.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), taaReactiveBoost(true, 0.1, 0.9, 0.0), 1e-7);
    // Non-finite depths fail closed (no boost on garbage).
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), taaReactiveBoost(true, std.math.nan(f32), 0.9, 1.0), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), taaReactiveBoost(true, 0.1, std.math.inf(f32), 1.0), 1e-7);
}

test "taa reactive boost ramps monotonically to full strength on edges" {
    // Below the LO knee: zero. Above the HI knee: exactly strength.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), taaReactiveBoost(true, 0.5, 0.50009, 0.7), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), taaReactiveBoost(true, 0.2, 0.8, 0.7), 1e-6);
    // Mid-ramp is strictly partial and monotonic in spread.
    const b1 = taaReactiveBoost(true, 0.5, 0.5006, 1.0);
    const b2 = taaReactiveBoost(true, 0.5, 0.5011, 1.0);
    try std.testing.expect(b1 > 0.0 and b1 < 1.0);
    try std.testing.expect(b2 > b1 and b2 < 1.0);
    // Strength scales linearly and clamps to [0, 1].
    try std.testing.expectApproxEqAbs(b2 * 0.5, taaReactiveBoost(true, 0.5, 0.5011, 0.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), taaReactiveBoost(true, 0.2, 0.8, 2.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), taaReactiveBoost(true, 0.2, 0.8, -1.0), 1e-7);
    // Depth order does not matter (min/max normalized).
    try std.testing.expectApproxEqAbs(
        taaReactiveBoost(true, 0.2, 0.8, 0.6),
        taaReactiveBoost(true, 0.8, 0.2, 0.6),
        1e-7,
    );
}

test "taa reactive lane packs zeros when inactive" {
    // Default (off) and TAA-off configs pack the zero lane.
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, taaReactiveParams(options.PostProcessOptions{}));
    try std.testing.expect(!taaReactiveActive(options.PostProcessOptions{}));
    var off = options.PostProcessOptions{
        .taa = .{
            .enabled = true,
            .reactive = .{
                .intensity = 0.5,
            },
        },
    };
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, taaReactiveParams(off));
    // Zero strength is inactive even when enabled.
    off.taa.reactive.enabled = true;
    off.taa.reactive.intensity = 0.0;
    try std.testing.expect(!taaReactiveActive(off));
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, taaReactiveParams(off));
    // Enabled packs (1, strength-clamped, marker-clamped, 0).
    var on = options.PostProcessOptions{
        .taa = .{
            .enabled = true,
            .reactive = .{
                .enabled = true,
                .intensity = 0.5,
            },
        },
    };
    try std.testing.expect(taaReactiveActive(on));
    try std.testing.expectEqual([4]f32{ 1.0, 0.5, 1.0, 0.0 }, taaReactiveParams(on));
    on.taa.reactive.intensity = 3.0;
    on.taa.reactive.marker_intensity = 3.0;
    try std.testing.expectEqual([4]f32{ 1.0, 1.0, 1.0, 0.0 }, taaReactiveParams(on));
    on.taa.reactive.marker_intensity = 0.25;
    try std.testing.expectEqual([4]f32{ 1.0, 1.0, 0.25, 0.0 }, taaReactiveParams(on));
}

test "taa velocity alpha classification: fallback, marker band, per-object" {
    // Clear value and the gap band (no writer) fail closed to fallback.
    try std.testing.expectEqual(TaaVelocityKind.fallback, taaVelocityKind(0.0));
    try std.testing.expectEqual(TaaVelocityKind.fallback, taaVelocityKind(0.14));
    try std.testing.expectEqual(TaaVelocityKind.fallback, taaVelocityKind(0.36));
    try std.testing.expectEqual(TaaVelocityKind.fallback, taaVelocityKind(0.5));
    // The stamped marker value classifies reactive (write/read contract).
    try std.testing.expectEqual(TaaVelocityKind.reactive, taaVelocityKind(taa.TAA_REACTIVE_MARKER_ALPHA));
    try std.testing.expectEqual(TaaVelocityKind.reactive, taaVelocityKind(0.15));
    try std.testing.expectEqual(TaaVelocityKind.reactive, taaVelocityKind(0.35));
    // Per-object gate unchanged (strictly above 0.5).
    try std.testing.expectEqual(TaaVelocityKind.per_object, taaVelocityKind(0.5001));
    try std.testing.expectEqual(TaaVelocityKind.per_object, taaVelocityKind(1.0));
    // Non-finite input never manufactures a marker.
    try std.testing.expectEqual(TaaVelocityKind.fallback, taaVelocityKind(std.math.nan(f32)));
    try std.testing.expectEqual(TaaVelocityKind.fallback, taaVelocityKind(std.math.inf(f32)));
}

test "taa reactive marker boost is the clamped strength, no edge ramp" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), taaReactiveMarkerBoost(0.0), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), taaReactiveMarkerBoost(0.5), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), taaReactiveMarkerBoost(1.0), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), taaReactiveMarkerBoost(2.0), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), taaReactiveMarkerBoost(-1.0), 1e-7);
}

test "taa reactive markers render only for an active TAA resolve" {
    const on = options.PostProcessOptions{
        .taa = .{
            .enabled = true,
            .reactive = .{
                .enabled = true,
                .intensity = 0.5,
            },
        },
    };
    const off = options.PostProcessOptions{};
    // TAA resolve active + reactive on: markers drawn.
    try std.testing.expect(taaReactiveMarkersActive(true, on));
    // Blur-only (TAA off): no markers — the zero-mask path stays bit-identical.
    try std.testing.expect(!taaReactiveMarkersActive(false, on));
    // Reactive off: no markers even under TAA.
    try std.testing.expect(!taaReactiveMarkersActive(true, off));
    var weak = on;
    weak.taa.reactive.intensity = 0.0;
    try std.testing.expect(!taaReactiveMarkersActive(true, weak));
}

test "taa marker resolve is bit-identical to rejection resolve when unmarked" {
    const current = [3]f32{ 0.55, 0.55, 0.55 };
    const neighbors = [8][3]f32{
        .{ 0.5, 0.5, 0.5 }, .{ 0.6, 0.6, 0.6 }, .{ 0.5, 0.5, 0.5 }, .{ 0.6, 0.6, 0.6 },
        .{ 0.5, 0.5, 0.5 }, .{ 0.6, 0.6, 0.6 }, .{ 0.5, 0.5, 0.5 }, .{ 0.6, 0.6, 0.6 },
    };
    const history = [3]f32{ 0.52, 0.52, 0.52 };
    const plain = taaResolvePixelWithRejection(current, neighbors, history, 0.9, 1.0, 0.0, 1.25);
    // Unmarked (or zero marker strength): exactly the v1 resolve.
    const unmarked = taaResolvePixelMarker(current, neighbors, history, 0.9, 1.0, 0.0, 1.25, false, 1.0);
    try std.testing.expectEqual(plain, unmarked);
    const zero = taaResolvePixelMarker(current, neighbors, history, 0.9, 1.0, 0.0, 1.25, true, 0.0);
    try std.testing.expectEqual(plain, zero);
}

test "taa marker resolve fully sheds history at strength 1, monotonic below" {
    const current = [3]f32{ 0.55, 0.55, 0.55 };
    const neighbors = [8][3]f32{
        .{ 0.5, 0.5, 0.5 }, .{ 0.6, 0.6, 0.6 }, .{ 0.5, 0.5, 0.5 }, .{ 0.6, 0.6, 0.6 },
        .{ 0.5, 0.5, 0.5 }, .{ 0.6, 0.6, 0.6 }, .{ 0.5, 0.5, 0.5 }, .{ 0.6, 0.6, 0.6 },
    };
    const history = [3]f32{ 0.52, 0.52, 0.52 };
    const plain = taaResolvePixelWithRejection(current, neighbors, history, 0.9, 1.0, 0.0, 1.25);
    const err_plain = @abs(plain[0] - current[0]);
    try std.testing.expect(err_plain > 0.0);
    // Marked pixels ignore the (flat, zero) depth heuristic and shed by the
    // marker lane instead: stronger treatment lands closer to current.
    const r_half = taaResolvePixelMarker(current, neighbors, history, 0.9, 1.0, 0.0, 1.25, true, 0.5);
    const r_full = taaResolvePixelMarker(current, neighbors, history, 0.9, 1.0, 0.0, 1.25, true, 1.0);
    const err_half = @abs(r_half[0] - current[0]);
    const err_full = @abs(r_full[0] - current[0]);
    try std.testing.expect(err_half < err_plain);
    try std.testing.expect(err_full < err_half);
    // Full marker strength rejects history exactly (near-zero blend).
    try std.testing.expectApproxEqAbs(current[0], r_full[0], 1e-6);
    try std.testing.expectApproxEqAbs(current[1], r_full[1], 1e-6);
    try std.testing.expectApproxEqAbs(current[2], r_full[2], 1e-6);
}

test "taa reactive resolve keeps static pixels bit-identical" {
    const current = [3]f32{ 0.6, 0.6, 0.6 };
    const neighbors = [_][3]f32{current} ** 8;
    const history = [3]f32{ 0.6, 0.6, 0.6 };
    const plain = taaResolvePixelWithRejection(current, neighbors, history, 0.9, 1.0, 0.0, 1.25);
    // Flat fallback (static pixel): zero boost, identical output.
    const flat = taaResolvePixelReactive(current, neighbors, history, 0.9, 1.0, 0.0, 1.25, true, 0.5, 0.5, 1.0);
    try std.testing.expectEqual(plain, flat);
    // Per-object-velocity path: zero boost even on a depth edge.
    const vel = taaResolvePixelReactive(current, neighbors, history, 0.9, 1.0, 0.0, 1.25, false, 0.1, 0.9, 1.0);
    try std.testing.expectEqual(plain, vel);
    // Disabled strength: zero boost even on fallback + edge.
    const off = taaResolvePixelReactive(current, neighbors, history, 0.9, 1.0, 0.0, 1.25, true, 0.1, 0.9, 0.0);
    try std.testing.expectEqual(plain, off);
}

test "taa reactive resolve sheds history on fallback edges, monotonic in strength" {
    // Slow-moving large translucent object: history color still sits inside
    // the variance box, so divergence rejection alone keeps 90% of it.
    const current = [3]f32{ 0.55, 0.55, 0.55 };
    const neighbors = [8][3]f32{
        .{ 0.5, 0.5, 0.5 }, .{ 0.6, 0.6, 0.6 }, .{ 0.5, 0.5, 0.5 }, .{ 0.6, 0.6, 0.6 },
        .{ 0.5, 0.5, 0.5 }, .{ 0.6, 0.6, 0.6 }, .{ 0.5, 0.5, 0.5 }, .{ 0.6, 0.6, 0.6 },
    };
    const history = [3]f32{ 0.52, 0.52, 0.52 };
    const plain = taaResolvePixelWithRejection(current, neighbors, history, 0.9, 1.0, 0.0, 1.25);
    const err_plain = @abs(plain[0] - current[0]);
    try std.testing.expect(err_plain > 0.0);
    // Fallback + hard depth edge: stronger treatment lands closer to current.
    const r_half = taaResolvePixelReactive(current, neighbors, history, 0.9, 1.0, 0.0, 1.25, true, 0.2, 0.8, 0.5);
    const r_full = taaResolvePixelReactive(current, neighbors, history, 0.9, 1.0, 0.0, 1.25, true, 0.2, 0.8, 1.0);
    const err_half = @abs(r_half[0] - current[0]);
    const err_full = @abs(r_full[0] - current[0]);
    try std.testing.expect(err_half < err_plain);
    try std.testing.expect(err_full < err_half);
    // Full strength on a hard edge sheds the stale history almost entirely.
    try std.testing.expect(err_full < err_plain * 0.6);
}
