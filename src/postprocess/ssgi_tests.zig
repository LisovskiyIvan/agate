//! Tests for `postprocess/ssgi.zig` — pin the gather weights the composite
//! shader mirrors (applySSGI in postprocess.glsl) and the packed lane.
const std = @import("std");
const testing = std.testing;
const ssgi = @import("ssgi.zig");
const PostProcessOptions = @import("options.zig").PostProcessOptions;

test "ssgiWeight mirrors cos_nd * falloff^3 with hard gates" {
    // Center sample, straight up the hemisphere: full cosine, full falloff.
    try testing.expectApproxEqAbs(@as(f32, 1.0), ssgi.ssgiWeight(1.0, 0.0, 1.0), 1e-6);
    // Half radius, 45 degrees: cos 0.7071 * (0.5)^3.
    try testing.expectApproxEqAbs(@as(f32, 0.7071 * 0.125), ssgi.ssgiWeight(0.7071, 0.5, 1.0), 1e-4);
    // Behind the hemisphere, at the radius, past it, degenerate radius: zero.
    try testing.expectEqual(@as(f32, 0.0), ssgi.ssgiWeight(-0.5, 0.2, 1.0));
    try testing.expectEqual(@as(f32, 0.0), ssgi.ssgiWeight(1.0, 1.0, 1.0));
    try testing.expectEqual(@as(f32, 0.0), ssgi.ssgiWeight(1.0, 1.2, 1.0));
    try testing.expectEqual(@as(f32, 0.0), ssgi.ssgiWeight(1.0, 0.2, 0.0));
}

test "ssgiNormalReliability trusts smooth surfaces and kills silhouettes" {
    // Smooth plane or grazing floor: both sides comparable, fully trusted.
    try testing.expectApproxEqAbs(@as(f32, 1.0), ssgi.ssgiNormalReliability(0.3, 0.3, 0.2, 0.2), 1e-6);
    // Moderate curvature asymmetry (below the 3x ramp start): still trusted.
    try testing.expectEqual(@as(f32, 1.0), ssgi.ssgiNormalReliability(0.1, 0.25, 0.2, 0.2));
    // Silhouette pixel: one side jumps across the edge (tens of times the
    // other side) — the reconstructed normal is garbage, gather is killed.
    try testing.expectEqual(@as(f32, 0.0), ssgi.ssgiNormalReliability(0.02, 1.2, 0.2, 0.2));
    // One axis smooth, the other crossing: the unreliable axis wins.
    try testing.expectEqual(@as(f32, 0.0), ssgi.ssgiNormalReliability(0.3, 0.3, 0.02, 1.2));
    // Degenerate flat taps (everything at the same depth): fully trusted.
    try testing.expectEqual(@as(f32, 1.0), ssgi.ssgiNormalReliability(0.0, 0.0, 0.0, 0.0));
}

test "ssgiCoverage saturates at a fully enclosed gather" {
    // Open scene: a couple of light taps among 12 steps barely bleed.
    try testing.expectApproxEqAbs(@as(f32, 0.025), ssgi.ssgiCoverage(0.2, 12), 1e-6);
    // Enclosed: total_w grows past steps/1.5 and clamps to 1.
    try testing.expectEqual(@as(f32, 1.0), ssgi.ssgiCoverage(12.0, 12));
    try testing.expectEqual(@as(f32, 0.0), ssgi.ssgiCoverage(5.0, 0));
}

test "ssgiBleedAdd caps fireflies and scales by coverage and intensity" {
    // Weighted sum 6.0 of red over total weight 6.0 -> normalized 1.0,
    // coverage 6/12*1.5 = 0.75, intensity 0.5: add = 0.375.
    const add = ssgi.ssgiBleedAdd(.{ 6.0, 0.0, 0.0 }, 6.0, 12, 0.5);
    try testing.expectApproxEqAbs(@as(f32, 0.375), add[0], 1e-6);
    try testing.expectEqual(@as(f32, 0.0), add[1]);
    try testing.expectEqual(@as(f32, 0.0), add[2]);
    // HDR firefly: the normalized color clamps per channel at SSGI_LUMA_CAP.
    const hot = ssgi.ssgiBleedAdd(.{ 600.0, 0.0, 0.0 }, 6.0, 12, 1.0);
    try testing.expectApproxEqAbs(ssgi.SSGI_LUMA_CAP * 0.75, hot[0], 1e-5);
    // Degenerate gather or zero intensity adds exactly nothing.
    const none = ssgi.ssgiBleedAdd(.{ 1.0, 1.0, 1.0 }, 0.0, 12, 1.0);
    try testing.expectEqual([3]f32{ 0, 0, 0 }, none);
}

test "ssgiReceiverTint and halo cap keep a white floor tap from glowing white" {
    // Green receiver, white normalized bleed: the add is green, scaled by
    // coverage 6/12*1.5 = 0.75, normalized by the receiver's peak, then
    // clamped per channel by the halo cap max(recv, 0.12)*0.75.
    const add = ssgi.ssgiBleedAddTinted(.{ 6.0, 6.0, 6.0 }, 6.0, 12, 1.0, .{ 0.2, 0.8, 0.2 }, 1.0, 1.0);
    try testing.expectApproxEqAbs(@as(f32, 0.15), add[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.6), add[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.15), add[2], 1e-5);

    // Dark receiver still gets fill, but only up to the 0.12*0.75 cap.
    const dark = ssgi.ssgiBleedAddTinted(.{ 6.0, 6.0, 6.0 }, 6.0, 12, 1.0, .{ 0.01, 0.01, 0.01 }, 1.0, 1.0);
    try testing.expectApproxEqAbs(@as(f32, 0.09), dark[0], 1e-5);

    // Zero reliability (silhouette receiver): the gather adds nothing.
    const edge = ssgi.ssgiBleedAddTinted(.{ 6.0, 6.0, 6.0 }, 6.0, 12, 1.0, .{ 0.2, 0.8, 0.2 }, 0.0, 1.0);
    try testing.expectEqual([3]f32{ 0, 0, 0 }, edge);

    // Contact occlusion: an AO-0.2 receiver at the gate keeps only
    // mix(1, 0.2, 0.8) = 0.36 of the capped add; AO 1.0 (or SSAO off) is
    // inert.
    const occluded = ssgi.ssgiBleedAddTinted(.{ 6.0, 6.0, 6.0 }, 6.0, 12, 1.0, .{ 0.2, 0.8, 0.2 }, 1.0, 0.2);
    try testing.expectApproxEqAbs(@as(f32, 0.15 * 0.36), occluded[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.0), ssgi.ssgiOcclusionGate(1.0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), ssgi.ssgiOcclusionGate(std.math.nan(f32)), 1e-6);

    // Degenerate/NaN receiver falls back to white tint, never NaN.
    const degenerate = ssgi.ssgiReceiverTint(.{ 0.0, 0.0, 0.0 });
    try testing.expectEqual([3]f32{ 1.0, 1.0, 1.0 }, degenerate);
    const nan = ssgi.ssgiReceiverTint(.{ std.math.nan(f32), 0.0, 0.0 });
    try testing.expectEqual([3]f32{ 1.0, 1.0, 1.0 }, nan);
}

test "ssgiParams packs the lane and zeros when inactive" {
    const on = PostProcessOptions{
        .ssgi = .{
            .enabled = true,
            .intensity = 0.7,
            .radius = 2.0,
            .steps = 16,
        },
    };
    try testing.expectEqual([4]f32{ 1.0, 0.7, 2.0, 16.0 }, ssgi.ssgiParams(on));
    // Zero intensity or disabled: all-zero lane (composite bit-identical).
    const quiet = PostProcessOptions{
        .ssgi = .{
            .enabled = true,
            .intensity = 0.0,
        },
    };
    try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, ssgi.ssgiParams(quiet));
    const off = PostProcessOptions{
        .ssgi = .{
            .enabled = false,
            .intensity = 1.0,
        },
    };
    try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, ssgi.ssgiParams(off));
    try testing.expect(!ssgi.ssgiActive(off));
    try testing.expect(ssgi.ssgiActive(on));
}

test "ssgiActive gates on the 0.001 intensity floor" {
    // At/below the floor the lane packs zeros (composite bit-identical).
    const floor_cfg = PostProcessOptions{
        .ssgi = .{
            .enabled = true,
            .intensity = 0.001,
        },
    };
    try testing.expect(!ssgi.ssgiActive(floor_cfg));
    try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, ssgi.ssgiParams(floor_cfg));
    const above = PostProcessOptions{
        .ssgi = .{
            .enabled = true,
            .intensity = 0.002,
        },
    };
    try testing.expect(ssgi.ssgiActive(above));
}

test "ssgi mode splits the inline and pass lanes without double-add" {
    // Default mode is inline: the v1 lane packs, the pass lane is zeros.
    const on = PostProcessOptions{
        .ssgi = .{
            .enabled = true,
            .intensity = 0.7,
            .radius = 2.0,
            .steps = 16,
        },
    };
    try testing.expect(on.ssgi.mode == .inline_gather);
    try testing.expect(ssgi.ssgiInlineActive(on));
    try testing.expect(!ssgi.ssgiPassActive(on));
    try testing.expectEqual([4]f32{ 1.0, 0.7, 2.0, 16.0 }, ssgi.ssgiParams(on));
    try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, ssgi.ssgiPassParams(on, true));
    // Pass mode: the inline lane zeros (no double-add), the pass gate opens.
    var pass_cfg = on;
    pass_cfg.ssgi.mode = .half_res_pass;
    try testing.expect(!ssgi.ssgiInlineActive(pass_cfg));
    try testing.expect(ssgi.ssgiPassActive(pass_cfg));
    try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, ssgi.ssgiParams(pass_cfg));
    // The pass-sample lane additionally needs a live denoised view.
    try testing.expectEqual([4]f32{ 1.0, 0.7, 0.0, 0.0 }, ssgi.ssgiPassParams(pass_cfg, true));
    try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, ssgi.ssgiPassParams(pass_cfg, false));
    // Master off or intensity floor kills both lanes in both modes.
    const off = PostProcessOptions{
        .ssgi = .{
            .enabled = false,
            .intensity = 1.0,
            .mode = .half_res_pass,
        },
    };
    try testing.expect(!ssgi.ssgiPassActive(off));
    try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, ssgi.ssgiPassParams(off, true));
    const floor_cfg = PostProcessOptions{
        .ssgi = .{
            .enabled = true,
            .intensity = 0.001,
            .mode = .half_res_pass,
        },
    };
    try testing.expect(!ssgi.ssgiPassActive(floor_cfg));
    try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, ssgi.ssgiPassParams(floor_cfg, true));
}

test "ssgiTargetSize is ceil-half of the effective size" {
    // Even sizes halve exactly; odd sizes round up (SSAO raw-target rule).
    try testing.expectEqual(ssgi.SsgiTargetSize{ .w = 640, .h = 360 }, ssgi.ssgiTargetSize(1280, 720));
    try testing.expectEqual(ssgi.SsgiTargetSize{ .w = 641, .h = 361 }, ssgi.ssgiTargetSize(1281, 721));
    try testing.expectEqual(ssgi.SsgiTargetSize{ .w = 1, .h = 1 }, ssgi.ssgiTargetSize(1, 1));
    // Degenerate inputs pass through (the caller guards, never allocates).
    try testing.expectEqual(ssgi.SsgiTargetSize{ .w = 0, .h = 720 }, ssgi.ssgiTargetSize(0, 720));
    // Two half-res targets (gather + denoised) at 8 Bpp RGBA16F.
    try testing.expectEqual(@as(usize, 2 * 640 * 360 * 8), ssgi.ssgiTargetBytes(1280, 720, 8));
    // Odd sizes use the rounded-up halves; degenerate sizes census zero.
    try testing.expectEqual(@as(usize, 2 * 641 * 361 * 8), ssgi.ssgiTargetBytes(1281, 721, 8));
    try testing.expectEqual(@as(usize, 0), ssgi.ssgiTargetBytes(0, 0, 8));
}

test "ssgiTemporalActive needs pass mode plus the opt-in flag" {
    // Default: temporal off in both modes (bit-identical to pre-temporal).
    const inline_cfg = PostProcessOptions{
        .ssgi = .{
            .enabled = true,
            .intensity = 0.5,
        },
    };
    try testing.expect(!ssgi.ssgiTemporalActive(inline_cfg));
    var pass_cfg = inline_cfg;
    pass_cfg.ssgi.mode = .half_res_pass;
    try testing.expect(!ssgi.ssgiTemporalActive(pass_cfg));
    // Flag on in pass mode: active. Flag on in inline mode: gated off
    // (the orchestrator warns once and the pass stays bit-identical).
    pass_cfg.ssgi.temporal.enabled = true;
    try testing.expect(ssgi.ssgiTemporalActive(pass_cfg));
    var inline_temporal = inline_cfg;
    inline_temporal.ssgi.temporal.enabled = true;
    try testing.expect(!ssgi.ssgiTemporalActive(inline_temporal));
    // Master off or intensity floor kills temporal with the pass gate.
    const off = PostProcessOptions{
        .ssgi = .{
            .enabled = false,
            .intensity = 1.0,
            .mode = .half_res_pass,
            .temporal = .{
                .enabled = true,
            },
        },
    };
    try testing.expect(!ssgi.ssgiTemporalActive(off));
}

test "ssgiTemporalBlend clamps and documents the sweet spot" {
    try testing.expectEqual(@as(f32, 0.9), ssgi.ssgiTemporalBlend(.{}));
    try testing.expectEqual(@as(f32, 0.0), ssgi.ssgiTemporalBlend(.{ .ssgi = .{ .temporal = .{ .blend = -1.0 } } }));
    try testing.expectEqual(@as(f32, 1.0), ssgi.ssgiTemporalBlend(.{ .ssgi = .{ .temporal = .{ .blend = 2.0 } } }));
    try testing.expectEqual(@as(f32, 0.87), ssgi.ssgiTemporalBlend(.{ .ssgi = .{ .temporal = .{ .blend = 0.87 } } }));
    // The recommended window sits strictly inside (0, 1).
    try testing.expect(ssgi.SSGI_TEMPORAL_BLEND_MIN > 0.0 and ssgi.SSGI_TEMPORAL_BLEND_MAX < 1.0);
    try testing.expect(ssgi.SSGI_TEMPORAL_BLEND_MIN < ssgi.SSGI_TEMPORAL_BLEND_DEFAULT and ssgi.SSGI_TEMPORAL_BLEND_DEFAULT < ssgi.SSGI_TEMPORAL_BLEND_MAX);
}

test "ssgiTemporalShouldReset mirrors the TAA reset shape" {
    try testing.expect(ssgi.ssgiTemporalShouldReset(.{ .first_frame = true }));
    try testing.expect(ssgi.ssgiTemporalShouldReset(.{ .toggled_on = true }));
    try testing.expect(ssgi.ssgiTemporalShouldReset(.{ .resized = true }));
    try testing.expect(ssgi.ssgiTemporalShouldReset(.{ .camera_cut = true }));
    try testing.expect(ssgi.ssgiTemporalShouldReset(.{ .explicit_reset = true }));
    try testing.expect(!ssgi.ssgiTemporalShouldReset(.{}));
}

test "ssgiTemporal ping-pong alternates by frame parity" {
    try testing.expectEqual(@as(u8, 0), ssgi.ssgiTemporalReadIndex(0));
    try testing.expectEqual(@as(u8, 1), ssgi.ssgiTemporalWriteIndex(0));
    try testing.expectEqual(@as(u8, 1), ssgi.ssgiTemporalReadIndex(1));
    try testing.expectEqual(@as(u8, 0), ssgi.ssgiTemporalWriteIndex(1));
    try testing.expect(ssgi.ssgiTemporalReadIndex(42) != ssgi.ssgiTemporalWriteIndex(42));
}

test "ssgiTemporalHistoryUv subtracts the velocity vector" {
    // Velocity stores uv_curr - uv_prev, so prev = uv - vel.
    const uv = ssgi.ssgiTemporalHistoryUv(.{ 0.5, 0.5 }, .{ 0.02, -0.01 });
    try testing.expectApproxEqAbs(@as(f32, 0.48), uv[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.51), uv[1], 1e-6);
    // Zero motion reprojects onto itself.
    try testing.expectEqual([2]f32{ 0.25, 0.75 }, ssgi.ssgiTemporalHistoryUv(.{ 0.25, 0.75 }, .{ 0.0, 0.0 }));
}

test "ssgiTemporal marker band matches the TAA velocity classification" {
    // The stamped particle/glass value falls inside; clear and per-object
    // do not. Non-finite input is never a marker (fails closed).
    try testing.expect(ssgi.ssgiTemporalIsMarker(0.25));
    try testing.expect(ssgi.ssgiTemporalIsMarker(0.15));
    try testing.expect(ssgi.ssgiTemporalIsMarker(0.35));
    try testing.expect(!ssgi.ssgiTemporalIsMarker(0.0));
    try testing.expect(!ssgi.ssgiTemporalIsMarker(0.5));
    try testing.expect(!ssgi.ssgiTemporalIsMarker(1.0));
    try testing.expect(!ssgi.ssgiTemporalIsMarker(std.math.nan(f32)));
    // Cross-pin against the TAA band: one writer (velocity.glsl fs_marker
    // stamps 0.25), two readers — they must never drift apart.
    const taa = @import("taa.zig");
    try testing.expectEqual(taa.TAA_REACTIVE_MARKER_LO, ssgi.SSGI_TEMPORAL_MARKER_LO);
    try testing.expectEqual(taa.TAA_REACTIVE_MARKER_HI, ssgi.SSGI_TEMPORAL_MARKER_HI);
}

test "ssgiTemporal depth gate follows the bilateral threshold" {
    // Same gate the denoise uses: max(0.001, radius * 0.2).
    try testing.expect(ssgi.ssgiTemporalDepthOk(5.0, 5.05, 1.5));
    try testing.expect(!ssgi.ssgiTemporalDepthOk(5.0, 5.5, 1.5));
    try testing.expect(!ssgi.ssgiTemporalDepthOk(5.0, 5.0 + 0.002, 0.001));
    try testing.expect(!ssgi.ssgiTemporalDepthOk(std.math.inf(f32), 5.0, 1.5));
}

test "ssgiTemporal normal gate needs agreeing surfaces" {
    try testing.expect(ssgi.ssgiTemporalNormalOk(.{ 0, 1, 0 }, .{ 0, 1, 0 }));
    try testing.expect(!ssgi.ssgiTemporalNormalOk(.{ 0, 1, 0 }, .{ 1, 0, 0 }));
    // Degenerate (zero) normals reject: no surface to agree on.
    try testing.expect(!ssgi.ssgiTemporalNormalOk(.{ 0, 0, 0 }, .{ 0, 1, 0 }));
}

test "ssgiTemporalResolve blends, clips, rejects and resets" {
    const flat: [8][3]f32 = @splat(.{ 1.0, 0.5, 0.25 });
    // Spread neighborhood so a nearby history lands inside the variance
    // box: mean == center, sigma > 0, history within mu +/- sigma.
    const spread: [8][3]f32 = .{
        .{ 0.8, 0.4, 0.2 }, .{ 1.2, 0.6, 0.3 },
        .{ 0.8, 0.4, 0.2 }, .{ 1.2, 0.6, 0.3 },
        .{ 0.8, 0.4, 0.2 }, .{ 1.2, 0.6, 0.3 },
        .{ 0.8, 0.4, 0.2 }, .{ 1.2, 0.6, 0.3 },
    };
    // Converged history near the current value blends by the weight.
    const near = ssgi.ssgiTemporalResolve(.{ 1.0, 0.5, 0.25 }, spread, .{ 1.1, 0.55, 0.3 }, true, 0.9, 1.0, true, true, false);
    for (0..3) |c| {
        const cur = ([3]f32{ 1.0, 0.5, 0.25 })[c];
        const hist = ([3]f32{ 1.1, 0.55, 0.3 })[c];
        try testing.expect(near[c] > @min(cur, hist) and near[c] < @max(cur, hist));
    }
    // Reset frames return the current gather untouched.
    try testing.expectEqual([3]f32{ 1.0, 0.5, 0.25 }, ssgi.ssgiTemporalResolve(.{ 1.0, 0.5, 0.25 }, flat, .{ 9.0, 9.0, 9.0 }, false, 0.9, 1.0, true, true, false));
    // A diverged history clamps into the box and barely blends (the
    // variance falloff drives the weight toward zero).
    const far = ssgi.ssgiTemporalResolve(.{ 1.0, 0.5, 0.25 }, flat, .{ 9.0, 9.0, 9.0 }, true, 0.9, 1.0, true, true, false);
    try testing.expectApproxEqAbs(@as(f32, 1.0), far[0], 0.05);
    // Depth/normal gates and reactive markers take exactly zero history.
    const gated = ssgi.ssgiTemporalResolve(.{ 1.0, 0.5, 0.25 }, flat, .{ 1.1, 0.55, 0.3 }, true, 0.9, 1.0, false, true, false);
    try testing.expectEqual([3]f32{ 1.0, 0.5, 0.25 }, gated);
    const marked = ssgi.ssgiTemporalResolve(.{ 1.0, 0.5, 0.25 }, flat, .{ 1.1, 0.55, 0.3 }, true, 0.9, 1.0, true, true, true);
    try testing.expectEqual([3]f32{ 1.0, 0.5, 0.25 }, marked);
}

test "ssgiTemporalTargetBytes counts the history pair" {
    // Two half-res RGBA16F targets at 8 Bpp, same halves the pass allocates.
    try testing.expectEqual(@as(usize, 2 * 640 * 360 * 8), ssgi.ssgiTemporalTargetBytes(1280, 720, 8));
    try testing.expectEqual(@as(usize, 2 * 641 * 361 * 8), ssgi.ssgiTemporalTargetBytes(1281, 721, 8));
    try testing.expectEqual(@as(usize, 0), ssgi.ssgiTemporalTargetBytes(0, 0, 8));
}
