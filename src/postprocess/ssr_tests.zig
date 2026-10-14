//! Tests for `postprocess/ssr.zig` — pin the inline raymarch weights the
//! composite mirrors and the half-res pass gates/lanes.
const std = @import("std");
const testing = std.testing;
const ssr = @import("ssr.zig");
const PostProcessOptions = @import("options.zig").PostProcessOptions;

test "ssr mode splits the inline and pass lanes without double-add" {
    const on = PostProcessOptions{
        .ssr = .{ .enabled = true, .intensity = 0.6, .steps = 32 },
    };
    try testing.expect(on.ssr.mode == .inline_);
    try testing.expect(ssr.ssrActive(on));
    try testing.expect(ssr.ssrInlineActive(on));
    try testing.expect(!ssr.ssrPassActive(on));
    try testing.expectEqual([4]f32{ 1.0, 0.6, 0.4, 25.0 }, ssr.ssrParams(on));
    try testing.expectEqual([4]f32{ 32.0, 0.0, 0.0, 0.0 }, ssr.ssrStepsParams(on));
    try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, ssr.ssrPassParams(on, true));

    var pass_cfg = on;
    pass_cfg.ssr.mode = .half_res_pass;
    try testing.expect(!ssr.ssrInlineActive(pass_cfg));
    try testing.expect(ssr.ssrPassActive(pass_cfg));
    try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, ssr.ssrParams(pass_cfg));
    try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, ssr.ssrStepsParams(pass_cfg));
    try testing.expectEqual([4]f32{ 1.0, 0.6, 0.0, 0.0 }, ssr.ssrPassParams(pass_cfg, true));
    try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, ssr.ssrPassParams(pass_cfg, false));

    const off = PostProcessOptions{ .ssr = .{ .enabled = false, .intensity = 1.0, .mode = .half_res_pass } };
    try testing.expect(!ssr.ssrPassActive(off));
    try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, ssr.ssrPassParams(off, true));
    const floor_cfg = PostProcessOptions{ .ssr = .{ .enabled = true, .intensity = 0.001, .mode = .half_res_pass } };
    try testing.expect(!ssr.ssrPassActive(floor_cfg));
}

test "ssrTargetSize is ceil-half of the effective size" {
    try testing.expectEqual(ssr.SsrTargetSize{ .w = 640, .h = 360 }, ssr.ssrTargetSize(1280, 720));
    try testing.expectEqual(ssr.SsrTargetSize{ .w = 641, .h = 361 }, ssr.ssrTargetSize(1281, 721));
    try testing.expectEqual(ssr.SsrTargetSize{ .w = 1, .h = 1 }, ssr.ssrTargetSize(1, 1));
    try testing.expectEqual(ssr.SsrTargetSize{ .w = 0, .h = 720 }, ssr.ssrTargetSize(0, 720));
    try testing.expectEqual(@as(usize, 2 * 640 * 360 * 8), ssr.ssrTargetBytes(1280, 720, 8));
    try testing.expectEqual(@as(usize, 0), ssr.ssrTargetBytes(0, 0, 8));
}

test "ssrFresnel matches the shader ramp" {
    try testing.expectApproxEqAbs(@as(f32, 0.9), ssr.ssrFresnel(0.0), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.15), ssr.ssrFresnel(1.0), 1e-3);
    // Grazing angles boost, facing angles floor.
    try testing.expect(ssr.ssrFresnel(0.1) > ssr.ssrFresnel(0.9));
    try testing.expectEqual(@as(f32, 0.15), ssr.ssrFresnel(std.math.nan(f32)));
}

test "ssrEdgeFade and ssrDistFade match the shader fades" {
    try testing.expectApproxEqAbs(@as(f32, 1.0), ssr.ssrEdgeFade(.{ 0.5, 0.5 }), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), ssr.ssrEdgeFade(.{ 0.0, 0.5 }), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), ssr.ssrDistFade(1, 32), 1e-6);
    const last = ssr.ssrDistFade(32, 32);
    try testing.expectApproxEqAbs(@as(f32, (1.0 - 31.0 / 32.0) * (1.0 - 31.0 / 32.0)), last, 1e-6);
    try testing.expectEqual(@as(f32, 0.0), ssr.ssrDistFade(0, 32));
}

test "ssrMix blends by intensity times weight" {
    const out = ssr.ssrMix(.{ 0.0, 0.0, 0.0 }, .{ 1.0, 1.0, 1.0 }, 0.5, 0.5);
    try testing.expectApproxEqAbs(@as(f32, 0.25), out[0], 1e-6);
    const w = ssr.ssrBlendWeight(1.0, 1.0, 0.5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), w, 1e-6);
    try testing.expectEqual(@as(f32, 0.0), ssr.ssrBlendWeight(1.0, std.math.nan(f32), 0.5));
}

test "ssrBilateralWeight gates on depth and falls off spatially" {
    const center = ssr.ssrBilateralWeight(.{ 0.0, 0.0 }, 0.0);
    try testing.expectApproxEqAbs(@as(f32, 1.0), center, 1e-6);
    try testing.expect(ssr.ssrBilateralWeight(.{ 1.0, 0.0 }, 0.0) < center);
    try testing.expectEqual(@as(f32, 0.0), ssr.ssrBilateralWeight(.{ 0.0, 0.0 }, ssr.SSR_BLUR_DEPTH_GATE));
    try testing.expectEqual(@as(f32, 0.0), ssr.ssrBilateralWeight(.{ 0.0, 0.0 }, 1.0));
}

test "ssrHalfResSize matches the ceil-half target rule" {
    // The composite derives the footprint with floor((w+1)/2); it must
    // land on the same halves the pass allocates (ssrTargetSize).
    const sizes = [_][2]i32{ .{ 1280, 720 }, .{ 1281, 721 }, .{ 720, 540 }, .{ 721, 541 }, .{ 2, 2 }, .{ 1, 1 } };
    for (sizes) |s| {
        const t = ssr.ssrTargetSize(s[0], s[1]);
        const f = ssr.ssrHalfResSize(@floatFromInt(s[0]), @floatFromInt(s[1]));
        try testing.expectApproxEqAbs(@as(f32, @floatFromInt(t.w)), f.w, 1e-6);
        try testing.expectApproxEqAbs(@as(f32, @floatFromInt(t.h)), f.h, 1e-6);
    }
}

test "ssrUpscaleResolve reconstructs without crossing silhouettes" {
    // Uniform footprint: returns the tap value (weight 1 everywhere).
    const flat_tap = ssr.SsrUpscaleTap{
        .rgb = .{ 0.8, 0.4, 0.2 },
        .weight_a = 0.5,
        .delta_texels = .{ 0.0, 0.0 },
        .depth_diff = 0.0,
    };
    const flat = ssr.ssrUpscaleResolve(.{ flat_tap, flat_tap, flat_tap, flat_tap });
    try testing.expectApproxEqAbs(@as(f32, 0.8), flat.rgb[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), flat.a, 1e-5);
    // One tap across a silhouette (depth gate): rejected, the other three
    // decide alone.
    const near_tap = ssr.SsrUpscaleTap{
        .rgb = .{ 1.0, 0.0, 0.0 },
        .weight_a = 1.0,
        .delta_texels = .{ 0.0, 0.0 },
        .depth_diff = 0.0,
    };
    const far_tap = ssr.SsrUpscaleTap{
        .rgb = .{ 0.0, 0.0, 1.0 },
        .weight_a = 1.0,
        .delta_texels = .{ 0.0, 0.0 },
        .depth_diff = 1.0,
    };
    const gated = ssr.ssrUpscaleResolve(.{ near_tap, near_tap, near_tap, far_tap });
    try testing.expectApproxEqAbs(@as(f32, 1.0), gated.rgb[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.0), gated.rgb[2], 1e-5);
    // All rejected: zeros (the caller mixes nothing).
    const none = ssr.ssrUpscaleResolve(.{ far_tap, far_tap, far_tap, far_tap });
    try testing.expectEqual([3]f32{ 0, 0, 0 }, none.rgb);
    try testing.expectEqual(@as(f32, 0.0), none.a);
}

test "ssrDenoiseResolve keeps reflection energy beside miss taps" {
    // 3x3 trace footprint: one valid center tap (refl 0.8, weight 1)
    // surrounded by 8 misses (0,0,0,0). Same-depth neighbors, so every
    // tap carries bilateral weight: center 1, edges exp(-0.5), corners
    // exp(-1); wsum = 1 + 4*exp(-0.5) + 4*exp(-1) ~= 4.89764.
    // Confidence normalization returns the full 0.8 (the old independent
    // rgb average returned 0.8/wsum ~= 0.163, and the composite mix by a
    // attenuated it a second time to ~= 0.033 at intensity 1).
    const center = ssr.SsrUpscaleTap{
        .rgb = .{ 0.8, 0.4, 0.2 },
        .weight_a = 1.0,
        .delta_texels = .{ 0.0, 0.0 },
        .depth_diff = 0.0,
    };
    const edge = ssr.SsrUpscaleTap{
        .rgb = .{ 0.0, 0.0, 0.0 },
        .weight_a = 0.0,
        .delta_texels = .{ 1.0, 0.0 },
        .depth_diff = 0.0,
    };
    const corner = ssr.SsrUpscaleTap{
        .rgb = .{ 0.0, 0.0, 0.0 },
        .weight_a = 0.0,
        .delta_texels = .{ 1.0, 1.0 },
        .depth_diff = 0.0,
    };
    const out = ssr.ssrDenoiseResolve(.{ center, edge, edge, edge, edge, corner, corner, corner, corner });
    try testing.expectApproxEqAbs(@as(f32, 0.8), out.rgb[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.4), out.rgb[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.0) / 4.89764, out.a, 1e-4);
    // Uniform valid footprint: value and weight pass through untouched.
    const full = ssr.SsrUpscaleTap{
        .rgb = .{ 0.5, 0.25, 0.125 },
        .weight_a = 0.7,
        .delta_texels = .{ 0.0, 0.0 },
        .depth_diff = 0.0,
    };
    const flat = ssr.ssrDenoiseResolve(.{full} ** 9);
    try testing.expectApproxEqAbs(@as(f32, 0.5), flat.rgb[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.7), flat.a, 1e-5);
    // All misses: zeros (the upscale then mixes nothing).
    const empty = ssr.ssrDenoiseResolve(.{corner} ** 9);
    try testing.expectEqual([3]f32{ 0, 0, 0 }, empty.rgb);
    try testing.expectEqual(@as(f32, 0.0), empty.a);
}
