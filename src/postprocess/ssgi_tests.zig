//! Tests for `postprocess/ssgi.zig` — pin the gather weights the composite
//! shader mirrors (applySSGI in postprocess.glsl) and the packed lane.
const std = @import("std");
const testing = std.testing;
const ssgi = @import("ssgi.zig");
const PostProcessOptions = @import("options.zig").PostProcessOptions;

test "ssgiWeight mirrors cos_nd * falloff^2 with hard gates" {
    // Center sample, straight up the hemisphere: full cosine, full falloff.
    try testing.expectApproxEqAbs(@as(f32, 1.0), ssgi.ssgiWeight(1.0, 0.0, 1.0), 1e-6);
    // Half radius, 45 degrees: cos 0.7071 * (0.5)^2.
    try testing.expectApproxEqAbs(@as(f32, 0.7071 * 0.25), ssgi.ssgiWeight(0.7071, 0.5, 1.0), 1e-4);
    // Behind the hemisphere, at the radius, past it, degenerate radius: zero.
    try testing.expectEqual(@as(f32, 0.0), ssgi.ssgiWeight(-0.5, 0.2, 1.0));
    try testing.expectEqual(@as(f32, 0.0), ssgi.ssgiWeight(1.0, 1.0, 1.0));
    try testing.expectEqual(@as(f32, 0.0), ssgi.ssgiWeight(1.0, 1.2, 1.0));
    try testing.expectEqual(@as(f32, 0.0), ssgi.ssgiWeight(1.0, 0.2, 0.0));
}

test "ssgiCoverage saturates at a fully enclosed gather" {
    // Open scene: a couple of light taps among 12 steps barely bleed.
    try testing.expectApproxEqAbs(@as(f32, 0.05), ssgi.ssgiCoverage(0.2, 12), 1e-6);
    // Enclosed: total_w grows past steps/3 and clamps to 1.
    try testing.expectEqual(@as(f32, 1.0), ssgi.ssgiCoverage(12.0, 12));
    try testing.expectEqual(@as(f32, 0.0), ssgi.ssgiCoverage(5.0, 0));
}

test "ssgiBleedAdd caps fireflies and scales by coverage and intensity" {
    // Weighted sum 6.0 of red over total weight 6.0 -> normalized 1.0,
    // coverage clamps to 1 (6/12*3), intensity 0.5: add = 0.5.
    const add = ssgi.ssgiBleedAdd(.{ 6.0, 0.0, 0.0 }, 6.0, 12, 0.5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), add[0], 1e-6);
    try testing.expectEqual(@as(f32, 0.0), add[1]);
    try testing.expectEqual(@as(f32, 0.0), add[2]);
    // HDR firefly: the normalized color clamps per channel at SSGI_LUMA_CAP.
    const hot = ssgi.ssgiBleedAdd(.{ 600.0, 0.0, 0.0 }, 6.0, 12, 1.0);
    try testing.expectApproxEqAbs(ssgi.SSGI_LUMA_CAP, hot[0], 1e-5);
    // Degenerate gather or zero intensity adds exactly nothing.
    const none = ssgi.ssgiBleedAdd(.{ 1.0, 1.0, 1.0 }, 0.0, 12, 1.0);
    try testing.expectEqual([3]f32{ 0, 0, 0 }, none);
}

test "ssgiParams packs the lane and zeros when inactive" {
    const on = PostProcessOptions{ .ssgi_enabled = true, .ssgi_intensity = 0.7, .ssgi_radius = 2.0, .ssgi_steps = 16 };
    try testing.expectEqual([4]f32{ 1.0, 0.7, 2.0, 16.0 }, ssgi.ssgiParams(on));
    // Zero intensity or disabled: all-zero lane (composite bit-identical).
    const quiet = PostProcessOptions{ .ssgi_enabled = true, .ssgi_intensity = 0.0 };
    try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, ssgi.ssgiParams(quiet));
    const off = PostProcessOptions{ .ssgi_enabled = false, .ssgi_intensity = 1.0 };
    try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, ssgi.ssgiParams(off));
    try testing.expect(!ssgi.ssgiActive(off));
    try testing.expect(ssgi.ssgiActive(on));
}

test "ssgiActive gates on the 0.001 intensity floor" {
    // At/below the floor the lane packs zeros (composite bit-identical).
    const floor_cfg = PostProcessOptions{ .ssgi_enabled = true, .ssgi_intensity = 0.001 };
    try testing.expect(!ssgi.ssgiActive(floor_cfg));
    try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, ssgi.ssgiParams(floor_cfg));
    const above = PostProcessOptions{ .ssgi_enabled = true, .ssgi_intensity = 0.002 };
    try testing.expect(ssgi.ssgiActive(above));
}

test "ssgi mode splits the inline and pass lanes without double-add" {
    // Default mode is inline: the v1 lane packs, the pass lane is zeros.
    const on = PostProcessOptions{ .ssgi_enabled = true, .ssgi_intensity = 0.7, .ssgi_radius = 2.0, .ssgi_steps = 16 };
    try testing.expect(on.ssgi_mode == .inline_gather);
    try testing.expect(ssgi.ssgiInlineActive(on));
    try testing.expect(!ssgi.ssgiPassActive(on));
    try testing.expectEqual([4]f32{ 1.0, 0.7, 2.0, 16.0 }, ssgi.ssgiParams(on));
    try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, ssgi.ssgiPassParams(on, true));
    // Pass mode: the inline lane zeros (no double-add), the pass gate opens.
    var pass_cfg = on;
    pass_cfg.ssgi_mode = .half_res_pass;
    try testing.expect(!ssgi.ssgiInlineActive(pass_cfg));
    try testing.expect(ssgi.ssgiPassActive(pass_cfg));
    try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, ssgi.ssgiParams(pass_cfg));
    // The pass-sample lane additionally needs a live denoised view.
    try testing.expectEqual([4]f32{ 1.0, 0.7, 0.0, 0.0 }, ssgi.ssgiPassParams(pass_cfg, true));
    try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, ssgi.ssgiPassParams(pass_cfg, false));
    // Master off or intensity floor kills both lanes in both modes.
    const off = PostProcessOptions{ .ssgi_enabled = false, .ssgi_intensity = 1.0, .ssgi_mode = .half_res_pass };
    try testing.expect(!ssgi.ssgiPassActive(off));
    try testing.expectEqual([4]f32{ 0, 0, 0, 0 }, ssgi.ssgiPassParams(off, true));
    const floor_cfg = PostProcessOptions{ .ssgi_enabled = true, .ssgi_intensity = 0.001, .ssgi_mode = .half_res_pass };
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
