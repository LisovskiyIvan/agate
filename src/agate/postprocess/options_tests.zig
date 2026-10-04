//! Tests for `postprocess/options.zig` (moved verbatim from inline blocks; production code unchanged).
const std = @import("std");
const prod = @import("options.zig");
const PostProcessOptions = prod.PostProcessOptions;
const TonemappingType = prod.TonemappingType;

test "config clamped sanitizes new fields" {
    var cfg = PostProcessOptions{
        .exposure = -2.0,
        .dof_focus_distance = -4.0,
        .dof_focus_range = -1.0,
        .dof_max_blur = -3.0,
        .grade_shadows = .{ 2.0, -2.0, 0.5 },
    };
    const out = cfg.clamped();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.exposure, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.dof_focus_distance, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.dof_focus_range, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.dof_max_blur, 1e-6);
    try std.testing.expectEqual([3]f32{ 1.0, -1.0, 0.5 }, out.grade_shadows);
}

test "ssr and motion blur clamp quality options" {
    var custom = PostProcessOptions{
        .ssr_steps = 1,
        .motion_blur_samples = 100,
    };
    const c = custom.clamped();
    try std.testing.expectEqual(@as(u32, 4), c.ssr_steps);
    try std.testing.expectEqual(@as(u32, 32), c.motion_blur_samples);
}

test "manual exposure is finite and bounded for every output frame" {
    const defaults = PostProcessOptions{};
    try std.testing.expectEqual(defaults, defaults.clamped());
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |value| {
        const cfg = (PostProcessOptions{ .exposure = value }).clamped();
        try std.testing.expectEqual(@as(f32, 1), cfg.exposure);
    }
    try std.testing.expectEqual(@as(f32, 0), (PostProcessOptions{ .exposure = -2 }).clamped().exposure);
    try std.testing.expectEqual(@as(f32, 65504), (PostProcessOptions{ .exposure = 1e20 }).clamped().exposure);
}

test "effect master preserves exposure and tone curve without mutating authored config" {
    const config = PostProcessOptions{ .exposure = 2, .tonemapping = .reinhard, .taa_enabled = true, .glow_enabled = true };
    const frame = config.forFrame();
    try std.testing.expectEqual(@as(f32, 2), frame.exposure);
    try std.testing.expectEqual(TonemappingType.reinhard, frame.tonemapping);
    try std.testing.expect(!frame.bloom_enabled and !frame.glow_enabled and !frame.taa_enabled and !frame.fxaa_enabled and !frame.fog_enabled and !frame.ssr_enabled);
    try std.testing.expectEqual(@as(f32, 1), frame.saturation);
    try std.testing.expectEqual(@as(f32, 1), frame.contrast);
    try std.testing.expect(config.taa_enabled and config.glow_enabled);
    const enabled = PostProcessOptions{ .enabled = true, .taa_enabled = true };
    try std.testing.expectEqual(enabled.clamped(), enabled.forFrame());
}
