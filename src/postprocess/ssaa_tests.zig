const std = @import("std");
const ssaa = @import("ssaa.zig");
const options = @import("options.zig");
const PostProcessOptions = options.PostProcessOptions;

test "ssaa: default options are disabled with factor 2.0" {
    const opts = ssaa.SsaaOptions{};
    try std.testing.expect(!opts.enabled);
    try std.testing.expectEqual(@as(f32, 2.0), opts.factor);
    try std.testing.expect(!ssaa.ssaaActive(opts));
    try std.testing.expectEqual(@as(f32, 1.0), ssaa.ssaaFactor(opts));
}

test "ssaa: factor clamping and NaN fallback" {
    var opts = ssaa.SsaaOptions{ .enabled = true, .factor = 0.5 };
    try std.testing.expectEqual(ssaa.MIN_SSAA_FACTOR, ssaa.ssaaFactor(opts));

    opts.factor = 3.5;
    try std.testing.expectEqual(ssaa.MAX_SSAA_FACTOR, ssaa.ssaaFactor(opts));

    opts.factor = 1.5;
    try std.testing.expectEqual(@as(f32, 1.5), ssaa.ssaaFactor(opts));

    opts.factor = std.math.nan(f32);
    try std.testing.expectEqual(ssaa.DEFAULT_SSAA_FACTOR, ssaa.ssaaFactor(opts));

    opts.factor = std.math.inf(f32);
    try std.testing.expectEqual(ssaa.DEFAULT_SSAA_FACTOR, ssaa.ssaaFactor(opts));
}

test "ssaa: effectiveRenderScale combines render_scale and ssaa" {
    var post = PostProcessOptions{};
    // Default: native 1.0
    try std.testing.expectEqual(@as(f32, 1.0), ssaa.effectiveRenderScale(post));

    // Dynamic resolution / downscale
    post.render_scale = 0.67;
    try std.testing.expectApproxEqAbs(@as(f32, 0.67), ssaa.effectiveRenderScale(post), 1e-5);

    // SSAA enabled overrides downscale
    post.ssaa.enabled = true;
    post.ssaa.factor = 1.5;
    try std.testing.expectEqual(@as(f32, 1.5), ssaa.effectiveRenderScale(post));

    post.ssaa.factor = 2.0;
    try std.testing.expectEqual(@as(f32, 2.0), ssaa.effectiveRenderScale(post));

    // When SSAA disabled, restores render_scale
    post.ssaa.enabled = false;
    try std.testing.expectApproxEqAbs(@as(f32, 0.67), ssaa.effectiveRenderScale(post), 1e-5);
}

test "ssaa: sceneDownscaleEnabled detects supersampled scene target" {
    // Standard 2x SSAA: scene 1920x1080 -> output 960x540
    try std.testing.expect(ssaa.sceneDownscaleEnabled(960, 540, 1920, 1080));

    // 1.5x SSAA: scene 1440x810 -> output 960x540
    try std.testing.expect(ssaa.sceneDownscaleEnabled(960, 540, 1440, 810));

    // Native 1:1 -> false (no downscale)
    try std.testing.expect(!ssaa.sceneDownscaleEnabled(960, 540, 960, 540));

    // Render scale < 1.0 -> false (upscale, not downscale)
    try std.testing.expect(!ssaa.sceneDownscaleEnabled(960, 540, 640, 360));

    // Zero / absent output size falls back to scene size -> false
    try std.testing.expect(!ssaa.sceneDownscaleEnabled(0, 0, 1920, 1080));
}

test "ssaa: downsampleOffsets quadrant symmetry" {
    const offsets = ssaa.downsampleOffsets(1920.0, 1080.0);
    const expected_x = 0.25 / 1920.0;
    const expected_y = 0.25 / 1080.0;

    try std.testing.expectApproxEqAbs(-expected_x, offsets[0][0], 1e-7);
    try std.testing.expectApproxEqAbs(-expected_y, offsets[0][1], 1e-7);

    try std.testing.expectApproxEqAbs(expected_x, offsets[1][0], 1e-7);
    try std.testing.expectApproxEqAbs(-expected_y, offsets[1][1], 1e-7);

    try std.testing.expectApproxEqAbs(-expected_x, offsets[2][0], 1e-7);
    try std.testing.expectApproxEqAbs(expected_y, offsets[2][1], 1e-7);

    try std.testing.expectApproxEqAbs(expected_x, offsets[3][0], 1e-7);
    try std.testing.expectApproxEqAbs(expected_y, offsets[3][1], 1e-7);

    // Sum of quadrant offsets must be zero (unbiased centroid)
    var sum_x: f32 = 0.0;
    var sum_y: f32 = 0.0;
    for (offsets) |o| {
        sum_x += o[0];
        sum_y += o[1];
    }
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sum_x, 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sum_y, 1e-7);
}

test "ssaa: downsampleFilter energy conservation and blend linearity" {
    // Uniform white -> uniform white
    const white: [4][3]f32 = @splat([3]f32{ 1.0, 1.0, 1.0 });
    const res_white = ssaa.downsampleFilter(white);
    try std.testing.expectEqual([3]f32{ 1.0, 1.0, 1.0 }, res_white);

    // Checkerboard (2 black, 2 white) -> 50% grey
    const checker: [4][3]f32 = .{
        .{ 1.0, 1.0, 1.0 },
        .{ 0.0, 0.0, 0.0 },
        .{ 0.0, 0.0, 0.0 },
        .{ 1.0, 1.0, 1.0 },
    };
    const res_checker = ssaa.downsampleFilter(checker);
    try std.testing.expectEqual([3]f32{ 0.5, 0.5, 0.5 }, res_checker);

    // 3 white, 1 black -> 0.75
    const edge: [4][3]f32 = .{
        .{ 1.0, 1.0, 1.0 },
        .{ 1.0, 1.0, 1.0 },
        .{ 1.0, 1.0, 1.0 },
        .{ 0.0, 0.0, 0.0 },
    };
    const res_edge = ssaa.downsampleFilter(edge);
    try std.testing.expectEqual([3]f32{ 0.75, 0.75, 0.75 }, res_edge);
}

test "ssaa: ssaaRenderSize and renderSizeForScale produce 2x dimensions with aspect preservation" {
    const orig_w: i32 = 1280;
    const orig_h: i32 = 720;
    const size = ssaa.ssaaRenderSize(orig_w, orig_h, 2.0);
    try std.testing.expectEqual(@as(i32, 2560), size.w);
    try std.testing.expectEqual(@as(i32, 1440), size.h);

    const for_scale = ssaa.renderSizeForScale(orig_w, orig_h, 1.5);
    try std.testing.expectEqual(@as(i32, 1920), for_scale.w);
    try std.testing.expectEqual(@as(i32, 1080), for_scale.h);
}
