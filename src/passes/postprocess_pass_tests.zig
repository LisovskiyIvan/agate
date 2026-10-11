const std = @import("std");
const testing = std.testing;
const sokol = @import("sokol");
const sg = sokol.gfx;
const postprocess = @import("../postprocess.zig");
const PostProcessOptions = postprocess.PostProcessOptions;
const PostProcessPass = @import("postprocess_pass.zig").PostProcessPass;

test "fullscreenPipelineDesc keeps display shape default, capture shape explicit 1x" {
    const shd: sg.Shader = .{};
    const display = PostProcessPass.fullscreenPipelineDesc(shd, .DEFAULT, 0, .DEFAULT);
    try testing.expectEqual(sg.PixelFormat.DEFAULT, display.colors[0].pixel_format);
    try testing.expectEqual(@as(i32, 0), display.sample_count);
    try testing.expectEqual(sg.PixelFormat.DEFAULT, display.depth.pixel_format);
    try testing.expect(display.depth.compare == .ALWAYS);
    try testing.expect(!display.depth.write_enabled);
    try testing.expect(display.cull_mode == .NONE);
    try testing.expect(display.index_type == .UINT16);

    const capture = PostProcessPass.fullscreenPipelineDesc(shd, .RGBA16F, 1, .NONE);
    try testing.expectEqual(sg.PixelFormat.RGBA16F, capture.colors[0].pixel_format);
    try testing.expectEqual(@as(i32, 1), capture.sample_count);
    try testing.expectEqual(sg.PixelFormat.NONE, capture.depth.pixel_format);
    // Same quad layout on both (shared helper, no duplicate layout).
    try testing.expectEqual(display.layout.buffers[0].stride, capture.layout.buffers[0].stride);
    try testing.expectEqual(
        display.layout.attrs[@import("postprocess_shader").ATTR_postprocess_position].format,
        capture.layout.attrs[@import("postprocess_shader").ATTR_postprocess_position].format,
    );
}

test "zero pass reports no valid targets and default shape metadata" {
    const p = PostProcessPass{};
    try testing.expectEqual(@as(i32, 0), p.width);
    try testing.expectEqual(@as(i32, 0), p.height);
    try testing.expectEqual(@as(i32, 1), p.sample_count);
    try testing.expectEqual(sg.PixelFormat.DEFAULT, p.color_format);
    try testing.expectEqual(sg.PixelFormat.DEFAULT, p.taa_format);
    try testing.expectEqual(@as(u32, 0), p.taa_pipeline.id);
    // Empty read view: callers bind a dummy placeholder, never this.
    try testing.expectEqual(@as(u32, 0), p.taaReadView().id);
    // Headless guards fail closed without touching sg resources.
    try testing.expect(!p.targetsValid());
    try testing.expect(!p.taaAvailable());
    var q = PostProcessPass{};
    try testing.expect(!q.resize(64, 64, 1));
    try testing.expect(!q.ensureTaaHistory(64, 64));
    try testing.expect(!q.taaAvailable());
}

test "ssaoParams packs the named lane with a zero spare" {
    // Enabled: flags on, intensity rides through raw (shader gates on x/y).
    try testing.expectEqual([4]f32{ 1.0, 0.0, 1.1, 0.0 }, postprocess.ssaoParams(true, false, 1.1));
    // Debug view without the AO multiply: y on, x off.
    try testing.expectEqual([4]f32{ 0.0, 1.0, 0.5, 0.0 }, postprocess.ssaoParams(false, true, 0.5));
    // Both off: flags zero, intensity still rides raw (unread while gated).
    try testing.expectEqual([4]f32{ 0.0, 0.0, 1.1, 0.0 }, postprocess.ssaoParams(false, false, 1.1));
}

test "fxaaParams packs the named lane with zero spares" {
    try testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 0.0 }, postprocess.fxaaParams(true));
    try testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, postprocess.fxaaParams(false));
}

test "ssrStepsParams packs the dedicated steps lane with zero spares" {
    // Enabled: the step count rides in x.
    var cfg = PostProcessOptions{ .ssr_enabled = true, .ssr_steps = 24 };
    try testing.expectEqual([4]f32{ 24.0, 0.0, 0.0, 0.0 }, postprocess.ssrStepsParams(cfg));
    // Disabled: zeros (the shader early-outs on ssr_params.x first).
    cfg.ssr_enabled = false;
    try testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, postprocess.ssrStepsParams(cfg));
}
