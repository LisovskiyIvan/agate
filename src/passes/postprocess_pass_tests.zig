const std = @import("std");
const testing = std.testing;
const sokol = @import("sokol");
const sg = sokol.gfx;
const PostProcessPass = @import("postprocess_pass.zig").PostProcessPass;

test "outputParamsFor packs manual-encode flag, yzw zero" {
    // UNORM backbuffer: manual encode on.
    try testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 0.0 }, PostProcessPass.outputParamsFor(.BGRA8));
    try testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 0.0 }, PostProcessPass.outputParamsFor(.RGBA8));
    try testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 0.0 }, PostProcessPass.outputParamsFor(.RGBA16F));
    try testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 0.0 }, PostProcessPass.outputParamsFor(.DEFAULT));
    try testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 0.0 }, PostProcessPass.outputParamsFor(.NONE));
    // sRGB backbuffer: hardware encodes, no manual pass.
    try testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, PostProcessPass.outputParamsFor(.SRGB8A8));
    try testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, PostProcessPass.outputParamsFor(.SBGR8A8));
}

test "isSrgbBackbuffer flags exactly the hardware-sRGB swapchain variants" {
    try testing.expect(PostProcessPass.isSrgbBackbuffer(.SRGB8A8));
    try testing.expect(PostProcessPass.isSrgbBackbuffer(.SBGR8A8));
    try testing.expect(!PostProcessPass.isSrgbBackbuffer(.RGBA8));
    try testing.expect(!PostProcessPass.isSrgbBackbuffer(.BGRA8));
    try testing.expect(!PostProcessPass.isSrgbBackbuffer(.RGBA16F));
    try testing.expect(!PostProcessPass.isSrgbBackbuffer(.DEFAULT));
    try testing.expect(!PostProcessPass.isSrgbBackbuffer(.NONE));
}

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
