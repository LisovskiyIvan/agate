const std = @import("std");
const testing = std.testing;
const sokol = @import("sokol");
const sg = sokol.gfx;
const post_shd = @import("postprocess_shader");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const Color3 = math.Color3;
const postprocess = @import("../postprocess.zig");
const PostProcessOptions = postprocess.PostProcessOptions;
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

fn packRepresentative(pass: *const PostProcessPass, config: PostProcessOptions) post_shd.FsParams {
    return pass.packCompositeParams(
        config,
        false,
        false,
        1.0,
        160,
        90,
        Mat4.identity,
        Mat4.identity,
        Mat4.identity,
        Vec3.new(1.0, 2.0, 3.0),
        Vec3.new(0.0, 1.0, 0.0),
        Color3.new(1.0, 1.0, 1.0),
        0.1,
        100.0,
        false,
        false,
        .BGRA8,
    );
}

test "packCompositeParams carries camera range with zeroed zw and ssr steps in its own lane" {
    const pass = PostProcessPass{};
    const params = packRepresentative(&pass, .{ .ssr_enabled = true, .ssr_steps = 24 });
    // camera_params is (near_z, far_z, 0, 0): no smuggled step count.
    try testing.expectEqual([4]f32{ 0.1, 100.0, 0.0, 0.0 }, params.camera_params);
    // The step count reaches the shader via the dedicated lane instead.
    try testing.expectEqual([4]f32{ 24.0, 0.0, 0.0, 0.0 }, params.ssr_params2);
    // Neighboring lanes ride through untouched by the move.
    try testing.expectEqual([4]f32{ 0.0, 0.0, 1.0, 0.0 }, params.ssao_params);
    try testing.expectEqual([4]f32{ 160.0, 90.0, 1.0 / 160.0, 1.0 / 90.0 }, params.resolution);
    try testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 0.0 }, params.output_params);
}

test "packCompositeParams zeroes the ssr steps lane while keeping camera range when ssr is off" {
    const pass = PostProcessPass{};
    const params = packRepresentative(&pass, .{ .ssr_enabled = false, .ssr_steps = 24 });
    try testing.expectEqual([4]f32{ 0.1, 100.0, 0.0, 0.0 }, params.camera_params);
    try testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, params.ssr_params2);
    try testing.expectEqual([4]f32{ 0.0, 0.55, 0.4, 25.0 }, params.ssr_params);
}

test "buildCompositeBindings falls back to placeholders for every empty optional view" {
    const resolve = sg.View{ .id = 11 };
    const zero = sg.View{ .id = 12 };
    const pass = PostProcessPass{
        .offscreen_resolve_tex_view = resolve,
        .default_zero_view = zero,
    };
    const ssao = sg.View{ .id = 21 };
    const depth = sg.View{ .id = 22 };
    const bind = pass.buildCompositeBindings(PostProcessOptions{}, ssao, depth, .{}, .{});
    try testing.expectEqual(ssao.id, bind.views[post_shd.VIEW_ssao_tex].id);
    try testing.expectEqual(depth.id, bind.views[post_shd.VIEW_depth_tex].id);
    try testing.expectEqual(resolve.id, bind.views[post_shd.VIEW_scene_tex].id);
    try testing.expectEqual(resolve.id, bind.views[post_shd.VIEW_bloom_tex].id);
    try testing.expectEqual(resolve.id, bind.views[post_shd.VIEW_glow_tex].id);
    try testing.expectEqual(resolve.id, bind.views[post_shd.VIEW_highlight_tex].id);
    try testing.expectEqual(resolve.id, bind.views[post_shd.VIEW_highlight_mask_tex].id);
    try testing.expectEqual(resolve.id, bind.views[post_shd.VIEW_shaft_tex].id);
    try testing.expectEqual(resolve.id, bind.views[post_shd.VIEW_lut_tex].id);
    try testing.expectEqual(resolve.id, bind.views[post_shd.VIEW_history_tex].id);
    // No velocity view fed: the zero mask (depth-reprojection fallback).
    try testing.expectEqual(zero.id, bind.views[post_shd.VIEW_velocity_tex].id);
}

test "buildCompositeBindings binds fed effect and velocity views instead of placeholders" {
    const resolve = sg.View{ .id = 11 };
    const zero = sg.View{ .id = 12 };
    var pass = PostProcessPass{
        .offscreen_resolve_tex_view = resolve,
        .default_zero_view = zero,
        .bloom_tex_view = .{ .id = 31 },
        .glow_tex_view = .{ .id = 32 },
        .highlight_tex_view = .{ .id = 33 },
        .highlight_mask_tex_view = .{ .id = 34 },
        .shaft_tex_view = .{ .id = 35 },
    };
    const history = sg.View{ .id = 41 };
    const velocity = sg.View{ .id = 42 };
    const bind = pass.buildCompositeBindings(PostProcessOptions{}, .{}, .{}, history, velocity);
    try testing.expectEqual(@as(u32, 31), bind.views[post_shd.VIEW_bloom_tex].id);
    try testing.expectEqual(@as(u32, 32), bind.views[post_shd.VIEW_glow_tex].id);
    try testing.expectEqual(@as(u32, 33), bind.views[post_shd.VIEW_highlight_tex].id);
    try testing.expectEqual(@as(u32, 34), bind.views[post_shd.VIEW_highlight_mask_tex].id);
    try testing.expectEqual(@as(u32, 35), bind.views[post_shd.VIEW_shaft_tex].id);
    try testing.expectEqual(history.id, bind.views[post_shd.VIEW_history_tex].id);
    try testing.expectEqual(velocity.id, bind.views[post_shd.VIEW_velocity_tex].id);
    // The fed bloom view also flips the composite bloom gate on.
    const params = packRepresentative(&pass, .{});
    try testing.expectEqual(@as(f32, 1.0), params.params3[2]);
}
