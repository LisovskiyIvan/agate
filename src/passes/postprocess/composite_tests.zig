//! Composite packing/binding tests exercise the unchanged PostProcessPass API.
const std = @import("std");
const testing = std.testing;
const sg = @import("sokol").gfx;
const post_shd = @import("postprocess_shader");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const Color3 = math.Color3;
const PostProcessOptions = @import("../../postprocess.zig").PostProcessOptions;
const PostProcessPass = @import("../postprocess_pass.zig").PostProcessPass;

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

fn packRepresentative(pass: *const PostProcessPass, config: PostProcessOptions) post_shd.FsParams {
    return pass.packCompositeParams(
        config,
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
        Mat4.identity,
    );
}

test "packCompositeParams carries camera range with zeroed zw and ssr steps in its own lane" {
    const pass = PostProcessPass{};
    const params = packRepresentative(&pass, .{ .ssr = .{ .enabled = true, .steps = 24 } });
    // camera_params is (near_z, far_z, 0, 0): no smuggled step count.
    try testing.expectEqual([4]f32{ 0.1, 100.0, 0.0, 0.0 }, params.camera_params);
    // The step count reaches the shader via the dedicated lane instead.
    try testing.expectEqual([4]f32{ 24.0, 0.0, 0.0, 0.0 }, params.ssr_params2);
    // Neighboring lanes ride through untouched by the move.
    try testing.expectEqual([4]f32{ 0.0, 0.0, 1.0, 0.0 }, params.ssao_params);
    try testing.expectEqual([4]f32{ 160.0, 90.0, 1.0 / 160.0, 1.0 / 90.0 }, params.resolution);
    try testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 0.0 }, params.output_params);
}

test "packCompositeParams separates scene texels from output-res TAA texels" {
    var pass = PostProcessPass{ .taa_width = 320, .taa_height = 180 };
    const native = packRepresentative(&pass, .{});
    // Scene taps remain at the scaled EFFECTIVE size...
    try testing.expectEqual([4]f32{ 160.0, 90.0, 1.0 / 160.0, 1.0 / 90.0 }, native.resolution);
    // ...while the TAA variance box steps by the display-sized history.
    try testing.expectEqual([4]f32{ 320.0, 180.0, 1.0 / 320.0, 1.0 / 180.0 }, native.taa_resolution);

    const no_history = PostProcessPass{};
    const fallback = packRepresentative(&no_history, .{});
    try testing.expectEqual(native.resolution, fallback.taa_resolution);
}

test "packCompositeParams zeroes the ssr steps lane while keeping camera range when ssr is off" {
    const pass = PostProcessPass{};
    const params = packRepresentative(&pass, .{ .ssr = .{ .enabled = false, .steps = 24 } });
    try testing.expectEqual([4]f32{ 0.1, 100.0, 0.0, 0.0 }, params.camera_params);
    try testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, params.ssr_params2);
    // Inactive inline lane packs zeros (shader gates on x; bit-identical).
    try testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, params.ssr_params);
    try testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, params.ssr_pass_params);
}

test "packCompositeParams splits ssr inline and pass lanes without double-add" {
    const pass = PostProcessPass{};
    const inline_params = packRepresentative(&pass, .{ .ssr = .{ .enabled = true, .intensity = 0.6, .steps = 32 } });
    try testing.expectEqual([4]f32{ 1.0, 0.6, 0.4, 25.0 }, inline_params.ssr_params);
    try testing.expectEqual([4]f32{ 32.0, 0.0, 0.0, 0.0 }, inline_params.ssr_params2);
    try testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, inline_params.ssr_pass_params);
    var fed = PostProcessPass{ .ssr_tex_view = .{ .id = 77 } };
    const pass_params = fed.packCompositeParams(
        .{ .ssr = .{ .enabled = true, .intensity = 0.6, .steps = 32, .mode = .half_res_pass } },
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
        Mat4.identity,
    );
    try testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pass_params.ssr_params);
    try testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pass_params.ssr_params2);
    try testing.expectEqual([4]f32{ 1.0, 0.6, 0.0, 0.0 }, pass_params.ssr_pass_params);
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
    try testing.expectEqual(resolve.id, bind.views[post_shd.VIEW_ssgi_tex].id);
    try testing.expectEqual(resolve.id, bind.views[post_shd.VIEW_ssr_tex].id);
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
