const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color3 = math.Color3;
const material = @import("../material.zig");
const uniforms = @import("uniforms.zig");
const ShadowState = uniforms.ShadowState;
const FrameContext = uniforms.FrameContext;
const FrameUniforms = uniforms.FrameUniforms;
const buildFrameUniforms = uniforms.buildFrameUniforms;
const alphaCutoffFor = uniforms.alphaCutoffFor;

test "buildFrameUniforms passes directional lanes through, legacy lanes intact" {
    const shadow = ShadowState{
        .ground_color = Color3.new(0.2, 0.25, 0.3),
        .enable_shadows = true,
        .mesh_receive_shadows = true,
        .bias = 0.0012,
        .intensity = 0.75,
        .normal_bias = 0.02,
        .softness = 1.5,
        .debug_cascades = false,
        .splits = .{ 10.0, 26.0, 65.0, 150.0 },
    };
    // Single-light default: slot 0 mirrors the sun, slots 1..3 zeroed.
    const ctx = FrameContext{
        .view_proj = Mat4.identity,
        .eye = Vec3.new(1.0, 2.0, 3.0),
        .sun_dir = Vec3.new(0.5, 1.0, 0.3),
        .sun_color = Color3.new(1.0, 0.9, 0.8),
        .sun_intensity = 2.0,
        .directional_dir = .{ .{ 0.5, 1.0, 0.3, 0.0 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 } },
        .directional_color_int = .{ .{ 1.0, 0.9, 0.8, 2.0 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 } },
        .cascades = [_]Mat4{Mat4.identity} ** 4,
        .light_counts = .{ 0.0, 0.0, 0.0, 0.0 },
        .point_pos_range = [_][4]f32{.{ 0, 0, 0, 0 }} ** 4,
        .point_color_int = [_][4]f32{.{ 0, 0, 0, 0 }} ** 4,
        .spot_pos_range = [_][4]f32{.{ 0, 0, 0, 0 }} ** 2,
        .spot_dir_inner = [_][4]f32{.{ 0, 0, 0, 0 }} ** 2,
        .spot_color_outer = [_][4]f32{.{ 0, 0, 0, 0 }} ** 2,
        .spot_intensity = [_][4]f32{.{ 0, 0, 0, 0 }} ** 2,
        .spot_view_proj = [_]Mat4{Mat4.identity} ** 2,
        .spot_shadow_params = [_][4]f32{.{ 0, 0, 0, 0 }} ** 2,
        .point_view_proj = [_]Mat4{Mat4.identity} ** 12,
        .point_shadow_params = [_][4]f32{.{ 0, 0, 0, 0 }} ** 4,
        .area_center_int = .{ .{ 1.0, 2.0, 3.0, 3.0 }, .{ 0, 0, 0, 0 } },
        .area_right = .{ .{ 2.0, 0.0, 0.0, 0.0 }, .{ 0, 0, 0, 0 } },
        .area_up = .{ .{ 0.0, 0.5, 0.0, 0.0 }, .{ 0, 0, 0, 0 } },
        .area_color = .{ .{ 1.0, 0.5, 0.25, 0.0 }, .{ 0, 0, 0, 0 } },
        .clustered_params = .{ 10.0, 6.0, 3.0, 1.0 },
        .clustered_viewport = .{ 640.0, 384.0, 64.0, 0.0 },
    };
    const f = buildFrameUniforms(shadow, &ctx);
    // Legacy lanes keep their exact legacy values...
    try std.testing.expectEqual([4]f32{ 0.5, 1.0, 0.3, 2048.0 }, f.light_dir);
    try std.testing.expectEqual([4]f32{ 1.0, 0.9, 0.8, 2.0 }, f.light_color);
    // ...and the directional lanes ride through verbatim.
    try std.testing.expectEqual(ctx.directional_dir, f.directional_dir);
    try std.testing.expectEqual(ctx.directional_color_int, f.directional_color_int);
    // ...and the area lanes ride through verbatim.
    try std.testing.expectEqual(ctx.area_center_int, f.area_center_int);
    try std.testing.expectEqual(ctx.area_right, f.area_right);
    try std.testing.expectEqual(ctx.area_up, f.area_up);
    try std.testing.expectEqual(ctx.area_color, f.area_color);
    // ...and the clustered tile descriptor rides through verbatim.
    try std.testing.expectEqual(ctx.clustered_params, f.clustered_params);
    try std.testing.expectEqual(ctx.clustered_viewport, f.clustered_viewport);
    // The shared per-view uniforms always carry a neutral probe lane (the
    // draw overwrites it per draw from the probe selection, or leaves it —
    // so the no-probe path is bit-identical to before this wave).
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, f.probe_params);
}

test "linear output contract: no output_params lanes on the frame structs" {
    // Scene shaders always write linear radiance (see
    // shaders/common/linear_output.glsl): no gamma/HDR mode lanes ride the
    // frame uniforms, and the display transfer lives in postprocess only.
    try std.testing.expect(!@hasField(FrameContext, "output_params"));
    try std.testing.expect(!@hasField(FrameUniforms, "output_params"));
}

test "FrameUniforms appends area lanes last (existing offsets unmoved)" {
    // Every pre-area field keeps its offset relative to the struct start
    // regardless of the appended lanes: spot-check the tail neighbors.
    try std.testing.expect(@offsetOf(FrameUniforms, "area_center_int") > @offsetOf(FrameUniforms, "probe_params"));
    try std.testing.expect(@offsetOf(FrameUniforms, "area_right") > @offsetOf(FrameUniforms, "area_center_int"));
    try std.testing.expect(@offsetOf(FrameUniforms, "area_up") > @offsetOf(FrameUniforms, "area_right"));
    try std.testing.expect(@offsetOf(FrameUniforms, "area_color") > @offsetOf(FrameUniforms, "area_up"));
    try std.testing.expect(@offsetOf(FrameUniforms, "directional_dir") < @offsetOf(FrameUniforms, "probe_params"));
}

test "FrameUniforms appends clustered lanes after the area lanes (offsets unmoved)" {
    // The clustered tile descriptor is the new tail: everything before it
    // (area lanes included) keeps its offset, so the empty pool uploads
    // bit-identical values on every pre-existing lane.
    try std.testing.expect(@offsetOf(FrameUniforms, "clustered_params") > @offsetOf(FrameUniforms, "area_color"));
    try std.testing.expect(@offsetOf(FrameUniforms, "clustered_viewport") > @offsetOf(FrameUniforms, "clustered_params"));
    // FrameContext carries the same appended tail for the shared packing.
    // Its *offsets* are deliberately NOT asserted: Zig's auto layout is free
    // to reorder equal-size fields (adding one lane did reshuffle the two
    // clustered lanes), and nothing reads this struct as bytes — every
    // FsParams is packed field by field — so only the field set is a
    // contract. The order that DOES matter is the shader block's, and that is
    // the GLSL source plus the `FsParams` contract test in scene/draw.zig.
    try std.testing.expect(@hasField(FrameContext, "clustered_params"));
    try std.testing.expect(@hasField(FrameContext, "clustered_viewport"));
}

test "alphaCutoffFor gates the cutoff on cutout mode" {
    // No material: opaque legacy, test disabled.
    try std.testing.expectEqual(@as(f32, 0.0), alphaCutoffFor(null));

    var pbr_mat = material.PBRMaterial.init("p");

    // Opaque and blend upload 0.0 even with a customized cutoff stored.
    pbr_mat.alpha_cutoff = 0.7;
    for ([_]material.AlphaMode{ .@"opaque", .blend }) |mode| {
        pbr_mat.alpha_mode = mode;
        try std.testing.expectEqual(@as(f32, 0.0), alphaCutoffFor(.{ .pbr = &pbr_mat }));
    }

    // Cutout uploads the material value (default 0.5).
    pbr_mat.alpha_mode = .cutout;
    try std.testing.expectEqual(@as(f32, 0.7), alphaCutoffFor(.{ .pbr = &pbr_mat }));

    var pbr_default = material.PBRMaterial.init("d");
    pbr_default.alpha_mode = .cutout;
    try std.testing.expectEqual(@as(f32, 0.5), alphaCutoffFor(.{ .pbr = &pbr_default }));
}
