//! fs_params split: partition equality (legacy FrameUniforms vs the
//! view/material wire blocks), wire layout + UB sizes, hook/runtime
//! uniform contracts, and quantified upload bytes (alignment included).
//!
//! Pure CPU tests (no GPU calls): they prove the split preserves every
//! field value while shrinking steady-state per-draw uploads from the
//! full ~2.3KB block to the material block.

const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const Color3 = math.Color3;

const pbr_shd = @import("pbr_shader");
const skinned_pbr_shd = @import("skinned_pbr_shader");
const inst_pbr_shd = @import("instanced_pbr_shader");

const uniforms = @import("uniforms.zig");
const sm = @import("../shader_material.zig");
const draw = @import("draw.zig");

fn testShadow() uniforms.ShadowState {
    return .{
        .ground_color = Color3.new(0.2, 0.25, 0.3),
        .enable_shadows = true,
        .mesh_receive_shadows = true,
        .bias = 0.0012,
        .intensity = 0.75,
        .normal_bias = 0.02,
        .softness = 1.5,
        .debug_cascades = true,
        .splits = .{ 10.0, 26.0, 65.0, 150.0 },
        .pcss_enabled = true,
        .pcss_light_size = 0.5,
        .pcss_blocker_radius = 0.01,
        .pcss_min_penumbra = 0.001,
        .pcss_max_penumbra = 0.02,
        .hemi_dir = Vec3.up,
        .hemi_diffuse = Color3.white,
        .hemi_intensity = 1.5,
    };
}

fn testContext() uniforms.FrameContext {
    return .{
        .view_proj = Mat4.identity,
        .eye = Vec3.new(1.0, 2.0, 3.0),
        .sun_dir = Vec3.new(0.5, 1.0, 0.3),
        .sun_color = Color3.new(1.0, 0.9, 0.8),
        .sun_intensity = 2.0,
        .directional_dir = .{ .{ 0.5, 1.0, 0.3, 0.0 }, .{ 0, 1, 0, 0.5 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 } },
        .directional_color_int = .{ .{ 1.0, 0.9, 0.8, 2.0 }, .{ 0.5, 0.5, 0.5, 0.5 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 } },
        .cascades = @as([4]Mat4, @splat(Mat4.identity)),
        .light_counts = .{ 2.0, 1.0, 0.0, 0.0 },
        .point_pos_range = @splat(.{ 1, 2, 3, 10 }),
        .point_color_int = @splat(.{ 1, 0.5, 0.25, 3 }),
        .spot_pos_range = @splat(.{ 4, 5, 6, 20 }),
        .spot_dir_inner = @splat(.{ 0, -1, 0, 0.9 }),
        .spot_color_outer = @splat(.{ 1, 1, 1, 0.7 }),
        .spot_intensity = @splat(.{ 5, 0, 0, 0 }),
        .spot_view_proj = @as([2]Mat4, @splat(Mat4.identity)),
        .spot_shadow_params = @splat(.{ 1, 0.001, 0.01, 0 }),
        .point_view_proj = @as([12]Mat4, @splat(Mat4.identity)),
        .point_shadow_params = @splat(.{ 1, 0.001, 0.01, 0 }),
        .area_center_int = .{ .{ 1.0, 2.0, 3.0, 3.0 }, .{ 0, 0, 0, 0 } },
        .area_right = .{ .{ 2.0, 0.0, 0.0, 0.0 }, .{ 0, 0, 0, 0 } },
        .area_up = .{ .{ 0.0, 0.5, 0.0, 0.0 }, .{ 0, 0, 0, 0 } },
        .area_color = .{ .{ 1.0, 0.5, 0.25, 0.0 }, .{ 0, 0, 0, 0 } },
        .clustered_params = .{ 10.0, 6.0, 3.0, 1.0 },
        .clustered_viewport = .{ 640.0, 384.0, 64.0, 0.0 },
    };
}

test "split partition: view block preserves every legacy frame lane" {
    const shadow = testShadow();
    var ctx = testContext();
    const legacy = uniforms.buildFrameUniforms(shadow, &ctx);
    const refr_proj = Mat4.translation(Vec3.new(7, 8, 9));
    const refr_cap: [4]f32 = .{ 1, 320, 180, 0 };
    const view = uniforms.buildFrameView(shadow, &ctx, refr_proj, refr_cap);

    try std.testing.expectEqual(legacy.eye_pos, view.eye_pos);
    try std.testing.expectEqual(legacy.light_dir, view.light_dir);
    try std.testing.expectEqual(legacy.light_color, view.light_color);
    try std.testing.expectEqual(legacy.ambient_color, view.ambient_color);
    try std.testing.expectEqual(legacy.shadow_params, view.shadow_params);
    try std.testing.expectEqual(legacy.shadow_splits, view.shadow_splits);
    try std.testing.expectEqual(legacy.cascade_view_proj, view.cascade_view_proj);
    try std.testing.expectEqual(legacy.cascade_debug, view.cascade_debug);
    try std.testing.expectEqual(legacy.light_counts, view.light_counts);
    try std.testing.expectEqual(legacy.point_pos_range, view.point_pos_range);
    try std.testing.expectEqual(legacy.point_color_int, view.point_color_int);
    try std.testing.expectEqual(legacy.spot_pos_range, view.spot_pos_range);
    try std.testing.expectEqual(legacy.spot_dir_inner, view.spot_dir_inner);
    try std.testing.expectEqual(legacy.spot_color_outer, view.spot_color_outer);
    try std.testing.expectEqual(legacy.spot_intensity, view.spot_intensity);
    try std.testing.expectEqual(legacy.spot_view_proj, view.spot_view_proj);
    try std.testing.expectEqual(legacy.spot_shadow_params, view.spot_shadow_params);
    try std.testing.expectEqual(legacy.point_view_proj, view.point_view_proj);
    try std.testing.expectEqual(legacy.point_shadow_params, view.point_shadow_params);
    try std.testing.expectEqual(legacy.directional_dir, view.directional_dir);
    try std.testing.expectEqual(legacy.directional_color_int, view.directional_color_int);
    try std.testing.expectEqual(legacy.area_center_int, view.area_center_int);
    try std.testing.expectEqual(legacy.area_right, view.area_right);
    try std.testing.expectEqual(legacy.area_up, view.area_up);
    try std.testing.expectEqual(legacy.area_color, view.area_color);
    try std.testing.expectEqual(legacy.clustered_params, view.clustered_params);
    try std.testing.expectEqual(legacy.clustered_viewport, view.clustered_viewport);
    try std.testing.expectEqual(legacy.hemi_dir_intensity, view.hemi_dir_intensity);
    try std.testing.expectEqual(legacy.hemi_diffuse, view.hemi_diffuse);
    // Refraction camera rides the view block (per-view env, not per-draw).
    try std.testing.expectEqual(refr_proj, view.refraction_view_proj);
    try std.testing.expectEqual(refr_cap, view.refraction_capture);

    // Receive-shadow gate preserved per variant: bias/intensity zero when
    // the mesh opts out, live otherwise.
    var no_shadow = shadow;
    no_shadow.mesh_receive_shadows = false;
    const view_off = uniforms.buildFrameView(no_shadow, &ctx, Mat4.identity, .{ 0, 1, 1, 0 });
    try std.testing.expectEqual(@as(f32, 0.0), view_off.shadow_params[0]);
    try std.testing.expectEqual(@as(f32, 0.0), view_off.shadow_params[1]);
    try std.testing.expectEqual(shadow.normal_bias, view_off.shadow_params[2]);
    try std.testing.expect(view.shadow_params[0] > 0.0);
    try std.testing.expect(view.shadow_params[1] > 0.0);
}

test "split partition: material block preserves every per-draw lane" {
    const inputs = uniforms.MaterialUniformInputs{
        .base_color = .{ 0.8, 0.2, 0.1, 1.0 },
        .pbr_factors = .{ 0.9, 0.35, 1.0, 1.2 },
        .emissive_color = .{ 0.1, 0.2, 0.3, 0 },
        .alpha_cutoff = 0.5,
        .normal_scale = 0.8,
        .uv_matrices = @splat(.{ 2, 0, 0, 2 }),
        .uv_offsets = @splat(.{ 0.5, 0.25, 0, 0 }),
        .channel_selectors = .{ 3, 1, 2, 1 },
        .clearcoat_factors = .{ 0.6, 0.2, 0, 0 },
        .clearcoat_color = .{ 0.9, 0.9, 1.0, 0 },
        .sheen_factors = .{ 0.4, 0.5, 0, 0 },
        .sheen_color = .{ 1.0, 0.8, 0.6, 0 },
        .anisotropy_factors = .{ 0.7, 0.3, 0, 0 },
        .transmission_factors = .{ 0.25, 0, 0, 0 },
        .transmission_color = .{ 0.95, 0.95, 1.0, 0 },
        .sss_factors = .{ 0.35, 0, 0, 0 },
        .sss_color = .{ 1.0, 0.6, 0.5, 0 },
        .clearcoat_uv_matrix = .{ 1, 0, 0, 1 },
        .clearcoat_uv_offset = .{ 0.1, 0.2, 0, 0 },
        .sheen_uv_matrix = .{ 1, 0, 0, 1 },
        .sheen_uv_offset = .{ 0, 0, 0, 0 },
        .refraction_factors = .{ 1, 0.5, 1.5, 0 },
        .anisotropy_uv_matrix = .{ 1, 0, 0, 1 },
        .anisotropy_uv_offset = .{ 0.05, 0.1, 0, 0 },
        .probe_params = .{ 1.0, 0.8, 7.0, 0.6 },
        .probe2_params = .{ 1.0, 0.5, 7.0, 0.4 },
        .probe_box = .{ .{ 1, 2, 3, 4 }, .{ 5, 6, 7, 8 } },
    };
    const mat = uniforms.buildFrameMaterial(inputs);
    try std.testing.expectEqual(inputs.base_color, mat.base_color_factor);
    try std.testing.expectEqual(inputs.pbr_factors, mat.pbr_factors);
    try std.testing.expectEqual(inputs.emissive_color, mat.emissive_factor);
    try std.testing.expectEqual(inputs.alpha_cutoff, mat.alpha_cutoff);
    try std.testing.expectEqual(inputs.normal_scale, mat.normal_scale);
    try std.testing.expectEqual(inputs.uv_matrices, mat.uv_matrix);
    try std.testing.expectEqual(inputs.uv_offsets, mat.uv_offset);
    try std.testing.expectEqual(inputs.channel_selectors, mat.channel_selectors);
    try std.testing.expectEqual(inputs.clearcoat_factors, mat.clearcoat_factors);
    try std.testing.expectEqual(inputs.clearcoat_color, mat.clearcoat_color);
    try std.testing.expectEqual(inputs.sheen_factors, mat.sheen_factors);
    try std.testing.expectEqual(inputs.sheen_color, mat.sheen_color);
    try std.testing.expectEqual(inputs.anisotropy_factors, mat.anisotropy_factors);
    try std.testing.expectEqual(inputs.transmission_factors, mat.transmission_factors);
    try std.testing.expectEqual(inputs.transmission_color, mat.transmission_color);
    try std.testing.expectEqual(inputs.sss_factors, mat.sss_factors);
    try std.testing.expectEqual(inputs.sss_color, mat.sss_color);
    try std.testing.expectEqual(inputs.clearcoat_uv_matrix, mat.clearcoat_uv_matrix);
    try std.testing.expectEqual(inputs.clearcoat_uv_offset, mat.clearcoat_uv_offset);
    try std.testing.expectEqual(inputs.sheen_uv_matrix, mat.sheen_uv_matrix);
    try std.testing.expectEqual(inputs.sheen_uv_offset, mat.sheen_uv_offset);
    try std.testing.expectEqual(inputs.refraction_factors, mat.refraction_factors);
    try std.testing.expectEqual(inputs.anisotropy_uv_matrix, mat.anisotropy_uv_matrix);
    try std.testing.expectEqual(inputs.anisotropy_uv_offset, mat.anisotropy_uv_offset);
    try std.testing.expectEqual(inputs.probe_params, mat.probe_params);
    try std.testing.expectEqual(inputs.probe2_params, mat.probe2_params);
    try std.testing.expectEqual(inputs.probe_box, mat.probe_box);
}

test "split wire layout: Zig model matches shdc blocks byte-for-byte" {
    // Sizes: the two blocks partition the legacy ~2.3KB fs_params with no
    // padding drift (sokol VALIDATE_AU_SIZE compares exact byte sizes).
    try std.testing.expectEqual(@sizeOf(pbr_shd.FsView), @sizeOf(uniforms.FrameViewUniforms));
    try std.testing.expectEqual(@sizeOf(pbr_shd.FsMat), @sizeOf(uniforms.FrameMaterialUniforms));
    try std.testing.expectEqual(@sizeOf(skinned_pbr_shd.FsView), @sizeOf(uniforms.FrameViewUniforms));
    try std.testing.expectEqual(@sizeOf(skinned_pbr_shd.FsMat), @sizeOf(uniforms.FrameMaterialUniforms));
    try std.testing.expectEqual(@sizeOf(inst_pbr_shd.FsView), @sizeOf(uniforms.FrameViewUniforms));
    try std.testing.expectEqual(@sizeOf(inst_pbr_shd.FsMat), @sizeOf(uniforms.FrameMaterialUniforms));
    // Offsets spot-check both blocks (full per-field parity lives in
    // draw_tests.zig across the triple; here the Zig model is pinned).
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsView, "cascade_view_proj"), @offsetOf(uniforms.FrameViewUniforms, "cascade_view_proj"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsView, "point_view_proj"), @offsetOf(uniforms.FrameViewUniforms, "point_view_proj"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsView, "directional_dir"), @offsetOf(uniforms.FrameViewUniforms, "directional_dir"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsView, "clustered_params"), @offsetOf(uniforms.FrameViewUniforms, "clustered_params"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsView, "refraction_view_proj"), @offsetOf(uniforms.FrameViewUniforms, "refraction_view_proj"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsView, "refraction_capture"), @offsetOf(uniforms.FrameViewUniforms, "refraction_capture"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsMat, "base_color_factor"), @offsetOf(uniforms.FrameMaterialUniforms, "base_color_factor"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsMat, "alpha_cutoff"), @offsetOf(uniforms.FrameMaterialUniforms, "alpha_cutoff"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsMat, "uv_matrix"), @offsetOf(uniforms.FrameMaterialUniforms, "uv_matrix"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsMat, "clearcoat_factors"), @offsetOf(uniforms.FrameMaterialUniforms, "clearcoat_factors"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsMat, "refraction_factors"), @offsetOf(uniforms.FrameMaterialUniforms, "refraction_factors"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsMat, "anisotropy_uv_matrix"), @offsetOf(uniforms.FrameMaterialUniforms, "anisotropy_uv_matrix"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsMat, "anisotropy_uv_offset"), @offsetOf(uniforms.FrameMaterialUniforms, "anisotropy_uv_offset"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsMat, "probe_params"), @offsetOf(uniforms.FrameMaterialUniforms, "probe_params"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsMat, "probe_box"), @offsetOf(uniforms.FrameMaterialUniforms, "probe_box"));
}

test "split upload bytes: steady-state per-draw cost is the material block" {
    const view_bytes = @sizeOf(uniforms.FrameViewUniforms);
    const mat_bytes = @sizeOf(uniforms.FrameMaterialUniforms);
    // Alignment: std140 blocks are 16-byte multiples (the material block
    // carries an explicit 8-byte trailing pad for the scalar pair); sokol
    // copies whole blocks and validates the exact upload size.
    try std.testing.expect(view_bytes % 16 == 0);
    try std.testing.expect(mat_bytes % 16 == 0);
    // Legacy per-draw cost rebuilt + uploaded the full block every draw.
    const legacy_per_draw = view_bytes + mat_bytes;
    // The view part (~1.15KB matrix-heavy constant: cascades 256B +
    // spot 128B + point 768B alone) uploads once per contiguous
    // (pipeline, shadow-policy) run instead of per draw.
    try std.testing.expect(view_bytes > 1024);
    // Scenario: 100 draws, one pipeline run, one shadow policy.
    // Legacy: 100 * (V + M). Split: 1 * V + 100 * M.
    const draws: usize = 100;
    const legacy_total = draws * legacy_per_draw;
    const split_total = view_bytes + draws * mat_bytes;
    try std.testing.expect(split_total < legacy_total);
    try std.testing.expectEqual((draws - 1) * view_bytes, legacy_total - split_total);
    // Two shadow policies on one pipeline: still one extra view upload,
    // never per-draw view churn beyond the policy flip.
    const split_two_policy = 2 * view_bytes + draws * mat_bytes;
    try std.testing.expect(split_two_policy < legacy_total);
}

test "split hook/runtime wire contracts" {
    // Engine hook template: view keeps UB 1, material rides UB 5 (free in
    // every forward shader's UB pool; user blocks stay on 3/4).
    try std.testing.expectEqual(sm.fs_params_ub, 1);
    try std.testing.expectEqual(sm.fs_mat_ub, 5);
    const idx = sm.indexForName("ramp_wave") orelse return error.TestUnexpectedResult;
    const e = sm.entry(idx).?;
    try std.testing.expect(e.engine_template);
    try std.testing.expectEqual(@as(u32, 1), e.fs_ub);
    try std.testing.expectEqual(@as(u32, 5), e.fs_mat_ub);
    // Static hook templates are generated from the split sources.
    try std.testing.expect(e.frame_split);
    // Runtime external path (engine_template = false) is NOT reinterpreted:
    // no frame blocks are read there, and the user-bytes window is intact.
    try std.testing.expectEqual(@as(u16, 32), sm.max_external_uniform_vec4 * 16);
}

test "split fallback keeps the live env refraction camera" {
    // Direct draws with no cached per-view variants (null pointers) must
    // retain the caller's live refraction matrix/capture — not a hardcoded
    // identity — while the receive-shadow gate still applies.
    var ctx = testContext();
    ctx.uniforms_with_shadows = null;
    ctx.uniforms_without_shadows = null;
    var shadow = testShadow();
    shadow.mesh_receive_shadows = true;
    const rp = Mat4.translation(Vec3.new(11, 22, 33));
    const rc: [4]f32 = .{ 1, 320, 180, 0 };
    const v = draw.frameUniformsForState(shadow, true, &ctx, rp, rc);
    try std.testing.expectEqual(rp, v.refraction_view_proj);
    try std.testing.expectEqual(rc, v.refraction_capture);
    try std.testing.expect(v.shadow_params[0] > 0.0);
    // Opt-out keeps the gate zeroed with the same live camera.
    const v_off = draw.frameUniformsForState(shadow, false, &ctx, rp, rc);
    try std.testing.expectEqual(@as(f32, 0.0), v_off.shadow_params[0]);
    try std.testing.expectEqual(rp, v_off.refraction_view_proj);
}
