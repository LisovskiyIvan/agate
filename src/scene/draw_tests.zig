const std = @import("std");
const sokol = @import("sokol");
const pbr_shd = @import("pbr_shader");
const skinned_pbr_shd = @import("skinned_pbr_shader");
const inst_pbr_shd = @import("instanced_pbr_shader");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;

const Texture = @import("../texture.zig").Texture;
const CubeTexture = @import("../texture.zig").CubeTexture;
const material_mod = @import("../material.zig");
const PBRMaterial = material_mod.PBRMaterial;
const Material = material_mod.Material;
const probe_layer = @import("probe_layer.zig");
const draw = @import("draw.zig");
const Environment = draw.Environment;
const probeForDraw = draw.probeForDraw;
const resolveCoat = draw.resolveCoat;
const pbrUvMatrices = draw.pbrUvMatrices;
const pbrUvOffsets = draw.pbrUvOffsets;
const pbrChannelSelectors = draw.pbrChannelSelectors;
const instancedDrawFlags = draw.instancedDrawFlags;

test "forward shader FsView/FsMat carry the split view/material uniforms" {
    comptime {
        for ([_]type{ pbr_shd.FsView, skinned_pbr_shd.FsView, inst_pbr_shd.FsView }) |P| {
            if (!@hasField(P, "eye_pos")) @compileError("FsView missing eye_pos");
            if (!@hasField(P, "light_dir")) @compileError("FsView missing light_dir");
            if (!@hasField(P, "light_color")) @compileError("FsView missing light_color");
            if (!@hasField(P, "ambient_color")) @compileError("FsView missing ambient_color");
            if (!@hasField(P, "shadow_params")) @compileError("FsView missing shadow_params");
            if (!@hasField(P, "shadow_splits")) @compileError("FsView missing shadow_splits");
            if (!@hasField(P, "cascade_view_proj")) @compileError("FsView missing cascade_view_proj");
            if (!@hasField(P, "cascade_debug")) @compileError("FsView missing cascade_debug");
            if (!@hasField(P, "light_counts")) @compileError("FsView missing light_counts");
            if (!@hasField(P, "point_pos_range")) @compileError("FsView missing point_pos_range");
            if (!@hasField(P, "point_color_int")) @compileError("FsView missing point_color_int");
            if (!@hasField(P, "spot_pos_range")) @compileError("FsView missing spot_pos_range");
            if (!@hasField(P, "spot_view_proj")) @compileError("FsView missing spot_view_proj");
            if (!@hasField(P, "point_view_proj")) @compileError("FsView missing point_view_proj");
            if (!@hasField(P, "point_shadow_params")) @compileError("FsView missing point_shadow_params");
            if (!@hasField(P, "directional_dir")) @compileError("FsView missing directional_dir");
            if (!@hasField(P, "directional_color_int")) @compileError("FsView missing directional_color_int");
            if (!@hasField(P, "area_center_int")) @compileError("FsView missing area_center_int");
            if (!@hasField(P, "area_right")) @compileError("FsView missing area_right");
            if (!@hasField(P, "area_up")) @compileError("FsView missing area_up");
            if (!@hasField(P, "area_color")) @compileError("FsView missing area_color");
            if (!@hasField(P, "clustered_params")) @compileError("FsView missing clustered_params");
            if (!@hasField(P, "clustered_viewport")) @compileError("FsView missing clustered_viewport");
            if (!@hasField(P, "hemi_dir_intensity")) @compileError("FsView missing hemi_dir_intensity");
            if (!@hasField(P, "hemi_diffuse")) @compileError("FsView missing hemi_diffuse");
            if (!@hasField(P, "refraction_view_proj")) @compileError("FsView missing refraction_view_proj");
            if (!@hasField(P, "refraction_capture")) @compileError("FsView missing refraction_capture");
            if (@hasField(P, "base_color_factor")) @compileError("FsView must not carry material lanes");
            if (@hasField(P, "probe_params")) @compileError("FsView must not carry probe lanes");
        }
        for ([_]type{ pbr_shd.FsMat, skinned_pbr_shd.FsMat, inst_pbr_shd.FsMat }) |P| {
            if (!@hasField(P, "base_color_factor")) @compileError("FsMat missing base_color_factor");
            if (!@hasField(P, "pbr_factors")) @compileError("FsMat missing pbr_factors");
            if (!@hasField(P, "emissive_factor")) @compileError("FsMat missing emissive_factor");
            if (!@hasField(P, "alpha_cutoff")) @compileError("FsMat missing alpha_cutoff");
            if (!@hasField(P, "normal_scale")) @compileError("FsMat missing normal_scale");
            if (!@hasField(P, "uv_matrix")) @compileError("FsMat missing uv_matrix");
            if (!@hasField(P, "uv_offset")) @compileError("FsMat missing uv_offset");
            if (!@hasField(P, "channel_selectors")) @compileError("FsMat missing channel_selectors");
            if (!@hasField(P, "clearcoat_factors")) @compileError("FsMat missing clearcoat_factors");
            if (!@hasField(P, "clearcoat_color")) @compileError("FsMat missing clearcoat_color");
            if (!@hasField(P, "sheen_factors")) @compileError("FsMat missing sheen_factors");
            if (!@hasField(P, "sheen_color")) @compileError("FsMat missing sheen_color");
            if (!@hasField(P, "anisotropy_factors")) @compileError("FsMat missing anisotropy_factors");
            if (!@hasField(P, "transmission_factors")) @compileError("FsMat missing transmission_factors");
            if (!@hasField(P, "transmission_color")) @compileError("FsMat missing transmission_color");
            if (!@hasField(P, "sss_factors")) @compileError("FsMat missing sss_factors");
            if (!@hasField(P, "sss_color")) @compileError("FsMat missing sss_color");
            if (!@hasField(P, "refraction_factors")) @compileError("FsMat missing refraction_factors");
            if (!@hasField(P, "probe_params")) @compileError("FsMat missing probe_params");
            if (!@hasField(P, "probe2_params")) @compileError("FsMat missing probe2_params");
            if (!@hasField(P, "probe_box")) @compileError("FsMat missing probe_box");
            if (@hasField(P, "output_params")) @compileError("FsMat still carries output_params");
            if (@hasField(P, "light_dir")) @compileError("FsMat must not carry view lanes");
            if (@hasField(P, "cascade_view_proj")) @compileError("FsMat must not carry view lanes");
            if (@hasField(P, "refraction_view_proj")) @compileError("FsMat must not carry the refraction camera");
        }
        // Wire slots: view keeps the legacy fs slot per family (pbr/inst
        // 1, skinned 2); material rides the free UB 5 everywhere.
        if (pbr_shd.UB_fs_view != 1) @compileError("pbr view slot moved");
        if (inst_pbr_shd.UB_fs_view != 1) @compileError("instanced view slot moved");
        if (skinned_pbr_shd.UB_fs_view != 2) @compileError("skinned view slot moved");
        if (pbr_shd.UB_fs_mat != 5) @compileError("pbr mat slot moved");
        if (skinned_pbr_shd.UB_fs_mat != 5) @compileError("skinned mat slot moved");
        if (inst_pbr_shd.UB_fs_mat != 5) @compileError("instanced mat slot moved");
        if (@hasDecl(pbr_shd, "UB_fs_params")) @compileError("stale UB_fs_params decl");
        for ([_]type{ pbr_shd, skinned_pbr_shd, inst_pbr_shd }) |M| {
            if (!@hasDecl(M, "VIEW_ssbo_cluster_lights")) @compileError("shader module missing VIEW_ssbo_cluster_lights");
            if (!@hasDecl(M, "VIEW_ssbo_cluster_tiles")) @compileError("shader module missing VIEW_ssbo_cluster_tiles");
            if (!@hasDecl(M, "VIEW_ssbo_cluster_indices")) @compileError("shader module missing VIEW_ssbo_cluster_indices");
            if (!@hasDecl(M, "VIEW_probe_tex")) @compileError("shader module missing VIEW_probe_tex");
            if (!@hasDecl(M, "VIEW_probe2_tex")) @compileError("shader module missing VIEW_probe2_tex");
        }
        if (pbr_shd.VIEW_ssbo_cluster_lights != 12) @compileError("clustered light slot moved");
        if (pbr_shd.VIEW_ssbo_cluster_tiles != 13) @compileError("clustered tile slot moved");
        if (inst_pbr_shd.VIEW_ssbo_cluster_indices != 14) @compileError("clustered index slot moved");
        if (pbr_shd.VIEW_probe2_tex != 19) @compileError("probe2_tex slot moved");
    }
}

test "pbr FsView/FsMat layouts stay identical across regular/skinned/instanced" {
    try std.testing.expectEqual(@sizeOf(pbr_shd.FsView), @sizeOf(skinned_pbr_shd.FsView));
    try std.testing.expectEqual(@sizeOf(pbr_shd.FsView), @sizeOf(inst_pbr_shd.FsView));
    try std.testing.expectEqual(@sizeOf(pbr_shd.FsMat), @sizeOf(skinned_pbr_shd.FsMat));
    try std.testing.expectEqual(@sizeOf(pbr_shd.FsMat), @sizeOf(inst_pbr_shd.FsMat));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsView, "cascade_view_proj"), @offsetOf(skinned_pbr_shd.FsView, "cascade_view_proj"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsView, "cascade_view_proj"), @offsetOf(inst_pbr_shd.FsView, "cascade_view_proj"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsView, "point_view_proj"), @offsetOf(skinned_pbr_shd.FsView, "point_view_proj"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsView, "directional_dir"), @offsetOf(inst_pbr_shd.FsView, "directional_dir"));
    try std.testing.expect(@offsetOf(pbr_shd.FsView, "area_center_int") > @offsetOf(pbr_shd.FsView, "directional_dir"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsView, "area_center_int"), @offsetOf(skinned_pbr_shd.FsView, "area_center_int"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsView, "area_color"), @offsetOf(inst_pbr_shd.FsView, "area_color"));
    try std.testing.expect(@offsetOf(pbr_shd.FsView, "clustered_params") > @offsetOf(pbr_shd.FsView, "area_color"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsView, "clustered_params"), @offsetOf(skinned_pbr_shd.FsView, "clustered_params"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsView, "refraction_view_proj"), @offsetOf(skinned_pbr_shd.FsView, "refraction_view_proj"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsView, "refraction_capture"), @offsetOf(inst_pbr_shd.FsView, "refraction_capture"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsMat, "clearcoat_factors"), @offsetOf(skinned_pbr_shd.FsMat, "clearcoat_factors"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsMat, "sheen_color"), @offsetOf(inst_pbr_shd.FsMat, "sheen_color"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsMat, "probe_params"), @offsetOf(skinned_pbr_shd.FsMat, "probe_params"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsMat, "probe_params"), @offsetOf(inst_pbr_shd.FsMat, "probe_params"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsMat, "probe2_params"), @offsetOf(skinned_pbr_shd.FsMat, "probe2_params"));
    try std.testing.expect(@offsetOf(pbr_shd.FsMat, "probe_box") > @offsetOf(pbr_shd.FsMat, "probe2_params"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsMat, "probe_box"), @offsetOf(inst_pbr_shd.FsMat, "probe_box"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsMat, "anisotropy_factors"), @offsetOf(skinned_pbr_shd.FsMat, "anisotropy_factors"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsMat, "transmission_color"), @offsetOf(inst_pbr_shd.FsMat, "transmission_color"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsMat, "anisotropy_uv_matrix"), @offsetOf(skinned_pbr_shd.FsMat, "anisotropy_uv_matrix"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsMat, "anisotropy_uv_matrix"), @offsetOf(inst_pbr_shd.FsMat, "anisotropy_uv_matrix"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsMat, "anisotropy_uv_offset"), @offsetOf(skinned_pbr_shd.FsMat, "anisotropy_uv_offset"));
    try std.testing.expectEqual(@offsetOf(pbr_shd.FsMat, "anisotropy_uv_offset"), @offsetOf(inst_pbr_shd.FsMat, "anisotropy_uv_offset"));
}

test "pbr layer mask texture slots are pinned across the triple" {
    comptime {
        for ([_]type{ pbr_shd, skinned_pbr_shd, inst_pbr_shd }) |M| {
            if (!@hasDecl(M, "VIEW_clearcoat_tex")) @compileError("shader module missing VIEW_clearcoat_tex");
            if (!@hasDecl(M, "VIEW_sheen_tex")) @compileError("shader module missing VIEW_sheen_tex");
            if (!@hasDecl(M, "VIEW_anisotropy_tex")) @compileError("shader module missing VIEW_anisotropy_tex");
            if (!@hasDecl(M, "VIEW_brdf_lut_tex")) @compileError("shader module missing VIEW_brdf_lut_tex");
            if (!@hasDecl(M, "SMP_brdf_lut_smp")) @compileError("shader module missing SMP_brdf_lut_smp");
        }
        if (pbr_shd.VIEW_clearcoat_tex != 15) @compileError("clearcoat texture slot moved");
        if (pbr_shd.VIEW_sheen_tex != 16) @compileError("sheen texture slot moved");
        if (pbr_shd.VIEW_brdf_lut_tex != 17) @compileError("brdf lut texture slot moved");
        if (pbr_shd.SMP_brdf_lut_smp != 7) @compileError("brdf lut sampler slot moved");
        if (pbr_shd.VIEW_anisotropy_tex != 22) @compileError("anisotropy texture slot moved");
        if (skinned_pbr_shd.VIEW_clearcoat_tex != 15) @compileError("clearcoat texture slot moved");
        if (skinned_pbr_shd.VIEW_brdf_lut_tex != 17) @compileError("brdf lut texture slot moved");
        if (skinned_pbr_shd.SMP_brdf_lut_smp != 7) @compileError("brdf lut sampler slot moved");
        if (skinned_pbr_shd.VIEW_anisotropy_tex != 22) @compileError("anisotropy texture slot moved");
        if (inst_pbr_shd.VIEW_sheen_tex != 16) @compileError("sheen texture slot moved");
        if (inst_pbr_shd.VIEW_brdf_lut_tex != 17) @compileError("brdf lut texture slot moved");
        if (inst_pbr_shd.SMP_brdf_lut_smp != 7) @compileError("brdf lut sampler slot moved");
        if (inst_pbr_shd.VIEW_anisotropy_tex != 22) @compileError("anisotropy texture slot moved");
    }
}

test "probeForDraw resolves the winning probe or the legacy fallback" {
    const empty_env = Environment{
        .pipelines = undefined,
        .stats = undefined,
        .default_white = std.mem.zeroes(Texture),
        .default_normal = std.mem.zeroes(Texture),
        .default_cube = std.mem.zeroes(CubeTexture),
        .sky_texture = null,
        .ibl_intensity = 1.0,
        .shadow_pass = undefined,
        .shadow_uniforms = undefined,
        .probes = &.{},
    };
    const off = probeForDraw(&empty_env, Mat4.identity);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, off.params);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, off.params2);
    try std.testing.expectEqual(@as(u32, 0), off.view.id);

    var entries = [_]probe_layer.ProbeFrameEntry{
        .{
            .position = Vec3.zero,
            .radius = 5.0,
            .enabled = true,
            .captured = true,
            .intensity = 0.5,
            .max_probe_lod = 7.0,
            .view = .{ .id = 31 },
            .sampler = .{ .id = 32 },
        },
    };
    var covered_env = empty_env;
    covered_env.probes = &entries;
    const on = probeForDraw(&covered_env, Mat4.identity);
    try std.testing.expectEqual([4]f32{ 1.0, 0.5, 7.0, 1.0 }, on.params);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, on.params2);
    try std.testing.expectEqual(@as(u32, 31), on.view.id);
    try std.testing.expectEqual(@as(u32, 32), on.sampler.id);

    var blend_entries = [_]probe_layer.ProbeFrameEntry{
        .{
            .position = Vec3.new(-2.0, 0, 0),
            .radius = 4.0,
            .enabled = true,
            .captured = true,
            .intensity = 1.0,
            .max_probe_lod = 7.0,
            .view = .{ .id = 41 },
            .sampler = .{ .id = 42 },
        },
        .{
            .position = Vec3.new(2.0, 0, 0),
            .radius = 4.0,
            .enabled = true,
            .captured = true,
            .intensity = 1.0,
            .max_probe_lod = 7.0,
            .view = .{ .id = 51 },
            .sampler = .{ .id = 52 },
        },
    };
    var blend_env = empty_env;
    blend_env.probes = &blend_entries;
    const blended = probeForDraw(&blend_env, Mat4.identity);
    try std.testing.expectEqual(@as(f32, 1.0), blended.params[0]);
    try std.testing.expectEqual(@as(f32, 1.0), blended.params2[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), blended.params[3], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), blended.params2[3], 0.01);
    try std.testing.expect(blended.view.id != 0);
    try std.testing.expect(blended.view2.id != 0);

    const far_model = Mat4.translation(Vec3.new(100.0, 0.0, 0.0));
    const far = probeForDraw(&covered_env, far_model);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, far.params);
}

test "resolveCoat falls back to neutral when the lobe is off" {
    const CoatParams = material_mod.CoatParams;
    const neutral = resolveCoat(&.{}, null);
    try std.testing.expectEqual(CoatParams.neutral, neutral);

    try std.testing.expectEqual(CoatParams.neutral, resolveCoat(&.{}, 7));

    const owned = [_]CoatParams{.{ .sheen_factors = .{ 0.5, 0.35, 0, 0 } }};
    const got = resolveCoat(&owned, 0);
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 0.35, 0, 0 }, &got.sheen_factors);
    try std.testing.expectEqual(CoatParams.neutral, resolveCoat(&owned, 1));
}

test "pbr uniform packing defaults are identity and glTF conventions" {
    const mats = pbrUvMatrices(null);
    const offs = pbrUvOffsets(null);
    for (0..5) |slot| {
        try std.testing.expectEqualSlices(f32, &.{ 1, 0, 0, 1 }, &mats[slot]);
        try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0 }, &offs[slot]);
    }
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 2, 0 }, &pbrChannelSelectors(null));

    var mat = PBRMaterial.init("m");
    mat.albedo_uv_transform = .{ .offset = .{ 0.5, 0 }, .scale = .{ 2, 2 } };
    mat.occlusion_channel = .a;
    const packed_mats = pbrUvMatrices(&mat);
    const packed_offs = pbrUvOffsets(&mat);
    try std.testing.expectEqualSlices(f32, &.{ 2, 0, 0, 2 }, &packed_mats[0]);
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 0, 0, 0 }, &packed_offs[0]);
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 0, 1 }, &packed_mats[1]);
    try std.testing.expectEqualSlices(f32, &.{ 3, 1, 2, 0 }, &pbrChannelSelectors(&mat));
}

test "instanced decal forces transparent + double-sided like regular decals" {
    var opaque_mat = PBRMaterial.init("opaque");
    const solid_mat: Material = .{ .pbr = &opaque_mat };
    const plain = instancedDrawFlags(null, false);
    try std.testing.expect(!plain.transparent and !plain.double_sided);
    const solid = instancedDrawFlags(solid_mat, false);
    try std.testing.expect(!solid.transparent and !solid.double_sided);
    const decal = instancedDrawFlags(solid_mat, true);
    try std.testing.expect(decal.transparent and decal.double_sided);
    opaque_mat.alpha_mode = .blend;
    const blended = instancedDrawFlags(solid_mat, false);
    try std.testing.expect(blended.transparent and !blended.double_sided);
}
