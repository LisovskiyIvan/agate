const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color3 = math.Color3;
const Material = @import("../material.zig").Material;
const pcss = @import("shadow_pcss.zig");

// Per-frame constants shared by the regular and instanced draw helpers.
pub const FrameContext = struct {
    view_proj: Mat4,
    eye: Vec3,
    sun_dir: Vec3,
    sun_color: Color3,
    sun_intensity: f32,
    // Up to 4 directional suns (slot 0 = primary shadow caster, slots 1..3
    // shadowless fills; unused/disabled slots zeroed). Mirrors
    // LightRig.FramePack; kept as explicit arrays (not folded into the sun
    // lanes) so the legacy sun fields stay bit-identical single-light inputs.
    directional_dir: [4][4]f32,
    directional_color_int: [4][4]f32,
    cascades: [4]Mat4,
    light_counts: [4]f32,
    point_pos_range: [4][4]f32,
    point_color_int: [4][4]f32,
    spot_pos_range: [2][4]f32,
    spot_dir_inner: [2][4]f32,
    spot_color_outer: [2][4]f32,
    spot_intensity: [2][4]f32,
    spot_view_proj: [2]Mat4,
    spot_shadow_params: [2][4]f32,
    point_view_proj: [12]Mat4,
    point_shadow_params: [4][4]f32,
    uniforms_with_shadows: ?*const FrameUniforms = null,
    uniforms_without_shadows: ?*const FrameUniforms = null,
};

// Fragment uniforms shared by the standard, PBR and instanced shaders:
// identical names, types and values in all three FsParams structs.
// Material-specific factors (diffuse_color / base_color_factor /
// pbr_factors / emissive_factor) stay with the caller.
pub const FrameUniforms = struct {
    eye_pos: [4]f32,
    light_dir: [4]f32,
    light_color: [4]f32,
    ambient_color: [4]f32,
    shadow_params: [4]f32,
    shadow_splits: [4]f32,
    cascade_view_proj: [4]Mat4,
    cascade_debug: [4]f32,
    light_counts: [4]f32,
    point_pos_range: [4][4]f32,
    point_color_int: [4][4]f32,
    spot_pos_range: [2][4]f32,
    spot_dir_inner: [2][4]f32,
    spot_color_outer: [2][4]f32,
    spot_intensity: [2][4]f32,
    spot_view_proj: [2]Mat4,
    spot_shadow_params: [2][4]f32,
    point_view_proj: [12]Mat4,
    point_shadow_params: [4][4]f32,
    // APPENDED LAST (multi-directional): 4 suns, slot 0 mirrors the primary
    // (light_dir/light_color above, the only shadow caster), slots 1..3 are
    // shadowless fills. A single sun leaves slots 1..3 zeroed and the
    // shader fill loop adds nothing.
    directional_dir: [4][4]f32,
    directional_color_int: [4][4]f32,
};

// Scene-derived inputs for the shared fragment uniforms. Keeping them in
// one struct makes the free function pure (no Scene import / no cycle)
// while the values stay bit-identical to the legacy per-shader literals
// (the PCSS lanes below extend previously constant-zero lanes; every legacy
// lane keeps its exact legacy value).
pub const ShadowState = struct {
    ground_color: Color3,
    enable_shadows: bool,
    mesh_receive_shadows: bool,
    bias: f32,
    intensity: f32,
    normal_bias: f32,
    softness: f32,
    debug_cascades: bool,
    splits: [4]f32,
    // PCSS (percentage-closer soft shadows) inputs. Only `pcss_enabled`
    // gates the shader branch; the rest ride free uniform lanes (see
    // buildFrameUniforms) and are ignored while disabled.
    pcss_enabled: bool = false,
    pcss_light_size: f32 = pcss.default_light_size,
    pcss_blocker_radius: f32 = pcss.default_blocker_radius,
    pcss_min_penumbra: f32 = pcss.default_min_penumbra,
    pcss_max_penumbra: f32 = pcss.default_max_penumbra,
};

// Packs the shared fragment uniforms once per draw. Legacy lanes are
// bit-identical to the legacy per-shader literals. PCSS rides free lanes
// with no fs_params layout change (no draw-path churn):
//   cascade_debug.yzw = (pcss_enabled, pcss_light_size, pcss_blocker_radius)
//   light_counts.zw   = (pcss_min_penumbra, pcss_max_penumbra)
pub fn buildFrameUniforms(shadow: ShadowState, ctx: *const FrameContext) FrameUniforms {
    return .{
        .eye_pos = .{ ctx.eye.x, ctx.eye.y, ctx.eye.z, 4.0 },
        .light_dir = .{ ctx.sun_dir.x, ctx.sun_dir.y, ctx.sun_dir.z, 2048.0 },
        .light_color = .{ ctx.sun_color.r, ctx.sun_color.g, ctx.sun_color.b, ctx.sun_intensity },
        .ambient_color = .{ shadow.ground_color.r, shadow.ground_color.g, shadow.ground_color.b, 1.0 },
        .shadow_params = .{
            if (shadow.enable_shadows and shadow.mesh_receive_shadows) shadow.bias else 0.0,
            if (shadow.enable_shadows and shadow.mesh_receive_shadows) shadow.intensity else 0.0,
            shadow.normal_bias,
            shadow.softness,
        },
        .shadow_splits = shadow.splits,
        .cascade_view_proj = ctx.cascades,
        .cascade_debug = .{
            if (shadow.debug_cascades) 1.0 else 0.0,
            if (shadow.pcss_enabled) 1.0 else 0.0,
            shadow.pcss_light_size,
            shadow.pcss_blocker_radius,
        },
        .light_counts = .{
            ctx.light_counts[0],
            ctx.light_counts[1],
            shadow.pcss_min_penumbra,
            shadow.pcss_max_penumbra,
        },
        .point_pos_range = ctx.point_pos_range,
        .point_color_int = ctx.point_color_int,
        .spot_pos_range = ctx.spot_pos_range,
        .spot_dir_inner = ctx.spot_dir_inner,
        .spot_color_outer = ctx.spot_color_outer,
        .spot_intensity = ctx.spot_intensity,
        .spot_view_proj = ctx.spot_view_proj,
        .spot_shadow_params = ctx.spot_shadow_params,
        .point_view_proj = ctx.point_view_proj,
        .point_shadow_params = ctx.point_shadow_params,
        .directional_dir = ctx.directional_dir,
        .directional_color_int = ctx.directional_color_int,
    };
}

// Value for the per-shader `alpha_cutoff` fragment uniform (appended last
// to every fs_params block). Only cutout materials upload a live cutoff;
// opaque/blend upload 0.0 so the shader alpha test never fires and their
// rendering is bit-identical to before. Meshes without a material render
// opaque legacy, so they upload 0.0 as well. Pure function (no GPU calls).
pub fn alphaCutoffFor(mat: ?Material) f32 {
    const m = mat orelse return 0.0;
    if (!m.isCutout()) return 0.0;
    return m.alphaCutoff();
}

test "buildFrameUniforms passes directional lanes through, legacy lanes intact" {
    const std = @import("std");
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
    };
    const f = buildFrameUniforms(shadow, &ctx);
    // Legacy lanes keep their exact legacy values...
    try std.testing.expectEqual([4]f32{ 0.5, 1.0, 0.3, 2048.0 }, f.light_dir);
    try std.testing.expectEqual([4]f32{ 1.0, 0.9, 0.8, 2.0 }, f.light_color);
    // ...and the directional lanes ride through verbatim.
    try std.testing.expectEqual(ctx.directional_dir, f.directional_dir);
    try std.testing.expectEqual(ctx.directional_color_int, f.directional_color_int);
}

test "alphaCutoffFor gates the cutoff on cutout mode" {
    const std = @import("std");
    const material = @import("../material.zig");

    // No material: opaque legacy, test disabled.
    try std.testing.expectEqual(@as(f32, 0.0), alphaCutoffFor(null));

    var std_mat = material.StandardMaterial.init("m");
    var pbr_mat = material.PBRMaterial.init("p");

    // Opaque and blend upload 0.0 even with a customized cutoff stored.
    std_mat.alpha_cutoff = 0.3;
    pbr_mat.alpha_cutoff = 0.7;
    for ([_]material.AlphaMode{ .@"opaque", .blend }) |mode| {
        std_mat.alpha_mode = mode;
        pbr_mat.alpha_mode = mode;
        try std.testing.expectEqual(@as(f32, 0.0), alphaCutoffFor(.{ .standard = &std_mat }));
        try std.testing.expectEqual(@as(f32, 0.0), alphaCutoffFor(.{ .pbr = &pbr_mat }));
    }

    // Cutout uploads the material value (default 0.5).
    std_mat.alpha_mode = .cutout;
    pbr_mat.alpha_mode = .cutout;
    try std.testing.expectEqual(@as(f32, 0.3), alphaCutoffFor(.{ .standard = &std_mat }));
    try std.testing.expectEqual(@as(f32, 0.7), alphaCutoffFor(.{ .pbr = &pbr_mat }));

    var std_default = material.StandardMaterial.init("d");
    std_default.alpha_mode = .cutout;
    try std.testing.expectEqual(@as(f32, 0.5), alphaCutoffFor(.{ .standard = &std_default }));
}
