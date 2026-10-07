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
    // Rect area lights (wave 26, v1): creation-order lanes mirroring
    // LightRig.FramePack (xyz + intensity, half-extent vectors, rgb).
    // Zeroed with zero lights, so the shader skip renders bit-identically.
    area_center_int: [2][4]f32,
    area_right: [2][4]f32,
    area_up: [2][4]f32,
    area_color: [2][4]f32,
    // Clustered forward point lights (wave 30, v1): tile-grid descriptor
    // mirroring the staged FramePack pool (the light data itself rides
    // storage buffers, never uniforms). Zeroed with an empty pool, so the
    // shader tile loop is gated off bit-identically.
    //   clustered_params:   x tiles_x, y tiles_y, z staged light count,
    //                       w gpu-live (1 = real storage views bound).
    //   clustered_viewport: x screen_w px, y screen_h px, z tile_size px, w 0.
    clustered_params: [4]f32,
    clustered_viewport: [4]f32,
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
    // APPENDED LAST (reflection probes, wave 25): per-draw probe state.
    // x: enabled (0/1), y: probe intensity, z: probe max lod, w: unused.
    // The shared per-view uniforms built here always carry zero (probe
    // off): the draw path copies this struct per draw and overwrites the
    // lane from the winning probe selection (or leaves it zeroed), so the
    // no-probe path uploads bit-identical values to before.
    probe_params: [4]f32,
    // APPENDED LAST (rect area lights, wave 26): creation-order lanes
    // mirroring LightRig.FramePack. xyz + intensity in area_center_int
    // (w = 0 when disabled/unused, so the shader skip costs nothing and
    // zero lights render bit-identically), half-extent vectors and rgb
    // alongside. Appended last so no existing offset shifts.
    area_center_int: [2][4]f32,
    area_right: [2][4]f32,
    area_up: [2][4]f32,
    area_color: [2][4]f32,
    // APPENDED LAST (clustered forward lights, wave 30): tile-grid
    // descriptor for the storage-buffer tile walk (light data itself is
    // never a uniform). Zeroed with an empty pool (and w = 0 until a live
    // GPU upload lands), so the shader gates the loop off bit-identically.
    // Appended last so no existing offset shifts.
    clustered_params: [4]f32,
    clustered_viewport: [4]f32,
    // APPENDED LAST (hemispheric light model): `ambient_color.rgb` holds the
    // hemispheric groundColor; these two lanes turn the old flat ambient into
    // Babylon's `HemisphericLight` irradiance
    // `mix(groundColor, diffuse * intensity, 0.5 + 0.5 * N·L)` (see
    // shaders/common/hemi.glsl). Appended last so no existing offset shifts.
    hemi_dir_intensity: [4]f32,
    hemi_diffuse: [4]f32,
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
    // Hemispheric light (Babylon model). Defaults mirror HemisphericLight's
    // own defaults (up, white, ground (0.2,0.2,0.2), intensity 1) so a
    // ShadowState built without a scene keeps the legacy ambient constant
    // readable and self-consistent. `ground_color` above is the hemispheric
    // groundColor — it is the same value the ambient lane always carried.
    hemi_dir: Vec3 = Vec3.up,
    hemi_diffuse: Color3 = Color3.white,
    hemi_intensity: f32 = 1.0,
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
        .probe_params = .{ 0.0, 0.0, 0.0, 0.0 },
        .area_center_int = ctx.area_center_int,
        .area_right = ctx.area_right,
        .area_up = ctx.area_up,
        .area_color = ctx.area_color,
        .clustered_params = ctx.clustered_params,
        .clustered_viewport = ctx.clustered_viewport,
        .hemi_dir_intensity = .{ shadow.hemi_dir.x, shadow.hemi_dir.y, shadow.hemi_dir.z, shadow.hemi_intensity },
        .hemi_diffuse = .{ shadow.hemi_diffuse.r, shadow.hemi_diffuse.g, shadow.hemi_diffuse.b, 0.0 },
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
    const std = @import("std");
    // Scene shaders always write linear radiance (see
    // shaders/common/linear_output.glsl): no gamma/HDR mode lanes ride the
    // frame uniforms, and the display transfer lives in postprocess only.
    try std.testing.expect(!@hasField(FrameContext, "output_params"));
    try std.testing.expect(!@hasField(FrameUniforms, "output_params"));
}

test "FrameUniforms appends area lanes last (existing offsets unmoved)" {
    const std = @import("std");
    // Every pre-area field keeps its offset relative to the struct start
    // regardless of the appended lanes: spot-check the tail neighbors.
    try std.testing.expect(@offsetOf(FrameUniforms, "area_center_int") > @offsetOf(FrameUniforms, "probe_params"));
    try std.testing.expect(@offsetOf(FrameUniforms, "area_right") > @offsetOf(FrameUniforms, "area_center_int"));
    try std.testing.expect(@offsetOf(FrameUniforms, "area_up") > @offsetOf(FrameUniforms, "area_right"));
    try std.testing.expect(@offsetOf(FrameUniforms, "area_color") > @offsetOf(FrameUniforms, "area_up"));
    try std.testing.expect(@offsetOf(FrameUniforms, "directional_dir") < @offsetOf(FrameUniforms, "probe_params"));
}

test "FrameUniforms appends clustered lanes after the area lanes (offsets unmoved)" {
    const std = @import("std");
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
    const std = @import("std");
    const material = @import("../material.zig");

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
