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
