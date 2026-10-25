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
    uniforms_with_shadows: ?*const FrameViewUniforms = null,
    uniforms_without_shadows: ?*const FrameViewUniforms = null,
};

// Fragment uniforms shared by the standard, PBR and instanced shaders:
// identical names, types and values in the generated FsView/FsMat structs
// (split of the legacy fs_params block, see below).
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
// with no fs_view layout change (no draw-path churn):
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

// Value for the per-shader `alpha_cutoff` fragment uniform (a member of
// the fs_mat block). Only cutout materials upload a live cutoff;
// opaque/blend upload 0.0 so the shader alpha test never fires and their
// rendering is bit-identical to before. Meshes without a material render
// opaque legacy, so they upload 0.0 as well. Pure function (no GPU calls).
pub fn alphaCutoffFor(mat: ?Material) f32 {
    const m = mat orelse return 0.0;
    if (!m.isCutout()) return 0.0;
    return m.alphaCutoff();
}

// ---------------------------------------------------------------------------
// fs_params split (view-global vs per-material/object wire blocks).
//
// The legacy `FrameUniforms`/`fs_params` single block (~2.3KB rebuilt and
// uploaded per draw, ~1.15KB of which is the constant view part) is
// partitioned into two uniform blocks with zero value change:
//   FrameViewUniforms     -> fs_view (UB 1, ex-fs_params slot) — light/
//     camera/cascade+spot+point matrices, clustered descriptor, hemispheric
//     terms, refraction camera. Contains the receive-shadow gate
//     (shadow_params/shadow_splits/cascade_*), so two variants exist per
//     view (with/without shadows, selected from FrameContext).
//   FrameMaterialUniforms -> fs_mat (UB 5, free in every forward shader) —
//     material factors, UV/channel packing, coat/fabric + layer lobes,
//     refraction MATERIAL gate, per-draw probe state.
// Both are `extern struct` in GLSL member order so @sizeOf/@offsetOf
// track the shdc-generated FsView/FsMat structs (asserted in
// scene/uniform_split_tests.zig). Pure data; no GPU calls.
// ---------------------------------------------------------------------------

/// View-global wire block (fs_view). Field order mirrors
/// shaders/common/frame_view.glsl.
pub const FrameViewUniforms = extern struct {
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
    directional_dir: [4][4]f32,
    directional_color_int: [4][4]f32,
    area_center_int: [2][4]f32,
    area_right: [2][4]f32,
    area_up: [2][4]f32,
    area_color: [2][4]f32,
    clustered_params: [4]f32,
    clustered_viewport: [4]f32,
    hemi_dir_intensity: [4]f32,
    hemi_diffuse: [4]f32,
    refraction_view_proj: Mat4,
    refraction_capture: [4]f32,
};

/// Per-material/object wire block (fs_mat). Field order mirrors
/// shaders/common/frame_mat.glsl.
pub const FrameMaterialUniforms = extern struct {
    base_color_factor: [4]f32,
    pbr_factors: [4]f32,
    emissive_factor: [4]f32,
    alpha_cutoff: f32,
    normal_scale: f32,
    // std140 pad: the scalar pair is followed by a vec4 array; shdc aligns
    // the array to 16 bytes (explicit 8-byte gap, zeroed, never read).
    // sokol validates the exact upload size, so the gap rides here.
    _pad_uv: [2]f32 = .{ 0, 0 },
    uv_matrix: [5][4]f32,
    uv_offset: [5][4]f32,
    channel_selectors: [4]f32,
    clearcoat_factors: [4]f32,
    clearcoat_color: [4]f32,
    sheen_factors: [4]f32,
    sheen_color: [4]f32,
    anisotropy_factors: [4]f32,
    transmission_factors: [4]f32,
    transmission_color: [4]f32,
    sss_factors: [4]f32,
    sss_color: [4]f32,
    clearcoat_uv_matrix: [4]f32,
    clearcoat_uv_offset: [4]f32,
    sheen_uv_matrix: [4]f32,
    sheen_uv_offset: [4]f32,
    refraction_factors: [4]f32,
    probe_params: [4]f32,
    probe2_params: [4]f32,
    probe_box: [2][4]f32,
};

/// Packs the view-global block. Value-identical to the corresponding lanes
/// of buildFrameUniforms; refraction camera rides here (per-view env).
pub fn buildFrameView(
    shadow: ShadowState,
    ctx: *const FrameContext,
    refraction_view_proj: Mat4,
    refraction_capture: [4]f32,
) FrameViewUniforms {
    const f = buildFrameUniforms(shadow, ctx);
    return .{
        .eye_pos = f.eye_pos,
        .light_dir = f.light_dir,
        .light_color = f.light_color,
        .ambient_color = f.ambient_color,
        .shadow_params = f.shadow_params,
        .shadow_splits = f.shadow_splits,
        .cascade_view_proj = f.cascade_view_proj,
        .cascade_debug = f.cascade_debug,
        .light_counts = f.light_counts,
        .point_pos_range = f.point_pos_range,
        .point_color_int = f.point_color_int,
        .spot_pos_range = f.spot_pos_range,
        .spot_dir_inner = f.spot_dir_inner,
        .spot_color_outer = f.spot_color_outer,
        .spot_intensity = f.spot_intensity,
        .spot_view_proj = f.spot_view_proj,
        .spot_shadow_params = f.spot_shadow_params,
        .point_view_proj = f.point_view_proj,
        .point_shadow_params = f.point_shadow_params,
        .directional_dir = f.directional_dir,
        .directional_color_int = f.directional_color_int,
        .area_center_int = f.area_center_int,
        .area_right = f.area_right,
        .area_up = f.area_up,
        .area_color = f.area_color,
        .clustered_params = f.clustered_params,
        .clustered_viewport = f.clustered_viewport,
        .hemi_dir_intensity = f.hemi_dir_intensity,
        .hemi_diffuse = f.hemi_diffuse,
        .refraction_view_proj = refraction_view_proj,
        .refraction_capture = refraction_capture,
    };
}

/// Material/object inputs for the per-draw block. Probe lanes are explicit
/// (draw-resolved per object); everything else is record/coat-owned.
pub const MaterialUniformInputs = struct {
    base_color: [4]f32,
    pbr_factors: [4]f32,
    emissive_color: [4]f32,
    alpha_cutoff: f32,
    normal_scale: f32,
    uv_matrices: [5][4]f32,
    uv_offsets: [5][4]f32,
    channel_selectors: [4]f32,
    clearcoat_factors: [4]f32,
    clearcoat_color: [4]f32,
    sheen_factors: [4]f32,
    sheen_color: [4]f32,
    anisotropy_factors: [4]f32,
    transmission_factors: [4]f32,
    transmission_color: [4]f32,
    sss_factors: [4]f32,
    sss_color: [4]f32,
    clearcoat_uv_matrix: [4]f32,
    clearcoat_uv_offset: [4]f32,
    sheen_uv_matrix: [4]f32,
    sheen_uv_offset: [4]f32,
    refraction_factors: [4]f32,
    probe_params: [4]f32,
    probe2_params: [4]f32,
    probe_box: [2][4]f32,
};

/// Packs the per-draw block from material inputs (field-for-field copy,
/// no value change vs the legacy per-draw FsParams lanes).
pub fn buildFrameMaterial(in: MaterialUniformInputs) FrameMaterialUniforms {
    return buildFrameMaterialRef(&in);
}

/// Packs the per-draw block from a pointer to material inputs to avoid copying.
pub fn buildFrameMaterialRef(in: *const MaterialUniformInputs) FrameMaterialUniforms {
    return .{
        .base_color_factor = in.base_color,
        .pbr_factors = in.pbr_factors,
        .emissive_factor = in.emissive_color,
        .alpha_cutoff = in.alpha_cutoff,
        .normal_scale = in.normal_scale,
        .uv_matrix = in.uv_matrices,
        .uv_offset = in.uv_offsets,
        .channel_selectors = in.channel_selectors,
        .clearcoat_factors = in.clearcoat_factors,
        .clearcoat_color = in.clearcoat_color,
        .sheen_factors = in.sheen_factors,
        .sheen_color = in.sheen_color,
        .anisotropy_factors = in.anisotropy_factors,
        .transmission_factors = in.transmission_factors,
        .transmission_color = in.transmission_color,
        .sss_factors = in.sss_factors,
        .sss_color = in.sss_color,
        .clearcoat_uv_matrix = in.clearcoat_uv_matrix,
        .clearcoat_uv_offset = in.clearcoat_uv_offset,
        .sheen_uv_matrix = in.sheen_uv_matrix,
        .sheen_uv_offset = in.sheen_uv_offset,
        .refraction_factors = in.refraction_factors,
        .probe_params = in.probe_params,
        .probe2_params = in.probe2_params,
        .probe_box = in.probe_box,
    };
}

// ---------------------------------------------------------------------------
// Legacy pre-split fs_params wire block (2560 bytes).
//
// FROZEN public-API preservation for runtime `engine_template = true`
// registrations compiled against the d6c5710 templates: field order and
// padding are byte-identical to the old generated `FsParams` (single UB 1
// block). New engine code — static hook templates included — uses the
// split FrameViewUniforms/FrameMaterialUniforms pair instead. Only the
// runtime-true legacy draw path constructs this, uploaded whole per draw
// by design (no view caching); the -78.6% wire saving does not apply to
// it. Do not extend: any new runtime shader should set
// RuntimeDesc.frame_split and follow the split contract.
// ---------------------------------------------------------------------------

/// Legacy wire block (fs_params @ UB 1, 2560 bytes). Field order mirrors
/// the d6c5710 `src/shaders/pbr.glsl` fs_params block exactly.
pub const LegacyFsParams = extern struct {
    eye_pos: [4]f32,
    light_dir: [4]f32,
    light_color: [4]f32,
    ambient_color: [4]f32,
    base_color_factor: [4]f32,
    pbr_factors: [4]f32,
    emissive_factor: [4]f32,
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
    alpha_cutoff: f32,
    normal_scale: f32,
    // Same std140 array-alignment gap as the split material block.
    _pad_legacy_uv: [2]f32 = .{ 0, 0 },
    uv_matrix: [5][4]f32,
    uv_offset: [5][4]f32,
    channel_selectors: [4]f32,
    clearcoat_factors: [4]f32,
    clearcoat_color: [4]f32,
    sheen_factors: [4]f32,
    sheen_color: [4]f32,
    point_view_proj: [12]Mat4,
    point_shadow_params: [4][4]f32,
    directional_dir: [4][4]f32,
    directional_color_int: [4][4]f32,
    probe_params: [4]f32,
    area_center_int: [2][4]f32,
    area_right: [2][4]f32,
    area_up: [2][4]f32,
    area_color: [2][4]f32,
    clustered_params: [4]f32,
    clustered_viewport: [4]f32,
    anisotropy_factors: [4]f32,
    transmission_factors: [4]f32,
    transmission_color: [4]f32,
    sss_factors: [4]f32,
    sss_color: [4]f32,
    hemi_dir_intensity: [4]f32,
    hemi_diffuse: [4]f32,
    clearcoat_uv_matrix: [4]f32,
    clearcoat_uv_offset: [4]f32,
    sheen_uv_matrix: [4]f32,
    sheen_uv_offset: [4]f32,
    refraction_factors: [4]f32,
    refraction_view_proj: Mat4,
    refraction_capture: [4]f32,
    probe2_params: [4]f32,
    probe_box: [2][4]f32,
};

/// Packs the legacy block from the split view + material inputs. The
/// refraction camera is explicit (hook legacy path passes identity /
/// disabled, mirroring the pre-split hook upload). Pure (no GPU calls).
pub fn buildLegacyFsParams(
    view: *const FrameViewUniforms,
    in: MaterialUniformInputs,
    refraction_view_proj: Mat4,
    refraction_capture: [4]f32,
) LegacyFsParams {
    return .{
        .eye_pos = view.eye_pos,
        .light_dir = view.light_dir,
        .light_color = view.light_color,
        .ambient_color = view.ambient_color,
        .base_color_factor = in.base_color,
        .pbr_factors = in.pbr_factors,
        .emissive_factor = in.emissive_color,
        .shadow_params = view.shadow_params,
        .shadow_splits = view.shadow_splits,
        .cascade_view_proj = view.cascade_view_proj,
        .cascade_debug = view.cascade_debug,
        .light_counts = view.light_counts,
        .point_pos_range = view.point_pos_range,
        .point_color_int = view.point_color_int,
        .spot_pos_range = view.spot_pos_range,
        .spot_dir_inner = view.spot_dir_inner,
        .spot_color_outer = view.spot_color_outer,
        .spot_intensity = view.spot_intensity,
        .spot_view_proj = view.spot_view_proj,
        .spot_shadow_params = view.spot_shadow_params,
        .alpha_cutoff = in.alpha_cutoff,
        .normal_scale = in.normal_scale,
        .uv_matrix = in.uv_matrices,
        .uv_offset = in.uv_offsets,
        .channel_selectors = in.channel_selectors,
        .clearcoat_factors = in.clearcoat_factors,
        .clearcoat_color = in.clearcoat_color,
        .sheen_factors = in.sheen_factors,
        .sheen_color = in.sheen_color,
        .point_view_proj = view.point_view_proj,
        .point_shadow_params = view.point_shadow_params,
        .directional_dir = view.directional_dir,
        .directional_color_int = view.directional_color_int,
        .probe_params = in.probe_params,
        .area_center_int = view.area_center_int,
        .area_right = view.area_right,
        .area_up = view.area_up,
        .area_color = view.area_color,
        .clustered_params = view.clustered_params,
        .clustered_viewport = view.clustered_viewport,
        .anisotropy_factors = in.anisotropy_factors,
        .transmission_factors = in.transmission_factors,
        .transmission_color = in.transmission_color,
        .sss_factors = in.sss_factors,
        .sss_color = in.sss_color,
        .hemi_dir_intensity = view.hemi_dir_intensity,
        .hemi_diffuse = view.hemi_diffuse,
        .clearcoat_uv_matrix = in.clearcoat_uv_matrix,
        .clearcoat_uv_offset = in.clearcoat_uv_offset,
        .sheen_uv_matrix = in.sheen_uv_matrix,
        .sheen_uv_offset = in.sheen_uv_offset,
        .refraction_factors = in.refraction_factors,
        .refraction_view_proj = refraction_view_proj,
        .refraction_capture = refraction_capture,
        .probe2_params = in.probe2_params,
        .probe_box = in.probe_box,
    };
}
