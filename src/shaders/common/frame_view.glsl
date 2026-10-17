// Shared forward view-global uniform block (wave: fs_params split).
//
// Included by the three PBR forward shaders (pbr/skinned_pbr/instanced_pbr)
// via `the frame_view include`. Single source so the layout
// stays bit-identical across the family by construction.
//
// Contract:
// - Wrapped by each shader with its legacy view slot (pbr/instanced:
//   binding 1, skinned: binding 2 — same slots as the legacy fs_params).
//   The SIZE changes so stale generated modules must be rebuilt — never
//   hand-edited.
// - Members are the view-global subset of the legacy fs_params block, in
//   the legacy relative order: light/camera/cascade+spot+point matrices,
//   clustered tile descriptor, hemispheric terms, refraction camera
//   (view_proj/capture). Per-material/object lanes (base/pbr factors,
//   UV/channel/coat/anisotropy/transmission/SSS/refraction gate,
//   probe state) live in common/frame_mat.glsl (UB 5).
// - shadow_params/shadow_splits/cascade_* carry the receive-shadow gate:
//   the draw uploads one of two variants per contiguous pipeline run
//   (FrameContext uniforms_with_shadows/without_shadows), selected by
//   mesh.receive_shadows — never a single blind once-per-view upload,
//   because sokol invalidates bindings AND uniform slots on
//   applyPipeline even for identical layouts (see sokol_gfx.h:
//   "bindings are not preserved across sg_apply_pipeline() calls, even
//   when the new pipeline uses the same 'bindings layout'" plus
//   VALIDATE_DRAW_REQUIRED_BINDINGS_OR_UNIFORMS_MISSING).
    vec4 eye_pos;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient_color;
    vec4 shadow_params; // x: bias, y: intensity, z: normal_bias, w: filter_radius
    vec4 shadow_splits; // x: split0, y: split1, z: split2, w: split3
    mat4 cascade_view_proj[4]; // 4 cascade light view-projection matrices
    vec4 cascade_debug; // x: debug_cascades, y: pcss_enabled, z: pcss_light_size, w: pcss_blocker_radius
    vec4 light_counts; // x: num_point_lights, y: num_spot_lights, z: pcss_min_penumbra, w: pcss_max_penumbra
    vec4 point_pos_range[4];
    vec4 point_color_int[4];
    vec4 spot_pos_range[2];
    vec4 spot_dir_inner[2];
    vec4 spot_color_outer[2];
    vec4 spot_intensity[2];
    mat4 spot_view_proj[2];
    vec4 spot_shadow_params[2]; // per packed spot light: x: shadow_slot+1 (0 = none), y: bias, z: normal_bias, w: source_radius
    mat4 point_view_proj[12];
    vec4 point_shadow_params[4]; // per packed point light: x: shadow_slot+1 (0 = none), y: bias, z: normal_bias, w: source_radius
    vec4 directional_dir[4];
    vec4 directional_color_int[4]; // rgb: color, a: intensity
    vec4 area_center_int[2];
    vec4 area_right[2];
    vec4 area_up[2];
    vec4 area_color[2];
    vec4 clustered_params;
    vec4 clustered_viewport;
    vec4 hemi_dir_intensity; // xyz: direction toward the light (normalized), w: intensity
    vec4 hemi_diffuse; // rgb: diffuse color, a: unused
    mat4 refraction_view_proj;
    vec4 refraction_capture;
    // APPENDED LAST (DDGI v1 spike, opt-in): probe-grid descriptor for
    // the atlas sample (common/ddgi_sample.glsl; CPU mirror:
    // gi/ddgi.gridLanes). Zeros when disabled, so the off path uploads
    // bit-identical values plus zero lanes the shader gates on.
    // origin.xyz grid origin, w enabled (0/1); counts.xyz probes, w
    // sample intensity; cell xyz, w tile texels; atlas size xy, tiles
    // per row z, w unused.
    vec4 ddgi_grid_origin;
    vec4 ddgi_grid_counts;
    vec4 ddgi_grid_step;
    vec4 ddgi_atlas;
