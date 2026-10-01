// Instanced Standard Shader with CSM and Multi-Lights for agate
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 view_proj;
};

// Per-vertex attributes (Buffer 0)
in vec3 position;
in vec3 normal;
in vec4 color0;
in vec2 texcoord0;

// Per-instance attributes (Buffer 1)
in vec4 inst_mat0;
in vec4 inst_mat1;
in vec4 inst_mat2;
in vec4 inst_mat3;

out vec3 v_world_pos;
out vec3 v_normal;
out vec4 v_color;
out vec2 v_uv;

void main() {
    mat4 model = mat4(inst_mat0, inst_mat1, inst_mat2, inst_mat3);
    vec4 world_pos = model * vec4(position, 1.0);
    v_world_pos = world_pos.xyz;
    gl_Position = view_proj * world_pos;
    v_normal = mat3(model) * normal;
    v_color = color0;
    v_uv = texcoord0;
}
@end

@fs fs
// Shadow atlas resolution. Must match ShadowPass.SHADOW_ATLAS_SIZE in
// passes/shadow_pass.zig (sokol-shdc --defines cannot carry a value).
#ifndef SHADOW_ATLAS_SIZE
#define SHADOW_ATLAS_SIZE 2048.0
#endif
layout(binding = 1) uniform fs_params {
    vec4 eye_pos; // xyz: camera position, w: cascade count
    vec4 light_dir; // xyz: light direction, w: shadow map size
    vec4 light_color; // rgb: color, a: intensity
    vec4 ambient_color;
    vec4 diffuse_color;
    vec4 shadow_params; // x: bias, y: intensity, z: normal_bias, w: filter_radius
    vec4 shadow_splits; // x: split0, y: split1, z: split2, w: split3
    mat4 cascade_view_proj[4]; // 4 cascade light view-projection matrices
    vec4 cascade_debug; // x: debug_cascades, y/z/w: unused
    vec4 light_counts; // x: num_point_lights, y: num_spot_lights, z/w: unused
    vec4 point_pos_range[4];
    vec4 point_color_int[4];
    vec4 spot_pos_range[2];
    vec4 spot_dir_inner[2];
    vec4 spot_color_outer[2];
    vec4 spot_intensity[2];
    mat4 spot_view_proj[2];
    vec4 spot_shadow_params[2]; // x: cast_shadows (0/1), y: bias, z: normal_bias, w: unused
    // APPENDED LAST: existing offsets above must not shift for old bindings.
    float alpha_cutoff; // cutout threshold; 0.0 disables the alpha test
    // APPENDED LAST (wave/ktx2): diffuse-slot KHR_texture_transform UV map.
    vec4 uv_matrix; // rotation*scale rows [m00, m01, m10, m11]
    vec4 uv_offset; // xy offset, zw unused
    // APPENDED LAST (point shadows): 2 shadow slots x 6 cube faces
    // (+X,-X,+Y,-Y,+Z,-Z). Zeroed by default: the shader early-outs with
    // zero cost when no point light casts shadows.
    mat4 point_view_proj[12];
    vec4 point_shadow_params[4]; // per point-light slot: x: shadow slot+1 (0 = none), y: bias, z: normal_bias, w: unused
    // APPENDED LAST (multi-directional): up to 4 suns; slot 0 mirrors the
    // primary (light_dir/light_color, the only shadow caster), slots 1..3
    // are shadowless fills. Unused/disabled slots are zeroed (intensity 0
    // skips in the fill loop below).
    vec4 directional_dir[4];
    vec4 directional_color_int[4]; // rgb: color, a: intensity
    // APPENDED LAST (reflection probes, wave 25): per-draw probe state.
    // x: enabled (0/1), y: probe intensity, z: probe max lod, w: unused.
    // Instanced draws always upload zero here — probes skip instanced
    // batches in v1 — but the lane must exist for layout parity with the
    // regular standard FsParams.
    vec4 probe_params;
    // APPENDED LAST (rect area lights, wave 26, v1): up to 2 rects in
    // creation order. area_center_int xyz = rect center, w = intensity
    // (0 when disabled/unused — the loop below skips, so zero lights
    // render bit-identically); area_right/area_up = half-extent vectors;
    // area_color rgb = emitted color. Appended last so no offset shifts.
    // No area-light shadows in v1 (unshadowed by design).
    vec4 area_center_int[2];
    vec4 area_right[2];
    vec4 area_up[2];
    vec4 area_color[2];
    // APPENDED LAST (clustered forward lights, wave 30, v1): tile-grid
    // descriptor for the storage-buffer tile walk below (the light data
    // itself is never a uniform). clustered_params = (tiles_x, tiles_y,
    // staged light count, gpu-live 0/1); clustered_viewport = (screen_w,
    // screen_h, tile_size px, unused). Zeroed with an empty pool (and w =
    // 0 until a live GPU upload lands), so the loop gates off
    // bit-identically. Appended last so no offset shifts.
    vec4 clustered_params;
    vec4 clustered_viewport;
    // APPENDED LAST (hemispheric light model): Babylon's HemisphericLight is
    // a light, not a flat ambient — its irradiance interpolates between
    // `ambient_color.rgb` (groundColor) and `hemi_diffuse.rgb * w` by
    // `0.5 + 0.5 * dot(N, hemi_dir_intensity.xyz)`. See common/hemi.glsl.
    vec4 hemi_dir_intensity; // xyz: direction toward the light (normalized), w: intensity
    vec4 hemi_diffuse; // rgb: diffuse color, a: unused
};

layout(binding = 0) uniform texture2D diffuse_tex;
layout(binding = 1) uniform texture2D shadow_tex;
@image_sample_type shadow_depth_tex unfilterable_float
layout(binding = 2) uniform texture2D shadow_depth_tex; // same atlas view as shadow_tex, raw-depth reads for PCSS
layout(binding = 3) uniform texture2D spot_shadow_tex;
// Point shadow atlas (passes/shadow_pass.zig POINT_SHADOW_* layout).
// Binding 4: the instanced vs uses no textures, fs uses 0..3.
// Sampled through the shared shadow_smp compare sampler.
layout(binding = 4) uniform texture2D point_shadow_tex;
layout(binding = 0) uniform sampler smp;
layout(binding = 1) uniform sampler shadow_smp;
@sampler_type depth_smp nonfiltering
layout(binding = 2) uniform sampler depth_smp;
// Reflection probe cube (wave 25): binding 11 is the next free texture
// slot in the shared pool (fs uses 0..4, the instanced vs uses no
// textures), binding 6 the next free sampler slot. Instanced draws bind
// the default cube with zeroed params (legacy path); the slots must still
// exist for layout parity with the regular standard family.
layout(binding = 11) uniform textureCube probe_tex;
layout(binding = 6) uniform sampler probe_smp;
// Clustered forward lights (wave 30, v1): storage buffers for the tile
// walk. Bindings 12..14 are free view slots in the shared pool of every
// forward shader (this family uses fs 0..4 and 11, vs no textures).
// Layouts mirror scene/clustered_lights.zig (ClusterLightGpu = 2x vec4,
// ClusterTileGpu = uvec2, indices = u32); every buffer block holds exactly
// one flexible array of a struct (sokol-shdc requirement). The draw always
// binds something valid here (real tile buffers when a GPU upload landed,
// the shared 16-byte dummy otherwise); the loop below only reads them when
// clustered_params gates it on.
// @include "common/cluster.glsl"

in vec3 v_world_pos;
in vec3 v_normal;
in vec4 v_color;
in vec2 v_uv;

out vec4 frag_color;

const vec2 POISSON_DISK[16] = vec2[](
    vec2(-0.94201624, -0.39906216),
    vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870),
    vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432),
    vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845),
    vec2( 0.97484398,  0.75648379),
    vec2( 0.44323325, -0.97511554),
    vec2( 0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023),
    vec2( 0.79197514,  0.19090188),
    vec2(-0.24188840,  0.99706507),
    vec2(-0.81409955,  0.91437590),
    vec2( 0.19984126,  0.78641367),
    vec2( 0.14383161, -0.14100790)
);

const vec2 CASCADE_OFFSETS[4] = vec2[](
    vec2(0.0, 0.0),
    vec2(0.5, 0.0),
    vec2(0.0, 0.5),
    vec2(0.5, 0.5)
);

// PCSS (percentage-closer soft shadows) for the 4-cascade sun atlas.
// Blocker search reads raw depths from shadow_depth_tex (same atlas view as
// shadow_tex, regular sampler): PCSS_BLOCKER_SAMPLES Poisson taps inside
// pcss_blocker_radius, average blocker depth -> penumbra ->
// (d_receiver - d_blocker) / d_blocker * light_size, clamped to
// [min_penumbra, max_penumbra]; the legacy Poisson PCF then runs with the
// KHR_texture_transform: uv' = matrix * uv + offset (identity uniforms are
// a no-op; see material.zig UvTransform for the packing).
// @include "common/uv_apply.glsl"

// penumbra as its disk radius. Params ride free uniform lanes (fs_params
// layout unchanged):
//   cascade_debug.y = pcss_enabled (0.0/1.0)
//   cascade_debug.z = pcss_light_size
//   cascade_debug.w = pcss_blocker_radius (atlas-UV search radius)
//   light_counts.z  = pcss_min_penumbra (atlas-UV clamp)
//   light_counts.w  = pcss_max_penumbra (atlas-UV clamp)
// Disabled (y <= 0.5): legacy fixed-radius 16x/8x Poisson PCF, bit-identical.
// Counts mirror scene/shadow_pcss.zig (blocker_sample_count).
#define PCSS_BLOCKER_SAMPLES 12

// @include "common/shadow_pcf.glsl"
// @include "common/hemi.glsl"

void main() {
    vec3 N = normalize(v_normal);

    // Alpha test (cutout): cutout materials discard sub-cutoff fragments
    // before any lighting work. Opaque/blend materials upload 0.0, so this
    // never fires for them (alpha is always >= 0.0).
    // KHR_texture_transform: uv' = matrix * uv + offset. Identity uniforms
    // make this a no-op, so materials without the extension sample unchanged.
    vec4 tex_val = texture(sampler2D(diffuse_tex, smp), uvApply(uv_matrix, uv_offset, v_uv));
    vec4 base = v_color * diffuse_color * tex_val;
    if (base.a < alpha_cutoff) discard;

    // Unlit mode: bypass all lighting and shadows
    if (uv_offset.z > 0.5) {
        frag_color = base;
        return;
    }

    // Primary directional light
    vec3 L = light_dir.xyz;
    float NdotL = max(dot(N, L), 0.0);
    vec3 debug_tint = vec3(0.0);
    float shadow = calculateShadow(v_world_pos, N, L, debug_tint);
    vec3 diffuse = light_color.rgb * (NdotL * light_color.a) * (1.0 - shadow);

    // Extra directional fills (slots 1..3, no shadows): zero intensity
    // (disabled/unused) skips, so a single sun renders bit-identically.
    for (int i = 1; i < 4; i++) {
        vec3 d_dir = directional_dir[i].xyz;
        vec3 d_col = directional_color_int[i].rgb;
        float d_int = directional_color_int[i].w;
        if (d_int <= 0.0) continue;
        float d_NdotL = max(dot(N, d_dir), 0.0);
        if (d_NdotL <= 0.0) continue;
        diffuse += d_col * (d_NdotL * d_int);
    }

    // Point Lights
    int num_points = int(light_counts.x);
    for (int i = 0; i < 4; i++) {
        if (i >= num_points) break;
        vec3 p_pos = point_pos_range[i].xyz;
        float p_range = point_pos_range[i].w;
        vec3 p_col = point_color_int[i].rgb;
        float p_int = point_color_int[i].w;

        vec3 p_to_light = p_pos - v_world_pos;
        float dist = length(p_to_light);
        if (dist >= p_range || dist < 0.0001) continue;

        vec3 p_L = p_to_light / dist;
        float p_NdotL = max(dot(N, p_L), 0.0);
        if (p_NdotL > 0.0) {
            float d_norm = dist / p_range;
            float factor = clamp(1.0 - d_norm * d_norm * d_norm * d_norm, 0.0, 1.0);
            float att = (factor * factor) / (dist * dist + 1.0);
            float point_shadow = calculatePointShadow(i, v_world_pos, N, p_L);
            diffuse += p_col * (p_NdotL * p_int * att * (1.0 - point_shadow));
        }
    }

    // Spot Lights
    int num_spots = int(light_counts.y);
    for (int i = 0; i < 2; i++) {
        if (i >= num_spots) break;
        vec3 s_pos = spot_pos_range[i].xyz;
        float s_range = spot_pos_range[i].w;
        vec3 s_dir = spot_dir_inner[i].xyz;
        float cos_inner = spot_dir_inner[i].w;
        vec3 s_col = spot_color_outer[i].rgb;
        float cos_outer = spot_color_outer[i].w;
        float s_int = spot_intensity[i].x;

        vec3 s_to_light = s_pos - v_world_pos;
        float dist = length(s_to_light);
        if (dist >= s_range || dist < 0.0001) continue;

        vec3 s_L = s_to_light / dist;
        float s_NdotL = max(dot(N, s_L), 0.0);
        if (s_NdotL > 0.0) {
            float d_norm = dist / s_range;
            float factor = clamp(1.0 - d_norm * d_norm * d_norm * d_norm, 0.0, 1.0);
            float dist_att = (factor * factor) / (dist * dist + 1.0);

            float cos_angle = dot(-s_L, s_dir);
            float cone_att = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001), 0.0, 1.0);
            cone_att *= cone_att;

            float spot_shadow = calculateSpotShadow(i, v_world_pos, N, s_L);
            diffuse += s_col * (s_NdotL * s_int * dist_att * cone_att * (1.0 - spot_shadow));
        }
    }

    // Rect area lights (up to 2, no shadows): zero-intensity lanes skip,
    // so zero lights add nothing bit-identically. Diffuse from the
    // closest-point direction plus a small Blinn-Phong boost — rect-shape
    // specular anisotropy is out of v1 scope (see areaLightFactor).
    for (int i = 0; i < 2; i++) {
        vec3 a_L;
        float a_NdotL;
        float a_factor = areaLightFactor(v_world_pos, N, i, a_L, a_NdotL);
        if (a_factor <= 0.0) continue;
        vec3 a_V = normalize(eye_pos.xyz - v_world_pos);
        vec3 a_H = normalize(a_V + a_L);
        float a_spec = pow(max(dot(N, a_H), 0.0), 32.0) * 0.25;
        diffuse += area_color[i].rgb * (a_factor * (a_NdotL + a_spec));
    }

    // Clustered forward point lights (up to 64, no shadows): pixel -> tile
    // -> tile light indices -> the same point-light math as the legacy
    // lanes above (windowed inverse-square, no shadow term). Gated on a
    // live tile grid (tiles_x/y > 0), a non-empty staged pool (count > 0)
    // and a landed GPU upload (w > 0.5); otherwise skipped entirely, so the
    // empty pool shades bit-identically. Tile id follows the CPU build
    // (scene/clustered_lights.zig tileNdcRect, bottom-left origin); Metal
    // and D3D run window-y down, hence the !SOKOL_GLSL flip. The tile
    // lookup is byte-identical in all five forward shaders (only the lobe
    // tail mirrors the host shader); sokol-shdc has no include mechanism,
    // so the duplication is deliberate and documented.
    if (clustered_params.x > 0.5 && clustered_params.y > 0.5 && clustered_params.z > 0.5 && clustered_params.w > 0.5) {
        vec2 c_px = gl_FragCoord.xy;
        #if !SOKOL_GLSL
            c_px.y = clustered_viewport.y - c_px.y;
        #endif
        int c_tx = clamp(int(c_px.x / clustered_viewport.z), 0, int(clustered_params.x) - 1);
        int c_ty = clamp(int(c_px.y / clustered_viewport.z), 0, int(clustered_params.y) - 1);
        uvec2 c_head = cluster_tiles[uint(c_ty) * uint(clustered_params.x) + uint(c_tx)].head;
        for (uint c_k = 0u; c_k < c_head.y; c_k++) {
            uint c_li = cluster_indices[c_head.x + c_k].light;
            vec3 c_pos = cluster_lights[c_li].pos_range.xyz;
            float c_range = cluster_lights[c_li].pos_range.w;
            vec3 c_col = cluster_lights[c_li].color_int.rgb;
            float c_int = cluster_lights[c_li].color_int.w;

            vec3 c_to_light = c_pos - v_world_pos;
            float c_dist = length(c_to_light);
            if (c_dist >= c_range || c_dist < 0.0001) continue;

            vec3 c_L = c_to_light / c_dist;
            float c_NdotL = max(dot(N, c_L), 0.0);
            if (c_NdotL > 0.0) {
                float c_d_norm = c_dist / c_range;
                float c_factor = clamp(1.0 - c_d_norm * c_d_norm * c_d_norm * c_d_norm, 0.0, 1.0);
                float c_att = (c_factor * c_factor) / (c_dist * c_dist + 1.0);
                diffuse += c_col * (c_NdotL * c_int * c_att);
            }
        }
    }

    // Directional ambient base. Reflection probe (wave 25): same
    // substitution as the regular standard shader (probe coarsest mip when
    // probe_params.x > 0.5). Instanced draws always upload zero here, so
    // this stays legacy.
    // Hemispheric base (Babylon model — see common/hemi.glsl).
    vec3 ambient = hemiIrradiance(N);
    if (probe_params.x > 0.5) {
        ambient = textureLod(samplerCube(probe_tex, probe_smp), N, probe_params.z).rgb * probe_params.y;
    }

    vec3 final_rgb = base.rgb * (ambient + diffuse) + debug_tint;
    frag_color = vec4(final_rgb, base.a);
}
@end

@program instanced vs fs

