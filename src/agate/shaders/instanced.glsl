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
vec2 uvApply(vec4 m, vec4 o, vec2 uv) {
    return vec2(m.x * uv.x + m.y * uv.y + o.x, m.z * uv.x + m.w * uv.y + o.y);
}

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

float hash01(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// PCSS blocker search: average depth of taps closer to the light than the
// receiver, or -1.0 when nothing blocks (caller early-outs to fully lit).
float pcssBlockerAverage(texture2D depth_tex, sampler depth_smp, vec2 atlas_uv, float receiver_depth, mat2 rot, float search_radius, vec2 quad_min, vec2 quad_max) {
    float blocker_sum = 0.0;
    int blocker_count = 0;
    for (int i = 0; i < PCSS_BLOCKER_SAMPLES; i++) {
        vec2 tap_uv = clamp(atlas_uv + rot * POISSON_DISK[i] * search_radius, quad_min, quad_max);
        float tap_depth = texture(sampler2D(depth_tex, depth_smp), tap_uv).r;
        if (tap_depth < receiver_depth) {
            blocker_sum += tap_depth;
            blocker_count += 1;
        }
    }
    if (blocker_count == 0) return -1.0;
    return blocker_sum / float(blocker_count);
}

// PCSS variable penumbra, atlas-UV radius for the PCF disk. Mirrors
// penumbraRadius in scene/shadow_pcss.zig (parallel rays: linear scaling without perspective division).
float pcssPenumbraRadius(float receiver_depth, float blocker_avg, float light_size, float min_penumbra, float max_penumbra) {
    float penumbra = (receiver_depth - blocker_avg) * light_size;
    return clamp(penumbra, min_penumbra, max_penumbra);
}

float sampleCascade(int cascade_idx, vec3 world_pos, vec3 N, vec3 L) {
    float cos_theta = max(dot(N, L), 0.0);
    float depth_bias = max(shadow_params.x * (1.0 - cos_theta), shadow_params.x * 0.2);
    vec3 normal_offset = N * (shadow_params.z * (1.0 - cos_theta));

    vec4 lpos = cascade_view_proj[cascade_idx] * vec4(world_pos + normal_offset, 1.0);
    #if !SOKOL_GLSL
        lpos.y = -lpos.y;
    #endif

    vec3 proj = lpos.xyz / lpos.w;
    if (proj.z > 1.0 || proj.z < 0.0) return 1.0;

    vec2 local_uv = (proj.xy + 1.0) * 0.5;
    if (local_uv.x < 0.0 || local_uv.x > 1.0 || local_uv.y < 0.0 || local_uv.y > 1.0) return 1.0;

    vec2 clamped_local_uv = clamp(local_uv, 0.003, 0.997);
    vec2 atlas_uv = clamped_local_uv * 0.5 + CASCADE_OFFSETS[cascade_idx];
    float depth = proj.z - depth_bias;

    vec2 quad_min = CASCADE_OFFSETS[cascade_idx] + vec2(0.003);
    vec2 quad_max = CASCADE_OFFSETS[cascade_idx] + vec2(0.497);

    float h = hash01(gl_FragCoord.xy);
    mat2 rot = mat2(1.0, 0.0, 0.0, 1.0);
    if (h > 0.75) {
        rot = mat2(0.0, 1.0, -1.0, 0.0);
    } else if (h > 0.5) {
        rot = mat2(-1.0, 0.0, 0.0, -1.0);
    } else if (h > 0.25) {
        rot = mat2(0.0, -1.0, 1.0, 0.0);
    }

    float filter_radius = (shadow_params.w / SHADOW_ATLAS_SIZE) * 0.5;

    int taps = 16;
    if (cascade_idx > 0) taps = 8;

    // PCSS: blocker search sets a receiver-dependent filter radius. Params
    // ride free lanes (cascade_debug.yzw / light_counts.zw); disabled keeps
    // the legacy fixed-radius path below bit-identical.
    if (cascade_debug.y > 0.5) {
        float blocker_avg = pcssBlockerAverage(shadow_depth_tex, depth_smp, atlas_uv, depth, rot, cascade_debug.w, quad_min, quad_max);
        if (blocker_avg < 0.0) return 1.0;
        filter_radius = pcssPenumbraRadius(depth, blocker_avg, cascade_debug.z, light_counts.z, light_counts.w);
    }

    float lit = 0.0;
    for (int i = 0; i < 16; i++) {
        if (i >= taps) break;
        vec2 offset = rot * POISSON_DISK[i] * filter_radius;
        vec2 sample_uv = clamp(atlas_uv + offset, quad_min, quad_max);
        lit += texture(sampler2DShadow(shadow_tex, shadow_smp), vec3(sample_uv, depth));
    }
    return lit / float(taps);
}

float calculateShadow(vec3 world_pos, vec3 N, vec3 L, out vec3 debug_color) {
    debug_color = vec3(0.0);
    if (shadow_params.y <= 0.001) return 0.0;

    float view_dist = length(world_pos - eye_pos.xyz);
    int cascade_idx = 3;
    if (view_dist < shadow_splits.x) {
        cascade_idx = 0;
    } else if (view_dist < shadow_splits.y) {
        cascade_idx = 1;
    } else if (view_dist < shadow_splits.z) {
        cascade_idx = 2;
    }

    if (cascade_debug.x > 0.5) {
        if (cascade_idx == 0) debug_color = vec3(0.25, 0.05, 0.05);
        else if (cascade_idx == 1) debug_color = vec3(0.05, 0.25, 0.05);
        else if (cascade_idx == 2) debug_color = vec3(0.05, 0.05, 0.25);
        else debug_color = vec3(0.25, 0.25, 0.05);
    }

    float lit = sampleCascade(cascade_idx, world_pos, N, L);

    float fade_start = shadow_splits.w * 0.85;
    if (view_dist > fade_start) {
        float fade = clamp((view_dist - fade_start) / max(shadow_splits.w - fade_start, 0.001), 0.0, 1.0);
        lit = mix(lit, 1.0, fade);
    }

    return (1.0 - lit) * shadow_params.y;
}

float calculateSpotShadow(int spot_idx, vec3 world_pos, vec3 N, vec3 L) {
    if (spot_shadow_params[spot_idx].x < 0.5) return 0.0;
    if (shadow_params.y <= 0.001) return 0.0;

    float cos_theta = max(dot(N, L), 0.0);
    float depth_bias = max(spot_shadow_params[spot_idx].y * (1.0 - cos_theta), spot_shadow_params[spot_idx].y * 0.2);
    vec3 normal_offset = N * (spot_shadow_params[spot_idx].z * (1.0 - cos_theta));

    vec4 lpos = spot_view_proj[spot_idx] * vec4(world_pos + normal_offset, 1.0);
    #if !SOKOL_GLSL
        lpos.y = -lpos.y;
    #endif

    vec3 proj = lpos.xyz / lpos.w;
    if (proj.z > 1.0 || proj.z < 0.0) return 0.0;

    vec2 local_uv = (proj.xy + 1.0) * 0.5;
    if (local_uv.x < 0.0 || local_uv.x > 1.0 || local_uv.y < 0.0 || local_uv.y > 1.0) return 0.0;

    vec2 clamped_uv = clamp(local_uv, 0.002, 0.998);
    vec2 atlas_uv = vec2(clamped_uv.x * 0.5 + float(spot_idx) * 0.5, clamped_uv.y);
    float depth = proj.z - depth_bias;

    vec2 texel = vec2(1.0 / 1024.0, 1.0 / 512.0);
    float lit = 0.0;
    lit += texture(sampler2DShadow(spot_shadow_tex, shadow_smp), vec3(atlas_uv + vec2(-texel.x, -texel.y), depth));
    lit += texture(sampler2DShadow(spot_shadow_tex, shadow_smp), vec3(atlas_uv + vec2( texel.x, -texel.y), depth));
    lit += texture(sampler2DShadow(spot_shadow_tex, shadow_smp), vec3(atlas_uv + vec2(-texel.x,  texel.y), depth));
    lit += texture(sampler2DShadow(spot_shadow_tex, shadow_smp), vec3(atlas_uv + vec2( texel.x,  texel.y), depth));
    lit *= 0.25;

    return (1.0 - lit) * shadow_params.y;
}

int pointFaceIndex(vec3 d) {
    vec3 a = abs(d);
    if (a.x >= a.y && a.x >= a.z) return d.x >= 0.0 ? 0 : 1;
    if (a.y >= a.x && a.y >= a.z) return d.y >= 0.0 ? 2 : 3;
    return d.z >= 0.0 ? 4 : 5;
}

// Point-light shadows: each shadow-casting point light owns 6 cube-face
// tiles (one atlas row, 256px tiles in the 1536x512 point atlas). The face
// is picked from the world-space direction to the light, then the fragment
// is projected by that face's view-projection matrix and PCF-sampled with
// the same 2D compare path as the spot atlas (4 taps + bias, shadow
// strength from shadow_params.y like every other shadow term here).
float calculatePointShadow(int light_idx, vec3 world_pos, vec3 N, vec3 L) {
    if (point_shadow_params[light_idx].x < 0.5) return 0.0;
    if (shadow_params.y <= 0.001) return 0.0;

    int slot = int(point_shadow_params[light_idx].x - 1.0);
    vec3 to_frag = world_pos - point_pos_range[light_idx].xyz;
    int face = pointFaceIndex(to_frag);

    float cos_theta = max(dot(N, L), 0.0);
    float bias = point_shadow_params[light_idx].y;
    float depth_bias = max(bias * (1.0 - cos_theta), bias * 0.2);
    vec3 normal_offset = N * (point_shadow_params[light_idx].z * (1.0 - cos_theta));

    vec4 lpos = point_view_proj[slot * 6 + face] * vec4(world_pos + normal_offset, 1.0);
    #if !SOKOL_GLSL
        lpos.y = -lpos.y;
    #endif

    vec3 proj = lpos.xyz / lpos.w;
    if (proj.z > 1.0 || proj.z < 0.0) return 0.0;

    vec2 local_uv = (proj.xy + 1.0) * 0.5;
    if (local_uv.x < 0.0 || local_uv.x > 1.0 || local_uv.y < 0.0 || local_uv.y > 1.0) return 0.0;

    // Point atlas layout (must match ShadowPass.pointTileOrigin):
    // 6 faces left-to-right, one row per shadow slot.
    vec2 clamped_uv = clamp(local_uv, 0.002, 0.998);
    vec2 atlas_uv = vec2((float(face) + clamped_uv.x) / 6.0, float(slot) * 0.5 + clamped_uv.y * 0.5);
    float depth = proj.z - depth_bias;

    vec2 texel = vec2(1.0 / 1536.0, 1.0 / 512.0);
    float lit = 0.0;
    lit += texture(sampler2DShadow(point_shadow_tex, shadow_smp), vec3(atlas_uv + vec2(-texel.x, -texel.y), depth));
    lit += texture(sampler2DShadow(point_shadow_tex, shadow_smp), vec3(atlas_uv + vec2( texel.x, -texel.y), depth));
    lit += texture(sampler2DShadow(point_shadow_tex, shadow_smp), vec3(atlas_uv + vec2(-texel.x,  texel.y), depth));
    lit += texture(sampler2DShadow(point_shadow_tex, shadow_smp), vec3(atlas_uv + vec2( texel.x,  texel.y), depth));
    lit *= 0.25;

    return (1.0 - lit) * shadow_params.y;
}

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

    // Directional ambient base. Reflection probe (wave 25): same
    // substitution as the regular standard shader (probe coarsest mip when
    // probe_params.x > 0.5). Instanced draws always upload zero here, so
    // this stays legacy.
    vec3 ambient = ambient_color.rgb * ambient_color.a;
    if (probe_params.x > 0.5) {
        ambient = textureLod(samplerCube(probe_tex, probe_smp), N, probe_params.z).rgb * probe_params.y;
    }

    vec3 final_rgb = base.rgb * (ambient + diffuse) + debug_tint;
    frag_color = vec4(final_rgb, base.a);
}
@end

@program instanced vs fs

