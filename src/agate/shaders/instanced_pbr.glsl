// Instanced Cook-Torrance PBR (Physically Based Rendering) with Multi-Lights for agate
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
in vec4 tangent;
in vec4 color0;
in vec2 texcoord0;

// Per-instance attributes (Buffer 1)
in vec4 inst_mat0;
in vec4 inst_mat1;
in vec4 inst_mat2;
in vec4 inst_mat3;

out vec3 v_world_pos;
out vec3 v_normal;
out vec3 v_tangent;
out vec3 v_bitangent;
out vec4 v_color;
out vec2 v_uv;

void main() {
    mat4 model = mat4(inst_mat0, inst_mat1, inst_mat2, inst_mat3);
    vec4 world_pos = model * vec4(position, 1.0);
    v_world_pos = world_pos.xyz;
    gl_Position = view_proj * world_pos;

    vec3 N = normalize(mat3(model) * normal);
    vec3 T = normalize(mat3(model) * tangent.xyz);
    // Gram-Schmidt orthogonalization
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T) * tangent.w;

    v_normal = N;
    v_tangent = T;
    v_bitangent = B;
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
    vec4 eye_pos;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient_color;
    vec4 base_color_factor;
    vec4 pbr_factors; // x: metallic, y: roughness, z: occlusion_strength, w: ibl_intensity
    vec4 emissive_factor; // xyz: emissive color, w: unused
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
    float normal_scale; // normal map xy scale (glTF normalTexture.scale)
    // APPENDED LAST (wave/ktx2): per-slot KHR_texture_transform UV maps and
    // manual channel selection. Slot order follows the texture bindings:
    // 0 albedo, 1 normal, 2 metallic-roughness, 3 emissive, 4 occlusion.
    vec4 uv_matrix[5]; // per slot: rotation*scale rows [m00, m01, m10, m11]
    vec4 uv_offset[5]; // per slot: xy offset, zw unused
    vec4 channel_selectors; // x occlusion, y roughness, z metallic (lane index), w unused
    // APPENDED LAST (clearcoat/sheen): scalar/color-only coat + fabric lobes
    // (Babylon parity, no textures this wave). Intensity 0 disables the lobe:
    // every added shader term scales by its intensity and the base-specular
    // attenuation becomes exactly (1 - 0), so disabled materials render
    // bit-identically to before. Appended last so no existing offset shifts.
    vec4 clearcoat_factors; // x: intensity (0 = off), y: roughness, z/w: unused
    vec4 clearcoat_color; // rgb: coat tint (white = untinted), w: unused
    vec4 sheen_factors; // x: intensity (0 = off), y: roughness, z/w: unused
    vec4 sheen_color; // rgb: fabric tint (white = untinted), w: unused
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
    // Zeroed when no probe applies: the shader then takes the legacy
    // ambient/IBL path bit-identically. Appended last so no offset shifts.
    // (Instanced draws always upload zero here — probes skip instanced
    // batches in v1 — but the lane must exist for layout parity with the
    // regular/skinned PBR FsParams.)
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
};

layout(binding = 0) uniform texture2D albedo_tex;
layout(binding = 1) uniform texture2D normal_tex;
layout(binding = 2) uniform texture2D metallic_roughness_tex;
layout(binding = 3) uniform texture2D emissive_tex;
layout(binding = 4) uniform texture2D occlusion_tex;
layout(binding = 5) uniform texture2D shadow_tex;
@image_sample_type shadow_depth_tex unfilterable_float
layout(binding = 7) uniform texture2D shadow_depth_tex; // same atlas view as shadow_tex, raw-depth reads for PCSS
layout(binding = 6) uniform textureCube env_tex;
layout(binding = 8) uniform texture2D spot_shadow_tex;
// Point shadow atlas (passes/shadow_pass.zig POINT_SHADOW_* layout).
// Binding 10: fs uses 0..8, the instanced vs uses no textures — same slot
// as the other PBR-family shaders. Sampled through shadow_smp.
layout(binding = 10) uniform texture2D point_shadow_tex;
layout(binding = 0) uniform sampler smp; // color slot: albedo (its own sampler)
// Data slots (normal / metallic-roughness / occlusion / emissive) sample
// through data_smp so their textures keep their OWN filter/wrap settings
// instead of inheriting the albedo sampler.
layout(binding = 5) uniform sampler data_smp;
layout(binding = 1) uniform sampler shadow_smp;
layout(binding = 2) uniform sampler env_smp;
@sampler_type depth_smp nonfiltering
layout(binding = 3) uniform sampler depth_smp;
// Reflection probe cube (wave 25): binding 11 is the next free texture
// slot in the shared pool (fs uses 0..8 and 10, the instanced vs uses no
// textures), binding 6 the next free sampler slot. Instanced draws bind
// the default cube with zeroed params (legacy path); the slots must still
// exist for layout parity with the regular PBR family.
layout(binding = 11) uniform textureCube probe_tex;
layout(binding = 6) uniform sampler probe_smp;

in vec3 v_world_pos;
in vec3 v_normal;
in vec3 v_tangent;
in vec3 v_bitangent;
in vec4 v_color;
in vec2 v_uv;

out vec4 frag_color;

const float PI = 3.14159265359;

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

    // Fixed 90-degree rotations keyed by screen hash: same dithering as a random
    // rotation, zero trig per fragment.
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

    // Full 16-tap PCF only for the near cascade; far cascades cover huge texels
    // where extra taps cost without visible quality.
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

// Rect area-light irradiance v1 (wave 26): analytic approximation, NOT LTC
// and NOT a multi-sample integration. The fragment is lit by the closest
// point Q on the rect (parallelogram projection onto the right/up
// half-extent axes); the returned factor is
//   emit * area / (dist^2 + area) * intensity
// with emit = clamp(dot(rect_normal, -L)) (single-sided front-face
// emission) and NdotL as an out-param for the caller's lobe. Limits,
// stated honestly: no LTC lobe, so large/close rects shade harder-edged
// than reality; no rect-shape specular anisotropy (the standard path uses
// a Blinn-Phong boost from the same representative direction L, the PBR
// path reuses its Cook-Torrance lobe); no shadows — an occluded area
// light still lights (v1 scope). Zero intensity or zero area returns 0,
// so zero lights are a bit-identical no-op.
float areaLightFactor(vec3 frag_pos, vec3 N, int area_idx, out vec3 L, out float NdotL) {
    float a_int = area_center_int[area_idx].w;
    vec3 r = area_right[area_idx].xyz;
    vec3 u = area_up[area_idx].xyz;
    vec3 naxis = cross(r, u);
    float rect_area = 4.0 * length(naxis);
    L = N;
    NdotL = 0.0;
    if (a_int <= 0.0 || rect_area <= 1e-8) return 0.0;
    vec3 nrect = naxis / (rect_area * 0.25);
    vec3 c = area_center_int[area_idx].xyz;
    vec3 d = frag_pos - c;
    float x = clamp(dot(d, r) / max(dot(r, r), 1e-6), -1.0, 1.0);
    float y = clamp(dot(d, u) / max(dot(u, u), 1e-6), -1.0, 1.0);
    vec3 to_light = (c + r * x + u * y) - frag_pos;
    float dist = length(to_light);
    L = to_light / max(dist, 1e-4);
    NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return 0.0;
    float emit = clamp(dot(nrect, -L), 0.0, 1.0);
    if (emit <= 0.0) return 0.0;
    float att = rect_area / (dist * dist + rect_area);
    return emit * att * a_int;
}

float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return num / max(denom, 0.0000001);
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec2 envBRDFApprox(float roughness, float NoV) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
    vec2 AB = vec2(-1.04, 1.04) * a004 + r.zw;
    return AB;
}

// Clearcoat + sheen helpers (scalar/color only, Babylon parity). The coat is
// a second GGX specular lobe with its own roughness and a dielectric
// F0 = 0.04 tinted by clearcoat_color; the base specular (direct + IBL) is
// attenuated by (1 - F_cc) for energy conservation while diffuse passes
// through. Sheen is an additive fabric lobe: Charlie distribution
// (Estevez-Kulla) + Neubelt visibility with an explicit grazing-angle
// weight on the IBL path (Karis-style); the direct path gets its grazing
// response from the Neubelt visibility. Intensity 0 zeroes every term, so
// disabled materials shade bit-identically to before.
vec3 clearcoatF0() {
    return vec3(0.04) * clearcoat_color.rgb;
}

float sheenDistributionCharlie(float roughness, float NoH) {
    float alpha = max(roughness * roughness, 0.0001);
    float inv_alpha = 1.0 / alpha;
    float cos2h = NoH * NoH;
    float sin2h = max(1.0 - cos2h, 0.0078125);
    return (2.0 + inv_alpha) * pow(sin2h, inv_alpha * 0.5) / (2.0 * PI);
}

float sheenVisibilityNeubelt(float NoV, float NoL) {
    return clamp(1.0 / (4.0 * (NoL + NoV - NoL * NoV)), 0.0, 1.0);
}

// Combined coat + sheen add-on for one punctual light. The caller multiplies
// `additive` by radiance * NdotL * shadow and scales its own BASE specular
// by `base_atten` (diffuse/kD untouched). Safe at NdotL = 0: the 0.0001
// guard matches the base lobe and the caller zeroes the contribution.
void coatSheenLight(vec3 N, vec3 V, vec3 L, vec3 H, float NdotV, float NdotL,
    float cc_rough, float cc_intensity, vec3 cc_F0,
    float sheen_rough, float sheen_intensity,
    out vec3 base_atten, out vec3 additive) {
    float cc_NDF = distributionGGX(N, H, cc_rough);
    float cc_G = geometrySmith(N, V, L, cc_rough);
    vec3 cc_F = fresnelSchlick(clamp(dot(H, V), 0.0, 1.0), cc_F0) * cc_intensity;
    base_atten = vec3(1.0) - cc_F;
    vec3 cc_spec = (cc_NDF * cc_G * cc_F) / (4.0 * NdotV * NdotL + 0.0001);
    float sheenD = sheenDistributionCharlie(sheen_rough, max(dot(N, H), 0.0));
    float sheenV = sheenVisibilityNeubelt(NdotV, NdotL);
    vec3 sheen_term = sheen_color.rgb * (sheenD * sheenV) * sheen_intensity;
    additive = cc_spec + sheen_term;
}

// KHR_texture_transform: uv' = matrix * uv + offset. Identity uniforms make
// this a no-op, so materials without the extension sample unchanged.
vec2 uvApply(vec4 m, vec4 o, vec2 uv) {
    return vec2(m.x * uv.x + m.y * uv.y + o.x, m.z * uv.x + m.w * uv.y + o.y);
}

// Manual channel selection (no glTF counterpart; see Channel in
// material.zig): picks one RGBA lane by uniform index through constant
// branches — SPIRV-Cross cannot flatten dynamic component indexing for
// legacy targets (HLSL5), same constraint as morphWeight.
float channelSelect(vec4 s, float lane) {
    if (lane < 0.5) return s.r;
    if (lane < 1.5) return s.g;
    if (lane < 2.5) return s.b;
    return s.a;
}

void main() {
    vec4 albedo_tex_val = texture(sampler2D(albedo_tex, smp), uvApply(uv_matrix[0], uv_offset[0], v_uv));
    vec4 albedo_rgba = v_color * base_color_factor * albedo_tex_val;
    // Alpha test (cutout): cutout materials discard sub-cutoff fragments
    // before any lighting work. Opaque/blend materials upload 0.0, so this
    // never fires for them (alpha is always >= 0.0).
    if (albedo_rgba.a < alpha_cutoff) discard;

    // Unlit mode: bypass all lighting, shadows, and IBL
    if (uv_offset[0].z > 0.5) {
        vec4 emissive_sample = texture(sampler2D(emissive_tex, data_smp), uvApply(uv_matrix[3], uv_offset[3], v_uv));
        vec3 emissive = emissive_factor.rgb * emissive_sample.rgb;
        frag_color = vec4(albedo_rgba.rgb + emissive, albedo_rgba.a);
        return;
    }
    vec3 albedo = albedo_rgba.rgb;

    vec4 mr_sample = texture(sampler2D(metallic_roughness_tex, data_smp), uvApply(uv_matrix[2], uv_offset[2], v_uv));
    float metallic = clamp(pbr_factors.x * channelSelect(mr_sample, channel_selectors.z), 0.0, 1.0);
    float roughness = clamp(pbr_factors.y * channelSelect(mr_sample, channel_selectors.y), 0.04, 1.0);

    // Normal mapping with TBN matrix; xy scaled by normal_scale (z stays
    // unsigned so the TBN projection keeps the hemisphere).
    vec3 map_n = texture(sampler2D(normal_tex, data_smp), uvApply(uv_matrix[1], uv_offset[1], v_uv)).xyz * 2.0 - 1.0;
    map_n.xy *= normal_scale;
    mat3 TBN = mat3(normalize(v_tangent), normalize(v_bitangent), normalize(v_normal));
    vec3 N = normalize(TBN * map_n);

    vec3 V = normalize(eye_pos.xyz - v_world_pos);
    float NdotV = max(dot(N, V), 0.0001);

    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    // Clearcoat + sheen factors (scalar/color only, no textures this wave).
    // Intensity 0 disables the lobe while keeping every legacy term exact.
    float cc_rough = clamp(clearcoat_factors.y, 0.03, 1.0);
    float cc_intensity = clamp(clearcoat_factors.x, 0.0, 1.0);
    vec3 cc_F0 = clearcoatF0();
    float sheen_rough = clamp(sheen_factors.y, 0.07, 1.0);
    float sheen_intensity = clamp(sheen_factors.x, 0.0, 1.0);

    // 1. Primary Directional Light (with shadow mapping)
    vec3 L = light_dir.xyz;
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);

    float NDF = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 specular = (NDF * G * F) / (4.0 * NdotV * NdotL + 0.0001);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

    // Coat + sheen add-on for the sun (base_atten scales the BASE specular
    // only — diffuse passes through unattenuated).
    vec3 sun_atten;
    vec3 sun_additive;
    coatSheenLight(N, V, L, H, NdotV, NdotL, cc_rough, cc_intensity, cc_F0, sheen_rough, sheen_intensity, sun_atten, sun_additive);

    vec3 debug_tint = vec3(0.0);
    float shadow = calculateShadow(v_world_pos, N, L, debug_tint);
    vec3 radiance = light_color.rgb * light_color.a;
    vec3 Lo = (kD * albedo / PI + specular * sun_atten + sun_additive) * radiance * NdotL * (1.0 - shadow);

    // Extra directional fills (slots 1..3, no shadows): the same
    // Cook-Torrance lobe as the sun (coat + sheen included), no shadow
    // term. Zero intensity (disabled/unused) skips, so a single sun
    // shades bit-identically.
    for (int i = 1; i < 4; i++) {
        vec3 d_dir = directional_dir[i].xyz;
        vec3 d_col = directional_color_int[i].rgb;
        float d_int = directional_color_int[i].w;
        if (d_int <= 0.0) continue;
        float d_NdotL = max(dot(N, d_dir), 0.0);
        if (d_NdotL <= 0.0) continue;
        vec3 d_H = normalize(V + d_dir);
        float d_NDF = distributionGGX(N, d_H, roughness);
        float d_G = geometrySmith(N, V, d_dir, roughness);
        vec3 d_F = fresnelSchlick(max(dot(d_H, V), 0.0), F0);
        vec3 d_spec = (d_NDF * d_G * d_F) / (4.0 * NdotV * d_NdotL + 0.0001);
        vec3 d_kD = (vec3(1.0) - d_F) * (1.0 - metallic);
        vec3 d_rad = d_col * d_int;
        vec3 d_atten;
        vec3 d_additive;
        coatSheenLight(N, V, d_dir, d_H, NdotV, d_NdotL, cc_rough, cc_intensity, cc_F0, sheen_rough, sheen_intensity, d_atten, d_additive);
        Lo += (d_kD * albedo / PI + d_spec * d_atten + d_additive) * d_rad * d_NdotL;
    }

    // 2. Point Lights (up to 4)
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
        vec3 p_H = normalize(V + p_L);

        // Windowed inverse-square attenuation (smooth cutoff at p_range)
        float d_norm = dist / p_range;
        float factor = clamp(1.0 - d_norm * d_norm * d_norm * d_norm, 0.0, 1.0);
        float att = (factor * factor) / (dist * dist + 1.0);

        float p_NdotL = max(dot(N, p_L), 0.0);
        if (p_NdotL > 0.0) {
            float p_NDF = distributionGGX(N, p_H, roughness);
            float p_G = geometrySmith(N, V, p_L, roughness);
            vec3 p_F = fresnelSchlick(max(dot(p_H, V), 0.0), F0);

            vec3 p_spec = (p_NDF * p_G * p_F) / (4.0 * NdotV * p_NdotL + 0.0001);
            vec3 p_kD = (vec3(1.0) - p_F) * (1.0 - metallic);
            vec3 p_rad = p_col * (p_int * att);
            vec3 p_atten;
            vec3 p_additive;
            coatSheenLight(N, V, p_L, p_H, NdotV, p_NdotL, cc_rough, cc_intensity, cc_F0, sheen_rough, sheen_intensity, p_atten, p_additive);

            float point_shadow = calculatePointShadow(i, v_world_pos, N, p_L);
            Lo += (p_kD * albedo / PI + p_spec * p_atten + p_additive) * p_rad * p_NdotL * (1.0 - point_shadow);
        }
    }

    // 3. Spot Lights (up to 2)
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
        vec3 s_H = normalize(V + s_L);

        // Distance attenuation
        float d_norm = dist / s_range;
        float factor = clamp(1.0 - d_norm * d_norm * d_norm * d_norm, 0.0, 1.0);
        float dist_att = (factor * factor) / (dist * dist + 1.0);

        // Cone attenuation
        float cos_angle = dot(-s_L, s_dir);
        float cone_att = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001), 0.0, 1.0);
        cone_att *= cone_att;

        float total_att = dist_att * cone_att;
        if (total_att <= 0.0) continue;

        float s_NdotL = max(dot(N, s_L), 0.0);
        if (s_NdotL > 0.0) {
            float s_NDF = distributionGGX(N, s_H, roughness);
            float s_G = geometrySmith(N, V, s_L, roughness);
            vec3 s_F = fresnelSchlick(max(dot(s_H, V), 0.0), F0);

            vec3 s_spec = (s_NDF * s_G * s_F) / (4.0 * NdotV * s_NdotL + 0.0001);
            vec3 s_kD = (vec3(1.0) - s_F) * (1.0 - metallic);
            vec3 s_rad = s_col * (s_int * total_att);
            vec3 s_atten;
            vec3 s_additive;
            coatSheenLight(N, V, s_L, s_H, NdotV, s_NdotL, cc_rough, cc_intensity, cc_F0, sheen_rough, sheen_intensity, s_atten, s_additive);

            float spot_shadow = calculateSpotShadow(i, v_world_pos, N, s_L);
            Lo += (s_kD * albedo / PI + s_spec * s_atten + s_additive) * s_rad * s_NdotL * (1.0 - spot_shadow);
        }
    }

    // Rect area lights (up to 2, no shadows): zero-intensity lanes skip,
    // so zero lights add nothing bit-identically. Same Cook-Torrance lobe
    // as the other punctual lights, driven by the closest-point
    // representative direction (no LTC in v1 — see areaLightFactor).
    for (int i = 0; i < 2; i++) {
        vec3 a_L;
        float a_NdotL;
        float a_factor = areaLightFactor(v_world_pos, N, i, a_L, a_NdotL);
        if (a_factor <= 0.0) continue;
        vec3 a_H = normalize(V + a_L);
        float a_NDF = distributionGGX(N, a_H, roughness);
        float a_G = geometrySmith(N, V, a_L, roughness);
        vec3 a_F = fresnelSchlick(max(dot(a_H, V), 0.0), F0);
        vec3 a_spec = (a_NDF * a_G * a_F) / (4.0 * NdotV * a_NdotL + 0.0001);
        vec3 a_kD = (vec3(1.0) - a_F) * (1.0 - metallic);
        vec3 a_rad = area_color[i].rgb * a_factor;
        vec3 a_atten;
        vec3 a_additive;
        coatSheenLight(N, V, a_L, a_H, NdotV, a_NdotL, cc_rough, cc_intensity, cc_F0, sheen_rough, sheen_intensity, a_atten, a_additive);
        Lo += (a_kD * albedo / PI + a_spec * a_atten + a_additive) * a_rad * a_NdotL;
    }

    // Ambient Occlusion
    float ao_sample = channelSelect(texture(sampler2D(occlusion_tex, data_smp), uvApply(uv_matrix[4], uv_offset[4], v_uv)), channel_selectors.x);
    float ao = 1.0 + pbr_factors.z * (ao_sample - 1.0);

    // Image-Based Lighting (IBL): two cube fetches + BRDF fit skipped when off.
    // Reflection probe (wave 25): same substitution as the regular PBR
    // shader (probe cube replaces env_tex when probe_params.x > 0.5).
    // Instanced draws always upload zero here, so this stays legacy.
    vec3 ibl = vec3(0.0);
    float ibl_intensity = pbr_factors.w;
    if (ibl_intensity > 0.001) {
        vec3 R = reflect(-V, N);
        float max_lod = 7.0;
        float lod = roughness * max_lod;
        vec3 prefiltered_spec;
        vec3 irradiance;
        if (probe_params.x > 0.5) {
            prefiltered_spec = textureLod(samplerCube(probe_tex, probe_smp), R, clamp(lod, 0.0, probe_params.z)).rgb * probe_params.y;
            irradiance = textureLod(samplerCube(probe_tex, probe_smp), N, probe_params.z).rgb * probe_params.y;
        } else {
            prefiltered_spec = textureLod(samplerCube(env_tex, env_smp), R, lod).rgb;
            irradiance = textureLod(samplerCube(env_tex, env_smp), N, max_lod).rgb;
        }

        vec3 F_ibl = fresnelSchlickRoughness(NdotV, F0, roughness);
        vec2 brdf = envBRDFApprox(roughness, NdotV);
        vec3 specular_ibl = prefiltered_spec * (F0 * brdf.x + brdf.y);

        // Clearcoat IBL: own roughness lobe; its fresnel attenuates the base
        // specular IBL (energy conservation). Sheen IBL: grazing-weighted
        // share of the diffuse irradiance. Both gated on intensity so the
        // disabled path keeps the legacy IBL bit-identical (and skips the
        // extra cube fetch).
        vec3 cc_F_ibl = fresnelSchlickRoughness(NdotV, cc_F0, cc_rough) * cc_intensity;
        vec3 cc_spec_ibl = vec3(0.0);
        if (cc_intensity > 0.001) {
            float cc_lod = cc_rough * max_lod;
            vec3 cc_prefiltered;
            if (probe_params.x > 0.5) {
                cc_prefiltered = textureLod(samplerCube(probe_tex, probe_smp), R, clamp(cc_lod, 0.0, probe_params.z)).rgb * probe_params.y;
            } else {
                cc_prefiltered = textureLod(samplerCube(env_tex, env_smp), R, cc_lod).rgb;
            }
            vec2 cc_brdf = envBRDFApprox(cc_rough, NdotV);
            cc_spec_ibl = cc_prefiltered * (cc_F0 * cc_brdf.x + cc_brdf.y) * cc_intensity;
        }
        specular_ibl = specular_ibl * (vec3(1.0) - cc_F_ibl) + cc_spec_ibl;

        vec3 kD_ibl = (vec3(1.0) - F_ibl) * (1.0 - metallic);
        vec3 diffuse_ibl = kD_ibl * irradiance * albedo;
        float sheen_grazing = pow(clamp(1.0 - NdotV, 0.0, 1.0), 5.0);
        vec3 sheen_ibl = sheen_color.rgb * sheen_intensity * irradiance * sheen_grazing;

        ibl = (diffuse_ibl + specular_ibl) * (ibl_intensity * ao) + sheen_ibl * (ibl_intensity * ao);
    }

    // Directional ambient base
    vec3 ambient = ambient_color.rgb * ambient_color.a * albedo * ao;

    // Emissive
    vec4 emissive_sample = texture(sampler2D(emissive_tex, data_smp), uvApply(uv_matrix[3], uv_offset[3], v_uv));
    vec3 emissive = emissive_factor.rgb * emissive_sample.rgb;

    vec3 final_color = ambient + ibl + Lo + emissive + debug_tint;

    frag_color = vec4(final_color, albedo_rgba.a);
}
@end

@program instanced_pbr vs fs
