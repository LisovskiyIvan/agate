// Skinned Cook-Torrance PBR (Physically Based Rendering) Shader with GPU Matrix Palette Skinning
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 mvp;
    mat4 model;
};
layout(binding = 1) uniform vs_skin {
    mat4 bones[64];
};

// GPU morph targets (opt-in, Mesh.morph_mode == .gpu): per-vertex deltas
// packed in an RGBA32F strip texture, texel index
//   vertex_index * 24 + target * 3 + slot   (slot: 0 pos, 1 normal, 2 tangent)
// texel.xyz carries the delta. Morph deltas are applied BEFORE the skin
// matrix (glTF: morphs deform the bind pose, skinning then transforms).
// Disabled draws bind a 1x1 zero texture with morph_params.x = 0 and zero
// weights. Slots stay unique across both stages (sokol requires a shared
// slot pool): UB 3 (vs_params 0, vs_skin 1, fs_params 2), texture 9
// (fs uses 0..8), sampler 5 (fs uses 0..3 plus data_smp at 5).
layout(binding = 3) uniform vs_morph {
    vec4 morph_weights0; // target weights 0..3
    vec4 morph_weights1; // target weights 4..7
    vec4 morph_params; // x: enabled (0/1), y: tex width, z: tex height, w: unused
};

@image_sample_type morph_tex unfilterable_float
layout(binding = 9) uniform texture2D morph_tex;
@sampler_type morph_smp nonfiltering
layout(binding = 4) uniform sampler morph_smp;

in vec3 position;
in vec3 normal;
in vec4 tangent;
in vec4 color0;
in vec2 texcoord0;
in vec4 joints;
in vec4 weights;

out vec3 v_world_pos;
out vec3 v_normal;
out vec3 v_tangent;
out vec3 v_bitangent;
out vec4 v_color;
out vec2 v_uv;

// One RGBA32F texel (xyz) at strip position idx; exact texel centers with
// NEAREST filtering, so no filtering support is needed for float textures.
vec3 morphTexel(float idx) {
    float u = (mod(idx, morph_params.y) + 0.5) / morph_params.y;
    float v = (floor(idx / morph_params.y) + 0.5) / morph_params.z;
    return texture(sampler2D(morph_tex, morph_smp), vec2(u, v)).xyz;
}

// Target weight lookup with constant vec4 lanes: SPIRV-Cross cannot
// flatten dynamic component indexing (morph_weights0[t]) for legacy
// targets (HLSL5), so the lane is selected via constant branches.
float morphWeight(int t) {
    if (t == 0) return morph_weights0.x;
    if (t == 1) return morph_weights0.y;
    if (t == 2) return morph_weights0.z;
    if (t == 3) return morph_weights0.w;
    if (t == 4) return morph_weights1.x;
    if (t == 5) return morph_weights1.y;
    if (t == 6) return morph_weights1.z;
    return morph_weights1.w;
}

// Mirrors morph_gpu.blendDeltas (mesh/morph_gpu.zig): per-vertex
// base + sum(weight * delta) accumulation in the same order.
void applyMorphDeltas(inout vec3 pos, inout vec3 nrm, inout vec3 tan_xyz, int vertex_id) {
    if (morph_params.x < 0.5) return;
    float base = float(vertex_id) * 24.0;
    for (int t = 0; t < 8; t++) {
        float w = morphWeight(t);
        if (w == 0.0) continue;
        float idx = base + float(t) * 3.0;
        pos += w * morphTexel(idx);
        nrm += w * morphTexel(idx + 1.0);
        tan_xyz += w * morphTexel(idx + 2.0);
    }
}

void main() {
    vec3 morphed_pos = position;
    vec3 morphed_nrm = normal;
    vec3 morphed_tan = tangent.xyz;
    applyMorphDeltas(morphed_pos, morphed_nrm, morphed_tan, gl_VertexIndex);

    ivec4 j = ivec4(joints);
    mat4 skin_mat = weights.x * bones[j.x] +
                    weights.y * bones[j.y] +
                    weights.z * bones[j.z] +
                    weights.w * bones[j.w];

    vec4 skinned_pos = skin_mat * vec4(morphed_pos, 1.0);
    vec3 skinned_norm = mat3(skin_mat) * morphed_nrm;
    vec3 skinned_tan = mat3(skin_mat) * morphed_tan;

    vec4 world_pos = model * skinned_pos;
    v_world_pos = world_pos.xyz;
    gl_Position = mvp * skinned_pos;

    vec3 N = normalize(mat3(model) * skinned_norm);
    vec3 T = normalize(mat3(model) * skinned_tan);
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
layout(binding = 2) uniform fs_params {
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
    // APPENDED LAST (clearcoat/sheen): coat + fabric lobes (scalar/color
    // plus optional R-mask / rgb-tint maps in pbr-layers v1, bindings 15/16).
    // Intensity 0 disables the lobe:
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
    // APPENDED LAST (pbr-layers v1): anisotropy / transmission / SSS.
    // All zeroed when unused: every new term gates on its factor (0 = off),
    // so legacy materials shade bit-identically. Appended last so no
    // existing offset shifts.
    vec4 anisotropy_factors; // x: intensity (0 = isotropic/off), y: tangent-plane rotation (rad), z/w: unused
    vec4 transmission_factors; // x: factor (0 = off), y/z/w: unused
    vec4 transmission_color; // rgb: throughput tint (white = untinted), w: unused
    vec4 sss_factors; // x: strength (0 = off), y/z/w: unused
    vec4 sss_color; // rgb: scatter tint (white = untinted), w: unused
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
// Binding 10: vs uses texture 9 (morph), fs uses 0..8 — next free slot in
// the shared pool. Sampled through the shared shadow_smp compare sampler.
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
// slot in the shared pool (fs uses 0..8 and 10, vs uses 9 for morph),
// binding 6 the next free sampler slot (fs uses 0..3 and 5, vs uses 4).
// The draw always binds something valid here (probe cube or default cube);
// the shader only samples it when probe_params.x > 0.5.
layout(binding = 11) uniform textureCube probe_tex;
layout(binding = 6) uniform sampler probe_smp;
// PBR layers v1: coat/fabric masks sampled through data_smp (each texture
// keeps its own filter/wrap, like the other data slots — no new sampler).
// Unset slots bind the default white texture: mask 1 / tint 1 = identity,
// so the scalar path stays bit-identical. Bindings 15/16 are free view
// slots in the shared pool (fs uses 0..8, 10, 11; vs uses 9 for morph;
// 12..14 are storage buffers, a separate array).
layout(binding = 15) uniform texture2D clearcoat_tex;
layout(binding = 16) uniform texture2D sheen_tex;
// Clustered forward lights (wave 30, v1): storage buffers for the tile
// walk. Bindings 12..14 are free view slots in the shared pool of every
// forward shader (this family uses fs 0..8, 10, 11 and vs 9 for morph).
// Layouts mirror scene/clustered_lights.zig (ClusterLightGpu = 2x vec4,
// ClusterTileGpu = uvec2, indices = u32); every buffer block holds exactly
// one flexible array of a struct (sokol-shdc requirement). The draw always
// binds something valid here (real tile buffers when a GPU upload landed,
// the shared 16-byte dummy otherwise); the loop below only reads them when
// clustered_params gates it on.
struct ClusterLight {
    vec4 pos_range; // xyz: position, w: radius
    vec4 color_int; // rgb: color, a: intensity
};
layout(std430, binding = 12) readonly buffer ssbo_cluster_lights {
    ClusterLight cluster_lights[];
};
struct ClusterTile {
    uvec2 head; // x: index-list offset, y: light count
};
layout(std430, binding = 13) readonly buffer ssbo_cluster_tiles {
    ClusterTile cluster_tiles[];
};
struct ClusterIndex {
    uint light;
};
layout(std430, binding = 14) readonly buffer ssbo_cluster_indices {
    ClusterIndex cluster_indices[];
};

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

// Anisotropy v1 (Heitz-style GGX): stretches the base-lobe NDF along the
// tangent frame; intensity 0 early-returns to the legacy isotropic NDF
// (bit-identical), geometry G stays isotropic and IBL stays isotropic
// (v1 scope, documented). CPU mirror: material.anisotropyAxes.
float anisoNDF(vec3 N, vec3 T, vec3 B, vec3 H, float roughness) {
    float aniso = clamp(anisotropy_factors.x, 0.0, 1.0);
    if (aniso <= 0.0) return distributionGGX(N, H, roughness);
    float rough2 = roughness * roughness;
    float aspect = sqrt(max(1.0 - 0.9 * aniso, 0.01));
    float ax = max(rough2 / aspect, 0.001);
    float ay = max(rough2 * aspect, 0.001);
    float TH = dot(T, H);
    float BH = dot(B, H);
    float NH = max(dot(N, H), 0.0);
    float denom = (TH * TH) / (ax * ax) + (BH * BH) / (ay * ay) + NH * NH;
    return 1.0 / max(PI * ax * ay * denom * denom, 0.0000001);
}

// Combined coat + sheen add-on for one punctual light. The caller multiplies
// `additive` by radiance * NdotL * shadow and scales its own BASE specular
// by `base_atten` (diffuse/kD untouched). Safe at NdotL = 0: the 0.0001
// guard matches the base lobe and the caller zeroes the contribution.
void coatSheenLight(vec3 N, vec3 V, vec3 L, vec3 H, float NdotV, float NdotL,
    float cc_rough, float cc_intensity, vec3 cc_F0,
    float sheen_rough, float sheen_intensity, vec3 sheen_tint,
    out vec3 base_atten, out vec3 additive) {
    float cc_NDF = distributionGGX(N, H, cc_rough);
    float cc_G = geometrySmith(N, V, L, cc_rough);
    vec3 cc_F = fresnelSchlick(clamp(dot(H, V), 0.0, 1.0), cc_F0) * cc_intensity;
    base_atten = vec3(1.0) - cc_F;
    vec3 cc_spec = (cc_NDF * cc_G * cc_F) / (4.0 * NdotV * NdotL + 0.0001);
    float sheenD = sheenDistributionCharlie(sheen_rough, max(dot(N, H), 0.0));
    float sheenV = sheenVisibilityNeubelt(NdotV, NdotL);
    vec3 sheen_term = sheen_tint * (sheenD * sheenV) * sheen_intensity;
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

    // PBR layers v1: coat/fabric masks sampled with the ALBEDO uv transform
    // (no per-slot coat transform yet — v1 scope). White fallback = 1, so
    // unset slots keep the scalar path exact; the branch only skips the
    // fetch when the lobe is off (mask * 0 == 0 either way).
    vec2 coat_uv = uvApply(uv_matrix[0], uv_offset[0], v_uv);
    float cc_mask = 1.0;
    if (cc_intensity > 0.0) {
        cc_mask = texture(sampler2D(clearcoat_tex, data_smp), coat_uv).r;
    }
    cc_intensity *= cc_mask;
    vec3 sheen_tint = sheen_color.rgb;
    if (sheen_intensity > 0.0) {
        sheen_tint *= texture(sampler2D(sheen_tex, data_smp), coat_uv).rgb;
    }

    // Anisotropy v1 frame: the existing TBN varyings (vertex tangent
    // attribute, Gram-Schmidt orthogonalized in vs) — NO new attribute.
    // Meshes without authored tangents carry the loader/builder fallback
    // (+X, see loader/mesh_spawn.zig): documented approximation.
    vec3 aniso_T = normalize(v_tangent);
    vec3 aniso_B = normalize(v_bitangent);
    float aniso_rot = anisotropy_factors.y;
    if (clamp(anisotropy_factors.x, 0.0, 1.0) > 0.0 && aniso_rot != 0.0) {
        float aniso_cr = cos(aniso_rot);
        float aniso_sr = sin(aniso_rot);
        vec3 aniso_rT = aniso_T * aniso_cr + aniso_B * aniso_sr;
        aniso_B = aniso_B * aniso_cr - aniso_T * aniso_sr;
        aniso_T = aniso_rT;
    }

    // Transmission v1: thin-slab approx (no refraction target — non-goal).
    // Diffuse throughput scales by (1 - factor) AFTER F0 so metals keep
    // their F0; the additive back-light term lands in the post-Lo block.
    // factor 0 skips: legacy albedo bit-identical.
    float transm_factor = clamp(transmission_factors.x, 0.0, 1.0);
    if (transm_factor > 0.0) {
        albedo *= (1.0 - transm_factor);
    }
    float sss_strength = clamp(sss_factors.x, 0.0, 1.0);

    // 1. Primary Directional Light (with shadow mapping)
    vec3 L = light_dir.xyz;
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);

    float NDF = anisoNDF(N, aniso_T, aniso_B, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 specular = (NDF * G * F) / (4.0 * NdotV * NdotL + 0.0001);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

    // Coat + sheen add-on for the sun (base_atten scales the BASE specular
    // only — diffuse passes through unattenuated).
    vec3 sun_atten;
    vec3 sun_additive;
    coatSheenLight(N, V, L, H, NdotV, NdotL, cc_rough, cc_intensity, cc_F0, sheen_rough, sheen_intensity, sheen_tint, sun_atten, sun_additive);

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
        float d_NDF = anisoNDF(N, aniso_T, aniso_B, d_H, roughness);
        float d_G = geometrySmith(N, V, d_dir, roughness);
        vec3 d_F = fresnelSchlick(max(dot(d_H, V), 0.0), F0);
        vec3 d_spec = (d_NDF * d_G * d_F) / (4.0 * NdotV * d_NdotL + 0.0001);
        vec3 d_kD = (vec3(1.0) - d_F) * (1.0 - metallic);
        vec3 d_rad = d_col * d_int;
        vec3 d_atten;
        vec3 d_additive;
        coatSheenLight(N, V, d_dir, d_H, NdotV, d_NdotL, cc_rough, cc_intensity, cc_F0, sheen_rough, sheen_intensity, sheen_tint, d_atten, d_additive);
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

        float d_norm = dist / p_range;
        float factor = clamp(1.0 - d_norm * d_norm * d_norm * d_norm, 0.0, 1.0);
        float att = (factor * factor) / (dist * dist + 1.0);

        float p_NdotL = max(dot(N, p_L), 0.0);
        if (p_NdotL > 0.0) {
            float p_NDF = anisoNDF(N, aniso_T, aniso_B, p_H, roughness);
            float p_G = geometrySmith(N, V, p_L, roughness);
            vec3 p_F = fresnelSchlick(max(dot(p_H, V), 0.0), F0);

            vec3 p_spec = (p_NDF * p_G * p_F) / (4.0 * NdotV * p_NdotL + 0.0001);
            vec3 p_kD = (vec3(1.0) - p_F) * (1.0 - metallic);
            vec3 p_rad = p_col * (p_int * att);
            vec3 p_atten;
            vec3 p_additive;
            coatSheenLight(N, V, p_L, p_H, NdotV, p_NdotL, cc_rough, cc_intensity, cc_F0, sheen_rough, sheen_intensity, sheen_tint, p_atten, p_additive);

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

        float d_norm = dist / s_range;
        float factor = clamp(1.0 - d_norm * d_norm * d_norm * d_norm, 0.0, 1.0);
        float dist_att = (factor * factor) / (dist * dist + 1.0);

        float cos_angle = dot(-s_L, s_dir);
        float cone_att = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001), 0.0, 1.0);
        cone_att *= cone_att;

        float total_att = dist_att * cone_att;
        if (total_att <= 0.0) continue;

        float s_NdotL = max(dot(N, s_L), 0.0);
        if (s_NdotL > 0.0) {
            float s_NDF = anisoNDF(N, aniso_T, aniso_B, s_H, roughness);
            float s_G = geometrySmith(N, V, s_L, roughness);
            vec3 s_F = fresnelSchlick(max(dot(s_H, V), 0.0), F0);

            vec3 s_spec = (s_NDF * s_G * s_F) / (4.0 * NdotV * s_NdotL + 0.0001);
            vec3 s_kD = (vec3(1.0) - s_F) * (1.0 - metallic);
            vec3 s_rad = s_col * (s_int * total_att);
            vec3 s_atten;
            vec3 s_additive;
            coatSheenLight(N, V, s_L, s_H, NdotV, s_NdotL, cc_rough, cc_intensity, cc_F0, sheen_rough, sheen_intensity, sheen_tint, s_atten, s_additive);

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
        float a_NDF = anisoNDF(N, aniso_T, aniso_B, a_H, roughness);
        float a_G = geometrySmith(N, V, a_L, roughness);
        vec3 a_F = fresnelSchlick(max(dot(a_H, V), 0.0), F0);
        vec3 a_spec = (a_NDF * a_G * a_F) / (4.0 * NdotV * a_NdotL + 0.0001);
        vec3 a_kD = (vec3(1.0) - a_F) * (1.0 - metallic);
        vec3 a_rad = area_color[i].rgb * a_factor;
        vec3 a_atten;
        vec3 a_additive;
        coatSheenLight(N, V, a_L, a_H, NdotV, a_NdotL, cc_rough, cc_intensity, cc_F0, sheen_rough, sheen_intensity, sheen_tint, a_atten, a_additive);
        Lo += (a_kD * albedo / PI + a_spec * a_atten + a_additive) * a_rad * a_NdotL;
    }

    // Clustered forward point lights (up to 64, no shadows): pixel -> tile
    // -> tile light indices -> the same Cook-Torrance lobe as the legacy
    // lanes above (coat + sheen included, no shadow term). Gated on a live
    // tile grid (tiles_x/y > 0), a non-empty staged pool (count > 0) and a
    // landed GPU upload (w > 0.5); otherwise skipped entirely, so the empty
    // pool shades bit-identically. Tile id follows the CPU build
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
            vec3 c_H = normalize(V + c_L);

            float c_d_norm = c_dist / c_range;
            float c_factor = clamp(1.0 - c_d_norm * c_d_norm * c_d_norm * c_d_norm, 0.0, 1.0);
            float c_att = (c_factor * c_factor) / (c_dist * c_dist + 1.0);

            float c_NdotL = max(dot(N, c_L), 0.0);
            if (c_NdotL > 0.0) {
                float c_NDF = anisoNDF(N, aniso_T, aniso_B, c_H, roughness);
                float c_G = geometrySmith(N, V, c_L, roughness);
                vec3 c_F = fresnelSchlick(max(dot(c_H, V), 0.0), F0);
                vec3 c_spec = (c_NDF * c_G * c_F) / (4.0 * NdotV * c_NdotL + 0.0001);
                vec3 c_kD = (vec3(1.0) - c_F) * (1.0 - metallic);
                vec3 c_rad = c_col * (c_int * c_att);
                vec3 c_atten;
                vec3 c_additive;
                coatSheenLight(N, V, c_L, c_H, NdotV, c_NdotL, cc_rough, cc_intensity, cc_F0, sheen_rough, sheen_intensity, sheen_tint, c_atten, c_additive);
                Lo += (c_kD * albedo / PI + c_spec * c_atten + c_additive) * c_rad * c_NdotL;
            }
        }
    }

    // Transmission v1 + SSS v1 post terms (sun + ambient driven; point /
    // spot / area / clustered punctuals do NOT contribute — v1 scope).
    // Both gate on a uniform branch: 0 adds exactly nothing bit-identical.
    if (transm_factor > 0.0) {
        float transm_back = clamp(dot(-N, L) * 0.5 + 0.5, 0.0, 1.0);
        vec3 transm_irr = light_color.rgb * light_color.a * transm_back + ambient_color.rgb * ambient_color.a * 0.5;
        Lo += transmission_color.rgb * transm_factor * albedo * transm_irr;
    }
    if (sss_strength > 0.0) {
        // Wrap mirror: material.wrapNdotL (unit-tested on CPU).
        float sss_wrap = sss_strength * 0.5;
        float wrap_nl = clamp((dot(N, L) + sss_wrap) / (1.0 + sss_wrap), 0.0, 1.0);
        float back_scatter = pow(clamp(dot(V, -L), 0.0, 1.0), 2.0);
        vec3 sss_irr = light_color.rgb * light_color.a * (wrap_nl * 0.6 + back_scatter * 0.4) + ambient_color.rgb * ambient_color.a * 0.25;
        Lo += sss_color.rgb * sss_strength * albedo * sss_irr;
    }

    // Ambient Occlusion
    float ao_sample = channelSelect(texture(sampler2D(occlusion_tex, data_smp), uvApply(uv_matrix[4], uv_offset[4], v_uv)), channel_selectors.x);
    float ao = 1.0 + pbr_factors.z * (ao_sample - 1.0);

    // Image-Based Lighting (IBL)
    // Reflection probe (wave 25): when a probe applies to this object
    // (probe_params.x > 0.5), the probe cube replaces env_tex as the IBL
    // source — specular via textureLod with roughness, diffuse via the
    // coarsest mip — scaled by the probe intensity. The hemispheric
    // ambient_color term below is unchanged. Disabled: legacy path,
    // bit-identical.
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
        vec3 sheen_ibl = sheen_tint * sheen_intensity * irradiance * sheen_grazing;

        ibl = (diffuse_ibl + specular_ibl) * (ibl_intensity * ao) + sheen_ibl * (ibl_intensity * ao);
    }

    vec3 ambient = ambient_color.rgb * ambient_color.a * albedo * ao;

    vec4 emissive_sample = texture(sampler2D(emissive_tex, data_smp), uvApply(uv_matrix[3], uv_offset[3], v_uv));
    vec3 emissive = emissive_factor.rgb * emissive_sample.rgb;

    vec3 final_color = ambient + ibl + Lo + emissive + debug_tint;

    frag_color = vec4(final_color, albedo_rgba.a);
}
@end

@program skinned_pbr vs fs
