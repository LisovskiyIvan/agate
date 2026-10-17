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
// slot pool): UB 3 (vs_params 0, vs_skin 1, fs_view 2, fs_mat 5), texture 9
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
in vec2 texcoord1;
in vec4 joints;
in vec4 weights;

out vec3 v_world_pos;
out vec3 v_normal;
out vec3 v_tangent;
out vec3 v_bitangent;
out vec4 v_color;
out vec2 v_uv;
out vec2 v_uv1;

// @include "common/morph_helpers.glsl"

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
    v_uv1 = texcoord1;
}
@end

@fs fs
// Shadow atlas resolution. Must match ShadowPass.SHADOW_ATLAS_SIZE in
// passes/shadow_pass.zig (sokol-shdc --defines cannot carry a value).
#ifndef SHADOW_ATLAS_SIZE
#define SHADOW_ATLAS_SIZE 2048.0
#endif
layout(binding = 2) uniform fs_view {
// @include "common/frame_view.glsl"
};
layout(binding = 5) uniform fs_mat {
// @include "common/frame_mat.glsl"
};

// @include "common/pbr_fs_bindings.glsl"
// @include "common/cluster.glsl"

in vec3 v_world_pos;
in vec3 v_normal;
in vec3 v_tangent;
in vec3 v_bitangent;
in vec4 v_color;
in vec2 v_uv;
in vec2 v_uv1;

out vec4 frag_color;
// @include "common/refraction.glsl"

// @include "common/pbr_shadow_consts.glsl"

// @include "common/shadow_pcf.glsl"

// @include "common/pbr_brdf.glsl"

// KHR_texture_transform: uv' = matrix * uv + offset. Identity uniforms make
// this a no-op, so materials without the extension sample unchanged.
// @include "common/uv_apply.glsl"

// Manual channel selection (no glTF counterpart; see Channel in
// material.zig): picks one RGBA lane by uniform index through constant
// branches — SPIRV-Cross cannot flatten dynamic component indexing for
// HLSL5 targets, same constraint as morphWeight.
// @include "common/channel_select.glsl"
// @include "common/hemi.glsl"
// @include "common/hemi_pbr.glsl"
// @include "common/ddgi_sample.glsl"
// @include "common/specular_aa.glsl"
// @include "common/linear_output.glsl"
// @include "common/box_project.glsl"

// @include "common/pbr_surface.glsl"

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
        frag_color = linearOutputColor(albedo_rgba.rgb + emissive, albedo_rgba.a);
        return;
    }
    frag_color = linearOutputColor(pbrSurfaceLighting(albedo_rgba, normalize(v_normal)), albedo_rgba.a);

}
@end

@program skinned_pbr vs fs
