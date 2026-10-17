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
in vec2 texcoord1;

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
out vec2 v_uv1;

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
    v_uv1 = texcoord1;
}
@end

@fs fs
// Shadow atlas resolution. Must match ShadowPass.SHADOW_ATLAS_SIZE in
// passes/shadow_pass.zig (sokol-shdc --defines cannot carry a value).
#ifndef SHADOW_ATLAS_SIZE
#define SHADOW_ATLAS_SIZE 2048.0
#endif
layout(binding = 1) uniform fs_view {
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

@program instanced_pbr vs fs
