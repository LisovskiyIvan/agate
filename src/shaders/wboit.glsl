// Weighted Blended Order-Independent Transparency (WBOIT) forward PBR shader
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs_rigid
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 mvp;
    mat4 model;
};

layout(binding = 3) uniform vs_morph {
    vec4 morph_weights[16];
    vec4 morph_params;
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

    vec4 world_pos = model * vec4(morphed_pos, 1.0);
    v_world_pos = world_pos.xyz;
    gl_Position = mvp * vec4(morphed_pos, 1.0);

    vec3 N = normalize(mat3(model) * morphed_nrm);
    vec3 T = normalize(mat3(model) * morphed_tan);
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

@vs vs_inst
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_inst_params {
    mat4 view_proj;
};

in vec3 position;
in vec3 normal;
in vec4 tangent;
in vec4 color0;
in vec2 texcoord0;
in vec2 texcoord1;

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

@vs vs_skin
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 mvp;
    mat4 model;
};
layout(binding = 1) uniform vs_skin {
    mat4 bones[64];
};

layout(binding = 3) uniform vs_morph {
    vec4 morph_weights[16];
    vec4 morph_params;
};

layout(binding = 9) uniform texture2D morph_tex;
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
    vec4 world_pos = model * skinned_pos;
    v_world_pos = world_pos.xyz;
    gl_Position = mvp * skinned_pos;

    mat3 normal_matrix = mat3(model) * mat3(skin_mat);
    vec3 N = normalize(normal_matrix * morphed_nrm);
    vec3 T = normalize(normal_matrix * morphed_tan);
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

@fs fs_wboit
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

layout(location = 0) out vec4 frag_accum;
layout(location = 1) out vec4 frag_reveal;

// @include "common/refraction.glsl"
// @include "common/pbr_shadow_consts.glsl"
// @include "common/shadow_pcf.glsl"
// @include "common/pbr_brdf.glsl"
// @include "common/uv_apply.glsl"
// @include "common/channel_select.glsl"
// @include "common/hemi.glsl"
// @include "common/hemi_pbr.glsl"
// @include "common/specular_aa.glsl"
// @include "common/linear_output.glsl"
// @include "common/box_project.glsl"
// @include "common/pbr_surface.glsl"
// @include "common/wboit.glsl"

void main() {
    vec4 albedo_sample = texture(sampler2D(albedo_tex, smp), uvApply(uv_matrix[0], uv_offset[0], v_uv));
    vec4 albedo_rgba = v_color * base_color_factor * albedo_sample;
    if (albedo_rgba.a < alpha_cutoff) discard;

    vec3 final_color;
    if (uv_offset[0].z > 0.5) {
        vec4 emissive_sample = texture(sampler2D(emissive_tex, data_smp), uvApply(uv_matrix[3], uv_offset[3], v_uv));
        vec3 emissive = emissive_factor.rgb * emissive_sample.rgb;
        final_color = albedo_rgba.rgb + emissive;
    } else {
        final_color = pbrSurfaceLighting(albedo_rgba, normalize(v_normal));
    }

    float dist = length(v_world_pos - eye_pos.xyz);
    wboitOutput(final_color, albedo_rgba.a, dist, frag_accum, frag_reveal);
}
@end

@program wboit_rigid vs_rigid fs_wboit
@program wboit_inst vs_inst fs_wboit
@program wboit_skin vs_skin fs_wboit
