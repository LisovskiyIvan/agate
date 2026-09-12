// Inverse-hull outline/highlight shader for agate (renders inside the main
// pass, after opaque geometry, like DebugPass/SkyboxPass).
// Three vertex stages share the same rim expansion and fragment stage:
// - outline:         rigid meshes (view_proj * model pre-multiplied into mvp)
// - outline_inst:    instanced meshes (per-instance model matrix, buffer 1)
// - outline_skinned: skinned meshes (matrix-palette skinning, joints/weights)
// The rim is a constant screen-space width by construction: the normal tip
// is projected next to the base vertex, the NDC delta is renormalized in
// pixel space, and the offset is scaled by clip depth, so the width does
// not shrink with camera distance. Back faces of the inflated hull are
// drawn (front faces culled); fragments covered by the original surface
// fail the LESS_EQUAL depth test, leaving only the silhouette rim.
// Non-uniform scale skews the silhouette direction slightly (documented
// limit, same as the rigid path).
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 mvp;
    mat4 model;
    vec4 color; // rgb: outline color, a: opacity (pipeline blends it)
    vec4 params; // x: width_px (clamped CPU-side), y: viewport_w, z: viewport_h, w: depth_bias (NDC)
};

in vec3 position;
in vec3 normal;

out vec4 v_color;

void main() {
    vec4 clip0 = mvp * vec4(position, 1.0);
    vec4 probe = mvp * vec4(position + normal, 1.0);

    // Renormalize in pixel space so the width is uniform on non-square
    // viewports, then convert back to NDC. Multiplying by clip depth keeps
    // the width constant in screen space at any camera distance.
    vec2 vp = max(params.yz, vec2(1.0));
    float w0 = max(clip0.w, 1e-6);
    float w1 = max(probe.w, 1e-6);
    vec2 dir_ndc = probe.xy / w1 - clip0.xy / w0;
    vec2 dir_px = dir_ndc * vp * 0.5;
    float len_px = max(length(dir_px), 1e-4);
    vec2 offset_ndc = dir_px / len_px * (params.x * 2.0 / vp);

    vec4 clip = clip0;
    clip.xy += offset_ndc * w0;
    clip.z -= params.w * w0;
    gl_Position = clip;
    v_color = color;
}
@end

@vs vs_inst
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 mvp; // view_proj (the model lives in the per-instance matrix)
    mat4 model; // unused: kept so all three stages share one uniform layout
    vec4 color;
    vec4 params;
};

in vec3 position;
in vec3 normal;
in vec4 inst_mat0;
in vec4 inst_mat1;
in vec4 inst_mat2;
in vec4 inst_mat3;

out vec4 v_color;

void main() {
    mat4 inst_model = mat4(inst_mat0, inst_mat1, inst_mat2, inst_mat3);
    vec4 world_pos = inst_model * vec4(position, 1.0);
    vec3 world_normal = mat3(inst_model) * normal;
    vec4 clip0 = mvp * world_pos;
    vec4 probe = mvp * (world_pos + vec4(world_normal, 0.0));

    vec2 vp = max(params.yz, vec2(1.0));
    float w0 = max(clip0.w, 1e-6);
    float w1 = max(probe.w, 1e-6);
    vec2 dir_ndc = probe.xy / w1 - clip0.xy / w0;
    vec2 dir_px = dir_ndc * vp * 0.5;
    float len_px = max(length(dir_px), 1e-4);
    vec2 offset_ndc = dir_px / len_px * (params.x * 2.0 / vp);

    vec4 clip = clip0;
    clip.xy += offset_ndc * w0;
    clip.z -= params.w * w0;
    gl_Position = clip;
    v_color = color;
}
@end

@vs vs_skinned
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 mvp; // view_proj * model (skin happens in mesh-local space)
    mat4 model; // unused: kept so all three stages share one uniform layout
    vec4 color;
    vec4 params;
};
layout(binding = 1) uniform vs_skin {
    mat4 bones[64]; // must match Skeleton.MAX_BONES and skinned_pbr.glsl
};

in vec3 position;
in vec3 normal;
in vec4 joints;
in vec4 weights;

out vec4 v_color;

void main() {
    ivec4 j = ivec4(joints);
    mat4 skin_mat = weights.x * bones[j.x] +
                    weights.y * bones[j.y] +
                    weights.z * bones[j.z] +
                    weights.w * bones[j.w];
    vec4 skinned_pos = skin_mat * vec4(position, 1.0);
    vec3 skinned_normal = mat3(skin_mat) * normal;
    vec4 clip0 = mvp * skinned_pos;
    vec4 probe = mvp * (skinned_pos + vec4(skinned_normal, 0.0));

    vec2 vp = max(params.yz, vec2(1.0));
    float w0 = max(clip0.w, 1e-6);
    float w1 = max(probe.w, 1e-6);
    vec2 dir_ndc = probe.xy / w1 - clip0.xy / w0;
    vec2 dir_px = dir_ndc * vp * 0.5;
    float len_px = max(length(dir_px), 1e-4);
    vec2 offset_ndc = dir_px / len_px * (params.x * 2.0 / vp);

    vec4 clip = clip0;
    clip.xy += offset_ndc * w0;
    clip.z -= params.w * w0;
    gl_Position = clip;
    v_color = color;
}
@end

@fs fs
in vec4 v_color;

out vec4 frag_color;

void main() {
    frag_color = v_color;
}
@end

@program outline vs fs
@program outline_inst vs_inst fs
@program outline_skinned vs_skinned fs
