// Inverse-hull outline/highlight shader for agate (renders inside the main
// pass, after opaque geometry, like DebugPass/SkyboxPass).
// - outline:         rigid meshes (view_proj * model pre-multiplied into mvp)
// - outline_inst:    instanced meshes (per-instance model matrix, buffer 1)
// - outline_skinned: skinned meshes (matrix-palette skinning, joints/weights)
// - outline_cutout:  alpha-cutout cards (foliage): the projected quad is
//   dilated away from its screen-space center and alpha-tested against the
//   albedo texture, so the halo follows the leaf silhouette, not the quad
//   border.
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


@vs vs_cutout
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 mvp; // view_proj * model
    mat4 model; // unused: kept so every stage shares one uniform layout
    vec4 color;
    vec4 params; // x: width_px, y: viewport_w, z: viewport_h, w: depth push-away bias (NDC)
};
layout(binding = 2) uniform vs_center {
    vec4 center_ndc; // xy: projected bounds center in NDC, z: unused, w: >0 when valid
};

in vec3 position;
in vec2 texcoord0;

out vec2 v_uv;
out vec4 v_color;

void main() {
    vec4 clip0 = mvp * vec4(position, 1.0);
    float w0 = max(clip0.w, 1e-6);
    vec2 vp = max(params.yz, vec2(1.0));
    vec2 px = params.x * 2.0 / vp;

    // Dilate the projected card away from its screen-space center. A flat
    // card has no interior silhouette for the inverse hull to grip, so the
    // expansion direction comes from the projected bounds center instead.
    vec2 dir = clip0.xy / w0 - center_ndc.xy;
    float len = max(length(dir), 1e-5);
    clip0.xy += (dir / len) * px * w0;

    // Push the dilated copy behind the card: the alpha-tested leaf wins the
    // depth test, so only the halo ring outside the leaf silhouette shows.
    clip0.z += params.w * w0;
    gl_Position = clip0;
    v_uv = texcoord0;
    v_color = color;
}
@end
@fs fs_cutout
layout(binding = 3) uniform fs_cutout_params {
    vec4 cutout; // x: alpha cutoff
};
layout(binding = 0) uniform texture2D albedo_tex;
layout(binding = 0) uniform sampler smp;

in vec2 v_uv;
in vec4 v_color;

out vec4 frag_color;

void main() {
    float a = texture(sampler2D(albedo_tex, smp), v_uv).a;
    if (a < cutout.x) discard;
    frag_color = v_color;
}
@end

@program outline_cutout vs_cutout fs_cutout