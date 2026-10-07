// Velocity buffer shader for rigid, instanced, and skinned geometry.
// Computes screen-space velocity vectors (uv_curr - uv_prev) into an RGBA16F buffer.
// Alpha-cutoff discard mirrors the main PBR path exactly (see pbr.glsl):
// alpha = v_color.a * base_color_factor.a * albedo_texel.a, discard iff
// alpha < alpha_cutoff. Cutout holes therefore never emit mask1; the
// shared EQUAL main-depth match is the second gate, not the only one.
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs_rigid
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 cur_mvp;
    mat4 prev_mvp;
};

in vec3 position;
in vec4 color0;
in vec2 texcoord0;
in vec2 texcoord1;

out vec4 v_cur_clip;
out vec4 v_prev_clip;
out vec4 v_color;
out vec2 v_uv;
out vec2 v_uv1;

void main() {
    vec4 pos = vec4(position, 1.0);
    v_cur_clip = cur_mvp * pos;
    v_prev_clip = prev_mvp * pos;
    v_color = color0;
    v_uv = texcoord0;
    v_uv1 = texcoord1;
    gl_Position = v_cur_clip;
}
@end

@vs vs_inst
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_inst_params {
    mat4 cur_view_proj;
    mat4 prev_view_proj;
};

in vec3 position;
in vec4 inst_mat0;
in vec4 inst_mat1;
in vec4 inst_mat2;
in vec4 inst_mat3;
in vec4 prev_inst_mat0;
in vec4 prev_inst_mat1;
in vec4 prev_inst_mat2;
in vec4 prev_inst_mat3;
in vec4 color0;
in vec2 texcoord0;
in vec2 texcoord1;

out vec4 v_cur_clip;
out vec4 v_prev_clip;
out vec4 v_color;
out vec2 v_uv;
out vec2 v_uv1;

void main() {
    mat4 cur_model = mat4(inst_mat0, inst_mat1, inst_mat2, inst_mat3);
    mat4 prev_model = mat4(prev_inst_mat0, prev_inst_mat1, prev_inst_mat2, prev_inst_mat3);
    vec4 pos = vec4(position, 1.0);
    v_cur_clip = cur_view_proj * (cur_model * pos);
    v_prev_clip = prev_view_proj * (prev_model * pos);
    v_color = color0;
    v_uv = texcoord0;
    v_uv1 = texcoord1;
    gl_Position = v_cur_clip;
}
@end

@vs vs_skinned
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 cur_mvp;
    mat4 prev_mvp;
};
layout(binding = 1) uniform vs_skin {
    mat4 bones[64];
};
layout(binding = 2) uniform vs_prev_skin {
    mat4 prev_bones[64];
};

in vec3 position;
in vec4 joints;
in vec4 weights;
in vec4 color0;
in vec2 texcoord0;
in vec2 texcoord1;

out vec4 v_cur_clip;
out vec4 v_prev_clip;
out vec4 v_color;
out vec2 v_uv;
out vec2 v_uv1;

void main() {
    ivec4 j = ivec4(joints);
    mat4 cur_skin = weights.x * bones[j.x] +
                    weights.y * bones[j.y] +
                    weights.z * bones[j.z] +
                    weights.w * bones[j.w];
    vec4 cur_pos = cur_skin * vec4(position, 1.0);
    v_cur_clip = cur_mvp * cur_pos;

    mat4 prev_skin = weights.x * prev_bones[j.x] +
                     weights.y * prev_bones[j.y] +
                     weights.z * prev_bones[j.z] +
                     weights.w * prev_bones[j.w];
    vec4 prev_pos = prev_skin * vec4(position, 1.0);
    v_prev_clip = prev_mvp * prev_pos;

    v_color = color0;
    v_uv = texcoord0;
    v_uv1 = texcoord1;
    gl_Position = v_cur_clip;
}
@end

@fs fs_velocity
layout(binding = 3) uniform fs_alpha {
    vec4 base_color_factor;
    vec4 uv_matrix;
    vec4 uv_offset;
    float alpha_cutoff;
};
layout(binding = 0) uniform texture2D albedo_tex;
layout(binding = 0) uniform sampler albedo_smp;

in vec4 v_cur_clip;
in vec4 v_prev_clip;
in vec4 v_color;
in vec2 v_uv;
in vec2 v_uv1;

out vec4 frag_velocity;

// Bit-identical to common/uv_apply.glsl uvApply (inlined: the velocity
// programs have no include lane): o.w selects the UV set, o.z (unlit)
// never alters the result.
vec2 velUvApply(vec4 m, vec4 o, vec2 uv) {
    vec2 pick = uv;
    if (o.w > 0.5) pick = v_uv1;
    return vec2(m.x * pick.x + m.y * pick.y + o.x, m.z * pick.x + m.w * pick.y + o.y);
}

void main() {
    // Behind-camera (or degenerate) projections carry no valid history:
    // report the zero mask so the composite falls back to depth
    // reprojection instead of a mirrored/absurd vector. Sign is preserved
    // (no abs-clamp): a flipped w must not masquerade as motion.
    if (v_cur_clip.w <= 0.0001 || v_prev_clip.w <= 0.0001) {
        frag_velocity = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    // Alpha test identical to the main PBR path (opaque/cutout materials
    // only ever reach this pass; hook materials are skipped upstream).
    // cutoff <= 0.0 keeps the no-discard branch: opaque materials upload
    // 0.0, so their fragments always survive exactly like the main pass.
    if (alpha_cutoff > 0.0) {
        float tex_alpha = texture(sampler2D(albedo_tex, albedo_smp), velUvApply(uv_matrix, uv_offset, v_uv)).a;
        float frag_alpha = v_color.a * base_color_factor.a * tex_alpha;
        if (frag_alpha < alpha_cutoff) discard;
    }
    vec2 cur_ndc = v_cur_clip.xy / v_cur_clip.w;
    vec2 prev_ndc = v_prev_clip.xy / v_prev_clip.w;
    vec2 cur_uv = vec2(cur_ndc.x * 0.5 + 0.5, 0.5 - cur_ndc.y * 0.5);
    vec2 prev_uv = vec2(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);
    frag_velocity = vec4(cur_uv - prev_uv, 0.0, 1.0);
}
@end

@program velocity vs_rigid fs_velocity
@program velocity_instanced vs_inst fs_velocity
@program velocity_skinned vs_skinned fs_velocity
