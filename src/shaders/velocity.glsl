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

// Reactive-marker writers (C.3 v2): particle coverage into the velocity
// buffer alpha for content that cannot carry velocity. The billboard math
// mirrors particle.glsl `vs` / `vs_gpu` exactly (same quad + instance
// layouts, same analytic integration) so the marker covers the exact pixels
// the main pass blended; the FS writes a constant band value (0.25) the TAA
// resolve classifies as reactive (see taaVelocityKind in postprocess/taa.zig).
// No texture sample: full-quad mark covers every blended pixel and can only
// over-mark transparent quad corners (bounded extra history shed, never a
// missed ghost). No loops: one quad per instance, zero tap cost.
@vs vs_marker_cpu
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_marker_params {
    mat4 view_proj;
    vec4 camera_right;
    vec4 camera_up;
};

// Quad corner (buffer 0) + CPU integrated instance record (buffer 1,
// ParticleInstanceData layout: only the center/size + rotation lanes are
// read; color/uv lanes ride along unconsumed in the stride).
in vec2 position;
in vec4 inst_pos_size;
in vec4 inst_rotation;

out vec4 v_clip;

void main() {
    float c = cos(inst_rotation.x);
    float s = sin(inst_rotation.x);
    vec2 rotated = vec2(c * position.x - s * position.y, s * position.x + c * position.y);
    vec3 world_pos = inst_pos_size.xyz +
        (camera_right.xyz * rotated.x + camera_up.xyz * rotated.y) * inst_pos_size.w;
    gl_Position = view_proj * vec4(world_pos, 1.0);
    v_clip = gl_Position;
}
@end

@vs vs_marker_gpu
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_marker_params {
    mat4 view_proj;
    vec4 camera_right;
    vec4 camera_up;
};
layout(binding = 1) uniform marker_gpu_params {
    // x = seconds since the system's epoch clock, y = exponential drag k
    // (1/s, 0 = off), zw unused. Mirrors particle.glsl gpu_params.
    vec4 time_drag;
    // xyz = world-space gravity acceleration, w unused.
    vec4 gravity;
    // x = spritesheet columns, y = rows, z = loops over lifetime, w unused
    // (frame math omitted here: the marker needs coverage only, and every
    // frame of one slot shares the same billboard center).
    vec4 sprite;
};

// Mirror of GpuParticleSlot (particles.zig), five FLOAT4 attributes.
in vec2 position;
in vec4 gpu_slot0;
in vec4 gpu_slot1;
in vec4 gpu_slot2;
in vec4 gpu_slot3;
in vec4 gpu_slot4;

out vec4 v_clip;

void main() {
    // Same analytic trajectory as particle.glsl vs_gpu (copy, not include:
    // this shader has no include lane): p = p0 + v0*s + g*s2.
    float age = time_drag.x - gpu_slot0.w;
    float t = age / max(gpu_slot1.w, 1.0e-4);
    if (t < 0.0 || t >= 1.0) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        v_clip = gl_Position;
        return;
    }
    float s;
    float s2;
    float k = time_drag.y;
    if (k > 1.0e-6) {
        s = (1.0 - exp(-k * age)) / k;
        s2 = (age - s) / k;
    } else {
        s = age;
        s2 = 0.5 * age * age;
    }
    vec3 center = gpu_slot0.xyz + gpu_slot1.xyz * s + gravity.xyz * s2;
    float size = mix(gpu_slot4.x, gpu_slot4.y, t);
    float rot = gpu_slot4.z + gpu_slot4.w * age;
    float c = cos(rot);
    float sn = sin(rot);
    vec2 rotated = vec2(c * position.x - sn * position.y, sn * position.x + c * position.y);
    vec3 world_pos = center + (camera_right.xyz * rotated.x + camera_up.xyz * rotated.y) * size;
    gl_Position = view_proj * vec4(world_pos, 1.0);
    v_clip = gl_Position;
}
@end

@fs fs_marker
in vec4 v_clip;

out vec4 frag_velocity;

void main() {
    // Behind-camera (or degenerate) projections carry no valid coverage:
    // report the zero mask so the composite falls back to depth
    // reprojection instead of marking mirrored garbage as reactive.
    if (v_clip.w <= 0.0001) {
        frag_velocity = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    // Reactive band value: must equal TAA_REACTIVE_MARKER_ALPHA in
    // postprocess/taa.zig (the resolve classifies 0.15..0.35 as reactive).
    frag_velocity = vec4(0.0, 0.0, 0.0, 0.25);
}
@end

@program reactive_marker vs_marker_cpu fs_marker
@program reactive_marker_gpu vs_marker_gpu fs_marker
