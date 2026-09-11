// Instanced Camera-Facing Billboard Particle Shader for agate
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 view_proj;
    vec4 camera_right;
    vec4 camera_up;
};

in vec2 position;
in vec2 texcoord0;
in vec4 inst_pos_size;
in vec4 inst_color;
// xy = UV offset of the spritesheet cell, zw = UV scale (1/columns, 1/rows).
in vec4 inst_uv_rect;
// x = billboard rotation in radians (counter-clockwise in billboard plane).
in vec4 inst_rotation;

out vec2 v_uv;
out vec4 v_color;

void main() {
    float c = cos(inst_rotation.x);
    float s = sin(inst_rotation.x);
    vec2 rotated = vec2(c * position.x - s * position.y, s * position.x + c * position.y);
    vec3 world_pos = inst_pos_size.xyz +
        (camera_right.xyz * rotated.x + camera_up.xyz * rotated.y) * inst_pos_size.w;
    gl_Position = view_proj * vec4(world_pos, 1.0);
    v_uv = inst_uv_rect.xy + texcoord0 * inst_uv_rect.zw;
    v_color = inst_color;
}
@end

@fs fs
layout(binding = 0) uniform texture2D particle_tex;
layout(binding = 0) uniform sampler smp;

in vec2 v_uv;
in vec4 v_color;

out vec4 frag_color;

void main() {
    vec4 tex = texture(sampler2D(particle_tex, smp), v_uv);
    frag_color = tex * v_color;
}
@end

@program particle vs fs
