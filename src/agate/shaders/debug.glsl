// Physics debug-line overlay shader for agate (renders inside the main pass,
// on top of already-bound color+depth attachments, like SkyboxPass).
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 mvp;
};

in vec3 position;
in vec4 color0;

out vec4 v_color;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    v_color = color0;
}
@end

@fs fs
in vec4 v_color;

out vec4 frag_color;

void main() {
    frag_color = v_color;
}
@end

@program debug vs fs
