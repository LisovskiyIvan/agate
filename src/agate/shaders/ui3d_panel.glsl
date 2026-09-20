// World-space 3D GUI panel quad (wave 28, v1): unlit/emissive textured quad
// sampling the panel's private color render target. One MVP uniform, one
// RGBA8 sample; alpha blends over the scene (see scene/gui3d_layer.zig for
// the depth-write-off / double-sided / blend state on the pipeline side).
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 mvp;
};

in vec3 position;
in vec2 texcoord0;

out vec2 v_uv;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    v_uv = texcoord0;
}
@end

@fs fs
layout(binding = 0) uniform texture2D panel_tex;
layout(binding = 0) uniform sampler smp;

in vec2 v_uv;

out vec4 frag_color;

void main() {
    frag_color = texture(sampler2D(panel_tex, smp), v_uv);
}
@end

@program ui3d_panel vs fs
