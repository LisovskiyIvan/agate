// Glow Threshold-Extract Shader for agate (glow layer v1).
// Soft bright-pass of the resolved scene color into the half-resolution
// glow target. Mirrors extractBright in postprocess.glsl (and glowExtract
// in postprocess.zig, which pins the formula for CPU tests); the separable
// blur stages (glow_blur.glsl) widen it into the halo.
@header const m = @import("math")

@vs vs
@glsl_options fixup_clipspace
in vec2 position;
in vec2 texcoord0;

out vec2 v_uv;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    #if !SOKOL_GLSL
        v_uv = vec2(texcoord0.x, 1.0 - texcoord0.y);
    #else
        v_uv = texcoord0;
    #endif
}
@end

@fs fs
layout(binding = 0) uniform fs_params {
    vec4 src_texel; // xy: texel size of the source (scene) level, zw: unused
    vec4 params; // x: luminance threshold, yzw: unused
};

layout(binding = 0) uniform texture2D src_tex;
layout(binding = 0) uniform sampler smp;

in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec3 c = texture(sampler2D(src_tex, smp), v_uv).rgb;
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    float factor = max(0.0, luma - params.x) / max(luma, 0.0001);
    frag_color = vec4(c * factor, 1.0);
}
@end

@program glow_extract vs fs
