// Bloom Upsample / Composite Shader for agate.
// Tent-filtered (3x3, weights 1-2-1 / 16) upscale of the coarser mip,
// added onto the finer mip. Intensity scaling stays in the final
// postprocess composite so this pass always uses blend = 1.0.
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
    vec4 texel; // xy: texel size of the low-res (coarse) mip, zw: unused
    vec4 params; // x: blend factor (1.0), yzw: unused
};

layout(binding = 0) uniform texture2D high_tex;
layout(binding = 1) uniform texture2D low_tex;
layout(binding = 0) uniform sampler smp;

in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec2 t = texel.xy;

    // 3x3 tent upscale of the coarse level.
    vec3 low = texture(sampler2D(low_tex, smp), v_uv).rgb * 4.0;
    low += texture(sampler2D(low_tex, smp), v_uv + vec2(-t.x,  0.0)).rgb * 2.0;
    low += texture(sampler2D(low_tex, smp), v_uv + vec2( t.x,  0.0)).rgb * 2.0;
    low += texture(sampler2D(low_tex, smp), v_uv + vec2( 0.0, -t.y)).rgb * 2.0;
    low += texture(sampler2D(low_tex, smp), v_uv + vec2( 0.0,  t.y)).rgb * 2.0;
    low += texture(sampler2D(low_tex, smp), v_uv + vec2(-t.x, -t.y)).rgb;
    low += texture(sampler2D(low_tex, smp), v_uv + vec2( t.x, -t.y)).rgb;
    low += texture(sampler2D(low_tex, smp), v_uv + vec2(-t.x,  t.y)).rgb;
    low += texture(sampler2D(low_tex, smp), v_uv + vec2( t.x,  t.y)).rgb;
    low /= 16.0;

    vec3 high = texture(sampler2D(high_tex, smp), v_uv).rgb;
    frag_color = vec4(high + low * params.x, 1.0);
}
@end

@program bloom_up vs fs
