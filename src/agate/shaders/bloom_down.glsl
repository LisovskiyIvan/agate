// Bloom Prefilter / Downsample Shader for agate.
// 13-tap Karis average: bright outliers get a 1/(1+luma) weight so single
// firefly pixels cannot dominate the downsampled average. The first pyramid
// level additionally applies the soft bright-pass threshold; deeper levels
// pass threshold < 0 to disable it.
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
    vec4 src_texel; // xy: texel size of the source level, zw: unused
    vec4 params; // x: bright-pass threshold (< 0 disables), yzw: unused
};

layout(binding = 0) uniform texture2D src_tex;
layout(binding = 0) uniform sampler smp;

in vec2 v_uv;
out vec4 frag_color;

float karisLuma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// One Karis-weighted tap: rgb holds color * weight, a holds the weight.
vec4 karisTap(vec2 uv) {
    vec3 c = texture(sampler2D(src_tex, smp), uv).rgb;
    float w = 1.0 / (1.0 + max(karisLuma(c), 0.0));
    return vec4(c * w, w);
}

void main() {
    vec2 o = src_texel.xy;
    // 3x3 core taps.
    vec4 acc = karisTap(v_uv + vec2(-o.x, -o.y));
    acc += karisTap(v_uv + vec2( 0.0, -o.y));
    acc += karisTap(v_uv + vec2( o.x, -o.y));
    acc += karisTap(v_uv + vec2(-o.x,  0.0));
    acc += karisTap(v_uv);
    acc += karisTap(v_uv + vec2( o.x,  0.0));
    acc += karisTap(v_uv + vec2(-o.x,  o.y));
    acc += karisTap(v_uv + vec2( 0.0,  o.y));
    acc += karisTap(v_uv + vec2( o.x,  o.y));
    // Wide axis taps widen the footprint to a 5x5-equivalent.
    acc += karisTap(v_uv + vec2(-2.0 * o.x, 0.0));
    acc += karisTap(v_uv + vec2( 2.0 * o.x, 0.0));
    acc += karisTap(v_uv + vec2(0.0, -2.0 * o.y));
    acc += karisTap(v_uv + vec2(0.0,  2.0 * o.y));

    vec3 c = acc.rgb / max(acc.a, 0.0001);

    // Soft bright-pass on the first level only.
    float thresh = params.x;
    if (thresh >= 0.0) {
        float luma = karisLuma(c);
        float factor = max(0.0, luma - thresh) / max(luma, 0.0001);
        c *= factor;
    }

    frag_color = vec4(c, 1.0);
}
@end

@program bloom_down vs fs
