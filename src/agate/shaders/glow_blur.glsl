// Glow Separable Gaussian Blur Shader for agate (glow layer v1).
// One axis per draw (params.x selects horizontal vs vertical); the parent
// runs it twice (H then V) for the full 9x9-equivalent halo. The 9-tap
// 1D Gaussian (offsets -4..4, sigma = glow_radius in glow-target texels,
// floored at 0.5) is wider than the bloom pyramid's tent upsample, which is
// what makes the glow halo read as a distinct effect. Mirrors
// glowGaussianWeight/glowKernelSum in postprocess.zig, which pin the exact
// formula (raw weights normalized by the 9-tap sum) for CPU tests.
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
    vec4 texel; // xy: texel size of the glow target, zw: unused
    vec4 params; // x: direction (0 = horizontal, 1 = vertical), y: sigma, zw: unused
};

layout(binding = 0) uniform texture2D src_tex;
layout(binding = 0) uniform sampler smp;

in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec2 step_uv = (params.x < 0.5) ? vec2(texel.x, 0.0) : vec2(0.0, texel.y);
    float sigma = max(params.y, 0.5);
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    for (int i = -4; i <= 4; i++) {
        float fi = float(i);
        float w = exp(-0.5 * (fi / sigma) * (fi / sigma));
        acc += texture(sampler2D(src_tex, smp), v_uv + step_uv * fi).rgb * w;
        wsum += w;
    }
    frag_color = vec4(acc / max(wsum, 0.0001), 1.0);
}
@end

@program glow_blur vs fs
