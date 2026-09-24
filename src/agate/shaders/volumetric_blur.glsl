// Volumetric bilateral blur for agate (shaft layer v1).
// One axis per draw (params.x selects horizontal vs vertical); the parent
// runs it twice (H then V). Each tap multiplies the 9-tap spatial Gaussian
// (same kernel as glow_blur, sigma floored at 0.5) by a raw-depth gate:
// taps whose scene depth differs from the center pixel's depth by more
// than a few edge sigmas contribute ~0, so shafts never bleed across
// silhouettes. The depth is the full-resolution scene (or MSAA-prepass)
// depth sampled at the shaft UV. Mirrors shaftBilateralWeight in
// postprocess.zig, which pins the exact formula for CPU tests.
@header const m = @import("math")
@ctype mat4 m.Mat4

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
    vec4 texel; // xy: texel size of the shaft target, zw: unused
    vec4 params; // x: direction (0 = horizontal, 1 = vertical), y: spatial sigma, z: edge sigma (raw depth), w: unused
};

layout(binding = 0) uniform texture2D src_tex;
@image_sample_type depth_tex unfilterable_float
layout(binding = 1) uniform texture2D depth_tex;
layout(binding = 0) uniform sampler smp;
@sampler_type depth_smp nonfiltering
layout(binding = 1) uniform sampler depth_smp;

in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec2 step_uv = (params.x < 0.5) ? vec2(texel.x, 0.0) : vec2(0.0, texel.y);
    float sigma = max(params.y, 0.5);
    float edge = params.z;
    float center_depth = texture(sampler2D(depth_tex, depth_smp), v_uv).r;
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    for (int i = -4; i <= 4; i++) {
        float fi = float(i);
        vec2 tap_uv = v_uv + step_uv * fi;
        float w = exp(-0.5 * (fi / sigma) * (fi / sigma));
        // Depth gate: edge <= 0 disables it (plain Gaussian, same as the
        // glow blur); otherwise taps across a depth step fade out.
        if (edge > 0.0) {
            float tap_depth = texture(sampler2D(depth_tex, depth_smp), tap_uv).r;
            float dd = (tap_depth - center_depth) / edge;
            w *= exp(-0.5 * dd * dd);
        }
        acc += texture(sampler2D(src_tex, smp), tap_uv).rgb * w;
        wsum += w;
    }
    frag_color = vec4(acc / max(wsum, 0.0001), 1.0);
}
@end

@program volumetric_blur vs fs
