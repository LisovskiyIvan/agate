// Screen-space reflections (v2) edge-preserving bilateral denoise for agate.
// Mirrors the ssao_blur/ssgi_blur structure: a 3x3 spatial Gaussian over
// the half-res trace output (reflection rgb + blend weight in alpha),
// gated by FULL-resolution scene depth so neighboring denoised texels
// never cross depth discontinuities. This is denoise only, NOT the final
// upscale: the fullscreen composite reconstructs full res with its own
// depth-aware 2x2 gather (sampleSSRPass in postprocess.glsl), so the
// silhouette gate holds at the final pixel too. Differences from
// ssgi_blur:
//
// - Input is RGBA: rgb = reflection color at the hit, a = blend weight
//   (edge * dist * fresnel, no intensity). rgb normalizes by the
//   reflection-confidence accumulator (sum rgb*a*w / sum a*w) while alpha
//   normalizes by the spatial/depth weight (sum a*w / sum w) — the same
//   representation the composite 2x2 gather (sampleSSRPass) consumes, so
//   a miss tap (0,0,0,0) dilutes coverage but never darkens a valid
//   reflection, and the composite mix applies the weight exactly once.
//   CPU mirror: ssrDenoiseResolve in postprocess/ssr.zig.
// - Fallback is ZERO weight (mix nothing), not 1.0: thin geometry missing
//   from the half-res buffer reflects nothing rather than borrowing a
//   reflection from an unrelated surface.
// - Depth gate is the fixed linearized threshold SSR_BLUR_DEPTH_GATE
//   (0.05 m, CPU mirror: ssr.SSR_BLUR_DEPTH_GATE): nearby surfaces
//   denoise together, silhouettes stay clean. Sky taps (raw depth >= 1.0)
//   never contribute.
// - Intensity is NOT applied here; the composite scales after this denoise.
@header const m = @import("math")

@vs vs
@glsl_options fixup_clipspace
// @include "common/fullscreen_vs.glsl"
@end

@fs fs
layout(binding = 0) uniform fs_params {
    vec4 resolution; // xy: half-res trace size, zw: trace texel size
    vec4 camera_params; // x: near, y: far, zw: unused
};

layout(binding = 0) uniform texture2D ssr_tex;
@image_sample_type depth_tex unfilterable_float
layout(binding = 1) uniform texture2D depth_tex;
@sampler_type smp nonfiltering
layout(binding = 0) uniform sampler smp;

in vec2 v_uv;
out vec4 frag_color;

float linearizeDepth(float d) {
    float near = camera_params.x;
    float far = camera_params.y;
    return (near * far) / max(far - d * (far - near), 0.0001);
}

void main() {
    vec2 texel = resolution.zw;
    float center_raw_d = texture(sampler2D(depth_tex, smp), v_uv).r;

    if (center_raw_d >= 1.0) {
        frag_color = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float center_lin_d = linearizeDepth(center_raw_d);
    float depth_threshold = 0.05;

    vec3 acc = vec3(0.0);
    float acc_a = 0.0;
    float wsum = 0.0;

    vec2 center_pixel = floor(v_uv * resolution.xy);
    vec2 depth_size = vec2(textureSize(sampler2D(depth_tex, smp), 0));
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            vec2 sample_uv = (center_pixel + vec2(x, y) + 0.5) * texel;
            if (any(lessThan(sample_uv, vec2(0.0))) || any(greaterThan(sample_uv, vec2(1.0)))) continue;

            vec2 depth_uv = (floor(sample_uv * depth_size) + 0.5) / depth_size;
            float sample_raw_d = texture(sampler2D(depth_tex, smp), depth_uv).r;
            if (sample_raw_d >= 1.0) continue;
            float sample_lin_d = linearizeDepth(sample_raw_d);

            float depth_diff = abs(center_lin_d - sample_lin_d);

            if (depth_diff < depth_threshold) {
                vec2 delta = (sample_uv - v_uv) * resolution.xy;
                float weight = exp(-dot(delta, delta) * 0.5) * (1.0 - depth_diff / depth_threshold);
                vec4 sample_ssr = texture(sampler2D(ssr_tex, smp), sample_uv);
                acc += sample_ssr.rgb * sample_ssr.a * weight;
                acc_a += sample_ssr.a * weight;
                wsum += weight;
            }
        }
    }

    // Confidence-normalized rgb (a miss tap carries no reflection energy),
    // coverage-normalized alpha; zero rgb with no confidence. Matches the
    // composite sampleSSRPass representation exactly.
    vec4 final_ssr = vec4(0.0);
    if (wsum > 0.0001) {
        float wa = acc_a / wsum;
        vec3 rgb = acc_a > 0.0001 ? acc / acc_a : vec3(0.0);
        final_ssr = vec4(rgb, wa);
    }
    frag_color = final_ssr;
}
@end

@program ssr_blur vs fs
