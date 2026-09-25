// Screen-Space Ambient Occlusion (SSAO) Edge-Preserving Bilateral Blur Shader
@header const m = @import("math")

@vs vs
@glsl_options fixup_clipspace
// @include "common/fullscreen_vs.glsl"
@end

@fs fs
layout(binding = 0) uniform fs_params {
    vec4 resolution; // xy: raw AO resolution, zw: raw AO texel size
    vec4 camera_params; // x: near, y: far, z: AO radius
};

layout(binding = 0) uniform texture2D ssao_tex;
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
        frag_color = vec4(1.0, 1.0, 1.0, 1.0);
        return;
    }

    float center_lin_d = linearizeDepth(center_raw_d);
    // Do not let distant foreground/background edges blend over metres.
    float depth_threshold = max(0.001, camera_params.z * 0.2);

    float result = 0.0;
    float total_weight = 0.0;

    vec2 center_pixel = floor(v_uv * resolution.xy);
    vec2 depth_size = vec2(textureSize(sampler2D(depth_tex, smp), 0));
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            // AO and depth must refer to the same low-resolution pixel centre.
            vec2 sample_uv = (center_pixel + vec2(x, y) + 0.5) * texel;
            if (any(lessThan(sample_uv, vec2(0.0))) || any(greaterThan(sample_uv, vec2(1.0)))) continue;

            vec2 depth_uv = (floor(sample_uv * depth_size) + 0.5) / depth_size;
            float sample_raw_d = texture(sampler2D(depth_tex, smp), depth_uv).r;
            if (sample_raw_d >= 1.0) continue;
            float sample_lin_d = linearizeDepth(sample_raw_d);

            float depth_diff = abs(center_lin_d - sample_lin_d);

            if (depth_diff < depth_threshold) {
                // Bilateral spatial Gaussian (sigma ~ 1.5)
                vec2 delta = (sample_uv - v_uv) * resolution.xy;
                float weight = exp(-dot(delta, delta) * 0.5) * (1.0 - depth_diff / depth_threshold);
                float sample_ao = texture(sampler2D(ssao_tex, smp), sample_uv).r;
                result += sample_ao * weight;
                total_weight += weight;
            }
        }
    }

    // Thin geometry may not exist in the half-resolution buffer: keep it lit
    // rather than borrowing occlusion from an unrelated surface.
    float final_ao = total_weight > 0.0001 ? result / total_weight : 1.0;
    frag_color = vec4(final_ao, final_ao, final_ao, 1.0);
}
@end

@program ssao_blur vs fs
