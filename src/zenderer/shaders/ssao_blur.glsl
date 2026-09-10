// Screen-Space Ambient Occlusion (SSAO) Edge-Preserving Bilateral Blur Shader
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
    vec4 resolution; // xy: resolution, zw: texel size
};

layout(binding = 0) uniform texture2D ssao_tex;
@image_sample_type depth_tex unfilterable_float
layout(binding = 1) uniform texture2D depth_tex;
@sampler_type smp nonfiltering
layout(binding = 0) uniform sampler smp;

in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec2 texel = resolution.zw;
    float center_depth = texture(sampler2D(depth_tex, smp), v_uv).r;

    if (center_depth >= 0.9999) {
        frag_color = vec4(1.0, 1.0, 1.0, 1.0);
        return;
    }

    float result = 0.0;
    float total_weight = 0.0;

    for (int x = -2; x <= 2; ++x) {
        for (int y = -2; y <= 2; ++y) {
            vec2 offset = vec2(float(x), float(y)) * texel;
            vec2 sample_uv = v_uv + offset;

            float sample_depth = texture(sampler2D(depth_tex, smp), sample_uv).r;
            float depth_diff = abs(center_depth - sample_depth);
            float weight = exp(-depth_diff * 350.0);

            float sample_ao = texture(sampler2D(ssao_tex, smp), sample_uv).r;
            result += sample_ao * weight;
            total_weight += weight;
        }
    }

    float final_ao = result / max(total_weight, 0.0001);
    frag_color = vec4(final_ao, final_ao, final_ao, 1.0);
}
@end

@program ssao_blur vs fs
