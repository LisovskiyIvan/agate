// GPU Hierarchical Depth Pyramid (Hi-Z) Downsampling Shader
@header const m = @import("math")

@vs vs
@glsl_options fixup_clipspace
// @include "common/fullscreen_vs.glsl"
@end

@fs fs
layout(binding = 0) uniform fs_params {
    vec4 src_resolution; // xy: source resolution, zw: texel size (1.0/width, 1.0/height)
};

@image_sample_type depth_tex unfilterable_float
layout(binding = 0) uniform texture2D depth_tex;
@sampler_type smp nonfiltering
layout(binding = 0) uniform sampler smp;

in vec2 v_uv;
out vec4 frag_color;

void main() {
    vec2 texel = src_resolution.zw;
    vec2 center = v_uv;

    // 2x2 conservative maximum depth reduction in [0, 1] depth range (LESS_EQUAL, clear 1.0).
    // The farthest point in the footprint gives the conservative occluder bound.
    vec2 uv00 = center + vec2(-0.5, -0.5) * texel;
    vec2 uv10 = center + vec2( 0.5, -0.5) * texel;
    vec2 uv01 = center + vec2(-0.5,  0.5) * texel;
    vec2 uv11 = center + vec2( 0.5,  0.5) * texel;

    float d00 = texture(sampler2D(depth_tex, smp), uv00).r;
    float d10 = texture(sampler2D(depth_tex, smp), uv10).r;
    float d01 = texture(sampler2D(depth_tex, smp), uv01).r;
    float d11 = texture(sampler2D(depth_tex, smp), uv11).r;

    float max_depth = max(max(d00, d10), max(d01, d11));

    // Handle odd source dimensions conservatively by sampling the edge row/column
    bool odd_x = (int(src_resolution.x) & 1) != 0;
    bool odd_y = (int(src_resolution.y) & 1) != 0;
    if (odd_x) {
        float d20 = texture(sampler2D(depth_tex, smp), center + vec2(1.5, -0.5) * texel).r;
        float d21 = texture(sampler2D(depth_tex, smp), center + vec2(1.5,  0.5) * texel).r;
        max_depth = max(max_depth, max(d20, d21));
    }
    if (odd_y) {
        float d02 = texture(sampler2D(depth_tex, smp), center + vec2(-0.5, 1.5) * texel).r;
        float d12 = texture(sampler2D(depth_tex, smp), center + vec2( 0.5, 1.5) * texel).r;
        max_depth = max(max_depth, max(d02, d12));
    }
    if (odd_x && odd_y) {
        float d22 = texture(sampler2D(depth_tex, smp), center + vec2(1.5, 1.5) * texel).r;
        max_depth = max(max_depth, d22);
    }

    frag_color = vec4(max_depth, max_depth, max_depth, 1.0);
}
@end

@program depth_pyramid vs fs
