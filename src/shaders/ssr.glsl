// Screen-space reflections (half-res trace, v2).
// Ports the v1 composite raymarch (postprocess.glsl applySSR) into a
// dedicated pass at half of the EFFECTIVE (render-scale) size:
//
// - SAME math: floor-only gate (N.y >= 0.45), reflection gate (R.y >=
//   0.02), clip-space march, thickness 0.4, max distance 25.0, edge fade
//   x10, distance fade squared, fresnel ramp x1.8 clamped [0.15, 0.9].
// - Tap budget is compile-time bounded: the loop runs to SSR_TAPS_MAX
//   (64) with an early break at the configured steps, mirroring the SSGI
//   gather pattern.
// - `resolution` carries the FULL effective size so the march footprint
//   matches v1; the output UV is half-res.
// - Output is (reflection rgb, blend weight) with NO intensity baked in
//   (CPU mirror: ssrBlendWeight); the composite scales by intensity after
//   the bilateral denoise, so intensity stays live-tunable.
// - Miss/sky/gated pixels output exactly zero weight (the composite mixes
//   nothing).
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
// @include "common/fullscreen_vs.glsl"
@end

@fs fs
layout(binding = 0) uniform fs_params {
    mat4 view_proj;
    mat4 inv_view_proj;
    vec4 camera_pos; // xyz: camera world pos, w: unused
    vec4 resolution; // xy: FULL effective size, zw: full texel (v1-identical march)
    vec4 ssr_params; // x: enabled (1/0), y: intensity (unused here), z: thickness, w: max distance
    vec4 ssr_params2; // x: raymarch steps, yzw: spare
};

layout(binding = 0) uniform texture2D scene_tex;
@image_sample_type depth_tex unfilterable_float
layout(binding = 1) uniform texture2D depth_tex;
layout(binding = 0) uniform sampler smp;
@sampler_type depth_smp nonfiltering
layout(binding = 1) uniform sampler depth_smp;

in vec2 v_uv;
out vec4 frag_color;

// Raymarch tap budget: the loop bound below. Mirrors SSR_TAPS_MAX in
// postprocess/ssr.zig, so a clamped config never overruns the loop.
const int SSR_TAPS_MAX = 64;

// Depth-derived world position + surface normal, shared with the
// fullscreen composite (postprocess.glsl applySSR path).
// @include "common/ssr_trace.glsl"

void main() {
    float raw_depth = texture(sampler2D(depth_tex, depth_smp), v_uv).r;
    if (raw_depth >= 0.9999) {
        frag_color = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    vec3 world_pos = reconstructWorldPos(v_uv, raw_depth);
    vec3 V = normalize(world_pos - camera_pos.xyz);
    vec3 N = reconstructWorldNormal(v_uv, world_pos);

    if (N.y < 0.45) {
        frag_color = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    vec3 R = reflect(V, N);
    if (R.y < 0.02) {
        frag_color = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float max_dist = ssr_params.w;
    float thickness = ssr_params.z;
    int steps = int(ssr_params2.x + 0.5);
    if (steps < 4) steps = 16;
    if (steps > SSR_TAPS_MAX) steps = SSR_TAPS_MAX;
    float step_size = max_dist / float(steps);

    vec3 ray_start = world_pos + N * 0.08;
    vec3 ray_dir_step = R * step_size;

    vec4 clip_start = view_proj * vec4(ray_start, 1.0);
    vec4 clip_step = view_proj * vec4(ray_dir_step, 0.0);

    for (int i = 1; i <= SSR_TAPS_MAX; i++) {
        if (i > steps) break;
        vec4 march_clip = clip_start + clip_step * float(i);
        if (march_clip.w <= 0.0001) break;

        vec3 ndc = march_clip.xyz / march_clip.w;
        vec2 march_uv = vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (march_uv.x < 0.01 || march_uv.x > 0.99 || march_uv.y < 0.01 || march_uv.y > 0.99) {
            break;
        }

        float scene_d = texture(sampler2D(depth_tex, depth_smp), march_uv).r;
        if (scene_d >= 0.9999) continue;
        if (ndc.z < scene_d) continue;

        vec3 ray_pos = ray_start + ray_dir_step * float(i);
        vec3 scene_pos = reconstructWorldPos(march_uv, scene_d);

        float ray_cam_dist = length(ray_pos - camera_pos.xyz);
        float scene_cam_dist = length(scene_pos - camera_pos.xyz);
        float depth_diff = ray_cam_dist - scene_cam_dist;
        float dist_to_surface = length(ray_pos - scene_pos);

        if (depth_diff >= 0.0 && dist_to_surface < thickness) {
            float edge_dist_x = min(march_uv.x, 1.0 - march_uv.x);
            float edge_dist_y = min(march_uv.y, 1.0 - march_uv.y);
            float edge_fade = clamp(min(edge_dist_x, edge_dist_y) * 10.0, 0.0, 1.0);

            float dist_fade = 1.0 - (float(i - 1) / float(steps));
            dist_fade *= dist_fade;

            float fresnel = 0.04 + 0.96 * pow(1.0 - max(0.0, dot(-V, N)), 5.0);
            fresnel = clamp(fresnel * 1.8, 0.15, 0.9);

            vec3 refl_color = texture(sampler2D(scene_tex, smp), march_uv).rgb;
            float weight = edge_fade * dist_fade * fresnel;
            frag_color = vec4(refl_color, weight);
            return;
        }
    }

    frag_color = vec4(0.0, 0.0, 0.0, 0.0);
}
@end

@program ssr vs fs
