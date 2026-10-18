// Screen-space one-bounce diffuse GI (v2 gather, half resolution).
// Ports the v1 composite gather (postprocess.glsl applySSGI) into a
// dedicated pass: the SAME jittered disk gather, cosine x falloff^2
// weights, per-sample luma cap, coverage normalization, and
// camera-distance-scaled disk. Differences from the inline version:
//
// - Runs at half of the EFFECTIVE (render-scale) size; `resolution` still
//   carries the FULL effective size so the world-space disk footprint is
//   identical to v1 (2..48 full-res px clamp unchanged).
// - Tap budget is compile-time bounded: the loop runs to
//   SSGI_STEPS_MAX (32) with an early break at
//   the configured steps, mirroring the volumetric raymarch pattern.
// - Jitter is the same IGN hash on gl_FragCoord (now half-res pixels).
// - Output is normalized bleed * coverage with NO intensity baked in
//   (CPU mirror: ssgiBleedAdd with intensity 1); the composite scales by
//   intensity after the bilateral denoise, so intensity stays live-tunable.
// - Sky / zero-coverage pixels output exactly zero (the composite adds).
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
// @include "common/fullscreen_vs.glsl"
@end

@fs fs
layout(binding = 0) uniform fs_params {
    mat4 inv_view_proj; // camera inverse view-projection (jittered when TAA runs)
    mat4 view_proj; // camera view-projection (fourth row = view-depth axis)
    vec4 camera_pos; // xyz: camera world pos, w: unused
    vec4 resolution; // xy: FULL effective size, zw: full texel (v1-identical disk)
    vec4 ssgi_params; // x: enabled (1/0), y: intensity (unused here), z: radius (m), w: gather steps
};

layout(binding = 0) uniform texture2D scene_tex;
@image_sample_type depth_tex unfilterable_float
layout(binding = 1) uniform texture2D depth_tex;
layout(binding = 0) uniform sampler smp;
@sampler_type depth_smp nonfiltering
layout(binding = 1) uniform sampler depth_smp;

in vec2 v_uv;
out vec4 frag_color;

// SSGI tap budget: the loop bound below. Mirrors SSGI_STEPS_MAX in
// postprocess/ssgi.zig, so a clamped config never overruns the loop.
const int SSGI_TAPS_MAX = 32;
// Per-sample HDR firefly cap (CPU mirror: ssgi.SSGI_LUMA_CAP, v1 value).
const float SSGI_LUMA_CAP = 8.0;
// Coverage gain and kernel falloff power (CPU mirrors:
// ssgi.SSGI_COVERAGE_GAIN / ssgi.SSGI_FALLOFF_POWER, v1-identical math).
const float SSGI_COVERAGE_GAIN = 1.5;
const float SSGI_FALLOFF_POWER = 3.0;

vec3 reconstructWorldPos(vec2 uv, float depth) {
    vec4 clip = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    vec4 world = inv_view_proj * clip;
    return world.xyz / world.w;
}

vec3 reconstructWorldNormal(vec2 uv, vec3 world_pos) {
    vec2 texel = resolution.zw;
    float d_r = texture(sampler2D(depth_tex, depth_smp), uv + vec2(texel.x, 0.0)).r;
    float d_l = texture(sampler2D(depth_tex, depth_smp), uv - vec2(texel.x, 0.0)).r;
    float d_u = texture(sampler2D(depth_tex, depth_smp), uv - vec2(0.0, texel.y)).r;
    float d_d = texture(sampler2D(depth_tex, depth_smp), uv + vec2(0.0, texel.y)).r;

    vec3 p_r = reconstructWorldPos(uv + vec2(texel.x, 0.0), d_r);
    vec3 p_l = reconstructWorldPos(uv - vec2(texel.x, 0.0), d_l);
    vec3 p_u = reconstructWorldPos(uv - vec2(0.0, texel.y), d_u);
    vec3 p_d = reconstructWorldPos(uv + vec2(0.0, texel.y), d_d);

    // Same continuous-side choice as the composite (view depth from the
    // perspective VP fourth row; CPU mirror: depthNormalFromNeighbors).
    vec3 view_forward = vec3(view_proj[0].w, view_proj[1].w, view_proj[2].w);
    vec3 dr = p_r - world_pos;
    vec3 dl = world_pos - p_l;
    vec3 du = p_u - world_pos;
    vec3 dd = world_pos - p_d;
    vec3 dx = (abs(dot(dr, view_forward)) < abs(dot(dl, view_forward))) ? dr : dl;
    vec3 dy = (abs(dot(du, view_forward)) < abs(dot(dd, view_forward))) ? du : dd;
    vec3 n = cross(dx, dy);
    float len_sq = dot(n, n);
    // Same relative degeneracy test as the composite (CPU mirror:
    // depthNormalFromGradients): invalid normals contribute nothing.
    float scale = dot(dx, dx) * dot(dy, dy);
    return (len_sq > max(scale * 1e-8, 1e-30)) ? (n * inversesqrt(len_sq)) : vec3(0.0);
}

// Receiver trust in [0, 1]: silhouette pixels have a garbage derivative
// normal (the closer-side pick jumps across the silhouette), so the
// cosine gate leaks bright taps and paints the white contact ring. Per
// axis the jump shows up as one side's view-depth delta dwarfing the
// other; edge factor ramps in over a 3x..9x ratio. Same math as the
// composite (common/ssr_trace.glsl ssgiNormalReliability; CPU mirror:
// postprocess/ssgi.zig ssgiNormalReliability).
float ssgiNormalReliability(vec2 uv, vec3 world_pos) {
    vec2 texel = resolution.zw;
    vec3 view_forward = vec3(view_proj[0].w, view_proj[1].w, view_proj[2].w);
    float d_r = texture(sampler2D(depth_tex, depth_smp), uv + vec2(texel.x, 0.0)).r;
    float d_l = texture(sampler2D(depth_tex, depth_smp), uv - vec2(texel.x, 0.0)).r;
    float d_u = texture(sampler2D(depth_tex, depth_smp), uv - vec2(0.0, texel.y)).r;
    float d_d = texture(sampler2D(depth_tex, depth_smp), uv + vec2(0.0, texel.y)).r;
    float f_r = abs(dot(reconstructWorldPos(uv + vec2(texel.x, 0.0), d_r) - world_pos, view_forward));
    float f_l = abs(dot(reconstructWorldPos(uv - vec2(texel.x, 0.0), d_l) - world_pos, view_forward));
    float f_u = abs(dot(reconstructWorldPos(uv - vec2(0.0, texel.y), d_u) - world_pos, view_forward));
    float f_d = abs(dot(reconstructWorldPos(uv + vec2(0.0, texel.y), d_d) - world_pos, view_forward));
    float edge_x = clamp((max(f_r, f_l) / (min(f_r, f_l) + 1e-6) - 3.0) / 6.0, 0.0, 1.0);
    float edge_y = clamp((max(f_u, f_d) / (min(f_u, f_d) + 1e-6) - 3.0) / 6.0, 0.0, 1.0);
    return 1.0 - max(edge_x, edge_y);
}

void main() {
    float raw_depth = texture(sampler2D(depth_tex, depth_smp), v_uv).r;
    if (raw_depth >= 0.9999) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    float radius = max(ssgi_params.z, 0.05);
    int steps = int(ssgi_params.w + 0.5);
    if (steps < 4) steps = 8;
    if (steps > SSGI_TAPS_MAX) steps = SSGI_TAPS_MAX;

    vec3 world_pos = reconstructWorldPos(v_uv, raw_depth);
    vec3 N = reconstructWorldNormal(v_uv, world_pos);
    // Silhouette trust (v1-identical): kill the gather on depth-edge
    // receivers where the derivative normal cannot be trusted.
    float reliability = ssgiNormalReliability(v_uv, world_pos);

    // Roughly constant WORLD radius: scale the pixel disk by camera
    // distance; clamp so the gather stays local (2..48 px of the FULL
    // effective size — v1-identical footprint).
    float cam_dist = max(length(world_pos - camera_pos.xyz), 0.05);
    float px_radius = clamp((radius / cam_dist) * resolution.y * 0.25, 2.0, 48.0);

    float jitter = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    vec3 bleed = vec3(0.0);
    float total_w = 0.0;
    for (int i = 0; i < SSGI_TAPS_MAX; i++) {
        if (i >= steps) break;
        float fi = float(i);
        float ang = 6.2831853 * (fi + jitter) / float(steps);
        float r = px_radius * sqrt((fi + 0.5) / float(steps));
        vec2 s_uv = v_uv + vec2(cos(ang), sin(ang)) * r * resolution.zw;
        if (s_uv.x <= 0.002 || s_uv.x >= 0.998 || s_uv.y <= 0.002 || s_uv.y >= 0.998) continue;

        float s_d = texture(sampler2D(depth_tex, depth_smp), s_uv).r;
        if (s_d >= 0.9999) continue;
        vec3 s_pos = reconstructWorldPos(s_uv, s_d);
        vec3 to_s = s_pos - world_pos;
        float dist = length(to_s);
        if (dist >= radius) continue;
        float cos_nd = dot(N, to_s / max(dist, 1e-4));
        if (cos_nd <= 0.0) continue;

        vec3 s_col = texture(sampler2D(scene_tex, smp), s_uv).rgb;
        float falloff = 1.0 - dist / radius;
        float w = cos_nd * falloff * falloff * falloff; // SSGI_FALLOFF_POWER
        bleed += min(s_col, vec3(SSGI_LUMA_CAP)) * w;
        total_w += w;
    }
    if (total_w <= 0.0001) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 bleed_color = bleed / total_w;
    float coverage = clamp(total_w / float(steps) * SSGI_COVERAGE_GAIN, 0.0, 1.0);
    frag_color = vec4(bleed_color * (coverage * reliability), 1.0);
}
@end

@program ssgi vs fs
