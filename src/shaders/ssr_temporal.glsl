// Screen-space reflections (v2.1) temporal accumulation for agate.
//
// Runs between the half-res trace and the bilateral denoise
// (trace -> temporal -> denoise -> composite) when
// ssr.temporal.enabled rides on half_res_pass mode. History holds the
// accumulated (reflection rgb, blend weight) signal with NO intensity baked
// in, so the bilateral denoise and composite work identically.
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
// @include "common/fullscreen_vs.glsl"
@end

@fs fs
// @include "common/screen_uv.glsl"
layout(binding = 0) uniform fs_params {
    mat4 inv_view_proj;
    mat4 view_proj;
    mat4 prev_view_proj;
    vec4 resolution; // xy: half-res size, zw: half-res texel
    vec4 full_resolution; // xy: full effective size, zw: full texel
    vec4 camera_params; // x: near, y: far, zw: unused
    vec4 temporal_params; // x: history_valid (1/0), y: velocity_valid (1/0), z: blend [0,1], w: gamma
};

layout(binding = 0) uniform texture2D trace_tex;
layout(binding = 1) uniform texture2D history_tex;
@image_sample_type depth_tex unfilterable_float
layout(binding = 2) uniform texture2D depth_tex;
layout(binding = 3) uniform texture2D velocity_tex;
layout(binding = 0) uniform sampler smp;
@sampler_type depth_smp nonfiltering
layout(binding = 1) uniform sampler depth_smp;
@sampler_type velocity_smp nonfiltering
layout(binding = 2) uniform sampler velocity_smp;

in vec2 v_uv;
out vec4 frag_color;

const float SSR_TEMPORAL_GAMMA_DEFAULT = 1.0;
const float SSR_TEMPORAL_NORMAL_MIN_DOT = 0.9;
const float SSR_TEMPORAL_MARKER_LO = 0.15;
const float SSR_TEMPORAL_MARKER_HI = 0.35;
const float SSR_TEMPORAL_DEPTH_GATE = 0.05;

float linearizeDepth(float d) {
    float near = camera_params.x;
    float far = camera_params.y;
    return (near * far) / max(far - d * (far - near), 0.0001);
}

vec3 reconstructWorldPos(vec2 uv, float depth) {
    vec4 clip = vec4(agateUvToNdc(uv), depth, 1.0);
    vec4 world = inv_view_proj * clip;
    return world.xyz / world.w;
}

vec3 reconstructWorldNormal(vec2 uv, vec3 world_pos) {
    vec2 texel = full_resolution.zw;
    float d_r = texture(sampler2D(depth_tex, depth_smp), uv + vec2(texel.x, 0.0)).r;
    float d_l = texture(sampler2D(depth_tex, depth_smp), uv - vec2(texel.x, 0.0)).r;
    float d_u = texture(sampler2D(depth_tex, depth_smp), uv - vec2(0.0, texel.y)).r;
    float d_d = texture(sampler2D(depth_tex, depth_smp), uv + vec2(0.0, texel.y)).r;

    vec3 p_r = reconstructWorldPos(uv + vec2(texel.x, 0.0), d_r);
    vec3 p_l = reconstructWorldPos(uv - vec2(texel.x, 0.0), d_l);
    vec3 p_u = reconstructWorldPos(uv - vec2(0.0, texel.y), d_u);
    vec3 p_d = reconstructWorldPos(uv + vec2(0.0, texel.y), d_d);

    vec3 view_forward = vec3(view_proj[0][2], view_proj[1][2], view_proj[2][2]);
    float z_c = dot(world_pos, view_forward);
    float z_r = dot(p_r, view_forward);
    float z_l = dot(p_l, view_forward);
    float z_u = dot(p_u, view_forward);
    float z_d = dot(p_d, view_forward);

    vec3 dx = abs(z_r - z_c) < abs(z_l - z_c) ? (p_r - world_pos) : (world_pos - p_l);
    vec3 dy = abs(z_d - z_c) < abs(z_u - z_c) ? (p_d - world_pos) : (world_pos - p_u);

    vec3 n = cross(dx, dy);
    float len = length(n);
    return len > 1e-5 ? (n / len) : vec3(0.0, 1.0, 0.0);
}

void main() {
    float raw_depth = texture(sampler2D(depth_tex, depth_smp), v_uv).r;
    vec4 current = texture(sampler2D(trace_tex, smp), v_uv);
    if (raw_depth >= 1.0) {
        frag_color = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }
    float lin_depth = linearizeDepth(raw_depth);

    if (temporal_params.x < 0.5) {
        frag_color = current;
        return;
    }

    vec4 vel_sample = vec4(0.0);
    if (temporal_params.y > 0.5) {
        vel_sample = texture(sampler2D(velocity_tex, velocity_smp), v_uv);
    }
    vec2 prev_uv;
    if (vel_sample.a > 0.5) {
        prev_uv = v_uv - vel_sample.xy;
    } else {
        vec4 clip = vec4(agateUvToNdc(v_uv), raw_depth, 1.0);
        vec4 prev_clip = prev_view_proj * (inv_view_proj * clip);
        if (prev_clip.w <= 0.0001) {
            frag_color = current;
            return;
        }
        vec2 prev_ndc = prev_clip.xy / prev_clip.w;
        prev_uv = agateNdcToUv(prev_ndc.xy);
    }
    if (prev_uv.x < 0.001 || prev_uv.x > 0.999 || prev_uv.y < 0.001 || prev_uv.y > 0.999) {
        frag_color = current;
        return;
    }

    vec4 hist = texture(sampler2D(history_tex, smp), prev_uv);

    // Variance clipping 3x3 over trace_tex
    vec2 texel = resolution.zw;
    vec4 box_min = current;
    vec4 box_max = current;
    vec4 sum = current;
    vec4 sum_sq = current * current;
    vec4 t;
    t = texture(sampler2D(trace_tex, smp), v_uv + vec2(-texel.x, -texel.y)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = texture(sampler2D(trace_tex, smp), v_uv + vec2(0.0, -texel.y)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = texture(sampler2D(trace_tex, smp), v_uv + vec2(texel.x, -texel.y)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = texture(sampler2D(trace_tex, smp), v_uv + vec2(-texel.x, 0.0)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = texture(sampler2D(trace_tex, smp), v_uv + vec2(texel.x, 0.0)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = texture(sampler2D(trace_tex, smp), v_uv + vec2(-texel.x, texel.y)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = texture(sampler2D(trace_tex, smp), v_uv + vec2(0.0, texel.y)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = texture(sampler2D(trace_tex, smp), v_uv + vec2(texel.x, texel.y)); box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;

    vec4 avg = sum / 9.0;
    vec4 sigma = sqrt(max(sum_sq / 9.0 - avg * avg, vec4(0.0)));
    float gamma = (temporal_params.w > 0.0) ? temporal_params.w : SSR_TEMPORAL_GAMMA_DEFAULT;
    box_min = max(box_min, avg - gamma * sigma);
    box_max = min(box_max, avg + gamma * sigma);

    vec4 span = max(box_max - box_min, vec4(1e-4));
    vec4 dist_min = max(box_min - hist, vec4(0.0)) / span;
    vec4 dist_max = max(hist - box_max, vec4(0.0)) / span;
    float max_dist = max(max(dist_min.x, dist_max.x), max(max(dist_min.y, dist_max.y), max(max(dist_min.z, dist_max.z), max(dist_min.w, dist_max.w))));
    float accept = 1.0 / (1.0 + max_dist * max_dist * 4.0);

    // Depth consistency against current scene surface
    float prev_raw_d = texture(sampler2D(depth_tex, depth_smp), prev_uv).r;
    if (prev_raw_d >= 0.9999) {
        accept = 0.0;
    } else {
        float prev_lin_d = linearizeDepth(prev_raw_d);
        if (abs(lin_depth - prev_lin_d) >= SSR_TEMPORAL_DEPTH_GATE) accept = 0.0;
    }

    // Normal consistency
    vec3 world_pos = reconstructWorldPos(v_uv, raw_depth);
    vec3 n_cur = reconstructWorldNormal(v_uv, world_pos);
    if (prev_raw_d < 0.9999) {
        vec3 prev_world = reconstructWorldPos(prev_uv, prev_raw_d);
        vec3 n_prev = reconstructWorldNormal(prev_uv, prev_world);
        if (dot(n_cur, n_prev) < SSR_TEMPORAL_NORMAL_MIN_DOT) accept = 0.0;
    }

    // Reactive marker check
    bool is_marker = (vel_sample.a > SSR_TEMPORAL_MARKER_LO && vel_sample.a < SSR_TEMPORAL_MARKER_HI);
    if (is_marker) accept = 0.0;

    vec4 hist_clamped = clamp(hist, box_min, box_max);
    float blend = clamp(temporal_params.z, 0.0, 1.0) * accept;
    frag_color = mix(current, hist_clamped, blend);
}
@end

@program ssr_temporal vs fs
