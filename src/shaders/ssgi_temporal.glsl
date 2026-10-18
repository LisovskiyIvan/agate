// Screen-space GI (v2.1) temporal accumulation for agate.
//
// Runs between the half-res gather and the bilateral denoise
// (gather -> temporal -> denoise -> composite) when
// ssgi_temporal_enabled rides on half_res_pass. History holds the RAW
// accumulated signal (normalized bleed * coverage, no intensity — the same
// lane the gather writes), so the denoise stays the final cleaner before
// the composite scales by intensity.
//
// Per pixel (half-res UV space, shared with the full-res buffers):
//  1. Current = gather tap; without valid history (first frame, toggle,
//     resize, camera cut — the parent passes history_valid = 0) the output
//     is the current gather plus its linear depth, priming history.
//  2. Reproject: per-object velocity (uv_curr - uv_prev in the buffer, so
//     prev = uv - vel) when the velocity view is live; camera-only depth
//     reprojection (prev_view_proj * inv_view_proj) otherwise.
//  3. Reject: variance divergence of history against the current 3x3
//     gather neighborhood (mean +/- gamma * sigma, same 1/(1+4d^2)
//     falloff as the TAA resolve), depth consistency against the
//     history-stored linear depth (the bilateral gate max(0.001,
//     radius * 0.2)), normal consistency (reconstructed house
//     depth-normal at both ends, cosine >= 0.9), and the reactive-marker
//     band (velocity alpha 0.15..0.35 = particles/glass -> zero history).
//  4. Blend: mix(current, variance-clipped history, blend * accept) with
//     full-strength clipping; sky pixels output exactly zero.
//  5. Alpha carries the CURRENT linear depth (meters) for the next frame's
//     depth gate, even on fully rejected pixels.
//
// CPU mirror: postprocess/ssgi.zig ssgiTemporal* (blend, gates, resolve).
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
// @include "common/fullscreen_vs.glsl"
@end

@fs fs
// @include "common/screen_uv.glsl"
layout(binding = 0) uniform fs_params {
    mat4 inv_view_proj; // camera inverse view-projection (jittered when TAA runs)
    mat4 view_proj; // camera view-projection (fourth row = view-depth axis)
    mat4 prev_view_proj; // previous frame view-projection (camera fallback path)
    vec4 resolution; // xy: half-res temporal size, zw: half-res texel
    vec4 full_resolution; // xy: FULL effective size, zw: full texel (normal reconstruction)
    vec4 camera_params; // x: near, y: far, z: SSGI radius (m), w: unused
    vec4 temporal_params; // x: history_valid (1/0), y: velocity_valid (1/0), z: blend [0,1], w: gamma
};

layout(binding = 0) uniform texture2D gather_tex;
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

// Variance-clip tightness default (CPU mirror:
// ssgi.SSGI_TEMPORAL_GAMMA); the parent uploads the same value.
const float SSGI_TEMPORAL_GAMMA_DEFAULT = 1.0;
// Normal-consistency cosine (CPU mirror: SSGI_TEMPORAL_NORMAL_MIN_DOT).
const float SSGI_TEMPORAL_NORMAL_MIN_DOT = 0.9;
// Reactive-marker band over the velocity alpha (CPU mirror:
// SSGI_TEMPORAL_MARKER_LO/HI; velocity.glsl fs_marker stamps 0.25).
const float SSGI_TEMPORAL_MARKER_LO = 0.15;
const float SSGI_TEMPORAL_MARKER_HI = 0.35;

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

    // Same continuous-side choice as the gather/composite (view depth from
    // the perspective VP fourth row; CPU mirror: depthNormalFromNeighbors).
    vec3 view_forward = vec3(view_proj[0].w, view_proj[1].w, view_proj[2].w);
    vec3 dr = p_r - world_pos;
    vec3 dl = world_pos - p_l;
    vec3 du = p_u - world_pos;
    vec3 dd = world_pos - p_d;
    vec3 dx = (abs(dot(dr, view_forward)) < abs(dot(dl, view_forward))) ? dr : dl;
    vec3 dy = (abs(dot(du, view_forward)) < abs(dot(dd, view_forward))) ? du : dd;
    vec3 n = cross(dx, dy);
    float len_sq = dot(n, n);
    float scale = dot(dx, dx) * dot(dy, dy);
    return (len_sq > max(scale * 1e-8, 1e-30)) ? (n * inversesqrt(len_sq)) : vec3(0.0);
}

void main() {
    vec3 current = texture(sampler2D(gather_tex, smp), v_uv).rgb;
    float raw_depth = texture(sampler2D(depth_tex, depth_smp), v_uv).r;
    float lin_depth = (raw_depth >= 1.0) ? 1e9 : linearizeDepth(raw_depth);

    // Sky (or missing depth) carries no bleed and no reprojectable surface:
    // zero output, like the gather and the denoise.
    if (raw_depth >= 0.9999) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // No valid history yet: pass the current gather through and prime the
    // depth lane for the next frame.
    if (temporal_params.x < 0.5) {
        frag_color = vec4(current, lin_depth);
        return;
    }

    // Reproject into history UV space.
    vec4 vel_sample = vec4(0.0, 0.0, 0.0, 0.0);
    if (temporal_params.y > 0.5) {
        vel_sample = texture(sampler2D(velocity_tex, velocity_smp), v_uv);
    }
    vec2 prev_uv;
    if (vel_sample.a > 0.5) {
        prev_uv = v_uv - vel_sample.xy;
    } else {
        vec4 clip = vec4(v_uv.x * 2.0 - 1.0, 1.0 - v_uv.y * 2.0, raw_depth, 1.0);
        vec4 prev_clip = prev_view_proj * (inv_view_proj * clip);
        if (prev_clip.w <= 0.0001) {
            frag_color = vec4(current, lin_depth);
            return;
        }
        vec2 prev_ndc = prev_clip.xy / prev_clip.w;
        prev_uv = agateNdcToUv(prev_ndc.xy);
    }
    if (prev_uv.x < 0.001 || prev_uv.x > 0.999 || prev_uv.y < 0.001 || prev_uv.y > 0.999) {
        frag_color = vec4(current, lin_depth);
        return;
    }

    vec4 hist_sample = texture(sampler2D(history_tex, smp), prev_uv);
    vec3 hist = hist_sample.rgb;

    // Current 3x3 gather neighborhood: variance box.
    vec2 texel = resolution.zw;
    vec3 box_min = current;
    vec3 box_max = current;
    vec3 sum = current;
    vec3 sum_sq = current * current;
    vec3 t;
    t = texture(sampler2D(gather_tex, smp), v_uv + vec2(-texel.x, -texel.y)).rgb; box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = texture(sampler2D(gather_tex, smp), v_uv + vec2(0.0, -texel.y)).rgb; box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = texture(sampler2D(gather_tex, smp), v_uv + vec2(texel.x, -texel.y)).rgb; box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = texture(sampler2D(gather_tex, smp), v_uv + vec2(-texel.x, 0.0)).rgb; box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = texture(sampler2D(gather_tex, smp), v_uv + vec2(texel.x, 0.0)).rgb; box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = texture(sampler2D(gather_tex, smp), v_uv + vec2(-texel.x, texel.y)).rgb; box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = texture(sampler2D(gather_tex, smp), v_uv + vec2(0.0, texel.y)).rgb; box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = texture(sampler2D(gather_tex, smp), v_uv + vec2(texel.x, texel.y)).rgb; box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    vec3 avg = sum / 9.0;
    vec3 sigma = sqrt(max(sum_sq / 9.0 - avg * avg, vec3(0.0)));
    float gamma = (temporal_params.w > 0.0) ? temporal_params.w : SSGI_TEMPORAL_GAMMA_DEFAULT;
    box_min = max(box_min, avg - gamma * sigma);
    box_max = min(box_max, avg + gamma * sigma);

    // Variance divergence (same falloff as the TAA resolve).
    vec3 span = max(box_max - box_min, vec3(1e-4));
    vec3 dist_min = max(box_min - hist, vec3(0.0)) / span;
    vec3 dist_max = max(hist - box_max, vec3(0.0)) / span;
    float max_dist = max(max(dist_min.x, dist_max.x), max(max(dist_min.y, dist_max.y), max(dist_min.z, dist_max.z)));
    float accept = 1.0 / (1.0 + max_dist * max_dist * 4.0);

    // Depth consistency against the history-stored linear depth (the
    // bilateral gate): silhouette crossings take the fresh gather.
    float depth_threshold = max(0.001, camera_params.z * 0.2);
    if (abs(lin_depth - hist_sample.a) >= depth_threshold) accept = 0.0;

    // Normal consistency: the house depth-normal reconstruction at both
    // ends must agree; degenerate normals reject.
    vec3 world_pos = reconstructWorldPos(v_uv, raw_depth);
    vec3 n_cur = reconstructWorldNormal(v_uv, world_pos);
    float prev_raw_d = texture(sampler2D(depth_tex, depth_smp), prev_uv).r;
    if (prev_raw_d >= 0.9999) {
        accept = 0.0;
    } else {
        vec3 prev_world = reconstructWorldPos(prev_uv, prev_raw_d);
        vec3 n_prev = reconstructWorldNormal(prev_uv, prev_world);
        if (dot(n_cur, n_prev) < SSGI_TEMPORAL_NORMAL_MIN_DOT) accept = 0.0;
    }

    // Reactive markers (particles/glass carry the background velocity, so
    // their true motion is invisible): stamped coverage takes zero history
    // with no edge ramp — same band the TAA resolve tests.
    bool is_marker = (vel_sample.a > SSGI_TEMPORAL_MARKER_LO && vel_sample.a < SSGI_TEMPORAL_MARKER_HI);
    if (is_marker) accept = 0.0;

    vec3 hist_clamped = clamp(hist, box_min, box_max);
    float blend = clamp(temporal_params.z, 0.0, 1.0) * accept;
    vec3 outc = mix(current, hist_clamped, blend);
    frag_color = vec4(outc, lin_depth);
}
@end

@program ssgi_temporal vs fs
