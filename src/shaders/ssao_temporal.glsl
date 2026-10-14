// Screen-space AO temporal accumulation for agate.
//
// Runs between the half-res gather and the bilateral upsample
// (gather -> temporal -> denoise) when ssao.temporal.enabled is set.
// History holds the RAW AO scalar (R), the oct-encoded CURRENT world
// normal (G,B) for the normal gate, and the current linear depth (A).
// The bilateral denoise stays the final cleaner.
//
// Per pixel (half-res UV space, shared with the full-res buffers):
//  1. Current = raw AO tap + current world normal (single 4-tap
//     reconstruction); without valid history (first frame, toggle,
//     mode switch, resize, camera cut, multi-camera, MSAA — the parent
//     passes history_valid = 0) the output is the current AO plus its
//     normal/depth lanes, priming history.
//  2. Reproject: per-object velocity (uv_curr - uv_prev in the buffer, so
//     prev = uv - vel) when the velocity view is live; camera-only depth
//     reprojection (prev_view_proj * world_pos) otherwise. The previous
//     clip is ALSO the expected-depth source for both paths (see 3).
//  3. Reject: variance divergence of history against the current 3x3
//     raw-AO neighborhood (mean +/- gamma * sigma, same 1/(1+4d^2)
//     falloff as the TAA/SSGI resolves), depth consistency of the
//     history-stored linear depth against the EXPECTED previous-camera
//     depth (current world pos through prev_view_proj, the bilateral
//     gate max(0.001, radius * 0.2)) — never against the current depth,
//     so a translating camera keeps static-geometry history; normal
//     consistency of the current normal against the history-DECODED
//     previous normal (cosine >= 0.9); and the reactive-marker band
//     (velocity alpha 0.15..0.35 = particles/glass -> zero history).
//  4. Blend: mix(current, variance-clipped history, blend * accept) with
//     full-strength clipping; sky pixels output exactly 1.0 (unoccluded).
//  5. G,B carry the CURRENT oct-encoded normal and alpha the CURRENT
//     linear depth for the next frame's gates, even on fully rejected
//     or primed pixels.
//
// Limits (conservative by design): under per-object motion the
// camera-expected depth over-rejects (safe direction: the fresh gather
// wins, no ghosting — the velocity UV still places the tap, only the
// depth confirmation is strict). The velocity buffer carries NO
// previous-depth lane (B is hard zero), so none is fabricated. A camera
// cut / mode switch / resize primes instead of blending (parent-driven
// history_valid = 0).
//
// CPU mirror: src/ssao.zig aoTemporal* (blend, gates, resolve),
// aoOctEncode/aoOctDecode, aoReproject.
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
    mat4 prev_view_proj; // previous frame view-projection (camera fallback + expected depth)
    vec4 resolution; // xy: half-res temporal size, zw: half-res texel
    vec4 full_resolution; // xy: FULL effective size, zw: full texel (normal reconstruction)
    vec4 camera_params; // x: near, y: far, z: AO radius (m), w: unused
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

// Variance-clip tightness default (CPU mirror: AO_TEMPORAL_GAMMA);
// the parent uploads the same value.
const float AO_TEMPORAL_GAMMA_DEFAULT = 1.0;
// Normal-consistency cosine (CPU mirror: AO_TEMPORAL_NORMAL_MIN_DOT).
const float AO_TEMPORAL_NORMAL_MIN_DOT = 0.9;
// Reactive-marker band over the velocity alpha (CPU mirror:
// AO_TEMPORAL_MARKER_LO/HI; velocity.glsl fs_marker stamps 0.25).
const float AO_TEMPORAL_MARKER_LO = 0.15;
const float AO_TEMPORAL_MARKER_HI = 0.35;

// Octahedral encode/decode, [0,1]^2 convention (CPU mirror:
// aoOctEncode/aoOctDecode; same convention as the DDGI codec).
// Degenerate input encodes to the (0.5, 0.5) sentinel, which decodes to
// +Z; the depth gate is the primary guard where that fires.
vec2 aoOctEncode(vec3 d) {
    float s = abs(d.x) + abs(d.y) + abs(d.z);
    float inv = (s > 1e-9) ? (1.0 / s) : 0.0;
    vec2 p = vec2(d.x * inv, d.y * inv);
    if (d.z < 0.0) {
        p = vec2((1.0 - abs(p.y)) * (p.x >= 0.0 ? 1.0 : -1.0),
                 (1.0 - abs(p.x)) * (p.y >= 0.0 ? 1.0 : -1.0));
    }
    return p * 0.5 + 0.5;
}

vec3 aoOctDecode(vec2 o) {
    vec2 p = o * 2.0 - 1.0;
    vec3 d = vec3(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    if (d.z < 0.0) {
        d.xy = vec2((1.0 - abs(d.y)) * (d.x >= 0.0 ? 1.0 : -1.0),
                    (1.0 - abs(d.x)) * (d.y >= 0.0 ? 1.0 : -1.0));
    }
    float len = length(d);
    return (len < 1e-9) ? vec3(0.0, 0.0, 1.0) : (d / len);
}

float linearizeDepth(float d) {
    float near = camera_params.x;
    float far = camera_params.y;
    return (near * far) / max(far - d * (far - near), 0.0001);
}

vec3 reconstructWorldPos(vec2 uv, float depth) {
    vec4 clip = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
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

    // Same continuous-side choice as the SSGI temporal stage (view depth
    // from the perspective VP fourth row; CPU mirror: aoTemporalNormalOk
    // compares the resulting cosines).
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
    float current = texture(sampler2D(gather_tex, smp), v_uv).r;
    float raw_depth = texture(sampler2D(depth_tex, depth_smp), v_uv).r;
    float lin_depth = (raw_depth >= 1.0) ? 1e9 : linearizeDepth(raw_depth);

    // Sky (or missing depth) is unoccluded and has no reprojectable
    // surface: output exactly 1.0 with the normal sentinel.
    if (raw_depth >= 0.9999) {
        frag_color = vec4(1.0, 0.5, 0.5, 1.0);
        return;
    }

    // Current surface state (needed for the output lanes on every path,
    // including priming: the next frame's normal gate reads what we
    // store now).
    vec3 world_pos = reconstructWorldPos(v_uv, raw_depth);
    vec3 n_cur = reconstructWorldNormal(v_uv, world_pos);
    vec2 enc = (dot(n_cur, n_cur) > 1e-12) ? aoOctEncode(n_cur * inversesqrt(dot(n_cur, n_cur))) : vec2(0.5, 0.5);

    // No valid history yet: pass the current AO through and prime the
    // normal + depth lanes for the next frame.
    if (temporal_params.x < 0.5) {
        frag_color = vec4(current, enc, lin_depth);
        return;
    }

    // Previous-camera projection of the current surface point: the
    // reprojection basis for the fallback path AND the expected-depth
    // source for both paths.
    vec4 prev_clip = prev_view_proj * vec4(world_pos, 1.0);
    if (prev_clip.w <= 0.0001) {
        frag_color = vec4(current, enc, lin_depth);
        return;
    }
    vec3 prev_ndc = prev_clip.xyz / prev_clip.w;
    vec2 cam_prev_uv = vec2(prev_ndc.x * 0.5 + 0.5, 0.5 - prev_ndc.y * 0.5);

    // Reproject into history UV space.
    vec4 vel_sample = vec4(0.0, 0.0, 0.0, 0.0);
    if (temporal_params.y > 0.5) {
        vel_sample = texture(sampler2D(velocity_tex, velocity_smp), v_uv);
    }
    vec2 prev_uv;
    if (vel_sample.a > 0.5) {
        prev_uv = v_uv - vel_sample.xy;
    } else {
        prev_uv = cam_prev_uv;
    }
    if (prev_uv.x < 0.001 || prev_uv.x > 0.999 || prev_uv.y < 0.001 || prev_uv.y > 0.999) {
        frag_color = vec4(current, enc, lin_depth);
        return;
    }

    vec4 hist_sample = texture(sampler2D(history_tex, smp), prev_uv);
    float hist = hist_sample.r;

    // Current 3x3 raw-AO neighborhood: scalar variance box.
    vec2 texel = resolution.zw;
    float box_min = current;
    float box_max = current;
    float sum = current;
    float sum_sq = current * current;
    float t;
    t = texture(sampler2D(gather_tex, smp), v_uv + vec2(-texel.x, -texel.y)).r; box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = texture(sampler2D(gather_tex, smp), v_uv + vec2(0.0, -texel.y)).r; box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = texture(sampler2D(gather_tex, smp), v_uv + vec2(texel.x, -texel.y)).r; box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = texture(sampler2D(gather_tex, smp), v_uv + vec2(-texel.x, 0.0)).r; box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = texture(sampler2D(gather_tex, smp), v_uv + vec2(texel.x, 0.0)).r; box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = texture(sampler2D(gather_tex, smp), v_uv + vec2(-texel.x, texel.y)).r; box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = texture(sampler2D(gather_tex, smp), v_uv + vec2(0.0, texel.y)).r; box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    t = texture(sampler2D(gather_tex, smp), v_uv + vec2(texel.x, texel.y)).r; box_min = min(box_min, t); box_max = max(box_max, t); sum += t; sum_sq += t * t;
    float avg = sum / 9.0;
    float sigma = sqrt(max(sum_sq / 9.0 - avg * avg, 0.0));
    float gamma = (temporal_params.w > 0.0) ? temporal_params.w : AO_TEMPORAL_GAMMA_DEFAULT;
    box_min = max(box_min, avg - gamma * sigma);
    box_max = min(box_max, avg + gamma * sigma);

    // Variance divergence (same falloff as the TAA/SSGI resolves).
    float span = max(box_max - box_min, 1e-4);
    float dist = 0.0;
    if (hist < box_min) dist = (box_min - hist) / span;
    else if (hist > box_max) dist = (hist - box_max) / span;
    float accept = 1.0 / (1.0 + dist * dist * 4.0);

    // Depth consistency of the history-stored linear depth against the
    // EXPECTED previous-camera depth (not the current depth): a
    // translating camera keeps static-geometry history, silhouette
    // crossings take the fresh gather. Surfaces the previous camera
    // could not see (expected raw >= 1) reject outright.
    float depth_threshold = max(0.001, camera_params.z * 0.2);
    if (prev_ndc.z >= 0.9999) {
        accept = 0.0;
    } else {
        float expected_lin = linearizeDepth(clamp(prev_ndc.z, 0.0, 1.0));
        if (abs(expected_lin - hist_sample.a) >= depth_threshold) accept = 0.0;
    }

    // Normal consistency: the current normal against the history-DECODED
    // previous normal. Degenerate current normals reject (no surface to
    // agree on); under per-object motion this gate is strict (safe
    // direction), like the depth gate above.
    vec3 n_prev = aoOctDecode(hist_sample.gb);
    if (dot(n_cur, n_cur) <= 1e-12) {
        accept = 0.0;
    } else if (dot(n_cur * inversesqrt(dot(n_cur, n_cur)), n_prev) < AO_TEMPORAL_NORMAL_MIN_DOT) {
        accept = 0.0;
    }

    // Reactive markers (particles/glass carry the background velocity, so
    // their true motion is invisible): stamped coverage takes zero history
    // with no edge ramp — same band the TAA/SSGI resolves test.
    bool is_marker = (vel_sample.a > AO_TEMPORAL_MARKER_LO && vel_sample.a < AO_TEMPORAL_MARKER_HI);
    if (is_marker) accept = 0.0;

    float hist_clamped = clamp(hist, box_min, box_max);
    float blend = clamp(temporal_params.z, 0.0, 1.0) * accept;
    float out_ao = mix(current, hist_clamped, blend);
    frag_color = vec4(out_ao, enc, lin_depth);
}
@end

@program ssao_temporal vs fs
