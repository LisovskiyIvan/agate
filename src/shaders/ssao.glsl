// Screen-Space Ambient Occlusion (SSAO) Generation Shader for agate
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
// @include "common/fullscreen_vs.glsl"
@end

@fs fs
layout(binding = 0) uniform fs_params {
    mat4 projection;
    mat4 inv_projection;
    vec4 kernel_samples[32];
    vec4 params; // x: radius, y: bias, z: intensity, w: power
    vec4 resolution; // xy: resolution, zw: texel size
    vec4 extra_params; // x: sample_count, y: mode (0 hemisphere, 1 horizon), z: temporal rotation radians (0 when off), w: unused
};

@image_sample_type depth_tex unfilterable_float
layout(binding = 0) uniform texture2D depth_tex;
layout(binding = 1) uniform texture2D noise_tex;
@sampler_type smp_depth nonfiltering
layout(binding = 0) uniform sampler smp_depth;
layout(binding = 1) uniform sampler smp_noise;

in vec2 v_uv;
out vec4 frag_color;

vec3 reconstructViewPos(vec2 uv, float depth) {
    vec2 ndc = uv * 2.0 - 1.0;
    #if !SOKOL_GLSL
        ndc.y = -ndc.y;
    #endif
    vec4 clip = vec4(ndc, depth, 1.0);
    vec4 view = inv_projection * clip;
    return view.xyz / view.w;
}

void main() {
    // Nearest depth comes from a full-resolution texel, not the AO pixel's
    // interpolated UV. Reconstruct at that texel centre to avoid false slopes.
    vec2 uv = (floor(v_uv * resolution.xy) + 0.5) * resolution.zw;
    float raw_depth = texture(sampler2D(depth_tex, smp_depth), uv).r;

    // Background / Skybox: no occlusion
    if (raw_depth >= 1.0 || params.x <= 0.0) {
        frag_color = vec4(1.0, 1.0, 1.0, 1.0);
        return;
    }

    vec3 pos = reconstructViewPos(uv, raw_depth);
    vec2 texel = resolution.zw;

    // Reconstruct view-space normal with edge-preserving centered difference
    vec3 p_r = reconstructViewPos(uv + vec2(texel.x, 0.0), texture(sampler2D(depth_tex, smp_depth), uv + vec2(texel.x, 0.0)).r);
    vec3 p_l = reconstructViewPos(uv - vec2(texel.x, 0.0), texture(sampler2D(depth_tex, smp_depth), uv - vec2(texel.x, 0.0)).r);
    vec3 p_u = reconstructViewPos(uv - vec2(0.0, texel.y), texture(sampler2D(depth_tex, smp_depth), uv - vec2(0.0, texel.y)).r);
    vec3 p_d = reconstructViewPos(uv + vec2(0.0, texel.y), texture(sampler2D(depth_tex, smp_depth), uv + vec2(0.0, texel.y)).r);

    vec3 dx = (abs(p_r.z - pos.z) < abs(p_l.z - pos.z)) ? (p_r - pos) : (pos - p_l);
    vec3 dy = (abs(p_u.z - pos.z) < abs(p_d.z - pos.z)) ? (p_u - pos) : (pos - p_d);

    vec3 normal_cross = cross(dx, dy);
    float normal_len_sq = dot(normal_cross, normal_cross);
    // Scale-relative degeneracy guard (CPU mirror: aoNormalFromGradients
    // in src/ssao.zig; same 1e-8 product-scale rule as
    // postprocess/depth_normal.zig): degenerate gradients carry no
    // surface, so there is nothing to occlude — return unoccluded
    // instead of inventing a camera-facing normal.
    float normal_scale = dot(dx, dx) * dot(dy, dy);
    if (normal_len_sq <= max(normal_scale * 1e-8, 1e-30)) {
        frag_color = vec4(1.0, 1.0, 1.0, 1.0);
        return;
    }
    vec3 normal = normal_cross * inversesqrt(normal_len_sq);
    if (dot(normal, -pos) < 0.0) {
        normal = -normal;
    }

    // High-frequency Interleaved Gradient Noise (IGN)
    // Eliminates all macro-tiling, 8x8 square blocks, and triangular bilinear saddle singularities
    float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    // Temporal jitter (opt-in): the parent uploads the bounded rotation
    // sequence via extra_params.z, exactly 0.0 when temporal is off or
    // fail-closed, so adding it preserves the default arithmetic exactly
    // while active frames integrate varying sample directions.
    float angle = ign * 6.28318530718 + extra_params.z;
    vec3 random_vec = vec3(cos(angle), sin(angle), 0.0) + texture(sampler2D(noise_tex, smp_noise), uv).xyz * 0.000001;

    // Tangent basis
    vec3 tangent = normalize(random_vec - normal * dot(random_vec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 tbn = mat3(tangent, bitangent, normal);

    float radius = params.x;
    float bias = params.y;

    int sample_count = int(extra_params.x > 0.5 ? extra_params.x : 24.0);
    sample_count = clamp(sample_count, 4, 32);
    int ao_mode = int(extra_params.y + 0.5);

    // Horizon search (opt-in GTAO class): 4 slices x 2 sides, steps per
    // side derived from sample_count so the tap budget matches the
    // hemisphere path (CPU mirror: aoHorizonStepsPerSide). Per-side
    // maxima (elevation * falloff) average to the slice coverage; a
    // single high occluder sets the horizon, samples behind it add
    // nothing. Same normal, radius, bias, horizon offset (0.15) and
    // power curve as the hemisphere path below.
    if (ao_mode == 1 && radius > 0.0) {
        float proj_scale = projection[1][1] * resolution.y * 0.5;
        float pixel_radius = (-pos.z > 1e-5) ? (radius * proj_scale / -pos.z) : 0.0;
        int steps = clamp(sample_count / 8, 2, 4);
        float step_px = (steps > 0) ? (pixel_radius / float(steps)) : 0.0;
        float slice_sum = 0.0;
        for (int s = 0; s < 4; s++) {
            float a = angle + float(s) * 0.78539816339;
            vec2 dir = vec2(cos(a), sin(a));
            float max_a = 0.0;
            float max_b = 0.0;
            for (int st = 1; st <= 4; st++) {
                // Step budget: stop at the configured step count BEFORE any
                // depth fetch (no multiply-by-zero wasted taps when steps is
                // 2 or 3). Loop stays statically bounded for all four slangs.
                if (st > steps) break;
                vec2 off_a = dir * step_px * float(st) * texel;
                vec2 off_b = -off_a;
                vec2 uv_a = uv + off_a;
                vec2 uv_b = uv + off_b;
                if (uv_a.x >= 0.0 && uv_a.x <= 1.0 && uv_a.y >= 0.0 && uv_a.y <= 1.0) {
                    vec2 suv_a = (floor(uv_a * resolution.xy) + 0.5) * resolution.zw;
                    float d_a = texture(sampler2D(depth_tex, smp_depth), suv_a).r;
                    if (d_a < 1.0) {
                        vec3 q_a = reconstructViewPos(suv_a, d_a);
                        vec3 diff_a = q_a - pos;
                        float dist_a = length(diff_a);
                        if (dist_a >= 0.03 && dist_a < radius && q_a.z >= pos.z + bias) {
                            float v_a = max(0.0, dot(normal, diff_a / dist_a) - 0.15) * (1.0 - dist_a / radius);
                            max_a = max(max_a, v_a);
                        }
                    }
                }
                if (uv_b.x >= 0.0 && uv_b.x <= 1.0 && uv_b.y >= 0.0 && uv_b.y <= 1.0) {
                    vec2 suv_b = (floor(uv_b * resolution.xy) + 0.5) * resolution.zw;
                    float d_b = texture(sampler2D(depth_tex, smp_depth), suv_b).r;
                    if (d_b < 1.0) {
                        vec3 q_b = reconstructViewPos(suv_b, d_b);
                        vec3 diff_b = q_b - pos;
                        float dist_b = length(diff_b);
                        if (dist_b >= 0.03 && dist_b < radius && q_b.z >= pos.z + bias) {
                            float v_b = max(0.0, dot(normal, diff_b / dist_b) - 0.15) * (1.0 - dist_b / radius);
                            max_b = max(max_b, v_b);
                        }
                    }
                }
            }
            slice_sum += (max_a + max_b) * 0.5;
        }
        float norm_occ = clamp(slice_sum / 4.0, 0.0, 1.0);
        float ao = clamp(pow(1.0 - norm_occ, params.w), 0.0, 1.0);
        frag_color = vec4(ao, ao, ao, 1.0);
        return;
    }

    // Cosine-weighted visibility estimate (CPU mirror:
    // aoKernelIndex/aoKernelCos/aoHemisphereTap/aoHemisphereResolve in
    // src/ssao.zig): each executed tap contributes its sampling weight
    // (the normalized kernel cosine) times distance falloff when it is a
    // real occluder, over the sum of the executed taps' weights. A wall
    // blocking half the cosine-weighted hemisphere keeps ~0.5 visibility
    // instead of going black; only near-full enclosure darkens deeply.
    // Reduced budgets use midpoint-stratified kernel taps across the
    // whole kernel, not the grazing/short-ray prefix.
    float occlusion = 0.0;
    float weight_sum = 0.0;

    vec4 pos_clip = projection * vec4(pos, 1.0);

    for (int i = 0; i < sample_count; i++) {
        // Midpoint stratification (CPU mirror: aoKernelIndex): N=32 is
        // the identity, smaller budgets spread over the full kernel.
        int k = ((2 * i + 1) * 32) / (2 * sample_count);
        vec3 kernel_dir = normalize(kernel_samples[k].xyz);
        float tap_weight = kernel_dir.z;
        // Fixed denominator over the EXECUTED tap set (skipped taps keep
        // their weight, matching the CPU aoHemisphereDenom): validity
        // misses dilute, exactly like the historical fixed divisor.
        weight_sum += tap_weight;
        vec3 sample_dir = tbn * kernel_samples[k].xyz;
        vec3 sample_view = pos + sample_dir * radius;

        vec4 sample_clip = pos_clip + projection * vec4(sample_dir * radius, 0.0);
        if (sample_clip.w <= 0.0) continue;
        sample_clip.xyz /= sample_clip.w;
        vec2 sample_uv = sample_clip.xy * 0.5 + 0.5;
        #if !SOKOL_GLSL
            sample_uv.y = 1.0 - sample_uv.y;
        #endif

        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            continue;
        }

        sample_uv = (floor(sample_uv * resolution.xy) + 0.5) * resolution.zw;
        float sample_raw_depth = texture(sampler2D(depth_tex, smp_depth), sample_uv).r;
        if (sample_raw_depth >= 1.0) continue;
        vec3 sample_frag_view = reconstructViewPos(sample_uv, sample_raw_depth);

        // Vector from current shading position to sampled surface geometry
        vec3 diff = sample_frag_view - pos;
        float dist = length(diff);

        // Skip the immediate local neighborhood (< 3cm, the shading
        // pixel's own triangle) and taps past the radius (zero falloff).
        if (dist < 0.03 || dist >= radius) continue;

        // Horizon gate (stays binary: the occluder-direction cosine must
        // clear ~9 degrees above the tangent plane, else grazing
        // self-hits would shadow flat surfaces). The sampling-weight
        // cosine above is a different distribution; the two are never
        // multiplied together.
        float n_dot = dot(normal, diff / dist);

        // Smooth linear distance attenuation
        float falloff = clamp(1.0 - (dist / radius), 0.0, 1.0);

        // In view space, camera looks down -Z. A surface closer to camera has higher z value.
        if (sample_frag_view.z >= sample_view.z + bias && n_dot > 0.15) {
            occlusion += tap_weight * falloff;
        }
    }

    // Cosine-weighted blocked fraction over the executed taps' weights
    // (CPU mirror: aoHemisphereDenom). Intensity is NOT applied here:
    // the composite (postprocess.glsl ssao_params.z) owns it exactly once.
    float norm_occ = clamp(occlusion / max(weight_sum, 1e-6), 0.0, 1.0);
    float ao = clamp(pow(1.0 - norm_occ, params.w), 0.0, 1.0);

    frag_color = vec4(ao, ao, ao, 1.0);
}
@end

@program ssao vs fs
