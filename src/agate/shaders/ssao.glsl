// Screen-Space Ambient Occlusion (SSAO) Generation Shader for agate
@header const m = @import("math")
@ctype mat4 m.Mat4

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
    mat4 projection;
    mat4 inv_projection;
    vec4 kernel_samples[32];
    vec4 params; // x: radius, y: bias, z: intensity, w: power
    vec4 resolution; // xy: resolution, zw: texel size
    vec4 extra_params; // x: sample_count, yzw: unused
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
    vec3 normal = dot(normal_cross, normal_cross) > 1e-20 ? normalize(normal_cross) : normalize(-pos);
    if (dot(normal, -pos) < 0.0) {
        normal = -normal;
    }

    // High-frequency Interleaved Gradient Noise (IGN)
    // Eliminates all macro-tiling, 8x8 square blocks, and triangular bilinear saddle singularities
    float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    float angle = ign * 6.28318530718;
    vec3 random_vec = vec3(cos(angle), sin(angle), 0.0) + texture(sampler2D(noise_tex, smp_noise), uv).xyz * 0.000001;

    // Tangent basis
    vec3 tangent = normalize(random_vec - normal * dot(random_vec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 tbn = mat3(tangent, bitangent, normal);

    float radius = params.x;
    float bias = params.y;
    float occlusion = 0.0;

    int sample_count = int(extra_params.x > 0.5 ? extra_params.x : 24.0);
    sample_count = clamp(sample_count, 4, 32);

    vec4 pos_clip = projection * vec4(pos, 1.0);

    for (int i = 0; i < sample_count; i++) {
        vec3 sample_dir = tbn * kernel_samples[i].xyz;
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

        // Skip immediate local neighborhood (< 3cm) to eliminate triangle mesh self-occlusion
        if (dist < 0.03) continue;

        // Angle above horizon: only occlude if geometry sticks up at least ~9 degrees above tangent plane
        float n_dot_v = max(0.0, dot(normal, diff / dist) - 0.15);

        // Smooth linear distance attenuation
        float falloff = clamp(1.0 - (dist / radius), 0.0, 1.0);

        // In view space, camera looks down -Z. A surface closer to camera has higher z value.
        if (sample_frag_view.z >= sample_view.z + bias) {
            occlusion += n_dot_v * falloff;
        }
    }

    // Normalize with solid angle factor for sample_count hemisphere samples
    float norm_occ = clamp(occlusion / (float(sample_count) * (2.8 / 32.0)), 0.0, 1.0);
    float ao = clamp(pow(1.0 - norm_occ, params.w), 0.0, 1.0);

    frag_color = vec4(ao, ao, ao, 1.0);
}
@end

@program ssao vs fs
