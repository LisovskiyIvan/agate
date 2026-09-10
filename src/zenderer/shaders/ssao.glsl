// Screen-Space Ambient Occlusion (SSAO) Generation Shader for zenderer
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
};

@image_sample_type depth_tex unfilterable_float
layout(binding = 0) uniform texture2D depth_tex;
layout(binding = 1) uniform texture2D noise_tex;
@sampler_type smp_depth nonfiltering
layout(binding = 0) uniform sampler smp_depth;
@sampler_type smp_noise nonfiltering
layout(binding = 1) uniform sampler smp_noise;

in vec2 v_uv;
out vec4 frag_color;

vec3 reconstructViewPos(vec2 uv, float depth) {
    vec4 clip = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    vec4 view = inv_projection * clip;
    return view.xyz / view.w;
}

void main() {
    vec2 uv = v_uv;
    float raw_depth = texture(sampler2D(depth_tex, smp_depth), uv).r;

    // Background / Skybox: no occlusion
    if (raw_depth >= 0.9999) {
        frag_color = vec4(1.0, 1.0, 1.0, 1.0);
        return;
    }

    vec3 pos = reconstructViewPos(uv, raw_depth);
    vec2 texel = resolution.zw;

    // Reconstruct view-space normal using depth-aware cross product
    vec3 p_r = reconstructViewPos(uv + vec2(texel.x, 0.0), texture(sampler2D(depth_tex, smp_depth), uv + vec2(texel.x, 0.0)).r);
    vec3 p_l = reconstructViewPos(uv - vec2(texel.x, 0.0), texture(sampler2D(depth_tex, smp_depth), uv - vec2(texel.x, 0.0)).r);
    vec3 p_u = reconstructViewPos(uv - vec2(0.0, texel.y), texture(sampler2D(depth_tex, smp_depth), uv - vec2(0.0, texel.y)).r);
    vec3 p_d = reconstructViewPos(uv + vec2(0.0, texel.y), texture(sampler2D(depth_tex, smp_depth), uv + vec2(0.0, texel.y)).r);

    vec3 dx = (abs(p_r.z - pos.z) < abs(p_l.z - pos.z)) ? (p_r - pos) : (pos - p_l);
    vec3 dy = (abs(p_u.z - pos.z) < abs(p_d.z - pos.z)) ? (p_u - pos) : (pos - p_d);

    vec3 normal = normalize(cross(dx, dy));
    if (normal.z < 0.0) {
        normal = -normal;
    }

    // Tiled 4x4 noise rotation vector
    vec2 noise_scale = resolution.xy / 4.0;
    vec3 random_vec = vec3(texture(sampler2D(noise_tex, smp_noise), uv * noise_scale).xy * 2.0 - 1.0, 0.0);

    // Tangent basis
    vec3 tangent = normalize(random_vec - normal * dot(random_vec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 tbn = mat3(tangent, bitangent, normal);

    float radius = params.x;
    float bias = params.y;
    float occlusion = 0.0;

    for (int i = 0; i < 32; i++) {
        vec3 sample_view = pos + (tbn * kernel_samples[i].xyz) * radius;

        vec4 sample_clip = projection * vec4(sample_view, 1.0);
        sample_clip.xyz /= sample_clip.w;
        vec2 sample_uv = vec2(sample_clip.x * 0.5 + 0.5, 0.5 - sample_clip.y * 0.5);

        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0) {
            continue;
        }

        float sample_raw_depth = texture(sampler2D(depth_tex, smp_depth), sample_uv).r;
        vec3 sample_frag_view = reconstructViewPos(sample_uv, sample_raw_depth);

        // Range check to avoid haloing on distant backgrounds
        float range_check = smoothstep(0.0, 1.0, radius / (abs(pos.z - sample_frag_view.z) + 0.001));

        // In view space, camera looks down -Z. A surface closer to camera has higher z value (e.g. -2.0 > -5.0).
        if (sample_frag_view.z >= sample_view.z + bias) {
            occlusion += range_check;
        }
    }

    occlusion = 1.0 - (occlusion / 32.0);
    float ao = clamp(pow(occlusion, params.w), 0.0, 1.0);

    frag_color = vec4(ao, ao, ao, 1.0);
}
@end

@program ssao vs fs
