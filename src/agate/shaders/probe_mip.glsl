// Reflection-probe mip prefilter (GGX importance sampling + diffuse irradiance, wave C.2).
// Renders one cube-face mip level from the source cubemap:
// - Mode 0 (params.z == 0): GGX Importance-Sampled specular prefilter with
//   roughness = params.y. Uses Hammersley low-discrepancy sequence with
//   solid-angle-based mip level selection to avoid aliasing and fireflies.
// - Mode 1 (params.z == 1): Cosine-weighted diffuse irradiance hemisphere
//   convolution for the coarsest mip (providing true diffuse ambient lighting).
// - Mode 2 (params.z == 2): Direct sampling / blit copy from source mip params.w.
//
// Orientation note: sampling is by world-space direction (samplerCube maps
// the direction to the right face automatically), so the per-face UV->dir
// table below only needs to cover its own face's directions.
// Must agree with scene/probe_layer.zig (faceDir order +X,-X,+Y,-Y,+Z,-Z).
// Note: no Y-flip (cube-face texel centers, not screen UVs) and no includes.
@vs vs
@glsl_options fixup_clipspace
in vec2 position;
in vec2 texcoord0;

out vec2 v_uv;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_uv = texcoord0;
}
@end

@fs fs
layout(binding = 0) uniform fs_params {
    vec4 params; // x: face 0..5 (+X,-X,+Y,-Y,+Z,-Z), y: roughness (0..1), z: mode (0=GGX, 1=irradiance, 2=copy), w: resolution or source mip
};

layout(binding = 0) uniform textureCube src_tex;
layout(binding = 0) uniform sampler smp;

in vec2 v_uv;
out vec4 frag_color;

vec3 probeFaceDir(float face, vec2 uv) {
    vec2 p = uv * 2.0 - 1.0;
    if (face < 0.5) return normalize(vec3(1.0, -p.y, -p.x)); // +X
    if (face < 1.5) return normalize(vec3(-1.0, -p.y, p.x)); // -X
    if (face < 2.5) return normalize(vec3(p.x, 1.0, p.y)); // +Y
    if (face < 3.5) return normalize(vec3(p.x, -1.0, -p.y)); // -Y
    if (face < 4.5) return normalize(vec3(p.x, -p.y, 1.0)); // +Z
    return normalize(vec3(-p.x, -p.y, -1.0)); // -Z
}

float radicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 hammersley(uint i, uint N) {
    return vec2(float(i) / float(N), radicalInverse_VdC(i));
}

vec3 importanceSampleGGX(vec2 Xi, vec3 N, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0 * 3.141592653589793 * Xi.x;
    float cosTheta = sqrt(clamp((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y), 0.0, 1.0));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

    vec3 H = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

    vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);

    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}

vec3 sampleCosineHemisphere(vec2 Xi, vec3 N) {
    float phi = 2.0 * 3.141592653589793 * Xi.x;
    float cosTheta = sqrt(clamp(1.0 - Xi.y, 0.0, 1.0));
    float sinTheta = sqrt(clamp(Xi.y, 0.0, 1.0));

    vec3 H = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

    vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);

    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}

void main() {
    vec3 N = probeFaceDir(params.x, v_uv);

    if (params.z < 0.5) {
        // Mode 0: GGX Specular Importance Sampling Prefilter
        float roughness = clamp(params.y, 0.0, 1.0);
        if (roughness < 0.005) {
            frag_color = vec4(textureLod(samplerCube(src_tex, smp), N, 0.0).rgb, 1.0);
            return;
        }
        float a = roughness * roughness;
        vec3 prefiltered_color = vec3(0.0);
        float total_weight = 0.0;
        const uint num_samples = 64u;
        float face_res = max(params.w, 1.0);
        float omega_p = (4.0 * 3.141592653589793) / (6.0 * face_res * face_res);

        for (uint i = 0u; i < num_samples; i++) {
            vec2 xi = hammersley(i, num_samples);
            vec3 H = importanceSampleGGX(xi, N, roughness);
            vec3 L = normalize(2.0 * dot(N, H) * H - N);
            float NdotL = max(dot(N, L), 0.0);
            if (NdotL > 0.0) {
                float NdotH = max(dot(N, H), 0.0);
                float VdotH = NdotH;
                float denom = (NdotH * NdotH * (a * a - 1.0) + 1.0);
                float D = (a * a) / (3.141592653589793 * denom * denom);
                float pdf = (D * NdotH) / (4.0 * VdotH) + 0.0001;
                float omega_s = 1.0 / (float(num_samples) * pdf);
                float mip_level = clamp(0.5 * log2(max(omega_s / omega_p, 0.0001)), 0.0, 7.0);

                prefiltered_color += textureLod(samplerCube(src_tex, smp), L, mip_level).rgb * NdotL;
                total_weight += NdotL;
            }
        }
        frag_color = vec4(total_weight > 0.0 ? prefiltered_color / total_weight : prefiltered_color, 1.0);
    } else if (params.z < 1.5) {
        // Mode 1: Cosine Diffuse Irradiance Convolution
        vec3 irradiance = vec3(0.0);
        const uint num_samples = 64u;
        float face_res = max(params.w, 1.0);
        float omega_p = (4.0 * 3.141592653589793) / (6.0 * face_res * face_res);

        for (uint i = 0u; i < num_samples; i++) {
            vec2 xi = hammersley(i, num_samples);
            vec3 L = sampleCosineHemisphere(xi, N);
            float NdotL = max(dot(N, L), 0.001);
            float pdf = NdotL / 3.141592653589793;
            float omega_s = 1.0 / (float(num_samples) * pdf);
            float mip_level = clamp(0.5 * log2(max(omega_s / omega_p, 0.0001)), 0.0, 7.0);

            irradiance += textureLod(samplerCube(src_tex, smp), L, mip_level).rgb;
        }
        frag_color = vec4(irradiance / float(num_samples), 1.0);
    } else {
        // Mode 2: Direct sample from source mip (blit copy)
        vec3 c = textureLod(samplerCube(src_tex, smp), N, params.w).rgb;
        frag_color = vec4(c, 1.0);
    }
}
@end

@program probe_mip vs fs
