// Standard Cook-Torrance PBR (Physically Based Rendering) for zenderer
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 mvp;
    mat4 model;
};

in vec3 position;
in vec3 normal;
in vec4 color0;
in vec2 texcoord0;

out vec3 v_world_pos;
out vec3 v_normal;
out vec4 v_color;
out vec2 v_uv;

void main() {
    vec4 world_pos = model * vec4(position, 1.0);
    v_world_pos = world_pos.xyz;
    gl_Position = mvp * vec4(position, 1.0);
    v_normal = normalize(mat3(model) * normal);
    v_color = color0;
    v_uv = texcoord0;
}
@end

@fs fs
layout(binding = 1) uniform fs_params {
    vec4 eye_pos;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient_color;
    vec4 base_color_factor;
    vec4 pbr_factors; // x: metallic, y: roughness, z: occlusion
};

layout(binding = 0) uniform texture2D albedo_tex;
layout(binding = 0) uniform sampler smp;

in vec3 v_world_pos;
in vec3 v_normal;
in vec4 v_color;
in vec2 v_uv;

out vec4 frag_color;

const float PI = 3.14159265359;

float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float num = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return num / max(denom, 0.0000001);
}

float geometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = geometrySchlickGGX(NdotV, roughness);
    float ggx1 = geometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

void main() {
    vec4 tex_val = texture(sampler2D(albedo_tex, smp), v_uv);
    vec4 albedo_rgba = v_color * base_color_factor * tex_val;
    vec3 albedo = albedo_rgba.rgb;

    float metallic = clamp(pbr_factors.x, 0.0, 1.0);
    float roughness = clamp(pbr_factors.y, 0.04, 1.0);

    vec3 N = normalize(v_normal);
    vec3 V = normalize(eye_pos.xyz - v_world_pos);
    vec3 L = normalize(light_dir.xyz);
    vec3 H = normalize(V + L);

    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);

    float NDF = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 numerator = NDF * G * F;
    float denominator = 4.0 * NdotV * NdotL + 0.0001;
    vec3 specular = numerator / denominator;

    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

    vec3 radiance = light_color.rgb * light_color.a;
    vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL;

    vec3 ambient = ambient_color.rgb * ambient_color.a * albedo;
    vec3 final_color = ambient + Lo;

    frag_color = vec4(final_color, albedo_rgba.a);
}
@end

@program pbr vs fs
