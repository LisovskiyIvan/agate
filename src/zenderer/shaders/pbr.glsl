// Standard Cook-Torrance PBR (Physically Based Rendering) for zenderer
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 mvp;
    mat4 model;
    mat4 light_view_proj;
};

in vec3 position;
in vec3 normal;
in vec4 color0;
in vec2 texcoord0;
in vec4 tangent;

out vec3 v_world_pos;
out vec3 v_normal;
out vec3 v_tangent;
out vec3 v_bitangent;
out vec4 v_color;
out vec2 v_uv;
out vec4 v_light_space_pos;

void main() {
    vec4 world_pos = model * vec4(position, 1.0);
    v_world_pos = world_pos.xyz;
    gl_Position = mvp * vec4(position, 1.0);
    vec4 lpos = light_view_proj * world_pos;
    #if !SOKOL_GLSL
        lpos.y = -lpos.y;
    #endif
    v_light_space_pos = lpos;

    vec3 N = normalize(mat3(model) * normal);
    vec3 T = normalize(mat3(model) * tangent.xyz);
    // Gram-Schmidt orthogonalization
    T = normalize(T - dot(T, N) * N);
    vec3 B = cross(N, T) * tangent.w;

    v_normal = N;
    v_tangent = T;
    v_bitangent = B;
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
    vec4 pbr_factors; // x: metallic, y: roughness, z: occlusion_strength, w: unused
    vec4 emissive_factor; // xyz: emissive color, w: unused
    vec4 shadow_params; // x: bias, y: intensity, z/w: unused
};

layout(binding = 0) uniform texture2D albedo_tex;
layout(binding = 1) uniform texture2D normal_tex;
layout(binding = 2) uniform texture2D metallic_roughness_tex;
layout(binding = 3) uniform texture2D emissive_tex;
layout(binding = 4) uniform texture2D occlusion_tex;
layout(binding = 5) uniform texture2D shadow_tex;
layout(binding = 0) uniform sampler smp;
layout(binding = 1) uniform sampler shadow_smp;


in vec3 v_world_pos;
in vec3 v_normal;
in vec3 v_tangent;
in vec3 v_bitangent;
in vec4 v_color;
in vec2 v_uv;
in vec4 v_light_space_pos;

out vec4 frag_color;

const float PI = 3.14159265359;

float calculateShadow(vec4 light_space_pos, vec3 N, vec3 L) {
    if (shadow_params.y <= 0.001) return 0.0;

    vec3 proj = light_space_pos.xyz / light_space_pos.w;
    if (proj.z > 1.0 || proj.z < 0.0) return 0.0;

    vec2 uv = (proj.xy + 1.0) * 0.5;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 0.0;

    float cos_theta = max(dot(N, L), 0.0);
    float bias = max(shadow_params.x * (1.0 - cos_theta), shadow_params.x * 0.2);
    float depth = proj.z - bias;

    vec2 texel_size = vec2(1.0 / 2048.0);
    float s0 = texture(sampler2DShadow(shadow_tex, shadow_smp), vec3(uv + vec2(-0.75, -0.75) * texel_size, depth));
    float s1 = texture(sampler2DShadow(shadow_tex, shadow_smp), vec3(uv + vec2( 0.75, -0.75) * texel_size, depth));
    float s2 = texture(sampler2DShadow(shadow_tex, shadow_smp), vec3(uv + vec2(-0.75,  0.75) * texel_size, depth));
    float s3 = texture(sampler2DShadow(shadow_tex, shadow_smp), vec3(uv + vec2( 0.75,  0.75) * texel_size, depth));
    float lit = (s0 + s1 + s2 + s3) * 0.25;

    return (1.0 - lit) * shadow_params.y;
}

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
    vec4 albedo_tex_val = texture(sampler2D(albedo_tex, smp), v_uv);
    vec4 albedo_rgba = v_color * base_color_factor * albedo_tex_val;
    vec3 albedo = albedo_rgba.rgb;

    vec4 mr_sample = texture(sampler2D(metallic_roughness_tex, smp), v_uv);
    float metallic = clamp(pbr_factors.x * mr_sample.b, 0.0, 1.0);
    float roughness = clamp(pbr_factors.y * mr_sample.g, 0.04, 1.0);

    // Normal mapping with TBN matrix
    vec3 map_n = texture(sampler2D(normal_tex, smp), v_uv).xyz * 2.0 - 1.0;
    mat3 TBN = mat3(normalize(v_tangent), normalize(v_bitangent), normalize(v_normal));
    vec3 N = normalize(TBN * map_n);

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

    float shadow = calculateShadow(v_light_space_pos, N, L);

    vec3 radiance = light_color.rgb * light_color.a;
    vec3 Lo = (kD * albedo / PI + specular) * radiance * NdotL * (1.0 - shadow);

    // Ambient Occlusion
    float ao_sample = texture(sampler2D(occlusion_tex, smp), v_uv).r;
    float ao = 1.0 + pbr_factors.z * (ao_sample - 1.0);
    vec3 ambient = ambient_color.rgb * ambient_color.a * albedo * ao;

    // Emissive
    vec4 emissive_sample = texture(sampler2D(emissive_tex, smp), v_uv);
    vec3 emissive = emissive_factor.rgb * emissive_sample.rgb;

    vec3 final_color = ambient + Lo + emissive;

    frag_color = vec4(final_color, albedo_rgba.a);
}

@end

@program pbr vs fs
