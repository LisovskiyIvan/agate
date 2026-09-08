// Standard Blinn-Phong/Diffuse shader for zenderer
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

out vec3 v_normal;
out vec4 v_color;
out vec2 v_uv;
out vec4 v_light_space_pos;

void main() {
    vec4 world_pos = model * vec4(position, 1.0);
    gl_Position = mvp * vec4(position, 1.0);
    v_normal = mat3(model) * normal;
    v_color = color0;
    v_uv = texcoord0;
    vec4 lpos = light_view_proj * world_pos;
    #if !SOKOL_GLSL
        lpos.y = -lpos.y;
    #endif
    v_light_space_pos = lpos;
}
@end

@fs fs
layout(binding = 1) uniform fs_params {
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient_color;
    vec4 diffuse_color;
    vec4 shadow_params; // x: bias, y: intensity (1.0=full, 0.0=off), z/w: unused
};

layout(binding = 0) uniform texture2D diffuse_tex;
layout(binding = 1) uniform texture2D shadow_tex;
layout(binding = 0) uniform sampler smp;
layout(binding = 1) uniform sampler shadow_smp;

in vec3 v_normal;
in vec4 v_color;
in vec2 v_uv;
in vec4 v_light_space_pos;

out vec4 frag_color;

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

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(light_dir.xyz);
    float NdotL = max(dot(N, L), 0.0);
    float shadow = calculateShadow(v_light_space_pos, N, L);

    vec3 diffuse = light_color.rgb * (NdotL * light_color.a) * (1.0 - shadow);
    vec3 ambient = ambient_color.rgb * ambient_color.a;

    vec4 tex_val = texture(sampler2D(diffuse_tex, smp), v_uv);
    vec4 base = v_color * diffuse_color * tex_val;
    vec3 final_rgb = base.rgb * (ambient + diffuse);
    frag_color = vec4(final_rgb, base.a);
}
@end

@program standard vs fs

