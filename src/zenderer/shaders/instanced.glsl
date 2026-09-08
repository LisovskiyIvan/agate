// Instanced Standard Shader with Multi-Lights for zenderer
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 view_proj;
    mat4 light_view_proj;
};

// Per-vertex attributes (Buffer 0)
in vec3 position;
in vec3 normal;
in vec4 color0;
in vec2 texcoord0;

// Per-instance attributes (Buffer 1)
in vec4 inst_mat0;
in vec4 inst_mat1;
in vec4 inst_mat2;
in vec4 inst_mat3;

out vec3 v_world_pos;
out vec3 v_normal;
out vec4 v_color;
out vec2 v_uv;
out vec4 v_light_space_pos;

void main() {
    mat4 model = mat4(inst_mat0, inst_mat1, inst_mat2, inst_mat3);
    vec4 world_pos = model * vec4(position, 1.0);
    v_world_pos = world_pos.xyz;
    gl_Position = view_proj * world_pos;
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
    vec4 shadow_params; // x: bias, y: intensity, z/w: unused
    vec4 light_counts; // x: num_point_lights, y: num_spot_lights, z/w: unused
    vec4 point_pos_range[4];
    vec4 point_color_int[4];
    vec4 spot_pos_range[2];
    vec4 spot_dir_inner[2];
    vec4 spot_color_outer[2];
    vec4 spot_intensity[2];
};

layout(binding = 0) uniform texture2D diffuse_tex;
layout(binding = 1) uniform texture2D shadow_tex;
layout(binding = 0) uniform sampler smp;
layout(binding = 1) uniform sampler shadow_smp;

in vec3 v_world_pos;
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

    // Primary directional light
    vec3 L = normalize(light_dir.xyz);
    float NdotL = max(dot(N, L), 0.0);
    float shadow = calculateShadow(v_light_space_pos, N, L);
    vec3 diffuse = light_color.rgb * (NdotL * light_color.a) * (1.0 - shadow);

    // Point Lights
    int num_points = int(light_counts.x);
    for (int i = 0; i < 4; i++) {
        if (i >= num_points) break;
        vec3 p_pos = point_pos_range[i].xyz;
        float p_range = point_pos_range[i].w;
        vec3 p_col = point_color_int[i].rgb;
        float p_int = point_color_int[i].w;

        vec3 p_to_light = p_pos - v_world_pos;
        float dist = length(p_to_light);
        if (dist >= p_range || dist < 0.0001) continue;

        vec3 p_L = p_to_light / dist;
        float p_NdotL = max(dot(N, p_L), 0.0);
        if (p_NdotL > 0.0) {
            float d_norm = dist / p_range;
            float factor = clamp(1.0 - d_norm * d_norm * d_norm * d_norm, 0.0, 1.0);
            float att = (factor * factor) / (dist * dist + 1.0);
            diffuse += p_col * (p_NdotL * p_int * att);
        }
    }

    // Spot Lights
    int num_spots = int(light_counts.y);
    for (int i = 0; i < 2; i++) {
        if (i >= num_spots) break;
        vec3 s_pos = spot_pos_range[i].xyz;
        float s_range = spot_pos_range[i].w;
        vec3 s_dir = normalize(spot_dir_inner[i].xyz);
        float cos_inner = spot_dir_inner[i].w;
        vec3 s_col = spot_color_outer[i].rgb;
        float cos_outer = spot_color_outer[i].w;
        float s_int = spot_intensity[i].x;

        vec3 s_to_light = s_pos - v_world_pos;
        float dist = length(s_to_light);
        if (dist >= s_range || dist < 0.0001) continue;

        vec3 s_L = s_to_light / dist;
        float s_NdotL = max(dot(N, s_L), 0.0);
        if (s_NdotL > 0.0) {
            float d_norm = dist / s_range;
            float factor = clamp(1.0 - d_norm * d_norm * d_norm * d_norm, 0.0, 1.0);
            float dist_att = (factor * factor) / (dist * dist + 1.0);

            float cos_angle = dot(-s_L, s_dir);
            float cone_att = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001), 0.0, 1.0);
            cone_att *= cone_att;

            diffuse += s_col * (s_NdotL * s_int * dist_att * cone_att);
        }
    }

    vec3 ambient = ambient_color.rgb * ambient_color.a;

    vec4 tex_val = texture(sampler2D(diffuse_tex, smp), v_uv);
    vec4 base = v_color * diffuse_color * tex_val;
    vec3 final_rgb = base.rgb * (ambient + diffuse);
    frag_color = vec4(final_rgb, base.a);
}
@end

@program instanced vs fs
