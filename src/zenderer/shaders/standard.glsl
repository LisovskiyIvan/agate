// Standard Blinn-Phong/Diffuse shader for zenderer
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

out vec3 v_normal;
out vec4 v_color;
out vec2 v_uv;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    v_normal = mat3(model) * normal;
    v_color = color0;
    v_uv = texcoord0;
}
@end

@fs fs
layout(binding = 1) uniform fs_params {
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient_color;
    vec4 diffuse_color;
};

layout(binding = 0) uniform texture2D diffuse_tex;
layout(binding = 0) uniform sampler smp;

in vec3 v_normal;
in vec4 v_color;
in vec2 v_uv;

out vec4 frag_color;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(light_dir.xyz);
    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = light_color.rgb * (NdotL * light_color.a);
    vec3 ambient = ambient_color.rgb * ambient_color.a;

    vec4 tex_val = texture(sampler2D(diffuse_tex, smp), v_uv);
    vec4 base = v_color * diffuse_color * tex_val;
    vec3 final_rgb = base.rgb * (ambient + diffuse);
    frag_color = vec4(final_rgb, base.a);
}
@end

@program standard vs fs
