// Screen-Space UI & Signed Distance Field (SDF) Text Shader for zenderer
@header const m = @import("math")

@vs vs
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    vec4 screen_size;
};

in vec2 position;
in vec2 texcoord0;
in vec4 color0;
in vec4 mode_params;

out vec2 v_uv;
out vec4 v_color;
out vec4 v_params;

void main() {
    float ndc_x = (2.0 * position.x) / screen_size.x - 1.0;
    float ndc_y = 1.0 - (2.0 * position.y) / screen_size.y;
    gl_Position = vec4(ndc_x, ndc_y, 0.0, 1.0);
    v_uv = texcoord0;
    v_color = color0;
    v_params = mode_params;
}
@end

@fs fs
layout(binding = 0) uniform texture2D font_tex;
layout(binding = 0) uniform sampler smp;

in vec2 v_uv;
in vec4 v_color;
in vec4 v_params;

out vec4 frag_color;

void main() {
    if (v_params.x < 0.5) {
        // Mode 0: Solid UI quad / button / panel / border
        frag_color = v_color;
    } else if (v_params.x < 1.5) {
        // Mode 1: Crisp bold SDF text with screen-space anti-aliasing
        float dist = texture(sampler2D(font_tex, smp), v_uv).r;
        float w = clamp(0.75 * fwidth(dist), 0.012, 0.045);
        float edge = clamp(0.32 - v_params.z, 0.10, 0.48);
        float alpha = smoothstep(edge - w, edge + w, dist);
        frag_color = vec4(v_color.rgb, v_color.a * alpha);
    } else {
        // Mode 2: SDF text with dark outline / drop shadow
        float dist = texture(sampler2D(font_tex, smp), v_uv).r;
        float w = clamp(0.75 * fwidth(dist), 0.012, 0.045);
        float edge = clamp(0.32 - v_params.z, 0.10, 0.48);
        float text_alpha = smoothstep(edge - w, edge + w, dist);
        float outline_edge = clamp(edge - v_params.y, 0.05, edge - 0.02);
        float outline_alpha = smoothstep(outline_edge - w, outline_edge + w, dist);
        vec3 outline_col = vec3(0.02, 0.03, 0.05);
        vec3 col = mix(outline_col, v_color.rgb, text_alpha);
        frag_color = vec4(col, max(text_alpha, outline_alpha * 0.90) * v_color.a);
    }
}
@end

@program ui vs fs
