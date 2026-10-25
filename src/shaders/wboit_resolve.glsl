// Fullscreen resolve pass for Weighted Blended Order-Independent Transparency (WBOIT)
@header const m = @import("math")

@vs vs
@glsl_options fixup_clipspace
// @include "common/fullscreen_vs.glsl"
@end

@fs fs
layout(binding = 0) uniform texture2D accum_tex;
layout(binding = 1) uniform texture2D reveal_tex;
layout(binding = 0) uniform sampler smp;

in vec2 v_uv;
out vec4 frag_color;

void main() {
    float reveal = texture(sampler2D(reveal_tex, smp), v_uv).r;
    if (reveal >= 0.99999) {
        discard;
    }
    vec4 accum = texture(sampler2D(accum_tex, smp), v_uv);
    vec3 avg_color = clamp(accum.rgb / max(accum.a, 0.00001), 0.0, 65500.0);
    float alpha = clamp(1.0 - reveal, 0.0, 1.0);
    frag_color = vec4(avg_color, alpha);
}
@end

@program wboit_resolve vs fs
