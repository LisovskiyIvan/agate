@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
layout(binding = 0) uniform vs_params {
    mat4 view_proj;
};

in vec3 position;
out vec3 v_dir;

void main() {
    v_dir = position;
    vec4 pos = view_proj * vec4(position, 1.0);
    gl_Position = pos.xyww;
}
@end

@fs fs
// Shared finite-radiance guard (fragment scope; parent enables the build
// include for this shader).
// @include "common/linear_output.glsl"
layout(binding = 1) uniform fs_params {
    vec4 params; // x: exposure / intensity, yzw: unused
};

layout(binding = 0) uniform textureCube sky_tex;
layout(binding = 0) uniform sampler smp;

in vec3 v_dir;
out vec4 frag_color;

void main() {
    vec4 color = texture(samplerCube(sky_tex, smp), v_dir);
    // Linear radiance scaled by intensity, finite-bound via the shared guard
    // so the scaled sky can neither feed NaN/Inf into additive blending nor
    // clamp >1 radiance (the composite owns exposure/tonemap).
    frag_color = vec4(boundRadiance(color.rgb * params.x), color.a);
}
@end

@program skybox vs fs
