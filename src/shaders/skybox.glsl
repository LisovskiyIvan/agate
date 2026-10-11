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

// APPENDED (analytic physical sky, opt-in): a NEW block, so the legacy
// texture path (params + sky_tex above) keeps its exact layout and the
// off path renders bit-identically. CPU mirror: postprocess/sky.zig
// (same constants and formulas; float-precision differences expected).
//   sky_sun:       xyz sun dir (toward sun, normalized), w turbidity
//   sky_ctrl:      x analytic_enabled (1/0), yzw spare (0)
//   sky_sun_color: xyz attenuated sun color, w sun intensity
layout(binding = 2) uniform sky_params {
    vec4 sky_sun;
    vec4 sky_ctrl;
    vec4 sky_sun_color;
};

in vec3 v_dir;
out vec4 frag_color;

float skyAirMass(float sin_elev) {
    float s = max(sin_elev, 0.02);
    float h = degrees(asin(clamp(sin_elev, -0.049, 1.0)));
    return 1.0 / (s + 0.50572 * pow(h + 6.07995, -1.6364));
}

vec3 skySunAtten(float sin_elev, float turb) {
    float m = skyAirMass(sin_elev);
    return exp(-vec3(0.011, 0.020, 0.042) * (turb * m));
}

float skyPerez(float cos_theta, float gamma, float turb) {
    float A = -(0.38 + 0.055 * turb);
    float B = -0.30;
    float C = 2.2 + 0.9 * turb;
    float D = -0.65;
    float E = 0.12;
    float ct = max(cos_theta, 0.012);
    float g = clamp(gamma, 0.0, 3.14159265);
    float cg = cos(g);
    return (1.0 + A * exp(B / ct)) * (1.0 + C * exp(D * g) + E * cg * cg);
}

// Returns vec4(tint.rgb, luminance).
vec4 skyZenithState(float s, float turb) {
    float sc = clamp(s, 0.02, 1.0);
    float lum = 1.15 * (0.22 + 0.78 * pow(sc, 0.55)) / (0.55 + 0.14 * turb);
    float whiten = clamp(0.18 + 0.075 * (turb - 2.0) - 0.55 * (sc - 0.25), 0.0, 1.0);
    return vec4(mix(vec3(0.30, 0.48, 1.0), vec3(0.86, 0.83, 0.79), whiten), lum);
}

// Returns vec4(tint.rgb, luminance); zen_lum anchors the horizon boost.
vec4 skyHorizonState(float s, float turb, float zen_lum) {
    float sc = clamp(s, 0.02, 1.0);
    float A = -(0.38 + 0.055 * turb);
    float boost = (1.0 / (1.0 + A * exp(-0.30))) * (0.85 + 0.10 * turb);
    float warmth = clamp(0.65 - 1.1 * sc + 0.06 * (turb - 2.0), 0.0, 1.0);
    return vec4(mix(vec3(0.55, 0.68, 0.85), vec3(1.0, 0.72, 0.48), warmth), zen_lum * boost);
}

vec3 skyAnalytic(vec3 dir) {
    float turb = sky_sun.w;
    vec3 sun_dir = normalize(sky_sun.xyz);
    float se = max(sun_dir.y, 0.02);
    vec4 zen = skyZenithState(se, turb);
    vec4 hor = skyHorizonState(se, turb, zen.a);
    if (dir.y < 0.0) return hor.rgb * 0.12;
    vec3 v = normalize(dir);
    float cos_g = clamp(dot(v, sun_dir), -1.0, 1.0);
    float gamma = acos(cos_g);
    float gamma_sun = acos(clamp(se, -1.0, 1.0));
    float f = skyPerez(max(v.y, 0.012), gamma, turb);
    float fz = skyPerez(1.0, gamma_sun, turb);
    float lum_r = zen.a * f / max(fz, 1e-4);
    vec3 col = mix(hor.rgb, zen.rgb, smoothstep(0.0, 0.5, v.y)) * lum_r;
    vec3 att = skySunAtten(se, turb);
    vec3 glow_tint = att / max(max(att.x, att.y), att.z);
    col += glow_tint * (pow(max(cos_g, 0.0), 24.0) * (0.35 + 0.45 * lum_r) * (0.8 + 0.15 * turb));
    if (cos_g > 0.9999893) {
        col += sky_sun_color.rgb * sky_sun_color.a * 8.0;
    }
    return max(col, vec3(0.0));
}

void main() {
    vec3 radiance;
    float alpha = 1.0;
    if (sky_ctrl.x > 0.5) {
        radiance = skyAnalytic(v_dir);
    } else {
        vec4 color = texture(samplerCube(sky_tex, smp), v_dir);
        radiance = color.rgb;
        alpha = color.a;
    }
    // Linear radiance scaled by intensity, finite-bound via the shared guard
    // so the scaled sky can neither feed NaN/Inf into additive blending nor
    // clamp >1 radiance (the composite owns exposure/tonemap).
    frag_color = vec4(boundRadiance(radiance * params.x), alpha);
}
@end

@program skybox vs fs
