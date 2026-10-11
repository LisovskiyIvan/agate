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
// APPENDED v2 (clouds + night, same binding 2: existing offsets untouched;
//   all-zero extras keep the day dome bit-identical):
//   sky_cloud:     x coverage 0..1, y scale, z time_sec, w cloud_enabled
//   sky_cloud2:    xy wind offset (uv, already dir*speed*time; scaled by
//                  coverage scale on use), z brightness, w silver strength
//   sky_night:     x night_amount 0..1, y star_intensity, z star_density,
//                  w moon_intensity
//   sky_night2:    x cloud horizon fade, y night horizon dim, z time_sec
//                  (star twinkle when clouds are off), w spare (0)
//   sky_moon:      xyz moon dir (toward moon, normalized), w moon disc cos
layout(binding = 2) uniform sky_params {
    vec4 sky_sun;
    vec4 sky_ctrl;
    vec4 sky_sun_color;
    vec4 sky_cloud;
    vec4 sky_cloud2;
    vec4 sky_night;
    vec4 sky_night2;
    vec4 sky_moon;
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

// Deterministic value-noise FBM for the dome cloud layer (hash has no
// textures/tables; identical input uv gives identical output on every
// backend up to float precision).
float skyHash21(vec2 p) {
    p = fract(p * vec2(234.34, 435.345));
    p += dot(p, p + 34.23);
    return fract(p.x * p.y);
}

float skyVNoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = skyHash21(i);
    float b = skyHash21(i + vec2(1.0, 0.0));
    float c = skyHash21(i + vec2(0.0, 1.0));
    float d = skyHash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float skyFbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 4; i++) {
        v += a * skyVNoise(p);
        p = p * 2.03 + vec2(17.3, 9.1);
        a *= 0.5;
    }
    return v * 1.0667; // 4-octave sum peaks at 0.9375; back to ~0..1
}

// Night horizon residue: today's horizon tint dimmed by the night horizon
// behavior lane (recomputed from the day state; no extra uniforms).
vec3 horNightHorizon() {
    float turb = sky_sun.w;
    vec3 sun_dir = normalize(sky_sun.xyz);
    float se = max(sun_dir.y, 0.02);
    vec4 zen = skyZenithState(se, turb);
    vec4 hor = skyHorizonState(se, turb, zen.a);
    return hor.rgb * hor.a * clamp(sky_night2.y, 0.0, 1.0) * 0.45;
}

// Night sky + clouds over an already-evaluated day dome. `ambient` is the
// Perez sky tint at this view (cloud base), `sun_tint` the attenuated sun
// color (silver-lining source). Visual-only: light/fog lanes untouched.
vec3 skyApplyCloudNight(vec3 col, vec3 v, vec3 sun_dir, vec3 sun_tint, vec3 ambient) {
    float na = clamp(sky_night.x, 0.0, 1.0);
    if (na > 0.0) {
        // Day sky sinks toward a near-black blue; the horizon keeps a dim
        // residue scaled by the night horizon behavior (0 = pitch black).
        vec3 night_sky = mix(horNightHorizon(), vec3(0.004, 0.006, 0.014), smoothstep(0.0, 0.5, v.y));
        col = mix(col, night_sky, na);
        // Hash star field: one cell one candidate, thresholded by density;
        // twinkle is a cheap sine (phase from the cell hash).
        float dens = clamp(sky_night.z, 0.0, 1.0);
        if (sky_night.y > 0.0 && dens > 0.0) {
            vec3 sp = v * 160.0;
            vec3 cell = floor(sp);
            float h = fract(sin(dot(cell, vec3(12.9898, 78.233, 37.719))) * 43758.5453);
            if (h > 1.0 - dens) {
                float d = length(fract(sp) - 0.5);
                float tsec = max(sky_cloud.z, sky_night2.z);
                float tw = 0.75 + 0.25 * sin(tsec * 3.0 + h * 40.0);
                float star = smoothstep(0.35, 0.0, d) * ((h - (1.0 - dens)) / max(dens, 1e-3)) * tw;
                col += vec3(0.9, 0.95, 1.0) * star * sky_night.y * na;
            }
        }
        // Analytic moon disc (full moon always: no phase term) with HDR gain
        // for bloom pickup, plus a tight halo lobe.
        vec3 moon_dir = normalize(sky_moon.xyz);
        float cos_m = dot(v, moon_dir);
        float disc = smoothstep(sky_moon.w, sky_moon.w + 0.0006, cos_m);
        col += vec3(0.92, 0.95, 1.0) * disc * sky_night.w * na;
        col += vec3(0.5, 0.6, 0.8) * pow(max(cos_m, 0.0), 800.0) * 0.3 * sky_night.w * na;
    }
    if (sky_cloud.w > 0.5) {
        // Dome compression: planar projection blows up at the horizon, so
        // the layer lives above the horizon band and fades into it.
        float up = max(v.y, 0.02);
        vec2 cuv = (v.xz / (up + 0.15) + sky_cloud2.xy) * sky_cloud.y;
        float n = clamp(skyFbm(cuv), 0.0, 1.0);
        float cov = clamp(sky_cloud.x, 0.0, 1.0);
        float m = smoothstep(1.0 - cov, 1.0 - cov + 0.25, n);
        m *= 1.0 - clamp(sky_night2.x, 0.0, 1.0) * (1.0 - smoothstep(0.0, 0.25, v.y));
        if (m > 0.0) {
            // Lit side: sun-tinted white; shaded side: ambient sky tint.
            // At night the sun term yields to a dim moon tint (fake, visual).
            vec3 lit = mix(sun_tint, vec3(0.55, 0.62, 0.78), na);
            float shade = 0.45 + 0.55 * n;
            vec3 cloud_col = (ambient * shade + lit * (0.75 + 0.45 * n)) * 0.5 * max(sky_cloud2.z, 0.0);
            cloud_col *= mix(1.0, 0.22, na);
            // Silver lining: forward-scatter lobe toward the sun/moon.
            vec3 silver_dir = normalize(mix(sun_dir, normalize(sky_moon.xyz), na));
            cloud_col += lit * pow(max(dot(v, silver_dir), 0.0), 6.0) * sky_cloud2.w * m * mix(1.0, 0.5, na);
            col = mix(col, max(cloud_col, vec3(0.0)), clamp(m, 0.0, 1.0));
        }
    }
    return max(col, vec3(0.0));
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
    col = max(col, vec3(0.0));
    // v2 extras (opt-in): skipped entirely unless clouds are enabled or the
    // night amount is positive, so the day dome stays bit-identical.
    if (sky_cloud.w > 0.5 || sky_night.x > 0.0) {
        col = skyApplyCloudNight(col, v, sun_dir, glow_tint, mix(hor.rgb, zen.rgb, smoothstep(0.0, 0.5, v.y)) * lum_r);
    }
    return col;
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
