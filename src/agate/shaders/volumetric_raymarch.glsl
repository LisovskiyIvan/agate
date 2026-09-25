// Volumetric light-shaft raymarch for agate (shaft layer v1).
// Low-resolution fullscreen march from the camera through each pixel to the
// scene-depth surface (sky pixels march the full span), sampling the SUN's
// CSM atlas once per step and accumulating single-scatter with a
// Henyey-Greenstein phase term and Beer-Lambert transmittance. The output
// is unscaled shaft radiance; intensity scaling and the additive composite
// happen in the postprocess composite (after the highlight block), so this
// pass needs no intensity input. Mirrors hgPhase/shaftCascadeIndex in
// postprocess.zig, which pin the exact formulas for CPU tests.
//
// Cascade selection, tile layout, and the non-GL NDC-Y flip mirror the
// forward shaders (standard.glsl sampleCascade/calculateShadow) exactly,
// so the march taps the same tile the surface shader uses. One raw-depth
// tap per step (no PCF in v1 — the bilateral blur eats the aliasing);
// outside all tiles or past the shadow range the sample reads lit (the
// forward fade-to-lit convention), never fake occlusion.
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
// @include "common/fullscreen_vs.glsl"
@end

@fs fs
layout(binding = 0) uniform fs_params {
    mat4 inv_view_proj; // camera inverse view-projection (jittered when TAA runs)
    mat4 cascade_vp[4]; // sun CSM cascade matrices (frame snapshot)
    vec4 camera_pos; // xyz: camera world pos, w: unused
    vec4 sun_dir; // xyz: sun direction TOWARD the sun (normalized), w: unused
    vec4 sun_color; // rgb: sun color, w: unused
    vec4 splits; // xyzw: cascade split distances (camera space)
    vec4 march_a; // x: steps, y: density (/m), z: max march distance (m), w: HG anisotropy g
    vec4 march_b; // x: depth bias, yzw: reserved
};

// 2x2 cascade tiles of the 2048 CSM atlas (passes/shadow/types.zig):
// cascade 0 at (0,0), 1 at (0.5,0), 2 at (0,0.5), 3 at (0.5,0.5).
// Same table as the forward shaders.
const vec2 CASCADE_OFFSETS[4] = vec2[](
    vec2(0.0, 0.0),
    vec2(0.5, 0.0),
    vec2(0.0, 0.5),
    vec2(0.5, 0.5)
);

@image_sample_type depth_tex unfilterable_float
layout(binding = 0) uniform texture2D depth_tex;
@image_sample_type shadow_tex unfilterable_float
layout(binding = 1) uniform texture2D shadow_tex;
@sampler_type depth_smp nonfiltering
layout(binding = 0) uniform sampler depth_smp;

in vec2 v_uv;
out vec4 frag_color;

// Henyey-Greenstein phase term. Mirrors hgPhase in postprocess.zig.
float hgPhase(float cos_theta, float g) {
    float gg = g * g;
    float denom = 1.0 + gg - 2.0 * g * cos_theta;
    return (1.0 - gg) / (4.0 * 3.14159265 * denom * sqrt(max(denom, 0.000001)));
}

// CSM cascade index for a camera-distance sample. Mirrors
// calculateShadow in the forward shaders (and shaftCascadeIndex in
// postprocess.zig).
int shaftCascade(float view_dist) {
    if (view_dist < splits.x) return 0;
    if (view_dist < splits.y) return 1;
    if (view_dist < splits.z) return 2;
    return 3;
}

// Single-tap CSM occlusion at a marched point: 0.0 blocked, 1.0 lit.
// Same tile math as forward sampleCascade (minus normal offset — march
// samples float in free space, not on a shaded surface).
float shaftShadow(vec3 p, float view_dist, float bias) {
    int ci = shaftCascade(view_dist);
    vec4 lpos = cascade_vp[ci] * vec4(p, 1.0);
    #if !SOKOL_GLSL
        lpos.y = -lpos.y;
    #endif
    vec3 proj = lpos.xyz / lpos.w;
    if (proj.z > 1.0 || proj.z < 0.0) return 1.0;
    vec2 local_uv = (proj.xy + 1.0) * 0.5;
    if (local_uv.x < 0.0 || local_uv.x > 1.0 || local_uv.y < 0.0 || local_uv.y > 1.0) return 1.0;
    vec2 atlas_uv = clamp(local_uv, 0.003, 0.997) * 0.5 + CASCADE_OFFSETS[ci];
    float stored = texture(sampler2D(shadow_tex, depth_smp), atlas_uv).r;
    float lit = (stored >= proj.z - bias) ? 1.0 : 0.0;
    // Forward fade-to-lit past the shadow range: no data, no occlusion.
    float fade_start = splits.w * 0.85;
    if (view_dist > fade_start) {
        float f = clamp((view_dist - fade_start) / max(splits.w - fade_start, 0.001), 0.0, 1.0);
        lit = mix(lit, 1.0, f);
    }
    return lit;
}

void main() {
    float raw = texture(sampler2D(depth_tex, depth_smp), v_uv).r;
    vec4 clip = vec4(v_uv.x * 2.0 - 1.0, 1.0 - v_uv.y * 2.0, raw, 1.0);
    vec4 world_h = inv_view_proj * clip;
    vec3 surface = world_h.xyz / max(world_h.w, 0.0001);
    vec3 to_surf = surface - camera_pos.xyz;
    float surf_dist = length(to_surf);
    vec3 dir = (surf_dist > 0.0001) ? (to_surf / surf_dist) : vec3(0.0, 0.0, 1.0);
    // Sky (cleared depth) marches the full span; surfaces clamp to the
    // nearer of the surface and the span cap.
    float dist = (raw >= 0.9999) ? march_a.z : min(surf_dist, march_a.z);

    int steps = int(march_a.x + 0.5);
    if (steps < 1 || dist <= 0.0) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    float dt = dist / float(steps);
    float phase = hgPhase(dot(dir, sun_dir.xyz), march_a.w);
    float density = max(march_a.y, 0.0);
    float bias = march_b.x;

    float transmit = 1.0;
    float scat = 0.0;
    for (int i = 0; i < 32; i++) {
        if (i >= steps) break;
        float t = (float(i) + 0.5) * dt;
        vec3 p = camera_pos.xyz + dir * t;
        // length(p - camera_pos) == t (dir is normalized): the cascade
        // select matches the forward view-distance convention exactly.
        float lit = shaftShadow(p, t, bias);
        scat += transmit * lit * density * dt;
        transmit *= exp(-density * dt);
    }
    frag_color = vec4(sun_color.rgb * (phase * scat), 1.0);
}
@end

@program volumetric_raymarch vs fs
