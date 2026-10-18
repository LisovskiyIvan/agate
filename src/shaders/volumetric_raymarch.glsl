// Volumetric light-shaft raymarch for agate (shaft layer v1).
// Low-resolution fullscreen march from the camera through each pixel to the
// scene-depth surface (sky pixels march the full span), sampling the SUN's
// CSM atlas with a 2x2 PCF footprint per step and accumulating single-scatter with a
// Henyey-Greenstein phase term and Beer-Lambert transmittance. The output
// is unscaled shaft radiance; intensity scaling and the additive composite
// happen in the postprocess composite (after the highlight block), so this
// pass needs no intensity input. Mirrors hgPhase/shaftCascadeIndex in
// postprocess.zig, which pin the exact formulas for CPU tests.
//
// Cascade selection, tile layout, and the non-GL NDC-Y flip mirror the
// forward shaders (common/shadow_pcf.glsl sampleCascade/calculateShadow) exactly,
// so the march taps the same tile the surface shader uses. Compare depths
// before bilinear interpolation: filtering raw depth invents occluders.
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

float shaftDepthCompare(vec2 uv, float depth) {
    float stored = texture(sampler2D(shadow_tex, depth_smp), uv).r;
    return stored >= depth ? 1.0 : 0.0;
}

// Bilinear 2x2 PCF at a marched point. Every fetch stays inside its own
// tile; averaging comparisons removes the nearest-texel rectangular steps.
float shaftShadowCascade(int ci, vec3 p, float bias, out bool in_bounds) {
    in_bounds = false;
    vec4 lpos = cascade_vp[ci] * vec4(p, 1.0);
    #if !SOKOL_GLSL
        lpos.y = -lpos.y;
    #endif
    vec3 proj = lpos.xyz / lpos.w;
    if (proj.z > 1.0 || proj.z < 0.0) return 1.0;
    vec2 local_uv = (proj.xy + 1.0) * 0.5;
    if (local_uv.x < 0.0 || local_uv.x > 1.0 || local_uv.y < 0.0 || local_uv.y > 1.0) return 1.0;
    in_bounds = true;
    vec2 tile_min = CASCADE_OFFSETS[ci] + vec2(0.5 / 2048.0);
    vec2 tile_max = CASCADE_OFFSETS[ci] + vec2(0.5 - 0.5 / 2048.0);
    vec2 atlas_uv = clamp(local_uv * 0.5 + CASCADE_OFFSETS[ci], tile_min, tile_max);
    vec2 texel_pos = atlas_uv * 2048.0 - 0.5;
    vec2 base = floor(texel_pos);
    vec2 weight = fract(texel_pos);
    vec2 uv00 = clamp((base + vec2(0.5, 0.5)) / 2048.0, tile_min, tile_max);
    vec2 uv10 = clamp((base + vec2(1.5, 0.5)) / 2048.0, tile_min, tile_max);
    vec2 uv01 = clamp((base + vec2(0.5, 1.5)) / 2048.0, tile_min, tile_max);
    vec2 uv11 = clamp((base + vec2(1.5, 1.5)) / 2048.0, tile_min, tile_max);
    float depth = proj.z - bias;
    return mix(mix(shaftDepthCompare(uv00, depth), shaftDepthCompare(uv10, depth), weight.x),
               mix(shaftDepthCompare(uv01, depth), shaftDepthCompare(uv11, depth), weight.x), weight.y);
}

float shaftShadow(vec3 p, float view_dist, float bias) {
    int ci = shaftCascade(view_dist);
    bool in_bounds;
    float lit = 1.0;
    for (int candidate = 0; candidate < 4; candidate++) {
        if (candidate < ci) continue;
        lit = shaftShadowCascade(candidate, p, bias, in_bounds);
        if (in_bounds) break;
    }

    if (ci < 3) {
        float split_dist = (ci == 0) ? splits.x : ((ci == 1) ? splits.y : splits.z);
        float blend_margin = split_dist * 0.12;
        if (view_dist > split_dist - blend_margin) {
            float blend_factor = clamp((view_dist - (split_dist - blend_margin)) / max(blend_margin, 0.001), 0.0, 1.0);
            // Walk the remaining cascades exactly like the forward
            // sampleCascade: the first in-bounds cascade above ci wins.
            // Testing only ci + 1 read "lit" whenever that cascade was out
            // of bounds while a farther one still covered the point, which
            // left a bright seam at split distances.
            bool next_bounds = false;
            float lit_next = 1.0;
            for (int candidate = ci + 1; candidate < 4; candidate++) {
                lit_next = shaftShadowCascade(candidate, p, bias, next_bounds);
                if (next_bounds) break;
            }
            if (next_bounds) {
                lit = mix(lit, lit_next, blend_factor);
            }
        }
    }

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
    float step_transmit = exp(-density * dt);
    float step_scatter = 1.0 - step_transmit;
    float jitter = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    for (int i = 0; i < 32; i++) {
        if (i >= steps) break;
        float t = (float(i) + jitter) * dt;
        vec3 p = camera_pos.xyz + dir * t;
        // length(p - camera_pos) == t (dir is normalized): the cascade
        // select matches the forward view-distance convention exactly.
        float lit = shaftShadow(p, t, bias);
        // Exact Beer-Lambert integral over this segment, not density*dt:
        // a fully-lit ray keeps the same energy at any march step count.
        scat += transmit * lit * step_scatter;
        transmit *= step_transmit;
    }
    frag_color = vec4(sun_color.rgb * (phase * scat), 1.0);
}
@end

@program volumetric_raymarch vs fs
