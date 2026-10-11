// DDGI v1 spike: probe-atlas update (fragment-based, no compute).
//
// One texel = one ray: the fragment derives its probe index + ray
// direction from v_uv (texel-exact on every backend: v_uv (0,0) addresses
// memory texel (0,0), the same texel the sample side computes UVs for —
// the SSGI v2 gather/composite pair relies on the same convention).
// Texel addressing mirrors gi/ddgi.zig exactly (atlasTileOrigin,
// probeRayDir, probePosition): probe p = tile (p%8, p/8), ray dir =
// octDecode(texel_center / 8).
//
// Per ray: march the proxy list (slab tests, DDGI_MAX_PROXIES loop bound
// with an early break at the uploaded count, SSGI-gather-shaped). Miss ->
// hemispheric sky ambient (the scene hemi lanes, same model as
// common/hemi.glsl). Hit -> proxy albedo x direct sun when the secondary
// march toward the sun is unblocked (proxies only, no shadow-map reads)
// + sky ambient. Single bounce only; alpha is 1.
//
// The pass renders with LOAD (atlas persists across frames) and a probe
// window uniform (ddgi_window): whole-grid refresh or a round-robin slice
// (CPU mirror: ddgi.budgetWindow). Texels outside this frame's window
// emit a no-op (same value cannot be re-emitted without a read, so the
// shader DISCARDS — with LOAD + discard the old value survives).
@header const m = @import("math")
@ctype mat4 m.Mat4

@vs vs
@glsl_options fixup_clipspace
// @include "common/fullscreen_vs.glsl"
@end

@fs fs
layout(binding = 0) uniform fs_params {
    vec4 ddgi_grid_origin; // xyz: grid origin, w: enabled (1; the pass only runs when on)
    vec4 ddgi_grid_counts; // xyz: probe counts, w: unused (intensity stays sample-side)
    vec4 ddgi_grid_step; // xyz: cell size, w: tile texels (8)
    vec4 ddgi_atlas; // xy: atlas size px, z: tiles per row, w: proxy count
    vec4 ddgi_window; // x: first probe this frame, y: probe count this frame, z: total probes, w: unused
    vec4 ddgi_sun_dir; // xyz: direction TO the sun, w: sun intensity
    vec4 ddgi_sun_color; // rgb: sun color, w: unused
    vec4 ddgi_sky; // rgb: hemi sky diffuse, w: hemi intensity
    vec4 ddgi_sky_ground; // rgb: hemi ground color, w: unused
    vec4 ddgi_sky_dir; // xyz: hemi direction, w: unused
    vec4 ddgi_proxy_min[64]; // xyz: box min (CPU mirror: ddgi.ProxyBox)
    vec4 ddgi_proxy_max[64]; // xyz: box max
    vec4 ddgi_proxy_albedo[64]; // rgb: linear proxy albedo
};

in vec2 v_uv;
out vec4 frag_color;

// Proxy cap: loop bound below (CPU mirror: ddgi.MAX_PROXIES). The
// uploaded count breaks early; the cap only bounds the worst case.
const int DDGI_MAX_PROXIES = 64;
// Self-hit guard (m): hits closer than this are ignored so probes
// resting on proxy surfaces do not shadow themselves (CPU mirror:
// ddgi.SELF_HIT_MIN_T).
const float DDGI_SELF_T = 0.05;

vec3 ddgiOctDecode(vec2 o) {
    vec2 p = o * 2.0 - 1.0;
    vec3 d = vec3(p.x, p.y, 1.0 - abs(p.x) - abs(p.y));
    if (d.z < 0.0) {
        float ox = (1.0 - abs(d.y)) * (d.x >= 0.0 ? 1.0 : -1.0);
        float oy = (1.0 - abs(d.x)) * (d.y >= 0.0 ? 1.0 : -1.0);
        d.x = ox;
        d.y = oy;
    }
    float len = length(d);
    return (len > 1e-9) ? (d / len) : vec3(0.0, 0.0, 1.0);
}

// Slab march over the proxy list: nearest t >= DDGI_SELF_T, hit proxy
// index in hit_idx (-1 on miss). inv_dir guards zero lanes like the CPU
// mirror (ddgi.rayAabbT).
float ddgiMarch(vec3 origin, vec3 inv_dir, int proxy_count, out int hit_idx) {
    float best_t = 1e30;
    hit_idx = -1;
    for (int i = 0; i < DDGI_MAX_PROXIES; i++) {
        if (i >= proxy_count) break;
        vec3 t0 = (ddgi_proxy_min[i].xyz - origin) * inv_dir;
        vec3 t1 = (ddgi_proxy_max[i].xyz - origin) * inv_dir;
        vec3 tmin3 = min(t0, t1);
        vec3 tmax3 = max(t0, t1);
        float tmin = max(max(tmin3.x, tmin3.y), tmin3.z);
        float tmax = min(min(tmax3.x, tmax3.y), tmax3.z);
        if (tmin <= tmax && tmax >= DDGI_SELF_T && tmin < best_t) {
            best_t = tmin;
            hit_idx = i;
        }
    }
    return best_t;
}

vec3 ddgiSkyAmbient(vec3 d) {
    float mix_factor = dot(d, ddgi_sky_dir.xyz) * 0.5 + 0.5;
    return mix(ddgi_sky_ground.rgb * ddgi_sky.w, ddgi_sky.rgb * ddgi_sky.w, mix_factor);
}

void main() {
    vec2 atlas_px = ddgi_atlas.xy;
    vec2 texel = floor(v_uv * atlas_px);
    float tile = ddgi_grid_step.w;
    float per_row = ddgi_atlas.z;
    vec2 tile_xy = floor(texel / tile);
    float probe_idx = tile_xy.x + tile_xy.y * per_row;
    if (probe_idx < ddgi_window.x || probe_idx >= ddgi_window.x + ddgi_window.y || probe_idx >= ddgi_window.z) {
        discard;
    }
    int proxy_count = int(ddgi_atlas.w + 0.5);

    // Probe center (CPU mirror: ddgi.probePosition, x-fastest).
    vec3 counts = ddgi_grid_counts.xyz;
    float pz = floor(probe_idx / (counts.x * counts.y));
    float py = floor((probe_idx - pz * counts.x * counts.y) / counts.x);
    float px = probe_idx - pz * counts.x * counts.y - py * counts.x;
    vec3 probe_pos = ddgi_grid_origin.xyz + (vec3(px, py, pz) + 0.5) * ddgi_grid_step.xyz;

    vec2 local = texel - tile_xy * tile;
    vec3 dir = ddgiOctDecode((local + 0.5) / tile);
    vec3 inv_dir = 1.0 / (abs(dir) + vec3(1e-9)) * sign(dir + vec3(1e-12));

    int hit_idx = -1;
    float t = ddgiMarch(probe_pos, inv_dir, proxy_count, hit_idx);
    if (hit_idx < 0 || t > 1e29) {
        frag_color = vec4(ddgiSkyAmbient(dir), 1.0);
        return;
    }
    vec3 hit_pos = probe_pos + dir * max(t, DDGI_SELF_T);
    // Face normal of the hit slab: the axis with the largest entry-plane
    // distance (recomputed, not interpolated — AABBs have flat faces).
    vec3 rel = (hit_pos - 0.5 * (ddgi_proxy_min[hit_idx].xyz + ddgi_proxy_max[hit_idx].xyz)) /
        max(0.5 * (ddgi_proxy_max[hit_idx].xyz - ddgi_proxy_min[hit_idx].xyz), vec3(1e-9));
    vec3 ax = abs(rel);
    vec3 face_n = (ax.x >= ax.y && ax.x >= ax.z) ? vec3(sign(rel.x), 0.0, 0.0)
        : (ax.y >= ax.z) ? vec3(0.0, sign(rel.y), 0.0)
        : vec3(0.0, 0.0, sign(rel.z));
    vec3 albedo = ddgi_proxy_albedo[hit_idx].rgb;
    // Secondary visibility march toward the sun (proxies only): blocked ->
    // ambient only, no direct sun. Self-hit guarded from the surface.
    vec3 sun_origin = hit_pos + face_n * DDGI_SELF_T * 2.0;
    vec3 sun_dir = normalize(ddgi_sun_dir.xyz);
    vec3 sun_inv = 1.0 / (abs(sun_dir) + vec3(1e-9)) * sign(sun_dir + vec3(1e-12));
    int sun_hit = -1;
    ddgiMarch(sun_origin, sun_inv, proxy_count, sun_hit);
    float sun_nl = max(dot(face_n, sun_dir), 0.0);
    vec3 direct = (sun_hit < 0) ? (albedo * ddgi_sun_color.rgb * ddgi_sun_dir.w * sun_nl) : vec3(0.0);
    // Sky ambient is analytic (unoccluded by construction, like the PBR
    // hemi term): full strength, albedo-modulated.
    frag_color = vec4(direct + albedo * ddgiSkyAmbient(face_n), 1.0);
}
@end

@program ddgi_update vs fs
