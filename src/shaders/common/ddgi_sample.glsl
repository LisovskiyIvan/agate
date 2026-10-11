// DDGI v1 spike: probe-atlas diffuse sample (shared PBR chunk).
//
// Included by the three PBR forward fragment shaders next to the ambient
// term. Single tap per probe along the surface normal, trilinearly
// blended over the 8 surrounding probes (CPU mirror: ddgi.probeBlend)
// — a diffuse proxy, NOT a hemisphere integral (documented spike limit,
// see docs/gi.md). NEAREST sampling, no guard bands: bilinear would
// bleed across tile borders.
//
// Uniform lanes (frame_view.glsl, appended last so no existing offset
// shifts; FrameViewUniforms mirrors; zeros when disabled):
//   ddgi_grid_origin: xyz grid origin, w enabled (0/1)
//   ddgi_grid_counts: xyz probe counts, w sample intensity
//   ddgi_grid_step:   xyz cell size, w tile texels (8)
//   ddgi_atlas:       xy atlas size px, z tiles per row, w unused
// Texture/sampler (pbr_fs_bindings.glsl): ddgi_tex (view 22), ddgi_smp
// (sampler 9, nearest+clamp). The draw always binds something valid
// (atlas view when the update pass ran, default-white otherwise); the
// shader only samples when the enabled lane is set.
vec2 ddgiOctEncodeS(vec3 d) {
    float inv_sum = 1.0 / max(abs(d.x) + abs(d.y) + abs(d.z), 1e-9);
    vec2 p = d.xy * inv_sum;
    if (d.z < 0.0) {
        float ox = (1.0 - abs(p.y)) * (p.x >= 0.0 ? 1.0 : -1.0);
        float oy = (1.0 - abs(p.x)) * (p.y >= 0.0 ? 1.0 : -1.0);
        p = vec2(ox, oy);
    }
    return p * 0.5 + 0.5;
}

vec2 ddgiProbeUv(float probe_idx, vec3 d) {
    float tile = ddgi_grid_step.w;
    float per_row = ddgi_atlas.z;
    vec2 o = clamp(ddgiOctEncodeS(d), vec2(0.0), vec2(1.0));
    vec2 local = clamp(floor(o * tile), vec2(0.0), vec2(tile - 1.0));
    float tx = mod(probe_idx, per_row);
    float ty = floor(probe_idx / per_row);
    vec2 texel = vec2(tx, ty) * tile + local;
    return (texel + 0.5) / ddgi_atlas.xy;
}

// Diffuse bounce irradiance at a world position for normal n, in
// surface-albedo units (the caller multiplies by albedo*(1-metallic),
// mirroring the hemi ambient term). Zero when disabled — before any
// texture fetch, so the off path is bit-identical.
vec3 ddgiBounce(vec3 world_pos, vec3 n) {
    if (ddgi_grid_origin.w < 0.5) {
        return vec3(0.0);
    }
    vec3 rel = (world_pos - ddgi_grid_origin.xyz) / max(ddgi_grid_step.xyz, vec3(1e-9)) - 0.5;
    vec3 counts = ddgi_grid_counts.xyz;
    vec3 gc = clamp(rel, vec3(0.0), max(counts - 1.0, vec3(0.0)));
    vec3 b = vec3(0.0);
    vec3 f = vec3(0.0);
    // Per-axis lower corner + fraction (single-probe axes do not blend).
    // Unrolled by axis: dynamic axis counts would need non-constant loop
    // bounds the HLSL5 floor cannot flatten (same constraint as
    // channel_select/morphWeight).
    b.x = (counts.x > 1.5) ? min(floor(gc.x), counts.x - 2.0) : 0.0;
    f.x = (counts.x > 1.5) ? clamp(gc.x - b.x, 0.0, 1.0) : 0.0;
    b.y = (counts.y > 1.5) ? min(floor(gc.y), counts.y - 2.0) : 0.0;
    f.y = (counts.y > 1.5) ? clamp(gc.y - b.y, 0.0, 1.0) : 0.0;
    b.z = (counts.z > 1.5) ? min(floor(gc.z), counts.z - 2.0) : 0.0;
    f.z = (counts.z > 1.5) ? clamp(gc.z - b.z, 0.0, 1.0) : 0.0;
    vec3 acc = vec3(0.0);
    for (int k = 0; k < 8; k++) {
        float dx = float(k & 1);
        float dy = float((k >> 1) & 1);
        float dz = float((k >> 2) & 1);
        float ix = min(b.x + dx, counts.x - 1.0);
        float iy = min(b.y + dy, counts.y - 1.0);
        float iz = min(b.z + dz, counts.z - 1.0);
        float probe_idx = ix + counts.x * (iy + counts.y * iz);
        float wx = (dx < 0.5) ? (1.0 - f.x) : f.x;
        float wy = (dy < 0.5) ? (1.0 - f.y) : f.y;
        float wz = (dz < 0.5) ? (1.0 - f.z) : f.z;
        acc += texture(sampler2D(ddgi_tex, ddgi_smp), ddgiProbeUv(probe_idx, n)).rgb * (wx * wy * wz);
    }
    return acc * ddgi_grid_counts.w;
}
