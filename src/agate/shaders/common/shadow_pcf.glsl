float hash01(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// PCSS blocker search: average depth of taps closer to the light than the
// receiver, or -1.0 when nothing blocks (caller early-outs to fully lit).
float pcssBlockerAverage(texture2D depth_tex, sampler depth_smp, vec2 atlas_uv, float receiver_depth, mat2 rot, float search_radius, vec2 quad_min, vec2 quad_max) {
    float blocker_sum = 0.0;
    int blocker_count = 0;
    for (int i = 0; i < PCSS_BLOCKER_SAMPLES; i++) {
        vec2 tap_uv = clamp(atlas_uv + rot * POISSON_DISK[i] * search_radius, quad_min, quad_max);
        float tap_depth = texture(sampler2D(depth_tex, depth_smp), tap_uv).r;
        if (tap_depth < receiver_depth) {
            blocker_sum += tap_depth;
            blocker_count += 1;
        }
    }
    if (blocker_count == 0) return -1.0;
    return blocker_sum / float(blocker_count);
}

// PCSS variable penumbra, atlas-UV radius for the PCF disk. Mirrors
// penumbraRadius in scene/shadow_pcss.zig (parallel rays: linear scaling without perspective division).
float pcssPenumbraRadius(float receiver_depth, float blocker_avg, float light_size, float min_penumbra, float max_penumbra) {
    float penumbra = (receiver_depth - blocker_avg) * light_size;
    return clamp(penumbra, min_penumbra, max_penumbra);
}

float sampleCascade(int cascade_idx, vec3 world_pos, vec3 N, vec3 L) {
    float cos_theta = max(dot(N, L), 0.0);
    float depth_bias = max(shadow_params.x * (1.0 - cos_theta), shadow_params.x * 0.2);
    vec3 normal_offset = N * (shadow_params.z * (1.0 - cos_theta));

    vec4 lpos = cascade_view_proj[cascade_idx] * vec4(world_pos + normal_offset, 1.0);
    #if !SOKOL_GLSL
        lpos.y = -lpos.y;
    #endif

    vec3 proj = lpos.xyz / lpos.w;
    if (proj.z > 1.0 || proj.z < 0.0) return 1.0;

    vec2 local_uv = (proj.xy + 1.0) * 0.5;
    if (local_uv.x < 0.0 || local_uv.x > 1.0 || local_uv.y < 0.0 || local_uv.y > 1.0) return 1.0;

    vec2 clamped_local_uv = clamp(local_uv, 0.003, 0.997);
    vec2 atlas_uv = clamped_local_uv * 0.5 + CASCADE_OFFSETS[cascade_idx];
    float depth = proj.z - depth_bias;

    vec2 quad_min = CASCADE_OFFSETS[cascade_idx] + vec2(0.003);
    vec2 quad_max = CASCADE_OFFSETS[cascade_idx] + vec2(0.497);

    // Fixed 90-degree rotations keyed by screen hash: same dithering as a random
    // rotation, zero trig per fragment.
    float h = hash01(gl_FragCoord.xy);
    mat2 rot = mat2(1.0, 0.0, 0.0, 1.0);
    if (h > 0.75) {
        rot = mat2(0.0, 1.0, -1.0, 0.0);
    } else if (h > 0.5) {
        rot = mat2(-1.0, 0.0, 0.0, -1.0);
    } else if (h > 0.25) {
        rot = mat2(0.0, -1.0, 1.0, 0.0);
    }

    float filter_radius = (shadow_params.w / SHADOW_ATLAS_SIZE) * 0.5;

    // Full 16-tap PCF only for the near cascade; far cascades cover huge texels
    // where extra taps cost without visible quality.
    int taps = 16;
    if (cascade_idx > 0) taps = 8;

    // PCSS: blocker search sets a receiver-dependent filter radius. Params
    // ride free lanes (cascade_debug.yzw / light_counts.zw); disabled keeps
    // the fixed-radius path below bit-identical.
    if (cascade_debug.y > 0.5) {
        float blocker_avg = pcssBlockerAverage(shadow_depth_tex, depth_smp, atlas_uv, depth, rot, cascade_debug.w, quad_min, quad_max);
        if (blocker_avg < 0.0) return 1.0;
        filter_radius = pcssPenumbraRadius(depth, blocker_avg, cascade_debug.z, light_counts.z, light_counts.w);
    }

    float lit = 0.0;
    for (int i = 0; i < 16; i++) {
        if (i >= taps) break;
        vec2 offset = rot * POISSON_DISK[i] * filter_radius;
        vec2 sample_uv = clamp(atlas_uv + offset, quad_min, quad_max);
        lit += texture(sampler2DShadow(shadow_tex, shadow_smp), vec3(sample_uv, depth));
    }
    return lit / float(taps);
}

float calculateShadow(vec3 world_pos, vec3 N, vec3 L, out vec3 debug_color) {
    debug_color = vec3(0.0);
    if (shadow_params.y <= 0.001) return 0.0;

    float view_dist = length(world_pos - eye_pos.xyz);
    int cascade_idx = 3;
    if (view_dist < shadow_splits.x) {
        cascade_idx = 0;
    } else if (view_dist < shadow_splits.y) {
        cascade_idx = 1;
    } else if (view_dist < shadow_splits.z) {
        cascade_idx = 2;
    }

    if (cascade_debug.x > 0.5) {
        if (cascade_idx == 0) debug_color = vec3(0.25, 0.05, 0.05);
        else if (cascade_idx == 1) debug_color = vec3(0.05, 0.25, 0.05);
        else if (cascade_idx == 2) debug_color = vec3(0.05, 0.05, 0.25);
        else debug_color = vec3(0.25, 0.25, 0.05);
    }

    float lit = sampleCascade(cascade_idx, world_pos, N, L);

    float fade_start = shadow_splits.w * 0.85;
    if (view_dist > fade_start) {
        float fade = clamp((view_dist - fade_start) / max(shadow_splits.w - fade_start, 0.001), 0.0, 1.0);
        lit = mix(lit, 1.0, fade);
    }

    return (1.0 - lit) * shadow_params.y;
}

float calculateSpotShadow(int spot_idx, vec3 world_pos, vec3 N, vec3 L) {
    if (spot_shadow_params[spot_idx].x < 0.5) return 0.0;
    if (shadow_params.y <= 0.001) return 0.0;

    float cos_theta = max(dot(N, L), 0.0);
    float depth_bias = max(spot_shadow_params[spot_idx].y * (1.0 - cos_theta), spot_shadow_params[spot_idx].y * 0.2);
    vec3 normal_offset = N * (spot_shadow_params[spot_idx].z * (1.0 - cos_theta));

    vec4 lpos = spot_view_proj[spot_idx] * vec4(world_pos + normal_offset, 1.0);
    #if !SOKOL_GLSL
        lpos.y = -lpos.y;
    #endif

    vec3 proj = lpos.xyz / lpos.w;
    if (proj.z > 1.0 || proj.z < 0.0) return 0.0;

    vec2 local_uv = (proj.xy + 1.0) * 0.5;
    if (local_uv.x < 0.0 || local_uv.x > 1.0 || local_uv.y < 0.0 || local_uv.y > 1.0) return 0.0;

    vec2 clamped_uv = clamp(local_uv, 0.002, 0.998);
    vec2 atlas_uv = vec2(clamped_uv.x * 0.5 + float(spot_idx) * 0.5, clamped_uv.y);
    float depth = proj.z - depth_bias;

    vec2 texel = vec2(1.0 / 1024.0, 1.0 / 512.0);
    float lit = 0.0;
    lit += texture(sampler2DShadow(spot_shadow_tex, shadow_smp), vec3(atlas_uv + vec2(-texel.x, -texel.y), depth));
    lit += texture(sampler2DShadow(spot_shadow_tex, shadow_smp), vec3(atlas_uv + vec2( texel.x, -texel.y), depth));
    lit += texture(sampler2DShadow(spot_shadow_tex, shadow_smp), vec3(atlas_uv + vec2(-texel.x,  texel.y), depth));
    lit += texture(sampler2DShadow(spot_shadow_tex, shadow_smp), vec3(atlas_uv + vec2( texel.x,  texel.y), depth));
    lit *= 0.25;

    return (1.0 - lit) * shadow_params.y;
}

int pointFaceIndex(vec3 d) {
    vec3 a = abs(d);
    if (a.x >= a.y && a.x >= a.z) return d.x >= 0.0 ? 0 : 1;
    if (a.y >= a.x && a.y >= a.z) return d.y >= 0.0 ? 2 : 3;
    return d.z >= 0.0 ? 4 : 5;
}

// Point-light shadows: each shadow-casting point light owns 6 cube-face
// tiles (one atlas row, 256px tiles in the 1536x512 point atlas). The face
// is picked from the world-space direction to the light, then the fragment
// is projected by that face's view-projection matrix and PCF-sampled with
// the same 2D compare path as the spot atlas (4 taps + bias, shadow
// strength from shadow_params.y like every other shadow term here).
float calculatePointShadow(int light_idx, vec3 world_pos, vec3 N, vec3 L) {
    if (point_shadow_params[light_idx].x < 0.5) return 0.0;
    if (shadow_params.y <= 0.001) return 0.0;

    int slot = int(point_shadow_params[light_idx].x - 1.0);
    vec3 to_frag = world_pos - point_pos_range[light_idx].xyz;
    int face = pointFaceIndex(to_frag);

    float cos_theta = max(dot(N, L), 0.0);
    float bias = point_shadow_params[light_idx].y;
    float depth_bias = max(bias * (1.0 - cos_theta), bias * 0.2);
    vec3 normal_offset = N * (point_shadow_params[light_idx].z * (1.0 - cos_theta));

    vec4 lpos = point_view_proj[slot * 6 + face] * vec4(world_pos + normal_offset, 1.0);
    #if !SOKOL_GLSL
        lpos.y = -lpos.y;
    #endif

    vec3 proj = lpos.xyz / lpos.w;
    if (proj.z > 1.0 || proj.z < 0.0) return 0.0;

    vec2 local_uv = (proj.xy + 1.0) * 0.5;
    if (local_uv.x < 0.0 || local_uv.x > 1.0 || local_uv.y < 0.0 || local_uv.y > 1.0) return 0.0;

    // Point atlas layout (must match ShadowPass.pointTileOrigin):
    // 6 faces left-to-right, one row per shadow slot.
    vec2 clamped_uv = clamp(local_uv, 0.002, 0.998);
    vec2 atlas_uv = vec2((float(face) + clamped_uv.x) / 6.0, float(slot) * 0.5 + clamped_uv.y * 0.5);
    float depth = proj.z - depth_bias;

    vec2 texel = vec2(1.0 / 1536.0, 1.0 / 512.0);
    float lit = 0.0;
    lit += texture(sampler2DShadow(point_shadow_tex, shadow_smp), vec3(atlas_uv + vec2(-texel.x, -texel.y), depth));
    lit += texture(sampler2DShadow(point_shadow_tex, shadow_smp), vec3(atlas_uv + vec2( texel.x, -texel.y), depth));
    lit += texture(sampler2DShadow(point_shadow_tex, shadow_smp), vec3(atlas_uv + vec2(-texel.x,  texel.y), depth));
    lit += texture(sampler2DShadow(point_shadow_tex, shadow_smp), vec3(atlas_uv + vec2( texel.x,  texel.y), depth));
    lit *= 0.25;

    return (1.0 - lit) * shadow_params.y;
}

// Rect area-light irradiance v1 (wave 26): analytic approximation, NOT LTC
// and NOT a multi-sample integration. The fragment is lit by the closest
// point Q on the rect (parallelogram projection onto the right/up
// half-extent axes); the returned factor is
//   emit * area / (dist^2 + area) * intensity
// with emit = clamp(dot(rect_normal, -L)) (single-sided front-face
// emission) and NdotL as an out-param for the caller's lobe. Limits,
// stated honestly: no LTC lobe, so large/close rects shade harder-edged
// than reality; no rect-shape specular anisotropy (the standard path uses
// a Blinn-Phong boost from the same representative direction L, the PBR
// path reuses its Cook-Torrance lobe); no shadows — an occluded area
// light still lights (v1 scope). Zero intensity or zero area returns 0,
// so zero lights are a bit-identical no-op.
float areaLightFactor(vec3 frag_pos, vec3 N, int area_idx, out vec3 L, out float NdotL) {
    float a_int = area_center_int[area_idx].w;
    vec3 r = area_right[area_idx].xyz;
    vec3 u = area_up[area_idx].xyz;
    vec3 naxis = cross(r, u);
    float rect_area = 4.0 * length(naxis);
    L = N;
    NdotL = 0.0;
    if (a_int <= 0.0 || rect_area <= 1e-8) return 0.0;
    vec3 nrect = naxis / (rect_area * 0.25);
    vec3 c = area_center_int[area_idx].xyz;
    vec3 d = frag_pos - c;
    float x = clamp(dot(d, r) / max(dot(r, r), 1e-6), -1.0, 1.0);
    float y = clamp(dot(d, u) / max(dot(u, u), 1e-6), -1.0, 1.0);
    vec3 to_light = (c + r * x + u * y) - frag_pos;
    float dist = length(to_light);
    L = to_light / max(dist, 1e-4);
    NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) return 0.0;
    float emit = clamp(dot(nrect, -L), 0.0, 1.0);
    if (emit <= 0.0) return 0.0;
    float att = rect_area / (dist * dist + rect_area);
    return emit * att * a_int;
}
