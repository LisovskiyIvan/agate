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

// Fixed 90-degree dither rotation keyed by screen hash: same dithering as a
// random rotation, zero trig per fragment. Spot-PCSS-only helper (the sun
// path keeps its inline rotation above untouched).
mat2 pcssDitherRotation(vec2 frag_coord) {
    float h = hash01(frag_coord);
    mat2 rot = mat2(1.0, 0.0, 0.0, 1.0);
    if (h > 0.75) {
        rot = mat2(0.0, 1.0, -1.0, 0.0);
    } else if (h > 0.5) {
        rot = mat2(-1.0, 0.0, 0.0, -1.0);
    } else if (h > 0.25) {
        rot = mat2(0.0, -1.0, 1.0, 0.0);
    }
    return rot;
}

// Spot-light PCSS blocker search: same math as pcssBlockerAverage above,
// but bounded by the spot tap budget (SPOT_PCSS_BLOCKER_SAMPLES) and
// clamped to the spot's own atlas page. Kept as a separate function so
// the sun path stays byte-identical. Returns the average blocker depth,
// or -1.0 when nothing blocks (caller early-outs to fully lit).
float pcssSpotBlockerAverage(texture2D depth_tex, sampler depth_smp, vec2 atlas_uv, float receiver_depth, mat2 rot, float search_radius, vec2 quad_min, vec2 quad_max) {
    float blocker_sum = 0.0;
    int blocker_count = 0;
    for (int i = 0; i < SPOT_PCSS_BLOCKER_SAMPLES; i++) {
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

    // Spot PCSS (source_radius > SPOT_PCSS_EPSILON): blocker search over the
    // spot's own atlas page -> average blocker depth -> variable penumbra via
    // the same formula as the sun path (pcssPenumbraRadius mirrors
    // penumbraRadius in scene/shadow_pcss.zig) -> variable-radius PCF.
    // Tap budgets (compile-time bounded): SPOT_PCSS_BLOCKER_SAMPLES blocker
    // taps + SPOT_PCSS_PCF_TAPS filter taps. source_radius at or below the
    // epsilon falls through to the legacy fixed 4-tap PCF bit-identically.
    float source_radius = spot_shadow_params[spot_idx].w;
    if (source_radius > SPOT_PCSS_EPSILON) {
        vec2 quad_min = vec2(float(spot_idx) * 0.5 + 0.002, 0.002);
        vec2 quad_max = vec2(float(spot_idx) * 0.5 + 0.498, 0.998);
        mat2 rot = pcssDitherRotation(gl_FragCoord.xy);
        float blocker_avg = pcssSpotBlockerAverage(spot_shadow_depth_tex, depth_smp, atlas_uv, depth, rot, SPOT_PCSS_BLOCKER_RADIUS, quad_min, quad_max);
        if (blocker_avg < 0.0) return 0.0;
        float filter_radius = pcssPenumbraRadius(depth, blocker_avg, source_radius, SPOT_PCSS_MIN_PENUMBRA, SPOT_PCSS_MAX_PENUMBRA);
        float pcss_lit = 0.0;
        for (int i = 0; i < SPOT_PCSS_PCF_TAPS; i++) {
            vec2 offset = rot * POISSON_DISK[i] * filter_radius;
            vec2 sample_uv = clamp(atlas_uv + offset, quad_min, quad_max);
            pcss_lit += texture(sampler2DShadow(spot_shadow_tex, shadow_smp), vec3(sample_uv, depth));
        }
        pcss_lit /= float(SPOT_PCSS_PCF_TAPS);
        return (1.0 - pcss_lit) * shadow_params.y;
    }

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

// Point-light PCSS v2 (border-correct) helpers. Taps are world-space disk
// offsets around the receiver; each tap selects its own cube face by
// dominant axis and projects through that face's matrix (all 12 face
// matrices are already fragment-shader uniforms, so no new uniforms were
// added — derived route, see calculatePointShadowSlot). Comparisons happen
// in the TAP's face depth space (the receiver is re-projected through the
// same face matrix), which is the cross-face depth normalization: view-space
// z differs per face orientation, so raw atlas depths from different faces
// are never compared directly.

// Projects a world point through one point-shadow face matrix, using the
// same SOKOL_GLSL y-flip convention as the receiver path. The caller
// checks .w > 0 before dividing.
vec4 pointPcssFaceProject(int proj_idx, vec3 world_pos) {
    vec4 lp = point_view_proj[proj_idx] * vec4(world_pos, 1.0);
    #if !SOKOL_GLSL
        lp.y = -lp.y;
    #endif
    return lp;
}

// Per-tap face selection + projection: picks the cube face from the tap
// direction, projects the tap through that face's matrix, and reports the
// atlas UV. Returns false (skip: the tap contributes nothing, it is never
// clamped into a neighboring texel) when the tap lands behind the tap-face
// camera, outside the shadow near/far range, or off the tap tile.
bool pointPcssTapAtlas(int slot, vec3 light_pos, vec3 tap_world, out int tap_face, out vec2 tap_atlas_uv) {
    tap_face = pointFaceIndex(tap_world - light_pos);
    vec4 lp = pointPcssFaceProject(slot * 6 + tap_face, tap_world);
    if (lp.w <= 0.0) return false;
    vec3 tp = lp.xyz / lp.w;
    if (tp.z > 1.0 || tp.z < 0.0) return false;
    vec2 tl = (tp.xy + 1.0) * 0.5;
    if (tl.x < 0.0 || tl.x > 1.0 || tl.y < 0.0 || tl.y > 1.0) return false;
    tap_atlas_uv = vec2((float(tap_face) + tl.x) / 6.0, float(slot) * 0.5 + tl.y * 0.5);
    return true;
}

// Receiver depth as seen by the tap's face (bias included), with validity.
// Comparing the stored tap depth against this value keeps both sides in one
// face space no matter which face the tap landed on. Returns false when the
// receiver itself falls outside the tap face frustum (comparison would be
// meaningless) — the tap is skipped.
bool pointPcssReceiverDepth(int slot, int tap_face, vec3 recv_world, float depth_bias, out float recv_depth) {
    vec4 lp = pointPcssFaceProject(slot * 6 + tap_face, recv_world);
    if (lp.w <= 0.0) return false;
    vec3 rp = lp.xyz / lp.w;
    if (rp.z > 1.0 || rp.z < 0.0) return false;
    vec2 rl = (rp.xy + 1.0) * 0.5;
    if (rl.x < 0.0 || rl.x > 1.0 || rl.y < 0.0 || rl.y > 1.0) return false;
    recv_depth = rp.z - depth_bias;
    return true;
}

// Point-light shadows: each shadow-casting point light owns 6 cube-face
// tiles (one atlas row, 256px tiles in the 1536x512 point atlas). The face
// is picked from the world-space direction to the light, then the fragment
// is projected by that face's view-projection matrix and PCF-sampled with
// the same 2D compare path as the spot atlas (4 taps + bias, shadow
float calculatePointShadowSlot(int slot, vec3 light_pos, vec3 world_pos, vec3 N, vec3 L) {
    if (slot < 0 || slot >= 2) return 0.0;
    if (shadow_params.y <= 0.001) return 0.0;

    vec3 to_frag = world_pos - light_pos;
    int face = pointFaceIndex(to_frag);

    float cos_theta = max(dot(N, L), 0.0);
    float bias = point_shadow_params[slot].y;
    float depth_bias = max(bias * (1.0 - cos_theta), bias * 0.2);
    vec3 normal_offset = N * (point_shadow_params[slot].z * (1.0 - cos_theta));

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

    // Point PCSS v2 (source_radius > POINT_PCSS_EPSILON, border-correct):
    // world-space disk taps around the receiver, per-tap face selection +
    // projection through that face's matrix, per-tap comparison in the tap
    // face's depth space (the cross-face normalization), blocker gaps
    // averaged as dimensionless (receiver - blocker) / receiver ratios and
    // anchored back at the receiver's own-face depth (identical to the v1
    // mean when every tap lands in one face — see pointEffectiveBlocker in
    // scene/shadow_pcss.zig), then the shared variable-penumbra formula and
    // a world-space PCF disk. No new uniforms: the 12 face matrices and the
    // light position were already available; tile UVs derive arithmetically
    // from the known atlas layout.
    // Tap budgets (compile-time bounded): POINT_PCSS_BLOCKER_SAMPLES
    // blocker taps + POINT_PCSS_PCF_TAPS filter taps, each doing 2 extra
    // mat4 x vec4 projections (tap + receiver-through-tap-face) and one
    // texture fetch. source_radius at or below the epsilon falls through
    // to the legacy fixed 4-tap PCF bit-identically.
    //
    // World-radius derivation (mirrors pointWorldRadiusFromUv): one tile
    // spans 1/6 of the atlas width and covers 2 x receiver_dist world units
    // at the receiver (90-degree cube faces), so one atlas-UV unit is
    // 12 x receiver_dist world units. The v1 calibration is preserved by
    // matching the tile-x texel footprint.
    //
    // Border correctness is NOT proven by screenshots — the q0/hdr gates
    // below only prove no-crash/no-validation-error. The correctness
    // evidence is the CPU mirror suite in scene/shadow_pcss_tests.zig
    // (face selection incl. ties, all-faces UV round-trip, skip predicate,
    // single-face mean reconstruction, cross-face edge equality).
    float source_radius = point_shadow_params[slot].w;
    if (source_radius > POINT_PCSS_EPSILON) {
        vec3 recv_world = world_pos + normal_offset;
        vec3 to_recv = recv_world - light_pos;
        float recv_dist = max(length(to_recv), 1e-4);
        vec3 fwd = to_recv / recv_dist;
        vec3 up_ref = abs(fwd.y) > 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
        vec3 disk_t = normalize(cross(up_ref, fwd));
        vec3 disk_b = cross(fwd, disk_t);
        mat2 rot = pcssDitherRotation(gl_FragCoord.xy);
        float world_search = POINT_PCSS_BLOCKER_RADIUS * 12.0 * recv_dist;
        float gap_sum = 0.0;
        int gap_count = 0;
        for (int i = 0; i < POINT_PCSS_BLOCKER_SAMPLES; i++) {
            vec2 disk_uv = rot * POISSON_DISK[i];
            vec3 tap_world = recv_world + (disk_t * disk_uv.x + disk_b * disk_uv.y) * world_search;
            int tap_face;
            vec2 tap_atlas_uv;
            if (!pointPcssTapAtlas(slot, light_pos, tap_world, tap_face, tap_atlas_uv)) continue;
            float tap_depth = texture(sampler2D(point_shadow_depth_tex, depth_smp), tap_atlas_uv).r;
            float tap_recv_depth;
            if (!pointPcssReceiverDepth(slot, tap_face, recv_world, depth_bias, tap_recv_depth)) continue;
            if (tap_depth < tap_recv_depth) {
                gap_sum += (tap_recv_depth - tap_depth) / max(tap_recv_depth, 1e-4);
                gap_count += 1;
            }
        }
        if (gap_count == 0) return 0.0;
        float blocker_eff = depth * (1.0 - gap_sum / float(gap_count));
        float filter_radius = pcssPenumbraRadius(depth, blocker_eff, source_radius, POINT_PCSS_MIN_PENUMBRA, POINT_PCSS_MAX_PENUMBRA);
        float world_filter = filter_radius * 12.0 * recv_dist;
        float pcss_lit = 0.0;
        int pcss_valid = 0;
        for (int i = 0; i < POINT_PCSS_PCF_TAPS; i++) {
            vec2 disk_uv = rot * POISSON_DISK[i];
            vec3 tap_world = recv_world + (disk_t * disk_uv.x + disk_b * disk_uv.y) * world_filter;
            int tap_face;
            vec2 tap_atlas_uv;
            if (!pointPcssTapAtlas(slot, light_pos, tap_world, tap_face, tap_atlas_uv)) continue;
            float tap_recv_depth;
            if (!pointPcssReceiverDepth(slot, tap_face, recv_world, depth_bias, tap_recv_depth)) continue;
            pcss_lit += texture(sampler2DShadow(point_shadow_tex, shadow_smp), vec3(tap_atlas_uv, tap_recv_depth));
            pcss_valid += 1;
        }
        if (pcss_valid == 0) return 0.0;
        pcss_lit /= float(pcss_valid);
        return (1.0 - pcss_lit) * shadow_params.y;
    }

    vec2 texel = vec2(1.0 / 1536.0, 1.0 / 512.0);
    float lit = 0.0;
    lit += texture(sampler2DShadow(point_shadow_tex, shadow_smp), vec3(atlas_uv + vec2(-texel.x, -texel.y), depth));
    lit += texture(sampler2DShadow(point_shadow_tex, shadow_smp), vec3(atlas_uv + vec2( texel.x, -texel.y), depth));
    lit += texture(sampler2DShadow(point_shadow_tex, shadow_smp), vec3(atlas_uv + vec2(-texel.x,  texel.y), depth));
    lit += texture(sampler2DShadow(point_shadow_tex, shadow_smp), vec3(atlas_uv + vec2( texel.x,  texel.y), depth));
    lit *= 0.25;

    return (1.0 - lit) * shadow_params.y;
}

// Point-light shadows: each shadow-casting point light owns 6 cube-face
// tiles (one atlas row, 256px tiles in the 1536x512 point atlas). The face
// is picked from the world-space direction to the light, then the fragment
// is projected by that face's view-projection matrix and PCF-sampled with
// the same 2D compare path as the spot atlas (4 taps + bias, shadow
// strength from shadow_params.y like every other shadow term here).
float calculatePointShadow(int light_idx, vec3 world_pos, vec3 N, vec3 L) {
    if (point_shadow_params[light_idx].x < 0.5) return 0.0;
    int slot = int(point_shadow_params[light_idx].x - 1.0);
    return calculatePointShadowSlot(slot, point_pos_range[light_idx].xyz, world_pos, N, L);
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
