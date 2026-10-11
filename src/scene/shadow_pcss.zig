// PCSS (percentage-closer soft shadows) CPU-side math for the cascaded sun shadow atlas
// and the spot-light shadow atlas. GPU-free: pure formulas mirrored by the forward shaders
// (shaders/{standard,pbr,skinned_pbr,instanced}.glsl), which run the same
// blocker search -> penumbra -> PCF pipeline on the GPU. No GPU imports here,
// so `zig build test` covers this file without a graphics context.

// Fixed PCF tap counts per cascade. Must match the taps logic in sampleCascade:
// cascade 0 keeps the legacy 16-tap Poisson PCF, cascades 1-3 keep 8 taps.
pub const pcf_taps_near: u32 = 16;
pub const pcf_taps_far: u32 = 8;

// Live defaults for the packed PCSS uniforms (atlas-UV units unless noted).
pub const default_light_size: f32 = 0.02;
pub const default_blocker_radius: f32 = 0.01;
pub const default_min_penumbra: f32 = 0.0005;
pub const default_max_penumbra: f32 = 0.01;

// Guards the (receiver - blocker) / blocker division on both sides.
pub const min_blocker_depth: f32 = 0.0001;

// Thresholds the packed pcss_enabled uniform lane (0.0/1.0) at one half,
// mirroring the `cascade_debug.y > 0.5` shader branch.
pub fn isEnabled(flag: f32) bool {
    return flag > 0.5;
}

// Mirrors the per-cascade PCF tap LOD in sampleCascade.
pub fn pcfTapCount(cascade_idx: u32) u32 {
    return if (cascade_idx == 0) pcf_taps_near else pcf_taps_far;
}

// Normalized weight of one PCF disk tap; the shader divides the lit sum by
// the tap count, so weight * taps == 1. Zero taps weigh nothing (no division).
pub fn tapWeight(taps: u32) f32 {
    if (taps == 0) return 0.0;
    return 1.0 / @as(f32, @floatFromInt(taps));
}

// Averages blocker depths; null when nothing blocked the receiver, in which
// case the shaders skip the PCF entirely and report fully lit.
pub fn averageBlocker(sum: f32, count: u32) ?f32 {
    if (count == 0) return null;
    return sum / @as(f32, @floatFromInt(count));
}

// Variable penumbra from the average blocker depth:
// (d_receiver - d_blocker) * light_size, clamped to
// [min_penumbra, max_penumbra]. Mirrors pcssPenumbraRadius in the shaders.
// For parallel rays (orthographic directional sun), penumbra scales linearly
// with the distance between blocker and receiver without perspective division.
pub fn penumbraRadius(receiver_depth: f32, blocker_avg: f32, light_size: f32, min_penumbra: f32, max_penumbra: f32) f32 {
    const raw = (receiver_depth - blocker_avg) * light_size;
    return @min(@max(raw, min_penumbra), max_penumbra);
}

// Maps the blocker-search outcome to the shadow factor: no blockers means
// fully lit (1.0) without running the PCF, otherwise the PCF result stands.
pub fn resolveLit(blocker_avg: ?f32, pcf_lit: f32) f32 {
    if (blocker_avg != null) return pcf_lit;
    return 1.0;
}

// Spot-light PCSS (local emitters): mirrors the spot branch of
// calculateSpotShadow in shaders/common/shadow_pcf.glsl. The penumbra
// formula is shared with the sun (penumbraRadius above); only the tap
// budgets and the search geometry differ (one 1024x512 atlas page per
// spot slot instead of the 4-cascade sun atlas).

// Fixed spot blocker-search tap count. Must match SPOT_PCSS_BLOCKER_SAMPLES
// in the shaders (pbr/instanced_pbr/skinned_pbr.glsl).
pub const spot_blocker_sample_count: u32 = 8;
// Fixed spot variable-radius PCF tap count. Must match SPOT_PCSS_PCF_TAPS
// in the shaders.
pub const spot_pcf_tap_count: u32 = 8;

// Spot blocker-search radius in atlas UV. Must match
// SPOT_PCSS_BLOCKER_RADIUS in the shaders.
pub const spot_blocker_radius: f32 = 0.01;
// Spot penumbra clamp range in atlas UV. Must match
// SPOT_PCSS_MIN_PENUMBRA / SPOT_PCSS_MAX_PENUMBRA in the shaders.
pub const spot_min_penumbra: f32 = 0.0005;
pub const spot_max_penumbra: f32 = 0.01;

// Thresholds the packed source_radius lane (spot_shadow_params[i].w) at
// this epsilon, mirroring the `source_radius > SPOT_PCSS_EPSILON` shader
// branch. Zero (the default) keeps the legacy fixed 4-tap PCF bit-identical.
pub const spot_pcss_epsilon: f32 = 0.0001;

// Whether a spot light runs the PCSS path: source_radius above epsilon.
pub fn spotPcssEnabled(source_radius: f32) bool {
    return source_radius > spot_pcss_epsilon;
}

// Point-light PCSS (cube-face atlas): mirrors the point branch of
// calculatePointShadowSlot in shaders/common/shadow_pcf.glsl. The penumbra
// formula is shared with the sun and the spots (penumbraRadius above); only
// the tap budgets and the search geometry differ (one 256px cube-face tile
// per face/slot instead of a spot atlas page).
//
// v2 (border-correct): taps are world-space disk offsets around the
// receiver in the plane perpendicular to the light direction, each tap
// selects its own cube face by dominant axis (pointFaceIndex below,
// mirroring the GLSL pointFaceIndex and passes/shadow pointFaceForDir)
// and projects through that face's matrix. The per-tap comparison happens
// in the TAP's face depth space (the receiver is re-projected through the
// same face matrix), so comparisons are exact without near/far uniforms —
// this is the cross-face depth normalization. The blocker average mixes
// faces, so taps accumulate the dimensionless relative gap
// (receiver - blocker) / receiver instead of raw depths; the effective
// blocker (pointEffectiveBlocker) reconstructs the v1 mean exactly when
// every tap lands in one face (pinned by tests), and degrades gracefully
// at borders (same-ray geometry gives the same view-depth ratio in every
// face, so only the perspective-mapping nonlinearity leaks across faces —
// second order, and usually clamped away by the penumbra clamps).
// Taps outside the shadow near/far range or off their tile are skipped
// (no contribution), never clamped into a neighboring texel.

// Fixed point blocker-search tap count. Must match POINT_PCSS_BLOCKER_SAMPLES
// in the shaders (pbr/instanced_pbr/skinned_pbr.glsl).
pub const point_blocker_sample_count: u32 = 8;
// Fixed point variable-radius PCF tap count. Must match POINT_PCSS_PCF_TAPS
// in the shaders.
pub const point_pcf_tap_count: u32 = 8;

// Point blocker-search radius in atlas UV. Must match
// POINT_PCSS_BLOCKER_RADIUS in the shaders.
pub const point_blocker_radius: f32 = 0.01;
// Point penumbra clamp range in atlas UV. Must match
// POINT_PCSS_MIN_PENUMBRA / POINT_PCSS_MAX_PENUMBRA in the shaders.
pub const point_min_penumbra: f32 = 0.0005;
pub const point_max_penumbra: f32 = 0.01;

// Thresholds the packed source_radius lane (point_shadow_params[i].w) at
// this epsilon, mirroring the `source_radius > POINT_PCSS_EPSILON` shader
// branch. Zero (the default) keeps the legacy fixed 4-tap PCF bit-identical.
pub const point_pcss_epsilon: f32 = 0.0001;

// Whether a point light runs the PCSS path: source_radius above epsilon.
pub fn pointPcssEnabled(source_radius: f32) bool {
    return source_radius > point_pcss_epsilon;
}

// ---- Point PCSS v2 world-space tap math (pure mirrors of the GLSL) ----

const math = @import("math");

fn shadowClip(vp: math.Mat4, p: math.Vec3, w: f32, flip_y: bool) [4]f32 {
    var clip: [4]f32 = undefined;
    for (0..4) |row| {
        clip[row] = vp.m[row] * p.x + vp.m[4 + row] * p.y + vp.m[8 + row] * p.z + vp.m[12 + row] * w;
    }
    if (flip_y) clip[1] = -clip[1];
    return clip;
}

/// Analytic dz/d(atlas UV) of a receiver plane, for orthographic or
/// perspective shadow projections. No screen derivatives or pixel-size
/// threshold: the result stays valid across cascade/face branches.
/// Mirrors computeShadowPlaneGradient in common/shadow_pcf.glsl.
pub fn receiverPlaneGradient(vp: math.Mat4, receiver: math.Vec3, normal: math.Vec3, atlas_scale: [2]f32, flip_y: bool) [2]f32 {
    const up_ref = if (@abs(normal.y) > 0.9) math.Vec3.new(1, 0, 0) else math.Vec3.up;
    const tangent = up_ref.cross(normal).normalize();
    const bitangent = normal.cross(tangent);
    const pos = shadowClip(vp, receiver, 1.0, flip_y);
    if (pos[3] <= 0.0) return .{ 0.0, 0.0 };
    const ct = shadowClip(vp, tangent, 0.0, flip_y);
    const cb = shadowClip(vp, bitangent, 0.0, flip_y);
    const proj = math.Vec3.new(pos[0], pos[1], pos[2]).scale(1.0 / pos[3]);
    const dt = math.Vec3.new(ct[0], ct[1], ct[2]).sub(proj.scale(ct[3]));
    const db = math.Vec3.new(cb[0], cb[1], cb[2]).sub(proj.scale(cb[3]));
    const plane = math.Vec3.new(dt.x * atlas_scale[0], dt.y * atlas_scale[1], dt.z).cross(
        math.Vec3.new(db.x * atlas_scale[0], db.y * atlas_scale[1], db.z),
    );
    const scale = @max(@max(@abs(plane.x), @abs(plane.y)), 1e-20);
    if (@abs(plane.z) <= scale * 1e-5) return .{ 0.0, 0.0 };
    return .{ -plane.x / plane.z, -plane.y / plane.z };
}

// Cube-face index for a light-space direction. Mirrors the GLSL
// pointFaceIndex in the forward shaders (and passes/shadow
// pointFaceForDir): the major axis wins, ties prefer X over Y over Z, the
// sign picks the positive/negative face (order +X, -X, +Y, -Y, +Z, -Z,
// matching PointLight face order). The zero vector falls into +X, exactly
// like the shader (0.0 >= 0.0 is true).
pub fn pointFaceIndex(d: math.Vec3) usize {
    const ax = @abs(d.x);
    const ay = @abs(d.y);
    const az = @abs(d.z);
    if (ax >= ay and ax >= az) return if (d.x >= 0.0) 0 else 1;
    if (ay >= ax and ay >= az) return if (d.y >= 0.0) 2 else 3;
    return if (d.z >= 0.0) 4 else 5;
}

// Atlas UV of a within-tile local UV for a (slot, face) tile. Mirrors the
// `atlas_uv` arithmetic in calculatePointShadowSlot: 6 faces left to right
// (tile width 1/6), one row per shadow slot (tile height 1/2).
// slot < 2, face < 6, local components in [0, 1].
pub fn pointAtlasUv(slot: usize, face: usize, local_uv: [2]f32) [2]f32 {
    return .{
        (@as(f32, @floatFromInt(face)) + local_uv[0]) / 6.0,
        @as(f32, @floatFromInt(slot)) * 0.5 + local_uv[1] * 0.5,
    };
}

/// Fetch UV keeps the whole bilinear comparison footprint inside the face.
/// Valid edge taps are retained, not discarded as a fully-lit seam strip.
/// Mirrors pointPcssTapAtlas and fixed point PCF tile bounds in the shader.
pub fn pointSampleAtlasUv(slot: usize, face: usize, local_uv: [2]f32) [2]f32 {
    const inset = 0.5 / 256.0;
    return pointAtlasUv(slot, face, .{
        @min(@max(local_uv[0], inset), 1.0 - inset),
        @min(@max(local_uv[1], inset), 1.0 - inset),
    });
}

// Inclusive atlas-UV bounds of a (slot, face) tile: taps strictly outside
// are skipped, taps exactly on the edge sample the edge texel (mirrors the
// `< 0.0 || > 1.0` local-UV skip in the shader).
pub fn pointTileBounds(slot: usize, face: usize) struct { min: [2]f32, max: [2]f32 } {
    return .{
        .min = .{
            @as(f32, @floatFromInt(face)) / 6.0,
            @as(f32, @floatFromInt(slot)) * 0.5,
        },
        .max = .{
            (@as(f32, @floatFromInt(face)) + 1.0) / 6.0,
            @as(f32, @floatFromInt(slot)) * 0.5 + 0.5,
        },
    };
}

// Per-tap validity predicate for the v2 world-space taps. Mirrors the skip
// chain in the shader: behind the tap face camera (w <= 0), outside the
// shadow near/far range (proj z outside [0, 1]), or off the tap tile
// (local UV outside [0, 1]). Skipped taps contribute nothing — they are
// never clamped into a neighboring texel.
pub fn pointTapSkipped(w: f32, proj_z: f32, local_uv: [2]f32) bool {
    if (w <= 0.0) return true;
    if (proj_z < 0.0 or proj_z > 1.0) return true;
    if (local_uv[0] < 0.0 or local_uv[0] > 1.0) return true;
    if (local_uv[1] < 0.0 or local_uv[1] > 1.0) return true;
    return false;
}

// Relative blocker gap of one tap, measured in the tap's own face depth
// space (both depths projected through the same face matrix, so the
// comparison is valid across faces). Null when the tap does not block the
// receiver (stored depth at or behind the receiver depth) — the tap is a
// non-blocker exactly like a non-closer tap in the sun/spot search.
pub fn pointBlockerGap(receiver_depth: f32, tap_depth: f32) ?f32 {
    if (tap_depth >= receiver_depth) return null;
    return (receiver_depth - tap_depth) / @max(receiver_depth, min_blocker_depth);
}

// Reconstructs the v1-style average blocker depth from the mean relative
// gap, anchored at the receiver depth in its own face. When every tap
// lands in one face the receiver depth is the same matrix projection for
// all taps, and this returns the plain arithmetic mean of the blocker
// depths bit-for-bit up to float association (pinned by tests).
pub fn pointEffectiveBlocker(receiver_depth_own: f32, avg_gap: f32) f32 {
    return receiver_depth_own * (1.0 - avg_gap);
}

// World-space radius of a PCSS disk from an atlas-UV radius at the
// receiver distance. One tile spans 1/6 of the atlas width and covers
// 2 * dist world units at the receiver (90-degree cube faces), so one
// atlas-UV unit is 12 * dist world units along the tile x axis. The v1
// calibration (blocker radius, penumbra clamps) is preserved by matching
// the x-axis texel footprint; the disk itself is isotropic in world space
// (v1's atlas-UV disk was elliptical in texels: 15.4 x 5.1 texels at the
// default radius — the y axis now matches x instead of being 3x tighter).
pub fn pointWorldRadiusFromUv(uv_radius: f32, receiver_dist: f32) f32 {
    return uv_radius * 12.0 * receiver_dist;
}

// Perspective depth (0..1) of a forward distance d for the point shadow
// faces. Mirrors math.Mat4.perspective (90-degree fov, aspect 1, OpenGL
// column-major convention: clip.z = far*(d-near)/(far-near)... over
// clip.w = d): d = near maps to 0, d -> far maps to 1.
pub fn pointPerspectiveDepth01(forward_dist: f32, near: f32, far: f32) f32 {
    return far * (forward_dist - near) / ((far - near) * forward_dist);
}

// Inverse of pointPerspectiveDepth01: forward distance from a stored
// atlas depth. Used by tests to build same-ray cross-face geometries.
pub fn pointViewDistance(depth01: f32, near: f32, far: f32) f32 {
    return near * far / (far - depth01 * (far - near));
}
