// Shared shadow-sampling constants for the PBR family (Poisson disk,
// cascade atlas tiling, PCSS tap budgets).
//
// Included by the three PBR forward fragment shaders (pbr, skinned_pbr,
// instanced_pbr) via `the pbr_shadow_consts include`,
// placed ahead of the shadow_pcf include whose functions
// consume them. Single source so a tap-budget or disk change lands in all
// three variants at once by construction.
//
// Contract:
// - Values and define names are load-bearing for visual parity: the disk
//   and offsets must match the CPU-side expectations (cascade atlas
//   layout in passes/shadow_pass.zig) and the tap budgets mirror
//   scene/shadow_pcss.zig (blocker_sample_count, spot/point budgets).
//   PI itself lives in common/pbr_brdf.glsl (single source for the BRDF).
// PI comes from common/pbr_brdf.glsl (single source for the BRDF chunk).

const vec2 POISSON_DISK[16] = vec2[](
    vec2(-0.94201624, -0.39906216),
    vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870),
    vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432),
    vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845),
    vec2( 0.97484398,  0.75648379),
    vec2( 0.44323325, -0.97511554),
    vec2( 0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023),
    vec2( 0.79197514,  0.19090188),
    vec2(-0.24188840,  0.99706507),
    vec2(-0.81409955,  0.91437590),
    vec2( 0.19984126,  0.78641367),
    vec2( 0.14383161, -0.14100790)
);

const vec2 CASCADE_OFFSETS[4] = vec2[](
    vec2(0.0, 0.0),
    vec2(0.5, 0.0),
    vec2(0.0, 0.5),
    vec2(0.5, 0.5)
);

// PCSS (percentage-closer soft shadows) for the 4-cascade sun atlas.
// Blocker search reads raw depths from shadow_depth_tex (same atlas view as
// shadow_tex, regular sampler): PCSS_BLOCKER_SAMPLES Poisson taps inside
// pcss_blocker_radius, average blocker depth -> penumbra ->
// (d_receiver - d_blocker) / d_blocker * light_size, clamped to
// [min_penumbra, max_penumbra]; the fixed-radius Poisson PCF then runs with the
// penumbra as its disk radius. Params ride free uniform lanes (fs_view
// layout unchanged):
//   cascade_debug.y = pcss_enabled (0.0/1.0)
//   cascade_debug.z = pcss_light_size
//   cascade_debug.w = pcss_blocker_radius (atlas-UV search radius)
//   light_counts.z  = pcss_min_penumbra (atlas-UV clamp)
//   light_counts.w  = pcss_max_penumbra (atlas-UV clamp)
// Disabled (y <= 0.5): fixed-radius 16x/8x Poisson PCF, bit-identical.
// Counts mirror scene/shadow_pcss.zig (blocker_sample_count).
#define PCSS_BLOCKER_SAMPLES 12

// Spot-light PCSS (local emitters): blocker search + variable penumbra for
// the spot atlas pages. source_radius rides spot_shadow_params[packed_spot_idx].w
// (0 = legacy fixed 4-tap PCF, bit-identical; > epsilon = PCSS path in
// calculateSpotShadow). Blocker search reads raw depths from
// spot_shadow_depth_tex (same atlas view as spot_shadow_tex) through the
// shared depth_smp; the penumbra reuses pcssPenumbraRadius (same formula as
// the sun path, see scene/shadow_pcss.zig). Tap budgets mirror
// scene/shadow_pcss.zig (spot_blocker_sample_count, spot_pcf_tap_count).
#define SPOT_PCSS_BLOCKER_SAMPLES 8
#define SPOT_PCSS_PCF_TAPS 8
#define SPOT_PCSS_EPSILON 0.0001
#define SPOT_PCSS_BLOCKER_RADIUS 0.01
#define SPOT_PCSS_MIN_PENUMBRA 0.0005
#define SPOT_PCSS_MAX_PENUMBRA 0.01

// Point-light PCSS (cube-face atlas): blocker search + variable penumbra
// per face tile. source_radius rides point_shadow_params[packed_light_idx].w
// (0 = legacy fixed 4-tap PCF, bit-identical; > epsilon = PCSS path in
// calculatePointShadowSlot). Blocker search reads raw depths from
// point_shadow_depth_tex (same atlas view as point_shadow_tex) through the
// shared depth_smp; the penumbra reuses pcssPenumbraRadius (same formula as
// the sun/spot paths, see scene/shadow_pcss.zig). Tap budgets mirror
// scene/shadow_pcss.zig (point_blocker_sample_count, point_pcf_tap_count).
// v2 is border-correct (world-space disk taps, per-tap face projection —
// see the v2 note in shadow_pcf.glsl).
#define POINT_PCSS_BLOCKER_SAMPLES 8
#define POINT_PCSS_PCF_TAPS 8
#define POINT_PCSS_EPSILON 0.0001
#define POINT_PCSS_BLOCKER_RADIUS 0.01
#define POINT_PCSS_MIN_PENUMBRA 0.0005
#define POINT_PCSS_MAX_PENUMBRA 0.01
