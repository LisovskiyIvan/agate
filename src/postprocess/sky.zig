//! Analytic physical sky (Preetham-class, opt-in).
//!
//! Derives a CONSISTENT triple from sun elevation + turbidity:
//! attenuated sun color/intensity, sky-dome radiance by view direction,
//! and hemispheric sky/ground ambient — plus a fog tint so aerial
//! perspective matches the sky. The model is strictly opt-in
//! (`SkyOptions.enabled = false` keeps every manual value bit-identical).
//!
//! Model class: Perez-distribution sky (Preetham et al. 1999 spirit) with
//! Beer-Lambert sun attenuation and a Kasten-Young-style air mass. The
//! Perez FUNCTIONAL FORM is exact; the turbidity-fit coefficients and the
//! zenith/horizon tint ramps are inline approximations, NOT a certified
//! Preetham implementation (Hosek-Wilkie was rejected: its coefficient
//! tables cannot be reproduced correctly from memory, and a correct
//! simpler model beats a broken fancier one). Golden CPU tests in
//! `sky_tests.zig` pin behavior (zenith/horizon, monotonicity, energy
//! sanity, fixed golden values); `skybox.glsl` mirrors the same formulas
//! on the GPU (float-precision differences expected and fine).
//!
//! Fog/shaft inputs REMAIN manual-overridable: the model PROVIDES
//! fog-tint/sun values through the existing uniform lanes, it never
//! forces them (per-output `drive_*` flags; fog/shaft enable switches
//! are untouched).

const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Color3 = math.Color3;

/// Default turbidity: clear mid-latitude day (Preetham range ~2..6).
pub const TURBIDITY_DEFAULT: f32 = 2.6;
pub const TURBIDITY_MIN: f32 = 1.0;
pub const TURBIDITY_MAX: f32 = 12.0;
/// Reference sun intensity scale (matches the ~2..4 manual sun range).
pub const SUN_REF_INTENSITY: f32 = 3.0;
/// Sun disk HDR gain over the attenuated sun radiance (bloom pickup).
pub const SUN_DISK_GAIN: f32 = 8.0;
/// Cosine of the solar angular radius (~0.265 deg): pixels inside the
/// disk carry the sun, outside carry sky.
pub const SUN_DISK_COS: f32 = 0.9999893;
/// Hemispheric ambient gain from mean sky radiance.
pub const AMBIENT_GAIN: f32 = 0.28;

/// Effective broadband extinction per channel per unit
/// turbidity-airmass. Inline fit (NOT laboratory cross-sections):
/// blue attenuates fastest, which is what reddens low sun; the red
/// lane still carries ~70% at 3 deg elevation, so sunset stays bright
/// instead of dying to black.
const beta_ray = [3]f32{ 0.011, 0.020, 0.042 };

/// Opt-in analytic atmosphere. All `drive_*` default true: when enabled
/// the model provides sun/ambient/fog values; setting a drive flag false
/// keeps that manual value while the rest stays model-driven.
pub const SkyOptions = struct {
    enabled: bool = false,
    turbidity: f32 = TURBIDITY_DEFAULT,
    /// Sun elevation override in degrees (null = keep the scene sun_dir).
    sun_elevation_deg: ?f32 = null,
    /// Sun azimuth override in degrees, same convention as
    /// lights.sunDirectionFromAngles (0 = +Z, 90 = +X; null = keep).
    sun_azimuth_deg: ?f32 = null,
    /// Lambertian ground albedo feeding the ground ambient lane.
    ground_albedo: [3]f32 = .{ 0.3, 0.3, 0.3 },
    drive_sun: bool = true,
    drive_ambient: bool = true,
    drive_fog: bool = true,
    /// Render the analytic dome through the skybox pass (needs no cubemap).
    dome_enabled: bool = false,
};

/// Clamp turbidity into the sane range; non-finite falls back to default.
pub fn sanitizeTurbidity(t: f32) f32 {
    if (!std.math.isFinite(t)) return TURBIDITY_DEFAULT;
    return std.math.clamp(t, TURBIDITY_MIN, TURBIDITY_MAX);
}

/// Kasten-Young-style relative air mass from sine of elevation. Floored at
/// 0.02 (never singular past the horizon); finite everywhere.
pub fn airMass(sin_elev: f32) f32 {
    const s = @max(sin_elev, 0.02);
    const h_deg = std.math.asin(std.math.clamp(sin_elev, -0.049, 1.0)) * 180.0 / std.math.pi;
    return 1.0 / (s + 0.50572 * std.math.pow(f32, h_deg + 6.07995, -1.6364));
}

/// Per-channel Beer-Lambert transmittance of direct sunlight.
pub fn sunAttenuation(sin_elev: f32, turbidity: f32) [3]f32 {
    const m = airMass(sin_elev);
    const t = sanitizeTurbidity(turbidity);
    var out: [3]f32 = undefined;
    for (0..3) |c| out[c] = @exp(-beta_ray[c] * t * m);
    return out;
}

/// Attenuated sun color (<= 1 per channel) + photometric-ish intensity.
/// Below-horizon fade kills the disk smoothly past ~-1 deg while the
/// air-mass floor keeps every lane finite.
pub const SunLight = struct {
    color: [3]f32,
    intensity: f32,
};

pub fn sunLight(sin_elev: f32, turbidity: f32) SunLight {
    const att = sunAttenuation(sin_elev, turbidity);
    const fade = std.math.clamp((sin_elev + 0.02) / 0.06, 0.0, 1.0);
    var peak: f32 = 0.0;
    for (att) |a| peak = @max(peak, a);
    return .{
        .color = .{ att[0] * fade, att[1] * fade, att[2] * fade },
        .intensity = SUN_REF_INTENSITY * peak * fade,
    };
}

/// Perez distribution coefficients (A, B, C, D, E) as a linear turbidity
/// fit in the Preetham spirit: negative A brightens the horizon, C grows
/// the circumsolar forward lobe with haze.
pub fn perezCoeffs(turbidity: f32) [5]f32 {
    const t = sanitizeTurbidity(turbidity);
    return .{ -(0.38 + 0.055 * t), -0.30, 2.2 + 0.9 * t, -0.65, 0.12 };
}

/// Perez sky-luminance distribution F(theta, gamma):
/// (1 + A exp(B/cosT)) (1 + C exp(D gamma) + E cos^2 gamma).
/// cosT is clamped (never singular at the horizon), gamma in radians.
pub fn perez(cos_theta: f32, gamma: f32, k: [5]f32) f32 {
    const ct = @max(cos_theta, 0.012);
    const g = std.math.clamp(gamma, 0.0, std.math.pi);
    const cg = @cos(g);
    return (1.0 + k[0] * @exp(k[1] / ct)) * (1.0 + k[2] * @exp(k[3] * g) + k[4] * cg * cg);
}

/// Zenith state: luminance falls with turbidity and low sun; the tint
/// whitens with haze and reddens toward sunset.
pub const ZenithState = struct {
    lum: f32,
    tint: [3]f32,
};

pub fn zenithState(sin_elev: f32, turbidity: f32) ZenithState {
    const s = std.math.clamp(sin_elev, 0.02, 1.0);
    const t = sanitizeTurbidity(turbidity);
    const lum = 1.15 * (0.22 + 0.78 * std.math.pow(f32, s, 0.55)) / (0.55 + 0.14 * t);
    const whiten = std.math.clamp(0.18 + 0.075 * (t - 2.0) - 0.55 * (s - 0.25), 0.0, 1.0);
    const deep = [3]f32{ 0.30, 0.48, 1.0 };
    const gray = [3]f32{ 0.86, 0.83, 0.79 };
    var tint: [3]f32 = undefined;
    for (0..3) |c| tint[c] = deep[c] + (gray[c] - deep[c]) * whiten;
    return .{ .lum = lum, .tint = tint };
}

/// Horizon state: anchored to the zenith through the Perez ratio, warmed
/// toward sunset and whitened with turbidity.
pub const HorizonState = struct {
    lum: f32,
    tint: [3]f32,
};

pub fn horizonState(sin_elev: f32, turbidity: f32) HorizonState {
    const s = std.math.clamp(sin_elev, 0.02, 1.0);
    const t = sanitizeTurbidity(turbidity);
    const zen = zenithState(s, t);
    const k = perezCoeffs(t);
    const boost = (1.0 / (1.0 + k[0] * @exp(k[1]))) * (0.85 + 0.10 * t);
    const warmth = std.math.clamp(0.65 - 1.1 * s + 0.06 * (t - 2.0), 0.0, 1.0);
    const cool = [3]f32{ 0.55, 0.68, 0.85 };
    const warm = [3]f32{ 1.0, 0.72, 0.48 };
    var tint: [3]f32 = undefined;
    for (0..3) |c| tint[c] = cool[c] + (warm[c] - cool[c]) * warmth;
    return .{ .lum = zen.lum * boost, .tint = tint };
}

fn dot3(a: [3]f32, b: [3]f32) f32 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

fn norm3(v: [3]f32) [3]f32 {
    const l = @sqrt(@max(dot3(v, v), 1e-12));
    return .{ v[0] / l, v[1] / l, v[2] / l };
}

fn smoothstep01(a: f32, b: f32, x: f32) f32 {
    const t = std.math.clamp((x - a) / (b - a), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

/// Sky-dome radiance for a view direction (CPU mirror of the skybox.glsl
/// analytic path; same formulas, f32 math). Below-horizon views return
/// dark ground haze. The sun disk (inside SUN_DISK_COS) adds HDR sun
/// radiance for bloom pickup. Never negative, never non-finite.
pub fn skyRadiance(view_dir: [3]f32, sun_dir: [3]f32, turbidity: f32) [3]f32 {
    const t = sanitizeTurbidity(turbidity);
    const v = norm3(view_dir);
    const d = norm3(sun_dir);
    const se = @max(d[1], 0.02);
    const zen = zenithState(se, t);
    const hor = horizonState(se, t);
    if (v[1] < 0.0) return .{ hor.tint[0] * 0.12, hor.tint[1] * 0.12, hor.tint[2] * 0.12 };
    const k = perezCoeffs(t);
    const cos_g = std.math.clamp(dot3(v, d), -1.0, 1.0);
    const gamma = std.math.acos(cos_g);
    const gamma_sun = std.math.acos(std.math.clamp(se, -1.0, 1.0));
    const f = perez(@max(v[1], 0.012), gamma, k);
    const fz = perez(1.0, gamma_sun, k);
    const lum = zen.lum * f / @max(fz, 1e-4);
    const mix_t = smoothstep01(0.0, 0.5, v[1]);
    var col: [3]f32 = undefined;
    for (0..3) |c| col[c] = (hor.tint[c] + (zen.tint[c] - hor.tint[c]) * mix_t) * lum;
    // Warm circumsolar glow tinted by the attenuated sun itself (white at
    // noon, orange at sunset — no extra tint table needed).
    const att = sunAttenuation(se, t);
    const peak = @max(att[0], @max(att[1], att[2]));
    const glow_tint = [3]f32{ att[0] / peak, att[1] / peak, att[2] / peak };
    const glow = std.math.pow(f32, @max(cos_g, 0.0), 24.0) * (0.35 + 0.45 * lum) * (0.8 + 0.15 * t);
    for (0..3) |c| col[c] += glow_tint[c] * glow;
    if (cos_g > SUN_DISK_COS) {
        const sun = sunLight(d[1], t);
        for (0..3) |c| col[c] += sun.color[c] * sun.intensity * SUN_DISK_GAIN;
    }
    return .{ @max(col[0], 0.0), @max(col[1], 0.0), @max(col[2], 0.0) };
}

/// Full consistent triple for one frame.
pub const SkyFrame = struct {
    sun_dir: [3]f32,
    sun_color: [3]f32,
    sun_intensity: f32,
    hemi_sky: [3]f32,
    hemi_ground: [3]f32,
    fog_tint: [3]f32,
    turbidity: f32,
    analytic: bool,
};

/// Manual values the model overrides selectively (snapshot-side staging).
pub const ManualFrame = struct {
    sun_color: [3]f32,
    sun_intensity: f32,
    hemi_diffuse: [3]f32,
    hemi_ground: [3]f32,
    hemi_intensity: f32,
    fog_color: [3]f32,
};

/// Azimuth of a direction under the sunDirectionFromAngles convention
/// (0 = +Z, +90 deg = +X). Degenerate XZ falls back to 0.
pub fn azimuthOf(dir: [3]f32) f32 {
    if (dir[0] * dir[0] + dir[2] * dir[2] < 1e-12) return 0.0;
    return std.math.atan2(dir[0], dir[2]);
}

/// Sun direction toward the sun from elevation/azimuth degrees.
pub fn sunDirFromDegrees(elevation_deg: f32, azimuth_deg: f32) [3]f32 {
    const el = elevation_deg * std.math.pi / 180.0;
    const az = azimuth_deg * std.math.pi / 180.0;
    const ce = @cos(el);
    const v = [3]f32{ ce * @sin(az), @sin(el), ce * @cos(az) };
    if (dot3(v, v) < 1e-12) return .{ 0.0, 1.0, 0.0 };
    return norm3(v);
}

/// Model sun direction: elevation/azimuth overrides win, otherwise the
/// scene sun_dir passes through untouched.
pub fn modelSunDir(opt: SkyOptions, fallback: Vec3) Vec3 {
    if (opt.sun_elevation_deg == null and opt.sun_azimuth_deg == null) return fallback;
    const f = [3]f32{ fallback.x, fallback.y, fallback.z };
    const se = @max(f[1], 0.02);
    const default_el = std.math.asin(std.math.clamp(se, -1.0, 1.0)) * 180.0 / std.math.pi;
    const default_az = azimuthOf(f) * 180.0 / std.math.pi;
    const d = sunDirFromDegrees(opt.sun_elevation_deg orelse default_el, opt.sun_azimuth_deg orelse default_az);
    return Vec3.new(d[0], d[1], d[2]);
}

/// Resolves the frame triple. Disabled returns the manual values verbatim
/// (analytic false) — the only branch the default path takes.
pub fn resolveFrame(opt: SkyOptions, sun_dir: Vec3, m: ManualFrame) SkyFrame {
    const d = [3]f32{ sun_dir.x, sun_dir.y, sun_dir.z };
    const t = sanitizeTurbidity(opt.turbidity);
    if (!opt.enabled) {
        return .{
            .sun_dir = d,
            .sun_color = m.sun_color,
            .sun_intensity = m.sun_intensity,
            .hemi_sky = m.hemi_diffuse,
            .hemi_ground = m.hemi_ground,
            .fog_tint = m.fog_color,
            .turbidity = t,
            .analytic = false,
        };
    }
    const se = @max(@min(d[1], 1.0), -1.0);
    const sun = sunLight(se, t);
    const zen = zenithState(@max(se, 0.02), t);
    const hor = horizonState(@max(se, 0.02), t);
    var sky_amb: [3]f32 = undefined;
    var fog: [3]f32 = undefined;
    for (0..3) |c| {
        sky_amb[c] = (zen.tint[c] * zen.lum * 0.35 + hor.tint[c] * hor.lum * 0.65) * AMBIENT_GAIN;
        fog[c] = hor.tint[c] * std.math.clamp(hor.lum * 0.45, 0.15, 1.0);
    }
    var ground_amb: [3]f32 = undefined;
    for (0..3) |c| ground_amb[c] = std.math.clamp(opt.ground_albedo[c], 0.0, 1.0) * sky_amb[c] * 0.6;
    return .{
        .sun_dir = d,
        .sun_color = if (opt.drive_sun) sun.color else m.sun_color,
        .sun_intensity = if (opt.drive_sun) sun.intensity else m.sun_intensity,
        .hemi_sky = if (opt.drive_ambient) sky_amb else m.hemi_diffuse,
        .hemi_ground = if (opt.drive_ambient) ground_amb else m.hemi_ground,
        .fog_tint = if (opt.drive_fog) fog else m.fog_color,
        .turbidity = t,
        .analytic = opt.dome_enabled,
    };
}

/// Uniform-block mirror for the skybox analytic path:
/// sky_sun = (sun dir, turbidity), sky_ctrl = (enabled, 0, 0, 0),
/// sky_sun_color = (sun rgb, intensity). Zeros disable bit-identically.
pub const AnalyticParams = struct {
    sky_sun: [4]f32,
    sky_ctrl: [4]f32,
    sky_sun_color: [4]f32,
};

pub fn analyticParams(sun_dir: Vec3, turbidity: f32, sun_color: Color3, sun_intensity: f32, enabled: bool) AnalyticParams {
    if (!enabled) return .{
        .sky_sun = .{ 0.0, 1.0, 0.0, TURBIDITY_DEFAULT },
        .sky_ctrl = .{ 0.0, 0.0, 0.0, 0.0 },
        .sky_sun_color = .{ 0.0, 0.0, 0.0, 0.0 },
    };
    return .{
        .sky_sun = .{ sun_dir.x, sun_dir.y, sun_dir.z, sanitizeTurbidity(turbidity) },
        .sky_ctrl = .{ 1.0, 0.0, 0.0, 0.0 },
        .sky_sun_color = .{ sun_color.r, sun_color.g, sun_color.b, sun_intensity },
    };
}
