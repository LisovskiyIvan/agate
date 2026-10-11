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
/// Cloud layer defaults: mid coverage, dome-friendly frequency, slow drift.
pub const CLOUD_COVERAGE_DEFAULT: f32 = 0.45;
pub const CLOUD_SCALE_DEFAULT: f32 = 3.0;
pub const CLOUD_SCALE_MIN: f32 = 0.5;
pub const CLOUD_SCALE_MAX: f32 = 16.0;
pub const WIND_SPEED_DEFAULT: f32 = 0.02;
pub const WIND_SPEED_MAX: f32 = 0.5;
/// Night defaults: moderate star field, small bright moon disc.
pub const STAR_DENSITY_DEFAULT: f32 = 0.3;
pub const STAR_INTENSITY_DEFAULT: f32 = 1.0;
pub const MOON_INTENSITY_DEFAULT: f32 = 2.0;
/// Moon angular RADIUS in degrees (full moon ~0.26 deg).
pub const MOON_RADIUS_DEG_DEFAULT: f32 = 0.26;
pub const MOON_RADIUS_DEG_MIN: f32 = 0.05;
pub const MOON_RADIUS_DEG_MAX: f32 = 3.0;
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
    /// 2D dome cloud layer (opt-in inside the analytic dome; off = no-op).
    clouds: CloudOptions = .{},
    /// Night sky: star field + analytic moon disc (auto-fades with the sun,
    /// forceable for day iteration; off/zero = no-op).
    night: NightOptions = .{},
    /// Sky clock in seconds feeding cloud wind advection (snapshot-captured;
    /// the app advances it, e.g. q0 accumulates dt while clouds are on).
    time_sec: f32 = 0.0,
};

/// 2D dome cloud layer: FBM value-noise on the view dome (NOT volumetric).
/// All strengths are multipliers around 1.0; every field is clamped.
pub const CloudOptions = struct {
    enabled: bool = false,
    /// 0 = clear, 1 = overcast.
    coverage: f32 = CLOUD_COVERAGE_DEFAULT,
    /// Dome UV frequency (higher = smaller puffs).
    scale: f32 = CLOUD_SCALE_DEFAULT,
    /// Advection direction in degrees (0 = +U, 90 = +V), same 0-at-+Z spirit
    /// as the sun azimuth convention (direction the wind blows TOWARD).
    wind_dir_deg: f32 = 45.0,
    /// UV drift per second at scale 1 (q0 default barely crawls).
    wind_speed: f32 = WIND_SPEED_DEFAULT,
    /// Cloud albedo multiplier (sun-tinted on GPU).
    brightness: f32 = 1.0,
    /// Circumsolar silver-lining gain (fake: forward-scatter lobe, no
    /// self-shadowing or phase function).
    silver_strength: f32 = 0.6,
    /// 0 = clouds reach the horizon, 1 = full fade below ~14 deg elevation.
    horizon_fade: f32 = 1.0,
};

/// Night sky: hash-based star field + analytic moon disc. Visual-only
/// (skybox dome): sun/ambient/fog lanes are untouched by night.
pub const NightOptions = struct {
    /// Force full night (day iteration). Otherwise night follows `auto`.
    enabled: bool = false,
    /// Fade stars/moon in automatically when the sun drops below ~+2 deg.
    auto: bool = true,
    /// Star HDR gain (bloom pickup at high values).
    star_intensity: f32 = STAR_INTENSITY_DEFAULT,
    /// 0 = sparse, 1 = dense (fraction of hash cells carrying a star).
    star_density: f32 = STAR_DENSITY_DEFAULT,
    /// Moon direction override in degrees (null = fixed default aloft).
    moon_elevation_deg: ?f32 = null,
    moon_azimuth_deg: ?f32 = null,
    /// Moon disc HDR gain (bloom pickup).
    moon_intensity: f32 = MOON_INTENSITY_DEFAULT,
    /// Moon angular radius in degrees.
    moon_radius_deg: f32 = MOON_RADIUS_DEG_DEFAULT,
    /// Night horizon multiplier (0 = pitch black, 1 = day horizon kept).
    horizon_dim: f32 = 0.15,
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
/// Cloud/night ride APPENDED vec4s in the same binding-2 block (existing
/// offsets untouched); all-zero extras keep the day dome bit-identical.
pub const AnalyticParams = struct {
    sky_sun: [4]f32,
    sky_ctrl: [4]f32,
    sky_sun_color: [4]f32,
    sky_cloud: [4]f32 = .{ 0.0, 0.0, 0.0, 0.0 },
    sky_cloud2: [4]f32 = .{ 0.0, 0.0, 0.0, 0.0 },
    sky_night: [4]f32 = .{ 0.0, 0.0, 0.0, 0.0 },
    sky_night2: [4]f32 = .{ 0.0, 0.0, 0.0, 0.0 },
    sky_moon: [4]f32 = .{ 0.0, 1.0, 0.0, 1.0 },
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

// --- Sky v2: clouds + night (opt-in inside the analytic dome) ---

/// Clamp coverage to 0..1; non-finite falls back to default.
pub fn sanitizeCoverage(c: f32) f32 {
    if (!std.math.isFinite(c)) return CLOUD_COVERAGE_DEFAULT;
    return std.math.clamp(c, 0.0, 1.0);
}

/// Clamp dome frequency into the sane range; non-finite falls back.
pub fn sanitizeCloudScale(s: f32) f32 {
    if (!std.math.isFinite(s)) return CLOUD_SCALE_DEFAULT;
    return std.math.clamp(s, CLOUD_SCALE_MIN, CLOUD_SCALE_MAX);
}

/// Clamp wind speed (signed: negative drifts backward); non-finite 0.
pub fn sanitizeWindSpeed(s: f32) f32 {
    if (!std.math.isFinite(s)) return 0.0;
    return std.math.clamp(s, -WIND_SPEED_MAX, WIND_SPEED_MAX);
}

/// Clamp a non-negative strength multiplier (brightness/silver/star/moon
/// intensity); non-finite falls back to `fallback`.
pub fn sanitizeStrength(v: f32, fallback: f32) f32 {
    if (!std.math.isFinite(v)) return fallback;
    return std.math.clamp(v, 0.0, 8.0);
}

/// Wind drift offset in dome-UV units: direction (toward) x speed x time.
/// Matches the GLSL advection (uv = (proj + wind*time) * scale) up to
/// float precision (the shader applies the coverage scale after the sum).
pub fn cloudWindOffset(wind_dir_deg: f32, wind_speed: f32, time_sec: f32) [2]f32 {
    const t = if (std.math.isFinite(time_sec)) time_sec else 0.0;
    const s = sanitizeWindSpeed(wind_speed);
    const a = (if (std.math.isFinite(wind_dir_deg)) wind_dir_deg else 0.0) * std.math.pi / 180.0;
    return .{ @cos(a) * s * t, @sin(a) * s * t };
}

/// Coverage mask from a 0..1 noise sample: smoothstep ramp of width 0.25
/// whose low edge slides with coverage (same formula in skybox.glsl).
/// Monotone in both inputs; coverage 0 pins clear exactly (low edge at 1.0,
/// noise never exceeds it); coverage 1 is full except the deepest noise
/// minima (ramp 0..0.25).
pub fn cloudMask(noise01: f32, coverage: f32) f32 {
    const n = std.math.clamp(noise01, 0.0, 1.0);
    const c = sanitizeCoverage(coverage);
    const t = std.math.clamp((n - (1.0 - c)) / 0.25, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

/// Dome compression shared with the shader: cloud UVs stay above the
/// horizon band (views at/below the horizon get no clouds).
pub fn cloudHorizonFade(view_y: f32, fade: f32) f32 {
    const f = std.math.clamp(if (std.math.isFinite(fade)) fade else 1.0, 0.0, 1.0);
    if (f <= 0.0) return 1.0;
    const t = std.math.clamp(view_y / 0.25, 0.0, 1.0);
    const s = t * t * (3.0 - 2.0 * t);
    return 1.0 - f * (1.0 - s);
}

/// Night amount 0..1 from sun elevation sine: full night at/below ~-6 deg,
/// full day at/above ~+2 deg, smooth in between. Forced to 1 when
/// `opt.enabled` (day iteration); 0 when neither enabled nor auto.
pub fn nightAmount(sun_sin_elev: f32, opt: NightOptions) f32 {
    if (opt.enabled) return 1.0;
    if (!opt.auto) return 0.0;
    const s = if (std.math.isFinite(sun_sin_elev)) sun_sin_elev else 1.0;
    const t = std.math.clamp((0.035 - s) / 0.14, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

/// Moon direction toward the moon: overrides win, otherwise a fixed
/// default aloft (elev 35, azim 200) — deterministic, never degenerate.
pub fn moonDir(opt: NightOptions) [3]f32 {
    const el = opt.moon_elevation_deg orelse 35.0;
    const az = opt.moon_azimuth_deg orelse 200.0;
    const e = if (std.math.isFinite(el)) el else 35.0;
    const a = if (std.math.isFinite(az)) az else 200.0;
    const d = sunDirFromDegrees(e, a);
    if (dot3(d, d) < 1e-12) return .{ 0.0, 1.0, 0.0 };
    return d;
}

/// Cosine of the moon angular radius (disc threshold shared with GLSL).
pub fn moonDiscCos(radius_deg: f32) f32 {
    const r = std.math.clamp(if (std.math.isFinite(radius_deg)) radius_deg else MOON_RADIUS_DEG_DEFAULT, MOON_RADIUS_DEG_MIN, MOON_RADIUS_DEG_MAX);
    return @cos(r * std.math.pi / 180.0);
}

/// Full cloud+night uniform mirror. Day-dome bit-identity: clouds disabled
/// AND night amount 0 reproduces `analyticParams(enabled)` extras exactly
/// (all-zero cloud/night lanes, moon parked at zenith).
pub fn analyticParamsFull(
    sun_dir: Vec3,
    turbidity: f32,
    sun_color: Color3,
    sun_intensity: f32,
    enabled: bool,
    opt: SkyOptions,
    sun_sin_elev: f32,
) AnalyticParams {
    var p = analyticParams(sun_dir, turbidity, sun_color, sun_intensity, enabled);
    if (!enabled) return p;
    if (opt.clouds.enabled) {
        const w = cloudWindOffset(opt.clouds.wind_dir_deg, opt.clouds.wind_speed, opt.time_sec);
        const tsec: f32 = if (std.math.isFinite(opt.time_sec)) opt.time_sec else 0.0;
        p.sky_cloud = .{ sanitizeCoverage(opt.clouds.coverage), sanitizeCloudScale(opt.clouds.scale), tsec, 1.0 };
        p.sky_cloud2 = .{ w[0], w[1], sanitizeStrength(opt.clouds.brightness, 1.0), sanitizeStrength(opt.clouds.silver_strength, 0.6) };
    }
    const na = nightAmount(sun_sin_elev, opt.night);
    // Bit-identity: lanes stay all-zero unless their feature is live, so a
    // day dome without clouds/night matches `analyticParams` exactly.
    const hf: f32 = if (opt.clouds.enabled) (if (std.math.isFinite(opt.clouds.horizon_fade)) std.math.clamp(opt.clouds.horizon_fade, 0.0, 1.0) else 1.0) else 0.0;
    const hd: f32 = if (na > 0.0) (if (std.math.isFinite(opt.night.horizon_dim)) std.math.clamp(opt.night.horizon_dim, 0.0, 1.0) else 0.15) else 0.0;
    if (na > 0.0) {
        const md = moonDir(opt.night);
        const dens = if (std.math.isFinite(opt.night.star_density)) std.math.clamp(opt.night.star_density, 0.0, 1.0) else STAR_DENSITY_DEFAULT;
        p.sky_night = .{ na, sanitizeStrength(opt.night.star_intensity, STAR_INTENSITY_DEFAULT), dens, sanitizeStrength(opt.night.moon_intensity, MOON_INTENSITY_DEFAULT) };
        p.sky_night2 = .{ hf, hd, 0.0, 0.0 };
        p.sky_moon = .{ md[0], md[1], md[2], moonDiscCos(opt.night.moon_radius_deg) };
    } else {
        p.sky_night2 = .{ hf, 0.0, 0.0, 0.0 };
    }
    return p;
}
