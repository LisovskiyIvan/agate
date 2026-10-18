//! Golden CPU tests for the analytic physical sky (postprocess/sky.zig).
//! Pins zenith/horizon behavior, sun monotonicity, energy sanity,
//! turbidity direction, drive-flag coupling, and fixed-input golden
//! values (skybox.glsl mirrors the same formulas; float-precision
//! differences on GPU are expected and fine).

const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Color3 = math.Color3;
const sky = @import("sky.zig");

fn lum(c: [3]f32) f32 {
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
}

fn sunDirElev(elev_deg: f32) Vec3 {
    const d = sky.sunDirFromDegrees(elev_deg, 0.0);
    return Vec3.new(d[0], d[1], d[2]);
}

fn manual() sky.ManualFrame {
    return .{
        .sun_color = .{ 1.0, 0.92, 0.8 },
        .sun_intensity = 2.6,
        .hemi_diffuse = .{ 1.0, 1.0, 1.0 },
        .hemi_ground = .{ 0.12, 0.1, 0.09 },
        .hemi_intensity = 0.18,
        .fog_color = .{ 0.72, 0.82, 0.92 },
    };
}

fn enabledOpt() sky.SkyOptions {
    return .{ .enabled = true, .turbidity = 2.6 };
}

test "horizon brighter than zenith away from the sun" {
    const t = 2.6;
    const sun = [3]f32{ 0.0, 0.76, 0.65 }; // ~49 deg elevation, south
    const zen = sky.skyRadiance(.{ 0.0, 1.0, 0.0 }, sun, t);
    const hor = sky.skyRadiance(.{ 0.87, 0.087, 0.0 }, sun, t); // east, 5 deg up
    try std.testing.expect(lum(hor) > lum(zen) * 1.2);
}

test "sun reddens and dims as elevation falls" {
    const hi = sky.sunLight(@sin(60.0 * std.math.pi / 180.0), 2.6);
    const lo = sky.sunLight(@sin(3.0 * std.math.pi / 180.0), 2.6);
    const hi_rb = hi.color[0] / @max(hi.color[2], 1e-6);
    const lo_rb = lo.color[0] / @max(lo.color[2], 1e-6);
    try std.testing.expect(lo_rb > hi_rb * 1.5);
    try std.testing.expect(lo.intensity < hi.intensity * 0.75);
}

test "turbidity dims the sun and zenith, brightens the horizon" {
    const se = @sin(40.0 * std.math.pi / 180.0);
    const clear = sky.sunLight(se, 2.0);
    const hazy = sky.sunLight(se, 8.0);
    try std.testing.expect(hazy.intensity < clear.intensity);
    const zc = sky.zenithState(se, 2.0);
    const zh = sky.zenithState(se, 8.0);
    try std.testing.expect(zh.lum < zc.lum);
    const hc = sky.horizonState(se, 2.0);
    const hh = sky.horizonState(se, 8.0);
    try std.testing.expect(hh.lum > hc.lum);
}

test "energy sanity sweep stays finite and non-negative" {
    const elevs = [_]f32{ -3.0, 0.0, 2.0, 15.0, 40.0, 70.0, 90.0 };
    const turbs = [_]f32{ 1.0, 2.6, 6.0, 12.0 };
    const views = [_][3]f32{
        .{ 0.0, 1.0, 0.0 },
        .{ 1.0, 0.05, 0.0 },
        .{ 0.0, 0.0, 1.0 },
        .{ 0.0, -0.5, 0.5 },
        .{ 0.3, 0.4, 0.86 },
    };
    for (elevs) |e| {
        const sun = sky.sunDirFromDegrees(e, 30.0);
        for (turbs) |tb| {
            const sl = sky.sunLight(@max(sun[1], -1.0), tb);
            for (sl.color) |c| {
                try std.testing.expect(std.math.isFinite(c) and c >= 0.0 and c <= 1.0);
            }
            try std.testing.expect(std.math.isFinite(sl.intensity) and sl.intensity >= 0.0);
            for (views) |v| {
                const r = sky.skyRadiance(v, sun, tb);
                for (r) |c| {
                    try std.testing.expect(std.math.isFinite(c) and c >= 0.0 and c < 500.0);
                }
            }
            const fr = sky.resolveFrame(.{ .enabled = true, .turbidity = tb }, Vec3.new(sun[0], sun[1], sun[2]), manual());
            for ([_][3]f32{ fr.sun_color, fr.hemi_sky, fr.hemi_ground, fr.fog_tint }) |lane| {
                for (lane) |c| try std.testing.expect(std.math.isFinite(c) and c >= 0.0);
            }
        }
    }
}

test "below-horizon sun floors gracefully" {
    const noon = sky.sunLight(1.0, 2.6);
    const set = sky.sunLight(@sin(-5.0 * std.math.pi / 180.0), 2.6);
    try std.testing.expect(set.intensity < noon.intensity * 0.05);
    try std.testing.expect(std.math.isFinite(set.intensity));
}

test "disabled model returns manual values verbatim" {
    const m = manual();
    const r = sky.resolveFrame(.{}, sunDirElev(49.0), m);
    try std.testing.expect(!r.analytic);
    try std.testing.expectEqual(m.sun_color, r.sun_color);
    try std.testing.expectEqual(m.sun_intensity, r.sun_intensity);
    try std.testing.expectEqual(m.hemi_diffuse, r.hemi_sky);
    try std.testing.expectEqual(m.hemi_ground, r.hemi_ground);
    try std.testing.expectEqual(m.fog_color, r.fog_tint);
}

test "drive flags keep manual values per output" {
    const m = manual();
    var opt = enabledOpt();
    opt.drive_sun = false;
    opt.drive_ambient = false;
    opt.drive_fog = false;
    const r = sky.resolveFrame(opt, sunDirElev(49.0), m);
    try std.testing.expect(r.analytic == false);
    try std.testing.expectEqual(m.sun_color, r.sun_color);
    try std.testing.expectEqual(m.hemi_diffuse, r.hemi_sky);
    try std.testing.expectEqual(m.fog_color, r.fog_tint);
    // Same switch, drives on: model values differ from manual.
    const on = sky.resolveFrame(enabledOpt(), sunDirElev(49.0), m);
    try std.testing.expect(!std.meta.eql(on.sun_color, m.sun_color));
    try std.testing.expect(on.sun_intensity != m.sun_intensity);
}

test "sun disk pixel carries HDR sun over sky" {
    const t = 2.6;
    const sun = sky.sunDirFromDegrees(49.0, 0.0);
    const disk = sky.skyRadiance(sun, sun, t);
    const off = sky.skyRadiance(.{ sun[0] + 0.05, sun[1], sun[2] - 0.05 }, sun, t);
    try std.testing.expect(lum(disk) > lum(off) * 5.0);
}

test "golden values at q0-like sun (T=2.6, elev 49 deg)" {
    const se = @sin(49.0 * std.math.pi / 180.0);
    const sl = sky.sunLight(se, 2.6);
    try std.testing.expectApproxEqAbs(sl.color[0], 0.9629, 1e-3);
    try std.testing.expectApproxEqAbs(sl.color[1], 0.9335, 1e-3);
    try std.testing.expectApproxEqAbs(sl.color[2], 0.8654, 1e-3);
    try std.testing.expectApproxEqAbs(sl.intensity, 2.8886, 1e-3);
    const zen = sky.zenithState(se, 2.6);
    try std.testing.expectApproxEqAbs(zen.lum, 1.1175, 1e-3);
    const hor = sky.horizonState(se, 2.6);
    try std.testing.expectApproxEqAbs(hor.lum, 2.0250, 1e-2);
    const r = sky.resolveFrame(enabledOpt(), sunDirElev(49.0), manual());
    try std.testing.expectApproxEqAbs(r.hemi_sky[0], 0.2355, 1e-2);
    try std.testing.expectApproxEqAbs(r.fog_tint[2], 0.7746, 1e-2);
}

test "analyticParams packs the skybox lane and zeros when off" {
    const off = sky.analyticParams(Vec3.new(0, 1, 0), 2.6, Color3.white, 1.0, false);
    try std.testing.expectEqual([4]f32{ 0, 0, 0, 0 }, off.sky_ctrl);
    try std.testing.expectEqual([4]f32{ 0, 0, 0, 0 }, off.sky_sun_color);
    const sun = sunDirElev(30.0);
    const on = sky.analyticParams(sun, 4.0, Color3.new(1, 0.9, 0.8), 2.0, true);
    try std.testing.expectEqual([4]f32{ 1, 0, 0, 0 }, on.sky_ctrl);
    try std.testing.expectApproxEqAbs(on.sky_sun[1], sun.y, 1e-6);
    try std.testing.expectApproxEqAbs(on.sky_sun[3], 4.0, 1e-6);
    try std.testing.expectEqual([4]f32{ 1.0, 0.9, 0.8, 2.0 }, on.sky_sun_color);
}

// --- Sky v2: clouds + night golden tests (table-driven house style) ---

test "cloud/night sanitizers clamp and fall back" {
    const cases = [_]struct { got: f32, want: f32 }{
        .{ .got = sky.sanitizeCoverage(0.45), .want = 0.45 },
        .{ .got = sky.sanitizeCoverage(-1.0), .want = 0.0 },
        .{ .got = sky.sanitizeCoverage(2.0), .want = 1.0 },
        .{ .got = sky.sanitizeCoverage(std.math.nan(f32)), .want = sky.CLOUD_COVERAGE_DEFAULT },
        .{ .got = sky.sanitizeCloudScale(4.0), .want = 4.0 },
        .{ .got = sky.sanitizeCloudScale(0.1), .want = sky.CLOUD_SCALE_MIN },
        .{ .got = sky.sanitizeCloudScale(99.0), .want = sky.CLOUD_SCALE_MAX },
        .{ .got = sky.sanitizeCloudScale(std.math.inf(f32)), .want = sky.CLOUD_SCALE_DEFAULT },
        .{ .got = sky.sanitizeWindSpeed(0.1), .want = 0.1 },
        .{ .got = sky.sanitizeWindSpeed(-9.0), .want = -sky.WIND_SPEED_MAX },
        .{ .got = sky.sanitizeWindSpeed(std.math.nan(f32)), .want = 0.0 },
        .{ .got = sky.sanitizeStrength(1.5, 1.0), .want = 1.5 },
        .{ .got = sky.sanitizeStrength(-2.0, 1.0), .want = 0.0 },
        .{ .got = sky.sanitizeStrength(99.0, 1.0), .want = 8.0 },
        .{ .got = sky.sanitizeStrength(std.math.nan(f32), 0.6), .want = 0.6 },
    };
    for (cases) |c| try std.testing.expectApproxEqAbs(c.got, c.want, 1e-6);
}

test "cloudMask pins clear, ramps, and stays monotone" {
    // Coverage 0: clear for every noise value (low edge sits at 1.0).
    for ([_]f32{ 0.0, 0.25, 0.5, 0.75, 1.0 }) |n| {
        try std.testing.expectApproxEqAbs(sky.cloudMask(n, 0.0), 0.0, 1e-6);
    }
    // Golden: (0.6 - 0.55) / 0.25 = 0.2 -> smoothstep 0.104.
    try std.testing.expectApproxEqAbs(sky.cloudMask(0.6, 0.45), 0.104, 1e-6);
    try std.testing.expectApproxEqAbs(sky.cloudMask(1.0, 0.45), 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(sky.cloudMask(0.5, 1.0), 1.0, 1e-6);
    // Monotone in noise (fixed coverage) and in coverage (fixed noise).
    var prev: f32 = 0.0;
    for ([_]f32{ 0.0, 0.3, 0.55, 0.7, 0.9, 1.0 }) |n| {
        const m = sky.cloudMask(n, 0.45);
        try std.testing.expect(m >= prev);
        prev = m;
    }
    prev = 0.0;
    for ([_]f32{ 0.0, 0.2, 0.45, 0.7, 1.0 }) |c| {
        const m = sky.cloudMask(0.7, c);
        try std.testing.expect(m >= prev);
        prev = m;
    }
}

test "cloudWindOffset matches dir*speed*time" {
    const w = sky.cloudWindOffset(45.0, 0.02, 10.0);
    try std.testing.expectApproxEqAbs(w[0], 0.14142136, 1e-6);
    try std.testing.expectApproxEqAbs(w[1], 0.14142136, 1e-6);
    const zero = sky.cloudWindOffset(45.0, 0.02, 0.0);
    try std.testing.expectApproxEqAbs(zero[0], 0.0, 1e-7);
    try std.testing.expectApproxEqAbs(zero[1], 0.0, 1e-7);
    const east = sky.cloudWindOffset(0.0, 0.1, 5.0);
    try std.testing.expectApproxEqAbs(east[0], 0.5, 1e-6);
    try std.testing.expectApproxEqAbs(east[1], 0.0, 1e-6);
    // Non-finite time parks the offset at zero (never NaN downstream).
    const bad = sky.cloudWindOffset(45.0, 0.02, std.math.nan(f32));
    try std.testing.expectApproxEqAbs(bad[0], 0.0, 1e-7);
    try std.testing.expectApproxEqAbs(bad[1], 0.0, 1e-7);
}

test "cloudHorizonFade keeps zenith, kills the horizon band" {
    try std.testing.expectApproxEqAbs(sky.cloudHorizonFade(1.0, 1.0), 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(sky.cloudHorizonFade(0.0, 1.0), 0.0, 1e-6);
    try std.testing.expectApproxEqAbs(sky.cloudHorizonFade(-1.0, 1.0), 0.0, 1e-6);
    // Fade 0 disables: clouds reach the horizon.
    for ([_]f32{ -1.0, 0.0, 0.1, 1.0 }) |y| {
        try std.testing.expectApproxEqAbs(sky.cloudHorizonFade(y, 0.0), 1.0, 1e-6);
    }
    // Monotone in view height at full fade.
    var prev: f32 = 0.0;
    for ([_]f32{ -0.5, 0.0, 0.1, 0.2, 0.3, 1.0 }) |y| {
        const m = sky.cloudHorizonFade(y, 1.0);
        try std.testing.expect(m >= prev);
        prev = m;
    }
}

test "nightAmount follows the sun unless forced or disabled" {
    const auto = sky.NightOptions{ .auto = true };
    try std.testing.expectApproxEqAbs(sky.nightAmount(1.0, auto), 0.0, 1e-6);
    try std.testing.expectApproxEqAbs(sky.nightAmount(0.035, auto), 0.0, 1e-6);
    try std.testing.expectApproxEqAbs(sky.nightAmount(-0.105, auto), 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(sky.nightAmount(-1.0, auto), 1.0, 1e-6);
    // Forced night ignores the sun; disabled auto stays 0 even at midnight.
    try std.testing.expectApproxEqAbs(sky.nightAmount(1.0, .{ .enabled = true }), 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(sky.nightAmount(-1.0, .{ .auto = false }), 0.0, 1e-6);
    // Monotone as the sun sinks.
    var prev: f32 = 0.0;
    for ([_]f32{ 0.5, 0.1, 0.0, -0.05, -0.1, -0.5 }) |s| {
        const a = sky.nightAmount(s, auto);
        try std.testing.expect(a >= prev);
        prev = a;
    }
}

test "moonDir defaults aloft and honors overrides" {
    const d = sky.moonDir(.{});
    try std.testing.expectApproxEqAbs(d[0] * d[0] + d[1] * d[1] + d[2] * d[2], 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(d[0], -0.2802, 1e-3);
    try std.testing.expectApproxEqAbs(d[1], 0.5736, 1e-3);
    try std.testing.expectApproxEqAbs(d[2], -0.7698, 1e-3);
    const up = sky.moonDir(.{ .moon_elevation_deg = 90.0, .moon_azimuth_deg = 0.0 });
    try std.testing.expectApproxEqAbs(up[1], 1.0, 1e-5);
}

test "moonDiscCos matches the angular radius" {
    try std.testing.expectApproxEqAbs(sky.moonDiscCos(0.26), 0.9999897, 1e-6);
    try std.testing.expect(sky.moonDiscCos(3.0) < sky.moonDiscCos(0.26));
    // Clamps hold: huge radii pin to the max, NaN to the default.
    try std.testing.expectApproxEqAbs(sky.moonDiscCos(90.0), sky.moonDiscCos(sky.MOON_RADIUS_DEG_MAX), 1e-7);
    try std.testing.expectApproxEqAbs(sky.moonDiscCos(std.math.nan(f32)), sky.moonDiscCos(sky.MOON_RADIUS_DEG_DEFAULT), 1e-7);
}

test "analyticParamsFull matches analyticParams with features off" {
    const sun = sunDirElev(49.0);
    const base = sky.analyticParams(sun, 2.6, Color3.new(1, 0.9, 0.8), 2.0, true);
    const full = sky.analyticParamsFull(sun, 2.6, Color3.new(1, 0.9, 0.8), 2.0, true, .{}, @sin(49.0 * std.math.pi / 180.0));
    try std.testing.expectEqual(base.sky_sun, full.sky_sun);
    try std.testing.expectEqual(base.sky_ctrl, full.sky_ctrl);
    try std.testing.expectEqual(base.sky_sun_color, full.sky_sun_color);
    try std.testing.expectEqual([4]f32{ 0, 0, 0, 0 }, full.sky_cloud);
    try std.testing.expectEqual([4]f32{ 0, 0, 0, 0 }, full.sky_cloud2);
    try std.testing.expectEqual([4]f32{ 0, 0, 0, 0 }, full.sky_night);
    try std.testing.expectEqual([4]f32{ 0, 0, 0, 0 }, full.sky_night2);
    // Disabled packs off bit-identically too.
    const off = sky.analyticParamsFull(sun, 2.6, Color3.white, 1.0, false, .{ .clouds = .{ .enabled = true } }, 1.0);
    try std.testing.expectEqual(sky.analyticParams(sun, 2.6, Color3.white, 1.0, false).sky_ctrl, off.sky_ctrl);
}

test "analyticParamsFull packs cloud and night lanes" {
    const sun = sunDirElev(49.0);
    var opt = sky.SkyOptions{
        .enabled = true,
        .clouds = .{ .enabled = true, .coverage = 0.5, .scale = 4.0, .wind_dir_deg = 90.0, .wind_speed = 0.1 },
        .night = .{ .enabled = true, .star_density = 0.4 },
        .time_sec = 5.0,
    };
    const p = sky.analyticParamsFull(sun, 2.6, Color3.new(1, 0.9, 0.8), 2.0, true, opt, 1.0);
    try std.testing.expectApproxEqAbs(p.sky_cloud[0], 0.5, 1e-6);
    try std.testing.expectApproxEqAbs(p.sky_cloud[1], 4.0, 1e-6);
    try std.testing.expectApproxEqAbs(p.sky_cloud[2], 5.0, 1e-6);
    try std.testing.expectApproxEqAbs(p.sky_cloud[3], 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(p.sky_cloud2[0], 0.0, 1e-6);
    try std.testing.expectApproxEqAbs(p.sky_cloud2[1], 0.5, 1e-6);
    try std.testing.expectApproxEqAbs(p.sky_night[0], 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(p.sky_night[2], 0.4, 1e-6);
    try std.testing.expect(p.sky_moon[3] > 0.999 and p.sky_moon[3] < 1.0);
    // Auto night at noon stays dark-lane zero even with clouds on.
    opt.night.enabled = false;
    const day = sky.analyticParamsFull(sun, 2.6, Color3.new(1, 0.9, 0.8), 2.0, true, opt, 1.0);
    try std.testing.expectEqual([4]f32{ 0, 0, 0, 0 }, day.sky_night);
    try std.testing.expectApproxEqAbs(day.sky_cloud[3], 1.0, 1e-6);
}

test "cloud/night sweep stays finite under hostile inputs" {
    const hostile = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), -9.0, 0.0, 9.0 };
    for (hostile) |h| {
        try std.testing.expect(sky.sanitizeCoverage(h) >= 0.0 and sky.sanitizeCoverage(h) <= 1.0);
        try std.testing.expect(std.math.isFinite(sky.sanitizeCloudScale(h)));
        try std.testing.expect(std.math.isFinite(sky.sanitizeWindSpeed(h)));
        try std.testing.expect(std.math.isFinite(sky.cloudMask(h, h)));
        try std.testing.expect(std.math.isFinite(sky.cloudHorizonFade(h, h)));
        try std.testing.expect(sky.nightAmount(h, .{}) >= 0.0 and sky.nightAmount(h, .{}) <= 1.0);
        try std.testing.expect(std.math.isFinite(sky.moonDiscCos(h)));
        const w = sky.cloudWindOffset(h, h, h);
        try std.testing.expect(std.math.isFinite(w[0]) and std.math.isFinite(w[1]));
        const md = sky.moonDir(.{ .moon_elevation_deg = h, .moon_azimuth_deg = h });
        for (md) |c| try std.testing.expect(std.math.isFinite(c));
    }
}
