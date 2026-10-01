//! Light tests (moved verbatim from `lights.zig`; only the header changed).
//! Reaches the API through the `../lights.zig` facade (same discipline as
//! `mesh/tests.zig` reaching `../mesh.zig`); the import exists only in test
//! builds via the facade's `test` block.
const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Color3 = math.Color3;
const lights_mod = @import("../lights.zig");
const HemisphericLight = lights_mod.HemisphericLight;
const DirectionalLight = lights_mod.DirectionalLight;
const PointLight = lights_mod.PointLight;
const SpotLight = lights_mod.SpotLight;
const AreaLight = lights_mod.AreaLight;
const ClusteredPointLight = lights_mod.ClusteredPointLight;
const max_directional_lights = lights_mod.max_directional_lights;
const max_fill_directionals = lights_mod.max_fill_directionals;
const max_area_lights = lights_mod.max_area_lights;
const max_clustered_lights = lights_mod.max_clustered_lights;
const resolveSunDirection = lights_mod.resolveSunDirection;
const resolveSunColor = lights_mod.resolveSunColor;
const resolveSunIntensity = lights_mod.resolveSunIntensity;
const sunDirectionFromAngles = lights_mod.sunDirectionFromAngles;
const colorTemperatureToRgb = lights_mod.colorTemperatureToRgb;

test "resolveSunDirection falls back to normalized hemi" {
    const hemi = HemisphericLight.init("hemi", .{ .direction = Vec3.new(2.0, 0.0, 0.0) });
    const dir = resolveSunDirection(null, hemi);
    try std.testing.expectApproxEqAbs(dir.x, 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(dir.y, 0.0, 1e-6);
    try std.testing.expectApproxEqAbs(dir.z, 0.0, 1e-6);
}

test "HemisphericLight defaults match Babylon (ground black, white diffuse, up, 1.0)" {
    // babylon.js: this.groundColor=lt(this,r,new Te(0,0,0)) -- the reference
    // bundle reads back [0,0,0] through a NullEngine probe; diffuse (1,1,1),
    // intensity 1, direction (0,1,0). Scene.init authors its own 0.2/0.25/0.3
    // ambient explicitly, so this default only affects callers who build a
    // light from `HemisphericLightOptions{}`.
    const defaults = HemisphericLight.init("hemi", .{});
    try std.testing.expectEqual(Color3.black, defaults.ground_color);
    try std.testing.expectEqual(Color3.white, defaults.diffuse);
    try std.testing.expectEqual(@as(f32, 1.0), defaults.intensity);
    try std.testing.expectEqual(Vec3.up, defaults.direction);
    // The options struct carries the same default.
    const options: lights_mod.HemisphericLightOptions = .{};
    try std.testing.expectEqual(Color3.black, options.ground_color);
}

test "resolveSunColor and resolveSunIntensity fall back to hemi" {
    const hemi = HemisphericLight.init("hemi", .{
        .diffuse = Color3.new(0.5, 0.25, 0.125),
        .intensity = 0.75,
    });
    const color = resolveSunColor(null, hemi);
    try std.testing.expectApproxEqAbs(color.r, 0.5, 1e-6);
    try std.testing.expectApproxEqAbs(color.g, 0.25, 1e-6);
    try std.testing.expectApproxEqAbs(color.b, 0.125, 1e-6);
    try std.testing.expectApproxEqAbs(resolveSunIntensity(null, hemi), 0.75, 1e-6);
}

test "directional light overrides sun resolvers" {
    const hemi = HemisphericLight.init("hemi", .{});
    var sun = DirectionalLight.init("sun", .{
        .direction = Vec3.new(0.0, -2.0, 0.0),
        .diffuse = Color3.new(1.0, 0.5, 0.25),
        .intensity = 2.0,
    });
    const dir = resolveSunDirection(&sun, hemi);
    try std.testing.expectApproxEqAbs(dir.x, 0.0, 1e-6);
    try std.testing.expectApproxEqAbs(dir.y, -1.0, 1e-6);
    try std.testing.expectApproxEqAbs(dir.z, 0.0, 1e-6);
    const color = resolveSunColor(&sun, hemi);
    try std.testing.expectApproxEqAbs(color.r, 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(color.g, 0.5, 1e-6);
    try std.testing.expectApproxEqAbs(color.b, 0.25, 1e-6);
    try std.testing.expectApproxEqAbs(resolveSunIntensity(&sun, hemi), 2.0, 1e-6);
}

test "resolveSunDirection with zero-length direction is safe" {
    const hemi = HemisphericLight.init("hemi", .{ .direction = Vec3.new(0.0, 1.0, 0.0) });
    var sun = DirectionalLight{ .name = "sun", .direction = Vec3.zero };
    const dir = resolveSunDirection(&sun, hemi);
    try std.testing.expect(std.math.isFinite(dir.x));
    try std.testing.expect(std.math.isFinite(dir.y));
    try std.testing.expect(std.math.isFinite(dir.z));
    try std.testing.expectApproxEqAbs(dir.y, 1.0, 1e-6);
}

test "sunDirectionFromAngles computes unit vector for zenith and horizon" {
    // Zenith: elevation = pi/2
    const zenith = sunDirectionFromAngles(0.0, std.math.pi * 0.5);
    try std.testing.expectApproxEqAbs(zenith.x, 0.0, 1e-5);
    try std.testing.expectApproxEqAbs(zenith.y, 1.0, 1e-5);
    try std.testing.expectApproxEqAbs(zenith.z, 0.0, 1e-5);

    // Horizon +Z: elevation = 0, azimuth = 0
    const horizon_z = sunDirectionFromAngles(0.0, 0.0);
    try std.testing.expectApproxEqAbs(horizon_z.x, 0.0, 1e-5);
    try std.testing.expectApproxEqAbs(horizon_z.y, 0.0, 1e-5);
    try std.testing.expectApproxEqAbs(horizon_z.z, 1.0, 1e-5);

    // Horizon +X: elevation = 0, azimuth = pi/2
    const horizon_x = sunDirectionFromAngles(std.math.pi * 0.5, 0.0);
    try std.testing.expectApproxEqAbs(horizon_x.x, 1.0, 1e-5);
    try std.testing.expectApproxEqAbs(horizon_x.y, 0.0, 1e-5);
    try std.testing.expectApproxEqAbs(horizon_x.z, 0.0, 1e-5);
}

test "colorTemperatureToRgb produces warm for low kelvin and cool for high kelvin" {
    const warm = colorTemperatureToRgb(2500.0);
    // Warm: red is 1.0, green ~ 0.62, blue ~ 0.27 (more red than blue)
    try std.testing.expect(warm.r > warm.g);
    try std.testing.expect(warm.g > warm.b);

    const neutral = colorTemperatureToRgb(6500.0);
    // Neutral daylight: balanced channels
    try std.testing.expectApproxEqAbs(neutral.r, 1.0, 0.05);
    try std.testing.expect(neutral.g > 0.9);
    try std.testing.expect(neutral.b > 0.9);

    const cool = colorTemperatureToRgb(10000.0);
    // Cool sky: blue is 1.0, more blue than red
    try std.testing.expect(cool.b >= cool.r);
}

test "disabled directional falls back to hemispheric sun" {
    const hemi = HemisphericLight.init("hemi", .{
        .direction = Vec3.new(0.0, 1.0, 0.0),
        .diffuse = Color3.new(0.5, 0.25, 0.125),
        .intensity = 0.75,
    });
    var sun = DirectionalLight.init("sun", .{
        .direction = Vec3.new(0.0, -1.0, 0.0),
        .diffuse = Color3.new(1.0, 0.5, 0.25),
        .intensity = 2.0,
    });
    sun.is_enabled = false;
    const dir = resolveSunDirection(&sun, hemi);
    try std.testing.expectApproxEqAbs(dir.y, 1.0, 1e-6);
    const color = resolveSunColor(&sun, hemi);
    try std.testing.expectApproxEqAbs(color.r, 0.5, 1e-6);
    try std.testing.expectApproxEqAbs(resolveSunIntensity(&sun, hemi), 0.75, 1e-6);
}

test "SpotLight.getShadowViewProj transforms points in front of spotlight" {
    const spot = SpotLight.init("spot", .{
        .position = Vec3.new(0, 10, 0),
        .direction = Vec3.new(0, -1, 0),
        .range = 20.0,
        .inner_angle_deg = 20.0,
        .outer_angle_deg = 45.0,
        .cast_shadows = true,
    });
    const vp = spot.getShadowViewProj();
    // A point at (0, 0, 0) is 10 units directly in front of the spotlight along -Y.
    const p_world = Vec3.new(0, 0, 0);
    const p_clip_x = vp.m[0] * p_world.x + vp.m[4] * p_world.y + vp.m[8] * p_world.z + vp.m[12];
    const p_clip_y = vp.m[1] * p_world.x + vp.m[5] * p_world.y + vp.m[9] * p_world.z + vp.m[13];
    const p_clip_z = vp.m[2] * p_world.x + vp.m[6] * p_world.y + vp.m[10] * p_world.z + vp.m[14];
    const p_clip_w = vp.m[3] * p_world.x + vp.m[7] * p_world.y + vp.m[11] * p_world.z + vp.m[15];

    try std.testing.expect(p_clip_w > 0.0);
    const ndc_x = p_clip_x / p_clip_w;
    const ndc_y = p_clip_y / p_clip_w;
    const ndc_z = p_clip_z / p_clip_w;

    // Center of cone maps to (0, 0) in XY NDC
    try std.testing.expectApproxEqAbs(ndc_x, 0.0, 1e-4);
    try std.testing.expectApproxEqAbs(ndc_y, 0.0, 1e-4);
    // Depth is within [0, 1]
    try std.testing.expect(ndc_z >= 0.0 and ndc_z <= 1.0);
}

test "PointLight shadows are off by default" {
    const pl = PointLight.init("lamp", .{});
    try std.testing.expect(!pl.cast_shadows);
    try std.testing.expectEqual(@as(f32, 0.002), pl.shadow_bias);
    try std.testing.expectEqual(@as(f32, 0.005), pl.shadow_normal_bias);
    try std.testing.expectEqual(@as(f32, 0.1), pl.shadow_near);
    // A literal without the new fields keeps the same defaults.
    const bare = PointLight{};
    try std.testing.expect(!bare.cast_shadows);
}

test "AreaLight normal and area follow the right/up half-extents" {
    const rect = AreaLight.init("rect", .{
        .center = Vec3.new(0, 2, 0),
        .right = Vec3.new(1, 0, 0),
        .up = Vec3.new(0, 0.5, 0),
    });
    const n = rect.normal();
    try std.testing.expectApproxEqAbs(n.x, 0.0, 1e-6);
    try std.testing.expectApproxEqAbs(n.y, 0.0, 1e-6);
    try std.testing.expectApproxEqAbs(n.z, 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(rect.area(), 2.0, 1e-6);

    // Degenerate rects (zero axis, parallel axes) are safe: zero normal
    // and zero area, never NaN.
    const flat = AreaLight.init("flat", .{ .right = Vec3.zero, .up = Vec3.new(0, 1, 0) });
    try std.testing.expectEqual(Vec3.zero, flat.normal());
    try std.testing.expectEqual(@as(f32, 0.0), flat.area());
    const parallel = AreaLight.init("par", .{ .right = Vec3.new(1, 0, 0), .up = Vec3.new(2, 0, 0) });
    try std.testing.expectEqual(Vec3.zero, parallel.normal());
    try std.testing.expectEqual(@as(f32, 0.0), parallel.area());
}

test "PointLight.getShadowFaceViewProj centers each axis on its face" {
    const pl = PointLight.init("lamp", .{
        .position = Vec3.new(1, 2, 3),
        .range = 10.0,
        .cast_shadows = true,
    });
    const targets = [_]Vec3{
        Vec3.new(2, 2, 3), // +X
        Vec3.new(0, 2, 3), // -X
        Vec3.new(1, 3, 3), // +Y
        Vec3.new(1, 1, 3), // -Y
        Vec3.new(1, 2, 4), // +Z
        Vec3.new(1, 2, 2), // -Z
    };
    for (targets, 0..) |t, face| {
        const vp = pl.getShadowFaceViewProj(face);
        const cx = vp.m[0] * t.x + vp.m[4] * t.y + vp.m[8] * t.z + vp.m[12];
        const cy = vp.m[1] * t.x + vp.m[5] * t.y + vp.m[9] * t.z + vp.m[13];
        const cz = vp.m[2] * t.x + vp.m[6] * t.y + vp.m[10] * t.z + vp.m[14];
        const cw = vp.m[3] * t.x + vp.m[7] * t.y + vp.m[11] * t.z + vp.m[15];
        try std.testing.expect(cw > 0.0);
        try std.testing.expectApproxEqAbs(cx / cw, 0.0, 1e-4);
        try std.testing.expectApproxEqAbs(cy / cw, 0.0, 1e-4);
        const ndc_z = cz / cw;
        try std.testing.expect(ndc_z >= 0.0 and ndc_z <= 1.0);
    }
}
