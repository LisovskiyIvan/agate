//! Tests for `light_rig.zig` (moved from `light_rig.zig` inline blocks).
const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color3 = math.Color3;
const lights = @import("../lights.zig");
const light_selection = @import("light_selection.zig");
const passes = @import("../passes/mod.zig");
const rig_mod = @import("light_rig.zig");
const LightRig = rig_mod.LightRig;

test "createDirectionalLight replaces the primary sun" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);

    const first = try rig.createDirectionalLight(allocator, "sun1", .{});
    try std.testing.expect(rig.directional == first);

    const second = try rig.createDirectionalLight(allocator, "sun2", .{ .intensity = 2.0 });
    try std.testing.expect(rig.directional == second);
    try std.testing.expectEqualStrings("sun2", rig.directional.?.name);
    try std.testing.expectEqual(@as(f32, 2.0), rig.directional.?.intensity);

    // Sun resolution prefers the directional once one is set.
    const sun_dir = rig.sunDirection();
    const want_dir = second.direction.normalize();
    try std.testing.expectEqual(want_dir.x, sun_dir.x);
    try std.testing.expectEqual(want_dir.y, sun_dir.y);
    try std.testing.expectEqual(want_dir.z, sun_dir.z);
}

test "single directional packs slot 0 with the rest zeroed (legacy-neutral)" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);
    rig.hysteresis_enabled = false;

    // No suns at all: all slots zeroed, sun falls back to hemispheric.
    var pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    for (0..4) |i| {
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.directional_dir[i]);
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.directional_color_int[i]);
    }
    try std.testing.expectEqual(@as(usize, 0), rig.directionalCount());
    try std.testing.expect(rig.directionalAt(0) == null);

    // Exactly one light (the default): slot 0 mirrors the primary with a
    // normalized direction, slots 1..3 stay zeroed — the shader fill loop
    // (1..3, skip on intensity <= 0) then adds nothing bit-identically.
    const sun = try rig.createDirectionalLight(allocator, "sun", .{
        .direction = Vec3.new(0.0, -2.0, 0.0),
        .diffuse = Color3.new(1.0, 0.5, 0.25),
        .intensity = 2.0,
    });
    _ = sun;
    pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual([4]f32{ 0.0, -1.0, 0.0, 0.0 }, pack.directional_dir[0]);
    try std.testing.expectEqual([4]f32{ 1.0, 0.5, 0.25, 2.0 }, pack.directional_color_int[0]);
    for (1..4) |i| {
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.directional_dir[i]);
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.directional_color_int[i]);
    }
    try std.testing.expectEqual(@as(usize, 1), rig.directionalCount());
}

test "fills pack in creation order, disabled entries zeroed" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);
    rig.hysteresis_enabled = false;

    _ = try rig.createDirectionalLight(allocator, "sun", .{
        .direction = Vec3.new(0.0, -1.0, 0.0),
        .intensity = 1.0,
    });
    const fill_a = try rig.addDirectionalLight(allocator, "fill-a", .{
        .direction = Vec3.new(2.0, 0.0, 0.0),
        .diffuse = Color3.new(0.0, 1.0, 0.0),
        .intensity = 0.5,
    });
    const fill_b = try rig.addDirectionalLight(allocator, "fill-b", .{
        .direction = Vec3.new(0.0, 0.0, 3.0),
        .diffuse = Color3.new(0.0, 0.0, 1.0),
        .intensity = 0.25,
    });
    fill_b.is_enabled = false;
    try std.testing.expectEqual(@as(usize, 3), rig.directionalCount());
    try std.testing.expect(rig.directionalAt(1) == fill_a);
    try std.testing.expect(rig.directionalAt(2) == fill_b);
    try std.testing.expect(rig.directionalAt(3) == null);

    const pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    // Slot 0 is the primary, slot 1 the enabled fill (normalized).
    try std.testing.expectEqual([4]f32{ 0.0, -1.0, 0.0, 0.0 }, pack.directional_dir[0]);
    try std.testing.expectEqual([4]f32{ 0.0, 1.0, 0.0, 0.5 }, pack.directional_color_int[1]);
    try std.testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 0.0 }, pack.directional_dir[1]);
    // Disabled fill zeroed, unused slot zeroed.
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.directional_dir[2]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.directional_color_int[2]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.directional_dir[3]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.directional_color_int[3]);
}

test "addDirectionalLight caps at three fills with a hard error" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);

    _ = try rig.createDirectionalLight(allocator, "sun", .{});
    _ = try rig.addDirectionalLight(allocator, "f1", .{});
    _ = try rig.addDirectionalLight(allocator, "f2", .{});
    _ = try rig.addDirectionalLight(allocator, "f3", .{});
    try std.testing.expectEqual(@as(usize, 4), rig.directionalCount());
    // Beyond 4 total: hard error, never silent clamp or replacement.
    try std.testing.expectError(error.TooManyDirectionalLights, rig.addDirectionalLight(allocator, "f4", .{}));
    try std.testing.expectEqual(@as(usize, 4), rig.directionalCount());
}

test "primary replacement keeps fills and sun resolution ignores fills" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);

    _ = try rig.createDirectionalLight(allocator, "sun1", .{
        .direction = Vec3.new(0.0, -1.0, 0.0),
        .diffuse = Color3.new(1.0, 0.0, 0.0),
        .intensity = 1.0,
    });
    const fill = try rig.addDirectionalLight(allocator, "fill", .{
        .direction = Vec3.new(1.0, 0.0, 0.0),
        .diffuse = Color3.new(0.0, 1.0, 0.0),
        .intensity = 5.0,
    });
    const sun2 = try rig.createDirectionalLight(allocator, "sun2", .{
        .direction = Vec3.new(0.0, 0.0, -1.0),
        .diffuse = Color3.new(0.0, 0.0, 1.0),
        .intensity = 2.0,
    });
    // Replacement swaps only slot 0; the fill pointer survives.
    try std.testing.expect(rig.directional == sun2);
    try std.testing.expect(rig.directionalAt(1) == fill);
    try std.testing.expectEqual(@as(usize, 2), rig.directionalCount());
    // Significance/ordering for the primary is unchanged: fills never win
    // the sun, no matter how intense.
    const dir = rig.sunDirection();
    try std.testing.expectApproxEqAbs(dir.z, -1.0, 1e-6);
    const col = rig.sunColor();
    try std.testing.expectApproxEqAbs(col.b, 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(rig.sunIntensity(), 2.0, 1e-6);
}

test "disabled primary zeroes slot 0 and falls back to hemispheric sun" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{ .direction = Vec3.new(0.0, 1.0, 0.0) });
    defer rig.deinit(allocator);
    rig.hysteresis_enabled = false;

    const sun = try rig.createDirectionalLight(allocator, "sun", .{
        .direction = Vec3.new(0.0, -1.0, 0.0),
        .intensity = 3.0,
    });
    sun.is_enabled = false;
    const pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.directional_dir[0]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.directional_color_int[0]);
    const dir = rig.sunDirection();
    try std.testing.expectApproxEqAbs(dir.y, 1.0, 1e-6);
}

test "packFrame packs selected point and spot lights into uniform arrays" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);
    // This test asserts the raw packing layout, not the fade transitions —
    // the legacy instant-swap path keeps it independent of fade timing.
    rig.hysteresis_enabled = false;

    const eye = Vec3.zero;
    // Empty rig: zero counts, zeroed shadow params.
    var pack = rig.packFrame(eye, true, 1.0 / 60.0);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.counts);
    try std.testing.expectEqual(@as(usize, 0), pack.num_spot_shadows);

    const pl = try rig.createPointLight(allocator, "lamp", .{
        .position = Vec3.new(1, 2, 3),
        .range = 10.0,
        .intensity = 2.0,
    });
    const sl = try rig.createSpotLight(allocator, "cone", .{
        .position = Vec3.new(0, 5, 0),
        .direction = Vec3.new(0, -1, 0),
        .range = 20.0,
        .intensity = 4.0,
    });
    sl.cast_shadows = true;
    sl.shadow_bias = 0.25;
    sl.shadow_normal_bias = 0.5;

    pack = rig.packFrame(eye, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(f32, 1.0), pack.counts[0]);
    try std.testing.expectEqual(@as(f32, 1.0), pack.counts[1]);
    try std.testing.expectEqual([4]f32{ pl.position.x, pl.position.y, pl.position.z, pl.range }, pack.point_pos_range[0]);
    try std.testing.expectEqual([4]f32{ 4.0, 0.0, 0.0, 0.0 }, pack.spot_intensity[0]);
    // Shadow-casting spot contributes one entry for the depth pass.
    try std.testing.expectEqual(@as(usize, 1), pack.num_spot_shadows);
    try std.testing.expectEqual([4]f32{ 1.0, 0.25, 0.5, 0.0 }, pack.spot_shadow_params[0]);

    // Disabled shadows clear the params but keep the light packing.
    pack = rig.packFrame(eye, false, 1.0 / 60.0);
    try std.testing.expectEqual(@as(usize, 0), pack.num_spot_shadows);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.spot_shadow_params[0]);
    try std.testing.expectEqual(@as(f32, 1.0), pack.counts[1]);
}

test "packFrame scales intensity by the slot fade factor" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);

    _ = try rig.createPointLight(allocator, "lamp", .{
        .position = Vec3.new(1, 0, 0),
        .range = 10.0,
        .intensity = 2.0,
    });
    const eye = Vec3.zero;
    const dt: f32 = 0.25 / 2.0; // two frames per half of the fade

    // First frame: the light just claimed its slot at factor 0 — color
    // channels pass through, the intensity component is scaled to zero.
    var pack = rig.packFrame(eye, false, dt);
    try std.testing.expectEqual(@as(f32, 1.0), pack.counts[0]);
    try std.testing.expectEqual(@as(f32, 0.0), pack.point_color_int[0][3]);

    // Halfway through the fade-in the packed intensity is scaled.
    pack = rig.packFrame(eye, false, dt);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), pack.point_color_int[0][3], 1e-6);

    // Fade complete: full intensity, steady from here on.
    pack = rig.packFrame(eye, false, dt);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), pack.point_color_int[0][3], 1e-6);
    pack = rig.packFrame(eye, false, dt);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), pack.point_color_int[0][3], 1e-6);
}

test "packFrame leaves point shadow uniforms zeroed when nothing casts" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);
    rig.hysteresis_enabled = false;

    // Empty rig: neutrality by construction.
    var pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(usize, 0), pack.num_point_shadows);
    for (pack.point_shadow_params) |p| {
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, p);
    }

    // A non-casting point light keeps the shadow lanes zeroed (off by
    // default): lighting still packs, the shader shadow path early-outs.
    _ = try rig.createPointLight(allocator, "lamp", .{
        .position = Vec3.new(1, 2, 3),
        .range = 10.0,
        .intensity = 2.0,
    });
    pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(f32, 1.0), pack.counts[0]);
    try std.testing.expectEqual(@as(usize, 0), pack.num_point_shadows);
    for (pack.point_shadow_params) |p| {
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, p);
    }
    for (pack.point_view_proj) |m| try std.testing.expectEqual(Mat4.identity, m);
}

test "packFrame caps point shadow casters at two by significance" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);
    rig.hysteresis_enabled = false;
    const eye = Vec3.zero;

    // Scores at the eye (intensity * range / (1 + dist^2)):
    // c0 at (1,0,0): 1*10/2 = 5; c1 at origin: 3*10/1 = 30;
    // c2 at (3,0,0): 4*10/10 = 4. Pack order (instant): c1, c0, c2.
    _ = try rig.createPointLight(allocator, "c0", .{ .position = Vec3.new(1, 0, 0), .intensity = 1.0, .range = 10.0, .cast_shadows = true });
    const c1 = try rig.createPointLight(allocator, "c1", .{ .position = Vec3.zero, .intensity = 3.0, .range = 10.0, .cast_shadows = true });
    _ = try rig.createPointLight(allocator, "c2", .{ .position = Vec3.new(3, 0, 0), .intensity = 4.0, .range = 10.0, .cast_shadows = true });
    c1.shadow_bias = 0.25;
    c1.shadow_normal_bias = 0.5;

    const pack = rig.packFrame(eye, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(f32, 3.0), pack.counts[0]);
    // Two shadow slots taken (6 face tiles each), the weakest caster drops.
    try std.testing.expectEqual(@as(usize, 12), pack.num_point_shadows);
    // Pack indices: 0 = c1 (strongest), 1 = c0, 2 = c2 (dropped).
    try std.testing.expectEqual([4]f32{ 1.0, 0.25, 0.5, 0.0 }, pack.point_shadow_params[0]);
    try std.testing.expectEqual(@as(f32, 2.0), pack.point_shadow_params[1][0]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.point_shadow_params[2]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.point_shadow_params[3]);
    // Tiles cover both slot rows: faces 0..5 at y 0 and y 256.
    for (pack.point_shadows[0..6], 0..) |t, f| {
        try std.testing.expectEqual(@as(i32, @intCast(f)) * 256, t.tile_x);
        try std.testing.expectEqual(@as(i32, 0), t.tile_y);
    }
    for (pack.point_shadows[6..12], 0..) |t, f| {
        try std.testing.expectEqual(@as(i32, @intCast(f)) * 256, t.tile_x);
        try std.testing.expectEqual(@as(i32, 256), t.tile_y);
    }
}

test "addAreaLight caps at two with a hard error" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);

    _ = try rig.addAreaLight(allocator, "a0", .{});
    _ = try rig.addAreaLight(allocator, "a1", .{});
    try std.testing.expectEqual(@as(usize, 2), rig.areaLightCount());
    // Beyond the cap: hard error, never silent clamp or replacement.
    try std.testing.expectError(error.TooManyAreaLights, rig.addAreaLight(allocator, "a2", .{}));
    try std.testing.expectEqual(@as(usize, 2), rig.areaLightCount());
}

test "area lights pack creation-order lanes, disabled entries zeroed" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);
    rig.hysteresis_enabled = false;

    // Empty rig: all area lanes zeroed (zero lights render bit-identically).
    var pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    for (0..lights.max_area_lights) |i| {
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.area_center_int[i]);
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.area_right[i]);
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.area_up[i]);
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.area_color[i]);
    }

    const a0 = try rig.addAreaLight(allocator, "key", .{
        .center = Vec3.new(1.0, 2.0, 3.0),
        .right = Vec3.new(2.0, 0.0, 0.0),
        .up = Vec3.new(0.0, 0.5, 0.0),
        .color = Color3.new(1.0, 0.5, 0.25),
        .intensity = 3.0,
    });
    const a1 = try rig.addAreaLight(allocator, "fill", .{
        .center = Vec3.new(-1.0, 0.0, 0.0),
        .intensity = 0.5,
    });
    a1.is_enabled = false;
    try std.testing.expect(rig.getAreaLight(0) == a0);
    try std.testing.expect(rig.getAreaLight(1) == a1);
    try std.testing.expect(rig.getAreaLight(2) == null);

    pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual([4]f32{ 1.0, 2.0, 3.0, 3.0 }, pack.area_center_int[0]);
    try std.testing.expectEqual([4]f32{ 2.0, 0.0, 0.0, 0.0 }, pack.area_right[0]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.5, 0.0, 0.0 }, pack.area_up[0]);
    try std.testing.expectEqual([4]f32{ 1.0, 0.5, 0.25, 0.0 }, pack.area_color[0]);
    // Disabled light: zeroed lane, unused w stays 0.
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.area_center_int[1]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.area_color[1]);

    // Re-enable: second lane packs verbatim (defaults: unit axes).
    a1.is_enabled = true;
    pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual([4]f32{ -1.0, 0.0, 0.0, 0.5 }, pack.area_center_int[1]);
    try std.testing.expectEqual([4]f32{ 0.5, 0.0, 0.0, 0.0 }, pack.area_right[1]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.5, 0.0, 0.0 }, pack.area_up[1]);
}

test "removeAreaLight destroys order-preserving, out-of-range is a no-op" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);

    const a0 = try rig.addAreaLight(allocator, "a0", .{});
    const a1 = try rig.addAreaLight(allocator, "a1", .{});
    rig.removeAreaLight(allocator, 7); // no-op
    try std.testing.expectEqual(@as(usize, 2), rig.areaLightCount());
    rig.removeAreaLight(allocator, 0);
    try std.testing.expectEqual(@as(usize, 1), rig.areaLightCount());
    try std.testing.expect(rig.getAreaLight(0) == a1);
    _ = a0;
    rig.removeAreaLight(allocator, 0);
    try std.testing.expectEqual(@as(usize, 0), rig.areaLightCount());
    try std.testing.expect(rig.getAreaLight(0) == null);
}

test "area packing is frame-independent (no fade, no selection)" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);
    // Hysteresis ON here on purpose: area lanes must still pack verbatim
    // on the very first frame (no enter-fade) and identically for any dt.
    _ = try rig.addAreaLight(allocator, "key", .{
        .center = Vec3.new(0.0, 3.0, 0.0),
        .intensity = 2.0,
    });
    const eye = Vec3.new(10.0, 0.0, 0.0);
    const p0 = rig.packFrame(eye, true, 0.016);
    const p1 = rig.packFrame(eye, true, 0.5);
    try std.testing.expectEqual([4]f32{ 0.0, 3.0, 0.0, 2.0 }, p0.area_center_int[0]);
    try std.testing.expectEqual(p0.area_center_int, p1.area_center_int);
    try std.testing.expectEqual(p0.area_right, p1.area_right);
    try std.testing.expectEqual(p0.area_up, p1.area_up);
    try std.testing.expectEqual(p0.area_color, p1.area_color);
}

test "packFrame gates point shadows on enabled lights and global shadows" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);
    rig.hysteresis_enabled = false;
    const eye = Vec3.zero;

    const caster = try rig.createPointLight(allocator, "caster", .{
        .position = Vec3.zero,
        .intensity = 5.0,
        .range = 10.0,
        .cast_shadows = true,
    });

    // Global shadows off: no tiles, zeroed params, light still packs.
    var pack = rig.packFrame(eye, false, 1.0 / 60.0);
    try std.testing.expectEqual(@as(f32, 1.0), pack.counts[0]);
    try std.testing.expectEqual(@as(usize, 0), pack.num_point_shadows);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.point_shadow_params[0]);

    // Disabled light: skipped even with shadows on.
    caster.is_enabled = false;
    pack = rig.packFrame(eye, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(f32, 0.0), pack.counts[0]);
    try std.testing.expectEqual(@as(usize, 0), pack.num_point_shadows);

    // Re-enabled: one slot, six face tiles.
    caster.is_enabled = true;
    pack = rig.packFrame(eye, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(usize, 6), pack.num_point_shadows);
    try std.testing.expectEqual(@as(f32, 1.0), pack.point_shadow_params[0][0]);
}

test "addClusteredPointLight caps at 64 with a hard error" {
    var rig = LightRig.init("hemi", .{});
    // Value array: no allocator traffic, but deinit stays idempotent.
    const allocator = std.testing.allocator;
    defer rig.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), rig.clusteredPointLightCount());
    try std.testing.expect(rig.getClusteredPointLight(0) == null);

    var i: usize = 0;
    while (i < lights.max_clustered_lights) : (i += 1) {
        const idx = try rig.addClusteredPointLight(Vec3.new(@floatFromInt(i), 0, 0), .{});
        try std.testing.expectEqual(i, idx);
    }
    try std.testing.expectEqual(lights.max_clustered_lights, rig.clusteredPointLightCount());
    // Beyond the cap: hard error, never silent clamp or replacement.
    try std.testing.expectError(error.TooManyClusteredLights, rig.addClusteredPointLight(Vec3.zero, .{}));
    try std.testing.expectEqual(lights.max_clustered_lights, rig.clusteredPointLightCount());
}

test "clustered lights pack creation-order lanes, disabled entries zeroed" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);

    // Empty pool: zero count, all lanes zeroed (the tile build writes empty
    // headers and the shader takes the exact legacy path).
    var pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(usize, 0), pack.clustered_count);
    for (0..lights.max_clustered_lights) |k| {
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.clustered_pos_range[k]);
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.clustered_color_int[k]);
    }

    const idx0 = try rig.addClusteredPointLight(Vec3.new(1.0, 2.0, 3.0), .{
        .color = Color3.new(1.0, 0.5, 0.25),
        .intensity = 3.0,
        .radius = 7.0,
    });
    const idx1 = try rig.addClusteredPointLight(Vec3.new(-1.0, 0.0, 0.0), .{ .intensity = 0.5 });
    try std.testing.expectEqual(@as(usize, 0), idx0);
    try std.testing.expectEqual(@as(usize, 1), idx1);
    try std.testing.expect(rig.getClusteredPointLight(0).?.position.x == 1.0);
    try std.testing.expect(rig.getClusteredPointLight(2) == null);

    // Packed index == creation index (stable identity for tile lists).
    pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(usize, 2), pack.clustered_count);
    try std.testing.expectEqual([4]f32{ 1.0, 2.0, 3.0, 7.0 }, pack.clustered_pos_range[0]);
    try std.testing.expectEqual([4]f32{ 1.0, 0.5, 0.25, 3.0 }, pack.clustered_color_int[0]);
    try std.testing.expectEqual([4]f32{ -1.0, 0.0, 0.0, 10.0 }, pack.clustered_pos_range[1]);

    // Disable: lane zeroes out (tile build + shader skip it), count still
    // stages the owned lanes.
    rig.getClusteredPointLight(0).?.is_enabled = false;
    pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(usize, 2), pack.clustered_count);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.clustered_pos_range[0]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.clustered_color_int[0]);
    try std.testing.expectEqual([4]f32{ -1.0, 0.0, 0.0, 10.0 }, pack.clustered_pos_range[1]);
    rig.getClusteredPointLight(0).?.is_enabled = true;

    // Legacy lanes are untouched by the clustered pool (append-last: the
    // legacy top-k selection never sees these lights).
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.counts);
}

test "removeClusteredPointLight destroys order-preserving, out-of-range is a no-op" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);

    _ = try rig.addClusteredPointLight(Vec3.new(1, 0, 0), .{});
    _ = try rig.addClusteredPointLight(Vec3.new(2, 0, 0), .{});
    _ = try rig.addClusteredPointLight(Vec3.new(3, 0, 0), .{});
    rig.removeClusteredPointLight(9); // no-op
    try std.testing.expectEqual(@as(usize, 3), rig.clusteredPointLightCount());
    rig.removeClusteredPointLight(0);
    try std.testing.expectEqual(@as(usize, 2), rig.clusteredPointLightCount());
    // Order-preserving: the tail shifted down (index 0 now holds x=2).
    try std.testing.expectEqual(@as(f32, 2.0), rig.getClusteredPointLight(0).?.position.x);
    try std.testing.expectEqual(@as(f32, 3.0), rig.getClusteredPointLight(1).?.position.x);
    rig.removeClusteredPointLight(1);
    rig.removeClusteredPointLight(0);
    try std.testing.expectEqual(@as(usize, 0), rig.clusteredPointLightCount());
    try std.testing.expect(rig.getClusteredPointLight(0) == null);
    // Empty again: pack returns to the zeroed legacy-neutral state.
    const pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(usize, 0), pack.clustered_count);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.clustered_pos_range[0]);
}

test "clustered packing is frame-independent (no fade, no selection)" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);
    // Hysteresis ON here on purpose: clustered lanes must still pack
    // verbatim on the very first frame (no enter-fade) and identically for
    // any dt — tiling culls per tile instead of fading slots.
    _ = try rig.addClusteredPointLight(Vec3.new(0.0, 3.0, 0.0), .{ .intensity = 2.0 });
    const eye = Vec3.new(10.0, 0.0, 0.0);
    const p0 = rig.packFrame(eye, true, 0.016);
    const p1 = rig.packFrame(eye, true, 0.5);
    try std.testing.expectEqual(@as(usize, 1), p0.clustered_count);
    try std.testing.expectEqual([4]f32{ 0.0, 3.0, 0.0, 10.0 }, p0.clustered_pos_range[0]);
    try std.testing.expectEqual([4]f32{ 1.0, 1.0, 1.0, 2.0 }, p0.clustered_color_int[0]);
    try std.testing.expectEqual(p0.clustered_pos_range, p1.clustered_pos_range);
    try std.testing.expectEqual(p0.clustered_color_int, p1.clustered_color_int);
}

test "addClusteredSpotLight stages creation-order lanes, get returns slot, cap at 32 errors" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);

    var pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(usize, 0), pack.clustered_spot_count);
    for (0..lights.max_clustered_spots) |k| {
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.clustered_spot_pos_range[k]);
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.clustered_spot_dir_inner[k]);
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.clustered_spot_color_outer[k]);
        try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, pack.clustered_spot_intensity[k]);
    }

    const idx0 = try rig.addClusteredSpotLight(Vec3.new(1.0, 2.0, 3.0), .{
        .direction = Vec3.new(0.0, -1.0, 0.0),
        .color = Color3.new(1.0, 0.5, 0.25),
        .intensity = 3.0,
        .range = 12.0,
        .inner_angle_deg = 20.0,
        .outer_angle_deg = 40.0,
    });
    const idx1 = try rig.addClusteredSpotLight(Vec3.new(-1.0, 0.0, 0.0), .{ .intensity = 0.5 });
    try std.testing.expectEqual(@as(usize, 0), idx0);
    try std.testing.expectEqual(@as(usize, 1), idx1);
    try std.testing.expect(rig.getClusteredSpotLight(0).?.position.x == 1.0);
    try std.testing.expect(rig.getClusteredSpotLight(2) == null);

    pack = rig.packFrame(Vec3.zero, true, 1.0 / 60.0);
    try std.testing.expectEqual(@as(usize, 2), pack.clustered_spot_count);
    try std.testing.expectEqual([4]f32{ 1.0, 2.0, 3.0, 12.0 }, pack.clustered_spot_pos_range[0]);
    const cos_inner = @cos(20.0 * (std.math.pi / 180.0));
    const cos_outer = @cos(40.0 * (std.math.pi / 180.0));
    try std.testing.expectApproxEqAbs(pack.clustered_spot_dir_inner[0][3], cos_inner, 1e-5);
    try std.testing.expectApproxEqAbs(pack.clustered_spot_color_outer[0][3], cos_outer, 1e-5);
    try std.testing.expectEqual([4]f32{ 3.0, 0.0, 0.0, 0.0 }, pack.clustered_spot_intensity[0]);

    // Cap at max_clustered_spots
    var k: usize = 2;
    while (k < lights.max_clustered_spots) : (k += 1) {
        _ = try rig.addClusteredSpotLight(Vec3.zero, .{});
    }
    try std.testing.expectEqual(lights.max_clustered_spots, rig.clusteredSpotLightCount());
    try std.testing.expectError(error.TooManyClusteredLights, rig.addClusteredSpotLight(Vec3.zero, .{}));
}

test "removeClusteredSpotLight destroys order-preserving, out-of-range is a no-op" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);

    _ = try rig.addClusteredSpotLight(Vec3.new(1, 0, 0), .{});
    _ = try rig.addClusteredSpotLight(Vec3.new(2, 0, 0), .{});
    _ = try rig.addClusteredSpotLight(Vec3.new(3, 0, 0), .{});
    rig.removeClusteredSpotLight(99); // no-op
    try std.testing.expectEqual(@as(usize, 3), rig.clusteredSpotLightCount());
    rig.removeClusteredSpotLight(0);
    try std.testing.expectEqual(@as(usize, 2), rig.clusteredSpotLightCount());
    try std.testing.expectEqual(@as(f32, 2.0), rig.getClusteredSpotLight(0).?.position.x);
    try std.testing.expectEqual(@as(f32, 3.0), rig.getClusteredSpotLight(1).?.position.x);
    rig.removeClusteredSpotLight(1);
    rig.removeClusteredSpotLight(0);
    try std.testing.expectEqual(@as(usize, 0), rig.clusteredSpotLightCount());
    try std.testing.expect(rig.getClusteredSpotLight(0) == null);
}

test "packSpotShadows selects up to 2 casters by significance with atlas page tile origins" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);
    rig.hysteresis_enabled = false;

    _ = try rig.createSpotLight(allocator, "close_bright", .{
        .position = Vec3.new(0, 2, 0),
        .direction = Vec3.new(0, -1, 0),
        .range = 20.0,
        .intensity = 10.0,
        .cast_shadows = true,
        .shadow_bias = 0.003,
        .shadow_normal_bias = 0.006,
    });
    _ = try rig.createSpotLight(allocator, "mid", .{
        .position = Vec3.new(0, 5, 0),
        .direction = Vec3.new(0, -1, 0),
        .range = 20.0,
        .intensity = 5.0,
        .cast_shadows = true,
        .shadow_bias = 0.002,
        .shadow_normal_bias = 0.004,
    });

    const eye = Vec3.zero;
    const pack = rig.packFrame(eye, true, 1.0 / 60.0);

    try std.testing.expectEqual(@as(usize, 2), pack.num_spot_shadows);

    try std.testing.expectEqual(@as(i32, 0), pack.spot_shadows[0].tile_x);
    try std.testing.expectEqual(@as(i32, 0), pack.spot_shadows[0].tile_y);
    try std.testing.expectEqual([4]f32{ 1.0, 0.003, 0.006, 0.0 }, pack.spot_shadow_params[pack.spot_shadows[0].spot_index]);

    try std.testing.expectEqual(@as(i32, 512), pack.spot_shadows[1].tile_x);
    try std.testing.expectEqual(@as(i32, 0), pack.spot_shadows[1].tile_y);
    try std.testing.expectEqual([4]f32{ 1.0, 0.002, 0.004, 0.0 }, pack.spot_shadow_params[pack.spot_shadows[1].spot_index]);
}

test "packFrame consolidates local points and spots into unified clustered pool with shadow tags" {
    const allocator = std.testing.allocator;
    var rig = LightRig.init("hemi", .{});
    defer rig.deinit(allocator);
    rig.hysteresis_enabled = false;

    // Create 2 point lights (1 casting shadows)
    _ = try rig.createPointLight(allocator, "p0", .{
        .position = Vec3.new(1, 2, 3),
        .range = 10.0,
        .intensity = 5.0,
        .cast_shadows = true,
        .shadow_bias = 0.005,
    });
    _ = try rig.createPointLight(allocator, "p1", .{
        .position = Vec3.new(4, 5, 6),
        .range = 8.0,
        .intensity = 2.0,
    });

    // Create 2 spot lights (1 casting shadows)
    _ = try rig.createSpotLight(allocator, "s0", .{
        .position = Vec3.new(0, 5, 0),
        .direction = Vec3.new(0, -1, 0),
        .range = 15.0,
        .intensity = 4.0,
        .inner_angle_deg = 20.0,
        .outer_angle_deg = 35.0,
        .cast_shadows = true,
        .shadow_bias = 0.004,
    });
    _ = try rig.createSpotLight(allocator, "s1", .{
        .position = Vec3.new(1, 1, 1),
        .range = 12.0,
        .intensity = 1.0,
    });

    const eye = Vec3.zero;
    const pack = rig.packFrame(eye, true, 1.0 / 60.0);

    // Total 4 lights consolidated into clustered pool
    try std.testing.expectEqual(@as(usize, 4), pack.clustered_count);

    // Light 0: Point light with shadow (slot 0)
    try std.testing.expectEqual([4]f32{ 1, 2, 3, 10 }, pack.clustered_lights[0].pos_range);
    try std.testing.expect(pack.clustered_lights[0].dir_inner[3] < -1.5); // point light marker
    try std.testing.expectEqual(@as(f32, 1.0), pack.clustered_lights[0].spot_params[1]); // point shadow type
    try std.testing.expectEqual(@as(f32, 0.0), pack.clustered_lights[0].spot_params[2]); // shadow slot 0
    try std.testing.expectEqual(@as(f32, 0.005), pack.clustered_lights[0].spot_params[3]); // shadow bias

    // Light 1: Point light without shadow
    try std.testing.expectEqual([4]f32{ 4, 5, 6, 8 }, pack.clustered_lights[1].pos_range);
    try std.testing.expect(pack.clustered_lights[1].dir_inner[3] < -1.5);
    try std.testing.expectEqual(@as(f32, 0.0), pack.clustered_lights[1].spot_params[1]); // no shadow

    // Light 2: Spot light with shadow (slot 0)
    try std.testing.expectEqual([4]f32{ 0, 5, 0, 15 }, pack.clustered_lights[2].pos_range);
    const cos_inner = @cos(20.0 * (std.math.pi / 180.0));
    const cos_outer = @cos(35.0 * (std.math.pi / 180.0));
    try std.testing.expectApproxEqAbs(cos_inner, pack.clustered_lights[2].dir_inner[3], 1e-5);
    try std.testing.expectApproxEqAbs(cos_outer, pack.clustered_lights[2].spot_params[0], 1e-5);
    try std.testing.expectEqual(@as(f32, 2.0), pack.clustered_lights[2].spot_params[1]); // spot shadow type
    try std.testing.expectEqual(@as(f32, 0.0), pack.clustered_lights[2].spot_params[2]); // shadow slot 0
    try std.testing.expectEqual(@as(f32, 0.004), pack.clustered_lights[2].spot_params[3]); // shadow bias

    // Light 3: Spot light without shadow
    try std.testing.expectEqual([4]f32{ 1, 1, 1, 12 }, pack.clustered_lights[3].pos_range);
    try std.testing.expectEqual(@as(f32, 0.0), pack.clustered_lights[3].spot_params[1]); // no shadow
}
