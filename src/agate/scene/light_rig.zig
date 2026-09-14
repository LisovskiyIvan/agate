const std = @import("std");

const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color3 = math.Color3;

const lights = @import("../lights.zig");
const HemisphericLight = lights.HemisphericLight;
const DirectionalLight = lights.DirectionalLight;
const DirectionalLightOptions = lights.DirectionalLightOptions;
const PointLight = lights.PointLight;
const PointLightOptions = lights.PointLightOptions;
const SpotLight = lights.SpotLight;
const SpotLightOptions = lights.SpotLightOptions;
const light_selection = @import("light_selection.zig");
const passes = @import("../passes/mod.zig");

/// Owns every light of the scene: the legacy hemispheric sun, the optional
/// directional sun, and the point/spot collections. Also packs the
/// shader-visible subset per frame (top-k selection + uniform arrays), so
/// light data has exactly one home and Scene never touches light pointers.
pub const LightRig = struct {
    // Legacy hemispheric sun: always present, also drives the ambient term.
    hemi: HemisphericLight = .{},
    // Optional scene sun. Null keeps the legacy hemispheric sun exactly.
    directional: ?*DirectionalLight = null,
    point_lights: std.ArrayListUnmanaged(*PointLight) = .empty,
    spot_lights: std.ArrayListUnmanaged(*SpotLight) = .empty,

    // Slot hysteresis (see light_selection.Hysteresis): without it two lights
    // trading the 4th/2nd slot pop every frame. `hysteresis_enabled = false`
    // restores the legacy instant-swap behavior bit-for-bit.
    hysteresis_enabled: bool = true,
    // A challenger must beat an incumbent's raw score by this factor to
    // displace it — the dampener that makes slot swaps rare and decisive.
    incumbency_bonus: f32 = 0.5,
    // Seconds for a slot hand-off (old light fades out, then the new one in).
    fade_time: f32 = 0.25,
    point_hysteresis: light_selection.Hysteresis(PointLight, point_slots, light_selection.scorePoint) = .{},
    spot_hysteresis: light_selection.Hysteresis(SpotLight, spot_slots, light_selection.scoreSpot) = .{},

    pub fn init(name: []const u8, options: lights.HemisphericLightOptions) LightRig {
        return .{ .hemi = HemisphericLight.init(name, options) };
    }

    pub fn setHemispheric(self: *LightRig, name: []const u8, options: lights.HemisphericLightOptions) HemisphericLight {
        self.hemi = HemisphericLight.init(name, options);
        return self.hemi;
    }

    pub fn createPointLight(self: *LightRig, allocator: std.mem.Allocator, name: []const u8, options: PointLightOptions) !*PointLight {
        const pl = try allocator.create(PointLight);
        pl.* = PointLight.init(name, options);
        try self.point_lights.append(allocator, pl);
        return pl;
    }

    pub fn createSpotLight(self: *LightRig, allocator: std.mem.Allocator, name: []const u8, options: SpotLightOptions) !*SpotLight {
        const sl = try allocator.create(SpotLight);
        sl.* = SpotLight.init(name, options);
        try self.spot_lights.append(allocator, sl);
        return sl;
    }

    /// Creates (or replaces) the single scene sun. Replacing destroys the
    /// previous light so at most one directional light is owned at a time.
    pub fn createDirectionalLight(self: *LightRig, allocator: std.mem.Allocator, name: []const u8, options: DirectionalLightOptions) !*DirectionalLight {
        if (self.directional) |old| {
            if (old.owns_name) allocator.free(old.name);
            allocator.destroy(old);
            self.directional = null;
        }
        const dl = try allocator.create(DirectionalLight);
        dl.* = DirectionalLight.init(name, options);
        self.directional = dl;
        return dl;
    }

    // Sun resolution helpers (directional override, hemispheric fallback).
    pub fn sunDirection(self: *const LightRig) Vec3 {
        return lights.resolveSunDirection(self.directional, self.hemi);
    }

    pub fn sunColor(self: *const LightRig) Color3 {
        return lights.resolveSunColor(self.directional, self.hemi);
    }

    pub fn sunIntensity(self: *const LightRig) f32 {
        return lights.resolveSunIntensity(self.directional, self.hemi);
    }

    /// Frees every owned light. Leaves the collections empty so a deinit
    /// path stays idempotent under testing allocators. Slot hysteresis state
    /// is cleared too — it holds raw light pointers.
    pub fn deinit(self: *LightRig, allocator: std.mem.Allocator) void {
        self.point_hysteresis.reset();
        self.spot_hysteresis.reset();

        for (self.point_lights.items) |pl| {
            if (pl.owns_name) allocator.free(pl.name);
            allocator.destroy(pl);
        }
        self.point_lights.deinit(allocator);

        for (self.spot_lights.items) |sl| {
            if (sl.owns_name) allocator.free(sl.name);
            allocator.destroy(sl);
        }
        self.spot_lights.deinit(allocator);

        if (self.directional) |dl| {
            if (dl.owns_name) allocator.free(dl.name);
            allocator.destroy(dl);
            self.directional = null;
        }
    }

    // Shader uniform arrays for point/spot lights. Sizes mirror the shader
    // declarations (4 point + 2 spot slots); selection picks the best subset.
    pub const point_slots = 4;
    pub const spot_slots = 2;

    /// Everything the renderer needs from the light rig for one frame:
    /// packed uniform arrays plus the spot shadow list for the depth pass.
    /// `shadows_enabled` gates spot shadow info generation exactly like the
    /// legacy inline render() code.
    pub const FramePack = struct {
        counts: [4]f32,
        point_pos_range: [4][4]f32,
        point_color_int: [4][4]f32,
        spot_pos_range: [2][4]f32,
        spot_dir_inner: [2][4]f32,
        spot_color_outer: [2][4]f32,
        spot_intensity: [2][4]f32,
        spot_view_proj: [2]Mat4,
        spot_shadow_params: [2][4]f32,
        spot_shadows: [2]passes.SpotShadowRenderInfo,
        num_spot_shadows: usize,
    };

    /// Packs point & spot lights for the camera. Directions are normalized
    /// once here so the fragment shaders can use them raw (no per-pixel
    /// normalize()). The directional light overrides the legacy hemispheric
    /// sun; ambient stays hemispheric ground_color. `dt` advances the slot
    /// fade state (pass 0 to freeze transitions).
    pub fn packFrame(self: *LightRig, eye: Vec3, shadows_enabled: bool, dt: f32) FramePack {
        var pack = FramePack{
            .counts = .{ 0.0, 0.0, 0.0, 0.0 },
            .point_pos_range = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 4,
            .point_color_int = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 4,
            .spot_pos_range = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2,
            .spot_dir_inner = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2,
            .spot_color_outer = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2,
            .spot_intensity = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2,
            .spot_view_proj = [_]Mat4{Mat4.identity} ** 2,
            .spot_shadow_params = [_][4]f32{.{ 0.0, 0.0, 0.0, 0.0 }} ** 2,
            .spot_shadows = undefined,
            .num_spot_shadows = 0,
        };

        // Pick the most relevant lights for the camera before packing; the
        // shader uniform arrays only hold 4 point + 2 spot slots. With
        // hysteresis the packed subset may differ from the raw top-k: a
        // displaced light keeps its slot fading out before the challenger
        // fades in, and intensities are scaled by the fade factor.
        if (self.hysteresis_enabled) {
            var point_packed: [point_slots]light_selection.Hysteresis(PointLight, point_slots, light_selection.scorePoint).Packed = undefined;
            const num_point = self.point_hysteresis.update(
                self.point_lights.items,
                eye,
                dt,
                self.incumbency_bonus,
                self.fade_time,
                &point_packed,
            );
            for (point_packed[0..num_point], 0..) |p, i| {
                const pl = p.light;
                pack.point_pos_range[i] = .{ pl.position.x, pl.position.y, pl.position.z, pl.range };
                pack.point_color_int[i] = .{ pl.color.r, pl.color.g, pl.color.b, pl.intensity * p.factor };
            }
            pack.counts[0] = @floatFromInt(num_point);

            var spot_packed: [spot_slots]light_selection.Hysteresis(SpotLight, spot_slots, light_selection.scoreSpot).Packed = undefined;
            const num_spot = self.spot_hysteresis.update(
                self.spot_lights.items,
                eye,
                dt,
                self.incumbency_bonus,
                self.fade_time,
                &spot_packed,
            );
            for (spot_packed[0..num_spot], 0..) |p, i| {
                const sl = p.light;
                const dir = sl.direction.normalize();
                const cos_inner = @cos(sl.inner_angle_deg * (std.math.pi / 180.0));
                const cos_outer = @cos(sl.outer_angle_deg * (std.math.pi / 180.0));
                const fade_intensity = sl.intensity * p.factor;
                pack.spot_pos_range[i] = .{ sl.position.x, sl.position.y, sl.position.z, sl.range };
                pack.spot_dir_inner[i] = .{ dir.x, dir.y, dir.z, cos_inner };
                pack.spot_color_outer[i] = .{ sl.color.r, sl.color.g, sl.color.b, cos_outer };
                pack.spot_intensity[i] = .{ fade_intensity, 0.0, 0.0, 0.0 };

                // A fading-out shadow caster keeps its shadow: dropping the
                // shadow earlier than the light itself would pop twice.
                if (sl.cast_shadows and shadows_enabled and sl.is_enabled) {
                    const svp = sl.getShadowViewProj();
                    pack.spot_view_proj[i] = svp;
                    pack.spot_shadow_params[i] = .{ 1.0, sl.shadow_bias, sl.shadow_normal_bias, 0.0 };
                    pack.spot_shadows[pack.num_spot_shadows] = .{
                        .spot_index = i,
                        .view_proj = svp,
                    };
                    pack.num_spot_shadows += 1;
                } else {
                    pack.spot_shadow_params[i] = .{ 0.0, 0.0, 0.0, 0.0 };
                }
            }
            pack.counts[1] = @floatFromInt(num_spot);
            return pack;
        }

        // Legacy path: instant top-k, no fade state (bit-identical to the
        // pre-hysteresis packing).
        var point_buf: [point_slots]*PointLight = undefined;
        const num_point = light_selection.selectPoint(self.point_lights.items, eye, &point_buf);
        for (point_buf[0..num_point], 0..) |pl, i| {
            pack.point_pos_range[i] = .{ pl.position.x, pl.position.y, pl.position.z, pl.range };
            pack.point_color_int[i] = .{ pl.color.r, pl.color.g, pl.color.b, pl.intensity };
        }
        pack.counts[0] = @floatFromInt(num_point);

        var spot_buf: [spot_slots]*SpotLight = undefined;
        const num_spot = light_selection.selectSpot(self.spot_lights.items, eye, &spot_buf);

        for (spot_buf[0..num_spot], 0..) |sl, i| {
            const dir = sl.direction.normalize();
            const cos_inner = @cos(sl.inner_angle_deg * (std.math.pi / 180.0));
            const cos_outer = @cos(sl.outer_angle_deg * (std.math.pi / 180.0));
            pack.spot_pos_range[i] = .{ sl.position.x, sl.position.y, sl.position.z, sl.range };
            pack.spot_dir_inner[i] = .{ dir.x, dir.y, dir.z, cos_inner };
            pack.spot_color_outer[i] = .{ sl.color.r, sl.color.g, sl.color.b, cos_outer };
            pack.spot_intensity[i] = .{ sl.intensity, 0.0, 0.0, 0.0 };

            if (sl.cast_shadows and shadows_enabled and sl.is_enabled) {
                const svp = sl.getShadowViewProj();
                pack.spot_view_proj[i] = svp;
                pack.spot_shadow_params[i] = .{ 1.0, sl.shadow_bias, sl.shadow_normal_bias, 0.0 };
                pack.spot_shadows[pack.num_spot_shadows] = .{
                    .spot_index = i,
                    .view_proj = svp,
                };
                pack.num_spot_shadows += 1;
            } else {
                pack.spot_shadow_params[i] = .{ 0.0, 0.0, 0.0, 0.0 };
            }
        }
        pack.counts[1] = @floatFromInt(num_spot);

        return pack;
    }
};

test "createDirectionalLight replaces and owns at most one sun" {
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
