const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color3 = math.Color3;

pub const HemisphericLightOptions = struct {
    direction: Vec3 = Vec3.up,
    diffuse: Color3 = Color3.white,
    ground_color: Color3 = Color3.new(0.2, 0.2, 0.2),
    intensity: f32 = 1.0,
};

pub const HemisphericLight = struct {
    name: []const u8 = "HemisphericLight",
    direction: Vec3 = Vec3.up,
    diffuse: Color3 = Color3.white,
    ground_color: Color3 = Color3.new(0.2, 0.2, 0.2),
    intensity: f32 = 1.0,

    pub fn init(name: []const u8, options: HemisphericLightOptions) HemisphericLight {
        return .{
            .name = name,
            .direction = options.direction.normalize(),
            .diffuse = options.diffuse,
            .ground_color = options.ground_color,
            .intensity = options.intensity,
        };
    }
};

pub const DirectionalLightOptions = struct {
    direction: Vec3 = Vec3.new(0.5, 1.0, 0.5),
    diffuse: Color3 = Color3.white,
    intensity: f32 = 1.0,
};

/// Maximum simultaneous directional lights (Babylon.js parity): index 0 is
/// the shadow-casting sun (CSM unchanged), indices 1..3 are shadowless
/// fills. See LightRig.addDirectionalLight for the creation cap.
pub const max_directional_lights: usize = 4;
/// Fills beyond the primary sun (slots 1..3).
pub const max_fill_directionals: usize = max_directional_lights - 1;

pub const DirectionalLight = struct {
    name: []const u8 = "DirectionalLight",
    /// True when `name` was heap-allocated by the glTF loader; the scene frees
    /// it on deinit/replacement. Programmatic lights keep string literals.
    owns_name: bool = false,
    direction: Vec3 = Vec3.new(0.5, 1.0, 0.5),
    diffuse: Color3 = Color3.white,
    intensity: f32 = 1.0,
    /// Disabled lights pack as zeroed slots (no contribution) and the sun
    /// resolvers below treat them as absent (hemispheric fallback).
    is_enabled: bool = true,

    pub fn init(name: []const u8, options: DirectionalLightOptions) DirectionalLight {
        return .{
            .name = name,
            .direction = options.direction.normalize(),
            .diffuse = options.diffuse,
            .intensity = options.intensity,
        };
    }
};

pub const PointLightOptions = struct {
    position: Vec3 = Vec3.zero,
    color: Color3 = Color3.white,
    intensity: f32 = 1.0,
    range: f32 = 10.0,
    cast_shadows: bool = false,
    shadow_bias: f32 = 0.002,
    shadow_normal_bias: f32 = 0.005,
    shadow_near: f32 = 0.1,
};

pub const PointLight = struct {
    name: []const u8 = "PointLight",
    /// See DirectionalLight.owns_name.
    owns_name: bool = false,
    position: Vec3 = Vec3.zero,
    color: Color3 = Color3.white,
    intensity: f32 = 1.0,
    range: f32 = 10.0,
    is_enabled: bool = true,
    cast_shadows: bool = false,
    shadow_bias: f32 = 0.002,
    shadow_normal_bias: f32 = 0.005,
    shadow_near: f32 = 0.1,

    pub fn init(name: []const u8, options: PointLightOptions) PointLight {
        return .{
            .name = name,
            .position = options.position,
            .color = options.color,
            .intensity = options.intensity,
            .range = options.range,
            .cast_shadows = options.cast_shadows,
            .shadow_bias = options.shadow_bias,
            .shadow_normal_bias = options.shadow_normal_bias,
            .shadow_near = options.shadow_near,
        };
    }

    /// Cube face order for point shadow tiles (matches the shader
    /// pointFaceIndex and ShadowPass.pointFaceForDir): +X, -X, +Y, -Y, +Z, -Z.
    pub const shadow_face_count: usize = 6;

    /// Computes the light view-projection matrix for one cube face of the
    /// point shadow atlas (90-degree perspective, aspect 1). The tile layout
    /// lives in passes/shadow_pass.zig (pointTileOrigin).
    pub fn getShadowFaceViewProj(self: PointLight, face: usize) Mat4 {
        const eye = self.position;
        const dirs = [_]Vec3{
            Vec3.new(1, 0, 0),
            Vec3.new(-1, 0, 0),
            Vec3.new(0, 1, 0),
            Vec3.new(0, -1, 0),
            Vec3.new(0, 0, 1),
            Vec3.new(0, 0, -1),
        };
        const ups = [_]Vec3{
            Vec3.up,
            Vec3.up,
            Vec3.new(0, 0, -1),
            Vec3.new(0, 0, 1),
            Vec3.up,
            Vec3.up,
        };
        const f = face % shadow_face_count;
        const view = Mat4.lookAt(eye, eye.add(dirs[f]), ups[f]);
        const near = @max(self.shadow_near, 0.05);
        const far = @max(self.range, near + 0.1);
        const proj = Mat4.perspective(90.0, 1.0, near, far);
        return Mat4.mul(proj, view);
    }
};

/// Maximum EXTRA point lights in the clustered forward pool (wave 30, v1).
/// These ride OUTSIDE the legacy 4-slot top-k lanes (LightRig.point_slots):
/// every owned clustered light packs verbatim into LightRig.FramePack and
/// the 2D screen-tile build culls them per tile. See
/// LightRig.addClusteredPointLight for the creation cap and
/// scene/clustered_lights.zig for the tiling design. No shadows in v1
/// (unshadowed by design, documented there).
pub const max_clustered_lights: usize = 64;

pub const ClusteredPointLightOptions = struct {
    color: Color3 = Color3.white,
    intensity: f32 = 1.0,
    /// Influence radius in world units (<= 0 packs as absent: the tile
    /// build skips the light and the shader range-gates it like a legacy
    /// lane with zero range).
    radius: f32 = 10.0,
    enabled: bool = true,
};

/// One extra forward point light for the clustered pool. Value type (no
/// name, no heap, no shadows in v1): LightRig owns a fixed array of these
/// plus a count, Scene exposes index-based add/remove/get/count, and the
/// GPU tile build reads the staged FramePack copy (1-frame lag).
pub const ClusteredPointLight = struct {
    position: Vec3 = Vec3.zero,
    color: Color3 = Color3.white,
    intensity: f32 = 1.0,
    radius: f32 = 10.0,
    /// Disabled lights pack as zeroed lanes (no contribution), mirroring
    /// the directional-fill contract.
    is_enabled: bool = true,

    pub fn init(position: Vec3, options: ClusteredPointLightOptions) ClusteredPointLight {
        return .{
            .position = position,
            .color = options.color,
            .intensity = options.intensity,
            .radius = options.radius,
            .is_enabled = options.enabled,
        };
    }
};

pub const AreaLightOptions = struct {
    center: Vec3 = Vec3.zero,
    /// Local +X half-extent vector: direction AND half-width encoded in one
    /// vector (corner = center +/- right +/- up). Must be non-zero and
    /// non-parallel to `up`; degenerate inputs emit nothing (see normal()).
    right: Vec3 = Vec3.new(0.5, 0.0, 0.0),
    /// Local +Y half-extent vector (direction and half-height).
    up: Vec3 = Vec3.new(0.0, 0.5, 0.0),
    color: Color3 = Color3.white,
    intensity: f32 = 1.0,
    is_enabled: bool = true,
};

/// Maximum simultaneous rect area lights (wave 26, v1). See
/// LightRig.addAreaLight for the creation cap.
pub const max_area_lights: usize = 2;

pub const AreaLight = struct {
    name: []const u8 = "AreaLight",
    /// True when `name` was heap-allocated; the scene frees it on
    /// deinit/removal. Programmatic lights keep string literals.
    owns_name: bool = false,
    center: Vec3 = Vec3.zero,
    right: Vec3 = Vec3.new(0.5, 0.0, 0.0),
    up: Vec3 = Vec3.new(0.0, 0.5, 0.0),
    color: Color3 = Color3.white,
    intensity: f32 = 1.0,
    /// Disabled lights pack as zeroed lanes (no contribution).
    is_enabled: bool = true,

    pub fn init(name: []const u8, options: AreaLightOptions) AreaLight {
        return .{
            .name = name,
            .center = options.center,
            .right = options.right,
            .up = options.up,
            .color = options.color,
            .intensity = options.intensity,
            .is_enabled = options.is_enabled,
        };
    }

    /// Emitting-face normal: normalize(cross(right, up)). Zero when the
    /// rect is degenerate (zero-area or parallel axes) — the shader then
    /// contributes nothing (area gate), and CPU code must treat zero as
    /// "no emission" rather than normalizing it into NaN.
    pub fn normal(self: AreaLight) Vec3 {
        const n = self.right.cross(self.up);
        if (n.lengthSq() <= 1e-12) return Vec3.zero;
        return n.normalize();
    }

    /// Emitting area (4 * |right x up|); zero for degenerate rects.
    pub fn area(self: AreaLight) f32 {
        return 4.0 * self.right.cross(self.up).length();
    }
};

pub const SpotLightOptions = struct {
    position: Vec3 = Vec3.zero,
    direction: Vec3 = Vec3.new(0, -1, 0),
    color: Color3 = Color3.white,
    intensity: f32 = 1.0,
    range: f32 = 15.0,
    inner_angle_deg: f32 = 15.0,
    outer_angle_deg: f32 = 30.0,
    cast_shadows: bool = false,
    shadow_bias: f32 = 0.002,
    shadow_normal_bias: f32 = 0.005,
    shadow_near: f32 = 0.1,
};

pub const SpotLight = struct {
    name: []const u8 = "SpotLight",
    /// See DirectionalLight.owns_name.
    owns_name: bool = false,
    position: Vec3 = Vec3.zero,
    direction: Vec3 = Vec3.new(0, -1, 0),
    color: Color3 = Color3.white,
    intensity: f32 = 1.0,
    range: f32 = 15.0,
    inner_angle_deg: f32 = 15.0,
    outer_angle_deg: f32 = 30.0,
    is_enabled: bool = true,
    cast_shadows: bool = false,
    shadow_bias: f32 = 0.002,
    shadow_normal_bias: f32 = 0.005,
    shadow_near: f32 = 0.1,

    pub fn init(name: []const u8, options: SpotLightOptions) SpotLight {
        return .{
            .name = name,
            .position = options.position,
            .direction = options.direction.normalize(),
            .color = options.color,
            .intensity = options.intensity,
            .range = options.range,
            .inner_angle_deg = options.inner_angle_deg,
            .outer_angle_deg = options.outer_angle_deg,
            .cast_shadows = options.cast_shadows,
            .shadow_bias = options.shadow_bias,
            .shadow_normal_bias = options.shadow_normal_bias,
            .shadow_near = options.shadow_near,
        };
    }

    /// Computes the light view-projection matrix for shadow map rendering.
    pub fn getShadowViewProj(self: SpotLight) Mat4 {
        const eye = self.position;
        const dir = if (self.direction.lengthSq() > 1e-6) self.direction.normalize() else Vec3.new(0, -1, 0);
        const target = eye.add(dir);
        const up = if (@abs(dir.y) > 0.99) Vec3.new(0, 0, 1) else Vec3.up;
        const view = Mat4.lookAt(eye, target, up);
        const fov_deg = std.math.clamp(self.outer_angle_deg * 2.0, 1.0, 175.0);
        const near = @max(self.shadow_near, 0.05);
        const far = @max(self.range, near + 0.1);
        const proj = Mat4.perspective(fov_deg, 1.0, near, far);
        return Mat4.mul(proj, view);
    }
};

// Active sun resolvers: an enabled scene DirectionalLight overrides the
// legacy hemispheric sun when present, otherwise hemispheric values are
// kept. A disabled directional counts as absent (same fallback).
pub fn resolveSunDirection(directional: ?*const DirectionalLight, hemi: HemisphericLight) Vec3 {
    if (directional) |d| {
        if (d.is_enabled and d.direction.lengthSq() > 1e-12) return d.direction.normalize();
    }
    return hemi.direction.normalize();
}

pub fn resolveSunColor(directional: ?*const DirectionalLight, hemi: HemisphericLight) Color3 {
    if (directional) |d| if (d.is_enabled) return d.diffuse;
    return hemi.diffuse;
}

pub fn resolveSunIntensity(directional: ?*const DirectionalLight, hemi: HemisphericLight) f32 {
    if (directional) |d| if (d.is_enabled) return d.intensity;
    return hemi.intensity;
}

/// Computes a normalized light direction vector from azimuth and elevation angles (radians).
/// Elevation is measured up from the horizontal XZ plane: 0 = horizon, pi/2 = directly overhead.
/// Azimuth rotates around +Y: 0 = along +Z, pi/2 = along +X.
pub fn sunDirectionFromAngles(azimuth_rad: f32, elevation_rad: f32) Vec3 {
    const cos_elev = @cos(elevation_rad);
    const sin_elev = @sin(elevation_rad);
    const dir = Vec3.new(
        cos_elev * @sin(azimuth_rad),
        sin_elev,
        cos_elev * @cos(azimuth_rad),
    );
    const len_sq = dir.lengthSq();
    if (len_sq > 1e-12) return dir.scale(1.0 / @sqrt(len_sq));
    return Vec3.up;
}

/// Converts a correlated color temperature in Kelvin (1000K to 12000K) to normalized linear RGB.
/// Uses the standard Tanner Helland Planckian approximation.
pub fn colorTemperatureToRgb(kelvin: f32) Color3 {
    const temp = std.math.clamp(kelvin, 1000.0, 12000.0) / 100.0;

    const r: f32 = if (temp <= 66.0)
        1.0
    else blk: {
        const val = 329.698727446 * std.math.pow(f32, temp - 60.0, -0.1332047592) / 255.0;
        break :blk std.math.clamp(val, 0.0, 1.0);
    };

    const g: f32 = if (temp <= 66.0) blk: {
        const val = (99.4708025861 * @log(temp) - 161.1195681661) / 255.0;
        break :blk std.math.clamp(val, 0.0, 1.0);
    } else blk: {
        const val = 288.1221695283 * std.math.pow(f32, temp - 60.0, -0.0755148492) / 255.0;
        break :blk std.math.clamp(val, 0.0, 1.0);
    };

    const b: f32 = if (temp >= 66.0)
        1.0
    else if (temp <= 19.0)
        0.0
    else blk: {
        const val = (138.5177312231 * @log(temp - 10.0) - 305.0447927307) / 255.0;
        break :blk std.math.clamp(val, 0.0, 1.0);
    };

    return Color3.new(r, g, b);
}

test "resolveSunDirection falls back to normalized hemi" {
    const hemi = HemisphericLight.init("hemi", .{ .direction = Vec3.new(2.0, 0.0, 0.0) });
    const dir = resolveSunDirection(null, hemi);
    try std.testing.expectApproxEqAbs(dir.x, 1.0, 1e-6);
    try std.testing.expectApproxEqAbs(dir.y, 0.0, 1e-6);
    try std.testing.expectApproxEqAbs(dir.z, 0.0, 1e-6);
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

test "directional light limits and array capacity" {
    try std.testing.expectEqual(@as(usize, 4), max_directional_lights);
    try std.testing.expectEqual(@as(usize, 3), max_fill_directionals);

    var lights_arr: [max_directional_lights]DirectionalLight = undefined;
    for (&lights_arr, 0..) |*l, i| {
        l.* = DirectionalLight.init("dir", .{
            .direction = Vec3.new(0, -1, @floatFromInt(i)),
            .intensity = 1.0 + @as(f32, @floatFromInt(i)),
        });
    }
    for (lights_arr, 0..) |l, i| {
        try std.testing.expect(l.is_enabled);
        try std.testing.expectApproxEqAbs(1.0 + @as(f32, @floatFromInt(i)), l.intensity, 1e-5);
    }
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

test "area light capacity and array bounds" {
    try std.testing.expectEqual(@as(usize, 2), max_area_lights);
    var area_arr: [max_area_lights]AreaLight = undefined;
    for (&area_arr, 0..) |*al, i| {
        al.* = AreaLight.init("area", .{
            .center = Vec3.new(@floatFromInt(i), 0, 0),
            .intensity = 2.0,
        });
    }
    for (area_arr, 0..) |al, i| {
        try std.testing.expectEqual(Vec3.new(@floatFromInt(i), 0, 0), al.center);
        try std.testing.expectEqual(@as(f32, 2.0), al.intensity);
    }
}

test "clustered pool cap is 64, defaults match the legacy point lane" {
    try std.testing.expectEqual(@as(usize, 64), max_clustered_lights);
    const l = ClusteredPointLight.init(Vec3.new(1, 2, 3), .{});
    try std.testing.expectEqual(Vec3.new(1, 2, 3), l.position);
    try std.testing.expectEqual(Color3.white, l.color);
    try std.testing.expectEqual(@as(f32, 1.0), l.intensity);
    try std.testing.expectEqual(@as(f32, 10.0), l.radius);
    try std.testing.expect(l.is_enabled);
    // Options map verbatim (enabled -> is_enabled, engine convention).
    const off = ClusteredPointLight.init(Vec3.zero, .{ .enabled = false, .intensity = 2.5, .radius = 3.0 });
    try std.testing.expect(!off.is_enabled);
    try std.testing.expectEqual(@as(f32, 2.5), off.intensity);
    try std.testing.expectEqual(@as(f32, 3.0), off.radius);
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
