const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
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

pub const DirectionalLight = struct {
    name: []const u8 = "DirectionalLight",
    direction: Vec3 = Vec3.new(0.5, 1.0, 0.5),
    diffuse: Color3 = Color3.white,
    intensity: f32 = 1.0,

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
};

pub const PointLight = struct {
    name: []const u8 = "PointLight",
    position: Vec3 = Vec3.zero,
    color: Color3 = Color3.white,
    intensity: f32 = 1.0,
    range: f32 = 10.0,
    is_enabled: bool = true,

    pub fn init(name: []const u8, options: PointLightOptions) PointLight {
        return .{
            .name = name,
            .position = options.position,
            .color = options.color,
            .intensity = options.intensity,
            .range = options.range,
        };
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
};

pub const SpotLight = struct {
    name: []const u8 = "SpotLight",
    position: Vec3 = Vec3.zero,
    direction: Vec3 = Vec3.new(0, -1, 0),
    color: Color3 = Color3.white,
    intensity: f32 = 1.0,
    range: f32 = 15.0,
    inner_angle_deg: f32 = 15.0,
    outer_angle_deg: f32 = 30.0,
    is_enabled: bool = true,

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
        };
    }
};

// Active sun resolvers: a scene DirectionalLight overrides the legacy
// hemispheric sun when present, otherwise hemispheric values are kept.
pub fn resolveSunDirection(directional: ?*const DirectionalLight, hemi: HemisphericLight) Vec3 {
    if (directional) |d| {
        if (d.direction.lengthSq() > 1e-12) return d.direction.normalize();
    }
    return hemi.direction.normalize();
}

pub fn resolveSunColor(directional: ?*const DirectionalLight, hemi: HemisphericLight) Color3 {
    if (directional) |d| return d.diffuse;
    return hemi.diffuse;
}

pub fn resolveSunIntensity(directional: ?*const DirectionalLight, hemi: HemisphericLight) f32 {
    if (directional) |d| return d.intensity;
    return hemi.intensity;
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
