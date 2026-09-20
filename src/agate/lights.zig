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

test "multi-directional cap is four (one sun plus three fills)" {
    try std.testing.expectEqual(@as(usize, 4), max_directional_lights);
    try std.testing.expectEqual(@as(usize, 3), max_fill_directionals);
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
