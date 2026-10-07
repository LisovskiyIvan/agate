//! Spot light (cone, optional shadow view-projection). Leaf of the
//! `lights.zig` facade; see the facade header for the module map and the
//! anti-cycle rule.
const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color3 = math.Color3;

pub const SpotLightOptions = struct {
    position: Vec3 = Vec3.zero,
    direction: Vec3 = Vec3.new(0, -1, 0),
    color: Color3 = Color3.white,
    intensity: f32 = 1.0,
    range: f32 = 15.0,
    inner_angle_deg: f32 = 15.0,
    outer_angle_deg: f32 = 30.0,
    /// Babylon's `SpotLight.exponent` (default 0) — the sharpening of the
    /// LEGACY cone its STANDARD material uses:
    /// `if (cosAngle >= cos(angle/2)) attenuation *= max(0, pow(cosAngle, exponent))`.
    /// Babylon's standard path ignores `innerAngle` entirely (only `angle` and
    /// `exponent`), and agate's standard shader follows it: `inner_angle_deg`
    /// is a PBR-family parameter here. `exponent = 0` (the default) makes the
    /// cone a hard edge, which is what Babylon renders by default. Rides the
    /// `spot_intensity.y` lane. See shaders/standard.glsl and PROBE.md §10.20.
    exponent: f32 = 0.0,
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
    /// See SpotLightOptions.exponent.
    exponent: f32 = 0.0,
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
            .exponent = options.exponent,
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
