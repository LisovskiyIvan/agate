//! Sun resolvers and color-temperature helpers. Leaf of the `lights.zig`
//! facade; see the facade header for the module map and the anti-cycle rule.
//! Imports the `directional` + `hemispheric` siblings directly (leaf-to-leaf,
//! same discipline as `particles/cpu.zig` reaching `subemitters`).
const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Color3 = math.Color3;

const directional = @import("directional.zig");
const DirectionalLight = directional.DirectionalLight;
const hemispheric = @import("hemispheric.zig");
const HemisphericLight = hemispheric.HemisphericLight;

// Active sun resolvers: an enabled scene DirectionalLight overrides the
// legacy hemispheric sun when present, otherwise hemispheric values are
// kept. A disabled directional counts as absent (same fallback).
pub fn resolveSunDirection(directional_light: ?*const DirectionalLight, hemi: HemisphericLight) Vec3 {
    if (directional_light) |d| {
        if (d.is_enabled and d.direction.lengthSq() > 1e-12) return d.direction.normalize();
    }
    return hemi.direction.normalize();
}

pub fn resolveSunColor(directional_light: ?*const DirectionalLight, hemi: HemisphericLight) Color3 {
    if (directional_light) |d| if (d.is_enabled) return d.diffuse;
    return hemi.diffuse;
}

pub fn resolveSunIntensity(directional_light: ?*const DirectionalLight, hemi: HemisphericLight) f32 {
    if (directional_light) |d| if (d.is_enabled) return d.intensity;
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
