//! Atmospheric Height Fog & Aerial Perspective (wave Q.5).
//!
//! Provides CPU golden calculations and verified formulas mirroring the GPU
//! postprocess atmospheric fog shader (postprocess.glsl applyAtmosphericFog).
//!
//! Features:
//! 1. Beer-Lambert transmittance with continuous optical depth integration.
//! 2. Exponential height falloff integrating average vertical density along rays.
//! 3. Directional sun inscattering (atmospheric Mie glow).
//! 4. Horizon haze for distant skybox blending.

const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;

/// Computes the average height density along the ray from `cam_y` to `world_y`
/// under an exponential atmosphere with falloff factor lambda (`falloff`).
/// Integral: (1 / dy) * integral(exp(-lambda * y) dy) = (exp(-lambda*y0) - exp(-lambda*y1)) / (lambda * dy).
pub fn heightDensity(cam_y: f32, world_y: f32, falloff: f32) f32 {
    const delta_y = world_y - cam_y;
    const density = if (@abs(delta_y) > 0.001)
        (@exp(-cam_y * falloff) - @exp(-world_y * falloff)) / (delta_y * falloff)
    else
        @exp(-cam_y * falloff);
    return std.math.clamp(density, 0.0, 5.0);
}

/// Optical depth tau along the ray segment beyond `fog_start` under base density `fog_density`.
pub fn opticalDepth(dist: f32, fog_start: f32, fog_density: f32, height_factor: f32) f32 {
    const eff_dist = @max(0.0, dist - fog_start);
    return eff_dist * fog_density * height_factor;
}

/// Physical Beer-Lambert extinction: 1 - exp(-tau).
/// Monotonically in [0, 1].
pub fn fogExtinction(tau: f32) f32 {
    if (tau <= 0.0) return 0.0;
    return std.math.clamp(1.0 - @exp(-tau), 0.0, 1.0);
}

/// Directional sun inscattering factor (forward Mie scattering peak).
pub fn sunInscatter(ray_dir: Vec3, sun_dir: Vec3, scattering: f32) f32 {
    const sun_dot = @max(0.0, ray_dir.dot(sun_dir));
    return std.math.pow(f32, sun_dot, 8.0) * scattering;
}

/// Horizon haze factor for blending the sky at the horizon.
pub fn skyHorizonHaze(ray_dir_y: f32, fog_density: f32) f32 {
    const horizon_haze = std.math.clamp(1.0 - @abs(ray_dir_y) * 4.0, 0.0, 1.0);
    const density_cap = std.math.clamp(fog_density * 15.0, 0.0, 0.7);
    return std.math.clamp(horizon_haze * density_cap, 0.0, 1.0);
}
