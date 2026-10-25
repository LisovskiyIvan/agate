const std = @import("std");
const options = @import("options.zig");

pub const LENS_FLARE_GHOSTS_MIN: u32 = 1;
pub const LENS_FLARE_GHOSTS_MAX: u32 = 8;
pub const LENS_FLARE_INTENSITY_MAX: f32 = 10.0;
pub const LENS_FLARE_SPACING_MIN: f32 = 0.05;
pub const LENS_FLARE_SPACING_MAX: f32 = 1.0;
pub const LENS_FLARE_HALO_RADIUS_MIN: f32 = 0.05;
pub const LENS_FLARE_HALO_RADIUS_MAX: f32 = 1.0;
pub const LENS_FLARE_HALO_THICKNESS_MIN: f32 = 0.01;
pub const LENS_FLARE_HALO_THICKNESS_MAX: f32 = 0.5;
pub const LENS_FLARE_DISPERSION_MIN: f32 = 0.0;
pub const LENS_FLARE_DISPERSION_MAX: f32 = 0.1;

/// Screen-space cinematic lens flare options.
/// Samples bloom_tex along the optical axis passing through the screen center,
/// simulating internal lens element reflections (ghosts), barrel halo, and
/// chromatic dispersion (refractive prism aberration).
pub const LensFlareOptions = struct {
    enabled: bool = false,
    /// Additive intensity scale of the flare reflections.
    intensity: f32 = 0.4,
    /// Number of ghost reflections along the optical axis (1..8).
    ghosts: u32 = 4,
    /// Spacing between consecutive ghosts along the optical axis.
    ghost_spacing: f32 = 0.35,
    /// Distance of the circular lens barrel halo from the screen center.
    halo_radius: f32 = 0.45,
    /// Thickness of the halo ring in normalized screen coordinates.
    halo_thickness: f32 = 0.1,
    /// Chromatic dispersion offset magnitude between R, G, and B wavelengths.
    chromatic_dispersion: f32 = 0.015,
};

/// Optical axis direction vector from current UV toward the screen center.
pub fn ghostVector(uv: [2]f32, spacing: f32) [2]f32 {
    const to_center = [2]f32{ 0.5 - uv[0], 0.5 - uv[1] };
    return .{ to_center[0] * spacing, to_center[1] * spacing };
}

/// UV sampling position for ghost reflection index `ghost_index` (0-based)
/// from the pixel at `uv`.
pub fn ghostUv(uv: [2]f32, ghost_index: u32, spacing: f32) [2]f32 {
    const flip_uv = [2]f32{ 1.0 - uv[0], 1.0 - uv[1] };
    const step_vec = ghostVector(uv, spacing);
    const fi: f32 = @floatFromInt(ghost_index);
    return .{
        flip_uv[0] + step_vec[0] * fi,
        flip_uv[1] + step_vec[1] * fi,
    };
}

/// Barrel vignette weight for a point `p`. Fades out smoothly towards
/// the screen edge so ghosts do not abruptly clip or artifact.
pub fn barrelWeight(p: [2]f32) f32 {
    const dx = 0.5 - p[0];
    const dy = 0.5 - p[1];
    const dist_center = @sqrt(dx * dx + dy * dy);
    const d = dist_center * 2.0;
    const w = std.math.clamp(1.0 - d, 0.0, 1.0);
    return w * w;
}

/// Halo ring sample UV coordinate from pixel at `uv`.
pub fn haloUv(uv: [2]f32, halo_radius: f32) [2]f32 {
    const to_center = [2]f32{ 0.5 - uv[0], 0.5 - uv[1] };
    const dist = @sqrt(to_center[0] * to_center[0] + to_center[1] * to_center[1]);
    const flip_uv = [2]f32{ 1.0 - uv[0], 1.0 - uv[1] };
    if (dist < 1e-5) return flip_uv;
    const norm_dir = [2]f32{ to_center[0] / dist, to_center[1] / dist };
    return .{
        flip_uv[0] + norm_dir[0] * halo_radius,
        flip_uv[1] + norm_dir[1] * halo_radius,
    };
}

/// Halo ring weight at pixel `uv` given the target radius and thickness.
pub fn haloWeight(uv: [2]f32, halo_radius: f32, halo_thickness: f32) f32 {
    const dx = 0.5 - uv[0];
    const dy = 0.5 - uv[1];
    const dist = @sqrt(dx * dx + dy * dy);
    const ring_dist = @abs(dist - halo_radius);
    const th = @max(halo_thickness, 0.001);
    const w = std.math.clamp(1.0 - ring_dist / th, 0.0, 1.0);
    return w * w * w;
}

/// Radial chromatic offset vector given optical axis direction and dispersion.
pub fn chromaticOffset(norm_dir: [2]f32, dispersion: f32) [2]f32 {
    return .{ norm_dir[0] * dispersion, norm_dir[1] * dispersion };
}

/// Whether lens flares are actively contributing to the composite pass.
pub fn lensFlareActive(post_enabled: bool, cfg: options.PostProcessOptions) bool {
    return post_enabled and cfg.lens_flares.enabled and cfg.bloom.enabled and cfg.lens_flares.intensity > 0.001;
}

/// Pack the composite `lens_flare_params` vec4:
/// (enabled 1/0, intensity, halo_radius, chromatic_dispersion).
/// Zeros when inactive, which keeps the composite bit-identical to the
/// lens-flares-off path.
pub fn lensFlareParams(cfg: options.PostProcessOptions) [4]f32 {
    const c = cfg.clamped();
    if (!c.lens_flares.enabled or !c.bloom.enabled or c.lens_flares.intensity <= 0.001) {
        return .{ 0.0, 0.0, 0.0, 0.0 };
    }
    return .{
        1.0,
        c.lens_flares.intensity,
        c.lens_flares.halo_radius,
        c.lens_flares.chromatic_dispersion,
    };
}

/// Pack the composite `lens_flare_params2` vec4:
/// (ghosts count, ghost_spacing, halo_thickness, spare 0).
/// Zeros when inactive.
pub fn lensFlareParams2(cfg: options.PostProcessOptions) [4]f32 {
    const c = cfg.clamped();
    if (!c.lens_flares.enabled or !c.bloom.enabled or c.lens_flares.intensity <= 0.001) {
        return .{ 0.0, 0.0, 0.0, 0.0 };
    }
    return .{
        @floatFromInt(c.lens_flares.ghosts),
        c.lens_flares.ghost_spacing,
        c.lens_flares.halo_thickness,
        0.0,
    };
}
