const std = @import("std");

pub const TonemappingType = enum(u32) {
    none = 0,
    aces = 1,
    reinhard = 2,
};

pub const PostProcessConfig = struct {
    enabled: bool = false,
    exposure: f32 = 1.0,
    tonemapping: TonemappingType = .aces,

    // Bloom (bright pass filter + soft Gaussian halo)
    bloom_enabled: bool = true,
    bloom_threshold: f32 = 0.8,
    bloom_intensity: f32 = 0.5,
    bloom_radius: f32 = 2.0,

    // Vignette (cinematic lens falloff)
    vignette_enabled: bool = true,
    vignette_intensity: f32 = 0.35,
    vignette_radius: f32 = 0.8,

    // Color Grading & Optics
    saturation: f32 = 1.05,
    contrast: f32 = 1.05,
    chromatic_aberration: f32 = 0.0,

    // Anti-Aliasing (FXAA 3.11 Sub-Pixel Edge Smoothing)
    fxaa_enabled: bool = true,

    // Atmospheric Depth & Height Fog
    fog_enabled: bool = true,
    fog_density: f32 = 0.015,
    fog_height_falloff: f32 = 0.08,
    fog_start_distance: f32 = 5.0,
    fog_color: [3]f32 = .{ 0.72, 0.82, 0.92 },
    fog_sun_scattering: f32 = 0.8,

    // Screen-Space Reflections (SSR)
    ssr_enabled: bool = true,
    ssr_intensity: f32 = 0.55,
    ssr_max_distance: f32 = 25.0,
    ssr_thickness: f32 = 0.4,

    // Sharpen (post-tonemap unsharp mask)
    sharpen_enabled: bool = false,
    sharpen_amount: f32 = 0.3,

    // Film Grain (post-tonemap hash noise, luminance-masked)
    grain_enabled: bool = false,
    grain_intensity: f32 = 0.05,

    // White Balance (post-tonemap channel gains, 0 = neutral)
    temperature: f32 = 0.0,
    tint: f32 = 0.0,
};
