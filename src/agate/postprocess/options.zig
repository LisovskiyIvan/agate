const std = @import("std");
const texture_mod = @import("../texture.zig");
const types = @import("types.zig");
const bloom = @import("bloom.zig");
const glow = @import("glow.zig");
const color_curves = @import("color_curves.zig");
const lut = @import("lut.zig");
const shafts = @import("shafts.zig");
const auto_exposure = @import("auto_exposure.zig");

pub const TonemappingType = types.TonemappingType;
pub const LutFormat = types.LutFormat;
pub const ShaftResolution = types.ShaftResolution;
pub const AutoExposureOptions = auto_exposure.AutoExposureOptions;

/// Post-processing chain knobs (exposure, tonemapping, SSAO/bloom/DOF
/// toggles and their parameters). A flat config read/written by tooling;
/// applied per frame by `Scene.postfx` (PostFXStack.renderChain).
pub const PostProcessOptions = struct {
    /// Artistic effects master switch. HDR output, exposure and tonemapping
    /// remain active even when effects are disabled.
    enabled: bool = false,
    exposure: f32 = 1.0,
    tonemapping: TonemappingType = .aces,

    // Auto-Exposure (adaptive luminance / histogram metering)
    auto_exposure_enabled: bool = false,
    auto_exposure_min: f32 = 0.01,
    auto_exposure_max: f32 = 16.0,
    auto_exposure_key_value: f32 = 0.18,
    auto_exposure_speed_up: f32 = 3.0,
    auto_exposure_speed_down: f32 = 1.0,
    auto_exposure_low_percentile: f32 = 0.10,
    auto_exposure_high_percentile: f32 = 0.90,
    auto_exposure_camera_cut: bool = false,

    // HDR bloom: Karis downsample + tent upsample pyramid.
    bloom_enabled: bool = true,
    bloom_threshold: f32 = 0.8,
    bloom_intensity: f32 = 0.5,
    /// Upsample tent radius in source-mip texels.
    bloom_radius: f32 = 2.0,
    bloom_pyramid_mips: u32 = 5,

    // Glow layer v1 (global halo, distinct from bloom): luminance threshold
    // extract + separable blur (GlowPass) + additive composite AFTER bloom
    // in postprocess.glsl. Default OFF: no glow passes run, the composite
    // branch is skipped, and rendering stays bit-identical to pre-glow.
    // Per-mesh glow weights are explicitly out of scope for v1 (they need
    // pipeline changes); this is the global pass only.
    glow_enabled: bool = false,
    // Luminance cut for the extract stage; >= 0 (unitless HDR-ish scale).
    glow_threshold: f32 = glow.GLOW_THRESHOLD_DEFAULT,
    // Additive composite scale; >= 0.
    glow_intensity: f32 = glow.GLOW_INTENSITY_DEFAULT,
    // Blur sigma in glow-target texels; >= 0, wider than bloom's radius.
    glow_radius: f32 = glow.GLOW_RADIUS_DEFAULT,
    // Composite color multiplier, per channel in [0, 1].
    glow_tint: [3]f32 = .{ 1.0, 1.0, 1.0 },

    // Vignette (cinematic lens falloff)
    vignette_enabled: bool = true,
    vignette_intensity: f32 = 0.35,
    vignette_radius: f32 = 0.8,

    // Color Grading & Optics
    saturation: f32 = 1.05,
    contrast: f32 = 1.05,
    chromatic_aberration: f32 = 0.0,

    // Depth of Field (gather blur driven by linearized depth)
    dof_enabled: bool = false,
    dof_focus_distance: f32 = 10.0,
    dof_focus_range: f32 = 5.0,
    dof_max_blur: f32 = 8.0,

    // Parametric color curves: per-channel additive lifts in [-1, 1]
    // weighted by shadows/midtones/highlights luminance zones.
    // Primary grading path (no texture dependency). A LUT texture
    // remains an opt-in parent-side extension, see report.
    grade_shadows: [3]f32 = .{ 0.0, 0.0, 0.0 },
    grade_midtones: [3]f32 = .{ 0.0, 0.0, 0.0 },
    grade_highlights: [3]f32 = .{ 0.0, 0.0, 0.0 },

    // Texture LUT color grading (Babylon.js parity), sampled after the
    // parametric curves. With lut_texture == null (the default) the shader
    // skips the LUT branch entirely and the composite path stays
    // bit-identical to pre-LUT. The handle is a by-value Texture (plain
    // GPU handles, no CPU refs), so the whole struct — including the LUT —
    // copies into SceneFrameSnapshot.post_process with a plain assignment
    // (scene.zig packFrameSnapshot) and needs no snapshot-side support.
    lut_enabled: bool = false,
    lut_strength: f32 = 1.0,
    lut_size: u8 = 16,
    lut_format: LutFormat = .strip_2d,
    lut_texture: ?texture_mod.Texture = null,

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
    ssr_steps: u32 = 16,

    // Hierarchical GPU Depth Pyramid (Hi-Z downsample mips for SSR, contact shadows, occlusion)
    depth_pyramid_enabled: bool = false,

    // Screen-Space Contact Shadows & Local Ambient Occlusion (short-range sun raymarch)
    contact_shadows_enabled: bool = false,
    contact_shadows_intensity: f32 = 0.5,
    contact_shadows_distance: f32 = 0.3,
    contact_shadows_thickness: f32 = 0.05,
    contact_shadows_steps: u32 = 12,

    // Sharpen (post-tonemap unsharp mask)
    sharpen_enabled: bool = false,
    sharpen_amount: f32 = 0.3,

    // Film Grain (post-tonemap hash noise, luminance-masked)
    grain_enabled: bool = false,
    grain_intensity: f32 = 0.05,

    // White Balance (post-tonemap channel gains, 0 = neutral)
    temperature: f32 = 0.0,
    tint: f32 = 0.0,

    // Camera Motion Blur (screen velocity gather blur)
    motion_blur_enabled: bool = false,
    motion_blur_intensity: f32 = 0.5,
    motion_blur_max_blur_px: f32 = 32.0,
    motion_blur_samples: u32 = 8,

    // Temporal Anti-Aliasing (sub-pixel Halton jitter + history resolve in
    // postprocess.glsl; history ping-pong in PostProcessPass; jittered VP
    // from Scene.render). Default OFF: the disabled shader path early-outs
    // before any history/depth sampling (bit-identical composite).
    taa_enabled: bool = false,
    // History weight for the temporal blend (1 = full history). 0.9 is the
    // standard compromise between stability and ghosting/disocclusion lag.
    taa_blend: f32 = 0.9,
    // Sub-pixel jitter amplitude multiplier (1 = +/-0.5 px Halton offsets).
    taa_jitter_scale: f32 = 1.0,
    // Post-blend unsharp amount (0 = off), re-clamped to the neighborhood
    // box so low amounts cannot ring.
    taa_sharpness: f32 = 0.0,
    // Neighborhood-clamp strength for ghosting control (1 = full clamp of
    // the reprojected history into the 3x3 current-frame box, 0 = raw).
    taa_clamp_strength: f32 = 1.0,
    // One-frame history reset request (camera teleport/cut/cinematic).
    // Snapshot-carried, so the update thread sets it safely for one frame.
    taa_camera_cut: bool = false,

    // Volumetric light shafts v1 (sun CSM-backed raymarch, low-res).
    // Screen-space raymarch from the camera through each low-res pixel to
    // the scene depth surface, sampling the SUN's CSM atlas at every step
    // (single raw-depth tap per step, no PCF in v1) and accumulating
    // single-scatter with a Henyey-Greenstein phase term and Beer-Lambert
    // transmittance; bilateral (depth-aware) H/V blur; additive composite
    // after the highlight block. Default OFF: zero passes run, the
    // composite branch is skipped, and rendering stays bit-identical.
    // Requires live CSM data (gated on shadows_enabled at the call site,
    // see shaftActive): without a rendered CSM atlas there is nothing
    // real to march against, and v1 refuses to fake it.
    shaft_enabled: bool = false,
    // Additive composite scale; >= 0.
    shaft_intensity: f32 = 1.0,
    // Raymarch steps per pixel; clamped to [4, 32] (the shader loop bound).
    shaft_steps: u32 = 12,
    // Extinction/scattering coefficient per meter; >= 0.
    shaft_density: f32 = 0.05,
    // Henyey-Greenstein anisotropy in [-0.9, 0.9]; > 0 forward scattering
    // (bright shafts looking toward the sun), 0 isotropic.
    shaft_anisotropy: f32 = 0.4,
    // Raymarch span cap in meters; >= 0 (sky pixels march the full span).
    shaft_max_distance: f32 = 60.0,
    // Raymarch/blur target resolution (half or quarter of the base size).
    shaft_resolution: ShaftResolution = .quarter,
    // Bilateral blur spatial sigma in shaft-target texels; >= 0.
    shaft_blur_sigma: f32 = 2.0,
    // Bilateral blur depth gate in raw-depth units; >= 0 (0 = no depth
    // weighting, plain Gaussian).
    shaft_edge_sigma: f32 = 0.02,

    // Return a copy with out-of-range values pulled into valid ranges.
    // Never fails; safe to apply on load or before uploading uniforms.
    pub fn clamped(self: PostProcessOptions) PostProcessOptions {
        var out = self;
        out.exposure = if (std.math.isFinite(self.exposure)) std.math.clamp(self.exposure, 0.0, 65504.0) else 1.0;
        out.auto_exposure_min = if (std.math.isFinite(self.auto_exposure_min)) @max(self.auto_exposure_min, 0.0) else 0.01;
        out.auto_exposure_max = if (std.math.isFinite(self.auto_exposure_max)) @max(self.auto_exposure_max, out.auto_exposure_min) else 16.0;
        out.auto_exposure_key_value = if (std.math.isFinite(self.auto_exposure_key_value)) @max(self.auto_exposure_key_value, 0.001) else 0.18;
        out.auto_exposure_speed_up = if (std.math.isFinite(self.auto_exposure_speed_up)) @max(self.auto_exposure_speed_up, 0.0) else 3.0;
        out.auto_exposure_speed_down = if (std.math.isFinite(self.auto_exposure_speed_down)) @max(self.auto_exposure_speed_down, 0.0) else 1.0;
        out.auto_exposure_low_percentile = if (std.math.isFinite(self.auto_exposure_low_percentile)) std.math.clamp(self.auto_exposure_low_percentile, 0.0, 1.0) else 0.10;
        out.auto_exposure_high_percentile = if (std.math.isFinite(self.auto_exposure_high_percentile)) std.math.clamp(self.auto_exposure_high_percentile, out.auto_exposure_low_percentile, 1.0) else 0.90;
        out.bloom_threshold = @max(self.bloom_threshold, 0.0);
        out.bloom_intensity = @max(self.bloom_intensity, 0.0);
        out.bloom_radius = if (std.math.isFinite(self.bloom_radius)) std.math.clamp(self.bloom_radius, 0.0, 16.0) else 2.0;
        out.bloom_pyramid_mips = bloom.clampBloomMips(self.bloom_pyramid_mips);
        out.glow_threshold = @max(self.glow_threshold, 0.0);
        out.glow_intensity = @max(self.glow_intensity, 0.0);
        out.glow_radius = @max(self.glow_radius, 0.0);
        out.glow_tint = glow.clampTint(self.glow_tint);
        out.dof_focus_distance = @max(self.dof_focus_distance, 0.0);
        out.dof_focus_range = @max(self.dof_focus_range, 0.0);
        out.dof_max_blur = @max(self.dof_max_blur, 0.0);
        out.ssr_steps = std.math.clamp(self.ssr_steps, 4, 64);
        out.contact_shadows_intensity = if (std.math.isFinite(self.contact_shadows_intensity)) std.math.clamp(self.contact_shadows_intensity, 0.0, 1.0) else 0.5;
        out.contact_shadows_distance = if (std.math.isFinite(self.contact_shadows_distance)) @max(self.contact_shadows_distance, 0.01) else 0.3;
        out.contact_shadows_thickness = if (std.math.isFinite(self.contact_shadows_thickness)) @max(self.contact_shadows_thickness, 0.001) else 0.05;
        out.contact_shadows_steps = std.math.clamp(self.contact_shadows_steps, 4, 32);
        out.motion_blur_samples = std.math.clamp(self.motion_blur_samples, 2, 32);
        out.motion_blur_intensity = std.math.clamp(self.motion_blur_intensity, 0.0, 3.0);
        out.motion_blur_max_blur_px = std.math.clamp(self.motion_blur_max_blur_px, 1.0, 128.0);
        out.taa_blend = std.math.clamp(self.taa_blend, 0.0, 1.0);
        out.taa_jitter_scale = std.math.clamp(self.taa_jitter_scale, 0.0, 4.0);
        out.taa_sharpness = std.math.clamp(self.taa_sharpness, 0.0, 1.0);
        out.taa_clamp_strength = std.math.clamp(self.taa_clamp_strength, 0.0, 1.0);
        out.shaft_intensity = @max(self.shaft_intensity, 0.0);
        out.shaft_steps = std.math.clamp(self.shaft_steps, shafts.SHAFT_STEPS_MIN, shafts.SHAFT_STEPS_MAX);
        out.shaft_density = @max(self.shaft_density, 0.0);
        out.shaft_anisotropy = std.math.clamp(self.shaft_anisotropy, -shafts.SHAFT_ANISOTROPY_MAX, shafts.SHAFT_ANISOTROPY_MAX);
        out.shaft_max_distance = @max(self.shaft_max_distance, 0.0);
        out.shaft_blur_sigma = @max(self.shaft_blur_sigma, 0.0);
        out.shaft_edge_sigma = @max(self.shaft_edge_sigma, 0.0);
        out.grade_shadows = color_curves.clampGrade(self.grade_shadows);
        out.grade_midtones = color_curves.clampGrade(self.grade_midtones);
        out.grade_highlights = color_curves.clampGrade(self.grade_highlights);
        out.lut_strength = std.math.clamp(self.lut_strength, 0.0, 1.0);
        // A binding without a live view, a supported size, or matching
        // strip dims can never be sampled; drop it so the pass keeps its
        // no-LUT path instead of binding a dead handle.
        if (out.lut_texture) |tex| {
            if (!lut.lutTextureValid(tex, out.lut_size)) out.lut_texture = null;
        }
        return out;
    }

    /// Render-owned copy: disable effects, never the linear HDR/output path.
    pub fn forFrame(self: PostProcessOptions) PostProcessOptions {
        var out = self.clamped();
        if (out.enabled) return out;
        out.bloom_enabled = false;
        out.glow_enabled = false;
        out.vignette_enabled = false;
        out.chromatic_aberration = 0;
        out.saturation = 1;
        out.contrast = 1;
        out.dof_enabled = false;
        out.grade_shadows = .{ 0, 0, 0 };
        out.grade_midtones = .{ 0, 0, 0 };
        out.grade_highlights = .{ 0, 0, 0 };
        out.lut_enabled = false;
        out.fxaa_enabled = false;
        out.fog_enabled = false;
        out.ssr_enabled = false;
        out.contact_shadows_enabled = false;
        out.sharpen_enabled = false;
        out.grain_enabled = false;
        out.temperature = 0;
        out.tint = 0;
        out.motion_blur_enabled = false;
        out.taa_enabled = false;
        out.shaft_enabled = false;
        return out;
    }

    /// Extract validated auto-exposure options for evaluation/adaptation.
    pub fn autoExposureOptions(self: PostProcessOptions) auto_exposure.AutoExposureOptions {
        const c = self.clamped();
        return .{
            .enabled = c.auto_exposure_enabled,
            .min_exposure = c.auto_exposure_min,
            .max_exposure = c.auto_exposure_max,
            .key_value = c.auto_exposure_key_value,
            .speed_up = c.auto_exposure_speed_up,
            .speed_down = c.auto_exposure_speed_down,
            .low_percentile = c.auto_exposure_low_percentile,
            .high_percentile = c.auto_exposure_high_percentile,
            .camera_cut = c.auto_exposure_camera_cut,
        };
    }

    /// Bind a 2D-strip LUT texture (Babylon.js parity) and enable grading.
    /// The strip geometry is validated (width == size*size, height == size,
    /// live view); invalid input — including null — clears the binding and
    /// disables grading, so the shader can never sample garbage. Consumed
    /// per frame by PostFXStack.renderChain via ChainParams.post, which is
    /// copied from the render-owned SceneFrameSnapshot; that is why the
    /// setter lives on the config struct and not on PostFXStack.
    pub fn setColorGradingLut(self: *PostProcessOptions, tex: ?texture_mod.Texture, size: u8) void {
        if (tex) |t| {
            if (lut.lutTextureValid(t, size)) {
                self.lut_texture = t;
                self.lut_size = size;
                self.lut_enabled = true;
                return;
            }
        }
        self.clearColorGradingLut();
    }

    /// Unbind the LUT and disable grading (back to the bit-identical
    /// no-LUT composite path).
    pub fn clearColorGradingLut(self: *PostProcessOptions) void {
        self.lut_texture = null;
        self.lut_enabled = false;
    }
};

// Options regression tests live in `options_tests.zig` (same directory,
// imported below so the test registry picks them up exactly once).

test {
    _ = @import("options_tests.zig");
}
