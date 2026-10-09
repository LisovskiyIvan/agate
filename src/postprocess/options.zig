const std = @import("std");
const texture_mod = @import("../texture.zig");
const types = @import("types.zig");
const bloom = @import("bloom.zig");
const glow = @import("glow.zig");
const color_curves = @import("color_curves.zig");
const lut = @import("lut.zig");
const shafts = @import("shafts.zig");
const ssgi_mod = @import("ssgi.zig");
const contact_shadows_mod = @import("contact_shadows.zig");
const auto_exposure = @import("auto_exposure.zig");

pub const TonemappingType = types.TonemappingType;
pub const LutFormat = types.LutFormat;
pub const ShaftResolution = types.ShaftResolution;
pub const AutoExposureOptions = auto_exposure.AutoExposureOptions;

/// One row of the clamped() policy table: the exact sanitize op for a field.
/// The op kinds reproduce the historical per-field behavior verbatim:
/// "guarded" lanes map non-finite floats to the spec default, "raw" lanes
/// keep the historical no-guard @max/clamp pass-through (NaN sorts low,
/// +Inf clamps to the bound — probed, not assumed).
const ClampSpec = union(enum) {
    /// finite? clamp(v, min, max) else default.
    guarded_clamp: struct { min: f32, max: f32, default: f32 },
    /// finite? max(v, min) else default.
    guarded_floor: struct { min: f32, default: f32 },
    /// @max(v, min), no finite guard.
    raw_floor: struct { min: f32 },
    /// std.math.clamp(v, min, max), no finite guard.
    raw_clamp: struct { min: f32, max: f32 },
    /// std.math.clamp(v, min, max) on a u32 count.
    uint_clamp: struct { min: u32, max: u32 },
    /// color_curves.clampGrade triplet.
    grade_triplet,
    /// glow.clampTint triplet.
    tint_triplet,
    /// bloom.clampBloomMips.
    bloom_mips,
};

const clamp_table = [_]struct { name: []const u8, spec: ClampSpec }{
    .{ .name = "exposure", .spec = .{ .guarded_clamp = .{ .min = 0.0, .max = 65504.0, .default = 1.0 } } },
    .{ .name = "auto_exposure_min", .spec = .{ .guarded_floor = .{ .min = 0.0, .default = 0.01 } } },
    .{ .name = "auto_exposure_key_value", .spec = .{ .guarded_floor = .{ .min = 0.001, .default = 0.18 } } },
    .{ .name = "auto_exposure_speed_up", .spec = .{ .guarded_floor = .{ .min = 0.0, .default = 3.0 } } },
    .{ .name = "auto_exposure_speed_down", .spec = .{ .guarded_floor = .{ .min = 0.0, .default = 1.0 } } },
    .{ .name = "auto_exposure_low_percentile", .spec = .{ .guarded_clamp = .{ .min = 0.0, .max = 1.0, .default = 0.10 } } },
    .{ .name = "bloom_threshold", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .name = "bloom_intensity", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .name = "bloom_radius", .spec = .{ .guarded_clamp = .{ .min = 0.0, .max = 16.0, .default = 2.0 } } },
    .{ .name = "bloom_pyramid_mips", .spec = .bloom_mips },
    .{ .name = "glow_threshold", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .name = "glow_intensity", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .name = "glow_radius", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .name = "glow_tint", .spec = .tint_triplet },
    .{ .name = "dof_focus_distance", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .name = "dof_focus_range", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .name = "dof_max_blur", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .name = "grade_shadows", .spec = .grade_triplet },
    .{ .name = "grade_midtones", .spec = .grade_triplet },
    .{ .name = "grade_highlights", .spec = .grade_triplet },
    .{ .name = "lut_strength", .spec = .{ .raw_clamp = .{ .min = 0.0, .max = 1.0 } } },
    .{ .name = "render_scale", .spec = .{ .guarded_clamp = .{ .min = min_render_scale, .max = 1.0, .default = 1.0 } } },
    .{ .name = "ssr_steps", .spec = .{ .uint_clamp = .{ .min = 4, .max = 64 } } },
    .{ .name = "ssgi_intensity", .spec = .{ .guarded_clamp = .{ .min = 0.0, .max = 1.0, .default = 0.5 } } },
    .{ .name = "ssgi_radius", .spec = .{ .guarded_clamp = .{ .min = ssgi_mod.SSGI_RADIUS_MIN, .max = ssgi_mod.SSGI_RADIUS_MAX, .default = 1.5 } } },
    .{ .name = "ssgi_steps", .spec = .{ .uint_clamp = .{ .min = ssgi_mod.SSGI_STEPS_MIN, .max = ssgi_mod.SSGI_STEPS_MAX } } },
    .{ .name = "contact_shadows_intensity", .spec = .{ .guarded_clamp = .{ .min = 0.0, .max = 1.0, .default = 0.5 } } },
    .{ .name = "contact_shadows_distance", .spec = .{ .guarded_floor = .{ .min = contact_shadows_mod.CONTACT_SHADOWS_DISTANCE_MIN, .default = 0.3 } } },
    .{ .name = "contact_shadows_thickness", .spec = .{ .guarded_floor = .{ .min = contact_shadows_mod.CONTACT_SHADOWS_THICKNESS_MIN, .default = 0.05 } } },
    .{ .name = "contact_shadows_steps", .spec = .{ .uint_clamp = .{ .min = contact_shadows_mod.CONTACT_SHADOWS_STEPS_MIN, .max = contact_shadows_mod.CONTACT_SHADOWS_STEPS_MAX } } },
    .{ .name = "local_tonemapping_intensity", .spec = .{ .guarded_clamp = .{ .min = 0.0, .max = 1.0, .default = 0.5 } } },
    .{ .name = "local_tonemapping_contrast", .spec = .{ .guarded_clamp = .{ .min = 0.0, .max = 1.0, .default = 0.3 } } },
    .{ .name = "motion_blur_intensity", .spec = .{ .raw_clamp = .{ .min = 0.0, .max = 3.0 } } },
    .{ .name = "motion_blur_max_blur_px", .spec = .{ .raw_clamp = .{ .min = 1.0, .max = 128.0 } } },
    .{ .name = "motion_blur_samples", .spec = .{ .uint_clamp = .{ .min = 2, .max = 32 } } },
    .{ .name = "taa_blend", .spec = .{ .raw_clamp = .{ .min = 0.0, .max = 1.0 } } },
    .{ .name = "taa_jitter_scale", .spec = .{ .raw_clamp = .{ .min = 0.0, .max = 4.0 } } },
    .{ .name = "taa_sharpness", .spec = .{ .raw_clamp = .{ .min = 0.0, .max = 1.0 } } },
    .{ .name = "taa_clamp_strength", .spec = .{ .raw_clamp = .{ .min = 0.0, .max = 1.0 } } },
    .{ .name = "shaft_intensity", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .name = "shaft_steps", .spec = .{ .uint_clamp = .{ .min = shafts.SHAFT_STEPS_MIN, .max = shafts.SHAFT_STEPS_MAX } } },
    .{ .name = "shaft_density", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .name = "shaft_anisotropy", .spec = .{ .raw_clamp = .{ .min = -shafts.SHAFT_ANISOTROPY_MAX, .max = shafts.SHAFT_ANISOTROPY_MAX } } },
    .{ .name = "shaft_max_distance", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .name = "shaft_blur_sigma", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .name = "shaft_edge_sigma", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
};

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

    // Render scale (Q4): scene + effects render into a scaled-down HDR
    // main target (RGBA16F), the composite/UI present at the swapchain
    // size. 1.0 = native (bit-identical to the pre-scale path: every
    // target keeps the window size); 0.5..0.75 trades sharpness for GPU
    // headroom with TAA reconstruction (see docs/graphics-roadmap.md Q4).
    // The TAA history/velocity/auxiliary targets all follow the scaled
    // main size; a scale change resizes them and resets TAA history the
    // same way a window resize does.
    render_scale: f32 = 1.0,

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

    // Screen-Space Global Illumination (v1 color bleed; mirrors in
    // postprocess/ssgi.zig, gather in postprocess.glsl applySSGI). Default
    // OFF: the disabled lane is zeros and the composite stays bit-identical
    // to the pre-SSGI path.
    ssgi_enabled: bool = false,
    ssgi_intensity: f32 = 0.5,
    // World-space influence radius in meters (pixel disk scales by camera
    // distance; clamped [0.1, 8]).
    ssgi_radius: f32 = 1.5,
    ssgi_steps: u32 = 12,

    // Screen-Space Contact Shadows & Local Ambient Occlusion (short-range sun raymarch)
    contact_shadows_enabled: bool = false,
    contact_shadows_intensity: f32 = 0.5,
    contact_shadows_distance: f32 = 0.3,
    contact_shadows_thickness: f32 = 0.05,
    contact_shadows_steps: u32 = 12,

    // Local Tonemapping & Contrast Adaptation (compresses wide dynamic range while preserving local details)
    local_tonemapping_enabled: bool = false,
    local_tonemapping_intensity: f32 = 0.5,
    local_tonemapping_contrast: f32 = 0.3,

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
    // Table-driven: one row per clamped field (see clamp_table above the
    // struct); a newly added field without a row, a manual handler, or an
    // exemption is a compile error (exhaustiveness check below the struct).
    pub fn clamped(self: PostProcessOptions) PostProcessOptions {
        var out = self;
        inline for (clamp_table) |row| {
            if (row.spec == .guarded_clamp) {
                const s = row.spec.guarded_clamp;
                const v: f32 = @field(self, row.name);
                @field(out, row.name) = if (std.math.isFinite(v)) std.math.clamp(v, s.min, s.max) else s.default;
            } else if (row.spec == .guarded_floor) {
                const s = row.spec.guarded_floor;
                const v: f32 = @field(self, row.name);
                @field(out, row.name) = if (std.math.isFinite(v)) @max(v, s.min) else s.default;
            } else if (row.spec == .raw_floor) {
                const s = row.spec.raw_floor;
                const v: f32 = @field(self, row.name);
                @field(out, row.name) = @max(v, s.min);
            } else if (row.spec == .raw_clamp) {
                const s = row.spec.raw_clamp;
                const v: f32 = @field(self, row.name);
                @field(out, row.name) = std.math.clamp(v, s.min, s.max);
            } else if (row.spec == .uint_clamp) {
                const s = row.spec.uint_clamp;
                const v: u32 = @field(self, row.name);
                @field(out, row.name) = std.math.clamp(v, s.min, s.max);
            } else if (row.spec == .grade_triplet) {
                const v: [3]f32 = @field(self, row.name);
                @field(out, row.name) = color_curves.clampGrade(v);
            } else if (row.spec == .tint_triplet) {
                const v: [3]f32 = @field(self, row.name);
                @field(out, row.name) = glow.clampTint(v);
            } else if (row.spec == .bloom_mips) {
                const v: u32 = @field(self, row.name);
                @field(out, row.name) = bloom.clampBloomMips(v);
            } else {
                @compileError("clamped() has no handler for clamp spec kind on field " ++ row.name);
            }
        }
        // Order-dependent leftovers (floor at the already-clamped neighbor).
        out.auto_exposure_max = if (std.math.isFinite(self.auto_exposure_max)) @max(self.auto_exposure_max, out.auto_exposure_min) else 16.0;
        out.auto_exposure_high_percentile = if (std.math.isFinite(self.auto_exposure_high_percentile)) std.math.clamp(self.auto_exposure_high_percentile, out.auto_exposure_low_percentile, 1.0) else 0.90;
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
        inline for (frame_disable_flags) |name| {
            @field(out, name) = false;
        }
        out.chromatic_aberration = 0;
        out.saturation = 1;
        out.contrast = 1;
        out.grade_shadows = .{ 0, 0, 0 };
        out.grade_midtones = .{ 0, 0, 0 };
        out.grade_highlights = .{ 0, 0, 0 };
        out.temperature = 0;
        out.tint = 0;
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

/// Fields clamped() handles outside the table (order-dependent or special):
/// auto_exposure_max floors at the already-clamped min, high_percentile at
/// the already-clamped low, and lut_texture is a handle-validity check.
const manual_clamp_fields = [_][]const u8{
    "auto_exposure_max",
    "auto_exposure_high_percentile",
    "lut_texture",
};

/// Fields clamped() intentionally leaves untouched: master/enable bools,
/// enums, the u8 LUT edge length (validated with the texture binding), and
/// floats the downstream shader math already handles over any value
/// (vignette/saturation/optics, fog + ssr shape knobs, sharpen/grain
/// amounts, white-balance gains).
const exempt_clamp_fields = [_][]const u8{
    "enabled",
    "tonemapping",
    "auto_exposure_enabled",
    "auto_exposure_camera_cut",
    "bloom_enabled",
    "glow_enabled",
    "vignette_enabled",
    "vignette_intensity",
    "vignette_radius",
    "saturation",
    "contrast",
    "chromatic_aberration",
    "dof_enabled",
    "lut_enabled",
    "lut_size",
    "lut_format",
    "fxaa_enabled",
    "fog_enabled",
    "fog_density",
    "fog_height_falloff",
    "fog_start_distance",
    "fog_color",
    "fog_sun_scattering",
    "ssr_enabled",
    "ssr_intensity",
    "ssr_max_distance",
    "ssr_thickness",
    "depth_pyramid_enabled",
    "ssgi_enabled",
    "contact_shadows_enabled",
    "local_tonemapping_enabled",
    "sharpen_enabled",
    "sharpen_amount",
    "grain_enabled",
    "grain_intensity",
    "temperature",
    "tint",
    "motion_blur_enabled",
    "taa_enabled",
    "taa_camera_cut",
    "shaft_enabled",
    "shaft_resolution",
};

/// Effect-enable flags forFrame() clears when the master `enabled` is off.
/// Typo-safe: @field on an unknown name is a compile error. Deliberately
/// NOT exhaustive over all bools: exposure/tonemapping/render_scale,
/// auto-exposure, the depth pyramid, and the LUT binding are render-owned
/// state, not artistic effects, and survive the master switch (as before).
const frame_disable_flags = [_][]const u8{
    "bloom_enabled",
    "glow_enabled",
    "vignette_enabled",
    "dof_enabled",
    "lut_enabled",
    "fxaa_enabled",
    "fog_enabled",
    "ssr_enabled",
    "contact_shadows_enabled",
    "ssgi_enabled",
    "local_tonemapping_enabled",
    "sharpen_enabled",
    "grain_enabled",
    "motion_blur_enabled",
    "taa_enabled",
    "shaft_enabled",
};

comptime {
    @setEvalBranchQuota(20000);
    const option_fields = @typeInfo(PostProcessOptions).@"struct".fields;
    for (option_fields) |f| {
        var covered = false;
        for (clamp_table) |row| {
            if (std.mem.eql(u8, row.name, f.name)) {
                covered = true;
                break;
            }
        }
        if (!covered) {
            for (manual_clamp_fields) |m| {
                if (std.mem.eql(u8, m, f.name)) {
                    covered = true;
                    break;
                }
            }
        }
        if (!covered) {
            for (exempt_clamp_fields) |e| {
                if (std.mem.eql(u8, e, f.name)) {
                    covered = true;
                    break;
                }
            }
        }
        if (!covered) @compileError("PostProcessOptions." ++ f.name ++ " needs a clamp-table row, a manual handler, or an exemption.");
    }
    // Reverse direction: every listed name must be a real field, so a
    // rename/typo fails here instead of silently dropping coverage.
    for (clamp_table) |row| {
        if (!@hasField(PostProcessOptions, row.name)) @compileError("clamp-table row for unknown field: " ++ row.name);
    }
    for (manual_clamp_fields) |m| {
        if (!@hasField(PostProcessOptions, m)) @compileError("manual clamp handler for unknown field: " ++ m);
    }
    for (exempt_clamp_fields) |e| {
        if (!@hasField(PostProcessOptions, e)) @compileError("clamp exemption for unknown field: " ++ e);
    }
    for (frame_disable_flags) |n| {
        if (!@hasField(PostProcessOptions, n)) @compileError("forFrame disable flag for unknown field: " ++ n);
    }
}

/// Lowest accepted `render_scale` (below this the upscale is mush and the
/// TAA reconstruction cannot keep up; raise the floor before lowering it).
pub const min_render_scale: f32 = 0.25;

/// Scaled main-target size for a window size and a render scale. Pure
/// math (no GPU), mirrored by PostFXStack.prepareMainTargets/resizeAll:
/// nearest rounding keeps the aspect error under half a source pixel, and
/// each dimension clamps to at least 1 so a degenerate window never
/// produces a zero-sized target. Pass an already-clamped scale.
pub const RenderSize = struct { w: i32, h: i32 };

pub fn scaledRenderSize(width: i32, height: i32, scale: f32) RenderSize {
    if (width <= 0 or height <= 0) return .{ .w = width, .h = height };
    const s = std.math.clamp(scale, min_render_scale, 1.0);
    const w: i32 = @intFromFloat(@max(1.0, @round(@as(f32, @floatFromInt(width)) * s)));
    const h: i32 = @intFromFloat(@max(1.0, @round(@as(f32, @floatFromInt(height)) * s)));
    return .{ .w = w, .h = h };
}

// Options regression tests live in `options_tests.zig` (same directory,
// imported below so the test registry picks them up exactly once).
