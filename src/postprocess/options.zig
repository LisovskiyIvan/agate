const std = @import("std");
const texture_mod = @import("../texture.zig");
const types = @import("types.zig");
const bloom = @import("bloom.zig");
const glow = @import("glow.zig");
const color_curves = @import("color_curves.zig");
const lut = @import("lut.zig");
const shafts = @import("shafts.zig");
const ssgi_mod = @import("ssgi.zig");
const ssr_mod = @import("ssr.zig");
const contact_shadows_mod = @import("contact_shadows.zig");
const auto_exposure = @import("auto_exposure.zig");

pub const TonemappingType = types.TonemappingType;
pub const ShaftResolution = types.ShaftResolution;

/// Auto-exposure (adaptive luminance / histogram metering). Defaults are
/// the historical flat-field values verbatim.
pub const AutoExposureOptions = struct {
    enabled: bool = false,
    min: f32 = 0.01,
    max: f32 = 16.0,
    key_value: f32 = 0.18,
    speed_up: f32 = 3.0,
    speed_down: f32 = 1.0,
    camera_cut: bool = false,
};

/// HDR bloom: Karis downsample + tent upsample pyramid.
pub const BloomOptions = struct {
    enabled: bool = true,
    threshold: f32 = 0.8,
    intensity: f32 = 0.5,
    /// Upsample tent radius in source-mip texels.
    radius: f32 = 2.0,
    /// Pyramid depth (was `bloom_pyramid_mips`).
    mips: u32 = 5,
};

/// Glow layer v1 (global halo, distinct from bloom): luminance threshold
/// extract + separable blur (GlowPass) + additive composite AFTER bloom
/// in postprocess.glsl. Default OFF: no glow passes run, the composite
/// branch is skipped, and rendering stays bit-identical to pre-glow.
/// Per-mesh glow weights are explicitly out of scope for v1 (they need
/// pipeline changes); this is the global pass only.
pub const GlowOptions = struct {
    enabled: bool = false,
    /// Luminance cut for the extract stage; >= 0 (unitless HDR-ish scale).
    threshold: f32 = glow.GLOW_THRESHOLD_DEFAULT,
    /// Additive composite scale; >= 0.
    intensity: f32 = glow.GLOW_INTENSITY_DEFAULT,
    /// Blur sigma in glow-target texels; >= 0, wider than bloom's radius.
    radius: f32 = glow.GLOW_RADIUS_DEFAULT,
    /// Composite color multiplier, per channel in [0, 1].
    tint: [3]f32 = .{ 1.0, 1.0, 1.0 },
};

/// Vignette (cinematic lens falloff).
pub const VignetteOptions = struct {
    enabled: bool = true,
    intensity: f32 = 0.35,
    radius: f32 = 0.8,
};

/// Parametric color grading and optics: saturation/contrast plus
/// per-channel additive lifts in [-1, 1] weighted by
/// shadows/midtones/highlights luminance zones. Primary grading path
/// (no texture dependency). A LUT texture remains an opt-in parent-side
/// extension (see LutOptions).
pub const GradeOptions = struct {
    saturation: f32 = 1.05,
    contrast: f32 = 1.05,
    chromatic_aberration: f32 = 0.0,
    shadows: [3]f32 = .{ 0.0, 0.0, 0.0 },
    midtones: [3]f32 = .{ 0.0, 0.0, 0.0 },
    highlights: [3]f32 = .{ 0.0, 0.0, 0.0 },
};

/// Depth of Field (gather blur driven by linearized depth).
pub const DofOptions = struct {
    enabled: bool = false,
    focus_distance: f32 = 10.0,
    focus_range: f32 = 5.0,
    max_blur: f32 = 8.0,
};

/// Texture LUT color grading (Babylon.js parity), sampled after the
/// parametric curves. With texture == null (the default) the shader
/// skips the LUT branch entirely and the composite path stays
/// bit-identical to pre-LUT. The handle is a by-value Texture (plain
/// GPU handles, no CPU refs), so the whole struct — including the LUT —
/// copies into SceneFrameSnapshot.post_process with a plain assignment
/// (scene.zig packFrameSnapshot) and needs no snapshot-side support.
pub const LutOptions = struct {
    enabled: bool = false,
    /// Grading strength (was `lut_strength`).
    intensity: f32 = 1.0,
    size: u8 = 16,
    texture: ?texture_mod.Texture = null,

    /// Bind a 2D-strip LUT texture (Babylon.js parity) and enable grading.
    /// The strip geometry is validated (width == size*size, height == size,
    /// live view); invalid input — including null — clears the binding and
    /// disables grading, so the shader can never sample garbage. Consumed
    /// per frame by PostFXStack.renderChain via ChainParams.post, which is
    /// copied from the render-owned SceneFrameSnapshot; that is why the
    /// setter lives on the config struct and not on PostFXStack.
    ///
    /// `enabled` is a plain bool kept in sync by these methods (NOT derived
    /// from `texture != null`): forFrame() clears the effect flag while the
    /// binding survives the master switch, exactly like the historical
    /// flat `lut_enabled`/`lut_texture` pair.
    pub fn setColorGradingLut(self: *LutOptions, tex: ?texture_mod.Texture, size: u8) void {
        if (tex) |t| {
            if (lut.lutTextureValid(t, size)) {
                self.texture = t;
                self.size = size;
                self.enabled = true;
                return;
            }
        }
        self.clearColorGradingLut();
    }

    /// Unbind the LUT and disable grading (back to the bit-identical
    /// no-LUT composite path).
    pub fn clearColorGradingLut(self: *LutOptions) void {
        self.texture = null;
        self.enabled = false;
    }
};

/// Anti-aliasing (FXAA 3.11 sub-pixel edge smoothing).
pub const FxaaOptions = struct {
    enabled: bool = true,
};

/// Atmospheric depth and height fog.
pub const FogOptions = struct {
    enabled: bool = true,
    density: f32 = 0.015,
    height_falloff: f32 = 0.08,
    start_distance: f32 = 5.0,
    color: [3]f32 = .{ 0.72, 0.82, 0.92 },
    sun_scattering: f32 = 0.8,
};

/// Screen-space reflections. The former `ssr_max_distance` (25.0) and
/// `ssr_thickness` (0.4) knobs were dropped (zero non-default setters);
/// the composite packs those historical defaults as constants.
pub const SsrOptions = struct {
    enabled: bool = true,
    intensity: f32 = 0.55,
    steps: u32 = 16,
    /// Execution mode: .inline_ runs the v1 raymarch inside the composite
    /// (default, bit-identical); .half_res_pass runs the dedicated
    /// half-res trace + bilateral denoise/upscale (SsrPass) and mixes the
    /// denoised reflection instead. An enum like shafts.resolution: no
    /// clamp row, survives forFrame() untouched (the enabled flag still
    /// gates both lanes).
    mode: ssr_mod.SsrMode = .inline_,
};

/// Temporal accumulation for the SSGI half_res_pass gather (v2.1, opt-in).
/// Default OFF: the pass runs gather -> denoise exactly as before
/// (bit-identical). Meaningful only with ssgi.mode == .half_res_pass;
/// in inline mode the flag warns once (orchestrator) and is ignored.
pub const SsgiTemporalOptions = struct {
    enabled: bool = false,
    /// History weight for the temporal blend (1 = full history). 0.9 is the
    /// standard compromise; the documented sweet spot is [0.85, 0.95].
    /// Clamped to [0, 1].
    blend: f32 = ssgi_mod.SSGI_TEMPORAL_BLEND_DEFAULT,
};

/// Screen-space global illumination (v1 color bleed; mirrors in
/// postprocess/ssgi.zig, gather in postprocess.glsl applySSGI). Default
/// OFF: the disabled lane is zeros and the composite stays bit-identical
/// to the pre-SSGI path.
pub const SsgiOptions = struct {
    enabled: bool = false,
    /// Execution mode: .inline_gather runs the v1 gather inside the
    /// composite; .half_res_pass runs the v2 dedicated half-res gather +
    /// bilateral denoise (SsgiPass) and samples the denoised texture
    /// instead. Default inline (bit-identical). An enum like
    /// shafts.resolution: no clamp row, survives forFrame() untouched (the
    /// enabled flag still gates both lanes).
    mode: ssgi_mod.SsgiMode = .inline_gather,
    intensity: f32 = 0.5,
    /// World-space influence radius in meters (pixel disk scales by camera
    /// distance; clamped [0.1, 8]).
    radius: f32 = 1.5,
    steps: u32 = 12,
    temporal: SsgiTemporalOptions = .{},
};

/// Screen-space contact shadows and local ambient occlusion (short-range
/// sun raymarch).
pub const ContactShadowsOptions = struct {
    enabled: bool = false,
    intensity: f32 = 0.5,
    distance: f32 = 0.3,
    thickness: f32 = 0.05,
    steps: u32 = 12,
};

/// Local tonemapping and contrast adaptation (compresses wide dynamic
/// range while preserving local details). The former
/// `local_tonemapping_contrast` knob (0.3) was dropped (zero non-default
/// setters); localTonemapParams() packs that historical default.
pub const LocalTonemapOptions = struct {
    enabled: bool = false,
    intensity: f32 = 0.5,
};

/// Sharpen (post-tonemap unsharp mask). `intensity` was `sharpen_amount`.
pub const SharpenOptions = struct {
    enabled: bool = false,
    intensity: f32 = 0.3,
};

/// Film grain (post-tonemap hash noise, luminance-masked).
pub const GrainOptions = struct {
    enabled: bool = false,
    intensity: f32 = 0.05,
};

/// White balance (post-tonemap channel gains, 0 = neutral).
pub const WhiteBalanceOptions = struct {
    temperature: f32 = 0.0,
    tint: f32 = 0.0,
};

/// Camera motion blur (screen velocity gather blur). `steps` was
/// `motion_blur_samples`. The former `motion_blur_intensity` knob (0.5)
/// was dropped (zero non-default setters); the composite packs that
/// historical default as a constant.
pub const MotionBlurOptions = struct {
    enabled: bool = false,
    max_blur_px: f32 = 32.0,
    steps: u32 = 8,
};

/// Reactive treatment for velocity-less content (particles, glass,
/// disocclusion over background): fallback-path pixels over an
/// inconsistent depth neighborhood shed history proportionally to
/// intensity (0 = off). Default OFF: the packed lane is zeros and the
/// composite stays bit-identical to the pre-reactive path.
pub const TaaReactiveOptions = struct {
    enabled: bool = false,
    /// Was `taa_reactive_strength`.
    intensity: f32 = 0.5,
    /// True-marker shed (C.3 v2) for velocity-pass particle coverage:
    /// marked pixels shed history by this amount with no edge ramp
    /// (1 = full history reject, 0 = markers drawn but ignored).
    /// Default 1.0 (full): the stamped coverage IS the signal.
    /// Master-gated by reactive.enabled like the heuristic lane (same
    /// packed vec4, z slot); a numeric amount, so forFrame() leaves it
    /// alone exactly like intensity (inert while the flag is off).
    /// Was `taa_reactive_marker_strength`.
    marker_intensity: f32 = 1.0,
};

/// Temporal anti-aliasing (sub-pixel Halton jitter + history resolve in
/// postprocess.glsl; history ping-pong in PostProcessPass; jittered VP
/// from Scene.render). Default OFF: the disabled shader path early-outs
/// before any history/depth sampling (bit-identical composite).
/// `clamp_intensity` was `taa_clamp_strength`.
pub const TaaOptions = struct {
    enabled: bool = false,
    /// History weight for the temporal blend (1 = full history). 0.9 is the
    /// standard compromise between stability and ghosting/disocclusion lag.
    blend: f32 = 0.9,
    /// Sub-pixel jitter amplitude multiplier (1 = +/-0.5 px Halton offsets).
    jitter_scale: f32 = 1.0,
    /// Post-blend unsharp amount (0 = off), re-clamped to the neighborhood
    /// box so low amounts cannot ring.
    sharpness: f32 = 0.0,
    /// Neighborhood-clamp amount for ghosting control (1 = full clamp of
    /// the reprojected history into the 3x3 current-frame box, 0 = raw).
    clamp_intensity: f32 = 1.0,
    reactive: TaaReactiveOptions = .{},
    /// One-frame history reset request (camera teleport/cut/cinematic).
    /// Snapshot-carried, so the update thread sets it safely for one frame.
    camera_cut: bool = false,
};

/// Volumetric light shafts v1 (sun CSM-backed raymarch, low-res).
/// Screen-space raymarch from the camera through each low-res pixel to
/// the scene depth surface, sampling the SUN's CSM atlas at every step
/// (single raw-depth tap per step, no PCF in v1) and accumulating
/// single-scatter with a Henyey-Greenstein phase term and Beer-Lambert
/// transmittance; bilateral (depth-aware) H/V blur; additive composite
/// after the highlight block. Default OFF: zero passes run, the
/// composite branch is skipped, and rendering stays bit-identical.
/// Requires live CSM data (gated on shadows_enabled at the call site,
/// see shaftActive): without a rendered CSM atlas there is nothing
/// real to march against, and v1 refuses to fake it.
pub const ShaftsOptions = struct {
    enabled: bool = false,
    /// Additive composite scale; >= 0.
    intensity: f32 = 1.0,
    /// Raymarch steps per pixel; clamped to [4, 32] (the shader loop bound).
    steps: u32 = 12,
    /// Extinction/scattering coefficient per meter; >= 0.
    density: f32 = 0.05,
    /// Henyey-Greenstein anisotropy in [-0.9, 0.9]; > 0 forward scattering
    /// (bright shafts looking toward the sun), 0 isotropic.
    anisotropy: f32 = 0.4,
    /// Raymarch span cap in meters; >= 0 (sky pixels march the full span).
    max_distance: f32 = 60.0,
    /// Raymarch/blur target resolution (half or quarter of the base size).
    resolution: ShaftResolution = .quarter,
    /// Bilateral blur spatial sigma in shaft-target texels; >= 0.
    blur_sigma: f32 = 2.0,
    /// Bilateral blur depth gate in raw-depth units; >= 0 (0 = no depth
    /// weighting, plain Gaussian).
    edge_sigma: f32 = 0.02,
};

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

/// Top-level (non-grouped) clamp rows. Only `exposure` and `render_scale`
/// live outside an effect group; everything else nests (see clamp_table).
const top_clamp_table = [_]struct { name: []const u8, spec: ClampSpec }{
    .{ .name = "exposure", .spec = .{ .guarded_clamp = .{ .min = 0.0, .max = 65504.0, .default = 1.0 } } },
    .{ .name = "render_scale", .spec = .{ .guarded_clamp = .{ .min = min_render_scale, .max = 1.0, .default = 1.0 } } },
};

/// Per-effect-group clamp rows, keyed by dotted path ("group.leaf").
/// Only depth-1 leaves live here (applied by the generic loop in
/// clamped()); the three depth-2 leaves (ssgi.temporal / taa.reactive)
/// are applied by direct lines right after the loop and listed under
/// manual_clamp_fields. One row per clamped leaf; a newly added leaf
/// without a row, a manual handler, or an exemption is a compile error
/// (the exhaustiveness check below the struct walks every nested leaf).
const clamp_table = [_]struct { path: []const u8, spec: ClampSpec }{
    .{ .path = "auto_exposure.min", .spec = .{ .guarded_floor = .{ .min = 0.0, .default = 0.01 } } },
    .{ .path = "auto_exposure.key_value", .spec = .{ .guarded_floor = .{ .min = 0.001, .default = 0.18 } } },
    .{ .path = "auto_exposure.speed_up", .spec = .{ .guarded_floor = .{ .min = 0.0, .default = 3.0 } } },
    .{ .path = "auto_exposure.speed_down", .spec = .{ .guarded_floor = .{ .min = 0.0, .default = 1.0 } } },
    .{ .path = "bloom.threshold", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .path = "bloom.intensity", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .path = "bloom.radius", .spec = .{ .guarded_clamp = .{ .min = 0.0, .max = 16.0, .default = 2.0 } } },
    .{ .path = "bloom.mips", .spec = .bloom_mips },
    .{ .path = "glow.threshold", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .path = "glow.intensity", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .path = "glow.radius", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .path = "glow.tint", .spec = .tint_triplet },
    .{ .path = "dof.focus_distance", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .path = "dof.focus_range", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .path = "dof.max_blur", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .path = "grade.shadows", .spec = .grade_triplet },
    .{ .path = "grade.midtones", .spec = .grade_triplet },
    .{ .path = "grade.highlights", .spec = .grade_triplet },
    .{ .path = "lut.intensity", .spec = .{ .raw_clamp = .{ .min = 0.0, .max = 1.0 } } },
    .{ .path = "ssr.steps", .spec = .{ .uint_clamp = .{ .min = 4, .max = 64 } } },
    .{ .path = "ssgi.intensity", .spec = .{ .guarded_clamp = .{ .min = 0.0, .max = 1.0, .default = 0.5 } } },
    .{ .path = "ssgi.radius", .spec = .{ .guarded_clamp = .{ .min = ssgi_mod.SSGI_RADIUS_MIN, .max = ssgi_mod.SSGI_RADIUS_MAX, .default = 1.5 } } },
    .{ .path = "ssgi.steps", .spec = .{ .uint_clamp = .{ .min = ssgi_mod.SSGI_STEPS_MIN, .max = ssgi_mod.SSGI_STEPS_MAX } } },
    .{ .path = "contact_shadows.intensity", .spec = .{ .guarded_clamp = .{ .min = 0.0, .max = 1.0, .default = 0.5 } } },
    .{ .path = "contact_shadows.distance", .spec = .{ .guarded_floor = .{ .min = contact_shadows_mod.CONTACT_SHADOWS_DISTANCE_MIN, .default = 0.3 } } },
    .{ .path = "contact_shadows.thickness", .spec = .{ .guarded_floor = .{ .min = contact_shadows_mod.CONTACT_SHADOWS_THICKNESS_MIN, .default = 0.05 } } },
    .{ .path = "contact_shadows.steps", .spec = .{ .uint_clamp = .{ .min = contact_shadows_mod.CONTACT_SHADOWS_STEPS_MIN, .max = contact_shadows_mod.CONTACT_SHADOWS_STEPS_MAX } } },
    .{ .path = "local_tonemapping.intensity", .spec = .{ .guarded_clamp = .{ .min = 0.0, .max = 1.0, .default = 0.5 } } },
    .{ .path = "motion_blur.max_blur_px", .spec = .{ .raw_clamp = .{ .min = 1.0, .max = 128.0 } } },
    .{ .path = "motion_blur.steps", .spec = .{ .uint_clamp = .{ .min = 2, .max = 32 } } },
    .{ .path = "taa.blend", .spec = .{ .raw_clamp = .{ .min = 0.0, .max = 1.0 } } },
    .{ .path = "taa.jitter_scale", .spec = .{ .raw_clamp = .{ .min = 0.0, .max = 4.0 } } },
    .{ .path = "taa.sharpness", .spec = .{ .raw_clamp = .{ .min = 0.0, .max = 1.0 } } },
    .{ .path = "taa.clamp_intensity", .spec = .{ .raw_clamp = .{ .min = 0.0, .max = 1.0 } } },
    .{ .path = "shafts.intensity", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .path = "shafts.steps", .spec = .{ .uint_clamp = .{ .min = shafts.SHAFT_STEPS_MIN, .max = shafts.SHAFT_STEPS_MAX } } },
    .{ .path = "shafts.density", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .path = "shafts.anisotropy", .spec = .{ .raw_clamp = .{ .min = -shafts.SHAFT_ANISOTROPY_MAX, .max = shafts.SHAFT_ANISOTROPY_MAX } } },
    .{ .path = "shafts.max_distance", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .path = "shafts.blur_sigma", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
    .{ .path = "shafts.edge_sigma", .spec = .{ .raw_floor = .{ .min = 0.0 } } },
};

/// Split a dotted option path ("group.leaf") into group and leaf.
/// Depth-2 paths never reach this helper: the three depth-2 leaves are
/// applied by direct lines, not the generic loop.
fn splitOptionPath(comptime path: []const u8) struct { group: []const u8, leaf: []const u8 } {
    const idx = std.mem.indexOfScalar(u8, path, '.') orelse @compileError("option path needs group.leaf form: " ++ path);
    return .{ .group = path[0..idx], .leaf = path[idx + 1 ..] };
}

/// Read a nested leaf by group and leaf name.
fn getLeaf(comptime T: type, cfg: PostProcessOptions, comptime group: []const u8, comptime leaf: []const u8) T {
    return @field(@field(cfg, group), leaf);
}

/// Write a nested leaf by group and leaf name.
fn setLeaf(comptime T: type, cfg: *PostProcessOptions, comptime group: []const u8, comptime leaf: []const u8, value: T) void {
    @field(@field(cfg.*, group), leaf) = value;
}

/// Post-processing chain knobs (exposure, tonemapping, per-effect option
/// groups). A nested config read/written by tooling; applied per frame by
/// `Scene.postfx` (PostFXStack.renderChain).
pub const PostProcessOptions = struct {
    /// Artistic effects master switch. HDR output, exposure and tonemapping
    /// remain active even when effects are disabled.
    enabled: bool = false,
    exposure: f32 = 1.0,
    tonemapping: TonemappingType = .aces,

    auto_exposure: AutoExposureOptions = .{},
    bloom: BloomOptions = .{},
    glow: GlowOptions = .{},
    vignette: VignetteOptions = .{},
    grade: GradeOptions = .{},
    dof: DofOptions = .{},
    lut: LutOptions = .{},
    fxaa: FxaaOptions = .{},
    /// Render scale (Q4): scene + effects render into a scaled-down HDR
    /// main target (RGBA16F), the composite/UI present at the swapchain
    /// size. 1.0 = native (bit-identical to the pre-scale path: every
    /// target keeps the window size); 0.5..0.75 trades sharpness for GPU
    /// headroom with TAA reconstruction (see docs/graphics-roadmap.md Q4).
    /// Velocity and the auxiliary effect targets follow the scaled main
    /// size. The TAA history intentionally stays at the display size: the
    /// composite resolves TAA at output pixels (temporal upscale v1) and
    /// reprojects with the low-res depth/velocity. A scale change resets
    /// TAA history explicitly, since the history size itself no longer
    /// changes with the scale.
    render_scale: f32 = 1.0,
    fog: FogOptions = .{},
    ssr: SsrOptions = .{},
    /// Hierarchical GPU Depth Pyramid (Hi-Z downsample mips for SSR, contact shadows, occlusion)
    depth_pyramid_enabled: bool = false,
    ssgi: SsgiOptions = .{},
    contact_shadows: ContactShadowsOptions = .{},
    local_tonemapping: LocalTonemapOptions = .{},
    sharpen: SharpenOptions = .{},
    grain: GrainOptions = .{},
    white_balance: WhiteBalanceOptions = .{},
    motion_blur: MotionBlurOptions = .{},
    taa: TaaOptions = .{},
    shafts: ShaftsOptions = .{},

    // Return a copy with out-of-range values pulled into valid ranges.
    // Never fails; safe to apply on load or before uploading uniforms.
    // Table-driven: one row per clamped field (top_clamp_table for the two
    // top-level leaves, clamp_table for every nested leaf); a newly added
    // field without a row, a manual handler, or an exemption is a compile
    // error (exhaustiveness check below the struct).
    pub fn clamped(self: PostProcessOptions) PostProcessOptions {
        var out = self;
        inline for (top_clamp_table) |row| {
            if (row.spec == .guarded_clamp) {
                const s = row.spec.guarded_clamp;
                const v: f32 = @field(self, row.name);
                @field(out, row.name) = if (std.math.isFinite(v)) std.math.clamp(v, s.min, s.max) else s.default;
            } else {
                @compileError("top-level clamp row must be a guarded_clamp: " ++ row.name);
            }
        }
        inline for (clamp_table) |row| {
            const p = comptime splitOptionPath(row.path);
            if (row.spec == .guarded_clamp) {
                const s = row.spec.guarded_clamp;
                const v: f32 = getLeaf(f32, self, p.group, p.leaf);
                setLeaf(f32, &out, p.group, p.leaf, if (std.math.isFinite(v)) std.math.clamp(v, s.min, s.max) else s.default);
            } else if (row.spec == .guarded_floor) {
                const s = row.spec.guarded_floor;
                const v: f32 = getLeaf(f32, self, p.group, p.leaf);
                setLeaf(f32, &out, p.group, p.leaf, if (std.math.isFinite(v)) @max(v, s.min) else s.default);
            } else if (row.spec == .raw_floor) {
                const s = row.spec.raw_floor;
                const v: f32 = getLeaf(f32, self, p.group, p.leaf);
                setLeaf(f32, &out, p.group, p.leaf, @max(v, s.min));
            } else if (row.spec == .raw_clamp) {
                const s = row.spec.raw_clamp;
                const v: f32 = getLeaf(f32, self, p.group, p.leaf);
                setLeaf(f32, &out, p.group, p.leaf, std.math.clamp(v, s.min, s.max));
            } else if (row.spec == .uint_clamp) {
                const s = row.spec.uint_clamp;
                const v: u32 = getLeaf(u32, self, p.group, p.leaf);
                setLeaf(u32, &out, p.group, p.leaf, std.math.clamp(v, s.min, s.max));
            } else if (row.spec == .grade_triplet) {
                const v: [3]f32 = getLeaf([3]f32, self, p.group, p.leaf);
                setLeaf([3]f32, &out, p.group, p.leaf, color_curves.clampGrade(v));
            } else if (row.spec == .tint_triplet) {
                const v: [3]f32 = getLeaf([3]f32, self, p.group, p.leaf);
                setLeaf([3]f32, &out, p.group, p.leaf, glow.clampTint(v));
            } else if (row.spec == .bloom_mips) {
                const v: u32 = getLeaf(u32, self, p.group, p.leaf);
                setLeaf(u32, &out, p.group, p.leaf, bloom.clampBloomMips(v));
            } else {
                @compileError("clamped() has no handler for clamp spec kind on path " ++ row.path);
            }
        }
        // Order-dependent leftover (floor at the already-clamped neighbor).
        out.auto_exposure.max = if (std.math.isFinite(self.auto_exposure.max)) @max(self.auto_exposure.max, out.auto_exposure.min) else 16.0;
        // Depth-2 leaves (the only groups nesting twice): raw [0, 1] lanes,
        // applied directly — same op the table would apply, minus the
        // generic loop (Zig keeps optional-capture indices runtime, so a
        // depth-2 generic @field does not evaluate at comptime).
        out.ssgi.temporal.blend = std.math.clamp(self.ssgi.temporal.blend, 0.0, 1.0);
        out.taa.reactive.intensity = std.math.clamp(self.taa.reactive.intensity, 0.0, 1.0);
        out.taa.reactive.marker_intensity = std.math.clamp(self.taa.reactive.marker_intensity, 0.0, 1.0);
        // A binding without a live view, a supported size, or matching
        // strip dims can never be sampled; drop it so the pass keeps its
        // no-LUT path instead of binding a dead handle.
        if (out.lut.texture) |tex| {
            if (!lut.lutTextureValid(tex, out.lut.size)) out.lut.texture = null;
        }
        return out;
    }

    /// Render-owned copy: disable effects, never the linear HDR/output path.
    pub fn forFrame(self: PostProcessOptions) PostProcessOptions {
        var out = self.clamped();
        if (out.enabled) return out;
        inline for (frame_disable_flags) |path| {
            const p = comptime splitOptionPath(path);
            setLeaf(bool, &out, p.group, p.leaf, false);
        }
        // Depth-2 effect flag (same family as the loop above, applied
        // directly for the same comptime reason as the depth-2 clamps).
        out.taa.reactive.enabled = false;
        out.grade.chromatic_aberration = 0;
        out.grade.saturation = 1;
        out.grade.contrast = 1;
        out.grade.shadows = .{ 0, 0, 0 };
        out.grade.midtones = .{ 0, 0, 0 };
        out.grade.highlights = .{ 0, 0, 0 };
        out.white_balance.temperature = 0;
        out.white_balance.tint = 0;
        return out;
    }

    /// Extract validated auto-exposure options for evaluation/adaptation.
    /// The histogram metering percentiles are fixed internal constants
    /// (0.10/0.90 — the historical defaults): zero non-default setters
    /// existed, so the knobs were dropped instead of nested.
    pub fn autoExposureOptions(self: PostProcessOptions) auto_exposure.AutoExposureOptions {
        const c = self.clamped();
        return .{
            .enabled = c.auto_exposure.enabled,
            .min_exposure = c.auto_exposure.min,
            .max_exposure = c.auto_exposure.max,
            .key_value = c.auto_exposure.key_value,
            .speed_up = c.auto_exposure.speed_up,
            .speed_down = c.auto_exposure.speed_down,
            .low_percentile = 0.10,
            .high_percentile = 0.90,
            .camera_cut = c.auto_exposure.camera_cut,
        };
    }
};

/// Fields clamped() handles outside the tables (order-dependent, special,
/// or depth-2): auto_exposure.max floors at the already-clamped min,
/// lut.texture is a handle-validity check, and the three depth-2 leaves
/// are raw [0, 1] lanes applied by direct lines. Dotted paths, like the
/// table.
const manual_clamp_fields = [_][]const u8{
    "auto_exposure.max",
    "lut.texture",
    "ssgi.temporal.blend",
    "taa.reactive.intensity",
    "taa.reactive.marker_intensity",
};

/// Leaves clamped() intentionally leaves untouched: master/enable bools,
/// enums, the u8 LUT edge length (validated with the texture binding), and
/// floats the downstream shader math already handles over any value
/// (vignette/saturation/optics, fog + ssr shape knobs, sharpen/grain
/// amounts, white-balance gains). Dotted paths for nested leaves, plain
/// names for the three top-level scalars.
const exempt_clamp_fields = [_][]const u8{
    "enabled",
    "tonemapping",
    "auto_exposure.enabled",
    "auto_exposure.camera_cut",
    "bloom.enabled",
    "glow.enabled",
    "vignette.enabled",
    "vignette.intensity",
    "vignette.radius",
    "grade.saturation",
    "grade.contrast",
    "grade.chromatic_aberration",
    "dof.enabled",
    "lut.enabled",
    "lut.size",
    "fxaa.enabled",
    "fog.enabled",
    "fog.density",
    "fog.height_falloff",
    "fog.start_distance",
    "fog.color",
    "fog.sun_scattering",
    "ssr.enabled",
    "ssr.intensity",
    "ssr.mode",
    "depth_pyramid_enabled",
    "ssgi.enabled",
    "ssgi.mode",
    "ssgi.temporal.enabled",
    "contact_shadows.enabled",
    "local_tonemapping.enabled",
    "sharpen.enabled",
    "sharpen.intensity",
    "grain.enabled",
    "grain.intensity",
    "white_balance.temperature",
    "white_balance.tint",
    "motion_blur.enabled",
    "taa.enabled",
    "taa.reactive.enabled",
    "taa.camera_cut",
    "shafts.enabled",
    "shafts.resolution",
};

/// Effect-enable flags forFrame() clears when the master `enabled` is off.
/// Typo-safe: an unknown path is a compile error. Deliberately NOT
/// exhaustive over all bools: exposure/tonemapping/render_scale,
/// auto-exposure, the depth pyramid, and the LUT binding are render-owned
/// state, not artistic effects, and survive the master switch (as before).
const frame_disable_flags = [_][]const u8{
    "bloom.enabled",
    "glow.enabled",
    "vignette.enabled",
    "dof.enabled",
    "lut.enabled",
    "fxaa.enabled",
    "fog.enabled",
    "ssr.enabled",
    "contact_shadows.enabled",
    "ssgi.enabled",
    "local_tonemapping.enabled",
    "sharpen.enabled",
    "grain.enabled",
    "motion_blur.enabled",
    "taa.enabled",
    "shafts.enabled",
};

/// True when a dotted/exempt path names a real leaf of PostProcessOptions
/// (recurses through nested groups, so a rename/typo fails at compile time
/// instead of silently dropping coverage).
fn hasOptionPath(comptime T: type, comptime path: []const u8) bool {
    const idx = std.mem.indexOfScalar(u8, path, '.') orelse return @hasField(T, path);
    if (!@hasField(T, path[0..idx])) return false;
    return hasOptionPath(@FieldType(T, path[0..idx]), path[idx + 1 ..]);
}

/// Require one leaf's coverage: a top-table row, a group-table row, a
/// manual handler, or an exemption.
fn requireLeafCovered(comptime key: []const u8) void {
    for (top_clamp_table) |row| {
        if (std.mem.eql(u8, row.name, key)) return;
    }
    for (clamp_table) |row| {
        if (std.mem.eql(u8, row.path, key)) return;
    }
    for (manual_clamp_fields) |m| {
        if (std.mem.eql(u8, m, key)) return;
    }
    for (exempt_clamp_fields) |e| {
        if (std.mem.eql(u8, e, key)) return;
    }
    @compileError("PostProcessOptions leaf " ++ key ++ " needs a clamp-table row, a manual handler, or an exemption.");
}

/// Recursively require coverage for every leaf of T, prefixing nested
/// paths with the group name (struct-typed fields recurse: those are
/// exactly the option groups — enums, optionals, and arrays are leaves).
fn requireCovered(comptime T: type, comptime prefix: []const u8) void {
    for (@typeInfo(T).@"struct".fields) |f| {
        const key = if (prefix.len == 0) f.name else prefix ++ "." ++ f.name;
        if (@typeInfo(f.type) == .@"struct") {
            requireCovered(f.type, key);
        } else {
            requireLeafCovered(key);
        }
    }
}

comptime {
    @setEvalBranchQuota(20000);
    requireCovered(PostProcessOptions, "");
    // Reverse direction: every listed path must be a real leaf, so a
    // rename/typo fails here instead of silently dropping coverage.
    for (top_clamp_table) |row| {
        if (!@hasField(PostProcessOptions, row.name)) @compileError("top clamp-table row for unknown field: " ++ row.name);
    }
    for (clamp_table) |row| {
        if (!hasOptionPath(PostProcessOptions, row.path)) @compileError("clamp-table row for unknown path: " ++ row.path);
    }
    for (manual_clamp_fields) |m| {
        if (!hasOptionPath(PostProcessOptions, m)) @compileError("manual clamp handler for unknown path: " ++ m);
    }
    for (exempt_clamp_fields) |e| {
        if (!hasOptionPath(PostProcessOptions, e)) @compileError("clamp exemption for unknown path: " ++ e);
    }
    for (frame_disable_flags) |n| {
        if (!hasOptionPath(PostProcessOptions, n)) @compileError("forFrame disable flag for unknown path: " ++ n);
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
