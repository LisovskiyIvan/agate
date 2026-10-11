const std = @import("std");

// Submodule namespaces
pub const types = @import("postprocess/types.zig");
pub const bloom = @import("postprocess/bloom.zig");
pub const glow = @import("postprocess/glow.zig");
pub const highlight = @import("postprocess/highlight.zig");
pub const lut = @import("postprocess/lut.zig");
pub const taa = @import("postprocess/taa.zig");
pub const shafts = @import("postprocess/shafts.zig");
pub const options = @import("postprocess/options.zig");
pub const hdr = @import("postprocess/hdr.zig");
pub const auto_exposure = @import("postprocess/auto_exposure.zig");
pub const depth_pyramid = @import("postprocess/depth_pyramid.zig");
pub const contact_shadows = @import("postprocess/contact_shadows.zig");
pub const ssgi = @import("postprocess/ssgi.zig");

// --- Core Types ---
pub const ShaftResolution = types.ShaftResolution;
pub const PostProcessOptions = options.PostProcessOptions;
pub const min_render_scale = options.min_render_scale;
pub const scaledRenderSize = options.scaledRenderSize;
pub const AutoExposureOptions = auto_exposure.AutoExposureOptions;
pub const AutoExposureState = auto_exposure.AutoExposureState;
pub const LuminanceHistogram = auto_exposure.LuminanceHistogram;
pub const calcLuminance = auto_exposure.calcLuminance;

// --- Bloom ---
pub const BLOOM_MAX_MIPS = bloom.BLOOM_MAX_MIPS;
pub const clampBloomMips = bloom.clampBloomMips;
pub const bloomMipSize = bloom.bloomMipSize;
pub const bloomPyramidActive = bloom.bloomPyramidActive;

// --- Glow ---
pub const GLOW_PASS_DRAWS = glow.GLOW_PASS_DRAWS;
pub const GLOW_THRESHOLD_DEFAULT = glow.GLOW_THRESHOLD_DEFAULT;
pub const GLOW_RADIUS_DEFAULT = glow.GLOW_RADIUS_DEFAULT;
pub const glowActive = glow.glowActive;
pub const glowParams = glow.glowParams;
pub const glowTintParams = glow.glowTintParams;

// --- Highlight ---
pub const highlightActive = highlight.highlightActive;
pub const highlightParams = highlight.highlightParams;

// --- Depth of Field ---

// --- Color Curves & Grading ---

// --- LUT ---
pub const buildIdentityLutStrip = lut.buildIdentityLutStrip;
pub const lutParams = lut.lutParams;

// --- Local Tonemapping ---
pub fn localTonemapParams(cfg: PostProcessOptions) [4]f32 {
    if (!cfg.local_tonemapping_enabled or cfg.local_tonemapping_intensity <= 0.001) {
        return .{ 0.0, 0.0, 0.0, 0.0 };
    }
    return .{
        1.0,
        cfg.local_tonemapping_intensity,
        // Former local_tonemapping_contrast knob (dropped: zero non-default
        // setters): the historical default, hardcoded.
        0.3,
        0.0,
    };
}

// --- SSAO / FXAA composite lanes (named split of the former params4 lane) ---
/// Pack the composite ssao_params vec4: (enabled 1/0, 0, intensity,
/// spare 0). Intensity rides through raw (the shader gates on x before
/// reading z), exactly like the former params4 lane.
pub fn ssaoParams(enabled: bool, intensity: f32) [4]f32 {
    return .{
        if (enabled) 1.0 else 0.0,
        0.0,
        intensity,
        0.0,
    };
}

/// Pack the composite fxaa_params vec4: (enabled 1/0, 0, 0, 0).
pub fn fxaaParams(enabled: bool) [4]f32 {
    return .{ if (enabled) 1.0 else 0.0, 0.0, 0.0, 0.0 };
}

/// Pack the composite ssr_params2 vec4: (steps, 0, 0, 0). Zeros when SSR is
/// off (the shader early-outs on ssr_params.x before reading the steps
/// lane), which keeps the composite identical to the SSR-off path. The
/// steps count used to ride in camera_params.z and now has its own lane.
pub fn ssrStepsParams(cfg: PostProcessOptions) [4]f32 {
    if (!cfg.ssr_enabled) return .{ 0.0, 0.0, 0.0, 0.0 };
    return .{ @floatFromInt(cfg.ssr_steps), 0.0, 0.0, 0.0 };
}

// --- Temporal Anti-Aliasing (TAA) ---
pub const taaJitter = taa.taaJitter;
pub const applyTaaJitterToViewProj = taa.applyTaaJitterToViewProj;
pub const taaReadIndex = taa.taaReadIndex;
pub const taaShouldReset = taa.taaShouldReset;
pub const taaReactiveParams = taa.taaReactiveParams;
pub const taaReactiveMarkersActive = taa.taaReactiveMarkersActive;
pub const taaParams = taa.taaParams;
pub const taaState = taa.taaState;

// --- Volumetric Light Shafts ---
pub const SHAFT_PASS_DRAWS = shafts.SHAFT_PASS_DRAWS;
pub const shaftTargetSize = shafts.shaftTargetSize;
pub const shaftActive = shafts.shaftActive;
pub const shaftParams = shafts.shaftParams;

// --- Depth Pyramid ---
pub const DEPTH_PYRAMID_MAX_MIPS = depth_pyramid.DEPTH_PYRAMID_MAX_MIPS;
pub const depthPyramidMipSize = depth_pyramid.depthPyramidMipSize;
pub const computeMipCount = depth_pyramid.computeMipCount;
pub const depthPyramidActive = depth_pyramid.depthPyramidActive;
// --- Contact Shadows ---
pub const contactShadowParams = contact_shadows.contactShadowParams;
pub const contactShadowStepsParams = contact_shadows.contactShadowStepsParams;
pub const ssgiParams = ssgi.ssgiParams;
pub const ssgiPassActive = ssgi.ssgiPassActive;
pub const ssgiPassParams = ssgi.ssgiPassParams;
pub const ssgiTargetSize = ssgi.ssgiTargetSize;
pub const ssgiTargetBytes = ssgi.ssgiTargetBytes;
pub const ssgiTemporalActive = ssgi.ssgiTemporalActive;
pub const ssgiTemporalBlend = ssgi.ssgiTemporalBlend;
pub const ssgiTemporalShouldReset = ssgi.ssgiTemporalShouldReset;
pub const ssgiTemporalTargetBytes = ssgi.ssgiTemporalTargetBytes;

// --- Analytic Physical Sky (opt-in, Preetham-class) ---
pub const sky = @import("postprocess/sky.zig");
pub const SkyOptions = sky.SkyOptions;

// --- Atmospheric Height Fog ---
