const std = @import("std");

// Submodule namespaces
pub const types = @import("postprocess/types.zig");
pub const bloom = @import("postprocess/bloom.zig");
pub const glow = @import("postprocess/glow.zig");
pub const highlight = @import("postprocess/highlight.zig");
pub const dof = @import("postprocess/dof.zig");
pub const color_curves = @import("postprocess/color_curves.zig");
pub const lut = @import("postprocess/lut.zig");
pub const taa = @import("postprocess/taa.zig");
pub const shafts = @import("postprocess/shafts.zig");
pub const options = @import("postprocess/options.zig");
pub const hdr = @import("postprocess/hdr.zig");
pub const auto_exposure = @import("postprocess/auto_exposure.zig");

// --- Core Types ---
pub const TonemappingType = types.TonemappingType;
pub const LutFormat = types.LutFormat;
pub const ShaftResolution = types.ShaftResolution;
pub const BloomMipSize = types.BloomMipSize;
pub const ShaftTargetSize = types.ShaftTargetSize;
pub const TaaReset = types.TaaReset;
pub const TaaBounds = types.TaaBounds;
pub const LutStripLayout = types.LutStripLayout;
pub const LutStripSample = types.LutStripSample;
pub const PostProcessOptions = options.PostProcessOptions;
pub const AutoExposureOptions = auto_exposure.AutoExposureOptions;
pub const AutoExposureState = auto_exposure.AutoExposureState;
pub const LuminanceHistogram = auto_exposure.LuminanceHistogram;
pub const calcLuminance = auto_exposure.calcLuminance;
pub const calcGeometricMeanLuminance = auto_exposure.calcGeometricMeanLuminance;
pub const calcTargetExposure = auto_exposure.calcTargetExposure;
pub const adaptExposure = auto_exposure.adaptExposure;

// --- Bloom ---
pub const BLOOM_PYRAMID_MIPS_MIN = bloom.BLOOM_PYRAMID_MIPS_MIN;
pub const BLOOM_PYRAMID_MIPS_MAX = bloom.BLOOM_PYRAMID_MIPS_MAX;
pub const BLOOM_MAX_MIPS = bloom.BLOOM_MAX_MIPS;
pub const clampBloomMips = bloom.clampBloomMips;
pub const bloomMipSize = bloom.bloomMipSize;
pub const karisWeight = bloom.karisWeight;
pub const tentWeight1D = bloom.tentWeight1D;
pub const bloomTentWeight = bloom.bloomTentWeight;
pub const bloomPyramidActive = bloom.bloomPyramidActive;

// --- Glow ---
pub const GLOW_BLUR_TAPS = glow.GLOW_BLUR_TAPS;
pub const GLOW_HALF_TAPS = glow.GLOW_HALF_TAPS;
pub const GLOW_PASS_DRAWS = glow.GLOW_PASS_DRAWS;
pub const GLOW_THRESHOLD_DEFAULT = glow.GLOW_THRESHOLD_DEFAULT;
pub const GLOW_INTENSITY_DEFAULT = glow.GLOW_INTENSITY_DEFAULT;
pub const GLOW_RADIUS_DEFAULT = glow.GLOW_RADIUS_DEFAULT;
pub const clampTint = glow.clampTint;
pub const validateGlow = glow.validateGlow;
pub const glowActive = glow.glowActive;
pub const glowExtract = glow.glowExtract;
pub const glowGaussianWeight = glow.glowGaussianWeight;
pub const glowKernelSum = glow.glowKernelSum;
pub const glowParams = glow.glowParams;
pub const glowTintParams = glow.glowTintParams;

// --- Highlight ---
pub const highlightActive = highlight.highlightActive;
pub const highlightParams = highlight.highlightParams;
pub const highlightInnerGlow = highlight.highlightInnerGlow;
pub const highlightComposite = highlight.highlightComposite;

// --- Depth of Field ---
pub const DOF_GOLDEN_ANGLE = dof.DOF_GOLDEN_ANGLE;
pub const DOF_TAPS = dof.DOF_TAPS;
pub const linearizeDepth = dof.linearizeDepth;
pub const circleOfConfusion = dof.circleOfConfusion;
pub const dofTapOffset = dof.dofTapOffset;

// --- Color Curves & Grading ---
pub const rgbLuma = color_curves.rgbLuma;
pub const clampGrade = color_curves.clampGrade;
pub const applyGrade = color_curves.applyGrade;

// --- LUT ---
pub const LUT_SIZE_MIN = lut.LUT_SIZE_MIN;
pub const LUT_SIZE_MAX = lut.LUT_SIZE_MAX;
pub const validLutSize = lut.validLutSize;
pub const lutTextureValid = lut.lutTextureValid;
pub const lutStripLayout = lut.lutStripLayout;
pub const lutStripUv = lut.lutStripUv;
pub const lutSampleUv = lut.lutSampleUv;
pub const applyLutStrip = lut.applyLutStrip;
pub const writeIdentityLutStrip = lut.writeIdentityLutStrip;
pub const buildIdentityLutStrip = lut.buildIdentityLutStrip;
pub const lutParams = lut.lutParams;

// --- Temporal Anti-Aliasing (TAA) ---
pub const TAA_JITTER_PERIOD = taa.TAA_JITTER_PERIOD;
pub const halton = taa.halton;
pub const taaJitter = taa.taaJitter;
pub const applyTaaJitterToViewProj = taa.applyTaaJitterToViewProj;
pub const taaReadIndex = taa.taaReadIndex;
pub const taaWriteIndex = taa.taaWriteIndex;
pub const taaShouldReset = taa.taaShouldReset;
pub const taaNeighborhoodBounds = taa.taaNeighborhoodBounds;
pub const taaNeighborhoodAvg = taa.taaNeighborhoodAvg;
pub const taaClampHistory = taa.taaClampHistory;
pub const taaResolve = taa.taaResolve;
pub const taaApplySharpen = taa.taaApplySharpen;
pub const taaResolvePixel = taa.taaResolvePixel;
pub const taaParams = taa.taaParams;
pub const taaState = taa.taaState;

// --- Volumetric Light Shafts ---
pub const SHAFT_STEPS_MIN = shafts.SHAFT_STEPS_MIN;
pub const SHAFT_STEPS_MAX = shafts.SHAFT_STEPS_MAX;
pub const SHAFT_MARCH_MAX = shafts.SHAFT_MARCH_MAX;
pub const SHAFT_PASS_DRAWS = shafts.SHAFT_PASS_DRAWS;
pub const SHAFT_ANISOTROPY_MAX = shafts.SHAFT_ANISOTROPY_MAX;
pub const SHAFT_BLUR_TAPS = shafts.SHAFT_BLUR_TAPS;
pub const SHAFT_BLUR_HALF_TAPS = shafts.SHAFT_BLUR_HALF_TAPS;
pub const shaftTargetSize = shafts.shaftTargetSize;
pub const shaftActive = shafts.shaftActive;
pub const shaftParams = shafts.shaftParams;
pub const validateShaft = shafts.validateShaft;
pub const hgPhase = shafts.hgPhase;
pub const shaftCascadeIndex = shafts.shaftCascadeIndex;
pub const shaftBilateralWeight = shafts.shaftBilateralWeight;
pub const shaftKernelSum = shafts.shaftKernelSum;

test {
    _ = types;
    _ = bloom;
    _ = glow;
    _ = highlight;
    _ = dof;
    _ = color_curves;
    _ = lut;
    _ = taa;
    _ = shafts;
    _ = options;
    _ = hdr;
    _ = auto_exposure;
}
