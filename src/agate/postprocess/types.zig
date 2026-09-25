const std = @import("std");

pub const TonemappingType = enum(u32) {
    none = 0,
    aces = 1,
    reinhard = 2,
};

/// LUT texture packing. Only the 2D strip is supported (an N*N x N RGBA8
/// image uploaded with mipmaps=false and CLAMP_TO_EDGE wraps); the enum
/// exists so future packings (3D texture, .cube decode) extend here rather
/// than in PostProcessOptions.
pub const LutFormat = enum(u8) {
    strip_2d = 0,
};

/// Raymarch target resolution: half or quarter of the base size.
pub const ShaftResolution = enum(u8) {
    half = 0,
    quarter = 1,
};

pub const BloomMipSize = struct {
    w: i32,
    h: i32,
};

pub const ShaftTargetSize = struct {
    w: i32,
    h: i32,
};

/// Reset-trigger bundle for one composite. Any set field invalidates history
/// for the frame (the shader then returns current without sampling history).
pub const TaaReset = struct {
    first_frame: bool = false,
    toggled_on: bool = false,
    resized: bool = false,
    camera_cut: bool = false,
    explicit_reset: bool = false,
};

pub const TaaBounds = struct {
    min: [3]f32,
    max: [3]f32,
};

/// Validated 2D strip geometry for one LUT.
pub const LutStripLayout = struct {
    /// Cube edge N (also the strip height).
    size: u32,
    /// Strip pixel width, always size * size.
    width: u32,
    /// Strip pixel height, always size.
    height: u32,
};

/// One manual-trilinear LUT strip lookup: two layer uvs plus the blend
/// weight between them.
pub const LutStripSample = struct {
    /// Strip uv inside layer floor(t) (blue axis).
    uv0: [2]f32,
    /// Strip uv inside the next layer up (same layer when b == 1).
    uv1: [2]f32,
    /// Linear blend weight from uv0's color toward uv1's color.
    blend: f32,
};
