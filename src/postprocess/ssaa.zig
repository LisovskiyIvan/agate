const std = @import("std");
const options = @import("options.zig");
const PostProcessOptions = options.PostProcessOptions;

pub const MIN_SSAA_FACTOR: f32 = 1.0;
pub const MAX_SSAA_FACTOR: f32 = 2.0;
pub const DEFAULT_SSAA_FACTOR: f32 = 2.0;

/// Full-Scene Super-Sample Anti-Aliasing (SSAA) options.
/// When enabled, renders the scene and post-processing stack at supersampled resolution
/// (up to 2.0x in each dimension, i.e. 4x pixel count) and resolves it down to display
/// resolution using a multi-tap quadrant box downsampler during composite presentation.
pub const SsaaOptions = struct {
    enabled: bool = false,
    /// Supersample factor along each axis. Clamped to [MIN_SSAA_FACTOR, MAX_SSAA_FACTOR].
    /// 2.0x SSAA renders 4x pixels for pristine edge, specular, and sub-pixel quality.
    factor: f32 = DEFAULT_SSAA_FACTOR,
};

/// True when SSAA is active and requesting supersampling (> 1.0x).
pub fn ssaaActive(opts: SsaaOptions) bool {
    return opts.enabled and opts.factor > MIN_SSAA_FACTOR;
}

/// Clamped supersample factor (1.0..2.0). When disabled, returns 1.0.
pub fn ssaaFactor(opts: SsaaOptions) f32 {
    if (!opts.enabled) return 1.0;
    if (std.math.isNan(opts.factor) or !std.math.isFinite(opts.factor)) return DEFAULT_SSAA_FACTOR;
    return std.math.clamp(opts.factor, MIN_SSAA_FACTOR, MAX_SSAA_FACTOR);
}

/// Effective scene render scale, combining `render_scale` (dynamic downscaling [0.25, 1.0])
/// and `ssaa` (full-scene supersampling [1.0, 2.0]).
/// When SSAA is enabled, the supersampling factor overrides the downscale floor,
/// rendering the HDR main target and scene effects at (width * factor, height * factor).
/// When SSAA is disabled, returns `render_scale` (bit-identical to legacy behavior).
pub fn effectiveRenderScale(opts: PostProcessOptions) f32 {
    if (opts.ssaa.enabled) {
        return ssaaFactor(opts.ssaa);
    }
    return std.math.clamp(opts.render_scale, options.min_render_scale, 1.0);
}

/// Target size for supersampling (scale in [MIN_SSAA_FACTOR, MAX_SSAA_FACTOR]).
pub fn ssaaRenderSize(width: i32, height: i32, scale: f32) options.RenderSize {
    if (width <= 0 or height <= 0) return .{ .w = width, .h = height };
    const s = std.math.clamp(scale, MIN_SSAA_FACTOR, MAX_SSAA_FACTOR);
    const w: i32 = @intFromFloat(@max(1.0, @round(@as(f32, @floatFromInt(width)) * s)));
    const h: i32 = @intFromFloat(@max(1.0, @round(@as(f32, @floatFromInt(height)) * s)));
    return .{ .w = w, .h = h };
}

/// Computes render size for any scale (downscale <= 1.0 or SSAA > 1.0).
pub fn renderSizeForScale(width: i32, height: i32, scale: f32) options.RenderSize {
    if (scale > 1.0) {
        return ssaaRenderSize(width, height, scale);
    }
    return options.scaledRenderSize(width, height, scale);
}

/// Effective render size combining render_scale and SSAA.
pub fn effectiveRenderSize(width: i32, height: i32, opts: PostProcessOptions) options.RenderSize {
    if (opts.ssaa.enabled) {
        return ssaaRenderSize(width, height, ssaaFactor(opts.ssaa));
    }
    return options.scaledRenderSize(width, height, opts.render_scale);
}

/// Whether the composite's scene base fetch downsamples (SSAA multi-tap box) or
/// reads through (bilinear): on iff the scene target size strictly exceeds the
/// output/history size (i.e. scene_w > output_w or scene_h > output_h).
pub fn sceneDownscaleEnabled(output_w: i32, output_h: i32, scene_w: i32, scene_h: i32) bool {
    const ow = if (output_w > 0) output_w else scene_w;
    const oh = if (output_h > 0) output_h else scene_h;
    return scene_w > ow or scene_h > oh;
}

/// UV sub-pixel quadrant offsets for 4-tap downsampling relative to output resolution.
pub fn downsampleOffsets(out_w: f32, out_h: f32) [4][2]f32 {
    const ox = if (out_w > 0) 0.25 / out_w else 0.0;
    const oy = if (out_h > 0) 0.25 / out_h else 0.0;
    return .{
        .{ -ox, -oy },
        .{ ox, -oy },
        .{ -ox, oy },
        .{ ox, oy },
    };
}

/// 4-tap box filter weights (equal quadrant distribution).
pub fn downsampleWeights() [4]f32 {
    return .{ 0.25, 0.25, 0.25, 0.25 };
}

/// Pure CPU evaluation of the composite SSAA 4-tap box filter.
pub fn downsampleFilter(samples: [4][3]f32) [3]f32 {
    return .{
        (samples[0][0] + samples[1][0] + samples[2][0] + samples[3][0]) * 0.25,
        (samples[0][1] + samples[1][1] + samples[2][1] + samples[3][1]) * 0.25,
        (samples[0][2] + samples[1][2] + samples[2][2] + samples[3][2]) * 0.25,
    };
}
