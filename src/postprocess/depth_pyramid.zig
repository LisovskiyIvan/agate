const std = @import("std");
const types = @import("types.zig");
const options = @import("options.zig");

// GPU Hierarchical Depth Pyramid (Hi-Z) math and dimension utilities.
// Pure module: headless-safe, no sokol calls.
// Evaluates conservative occluder depth mip levels for hierarchical raymarching
// (SSR / contact shadows), SSAO coarsening, and GPU occlusion culling.

/// Maximum number of downsample levels in the GPU depth pyramid (down to ~1x1).
pub const DEPTH_PYRAMID_MAX_MIPS: usize = 8;

pub const DepthPyramidMipSize = struct {
    w: i32,
    h: i32,
};

/// Computes the dimensions for a given level in the depth pyramid.
/// Level 0 is half of the base resolution, and each subsequent level halves again,
/// clamped to a minimum dimension of 1x1.
pub fn depthPyramidMipSize(base_w: i32, base_h: i32, level: u32) DepthPyramidMipSize {
    var w: i32 = @max(1, @divTrunc(base_w, 2));
    var h: i32 = @max(1, @divTrunc(base_h, 2));
    var i: u32 = 0;
    while (i < level) : (i += 1) {
        w = @max(1, @divTrunc(w, 2));
        h = @max(1, @divTrunc(h, 2));
    }
    return .{ .w = w, .h = h };
}

/// Computes the effective number of mip levels for a base resolution (clamped to DEPTH_PYRAMID_MAX_MIPS).
pub fn computeMipCount(base_w: i32, base_h: i32) u32 {
    if (base_w <= 0 or base_h <= 0) return 0;
    const max_dim = @max(@divTrunc(base_w, 2), @divTrunc(base_h, 2));
    if (max_dim <= 0) return 1;
    var count: u32 = 1;
    var cur = max_dim;
    while (cur > 1 and count < DEPTH_PYRAMID_MAX_MIPS) : (count += 1) {
        cur = @divTrunc(cur, 2);
    }
    return count;
}

/// Conservative max depth reduction of 4 neighboring samples in [0, 1] depth range.
/// For LESS_EQUAL depth test where 1.0 is the far plane, conservative occlusion culling
/// requires the farthest depth in the footprint.
pub fn reduceConservativeDepth(d00: f32, d10: f32, d01: f32, d11: f32) f32 {
    return @max(@max(d00, d10), @max(d01, d11));
}

/// Pass-construction decision (pure; PostFXStack gates the GPU pass on this).
pub fn depthPyramidActive(post_enabled: bool, cfg: options.PostProcessOptions) bool {
    return post_enabled and (cfg.depth_pyramid_enabled or cfg.ssr_enabled);
}
