const std = @import("std");
const types = @import("types.zig");
const options = @import("options.zig");

pub const BloomMipSize = types.BloomMipSize;

// Valid range for the bloom mip pyramid depth (see clampBloomMips).
pub const BLOOM_PYRAMID_MIPS_MIN: u32 = 3;
pub const BLOOM_PYRAMID_MIPS_MAX: u32 = 7;
// Hard capacity of BloomPass target arrays; always >= MAX.
// Sized as usize so passes can index target arrays directly.
pub const BLOOM_MAX_MIPS: usize = 7;

/// Clamp the requested pyramid depth into [3, 7].
pub fn clampBloomMips(mips: u32) u32 {
    return std.math.clamp(mips, BLOOM_PYRAMID_MIPS_MIN, BLOOM_PYRAMID_MIPS_MAX);
}

/// Size of one bloom pyramid level. Level 0 is half resolution, each
/// further level halves again, clamped to 1x1. Mirrors BloomPass.resize.
pub fn bloomMipSize(base_w: i32, base_h: i32, level: u32) BloomMipSize {
    var w: i32 = @max(1, @divTrunc(base_w, 2));
    var h: i32 = @max(1, @divTrunc(base_h, 2));
    var i: u32 = 0;
    while (i < level) : (i += 1) {
        w = @max(1, @divTrunc(w, 2));
        h = @max(1, @divTrunc(h, 2));
    }
    return .{ .w = w, .h = h };
}

/// Karis weighting for firefly suppression: bright outliers contribute
/// less to the downsampled average. Mirrors bloom_down.glsl.
pub fn karisWeight(luma: f32) f32 {
    return 1.0 / (1.0 + @max(luma, 0.0));
}

/// 1D tent (triangle) filter weight. Mirrors the separable form of the
/// 3x3 tent kernel used by bloom_up.glsl.
pub fn tentWeight1D(x: f32) f32 {
    return @max(0.0, 1.0 - @abs(x));
}

/// 3x3 tent kernel weight for integer offsets in [-1, 1], normalized so
/// the kernel sums to 1 (center 4/16, edges 2/16, corners 1/16).
/// Returns 0 for offsets outside the kernel.
pub fn bloomTentWeight(ix: i32, iy: i32) f32 {
    if (ix < -1 or ix > 1 or iy < -1 or iy > 1) return 0.0;
    const ax: f32 = if (ix == 0) 2.0 else 1.0;
    const ay: f32 = if (iy == 0) 2.0 else 1.0;
    return (ax * ay) / 16.0;
}

/// Pass-construction decision (pure; PostFXStack.renderChain gates the GPU
/// passes on this).
pub fn bloomPyramidActive(post_enabled: bool, cfg: options.PostProcessOptions) bool {
    return post_enabled and cfg.bloom_enabled and cfg.bloom_intensity > 0;
}

// Bloom regression tests live in `bloom_tests.zig` (same directory,
// imported below so the test registry picks them up exactly once).

test {
    _ = @import("bloom_tests.zig");
}
