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

test "bloom mips clamp" {
    try std.testing.expectEqual(@as(u32, 3), clampBloomMips(0));
    try std.testing.expectEqual(@as(u32, 3), clampBloomMips(3));
    try std.testing.expectEqual(@as(u32, 5), clampBloomMips(5));
    try std.testing.expectEqual(@as(u32, 7), clampBloomMips(7));
    try std.testing.expectEqual(@as(u32, 7), clampBloomMips(42));

    var cfg = options.PostProcessOptions{ .bloom_pyramid_mips = 99 };
    try std.testing.expectEqual(@as(u32, 7), cfg.clamped().bloom_pyramid_mips);
    cfg.bloom_pyramid_mips = 1;
    try std.testing.expectEqual(@as(u32, 3), cfg.clamped().bloom_pyramid_mips);
}

test "bloom mip sizes" {
    try std.testing.expectEqual(BloomMipSize{ .w = 640, .h = 360 }, bloomMipSize(1280, 720, 0));
    try std.testing.expectEqual(BloomMipSize{ .w = 320, .h = 180 }, bloomMipSize(1280, 720, 1));
    try std.testing.expectEqual(BloomMipSize{ .w = 40, .h = 22 }, bloomMipSize(1280, 720, 4));
    // Odd dimensions round down but never below 1x1.
    try std.testing.expectEqual(BloomMipSize{ .w = 1, .h = 1 }, bloomMipSize(3, 3, 3));
    const a = bloomMipSize(1920, 1080, 6);
    try std.testing.expect(a.w >= 1 and a.h >= 1);
}

test "tent weights" {
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), tentWeight1D(0.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), tentWeight1D(0.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), tentWeight1D(1.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), tentWeight1D(2.0), 1e-6);
    // 3x3 kernel shape and normalization.
    try std.testing.expectApproxEqAbs(@as(f32, 4.0 / 16.0), bloomTentWeight(0, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 16.0), bloomTentWeight(1, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 16.0), bloomTentWeight(0, -1), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 16.0), bloomTentWeight(1, 1), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), bloomTentWeight(2, 0), 1e-6);
    var sum: f32 = 0.0;
    var ix: i32 = -1;
    while (ix <= 1) : (ix += 1) {
        var iy: i32 = -1;
        while (iy <= 1) : (iy += 1) sum += bloomTentWeight(ix, iy);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-6);
}

test "karis weight" {
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), karisWeight(0.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), karisWeight(1.0), 1e-6);
    // Outliers are suppressed but never reach zero or negative.
    try std.testing.expect(karisWeight(10.0) < karisWeight(1.0));
    try std.testing.expect(karisWeight(10.0) > 0.0);
}
