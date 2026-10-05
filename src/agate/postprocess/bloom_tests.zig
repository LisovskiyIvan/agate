//! Tests for `postprocess/bloom.zig` (moved verbatim from inline blocks; production code unchanged).
const std = @import("std");
const options = @import("options.zig");
const prod = @import("bloom.zig");
const BloomMipSize = prod.BloomMipSize;
const clampBloomMips = prod.clampBloomMips;
const bloomMipSize = prod.bloomMipSize;
const karisWeight = prod.karisWeight;
const tentWeight1D = prod.tentWeight1D;
const bloomTentWeight = prod.bloomTentWeight;

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
