const std = @import("std");
const types = @import("types.zig");
const bloom = @import("bloom.zig");
const options = @import("options.zig");
const glow = @import("glow.zig");

test "glow defaults are off and neutral" {
    const cfg = options.PostProcessOptions{};
    // Default OFF: disabled glow runs zero passes and packs zero uniforms,
    // so the composite stays bit-identical to the pre-glow path.
    try std.testing.expect(!cfg.glow.enabled);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cfg.glow.threshold, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), cfg.glow.intensity, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), cfg.glow.radius, 1e-6);
    try std.testing.expectEqual([3]f32{ 1.0, 1.0, 1.0 }, cfg.glow.tint);
    // Distinct from bloom: threshold above bloom's 0.8, radius wider than
    // bloom's 2.0.
    try std.testing.expect(cfg.glow.threshold > 0.8);
    try std.testing.expect(cfg.glow.radius > 2.0);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, glow.glowParams(cfg));
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, glow.glowTintParams(cfg));
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, glow.glowParams(cfg.clamped()));
    try std.testing.expect(!glow.glowActive(true, cfg));
    try std.testing.expect(!glow.glowActive(false, cfg));
}

test "glow clamped sanitizes ranges like bloom" {
    var cfg = options.PostProcessOptions{
        .glow = .{
            .threshold = -1.0,
            .intensity = -0.5,
            .radius = -4.0,
            .tint = .{ 2.0, -1.0, 0.5 },
        },
    };
    const out = cfg.clamped();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.glow.threshold, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.glow.intensity, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.glow.radius, 1e-6);
    try std.testing.expectEqual([3]f32{ 1.0, 0.0, 0.5 }, out.glow.tint);
    // Bloom's own sanitization is untouched by the glow fields.
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), out.bloom.threshold, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), out.bloom.intensity, 1e-6);
}

test "glow validate rejects non-finite and out-of-range" {
    // Defaults validate clean.
    try glow.validateGlow(options.PostProcessOptions{});
    try glow.validateGlow(options.PostProcessOptions{
        .glow = .{
            .enabled = true,
        },
    });

    // NaN/Inf can never sanitize: hard errors, never silent clamps.
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    try std.testing.expectError(error.InvalidGlowOptions, glow.validateGlow(.{ .glow = .{ .threshold = nan } }));
    try std.testing.expectError(error.InvalidGlowOptions, glow.validateGlow(.{ .glow = .{ .intensity = inf } }));
    try std.testing.expectError(error.InvalidGlowOptions, glow.validateGlow(.{ .glow = .{ .radius = nan } }));
    try std.testing.expectError(error.InvalidGlowOptions, glow.validateGlow(.{ .glow = .{ .tint = .{ 1.0, nan, 1.0 } } }));

    // Finite out-of-range values are rejected here (clamped() is the
    // sanitizing path for those, mirroring bloom).
    try std.testing.expectError(error.InvalidGlowOptions, glow.validateGlow(.{ .glow = .{ .threshold = -0.1 } }));
    try std.testing.expectError(error.InvalidGlowOptions, glow.validateGlow(.{ .glow = .{ .intensity = -1.0 } }));
    try std.testing.expectError(error.InvalidGlowOptions, glow.validateGlow(.{ .glow = .{ .radius = -2.0 } }));
    try std.testing.expectError(error.InvalidGlowOptions, glow.validateGlow(.{ .glow = .{ .tint = .{ 1.0, 1.0, 1.5 } } }));
    try std.testing.expectError(error.InvalidGlowOptions, glow.validateGlow(.{ .glow = .{ .tint = .{ -0.1, 0.0, 0.0 } } }));
}

test "glow extract and gaussian kernel math" {
    // Below threshold extracts black (soft bright-pass).
    try std.testing.expectEqual([3]f32{ 0.0, 0.0, 0.0 }, glow.glowExtract(.{ 0.2, 0.3, 0.4 }, 1.0));
    // At double the threshold the over-threshold fraction is 1/2.
    const hot = glow.glowExtract(.{ 2.0, 2.0, 2.0 }, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), hot[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), hot[1], 1e-5);
    // Black stays black for any threshold (no divide-by-zero).
    try std.testing.expectEqual([3]f32{ 0.0, 0.0, 0.0 }, glow.glowExtract(.{ 0.0, 0.0, 0.0 }, 0.0));

    // Gaussian kernel: center is the peak, symmetric, normalized sum is 1.
    const sigma: f32 = glow.GLOW_RADIUS_DEFAULT;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), glow.glowGaussianWeight(0.0, sigma), 1e-6);
    try std.testing.expect(glow.glowGaussianWeight(1.0, sigma) < 1.0);
    try std.testing.expectApproxEqAbs(
        glow.glowGaussianWeight(2.0, sigma),
        glow.glowGaussianWeight(-2.0, sigma),
        1e-6,
    );
    try std.testing.expect(glow.glowGaussianWeight(4.0, sigma) > 0.0);
    var norm: f32 = 0.0;
    var i: i32 = -glow.GLOW_HALF_TAPS;
    while (i <= glow.GLOW_HALF_TAPS) : (i += 1) {
        norm += glow.glowGaussianWeight(@floatFromInt(i), sigma) / glow.glowKernelSum(sigma);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, 1e-6);
    // Wider radius spreads the weight outward (edge taps gain share).
    const tight = glow.glowGaussianWeight(4.0, 1.0) / glow.glowKernelSum(1.0);
    const wide = glow.glowGaussianWeight(4.0, 8.0) / glow.glowKernelSum(8.0);
    try std.testing.expect(wide > tight);
}

test "glow params and tint packing, independent of bloom" {
    // Enabled packs the clamped intensity and tint.
    const on = options.PostProcessOptions{
        .glow = .{
            .enabled = true,
        },
    };
    try std.testing.expectEqual([4]f32{ 1.0, 0.5, 0.0, 0.0 }, glow.glowParams(on));
    try std.testing.expectEqual([4]f32{ 1.0, 1.0, 1.0, 0.0 }, glow.glowTintParams(on));

    // Out-of-range packs clamped (the shader additionally gates on
    // intensity > 0.001, so zero intensity is a no-op even when enabled).
    var hot = options.PostProcessOptions{
        .glow = .{
            .enabled = true,
            .intensity = 9.0,
            .tint = .{ 3.0, -2.0, 0.25 },
        },
    };
    const hp = glow.glowParams(hot);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), hp[1], 1e-6);
    const ht = glow.glowTintParams(hot);
    try std.testing.expectEqual([4]f32{ 1.0, 0.0, 0.25, 0.0 }, ht);
    hot.glow.intensity = -4.0;
    try std.testing.expectEqual(@as(f32, 0.0), glow.glowParams(hot)[1]);

    // Bloom knobs never leak into the glow uniforms and vice versa: the two
    // effects stay independently toggleable.
    const bloom_only = options.PostProcessOptions{
        .bloom = .{
            .enabled = true,
            .intensity = 0.9,
        },
    };
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, glow.glowParams(bloom_only));
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, glow.glowTintParams(bloom_only));
    try std.testing.expect(bloom.bloomPyramidActive(true, bloom_only));
    try std.testing.expect(!glow.glowActive(true, bloom_only));
}

test "glow and bloom pass decisions are independent" {
    const def = options.PostProcessOptions{};
    // Bloom uses one pyramid implementation; glow remains a separate effect.
    try std.testing.expect(bloom.bloomPyramidActive(true, def));
    try std.testing.expect(!bloom.bloomPyramidActive(false, def));
    try std.testing.expect(!glow.glowActive(true, def));
    // Post off disables both regardless of their own toggles.
    const both_on = options.PostProcessOptions{
        .glow = .{
            .enabled = true,
        },
    };
    try std.testing.expect(!bloom.bloomPyramidActive(false, both_on));
    try std.testing.expect(!glow.glowActive(false, both_on));
    try std.testing.expect(bloom.bloomPyramidActive(true, both_on));
    try std.testing.expect(glow.glowActive(true, both_on));

    // Toggling glow never changes the bloom decision (and vice versa).
    var cfg = options.PostProcessOptions{};
    const bloom_before = bloom.bloomPyramidActive(true, cfg);
    cfg.glow.enabled = true;
    try std.testing.expectEqual(bloom_before, bloom.bloomPyramidActive(true, cfg));
    try std.testing.expect(glow.glowActive(true, cfg));
    cfg.glow.enabled = false;
    try std.testing.expect(!glow.glowActive(true, cfg));
    try std.testing.expectEqual(bloom_before, bloom.bloomPyramidActive(true, cfg));
    cfg.bloom.enabled = false;
    try std.testing.expect(!bloom.bloomPyramidActive(true, cfg));
    try std.testing.expect(!glow.glowActive(true, cfg));

    // One glow frame is exactly extract + H + V draws (ordering constant
    // the renderChain stats pin); the glow target reuses bloom's half-res
    // level-0 sizing.
    try std.testing.expectEqual(@as(u32, 3), glow.GLOW_PASS_DRAWS);
    try std.testing.expectEqual(types.BloomMipSize{ .w = 640, .h = 360 }, bloom.bloomMipSize(1280, 720, 0));
}
