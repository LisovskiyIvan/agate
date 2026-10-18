const std = @import("std");
const types = @import("types.zig");
const glow = @import("glow.zig");
const options = @import("options.zig");
const shafts = @import("shafts.zig");

const ShaftResolution = shafts.ShaftResolution;
const ShaftTargetSize = shafts.ShaftTargetSize;
const SHAFT_STEPS_MIN = shafts.SHAFT_STEPS_MIN;
const SHAFT_STEPS_MAX = shafts.SHAFT_STEPS_MAX;
const SHAFT_ANISOTROPY_MAX = shafts.SHAFT_ANISOTROPY_MAX;
const SHAFT_BLUR_HALF_TAPS = shafts.SHAFT_BLUR_HALF_TAPS;
const shaftTargetSize = shafts.shaftTargetSize;
const shaftActive = shafts.shaftActive;
const shaftParams = shafts.shaftParams;
const validateShaft = shafts.validateShaft;
const hgPhase = shafts.hgPhase;
const shaftCascadeIndex = shafts.shaftCascadeIndex;
const shaftBilateralWeight = shafts.shaftBilateralWeight;
const shaftKernelSum = shafts.shaftKernelSum;

test "shaft defaults are off and bit-identical" {
    const cfg = options.PostProcessOptions{};
    try std.testing.expect(!cfg.shafts.enabled);
    // Off packs zeros: the composite never samples the shaft texture.
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, shaftParams(cfg, true));
    try std.testing.expect(!shaftActive(true, cfg, true));
    try std.testing.expect(!shaftActive(false, options.PostProcessOptions{
        .shafts = .{
            .enabled = true,
        },
    }, true));
    try std.testing.expect(!shaftActive(true, options.PostProcessOptions{
        .shafts = .{
            .enabled = true,
        },
    }, false));
    try std.testing.expect(shaftActive(true, options.PostProcessOptions{
        .shafts = .{
            .enabled = true,
        },
    }, true));
    // A configured effect without a successful CSM-backed result must not
    // sample the scene-view placeholder as if it were shaft radiance.
    try std.testing.expectEqual(
        [4]f32{ 0.0, 0.0, 0.0, 0.0 },
        shaftParams(options.PostProcessOptions{
            .shafts = .{
                .enabled = true,
            },
        }, false),
    );
}

test "shaft config clamped sanitizes ranges" {
    var cfg = options.PostProcessOptions{
        .shafts = .{
            .intensity = -1.0,
            .steps = 100,
            .density = -0.5,
            .anisotropy = 2.0,
            .max_distance = -10.0,
            .blur_sigma = -3.0,
            .edge_sigma = -0.1,
        },
    };
    const out = cfg.clamped();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.shafts.intensity, 1e-6);
    try std.testing.expectEqual(SHAFT_STEPS_MAX, out.shafts.steps);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.shafts.density, 1e-6);
    try std.testing.expectApproxEqAbs(SHAFT_ANISOTROPY_MAX, out.shafts.anisotropy, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.shafts.max_distance, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.shafts.blur_sigma, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.shafts.edge_sigma, 1e-6);
    cfg.shafts.steps = 0;
    try std.testing.expectEqual(SHAFT_STEPS_MIN, cfg.clamped().shafts.steps);
    cfg.shafts.anisotropy = -2.0;
    try std.testing.expectApproxEqAbs(-SHAFT_ANISOTROPY_MAX, cfg.clamped().shafts.anisotropy, 1e-6);
    // Enabled packs the clamped intensity.
    const on = options.PostProcessOptions{
        .shafts = .{
            .enabled = true,
            .intensity = 2.5,
        },
    };
    try std.testing.expectEqual([4]f32{ 1.0, 2.5, 0.0, 0.0 }, shaftParams(on, true));
}

test "shaft validation rejects non-finite and out-of-range" {
    const ok = options.PostProcessOptions{
        .shafts = .{
            .enabled = true,
        },
    };
    try validateShaft(ok);
    var bad = ok;
    bad.shafts.intensity = std.math.nan(f32);
    try std.testing.expectError(error.InvalidShaftOptions, validateShaft(bad));
    bad = ok;
    bad.shafts.anisotropy = 1.0;
    try std.testing.expectError(error.InvalidShaftOptions, validateShaft(bad));
    bad = ok;
    bad.shafts.steps = 3;
    try std.testing.expectError(error.InvalidShaftOptions, validateShaft(bad));
    bad = ok;
    bad.shafts.steps = 33;
    try std.testing.expectError(error.InvalidShaftOptions, validateShaft(bad));
    bad = ok;
    bad.shafts.density = -1.0;
    try std.testing.expectError(error.InvalidShaftOptions, validateShaft(bad));
    bad = ok;
    bad.shafts.edge_sigma = std.math.inf(f32);
    try std.testing.expectError(error.InvalidShaftOptions, validateShaft(bad));
}

test "hg phase golden values" {
    // Isotropic (g = 0): uniform 1/(4π) whatever the angle.
    const iso = 1.0 / (4.0 * std.math.pi);
    try std.testing.expectApproxEqAbs(iso, hgPhase(1.0, 0.0), 1e-6);
    try std.testing.expectApproxEqAbs(iso, hgPhase(-1.0, 0.0), 1e-6);
    try std.testing.expectApproxEqAbs(iso, hgPhase(0.0, 0.0), 1e-6);
    // Forward scattering peaks looking toward the sun.
    const fwd = hgPhase(1.0, 0.4);
    try std.testing.expect(fwd > hgPhase(0.0, 0.4));
    try std.testing.expect(hgPhase(0.0, 0.4) > hgPhase(-1.0, 0.4));
    // Negative g mirrors: back-scatter peak.
    try std.testing.expectApproxEqAbs(fwd, hgPhase(-1.0, -0.4), 1e-5);
    // Spot value: g = 0.4 straight into the sun: (1-0.16)/(4π·0.36^1.5).
    try std.testing.expectApproxEqAbs(@as(f32, 0.3095), fwd, 1e-3);
    // Always finite and positive inside the clamp range.
    var g: f32 = -0.9;
    while (g <= 0.9) : (g += 0.1) {
        const p = hgPhase(0.7, g);
        try std.testing.expect(std.math.isFinite(p) and p > 0.0);
    }
}

test "shaft march preserves Beer-Lambert energy at every step budget" {
    for ([_]f32{ 0.0, 0.05, 1.0 }) |density| {
        const distance: f32 = 20.0;
        const expected = 1.0 - @exp(-density * distance);
        for ([_]u32{ 4, 12, 32 }) |steps| {
            const opacity = shafts.shaftStepOpacity(density, distance / @as(f32, @floatFromInt(steps)));
            var transmit: f32 = 1.0;
            var scatter: f32 = 0.0;
            for (0..steps) |_| {
                scatter += transmit * opacity;
                transmit *= 1.0 - opacity;
            }
            try std.testing.expect(scatter >= 0.0 and scatter <= 1.0 + 1e-6);
            try std.testing.expectApproxEqAbs(expected, scatter, 2e-6);
        }
    }
}

test "shaft cascade index mirrors the forward splits" {
    const splits = [4]f32{ 10.0, 26.0, 65.0, 150.0 };
    try std.testing.expectEqual(@as(usize, 0), shaftCascadeIndex(5.0, splits));
    try std.testing.expectEqual(@as(usize, 0), shaftCascadeIndex(9.99, splits));
    try std.testing.expectEqual(@as(usize, 1), shaftCascadeIndex(10.0, splits));
    try std.testing.expectEqual(@as(usize, 1), shaftCascadeIndex(25.99, splits));
    try std.testing.expectEqual(@as(usize, 2), shaftCascadeIndex(26.0, splits));
    try std.testing.expectEqual(@as(usize, 3), shaftCascadeIndex(65.0, splits));
    try std.testing.expectEqual(@as(usize, 3), shaftCascadeIndex(1000.0, splits));
}

test "shaft target sizes halve and quarter" {
    try std.testing.expectEqual(ShaftTargetSize{ .w = 640, .h = 360 }, shaftTargetSize(1280, 720, .half));
    try std.testing.expectEqual(ShaftTargetSize{ .w = 320, .h = 180 }, shaftTargetSize(1280, 720, .quarter));
    // Quarter of half-res level 0, never below 1x1.
    try std.testing.expectEqual(ShaftTargetSize{ .w = 1, .h = 1 }, shaftTargetSize(3, 3, .quarter));
}

test "shaft bilateral weights gate on depth edges" {
    // Center tap with no depth step is exactly 1.
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), shaftBilateralWeight(0.0, 2.0, 0.0, 0.02), 1e-6);
    // Spatial falloff matches the glow Gaussian at zero depth difference.
    try std.testing.expectApproxEqAbs(glow.glowGaussianWeight(2.0, 2.0), shaftBilateralWeight(2.0, 2.0, 0.0, 0.02), 1e-6);
    // A depth step far beyond the edge sigma kills the tap.
    try std.testing.expect(shaftBilateralWeight(0.0, 2.0, 1.0, 0.02) < 1e-6);
    // Zero edge sigma disables the gate: plain Gaussian everywhere.
    try std.testing.expectApproxEqAbs(glow.glowGaussianWeight(1.0, 2.0), shaftBilateralWeight(1.0, 2.0, 1.0, 0.0), 1e-6);
    // Kernel is normalized the same way (center normalization).
    var sum: f32 = 0.0;
    var i: i32 = -SHAFT_BLUR_HALF_TAPS;
    while (i <= SHAFT_BLUR_HALF_TAPS) : (i += 1) {
        sum += glow.glowGaussianWeight(@floatFromInt(i), 2.0);
    }
    try std.testing.expectApproxEqAbs(sum, shaftKernelSum(2.0), 1e-6);
}
