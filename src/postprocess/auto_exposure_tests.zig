const std = @import("std");
const hdr = @import("hdr.zig");
const types = @import("types.zig");
const auto_exp = @import("auto_exposure.zig");

const LUMA_R = auto_exp.LUMA_R;
const LUMA_G = auto_exp.LUMA_G;
const LUMA_B = auto_exp.LUMA_B;
const AutoExposureOptions = auto_exp.AutoExposureOptions;
const AutoExposureState = auto_exp.AutoExposureState;
const LuminanceHistogram = auto_exp.LuminanceHistogram;
const calcLuminance = auto_exp.calcLuminance;
const calcGeometricMeanLuminance = auto_exp.calcGeometricMeanLuminance;
const calcTargetExposure = auto_exp.calcTargetExposure;
const adaptExposure = auto_exp.adaptExposure;

test "auto-exposure: Rec.709 luminance golden weights and sanitization" {
    // Pure primaries hit exact Rec.709 weights
    try std.testing.expectApproxEqAbs(LUMA_R, calcLuminance(.{ 1.0, 0.0, 0.0 }), 1e-6);
    try std.testing.expectApproxEqAbs(LUMA_G, calcLuminance(.{ 0.0, 1.0, 0.0 }), 1e-6);
    try std.testing.expectApproxEqAbs(LUMA_B, calcLuminance(.{ 0.0, 0.0, 1.0 }), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), calcLuminance(.{ 1.0, 1.0, 1.0 }), 1e-6);

    // Negative and NaN inputs sanitize to 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), calcLuminance(.{ -1.0, -2.0, -3.0 }), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), calcLuminance(.{ std.math.nan(f32), 0.0, 0.0 }), 1e-6);
}

test "auto-exposure: geometric mean luminance on known distributions" {
    // Uniform samples have identical mean
    const uni = [_][3]f32{ .{ 2.0, 2.0, 2.0 }, .{ 2.0, 2.0, 2.0 }, .{ 2.0, 2.0, 2.0 } };
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), calcGeometricMeanLuminance(&uni, null), 1e-4);

    // Geometric mean of 1.0 and 4.0 is sqrt(1 * 4) = 2.0
    const pair = [_][3]f32{ .{ 1.0, 1.0, 1.0 }, .{ 4.0, 4.0, 4.0 } };
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), calcGeometricMeanLuminance(&pair, null), 1e-4);

    // Empty buffer falls back to 0.18
    try std.testing.expectApproxEqAbs(@as(f32, 0.18), calcGeometricMeanLuminance(&.{}, null), 1e-6);
}

test "auto-exposure: histogram binning and percentile cutoff" {
    var hist = LuminanceHistogram{};

    // Add 10 dark samples (0.01), 80 mid samples (4.0), and 10 blown-out sun samples (1000.0)
    for (0..10) |_| hist.addSample(0.01, 1.0);
    for (0..80) |_| hist.addSample(4.0, 1.0);
    for (0..10) |_| hist.addSample(1000.0, 1.0);

    // With 10% low and 90% high percentiles, the 10 dark and 10 bright samples are trimmed,
    // leaving solely the 4.0 midtone samples (within half-bin discretization error of 64 bins over 16 EV).
    const trimmed_avg = hist.evalAverageLuminance(0.10, 0.90);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), trimmed_avg, 0.4);
}

test "auto-exposure: target exposure middle-gray calibration and EV bounds" {
    const min_e = 0.01;
    const max_e = 16.0;
    const key = 0.18;

    // Normal midtone: target = 0.18 / 1.0 = 0.18
    try std.testing.expectApproxEqAbs(@as(f32, 0.18), calcTargetExposure(1.0, key, min_e, max_e), 1e-4);

    // High brightness: target = 0.18 / 18.0 = 0.01 (min clamp)
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), calcTargetExposure(20.0, key, min_e, max_e), 1e-4);

    // Pitch black: target clamps to max_exposure
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), calcTargetExposure(0.0, key, min_e, max_e), 1e-4);
}

test "auto-exposure: temporal adaptation golden steps and camera cut" {
    // Initial state: starts at 1.0
    var state = AutoExposureState{};
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), state.adapted_exposure, 1e-6);

    const opts = AutoExposureOptions{
        .enabled = true,
        .key_value = 0.18,
        .speed_up = 2.0,
        .speed_down = 2.0,
        .min_exposure = 0.01,
        .max_exposure = 16.0,
    };

    // Frame 1: Scene luminance 0.18 -> target is exactly 1.0. First frame initializes immediately.
    const e1 = state.update(0.18, opts, 0.016);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), e1, 1e-4);
    try std.testing.expect(state.has_history);

    // Frame 2: Scene suddenly jumps to 0.36 (twice as bright -> target = 0.5).
    // Golden step: dt = 0.1s, speed = 2.0 -> factor = 1 - exp(-0.2) = 0.181269
    // Expected adapted = 1.0 + (0.5 - 1.0) * 0.181269 = 0.909365
    const e2 = state.update(0.36, opts, 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.909365), e2, 1e-3);

    // Step across 50 ticks of 0.1s (5 seconds = 10 time constants) -> converges smoothly to target 0.500
    var step: usize = 0;
    while (step < 50) : (step += 1) {
        _ = state.update(0.36, opts, 0.1);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 0.500), state.adapted_exposure, 1e-3);

    // Camera cut: instant jump without lag
    var cut_opts = opts;
    cut_opts.camera_cut = true;
    const e_cut = state.update(1.8, cut_opts, 0.016); // target = 0.18 / 1.8 = 0.1
    try std.testing.expectApproxEqAbs(@as(f32, 0.100), e_cut, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.100), state.adapted_exposure, 1e-4);
}

test "auto-exposure gate: scene 1 / 4 / 16 without clipping" {
    // Golden test: CPU steps + scenes with radiance 1, 4, 16 without clipping.
    // In scenes with average radiance 1.0, 4.0, and 16.0:
    // With auto-exposure, all three scenes adapt to produce the calibrated middle-gray
    // target without highlight blow-out or channel clipping.

    const tonemap_mode = types.TonemappingType.aces;
    const key = 0.18;

    // --- Scene 1: Moderate interior (radiance 1.0) ---
    const rad1: [3]f32 = .{ 1.0, 1.0, 1.0 };
    const exp1 = calcTargetExposure(calcLuminance(rad1), key, 0.001, 16.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.18), exp1, 1e-4);
    const exposed1 = [3]f32{ rad1[0] * exp1, rad1[1] * exp1, rad1[2] * exp1 };
    const tonemapped1 = hdr.tonemap(rad1, exp1, tonemap_mode);

    // --- Scene 4: Bright room / studio (radiance 4.0) ---
    const rad4: [3]f32 = .{ 4.0, 4.0, 4.0 };
    const exp4 = calcTargetExposure(calcLuminance(rad4), key, 0.001, 16.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.045), exp4, 1e-4);
    const exposed4 = [3]f32{ rad4[0] * exp4, rad4[1] * exp4, rad4[2] * exp4 };
    const tonemapped4 = hdr.tonemap(rad4, exp4, tonemap_mode);

    // --- Scene 16: Sunlit exterior (radiance 16.0) ---
    const rad16: [3]f32 = .{ 16.0, 16.0, 16.0 };
    const exp16 = calcTargetExposure(calcLuminance(rad16), key, 0.001, 16.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01125), exp16, 1e-5);
    const exposed16 = [3]f32{ rad16[0] * exp16, rad16[1] * exp16, rad16[2] * exp16 };
    const tonemapped16 = hdr.tonemap(rad16, exp16, tonemap_mode);

    // Verification 1: Exposed radiance is normalized to exactly middle gray (0.18) across all 3 scenes
    try std.testing.expectApproxEqAbs(@as(f32, 0.18), exposed1[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.18), exposed4[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.18), exposed16[0], 1e-4);

    // Verification 2: Tonemapped output is identically preserved across 1, 4, 16 without clipping
    try std.testing.expectApproxEqAbs(tonemapped1[0], tonemapped4[0], 1e-4);
    try std.testing.expectApproxEqAbs(tonemapped1[0], tonemapped16[0], 1e-4);

    // Verification 3: Output is safely in non-clipped midtone range (~0.25 on ACES curve)
    try std.testing.expect(tonemapped16[0] > 0.20 and tonemapped16[0] < 0.35);

    // Verification 4: Contrast against unadapted manual exposure 1.0 (which clips/saturates scene 16 at 0.992)
    const clipped16 = hdr.tonemap(rad16, 1.0, tonemap_mode);
    try std.testing.expect(clipped16[0] > 0.99); // completely blown out / clipped without auto-exposure!
}
