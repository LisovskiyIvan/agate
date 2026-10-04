const std = @import("std");
const hdr = @import("hdr.zig");
const types = @import("types.zig");

// Auto-exposure math and temporal adaptation (pure CPU; headless-safe).
// Evaluates scene luminance through geometric mean or log-luminance histogram
// with percentile trimming, computes target middle-gray exposure within EV bounds,
// and smoothly adapts exposure over time with camera-cut support.

/// Standard ITU-R BT.709 luminance coefficients for linear sRGB/Rec.709.
pub const LUMA_R: f32 = 0.2126;
pub const LUMA_G: f32 = 0.7152;
pub const LUMA_B: f32 = 0.0722;

/// Epsilon floor for luminance to avoid log(0) and division by zero.
pub const LUMA_EPSILON: f32 = 1e-4;

/// Default number of histogram bins over the EV range.
pub const HISTOGRAM_BINS: usize = 64;

/// Configuration parameters for auto-exposure.
pub const AutoExposureOptions = struct {
    enabled: bool = false,
    /// Minimum allowed exposure multiplier (clamping lower limit).
    min_exposure: f32 = 0.01,
    /// Maximum allowed exposure multiplier (clamping upper limit).
    max_exposure: f32 = 16.0,
    /// Calibrated middle-gray target luminance (default 0.18 = 18% gray).
    key_value: f32 = 0.18,
    /// Adaptation speed when adjusting to a brighter scene (in 1/sec).
    speed_up: f32 = 3.0,
    /// Adaptation speed when adjusting to a darker scene (in 1/sec).
    speed_down: f32 = 1.0,
    /// Lower percentile to discard in histogram evaluation (e.g. 0.10 = drop bottom 10%).
    low_percentile: f32 = 0.10,
    /// Upper percentile to discard in histogram evaluation (e.g. 0.90 = drop top 10%).
    high_percentile: f32 = 0.90,
    /// Instant snap to target exposure, bypassing temporal adaptation for this frame.
    camera_cut: bool = false,
};

/// Computes linear luminance (Rec.709) from RGB radiance, sanitizing non-finite values.
pub fn calcLuminance(rgb: [3]f32) f32 {
    const r = @max(hdr.sanitizeFinite(rgb[0]), 0.0);
    const g = @max(hdr.sanitizeFinite(rgb[1]), 0.0);
    const b = @max(hdr.sanitizeFinite(rgb[2]), 0.0);
    return LUMA_R * r + LUMA_G * g + LUMA_B * b;
}

/// Computes geometric mean (log-average luminance) across an RGB radiance buffer,
/// with optional per-sample weights.
pub fn calcGeometricMeanLuminance(buffer: []const [3]f32, weights: ?[]const f32) f32 {
    if (buffer.len == 0) return 0.18;
    var sum_log: f32 = 0.0;
    var total_weight: f32 = 0.0;

    for (buffer, 0..) |rgb, i| {
        const w = if (weights) |ws| (if (i < ws.len) ws[i] else 1.0) else 1.0;
        if (w <= 0.0 or !std.math.isFinite(w)) continue;
        const lum = @max(calcLuminance(rgb), LUMA_EPSILON);
        sum_log += w * @log(lum);
        total_weight += w;
    }

    if (total_weight <= 0.0) return 0.18;
    const avg_log = sum_log / total_weight;
    return @exp(avg_log);
}

/// 64-bin log-luminance histogram over an EV range.
pub const LuminanceHistogram = struct {
    bins: [HISTOGRAM_BINS]f32 = [_]f32{0.0} ** HISTOGRAM_BINS,
    min_ev: f32 = -8.0,
    max_ev: f32 = 8.0,

    pub fn reset(self: *LuminanceHistogram) void {
        @memset(&self.bins, 0.0);
    }

    /// Maps a luminance value into [0, HISTOGRAM_BINS - 1].
    pub fn binForLuminance(self: *const LuminanceHistogram, lum: f32) usize {
        const clamped_lum = @max(lum, LUMA_EPSILON);
        const log2_lum = @log2(clamped_lum);
        const span = self.max_ev - self.min_ev;
        if (span <= 1e-6) return 0;
        const norm = (log2_lum - self.min_ev) / span;
        const clamped_norm = std.math.clamp(norm, 0.0, 0.99999);
        const bin_f = clamped_norm * @as(f32, @floatFromInt(HISTOGRAM_BINS));
        return @intFromFloat(bin_f);
    }

    /// Evaluates the center log2 luminance value of a given bin.
    pub fn binCenterLog2(self: *const LuminanceHistogram, bin_index: usize) f32 {
        const span = self.max_ev - self.min_ev;
        const t = (@as(f32, @floatFromInt(bin_index)) + 0.5) / @as(f32, @floatFromInt(HISTOGRAM_BINS));
        return self.min_ev + t * span;
    }

    /// Adds a single luminance sample with weight to the histogram.
    pub fn addSample(self: *LuminanceHistogram, lum: f32, weight: f32) void {
        if (weight <= 0.0 or !std.math.isFinite(weight)) return;
        const bin = self.binForLuminance(lum);
        self.bins[bin] += weight;
    }

    /// Populates histogram from a radiance buffer.
    pub fn buildFromBuffer(self: *LuminanceHistogram, buffer: []const [3]f32, weights: ?[]const f32) void {
        self.reset();
        for (buffer, 0..) |rgb, i| {
            const w = if (weights) |ws| (if (i < ws.len) ws[i] else 1.0) else 1.0;
            self.addSample(calcLuminance(rgb), w);
        }
    }

    /// Evaluates trimmed log-average luminance ignoring outlier low/high percentiles.
    pub fn evalAverageLuminance(self: *const LuminanceHistogram, low_percentile: f32, high_percentile: f32) f32 {
        var total_weight: f32 = 0.0;
        for (self.bins) |b| total_weight += b;
        if (total_weight <= 0.0) return 0.18;

        const low_p = std.math.clamp(low_percentile, 0.0, 1.0);
        const high_p = std.math.clamp(high_percentile, low_p, 1.0);

        const low_thresh = total_weight * low_p;
        const high_thresh = total_weight * high_p;

        var cum_weight: f32 = 0.0;
        var weighted_sum_log2: f32 = 0.0;
        var included_weight: f32 = 0.0;

        for (self.bins, 0..) |bin_weight, i| {
            if (bin_weight <= 0.0) continue;
            const bin_start = cum_weight;
            const bin_end = cum_weight + bin_weight;
            cum_weight = bin_end;

            // Compute overlapping weight within [low_thresh, high_thresh]
            const overlap_start = @max(bin_start, low_thresh);
            const overlap_end = @min(bin_end, high_thresh);
            if (overlap_end > overlap_start) {
                const w = overlap_end - overlap_start;
                weighted_sum_log2 += w * self.binCenterLog2(i);
                included_weight += w;
            }
        }

        if (included_weight <= 0.0) return 0.18;
        const avg_log2 = weighted_sum_log2 / included_weight;
        return std.math.pow(f32, 2.0, avg_log2);
    }
};

/// Computes target exposure from average scene luminance and calibrated middle gray (key value),
/// clamped within [min_exposure, max_exposure].
pub fn calcTargetExposure(avg_lum: f32, key_value: f32, min_exp: f32, max_exp: f32) f32 {
    const lum = @max(hdr.sanitizeFinite(avg_lum), LUMA_EPSILON);
    const key = @max(hdr.sanitizeFinite(key_value), 0.001);
    const min_e = @max(hdr.sanitizeFinite(min_exp), 0.0);
    const max_e = @max(hdr.sanitizeFinite(max_exp), min_e);

    const target = key / lum;
    return std.math.clamp(target, min_e, max_e);
}

/// Applies exponential eye adaptation over dt between current and target exposure.
pub fn adaptExposure(
    current_exp: f32,
    target_exp: f32,
    dt: f32,
    speed_up: f32,
    speed_down: f32,
    camera_cut: bool,
) f32 {
    const cur = hdr.sanitizeExposure(current_exp);
    const target = hdr.sanitizeExposure(target_exp);
    if (camera_cut or dt <= 0.0 or !std.math.isFinite(dt)) return target;

    const speed = if (target > cur) @max(speed_up, 0.0) else @max(speed_down, 0.0);
    const factor = 1.0 - @exp(-dt * speed);
    const adapted = cur + (target - cur) * factor;
    return hdr.sanitizeExposure(adapted);
}

/// Stateful auto-exposure tracker maintaining temporal history across frames.
pub const AutoExposureState = struct {
    adapted_exposure: f32 = 1.0,
    has_history: bool = false,

    pub fn reset(self: *AutoExposureState) void {
        self.adapted_exposure = 1.0;
        self.has_history = false;
    }

    /// Updates exposure from average luminance with temporal smoothing.
    pub fn update(self: *AutoExposureState, avg_lum: f32, opts: AutoExposureOptions, dt: f32) f32 {
        if (!opts.enabled) {
            self.reset();
            return 1.0;
        }

        const target = calcTargetExposure(avg_lum, opts.key_value, opts.min_exposure, opts.max_exposure);
        const is_first = !self.has_history or opts.camera_cut;
        const adapted = if (is_first) target else adaptExposure(self.adapted_exposure, target, dt, opts.speed_up, opts.speed_down, false);

        self.adapted_exposure = adapted;
        self.has_history = true;
        return adapted;
    }

    /// Evaluates luminance buffer and updates adapted exposure.
    pub fn updateFromBuffer(self: *AutoExposureState, buffer: []const [3]f32, weights: ?[]const f32, opts: AutoExposureOptions, dt: f32) f32 {
        const avg_lum = calcGeometricMeanLuminance(buffer, weights);
        return self.update(avg_lum, opts, dt);
    }

    /// Evaluates histogram with percentile trim and updates adapted exposure.
    pub fn updateFromHistogram(self: *AutoExposureState, hist: *const LuminanceHistogram, opts: AutoExposureOptions, dt: f32) f32 {
        const avg_lum = hist.evalAverageLuminance(opts.low_percentile, opts.high_percentile);
        return self.update(avg_lum, opts, dt);
    }
};

// ===========================================================================
// Tests: CPU golden steps + scene 1/4/16 without clipping gate
// ===========================================================================

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
    // Roadmap Gate: "Гейт: CPU golden steps + сцена 1/4/16 без клипа."
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
