// PCSS (percentage-closer soft shadows) CPU-side math for the cascaded sun shadow atlas.
// GPU-free: pure formulas mirrored by the forward shaders
// (shaders/{standard,pbr,skinned_pbr,instanced}.glsl), which run the same
// blocker search -> penumbra -> PCF pipeline on the GPU. No GPU imports here,
// so `zig build test` covers this file without a graphics context.

// Fixed blocker-search tap count. Must match PCSS_BLOCKER_SAMPLES in the shaders.
pub const blocker_sample_count: u32 = 12;
// Fixed PCF tap counts per cascade. Must match the taps logic in sampleCascade:
// cascade 0 keeps the legacy 16-tap Poisson PCF, cascades 1-3 keep 8 taps.
pub const pcf_taps_near: u32 = 16;
pub const pcf_taps_far: u32 = 8;

// Live defaults for the packed PCSS uniforms (atlas-UV units unless noted).
pub const default_light_size: f32 = 0.02;
pub const default_blocker_radius: f32 = 0.01;
pub const default_min_penumbra: f32 = 0.0005;
pub const default_max_penumbra: f32 = 0.01;

// Sentinel for "no blocker found". The shaders return it from the blocker
// search (blocker_count == 0) and early-out to fully lit (see resolveLit).
pub const no_blockers: f32 = -1.0;

// Guards the (receiver - blocker) / blocker division on both sides.
pub const min_blocker_depth: f32 = 0.0001;

// Thresholds the packed pcss_enabled uniform lane (0.0/1.0) at one half,
// mirroring the `cascade_debug.y > 0.5` shader branch.
pub fn isEnabled(flag: f32) bool {
    return flag > 0.5;
}

// Mirrors the per-cascade PCF tap LOD in sampleCascade.
pub fn pcfTapCount(cascade_idx: u32) u32 {
    return if (cascade_idx == 0) pcf_taps_near else pcf_taps_far;
}

// Normalized weight of one PCF disk tap; the shader divides the lit sum by
// the tap count, so weight * taps == 1. Zero taps weigh nothing (no division).
pub fn tapWeight(taps: u32) f32 {
    if (taps == 0) return 0.0;
    return 1.0 / @as(f32, @floatFromInt(taps));
}

// Averages blocker depths; null when nothing blocked the receiver, in which
// case the shaders skip the PCF entirely and report fully lit.
pub fn averageBlocker(sum: f32, count: u32) ?f32 {
    if (count == 0) return null;
    return sum / @as(f32, @floatFromInt(count));
}

// Variable penumbra from the average blocker depth:
// (d_receiver - d_blocker) * light_size, clamped to
// [min_penumbra, max_penumbra]. Mirrors pcssPenumbraRadius in the shaders.
// For parallel rays (orthographic directional sun), penumbra scales linearly
// with the distance between blocker and receiver without perspective division.
pub fn penumbraRadius(receiver_depth: f32, blocker_avg: f32, light_size: f32, min_penumbra: f32, max_penumbra: f32) f32 {
    const raw = (receiver_depth - blocker_avg) * light_size;
    return @min(@max(raw, min_penumbra), max_penumbra);
}

// Maps the blocker-search outcome to the shadow factor: no blockers means
// fully lit (1.0) without running the PCF, otherwise the PCF result stands.
pub fn resolveLit(blocker_avg: ?f32, pcf_lit: f32) f32 {
    if (blocker_avg != null) return pcf_lit;
    return 1.0;
}

test "penumbraRadius matches the analytic formula" {
    const std = @import("std");
    // (0.6 - 0.5) * 0.02 = 0.002, inside wide clamps: raw value passes through.
    try std.testing.expectApproxEqAbs(@as(f32, 0.002), penumbraRadius(0.6, 0.5, 0.02, 0.0, 1.0), 1e-6);
    // (0.9 - 0.3) * 0.05 = 0.03.
    try std.testing.expectApproxEqAbs(@as(f32, 0.03), penumbraRadius(0.9, 0.3, 0.05, 0.0, 1.0), 1e-6);
}

test "penumbraRadius clamps below min and above max" {
    const std = @import("std");
    // Near-contact: (0.5001 - 0.5) * 0.02 = 2e-6 -> min.
    try std.testing.expectEqual(@as(f32, 0.0005), penumbraRadius(0.5001, 0.5, 0.02, 0.0005, 0.01));
    // Far blocker: (0.9 - 0.1) * 0.02 = 0.016 -> max.
    try std.testing.expectEqual(@as(f32, 0.01), penumbraRadius(0.9, 0.1, 0.02, 0.0005, 0.01));
}

test "penumbraRadius guards zero and negative blocker depth" {
    const std = @import("std");
    try std.testing.expectEqual(@as(f32, 0.01), penumbraRadius(0.5, 0.0, 0.02, 0.0005, 0.01));
    try std.testing.expectEqual(@as(f32, 0.01), penumbraRadius(0.5, -0.3, 0.02, 0.0005, 0.01));
}

test "penumbraRadius floors receiver at or behind blocker to min" {
    const std = @import("std");
    // Coincident surfaces: raw 0.0 -> min.
    try std.testing.expectEqual(@as(f32, 0.0005), penumbraRadius(0.4, 0.4, 0.02, 0.0005, 0.01));
    // Receiver behind the blocker (float noise): raw negative -> min.
    try std.testing.expectEqual(@as(f32, 0.0005), penumbraRadius(0.3, 0.5, 0.02, 0.0005, 0.01));
}

test "averageBlocker is null without blockers" {
    const std = @import("std");
    try std.testing.expect(averageBlocker(0.0, 0) == null);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), averageBlocker(1.5, 3).?, 1e-6);
}

test "averageBlocker handles zero and maximum spread" {
    const std = @import("std");
    // Zero spread: three identical blockers average exactly.
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), averageBlocker(1.2, 3).?, 1e-6);
    // Maximum spread: blockers at the depth range extremes average to mid.
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), averageBlocker(1.0, 2).?, 1e-6);
}

test "tapWeight normalizes the PCF disk" {
    const std = @import("std");
    try std.testing.expectApproxEqAbs(@as(f32, 0.0625), tapWeight(16), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), tapWeight(16) * 16.0, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), tapWeight(8) * 8.0, 1e-6);
    try std.testing.expectEqual(@as(f32, 0.0), tapWeight(0));
}

test "pcfTapCount matches the shader cascade LOD" {
    const std = @import("std");
    try std.testing.expectEqual(@as(u32, 16), pcfTapCount(0));
    try std.testing.expectEqual(@as(u32, 8), pcfTapCount(1));
    try std.testing.expectEqual(@as(u32, 8), pcfTapCount(2));
    try std.testing.expectEqual(@as(u32, 8), pcfTapCount(3));
}

test "resolveLit is fully lit without blockers" {
    const std = @import("std");
    // No blockers: fully lit even if a stale PCF value says shadowed.
    try std.testing.expectEqual(@as(f32, 1.0), resolveLit(null, 0.0));
    try std.testing.expectEqual(@as(f32, 1.0), resolveLit(null, 0.37));
    // Blockers present: the PCF result stands.
    try std.testing.expectEqual(@as(f32, 0.25), resolveLit(0.4, 0.25));
}

test "isEnabled thresholds the packed flag at one half" {
    const std = @import("std");
    try std.testing.expect(!isEnabled(0.0));
    try std.testing.expect(!isEnabled(0.5));
    try std.testing.expect(isEnabled(0.5001));
    try std.testing.expect(isEnabled(1.0));
}
