const std = @import("std");
const pcss = @import("shadow_pcss.zig");

test "penumbraRadius matches the analytic formula" {
    // (0.6 - 0.5) * 0.02 = 0.002, inside wide clamps: raw value passes through.
    try std.testing.expectApproxEqAbs(@as(f32, 0.002), pcss.penumbraRadius(0.6, 0.5, 0.02, 0.0, 1.0), 1e-6);
    // (0.9 - 0.3) * 0.05 = 0.03.
    try std.testing.expectApproxEqAbs(@as(f32, 0.03), pcss.penumbraRadius(0.9, 0.3, 0.05, 0.0, 1.0), 1e-6);
}

test "penumbraRadius clamps below min and above max" {
    // Near-contact: (0.5001 - 0.5) * 0.02 = 2e-6 -> min.
    try std.testing.expectEqual(@as(f32, 0.0005), pcss.penumbraRadius(0.5001, 0.5, 0.02, 0.0005, 0.01));
    // Far blocker: (0.9 - 0.1) * 0.02 = 0.016 -> max.
    try std.testing.expectEqual(@as(f32, 0.01), pcss.penumbraRadius(0.9, 0.1, 0.02, 0.0005, 0.01));
}

test "penumbraRadius guards zero and negative blocker depth" {
    try std.testing.expectEqual(@as(f32, 0.01), pcss.penumbraRadius(0.5, 0.0, 0.02, 0.0005, 0.01));
    try std.testing.expectEqual(@as(f32, 0.01), pcss.penumbraRadius(0.5, -0.3, 0.02, 0.0005, 0.01));
}

test "penumbraRadius floors receiver at or behind blocker to min" {
    // Coincident surfaces: raw 0.0 -> min.
    try std.testing.expectEqual(@as(f32, 0.0005), pcss.penumbraRadius(0.4, 0.4, 0.02, 0.0005, 0.01));
    // Receiver behind the blocker (float noise): raw negative -> min.
    try std.testing.expectEqual(@as(f32, 0.0005), pcss.penumbraRadius(0.3, 0.5, 0.02, 0.0005, 0.01));
}

test "averageBlocker is null without blockers" {
    try std.testing.expect(pcss.averageBlocker(0.0, 0) == null);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), pcss.averageBlocker(1.5, 3).?, 1e-6);
}

test "averageBlocker handles zero and maximum spread" {
    // Zero spread: three identical blockers average exactly.
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), pcss.averageBlocker(1.2, 3).?, 1e-6);
    // Maximum spread: blockers at the depth range extremes average to mid.
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), pcss.averageBlocker(1.0, 2).?, 1e-6);
}

test "tapWeight normalizes the PCF disk" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0625), pcss.tapWeight(16), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), pcss.tapWeight(16) * 16.0, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), pcss.tapWeight(8) * 8.0, 1e-6);
    try std.testing.expectEqual(@as(f32, 0.0), pcss.tapWeight(0));
}

test "pcfTapCount matches the shader cascade LOD" {
    try std.testing.expectEqual(@as(u32, 16), pcss.pcfTapCount(0));
    try std.testing.expectEqual(@as(u32, 8), pcss.pcfTapCount(1));
    try std.testing.expectEqual(@as(u32, 8), pcss.pcfTapCount(2));
    try std.testing.expectEqual(@as(u32, 8), pcss.pcfTapCount(3));
}

test "resolveLit is fully lit without blockers" {
    // No blockers: fully lit even if a stale PCF value says shadowed.
    try std.testing.expectEqual(@as(f32, 1.0), pcss.resolveLit(null, 0.0));
    try std.testing.expectEqual(@as(f32, 1.0), pcss.resolveLit(null, 0.37));
    // Blockers present: the PCF result stands.
    try std.testing.expectEqual(@as(f32, 0.25), pcss.resolveLit(0.4, 0.25));
}

test "isEnabled thresholds the packed flag at one half" {
    try std.testing.expect(!pcss.isEnabled(0.0));
    try std.testing.expect(!pcss.isEnabled(0.5));
    try std.testing.expect(pcss.isEnabled(0.5001));
    try std.testing.expect(pcss.isEnabled(1.0));
}
