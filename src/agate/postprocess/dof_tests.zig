//! Tests for `postprocess/dof.zig` (moved verbatim from inline blocks; production code unchanged).
const std = @import("std");
const prod = @import("dof.zig");
const DOF_TAPS = prod.DOF_TAPS;
const linearizeDepth = prod.linearizeDepth;
const circleOfConfusion = prod.circleOfConfusion;
const dofTapOffset = prod.dofTapOffset;

test "circle of confusion" {
    // In focus -> no blur.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), circleOfConfusion(10.0, 10.0, 5.0, 8.0), 1e-6);
    // Halfway to the ramp edge -> half of max blur.
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), circleOfConfusion(12.5, 10.0, 5.0, 8.0), 1e-5);
    // Beyond the range -> clamped to max blur (both sides).
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), circleOfConfusion(100.0, 10.0, 5.0, 8.0), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), circleOfConfusion(0.0, 10.0, 5.0, 8.0), 1e-5);
    // Degenerate range never divides by zero.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), circleOfConfusion(10.0, 10.0, 0.0, 8.0), 1e-6);
    try std.testing.expect(circleOfConfusion(11.0, 10.0, 0.0, 8.0) >= 0.0);
}

test "linearize depth" {
    // Near maps to near, far maps to far.
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), linearizeDepth(0.0, 1.0, 100.0), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), linearizeDepth(1.0, 1.0, 100.0), 1e-2);
    // Monotonic in between.
    const a = linearizeDepth(0.5, 0.1, 50.0);
    const b = linearizeDepth(0.9, 0.1, 50.0);
    try std.testing.expect(a < b);
}

test "dof tap offsets" {
    // First tap points along +X with the expected spiral radius.
    const t0 = dofTapOffset(0, DOF_TAPS, 8.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5 / 14.0 * 8.0), t0[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), t0[1], 1e-5);
    // All taps stay inside the gather radius.
    for (0..DOF_TAPS) |i| {
        const t = dofTapOffset(@intCast(i), DOF_TAPS, 8.0);
        try std.testing.expect(@sqrt(t[0] * t[0] + t[1] * t[1]) <= 8.0 + 1e-5);
    }
    // Zero radius collapses every tap to the center pixel.
    const z = dofTapOffset(7, DOF_TAPS, 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), z[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), z[1], 1e-6);
}
