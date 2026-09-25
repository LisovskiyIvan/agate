const std = @import("std");

/// Golden angle (radians) used by the DoF spiral gather. Shared with
/// postprocess.glsl applyDoF so CPU tests mirror the shader exactly.
pub const DOF_GOLDEN_ANGLE: f32 = 2.3999632;
pub const DOF_TAPS: u32 = 14;

/// Linearize a [0, 1] depth buffer value. Mirrors linearize helpers in
/// ssao_blur.glsl and postprocess.glsl. Returns far for degenerate input.
pub fn linearizeDepth(raw_depth: f32, near: f32, far: f32) f32 {
    if (far <= near) return far;
    const denom = far - raw_depth * (far - near);
    return (near * far) / @max(denom, 0.0001);
}

/// Circle of confusion in pixels for a linearized view distance.
/// 0 inside the focal plane, ramps to max_blur at focus_range away.
/// Mirrors postprocess.glsl applyDoF exactly (including the epsilon guard).
pub fn circleOfConfusion(depth_linear: f32, focus_distance: f32, focus_range: f32, max_blur: f32) f32 {
    const fr = @max(focus_range, 0.0001);
    const coc = @abs(depth_linear - focus_distance) / fr;
    return @min(coc, 1.0) * @max(max_blur, 0.0);
}

/// One DoF spiral tap offset in pixels for tap `index` of `taps` taps at
/// gather radius `radius_px`. Mirrors postprocess.glsl applyDoF.
pub fn dofTapOffset(index: u32, taps: u32, radius_px: f32) [2]f32 {
    const fi: f32 = @floatFromInt(index);
    const ft: f32 = @floatFromInt(@max(taps, 1));
    const ang = fi * DOF_GOLDEN_ANGLE;
    const rr = (fi + 0.5) / ft * radius_px;
    return .{ @cos(ang) * rr, @sin(ang) * rr };
}

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
