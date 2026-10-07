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

// DoF regression tests live in `dof_tests.zig` (same directory,
// imported below so the test registry picks them up exactly once).

test {
    _ = @import("dof_tests.zig");
}
