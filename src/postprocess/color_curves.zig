const std = @import("std");

pub fn rgbLuma(c: [3]f32) f32 {
    return c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722;
}

/// Clamp one grade triplet into [-1, 1] per channel.
pub fn clampGrade(v: [3]f32) [3]f32 {
    return .{
        std.math.clamp(v[0], -1.0, 1.0),
        std.math.clamp(v[1], -1.0, 1.0),
        std.math.clamp(v[2], -1.0, 1.0),
    };
}

/// Parametric zone grade. Mirrors applyColorCurves in postprocess.glsl:
/// each lift is weighted by its luminance zone (shadows ramp out by
/// l=0.5, highlights ramp in from l=0.5, midtones peak at l=0.5).
pub fn applyGrade(color: [3]f32, shadows: [3]f32, midtones: [3]f32, highlights: [3]f32) [3]f32 {
    const l = rgbLuma(color);
    const w_s = std.math.clamp(1.0 - l * 2.0, 0.0, 1.0);
    const w_h = std.math.clamp((l - 0.5) * 2.0, 0.0, 1.0);
    const w_m = std.math.clamp(1.0 - @abs(l - 0.5) * 2.0, 0.0, 1.0);
    return .{
        color[0] + shadows[0] * w_s + midtones[0] * w_m + highlights[0] * w_h,
        color[1] + shadows[1] * w_s + midtones[1] * w_m + highlights[1] * w_h,
        color[2] + shadows[2] * w_s + midtones[2] * w_m + highlights[2] * w_h,
    };
}
