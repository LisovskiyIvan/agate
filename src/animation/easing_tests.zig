const std = @import("std");
const easing = @import("easing.zig");
const EasingType = easing.EasingType;
const evaluate = easing.evaluate;
const easingName = easing.easingName;
const easingFromName = easing.easingFromName;
const easing_names = easing.easing_names;

test "easing boundaries are exact for all types" {
    const info = @typeInfo(EasingType).@"enum";
    const fields = info.field_names;
    try std.testing.expectEqual(fields.len, easing_names.len);
    inline for (info.field_values) |value| {
        const t: EasingType = @fromBackingInt(value);
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), evaluate(t, 0.0), 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), evaluate(t, 1.0), 1e-6);
    }
}

test "easing clamps inputs outside 0..1" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), evaluate(.linear, -2.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), evaluate(.linear, 2.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), evaluate(.ease_in_quad, -0.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), evaluate(.ease_out_cubic, 1.5), 1e-6);
}

test "easing midpoints match known values" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), evaluate(.ease_in_quad, 0.5), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), evaluate(.ease_out_quad, 0.5), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), evaluate(.ease_in_out_quad, 0.5), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.125), evaluate(.ease_in_cubic, 0.5), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.875), evaluate(.ease_out_cubic, 0.5), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), evaluate(.ease_in_out_cubic, 0.5), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.29289322), evaluate(.ease_in_sine, 0.5), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.70710678), evaluate(.ease_out_sine, 0.5), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), evaluate(.ease_in_out_sine, 0.5), 1e-5);
}

test "easing name table round-trips" {
    try std.testing.expectEqualStrings("Linear", easingName(.linear));
    try std.testing.expectEqualStrings("Ease In Out Sine", easingName(.ease_in_out_sine));
    try std.testing.expectEqual(EasingType.ease_out_quad, easingFromName("Ease Out Quad").?);
    try std.testing.expect(easingFromName("No Such Easing") == null);
    try std.testing.expect(easingFromName("") == null);
}
