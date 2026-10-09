const std = @import("std");

// Easing functions for node transform animations.
// All functions map the unit interval onto itself: f(0) = 0, f(1) = 1.
// Inputs outside [0, 1] are clamped before evaluation.
pub const EasingType = enum {
    linear,
    ease_in_quad,
    ease_out_quad,
    ease_in_out_quad,
    ease_in_cubic,
    ease_out_cubic,
    ease_in_out_cubic,
    ease_in_sine,
    ease_out_sine,
    ease_in_out_sine,
};

// Display names for UI dropdowns, in enum order.
pub const easing_names: [10][]const u8 = .{
    "Linear",
    "Ease In Quad",
    "Ease Out Quad",
    "Ease In Out Quad",
    "Ease In Cubic",
    "Ease Out Cubic",
    "Ease In Out Cubic",
    "Ease In Sine",
    "Ease Out Sine",
    "Ease In Out Sine",
};

pub fn easingName(t: EasingType) []const u8 {
    return easing_names[@intFromEnum(t)];
}

pub fn easingFromName(name: []const u8) ?EasingType {
    for (easing_names, 0..) |n, i| {
        if (std.mem.eql(u8, n, name)) return @enumFromInt(i);
    }
    return null;
}

pub fn evaluate(t: EasingType, x: f32) f32 {
    const cx = std.math.clamp(x, 0.0, 1.0);
    return switch (t) {
        .linear => cx,
        .ease_in_quad => cx * cx,
        .ease_out_quad => 1.0 - (1.0 - cx) * (1.0 - cx),
        .ease_in_out_quad => if (cx < 0.5)
            2.0 * cx * cx
        else
            1.0 - (-2.0 * cx + 2.0) * (-2.0 * cx + 2.0) / 2.0,
        .ease_in_cubic => cx * cx * cx,
        .ease_out_cubic => 1.0 - (1.0 - cx) * (1.0 - cx) * (1.0 - cx),
        .ease_in_out_cubic => if (cx < 0.5)
            4.0 * cx * cx * cx
        else
            1.0 - (-2.0 * cx + 2.0) * (-2.0 * cx + 2.0) * (-2.0 * cx + 2.0) / 2.0,
        .ease_in_sine => 1.0 - std.math.cos(cx * std.math.pi / 2.0),
        .ease_out_sine => std.math.sin(cx * std.math.pi / 2.0),
        .ease_in_out_sine => -(std.math.cos(std.math.pi * cx) - 1.0) / 2.0,
    };
}
