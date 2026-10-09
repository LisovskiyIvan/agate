const std = @import("std");
const testing = std.testing;
const math = @import("math");
const Color4 = math.Color4;
const transition = @import("transition.zig");
const UIStyle = @import("types.zig").UIStyle;
const UIStyleTransition = transition.UIStyleTransition;
const styleEql = transition.styleEql;
const lerpStyle = transition.lerpStyle;
const sample = transition.sample;

fn styleWith(bg: Color4, width: f32, opacity: f32) UIStyle {
    return .{ .background = bg, .border_width = width, .opacity = opacity };
}

test "styleEql is exact field-by-field equality" {
    const a = styleWith(Color4.new(0.1, 0.2, 0.3, 0.4), 2.0, 1.0);
    try testing.expect(styleEql(a, a));
    try testing.expect(styleEql(UIStyle{}, UIStyle{}));

    var b = a;
    b.background.a += 0.01;
    try testing.expect(!styleEql(a, b));
    b = a;
    b.shadow = .{};
    try testing.expect(!styleEql(a, b));
    b.shadow.?.blur = 5.0;
    try testing.expect(!styleEql(a, b));

    var c = a;
    c.gradient = .{ .top = Color4.white, .bottom = Color4.black };
    try testing.expect(!styleEql(a, c));
    var c2 = c;
    c2.gradient.?.bottom = Color4.white;
    try testing.expect(!styleEql(c, c2));
}

test "lerpStyle interpolates scalars and colors" {
    const a = styleWith(Color4.new(0, 0, 0, 1), 0.0, 1.0);
    const b = styleWith(Color4.new(1, 1, 1, 1), 10.0, 0.0);
    const mid = lerpStyle(a, b, 0.5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), mid.background.r, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 5.0), mid.border_width, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), mid.opacity, 1e-5);

    // Endpoints are exact (t=1 returns b's values).
    const end = lerpStyle(a, b, 1.0);
    try testing.expect(styleEql(b, end));

    // Optional gradient cross-fades from the side's background when missing.
    var g = b;
    g.gradient = .{ .top = Color4.new(1, 0, 0, 1), .bottom = Color4.new(0, 0, 1, 1) };
    const gm = lerpStyle(a, g, 0.25);
    try testing.expect(gm.gradient != null);
    // Flat black background fades toward the red top stop.
    try testing.expectApproxEqAbs(@as(f32, 0.25), gm.gradient.?.top.r, 1e-5);

    // Two nulls stay null.
    try testing.expect(lerpStyle(a, b, 0.3).gradient == null);
    try testing.expect(lerpStyle(a, b, 0.3).shadow == null);
    try testing.expect(lerpStyle(a, b, 0.3).accent == null);
}

test "sample: linear progress across time and exact endpoints" {
    const from = styleWith(Color4.new(0, 0, 0, 1), 0.0, 1.0);
    const to = styleWith(Color4.new(1, 0, 0, 1), 8.0, 0.0);
    const entry = UIStyleTransition{
        .used = true,
        .from = from,
        .to = to,
        .started_ms = 100.0,
        .duration_ms = 200.0,
        .easing = .linear,
    };
    // Before start: exact source. After end: exact target.
    try testing.expect(styleEql(from, sample(&entry, 50.0)));
    try testing.expect(styleEql(to, sample(&entry, 300.0)));
    try testing.expect(styleEql(to, sample(&entry, 100000.0)));

    // Halfway through a linear transition: halfway values.
    const mid = sample(&entry, 200.0);
    try testing.expectApproxEqAbs(@as(f32, 0.5), mid.background.r, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 4.0), mid.border_width, 1e-5);
}

test "sample: easing shapes the blend, zero duration snaps" {
    const from = styleWith(Color4.new(0, 0, 0, 1), 0.0, 1.0);
    const to = styleWith(Color4.new(1, 0, 0, 1), 0.0, 1.0);
    const eased = UIStyleTransition{
        .used = true,
        .from = from,
        .to = to,
        .started_ms = 0.0,
        .duration_ms = 100.0,
        .easing = .ease_in_quad,
    };
    // Half the time with ease-in-quad: a quarter of the way.
    const mid = sample(&eased, 50.0);
    try testing.expectApproxEqAbs(@as(f32, 0.25), mid.background.r, 1e-5);

    // Zero duration always reports the target directly.
    const instant = UIStyleTransition{ .used = true, .from = from, .to = to, .duration_ms = 0.0 };
    try testing.expect(styleEql(to, sample(&instant, 0.0)));
    try testing.expect(styleEql(to, sample(&instant, 12345.0)));
}
