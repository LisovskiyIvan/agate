//! Pure style-transition math: interpolation between two resolved
//! `UIStyle` values, target equality and time sampling. Storage lives on
//! `UICanvas` (one fixed slot table keyed by widget hash); keeping the
//! math here makes the interpolation rules unit-testable without a canvas.

const std = @import("std");

const math = @import("math");
const Color4 = math.Color4;

const easing = @import("../animation/easing.zig");
const types = @import("types.zig");

const UIStyle = types.UIStyle;
const UIGradient = types.UIGradient;
const UIShadow = types.UIShadow;
const TransitionConfig = types.TransitionConfig;

/// Fixed size of the per-canvas transition table. UI screens animate a
/// couple dozen widgets at once; past the cap the stalest entry is recycled.
pub const max_style_transitions = 64;

/// One retained transition per animated widget. The canvas keeps a fixed
/// table of these; `used` marks occupied slots so the table can live in
/// the canvas struct without an indirection.
pub const UIStyleTransition = struct {
    used: bool = false,
    key: u64 = 0,
    from: UIStyle = .{},
    to: UIStyle = .{},
    started_ms: f64 = 0,
    duration_ms: f32 = 0,
    easing: easing.EasingType = .ease_out_quad,
    /// Last frame this entry was touched; used to evict stale entries
    /// when the table is full.
    last_touch_ms: f64 = 0,
};

/// Exact field-by-field equality of two resolved styles. Targets come from
/// the same deterministic cascade every frame, so identical inputs produce
/// bit-identical styles and exact comparison is the right "did anything
/// change" test.
pub fn styleEql(a: UIStyle, b: UIStyle) bool {
    if (!colorEql(a.background, b.background)) return false;
    if (!gradientEql(a.gradient, b.gradient)) return false;
    if (!colorEql(a.border_color, b.border_color)) return false;
    if (a.border_width != b.border_width) return false;
    if (a.corner_radius != b.corner_radius) return false;
    if (a.padding != b.padding) return false;
    if (a.margin != b.margin) return false;
    if (!shadowEql(a.shadow, b.shadow)) return false;
    if (!colorEql(a.text_color, b.text_color)) return false;
    if (!accentEql(a.accent, b.accent)) return false;
    if (a.opacity != b.opacity) return false;
    return true;
}

/// Interpolates every numeric/color component of a style. Optional fields
/// (gradient, shadow, accent) cross-fade component-wise; a missing side is
/// seeded from the present side (flat background stops / fully transparent
/// shadow), so none -> some fades in instead of popping. Two nulls stay
/// null.
pub fn lerpStyle(a: UIStyle, b: UIStyle, t: f32) UIStyle {
    return .{
        .background = Color4.lerp(a.background, b.background, t),
        .gradient = lerpGradient(a.gradient, b.gradient, a.background, b.background, t),
        .border_color = Color4.lerp(a.border_color, b.border_color, t),
        .border_width = lerpF32(a.border_width, b.border_width, t),
        .corner_radius = lerpF32(a.corner_radius, b.corner_radius, t),
        .padding = lerpF32(a.padding, b.padding, t),
        .margin = lerpF32(a.margin, b.margin, t),
        .shadow = lerpShadow(a.shadow, b.shadow, t),
        .text_color = Color4.lerp(a.text_color, b.text_color, t),
        .accent = lerpOptColor(a.accent, b.accent, t),
        .opacity = lerpF32(a.opacity, b.opacity, t),
    };
}

/// Samples a transition at canvas time `now_ms`. Finished transitions
/// return the exact target (no easing overshoot past 1), not-yet-started
/// ones the exact source, so both endpoints are stable values.
pub fn sample(entry: *const UIStyleTransition, now_ms: f64) UIStyle {
    if (entry.duration_ms <= 0.0) return entry.to;
    const elapsed: f64 = now_ms - entry.started_ms;
    if (elapsed <= 0.0) return entry.from;
    const dur: f64 = @floatCast(entry.duration_ms);
    if (elapsed >= dur) return entry.to;
    const t = easing.evaluate(entry.easing, @floatCast(elapsed / dur));
    return lerpStyle(entry.from, entry.to, t);
}

fn lerpF32(a: f32, b: f32, t: f32) f32 {
    return a + (b - a) * t;
}

fn colorEql(a: Color4, b: Color4) bool {
    return a.r == b.r and a.g == b.g and a.b == b.b and a.a == b.a;
}

fn gradientEql(a: ?UIGradient, b: ?UIGradient) bool {
    if (a == null and b == null) return true;
    if (a == null or b == null) return false;
    return colorEql(a.?.top, b.?.top) and colorEql(a.?.bottom, b.?.bottom);
}

fn shadowEql(a: ?UIShadow, b: ?UIShadow) bool {
    if (a == null and b == null) return true;
    if (a == null or b == null) return false;
    return colorEql(a.?.color, b.?.color) and a.?.offset_x == b.?.offset_x and
        a.?.offset_y == b.?.offset_y and a.?.blur == b.?.blur;
}

fn accentEql(a: ?Color4, b: ?Color4) bool {
    if (a == null and b == null) return true;
    if (a == null or b == null) return false;
    return colorEql(a.?, b.?);
}

fn lerpGradient(a: ?UIGradient, b: ?UIGradient, bg_a: Color4, bg_b: Color4, t: f32) ?UIGradient {
    if (a == null and b == null) return null;
    // A missing gradient renders as a flat fill, so it seeds from that
    // side's background color: background <-> gradient cross-fades smoothly.
    const ga = a orelse UIGradient{ .top = bg_a, .bottom = bg_a };
    const gb = b orelse UIGradient{ .top = bg_b, .bottom = bg_b };
    return .{
        .top = Color4.lerp(ga.top, gb.top, t),
        .bottom = Color4.lerp(ga.bottom, gb.bottom, t),
    };
}

fn lerpShadow(a: ?UIShadow, b: ?UIShadow, t: f32) ?UIShadow {
    if (a == null and b == null) return null;
    // A missing shadow is "invisible shadow": keep the present side's
    // geometry but zero alpha, so the shadow fades in/out in place.
    const sa = a orelse faded(b.?);
    const sb = b orelse faded(a.?);
    return .{
        .color = Color4.lerp(sa.color, sb.color, t),
        .offset_x = lerpF32(sa.offset_x, sb.offset_x, t),
        .offset_y = lerpF32(sa.offset_y, sb.offset_y, t),
        .blur = lerpF32(sa.blur, sb.blur, t),
    };
}

fn faded(s: UIShadow) UIShadow {
    return .{ .color = Color4.new(0, 0, 0, 0), .offset_x = s.offset_x, .offset_y = s.offset_y, .blur = s.blur };
}

fn lerpOptColor(a: ?Color4, b: ?Color4, t: f32) ?Color4 {
    if (a == null and b == null) return null;
    // Missing accents fade from fully transparent rather than snapping.
    const ca = a orelse Color4.new(0, 0, 0, 0);
    const cb = b orelse Color4.new(0, 0, 0, 0);
    return Color4.lerp(ca, cb, t);
}

// ----------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------

const testing = std.testing;

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
