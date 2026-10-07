//! UI styled rendering: the CSS-like cascade (theme default < named class
//! < per-call override, state deltas per layer), the retained per-widget
//! transition storage, and the resolved-style rasterization (rounded rects,
//! gradients, shadows, borders) plus the `drawStyled*` widget entry points.
//!
//! Split out of `ui.zig` (facade). Functions take a generic `canvas: anytype`
//! (a `*UICanvas` in practice) so this module never imports `../ui.zig` —
//! following the `scene/` precedent, subsystems never import the facade;
//! the canvas passes itself. Primitive emission (`drawRect`, `drawText`, ...)
//! is reached through the canvas forwarders (same discipline as `widgets.zig`);
//! the pure value math comes from the sibling leaves directly. `ui.zig`
//! owns the `UICanvas` type and provides thin forwarders (`canvas.drawStyledButton(...)`
//! keeps working exactly as before).

const std = @import("std");

const math = @import("math");
const Color4 = math.Color4;

const ui_types = @import("types.zig");
const ui_theme_mod = @import("theme.zig");
const ui_transition = @import("transition.zig");
const css_parser = @import("css_parser.zig");

const UIState = ui_types.UIState;
const UIStyle = ui_types.UIStyle;
const UIStyleOverride = ui_types.UIStyleOverride;
const UIStyleSet = ui_types.UIStyleSet;
const TransitionOptions = ui_types.TransitionOptions;
const UIStyleKind = ui_types.UIStyleKind;
const UIStyledOptions = ui_types.UIStyledOptions;
const UIStyleRequest = ui_types.UIStyleRequest;
const UIStyleClass = ui_types.UIStyleClass;
const UITheme = ui_theme_mod.UITheme;
const UIStyleTransition = ui_transition.UIStyleTransition;
const mulAlpha = ui_types.mulAlpha;

/// Band height (px) for approximating rounded corner arcs with solid quads.
const corner_band_px: f32 = 2.0;

/// Registers (or replaces) a named style class. Registration order is
/// irrelevant; replacing keeps the original slot.
pub fn setStyleClass(canvas: anytype, name: []const u8, set: UIStyleSet) void {
    for (canvas.style_classes[0..canvas.style_class_count]) |*sc| {
        if (std.mem.eql(u8, sc.name, name)) {
            sc.set = set;
            return;
        }
    }
    if (canvas.style_class_count >= ui_types.max_style_classes) return;
    canvas.style_classes[canvas.style_class_count] = .{ .name = name, .set = set };
    canvas.style_class_count += 1;
}

/// Looks up a registered class by name.
pub fn styleClass(canvas: anytype, name: []const u8) ?UIStyleSet {
    for (canvas.style_classes[0..canvas.style_class_count]) |sc| {
        if (std.mem.eql(u8, sc.name, name)) return sc.set;
    }
    return null;
}

/// Full cascade resolution: theme default for the widget kind, then the
/// named class (if registered), then the per-call override. Each layer's
/// state delta applies within that layer, so e.g. a hover delta from the
/// theme still shows through a class that only sets a border.
pub fn resolveStyle(canvas: anytype, request: UIStyleRequest) UIStyle {
    const set = canvas.theme.setFor(request.kind).*;
    var s = set.resolve(UIStyle{}, request.state);
    if (request.class) |name| {
        if (styleClass(canvas, name)) |cs| s = cs.resolve(s, request.state);
    }
    if (request.override) |o| s = o.apply(s);
    return s;
}

/// `resolveStyle` plus the transition layer: with `opts.anim_key` set,
/// the widget animates from its previously drawn style toward the newly
/// resolved target over the cascade-resolved TransitionOptions. State
/// changes (hover/active/focus/disabled) trigger transitions implicitly —
/// they simply change the resolved target. Without `anim_key` this is
/// exactly resolveStyle (stateless, no per-widget storage).
pub fn resolveAnimatedStyle(canvas: anytype, kind: UIStyleKind, opts: UIStyledOptions) UIStyle {
    const target = resolveStyle(canvas, .{ .kind = kind, .class = opts.class, .override = opts.style, .state = opts.state });
    const key = opts.anim_key orelse return target;
    const cfg = opts.transition orelse cascadeTransition(canvas, kind, opts.class);
    return animateStyle(canvas, ui_types.animKeyHash(key), target, cfg);
}

/// Transition config from the cascade (per-call override already won in
/// resolveAnimatedStyle): a class wins when it defines a duration,
/// otherwise the theme kind's set provides it.
fn cascadeTransition(canvas: anytype, kind: UIStyleKind, class: ?[]const u8) TransitionOptions {
    if (class) |name| {
        if (styleClass(canvas, name)) |cs| {
            if (cs.transition.duration_ms > 0.0) return cs.transition;
        }
    }
    return canvas.theme.setFor(kind).transition;
}

/// Advances (or starts) the retained transition `key` toward `target`
/// and returns the style to draw this frame.
fn animateStyle(canvas: anytype, key: u64, target: UIStyle, cfg: TransitionOptions) UIStyle {
    const now = canvas.style_time_ms;
    if (cfg.duration_ms <= 0.0) {
        // Instant config: snap and release any retained transition.
        if (styleTransitionSlot(canvas, key)) |s| s.used = false;
        return target;
    }
    const s = styleTransitionSlot(canvas, key) orelse acquireStyleSlot(canvas);
    const restart = !s.used or s.key != key or !ui_transition.styleEql(s.to, target);
    if (restart) {
        // Animate from whatever the widget currently shows: sampling the
        // live entry (instead of jumping to the old target) keeps
        // mid-flight retargets jump-free.
        const from = if (s.used and s.key == key) ui_transition.sample(s, now) else target;
        s.* = .{
            .used = true,
            .key = key,
            .from = from,
            .to = target,
            .started_ms = now,
            .duration_ms = cfg.duration_ms,
            .easing = cfg.easing,
            .last_touch_ms = now,
        };
    } else {
        // Same target: keep animating; config edits apply live.
        s.last_touch_ms = now;
        s.duration_ms = cfg.duration_ms;
        s.easing = cfg.easing;
    }
    return ui_transition.sample(s, now);
}

fn styleTransitionSlot(canvas: anytype, key: u64) ?*UIStyleTransition {
    for (&canvas.style_transitions) |*s| {
        if (s.used and s.key == key) return s;
    }
    return null;
}

/// Storage for a new transition: the first free slot, else the least
/// recently touched entry is recycled (UI screens animate few widgets;
/// LRU over a fixed table keeps the canvas allocation-free).
fn acquireStyleSlot(canvas: anytype) *UIStyleTransition {
    var oldest = &canvas.style_transitions[0];
    for (&canvas.style_transitions) |*s| {
        if (!s.used) return s;
        if (s.last_touch_ms < oldest.last_touch_ms) oldest = s;
    }
    return oldest;
}

/// Installs a parsed CSS theme: replaces the kind defaults wholesale and
/// registers/updates every parsed class (per class name, latest parse
/// wins). Diagnostics are the caller's to inspect — a parse with errors
/// still yields a usable partial theme.
pub fn applyCssTheme(canvas: anytype, parsed: css_parser.CssTheme) void {
    canvas.theme = parsed.theme;
    for (parsed.classes) |c| setStyleClass(canvas, c.name, c.set);
}

/// Renders one resolved style: shadow, background (flat or gradient
/// bands) and border. Fully transparent styles draw nothing, so
/// containers can render their style unconditionally.
pub fn drawStyleRect(canvas: anytype, rect: [4]f32, style: UIStyle) void {
    const w = rect[2];
    const h = rect[3];
    if (w <= 0.0 or h <= 0.0) return;
    const op = std.math.clamp(style.opacity, 0.0, 1.0);
    if (op <= 0.001) return;
    const x = rect[0];
    const y = rect[1];

    // Shadow first (behind everything): stacked expanding translucent
    // rounded rects approximate a blur without a dedicated shader pass.
    if (style.shadow) |sh| {
        const layers = 3;
        var j: usize = layers;
        while (j > 0) : (j -= 1) {
            const t = @as(f32, @floatFromInt(j)) / layers; // widest first
            const grow = sh.blur * t;
            const a = sh.color.a * (0.36 - 0.24 * t);
            canvas.drawRectRoundedFill(
                x - grow + sh.offset_x,
                y - grow + sh.offset_y,
                w + 2.0 * grow,
                h + 2.0 * grow,
                style.corner_radius + grow,
                mulAlpha(sh.color, a),
            );
        }
    }

    if (style.gradient) |g| {
        // Vertical two-stop gradient rasterized into fixed bands; a
        // small horizontal overlap hides seams between solid quads.
        const bands = 8;
        const band_h = h / @as(f32, bands);
        var i: usize = 0;
        while (i < bands) : (i += 1) {
            const f0 = @as(f32, @floatFromInt(i)) / @as(f32, bands);
            const fm = (@as(f32, @floatFromInt(i)) + 0.5) / @as(f32, bands);
            const bh = if (i + 1 == bands) band_h else band_h + 0.4;
            canvas.drawRect(x, y + f0 * h, w, bh, mulAlpha(Color4.lerp(g.top, g.bottom, fm), op));
        }
    } else {
        const bg = mulAlpha(style.background, op);
        if (bg.a > 0.001) {
            canvas.drawRectRoundedFill(x, y, w, h, style.corner_radius, bg);
        }
    }

    const bc = mulAlpha(style.border_color, op);
    if (style.border_width > 0.0 and bc.a > 0.001) {
        canvas.drawRectRoundedOutline(x, y, w, h, style.corner_radius, style.border_width, bc);
    }
}

/// Fills a rounded rectangle with solid quads: one middle rect plus thin
/// horizontal bands approximating the corner arcs (2px resolution).
pub fn drawRectRoundedFill(canvas: anytype, x: f32, y: f32, w: f32, h: f32, radius: f32, color: Color4) void {
    if (w <= 0.0 or h <= 0.0 or color.a <= 0.001) return;
    const r = std.math.clamp(radius, 0.0, @min(w * 0.5, h * 0.5));
    if (r <= 0.5) {
        canvas.drawRect(x, y, w, h, color);
        return;
    }
    canvas.drawRect(x, y + r, w, @max(h - 2.0 * r, 0.0), color);
    var t: f32 = 0.0;
    while (t < r) : (t += corner_band_px) {
        const bh = @min(corner_band_px, r - t);
        const dy = r - (t + bh * 0.5);
        const inset = r - @sqrt(@max(r * r - dy * dy, 0.0));
        canvas.drawRect(x + inset, y + t, @max(w - 2.0 * inset, 0.0), bh, color);
        canvas.drawRect(x + inset, y + h - t - bh, @max(w - 2.0 * inset, 0.0), bh, color);
    }
}

/// Strokes a rounded-rectangle border: four straight segments between
/// the corner arcs plus per-band arc segments. The inner edge is
/// concentric with the outer arc (inset by the border width); rows the
/// inner arc does not reach are solid border strips.
pub fn drawRectRoundedOutline(canvas: anytype, x: f32, y: f32, w: f32, h: f32, radius: f32, thickness: f32, color: Color4) void {
    if (w <= 0.0 or h <= 0.0 or thickness <= 0.0 or color.a <= 0.001) return;
    const t = @min(thickness, @min(w * 0.5, h * 0.5));
    const r = std.math.clamp(radius, 0.0, @min(w * 0.5, h * 0.5));
    if (r <= 0.5) {
        canvas.drawRectOutline(x, y, w, h, t, color);
        return;
    }
    canvas.drawRect(x + r, y, @max(w - 2.0 * r, 0.0), t, color); // top
    canvas.drawRect(x + r, y + h - t, @max(w - 2.0 * r, 0.0), t, color); // bottom
    canvas.drawRect(x, y + r, t, @max(h - 2.0 * r, 0.0), color); // left
    canvas.drawRect(x + w - t, y + r, t, @max(h - 2.0 * r, 0.0), color); // right

    const ri = @max(r - t, 0.0);
    var tc: f32 = 0.0;
    while (tc < r) : (tc += corner_band_px) {
        const bh = @min(corner_band_px, r - tc);
        const d = r - (tc + bh * 0.5); // band center below the arc center
        const o = r - @sqrt(@max(r * r - d * d, 0.0)); // outer edge inset
        const inner: f32 = if (d < ri) r - @sqrt(@max(ri * ri - d * d, 0.0)) else 0.0;
        const y_top = y + tc;
        const y_bot = y + h - tc - bh;
        if (inner <= o + 0.01) {
            // Border wider than the arc at this row: solid strip.
            const sw = @max(w - 2.0 * o, 0.0);
            canvas.drawRect(x + o, y_top, sw, bh, color);
            canvas.drawRect(x + o, y_bot, sw, bh, color);
        } else {
            const bw = inner - o;
            canvas.drawRect(x + o, y_top, bw, bh, color);
            canvas.drawRect(x + w - inner, y_top, bw, bh, color);
            canvas.drawRect(x + o, y_bot, bw, bh, color);
            canvas.drawRect(x + w - inner, y_bot, bw, bh, color);
        }
    }
}

/// Styled panel: renders the resolved panel style into `rect`.
pub fn drawStyledPanel(canvas: anytype, rect: [4]f32, opts: UIStyledOptions) void {
    const s = resolveAnimatedStyle(canvas, .panel, opts);
    drawStyleRect(canvas, rect, s);
}

/// Styled button: resolved button style plus centered outlined text.
/// Pairs with LayoutStack: `const r = ls.place(w, h); canvas.drawStyledButton("Ok", r, 14, .{});`
pub fn drawStyledButton(canvas: anytype, text: []const u8, rect: [4]f32, font_size: f32, opts: UIStyledOptions) void {
    const s = resolveAnimatedStyle(canvas, .button, opts);
    drawStyleRect(canvas, rect, s);
    const op = std.math.clamp(s.opacity, 0.0, 1.0);
    const text_w = @as(f32, @floatFromInt(text.len)) * font_size * 0.5;
    const tx = rect[0] + (rect[2] - text_w) * 0.5;
    const ty = rect[1] + (rect[3] - font_size) * 0.5;
    canvas.drawTextWithOutline(text, tx, ty, font_size, mulAlpha(s.text_color, op), 0.16);
}

/// Styled checkbox: box from the resolved style (state via `opts`),
/// white check mark as in the legacy widget, optional label to the right.
pub fn drawStyledCheckbox(canvas: anytype, rect: [4]f32, checked: bool, label: ?[]const u8, label_size: f32, opts: UIStyledOptions) void {
    const s = resolveAnimatedStyle(canvas, .checkbox, opts);
    const op = std.math.clamp(s.opacity, 0.0, 1.0);
    const size = @min(rect[2], rect[3]);
    drawStyleRect(canvas, .{ rect[0], rect[1], size, size }, s);
    if (checked) {
        const m = size * 0.25;
        canvas.drawRect(rect[0] + m, rect[1] + m, size - 2.0 * m, size - 2.0 * m, mulAlpha(Color4.white, op));
    }
    if (label) |text| {
        const ty = rect[1] + (size - label_size) * 0.5;
        canvas.drawText(text, rect[0] + size + 8.0, ty, label_size, mulAlpha(s.text_color, op));
    }
}

/// Styled horizontal slider. Track from the resolved style, fill from
/// the accent (style override wins over the theme accent). Returns the
/// clamped value like the legacy widget.
pub fn drawStyledSlider(canvas: anytype, rect: [4]f32, value: f32, opts: UIStyledOptions) f32 {
    const s = resolveAnimatedStyle(canvas, .slider, opts);
    const v = std.math.clamp(value, 0.0, 1.0);
    const op = std.math.clamp(s.opacity, 0.0, 1.0);
    drawStyleRect(canvas, rect, s);
    if (v > 0.001) {
        drawRectRoundedFill(canvas, rect[0], rect[1], rect[2] * v, rect[3], s.corner_radius, mulAlpha(s.accent orelse canvas.theme.accent, op));
    }
    // Knob: small square centered on the fill edge (legacy look).
    const knob_size = @max(rect[3] + 6.0, 10.0);
    const cx = rect[0] + v * rect[2];
    const kx = if (rect[2] <= knob_size)
        rect[0] + (rect[2] - knob_size) * 0.5
    else
        std.math.clamp(cx - knob_size * 0.5, rect[0], rect[0] + rect[2] - knob_size);
    const ky = rect[1] + rect[3] * 0.5 - knob_size * 0.5;
    canvas.drawPanel(kx, ky, knob_size, knob_size, mulAlpha(Color4.new(0.52, 0.60, 0.72, 1.0), op), Color4.new(0.3, 0.36, 0.46, 0.9), 1.5);
    return v;
}

/// Styled badge: resolved style plus text at the badge padding. The
/// rect is caller-provided (measure the text and use LayoutStack.place).
pub fn drawStyledBadge(canvas: anytype, text: []const u8, rect: [4]f32, font_size: f32, opts: UIStyledOptions) void {
    const s = resolveAnimatedStyle(canvas, .badge, opts);
    drawStyleRect(canvas, rect, s);
    const op = std.math.clamp(s.opacity, 0.0, 1.0);
    canvas.drawText(text, rect[0] + font_size * 0.4, rect[1] + font_size * 0.25, font_size, mulAlpha(s.text_color, op));
}

// ---------------------------------------------------------------------------
// Test helpers + moved unit tests (same test names, same assertions; only the
// callee paths changed). The canvas is constructed inline per test through a
// block-scoped import of the owner type (same discipline as `profiler/`
// leaves reaching `core.Profiler`); the helpers below stay generic over the
// canvas shape so no leaf-to-owner edge exists at file scope.
// ---------------------------------------------------------------------------

/// Frees the CPU-side quad buffers of a headless test canvas (no GPU state:
/// the full `deinit` would touch the undefined sokol handles).
fn freeTestCanvas(canvas: anytype) void {
    canvas.vertices.deinit(canvas.allocator);
    canvas.indices.deinit(canvas.allocator);
}

fn quadCount(canvas: anytype) usize {
    return canvas.vertices.items.len / 4;
}

/// Exact color equality for test assertions (transition.styleEql compares
/// whole styles, these tests assert single color fields).
fn testColorEql(a: Color4, b: Color4) bool {
    return a.r == b.r and a.g == b.g and a.b == b.b and a.a == b.a;
}

fn usedTransitionSlots(canvas: anytype) usize {
    var n: usize = 0;
    for (&canvas.style_transitions) |*s| {
        if (s.used) n += 1;
    }
    return n;
}

test "style cascade: override beats class beats theme default" {
    const UICanvas = @import("canvas.zig").UICanvas;
    const t = std.testing;
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);
    const theme_bg = Color4.new(0.14, 0.18, 0.25, 0.85);
    const class_bg = Color4.new(0.9, 0.1, 0.1, 0.5);
    const call_bg = Color4.new(0.0, 1.0, 0.0, 1.0);

    // Theme default only.
    var s = canvas.resolveStyle(.{ .kind = .button });
    try t.expectApproxEqAbs(theme_bg.r, s.background.r, 1e-5);
    try t.expectApproxEqAbs(theme_bg.a, s.background.a, 1e-5);

    // Class overrides background, theme-only fields still inherit.
    canvas.setStyleClass("danger", .{ .normal = .{ .background = class_bg, .border_width = 3.0 } });
    s = canvas.resolveStyle(.{ .kind = .button, .class = "danger" });
    try t.expectApproxEqAbs(class_bg.r, s.background.r, 1e-5);
    try t.expectApproxEqAbs(@as(f32, 3.0), s.border_width, 1e-5);
    try t.expectApproxEqAbs(@as(f32, 4.0), s.corner_radius, 1e-5); // theme radius kept

    // Per-call override wins over the class.
    s = canvas.resolveStyle(.{ .kind = .button, .class = "danger", .override = .{ .background = call_bg } });
    try t.expectApproxEqAbs(call_bg.r, s.background.r, 1e-5);
    try t.expectApproxEqAbs(@as(f32, 3.0), s.border_width, 1e-5);

    // Unknown class resolves to the theme only.
    s = canvas.resolveStyle(.{ .kind = .button, .class = "missing" });
    try t.expectApproxEqAbs(theme_bg.r, s.background.r, 1e-5);

    // Registering the same name replaces the class.
    canvas.setStyleClass("danger", .{ .normal = .{ .background = call_bg } });
    s = canvas.resolveStyle(.{ .kind = .button, .class = "danger" });
    try t.expectApproxEqAbs(call_bg.r, s.background.r, 1e-5);
}

test "style state deltas compose across cascade layers" {
    const UICanvas = @import("canvas.zig").UICanvas;
    const t = std.testing;
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);
    const theme_hover_bg = Color4.new(0.24, 0.32, 0.44, 0.92);
    const theme_active_bg = Color4.new(0.18, 0.42, 0.78, 0.95);

    // Theme hover delta applies on top of the theme normal.
    var s = canvas.resolveStyle(.{ .kind = .button, .state = .hover });
    try t.expectApproxEqAbs(theme_hover_bg.r, s.background.r, 1e-5);
    s = canvas.resolveStyle(.{ .kind = .button, .state = .active });
    try t.expectApproxEqAbs(theme_active_bg.r, s.background.r, 1e-5);

    // A class without a hover delta inherits the theme hover...
    canvas.setStyleClass("border_only", .{ .normal = .{ .border_width = 2.0 } });
    s = canvas.resolveStyle(.{ .kind = .button, .class = "border_only", .state = .hover });
    try t.expectApproxEqAbs(theme_hover_bg.r, s.background.r, 1e-5);
    try t.expectApproxEqAbs(@as(f32, 2.0), s.border_width, 1e-5);

    // ...while a class hover delta wins over the theme hover.
    const class_hover = Color4.new(0.5, 0.0, 0.5, 1.0);
    canvas.setStyleClass("purple", .{ .hover = .{ .background = class_hover } });
    s = canvas.resolveStyle(.{ .kind = .button, .class = "purple", .state = .hover });
    try t.expectApproxEqAbs(class_hover.r, s.background.r, 1e-5);
    // Per-call override beats every state delta (CSS inline-style semantics).
    const call_bg = Color4.new(0.0, 1.0, 0.0, 1.0);
    s = canvas.resolveStyle(.{ .kind = .button, .class = "purple", .state = .hover, .override = .{ .background = call_bg } });
    try t.expectApproxEqAbs(call_bg.r, s.background.r, 1e-5);
}

test "UIState.fromFlags and disabled opacity" {
    const UICanvas = @import("canvas.zig").UICanvas;
    const t = std.testing;
    try t.expectEqual(UIState.hover, UIState.fromFlags(true, false, false, false));
    try t.expectEqual(UIState.active, UIState.fromFlags(true, true, false, false));
    try t.expectEqual(UIState.focus, UIState.fromFlags(true, false, true, false));
    try t.expectEqual(UIState.disabled, UIState.fromFlags(true, true, true, true));
    try t.expectEqual(UIState.normal, UIState.fromFlags(false, false, false, false));

    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);
    const s = canvas.resolveStyle(.{ .kind = .button, .state = .disabled });
    try t.expectApproxEqAbs(@as(f32, 0.45), s.opacity, 1e-5);
    // Inherit: non-disabled states keep full opacity.
    const n = canvas.resolveStyle(.{ .kind = .button });
    try t.expectApproxEqAbs(@as(f32, 1.0), n.opacity, 1e-5);
}

test "corner radius, gradient, shadow and opacity change emitted geometry" {
    const UICanvas = @import("canvas.zig").UICanvas;
    const t = std.testing;
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);
    const red = Color4.new(1, 0, 0, 1);

    // Square background: one quad.
    canvas.begin();
    canvas.drawStyleRect(.{ 0, 0, 100, 50 }, .{ .background = red });
    try t.expectEqual(@as(usize, 1), quadCount(&canvas));

    // Radius 8 with 2px corner bands: middle + 4 top + 4 bottom bands.
    canvas.begin();
    canvas.drawStyleRect(.{ 0, 0, 100, 50 }, .{ .background = red, .corner_radius = 8 });
    try t.expectEqual(@as(usize, 9), quadCount(&canvas));
    // All vertices stay inside the rect.
    for (canvas.vertices.items) |v| {
        try t.expect(v.position[0] >= 0.0 and v.position[0] <= 100.0);
        try t.expect(v.position[1] >= 0.0 and v.position[1] <= 50.0);
    }
    // The top corner band is inset: nothing is drawn in the outermost
    // corner square (radius 8, band centers leave the corners empty).
    var min_x_near_top: f32 = 100.0;
    for (canvas.vertices.items) |v| {
        if (v.position[1] < 2.0) min_x_near_top = @min(min_x_near_top, v.position[0]);
    }
    try t.expect(min_x_near_top > 3.0); // sqrt(64-49) ~= 3.87 inset
    try t.expect(min_x_near_top < 8.0);

    // Gradient rasterizes into 8 bands.
    canvas.begin();
    canvas.drawStyleRect(.{ 0, 0, 100, 50 }, .{ .gradient = .{ .top = Color4.white, .bottom = Color4.black } });
    try t.expectEqual(@as(usize, 8), quadCount(&canvas));

    // Shadow adds 3 stacked expanding layers behind the fill (each layer is
    // itself a banded rounded fill, so far more than 4 quads total).
    canvas.begin();
    canvas.drawStyleRect(.{ 0, 0, 100, 50 }, .{ .background = red, .shadow = .{} });
    try t.expect(quadCount(&canvas) > 4);

    // Square border: fill + 4 outline segments.
    canvas.begin();
    canvas.drawStyleRect(.{ 0, 0, 100, 50 }, .{ .background = red, .border_color = Color4.new(0, 0, 1, 1), .border_width = 2 });
    try t.expectEqual(@as(usize, 5), quadCount(&canvas));

    // Rounded border emits bands for the arcs (fill 9 + arcs + 4 straight).
    canvas.begin();
    canvas.drawStyleRect(.{ 0, 0, 100, 50 }, .{
        .background = red,
        .border_color = Color4.new(0, 0, 1, 1),
        .border_width = 2,
        .corner_radius = 8,
    });
    try t.expect(quadCount(&canvas) > 9 + 4);

    // Opacity scales the emitted alpha.
    canvas.begin();
    canvas.drawStyleRect(.{ 0, 0, 10, 10 }, .{ .background = Color4.new(1, 0, 0, 0.8), .opacity = 0.5 });
    try t.expectApproxEqAbs(@as(f32, 0.4), canvas.vertices.items[0].color[3], 1e-5);
}

test "container style class draws background and drives padding" {
    const UICanvas = @import("canvas.zig").UICanvas;
    const PinnedStack = @import("stack.zig").LayoutStack(UICanvas);
    const t = std.testing;
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);
    canvas.setStyleClass("padded", .{ .normal = .{
        .padding = 10.0,
        .background = Color4.new(0.2, 0.2, 0.3, 0.8),
    } });

    var ls = PinnedStack.init(&canvas);
    defer ls.reset();
    canvas.begin();
    try t.expect(ls.beginVStack(.{ 0, 0, 100, 100 }, .{ .class = "padded" }));
    const r = ls.place(50, 20);
    try t.expectApproxEqAbs(@as(f32, 10), r[0], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 10), r[1], 1e-5);
    // The container drew its styled background (1 quad for the square bg).
    try t.expect(quadCount(&canvas) >= 1);
    ls.end();

    // The larger of layout padding and style padding wins.
    try t.expect(ls.beginVStack(.{ 0, 0, 100, 100 }, .{ .class = "padded", .padding = 20 }));
    const r2 = ls.place(50, 20);
    try t.expectApproxEqAbs(@as(f32, 20), r2[0], 1e-5);
    ls.end();
}

test "styled widgets emit geometry and resolve their state styles" {
    const UICanvas = @import("canvas.zig").UICanvas;
    const t = std.testing;
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);

    // Button: styled background (radius 4 bands) + per-glyph text quads.
    canvas.begin();
    canvas.drawStyledButton("Ok", .{ 0, 0, 80, 24 }, 14, .{ .state = .hover });
    try t.expect(quadCount(&canvas) >= 7);

    // Panel with default theme style: radius-6 fill bands plus the theme
    // border (straight segments and arc bands).
    canvas.begin();
    canvas.drawStyledPanel(.{ 0, 0, 100, 100 }, .{});
    try t.expect(quadCount(&canvas) > 8);

    // Slider clamps and draws track/fill/knob.
    canvas.begin();
    const v = canvas.drawStyledSlider(.{ 0, 0, 100, 20 }, 2.0, .{});
    try t.expectApproxEqAbs(@as(f32, 1.0), v, 1e-5);
    try t.expect(quadCount(&canvas) >= 3);

    // Checkbox with label, badge with text.
    canvas.begin();
    canvas.drawStyledCheckbox(.{ 0, 0, 20, 20 }, true, "label", 12, .{});
    try t.expect(quadCount(&canvas) >= 3);
    canvas.begin();
    canvas.drawStyledBadge("FPS", .{ 0, 0, 40, 20 }, 12, .{});
    try t.expect(quadCount(&canvas) >= 2);

    // Styled widgets honor per-call overrides: with background and border
    // stripped, only the per-glyph text quads remain.
    canvas.begin();
    canvas.drawStyledButton("Hi", .{ 0, 0, 80, 24 }, 14, .{
        .style = .{ .background = Color4.transparent, .border_width = 0 },
    });
    try t.expectEqual(@as(usize, 2), quadCount(&canvas));
}

test "applyCssTheme flows parsed theme and classes through the cascade" {
    const UICanvas = @import("canvas.zig").UICanvas;
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);

    const parsed = try css_parser.parseCss(arena.allocator(),
        \\theme { accent: #7cb3ff; }
        \\button { background: #141925d9; }
        \\.danger { background: #b3261e; }
        \\.danger:disabled { opacity: 0.4; }
        \\text_input:focus { border_color: #66bfff; }
    );
    try t.expectEqual(@as(usize, 0), parsed.diags.len);
    canvas.applyCssTheme(parsed);

    // Parsed kind slot replaces the theme default for `button`...
    const btn = canvas.resolveStyle(.{ .kind = .button });
    try t.expectApproxEqAbs(@as(f32, 0x14) / 255.0, btn.background.r, 1e-5);
    try t.expectApproxEqAbs(@as(f32, 0xd9) / 255.0, btn.background.a, 1e-5);
    // ...while untouched kinds keep the built-in defaults.
    try t.expectApproxEqAbs(@as(f32, 6.0), canvas.resolveStyle(.{ .kind = .panel }).corner_radius, 1e-5);
    // New kinds resolve too (theme kind switch covers all slots).
    const ti = canvas.resolveStyle(.{ .kind = .text_input, .state = .focus });
    try t.expectApproxEqAbs(@as(f32, 0x66) / 255.0, ti.border_color.r, 1e-5);

    // Parsed classes participate in the cascade with state deltas.
    const danger = canvas.resolveStyle(.{ .kind = .button, .class = "danger", .state = .disabled });
    try t.expectApproxEqAbs(@as(f32, 0xb3) / 255.0, danger.background.r, 1e-5);
    try t.expectApproxEqAbs(@as(f32, 0.4), danger.opacity, 1e-5);
    // Parsed theme accent is the slider fill fallback.
    try t.expectApproxEqAbs(@as(f32, 0x7c) / 255.0, canvas.theme.accent.r, 1e-5);
}

test "style transitions interpolate toward the target over time" {
    const UICanvas = @import("canvas.zig").UICanvas;
    const t = std.testing;
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);
    canvas.frame_dt_ms = 100;
    const normal_bg = Color4.new(0.14, 0.18, 0.25, 0.85);
    const hover_bg = Color4.new(0.24, 0.32, 0.44, 0.92);
    const opts = UIStyledOptions{
        .anim_key = "play_btn",
        .transition = .{ .duration_ms = 200, .easing = .linear },
    };

    canvas.begin(); // t=100: first sight, fresh entry starts exactly on target
    var s = canvas.resolveAnimatedStyle(.button, opts);
    try t.expect(testColorEql(normal_bg, s.background));

    canvas.begin(); // t=200: hover retargets; at elapsed 0 the drawn style is still normal
    s = canvas.resolveAnimatedStyle(.button, .{ .anim_key = "play_btn", .transition = .{ .duration_ms = 200, .easing = .linear }, .state = .hover });
    try t.expect(testColorEql(normal_bg, s.background));

    canvas.begin(); // t=300: halfway through a linear 200ms transition
    s = canvas.resolveAnimatedStyle(.button, .{ .anim_key = "play_btn", .transition = .{ .duration_ms = 200, .easing = .linear }, .state = .hover });
    try t.expectApproxEqAbs((normal_bg.r + hover_bg.r) * 0.5, s.background.r, 1e-5);

    canvas.begin(); // t=400: finished -> exact target
    s = canvas.resolveAnimatedStyle(.button, .{ .anim_key = "play_btn", .transition = .{ .duration_ms = 200, .easing = .linear }, .state = .hover });
    try t.expect(testColorEql(hover_bg, s.background));

    canvas.begin(); // t=500: finished transitions stay on the target (no restart drift)
    s = canvas.resolveAnimatedStyle(.button, .{ .anim_key = "play_btn", .transition = .{ .duration_ms = 200, .easing = .linear }, .state = .hover });
    try t.expect(testColorEql(hover_bg, s.background));
}

test "retargeting mid-flight restarts from the currently drawn style" {
    const UICanvas = @import("canvas.zig").UICanvas;
    const t = std.testing;
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);
    canvas.frame_dt_ms = 100;
    const normal_bg = Color4.new(0.14, 0.18, 0.25, 0.85);
    const hover_bg = Color4.new(0.24, 0.32, 0.44, 0.92);

    canvas.begin(); // t=100: seed the entry in the normal state
    _ = canvas.resolveAnimatedStyle(.button, .{ .anim_key = "r", .transition = .{ .duration_ms = 300, .easing = .linear } });
    canvas.begin(); // t=200: hover starts from normal
    _ = canvas.resolveAnimatedStyle(.button, .{ .anim_key = "r", .transition = .{ .duration_ms = 300, .easing = .linear }, .state = .hover });
    canvas.begin(); // t=300: 100/300 through the hover animation
    const mid = canvas.resolveAnimatedStyle(.button, .{ .anim_key = "r", .transition = .{ .duration_ms = 300, .easing = .linear }, .state = .hover });
    try t.expectApproxEqAbs(ui_transition.lerpStyle(UIStyle{ .background = normal_bg }, UIStyle{ .background = hover_bg }, 1.0 / 3.0).background.r, mid.background.r, 1e-5);

    canvas.begin(); // t=400: back to normal mid-flight; the drawn style (200/300) is the new source
    const back = canvas.resolveAnimatedStyle(.button, .{ .anim_key = "r", .transition = .{ .duration_ms = 300, .easing = .linear } });
    try t.expectApproxEqAbs(normal_bg.r + (hover_bg.r - normal_bg.r) * (2.0 / 3.0), back.background.r, 1e-5);

    canvas.begin(); // t=500: now animating from that point toward normal
    const later = canvas.resolveAnimatedStyle(.button, .{ .anim_key = "r", .transition = .{ .duration_ms = 300, .easing = .linear } });
    const from_r = normal_bg.r + (hover_bg.r - normal_bg.r) * (2.0 / 3.0);
    try t.expectApproxEqAbs(from_r + (normal_bg.r - from_r) * (1.0 / 3.0), later.background.r, 1e-5);
}

test "class-level transition config drives the animation without per-call override" {
    const UICanvas = @import("canvas.zig").UICanvas;
    const t = std.testing;
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);
    canvas.frame_dt_ms = 50;
    const a = Color4.new(0.1, 0.0, 0.0, 1.0);
    canvas.setStyleClass("anim", .{
        .normal = .{ .background = a },
        .transition = .{ .duration_ms = 100, .easing = .linear },
    });
    const class_bg = Color4.new(0.9, 0.2, 0.1, 1.0);

    canvas.begin(); // t=50
    var s = canvas.resolveAnimatedStyle(.panel, .{ .anim_key = "p", .class = "anim" });
    try t.expect(testColorEql(a, s.background));

    canvas.begin(); // t=100: per-call override retargets the same animated widget
    s = canvas.resolveAnimatedStyle(.panel, .{ .anim_key = "p", .class = "anim", .style = .{ .background = class_bg } });
    try t.expect(testColorEql(a, s.background));

    canvas.begin(); // t=150: halfway a -> class_bg
    s = canvas.resolveAnimatedStyle(.panel, .{ .anim_key = "p", .class = "anim", .style = .{ .background = class_bg } });
    try t.expectApproxEqAbs((a.r + class_bg.r) * 0.5, s.background.r, 1e-4);
}

test "transition table recycles slots without disturbing active targets" {
    const UICanvas = @import("canvas.zig").UICanvas;
    const t = std.testing;
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);
    canvas.frame_dt_ms = 100;
    const bg = Color4.new(0.2, 0.4, 0.6, 1.0);

    canvas.begin(); // t=100: fill every slot
    var i: usize = 0;
    while (i < ui_transition.max_style_transitions) : (i += 1) {
        var name_buf: [16]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "w{d}", .{i}) catch unreachable;
        const s = canvas.resolveAnimatedStyle(.badge, .{
            .anim_key = name,
            .transition = .{ .duration_ms = 500, .easing = .linear },
            .style = .{ .background = bg },
        });
        try t.expect(testColorEql(bg, s.background));
    }
    try t.expectEqual(@as(usize, ui_transition.max_style_transitions), usedTransitionSlots(&canvas));

    canvas.begin(); // t=200: one more widget evicts the stalest entry, all still resolve
    const extra = canvas.resolveAnimatedStyle(.badge, .{
        .anim_key = "extra",
        .transition = .{ .duration_ms = 500, .easing = .linear },
        .style = .{ .background = bg },
    });
    try t.expect(testColorEql(bg, extra.background));
    try t.expectEqual(@as(usize, ui_transition.max_style_transitions), usedTransitionSlots(&canvas));

    // Zero-duration config snaps and releases its slot.
    const snap = canvas.resolveAnimatedStyle(.badge, .{
        .anim_key = "extra",
        .transition = .{},
        .style = .{ .background = bg },
    });
    try t.expect(testColorEql(bg, snap.background));
    try t.expectEqual(@as(usize, ui_transition.max_style_transitions - 1), usedTransitionSlots(&canvas));
}

test "animated styled widgets draw their interpolated style" {
    const UICanvas = @import("canvas.zig").UICanvas;
    const t = std.testing;
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);
    canvas.frame_dt_ms = 100;
    canvas.begin();

    // Widgets without anim_key stay allocation-free and stateless.
    canvas.drawStyledButton("No", .{ 0, 0, 80, 24 }, 14, .{});
    try t.expectEqual(@as(usize, 0), usedTransitionSlots(&canvas));

    // A styled button with an anim_key goes through the transition layer and
    // still emits its full geometry (radius-4 fill bands + text glyphs).
    canvas.begin();
    canvas.drawStyledButton("Ok", .{ 0, 0, 80, 24 }, 14, .{ .anim_key = "btn", .transition = .{ .duration_ms = 100 } });
    try t.expect(quadCount(&canvas) >= 7);
    try t.expectEqual(@as(usize, 1), usedTransitionSlots(&canvas));
}
