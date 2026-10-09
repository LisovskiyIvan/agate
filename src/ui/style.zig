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
const text_mod = @import("text.zig");

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
    const text_w = text_mod.measureForCanvas(canvas, text, font_size).x;
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
