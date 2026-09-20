//! UI widgets: stateless immediate-mode controls (button, checkbox, slider,
//! dropdown, scrollbar, text input, divider, arrow, badge).
//!
//! Split out of `ui.zig` (facade). Functions take a generic `canvas: anytype`
//! (a `*UICanvas` in practice) so this module never imports `../ui.zig` —
//! primitives (`drawPanel`, `drawText`, ...) are reached through the canvas
//! forwarders, and the pure geometry (dropdown rows, scrollbar thumb) comes
//! from `input_state.zig`. `ui.zig` forwards the `UICanvas` widget methods so
//! the public API is unchanged.
//!
//! Styled variants (`drawStyledButton`, ...) live in `style.zig`: they couple
//! to the theme cascade and the retained transition storage owned by
//! `UICanvas` (reached through the canvas forwarders like every other leaf).

const std = @import("std");

const math = @import("math");
const Color4 = math.Color4;

const input_state = @import("input_state.zig");
const TextInputState = input_state.TextInputState;
const text_mod = @import("text.zig");

/// Draws an interactive styled button
pub fn drawButton(canvas: anytype, text: []const u8, x: f32, y: f32, w: f32, h: f32, font_size: f32, is_hovered: bool, is_pressed: bool) void {
    const bg = if (is_pressed)
        Color4.new(0.18, 0.42, 0.78, 0.95)
    else if (is_hovered)
        Color4.new(0.24, 0.32, 0.44, 0.92)
    else
        Color4.new(0.14, 0.18, 0.25, 0.85);

    const border = if (is_pressed)
        Color4.new(0.4, 0.75, 1.0, 1.0)
    else if (is_hovered)
        Color4.new(0.65, 0.85, 1.0, 0.95)
    else
        Color4.new(0.3, 0.4, 0.52, 0.75);

    canvas.drawPanel(x, y, w, h, bg, border, 1.5);

    const text_w = @as(f32, @floatFromInt(text.len)) * font_size * 0.5;
    const tx = x + (w - text_w) * 0.5;
    const ty = y + (h - font_size) * 0.5;
    canvas.drawTextWithOutline(text, tx, ty, font_size, Color4.white, 0.16);
}

/// Draws a compact pill-shaped badge with text (e.g. status tags, FPS counter badge)
pub fn drawBadge(canvas: anytype, text: []const u8, x: f32, y: f32, font_size: f32, bg_col: Color4, text_col: Color4) void {
    const text_w = @as(f32, @floatFromInt(text.len)) * font_size * 0.5;
    const pad_x = font_size * 0.4;
    const pad_y = font_size * 0.25;
    const w = text_w + pad_x * 2.0;
    const h = font_size + pad_y * 2.0;

    canvas.drawPanel(x, y, w, h, bg_col, Color4.new(bg_col.r * 1.3, bg_col.g * 1.3, bg_col.b * 1.3, 0.9), 1.0);
    canvas.drawText(text, x + pad_x, y + pad_y, font_size, text_col);
}

/// Draws a stateless checkbox box with an optional label to the right
pub fn drawCheckbox(canvas: anytype, x: f32, y: f32, size: f32, checked: bool, is_hovered: bool, label: ?[]const u8, label_size: f32) void {
    const bg = if (checked)
        (if (is_hovered) Color4.new(0.24, 0.50, 0.86, 0.95) else Color4.new(0.18, 0.42, 0.78, 0.95))
    else
        (if (is_hovered) Color4.new(0.24, 0.32, 0.44, 0.92) else Color4.new(0.14, 0.18, 0.25, 0.85));
    const border = if (is_hovered)
        Color4.new(0.65, 0.85, 1.0, 0.95)
    else
        Color4.new(0.3, 0.4, 0.52, 0.75);

    canvas.drawPanel(x, y, size, size, bg, border, 1.5);
    if (checked) {
        const m = size * 0.25;
        canvas.drawRect(x + m, y + m, size - 2.0 * m, size - 2.0 * m, Color4.white);
    }
    if (label) |txt| {
        const ty = y + (size - label_size) * 0.5;
        canvas.drawText(txt, x + size + 8.0, ty, label_size, Color4.white);
    }
}

/// Draws a stateless horizontal slider, returns the clamped value
pub fn drawSlider(canvas: anytype, x: f32, y: f32, w: f32, h: f32, value: f32, is_hovered: bool, is_dragging: bool) f32 {
    const v = std.math.clamp(value, 0.0, 1.0);
    const track_bg = Color4.new(0.10, 0.12, 0.18, 0.9);
    const fill_col = if (is_dragging)
        Color4.new(0.30, 0.62, 1.0, 1.0)
    else if (is_hovered)
        Color4.new(0.26, 0.56, 0.94, 1.0)
    else
        Color4.new(0.20, 0.46, 0.82, 0.95);

    canvas.drawRect(x, y, w, h, track_bg);
    if (v > 0.001) {
        canvas.drawRect(x, y, w * v, h, fill_col);
    }
    canvas.drawRectOutline(x, y, w, h, 1.0, Color4.new(0.45, 0.5, 0.6, 0.7));

    // Knob: small square centered on the fill edge, slightly taller than the track
    const knob_size = @max(h + 6.0, 10.0);
    const cx = x + v * w;
    const kx = if (w <= knob_size)
        x + (w - knob_size) * 0.5
    else
        std.math.clamp(cx - knob_size * 0.5, x, x + w - knob_size);
    const ky = y + h * 0.5 - knob_size * 0.5;
    const knob_bg = if (is_dragging)
        Color4.new(0.75, 0.87, 1.0, 1.0)
    else if (is_hovered)
        Color4.new(0.62, 0.72, 0.86, 1.0)
    else
        Color4.new(0.52, 0.60, 0.72, 1.0);
    const knob_border = if (is_dragging or is_hovered) Color4.white else Color4.new(0.3, 0.36, 0.46, 0.9);
    canvas.drawPanel(kx, ky, knob_size, knob_size, knob_bg, knob_border, 1.5);
    return v;
}

/// Draws a thin horizontal separator line
pub fn drawDivider(canvas: anytype, x: f32, y: f32, w: f32, thickness: f32, color: Color4) void {
    if (w <= 0.0 or thickness <= 0.0) return;
    canvas.drawRect(x, y, w, thickness, color);
}

/// Draws a small down-triangle arrow (dropdown chevron) from stacked solid quads
pub fn drawArrowDown(canvas: anytype, x: f32, y: f32, size: f32, color: Color4) void {
    if (size <= 0.0) return;
    const n: usize = 4;
    const nf: f32 = @floatFromInt(n);
    const row_h = size / nf;
    for (0..n) |i| {
        const fi: f32 = @floatFromInt(i);
        const row_w = size * (1.0 - fi / nf);
        const ox = (size - row_w) * 0.5;
        canvas.drawRect(x + ox, y + fi * row_h, row_w, row_h, color);
    }
}

/// Draws the closed button (pressed-look while open) plus, when open,
/// the item list below it. `label` is the button caption (usually the
/// selected item or a placeholder). The selected item is highlighted;
/// `selected`/`hover_index` may be null. Reuses drawButton/drawPanel/
/// drawText/drawArrowDown. No clipping: keep lists short or pair with
/// scroll state. The centered button caption may underlap the chevron
/// on narrow buttons; pass a label with a trailing gap if it matters.
///
/// Call order per frame:
///   1. drawDropdown(...) to emit geometry;
///   2. on click inside the button rect (isPointInRect) toggle `open`;
///   3. while open, on click use dropdownHit(...) to pick the item
///      (a click elsewhere, incl. the button, closes without picking).
pub fn drawDropdown(
    canvas: anytype,
    rect: [4]f32,
    label: []const u8,
    items: []const []const u8,
    selected: ?usize,
    open: bool,
    hover_index: ?usize,
    font_size: f32,
) void {
    const x = rect[0];
    const y = rect[1];
    const w = rect[2];
    const h = rect[3];
    canvas.drawButton(label, x, y, w, h, font_size, false, open);
    const arrow_size = @min(h * 0.4, 12.0);
    if (arrow_size > 0.0 and w > arrow_size + 12.0) {
        canvas.drawArrowDown(x + w - arrow_size - 8.0, y + (h - arrow_size) * 0.5, arrow_size, Color4.white);
    }
    if (!open) return;
    const item_h = input_state.dropdownItemHeight(font_size);
    for (items, 0..) |item, i| {
        const r = input_state.dropdownItemRect(rect, item_h, i);
        const is_sel = if (selected) |s| s == i else false;
        const is_hov = if (hover_index) |hv| hv == i else false;
        const bg = if (is_sel)
            Color4.new(0.18, 0.42, 0.78, 0.95)
        else if (is_hov)
            Color4.new(0.24, 0.32, 0.44, 0.92)
        else
            Color4.new(0.14, 0.18, 0.25, 0.85);
        canvas.drawPanel(r[0], r[1], r[2], r[3], bg, Color4.new(0.3, 0.4, 0.52, 0.75), 1.0);
        canvas.drawText(item, r[0] + 6.0, r[1] + (item_h - font_size) * 0.5, font_size, Color4.white);
    }
}

/// Draws the scrollbar track + thumb (slider-like colors).
pub fn drawScrollbar(canvas: anytype, track: [4]f32, content_h: f32, view_h: f32, offset: f32) void {
    canvas.drawRect(track[0], track[1], track[2], track[3], Color4.new(0.10, 0.12, 0.18, 0.9));
    const thumb = input_state.scrollbarThumbRect(track, content_h, view_h, offset);
    canvas.drawPanel(thumb[0], thumb[1], thumb[2], thumb[3], Color4.new(0.52, 0.60, 0.72, 1.0), Color4.new(0.3, 0.36, 0.46, 0.9), 1.0);
}

/// Single-line field: panel, text, and a 2px cursor bar at the cursor
/// byte offset when focused. No blink timer (static line) and no
/// clipping: overlong text overflows the frame, the caller may shorten
/// or scroll it. The cursor x reuses canvas-aware measurement so it
/// matches the drawText advances exactly, byte for byte (TTF metrics
/// when a font is set, legacy monospace otherwise).
pub fn drawTextInput(canvas: anytype, rect: [4]f32, state: *const TextInputState, focused: bool, font_size: f32) void {
    const bg = if (focused) Color4.new(0.09, 0.11, 0.16, 0.95) else Color4.new(0.10, 0.12, 0.18, 0.9);
    const border = if (focused) Color4.new(0.4, 0.75, 1.0, 1.0) else Color4.new(0.3, 0.4, 0.52, 0.75);
    canvas.drawPanel(rect[0], rect[1], rect[2], rect[3], bg, border, 1.5);
    const pad_x: f32 = 6.0;
    const tx = rect[0] + pad_x;
    const ty = rect[1] + (rect[3] - font_size) * 0.5;
    canvas.drawText(state.text(), tx, ty, font_size, Color4.white);
    if (focused) {
        const cur = @min(state.cursor, state.len);
        // Canvas-aware measurement: matches whichever font drawText used
        // (TTF advances when a font is set, legacy monospace otherwise).
        const cx = tx + text_mod.measureForCanvas(canvas, state.buf[0..cur], font_size).x;
        canvas.drawRect(cx, ty, 2.0, font_size, Color4.white);
    }
}
