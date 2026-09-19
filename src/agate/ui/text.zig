//! UI text rendering: SDF glyph atlas lookup plus the monospace text
//! drawing, measuring and consistency helpers.
//!
//! Split out of `ui.zig` (facade). Drawing functions take a generic
//! `canvas: anytype` (a `*UICanvas` in practice) so this module never imports
//! `../ui.zig` — quads are emitted through `canvas.addQuad(...)`, which
//! `ui.zig` forwards to `draw.zig`. `ui.zig` re-exports `GlyphUV`/`getGlyphUV`
//! and forwards the `UICanvas` text methods so the public API is unchanged.

const std = @import("std");

const math = @import("math");
const Color4 = math.Color4;
const Vec2 = math.Vec2;

pub const GlyphUV = struct {
    u_min: f32,
    v_min: f32,
    u_max: f32,
    v_max: f32,
};

/// Computes UV texture coordinates in the 512x512 Signed Distance Field atlas (16 cols x 8 rows)
pub fn getGlyphUV(char_code: u8) GlyphUV {
    const code: usize = if (char_code >= 32 and char_code <= 126) char_code - 32 else 0;
    const col: f32 = @floatFromInt(code % 16);
    const row: f32 = @floatFromInt(code / 16);
    return .{
        .u_min = (col * 32.0) / 512.0,
        .v_min = (row * 64.0) / 512.0,
        .u_max = ((col + 1.0) * 32.0) / 512.0,
        .v_max = ((row + 1.0) * 64.0) / 512.0,
    };
}

/// Draws crisp Signed Distance Field (SDF) text
pub fn drawText(canvas: anytype, text: []const u8, x: f32, y: f32, font_size: f32, color: Color4) void {
    drawTextInternal(canvas, text, x, y, font_size, color, 1.0, 0.0, 0.0);
}

/// Draws bold Signed Distance Field (SDF) text
pub fn drawTextBold(canvas: anytype, text: []const u8, x: f32, y: f32, font_size: f32, color: Color4, extra_boldness: f32) void {
    drawTextInternal(canvas, text, x, y, font_size, color, 1.0, 0.0, extra_boldness);
}

/// Draws SDF text with a high-contrast dark outline / shadow
pub fn drawTextWithOutline(canvas: anytype, text: []const u8, x: f32, y: f32, font_size: f32, color: Color4, outline_width: f32) void {
    drawTextInternal(canvas, text, x, y, font_size, color, 2.0, outline_width, 0.0);
}

fn drawTextInternal(canvas: anytype, text: []const u8, start_x: f32, start_y: f32, font_size: f32, color: Color4, mode: f32, outline_width: f32, boldness: f32) void {
    const char_w = font_size * 0.5;
    const char_h = font_size;
    var cur_x = start_x;
    var cur_y = start_y;

    for (text) |c| {
        if (c == '\n') {
            cur_x = start_x;
            cur_y += char_h * 1.15;
            continue;
        }
        if (c == ' ') {
            cur_x += char_w;
            continue;
        }
        const uv = getGlyphUV(c);
        canvas.addQuad(
            cur_x,
            cur_y,
            char_w,
            char_h,
            uv.u_min,
            uv.v_min,
            uv.u_max,
            uv.v_max,
            color,
            .{ mode, outline_width, boldness, 0.0 },
        );
        cur_x += char_w;
    }
}

/// Returns the pixel dimensions of a text string
pub fn measureText(text: []const u8, font_size: f32) Vec2 {
    const char_w = font_size * 0.5;
    const char_h = font_size;
    var max_w: f32 = 0.0;
    var cur_w: f32 = 0.0;
    var total_h: f32 = char_h;

    for (text) |c| {
        if (c == '\n') {
            max_w = @max(max_w, cur_w);
            cur_w = 0.0;
            total_h += char_h * 1.15;
        } else {
            cur_w += char_w;
        }
    }
    max_w = @max(max_w, cur_w);
    return Vec2.new(max_w, total_h);
}

test "getGlyphUV layout" {
    // Space char (32) should be at col 0, row 0
    const space_uv = getGlyphUV(' ');
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), space_uv.u_min, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), space_uv.v_min, 1e-5);

    // '!' char (33) should be at col 1, row 0
    const excl_uv = getGlyphUV('!');
    try std.testing.expectApproxEqAbs(@as(f32, 32.0 / 512.0), excl_uv.u_min, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), excl_uv.v_min, 1e-5);
}

test "UICanvas measureText" {
    const size = measureText("Hello World", 20.0);
    try std.testing.expectApproxEqAbs(@as(f32, 110.0), size.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), size.y, 1e-5);
}

test "UICanvas measureText consistency" {
    // Monospace advance: measureText matches the per-char accumulation
    // drawText (and the drawTextInput cursor) uses.
    const a = measureText("a", 16.0);
    const ab = measureText("ab", 16.0);
    try std.testing.expectApproxEqAbs(a.x * 2.0, ab.x, 1e-5);
    try std.testing.expectApproxEqAbs(a.y, ab.y, 1e-5);
    const empty = measureText("", 16.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), empty.x, 1e-5);

    // Dropdown list total height = count * shared item height.
    const input_state = @import("input_state.zig");
    const ih = input_state.dropdownItemHeight(16.0);
    try std.testing.expectApproxEqAbs(@as(f32, 24.0), ih, 1e-5);
    const btn: [4]f32 = .{ 0, 0, 100, 20 };
    const last = input_state.dropdownItemRect(btn, ih, 3);
    try std.testing.expectApproxEqAbs(@as(f32, 20 + 3 * 24), last[1], 1e-5);
}
