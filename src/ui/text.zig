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

const ttf = @import("../ttf.zig");
const ui_draw = @import("draw.zig");

/// Shader mode for TrueType coverage text (ui.glsl mode 3: white RGB +
/// alpha coverage sampling). Same UIVertex layout as every other quad.
pub const ttf_text_mode: f32 = 3.0;

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
    // TTF override: when the canvas carries a TrueType font, every text
    // entry point (plain/bold/outline) draws coverage quads from the TTF
    // atlas instead of SDF cells. Bold/outline parameters have no meaning
    // for baked coverage and are ignored (documented in ttf.zig).
    if (canvasTtfFont(canvas)) |font| {
        drawTextTtf(canvas, font, text, start_x, start_y, font_size, color);
        return;
    }
    const char_w = font_size * 0.5;
    const char_h = font_size;
    var cur_x = start_x;
    var cur_y = start_y;

    var i: usize = 0;
    while (i < text.len) {
        const dc = decodeCodepoint(text, i);
        i += dc.len;
        if (dc.cp == '\n') {
            cur_x = start_x;
            cur_y += char_h * 1.15;
            continue;
        }
        if (dc.cp == ' ') {
            cur_x += char_w;
            continue;
        }
        const c: u8 = if (dc.cp <= 127) @intCast(dc.cp) else '?';
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

    var i: usize = 0;
    while (i < text.len) {
        const dc = decodeCodepoint(text, i);
        i += dc.len;
        if (dc.cp == '\n') {
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

// ---------------------------------------------------------------------------
// TrueType text path (see ../ttf.zig). Same UIVertex quads as the SDF
// path, but UVs address the TTF coverage atlas and advances/kern come
// from hmtx/kern at the baked pixel size, scaled to the draw size.
// ---------------------------------------------------------------------------

/// Returns the canvas TTF font when the (generic) canvas carries one.
/// Real `UICanvas` values always have the field; foreign canvas shapes
/// without it simply take the legacy path.
fn canvasTtfFont(canvas: anytype) ?*const ttf.TtfFont {
    const Child = switch (@typeInfo(@TypeOf(canvas))) {
        .pointer => |p| p.child,
        else => return null,
    };
    if (!@hasField(Child, "ttf_font")) return null;
    return canvas.ttf_font;
}

/// Decodes one UTF-8 codepoint at `text[i]`; invalid bytes yield U+FFFD
/// and consume one byte (they render .notdef, never break the loop).
fn decodeCodepoint(text: []const u8, i: usize) struct { cp: u21, len: usize } {
    const b0 = text[i];
    if (b0 < 0x80) return .{ .cp = b0, .len = 1 };
    const rest = text.len - i;
    if (b0 & 0xE0 == 0xC0 and rest >= 2) {
        const b1 = text[i + 1];
        if (b1 & 0xC0 == 0x80) {
            const cp: u21 = (@as(u21, b0 & 0x1F) << 6) | (b1 & 0x3F);
            if (cp >= 0x80) return .{ .cp = cp, .len = 2 };
        }
    } else if (b0 & 0xF0 == 0xE0 and rest >= 3) {
        const b1 = text[i + 1];
        const b2 = text[i + 2];
        if (b1 & 0xC0 == 0x80 and b2 & 0xC0 == 0x80) {
            const cp: u21 = (@as(u21, b0 & 0x0F) << 12) | (@as(u21, b1 & 0x3F) << 6) | (b2 & 0x3F);
            if (cp >= 0x800 and (cp < 0xD800 or cp > 0xDFFF)) return .{ .cp = cp, .len = 3 };
        }
    } else if (b0 & 0xF8 == 0xF0 and rest >= 4) {
        const b1 = text[i + 1];
        const b2 = text[i + 2];
        const b3 = text[i + 3];
        if (b1 & 0xC0 == 0x80 and b2 & 0xC0 == 0x80 and b3 & 0xC0 == 0x80) {
            const cp: u21 = (@as(u21, b0 & 0x07) << 18) | (@as(u21, b1 & 0x3F) << 12) |
                (@as(u21, b2 & 0x3F) << 6) | (b3 & 0x3F);
            if (cp >= 0x10000 and cp <= 0x10FFFF) return .{ .cp = cp, .len = 4 };
        }
    }
    return .{ .cp = 0xFFFD, .len = 1 };
}

/// Draws coverage (mode 3) TTF text: `x, y` is the top of the first line
/// (same convention as the SDF path), `font_size` scales the baked atlas.
/// Newlines reset the pen and advance the baseline by the font line
/// height; kern applies between consecutive baked glyphs. Empty glyphs
/// (space) advance the pen without emitting a quad.
pub fn drawTextTtf(
    canvas: anytype,
    font: *const ttf.TtfFont,
    text: []const u8,
    x: f32,
    y: f32,
    font_size: f32,
    color: Color4,
) void {
    if (font_size <= 0.0) return;
    const s = font_size / font.pixel_size;
    var pen_x = x;
    var baseline = y + font.ascent * s;
    const line_h = font.line_height * s;
    var prev: ?ttf.GlyphInfo = null;
    var i: usize = 0;
    while (i < text.len) {
        const dc = decodeCodepoint(text, i);
        i += dc.len;
        if (dc.cp == '\n') {
            pen_x = x;
            baseline += line_h;
            prev = null;
            continue;
        }
        const g = font.lookup(dc.cp) orelse {
            prev = null;
            continue;
        };
        if (prev) |p| pen_x += font.kernPx(p, g) * s;
        prev = g;
        if (g.w > 0 and g.h > 0) {
            const uv = font.glyphUv(g);
            canvas.addQuad(
                pen_x + g.bearing_x * s,
                baseline - g.bearing_y * s,
                @as(f32, @floatFromInt(g.w)) * s,
                @as(f32, @floatFromInt(g.h)) * s,
                uv[0],
                uv[1],
                uv[2],
                uv[3],
                color,
                .{ ttf_text_mode, 0.0, 0.0, 0.0 },
            );
        }
        pen_x += g.advance * s;
    }
}

/// TTF text measurement: the exact advance/kerning/line math
/// `drawTextTtf` uses, so layout and the text-input cursor match the
/// drawn glyphs byte for byte (same contract as the SDF pair).
pub fn measureTextTtf(font: *const ttf.TtfFont, text: []const u8, font_size: f32) Vec2 {
    if (font_size <= 0.0) return Vec2.new(0.0, 0.0);
    const s = font_size / font.pixel_size;
    const line_h = font.line_height * s;
    var max_w: f32 = 0.0;
    var pen: f32 = 0.0;
    var total_h: f32 = line_h;
    var prev: ?ttf.GlyphInfo = null;
    var i: usize = 0;
    while (i < text.len) {
        const dc = decodeCodepoint(text, i);
        i += dc.len;
        if (dc.cp == '\n') {
            max_w = @max(max_w, pen);
            pen = 0.0;
            total_h += line_h;
            prev = null;
            continue;
        }
        const g = font.lookup(dc.cp) orelse {
            prev = null;
            continue;
        };
        if (prev) |p| pen += font.kernPx(p, g) * s;
        prev = g;
        pen += g.advance * s;
    }
    max_w = @max(max_w, pen);
    return Vec2.new(max_w, total_h);
}

/// Canvas-aware measurement for shared widgets (text-input cursor):
/// TTF metrics when the canvas carries a font, legacy otherwise.
pub fn measureForCanvas(canvas: anytype, text: []const u8, font_size: f32) Vec2 {
    if (canvasTtfFont(canvas)) |font| return measureTextTtf(font, text, font_size);
    return measureText(text, font_size);
}
