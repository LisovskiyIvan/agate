//! UI input state: caller-owned widget state plus pure hit-test and scroll
//! geometry. No canvas, no sokol, no allocation — everything here is pure
//! data + pure helpers, unit-testable without a GPU context.
//!
//! Split out of `ui.zig` (facade): `ui.zig` re-exports the types and
//! forwards the `UICanvas` static helpers (`isPointInRect`, `scrollClamp`,
//! `dropdownHit`, ...) so the public API is unchanged. Stateful drawing
//! (scrollbar track/thumb, text-input field, dropdown list) lives in
//! `widgets.zig` and calls back into these pure helpers through the canvas.

const std = @import("std");

/// Vertical scroll state. Caller-owned; UICanvas stays stateless
/// (the sandbox will keep one ScrollState per scrollable list).
pub const ScrollState = struct {
    offset: f32 = 0.0,
    content_h: f32 = 0.0,
    view_h: f32 = 0.0,

    /// Maximum legal offset (0 when the content fits in the view).
    pub fn maxOffset(self: *const ScrollState) f32 {
        return @max(self.content_h - self.view_h, 0.0);
    }
};

fn isContinuationByte(b: u8) bool {
    return (b & 0xC0) == 0x80;
}

/// Single-line text input state. Caller-owned fixed buffer of UTF-8 bytes.
/// `cursor` is a byte index always kept on a codepoint boundary, so editing
/// never splits a multibyte sequence. Rendering via UICanvas.drawText is
/// ASCII-only (the font atlas covers codes 32..126): multibyte codepoints
/// are stored and edited safely but draw as fallback glyphs.
pub const TextInputState = struct {
    buf: [128]u8 = undefined,
    len: usize = 0,
    cursor: usize = 0,

    /// Current contents as a byte slice.
    pub fn text(self: *const TextInputState) []const u8 {
        return self.buf[0..self.len];
    }

    /// Replaces the whole buffer; overlong input is truncated on a codepoint
    /// boundary (no split multibyte sequence). Cursor moves to the end.
    pub fn setText(self: *TextInputState, s: []const u8) void {
        var n: usize = @min(s.len, self.buf.len);
        if (n < s.len) {
            while (n > 0) {
                var start = n - 1;
                while (start > 0 and isContinuationByte(s[start])) start -= 1;
                const seq_len: usize = std.unicode.utf8ByteSequenceLength(s[start]) catch 1;
                if (start + seq_len > n) {
                    n = start;
                } else break;
            }
        }
        @memcpy(self.buf[0..n], s[0..n]);
        self.len = n;
        self.cursor = n;
    }

    /// Inserts a Unicode scalar value at the cursor (UTF-8 encoded).
    /// Returns false (no change) when the buffer is full or the scalar
    /// is not encodable (e.g. a surrogate half).
    pub fn insertChar(self: *TextInputState, cp: u21) bool {
        var tmp: [4]u8 = undefined;
        const n: usize = std.unicode.utf8Encode(cp, &tmp) catch return false;
        if (self.len + n > self.buf.len) return false;
        std.mem.copyBackwards(u8, self.buf[self.cursor + n .. self.len + n], self.buf[self.cursor..self.len]);
        @memcpy(self.buf[self.cursor .. self.cursor + n], tmp[0..n]);
        self.len += n;
        self.cursor += n;
        return true;
    }

    /// Deletes the codepoint before the cursor. Returns false at position 0.
    pub fn backspace(self: *TextInputState) bool {
        if (self.cursor == 0) return false;
        var start = self.cursor - 1;
        while (start > 0 and isContinuationByte(self.buf[start])) start -= 1;
        const rm = self.cursor - start;
        std.mem.copyForwards(u8, self.buf[start .. self.len - rm], self.buf[self.cursor..self.len]);
        self.len -= rm;
        self.cursor = start;
        return true;
    }

    /// Deletes the codepoint after the cursor. Returns false at end of text.
    pub fn deleteForward(self: *TextInputState) bool {
        if (self.cursor >= self.len) return false;
        var tail = self.cursor + 1;
        while (tail < self.len and isContinuationByte(self.buf[tail])) tail += 1;
        const rm = tail - self.cursor;
        std.mem.copyForwards(u8, self.buf[self.cursor .. self.len - rm], self.buf[tail..self.len]);
        self.len -= rm;
        return true;
    }

    /// Moves one codepoint left/right. Returns false when already at the edge.
    pub fn moveLeft(self: *TextInputState) bool {
        if (self.cursor == 0) return false;
        var next = self.cursor - 1;
        while (next > 0 and isContinuationByte(self.buf[next])) next -= 1;
        self.cursor = next;
        return true;
    }

    pub fn moveRight(self: *TextInputState) bool {
        if (self.cursor >= self.len) return false;
        var next = self.cursor + 1;
        while (next < self.len and isContinuationByte(self.buf[next])) next += 1;
        self.cursor = next;
        return true;
    }

    pub fn home(self: *TextInputState) void {
        self.cursor = 0;
    }

    pub fn end(self: *TextInputState) void {
        self.cursor = self.len;
    }
};

/// Hit test helper: checks if a 2D screen coordinate (e.g. mouse cursor) is inside a rectangle
pub fn isPointInRect(px: f32, py: f32, x: f32, y: f32, w: f32, h: f32) bool {
    return px >= x and px <= (x + w) and py >= y and py <= (y + h);
}

/// Maps a mouse x coordinate to a 0..1 slider value, clamped
pub fn sliderValueAt(x: f32, w: f32, mouse_x: f32) f32 {
    if (w <= 0.0) return 0.0;
    return std.math.clamp((mouse_x - x) / w, 0.0, 1.0);
}

/// Returns the checkbox hit rect as [x, y, w, h] for use with isPointInRect
pub fn checkboxHitRect(x: f32, y: f32, size: f32) [4]f32 {
    return .{ x, y, size, size };
}

/// Applies a wheel delta and clamps offset into 0..content-view.
/// Resets offset to 0 when the content fits in the view.
pub fn scrollClamp(state: *ScrollState, delta: f32) void {
    const max_off = state.maxOffset();
    if (max_off <= 0.0) {
        state.offset = 0.0;
        return;
    }
    state.offset = std.math.clamp(state.offset + delta, 0.0, max_off);
}

/// Offset that makes [item_y, item_y+item_h] visible with minimal movement.
/// Returns 0 when the content fits in the view.
pub fn scrollOffsetForItem(offset: f32, item_y: f32, item_h: f32, view_h: f32, content_h: f32) f32 {
    const max_off = @max(content_h - view_h, 0.0);
    if (max_off <= 0.0) return 0.0;
    var o = std.math.clamp(offset, 0.0, max_off);
    if (item_y < o) {
        o = item_y;
    } else if (item_y + item_h > o + view_h) {
        o = item_y + item_h - view_h;
    }
    return std.math.clamp(o, 0.0, max_off);
}

/// Thumb rect inside a vertical track [x, y, w, h]. Full track when the
/// content fits; otherwise the thumb height is proportional to
/// view/content (16px minimum) and its position maps the offset
/// linearly over 0..content-view.
pub fn scrollbarThumbRect(track: [4]f32, content_h: f32, view_h: f32, offset: f32) [4]f32 {
    if (content_h <= view_h or content_h <= 0.0 or view_h <= 0.0 or track[3] <= 0.0) return track;
    const capped_min = @min(@as(f32, 16.0), track[3]);
    const thumb_h = std.math.clamp(track[3] * (view_h / content_h), capped_min, track[3]);
    const max_off = content_h - view_h;
    const t = std.math.clamp(offset / max_off, 0.0, 1.0);
    return .{ track[0], track[1] + (track[3] - thumb_h) * t, track[2], thumb_h };
}

/// Item row height derived from the font size; shared by drawing and
/// hit-testing so geometry always matches. 4px padding above/below text.
pub fn dropdownItemHeight(font_size: f32) f32 {
    return font_size + 8.0;
}

/// Rect [x, y, w, h] of an open-list item stacked directly below the
/// closed button rect.
pub fn dropdownItemRect(rect: [4]f32, item_h: f32, index: usize) [4]f32 {
    const fi: f32 = @floatFromInt(index);
    return .{ rect[0], rect[1] + rect[3] + fi * item_h, rect[2], item_h };
}

/// Hit-tests ONLY the open list stacked under the button rect.
/// Points over the closed button (or outside the list) return null;
/// hit-test the button itself with isPointInRect.
/// Rows are half-open [y0, y1); the list bottom edge maps to the last item.
pub fn dropdownHit(rect: [4]f32, item_h: f32, count: usize, mx: f32, my: f32) ?usize {
    if (count == 0 or item_h <= 0.0 or rect[2] <= 0.0) return null;
    if (mx < rect[0] or mx > rect[0] + rect[2]) return null;
    const list_y = rect[1] + rect[3];
    const list_h = item_h * @as(f32, @floatFromInt(count));
    if (my < list_y or my > list_y + list_h) return null;
    var idx: usize = @intFromFloat((my - list_y) / item_h);
    if (idx >= count) idx = count - 1; // bottom edge inclusive
    return idx;
}
