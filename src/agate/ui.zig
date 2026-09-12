const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const sapp = sokol.app;

const math = @import("math");
const Vec2 = math.Vec2;
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color4 = math.Color4;

const Texture = @import("texture.zig").Texture;
const ui_shd = @import("ui_shader");

const font_png_data = @embedFile("assets/font_sdf.png");

pub const UIVertex = extern struct {
    position: [2]f32,
    uv: [2]f32,
    color: [4]f32,
    mode_params: [4]f32, // x: mode (0=solid, 1=sdf_text, 2=sdf_outline), y: outline_width, z: softness, w: extra
};

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

pub const UICanvas = struct {
    allocator: std.mem.Allocator,
    vertices: std.ArrayListUnmanaged(UIVertex) = .empty,
    indices: std.ArrayListUnmanaged(u16) = .empty,

    vertex_buffer: sg.Buffer = .{},
    index_buffer: sg.Buffer = .{},
    pipeline: sg.Pipeline = .{},
    font_texture: Texture,

    capacity_vertices: usize = 32768,
    capacity_indices: usize = 49152,

    // Shared solid-quad constants (same values as the previous per-call literals).
    const solid_mode: [4]f32 = .{ 0.0, 0.0, 0.0, 0.0 };
    const solid_uv: f32 = 1.0 / 512.0;

    // Corner math using the already-computed segment length (single sqrt per line).
    fn lineCornersWithLen(x0: f32, y0: f32, x1: f32, y1: f32, dx: f32, dy: f32, len: f32, thickness: f32) [4][2]f32 {
        if (len < 1e-6 or thickness <= 0.0) {
            return .{
                .{ x0, y0 },
                .{ x0, y0 },
                .{ x1, y1 },
                .{ x1, y1 },
            };
        }
        const nx = -dy / len;
        const ny = dx / len;
        const hx = nx * thickness * 0.5;
        const hy = ny * thickness * 0.5;
        return .{
            .{ x0 - hx, y0 - hy },
            .{ x0 + hx, y0 + hy },
            .{ x1 + hx, y1 + hy },
            .{ x1 - hx, y1 - hy },
        };
    }

    pub fn init(allocator: std.mem.Allocator) !UICanvas {
        const tex = try Texture.fromMemory(allocator, font_png_data, .{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
            // Box-filtered mips blur the distance field beyond legibility.
            .mipmaps = false,
        });

        const max_v: usize = 32768;
        const max_i: usize = 49152;

        const vb = sg.makeBuffer(.{
            .usage = .{ .vertex_buffer = true, .dynamic_update = true },
            .size = max_v * @sizeOf(UIVertex),
        });

        const ib = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true, .dynamic_update = true },
            .size = max_i * @sizeOf(u16),
        });

        var pip_desc = sg.PipelineDesc{
            .shader = sg.makeShader(ui_shd.uiShaderDesc(sg.queryBackend())),
            .index_type = .UINT16,
            .depth = .{
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
        };

        pip_desc.colors[0].blend = .{
            .enabled = true,
            .src_factor_rgb = .SRC_ALPHA,
            .dst_factor_rgb = .ONE_MINUS_SRC_ALPHA,
            .src_factor_alpha = .ONE,
            .dst_factor_alpha = .ONE_MINUS_SRC_ALPHA,
        };

        pip_desc.layout.buffers[0] = .{ .stride = @sizeOf(UIVertex) };
        pip_desc.layout.attrs[ui_shd.ATTR_ui_position] = .{
            .buffer_index = 0,
            .format = .FLOAT2,
            .offset = @offsetOf(UIVertex, "position"),
        };
        pip_desc.layout.attrs[ui_shd.ATTR_ui_texcoord0] = .{
            .buffer_index = 0,
            .format = .FLOAT2,
            .offset = @offsetOf(UIVertex, "uv"),
        };
        pip_desc.layout.attrs[ui_shd.ATTR_ui_color0] = .{
            .buffer_index = 0,
            .format = .FLOAT4,
            .offset = @offsetOf(UIVertex, "color"),
        };
        pip_desc.layout.attrs[ui_shd.ATTR_ui_mode_params] = .{
            .buffer_index = 0,
            .format = .FLOAT4,
            .offset = @offsetOf(UIVertex, "mode_params"),
        };

        const pip = sg.makePipeline(pip_desc);

        return .{
            .allocator = allocator,
            .vertex_buffer = vb,
            .index_buffer = ib,
            .pipeline = pip,
            .font_texture = tex,
            .capacity_vertices = max_v,
            .capacity_indices = max_i,
        };
    }

    pub fn deinit(self: *UICanvas) void {
        self.vertices.deinit(self.allocator);
        self.indices.deinit(self.allocator);
        if (self.vertex_buffer.id != 0) sg.destroyBuffer(self.vertex_buffer);
        if (self.index_buffer.id != 0) sg.destroyBuffer(self.index_buffer);
        if (self.pipeline.id != 0) sg.destroyPipeline(self.pipeline);
        self.font_texture.deinit();
    }

    /// Prepares canvas for a new frame of 2D/3D UI rendering
    pub fn begin(self: *UICanvas) void {
        self.vertices.clearRetainingCapacity();
        self.indices.clearRetainingCapacity();
    }

    /// Helper to add a textured / colored quad
    pub fn addQuad(
        self: *UICanvas,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        u_min: f32,
        v_min: f32,
        u_max: f32,
        v_max: f32,
        color: Color4,
        mode_params: [4]f32,
    ) void {
        if (self.vertices.items.len + 4 > self.capacity_vertices) return;
        if (self.indices.items.len + 6 > self.capacity_indices) return;

        const base_idx: u16 = @intCast(self.vertices.items.len);
        const col_arr = color.toArray();

        self.vertices.appendSlice(self.allocator, &[_]UIVertex{
            .{ .position = .{ x, y }, .uv = .{ u_min, v_min }, .color = col_arr, .mode_params = mode_params },
            .{ .position = .{ x + w, y }, .uv = .{ u_max, v_min }, .color = col_arr, .mode_params = mode_params },
            .{ .position = .{ x + w, y + h }, .uv = .{ u_max, v_max }, .color = col_arr, .mode_params = mode_params },
            .{ .position = .{ x, y + h }, .uv = .{ u_min, v_max }, .color = col_arr, .mode_params = mode_params },
        }) catch return;

        self.indices.appendSlice(self.allocator, &[_]u16{
            base_idx + 0, base_idx + 1, base_idx + 2,
            base_idx + 0, base_idx + 2, base_idx + 3,
        }) catch return;
    }

    /// Draws a solid rectangle in screen pixel coordinates
    pub fn drawRect(self: *UICanvas, x: f32, y: f32, w: f32, h: f32, color: Color4) void {
        self.addQuad(x, y, w, h, solid_uv, solid_uv, solid_uv, solid_uv, color, solid_mode);
    }

    /// Draws a rectangle outline with specified border thickness
    pub fn drawRectOutline(self: *UICanvas, x: f32, y: f32, w: f32, h: f32, thickness: f32, color: Color4) void {
        const t = @min(thickness, @min(w * 0.5, h * 0.5));
        self.drawRect(x, y, w, t, color); // Top
        self.drawRect(x, y + h - t, w, t, color); // Bottom
        self.drawRect(x, y + t, t, h - 2.0 * t, color); // Left
        self.drawRect(x + w - t, y + t, t, h - 2.0 * t, color); // Right
    }

    /// Draws a styled UI panel (filled rectangle + border)
    pub fn drawPanel(self: *UICanvas, x: f32, y: f32, w: f32, h: f32, bg_col: Color4, border_col: Color4, border_width: f32) void {
        self.drawRect(x, y, w, h, bg_col);
        if (border_width > 0.0) {
            self.drawRectOutline(x, y, w, h, border_width, border_col);
        }
    }

    /// Draws crisp Signed Distance Field (SDF) text
    pub fn drawText(self: *UICanvas, text: []const u8, x: f32, y: f32, font_size: f32, color: Color4) void {
        self.drawTextInternal(text, x, y, font_size, color, 1.0, 0.0, 0.0);
    }

    /// Draws bold Signed Distance Field (SDF) text
    pub fn drawTextBold(self: *UICanvas, text: []const u8, x: f32, y: f32, font_size: f32, color: Color4, extra_boldness: f32) void {
        self.drawTextInternal(text, x, y, font_size, color, 1.0, 0.0, extra_boldness);
    }

    /// Draws SDF text with a high-contrast dark outline / shadow
    pub fn drawTextWithOutline(self: *UICanvas, text: []const u8, x: f32, y: f32, font_size: f32, color: Color4, outline_width: f32) void {
        self.drawTextInternal(text, x, y, font_size, color, 2.0, outline_width, 0.0);
    }

    fn drawTextInternal(self: *UICanvas, text: []const u8, start_x: f32, start_y: f32, font_size: f32, color: Color4, mode: f32, outline_width: f32, boldness: f32) void {
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
            self.addQuad(
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

    /// Draws a smooth horizontal progress / health bar
    pub fn drawProgressBar(self: *UICanvas, x: f32, y: f32, w: f32, h: f32, progress: f32, bg_col: Color4, fill_col: Color4) void {
        self.drawRect(x, y, w, h, bg_col);
        const clamped_p = std.math.clamp(progress, 0.0, 1.0);
        if (clamped_p > 0.001) {
            self.drawRect(x, y, w * clamped_p, h, fill_col);
        }
        self.drawRectOutline(x, y, w, h, 1.0, Color4.new(0.45, 0.5, 0.6, 0.7));
    }

    /// Draws an interactive styled button
    pub fn drawButton(self: *UICanvas, text: []const u8, x: f32, y: f32, w: f32, h: f32, font_size: f32, is_hovered: bool, is_pressed: bool) void {
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

        self.drawPanel(x, y, w, h, bg, border, 1.5);

        const text_w = @as(f32, @floatFromInt(text.len)) * font_size * 0.5;
        const tx = x + (w - text_w) * 0.5;
        const ty = y + (h - font_size) * 0.5;
        self.drawTextWithOutline(text, tx, ty, font_size, Color4.white, 0.16);
    }

    /// Draws a compact pill-shaped badge with text (e.g. status tags, FPS counter badge)
    pub fn drawBadge(self: *UICanvas, text: []const u8, x: f32, y: f32, font_size: f32, bg_col: Color4, text_col: Color4) void {
        const text_w = @as(f32, @floatFromInt(text.len)) * font_size * 0.5;
        const pad_x = font_size * 0.4;
        const pad_y = font_size * 0.25;
        const w = text_w + pad_x * 2.0;
        const h = font_size + pad_y * 2.0;

        self.drawPanel(x, y, w, h, bg_col, Color4.new(bg_col.r * 1.3, bg_col.g * 1.3, bg_col.b * 1.3, 0.9), 1.0);
        self.drawText(text, x + pad_x, y + pad_y, font_size, text_col);
    }

    /// Draws a stateless checkbox box with an optional label to the right
    pub fn drawCheckbox(self: *UICanvas, x: f32, y: f32, size: f32, checked: bool, is_hovered: bool, label: ?[]const u8, label_size: f32) void {
        const bg = if (checked)
            (if (is_hovered) Color4.new(0.24, 0.50, 0.86, 0.95) else Color4.new(0.18, 0.42, 0.78, 0.95))
        else
            (if (is_hovered) Color4.new(0.24, 0.32, 0.44, 0.92) else Color4.new(0.14, 0.18, 0.25, 0.85));
        const border = if (is_hovered)
            Color4.new(0.65, 0.85, 1.0, 0.95)
        else
            Color4.new(0.3, 0.4, 0.52, 0.75);

        self.drawPanel(x, y, size, size, bg, border, 1.5);
        if (checked) {
            const m = size * 0.25;
            self.drawRect(x + m, y + m, size - 2.0 * m, size - 2.0 * m, Color4.white);
        }
        if (label) |text| {
            const ty = y + (size - label_size) * 0.5;
            self.drawText(text, x + size + 8.0, ty, label_size, Color4.white);
        }
    }

    /// Draws a stateless horizontal slider, returns the clamped value
    pub fn drawSlider(self: *UICanvas, x: f32, y: f32, w: f32, h: f32, value: f32, is_hovered: bool, is_dragging: bool) f32 {
        const v = std.math.clamp(value, 0.0, 1.0);
        const track_bg = Color4.new(0.10, 0.12, 0.18, 0.9);
        const fill_col = if (is_dragging)
            Color4.new(0.30, 0.62, 1.0, 1.0)
        else if (is_hovered)
            Color4.new(0.26, 0.56, 0.94, 1.0)
        else
            Color4.new(0.20, 0.46, 0.82, 0.95);

        self.drawRect(x, y, w, h, track_bg);
        if (v > 0.001) {
            self.drawRect(x, y, w * v, h, fill_col);
        }
        self.drawRectOutline(x, y, w, h, 1.0, Color4.new(0.45, 0.5, 0.6, 0.7));

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
        self.drawPanel(kx, ky, knob_size, knob_size, knob_bg, knob_border, 1.5);
        return v;
    }

    /// Draws a thin horizontal separator line
    pub fn drawDivider(self: *UICanvas, x: f32, y: f32, w: f32, thickness: f32, color: Color4) void {
        if (w <= 0.0 or thickness <= 0.0) return;
        self.drawRect(x, y, w, thickness, color);
    }

    /// Pure corner math for drawLine: returns the 4 quad corners
    /// (p0-left, p0-right, p1-right, p1-left) offset perpendicular
    /// to the segment by half the thickness. Zero-area on degenerate input.
    pub fn lineCorners(x0: f32, y0: f32, x1: f32, y1: f32, thickness: f32) [4][2]f32 {
        const dx = x1 - x0;
        const dy = y1 - y0;
        const len = @sqrt(dx * dx + dy * dy);
        return lineCornersWithLen(x0, y0, x1, y1, dx, dy, len, thickness);
    }

    /// Draws a solid thick line in screen pixel coordinates (no depth test).
    pub fn drawLine(self: *UICanvas, x0: f32, y0: f32, x1: f32, y1: f32, thickness: f32, color: Color4) void {
        const dx = x1 - x0;
        const dy = y1 - y0;
        const len = @sqrt(dx * dx + dy * dy);
        if (len < 1e-6 or thickness <= 0.0) return;
        if (self.vertices.items.len + 4 > self.capacity_vertices) return;
        if (self.indices.items.len + 6 > self.capacity_indices) return;

        const corners = lineCornersWithLen(x0, y0, x1, y1, dx, dy, len, thickness);
        const base_idx: u16 = @intCast(self.vertices.items.len);
        const col_arr = color.toArray();

        self.vertices.appendSlice(self.allocator, &[_]UIVertex{
            .{ .position = corners[0], .uv = .{ solid_uv, solid_uv }, .color = col_arr, .mode_params = solid_mode },
            .{ .position = corners[1], .uv = .{ solid_uv, solid_uv }, .color = col_arr, .mode_params = solid_mode },
            .{ .position = corners[2], .uv = .{ solid_uv, solid_uv }, .color = col_arr, .mode_params = solid_mode },
            .{ .position = corners[3], .uv = .{ solid_uv, solid_uv }, .color = col_arr, .mode_params = solid_mode },
        }) catch return;

        self.indices.appendSlice(self.allocator, &[_]u16{
            base_idx + 0, base_idx + 1, base_idx + 2,
            base_idx + 0, base_idx + 2, base_idx + 3,
        }) catch return;
    }

    /// Draws a small down-triangle arrow (dropdown chevron) from stacked solid quads
    pub fn drawArrowDown(self: *UICanvas, x: f32, y: f32, size: f32, color: Color4) void {
        if (size <= 0.0) return;
        const n: usize = 4;
        const nf: f32 = @floatFromInt(n);
        const row_h = size / nf;
        for (0..n) |i| {
            const fi: f32 = @floatFromInt(i);
            const row_w = size * (1.0 - fi / nf);
            const ox = (size - row_w) * 0.5;
            self.drawRect(x + ox, y + fi * row_h, row_w, row_h, color);
        }
    }

    // ------------------------------------------------------------------
    // Dropdown (stateless immediate-mode; caller owns selected/open/hover)
    // ------------------------------------------------------------------

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
        self: *UICanvas,
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
        self.drawButton(label, x, y, w, h, font_size, false, open);
        const arrow_size = @min(h * 0.4, 12.0);
        if (arrow_size > 0.0 and w > arrow_size + 12.0) {
            self.drawArrowDown(x + w - arrow_size - 8.0, y + (h - arrow_size) * 0.5, arrow_size, Color4.white);
        }
        if (!open) return;
        const item_h = UICanvas.dropdownItemHeight(font_size);
        for (items, 0..) |item, i| {
            const r = UICanvas.dropdownItemRect(rect, item_h, i);
            const is_sel = if (selected) |s| s == i else false;
            const is_hov = if (hover_index) |hv| hv == i else false;
            const bg = if (is_sel)
                Color4.new(0.18, 0.42, 0.78, 0.95)
            else if (is_hov)
                Color4.new(0.24, 0.32, 0.44, 0.92)
            else
                Color4.new(0.14, 0.18, 0.25, 0.85);
            self.drawPanel(r[0], r[1], r[2], r[3], bg, Color4.new(0.3, 0.4, 0.52, 0.75), 1.0);
            self.drawText(item, r[0] + 6.0, r[1] + (item_h - font_size) * 0.5, font_size, Color4.white);
        }
    }

    // ------------------------------------------------------------------
    // Scroll (caller owns ScrollState; UICanvas stays stateless)
    // ------------------------------------------------------------------

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

    /// Draws the scrollbar track + thumb (slider-like colors).
    pub fn drawScrollbar(self: *UICanvas, track: [4]f32, content_h: f32, view_h: f32, offset: f32) void {
        self.drawRect(track[0], track[1], track[2], track[3], Color4.new(0.10, 0.12, 0.18, 0.9));
        const thumb = UICanvas.scrollbarThumbRect(track, content_h, view_h, offset);
        self.drawPanel(thumb[0], thumb[1], thumb[2], thumb[3], Color4.new(0.52, 0.60, 0.72, 1.0), Color4.new(0.3, 0.36, 0.46, 0.9), 1.0);
    }

    // ------------------------------------------------------------------
    // Text input (caller owns TextInputState + the focus flag)
    // ------------------------------------------------------------------

    /// Single-line field: panel, text, and a 2px cursor bar at the cursor
    /// byte offset when focused. No blink timer (static line) and no
    /// clipping: overlong text overflows the frame, the caller may shorten
    /// or scroll it. The cursor x reuses measureText so it matches the
    /// drawText advances exactly, byte for byte.
    pub fn drawTextInput(self: *UICanvas, rect: [4]f32, state: *const TextInputState, focused: bool, font_size: f32) void {
        const bg = if (focused) Color4.new(0.09, 0.11, 0.16, 0.95) else Color4.new(0.10, 0.12, 0.18, 0.9);
        const border = if (focused) Color4.new(0.4, 0.75, 1.0, 1.0) else Color4.new(0.3, 0.4, 0.52, 0.75);
        self.drawPanel(rect[0], rect[1], rect[2], rect[3], bg, border, 1.5);
        const pad_x: f32 = 6.0;
        const tx = rect[0] + pad_x;
        const ty = rect[1] + (rect[3] - font_size) * 0.5;
        self.drawText(state.text(), tx, ty, font_size, Color4.white);
        if (focused) {
            const cur = @min(state.cursor, state.len);
            const cx = tx + measureText(state.buf[0..cur], font_size).x;
            self.drawRect(cx, ty, 2.0, font_size, Color4.white);
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

    /// Uploads dynamic batch buffers and executes the UI render pass
    pub fn render(self: *UICanvas, screen_w: f32, screen_h: f32) void {
        if (self.vertices.items.len == 0 or self.indices.items.len == 0) return;
        if (screen_w <= 0.0 or screen_h <= 0.0) return;

        const vert_count = @min(self.vertices.items.len, std.math.maxInt(u16));
        if (vert_count > self.capacity_vertices) {
            if (self.vertex_buffer.id != 0) sg.destroyBuffer(self.vertex_buffer);
            self.capacity_vertices = @max(self.capacity_vertices * 2, vert_count);
            self.vertex_buffer = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = self.capacity_vertices * @sizeOf(UIVertex),
            });
        }
        if (self.indices.items.len > self.capacity_indices) {
            if (self.index_buffer.id != 0) sg.destroyBuffer(self.index_buffer);
            self.capacity_indices = @max(self.capacity_indices * 2, self.indices.items.len);
            self.index_buffer = sg.makeBuffer(.{
                .usage = .{ .index_buffer = true, .dynamic_update = true },
                .size = self.capacity_indices * @sizeOf(u16),
            });
        }

        sg.updateBuffer(self.vertex_buffer, sg.asRange(self.vertices.items[0..vert_count]));
        sg.updateBuffer(self.index_buffer, sg.asRange(self.indices.items));

        if (self.pipeline.id == 0) return;
        sg.applyPipeline(self.pipeline);

        var bind = sg.Bindings{};
        bind.vertex_buffers[0] = self.vertex_buffer;
        bind.index_buffer = self.index_buffer;
        bind.views[ui_shd.VIEW_font_tex] = self.font_texture.view;
        bind.samplers[ui_shd.SMP_smp] = self.font_texture.sampler;
        sg.applyBindings(bind);

        const vs_params = ui_shd.VsParams{
            .screen_size = .{ screen_w, screen_h, 0.0, 0.0 },
        };
        sg.applyUniforms(ui_shd.UB_vs_params, sg.asRange(&vs_params));

        sg.draw(0, @intCast(self.indices.items.len), 1);
    }
};

// Top-level aliases for the stateless UI helpers above, so calling code can
// use them without spelling `UICanvas.` Both paths share one implementation.
pub fn dropdownItemHeight(font_size: f32) f32 {
    return UICanvas.dropdownItemHeight(font_size);
}

pub fn dropdownItemRect(rect: [4]f32, item_h: f32, index: usize) [4]f32 {
    return UICanvas.dropdownItemRect(rect, item_h, index);
}

pub fn dropdownHit(rect: [4]f32, item_h: f32, count: usize, mx: f32, my: f32) ?usize {
    return UICanvas.dropdownHit(rect, item_h, count, mx, my);
}

pub fn scrollClamp(state: *ScrollState, delta: f32) void {
    UICanvas.scrollClamp(state, delta);
}

pub fn scrollOffsetForItem(offset: f32, item_y: f32, item_h: f32, view_h: f32, content_h: f32) f32 {
    return UICanvas.scrollOffsetForItem(offset, item_y, item_h, view_h, content_h);
}

pub fn scrollbarThumbRect(track: [4]f32, content_h: f32, view_h: f32, offset: f32) [4]f32 {
    return UICanvas.scrollbarThumbRect(track, content_h, view_h, offset);
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
    const size = UICanvas.measureText("Hello World", 20.0);
    try std.testing.expectApproxEqAbs(@as(f32, 110.0), size.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), size.y, 1e-5);
}

test "UICanvas isPointInRect" {
    try std.testing.expect(UICanvas.isPointInRect(50, 50, 0, 0, 100, 100));
    try std.testing.expect(!UICanvas.isPointInRect(150, 50, 0, 0, 100, 100));
}

test "UICanvas sliderValueAt" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), UICanvas.sliderValueAt(10, 100, 60), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), UICanvas.sliderValueAt(10, 100, 10), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), UICanvas.sliderValueAt(10, 100, 110), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), UICanvas.sliderValueAt(10, 100, -50), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), UICanvas.sliderValueAt(10, 100, 500), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), UICanvas.sliderValueAt(10, 0, 60), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), UICanvas.sliderValueAt(10, -20, 60), 1e-5);
}

test "UICanvas checkboxHitRect" {
    const r = UICanvas.checkboxHitRect(10, 20, 24);
    try std.testing.expectApproxEqAbs(@as(f32, 10), r[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 20), r[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 24), r[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 24), r[3], 1e-5);
    try std.testing.expect(UICanvas.isPointInRect(15, 25, r[0], r[1], r[2], r[3]));
    try std.testing.expect(!UICanvas.isPointInRect(100, 100, r[0], r[1], r[2], r[3]));
}

test "UICanvas lineCorners" {
    // Horizontal segment: thickness extends along +/-Y.
    const h = UICanvas.lineCorners(0, 0, 10, 0, 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), h[0][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), h[0][1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), h[1][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), h[1][1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), h[2][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), h[2][1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), h[3][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), h[3][1], 1e-5);

    // Vertical segment: thickness extends along +/-X.
    const v = UICanvas.lineCorners(0, 0, 0, 8, 4.0);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), v[0][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), v[0][1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), v[1][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), v[2][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), v[2][1], 1e-5);

    // Quad width matches thickness (diagonal case).
    const d = UICanvas.lineCorners(0, 0, 3, 4, 2.0);
    const w0x = d[1][0] - d[0][0];
    const w0y = d[1][1] - d[0][1];
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), @sqrt(w0x * w0x + w0y * w0y), 1e-5);

    // Degenerate segment collapses to the endpoints.
    const z = UICanvas.lineCorners(5, 5, 5, 5, 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), z[0][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), z[2][0], 1e-5);
}

test "UICanvas dropdownItemRect" {
    const btn: [4]f32 = .{ 10, 20, 120, 28 };
    const r0 = UICanvas.dropdownItemRect(btn, 24, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 10), r0[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 48), r0[1], 1e-5); // stacked below: 20 + 28
    try std.testing.expectApproxEqAbs(@as(f32, 120), r0[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 24), r0[3], 1e-5);

    const r2 = UICanvas.dropdownItemRect(btn, 24, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 48 + 2 * 24), r2[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 120), r2[2], 1e-5);

    // Shared item height: draw + hit-test geometry always agree.
    try std.testing.expectApproxEqAbs(@as(f32, 24.0), UICanvas.dropdownItemHeight(16.0), 1e-5);
}

test "UICanvas dropdownHit" {
    const btn: [4]f32 = .{ 10, 20, 120, 28 };
    const item_h: f32 = 24; // open list spans y = 48..120 for count 3
    try std.testing.expectEqual(@as(?usize, 0), UICanvas.dropdownHit(btn, item_h, 3, 50, 48)); // top edge
    try std.testing.expectEqual(@as(?usize, 0), UICanvas.dropdownHit(btn, item_h, 3, 50, 60));
    try std.testing.expectEqual(@as(?usize, 1), UICanvas.dropdownHit(btn, item_h, 3, 50, 72)); // row boundary -> next row
    try std.testing.expectEqual(@as(?usize, 2), UICanvas.dropdownHit(btn, item_h, 3, 50, 119));
    try std.testing.expectEqual(@as(?usize, 2), UICanvas.dropdownHit(btn, item_h, 3, 50, 120)); // bottom edge inclusive
    try std.testing.expectEqual(@as(?usize, 0), UICanvas.dropdownHit(btn, item_h, 3, 10, 60)); // x edges inclusive
    try std.testing.expectEqual(@as(?usize, 2), UICanvas.dropdownHit(btn, item_h, 3, 130, 100));

    // The closed button is NOT part of the list hit area.
    try std.testing.expectEqual(@as(?usize, null), UICanvas.dropdownHit(btn, item_h, 3, 50, 30));
    try std.testing.expectEqual(@as(?usize, null), UICanvas.dropdownHit(btn, item_h, 3, 50, 47.9));
    // Outside the list: x miss, below the list, empty list, degenerate height.
    try std.testing.expectEqual(@as(?usize, null), UICanvas.dropdownHit(btn, item_h, 3, 9, 60));
    try std.testing.expectEqual(@as(?usize, null), UICanvas.dropdownHit(btn, item_h, 3, 131, 60));
    try std.testing.expectEqual(@as(?usize, null), UICanvas.dropdownHit(btn, item_h, 3, 50, 121));
    try std.testing.expectEqual(@as(?usize, null), UICanvas.dropdownHit(btn, item_h, 0, 50, 60));
    try std.testing.expectEqual(@as(?usize, null), UICanvas.dropdownHit(btn, 0, 3, 50, 60));
}

test "UICanvas scrollClamp" {
    var s = ScrollState{ .offset = 0, .content_h = 500, .view_h = 200 }; // max 300
    UICanvas.scrollClamp(&s, 100);
    try std.testing.expectApproxEqAbs(@as(f32, 100), s.offset, 1e-5);
    UICanvas.scrollClamp(&s, 500); // clamp at the bottom
    try std.testing.expectApproxEqAbs(@as(f32, 300), s.offset, 1e-5);
    UICanvas.scrollClamp(&s, -1000); // clamp at the top
    try std.testing.expectApproxEqAbs(@as(f32, 0), s.offset, 1e-5);

    // Content smaller than the view: no scrolling, offset resets to 0.
    var small = ScrollState{ .offset = 50, .content_h = 100, .view_h = 200 };
    UICanvas.scrollClamp(&small, 10);
    try std.testing.expectApproxEqAbs(@as(f32, 0), small.offset, 1e-5);

    // scrollOffsetForItem: visible item keeps the offset, hidden item scrolls minimally.
    try std.testing.expectApproxEqAbs(@as(f32, 100), UICanvas.scrollOffsetForItem(100, 150, 20, 200, 500), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10), UICanvas.scrollOffsetForItem(100, 10, 20, 200, 500), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 220), UICanvas.scrollOffsetForItem(100, 400, 20, 200, 500), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), UICanvas.scrollOffsetForItem(100, 400, 20, 200, 100), 1e-5);
}

test "UICanvas scrollbarThumbRect" {
    const track: [4]f32 = .{ 0, 0, 12, 200 };
    // Content fits: thumb covers the full track.
    const full = UICanvas.scrollbarThumbRect(track, 100, 200, 0);
    try std.testing.expectApproxEqAbs(track[0], full[0], 1e-5);
    try std.testing.expectApproxEqAbs(track[1], full[1], 1e-5);
    try std.testing.expectApproxEqAbs(track[2], full[2], 1e-5);
    try std.testing.expectApproxEqAbs(track[3], full[3], 1e-5);

    // Half visible: half-height thumb, top at offset 0...
    const top = UICanvas.scrollbarThumbRect(track, 400, 200, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 100), top[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), top[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 12), top[2], 1e-5);
    // ...pinned to the track bottom at max offset.
    const bottom = UICanvas.scrollbarThumbRect(track, 400, 200, 200);
    try std.testing.expectApproxEqAbs(@as(f32, 100), bottom[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 100), bottom[1], 1e-5); // 200 - 100

    // Tiny view ratio: thumb clamped to the 16px minimum.
    const tiny = UICanvas.scrollbarThumbRect(track, 4000, 200, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 16), tiny[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), tiny[1], 1e-5);
}

test "UICanvas TextInput editing" {
    var st = TextInputState{};
    try std.testing.expect(st.insertChar('h'));
    try std.testing.expect(st.insertChar('i'));
    try std.testing.expectEqualStrings("hi", st.text());
    try std.testing.expectEqual(@as(usize, 2), st.cursor);

    _ = st.moveLeft();
    try std.testing.expect(st.insertChar('e'));
    try std.testing.expectEqualStrings("hei", st.text());
    _ = st.moveRight();
    try std.testing.expectEqual(@as(usize, 3), st.cursor);

    try std.testing.expect(st.backspace());
    try std.testing.expectEqualStrings("he", st.text());
    st.home();
    try std.testing.expect(st.deleteForward());
    try std.testing.expectEqualStrings("e", st.text());
    st.end();
    try std.testing.expectEqual(@as(usize, 1), st.cursor);

    // Edges are no-ops reporting false.
    st.home();
    try std.testing.expect(!st.backspace());
    try std.testing.expect(!st.moveLeft());
    st.end();
    try std.testing.expect(!st.deleteForward());
    try std.testing.expect(!st.moveRight());

    // setText replaces the buffer and moves the cursor to the end.
    st.setText("hello");
    try std.testing.expectEqualStrings("hello", st.text());
    try std.testing.expectEqual(@as(usize, 5), st.cursor);
}

test "UICanvas TextInput UTF-8 boundaries" {
    var st = TextInputState{};
    try std.testing.expect(st.insertChar('ж')); // U+0436, 2 bytes in UTF-8
    try std.testing.expectEqual(@as(usize, 2), st.len);
    try std.testing.expectEqual(@as(usize, 2), st.cursor);

    // Cursor moves step over the whole codepoint, never splitting it.
    try std.testing.expect(st.moveLeft());
    try std.testing.expectEqual(@as(usize, 0), st.cursor);
    try std.testing.expect(st.moveRight());
    try std.testing.expectEqual(@as(usize, 2), st.cursor);

    // Backspace removes both bytes at once.
    try std.testing.expect(st.backspace());
    try std.testing.expectEqual(@as(usize, 0), st.len);
    try std.testing.expect(!st.backspace());

    // deleteForward removes a whole multibyte codepoint ("aжb" -> "ab").
    st.setText("aжb");
    try std.testing.expectEqual(@as(usize, 4), st.len);
    st.home();
    _ = st.moveRight(); // cursor 1, right before ж
    try std.testing.expect(st.deleteForward());
    try std.testing.expectEqualStrings("ab", st.text());

    // Truncation never splits a codepoint: 127 x 'a' + 2 x 'ж' (131 bytes)
    // keeps exactly the 127 ASCII bytes.
    var big: [140]u8 = undefined;
    @memset(big[0..127], 'a');
    big[127] = 0xD0;
    big[128] = 0xB6; // ж
    big[129] = 0xD0;
    big[130] = 0xB6; // ж
    st.setText(big[0..131]);
    try std.testing.expectEqual(@as(usize, 127), st.len);
    try std.testing.expectEqual(@as(usize, 127), st.cursor);
    try std.testing.expectEqual(@as(u8, 'a'), st.text()[126]);

    // Full buffer rejects further input without modification.
    st.setText(big[0..128]);
    try std.testing.expectEqual(@as(usize, 128), st.len);
    try std.testing.expect(!st.insertChar('b'));
    try std.testing.expect(!st.insertChar('ж'));
    try std.testing.expectEqual(@as(usize, 128), st.len);
}

test "UICanvas measureText consistency" {
    // Monospace advance: measureText matches the per-char accumulation
    // drawText (and the drawTextInput cursor) uses.
    const a = UICanvas.measureText("a", 16.0);
    const ab = UICanvas.measureText("ab", 16.0);
    try std.testing.expectApproxEqAbs(a.x * 2.0, ab.x, 1e-5);
    try std.testing.expectApproxEqAbs(a.y, ab.y, 1e-5);
    const empty = UICanvas.measureText("", 16.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), empty.x, 1e-5);

    // Dropdown list total height = count * shared item height.
    const ih = UICanvas.dropdownItemHeight(16.0);
    try std.testing.expectApproxEqAbs(@as(f32, 24.0), ih, 1e-5);
    const btn: [4]f32 = .{ 0, 0, 100, 20 };
    const last = UICanvas.dropdownItemRect(btn, ih, 3);
    try std.testing.expectApproxEqAbs(@as(f32, 20 + 3 * 24), last[1], 1e-5);
}
