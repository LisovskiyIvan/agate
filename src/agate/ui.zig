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

// Style system modules (phase 2): shared value types, CSS-subset theme
// parsing, and the pure transition math. The public type names below are
// aliases into them so facade callers keep their old paths.
const ui_types = @import("ui/types.zig");
const ui_theme_mod = @import("ui/theme.zig");
const ui_transition = @import("ui/transition.zig");
const css_parser = @import("ui/css_parser.zig");
const ui_layout = @import("ui/layout.zig");
const upload_meter = @import("gpu_upload_meter.zig");

// Re-exports from layout module
pub const UISize = ui_layout.UISize;
pub const UIEdges = ui_layout.UIEdges;
pub const UIAnchor = ui_layout.UIAnchor;
pub const UIDock = ui_layout.UIDock;
pub const anchorRect = ui_layout.anchorRect;
pub const dockRect = ui_layout.dockRect;
pub const FlexDirection = ui_layout.FlexDirection;
pub const JustifyContent = ui_layout.JustifyContent;
pub const AlignItems = ui_layout.AlignItems;
pub const LayoutItem = ui_layout.LayoutItem;
pub const solveFlex = ui_layout.solveFlex;
pub const GridTrack = ui_layout.GridTrack;
pub const solveGridTracks = ui_layout.solveGridTracks;
pub const AdvancedGridSpec = ui_layout.AdvancedGridSpec;

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

    // Style system state (see the Styling section below the canvas).
    /// Default styles per widget kind; overridden wholesale by assignment
    /// (theme = "all widgets at once") or per kind (`canvas.theme.button = ...`).
    theme: UITheme = UITheme.defaults(),
    /// Named style classes ("panel.dark"); name slices are caller-owned
    /// (typically comptime literals — immediate mode never copies strings).
    style_classes: [max_style_classes]UIStyleClass = [_]UIStyleClass{.{ .name = "", .set = .{} }} ** max_style_classes,
    style_class_count: usize = 0,
    /// Retained per-widget style transitions (see resolveAnimatedStyle).
    style_transitions: [max_style_transitions]UIStyleTransition = [_]UIStyleTransition{.{}} ** max_style_transitions,
    /// Canvas clock for style transitions; advanced by begin().
    style_time_ms: f64 = 0,
    /// Frame delta begin() adds to style_time_ms. Fixed at 60 Hz by default
    /// (immediate-mode callers have no global clock); variable-rate callers
    /// set it before begin() each frame.
    frame_dt_ms: f32 = 1000.0 / 60.0,
    /// Mouse input state for immediate-mode layout widgets (button, slider, checkbox, etc.)
    mouse_pos: [2]f32 = .{ -1000.0, -1000.0 },
    mouse_down: bool = false,
    mouse_clicked: bool = false,

    pub fn setInput(self: *UICanvas, mx: f32, my: f32, is_down: bool, is_clicked: bool) void {
        self.mouse_pos = .{ mx, my };
        self.mouse_down = is_down;
        self.mouse_clicked = is_clicked;
    }

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
        // The frame boundary is the transition clock tick: no separate
        // update call for callers to forget.
        self.style_time_ms += self.frame_dt_ms;
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

    // ------------------------------------------------------------------
    // Styled rendering (CSS-like cascade; types declared below the canvas)
    // ------------------------------------------------------------------

    /// Registers (or replaces) a named style class. Registration order is
    /// irrelevant; replacing keeps the original slot.
    pub fn setStyleClass(self: *UICanvas, name: []const u8, set: UIStyleSet) void {
        for (self.style_classes[0..self.style_class_count]) |*sc| {
            if (std.mem.eql(u8, sc.name, name)) {
                sc.set = set;
                return;
            }
        }
        if (self.style_class_count >= max_style_classes) return;
        self.style_classes[self.style_class_count] = .{ .name = name, .set = set };
        self.style_class_count += 1;
    }

    /// Looks up a registered class by name.
    pub fn styleClass(self: *const UICanvas, name: []const u8) ?UIStyleSet {
        for (self.style_classes[0..self.style_class_count]) |sc| {
            if (std.mem.eql(u8, sc.name, name)) return sc.set;
        }
        return null;
    }

    /// Full cascade resolution: theme default for the widget kind, then the
    /// named class (if registered), then the per-call override. Each layer's
    /// state delta applies within that layer, so e.g. a hover delta from the
    /// theme still shows through a class that only sets a border.
    pub fn resolveStyle(self: *const UICanvas, request: UIStyleRequest) UIStyle {
        const set = self.theme.setFor(request.kind).*;
        var s = set.resolve(UIStyle{}, request.state);
        if (request.class) |name| {
            if (self.styleClass(name)) |cs| s = cs.resolve(s, request.state);
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
    pub fn resolveAnimatedStyle(self: *UICanvas, kind: UIStyleKind, opts: UIStyledOptions) UIStyle {
        const target = self.resolveStyle(.{ .kind = kind, .class = opts.class, .override = opts.style, .state = opts.state });
        const key = opts.anim_key orelse return target;
        const cfg = opts.transition orelse self.cascadeTransition(kind, opts.class);
        return self.animateStyle(ui_types.animKeyHash(key), target, cfg);
    }

    /// Transition config from the cascade (per-call override already won in
    /// resolveAnimatedStyle): a class wins when it defines a duration,
    /// otherwise the theme kind's set provides it.
    fn cascadeTransition(self: *UICanvas, kind: UIStyleKind, class: ?[]const u8) TransitionOptions {
        if (class) |name| {
            if (self.styleClass(name)) |cs| {
                if (cs.transition.duration_ms > 0.0) return cs.transition;
            }
        }
        return self.theme.setFor(kind).transition;
    }

    /// Advances (or starts) the retained transition `key` toward `target`
    /// and returns the style to draw this frame.
    fn animateStyle(self: *UICanvas, key: u64, target: UIStyle, cfg: TransitionOptions) UIStyle {
        const now = self.style_time_ms;
        if (cfg.duration_ms <= 0.0) {
            // Instant config: snap and release any retained transition.
            if (self.styleTransitionSlot(key)) |s| s.used = false;
            return target;
        }
        const s = self.styleTransitionSlot(key) orelse self.acquireStyleSlot();
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

    fn styleTransitionSlot(self: *UICanvas, key: u64) ?*UIStyleTransition {
        for (&self.style_transitions) |*s| {
            if (s.used and s.key == key) return s;
        }
        return null;
    }

    /// Storage for a new transition: the first free slot, else the least
    /// recently touched entry is recycled (UI screens animate few widgets;
    /// LRU over a fixed table keeps the canvas allocation-free).
    fn acquireStyleSlot(self: *UICanvas) *UIStyleTransition {
        var oldest = &self.style_transitions[0];
        for (&self.style_transitions) |*s| {
            if (!s.used) return s;
            if (s.last_touch_ms < oldest.last_touch_ms) oldest = s;
        }
        return oldest;
    }

    /// Installs a parsed CSS theme: replaces the kind defaults wholesale and
    /// registers/updates every parsed class (per class name, latest parse
    /// wins). Diagnostics are the caller's to inspect — a parse with errors
    /// still yields a usable partial theme.
    pub fn applyCssTheme(self: *UICanvas, parsed: css_parser.CssTheme) void {
        self.theme = parsed.theme;
        for (parsed.classes) |c| self.setStyleClass(c.name, c.set);
    }

    /// Renders one resolved style: shadow, background (flat or gradient
    /// bands) and border. Fully transparent styles draw nothing, so
    /// containers can render their style unconditionally.
    pub fn drawStyleRect(self: *UICanvas, rect: [4]f32, style: UIStyle) void {
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
                self.drawRectRoundedFill(
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
                self.drawRect(x, y + f0 * h, w, bh, mulAlpha(Color4.lerp(g.top, g.bottom, fm), op));
            }
        } else {
            const bg = mulAlpha(style.background, op);
            if (bg.a > 0.001) {
                self.drawRectRoundedFill(x, y, w, h, style.corner_radius, bg);
            }
        }

        const bc = mulAlpha(style.border_color, op);
        if (style.border_width > 0.0 and bc.a > 0.001) {
            self.drawRectRoundedOutline(x, y, w, h, style.corner_radius, style.border_width, bc);
        }
    }

    /// Fills a rounded rectangle with solid quads: one middle rect plus thin
    /// horizontal bands approximating the corner arcs (2px resolution).
    pub fn drawRectRoundedFill(self: *UICanvas, x: f32, y: f32, w: f32, h: f32, radius: f32, color: Color4) void {
        if (w <= 0.0 or h <= 0.0 or color.a <= 0.001) return;
        const r = std.math.clamp(radius, 0.0, @min(w * 0.5, h * 0.5));
        if (r <= 0.5) {
            self.drawRect(x, y, w, h, color);
            return;
        }
        self.drawRect(x, y + r, w, @max(h - 2.0 * r, 0.0), color);
        var t: f32 = 0.0;
        while (t < r) : (t += corner_band_px) {
            const bh = @min(corner_band_px, r - t);
            const dy = r - (t + bh * 0.5);
            const inset = r - @sqrt(@max(r * r - dy * dy, 0.0));
            self.drawRect(x + inset, y + t, @max(w - 2.0 * inset, 0.0), bh, color);
            self.drawRect(x + inset, y + h - t - bh, @max(w - 2.0 * inset, 0.0), bh, color);
        }
    }

    /// Strokes a rounded-rectangle border: four straight segments between
    /// the corner arcs plus per-band arc segments. The inner edge is
    /// concentric with the outer arc (inset by the border width); rows the
    /// inner arc does not reach are solid border strips.
    pub fn drawRectRoundedOutline(self: *UICanvas, x: f32, y: f32, w: f32, h: f32, radius: f32, thickness: f32, color: Color4) void {
        if (w <= 0.0 or h <= 0.0 or thickness <= 0.0 or color.a <= 0.001) return;
        const t = @min(thickness, @min(w * 0.5, h * 0.5));
        const r = std.math.clamp(radius, 0.0, @min(w * 0.5, h * 0.5));
        if (r <= 0.5) {
            self.drawRectOutline(x, y, w, h, t, color);
            return;
        }
        self.drawRect(x + r, y, @max(w - 2.0 * r, 0.0), t, color); // top
        self.drawRect(x + r, y + h - t, @max(w - 2.0 * r, 0.0), t, color); // bottom
        self.drawRect(x, y + r, t, @max(h - 2.0 * r, 0.0), color); // left
        self.drawRect(x + w - t, y + r, t, @max(h - 2.0 * r, 0.0), color); // right

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
                self.drawRect(x + o, y_top, sw, bh, color);
                self.drawRect(x + o, y_bot, sw, bh, color);
            } else {
                const bw = inner - o;
                self.drawRect(x + o, y_top, bw, bh, color);
                self.drawRect(x + w - inner, y_top, bw, bh, color);
                self.drawRect(x + o, y_bot, bw, bh, color);
                self.drawRect(x + w - inner, y_bot, bw, bh, color);
            }
        }
    }

    /// Styled panel: renders the resolved panel style into `rect`.
    pub fn drawStyledPanel(self: *UICanvas, rect: [4]f32, opts: UIStyledOptions) void {
        const s = self.resolveAnimatedStyle(.panel, opts);
        self.drawStyleRect(rect, s);
    }

    /// Styled button: resolved button style plus centered outlined text.
    /// Pairs with LayoutStack: `const r = ls.place(w, h); canvas.drawStyledButton("Ok", r, 14, .{});`
    pub fn drawStyledButton(self: *UICanvas, text: []const u8, rect: [4]f32, font_size: f32, opts: UIStyledOptions) void {
        const s = self.resolveAnimatedStyle(.button, opts);
        self.drawStyleRect(rect, s);
        const op = std.math.clamp(s.opacity, 0.0, 1.0);
        const text_w = @as(f32, @floatFromInt(text.len)) * font_size * 0.5;
        const tx = rect[0] + (rect[2] - text_w) * 0.5;
        const ty = rect[1] + (rect[3] - font_size) * 0.5;
        self.drawTextWithOutline(text, tx, ty, font_size, mulAlpha(s.text_color, op), 0.16);
    }

    /// Styled checkbox: box from the resolved style (state via `opts`),
    /// white check mark as in the legacy widget, optional label to the right.
    pub fn drawStyledCheckbox(self: *UICanvas, rect: [4]f32, checked: bool, label: ?[]const u8, label_size: f32, opts: UIStyledOptions) void {
        const s = self.resolveAnimatedStyle(.checkbox, opts);
        const op = std.math.clamp(s.opacity, 0.0, 1.0);
        const size = @min(rect[2], rect[3]);
        self.drawStyleRect(.{ rect[0], rect[1], size, size }, s);
        if (checked) {
            const m = size * 0.25;
            self.drawRect(rect[0] + m, rect[1] + m, size - 2.0 * m, size - 2.0 * m, mulAlpha(Color4.white, op));
        }
        if (label) |text| {
            const ty = rect[1] + (size - label_size) * 0.5;
            self.drawText(text, rect[0] + size + 8.0, ty, label_size, mulAlpha(s.text_color, op));
        }
    }

    /// Styled horizontal slider. Track from the resolved style, fill from
    /// the accent (style override wins over the theme accent). Returns the
    /// clamped value like the legacy widget.
    pub fn drawStyledSlider(self: *UICanvas, rect: [4]f32, value: f32, opts: UIStyledOptions) f32 {
        const s = self.resolveAnimatedStyle(.slider, opts);
        const v = std.math.clamp(value, 0.0, 1.0);
        const op = std.math.clamp(s.opacity, 0.0, 1.0);
        self.drawStyleRect(rect, s);
        if (v > 0.001) {
            self.drawRectRoundedFill(rect[0], rect[1], rect[2] * v, rect[3], s.corner_radius, mulAlpha(s.accent orelse self.theme.accent, op));
        }
        // Knob: small square centered on the fill edge (legacy look).
        const knob_size = @max(rect[3] + 6.0, 10.0);
        const cx = rect[0] + v * rect[2];
        const kx = if (rect[2] <= knob_size)
            rect[0] + (rect[2] - knob_size) * 0.5
        else
            std.math.clamp(cx - knob_size * 0.5, rect[0], rect[0] + rect[2] - knob_size);
        const ky = rect[1] + rect[3] * 0.5 - knob_size * 0.5;
        self.drawPanel(kx, ky, knob_size, knob_size, mulAlpha(Color4.new(0.52, 0.60, 0.72, 1.0), op), Color4.new(0.3, 0.36, 0.46, 0.9), 1.5);
        return v;
    }

    /// Styled badge: resolved style plus text at the badge padding. The
    /// rect is caller-provided (measure the text and use LayoutStack.place).
    pub fn drawStyledBadge(self: *UICanvas, text: []const u8, rect: [4]f32, font_size: f32, opts: UIStyledOptions) void {
        const s = self.resolveAnimatedStyle(.badge, opts);
        self.drawStyleRect(rect, s);
        const op = std.math.clamp(s.opacity, 0.0, 1.0);
        self.drawText(text, rect[0] + font_size * 0.4, rect[1] + font_size * 0.25, font_size, mulAlpha(s.text_color, op));
    }

    /// Bytes handed to sg by one rendered UI batch (clamped vertex prefix +
    /// full index list). Must stay in usize: the vertex cap times
    /// @sizeOf(UIVertex) overflows u16 arithmetic, so this must never run in
    /// the clamped vertex type.
    fn batchUploadBytes(vert_count: usize, index_count: usize) usize {
        return vert_count * @sizeOf(UIVertex) + index_count * @sizeOf(u16);
    }

    /// Uploads dynamic batch buffers and executes the UI render pass
    pub fn render(self: *UICanvas, screen_w: f32, screen_h: f32) void {
        if (self.vertices.items.len == 0 or self.indices.items.len == 0) return;
        if (screen_w <= 0.0 or screen_h <= 0.0) return;

        // usize on purpose: `@min(usize, u16)` resolves to u16 in Zig 0.16, and
        // 48 B/vertex would then overflow the u16 multiply at 1366 vertices.
        const vert_count: usize = @min(self.vertices.items.len, @as(usize, std.math.maxInt(u16)));
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
        // Учёт динамики: весь UI-батч кадра (вершины + u16-индексы).
        upload_meter.record(batchUploadBytes(vert_count, self.indices.items.len));

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

// CSS-subset theme parsing (see ui/css_parser.zig). `UITheme.parseCss` is
// the same parser as a method on the theme type; install a parsed theme on
// a canvas with UICanvas.applyCssTheme.
pub const CssTheme = css_parser.CssTheme;
pub const CssClass = css_parser.CssClass;
pub const CssDiag = css_parser.CssDiag;
pub const CssDiagKind = css_parser.CssDiagKind;
pub const parseCss = css_parser.parseCss;
pub const loadThemeFile = css_parser.loadThemeFile;

// ---------------------------------------------------------------------
// Styling (CSS-like, immediate-mode; no string parsing, no allocation)
// ---------------------------------------------------------------------
//
// The model follows the spirit of clay.h / Dear ImGui styles: concrete
// style values live in plain structs, and widgets resolve one concrete
// `UIStyle` per call through a cascade:
//
//   engine fallback < theme default (per widget kind) < named class
//                    < per-call override < state deltas applied per layer
//
// - `UIStyle` is fully resolved (all fields concrete).
// - `UIStyleOverride` is a partial (all fields optional; null = inherit).
// - `UIStyleSet` is a partial "normal" look plus optional per-state deltas
//   (the pseudo-class analog: :hover / :active / :focus / :disabled).
// - `UICanvas.theme` holds the defaults per widget kind; named classes are
//   registered on the canvas with `setStyleClass` and referenced by name.

// The style value types live in ui/types.zig, the theme in ui/theme.zig and
// the transition math in ui/transition.zig; the facade re-exports them under
// their historical names so callers keep one import path.
pub const UIState = ui_types.UIState;
pub const UIGradient = ui_types.UIGradient;
pub const UIShadow = ui_types.UIShadow;
pub const UIStyle = ui_types.UIStyle;
pub const UIStyleOverride = ui_types.UIStyleOverride;
pub const UIStyleSet = ui_types.UIStyleSet;
pub const TransitionOptions = ui_types.TransitionOptions;
pub const UIStyleKind = ui_types.UIStyleKind;
pub const UIStyledOptions = ui_types.UIStyledOptions;
pub const UIStyleRequest = ui_types.UIStyleRequest;
pub const UIStyleClass = ui_types.UIStyleClass;
pub const UIBoxStyle = ui_types.UIBoxStyle;
pub const max_style_classes = ui_types.max_style_classes;
pub const UITheme = ui_theme_mod.UITheme;
pub const UIStyleTransition = ui_transition.UIStyleTransition;
pub const max_style_transitions = ui_transition.max_style_transitions;
pub const lerpStyle = ui_transition.lerpStyle;
pub const styleEql = ui_transition.styleEql;

const mulAlpha = ui_types.mulAlpha;

/// Band height (px) for approximating rounded corner arcs with solid quads.
const corner_band_px: f32 = 2.0;

// ---------------------------------------------------------------------
// Layout containers (immediate-mode cursor; no retained tree)
// ---------------------------------------------------------------------
//
// `LayoutStack` is a small stack of container frames. A widget is placed
// with `place(w, h)` which returns the rect to draw it into and advances
// the container cursor — existing widgets keep their absolute-coordinate
// signatures, the caller just feeds them the returned rect. Nesting is
// placing a rect and opening the next container on it:
//
//   var ls = LayoutStack.init(canvas);
//   defer ls.reset();
//   if (ls.beginVStack(panel_rect, .{ .padding = 8, .spacing = 4 })) {
//       const b = ls.place(120, 24);
//       canvas.drawButton("Ok", b[0], b[1], b[2], b[3], 14, false, false);
//       const row = ls.place(120, 24);
//       _ = ls.beginHStack(row, .{ .spacing = 8 });
//       ...
//       ls.end();
//       ls.end();
//   }

/// Alignment along the main axis (or inside a grid cell).
pub const LayoutAlign = enum { start, center, end };

/// Alignment along the cross axis of a flow container. `stretch` expands
/// the widget to the container's inner cross extent.
pub const LayoutAlignCross = enum { start, center, end, stretch };

/// Offset of a `size` item inside an `extent` slot. Oversized items clamp
/// to the slot start (overflow never produces negative offsets, matching
/// the clamping spirit of the rest of the UI helpers).
pub fn layoutAlignOffset(extent: f32, size: f32, alignment: LayoutAlign) f32 {
    return switch (alignment) {
        .start => 0.0,
        .center => @max((extent - size) * 0.5, 0.0),
        .end => @max(extent - size, 0.0),
    };
}

/// Size of one grid track. Cells past the end of an explicit sizes slice
/// share the leftover extent (total minus explicit sizes minus spacing)
/// equally; without explicit sizes every track is an equal share.
pub fn gridExtentSize(sizes: ?[]const f32, count: usize, total: f32, spacing: f32, index: usize) f32 {
    const n = @max(count, 1);
    if (sizes) |s| {
        if (index < s.len and index < n) return @max(s[index], 0.0);
        var explicit: f32 = 0.0;
        for (s[0..@min(s.len, n)]) |v| explicit += @max(v, 0.0);
        const rest_cells = n - @min(s.len, n);
        if (rest_cells == 0) return 0.0;
        const leftover = @max(total - explicit - spacing * @as(f32, @floatFromInt(n - 1)), 0.0);
        return leftover / @as(f32, @floatFromInt(rest_cells));
    }
    const leftover = @max(total - spacing * @as(f32, @floatFromInt(n - 1)), 0.0);
    return leftover / @as(f32, @floatFromInt(n));
}

/// Start position of grid track `index` relative to the inner origin.
pub fn gridExtentOffset(sizes: ?[]const f32, count: usize, total: f32, spacing: f32, index: usize) f32 {
    var off: f32 = 0.0;
    var i: usize = 0;
    while (i < index) : (i += 1) {
        off += gridExtentSize(sizes, count, total, spacing, i) + spacing;
    }
    return off;
}

/// Pure grid geometry: cell rects for a columns x rows grid inside an inner
/// content rect. Also usable standalone (unit tests, one-off queries).
pub const LayoutGridSpec = struct {
    /// Content rect with padding already applied.
    inner: [4]f32,
    columns: usize,
    rows: usize,
    spacing: f32 = 0.0,
    column_widths: ?[]const f32 = null,
    row_heights: ?[]const f32 = null,
    align_main: LayoutAlign = .start,
    align_cross: LayoutAlign = .start,

    /// Raw cell rect (before per-item alignment). Out-of-range col/row
    /// clamp to the last track so overflow stays well-defined.
    pub fn cellRect(self: LayoutGridSpec, col_in: usize, row_in: usize) [4]f32 {
        const cols = @max(self.columns, 1);
        const rows = @max(self.rows, 1);
        const col = @min(col_in, cols - 1);
        const row = @min(row_in, rows - 1);
        const x = self.inner[0] + gridExtentOffset(self.column_widths, cols, self.inner[2], self.spacing, col);
        const w = gridExtentSize(self.column_widths, cols, self.inner[2], self.spacing, col);
        const y = self.inner[1] + gridExtentOffset(self.row_heights, rows, self.inner[3], self.spacing, row);
        const h = gridExtentSize(self.row_heights, rows, self.inner[3], self.spacing, row);
        return .{ x, y, w, h };
    }

    /// Cell rect with a w x h widget aligned inside it (row-major `index`).
    pub fn placedRect(self: LayoutGridSpec, index: usize, w: f32, h: f32) [4]f32 {
        const cols = @max(self.columns, 1);
        const cell = self.cellRect(index % cols, index / cols);
        var r = cell;
        r[0] += layoutAlignOffset(cell[2], w, self.align_main);
        r[2] = w;
        r[1] += layoutAlignOffset(cell[3], h, self.align_cross);
        r[3] = h;
        return r;
    }
};

fn effectivePadding(layout_pad: f32, style_pad: f32) f32 {
    return @max(@max(layout_pad, style_pad), 0.0);
}

pub const LayoutFlowOptions = struct {
    padding: f32 = 0.0,
    padding_edges: ?UIEdges = null,
    spacing: f32 = 0.0,
    align_cross: LayoutAlignCross = .start,
    class: ?[]const u8 = null,
    style: ?UIStyleOverride = null,

    pub fn getEdges(self: LayoutFlowOptions, style_pad: f32) UIEdges {
        if (self.padding_edges) |e| return e;
        const p = effectivePadding(self.padding, style_pad);
        return UIEdges.all(p);
    }
};

pub const LayoutGridOptions = struct {
    padding: f32 = 0.0,
    padding_edges: ?UIEdges = null,
    spacing: f32 = 0.0,
    columns: usize,
    rows: usize,
    column_widths: ?[]const f32 = null,
    row_heights: ?[]const f32 = null,
    align_main: LayoutAlign = .start,
    align_cross: LayoutAlign = .start,
    class: ?[]const u8 = null,
    style: ?UIStyleOverride = null,

    pub fn getEdges(self: LayoutGridOptions, style_pad: f32) UIEdges {
        if (self.padding_edges) |e| return e;
        const p = effectivePadding(self.padding, style_pad);
        return UIEdges.all(p);
    }
};

pub const LayoutFlexOptions = struct {
    direction: FlexDirection = .row,
    padding: f32 = 0.0,
    padding_edges: ?UIEdges = null,
    spacing: f32 = 0.0,
    align_cross: LayoutAlignCross = .start,
    class: ?[]const u8 = null,
    style: ?UIStyleOverride = null,

    pub fn getEdges(self: LayoutFlexOptions, style_pad: f32) UIEdges {
        if (self.padding_edges) |e| return e;
        const p = effectivePadding(self.padding, style_pad);
        return UIEdges.all(p);
    }
};

pub const LayoutLabelOptions = struct {
    font_size: f32 = 13.0,
    color: Color4 = Color4.white,
    width: UISize = .auto,
    height: UISize = .auto,
    outline_width: f32 = 0.16,
};

pub const LayoutButtonOptions = struct {
    width: UISize = .auto,
    height: UISize = .px(28.0),
    font_size: f32 = 12.0,
    is_hovered: ?bool = null,
    is_pressed: ?bool = null,
    class: ?[]const u8 = null,
    style: ?UIStyleOverride = null,
};

pub const LayoutCheckboxOptions = struct {
    size: f32 = 18.0,
    label_size: f32 = 12.5,
    is_hovered: ?bool = null,
    class: ?[]const u8 = null,
    style: ?UIStyleOverride = null,
};

pub const LayoutSliderOptions = struct {
    width: UISize = .fill,
    height: UISize = .px(20.0),
    is_hovered: ?bool = null,
    is_dragging: ?bool = null,
    class: ?[]const u8 = null,
    style: ?UIStyleOverride = null,
};

pub const LayoutProgressOptions = struct {
    width: UISize = .fill,
    height: UISize = .px(16.0),
    bg_color: Color4 = Color4.new(0.1, 0.12, 0.18, 0.8),
    fill_color: Color4 = Color4.new(0.2, 0.6, 1.0, 0.95),
};

pub const LayoutDividerOptions = struct {
    thickness: f32 = 1.0,
    color: Color4 = Color4.new(0.3, 0.4, 0.5, 0.5),
};

pub const LayoutBadgeOptions = struct {
    font_size: f32 = 11.0,
    bg_color: Color4 = Color4.new(0.14, 0.25, 0.4, 0.9),
    text_color: Color4 = Color4.white,
};

pub const LayoutStack = struct {
    /// Fixed frame stack: UI nesting is shallow; exceeding the cap makes
    /// `begin*` report false instead of growing memory.
    pub const max_depth = 16;

    const FrameKind = enum { hstack, vstack, grid };

    const Frame = struct {
        kind: FrameKind,
        // Outer container rect (clamps for overflow).
        x: f32 = 0.0,
        y: f32 = 0.0,
        w: f32 = 0.0,
        h: f32 = 0.0,
        // Inner content rect (padding applied).
        ix: f32 = 0.0,
        iy: f32 = 0.0,
        iw: f32 = 0.0,
        ih: f32 = 0.0,
        // Flow frames: next main-axis origin (absolute).
        cursor: f32 = 0.0,
        spacing: f32 = 0.0,
        align_cross: LayoutAlignCross = .start,
        // Grid frame.
        cell: usize = 0,
        grid: LayoutGridSpec = .{ .inner = .{ 0, 0, 0, 0 }, .columns = 1, .rows = 1 },
    };

    canvas: *UICanvas,
    frames: [max_depth]Frame = undefined,
    depth: usize = 0,

    pub fn init(canvas: *UICanvas) LayoutStack {
        return .{ .canvas = canvas };
    }

    /// Drops all open frames (start of a new UI frame).
    pub fn reset(self: *LayoutStack) void {
        self.depth = 0;
    }

    pub fn beginHStack(self: *LayoutStack, rect: [4]f32, opts: LayoutFlowOptions) bool {
        return self.beginFlow(rect, opts, .hstack);
    }

    pub fn beginVStack(self: *LayoutStack, rect: [4]f32, opts: LayoutFlowOptions) bool {
        return self.beginFlow(rect, opts, .vstack);
    }

    pub fn beginFlex(self: *LayoutStack, rect: [4]f32, opts: LayoutFlexOptions) bool {
        return self.beginFlow(rect, .{
            .padding = opts.padding,
            .padding_edges = opts.padding_edges,
            .spacing = opts.spacing,
            .align_cross = opts.align_cross,
            .class = opts.class,
            .style = opts.style,
        }, if (opts.direction.isRow()) .hstack else .vstack);
    }

    pub fn beginGrid(self: *LayoutStack, rect: [4]f32, opts: LayoutGridOptions) bool {
        if (self.depth >= max_depth) return false;
        const s = self.resolveContainerStyle(opts.class, opts.style);
        const pad = opts.getEdges(s.padding);
        self.canvas.drawStyleRect(rect, s);
        const inner_w = @max(rect[2] - pad.hTotal(), 0.0);
        const inner_h = @max(rect[3] - pad.vTotal(), 0.0);
        const f = Frame{
            .kind = .grid,
            .x = rect[0],
            .y = rect[1],
            .w = rect[2],
            .h = rect[3],
            .ix = rect[0] + pad.left,
            .iy = rect[1] + pad.top,
            .iw = inner_w,
            .ih = inner_h,
            .cell = 0,
            .grid = .{
                .inner = .{ rect[0] + pad.left, rect[1] + pad.top, inner_w, inner_h },
                .columns = @max(opts.columns, 1),
                .rows = @max(opts.rows, 1),
                .spacing = @max(opts.spacing, 0.0),
                .column_widths = opts.column_widths,
                .row_heights = opts.row_heights,
                .align_main = opts.align_main,
                .align_cross = opts.align_cross,
            },
        };
        self.frames[self.depth] = f;
        self.depth += 1;
        return true;
    }

    pub fn end(self: *LayoutStack) void {
        if (self.depth > 0) self.depth -= 1;
    }
    pub const endFlow = end;
    pub const endGrid = end;

    /// Returns the inner content bounds [x, y, w, h] of the current container.
    pub fn innerRect(self: *const LayoutStack) [4]f32 {
        if (self.depth == 0) return .{ 0, 0, 0, 0 };
        const f = &self.frames[self.depth - 1];
        return .{ f.ix, f.iy, f.iw, f.ih };
    }

    /// Places the next widget with the container's cross alignment.
    /// Returns the rect to draw the widget into.
    pub fn place(self: *LayoutStack, w: f32, h: f32) [4]f32 {
        return self.placeAligned(w, h, null);
    }

    /// `place` with a per-call cross-axis alignment override.
    pub fn placeAligned(self: *LayoutStack, w: f32, h: f32, align_cross: ?LayoutAlignCross) [4]f32 {
        if (self.depth == 0) return .{ 0, 0, 0, 0 };
        const f = &self.frames[self.depth - 1];
        switch (f.kind) {
            .hstack, .vstack => {
                const cross = align_cross orelse f.align_cross;
                const r = self.flowRect(w, h, cross);
                // The cursor advances by the requested size + spacing even
                // when the returned rect got clamped: placement stays
                // deterministic no matter what overflow happened.
                f.cursor += (if (f.kind == .hstack) w else h) + f.spacing;
                return clampToFrame(r, f);
            },
            .grid => {
                const r = f.grid.placedRect(f.cell, w, h);
                f.cell += 1;
                return clampToFrame(r, f);
            },
        }
    }

    /// Places a widget dimensioned via UISize (fixed, percent, flex, or auto).
    pub fn placeSize(self: *LayoutStack, w: UISize, h: UISize) [4]f32 {
        return self.placeSizeWithAuto(w, h, 0.0, 0.0);
    }

    /// Places a widget dimensioned via UISize with explicit auto content dimensions.
    pub fn placeSizeWithAuto(self: *LayoutStack, w: UISize, h: UISize, auto_w: f32, auto_h: f32) [4]f32 {
        if (self.depth == 0) return .{ 0, 0, 0, 0 };
        const f = &self.frames[self.depth - 1];
        const rem_main = if (f.kind == .hstack) @max(f.ix + f.iw - f.cursor, 0.0) else @max(f.iy + f.ih - f.cursor, 0.0);

        const actual_w: f32 = switch (w) {
            .fixed => |v| v,
            .percent => |p| f.iw * (p * 0.01),
            .auto => auto_w,
            .flex => if (f.kind == .hstack) rem_main else f.iw,
        };

        const actual_h: f32 = switch (h) {
            .fixed => |v| v,
            .percent => |p| f.ih * (p * 0.01),
            .auto => auto_h,
            .flex => if (f.kind == .vstack) rem_main else f.ih,
        };

        return self.place(actual_w, actual_h);
    }

    /// Places a widget that expands across remaining space along the main axis.
    pub fn placeFlex(self: *LayoutStack, weight: f32) [4]f32 {
        _ = weight;
        return self.placeSize(.fill, .fill);
    }

    /// Advances the layout cursor by taking up all remaining main-axis space.
    pub fn spacer(self: *LayoutStack) [4]f32 {
        return self.spacerWeight(1.0);
    }

    pub fn spacerWeight(self: *LayoutStack, weight: f32) [4]f32 {
        _ = weight;
        if (self.depth == 0) return .{ 0, 0, 0, 0 };
        const f = &self.frames[self.depth - 1];
        if (f.kind == .hstack) {
            const rem = @max(f.ix + f.iw - f.cursor, 0.0);
            return self.place(rem, 0.0);
        } else if (f.kind == .vstack) {
            const rem = @max(f.iy + f.ih - f.cursor, 0.0);
            return self.place(0.0, rem);
        }
        return .{ 0, 0, 0, 0 };
    }

    /// Places a widget in a grid container spanning `col_span` columns and `row_span` rows.
    pub fn placeGridSpan(
        self: *LayoutStack,
        col: usize,
        row: usize,
        col_span: usize,
        row_span: usize,
        w: UISize,
        h: UISize,
    ) [4]f32 {
        if (self.depth == 0) return .{ 0, 0, 0, 0 };
        const f = &self.frames[self.depth - 1];
        if (f.kind != .grid) return .{ 0, 0, 0, 0 };

        const c0 = f.grid.cellRect(col, row);
        const cols = @max(f.grid.columns, 1);
        const rows = @max(f.grid.rows, 1);
        const max_c = @min(col + @max(col_span, 1) - 1, cols - 1);
        const max_r = @min(row + @max(row_span, 1) - 1, rows - 1);
        const c1 = f.grid.cellRect(max_c, max_r);

        const spanned_w = (c1[0] + c1[2]) - c0[0];
        const spanned_h = (c1[1] + c1[3]) - c0[1];

        const actual_w = w.resolve(spanned_w, spanned_w);
        const actual_h = h.resolve(spanned_h, spanned_h);

        var r: [4]f32 = .{ c0[0], c0[1], actual_w, actual_h };
        r[0] += layoutAlignOffset(spanned_w, actual_w, f.grid.align_main);
        r[1] += layoutAlignOffset(spanned_h, actual_h, f.grid.align_cross);
        return clampToFrame(r, f);
    }

    /// Positions a widget relative to the container frame using 9-point anchor.
    pub fn anchor(self: *LayoutStack, w: f32, h: f32, anchor_pt: UIAnchor, margin: UIEdges) [4]f32 {
        if (self.depth == 0) return .{ 0, 0, 0, 0 };
        const f = &self.frames[self.depth - 1];
        return anchorRect(.{ f.ix, f.iy, f.iw, f.ih }, w, h, anchor_pt, margin);
    }

    /// Docks an element to a side of the current frame and shrinks the remaining inner area.
    pub fn dock(self: *LayoutStack, dock_side: UIDock, size: f32, margin: UIEdges) [4]f32 {
        if (self.depth == 0) return .{ 0, 0, 0, 0 };
        const f = &self.frames[self.depth - 1];
        var current: [4]f32 = .{ f.ix, f.iy, f.iw, f.ih };
        const r = dockRect(&current, dock_side, size, margin);
        f.ix = current[0];
        f.iy = current[1];
        f.iw = current[2];
        f.ih = current[3];
        return r;
    }

    // ========================================================================
    // Immediate-Mode Layout Widgets
    // ========================================================================

    /// Places and renders a label widget inside the current container.
    pub fn label(self: *LayoutStack, text: []const u8, opts: LayoutLabelOptions) void {
        const auto_w = @as(f32, @floatFromInt(text.len)) * opts.font_size * 0.5;
        const auto_h = opts.font_size;
        const r = self.placeSizeWithAuto(opts.width, opts.height, auto_w, auto_h);
        const tx = r[0] + @max((r[2] - auto_w) * 0.5, 0.0);
        const ty = r[1] + @max((r[3] - auto_h) * 0.5, 0.0);
        self.canvas.drawTextWithOutline(text, tx, ty, opts.font_size, opts.color, opts.outline_width);
    }

    /// Places and renders a button widget. Returns true if clicked.
    pub fn button(self: *LayoutStack, text: []const u8, opts: LayoutButtonOptions) bool {
        const auto_w = @as(f32, @floatFromInt(text.len)) * opts.font_size * 0.5 + 24.0;
        const auto_h = opts.font_size + 14.0;
        const r = self.placeSizeWithAuto(opts.width, opts.height, auto_w, auto_h);
        const hov = opts.is_hovered orelse UICanvas.isPointInRect(self.canvas.mouse_pos[0], self.canvas.mouse_pos[1], r[0], r[1], r[2], r[3]);
        const press = opts.is_pressed orelse (hov and self.canvas.mouse_down);

        if (opts.class != null or opts.style != null) {
            const state = UIState.fromFlags(hov, press, false, false);
            self.canvas.drawStyledButton(text, r, opts.font_size, .{
                .class = opts.class,
                .style = opts.style,
                .state = state,
            });
        } else {
            self.canvas.drawButton(text, r[0], r[1], r[2], r[3], opts.font_size, hov, press);
        }

        return hov and self.canvas.mouse_clicked;
    }

    /// Places and renders a checkbox widget. Toggles state and returns true if clicked.
    pub fn checkbox(self: *LayoutStack, label_text: ?[]const u8, checked: *bool, opts: LayoutCheckboxOptions) bool {
        const text_w = if (label_text) |t| @as(f32, @floatFromInt(t.len)) * opts.label_size * 0.5 + 8.0 else 0.0;
        const total_w = opts.size + text_w;
        const r = self.place(total_w, @max(opts.size, opts.label_size));
        const hov = opts.is_hovered orelse UICanvas.isPointInRect(self.canvas.mouse_pos[0], self.canvas.mouse_pos[1], r[0], r[1], r[2], r[3]);

        if (opts.class != null or opts.style != null) {
            const state = UIState.fromFlags(hov, false, false, false);
            self.canvas.drawStyledCheckbox(r, checked.*, label_text, opts.label_size, .{
                .class = opts.class,
                .style = opts.style,
                .state = state,
            });
        } else {
            self.canvas.drawCheckbox(r[0], r[1], opts.size, checked.*, hov, label_text, opts.label_size);
        }

        if (hov and self.canvas.mouse_clicked) {
            checked.* = !checked.*;
            return true;
        }
        return false;
    }

    /// Places and renders an interactive slider widget. Returns the new value.
    pub fn slider(self: *LayoutStack, value: f32, min_val: f32, max_val: f32, opts: LayoutSliderOptions) f32 {
        const r = self.placeSizeWithAuto(opts.width, opts.height, 120.0, 20.0);
        const hov = opts.is_hovered orelse UICanvas.isPointInRect(self.canvas.mouse_pos[0], self.canvas.mouse_pos[1], r[0], r[1], r[2], r[3]);
        const drag = opts.is_dragging orelse (hov and self.canvas.mouse_down);

        const norm = if (max_val > min_val) std.math.clamp((value - min_val) / (max_val - min_val), 0.0, 1.0) else 0.0;
        var new_norm = norm;

        if (drag) {
            new_norm = UICanvas.sliderValueAt(r[0], r[2], self.canvas.mouse_pos[0]);
        }

        if (opts.class != null or opts.style != null) {
            const state = UIState.fromFlags(hov, drag, false, false);
            _ = self.canvas.drawStyledSlider(r, new_norm, .{
                .class = opts.class,
                .style = opts.style,
                .state = state,
            });
        } else {
            _ = self.canvas.drawSlider(r[0], r[1], r[2], r[3], new_norm, hov, drag);
        }

        return min_val + new_norm * (max_val - min_val);
    }

    /// Places and renders a progress bar widget.
    pub fn progressBar(self: *LayoutStack, fraction: f32, opts: LayoutProgressOptions) void {
        const r = self.placeSizeWithAuto(opts.width, opts.height, 100.0, 16.0);
        self.canvas.drawProgressBar(r[0], r[1], r[2], r[3], fraction, opts.bg_color, opts.fill_color);
    }

    /// Places and renders a dividing line widget.
    pub fn divider(self: *LayoutStack, opts: LayoutDividerOptions) void {
        if (self.depth == 0) return;
        const f = &self.frames[self.depth - 1];
        if (f.kind == .hstack) {
            const r = self.place(opts.thickness, f.ih);
            self.canvas.drawLine(r[0], r[1], r[0], r[1] + r[3], opts.thickness, opts.color);
        } else {
            const r = self.place(f.iw, opts.thickness);
            self.canvas.drawDivider(r[0], r[1], r[2], opts.thickness, opts.color);
        }
    }

    /// Places and renders a badge tag widget.
    pub fn badge(self: *LayoutStack, text: []const u8, opts: LayoutBadgeOptions) void {
        const auto_w = @as(f32, @floatFromInt(text.len)) * opts.font_size * 0.5 + 14.0;
        const auto_h = opts.font_size + 8.0;
        const r = self.place(auto_w, auto_h);
        self.canvas.drawBadge(text, r[0], r[1], opts.font_size, opts.bg_color, opts.text_color);
    }

    /// Places a widget box with a resolved style `margin` around it (the
    /// CSS-ish spacing between a box and its slot). Returns the content
    /// rect; the cursor advances by margin + size + margin + spacing.
    pub fn placeBox(self: *LayoutStack, w: f32, h: f32, box: ?UIBoxStyle) [4]f32 {
        if (self.depth == 0) return .{ 0, 0, 0, 0 };
        const bs = box orelse UIBoxStyle{};
        const s = self.resolveContainerStyle(bs.class, bs.style);
        const m = @max(s.margin, 0.0);
        const f = &self.frames[self.depth - 1];
        switch (f.kind) {
            .grid => {
                // A grid slot is explicit: the margin insets the widget
                // inside its cell instead of moving the walk.
                const cell = f.grid.placedRect(f.cell, w, h);
                f.cell += 1;
                var r = cell;
                r[0] += m;
                r[1] += m;
                r[2] = @max(r[2] - 2.0 * m, 0.0);
                r[3] = @max(r[3] - 2.0 * m, 0.0);
                return clampToFrame(r, f);
            },
            .hstack, .vstack => {
                f.cursor += m;
                const r = self.flowRect(w, h, f.align_cross);
                f.cursor += (if (f.kind == .hstack) w else h) + f.spacing + m;
                return clampToFrame(r, f);
            },
        }
    }

    fn beginFlow(self: *LayoutStack, rect: [4]f32, opts: LayoutFlowOptions, kind: FrameKind) bool {
        if (self.depth >= max_depth) return false;
        const s = self.resolveContainerStyle(opts.class, opts.style);
        const pad = opts.getEdges(s.padding);
        self.canvas.drawStyleRect(rect, s);
        const inner_w = @max(rect[2] - pad.hTotal(), 0.0);
        const inner_h = @max(rect[3] - pad.vTotal(), 0.0);
        const f = Frame{
            .kind = kind,
            .x = rect[0],
            .y = rect[1],
            .w = rect[2],
            .h = rect[3],
            .ix = rect[0] + pad.left,
            .iy = rect[1] + pad.top,
            .iw = inner_w,
            .ih = inner_h,
            .cursor = if (kind == .hstack) rect[0] + pad.left else rect[1] + pad.top,
            .spacing = @max(opts.spacing, 0.0),
            .align_cross = opts.align_cross,
        };
        self.frames[self.depth] = f;
        self.depth += 1;
        return true;
    }

    fn resolveContainerStyle(self: *LayoutStack, class: ?[]const u8, override: ?UIStyleOverride) UIStyle {
        return self.canvas.resolveStyle(.{
            .kind = .container,
            .class = class,
            .override = override,
        });
    }

    /// Next flow-frame rect: main axis at the cursor, cross axis aligned
    /// inside the inner extent (`stretch` expands to the inner extent).
    fn flowRect(self: *LayoutStack, w: f32, h: f32, cross: LayoutAlignCross) [4]f32 {
        const f = &self.frames[self.depth - 1];
        if (f.kind == .hstack) {
            return .{
                f.cursor,
                alignCrossPos(f.iy, f.ih, h, cross),
                w,
                if (cross == .stretch) f.ih else h,
            };
        }
        return .{
            alignCrossPos(f.ix, f.iw, w, cross),
            f.cursor,
            if (cross == .stretch) f.iw else w,
            h,
        };
    }

    fn alignCrossPos(origin: f32, extent: f32, size: f32, alignment: LayoutAlignCross) f32 {
        return switch (alignment) {
            .start, .stretch => origin,
            .center => origin + @max((extent - size) * 0.5, 0.0),
            .end => origin + @max(extent - size, 0.0),
        };
    }

    /// Overflow clamp: a widget that does not fit is kept inside the
    /// container bounds (pinned to the origin when it is larger than the
    /// container), mirroring the clamps used by the other UI helpers.
    fn clampToFrame(rect: [4]f32, f: *const Frame) [4]f32 {
        var r = rect;
        r[0] = clampExtent(r[0], r[2], f.x, f.w);
        r[1] = clampExtent(r[1], r[3], f.y, f.h);
        return r;
    }

    fn clampExtent(pos: f32, size: f32, origin: f32, extent: f32) f32 {
        if (extent <= 0.0 or size <= 0.0) return pos;
        if (size >= extent) return origin;
        return std.math.clamp(pos, origin, origin + extent - size);
    }
};

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
    // p1-right = end + half-normal; the normal is (-1, 0) here, mirroring
    // the horizontal case (p1-right keeps the +Y side there).
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), v[2][0], 1e-5);
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

// --- Layout + styling tests ---
//
// The test canvas constructs without a GPU: drawStyleRect and the styled
// widgets only push quads into the CPU-side vertex list, so emitted geometry
// is assertable directly.

fn testCanvas(alloc: std.mem.Allocator) UICanvas {
    return .{ .allocator = alloc, .font_texture = undefined };
}

/// Frees the CPU-side quad buffers of a `testCanvas` (no GPU state: the full
/// `deinit` would touch the undefined sokol handles).
fn freeTestCanvas(canvas: *UICanvas) void {
    canvas.vertices.deinit(canvas.allocator);
    canvas.indices.deinit(canvas.allocator);
}

fn quadCount(canvas: *const UICanvas) usize {
    return canvas.vertices.items.len / 4;
}

test "layoutAlignOffset and grid extent helpers" {
    const t = std.testing;
    try t.expectApproxEqAbs(@as(f32, 0.0), layoutAlignOffset(100, 20, .start), 1e-5);
    try t.expectApproxEqAbs(@as(f32, 40.0), layoutAlignOffset(100, 20, .center), 1e-5);
    try t.expectApproxEqAbs(@as(f32, 80.0), layoutAlignOffset(100, 20, .end), 1e-5);
    // Oversized items clamp to the slot start: no negative offsets.
    try t.expectApproxEqAbs(@as(f32, 0.0), layoutAlignOffset(20, 100, .center), 1e-5);
    try t.expectApproxEqAbs(@as(f32, 0.0), layoutAlignOffset(20, 100, .end), 1e-5);

    // Equal split accounts for spacing.
    try t.expectApproxEqAbs(@as(f32, 45.0), gridExtentSize(null, 2, 100, 10, 0), 1e-5);
    try t.expectApproxEqAbs(@as(f32, 45.0), gridExtentSize(null, 2, 100, 10, 1), 1e-5);
    // Explicit widths; columns past the slice share the leftover.
    const widths = [_]f32{ 30, 50 };
    try t.expectApproxEqAbs(@as(f32, 30.0), gridExtentSize(&widths, 3, 100, 0, 0), 1e-5);
    try t.expectApproxEqAbs(@as(f32, 50.0), gridExtentSize(&widths, 3, 100, 0, 1), 1e-5);
    try t.expectApproxEqAbs(@as(f32, 20.0), gridExtentSize(&widths, 3, 100, 0, 2), 1e-5);
    try t.expectApproxEqAbs(@as(f32, 0.0), gridExtentOffset(&widths, 3, 100, 5, 0), 1e-5);
    try t.expectApproxEqAbs(@as(f32, 35.0), gridExtentOffset(&widths, 3, 100, 5, 1), 1e-5);
    try t.expectApproxEqAbs(@as(f32, 90.0), gridExtentOffset(&widths, 3, 100, 5, 2), 1e-5);
}

test "LayoutGridSpec cell and placed rects" {
    const t = std.testing;
    const spec = LayoutGridSpec{ .inner = .{ 10, 20, 110, 60 }, .columns = 2, .rows = 2, .spacing = 10 };
    // Equal cells: 50x25 each.
    const c00 = spec.cellRect(0, 0);
    try t.expectApproxEqAbs(@as(f32, 10), c00[0], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 20), c00[1], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 50), c00[2], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 25), c00[3], 1e-5);
    const c10 = spec.cellRect(1, 0);
    try t.expectApproxEqAbs(@as(f32, 70), c10[0], 1e-5);
    const c01 = spec.cellRect(0, 1);
    try t.expectApproxEqAbs(@as(f32, 55), c01[1], 1e-5);
    // Out-of-range indices clamp to the last track.
    const last = spec.cellRect(5, 9);
    const expected_last = spec.cellRect(1, 1);
    try t.expectEqual(expected_last[0], last[0]);
    try t.expectEqual(expected_last[1], last[1]);

    // Placed rect centers a smaller widget in the cell (explicit alignment).
    const centered = LayoutGridSpec{
        .inner = .{ 10, 20, 110, 60 },
        .columns = 2,
        .rows = 2,
        .spacing = 10,
        .align_main = .center,
        .align_cross = .center,
    };
    const p = centered.placedRect(0, 30, 10);
    try t.expectApproxEqAbs(@as(f32, 20), p[0], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 27.5), p[1], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 30), p[2], 1e-5);

    // Explicit column widths: [30, 50, rest].
    const spec2 = LayoutGridSpec{
        .inner = .{ 0, 0, 100, 50 },
        .columns = 3,
        .rows = 1,
        .column_widths = &.{ 30, 50 },
    };
    try t.expectApproxEqAbs(@as(f32, 30), spec2.cellRect(0, 0)[2], 1e-5);
    const c2 = spec2.cellRect(2, 0);
    try t.expectApproxEqAbs(@as(f32, 80), c2[0], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 20), c2[2], 1e-5);
}

test "LayoutStack HStack places with spacing and cross alignment" {
    const t = std.testing;
    var canvas = testCanvas(std.testing.allocator);
    defer freeTestCanvas(&canvas);
    var ls = LayoutStack.init(&canvas);
    defer ls.reset();
    try t.expect(ls.beginHStack(.{ 10, 20, 300, 100 }, .{ .spacing = 10, .align_cross = .center }));

    // First item at the inner origin, vertically centered (100 - 20) / 2.
    const r0 = ls.place(50, 20);
    try t.expectApproxEqAbs(@as(f32, 10), r0[0], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 60), r0[1], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 50), r0[2], 1e-5);
    // Second item advances by size + spacing.
    const r1 = ls.place(50, 20);
    try t.expectApproxEqAbs(@as(f32, 70), r1[0], 1e-5);
    // Per-call override beats the container alignment.
    const r2 = ls.placeAligned(40, 30, .start);
    try t.expectApproxEqAbs(@as(f32, 130), r2[0], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 20), r2[1], 1e-5);
    // Stretch expands to the inner cross extent.
    const r3 = ls.placeAligned(40, 5, .stretch);
    try t.expectApproxEqAbs(@as(f32, 180), r3[0], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 20), r3[1], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 100), r3[3], 1e-5);

    ls.end();
    // Extra end() past the bottom is a no-op; place() without a container
    // returns a zero rect.
    ls.end();
    ls.end();
    const r = ls.place(10, 10);
    try t.expectEqual(@as(f32, 0), r[0]);
    try t.expectEqual(@as(f32, 0), r[3]);
}

test "LayoutStack VStack padding and end alignment" {
    const t = std.testing;
    var canvas = testCanvas(std.testing.allocator);
    defer freeTestCanvas(&canvas);
    var ls = LayoutStack.init(&canvas);
    defer ls.reset();
    try t.expect(ls.beginVStack(.{ 0, 0, 200, 200 }, .{ .padding = 8, .spacing = 4 }));
    const r0 = ls.place(100, 20);
    try t.expectApproxEqAbs(@as(f32, 8), r0[0], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 8), r0[1], 1e-5);
    const r1 = ls.place(100, 20);
    try t.expectApproxEqAbs(@as(f32, 32), r1[1], 1e-5); // 8 + 20 + 4
    ls.end();

    // Cross-axis end alignment insets horizontally from the inner right edge.
    try t.expect(ls.beginVStack(.{ 0, 0, 200, 200 }, .{ .padding = 8, .align_cross = .end }));
    const re = ls.place(50, 20);
    try t.expectApproxEqAbs(@as(f32, 200 - 8 - 50), re[0], 1e-5);
    ls.end();
}

test "LayoutStack nested containers compose" {
    const t = std.testing;
    var canvas = testCanvas(std.testing.allocator);
    defer freeTestCanvas(&canvas);
    var ls = LayoutStack.init(&canvas);
    defer ls.reset();
    try t.expect(ls.beginVStack(.{ 0, 0, 100, 300 }, .{ .spacing = 10 }));

    const header = ls.place(80, 20);
    try t.expectApproxEqAbs(@as(f32, 0), header[1], 1e-5);

    // Place a rect, then open a nested HStack exactly on it.
    const row = ls.place(80, 40);
    try t.expectApproxEqAbs(@as(f32, 30), row[1], 1e-5);
    try t.expect(ls.beginHStack(row, .{ .spacing = 5 }));
    const b0 = ls.place(30, 40);
    try t.expectApproxEqAbs(@as(f32, 0), b0[0], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 30), b0[1], 1e-5);
    const b1 = ls.place(30, 40);
    try t.expectApproxEqAbs(@as(f32, 35), b1[0], 1e-5);
    ls.end();

    // After the nested container closes, the outer cursor continues.
    const b2 = ls.place(80, 20);
    try t.expectApproxEqAbs(@as(f32, 80), b2[1], 1e-5); // 30 + 40 + 10
    ls.end();
}

test "LayoutStack grid walk and overflow clamp" {
    const t = std.testing;
    var canvas = testCanvas(std.testing.allocator);
    defer freeTestCanvas(&canvas);
    var ls = LayoutStack.init(&canvas);
    defer ls.reset();
    try t.expect(ls.beginGrid(.{ 0, 0, 120, 60 }, .{ .columns = 3, .rows = 2 }));

    // Row-major walk over equal 40x30 cells.
    const c0 = ls.place(20, 10);
    try t.expectApproxEqAbs(@as(f32, 0), c0[0], 1e-5);
    const c1 = ls.place(20, 10);
    try t.expectApproxEqAbs(@as(f32, 40), c1[0], 1e-5);
    _ = ls.place(20, 10);
    const c3 = ls.place(20, 10);
    try t.expectApproxEqAbs(@as(f32, 30), c3[1], 1e-5);
    const c4 = ls.place(20, 10);
    try t.expectApproxEqAbs(@as(f32, 40), c4[0], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 30), c4[1], 1e-5);

    // A widget bigger than its cell clamps inside the container bounds.
    const big = ls.place(60, 40);
    try t.expectApproxEqAbs(@as(f32, 60), big[0], 1e-5); // 120 - 60
    try t.expectApproxEqAbs(@as(f32, 20), big[1], 1e-5); // 60 - 40
    ls.end();

    // A widget bigger than the whole container pins to the origin.
    try t.expect(ls.beginHStack(.{ 0, 0, 50, 50 }, .{}));
    const huge = ls.place(80, 80);
    try t.expectApproxEqAbs(@as(f32, 0), huge[0], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 0), huge[1], 1e-5);
    ls.end();
}

test "LayoutStack depth cap and reset" {
    const t = std.testing;
    var canvas = testCanvas(std.testing.allocator);
    defer freeTestCanvas(&canvas);
    var ls = LayoutStack.init(&canvas);
    var opened: usize = 0;
    for (0..LayoutStack.max_depth + 4) |_| {
        if (ls.beginHStack(.{ 0, 0, 500, 500 }, .{})) opened += 1;
    }
    try t.expectEqual(LayoutStack.max_depth, opened);
    // Placements past the cap operate on the top frame instead of crashing.
    _ = ls.place(10, 10);
    ls.reset();
    try t.expectEqual(@as(usize, 0), ls.depth);
    ls.end(); // no-op at depth 0
}

test "style cascade: override beats class beats theme default" {
    const t = std.testing;
    var canvas = testCanvas(std.testing.allocator);
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
    const t = std.testing;
    var canvas = testCanvas(std.testing.allocator);
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
    const t = std.testing;
    try t.expectEqual(UIState.hover, UIState.fromFlags(true, false, false, false));
    try t.expectEqual(UIState.active, UIState.fromFlags(true, true, false, false));
    try t.expectEqual(UIState.focus, UIState.fromFlags(true, false, true, false));
    try t.expectEqual(UIState.disabled, UIState.fromFlags(true, true, true, true));
    try t.expectEqual(UIState.normal, UIState.fromFlags(false, false, false, false));

    var canvas = testCanvas(std.testing.allocator);
    defer freeTestCanvas(&canvas);
    const s = canvas.resolveStyle(.{ .kind = .button, .state = .disabled });
    try t.expectApproxEqAbs(@as(f32, 0.45), s.opacity, 1e-5);
    // Inherit: non-disabled states keep full opacity.
    const n = canvas.resolveStyle(.{ .kind = .button });
    try t.expectApproxEqAbs(@as(f32, 1.0), n.opacity, 1e-5);
}

test "corner radius, gradient, shadow and opacity change emitted geometry" {
    const t = std.testing;
    var canvas = testCanvas(std.testing.allocator);
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
    const t = std.testing;
    var canvas = testCanvas(std.testing.allocator);
    defer freeTestCanvas(&canvas);
    canvas.setStyleClass("padded", .{ .normal = .{
        .padding = 10.0,
        .background = Color4.new(0.2, 0.2, 0.3, 0.8),
    } });

    var ls = LayoutStack.init(&canvas);
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

test "placeBox applies resolved margin around the box" {
    const t = std.testing;
    var canvas = testCanvas(std.testing.allocator);
    defer freeTestCanvas(&canvas);
    canvas.setStyleClass("gappy", .{ .normal = .{ .margin = 5.0 } });

    var ls = LayoutStack.init(&canvas);
    defer ls.reset();
    try t.expect(ls.beginVStack(.{ 0, 0, 100, 100 }, .{}));
    const r0 = ls.placeBox(40, 10, .{ .class = "gappy" });
    try t.expectApproxEqAbs(@as(f32, 0), r0[0], 1e-5);
    try t.expectApproxEqAbs(@as(f32, 5), r0[1], 1e-5); // top margin shifts
    // Cursor: margin 5 + height 10 + spacing 0 + margin 5 = 20.
    const r1 = ls.placeBox(40, 10, null);
    try t.expectApproxEqAbs(@as(f32, 20), r1[1], 1e-5);
    ls.end();
}

test "styled widgets emit geometry and resolve their state styles" {
    const t = std.testing;
    var canvas = testCanvas(std.testing.allocator);
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

// --- CSS theme parsing + style transition integration tests ---

/// Exact color equality for test assertions (transition.styleEql compares
/// whole styles, these tests assert single color fields).
fn testColorEql(a: Color4, b: Color4) bool {
    return a.r == b.r and a.g == b.g and a.b == b.b and a.a == b.a;
}

test "applyCssTheme flows parsed theme and classes through the cascade" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var canvas = testCanvas(t.allocator);
    defer freeTestCanvas(&canvas);

    const parsed = try parseCss(arena.allocator(),
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
    const t = std.testing;
    var canvas = testCanvas(t.allocator);
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
    const t = std.testing;
    var canvas = testCanvas(t.allocator);
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
    try t.expectApproxEqAbs(lerpStyle(UIStyle{ .background = normal_bg }, UIStyle{ .background = hover_bg }, 1.0 / 3.0).background.r, mid.background.r, 1e-5);

    canvas.begin(); // t=400: back to normal mid-flight; the drawn style (200/300) is the new source
    const back = canvas.resolveAnimatedStyle(.button, .{ .anim_key = "r", .transition = .{ .duration_ms = 300, .easing = .linear } });
    try t.expectApproxEqAbs(normal_bg.r + (hover_bg.r - normal_bg.r) * (2.0 / 3.0), back.background.r, 1e-5);

    canvas.begin(); // t=500: now animating from that point toward normal
    const later = canvas.resolveAnimatedStyle(.button, .{ .anim_key = "r", .transition = .{ .duration_ms = 300, .easing = .linear } });
    const from_r = normal_bg.r + (hover_bg.r - normal_bg.r) * (2.0 / 3.0);
    try t.expectApproxEqAbs(from_r + (normal_bg.r - from_r) * (1.0 / 3.0), later.background.r, 1e-5);
}

test "class-level transition config drives the animation without per-call override" {
    const t = std.testing;
    var canvas = testCanvas(t.allocator);
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
    const t = std.testing;
    var canvas = testCanvas(t.allocator);
    defer freeTestCanvas(&canvas);
    canvas.frame_dt_ms = 100;
    const bg = Color4.new(0.2, 0.4, 0.6, 1.0);

    canvas.begin(); // t=100: fill every slot
    var i: usize = 0;
    while (i < max_style_transitions) : (i += 1) {
        var name_buf: [16]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "w{d}", .{i}) catch unreachable;
        const s = canvas.resolveAnimatedStyle(.badge, .{
            .anim_key = name,
            .transition = .{ .duration_ms = 500, .easing = .linear },
            .style = .{ .background = bg },
        });
        try t.expect(testColorEql(bg, s.background));
    }
    try t.expectEqual(@as(usize, max_style_transitions), usedTransitionSlots(&canvas));

    canvas.begin(); // t=200: one more widget evicts the stalest entry, all still resolve
    const extra = canvas.resolveAnimatedStyle(.badge, .{
        .anim_key = "extra",
        .transition = .{ .duration_ms = 500, .easing = .linear },
        .style = .{ .background = bg },
    });
    try t.expect(testColorEql(bg, extra.background));
    try t.expectEqual(@as(usize, max_style_transitions), usedTransitionSlots(&canvas));

    // Zero-duration config snaps and releases its slot.
    const snap = canvas.resolveAnimatedStyle(.badge, .{
        .anim_key = "extra",
        .transition = .{},
        .style = .{ .background = bg },
    });
    try t.expect(testColorEql(bg, snap.background));
    try t.expectEqual(@as(usize, max_style_transitions - 1), usedTransitionSlots(&canvas));
}

fn usedTransitionSlots(canvas: *const UICanvas) usize {
    var n: usize = 0;
    for (&canvas.style_transitions) |*s| {
        if (s.used) n += 1;
    }
    return n;
}

test "animated styled widgets draw their interpolated style" {
    const t = std.testing;
    var canvas = testCanvas(t.allocator);
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

test "LayoutStack with UIEdges padding and spacer in HStack and VStack" {
    const t = std.testing;
    var canvas = testCanvas(t.allocator);
    defer freeTestCanvas(&canvas);

    var stack = LayoutStack.init(&canvas);
    _ = stack.beginHStack(.{ 0, 0, 300, 100 }, .{
        .padding_edges = UIEdges.trbl(10, 20, 10, 15),
        .spacing = 10,
    });

    const r1 = stack.placeSize(UISize.px(50), UISize.px(30));
    try t.expectEqual(@as(f32, 15.0), r1[0]); // x = 0 + left(15)
    try t.expectEqual(@as(f32, 10.0), r1[1]); // y = 0 + top(10)
    try t.expectEqual(@as(f32, 50.0), r1[2]);
    try t.expectEqual(@as(f32, 30.0), r1[3]);

    // Push spacer: takes remaining inner space
    const sp = stack.spacer();
    try t.expect(sp[2] > 0.0);

    const r2 = stack.place(40, 30);
    // When placed after a spacer taking full inner width, item clamps to container frame right (300 - 40 = 260)
    try t.expectEqual(@as(f32, 260.0), r2[0]);

    stack.endFlow();
}

test "LayoutStack beginFlex and placeFlex proportional sizing" {
    const t = std.testing;
    var canvas = testCanvas(t.allocator);
    defer freeTestCanvas(&canvas);

    var stack = LayoutStack.init(&canvas);
    _ = stack.beginFlex(.{ 0, 0, 400, 100 }, .{
        .direction = .row,
        .padding = 10,
        .spacing = 10,
    });

    const r1 = stack.placeSize(UISize.px(80), UISize.px(40));
    try t.expectEqual(@as(f32, 10.0), r1[0]);
    try t.expectEqual(@as(f32, 80.0), r1[2]);

    const r_flex = stack.placeFlex(1.0);
    // Inner w = 400 - 20 = 380. Cursor is at 10 + 80 + 10 = 100.
    // Remaining = (10 + 380) - 100 = 290.
    try t.expectEqual(@as(f32, 100.0), r_flex[0]);
    try t.expectEqual(@as(f32, 290.0), r_flex[2]);

    stack.endFlow();
}

test "LayoutStack placeGridSpan multi-cell spanning" {
    const t = std.testing;
    var canvas = testCanvas(t.allocator);
    defer freeTestCanvas(&canvas);

    var stack = LayoutStack.init(&canvas);
    _ = stack.beginGrid(.{ 0, 0, 210, 110 }, .{
        .columns = 2,
        .rows = 2,
        .spacing = 10,
        .padding = 0,
    });

    // 2x2 grid in 210x110: each cell is 100x50 with 10px spacing.
    // Span cols 0..1 (2 cols) and row 0 (1 row) -> width is 100 + 10 + 100 = 210.
    const span_rect = stack.placeGridSpan(0, 0, 2, 1, .auto, .auto);
    try t.expectEqual(@as(f32, 0.0), span_rect[0]);
    try t.expectEqual(@as(f32, 0.0), span_rect[1]);
    try t.expectEqual(@as(f32, 210.0), span_rect[2]);
    try t.expectEqual(@as(f32, 50.0), span_rect[3]);

    stack.endGrid();
}

test "LayoutStack anchor and dock placement" {
    const t = std.testing;
    var canvas = testCanvas(t.allocator);
    defer freeTestCanvas(&canvas);

    var stack = LayoutStack.init(&canvas);
    _ = stack.beginVStack(.{ 0, 0, 800, 600 }, .{});

    // Dock top navigation bar
    const nav = stack.dock(.top, 60.0, UIEdges.zero);
    try t.expectEqual(@as(f32, 0.0), nav[0]);
    try t.expectEqual(@as(f32, 0.0), nav[1]);
    try t.expectEqual(@as(f32, 800.0), nav[2]);
    try t.expectEqual(@as(f32, 60.0), nav[3]);

    // Anchor floating modal dialog in the remaining screen center
    const modal = stack.anchor(300, 200, .center, UIEdges.all(20));
    try t.expectEqual(@as(f32, 250.0), modal[0]); // (800 - 300) / 2
    try t.expectEqual(@as(f32, 230.0), modal[1]); // 60 + (540 - 200) / 2
    try t.expectEqual(@as(f32, 300.0), modal[2]);
    try t.expectEqual(@as(f32, 200.0), modal[3]);

    stack.end();
}

test "LayoutStack immediate mode widgets emit geometry and handle input" {
    const t = std.testing;
    var canvas = testCanvas(t.allocator);
    defer freeTestCanvas(&canvas);

    // Simulate mouse clicked at (50, 60)
    canvas.setInput(50.0, 60.0, true, true);

    var stack = LayoutStack.init(&canvas);
    _ = stack.beginVStack(.{ 0, 0, 200, 400 }, .{ .padding = 10, .spacing = 10 });

    // Label: takes y in [10..26]
    stack.label("Test Title", .{ .font_size = 14.0 });

    // Button: at y in [36..76] -> contains mouse (50, 60)!
    const clicked = stack.button("Click Me", .{
        .width = UISize.px(120),
        .height = UISize.px(40),
    });
    try t.expect(clicked);

    // Checkbox: at y in [86..106] -> does NOT contain mouse (50, 60)
    var is_checked = false;
    const cb_clicked = stack.checkbox("Enable", &is_checked, .{});
    try t.expect(!cb_clicked);
    try t.expect(!is_checked);

    // Slider: value update
    var val: f32 = 0.25;
    val = stack.slider(val, 0.0, 1.0, .{});

    // Progress bar
    stack.progressBar(0.5, .{});

    // Divider
    stack.divider(.{});

    // Badge
    stack.badge("HOT", .{});

    stack.endFlow();

    // Verify all widgets generated vertex geometry
    try t.expect(quadCount(&canvas) > 10);
}

test "batchUploadBytes keeps the u16 vertex cap in usize arithmetic" {
    // 65535 vertices x 48 B ~= 3.1 MB: u16 arithmetic would already trap at
    // 1366 vertices, so the helper must compute in usize in every build mode.
    const verts: usize = std.math.maxInt(u16);
    const idx: usize = std.math.maxInt(u16);
    const expected = verts * @sizeOf(UIVertex) + idx * @sizeOf(u16);
    try std.testing.expectEqual(expected, UICanvas.batchUploadBytes(verts, idx));
}
