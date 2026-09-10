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

pub const UICanvas = struct {
    allocator: std.mem.Allocator,
    vertices: std.ArrayListUnmanaged(UIVertex) = .empty,
    indices: std.ArrayListUnmanaged(u16) = .empty,

    vertex_buffer: sg.Buffer = .{},
    index_buffer: sg.Buffer = .{},
    pipeline: sg.Pipeline = .{},
    font_texture: Texture,

    capacity_vertices: usize = 16384,
    capacity_indices: usize = 24576,

    pub fn init(allocator: std.mem.Allocator) !UICanvas {
        const tex = try Texture.fromMemory(allocator, font_png_data, .{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
            // Box-filtered mips blur the distance field beyond legibility.
            .mipmaps = false,
        });

        const max_v: usize = 16384;
        const max_i: usize = 24576;

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
        const u = 1.0 / 512.0;
        const v = 1.0 / 512.0;
        self.addQuad(x, y, w, h, u, v, u, v, color, .{ 0.0, 0.0, 0.0, 0.0 });
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

    /// Uploads dynamic batch buffers and executes the UI render pass
    pub fn render(self: *UICanvas, screen_w: f32, screen_h: f32) void {
        if (self.vertices.items.len == 0 or self.indices.items.len == 0) return;
        if (screen_w <= 0.0 or screen_h <= 0.0) return;

        sg.updateBuffer(self.vertex_buffer, sg.asRange(self.vertices.items));
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
