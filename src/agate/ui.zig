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
// Split leaf modules (phase 3): draw primitives, text, widgets and input
// state. They take a generic canvas (`anytype`) so they never import this
// facade back (same discipline as `scene/`); this file owns `UICanvas` and
// forwards each moved method (Zig 0.16 has no usingnamespace).
const ui_draw = @import("ui/draw.zig");
const ui_text = @import("ui/text.zig");
const ui_widgets = @import("ui/widgets.zig");
const ui_input = @import("ui/input_state.zig");

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

// Re-exports from the split leaf modules (public API unchanged).
pub const UIVertex = ui_draw.UIVertex;
pub const GlyphUV = ui_text.GlyphUV;
pub const getGlyphUV = ui_text.getGlyphUV;
pub const ScrollState = ui_input.ScrollState;
pub const TextInputState = ui_input.TextInputState;

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

    /// P6 same-frame window (automatic, no manual reset): commit watermark
    /// of the last successful upload into the canvas-owned pair, read from
    /// `sg.queryStats().prev_frame.frame_index` (rotates at every
    /// `sg.commit`, even with stats disabled). `isUploadOpen` compares the
    /// stored watermark with the current marker (EQUALITY, wrap-safe): a
    /// real commit changes the marker and reopens the window with no caller
    /// action. Buffer state lives with the buffers (canvas-owned); frames
    /// only borrow the IDs. `armed` marks a completed upload; headless (no
    /// commits exist) it alone decides, which doubles as the test override.
    /// `upload_seq` counts every successful upload to this canvas (any
    /// path) and pairs with the buffer IDs as the committed-upload identity
    /// a frame stamps: any other writer between a frame's upload and its
    /// next capture invalidates the borrowed packet (see `UiFrame`).
    ui_upload_commit: u32 = 0,
    ui_upload_armed: bool = false,
    ui_upload_seq: u64 = 0,

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

    /// Helper to add a textured / colored quad (see ui/draw.zig).
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
        ui_draw.addQuad(self, x, y, w, h, u_min, v_min, u_max, v_max, color, mode_params);
    }

    /// Draws a solid rectangle in screen pixel coordinates (see ui/draw.zig).
    pub fn drawRect(self: *UICanvas, x: f32, y: f32, w: f32, h: f32, color: Color4) void {
        ui_draw.drawRect(self, x, y, w, h, color);
    }

    /// Draws a rectangle outline with specified border thickness (see ui/draw.zig).
    pub fn drawRectOutline(self: *UICanvas, x: f32, y: f32, w: f32, h: f32, thickness: f32, color: Color4) void {
        ui_draw.drawRectOutline(self, x, y, w, h, thickness, color);
    }

    /// Draws a styled UI panel (filled rectangle + border, see ui/draw.zig).
    pub fn drawPanel(self: *UICanvas, x: f32, y: f32, w: f32, h: f32, bg_col: Color4, border_col: Color4, border_width: f32) void {
        ui_draw.drawPanel(self, x, y, w, h, bg_col, border_col, border_width);
    }

    /// Draws crisp Signed Distance Field (SDF) text (see ui/text.zig).
    pub fn drawText(self: *UICanvas, text: []const u8, x: f32, y: f32, font_size: f32, color: Color4) void {
        ui_text.drawText(self, text, x, y, font_size, color);
    }

    /// Draws bold Signed Distance Field (SDF) text (see ui/text.zig).
    pub fn drawTextBold(self: *UICanvas, text: []const u8, x: f32, y: f32, font_size: f32, color: Color4, extra_boldness: f32) void {
        ui_text.drawTextBold(self, text, x, y, font_size, color, extra_boldness);
    }

    /// Draws SDF text with a high-contrast dark outline / shadow (see ui/text.zig).
    pub fn drawTextWithOutline(self: *UICanvas, text: []const u8, x: f32, y: f32, font_size: f32, color: Color4, outline_width: f32) void {
        ui_text.drawTextWithOutline(self, text, x, y, font_size, color, outline_width);
    }

    /// Draws a smooth horizontal progress / health bar (see ui/draw.zig).
    pub fn drawProgressBar(self: *UICanvas, x: f32, y: f32, w: f32, h: f32, progress: f32, bg_col: Color4, fill_col: Color4) void {
        ui_draw.drawProgressBar(self, x, y, w, h, progress, bg_col, fill_col);
    }

    /// Draws an interactive styled button (see ui/widgets.zig).
    pub fn drawButton(self: *UICanvas, text: []const u8, x: f32, y: f32, w: f32, h: f32, font_size: f32, is_hovered: bool, is_pressed: bool) void {
        ui_widgets.drawButton(self, text, x, y, w, h, font_size, is_hovered, is_pressed);
    }

    /// Draws a compact pill-shaped badge with text (see ui/widgets.zig).
    pub fn drawBadge(self: *UICanvas, text: []const u8, x: f32, y: f32, font_size: f32, bg_col: Color4, text_col: Color4) void {
        ui_widgets.drawBadge(self, text, x, y, font_size, bg_col, text_col);
    }

    /// Draws a stateless checkbox box with an optional label to the right (see ui/widgets.zig).
    pub fn drawCheckbox(self: *UICanvas, x: f32, y: f32, size: f32, checked: bool, is_hovered: bool, label: ?[]const u8, label_size: f32) void {
        ui_widgets.drawCheckbox(self, x, y, size, checked, is_hovered, label, label_size);
    }

    /// Draws a stateless horizontal slider, returns the clamped value (see ui/widgets.zig).
    pub fn drawSlider(self: *UICanvas, x: f32, y: f32, w: f32, h: f32, value: f32, is_hovered: bool, is_dragging: bool) f32 {
        return ui_widgets.drawSlider(self, x, y, w, h, value, is_hovered, is_dragging);
    }

    /// Draws a thin horizontal separator line (see ui/widgets.zig).
    pub fn drawDivider(self: *UICanvas, x: f32, y: f32, w: f32, thickness: f32, color: Color4) void {
        ui_widgets.drawDivider(self, x, y, w, thickness, color);
    }

    /// Pure corner math for drawLine: returns the 4 quad corners
    /// (p0-left, p0-right, p1-right, p1-left) offset perpendicular
    /// to the segment by half the thickness. Zero-area on degenerate input.
    /// (See ui/draw.zig.)
    pub fn lineCorners(x0: f32, y0: f32, x1: f32, y1: f32, thickness: f32) [4][2]f32 {
        return ui_draw.lineCorners(x0, y0, x1, y1, thickness);
    }

    /// Draws a solid thick line in screen pixel coordinates (no depth test; see ui/draw.zig).
    pub fn drawLine(self: *UICanvas, x0: f32, y0: f32, x1: f32, y1: f32, thickness: f32, color: Color4) void {
        ui_draw.drawLine(self, x0, y0, x1, y1, thickness, color);
    }

    /// Draws a small down-triangle arrow (dropdown chevron) from stacked solid quads (see ui/widgets.zig).
    pub fn drawArrowDown(self: *UICanvas, x: f32, y: f32, size: f32, color: Color4) void {
        ui_widgets.drawArrowDown(self, x, y, size, color);
    }

    // ------------------------------------------------------------------
    // Dropdown (stateless immediate-mode; caller owns selected/open/hover)
    // ------------------------------------------------------------------

    /// Item row height derived from the font size; shared by drawing and
    /// hit-testing so geometry always matches. 4px padding above/below text.
    /// (See ui/input_state.zig.)
    pub fn dropdownItemHeight(font_size: f32) f32 {
        return ui_input.dropdownItemHeight(font_size);
    }

    /// Rect [x, y, w, h] of an open-list item stacked directly below the
    /// closed button rect. (See ui/input_state.zig.)
    pub fn dropdownItemRect(rect: [4]f32, item_h: f32, index: usize) [4]f32 {
        return ui_input.dropdownItemRect(rect, item_h, index);
    }

    /// Hit-tests ONLY the open list stacked under the button rect.
    /// Points over the closed button (or outside the list) return null;
    /// hit-test the button itself with isPointInRect.
    /// Rows are half-open [y0, y1); the list bottom edge maps to the last item.
    /// (See ui/input_state.zig.)
    pub fn dropdownHit(rect: [4]f32, item_h: f32, count: usize, mx: f32, my: f32) ?usize {
        return ui_input.dropdownHit(rect, item_h, count, mx, my);
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
        ui_widgets.drawDropdown(self, rect, label, items, selected, open, hover_index, font_size);
    }

    // ------------------------------------------------------------------
    // Scroll (caller owns ScrollState; UICanvas stays stateless)
    // ------------------------------------------------------------------

    /// Applies a wheel delta and clamps offset into 0..content-view.
    /// Resets offset to 0 when the content fits in the view.
    /// (See ui/input_state.zig.)
    pub fn scrollClamp(state: *ScrollState, delta: f32) void {
        ui_input.scrollClamp(state, delta);
    }

    /// Offset that makes [item_y, item_y+item_h] visible with minimal movement.
    /// Returns 0 when the content fits in the view.
    /// (See ui/input_state.zig.)
    pub fn scrollOffsetForItem(offset: f32, item_y: f32, item_h: f32, view_h: f32, content_h: f32) f32 {
        return ui_input.scrollOffsetForItem(offset, item_y, item_h, view_h, content_h);
    }

    /// Thumb rect inside a vertical track [x, y, w, h]. Full track when the
    /// content fits; otherwise the thumb height is proportional to
    /// view/content (16px minimum) and its position maps the offset
    /// linearly over 0..content-view.
    /// (See ui/input_state.zig.)
    pub fn scrollbarThumbRect(track: [4]f32, content_h: f32, view_h: f32, offset: f32) [4]f32 {
        return ui_input.scrollbarThumbRect(track, content_h, view_h, offset);
    }

    /// Draws the scrollbar track + thumb (slider-like colors; see ui/widgets.zig).
    pub fn drawScrollbar(self: *UICanvas, track: [4]f32, content_h: f32, view_h: f32, offset: f32) void {
        ui_widgets.drawScrollbar(self, track, content_h, view_h, offset);
    }

    // ------------------------------------------------------------------
    // Text input (caller owns TextInputState + the focus flag)
    // ------------------------------------------------------------------

    /// Single-line field: panel, text, and a 2px cursor bar at the cursor
    /// byte offset when focused. No blink timer (static line) and no
    /// clipping: overlong text overflows the frame, the caller may shorten
    /// or scroll it. The cursor x reuses measureText so it matches the
    /// drawText advances exactly, byte for byte.
    /// (See ui/widgets.zig.)
    pub fn drawTextInput(self: *UICanvas, rect: [4]f32, state: *const TextInputState, focused: bool, font_size: f32) void {
        ui_widgets.drawTextInput(self, rect, state, focused, font_size);
    }

    /// Returns the pixel dimensions of a text string (see ui/text.zig).
    pub fn measureText(text: []const u8, font_size: f32) Vec2 {
        return ui_text.measureText(text, font_size);
    }

    /// Hit test helper: checks if a 2D screen coordinate (e.g. mouse cursor) is inside a rectangle
    /// (see ui/input_state.zig).
    pub fn isPointInRect(px: f32, py: f32, x: f32, y: f32, w: f32, h: f32) bool {
        return ui_input.isPointInRect(px, py, x, y, w, h);
    }

    /// Maps a mouse x coordinate to a 0..1 slider value, clamped
    /// (see ui/input_state.zig).
    pub fn sliderValueAt(x: f32, w: f32, mouse_x: f32) f32 {
        return ui_input.sliderValueAt(x, w, mouse_x);
    }

    /// Returns the checkbox hit rect as [x, y, w, h] for use with isPointInRect
    /// (see ui/input_state.zig).
    pub fn checkboxHitRect(x: f32, y: f32, size: f32) [4]f32 {
        return ui_input.checkboxHitRect(x, y, size);
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
    /// the clamped vertex type. Shared with the P6 frame path (same bytes
    /// whether the upload runs in prepare or in the legacy render).
    /// (See ui/draw.zig.)
    pub fn batchUploadBytes(vert_count: usize, index_count: usize) usize {
        return ui_draw.batchUploadBytes(vert_count, index_count);
    }

    /// True when this sokol frame's single `updateBuffer` per buffer is
    /// already spent on the canvas-owned pair. Read-only SDK metadata
    /// (`queryStats`, no GPU writes), `isvalid`-gated and headless-safe:
    /// without a context no commit can exist, so the armed flag alone
    /// decides (never-uploaded canvases stay closed; tests override it
    /// directly to simulate an open window).
    pub fn isUploadOpen(self: *const UICanvas) bool {
        if (!self.ui_upload_armed) return false;
        if (!sg.isvalid()) return true;
        return sg.queryStats().prev_frame.frame_index == self.ui_upload_commit;
    }

    /// Marks a successful upload: arms the window with the current commit
    /// watermark and bumps the upload sequence. Called by both upload paths
    /// (frame + legacy immediate) right after their `updateBuffer` pair
    /// lands — so the sequence identifies the last writer unambiguously,
    /// even for two uploads inside one sokol frame (the watermark alone
    /// cannot: the marker rotates only at commit).
    pub fn markUiUploaded(self: *UICanvas) void {
        self.ui_upload_commit = if (sg.isvalid()) sg.queryStats().prev_frame.frame_index else 0;
        self.ui_upload_armed = true;
        self.ui_upload_seq +%= 1;
    }

    /// u16-clamped drawable vertex prefix (indices address vertices as
    /// u16). Pure for tests; shared by the legacy render and the P6 frame.
    /// (See ui/draw.zig.)
    pub fn clampedVertCount(len: usize) usize {
        return ui_draw.clampedVertCount(len);
    }

    /// Replacement capacity on growth (same formula as the legacy path).
    /// Pure for tests. (See ui/draw.zig.)
    pub fn grownCapacity(current: usize, need: usize) usize {
        return ui_draw.grownCapacity(current, need);
    }

    /// Outcome of `ensureUiBufferPair`: buffers to upload into + install,
    /// with per-buffer replacement flags. On `ok == false` the current
    /// pair is untouched (only uncommitted new handles were destroyed).
    pub const UiBufferEnsure = struct {
        ok: bool = false,
        vertex_buffer: sg.Buffer = .{},
        index_buffer: sg.Buffer = .{},
        capacity_vertices: usize = 0,
        capacity_indices: usize = 0,
        replaced_vb: bool = false,
        replaced_ib: bool = false,
    };

    /// Ensures canvas GPU buffers cover the batch: creates + VALIDates ALL
    /// needed replacements BEFORE returning, never installs a FAILED
    /// handle. Current handles are VALID-checked too — an id==0 (or
    /// otherwise invalid) current with a large capacity is recreated, never
    /// updated blindly. On any failure only the uncommitted new handle(s)
    /// are destroyed and `ok` is false with the current pair untouched.
    /// No retire here: the caller installs the pair and retires/destroys
    /// the replaced handles itself (queue for the Scene frame path with
    /// live epochs; immediate destroy for the standalone immediate path,
    /// whose caller asserts no outstanding snapshots). Caller must hold a
    /// sokol context (all `sg.*` below assert it).
    pub fn ensureUiBufferPair(
        cur_vb: sg.Buffer,
        cur_ib: sg.Buffer,
        cap_v: usize,
        cap_i: usize,
        need_v: usize,
        need_i: usize,
    ) UiBufferEnsure {
        // A failed makeBuffer may hand out a nonzero FAILED id (pool
        // exhaustion is id == 0 only): validity is the state query, here
        // and for the current pair.
        const want_vb = need_v > cap_v or cur_vb.id == 0 or sg.queryBufferState(cur_vb) != .VALID;
        const want_ib = need_i > cap_i or cur_ib.id == 0 or sg.queryBufferState(cur_ib) != .VALID;
        const target_cap_v = if (need_v > cap_v) grownCapacity(cap_v, need_v) else cap_v;
        const target_cap_i = if (need_i > cap_i) grownCapacity(cap_i, need_i) else cap_i;
        var new_vb: sg.Buffer = .{};
        var new_ib: sg.Buffer = .{};
        if (want_vb) {
            new_vb = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = target_cap_v * @sizeOf(UIVertex),
            });
            if (new_vb.id == 0 or sg.queryBufferState(new_vb) != .VALID) {
                if (new_vb.id != 0) sg.destroyBuffer(new_vb);
                return .{ .ok = false };
            }
        }
        if (want_ib) {
            new_ib = sg.makeBuffer(.{
                .usage = .{ .index_buffer = true, .dynamic_update = true },
                .size = target_cap_i * @sizeOf(u16),
            });
            if (new_ib.id == 0 or sg.queryBufferState(new_ib) != .VALID) {
                if (new_ib.id != 0) sg.destroyBuffer(new_ib);
                if (new_vb.id != 0) sg.destroyBuffer(new_vb);
                return .{ .ok = false };
            }
        }
        return .{
            .ok = true,
            .vertex_buffer = if (want_vb) new_vb else cur_vb,
            .index_buffer = if (want_ib) new_ib else cur_ib,
            .capacity_vertices = target_cap_v,
            .capacity_indices = target_cap_i,
            .replaced_vb = want_vb,
            .replaced_ib = want_ib,
        };
    }

    /// Uploads one UI batch into the given buffers (the single allowed
    /// update per buffer per sokol frame) and records the exact bytes.
    /// Shared by the legacy immediate render and the P6 frame upload; both
    /// callers resolve WHICH buffers first (growth ownership differs).
    pub fn uploadUiBuffers(
        vertex_buffer: sg.Buffer,
        index_buffer: sg.Buffer,
        verts: []const UIVertex,
        indices: []const u16,
    ) void {
        sg.updateBuffer(vertex_buffer, sg.asRange(verts));
        sg.updateBuffer(index_buffer, sg.asRange(indices));
        // Учёт динамики: весь UI-батч кадра (вершины + u16-индексы).
        upload_meter.record(batchUploadBytes(verts.len, indices.len));
    }

    /// Draws an already-uploaded UI batch (no upload, no meter). Shared by
    /// the legacy immediate render and the P6 upload-free frame draw.
    pub fn drawUiBuffers(
        pipeline: sg.Pipeline,
        vertex_buffer: sg.Buffer,
        index_buffer: sg.Buffer,
        font_view: sg.View,
        font_sampler: sg.Sampler,
        screen_w: f32,
        screen_h: f32,
        index_count: usize,
    ) void {
        if (pipeline.id == 0) return;
        sg.applyPipeline(pipeline);

        var bind = sg.Bindings{};
        bind.vertex_buffers[0] = vertex_buffer;
        bind.index_buffer = index_buffer;
        bind.views[ui_shd.VIEW_font_tex] = font_view;
        bind.samplers[ui_shd.SMP_smp] = font_sampler;
        sg.applyBindings(bind);

        const vs_params = ui_shd.VsParams{
            .screen_size = .{ screen_w, screen_h, 0.0, 0.0 },
        };
        sg.applyUniforms(ui_shd.UB_vs_params, sg.asRange(&vs_params));

        sg.draw(0, @intCast(index_count), 1);
    }

    /// Uploads dynamic batch buffers and executes the UI render pass.
    /// Standalone immediate path, signature unchanged: the caller owns the
    /// frame discipline (a `sg.commit` between renders reopens the window
    /// automatically — no manual reset, standalone `render();
    /// sg.commit(); render()` draws twice).
    ///
    /// Borrower lifetime: this call is a buffer WRITER — it invalidates any
    /// previously committed `UiFrame` on this canvas (same buffers
    /// overwritten, or replaced on growth). Drawing an old frame after this
    /// render without a recapture is invalid; Scene fail-closes that mix via
    /// the upload identity, standalone callers own the discipline. No
    /// blanket ban: epochs between uploads are freely usable — only
    /// cross-writer consumption without recapture is invalid, and retired /
    /// destroyed snapshots must never be consumed.
    pub fn render(self: *UICanvas, screen_w: f32, screen_h: f32) void {
        if (self.vertices.items.len == 0 or self.indices.items.len == 0) return;
        if (screen_w <= 0.0 or screen_h <= 0.0) return;
        // P6 same-frame guard: this sokol frame's single update on these
        // buffers is already spent — fail close (no upload, no draw: the
        // live lists may differ from the GPU-resident data). Checked before
        // the context gate so the policy holds headless too.
        if (self.isUploadOpen()) return;
        // Headless/tools (no sokol context): safe no-op instead of an
        // assert trap inside sg.*.
        if (!sg.isvalid()) return;

        // usize on purpose: `@min(usize, u16)` resolves to u16 in Zig 0.16, and
        // 48 B/vertex would then overflow the u16 multiply at 1366 vertices.
        const vert_count: usize = clampedVertCount(self.vertices.items.len);
        // Shared validate-all routine (same as the Scene frame path):
        // every needed replacement is created + VALIDated before any
        // upload/install, so a FAILED handle is never installed and the
        // current pair survives a pair failure untouched.
        const ensured = ensureUiBufferPair(
            self.vertex_buffer,
            self.index_buffer,
            self.capacity_vertices,
            self.capacity_indices,
            vert_count,
            self.indices.items.len,
        );
        if (!ensured.ok) return;
        // Standalone ownership: replaced handles are destroyed immediately —
        // the caller asserts no outstanding snapshots reference them (the
        // Scene frame path retires through the epoch queue instead).
        if (ensured.replaced_vb) {
            if (self.vertex_buffer.id != 0) sg.destroyBuffer(self.vertex_buffer);
            self.vertex_buffer = ensured.vertex_buffer;
            self.capacity_vertices = ensured.capacity_vertices;
        }
        if (ensured.replaced_ib) {
            if (self.index_buffer.id != 0) sg.destroyBuffer(self.index_buffer);
            self.index_buffer = ensured.index_buffer;
            self.capacity_indices = ensured.capacity_indices;
        }

        uploadUiBuffers(self.vertex_buffer, self.index_buffer, self.vertices.items[0..vert_count], self.indices.items);
        self.markUiUploaded();

        drawUiBuffers(
            self.pipeline,
            self.vertex_buffer,
            self.index_buffer,
            self.font_texture.view,
            self.font_texture.sampler,
            screen_w,
            screen_h,
            self.indices.items.len,
        );
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

// Moved unit tests: glyph/text tests live in ui/text.zig, hit-test/scroll/
// text-input tests in ui/input_state.zig, corner/batch tests in ui/draw.zig
// (same test names, same assertions; only the callee paths changed).

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

// Note: the "batchUploadBytes keeps the u16 vertex cap in usize arithmetic"
// test moved to ui/draw.zig with the helper (same name, same assertions).
