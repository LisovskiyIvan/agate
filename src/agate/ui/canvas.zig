//! UI canvas: owns the `UICanvas` type — the immediate-mode draw state,
//! GPU resources, style cascade storage, and input state — plus the trivial
//! lifecycle (`setInput` / `initCpuOnly` / `init` / `deinit` / `begin`) and
//! thin forwarders into the focused siblings, so every call site keeps
//! working unchanged:
//!
//! - `draw.zig` — draw primitives (`drawRect`, `drawLine`, ...).
//! - `text.zig` — SDF + TrueType text (`drawText`, `measureText`, ...).
//! - `widgets.zig` — stateless controls (`drawButton`, `drawDropdown`, ...).
//! - `input_state.zig` — hit-testing and scroll geometry.
//! - `style.zig` — cascade, transitions, styled drawing (`drawStyledButton`, ...).
//! - `font.zig` — font atlas creation and the TrueType override.
//! - `gpu.zig` — buffer-pair ensure, upload/draw, same-frame guard, `render`.
//!
//! Anti-cycle rule (same as `scene/`, `profiler/`): siblings take the canvas
//! as `anytype` and never import this module or the `ui.zig` facade back;
//! this module passes `self` straight through. `ui.zig` re-exports `UICanvas`
//! under its historical path.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const sapp = sokol.app;

const math = @import("math");
const Vec2 = math.Vec2;
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color4 = math.Color4;

const Texture = @import("../texture.zig").Texture;
const ui_shd = @import("ui_shader");

const ui_types = @import("types.zig");
const ui_theme_mod = @import("theme.zig");
const ui_transition = @import("transition.zig");
const css_parser = @import("css_parser.zig");
const ui_layout = @import("layout.zig");
const ui_draw = @import("draw.zig");
const ui_text = @import("text.zig");
const ui_widgets = @import("widgets.zig");
const ui_input = @import("input_state.zig");
const ui_font = @import("font.zig");
const ui_style = @import("style.zig");
const ui_gpu = @import("gpu.zig");

const UIVertex = ui_draw.UIVertex;
const ScrollState = ui_input.ScrollState;
const TextInputState = ui_input.TextInputState;
const UIStyleSet = ui_types.UIStyleSet;
const UIStyle = ui_types.UIStyle;
const UIStyleRequest = ui_types.UIStyleRequest;
const UIStyleKind = ui_types.UIStyleKind;
const UIStyledOptions = ui_types.UIStyledOptions;
const UITheme = ui_theme_mod.UITheme;
const UIStyleTransition = ui_transition.UIStyleTransition;
const TtfFont = ui_font.TtfFont;

pub const UICanvas = struct {
    allocator: std.mem.Allocator,
    vertices: std.ArrayListUnmanaged(UIVertex) = .empty,
    indices: std.ArrayListUnmanaged(u16) = .empty,

    vertex_buffer: sg.Buffer = .{},
    index_buffer: sg.Buffer = .{},
    pipeline: sg.Pipeline = .{},
    font_texture: Texture,

    /// Optional TrueType font override (borrowed; the caller owns the
    /// `TtfFont` lifetime and must keep both it and its sfnt bytes alive
    /// while set). When non-null, every drawText* call emits mode-3
    /// coverage quads from the TTF atlas and canvas-aware measurement
    /// uses TTF advances; null (default) keeps the bitmap/SDF font
    /// bit-identical. Switch with setFontTtf/clearFontTtf.
    ttf_font: ?*const TtfFont = null,
    /// GPU upload of the TTF atlas, created by setFontTtf when a sokol
    /// context is live (headless/tests keep CPU-only state and still
    /// emit TTF vertices). Destroyed by clearFontTtf/deinit.
    ttf_texture: ?Texture = null,

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

    // Style system state (see style.zig).
    /// Default styles per widget kind; overridden wholesale by assignment
    /// (theme = "all widgets at once") or per kind (`canvas.theme.button = ...`).
    theme: UITheme = UITheme.defaults(),
    /// Named style classes ("panel.dark"); name slices are caller-owned
    /// (typically comptime literals — immediate mode never copies strings).
    style_classes: [ui_types.max_style_classes]ui_types.UIStyleClass = [_]ui_types.UIStyleClass{.{ .name = "", .set = .{} }} ** ui_types.max_style_classes,
    style_class_count: usize = 0,
    /// Retained per-widget style transitions (see resolveAnimatedStyle).
    style_transitions: [ui_transition.max_style_transitions]UIStyleTransition = [_]UIStyleTransition{.{}} ** ui_transition.max_style_transitions,
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

    /// Uploads the embedded SDF bitmap font to a GPU texture (see ui/font.zig).
    pub fn makeFontTexture(allocator: std.mem.Allocator) !Texture {
        return ui_font.makeFontTexture(allocator);
    }

    /// CPU-only canvas: no GPU handles (zero buffers/pipeline/font texture),
    /// usable headless for drawing into the CPU-side lists and driving input
    /// state. `render`/capture/upload need a live context and real handles;
    /// either complete it with `init`-style GPU creation on the context
    /// thread or hand its lists to an owner that uploads them itself (the
    /// 3D-GUI layer path). Never call `deinit` on a canvas whose
    /// `font_texture` was not created via `makeFontTexture`/`init`
    /// (`Texture.deinit` issues `sg.destroy*` unconditionally) — free the
    /// CPU lists directly instead.
    pub fn initCpuOnly(allocator: std.mem.Allocator) UICanvas {
        return .{
            .allocator = allocator,
            .font_texture = std.mem.zeroes(Texture),
        };
    }

    pub fn init(allocator: std.mem.Allocator) !UICanvas {
        const tex = try ui_font.makeFontTexture(allocator);

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
        if (self.ttf_texture) |*t| t.deinit();
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
    /// or scroll it. The cursor x reuses canvas-aware measurement so it
    /// matches the drawText advances exactly, byte for byte.
    /// (See ui/widgets.zig.)
    pub fn drawTextInput(self: *UICanvas, rect: [4]f32, state: *const TextInputState, focused: bool, font_size: f32) void {
        ui_widgets.drawTextInput(self, rect, state, focused, font_size);
    }

    /// Returns the pixel dimensions of a text string (see ui/text.zig).
    pub fn measureText(text: []const u8, font_size: f32) Vec2 {
        return ui_text.measureText(text, font_size);
    }

    /// Switches text drawing to a TrueType font (`null` restores the
    /// bitmap/SDF default). The font is borrowed, not copied (see ui/font.zig).
    pub fn setFontTtf(self: *UICanvas, font: ?*const TtfFont) void {
        ui_font.setFontTtf(self, font);
    }

    /// Restores the bitmap/SDF font (see setFontTtf).
    pub fn clearFontTtf(self: *UICanvas) void {
        ui_font.clearFontTtf(self);
    }

    /// True when a TrueType font is installed.
    pub fn hasTtfFont(self: *const UICanvas) bool {
        return ui_font.hasTtfFont(self);
    }

    /// Font view the draw binds: the TTF atlas upload when a font is set
    /// and uploaded, else the bitmap/SDF atlas. Shared by the legacy
    /// render and the P6 frame capture so both bind the same font.
    pub fn activeFontView(self: *const UICanvas) sg.View {
        return ui_font.activeFontView(self);
    }

    /// Sampler matching `activeFontView`.
    pub fn activeFontSampler(self: *const UICanvas) sg.Sampler {
        return ui_font.activeFontSampler(self);
    }

    /// Canvas-aware measurement: TTF advances when a font is set (so UI
    /// layout matches the drawn TTF glyphs), the legacy monospace math
    /// otherwise. Static `measureText` keeps the legacy contract for
    /// callers that never install a font.
    pub fn measureTextCurrent(self: *const UICanvas, text: []const u8, font_size: f32) Vec2 {
        return ui_font.measureTextCurrent(self, text, font_size);
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
    // Styled rendering (CSS-like cascade; see ui/style.zig)
    // ------------------------------------------------------------------

    /// Registers (or replaces) a named style class. Registration order is
    /// irrelevant; replacing keeps the original slot.
    pub fn setStyleClass(self: *UICanvas, name: []const u8, set: UIStyleSet) void {
        ui_style.setStyleClass(self, name, set);
    }

    /// Looks up a registered class by name.
    pub fn styleClass(self: *const UICanvas, name: []const u8) ?UIStyleSet {
        return ui_style.styleClass(self, name);
    }

    /// Full cascade resolution: theme default for the widget kind, then the
    /// named class (if registered), then the per-call override. Each layer's
    /// state delta applies within that layer, so e.g. a hover delta from the
    /// theme still shows through a class that only sets a border.
    pub fn resolveStyle(self: *const UICanvas, request: UIStyleRequest) UIStyle {
        return ui_style.resolveStyle(self, request);
    }

    /// `resolveStyle` plus the transition layer: with `opts.anim_key` set,
    /// the widget animates from its previously drawn style toward the newly
    /// resolved target over the cascade-resolved TransitionOptions. State
    /// changes (hover/active/focus/disabled) trigger transitions implicitly —
    /// they simply change the resolved target. Without `anim_key` this is
    /// exactly resolveStyle (stateless, no per-widget storage).
    pub fn resolveAnimatedStyle(self: *UICanvas, kind: UIStyleKind, opts: UIStyledOptions) UIStyle {
        return ui_style.resolveAnimatedStyle(self, kind, opts);
    }

    /// Installs a parsed CSS theme: replaces the kind defaults wholesale and
    /// registers/updates every parsed class (per class name, latest parse
    /// wins). Diagnostics are the caller's to inspect — a parse with errors
    /// still yields a usable partial theme.
    pub fn applyCssTheme(self: *UICanvas, parsed: css_parser.CssTheme) void {
        ui_style.applyCssTheme(self, parsed);
    }

    /// Renders one resolved style: shadow, background (flat or gradient
    /// bands) and border. Fully transparent styles draw nothing, so
    /// containers can render their style unconditionally.
    pub fn drawStyleRect(self: *UICanvas, rect: [4]f32, style: UIStyle) void {
        ui_style.drawStyleRect(self, rect, style);
    }

    /// Fills a rounded rectangle with solid quads: one middle rect plus thin
    /// horizontal bands approximating the corner arcs (2px resolution).
    pub fn drawRectRoundedFill(self: *UICanvas, x: f32, y: f32, w: f32, h: f32, radius: f32, color: Color4) void {
        ui_style.drawRectRoundedFill(self, x, y, w, h, radius, color);
    }

    /// Strokes a rounded-rectangle border: four straight segments between
    /// the corner arcs plus per-band arc segments. The inner edge is
    /// concentric with the outer arc (inset by the border width); rows the
    /// inner arc does not reach are solid border strips.
    pub fn drawRectRoundedOutline(self: *UICanvas, x: f32, y: f32, w: f32, h: f32, radius: f32, thickness: f32, color: Color4) void {
        ui_style.drawRectRoundedOutline(self, x, y, w, h, radius, thickness, color);
    }

    /// Styled panel: renders the resolved panel style into `rect`.
    pub fn drawStyledPanel(self: *UICanvas, rect: [4]f32, opts: UIStyledOptions) void {
        ui_style.drawStyledPanel(self, rect, opts);
    }

    /// Styled button: resolved button style plus centered outlined text.
    /// Pairs with LayoutStack: `const r = ls.place(w, h); canvas.drawStyledButton("Ok", r, 14, .{});`
    pub fn drawStyledButton(self: *UICanvas, text: []const u8, rect: [4]f32, font_size: f32, opts: UIStyledOptions) void {
        ui_style.drawStyledButton(self, text, rect, font_size, opts);
    }

    /// Styled checkbox: box from the resolved style (state via `opts`),
    /// white check mark as in the legacy widget, optional label to the right.
    pub fn drawStyledCheckbox(self: *UICanvas, rect: [4]f32, checked: bool, label: ?[]const u8, label_size: f32, opts: UIStyledOptions) void {
        ui_style.drawStyledCheckbox(self, rect, checked, label, label_size, opts);
    }

    /// Styled horizontal slider. Track from the resolved style, fill from
    /// the accent (style override wins over the theme accent). Returns the
    /// clamped value like the legacy widget.
    pub fn drawStyledSlider(self: *UICanvas, rect: [4]f32, value: f32, opts: UIStyledOptions) f32 {
        return ui_style.drawStyledSlider(self, rect, value, opts);
    }

    /// Styled badge: resolved style plus text at the badge padding. The
    /// rect is caller-provided (measure the text and use LayoutStack.place).
    pub fn drawStyledBadge(self: *UICanvas, text: []const u8, rect: [4]f32, font_size: f32, opts: UIStyledOptions) void {
        ui_style.drawStyledBadge(self, text, rect, font_size, opts);
    }

    // ------------------------------------------------------------------
    // GPU upload + draw path (see ui/gpu.zig)
    // ------------------------------------------------------------------

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
    /// already spent on the canvas-owned pair (see ui/gpu.zig).
    pub fn isUploadOpen(self: *const UICanvas) bool {
        return ui_gpu.isUploadOpen(self);
    }

    /// Marks a successful upload: arms the window with the current commit
    /// watermark and bumps the upload sequence (see ui/gpu.zig).
    pub fn markUiUploaded(self: *UICanvas) void {
        ui_gpu.markUiUploaded(self);
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
    pub const UiBufferEnsure = ui_gpu.UiBufferEnsure;

    /// Ensures canvas GPU buffers cover the batch (see ui/gpu.zig).
    pub fn ensureUiBufferPair(
        cur_vb: sg.Buffer,
        cur_ib: sg.Buffer,
        cap_v: usize,
        cap_i: usize,
        need_v: usize,
        need_i: usize,
    ) UiBufferEnsure {
        return ui_gpu.ensureUiBufferPair(cur_vb, cur_ib, cap_v, cap_i, need_v, need_i);
    }

    /// Uploads one UI batch into the given buffers (see ui/gpu.zig).
    pub fn uploadUiBuffers(
        vertex_buffer: sg.Buffer,
        index_buffer: sg.Buffer,
        verts: []const UIVertex,
        indices: []const u16,
    ) void {
        ui_gpu.uploadUiBuffers(vertex_buffer, index_buffer, verts, indices);
    }

    /// Draws an already-uploaded UI batch (see ui/gpu.zig).
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
        ui_gpu.drawUiBuffers(pipeline, vertex_buffer, index_buffer, font_view, font_sampler, screen_w, screen_h, index_count);
    }

    /// Uploads dynamic batch buffers and executes the UI render pass.
    /// Standalone immediate path (see ui/gpu.zig for the lifetime contract).
    pub fn render(self: *UICanvas, screen_w: f32, screen_h: f32) void {
        ui_gpu.render(self, screen_w, screen_h);
    }
};
