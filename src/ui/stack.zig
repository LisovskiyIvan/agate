//! UI layout containers: the immediate-mode `LayoutStack` cursor (hstack,
//! vstack, flex, grid), the pure grid/alignment helpers, and the per-widget
//! layout option structs.
//!
//! `LayoutStack` is generic over the canvas pointer type (`LayoutStack(*UICanvas)`
//! in practice, pinned by the `ui.zig` facade) so this module never imports
//! `../ui.zig` or `canvas.zig` — following the `scene/` precedent, subsystems
//! never import the facade. Drawing and style resolution are reached through
//! the canvas value's own methods (same discipline as `widgets.zig`); the pure
//! sizing math comes from the sibling leaves directly (`layout.zig` for
//! `UISize`/`UIEdges`/anchors, `input_state.zig` for hit-testing). `ui.zig`
//! re-exports every public name under its historical path so all call sites
//! keep working unchanged.
//!
//! `LayoutStack` is a small stack of container frames. A widget is placed
//! with `place(w, h)` which returns the rect to draw it into and advances
//! the container cursor — existing widgets keep their absolute-coordinate
//! signatures, the caller just feeds them the returned rect. Nesting is
//! placing a rect and opening the next container on it:
//!
//!   var ls = LayoutStack(UICanvas).init(canvas);
//!   defer ls.reset();
//!   if (ls.beginVStack(panel_rect, .{ .padding = 8, .spacing = 4 })) {
//!       const b = ls.place(120, 24);
//!       canvas.drawButton("Ok", b[0], b[1], b[2], b[3], 14, false, false);
//!       const row = ls.place(120, 24);
//!       _ = ls.beginHStack(row, .{ .spacing = 8 });
//!       ...
//!       ls.end();
//!       ls.end();
//!   }

const std = @import("std");

const math = @import("math");
const Color4 = math.Color4;

const ui_layout = @import("layout.zig");
const ui_types = @import("types.zig");
const ui_input = @import("input_state.zig");
const text_mod = @import("text.zig");

const UISize = ui_layout.UISize;
const UIEdges = ui_layout.UIEdges;
const UIAnchor = ui_layout.UIAnchor;
const UIDock = ui_layout.UIDock;
const anchorRect = ui_layout.anchorRect;
const dockRect = ui_layout.dockRect;
const FlexDirection = ui_layout.FlexDirection;
const UIState = ui_types.UIState;
const UIStyle = ui_types.UIStyle;
const UIStyleOverride = ui_types.UIStyleOverride;
const UIStyledOptions = ui_types.UIStyledOptions;
const UIBoxStyle = ui_types.UIBoxStyle;

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

pub fn LayoutStack(comptime Canvas: type) type {
    return struct {
        const Self = @This();
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

        canvas: *Canvas,
        frames: [max_depth]Frame = undefined,
        depth: usize = 0,

        pub fn init(canvas: *Canvas) Self {
            return .{ .canvas = canvas };
        }

        /// Drops all open frames (start of a new UI frame).
        pub fn reset(self: *Self) void {
            self.depth = 0;
        }

        pub fn beginHStack(self: *Self, rect: [4]f32, opts: LayoutFlowOptions) bool {
            return self.beginFlow(rect, opts, .hstack);
        }

        pub fn beginVStack(self: *Self, rect: [4]f32, opts: LayoutFlowOptions) bool {
            return self.beginFlow(rect, opts, .vstack);
        }

        pub fn beginFlex(self: *Self, rect: [4]f32, opts: LayoutFlexOptions) bool {
            return self.beginFlow(rect, .{
                .padding = opts.padding,
                .padding_edges = opts.padding_edges,
                .spacing = opts.spacing,
                .align_cross = opts.align_cross,
                .class = opts.class,
                .style = opts.style,
            }, if (opts.direction.isRow()) .hstack else .vstack);
        }

        pub fn beginGrid(self: *Self, rect: [4]f32, opts: LayoutGridOptions) bool {
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

        pub fn end(self: *Self) void {
            if (self.depth > 0) self.depth -= 1;
        }
        pub const endFlow = end;
        pub const endGrid = end;

        /// Returns the inner content bounds [x, y, w, h] of the current container.
        pub fn innerRect(self: *const Self) [4]f32 {
            if (self.depth == 0) return .{ 0, 0, 0, 0 };
            const f = &self.frames[self.depth - 1];
            return .{ f.ix, f.iy, f.iw, f.ih };
        }

        /// Places the next widget with the container's cross alignment.
        /// Returns the rect to draw the widget into.
        pub fn place(self: *Self, w: f32, h: f32) [4]f32 {
            return self.placeAligned(w, h, null);
        }

        /// `place` with a per-call cross-axis alignment override.
        pub fn placeAligned(self: *Self, w: f32, h: f32, align_cross: ?LayoutAlignCross) [4]f32 {
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
        pub fn placeSize(self: *Self, w: UISize, h: UISize) [4]f32 {
            return self.placeSizeWithAuto(w, h, 0.0, 0.0);
        }

        /// Places a widget dimensioned via UISize with explicit auto content dimensions.
        pub fn placeSizeWithAuto(self: *Self, w: UISize, h: UISize, auto_w: f32, auto_h: f32) [4]f32 {
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
        pub fn placeFlex(self: *Self, weight: f32) [4]f32 {
            _ = weight;
            return self.placeSize(.fill, .fill);
        }

        /// Advances the layout cursor by taking up all remaining main-axis space.
        pub fn spacer(self: *Self) [4]f32 {
            return self.spacerWeight(1.0);
        }

        pub fn spacerWeight(self: *Self, weight: f32) [4]f32 {
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
            self: *Self,
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
        pub fn anchor(self: *Self, w: f32, h: f32, anchor_pt: UIAnchor, margin: UIEdges) [4]f32 {
            if (self.depth == 0) return .{ 0, 0, 0, 0 };
            const f = &self.frames[self.depth - 1];
            return anchorRect(.{ f.ix, f.iy, f.iw, f.ih }, w, h, anchor_pt, margin);
        }

        /// Docks an element to a side of the current frame and shrinks the remaining inner area.
        pub fn dock(self: *Self, dock_side: UIDock, size: f32, margin: UIEdges) [4]f32 {
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
        pub fn label(self: *Self, text: []const u8, opts: LayoutLabelOptions) void {
            const auto_w = text_mod.measureForCanvas(self.canvas, text, opts.font_size).x;
            const auto_h = opts.font_size;
            const r = self.placeSizeWithAuto(opts.width, opts.height, auto_w, auto_h);
            const tx = r[0] + @max((r[2] - auto_w) * 0.5, 0.0);
            const ty = r[1] + @max((r[3] - auto_h) * 0.5, 0.0);
            self.canvas.drawTextWithOutline(text, tx, ty, opts.font_size, opts.color, opts.outline_width);
        }

        /// Places and renders a button widget. Returns true if clicked.
        pub fn button(self: *Self, text: []const u8, opts: LayoutButtonOptions) bool {
            const auto_w = text_mod.measureForCanvas(self.canvas, text, opts.font_size).x + 24.0;
            const auto_h = opts.font_size + 14.0;
            const r = self.placeSizeWithAuto(opts.width, opts.height, auto_w, auto_h);
            const hov = opts.is_hovered orelse ui_input.isPointInRect(self.canvas.mouse_pos[0], self.canvas.mouse_pos[1], r[0], r[1], r[2], r[3]);
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
        pub fn checkbox(self: *Self, label_text: ?[]const u8, checked: *bool, opts: LayoutCheckboxOptions) bool {
            const text_w = if (label_text) |t| text_mod.measureForCanvas(self.canvas, t, opts.label_size).x + 8.0 else 0.0;
            const total_w = opts.size + text_w;
            const r = self.place(total_w, @max(opts.size, opts.label_size));
            const hov = opts.is_hovered orelse ui_input.isPointInRect(self.canvas.mouse_pos[0], self.canvas.mouse_pos[1], r[0], r[1], r[2], r[3]);

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
        pub fn slider(self: *Self, value: f32, min_val: f32, max_val: f32, opts: LayoutSliderOptions) f32 {
            const r = self.placeSizeWithAuto(opts.width, opts.height, 120.0, 20.0);
            const hov = opts.is_hovered orelse ui_input.isPointInRect(self.canvas.mouse_pos[0], self.canvas.mouse_pos[1], r[0], r[1], r[2], r[3]);
            const drag = opts.is_dragging orelse (hov and self.canvas.mouse_down);

            const norm = if (max_val > min_val) std.math.clamp((value - min_val) / (max_val - min_val), 0.0, 1.0) else 0.0;
            var new_norm = norm;

            if (drag) {
                new_norm = ui_input.sliderValueAt(r[0], r[2], self.canvas.mouse_pos[0]);
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
        pub fn progressBar(self: *Self, fraction: f32, opts: LayoutProgressOptions) void {
            const r = self.placeSizeWithAuto(opts.width, opts.height, 100.0, 16.0);
            self.canvas.drawProgressBar(r[0], r[1], r[2], r[3], fraction, opts.bg_color, opts.fill_color);
        }

        /// Places and renders a dividing line widget.
        pub fn divider(self: *Self, opts: LayoutDividerOptions) void {
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
        pub fn badge(self: *Self, text: []const u8, opts: LayoutBadgeOptions) void {
            const auto_w = text_mod.measureForCanvas(self.canvas, text, opts.font_size).x + 14.0;
            const auto_h = opts.font_size + 8.0;
            const r = self.place(auto_w, auto_h);
            self.canvas.drawBadge(text, r[0], r[1], opts.font_size, opts.bg_color, opts.text_color);
        }

        /// Places a widget box with a resolved style `margin` around it (the
        /// CSS-ish spacing between a box and its slot). Returns the content
        /// rect; the cursor advances by margin + size + margin + spacing.
        pub fn placeBox(self: *Self, w: f32, h: f32, box: ?UIBoxStyle) [4]f32 {
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

        fn beginFlow(self: *Self, rect: [4]f32, opts: LayoutFlowOptions, kind: FrameKind) bool {
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

        fn resolveContainerStyle(self: *Self, class: ?[]const u8, override: ?UIStyleOverride) UIStyle {
            return self.canvas.resolveStyle(.{
                .kind = .container,
                .class = class,
                .override = override,
            });
        }

        /// Next flow-frame rect: main axis at the cursor, cross axis aligned
        /// inside the inner extent (`stretch` expands to the inner extent).
        fn flowRect(self: *Self, w: f32, h: f32, cross: LayoutAlignCross) [4]f32 {
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
}
