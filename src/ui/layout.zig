//! Agate UI Layout System: Flexible containers, anchors, docking, and CSS-style grid.
//!
//! Features:
//! - `UISize`: fixed pixels, percentages (0..100%), flex grow factors, and auto/fit-content.
//! - `UIEdges`: per-side padding and margin (top, right, bottom, left) with symmetric and uniform helpers.
//! - `UIAnchor`: 9-point anchor positioning (top_left, center, bottom_right, etc.) with edge margins.
//! - `UIDock`: screen and panel docking (top, bottom, left, right, fill) with sequential container carving.
//! - `FlexLayout` / `solveFlex`: Flexbox-style flow layout supporting row/column, reverse directions,
//!   `justify_content` (start, center, end, space_between, space_around, space_evenly),
//!   `align_items` (start, center, end, stretch), and flex proportional expansion.
//! - `GridLayout` / `solveGridTracks`: track sizing with pixels, percentages, and `fr` flexible fractions,
//!   multi-cell spans (colspan, rowspan), and item alignment.
//! - Zero dynamic allocations in layout solving: pure deterministic math with caller-provided buffers.

const std = @import("std");
const math = @import("math");
const Color4 = math.Color4;

// ============================================================================
// 1. Sizing Dimensions & Per-Edge Insets
// ============================================================================

/// Dimension specification for UI widgets and container tracks.
pub const UISize = union(enum) {
    auto,
    fixed: f32,
    percent: f32, // 0.0 .. 100.0
    flex: f32, // Proportional grow factor (e.g. 1.0, 2.0)

    pub const fill = UISize{ .flex = 1.0 };

    pub fn px(val: f32) UISize {
        return .{ .fixed = @max(val, 0.0) };
    }

    pub fn pct(val: f32) UISize {
        return .{ .percent = std.math.clamp(val, 0.0, 100.0) };
    }

    pub fn grow(factor: f32) UISize {
        return .{ .flex = @max(factor, 0.0) };
    }

    /// Resolves size into concrete pixels against available space.
    pub fn resolve(self: UISize, available: f32, auto_content: f32) f32 {
        return switch (self) {
            .auto => @max(auto_content, 0.0),
            .fixed => |v| @max(v, 0.0),
            .percent => |p| @max(available * (p * 0.01), 0.0),
            .flex => |f| if (f > 0.0) @max(available, 0.0) else 0.0,
        };
    }
};

/// 4-edge insets for margins, paddings, and borders (Top, Right, Bottom, Left).
pub const UIEdges = struct {
    top: f32 = 0.0,
    right: f32 = 0.0,
    bottom: f32 = 0.0,
    left: f32 = 0.0,

    pub const zero = UIEdges{};

    pub fn all(val: f32) UIEdges {
        const v = @max(val, 0.0);
        return .{ .top = v, .right = v, .bottom = v, .left = v };
    }

    pub fn symmetric(horizontal_val: f32, vertical_val: f32) UIEdges {
        const h = @max(horizontal_val, 0.0);
        const v = @max(vertical_val, 0.0);
        return .{ .top = v, .bottom = v, .left = h, .right = h };
    }

    pub fn trbl(t: f32, r: f32, b: f32, l: f32) UIEdges {
        return .{
            .top = @max(t, 0.0),
            .right = @max(r, 0.0),
            .bottom = @max(b, 0.0),
            .left = @max(l, 0.0),
        };
    }

    pub fn horizontal(val: f32) UIEdges {
        const h = @max(val, 0.0);
        return .{ .left = h, .right = h };
    }

    pub fn vertical(val: f32) UIEdges {
        const v = @max(val, 0.0);
        return .{ .top = v, .bottom = v };
    }

    pub fn hTotal(self: UIEdges) f32 {
        return self.left + self.right;
    }

    pub fn vTotal(self: UIEdges) f32 {
        return self.top + self.bottom;
    }

    /// Insets a rect [x, y, w, h] by edges, clamping w and h to 0.
    pub fn inset(self: UIEdges, rect: [4]f32) [4]f32 {
        const w = @max(rect[2] - self.hTotal(), 0.0);
        const h = @max(rect[3] - self.vTotal(), 0.0);
        return .{ rect[0] + self.left, rect[1] + self.top, w, h };
    }

    /// Expands a rect [x, y, w, h] outwards by edges.
    pub fn outset(self: UIEdges, rect: [4]f32) [4]f32 {
        return .{
            rect[0] - self.left,
            rect[1] - self.top,
            rect[2] + self.hTotal(),
            rect[3] + self.vTotal(),
        };
    }
};

// ============================================================================
// 2. Anchors & Docking
// ============================================================================

/// 9-point 2D container anchor positions.
pub const UIAnchor = enum {
    top_left,
    top_center,
    top_right,
    center_left,
    center,
    center_right,
    bottom_left,
    bottom_center,
    bottom_right,
};

/// Computes a [x, y, w, h] rectangle placed inside `container` according to `anchor` and `margin`.
pub fn anchorRect(container: [4]f32, w_in: f32, h_in: f32, anchor: UIAnchor, margin: UIEdges) [4]f32 {
    const inner = margin.inset(container);
    const w = @min(w_in, inner[2]);
    const h = @min(h_in, inner[3]);

    const x: f32 = switch (anchor) {
        .top_left, .center_left, .bottom_left => inner[0],
        .top_center, .center, .bottom_center => inner[0] + @max((inner[2] - w) * 0.5, 0.0),
        .top_right, .center_right, .bottom_right => inner[0] + @max(inner[2] - w, 0.0),
    };

    const y: f32 = switch (anchor) {
        .top_left, .top_center, .top_right => inner[1],
        .center_left, .center, .center_right => inner[1] + @max((inner[3] - h) * 0.5, 0.0),
        .bottom_left, .bottom_center, .bottom_right => inner[1] + @max(inner[3] - h, 0.0),
    };

    return .{ x, y, w, h };
}

/// Docking edge placement.
pub const UIDock = enum {
    top,
    bottom,
    left,
    right,
    fill,
};

/// Carves a docked rectangle of `size` pixels from `container`, mutating `container`
/// to reflect the remaining space. Useful for toolbars, sidebars, and status strips.
pub fn dockRect(container: *[4]f32, dock: UIDock, size_in: f32, margin: UIEdges) [4]f32 {
    const size = @max(size_in, 0.0);
    switch (dock) {
        .top => {
            const h = @min(size, container.*[3]);
            const r = UIEdges.trbl(margin.top, margin.right, margin.bottom, margin.left).inset(.{
                container.*[0],
                container.*[1],
                container.*[2],
                h,
            });
            container.*[1] += h;
            container.*[3] = @max(container.*[3] - h, 0.0);
            return r;
        },
        .bottom => {
            const h = @min(size, container.*[3]);
            const y = container.*[1] + container.*[3] - h;
            const r = UIEdges.trbl(margin.top, margin.right, margin.bottom, margin.left).inset(.{
                container.*[0],
                y,
                container.*[2],
                h,
            });
            container.*[3] = @max(container.*[3] - h, 0.0);
            return r;
        },
        .left => {
            const w = @min(size, container.*[2]);
            const r = UIEdges.trbl(margin.top, margin.right, margin.bottom, margin.left).inset(.{
                container.*[0],
                container.*[1],
                w,
                container.*[3],
            });
            container.*[0] += w;
            container.*[2] = @max(container.*[2] - w, 0.0);
            return r;
        },
        .right => {
            const w = @min(size, container.*[2]);
            const x = container.*[0] + container.*[2] - w;
            const r = UIEdges.trbl(margin.top, margin.right, margin.bottom, margin.left).inset(.{
                x,
                container.*[1],
                w,
                container.*[3],
            });
            container.*[2] = @max(container.*[2] - w, 0.0);
            return r;
        },
        .fill => {
            const r = margin.inset(container.*);
            container.*[2] = 0.0;
            container.*[3] = 0.0;
            return r;
        },
    }
}

// ============================================================================
// 3. Flexbox Flow Layout Solver
// ============================================================================

pub const FlexDirection = enum {
    row,
    column,
    row_reverse,
    column_reverse,

    pub fn isRow(self: FlexDirection) bool {
        return self == .row or self == .row_reverse;
    }

    pub fn isReverse(self: FlexDirection) bool {
        return self == .row_reverse or self == .column_reverse;
    }
};

pub const JustifyContent = enum {
    start,
    center,
    end,
    space_between,
    space_around,
    space_evenly,
};

pub const AlignItems = enum {
    start,
    center,
    end,
    stretch,
};

pub const LayoutItem = struct {
    width: UISize = .auto,
    height: UISize = .auto,
    margin: UIEdges = .zero,
    align_self: ?AlignItems = null,
    min_width: f32 = 0.0,
    max_width: f32 = std.math.inf(f32),
    min_height: f32 = 0.0,
    max_height: f32 = std.math.inf(f32),
    auto_width: f32 = 0.0,
    auto_height: f32 = 0.0,
};

/// Solves 1D Flexbox layout along `direction`. Zero allocations: writes directly into `out_rects`.
pub fn solveFlex(
    container: [4]f32,
    direction: FlexDirection,
    items: []const LayoutItem,
    spacing: f32,
    justify: JustifyContent,
    align_items: AlignItems,
    out_rects: [][4]f32,
) void {
    const n = @min(items.len, out_rects.len);
    if (n == 0) return;

    const is_row = direction.isRow();
    const is_rev = direction.isReverse();
    const main_origin = if (is_row) container[0] else container[1];
    const cross_origin = if (is_row) container[1] else container[0];
    const main_avail = if (is_row) container[2] else container[3];
    const cross_avail = if (is_row) container[3] else container[2];

    // Pass 1: measure fixed, percent, and count total flex weights
    var total_fixed_main: f32 = 0.0;
    var total_flex: f32 = 0.0;
    var main_sizes: [64]f32 = [_]f32{0.0} ** 64;
    var cross_sizes: [64]f32 = [_]f32{0.0} ** 64;

    const count = @min(n, 64);
    for (0..count) |i| {
        const it = items[i];
        const main_sz = if (is_row) it.width else it.height;
        const cross_sz = if (is_row) it.height else it.width;
        const main_auto = if (is_row) it.auto_width else it.auto_height;
        const cross_auto = if (is_row) it.auto_height else it.auto_width;
        const main_min = if (is_row) it.min_width else it.min_height;
        const main_max = if (is_row) it.max_width else it.max_height;
        const cross_min = if (is_row) it.min_height else it.min_width;
        const cross_max = if (is_row) it.max_height else it.max_width;
        const m_extra = if (is_row) it.margin.hTotal() else it.margin.vTotal();
        const c_extra = if (is_row) it.margin.vTotal() else it.margin.hTotal();

        // Cross-axis sizing
        const cross_val = switch (cross_sz) {
            .fixed => |v| std.math.clamp(v, cross_min, cross_max),
            .percent => |p| std.math.clamp(cross_avail * (p * 0.01), cross_min, cross_max),
            .auto => std.math.clamp(cross_auto, cross_min, cross_max),
            .flex => cross_avail - c_extra,
        };
        cross_sizes[i] = cross_val;

        // Main-axis sizing
        switch (main_sz) {
            .fixed => |v| {
                const s = std.math.clamp(v, main_min, main_max);
                main_sizes[i] = s;
                total_fixed_main += s + m_extra;
            },
            .percent => |p| {
                const s = std.math.clamp(main_avail * (p * 0.01), main_min, main_max);
                main_sizes[i] = s;
                total_fixed_main += s + m_extra;
            },
            .auto => {
                const s = std.math.clamp(main_auto, main_min, main_max);
                main_sizes[i] = s;
                total_fixed_main += s + m_extra;
            },
            .flex => |f| {
                total_flex += @max(f, 0.0);
                total_fixed_main += m_extra;
            },
        }
    }

    const total_spacing = @as(f32, @floatFromInt(count - 1)) * @max(spacing, 0.0);
    const leftover_main = @max(main_avail - total_fixed_main - total_spacing, 0.0);

    // Pass 2: distribute leftover to flex items
    if (total_flex > 0.0 and leftover_main > 0.0) {
        for (0..count) |i| {
            const it = items[i];
            const main_sz = if (is_row) it.width else it.height;
            const main_min = if (is_row) it.min_width else it.min_height;
            const main_max = if (is_row) it.max_width else it.max_height;

            if (main_sz == .flex) {
                const weight = @max(main_sz.flex, 0.0);
                const share = leftover_main * (weight / total_flex);
                main_sizes[i] = std.math.clamp(share, main_min, main_max);
            }
        }
    }

    // Pass 3: compute justify offset & gaps if no flex items
    var start_off: f32 = 0.0;
    var actual_gap = @max(spacing, 0.0);

    if (total_flex <= 0.0 and leftover_main > 0.0) {
        switch (justify) {
            .start => {},
            .center => start_off = leftover_main * 0.5,
            .end => start_off = leftover_main,
            .space_between => {
                if (count > 1) {
                    actual_gap += leftover_main / @as(f32, @floatFromInt(count - 1));
                }
            },
            .space_around => {
                const step = leftover_main / @as(f32, @floatFromInt(count));
                start_off = step * 0.5;
                actual_gap += step;
            },
            .space_evenly => {
                const step = leftover_main / @as(f32, @floatFromInt(count + 1));
                start_off = step;
                actual_gap += step;
            },
        }
    }

    // Pass 4: compute final rects
    var cur_main = main_origin + start_off;

    for (0..count) |idx| {
        const i = if (is_rev) count - 1 - idx else idx;
        const it = items[i];
        const m_start = if (is_row) it.margin.left else it.margin.top;
        const m_end = if (is_row) it.margin.right else it.margin.bottom;
        const c_start = if (is_row) it.margin.top else it.margin.left;
        const c_size = cross_sizes[i];
        const m_size = main_sizes[i];

        // Cross-axis alignment
        const align_mode = it.align_self orelse align_items;
        var c_pos = cross_origin + c_start;
        var c_render_size = c_size;

        switch (align_mode) {
            .start => {},
            .center => c_pos += @max((cross_avail - c_size - (if (is_row) it.margin.vTotal() else it.margin.hTotal())) * 0.5, 0.0),
            .end => c_pos += @max(cross_avail - c_size - (if (is_row) it.margin.bottom else it.margin.right), 0.0),
            .stretch => {
                c_render_size = @max(cross_avail - (if (is_row) it.margin.vTotal() else it.margin.hTotal()), 0.0);
            },
        }

        const m_pos = cur_main + m_start;

        if (is_row) {
            out_rects[i] = .{ m_pos, c_pos, m_size, c_render_size };
        } else {
            out_rects[i] = .{ c_pos, m_pos, c_render_size, m_size };
        }

        cur_main += m_start + m_size + m_end + actual_gap;
    }
}

// ============================================================================
// 4. CSS-style Grid Track Solver with FR Units & Spans
// ============================================================================

/// Grid track size definition (pixels, flexible fractions, or percentages).
pub const GridTrack = union(enum) {
    px: f32,
    fr: f32,
    percent: f32,
    auto,

    pub fn pixel(v: f32) GridTrack {
        return .{ .px = @max(v, 0.0) };
    }

    pub fn frUnit(factor: f32) GridTrack {
        return .{ .fr = @max(factor, 0.0) };
    }

    pub fn pct(percent_val: f32) GridTrack {
        return .{ .percent = std.math.clamp(percent_val, 0.0, 100.0) };
    }
};

/// Solves 1D track sizes (columns or rows) taking into account `fr`, `px`, and `percent`.
pub fn solveGridTracks(
    tracks: []const GridTrack,
    total_extent: f32,
    spacing: f32,
    out_sizes: []f32,
    out_offsets: []f32,
) void {
    const n = @min(tracks.len, @min(out_sizes.len, out_offsets.len));
    if (n == 0) return;

    var total_fixed: f32 = 0.0;
    var total_fr: f32 = 0.0;

    for (0..n) |i| {
        switch (tracks[i]) {
            .px => |v| {
                out_sizes[i] = v;
                total_fixed += v;
            },
            .percent => |p| {
                const sz = total_extent * (p * 0.01);
                out_sizes[i] = sz;
                total_fixed += sz;
            },
            .auto => {
                out_sizes[i] = 0.0;
            },
            .fr => |f| {
                total_fr += f;
                out_sizes[i] = 0.0;
            },
        }
    }

    const total_spacing = @as(f32, @floatFromInt(n - 1)) * spacing;
    const leftover = @max(total_extent - total_fixed - total_spacing, 0.0);

    if (total_fr > 0.0 and leftover > 0.0) {
        for (0..n) |i| {
            if (tracks[i] == .fr) {
                out_sizes[i] = leftover * (tracks[i].fr / total_fr);
            }
        }
    }

    var off: f32 = 0.0;
    for (0..n) |i| {
        out_offsets[i] = off;
        off += out_sizes[i] + spacing;
    }
}

/// Advanced Grid layout specification with spans and track resolutions.
pub const AdvancedGridSpec = struct {
    inner: [4]f32,
    col_sizes: []const f32,
    col_offsets: []const f32,
    row_sizes: []const f32,
    row_offsets: []const f32,

    pub fn getCellRect(
        self: AdvancedGridSpec,
        col: usize,
        row: usize,
        col_span_in: usize,
        row_span_in: usize,
    ) [4]f32 {
        if (self.col_sizes.len == 0 or self.row_sizes.len == 0) return self.inner;

        const c = @min(col, self.col_sizes.len - 1);
        const r = @min(row, self.row_sizes.len - 1);
        const c_span = @min(@max(col_span_in, 1), self.col_sizes.len - c);
        const r_span = @min(@max(row_span_in, 1), self.row_sizes.len - r);

        const x = self.inner[0] + self.col_offsets[c];
        const y = self.inner[1] + self.row_offsets[r];

        var w: f32 = 0.0;
        for (0..c_span) |ci| {
            w += self.col_sizes[c + ci];
        }
        if (c_span > 1 and self.col_offsets.len > c + c_span - 1) {
            // Include intermediate track spacings
            const spacing_w = (self.col_offsets[c + c_span - 1] - self.col_offsets[c]) - (w - self.col_sizes[c + c_span - 1]);
            w += @max(spacing_w, 0.0);
        }

        var h: f32 = 0.0;
        for (0..r_span) |ri| {
            h += self.row_sizes[r + ri];
        }
        if (r_span > 1 and self.row_offsets.len > r + r_span - 1) {
            const spacing_h = (self.row_offsets[r + r_span - 1] - self.row_offsets[r]) - (h - self.row_sizes[r + r_span - 1]);
            h += @max(spacing_h, 0.0);
        }

        return .{ x, y, w, h };
    }
};

// ============================================================================
// Layout Unit Tests
// ============================================================================

test "UISize resolves fixed, percent, flex, and auto sizes" {
    const t = std.testing;

    const fixed = UISize.px(64.0);
    try t.expectEqual(@as(f32, 64.0), fixed.resolve(500.0, 10.0));

    const percent = UISize.pct(50.0);
    try t.expectEqual(@as(f32, 250.0), percent.resolve(500.0, 10.0));

    const flex = UISize.grow(1.0);
    try t.expectEqual(@as(f32, 500.0), flex.resolve(500.0, 10.0));

    const auto_sz: UISize = .auto;
    try t.expectEqual(@as(f32, 32.0), auto_sz.resolve(500.0, 32.0));
}

test "UIEdges calculations, insetting and outsetting" {
    const t = std.testing;

    const e_all = UIEdges.all(12.0);
    try t.expectEqual(@as(f32, 24.0), e_all.hTotal());
    try t.expectEqual(@as(f32, 24.0), e_all.vTotal());

    const e_sym = UIEdges.symmetric(16.0, 8.0);
    try t.expectEqual(@as(f32, 32.0), e_sym.hTotal());
    try t.expectEqual(@as(f32, 16.0), e_sym.vTotal());

    const e_trbl = UIEdges.trbl(5.0, 10.0, 15.0, 20.0);
    try t.expectEqual(@as(f32, 30.0), e_trbl.hTotal());
    try t.expectEqual(@as(f32, 20.0), e_trbl.vTotal());

    const rect: [4]f32 = .{ 100.0, 100.0, 200.0, 100.0 };
    const inset_rect = e_trbl.inset(rect);
    try t.expectEqual(@as(f32, 120.0), inset_rect[0]); // 100 + left(20)
    try t.expectEqual(@as(f32, 105.0), inset_rect[1]); // 100 + top(5)
    try t.expectEqual(@as(f32, 170.0), inset_rect[2]); // 200 - 30
    try t.expectEqual(@as(f32, 80.0), inset_rect[3]); // 100 - 20

    const outset_rect = e_trbl.outset(inset_rect);
    try t.expectEqual(@as(f32, 100.0), outset_rect[0]);
    try t.expectEqual(@as(f32, 100.0), outset_rect[1]);
    try t.expectEqual(@as(f32, 200.0), outset_rect[2]);
    try t.expectEqual(@as(f32, 100.0), outset_rect[3]);
}

test "UIAnchor anchorRect positions correctly across 9 points" {
    const t = std.testing;

    const container: [4]f32 = .{ 100.0, 100.0, 400.0, 300.0 };
    const margin = UIEdges.all(10.0); // inner rect: [110, 110, 380, 280]

    // Top-Left
    const tl = anchorRect(container, 80.0, 40.0, .top_left, margin);
    try t.expectEqual(@as(f32, 110.0), tl[0]);
    try t.expectEqual(@as(f32, 110.0), tl[1]);
    try t.expectEqual(@as(f32, 80.0), tl[2]);
    try t.expectEqual(@as(f32, 40.0), tl[3]);

    // Center
    const c = anchorRect(container, 80.0, 40.0, .center, margin);
    try t.expectEqual(@as(f32, 260.0), c[0]); // 110 + (380 - 80) / 2
    try t.expectEqual(@as(f32, 230.0), c[1]); // 110 + (280 - 40) / 2
    try t.expectEqual(@as(f32, 80.0), c[2]);
    try t.expectEqual(@as(f32, 40.0), c[3]);

    // Bottom-Right
    const br = anchorRect(container, 80.0, 40.0, .bottom_right, margin);
    try t.expectEqual(@as(f32, 410.0), br[0]); // 110 + (380 - 80)
    try t.expectEqual(@as(f32, 350.0), br[1]); // 110 + (280 - 40)
    try t.expectEqual(@as(f32, 80.0), br[2]);
    try t.expectEqual(@as(f32, 40.0), br[3]);
}

test "UIDock dockRect carves container sequentially" {
    const t = std.testing;

    var container: [4]f32 = .{ 0.0, 0.0, 500.0, 400.0 };

    // Dock top toolbar of 50px
    const top_bar = dockRect(&container, .top, 50.0, .zero);
    try t.expectEqual(@as(f32, 0.0), top_bar[0]);
    try t.expectEqual(@as(f32, 0.0), top_bar[1]);
    try t.expectEqual(@as(f32, 500.0), top_bar[2]);
    try t.expectEqual(@as(f32, 50.0), top_bar[3]);
    try t.expectEqual(@as(f32, 50.0), container[1]);
    try t.expectEqual(@as(f32, 350.0), container[3]);

    // Dock left sidebar of 120px
    const left_bar = dockRect(&container, .left, 120.0, .zero);
    try t.expectEqual(@as(f32, 0.0), left_bar[0]);
    try t.expectEqual(@as(f32, 50.0), left_bar[1]);
    try t.expectEqual(@as(f32, 120.0), left_bar[2]);
    try t.expectEqual(@as(f32, 350.0), left_bar[3]);
    try t.expectEqual(@as(f32, 120.0), container[0]);
    try t.expectEqual(@as(f32, 380.0), container[2]);

    // Dock bottom statusbar of 30px
    const bot_bar = dockRect(&container, .bottom, 30.0, .zero);
    try t.expectEqual(@as(f32, 120.0), bot_bar[0]);
    try t.expectEqual(@as(f32, 370.0), bot_bar[1]); // 50 + 350 - 30
    try t.expectEqual(@as(f32, 380.0), bot_bar[2]);
    try t.expectEqual(@as(f32, 30.0), bot_bar[3]);
    try t.expectEqual(@as(f32, 320.0), container[3]);

    // Dock fill remaining central content
    const fill_rect = dockRect(&container, .fill, 0.0, .zero);
    try t.expectEqual(@as(f32, 120.0), fill_rect[0]);
    try t.expectEqual(@as(f32, 50.0), fill_rect[1]);
    try t.expectEqual(@as(f32, 380.0), fill_rect[2]);
    try t.expectEqual(@as(f32, 320.0), fill_rect[3]);
    try t.expectEqual(@as(f32, 0.0), container[2]);
    try t.expectEqual(@as(f32, 0.0), container[3]);
}

test "solveFlex distributes space with flex weights and spacing" {
    const t = std.testing;

    const container: [4]f32 = .{ 10.0, 20.0, 320.0, 100.0 };
    const items = [_]LayoutItem{
        .{ .width = .{ .fixed = 50.0 }, .height = .{ .fixed = 40.0 } },
        .{ .width = .{ .flex = 1.0 }, .height = .{ .fixed = 40.0 } },
        .{ .width = .{ .flex = 2.0 }, .height = .{ .fixed = 40.0 } },
    };
    var rects: [3][4]f32 = undefined;

    // Available main = 320
    // Fixed = 50
    // Spacing = 2 * 10 = 20
    // Leftover = 320 - 50 - 20 = 250
    // Total flex = 3.0
    // Item 1 width: 250 * 1/3 = 83.33333
    // Item 2 width: 250 * 2/3 = 166.66667
    solveFlex(container, .row, &items, 10.0, .start, .start, &rects);

    try t.expectEqual(@as(f32, 10.0), rects[0][0]);
    try t.expectEqual(@as(f32, 20.0), rects[0][1]);
    try t.expectEqual(@as(f32, 50.0), rects[0][2]);

    try t.expectEqual(@as(f32, 70.0), rects[1][0]); // 10 + 50 + 10
    try t.expectApproxEqAbs(@as(f32, 83.33333), rects[1][2], 1e-4);

    try t.expectApproxEqAbs(@as(f32, 163.33333), rects[2][0], 1e-4); // 70 + 83.33333 + 10
    try t.expectApproxEqAbs(@as(f32, 166.66667), rects[2][2], 1e-4);
}

test "solveFlex justify_content modes without flex items" {
    const t = std.testing;

    const container: [4]f32 = .{ 0.0, 0.0, 200.0, 50.0 };
    const items = [_]LayoutItem{
        .{ .width = .{ .fixed = 40.0 }, .height = .{ .fixed = 30.0 } },
        .{ .width = .{ .fixed = 40.0 }, .height = .{ .fixed = 30.0 } },
    };
    var rects: [2][4]f32 = undefined;

    // Total fixed = 80. Leftover = 120.
    // Center: start_off = 60
    solveFlex(container, .row, &items, 0.0, .center, .start, &rects);
    try t.expectEqual(@as(f32, 60.0), rects[0][0]);
    try t.expectEqual(@as(f32, 100.0), rects[1][0]);

    // End: start_off = 120
    solveFlex(container, .row, &items, 0.0, .end, .start, &rects);
    try t.expectEqual(@as(f32, 120.0), rects[0][0]);
    try t.expectEqual(@as(f32, 160.0), rects[1][0]);

    // Space between: gap = 120
    solveFlex(container, .row, &items, 0.0, .space_between, .start, &rects);
    try t.expectEqual(@as(f32, 0.0), rects[0][0]);
    try t.expectEqual(@as(f32, 160.0), rects[1][0]);

    // Space evenly: step = 120 / 3 = 40
    solveFlex(container, .row, &items, 0.0, .space_evenly, .start, &rects);
    try t.expectEqual(@as(f32, 40.0), rects[0][0]);
    try t.expectEqual(@as(f32, 120.0), rects[1][0]);
}

test "solveFlex column direction and cross-axis alignment" {
    const t = std.testing;

    const container: [4]f32 = .{ 0.0, 0.0, 200.0, 300.0 };
    const items = [_]LayoutItem{
        .{ .width = .{ .fixed = 80.0 }, .height = .{ .fixed = 50.0 }, .align_self = .center },
        .{ .width = .{ .fixed = 80.0 }, .height = .{ .fixed = 50.0 }, .align_self = .stretch },
    };
    var rects: [2][4]f32 = undefined;

    solveFlex(container, .column, &items, 10.0, .start, .start, &rects);

    // Item 0: center on cross axis (x = (200 - 80) / 2 = 60)
    try t.expectEqual(@as(f32, 60.0), rects[0][0]);
    try t.expectEqual(@as(f32, 0.0), rects[0][1]);
    try t.expectEqual(@as(f32, 80.0), rects[0][2]);
    try t.expectEqual(@as(f32, 50.0), rects[0][3]);

    // Item 1: stretch on cross axis (x = 0, width = 200)
    try t.expectEqual(@as(f32, 0.0), rects[1][0]);
    try t.expectEqual(@as(f32, 60.0), rects[1][1]); // 0 + 50 + 10
    try t.expectEqual(@as(f32, 200.0), rects[1][2]);
    try t.expectEqual(@as(f32, 50.0), rects[1][3]);
}

test "solveGridTracks with mix of px, percent, and fr" {
    const t = std.testing;

    const tracks = [_]GridTrack{
        GridTrack.pixel(50.0),
        GridTrack.pct(25.0), // 25% of 400 = 100
        GridTrack.frUnit(1.0), // leftover: 400 - 50 - 100 - (2 * 10) = 230
    };
    var sizes: [3]f32 = undefined;
    var offsets: [3]f32 = undefined;

    solveGridTracks(&tracks, 400.0, 10.0, &sizes, &offsets);

    try t.expectEqual(@as(f32, 50.0), sizes[0]);
    try t.expectEqual(@as(f32, 0.0), offsets[0]);

    try t.expectEqual(@as(f32, 100.0), sizes[1]);
    try t.expectEqual(@as(f32, 60.0), offsets[1]); // 50 + 10

    try t.expectEqual(@as(f32, 230.0), sizes[2]);
    try t.expectEqual(@as(f32, 170.0), offsets[2]); // 60 + 100 + 10
}

test "AdvancedGridSpec getCellRect multi-cell spanning" {
    const t = std.testing;

    const col_sizes = [_]f32{ 100.0, 100.0, 100.0 };
    const col_offsets = [_]f32{ 0.0, 110.0, 220.0 }; // 10px spacing
    const row_sizes = [_]f32{ 50.0, 50.0 };
    const row_offsets = [_]f32{ 0.0, 60.0 }; // 10px spacing

    const spec = AdvancedGridSpec{
        .inner = .{ 10.0, 20.0, 320.0, 110.0 },
        .col_sizes = &col_sizes,
        .col_offsets = &col_offsets,
        .row_sizes = &row_sizes,
        .row_offsets = &row_offsets,
    };

    // Span col 0..1 (2 cols) and row 0..1 (2 rows)
    const spanned = spec.getCellRect(0, 0, 2, 2);
    try t.expectEqual(@as(f32, 10.0), spanned[0]);
    try t.expectEqual(@as(f32, 20.0), spanned[1]);
    try t.expectEqual(@as(f32, 210.0), spanned[2]); // 100 + 10 + 100
    try t.expectEqual(@as(f32, 110.0), spanned[3]); // 50 + 10 + 50
}
