const std = @import("std");
const UICanvas = @import("canvas.zig").UICanvas;
const stack_mod = @import("stack.zig");
const LayoutStack = stack_mod.LayoutStack;
const LayoutGridSpec = stack_mod.LayoutGridSpec;
const layoutAlignOffset = stack_mod.layoutAlignOffset;
const gridExtentSize = stack_mod.gridExtentSize;
const gridExtentOffset = stack_mod.gridExtentOffset;
const ui_layout = @import("layout.zig");
const UIEdges = ui_layout.UIEdges;
const UISize = ui_layout.UISize;

/// Frees the CPU-side quad buffers of a headless test canvas (no GPU state:
/// the full `deinit` would touch the undefined sokol handles).
fn freeTestCanvas(canvas: anytype) void {
    canvas.vertices.deinit(canvas.allocator);
    canvas.indices.deinit(canvas.allocator);
}

fn quadCount(canvas: anytype) usize {
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
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);
    var ls = LayoutStack(UICanvas).init(&canvas);
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
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);
    var ls = LayoutStack(UICanvas).init(&canvas);
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
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);
    var ls = LayoutStack(UICanvas).init(&canvas);
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
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);
    var ls = LayoutStack(UICanvas).init(&canvas);
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
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);
    var ls = LayoutStack(UICanvas).init(&canvas);
    var opened: usize = 0;
    for (0..LayoutStack(UICanvas).max_depth + 4) |_| {
        if (ls.beginHStack(.{ 0, 0, 500, 500 }, .{})) opened += 1;
    }
    try t.expectEqual(LayoutStack(UICanvas).max_depth, opened);
    // Placements past the cap operate on the top frame instead of crashing.
    _ = ls.place(10, 10);
    ls.reset();
    try t.expectEqual(@as(usize, 0), ls.depth);
    ls.end(); // no-op at depth 0
}

test "placeBox applies resolved margin around the box" {
    const t = std.testing;
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);
    canvas.setStyleClass("gappy", .{ .normal = .{ .margin = 5.0 } });

    var ls = LayoutStack(UICanvas).init(&canvas);
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

test "LayoutStack with UIEdges padding and spacer in HStack and VStack" {
    const t = std.testing;
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);

    var stack = LayoutStack(UICanvas).init(&canvas);
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
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);

    var stack = LayoutStack(UICanvas).init(&canvas);
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
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);

    var stack = LayoutStack(UICanvas).init(&canvas);
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
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);

    var stack = LayoutStack(UICanvas).init(&canvas);
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
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);

    // Simulate mouse clicked at (50, 60)
    canvas.setInput(50.0, 60.0, true, true);

    var stack = LayoutStack(UICanvas).init(&canvas);
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
