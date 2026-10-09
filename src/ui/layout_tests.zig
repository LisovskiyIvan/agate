const std = @import("std");
const layout = @import("layout.zig");

const UISize = layout.UISize;
const UIEdges = layout.UIEdges;
const anchorRect = layout.anchorRect;
const dockRect = layout.dockRect;
const LayoutItem = layout.LayoutItem;
const solveFlex = layout.solveFlex;
const GridTrack = layout.GridTrack;
const solveGridTracks = layout.solveGridTracks;
const AdvancedGridSpec = layout.AdvancedGridSpec;

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
