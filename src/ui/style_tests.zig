const std = @import("std");
const math = @import("math");
const Color4 = math.Color4;
const UICanvas = @import("canvas.zig").UICanvas;
const ui_types = @import("types.zig");
const UIStyle = ui_types.UIStyle;
const UIStyledOptions = ui_types.UIStyledOptions;
const UIState = ui_types.UIState;
const css_parser = @import("css_parser.zig");
const ui_transition = @import("transition.zig");

fn freeTestCanvas(canvas: anytype) void {
    canvas.vertices.deinit(canvas.allocator);
    canvas.indices.deinit(canvas.allocator);
}

fn quadCount(canvas: anytype) usize {
    return canvas.vertices.items.len / 4;
}

fn testColorEql(a: Color4, b: Color4) bool {
    return a.r == b.r and a.g == b.g and a.b == b.b and a.a == b.a;
}

fn usedTransitionSlots(canvas: anytype) usize {
    var n: usize = 0;
    for (&canvas.style_transitions) |*s| {
        if (s.used) n += 1;
    }
    return n;
}

test "style cascade: override beats class beats theme default" {
    const t = std.testing;
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
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
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
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

    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);
    const s = canvas.resolveStyle(.{ .kind = .button, .state = .disabled });
    try t.expectApproxEqAbs(@as(f32, 0.45), s.opacity, 1e-5);
    // Inherit: non-disabled states keep full opacity.
    const n = canvas.resolveStyle(.{ .kind = .button });
    try t.expectApproxEqAbs(@as(f32, 1.0), n.opacity, 1e-5);
}

test "corner radius, gradient, shadow and opacity change emitted geometry" {
    const t = std.testing;
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
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
    const PinnedStack = @import("stack.zig").LayoutStack(UICanvas);
    const t = std.testing;
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);
    canvas.setStyleClass("padded", .{ .normal = .{
        .padding = 10.0,
        .background = Color4.new(0.2, 0.2, 0.3, 0.8),
    } });

    var ls = PinnedStack.init(&canvas);
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

test "styled widgets emit geometry and resolve their state styles" {
    const t = std.testing;
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
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

test "applyCssTheme flows parsed theme and classes through the cascade" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);

    const parsed = try css_parser.parseCss(arena.allocator(),
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
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
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
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
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
    try t.expectApproxEqAbs(ui_transition.lerpStyle(UIStyle{ .background = normal_bg }, UIStyle{ .background = hover_bg }, 1.0 / 3.0).background.r, mid.background.r, 1e-5);

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
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
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
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer freeTestCanvas(&canvas);
    canvas.frame_dt_ms = 100;
    const bg = Color4.new(0.2, 0.4, 0.6, 1.0);

    canvas.begin(); // t=100: fill every slot
    var i: usize = 0;
    while (i < ui_transition.max_style_transitions) : (i += 1) {
        var name_buf: [16]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "w{d}", .{i}) catch unreachable;
        const s = canvas.resolveAnimatedStyle(.badge, .{
            .anim_key = name,
            .transition = .{ .duration_ms = 500, .easing = .linear },
            .style = .{ .background = bg },
        });
        try t.expect(testColorEql(bg, s.background));
    }
    try t.expectEqual(@as(usize, ui_transition.max_style_transitions), usedTransitionSlots(&canvas));

    canvas.begin(); // t=200: one more widget evicts the stalest entry, all still resolve
    const extra = canvas.resolveAnimatedStyle(.badge, .{
        .anim_key = "extra",
        .transition = .{ .duration_ms = 500, .easing = .linear },
        .style = .{ .background = bg },
    });
    try t.expect(testColorEql(bg, extra.background));
    try t.expectEqual(@as(usize, ui_transition.max_style_transitions), usedTransitionSlots(&canvas));

    // Zero-duration config snaps and releases its slot.
    const snap = canvas.resolveAnimatedStyle(.badge, .{
        .anim_key = "extra",
        .transition = .{},
        .style = .{ .background = bg },
    });
    try t.expect(testColorEql(bg, snap.background));
    try t.expectEqual(@as(usize, ui_transition.max_style_transitions - 1), usedTransitionSlots(&canvas));
}

test "animated styled widgets draw their interpolated style" {
    const t = std.testing;
    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
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
