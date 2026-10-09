const std = @import("std");
const math = @import("math");
const Color4 = math.Color4;
const easing = @import("../animation/easing.zig");
const css = @import("css_parser.zig");
const parseCss = css.parseCss;
const loadThemeFile = css.loadThemeFile;
const CssDiagKind = css.CssDiagKind;
const parseHexColor = css.parseHexColor;
const parseRgbColor = css.parseRgbColor;
const parseColor = css.parseColor;
const parseLength = css.parseLength;
const parseDurationMs = css.parseDurationMs;
const parseEasing = css.parseEasing;

const testing = std.testing;

const fixture_css =
    \\/* engine theme */
    \\theme { accent: #7cb3ff; transition_duration: 120ms; }
    \\button { background: #141925d9; corner_radius: 4px; }
    \\button:hover { background: rgb(36, 48, 66); border_color: #a6d4ff; }
    \\.danger { background: #b3261e; transition_easing: ease-in-cubic; }
    \\.danger:disabled { opacity: 0.4; }
;

fn expectColorEql(a: Color4, b: Color4) !void {
    try testing.expectApproxEqAbs(a.r, b.r, 1e-4);
    try testing.expectApproxEqAbs(a.g, b.g, 1e-4);
    try testing.expectApproxEqAbs(a.b, b.b, 1e-4);
    try testing.expectApproxEqAbs(a.a, b.a, 1e-4);
}

test "parseCss: valid stylesheet yields theme slots, classes and state deltas" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const parsed = try parseCss(arena.allocator(), fixture_css);

    try testing.expectEqual(@as(usize, 0), parsed.diags.len);
    try testing.expect(parsed.accent_set);
    try expectColorEql(Color4.new(@as(f32, 0x7c) / 255.0, @as(f32, 0xb3) / 255.0, @as(f32, 0xff) / 255.0, 1.0), parsed.theme.accent);

    try testing.expectApproxEqAbs(@as(f32, 120.0), parsed.theme.button.transition.duration_ms, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 120.0), parsed.theme.panel.transition.duration_ms, 1e-5);

    const btn = parsed.theme.button;
    try expectColorEql(Color4.new(@as(f32, 0x14) / 255.0, @as(f32, 0x19) / 255.0, @as(f32, 0x25) / 255.0, @as(f32, 0xd9) / 255.0), btn.normal.background.?);
    try testing.expectApproxEqAbs(@as(f32, 4.0), btn.normal.corner_radius.?, 1e-5);

    try expectColorEql(Color4.new(36.0 / 255.0, 48.0 / 255.0, 66.0 / 255.0, 1.0), btn.hover.?.background.?);
    try expectColorEql(Color4.new(@as(f32, 0xa6) / 255.0, @as(f32, 0xd4) / 255.0, @as(f32, 0xff) / 255.0, 1.0), btn.hover.?.border_color.?);
    try testing.expectApproxEqAbs(@as(f32, 0.18), btn.active.?.background.?.r, 1e-4);

    try testing.expectEqual(@as(usize, 1), parsed.classes.len);
    try testing.expectEqualStrings("danger", parsed.classes[0].name);
    const danger = parsed.classes[0].set;
    try expectColorEql(Color4.new(@as(f32, 0xb3) / 255.0, @as(f32, 0x26) / 255.0, @as(f32, 0x1e) / 255.0, 1.0), danger.normal.background.?);
    try testing.expectApproxEqAbs(@as(f32, 0.4), danger.disabled.?.opacity.?, 1e-5);
    try testing.expectEqual(easing.EasingType.ease_in_cubic, danger.transition.easing);

    const resolved = danger.resolve(.{}, .disabled);
    try testing.expectApproxEqAbs(@as(f32, 0.4), resolved.opacity, 1e-5);
    try expectColorEql(Color4.new(@as(f32, 0xb3) / 255.0, @as(f32, 0x26) / 255.0, @as(f32, 0x1e) / 255.0, 1.0), resolved.background);
}

test "parseCss: syntax error reports a diagnostic with line/column" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const src = "button { background: #101828; }\npanel { corner_radius 8px; }";
    const parsed = try parseCss(arena.allocator(), src);
    try testing.expect(parsed.diags.len >= 1);
    const d = parsed.diags[0];
    try testing.expectEqual(CssDiagKind.syntax, d.kind);
    try testing.expectEqual(@as(usize, 2), d.line);
    try testing.expect(d.col >= 8 and d.col <= 30);

    try testing.expect(parsed.theme.button.normal.background != null);
}

test "parseCss: unknown selector and unknown property warn but do not abort" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const src =
        \\frobnicator { background: #fff; }
        \\panel { background: #101828; glow_radius: 7px; padding: 4px; }
    ;
    const parsed = try parseCss(arena.allocator(), src);
    try testing.expectEqual(@as(usize, 2), parsed.diags.len);
    try testing.expectEqual(CssDiagKind.unknown_selector, parsed.diags[0].kind);
    try testing.expectEqualStrings("frobnicator", parsed.diags[0].message);
    try testing.expectEqual(CssDiagKind.unknown_property, parsed.diags[1].kind);
    try testing.expectEqualStrings("glow_radius", parsed.diags[1].message);

    try testing.expectApproxEqAbs(@as(f32, 4.0), parsed.theme.panel.normal.padding.?, 1e-5);
    try testing.expect(parsed.theme.panel.normal.background != null);
    try testing.expect(parsed.theme.button.normal.background.?.r < 0.5);
}

test "value parsers: hex shorthand, rgb/rgba, lengths, durations, easing" {
    try expectColorEql(Color4.new(@as(f32, 0xaa) / 255.0, @as(f32, 0xbb) / 255.0, @as(f32, 0xcc) / 255.0, 1.0), parseHexColor("#abc").?);
    try expectColorEql(Color4.new(@as(f32, 0xaa) / 255.0, @as(f32, 0xbb) / 255.0, @as(f32, 0xcc) / 255.0, @as(f32, 0xdd) / 255.0), parseHexColor("#abcd").?);
    try expectColorEql(Color4.new(@as(f32, 0x10) / 255.0, @as(f32, 0x18) / 255.0, @as(f32, 0x28) / 255.0, 1.0), parseHexColor("#101828").?);
    try testing.expect(parseHexColor("#12345") == null);
    try testing.expect(parseHexColor("101828") == null);

    try expectColorEql(Color4.new(1.0 / 255.0, 2.0 / 255.0, 3.0 / 255.0, 1.0), parseRgbColor("rgb(1,2,3)").?);
    try expectColorEql(Color4.new(0, 0, 0, 0.5), parseRgbColor("rgba(0, 0, 0, 0.5)").?);
    try expectColorEql(Color4.new(1.0, 0.0, 1.0, 1.0), parseRgbColor("rgb(300,-5,255)").?);

    try expectColorEql(Color4.new(1, 0, 0, 1), parseColor("#f00").?);
    try expectColorEql(Color4.new(0, 1, 0, 1), parseColor("rgb(0,255,0)").?);
    try testing.expect(parseColor("null") == null);

    try testing.expectApproxEqAbs(@as(f32, 8.0), parseLength("8px").?, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.5), parseLength("1.5").?, 1e-6);
    try testing.expect(parseLength("px") == null);

    try testing.expectApproxEqAbs(@as(f32, 150.0), parseDurationMs("150ms").?, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 2000.0), parseDurationMs("2s").?, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 90.0), parseDurationMs("90").?, 1e-6);

    try testing.expectEqual(easing.EasingType.ease_out_quad, parseEasing("ease-out-quad").?);
    try testing.expectEqual(easing.EasingType.linear, parseEasing("linear").?);
    try testing.expect(parseEasing("no-such-easing") == null);
}

test "loadThemeFile: reads, parses and copies a theme file" {
    const io = std.Io.Threaded.global_single_threaded.io();
    const path = "agate_ui_css_parser_test.tmp.css";
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = fixture_css });
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const parsed = try loadThemeFile(arena.allocator(), path);
    try testing.expectEqual(@as(usize, 0), parsed.diags.len);
    try testing.expectEqual(@as(usize, 1), parsed.classes.len);
    try testing.expectEqualStrings("danger", parsed.classes[0].name);
    try testing.expect(parsed.theme.button.hover != null);
    try testing.expect(parsed.accent_set);
}

test "loadThemeFile: missing file is a file error, not a parse error" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const result = loadThemeFile(arena.allocator(), "agate_ui_css_parser_no_such_file.css");
    try testing.expectError(error.FileNotFound, result);
}
