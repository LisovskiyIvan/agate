const std = @import("std");
const math = @import("math");
const Color4 = math.Color4;
const Vec2 = math.Vec2;

const ttf = @import("../ttf.zig");
const ui_draw = @import("draw.zig");
const text_mod = @import("text.zig");

const ttf_text_mode = text_mod.ttf_text_mode;
const GlyphUV = text_mod.GlyphUV;
const getGlyphUV = text_mod.getGlyphUV;
const drawText = text_mod.drawText;
const measureText = text_mod.measureText;
const drawTextTtf = text_mod.drawTextTtf;
const measureTextTtf = text_mod.measureTextTtf;
const measureForCanvas = text_mod.measureForCanvas;

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
    const size = measureText("Hello World", 20.0);
    try std.testing.expectApproxEqAbs(@as(f32, 110.0), size.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), size.y, 1e-5);
}

test "UICanvas measureText consistency" {
    // Monospace advance: measureText matches the per-char accumulation
    // drawText (and the drawTextInput cursor) uses.
    const a = measureText("a", 16.0);
    const ab = measureText("ab", 16.0);
    try std.testing.expectApproxEqAbs(a.x * 2.0, ab.x, 1e-5);
    try std.testing.expectApproxEqAbs(a.y, ab.y, 1e-5);
    const empty = measureText("", 16.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), empty.x, 1e-5);

    // UTF-8 multibyte characters measure per codepoint, not per byte:
    const utf8_two_chars = measureText("Привет", 16.0); // 6 Cyrillic characters (12 bytes)
    try std.testing.expectApproxEqAbs(a.x * 6.0, utf8_two_chars.x, 1e-5);

    // Dropdown list total height = count * shared item height.
    const input_state = @import("input_state.zig");
    const ih = input_state.dropdownItemHeight(16.0);
    try std.testing.expectApproxEqAbs(@as(f32, 24.0), ih, 1e-5);
    const btn: [4]f32 = .{ 0, 0, 100, 20 };
    const last = input_state.dropdownItemRect(btn, ih, 3);
    try std.testing.expectApproxEqAbs(@as(f32, 20 + 3 * 24), last[1], 1e-5);
}

// ---------------------------------------------------------------------------
// TTF UI-path tests: a headless fake canvas (same vertex/quad shape as
// UICanvas) plus a programmatic fixture font (no font files).
// ---------------------------------------------------------------------------

const FakeCanvas = struct {
    allocator: std.mem.Allocator = undefined,
    vertices: std.ArrayListUnmanaged(ui_draw.UIVertex) = .empty,
    indices: std.ArrayListUnmanaged(u16) = .empty,
    capacity_vertices: usize = 32768,
    capacity_indices: usize = 49152,
    ttf_font: ?*const ttf.TtfFont = null,

    fn init(alloc: std.mem.Allocator) FakeCanvas {
        return .{ .allocator = alloc };
    }

    fn deinit(self: *FakeCanvas) void {
        self.vertices.deinit(self.allocator);
        self.indices.deinit(self.allocator);
    }

    pub fn addQuad(
        self: *FakeCanvas,
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
};

fn ttfTestFont(alloc: std.mem.Allocator, file: []const u8) !ttf.TtfFont {
    return ttf.TtfFont.init(alloc, file, 20.0, &.{ 'A', 'B', 'C' });
}

test "TTF UI path emits coverage quads with atlas UVs" {
    const alloc = std.testing.allocator;
    const file = try ttf.buildFixture(alloc, .{});
    defer alloc.free(file);
    var font = try ttfTestFont(alloc, file);
    defer font.deinit();

    var canvas = FakeCanvas.init(alloc);
    defer canvas.deinit();
    canvas.ttf_font = &font;
    drawText(&canvas, "ABC", 10.0, 20.0, 20.0, Color4.white);

    // One quad per baked glyph (all three have bitmaps).
    try std.testing.expectEqual(@as(usize, 12), canvas.vertices.items.len);
    try std.testing.expectEqual(@as(usize, 18), canvas.indices.items.len);
    for (canvas.vertices.items) |v| {
        try std.testing.expectApproxEqAbs(ttf_text_mode, v.mode_params[0], 1e-6);
    }
    // First quad UVs are glyph A's atlas cell exactly.
    const ga = font.lookup('A').?;
    const uv = font.glyphUv(ga);
    try std.testing.expectApproxEqAbs(uv[0], canvas.vertices.items[0].uv[0], 1e-6);
    try std.testing.expectApproxEqAbs(uv[1], canvas.vertices.items[0].uv[1], 1e-6);
    try std.testing.expectApproxEqAbs(uv[2], canvas.vertices.items[2].uv[0], 1e-6);
    // Quad position honors bearing: x = pen + bearing_x (scale 1 here).
    try std.testing.expectApproxEqAbs(@as(f32, 10.0) + ga.bearing_x, canvas.vertices.items[0].position[0], 1e-4);
    // Second glyph starts after A's advance + the A-B kern (-1.6px).
    const gb = font.lookup('B').?;
    const want_bx = 10.0 + ga.advance + font.kernPx(ga, gb) + gb.bearing_x;
    try std.testing.expectApproxEqAbs(want_bx, canvas.vertices.items[4].position[0], 1e-4);
}

test "TTF measureText uses TTF metrics and differs from the bitmap font" {
    const alloc = std.testing.allocator;
    const file = try ttf.buildFixture(alloc, .{});
    defer alloc.free(file);
    var font = try ttfTestFont(alloc, file);
    defer font.deinit();

    const m = measureTextTtf(&font, "ABC", 20.0);
    // Advances 14 + (13 - 1.6 kern) + 15 at scale 1.
    try std.testing.expectApproxEqAbs(@as(f32, 14.0 + 11.4 + 15.0), m.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), m.y, 1e-4);
    // The bitmap font would report 3 * 10 = 30px: the TTF width differs.
    const legacy = measureText("ABC", 20.0);
    try std.testing.expectApproxEqAbs(@as(f32, 30.0), legacy.x, 1e-5);
    try std.testing.expect(m.x != legacy.x);

    // Newlines: height scales by whole line heights, width is the max line.
    const ml = measureTextTtf(&font, "AB\nC", 20.0);
    try std.testing.expectApproxEqAbs(@as(f32, 44.0), ml.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 14.0 + 11.4), ml.x, 1e-4);

    // measureForCanvas dispatches on the canvas font.
    var canvas = FakeCanvas.init(alloc);
    defer canvas.deinit();
    canvas.ttf_font = &font;
    const mc = measureForCanvas(&canvas, "ABC", 20.0);
    try std.testing.expectApproxEqAbs(m.x, mc.x, 1e-5);
    canvas.ttf_font = null;
    const ml2 = measureForCanvas(&canvas, "ABC", 20.0);
    try std.testing.expectApproxEqAbs(legacy.x, ml2.x, 1e-5);
}

test "TTF draw output is stable across two draws" {
    const alloc = std.testing.allocator;
    const file = try ttf.buildFixture(alloc, .{});
    defer alloc.free(file);
    var font = try ttfTestFont(alloc, file);
    defer font.deinit();

    var a = FakeCanvas.init(alloc);
    defer a.deinit();
    var b = FakeCanvas.init(alloc);
    defer b.deinit();
    a.ttf_font = &font;
    b.ttf_font = &font;
    drawText(&a, "ABC", 10.0, 20.0, 20.0, Color4.white);
    drawText(&b, "ABC", 10.0, 20.0, 20.0, Color4.white);
    try std.testing.expectEqual(a.vertices.items.len, b.vertices.items.len);
    try std.testing.expectEqualSlices(u16, a.indices.items, b.indices.items);
    for (a.vertices.items, b.vertices.items) |va, vb| {
        try std.testing.expectApproxEqAbs(va.position[0], vb.position[0], 0.0);
        try std.testing.expectApproxEqAbs(va.uv[0], vb.uv[0], 0.0);
    }
}

test "no-TTF path stays bit-identical (legacy SDF quads)" {
    const alloc = std.testing.allocator;
    var canvas = FakeCanvas.init(alloc);
    defer canvas.deinit();
    drawText(&canvas, "AB", 0.0, 0.0, 20.0, Color4.white);
    // Legacy: char_w = 10, char_h = 20, two quads, mode 1.
    try std.testing.expectEqual(@as(usize, 8), canvas.vertices.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), canvas.vertices.items[0].position[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), canvas.vertices.items[4].position[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), canvas.vertices.items[0].mode_params[0], 1e-6);
}
