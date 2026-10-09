const std = @import("std");
const math = @import("math");
const Color4 = math.Color4;
const UICanvas = @import("canvas.zig").UICanvas;
const ttf_mod = @import("../ttf.zig");
const TtfFont = ttf_mod.TtfFont;
const ui_text = @import("text.zig");

test "setFontTtf switches text drawing and measurement, clear restores legacy" {
    const t = std.testing;
    const file = try ttf_mod.buildFixture(t.allocator, .{});
    defer t.allocator.free(file);
    var font = try TtfFont.init(t.allocator, file, 20.0, &.{ 'A', 'B', 'C' });
    defer font.deinit();

    var canvas: UICanvas = .{ .allocator = t.allocator, .font_texture = undefined };
    defer canvas.vertices.deinit(canvas.allocator);
    defer canvas.indices.deinit(canvas.allocator);
    try t.expect(!canvas.hasTtfFont());

    // Legacy baseline: "AB" at 20px -> 2 monospace quads, mode 1.
    canvas.drawText("AB", 0, 0, 20.0, Color4.white);
    try t.expectEqual(@as(usize, 8), canvas.vertices.items.len);
    try t.expectApproxEqAbs(@as(f32, 1.0), canvas.vertices.items[0].mode_params[0], 1e-6);
    const legacy_w = canvas.measureTextCurrent("AB", 20.0).x;
    try t.expectApproxEqAbs(@as(f32, 20.0), legacy_w, 1e-5);
    canvas.begin();

    // Install the TTF font (headless: no GPU upload, CPU switch only).
    canvas.setFontTtf(&font);
    try t.expect(canvas.hasTtfFont());
    try t.expect(canvas.ttf_texture == null);
    canvas.drawText("AB", 0, 0, 20.0, Color4.white);
    try t.expectEqual(@as(usize, 8), canvas.vertices.items.len);
    try t.expectApproxEqAbs(ui_text.ttf_text_mode, canvas.vertices.items[0].mode_params[0], 1e-6);
    // TTF advances (14 + 13 - 1.6 kern) differ from the 20px legacy width.
    const ttf_w = canvas.measureTextCurrent("AB", 20.0).x;
    try t.expectApproxEqAbs(@as(f32, 14.0 + 11.4), ttf_w, 1e-4);
    try t.expect(ttf_w != legacy_w);
    // Second draw is stable (same quad count, same first vertex).
    const first = canvas.vertices.items[0];
    canvas.begin();
    canvas.drawText("AB", 0, 0, 20.0, Color4.white);
    try t.expectEqual(@as(usize, 8), canvas.vertices.items.len);
    try t.expectApproxEqAbs(first.position[0], canvas.vertices.items[0].position[0], 0.0);
    try t.expectApproxEqAbs(first.uv[0], canvas.vertices.items[0].uv[0], 0.0);

    // Clear: legacy path returns bit-identical (mode 1, monospace advance).
    canvas.clearFontTtf();
    try t.expect(!canvas.hasTtfFont());
    canvas.begin();
    canvas.drawText("AB", 0, 0, 20.0, Color4.white);
    try t.expectApproxEqAbs(@as(f32, 1.0), canvas.vertices.items[0].mode_params[0], 1e-6);
    try t.expectApproxEqAbs(@as(f32, 10.0), canvas.vertices.items[4].position[0], 1e-6);
}
