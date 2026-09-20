//! UI font plumbing: the embedded SDF bitmap font upload plus the optional
//! TrueType override (install, GPU atlas binding, canvas-aware measurement).
//!
//! Split out of `ui.zig` (facade). Functions take a generic `canvas: anytype`
//! (a `*UICanvas` in practice) so this module never imports `../ui.zig` —
//! following the `scene/` precedent, subsystems never import the facade;
//! the canvas passes itself. `ui.zig` owns the `UICanvas` type and provides
//! thin forwarders (`canvas.setFontTtf(...)` keeps working exactly as before).

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const math = @import("math");
const Color4 = math.Color4;
const Vec2 = math.Vec2;

const Texture = @import("../texture.zig").Texture;
const ttf_mod = @import("../ttf.zig");
const ui_text = @import("text.zig");

/// TrueType font handle for the UI text path (see ../ttf.zig).
pub const TtfFont = ttf_mod.TtfFont;

const font_png_data = @embedFile("../assets/font_sdf.png");

/// Uploads the embedded SDF bitmap font to a GPU texture (context thread
/// only, like every other Texture upload). Shared by canvas `init` and by
/// out-of-canvas owners that need the same atlas without a full canvas
/// (the 3D-GUI layer's shared font, see scene/gui3d_layer.zig).
pub fn makeFontTexture(allocator: std.mem.Allocator) !Texture {
    return Texture.fromMemory(allocator, font_png_data, .{
        .min_filter = .LINEAR,
        .mag_filter = .LINEAR,
        .wrap_u = .CLAMP_TO_EDGE,
        .wrap_v = .CLAMP_TO_EDGE,
        // Box-filtered mips blur the distance field beyond legibility.
        .mipmaps = false,
    });
}

/// Switches text drawing to a TrueType font (`null` restores the
/// bitmap/SDF default). The font is borrowed, not copied: the caller
/// owns the `TtfFont` (and its sfnt bytes) until clearFontTtf/deinit.
/// Parse failures surface at `TtfFont.init`, never here — an installed
/// font always draws. With a live sokol context the atlas uploads to
/// a GPU texture immediately (context thread only, like every other
/// Texture upload); headless, only the CPU-side switch happens and the
/// upload is skipped (vertices still emit TTF UVs, so tests can
/// assert geometry without a GPU).
pub fn setFontTtf(canvas: anytype, font: ?*const TtfFont) void {
    if (canvas.ttf_texture) |*t| {
        t.deinit();
        canvas.ttf_texture = null;
    }
    canvas.ttf_font = font;
    if (font) |f| {
        if (sg.isvalid()) {
            canvas.ttf_texture = Texture.initRaw(ttf_mod.atlas_size, ttf_mod.atlas_size, f.atlas_pixels, .{
                .min_filter = .LINEAR,
                .mag_filter = .LINEAR,
                .wrap_u = .CLAMP_TO_EDGE,
                .wrap_v = .CLAMP_TO_EDGE,
                .mipmaps = false,
            });
        }
    }
}

/// Restores the bitmap/SDF font (see setFontTtf).
pub fn clearFontTtf(canvas: anytype) void {
    setFontTtf(canvas, null);
}

/// True when a TrueType font is installed.
pub fn hasTtfFont(canvas: anytype) bool {
    return canvas.ttf_font != null;
}

/// Font view the draw binds: the TTF atlas upload when a font is set
/// and uploaded, else the bitmap/SDF atlas. Shared by the legacy
/// render and the P6 frame capture so both bind the same font.
pub fn activeFontView(canvas: anytype) sg.View {
    if (canvas.ttf_font != null) {
        if (canvas.ttf_texture) |*t| {
            if (t.view.id != 0) return t.view;
        }
    }
    return canvas.font_texture.view;
}

/// Sampler matching `activeFontView`.
pub fn activeFontSampler(canvas: anytype) sg.Sampler {
    if (canvas.ttf_font != null) {
        if (canvas.ttf_texture) |*t| {
            if (t.sampler.id != 0) return t.sampler;
        }
    }
    return canvas.font_texture.sampler;
}

/// Canvas-aware measurement: TTF advances when a font is set (so UI
/// layout matches the drawn TTF glyphs), the legacy monospace math
/// otherwise. Static `measureText` keeps the legacy contract for
/// callers that never install a font.
pub fn measureTextCurrent(canvas: anytype, text: []const u8, font_size: f32) Vec2 {
    if (canvas.ttf_font) |f| return ui_text.measureTextTtf(f, text, font_size);
    return ui_text.measureText(text, font_size);
}

test "setFontTtf switches text drawing and measurement, clear restores legacy" {
    const UICanvas = @import("canvas.zig").UICanvas;
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
