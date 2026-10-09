const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const glow_mod = @import("glow_pass.zig");
const GlowPass = glow_mod.GlowPass;
const pp = @import("../postprocess.zig");
const upload_meter = @import("../gpu_upload_meter.zig");

test "glow pass fail-closes headless with no state touched" {
    // Zero-initialized pass (never init'ed: no sg context headless, same as
    // the BloomPass fail-closed shape) must return an empty view before any
    // sg.* call — disabled glow touches nothing.
    var pass: GlowPass = .{};
    _ = upload_meter.takeAndReset();
    const empty = pass.render(.{}, pp.GLOW_THRESHOLD_DEFAULT, pp.GLOW_RADIUS_DEFAULT, 1280, 720);
    try std.testing.expectEqual(@as(u32, 0), empty.id);
    // Empty source, degenerate size, and missing pipelines all fail closed
    // the same way (guard order: pipelines first, then source, then size).
    try std.testing.expectEqual(@as(u32, 0), pass.render(.{ .id = 9 }, 1.0, 4.0, 1280, 720).id);
    try std.testing.expectEqual(@as(u32, 0), pass.render(.{ .id = 9 }, 1.0, 4.0, 0, 720).id);
    // Fail-closed render records no GPU uploads (uniform-only pass: replay
    // in renderReuse stays upload-free by construction).
    try std.testing.expectEqual(@as(u64, 0), upload_meter.takeAndReset());
    // Base size untouched: no resize happened.
    try std.testing.expectEqual(@as(i32, 0), pass.base_width);
    try std.testing.expectEqual(@as(i32, 0), pass.base_height);
    // The sole-contract format helpers are pure (no sg calls).
    try std.testing.expectEqual(sg.PixelFormat.RGBA16F, GlowPass.glowPixelFormat());
    try std.testing.expectEqual(@as(usize, 8), GlowPass.glowBytesPerPixel());
}

test "glow target bytes account three half-res targets" {
    // Pure byte math (no sg calls): exact and deterministic headless. The
    // RGBA16F leg (8 Bpp, via the pure glowBytesPerPixel) shares the same
    // formula; the RGBA8 leg (4 Bpp) is pinned for the census shape.
    try std.testing.expectEqual(@as(usize, 3 * 640 * 360 * 4), GlowPass.targetBytes(1280, 720, 4));
    try std.testing.expectEqual(@as(usize, 3 * 640 * 360 * 8), GlowPass.targetBytes(1280, 720, 8));
    // Doubling both dims quadruples the census (area scaling).
    try std.testing.expectEqual(GlowPass.targetBytes(640, 360, 4) * 4, GlowPass.targetBytes(1280, 720, 4));
    // Degenerate sizes clamp to 1x1 targets, never zero.
    try std.testing.expectEqual(@as(usize, 3 * 1 * 1 * 4), GlowPass.targetBytes(1, 1, 4));
    try std.testing.expectEqual(@as(usize, 3 * 1 * 1 * 4), GlowPass.targetBytes(0, 0, 4));
    // Matches the half-res level-0 sizing the pass allocates.
    const size = pp.bloomMipSize(1280, 720, 0);
    try std.testing.expectEqual(@as(usize, 3 * @as(usize, @intCast(size.w)) * @as(usize, @intCast(size.h)) * 4), GlowPass.targetBytes(1280, 720, 4));
}
