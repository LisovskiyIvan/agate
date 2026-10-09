const std = @import("std");
const volumetric_mod = @import("volumetric_pass.zig");
const VolumetricPass = volumetric_mod.VolumetricPass;
const pp = @import("../postprocess.zig");
const upload_meter = @import("../gpu_upload_meter.zig");

test "volumetric pass fail-closes headless with no state touched" {
    // Zero-initialized pass (never init'ed: no sg context headless, same
    // as the GlowPass fail-closed shape) must return an empty view before
    // any sg.* call.
    var pass: VolumetricPass = .{};
    _ = upload_meter.takeAndReset();
    const empty = pass.render(.{});
    try std.testing.expectEqual(@as(u32, 0), empty.id);
    // Missing pipelines, missing depth/shadow views, and degenerate sizes
    // all fail closed the same way (guard order: pipelines, views, size).
    try std.testing.expectEqual(@as(u32, 0), pass.render(.{ .depth_view = .{ .id = 9 }, .shadow_view = .{ .id = 10 }, .base_w = 1280, .base_h = 720 }).id);
    try std.testing.expectEqual(@as(u32, 0), pass.render(.{ .depth_view = .{ .id = 9 }, .base_w = 1280, .base_h = 720 }).id);
    try std.testing.expectEqual(@as(u32, 0), pass.render(.{ .depth_view = .{ .id = 9 }, .shadow_view = .{ .id = 10 }, .base_w = 0, .base_h = 720 }).id);
    // Fail-closed render records no GPU uploads and sizes nothing.
    try std.testing.expectEqual(@as(u64, 0), upload_meter.takeAndReset());
    try std.testing.expectEqual(@as(i32, 0), pass.base_width);
    try std.testing.expectEqual(@as(i32, 0), pass.base_height);
    try std.testing.expectEqual(@as(u32, 0), pass.raymarch_image.id);
}

test "volumetric target bytes account three shaft-res targets" {
    // Pure byte math (no sg calls): exact and deterministic headless. The
    // RGBA16F leg (8 Bpp, via the pure shaftBytesPerPixel) shares the same
    // formula; the RGBA8 leg (4 Bpp) is pinned for the census shape.
    try std.testing.expectEqual(@as(usize, 3 * 320 * 180 * 4), VolumetricPass.targetBytes(1280, 720, 4, .quarter));
    try std.testing.expectEqual(@as(usize, 3 * 640 * 360 * 4), VolumetricPass.targetBytes(1280, 720, 4, .half));
    try std.testing.expectEqual(@as(usize, 3 * 320 * 180 * 8), VolumetricPass.targetBytes(1280, 720, 8, .quarter));
    // Quarter is exactly a fourth of half (area scaling).
    try std.testing.expectEqual(VolumetricPass.targetBytes(1280, 720, 4, .half) / 4, VolumetricPass.targetBytes(1280, 720, 4, .quarter));
    // Degenerate sizes clamp to 1x1 targets, never zero.
    try std.testing.expectEqual(@as(usize, 3 * 1 * 1 * 4), VolumetricPass.targetBytes(1, 1, 4, .quarter));
    try std.testing.expectEqual(@as(usize, 3 * 1 * 1 * 4), VolumetricPass.targetBytes(0, 0, 4, .half));
    // Matches the shaft sizing the pass allocates.
    const size = pp.shaftTargetSize(1280, 720, .quarter);
    try std.testing.expectEqual(@as(usize, 3 * @as(usize, @intCast(size.w)) * @as(usize, @intCast(size.h)) * 4), VolumetricPass.targetBytes(1280, 720, 4, .quarter));
}
