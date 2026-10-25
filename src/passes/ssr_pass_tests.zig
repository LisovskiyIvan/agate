const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const ssr_mod = @import("ssr_pass.zig");
const SsrPass = ssr_mod.SsrPass;
const pp = @import("../postprocess.zig");

test "ssr pass fail-closes headless with no state touched" {
    var pass: SsrPass = .{};
    const empty = pass.render(.{});
    try std.testing.expectEqual(@as(u32, 0), empty.view.id);
    try std.testing.expect(!empty.temporal_executed);

    try std.testing.expectEqual(@as(u32, 0), pass.render(.{ .scene_view = .{ .id = 9 }, .depth_view = .{ .id = 10 }, .base_w = 1280, .base_h = 720 }).view.id);
    try std.testing.expectEqual(@as(u32, 0), pass.render(.{ .scene_view = .{ .id = 9 }, .base_w = 1280, .base_h = 720 }).view.id);
    try std.testing.expectEqual(@as(u32, 0), pass.render(.{ .scene_view = .{ .id = 9 }, .depth_view = .{ .id = 10 }, .base_w = 0, .base_h = 720 }).view.id);

    // Inline mode never runs the pass even with live-looking inputs
    const inline_cfg = pp.PostProcessOptions{
        .enabled = true,
        .ssr = .{
            .enabled = true,
            .intensity = 0.5,
            .mode = .inline_,
        },
    };
    try std.testing.expectEqual(@as(u32, 0), pass.render(.{ .scene_view = .{ .id = 9 }, .depth_view = .{ .id = 10 }, .config = inline_cfg, .base_w = 1280, .base_h = 720 }).view.id);

    try std.testing.expectEqual(@as(i32, 0), pass.width);
    try std.testing.expectEqual(@as(i32, 0), pass.height);
    try std.testing.expectEqual(sg.PixelFormat.RGBA16F, SsrPass.ssrPixelFormat());
    try std.testing.expectEqual(@as(usize, 8), SsrPass.ssrBytesPerPixel());
}

test "ssr target bytes account two half-res targets" {
    try std.testing.expectEqual(@as(usize, 2 * 640 * 360 * 8), SsrPass.targetBytes(1280, 720, 8));
    try std.testing.expectEqual(@as(usize, 2 * 640 * 360 * 4), SsrPass.targetBytes(1280, 720, 4));
    try std.testing.expectEqual(SsrPass.targetBytes(640, 360, 8) * 4, SsrPass.targetBytes(1280, 720, 8));
    try std.testing.expectEqual(@as(usize, 2 * 641 * 361 * 8), SsrPass.targetBytes(1281, 721, 8));
    try std.testing.expectEqual(@as(usize, 0), SsrPass.targetBytes(0, 0, 8));

    const size = pp.ssrTargetSize(1280, 720);
    try std.testing.expectEqual(@as(usize, 2 * @as(usize, @intCast(size.w)) * @as(usize, @intCast(size.h)) * 8), SsrPass.targetBytes(1280, 720, 8));
}

test "ssr temporal pair is unsized until the first temporal render" {
    const pass: SsrPass = .{};
    try std.testing.expect(!pass.temporalSized());
    try std.testing.expectEqual(@as(usize, 2 * 640 * 360 * 8), SsrPass.temporalTargetBytes(1280, 720, 8));
    try std.testing.expectEqual(@as(usize, 2 * 641 * 361 * 8), SsrPass.temporalTargetBytes(1281, 721, 8));
    try std.testing.expectEqual(@as(usize, 0), SsrPass.temporalTargetBytes(0, 0, 8));
}

test "ssr resetTemporalHistory invalidates the history headless" {
    var pass: SsrPass = .{};
    pass.temp_frame = 5;
    pass.temp_has_history = true;
    pass.temp_enabled_prev = true;
    pass.resetTemporalHistory();
    try std.testing.expectEqual(@as(u64, 0), pass.temp_frame);
    try std.testing.expect(!pass.temp_has_history);
    try std.testing.expect(!pass.temp_enabled_prev);
    try std.testing.expect(!pass.temporalSized());
}
