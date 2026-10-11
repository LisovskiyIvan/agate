const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const ssgi_mod = @import("ssgi_pass.zig");
const SsgiPass = ssgi_mod.SsgiPass;
const pp = @import("../postprocess.zig");
const upload_meter = @import("../gpu_upload_meter.zig");

test "ssgi pass fail-closes headless with no state touched" {
    // Zero-initialized pass (never init'ed: no sg context headless, same as
    // the GlowPass fail-closed shape) must return an empty view before any
    // sg.* call — disabled pass mode touches nothing.
    var pass: SsgiPass = .{};
    _ = upload_meter.takeAndReset();
    const empty = pass.render(.{});
    try std.testing.expectEqual(@as(u32, 0), empty.id);
    // Missing pipelines, empty inputs, and degenerate sizes all fail
    // closed the same way (guard order: pipelines first, then inputs,
    // then size — before the clamped() gate and any resize).
    try std.testing.expectEqual(@as(u32, 0), pass.render(.{ .scene_view = .{ .id = 9 }, .depth_view = .{ .id = 10 }, .base_w = 1280, .base_h = 720 }).id);
    try std.testing.expectEqual(@as(u32, 0), pass.render(.{ .scene_view = .{ .id = 9 }, .base_w = 1280, .base_h = 720 }).id);
    try std.testing.expectEqual(@as(u32, 0), pass.render(.{ .scene_view = .{ .id = 9 }, .depth_view = .{ .id = 10 }, .base_w = 0, .base_h = 720 }).id);
    // Inline mode never runs the pass even with live-looking inputs.
    const inline_cfg = pp.PostProcessOptions{
        .enabled = true,
        .ssgi = .{
            .enabled = true,
            .intensity = 0.5,
        },
    };
    try std.testing.expectEqual(@as(u32, 0), pass.render(.{ .scene_view = .{ .id = 9 }, .depth_view = .{ .id = 10 }, .config = inline_cfg, .base_w = 1280, .base_h = 720 }).id);
    // Fail-closed render records no GPU uploads (uniform-only pass: replay
    // in renderReuse stays upload-free by construction).
    try std.testing.expectEqual(@as(u64, 0), upload_meter.takeAndReset());
    // Base size untouched: no resize happened.
    try std.testing.expectEqual(@as(i32, 0), pass.width);
    try std.testing.expectEqual(@as(i32, 0), pass.height);
    // The sole-contract format helpers are pure (no sg calls).
    try std.testing.expectEqual(sg.PixelFormat.RGBA16F, SsgiPass.ssgiPixelFormat());
    try std.testing.expectEqual(@as(usize, 8), SsgiPass.ssgiBytesPerPixel());
}

test "ssgi target bytes account two half-res targets" {
    // Pure byte math (no sg calls): exact and deterministic headless. Two
    // half-res RGBA16F targets (gather + denoised) at 8 Bpp.
    try std.testing.expectEqual(@as(usize, 2 * 640 * 360 * 8), SsgiPass.targetBytes(1280, 720, 8));
    try std.testing.expectEqual(@as(usize, 2 * 640 * 360 * 4), SsgiPass.targetBytes(1280, 720, 4));
    // Doubling both dims quadruples the census (area scaling).
    try std.testing.expectEqual(SsgiPass.targetBytes(640, 360, 8) * 4, SsgiPass.targetBytes(1280, 720, 8));
    // Odd sizes round halves up (same rule the pass allocates).
    try std.testing.expectEqual(@as(usize, 2 * 641 * 361 * 8), SsgiPass.targetBytes(1281, 721, 8));
    // Degenerate sizes census zero, never negative wrap.
    try std.testing.expectEqual(@as(usize, 0), SsgiPass.targetBytes(0, 0, 8));
    // Matches the ssgiTargetSize halves the pass allocates.
    const size = pp.ssgiTargetSize(1280, 720);
    try std.testing.expectEqual(@as(usize, 2 * @as(usize, @intCast(size.w)) * @as(usize, @intCast(size.h)) * 8), SsgiPass.targetBytes(1280, 720, 8));
}

test "ssgi temporal pair is unsized until the first temporal render" {
    // Zero pass: no history targets, census gate quiet (headless-safe,
    // pure handle checks — no sg calls).
    const pass: SsgiPass = .{};
    try std.testing.expect(!pass.temporalSized());
    try std.testing.expectEqual(@as(u64, 0), pass.temp_frame);
    try std.testing.expect(!pass.temp_has_history);
    try std.testing.expect(!pass.temp_enabled_prev);
    // History pair census: two half-res RGBA16F targets, same halves.
    try std.testing.expectEqual(@as(usize, 2 * 640 * 360 * 8), SsgiPass.temporalTargetBytes(1280, 720, 8));
    try std.testing.expectEqual(@as(usize, 2 * 641 * 361 * 8), SsgiPass.temporalTargetBytes(1281, 721, 8));
    try std.testing.expectEqual(@as(usize, 0), SsgiPass.temporalTargetBytes(0, 0, 8));
}

test "ssgi render args default to no-velocity, no-cut" {
    // A caller that only fills scene/depth/config gets the camera-fallback
    // reprojection with no history reset — the fail-closed shape relies on
    // these defaults (empty velocity view + false flags).
    const args = SsgiPass.RenderArgs{};
    try std.testing.expectEqual(@as(u32, 0), args.velocity_view.id);
    try std.testing.expect(!args.velocity_valid);
    try std.testing.expect(!args.camera_cut);
}
