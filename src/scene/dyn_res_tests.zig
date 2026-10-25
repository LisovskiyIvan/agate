//! Headless tests for the Scene-side dynamic resolution wiring
//! (`enableDynamicResolution` / `observeFrameTime` / `disableDynamicResolution`).
//! Pure CPU: the controller is pure, and the wiring only touches
//! `post_process.render_scale` + the one-frame `taa.camera_cut` reset.

const std = @import("std");
const Scene = @import("../scene.zig").Scene;
const testScene = @import("../testing.zig").testScene;

const testing = std.testing;

fn deinitFixture(scene: *Scene) void {
    const alloc = scene.allocator;
    scene.lights.deinit(alloc);
    scene.cameras.deinit(alloc);
    scene.draws.deinit(alloc);
    scene.gpu_retire.deinit(alloc);
    scene.profiler.deinit();
}

test "dyn res: disabled by default — observeFrameTime is a no-op" {
    var scene = testScene(testing.allocator);
    defer deinitFixture(&scene);
    try testing.expect(scene.dyn_res == null);
    try testing.expectEqual(@as(?f32, null), scene.observeFrameTime(100.0));
    try testing.expectEqual(@as(f32, 1.0), scene.post_process.render_scale);
    try testing.expect(!scene.post_process.taa.camera_cut);
}

test "dyn res: sustained overrun applies the scale and raises a ONE-frame reset" {
    var scene = testScene(testing.allocator);
    defer deinitFixture(&scene);

    scene.enableDynamicResolution(.{
        .target_frame_ms = 10.0,
        .warmup_frames = 0,
        .cooldown_frames = 0,
        .sustained_frames = 2,
        .max_down_step = 0.1,
    });

    // No change frames: null, flag untouched.
    _ = scene.observeFrameTime(30.0);
    // Change frame (sustained=2): scale applied + camera_cut raised.
    const s1 = scene.observeFrameTime(30.0) orelse return error.TestUnexpectedResult;
    try testing.expectApproxEqAbs(@as(f32, 0.9), s1, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.9), scene.post_process.render_scale, 1e-6);
    try testing.expect(scene.post_process.taa.camera_cut);

    // The NEXT observe clears the reset (one-frame semantics) even though
    // this frame itself changes nothing yet (cooldown).
    try testing.expectEqual(@as(?f32, null), scene.observeFrameTime(30.0));
    try testing.expect(!scene.post_process.taa.camera_cut);
}

test "dyn res: recovery up to native and quiet at the bounds" {
    var scene = testScene(testing.allocator);
    defer deinitFixture(&scene);

    scene.enableDynamicResolution(.{
        .target_frame_ms = 10.0,
        .warmup_frames = 0,
        .cooldown_frames = 0,
        .sustained_frames = 1,
        .max_up_step = 0.05,
        .min_scale = 0.5,
    });
    scene.post_process.render_scale = 0.5;
    scene.dyn_res.?.scale = 0.5;

    var guard: usize = 0;
    while (scene.post_process.render_scale < 1.0) : (guard += 1) {
        _ = scene.observeFrameTime(1.0);
        if (guard > 100) return error.TestUnexpectedResult;
    }
    try testing.expectApproxEqAbs(@as(f32, 1.0), scene.post_process.render_scale, 1e-6);
    // At native with fast frames: quiet, and no lingering reset.
    _ = scene.observeFrameTime(1.0);
    try testing.expect(!scene.post_process.taa.camera_cut);
}

test "dyn res: disable stops managing the scale and the cut bit" {
    var scene = testScene(testing.allocator);
    defer deinitFixture(&scene);

    scene.enableDynamicResolution(.{ .target_frame_ms = 10.0, .warmup_frames = 0, .sustained_frames = 1 });
    scene.post_process.render_scale = 0.75;
    scene.dyn_res.?.scale = 0.75;

    scene.disableDynamicResolution();
    try testing.expect(scene.dyn_res == null);
    // Kept (host decides what to restore to).
    try testing.expectApproxEqAbs(@as(f32, 0.75), scene.post_process.render_scale, 1e-6);
    // A slow frame no longer moves anything.
    try testing.expectEqual(@as(?f32, null), scene.observeFrameTime(100.0));
    try testing.expectApproxEqAbs(@as(f32, 0.75), scene.post_process.render_scale, 1e-6);
    try testing.expect(!scene.post_process.taa.camera_cut);
}

test "ssaa: scene api enable, disable, and toggle" {
    var scene = testScene(testing.allocator);
    defer deinitFixture(&scene);

    try testing.expect(!scene.isSsaaEnabled());
    try testing.expectEqual(@as(f32, 2.0), scene.post_process.ssaa.factor);

    scene.enableSsaa(1.5);
    try testing.expect(scene.isSsaaEnabled());
    try testing.expectEqual(@as(f32, 1.5), scene.post_process.ssaa.factor);

    scene.disableSsaa();
    try testing.expect(!scene.isSsaaEnabled());

    scene.toggleSsaa();
    try testing.expect(scene.isSsaaEnabled());

    scene.toggleSsaa();
    try testing.expect(!scene.isSsaaEnabled());
}
