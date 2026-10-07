//! Screen-space slab refraction v1 (linear HDR). No work/resources without
//! opt-in draws. The background target is half-res linear HDR (RGBA16F) and
//! renders the same clipped opaque geometry the main view draws.
const sg = @import("sokol").gfx;
const target = @import("../render_target.zig");
const draw = @import("draw.zig");
const snapshot = @import("snapshot.zig");
const queues_mod = @import("render_queue.zig");

pub const RefractionCapture = struct {
    target: target.RenderTarget = .{},
    captured_frame: ?u64 = null,

    pub fn deinit(self: *RefractionCapture) void {
        self.target.deinit();
        self.captured_frame = null;
    }
};

pub fn needsCapture(queues: *const queues_mod.RenderQueues) bool {
    for (queues.transparent.items) |item| if (item.draw_record.refractive) return true;
    for (queues.transparent_instanced.items) |batch| if (batch.draw_record.refractive) return true;
    return false;
}

pub fn capture(scene: anytype, draws: anytype, snap: *const snapshot.SceneFrameSnapshot, env: *draw.Environment) void {
    // One camera in v1: never feed a PIP camera another view's background.
    // Reuse re-presents the previous image without uploads or recapture.
    if (!sg.isvalid() or !snap.has_camera or snap.enable_multi_camera or
        snap.screen_w <= 0 or snap.screen_h <= 0 or !needsCapture(&draws.primary)) return;
    const rt = &scene.refraction.target;
    const cam = snap.primary_cam;
    const rect = cam.viewport.toPixelRect(snap.screen_w, snap.screen_h);
    if (rect.width <= 0 or rect.height <= 0) return;
    const width: u32 = @intCast(@max(1, @divTrunc(rect.width, 2)));
    const height: u32 = @intCast(@max(1, @divTrunc(rect.height, 2)));
    if (!scene.rendering_reuse) {
        scene.refraction.captured_frame = null;
        if (!rt.isValid()) {
            rt.* = target.RenderTarget.create(.{ .width = width, .height = height, .color_format = .RGBA16F }) catch return;
        } else if (!rt.resize(width, height)) return;
        if (rt.queuesSampleSelf(&draws.primary, false)) return;
        // Resolve the HDR forward set BEFORE opening the pass: forwardFor
        // may recreate the twin, which must never happen mid-pass.
        const fwd = scene.forwardFor(1, .RGBA16F);
        if (!rt.begin(snap.clear_color, 1)) return;
        var capture_env = env.*;
        capture_env.pipelines = fwd;
        capture_env.capture_opaque_only = true;
        var capture_snap = snap.*;
        capture_snap.screen_w = @intCast(width);
        capture_snap.screen_h = @intCast(height);
        var capture_cam = cam;
        capture_cam.viewport = .{};
        scene.renderSceneView(capture_cam, &draws.primary, &.{}, &.{}, 1, &capture_snap, capture_env, @import("clustered_lights.zig").REFRACTION_VIEW_SLOT);
        rt.end();
        scene.refraction.captured_frame = snap.frame_id;
    }
    if (scene.refraction.captured_frame != snap.frame_id or !rt.isValid() or rt.width != width or rt.height != height) return;
    env.refraction_view = rt.sampleView();
    env.refraction_sampler = rt.sampleSampler();
    env.refraction_view_proj = cam.view_proj;
    env.refraction_capture = .{ 1, @floatFromInt(width), @floatFromInt(height), 0 };
}

test "refraction empty queues need no capture" {
    const std = @import("std");
    const queues = queues_mod.RenderQueues{};
    try std.testing.expect(!needsCapture(&queues));
}
