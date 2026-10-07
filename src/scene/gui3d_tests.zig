const std = @import("std");
const Scene = @import("../scene.zig").Scene;
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const gpu_thread = @import("../gpu_thread.zig");

test "Scene 3D-GUI panel add/remove/count/dirty" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.gui3d.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);

    const a = try scene.addUi3dPanel("hud", Vec3.new(0, 1, 5), .{});
    const b = try scene.addUi3dPanel("map", Vec3.new(3, 1, 5), .{ .canvas_width = 256, .canvas_height = 256 });
    try std.testing.expectEqual(@as(usize, 0), a);
    try std.testing.expectEqual(@as(usize, 1), b);
    try std.testing.expectEqual(@as(usize, 2), scene.ui3dPanelCount());
    try std.testing.expectEqualStrings("map", scene.getUi3dPanel(1).?.name);
    try std.testing.expectEqualStrings("hud", scene.getUi3dPanelByName("hud").?.name);
    try std.testing.expect(scene.getUi3dPanelByName("missing") == null);
    try std.testing.expect(scene.getUi3dPanel(7) == null);

    // Fresh panels start dirty (scheduled for on-demand capture).
    try std.testing.expectEqual(@as(usize, 2), scene.ui3dDirtyCount());
    scene.markUi3dPanelDirty(0);
    scene.markAllUi3dPanelsDirty();
    try std.testing.expectEqual(@as(usize, 2), scene.ui3dDirtyCount());

    // Cap: 4 max, hard error past it.
    _ = try scene.addUi3dPanel("c", Vec3.zero, .{});
    _ = try scene.addUi3dPanel("d", Vec3.zero, .{});
    try std.testing.expectError(error.TooManyUi3dPanels, scene.addUi3dPanel("e", Vec3.zero, .{}));
    try std.testing.expectEqual(@as(usize, 4), scene.ui3dPanelCount());

    // Removal retires through the epoch queue and keeps index order.
    scene.removeUi3dPanel(0);
    try std.testing.expectEqual(@as(usize, 3), scene.ui3dPanelCount());
    try std.testing.expectEqualStrings("map", scene.getUi3dPanel(0).?.name);
    try std.testing.expectEqual(@as(usize, 1), scene.gpu_retire.retainedCount());
    // Out-of-range removal is a no-op (never retires).
    scene.removeUi3dPanel(42);
    try std.testing.expectEqual(@as(usize, 3), scene.ui3dPanelCount());
    try std.testing.expectEqual(@as(usize, 1), scene.gpu_retire.retainedCount());
}

test "Scene pickUi3dPanel uses the staged snapshot camera" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.gui3d.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);

    // No staged camera: clean miss (never a live ray into the scene).
    try std.testing.expect(scene.pickUi3dPanel(400, 300) == null);

    // Stage a snapshot camera headlessly: identity view-proj over 800x600,
    // so the screen-center ray is origin +Z by construction.
    const front = scene.draws.front;
    scene.draws.slots[front].snapshot.has_camera = true;
    scene.draws.slots[front].snapshot.screen_w = 800;
    scene.draws.slots[front].snapshot.screen_h = 600;
    scene.draws.slots[front].snapshot.primary_cam.view_proj = Mat4.identity;

    // Still nothing: no panels exist.
    try std.testing.expect(scene.pickUi3dPanel(400, 300) == null);

    _ = try scene.addUi3dPanel("hud", Vec3.new(0, 0, 5), .{
        .width = 2.0,
        .height = 2.0,
        .yaw_deg = 180.0,
        .canvas_width = 512,
        .canvas_height = 256,
    });
    // Uncaptured panel: invisible to picking until the first capture lands.
    try std.testing.expect(scene.pickUi3dPanel(400, 300) == null);
    scene.gui3d.notifyCaptured(0);

    // Screen center hits the quad center: panel 0, mid-canvas pixels.
    const hit = scene.pickUi3dPanel(400, 300).?;
    try std.testing.expectEqual(@as(usize, 0), hit.panel_index);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), hit.u, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), hit.v, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 256.0), hit.canvas_x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 128.0), hit.canvas_y, 1e-4);

    // The app routes the pick result into the canvas input state itself.
    const panel = scene.getUi3dPanel(hit.panel_index).?;
    panel.injectPointer(hit.canvas_x, hit.canvas_y, true);
    try std.testing.expect(panel.canvas.?.mouse_down);
    panel.injectRelease();
    try std.testing.expect(!panel.canvas.?.mouse_down);

    // Off-panel cursor: clean miss. Disabled panel: skipped entirely.
    try std.testing.expect(scene.pickUi3dPanel(-100, 300) == null);
    panel.enabled = false;
    try std.testing.expect(scene.pickUi3dPanel(-100, 300) == null);
}
