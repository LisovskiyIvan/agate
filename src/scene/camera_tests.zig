const std = @import("std");
const Scene = @import("../scene.zig").Scene;
const camera_mod = @import("../camera.zig");
const Camera = camera_mod.Camera;

test "Scene camera switching and cycling" {
    const ally = std.testing.allocator;
    var scene: Scene = undefined;
    scene.allocator = ally;
    scene.cameras = .empty;
    scene.active_camera_index = null;
    scene.active_camera = null;
    scene.active_camera_owned_name = null;
    scene.enable_multi_camera = true;

    defer {
        for (scene.cameras.items) |entry| {
            if (entry.owns_name) ally.free(entry.name);
        }
        scene.cameras.deinit(ally);
    }

    const cam1 = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    const cam2 = Camera{ .arc_rotate = camera_mod.ArcRotateCamera.init("Cam2", .{}) };
    const cam3 = Camera{ .fly = camera_mod.FlyCamera.init("Cam3", .{}) };

    const idx0 = try scene.addCamera(.{ .name = "Cam1", .camera = cam1 });
    try std.testing.expectEqual(@as(usize, 0), idx0);
    try std.testing.expectEqual(@as(usize, 0), scene.getActiveCameraIndex().?);
    try std.testing.expectEqualStrings("Cam1", scene.getActiveCameraName().?);

    _ = try scene.addCamera(.{ .name = "Cam2", .camera = cam2 });
    _ = try scene.addCamera(.{ .name = "Cam3", .camera = cam3 });
    try std.testing.expectEqual(@as(usize, 3), scene.getCameraCount());

    // Cycle next: 0 -> 1 -> 2 -> 0
    scene.nextCamera();
    try std.testing.expectEqual(@as(usize, 1), scene.getActiveCameraIndex().?);
    try std.testing.expectEqualStrings("Cam2", scene.getActiveCameraName().?);

    scene.nextCamera();
    try std.testing.expectEqual(@as(usize, 2), scene.getActiveCameraIndex().?);
    try std.testing.expectEqualStrings("Cam3", scene.getActiveCameraName().?);

    scene.nextCamera();
    try std.testing.expectEqual(@as(usize, 0), scene.getActiveCameraIndex().?);

    // Cycle prev: 0 -> 2 -> 1
    scene.prevCamera();
    try std.testing.expectEqual(@as(usize, 2), scene.getActiveCameraIndex().?);

    // Switch by name
    try std.testing.expect(scene.switchCameraByName("Cam2"));
    try std.testing.expectEqual(@as(usize, 1), scene.getActiveCameraIndex().?);
    try std.testing.expect(!scene.switchCameraByName("NonExistent"));
}
