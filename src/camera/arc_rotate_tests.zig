const std = @import("std");
const sokol = @import("sokol");
const arc_rotate_mod = @import("arc_rotate.zig");
const ArcRotateCamera = arc_rotate_mod.ArcRotateCamera;

test "ArcRotateCamera inertia disabled behaves instantly" {
    var cam = ArcRotateCamera.init("orbit", .{
        .alpha = 0.0,
        .beta = std.math.pi / 2.0,
        .radius = 10.0,
        .inertia = 0.0,
    });
    var down_ev = sokol.app.Event{
        .type = .MOUSE_DOWN,
        .mouse_button = .LEFT,
        .mouse_x = 100.0,
        .mouse_y = 100.0,
    };
    cam.handleEvent(&down_ev);

    var move_ev = sokol.app.Event{
        .type = .MOUSE_MOVE,
        .mouse_x = 110.0,
        .mouse_y = 105.0,
    };
    cam.handleEvent(&move_ev);

    // With inertia == 0, alpha and beta move immediately
    try std.testing.expect(cam.alpha != 0.0);
    try std.testing.expect(cam.beta != std.math.pi / 2.0);
    try std.testing.expectEqual(@as(f32, 0.0), cam.inertial_alpha_offset);
    try std.testing.expectEqual(@as(f32, 0.0), cam.inertial_beta_offset);
}

test "ArcRotateCamera inertia enables smooth damping across update ticks" {
    var cam = ArcRotateCamera.init("orbit", .{
        .alpha = 0.0,
        .beta = std.math.pi / 2.0,
        .radius = 10.0,
        .inertia = 0.9,
    });
    var down_ev = sokol.app.Event{
        .type = .MOUSE_DOWN,
        .mouse_button = .LEFT,
        .mouse_x = 100.0,
        .mouse_y = 100.0,
    };
    cam.handleEvent(&down_ev);

    var move_ev = sokol.app.Event{
        .type = .MOUSE_MOVE,
        .mouse_x = 120.0,
        .mouse_y = 110.0,
    };
    cam.handleEvent(&move_ev);

    // With inertia > 0, angle does not change immediately during drag
    try std.testing.expectEqual(@as(f32, 0.0), cam.alpha);
    try std.testing.expectEqual(@as(f32, std.math.pi / 2.0), cam.beta);
    try std.testing.expect(cam.inertial_alpha_offset != 0.0);
    try std.testing.expect(cam.inertial_beta_offset != 0.0);

    // Step 1: update moves the camera towards the target and decays offsets
    cam.update(1.0 / 60.0);
    const alpha_step1 = cam.alpha;
    try std.testing.expect(alpha_step1 < 0.0); // Moved
    const offset_step1 = cam.inertial_alpha_offset;
    try std.testing.expect(offset_step1 < 20.0 * cam.angular_sensitivity); // Decayed
    // Step 2: continues to move smoothly
    cam.update(1.0 / 60.0);
    try std.testing.expect(cam.alpha < alpha_step1);
    try std.testing.expect(cam.inertial_alpha_offset < offset_step1);
}
