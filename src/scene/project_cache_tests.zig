const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Camera = @import("../camera.zig").Camera;
const scene_projection = @import("projection.zig");
const project_cache = @import("project_cache.zig");
const ProjectCache = project_cache.ProjectCache;

test "project VP cache: equal cameras hit" {
    const a: Camera = .{ .arc_rotate = .{ .alpha = 0.5, .beta = 1.0, .radius = 8.0 } };
    const b: Camera = .{ .arc_rotate = .{ .alpha = 0.5, .beta = 1.0, .radius = 8.0 } };
    try std.testing.expect(scene_projection.camerasEqualForProjection(a, b));
    // Name pointers and input state do not affect the matrices.
    const c: Camera = .{ .arc_rotate = .{ .alpha = 0.5, .beta = 1.0, .radius = 8.0, .name = "other", .is_dragging = true } };
    try std.testing.expect(scene_projection.camerasEqualForProjection(a, c));

    const fa: Camera = .{ .free = .{ .position = Vec3.new(1.0, 2.0, 3.0) } };
    const fb: Camera = .{ .free = .{ .position = Vec3.new(1.0, 2.0, 3.0) } };
    try std.testing.expect(scene_projection.camerasEqualForProjection(fa, fb));

    const ga: Camera = .{ .follow = .{ .target_position = Vec3.new(1.0, 0.0, 0.0), .radius = 5.0 } };
    const gb: Camera = .{ .follow = .{ .target_position = Vec3.new(1.0, 0.0, 0.0), .radius = 5.0 } };
    try std.testing.expect(scene_projection.camerasEqualForProjection(ga, gb));
}

test "project VP cache: changed field misses" {
    const a: Camera = .{ .arc_rotate = .{ .radius = 8.0 } };
    const b: Camera = .{ .arc_rotate = .{ .radius = 9.0 } };
    try std.testing.expect(!scene_projection.camerasEqualForProjection(a, b));

    const fa: Camera = .{ .free = .{ .position = Vec3.new(1.0, 2.0, 3.0) } };
    const fb: Camera = .{ .free = .{ .position = Vec3.new(1.0, 2.0, 4.0) } };
    try std.testing.expect(!scene_projection.camerasEqualForProjection(fa, fb));

    const ga: Camera = .{ .follow = .{ .radius = 5.0 } };
    const gb: Camera = .{ .follow = .{ .radius = 6.0 } };
    try std.testing.expect(!scene_projection.camerasEqualForProjection(ga, gb));

    // Distinct target meshes miss even when every scalar matches.
    var m1: @import("../mesh.zig").Mesh = undefined;
    var m2: @import("../mesh.zig").Mesh = undefined;
    const ha: Camera = .{ .follow = .{ .target_mesh = &m1 } };
    const hb: Camera = .{ .follow = .{ .target_mesh = &m2 } };
    try std.testing.expect(!scene_projection.camerasEqualForProjection(ha, hb));
}

test "project VP cache: different union variants miss" {
    const arc: Camera = .{ .arc_rotate = .{} };
    const free: Camera = .{ .free = .{} };
    const follow: Camera = .{ .follow = .{} };
    try std.testing.expect(!scene_projection.camerasEqualForProjection(arc, free));
    try std.testing.expect(!scene_projection.camerasEqualForProjection(free, follow));
    try std.testing.expect(!scene_projection.camerasEqualForProjection(follow, arc));
}

test "project VP cache: target and fly cameras" {
    // Equal target pairs hit; tuning-only (smoothing) differences still hit.
    const ta: Camera = .{ .target = .{ .position = Vec3.new(0.0, 0.0, 5.0), .target = Vec3.zero } };
    const tb: Camera = .{ .target = .{ .position = Vec3.new(0.0, 0.0, 5.0), .target = Vec3.zero } };
    try std.testing.expect(scene_projection.camerasEqualForProjection(ta, tb));
    const tc: Camera = .{ .target = .{ .position = Vec3.new(0.0, 0.0, 5.0), .target = Vec3.zero, .smoothing = 0.0 } };
    try std.testing.expect(scene_projection.camerasEqualForProjection(ta, tc));

    // Moved target point, moved position, pending goal, and changed up miss.
    const td: Camera = .{ .target = .{ .position = Vec3.new(0.0, 0.0, 5.0), .target = Vec3.new(1.0, 0.0, 0.0) } };
    try std.testing.expect(!scene_projection.camerasEqualForProjection(ta, td));
    const te: Camera = .{ .target = .{ .position = Vec3.new(0.0, 1.0, 5.0), .target = Vec3.zero } };
    try std.testing.expect(!scene_projection.camerasEqualForProjection(ta, te));
    var tf: Camera = .{ .target = .{ .position = Vec3.new(0.0, 0.0, 5.0), .target = Vec3.zero } };
    tf.target.desired_target = Vec3.new(0.0, 1.0, 0.0);
    try std.testing.expect(!scene_projection.camerasEqualForProjection(ta, tf));
    const tg: Camera = .{ .target = .{ .position = Vec3.new(0.0, 5.0, 0.0), .target = Vec3.zero, .up = Vec3.new(0.0, 0.0, 1.0) } };
    const th: Camera = .{ .target = .{ .position = Vec3.new(0.0, 5.0, 0.0), .target = Vec3.zero } };
    try std.testing.expect(!scene_projection.camerasEqualForProjection(tg, th));

    // Equal fly pairs hit; roll-only differences miss (roll tilts view up).
    const fa: Camera = .{ .fly = .{ .position = Vec3.new(1.0, 2.0, 3.0) } };
    const fb: Camera = .{ .fly = .{ .position = Vec3.new(1.0, 2.0, 3.0) } };
    try std.testing.expect(scene_projection.camerasEqualForProjection(fa, fb));
    const fc: Camera = .{ .fly = .{ .position = Vec3.new(1.0, 2.0, 3.0), .rotation = Vec3.new(0.0, 0.0, 90.0) } };
    try std.testing.expect(!scene_projection.camerasEqualForProjection(fa, fc));
    const fd: Camera = .{ .fly = .{ .position = Vec3.new(1.0, 2.0, 4.0) } };
    try std.testing.expect(!scene_projection.camerasEqualForProjection(fa, fd));

    // Cross-variant pairs (old and new) always miss.
    const arc: Camera = .{ .arc_rotate = .{} };
    const free: Camera = .{ .free = .{} };
    const follow: Camera = .{ .follow = .{} };
    try std.testing.expect(!scene_projection.camerasEqualForProjection(ta, fa));
    try std.testing.expect(!scene_projection.camerasEqualForProjection(fa, ta));
    try std.testing.expect(!scene_projection.camerasEqualForProjection(arc, ta));
    try std.testing.expect(!scene_projection.camerasEqualForProjection(ta, arc));
    try std.testing.expect(!scene_projection.camerasEqualForProjection(free, fa));
    try std.testing.expect(!scene_projection.camerasEqualForProjection(fa, free));
    try std.testing.expect(!scene_projection.camerasEqualForProjection(follow, ta));
    try std.testing.expect(!scene_projection.camerasEqualForProjection(fa, follow));
}

test "project cache reuses vp until camera or viewport changes" {
    var cache: ProjectCache = .{};
    try std.testing.expect(!cache.vp_valid);

    const cam_a: Camera = .{ .arc_rotate = .{ .alpha = 0.5, .beta = 1.0, .radius = 8.0 } };
    const vp1 = cache.viewProjection(cam_a, 1280.0, 720.0);
    try std.testing.expect(cache.vp_valid);
    // Equal camera + viewport hits the cache (same bits out).
    const vp2 = cache.viewProjection(cam_a, 1280.0, 720.0);
    for (vp1.m, vp2.m) |x, y| try std.testing.expectEqual(x, y);

    // Viewport change invalidates.
    _ = cache.viewProjection(cam_a, 640.0, 720.0);
    try std.testing.expectEqual(@as(f32, 640.0), cache.w);

    // Camera movement invalidates and reflects the new matrices.
    const cam_b: Camera = .{ .arc_rotate = .{ .alpha = 0.5, .beta = 1.5, .radius = 8.0 } };
    const vp3 = cache.viewProjection(cam_b, 640.0, 720.0);
    const vp3_fresh = cam_b.getViewProjection(640.0 / 720.0);
    for (vp3.m, vp3_fresh.m) |x, y| try std.testing.expectEqual(x, y);
}
