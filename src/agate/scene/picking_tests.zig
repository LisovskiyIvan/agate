//! Tests for `picking.zig` (moved from `picking.zig` inline blocks).
const std = @import("std");
const sokol = @import("sokol");
const sapp = sokol.app;
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Ray = math.Ray;
const RayHit = math.RayHit;
const physics = @import("../physics.zig");
const Camera = @import("../camera.zig").Camera;
const Viewport = @import("../camera.zig").Viewport;
const Mesh = @import("../mesh.zig").Mesh;
const InstancedMesh = @import("../mesh.zig").InstancedMesh;
const PickingInfo = physics.PickingInfo;
const PhysicsWorld = physics.PhysicsWorld;
const pick = @import("picking.zig");
const createPickingRayViewport = pick.createPickingRayViewport;
const viewportContainsPoint = pick.viewportContainsPoint;
const selectPickIndex = pick.selectPickIndex;
const pickWithRay = pick.pickWithRay;

test "viewport ray centers on the viewport, not the window" {
    const FreeCamera = @import("../camera.zig").FreeCamera;
    const cam: Camera = .{ .free = FreeCamera.init("test", .{ .position = Vec3.new(0, 0, 5) }) };
    const fb_w: f32 = 800.0;
    const fb_h: f32 = 600.0;
    // Right-half PIP viewport: its center is at x=600 in window pixels.
    const pip = Viewport{ .x = 0.5, .y = 0.0, .width = 0.5, .height = 1.0 };
    const r = createPickingRayViewport(cam, 600.0, 300.0, fb_w, fb_h, pip);
    // Viewport center looks straight ahead (-Z for the default free camera).
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), r.direction.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), r.direction.y, 1e-4);
    try std.testing.expect(r.direction.z < -0.9);
    // Same cursor through the whole-window formula points off-center instead.
    const whole = createPickingRayViewport(cam, 600.0, 300.0, fb_w, fb_h, .{});
    try std.testing.expect(whole.direction.x > 0.1);
    // Containment follows the pixel rect (right half only).
    try std.testing.expect(viewportContainsPoint(pip, 600.0, 300.0, fb_w, fb_h));
    try std.testing.expect(!viewportContainsPoint(pip, 200.0, 300.0, fb_w, fb_h));
}

test "overlay camera wins under the cursor, fallback is null" {
    const entries = [_]struct { viewport: Viewport, enabled: bool }{
        .{ .viewport = .{}, .enabled = true },
        .{ .viewport = .{ .x = 0.5, .y = 0.0, .width = 0.5, .height = 1.0 }, .enabled = true },
    };
    // Cursor inside the PIP overlay: topmost (index 1) wins over fullscreen primary.
    try std.testing.expectEqual(@as(?usize, 1), selectPickIndex(&entries, 0, 600.0, 300.0, 800.0, 600.0));
    // Cursor in the left half: only the primary covers it.
    try std.testing.expectEqual(@as(?usize, 0), selectPickIndex(&entries, 0, 200.0, 300.0, 800.0, 600.0));
    // Disabled overlay falls through to the primary.
    const disabled = [_]struct { viewport: Viewport, enabled: bool }{
        .{ .viewport = .{}, .enabled = true },
        .{ .viewport = .{ .x = 0.5, .y = 0.0, .width = 0.5, .height = 1.0 }, .enabled = false },
    };
    try std.testing.expectEqual(@as(?usize, 0), selectPickIndex(&disabled, 0, 600.0, 300.0, 800.0, 600.0));
    // No coverage at all: sensible null fallback (Scene maps this to PickingInfo{}).
    const halves = [_]struct { viewport: Viewport, enabled: bool }{
        .{ .viewport = .{ .x = 0.0, .y = 0.0, .width = 0.5, .height = 1.0 }, .enabled = true },
        .{ .viewport = .{ .x = 0.5, .y = 0.0, .width = 0.5, .height = 0.5 }, .enabled = true },
    };
    try std.testing.expectEqual(@as(?usize, null), selectPickIndex(&halves, 0, 600.0, 500.0, 800.0, 600.0));
}

test "picking containment matches the rounded toPixelRect edges" {
    // 800/3 = 266.67: rendering rounds the rect to x=267 w=267, so picking
    // must use the same edges (unrounded math would admit 266.9 and drop
    // 533.5, disagreeing with the drawn viewport by a pixel).
    const vp = Viewport{ .x = 1.0 / 3.0, .y = 0.0, .width = 1.0 / 3.0, .height = 1.0 };
    const rect = vp.toPixelRect(800, 600);
    try std.testing.expectEqual(@as(i32, 267), rect.x);
    try std.testing.expectEqual(@as(i32, 267), rect.width);
    try std.testing.expect(!viewportContainsPoint(vp, 266.9, 300.0, 800.0, 600.0));
    try std.testing.expect(viewportContainsPoint(vp, 267.0, 300.0, 800.0, 600.0));
    try std.testing.expect(viewportContainsPoint(vp, 533.5, 300.0, 800.0, 600.0));
    try std.testing.expect(!viewportContainsPoint(vp, 534.0, 300.0, 800.0, 600.0));
}

test "sphere picking follows the world transform for parented meshes" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var parent = Mesh{
        .name = "parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(10, 0, 0),
    };
    var child = Mesh{
        .name = "child",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .parent = &parent,
    };
    const body = try pw.createBody(&child, .sphere, 0.0);
    body.sphere_radius = 1.0;
    var list = [_]*Mesh{&child};

    // Ray through the world position hits (the old local-space code tested
    // mesh.position = origin and missed).
    const hit_info = pickWithRay(&list, &pw, Ray.new(Vec3.new(10, 0, -5), Vec3.new(0, 0, 1)));
    try std.testing.expect(hit_info.hit);
    try std.testing.expectEqual(&child, hit_info.picked_mesh.?);

    // Ray through the old local-space center misses.
    try std.testing.expect(!pickWithRay(&list, &pw, Ray.new(Vec3.new(0, 0, -5), Vec3.new(0, 0, 1))).hit);
}

test "sphere picking follows rotated parents" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var parent = Mesh{
        .name = "parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .rotation = Vec3.new(0, 0, 90),
    };
    var child = Mesh{
        .name = "child",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(1, 0, 0),
        .parent = &parent,
    };
    const body = try pw.createBody(&child, .sphere, 0.0);
    body.sphere_radius = 1.0;
    var list = [_]*Mesh{&child};

    // Rz(90deg) * (1,0,0) = (0,1,0): the rotated world center hits.
    const hit_info = pickWithRay(&list, &pw, Ray.new(Vec3.new(0, 1, -5), Vec3.new(0, 0, 1)));
    try std.testing.expect(hit_info.hit);
    try std.testing.expectEqual(&child, hit_info.picked_mesh.?);
    // The unrotated local offset misses.
    try std.testing.expect(!pickWithRay(&list, &pw, Ray.new(Vec3.new(1, 0, -5), Vec3.new(0, 0, 1))).hit);
}

test "sphere picking intersects the true scaled ellipsoid" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var mesh = Mesh{
        .name = "scaled",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(10, 0, 0),
        .scaling = Vec3.new(3, 1, 1),
    };
    const body = try pw.createBody(&mesh, .sphere, 0.0);
    body.sphere_radius = 1.0;
    var list = [_]*Mesh{&mesh};

    // Exact local-space test: scaling (3,1,1) maps the unit sphere to the
    // ellipsoid with semi-axes (3,1,1). y-offset 2 is outside the y
    // semi-axis and must miss (a world-space sphere approximation would
    // false-hit here); x-offset 2.5 is inside the x semi-axis and hits.
    try std.testing.expect(!pickWithRay(&list, &pw, Ray.new(Vec3.new(10, 2, -5), Vec3.new(0, 0, 1))).hit);
    try std.testing.expect(pickWithRay(&list, &pw, Ray.new(Vec3.new(12.5, 0, -5), Vec3.new(0, 0, 1))).hit);
    // Beyond the x semi-axis still misses.
    try std.testing.expect(!pickWithRay(&list, &pw, Ray.new(Vec3.new(14, 0, -5), Vec3.new(0, 0, 1))).hit);
}

test "sphere picking composes rotation with non-uniform parent scale" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    // Parent scale (3,1,1) + child Rz(45°): the world linear part is S·R
    // with singular values (3,1,1), so the ellipsoid's extent along world
    // y is still 1 while a max-column world radius would be sqrt(5) ≈ 2.24
    // and false-hit at y-offset 2 (the under-hit class this test locks).
    var parent = Mesh{
        .name = "scaled_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .scaling = Vec3.new(3, 1, 1),
    };
    var child = Mesh{
        .name = "rotated_child",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(1, 0, 0),
        .rotation = Vec3.new(0, 0, 45),
        .parent = &parent,
    };
    const body = try pw.createBody(&child, .sphere, 0.0);
    body.sphere_radius = 1.0;
    var list = [_]*Mesh{&child};

    const center = child.getWorldMatrix().getTranslation();
    // Ray through the transformed center hits.
    try std.testing.expect(pickWithRay(&list, &pw, Ray.new(center.add(Vec3.new(0, 0, -5)), Vec3.new(0, 0, 1))).hit);
    // Extent along world y is 1: offset 2 is outside and must miss.
    try std.testing.expect(!pickWithRay(&list, &pw, Ray.new(center.add(Vec3.new(0, 2, -5)), Vec3.new(0, 0, 1))).hit);
}

test "sphere picking normal uses the inverse-transpose under rotation" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    // Rz(90°): a ray offset in x hits off-center, so the normal has both x
    // and z components. The world normal must be the local normal mapped by
    // the inverse-transpose of the linear part — (0.5, 0, -0.866) here.
    // Mapping by the matrix itself (a common mix-up: rows vs columns of the
    // column-major inverse) would flip the x sign to -0.5.
    var mesh = Mesh{
        .name = "rot",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .rotation = Vec3.new(0, 0, 90),
    };
    const body = try pw.createBody(&mesh, .sphere, 0.0);
    body.sphere_radius = 1.0;
    var list = [_]*Mesh{&mesh};

    const hit_info = pickWithRay(&list, &pw, Ray.new(Vec3.new(0.5, 0, -5), Vec3.new(0, 0, 1)));
    try std.testing.expect(hit_info.hit);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), hit_info.picked_normal.x, 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), hit_info.picked_normal.y, 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, -0.86603), hit_info.picked_normal.z, 1e-3);
}

test "picking ray with degenerate framebuffer returns a defined forward ray" {
    const FreeCamera = @import("../camera.zig").FreeCamera;
    const cam: Camera = .{ .free = FreeCamera.init("test", .{ .position = Vec3.new(0, 0, 5) }) };
    const origin = cam.getPosition();
    const dims = [_][2]f32{ .{ 0, 600 }, .{ 800, 0 }, .{ -1, 600 }, .{ 800, -2 } };
    for (dims) |d| {
        const rr = createPickingRayViewport(cam, 100.0, 100.0, d[0], d[1], .{});
        try std.testing.expectEqual(origin.x, rr.origin.x);
        try std.testing.expectEqual(origin.y, rr.origin.y);
        try std.testing.expectEqual(origin.z, rr.origin.z);
        try std.testing.expectApproxEqAbs(Vec3.forward.x, rr.direction.x, 1e-6);
        try std.testing.expectApproxEqAbs(Vec3.forward.y, rr.direction.y, 1e-6);
        try std.testing.expectApproxEqAbs(Vec3.forward.z, rr.direction.z, 1e-6);
    }
    // Degenerate containment stays a defined false, never a crash.
    try std.testing.expect(!viewportContainsPoint(.{}, 100.0, 100.0, 0.0, 600.0));
    try std.testing.expect(!viewportContainsPoint(.{}, 100.0, 100.0, 800.0, 0.0));
}

test "instance picking hits an offset instance and reports its index" {
    var src = Mesh{
        .name = "src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .local_bounding_box = math.BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "i0", .source_mesh = &src, .position = Vec3.new(5, 0, 0) };
    var ptrs = [_]*InstancedMesh{&inst0};
    src.instances = .{ .items = ptrs[0..], .capacity = 1 };
    var list = [_]*Mesh{&src};

    const hit_info = pickWithRay(&list, null, Ray.new(Vec3.new(5, 0, -5), Vec3.new(0, 0, 1)));
    try std.testing.expect(hit_info.hit);
    try std.testing.expectEqual(&src, hit_info.picked_mesh.?);
    try std.testing.expectEqual(@as(?usize, 0), hit_info.picked_instance);

    // The source origin holds no instance, so a ray through it misses even
    // though the source mesh itself is under the cursor.
    try std.testing.expect(!pickWithRay(&list, null, Ray.new(Vec3.new(0, 0, -5), Vec3.new(0, 0, 1))).hit);
}

test "instance picking reports the nearer of two instances" {
    var src = Mesh{
        .name = "src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .local_bounding_box = math.BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "i0", .source_mesh = &src, .position = Vec3.new(0, 0, 0) };
    var inst1 = InstancedMesh{ .name = "i1", .source_mesh = &src, .position = Vec3.new(0, 0, 5) };
    var ptrs = [_]*InstancedMesh{ &inst0, &inst1 };
    src.instances = .{ .items = ptrs[0..], .capacity = 2 };
    var list = [_]*Mesh{&src};

    const from_front = pickWithRay(&list, null, Ray.new(Vec3.new(0, 0, -5), Vec3.new(0, 0, 1)));
    try std.testing.expect(from_front.hit);
    try std.testing.expectEqual(@as(?usize, 0), from_front.picked_instance);

    const from_back = pickWithRay(&list, null, Ray.new(Vec3.new(0, 0, 10), Vec3.new(0, 0, -1)));
    try std.testing.expect(from_back.hit);
    try std.testing.expectEqual(@as(?usize, 1), from_back.picked_instance);
}

test "instance picking skips invisible instances" {
    var src = Mesh{
        .name = "src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .local_bounding_box = math.BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "i0", .source_mesh = &src, .position = Vec3.new(0, 0, 0), .is_visible = false };
    var inst1 = InstancedMesh{ .name = "i1", .source_mesh = &src, .position = Vec3.new(0, 0, 5) };
    var ptrs = [_]*InstancedMesh{ &inst0, &inst1 };
    src.instances = .{ .items = ptrs[0..], .capacity = 2 };
    var list = [_]*Mesh{&src};

    // The nearer instance is invisible, so the farther visible one wins.
    const hit_info = pickWithRay(&list, null, Ray.new(Vec3.new(0, 0, -5), Vec3.new(0, 0, 1)));
    try std.testing.expect(hit_info.hit);
    try std.testing.expectEqual(@as(?usize, 1), hit_info.picked_instance);

    // All instances invisible behaves like a miss.
    inst1.is_visible = false;
    try std.testing.expect(!pickWithRay(&list, null, Ray.new(Vec3.new(0, 0, -5), Vec3.new(0, 0, 1))).hit);
}

test "instance picking matches the drawn instance matrix" {
    // The instanced draw path uses TRS(instance) * source.base_matrix as the
    // model matrix; the source mesh's own TRS and parent chain never reach
    // the draw. Picking must mirror that: a parent on the source does NOT
    // move instances, and picks land where instances actually render.
    var parent = Mesh{
        .name = "parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(10, 0, 0),
    };
    var src = Mesh{
        .name = "src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .parent = &parent,
        .local_bounding_box = math.BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "i0", .source_mesh = &src };
    var ptrs = [_]*InstancedMesh{&inst0};
    src.instances = .{ .items = ptrs[0..], .capacity = 1 };
    var list = [_]*Mesh{&src};

    // Instance at its own local origin draws (and picks) at (0,0,0)...
    const hit_info = pickWithRay(&list, null, Ray.new(Vec3.new(0, 0, -5), Vec3.new(0, 0, 1)));
    try std.testing.expect(hit_info.hit);
    try std.testing.expectEqual(@as(?usize, 0), hit_info.picked_instance);
    // ...not at the source parent's position (the draw ignores it).
    try std.testing.expect(!pickWithRay(&list, null, Ray.new(Vec3.new(10, 0, -5), Vec3.new(0, 0, 1))).hit);
}

test "instance picking sees instances of a hidden source" {
    // The instanced draw ignores the source template's own is_visible flag
    // (only per-instance visibility gates it), so a hidden source with a
    // visible instance must still be pickable — exactly where it draws.
    var src = Mesh{
        .name = "src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .is_visible = false,
        .local_bounding_box = math.BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "i0", .source_mesh = &src, .position = Vec3.new(3, 0, 0) };
    var ptrs = [_]*InstancedMesh{&inst0};
    src.instances = .{ .items = ptrs[0..], .capacity = 1 };
    var list = [_]*Mesh{&src};

    const hit_info = pickWithRay(&list, null, Ray.new(Vec3.new(3, 0, -5), Vec3.new(0, 0, 1)));
    try std.testing.expect(hit_info.hit);
    try std.testing.expectEqual(@as(?usize, 0), hit_info.picked_instance);
}

test "plain mesh picking still reports a null instance index" {
    var src = Mesh{
        .name = "src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .local_bounding_box = math.BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var inst0 = InstancedMesh{ .name = "i0", .source_mesh = &src, .position = Vec3.new(0, 0, 0) };
    var ptrs = [_]*InstancedMesh{&inst0};
    src.instances = .{ .items = ptrs[0..], .capacity = 1 };
    var plain = Mesh{
        .name = "plain",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0, 0, -4),
        .local_bounding_box = math.BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var list = [_]*Mesh{ &src, &plain };

    // Plain box spans z in [-5,-3], instance box spans [-1,1]: from z=-10 the
    // plain mesh is nearer and must win with a null instance index.
    const plain_wins = pickWithRay(&list, null, Ray.new(Vec3.new(0, 0, -10), Vec3.new(0, 0, 1)));
    try std.testing.expect(plain_wins.hit);
    try std.testing.expectEqual(&plain, plain_wins.picked_mesh.?);
    try std.testing.expectEqual(@as(?usize, null), plain_wins.picked_instance);

    // From z=-2 going +z the plain box [-5,-3] is behind the origin, so the
    // instance wins and its index is reported.
    const inst_wins = pickWithRay(&list, null, Ray.new(Vec3.new(0, 0, -2), Vec3.new(0, 0, 1)));
    try std.testing.expect(inst_wins.hit);
    try std.testing.expectEqual(&src, inst_wins.picked_mesh.?);
    try std.testing.expectEqual(@as(?usize, 0), inst_wins.picked_instance);

    // A lone plain mesh hit also reports null.
    var lone = [_]*Mesh{&plain};
    const lone_hit = pickWithRay(&lone, null, Ray.new(Vec3.new(0, 0, -10), Vec3.new(0, 0, 1)));
    try std.testing.expect(lone_hit.hit);
    try std.testing.expectEqual(@as(?usize, null), lone_hit.picked_instance);
}
