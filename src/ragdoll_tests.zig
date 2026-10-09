const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const physics = @import("physics.zig");
const PhysicsWorld = physics.PhysicsWorld;
const ragdoll_mod = @import("ragdoll.zig");
const Ragdoll = ragdoll_mod.Ragdoll;
const part_count = ragdoll_mod.part_count;
const joint_count = ragdoll_mod.joint_count;

fn isFiniteVec(v: Vec3) bool {
    return std.math.isFinite(v.x) and std.math.isFinite(v.y) and std.math.isFinite(v.z);
}

test "Ragdoll init creates bodies and joints" {
    var world = PhysicsWorld.init(std.testing.allocator);
    defer world.deinit();

    var doll = try Ragdoll.init(std.testing.allocator, &world, null, .{});
    defer doll.deinit(&world);

    try std.testing.expectEqual(part_count, doll.bodies.items.len);
    try std.testing.expectEqual(joint_count, doll.joints.items.len);
    try std.testing.expectEqual(part_count, doll.meshes.items.len);
    for (doll.joints.items) |jid| {
        try std.testing.expect(world.isJointValid(jid));
    }
    // Spot-check masses against defaults.
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), doll.getPart(.pelvis).mass, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), doll.getPart(.chest).mass, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), doll.getPart(.head).mass, 1e-6);
}

test "Ragdoll stays finite after steps" {
    var world = PhysicsWorld.init(std.testing.allocator);
    defer world.deinit();

    var doll = try Ragdoll.init(std.testing.allocator, &world, null, .{});
    defer doll.deinit(&world);

    for (0..120) |_| world.step(0.016);
    for (doll.bodies.items) |body| {
        try std.testing.expect(isFiniteVec(body.mesh.position));
        try std.testing.expect(isFiniteVec(body.mesh.rotation));
        try std.testing.expect(isFiniteVec(body.velocity));
    }
}

test "Ragdoll applyImpulse moves bodies" {
    var world = PhysicsWorld.init(std.testing.allocator);
    defer world.deinit();

    var doll = try Ragdoll.init(std.testing.allocator, &world, null, .{});
    defer doll.deinit(&world);

    const before = doll.getPart(.pelvis).mesh.position;
    doll.applyImpulse(Vec3.new(20.0, 5.0, 0.0));
    for (0..10) |_| world.step(0.016);
    const after = doll.getPart(.pelvis).mesh.position;
    try std.testing.expect(before.sub(after).length() > 0.1);
}

test "Ragdoll reset restores spawn pose" {
    var world = PhysicsWorld.init(std.testing.allocator);
    defer world.deinit();

    var doll = try Ragdoll.init(std.testing.allocator, &world, null, .{});
    defer doll.deinit(&world);

    for (0..90) |_| world.step(0.016);
    doll.reset();
    for (doll.bodies.items, 0..) |body, i| {
        try std.testing.expectEqual(doll.home_positions.items[i], body.mesh.position);
        try std.testing.expectEqual(doll.home_rotations.items[i], body.mesh.rotation);
        try std.testing.expectEqual(Vec3.zero, body.velocity);
        try std.testing.expectEqual(Vec3.zero, body.angular_velocity);
    }
    // The restored pose must survive stepping (solver picked up the teleport).
    for (0..5) |_| world.step(0.016);
    for (doll.bodies.items, 0..) |body, i| {
        try std.testing.expect(body.mesh.position.sub(doll.home_positions.items[i]).length() < 0.1);
    }
}

test "Ragdoll syncMeshes copies body transforms" {
    var world = PhysicsWorld.init(std.testing.allocator);
    defer world.deinit();

    var doll = try Ragdoll.init(std.testing.allocator, &world, null, .{});
    defer doll.deinit(&world);

    for (0..30) |_| world.step(0.016);
    // Corrupt the mesh transforms, then restore them from the bodies.
    for (doll.bodies.items) |body| {
        body.mesh.position = Vec3.new(1234.0, 5678.0, 91011.0);
    }
    doll.syncMeshes();
    for (doll.bodies.items) |body| {
        try std.testing.expectEqual(body.last_pos, body.mesh.position);
        try std.testing.expectEqual(body.last_rot, body.mesh.rotation);
    }
}
