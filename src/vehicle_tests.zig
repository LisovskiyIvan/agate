const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const physics = @import("physics.zig");
const PhysicsWorld = physics.PhysicsWorld;
const vehicle_mod = @import("vehicle.zig");
const RaycastVehicle = vehicle_mod.RaycastVehicle;
const wheel_count = vehicle_mod.wheel_count;

fn isFiniteVec(v: Vec3) bool {
    return std.math.isFinite(v.x) and std.math.isFinite(v.y) and std.math.isFinite(v.z);
}

test "RaycastVehicle init creates chassis, wheels and joints" {
    var world = PhysicsWorld.init(std.testing.allocator);
    defer world.deinit();

    var car = try RaycastVehicle.init(std.testing.allocator, &world, null, .{});
    defer car.deinit(&world);

    // 1 chassis + 4 wheels, 4 wheel joints, 5 meshes.
    try std.testing.expectEqual(@as(usize, 5), car.bodies.items.len);
    try std.testing.expectEqual(@as(usize, 4), car.joints.items.len);
    try std.testing.expectEqual(@as(usize, 5), car.meshes.items.len);
    for (car.joints.items) |jid| {
        try std.testing.expect(world.isJointValid(jid));
    }
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), car.chassisBody().mass, 1e-6);
    for (0..wheel_count) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.8), car.wheelBody(i).mass, 1e-6);
    }
}

test "RaycastVehicle steps finite and throttle spins wheels" {
    var world = PhysicsWorld.init(std.testing.allocator);
    defer world.deinit();

    var car = try RaycastVehicle.init(std.testing.allocator, &world, null, .{
        .position = Vec3.new(0.0, 2.0, 0.0),
    });
    defer car.deinit(&world);

    car.setThrottle(1.0);
    car.setSteering(0.5);
    for (0..60) |_| {
        car.update(0.016);
        world.step(0.016);
    }
    for (car.bodies.items) |body| {
        try std.testing.expect(isFiniteVec(body.mesh.position));
        try std.testing.expect(isFiniteVec(body.mesh.rotation));
    }
    // Drive motor must have spun the wheels up.
    for (0..wheel_count) |i| {
        const speed = world.wheelSpinSpeed(car.wheelJoint(i));
        try std.testing.expect(std.math.isFinite(speed));
        try std.testing.expect(@abs(speed) > 0.5);
    }
    // Front wheels (offsets 1 and 3 have +X) must have steered.
    for ([2]usize{ 1, 3 }) |i| {
        try std.testing.expect(world.wheelSteeringAngle(car.wheelJoint(i)) > 0.1);
    }
}

test "RaycastVehicle brake slows the wheels" {
    var world = PhysicsWorld.init(std.testing.allocator);
    defer world.deinit();

    var car = try RaycastVehicle.init(std.testing.allocator, &world, null, .{
        .position = Vec3.new(0.0, 2.0, 0.0),
    });
    defer car.deinit(&world);

    car.setThrottle(1.0);
    for (0..60) |_| {
        car.update(0.016);
        world.step(0.016);
    }
    var rolling: f32 = 0.0;
    for (0..wheel_count) |i| rolling += @abs(world.wheelSpinSpeed(car.wheelJoint(i)));

    car.setThrottle(0.0);
    car.setBrake(1.0);
    for (0..60) |_| {
        car.update(0.016);
        world.step(0.016);
    }
    var braked: f32 = 0.0;
    for (0..wheel_count) |i| braked += @abs(world.wheelSpinSpeed(car.wheelJoint(i)));
    try std.testing.expect(braked < rolling);
}

test "RaycastVehicle syncMeshes copies body transforms" {
    var world = PhysicsWorld.init(std.testing.allocator);
    defer world.deinit();

    var car = try RaycastVehicle.init(std.testing.allocator, &world, null, .{});
    defer car.deinit(&world);

    for (0..30) |_| world.step(0.016);
    for (car.bodies.items) |body| {
        body.mesh.position = Vec3.new(-777.0, -777.0, -777.0);
    }
    car.syncMeshes();
    for (car.bodies.items) |body| {
        try std.testing.expectEqual(body.last_pos, body.mesh.position);
        try std.testing.expectEqual(body.last_rot, body.mesh.rotation);
    }
}
