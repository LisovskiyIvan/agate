const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const crowd_mod = @import("crowd.zig");
const Crowd = crowd_mod.Crowd;

test "Crowd basic agent addition, pathfinding, and arrival" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var crowd = Crowd.init(allocator, null);
    defer crowd.deinit();

    const id1 = try crowd.addAgent(Vec3.new(0, 0, 0), .{ .max_speed = 4.0, .stopping_distance = 0.2 });
    try testing.expectEqual(@as(usize, 1), crowd.agentCount());

    const agent1 = crowd.getAgent(id1).?;
    try testing.expect(agent1.arrived);

    _ = try crowd.setAgentDestination(id1, Vec3.new(10, 0, 0));
    try testing.expect(!agent1.arrived);

    // Simulate 4 seconds
    var t: f32 = 0.0;
    while (t < 4.0) : (t += 0.1) {
        crowd.update(0.1);
    }

    // Should have arrived near (10, 0, 0)
    try testing.expect(agent1.arrived);
    try testing.expectApproxEqAbs(@as(f32, 10.0), agent1.position.x, 0.5);
    try testing.expectApproxEqAbs(@as(f32, 0.0), agent1.position.z, 0.5);
}

test "Crowd reciprocal collision avoidance (two agents moving head-on)" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var crowd = Crowd.init(allocator, null);
    defer crowd.deinit();

    // Agent 1 starts at (-5, 0, 0) and moves toward (5, 0, 0)
    const id1 = try crowd.addAgent(Vec3.new(-5, 0, 0), .{
        .radius = 0.5,
        .max_speed = 2.0,
        .stopping_distance = 0.2,
    });
    _ = try crowd.setAgentDestination(id1, Vec3.new(5, 0, 0));

    // Agent 2 starts at (5, 0, 0) and moves toward (-5, 0, 0)
    const id2 = try crowd.addAgent(Vec3.new(5, 0, 0), .{
        .radius = 0.5,
        .max_speed = 2.0,
        .stopping_distance = 0.2,
    });
    _ = try crowd.setAgentDestination(id2, Vec3.new(-5, 0, 0));

    const a1 = crowd.getAgent(id1).?;
    const a2 = crowd.getAgent(id2).?;

    // Simulate agents passing each other
    var min_dist: f32 = 100.0;
    var t: f32 = 0.0;
    while (t < 6.0) : (t += 0.05) {
        crowd.update(0.05);

        const dx = a1.position.x - a2.position.x;
        const dz = a1.position.z - a2.position.z;
        const dist = @sqrt(dx * dx + dz * dz);
        if (dist < min_dist) min_dist = dist;
    }

    try testing.expect(min_dist > 0.35);
    try testing.expect(a1.position.x > 3.0);
    try testing.expect(a2.position.x < -3.0);
}
