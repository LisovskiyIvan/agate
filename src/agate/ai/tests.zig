const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;

const NavMesh = @import("navmesh.zig").NavMesh;
const funnel = @import("funnel.zig");
const Portal = funnel.Portal;
const Pathfinding = @import("pathfinding.zig").Pathfinding;
const NavAgent = @import("agent.zig").NavAgent;

test "NavMesh: build from simple 2-triangle quad with edge connection" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Two triangles forming a 10x10 quad on the XZ plane:
    // v0=(0,0,0), v1=(10,0,0), v2=(10,0,10), v3=(0,0,10)
    // Tri 0: 0, 1, 2
    // Tri 1: 0, 2, 3
    // Shared edge: (0, 2)
    const positions = [_][3]f32{
        .{ 0.0, 0.0, 0.0 },
        .{ 10.0, 0.0, 0.0 },
        .{ 10.0, 0.0, 10.0 },
        .{ 0.0, 0.0, 10.0 },
    };
    const indices = [_]u32{
        0, 1, 2,
        0, 2, 3,
    };

    var nav = try NavMesh.buildFromTriangles(allocator, &positions, &indices, std.math.pi * 0.4);
    defer nav.deinit();

    try testing.expectEqual(@as(usize, 2), nav.nodes.len);

    // Nodes must be adjacent
    var has_neighbor0 = false;
    for (nav.nodes[0].neighbors) |nbr| {
        if (nbr != null and nbr.? == 1) has_neighbor0 = true;
    }
    try testing.expect(has_neighbor0);

    var has_neighbor1 = false;
    for (nav.nodes[1].neighbors) |nbr| {
        if (nbr != null and nbr.? == 0) has_neighbor1 = true;
    }
    try testing.expect(has_neighbor1);

    // Point in triangle test
    const node_idx = nav.findNode(Vec3.new(2.0, 0.0, 8.0));
    try testing.expect(node_idx != null);
    try testing.expectEqual(@as(u32, 1), node_idx.?);
}

test "NavMesh: filter out steep vertical walls" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Triangle 0: flat horizontal floor
    // Triangle 1: vertical wall (normal pointing horizontally)
    const positions = [_][3]f32{
        // Floor
        .{ 0.0, 0.0, 0.0 },
        .{ 5.0, 0.0, 0.0 },
        .{ 5.0, 0.0, 5.0 },
        // Wall
        .{ 0.0, 0.0, 0.0 },
        .{ 0.0, 5.0, 0.0 },
        .{ 5.0, 5.0, 0.0 },
    };
    const indices = [_]u32{
        0, 1, 2,
        3, 4, 5,
    };

    var nav = try NavMesh.buildFromTriangles(allocator, &positions, &indices, std.math.pi * 0.25);
    defer nav.deinit();

    // Only the horizontal floor should be included
    try testing.expectEqual(@as(usize, 1), nav.nodes.len);
}

test "NavMesh: build grid with obstacle" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // 4x4 grid from (0,0) to (40,40)
    // Center box obstacle from (15, -1, 15) to (25, 2, 25)
    const obstacle = BoundingBox.init(
        Vec3.new(15.0, -1.0, 15.0),
        Vec3.new(25.0, 2.0, 25.0),
    );

    var nav = try NavMesh.buildGrid(
        allocator,
        0.0,
        40.0,
        0.0,
        40.0,
        0.0,
        4,
        4,
        &.{obstacle},
    );
    defer nav.deinit();

    // 4x4 quads = 16 quads total. The 4 central quads [10..20, 20..30] intersect the obstacle.
    // So fewer than 32 triangles should be generated.
    try testing.expect(nav.nodes.len > 0);
    try testing.expect(nav.nodes.len < 32);

    // Finding node in the center should be null (blocked)
    const center_node = nav.findNode(Vec3.new(20.0, 0.0, 20.0));
    try testing.expect(center_node == null);

    // Finding node in corner (5, 5) should succeed
    const corner_node = nav.findNode(Vec3.new(5.0, 0.0, 5.0));
    try testing.expect(corner_node != null);
}

test "Funnel: string pulling around L-shaped corner" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Start at (0, 0, 0)
    // Portal 0: left=(2, 0, 1), right=(2, 0, -1)
    // Portal 1 (corner): left=(5, 0, 5), right=(3, 0, 1)
    // Portal 2: left=(5, 0, 10), right=(3, 0, 10)
    // End at (4, 0, 12)
    const portals = [_]Portal{
        .{ .left = Vec3.new(2.0, 0.0, 1.0), .right = Vec3.new(2.0, 0.0, -1.0) },
        .{ .left = Vec3.new(5.0, 0.0, 5.0), .right = Vec3.new(3.0, 0.0, 1.0) },
        .{ .left = Vec3.new(5.0, 0.0, 10.0), .right = Vec3.new(3.0, 0.0, 10.0) },
    };

    const start = Vec3.new(0.0, 0.0, 0.0);
    const end = Vec3.new(4.0, 0.0, 12.0);

    const path = try funnel.stringPull(allocator, start, end, &portals);
    defer allocator.free(path);

    try testing.expect(path.len >= 2);
    try testing.expectEqual(start, path[0]);
    try testing.expectEqual(end, path[path.len - 1]);
}

test "Pathfinding: A* around obstacle in NavMesh grid" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // 5x5 grid from (0,0) to (50,50)
    // Obstacle blocking the center from (18, -1, 10) to (32, 2, 40)
    const obstacle = BoundingBox.init(
        Vec3.new(18.0, -1.0, 10.0),
        Vec3.new(32.0, 2.0, 40.0),
    );

    var nav = try NavMesh.buildGrid(
        allocator,
        0.0,
        50.0,
        0.0,
        50.0,
        0.0,
        5,
        5,
        &.{obstacle},
    );
    defer nav.deinit();

    const start_pos = Vec3.new(5.0, 0.0, 25.0);
    const end_pos = Vec3.new(45.0, 0.0, 25.0);

    const path = try Pathfinding.findPath(&nav, start_pos, end_pos, allocator);
    defer allocator.free(path);

    try testing.expect(path.len >= 2);
    try testing.expectEqual(start_pos, path[0]);
    try testing.expectEqual(end_pos, path[path.len - 1]);

    // Ensure path goes around the obstacle and not through its X coordinates [18..32] at Z=25
    var went_around = false;
    for (path) |wp| {
        if (wp.z < 12.0 or wp.z > 38.0) {
            went_around = true;
        }
    }
    try testing.expect(went_around);
}

test "NavAgent: set destination and advance along path" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const positions = [_][3]f32{
        .{ 0.0, 0.0, 0.0 },
        .{ 20.0, 0.0, 0.0 },
        .{ 20.0, 0.0, 20.0 },
        .{ 0.0, 0.0, 20.0 },
    };
    const indices = [_]u32{
        0, 1, 2,
        0, 2, 3,
    };

    var nav = try NavMesh.buildFromTriangles(allocator, &positions, &indices, std.math.pi * 0.4);
    defer nav.deinit();

    var agent = NavAgent.init(allocator, &nav, Vec3.new(2.0, 0.0, 2.0));
    defer agent.deinit();

    agent.speed = 10.0;
    agent.stopping_distance = 0.2;

    const success = try agent.setDestination(Vec3.new(18.0, 0.0, 18.0));
    try testing.expect(success);
    try testing.expect(!agent.arrived);

    // Step agent forward in time
    var total_time: f32 = 0.0;
    while (!agent.arrived and total_time < 5.0) {
        agent.update(0.1);
        total_time += 0.1;
    }

    try testing.expect(agent.arrived);
    const dist_to_target = agent.position.sub(Vec3.new(18.0, 0.0, 18.0)).length();
    try testing.expect(dist_to_target < 0.5);
}

test "NavMesh: empty mesh pathfinding handled gracefully" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var nav = NavMesh.init(allocator, &.{}, BoundingBox.zero);
    defer nav.deinit();

    const path = try Pathfinding.findPath(&nav, Vec3.zero, Vec3.one, allocator);
    defer allocator.free(path);

    try testing.expectEqual(@as(usize, 0), path.len);
}

test "Funnel: straight path with collinear portals produces 2 points" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const portals = [_]Portal{
        .{ .left = Vec3.new(-2.0, 0.0, 5.0), .right = Vec3.new(2.0, 0.0, 5.0) },
        .{ .left = Vec3.new(-2.0, 0.0, 10.0), .right = Vec3.new(2.0, 0.0, 10.0) },
        .{ .left = Vec3.new(-2.0, 0.0, 15.0), .right = Vec3.new(2.0, 0.0, 15.0) },
    };

    const start = Vec3.new(0.0, 0.0, 0.0);
    const end = Vec3.new(0.0, 0.0, 20.0);

    const path = try funnel.stringPull(allocator, start, end, &portals);
    defer allocator.free(path);

    // Completely straight line through open wide portals: direct line from start to end!
    try testing.expectEqual(@as(usize, 2), path.len);
    try testing.expectEqual(start, path[0]);
    try testing.expectEqual(end, path[1]);
}

test "NavAgent: teleport and stop" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const positions = [_][3]f32{
        .{ 0.0, 0.0, 0.0 },
        .{ 10.0, 0.0, 0.0 },
        .{ 10.0, 0.0, 10.0 },
        .{ 0.0, 0.0, 10.0 },
    };
    const indices = [_]u32{
        0, 1, 2,
        0, 2, 3,
    };

    var nav = try NavMesh.buildFromTriangles(allocator, &positions, &indices, std.math.pi * 0.4);
    defer nav.deinit();

    var agent = NavAgent.init(allocator, &nav, Vec3.new(1.0, 0.0, 1.0));
    defer agent.deinit();

    _ = try agent.setDestination(Vec3.new(9.0, 0.0, 9.0));
    try testing.expect(!agent.arrived);

    agent.stop();
    try testing.expect(agent.arrived);
    try testing.expectEqual(Vec3.zero, agent.velocity);

    agent.teleport(Vec3.new(4.0, 0.0, 4.0));
    try testing.expectEqual(@as(f32, 4.0), agent.position.x);
    try testing.expectEqual(@as(f32, 4.0), agent.position.z);
}

test "NavAgent: smooth acceleration, cornering, and arrival" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const positions = [_][3]f32{
        .{ 0.0, 0.0, 0.0 },
        .{ 20.0, 0.0, 0.0 },
        .{ 20.0, 0.0, 20.0 },
        .{ 0.0, 0.0, 20.0 },
    };
    const indices = [_]u32{
        0, 1, 2,
        0, 2, 3,
    };

    var nav = try NavMesh.buildFromTriangles(allocator, &positions, &indices, std.math.pi * 0.4);
    defer nav.deinit();

    var agent = NavAgent.init(allocator, &nav, Vec3.new(2.0, 0.0, 2.0));
    defer agent.deinit();

    agent.speed = 4.0;
    agent.acceleration = 6.0;
    agent.waypoint_radius = 0.8;
    agent.slowdown_distance = 1.5;

    _ = try agent.setDestination(Vec3.new(10.0, 0.0, 2.0));

    // After a very small step (1 frame at 60 FPS), velocity should ramp up smoothly, not jump instantly to 4.0
    agent.update(0.016);
    const initial_speed = agent.velocity.length();
    try testing.expect(initial_speed > 0.0);
    try testing.expect(initial_speed < 1.0);

    // Step forward until arrival
    var steps: usize = 0;
    while (!agent.arrived and steps < 400) : (steps += 1) {
        agent.update(0.016);
    }
    try testing.expect(agent.arrived);
    try testing.expect(agent.position.sub(Vec3.new(10.0, 0.0, 2.0)).length() < 0.25);
}
