const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;

const NavMesh = @import("navmesh.zig").NavMesh;
const NavNode = @import("navmesh.zig").NavNode;
const funnel = @import("funnel.zig");
const Portal = funnel.Portal;
const Pathfinding = @import("pathfinding.zig").Pathfinding;
const NavAgent = @import("agent.zig").NavAgent;

fn refFindNode(nodes: []const NavNode, pt: Vec3) ?u32 {
    // Independent reference specification of findNode/findClosestNode semantics.
    for (nodes, 0..) |node, i| {
        if (node.containsPointXZ(pt)) {
            const y = node.getYAtXZ(pt);
            if (@abs(y - pt.y) < 3.0) return @intCast(i);
        }
    }
    return null;
}

fn refFindClosestNode(nodes: []const NavNode, pt: Vec3) ?u32 {
    if (refFindNode(nodes, pt)) |idx| return idx;
    if (nodes.len == 0) return null;
    var best_idx: u32 = 0;
    var best_dist_sq: f32 = std.math.inf(f32);
    for (nodes, 0..) |node, i| {
        const dx = node.centroid.x - pt.x;
        const dy = node.centroid.y - pt.y;
        const dz = node.centroid.z - pt.z;
        const dist_sq = dx * dx + dy * dy + dz * dz;
        if (dist_sq < best_dist_sq) {
            best_dist_sq = dist_sq;
            best_idx = @intCast(i);
        }
    }
    return best_idx;
}

fn flatTri(x0: f32, z0: f32, x1: f32, z1: f32, x2: f32, z2: f32, y: f32) NavNode {
    return .{
        .vertices = .{
            Vec3.new(x0, y, z0),
            Vec3.new(x1, y, z1),
            Vec3.new(x2, y, z2),
        },
        .centroid = Vec3.new(
            (x0 + x1 + x2) / 3.0,
            y,
            (z0 + z1 + z2) / 3.0,
        ),
        .normal = Vec3.new(0.0, 1.0, 0.0),
    };
}

test "NavMesh findNode: first-match parity on random grid queries" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var nav = try NavMesh.buildGrid(allocator, 0.0, 100.0, 0.0, 100.0, 0.0, 20, 20, &.{});
    defer nav.deinit();
    try testing.expect(nav.nodes.len == 800);

    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rand = prng.random();
    var i: usize = 0;
    while (i < 2000) : (i += 1) {
        const pt = Vec3.new(
            rand.float(f32) * 120.0 - 10.0,
            rand.float(f32) * 10.0 - 5.0,
            rand.float(f32) * 120.0 - 10.0,
        );
        try testing.expectEqual(refFindNode(nav.nodes, pt), nav.findNode(pt));
        try testing.expectEqual(refFindClosestNode(nav.nodes, pt), nav.findClosestNode(pt));
    }
}

test "NavMesh findNode: parity on negative-coordinate grid" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var nav = try NavMesh.buildGrid(allocator, -60.0, -10.0, -60.0, -10.0, 2.0, 12, 12, &.{});
    defer nav.deinit();

    var prng = std.Random.DefaultPrng.init(0x9E61);
    const rand = prng.random();
    var i: usize = 0;
    while (i < 500) : (i += 1) {
        const pt = Vec3.new(
            rand.float(f32) * 70.0 - 70.0,
            rand.float(f32) * 10.0 - 3.0,
            rand.float(f32) * 70.0 - 70.0,
        );
        try testing.expectEqual(refFindNode(nav.nodes, pt), nav.findNode(pt));
        try testing.expectEqual(refFindClosestNode(nav.nodes, pt), nav.findClosestNode(pt));
    }
}

test "NavMesh findNode: overlapping layers keep lowest qualifying index" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // 40 quads x 2 tris x 2 layers (y=0 and y=1) = 160 nodes, index active.
    // Layer 0 owns even pairs, layer 1 the same footprint shifted by +80.
    var list = std.ArrayList(NavNode).empty;
    defer list.deinit(allocator);
    var q: usize = 0;
    while (q < 40) : (q += 1) {
        const x0: f32 = @floatFromInt(q * 2);
        try list.append(allocator, flatTri(x0, 0, x0 + 2, 0, x0 + 2, 2, 0.0));
        try list.append(allocator, flatTri(x0, 0, x0 + 2, 2, x0, 2, 0.0));
        try list.append(allocator, flatTri(x0, 0, x0 + 2, 0, x0 + 2, 2, 1.0));
        try list.append(allocator, flatTri(x0, 0, x0 + 2, 2, x0, 2, 1.0));
    }
    const owned = try list.toOwnedSlice(allocator);
    var nav = NavMesh.init(allocator, owned, BoundingBox.zero);
    defer nav.deinit();
    try testing.expect(nav.nodes.len == 160);

    // Quad centers: both layers within <3m elevation -> first-layer index wins.
    var k: usize = 0;
    while (k < 40) : (k += 1) {
        const cx: f32 = @as(f32, @floatFromInt(k * 2)) + 1.0;
        const pt = Vec3.new(cx, 0.5, 1.0);
        try testing.expectEqual(refFindNode(nav.nodes, pt), nav.findNode(pt));
        try testing.expectEqual(@as(u32, @intCast(k * 4)), nav.findNode(pt).?);
        // y=2: layer 0 diff 2 accepted, still first.
        try testing.expectEqual(@as(u32, @intCast(k * 4)), nav.findNode(Vec3.new(cx, 2.0, 1.0)).?);
        // y=3.5: layer 0 diff 3.5 rejected, layer 1 diff 2.5 accepted.
        try testing.expectEqual(@as(u32, @intCast(k * 4 + 2)), nav.findNode(Vec3.new(cx, 3.5, 1.0)).?);
        // y=4.5: both rejected.
        try testing.expectEqual(@as(?u32, null), nav.findNode(Vec3.new(cx, 4.5, 1.0)));
    }

    // Degenerate node appended manually: parity must hold on the grown slice.
    const degen = flatTri(500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 0.0);
    const grown = try allocator.realloc(owned, owned.len + 1);
    grown[grown.len - 1] = degen;
    nav.nodes = grown;
    try testing.expectEqual(refFindNode(nav.nodes, Vec3.new(500.0, 0.0, 500.0)), nav.findNode(Vec3.new(500.0, 0.0, 500.0)));
    try testing.expectEqual(refFindClosestNode(nav.nodes, Vec3.new(500.0, 0.0, 500.0)), nav.findClosestNode(Vec3.new(500.0, 0.0, 500.0)));
}

test "NavMesh findNode: empty/degen/outside/fallback ties/non-finite" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // Empty mesh stays null everywhere.
    var empty = NavMesh.init(allocator, &.{}, BoundingBox.zero);
    defer empty.deinit();
    try testing.expectEqual(@as(?u32, null), empty.findNode(Vec3.zero));
    try testing.expectEqual(@as(?u32, null), empty.findClosestNode(Vec3.zero));

    // Symmetric pair: midpoint query misses both, centroid fallback ties -> index 0.
    const pair = try allocator.dupe(NavNode, &[_]NavNode{
        flatTri(-4.0, -1.0, -2.0, -1.0, -2.0, 1.0, 0.0),
        flatTri(2.0, -1.0, 4.0, -1.0, 4.0, 1.0, 0.0),
    });
    var nav = NavMesh.init(allocator, pair, BoundingBox.zero);
    defer nav.deinit();
    const mid = Vec3.new(0.0, 0.0, 0.0);
    try testing.expectEqual(@as(?u32, null), nav.findNode(mid));
    try testing.expectEqual(@as(?u32, 0), nav.findClosestNode(mid));

    // Far-outside point: findNode null, closest matches brute force.
    var grid = try NavMesh.buildGrid(allocator, 0.0, 40.0, 0.0, 40.0, 0.0, 10, 10, &.{});
    defer grid.deinit();
    const far = Vec3.new(500.0, 0.0, -300.0);
    try testing.expectEqual(@as(?u32, null), grid.findNode(far));
    try testing.expectEqual(refFindClosestNode(grid.nodes, far), grid.findClosestNode(far));

    // Non-finite inputs fail safe (no trap, legacy-equivalent results).
    const inf_pt = Vec3.new(std.math.inf(f32), 0.0, 0.0);
    try testing.expectEqual(refFindNode(grid.nodes, inf_pt), grid.findNode(inf_pt));
    try testing.expectEqual(refFindClosestNode(grid.nodes, inf_pt), grid.findClosestNode(inf_pt));
    const nan_pt = Vec3.new(std.math.nan(f32), 0.0, std.math.nan(f32));
    try testing.expectEqual(refFindNode(grid.nodes, nan_pt), grid.findNode(nan_pt));
    try testing.expectEqual(refFindClosestNode(grid.nodes, nan_pt), grid.findClosestNode(nan_pt));

    // Boundary points exactly on shared triangle edges stay deterministic.
    const edge_pt = Vec3.new(4.0, 0.0, 4.0);
    try testing.expectEqual(refFindNode(grid.nodes, edge_pt), grid.findNode(edge_pt));
}

test "NavMesh findNode: exact after in-place geometry mutation" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var nav = try NavMesh.buildGrid(allocator, 0.0, 100.0, 0.0, 100.0, 0.0, 20, 20, &.{});
    defer nav.deinit();

    // Drag node 5 far away in place. Parity must hold with no refresh call.
    for (&nav.nodes[5].vertices) |*v| {
        v.x += 1000.0;
        v.z += 1000.0;
    }
    nav.nodes[5].centroid.x += 1000.0;
    nav.nodes[5].centroid.z += 1000.0;

    var prng = std.Random.DefaultPrng.init(0xBEA7);
    const rand = prng.random();
    var i: usize = 0;
    while (i < 300) : (i += 1) {
        const pt = Vec3.new(
            rand.float(f32) * 120.0 - 10.0,
            rand.float(f32) * 6.0 - 3.0,
            rand.float(f32) * 120.0 - 10.0,
        );
        try testing.expectEqual(refFindNode(nav.nodes, pt), nav.findNode(pt));
        try testing.expectEqual(refFindClosestNode(nav.nodes, pt), nav.findClosestNode(pt));
    }
}

test "NavMesh findNode: parity at large coordinate offsets" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // +/-1e6 offsets, 1e6 spans: f32 ulp (~0.06) dwarfs the 1e-4 tolerances.
    // 10x10 subdivisions over a 1e6 span: all grid coordinates are f32-exact.
    var pos = try NavMesh.buildGrid(allocator, 1.0e6, 2.0e6, 1.0e6, 2.0e6, 0.0, 10, 10, &.{});
    defer pos.deinit();
    var neg = try NavMesh.buildGrid(allocator, -2.0e6, -1.0e6, -2.0e6, -1.0e6, 5.0, 10, 10, &.{});
    defer neg.deinit();

    var prng = std.Random.DefaultPrng.init(0xB16_0FF5);
    const rand = prng.random();
    var i: usize = 0;
    while (i < 500) : (i += 1) {
        const px = 1.0e6 + rand.float(f32) * 1.0e6;
        const pz = 1.0e6 + rand.float(f32) * 1.0e6;
        const pt = Vec3.new(px, rand.float(f32) * 6.0 - 3.0, pz);
        try testing.expectEqual(refFindNode(pos.nodes, pt), pos.findNode(pt));
        const npt = Vec3.new(-px, 5.0, -pz);
        try testing.expectEqual(refFindNode(neg.nodes, npt), neg.findNode(npt));
    }

    // Exact cell/quad boundaries (shared edges): lowest index must win.
    var k: u32 = 0;
    while (k <= 10) : (k += 1) {
        const e: f32 = 1.0e6 + @as(f32, @floatFromInt(k)) * 1.0e5;
        const pt = Vec3.new(e, 0.0, e);
        try testing.expectEqual(refFindNode(pos.nodes, pt), pos.findNode(pt));
    }
}

test "NavMesh findNode: same-slice geometry edit stays exact" {
    const testing = std.testing;
    const allocator = testing.allocator;

    // 10x10 grid over (0,0)-(50,50): 200 nodes, index active.
    var nav = try NavMesh.buildGrid(allocator, 0.0, 50.0, 0.0, 50.0, 0.0, 10, 10, &.{});
    defer nav.deinit();

    // Point deep in the last quad; answered by a high node index.
    const pt = Vec3.new(47.5, 0.0, 47.5);
    const before = nav.findNode(pt);
    try testing.expect(before != null and before.? != 0);

    // Public in-place mutation on the SAME slice (ptr+len unchanged):
    // drag node 0 (a non-candidate for pt's cell) so it covers pt.
    // The historical mutable contract holds automatically;
    // the earlier index must now win.
    nav.nodes[0].vertices = .{
        Vec3.new(46.0, 0.0, 46.0),
        Vec3.new(49.0, 0.0, 46.0),
        Vec3.new(47.5, 0.0, 49.0),
    };
    nav.nodes[0].centroid = Vec3.new(47.5, 0.0, 47.33);
    try testing.expectEqual(@as(u32, 0), nav.findNode(pt).?);
    try testing.expectEqual(refFindNode(nav.nodes, pt), nav.findNode(pt));
    try testing.expectEqual(refFindClosestNode(nav.nodes, pt), nav.findClosestNode(pt));
}

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

/// Test-only counting allocator: forwards everything to the backing
/// allocator while counting allocs vs frees. Catches leaks the backing
/// GPA silently ignores — notably zero-length allocations, which is exactly
/// what `findPath` returns on its empty-result early-outs.
const CountAlloc = struct {
    backing: std.mem.Allocator,
    allocs: usize = 0,
    frees: usize = 0,

    fn init(backing: std.mem.Allocator) CountAlloc {
        return .{ .backing = backing };
    }

    fn allocator(self: *CountAlloc) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = allocFn,
                .resize = resizeFn,
                .remap = remapFn,
                .free = freeFn,
            },
        };
    }

    fn allocFn(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *CountAlloc = @ptrCast(@alignCast(ctx));
        const p = self.backing.rawAlloc(len, alignment, ra) orelse return null;
        self.allocs += 1;
        return p;
    }

    fn resizeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) bool {
        const self: *CountAlloc = @ptrCast(@alignCast(ctx));
        return self.backing.rawResize(memory, alignment, new_len, ra);
    }

    fn remapFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ra: usize) ?[*]u8 {
        const self: *CountAlloc = @ptrCast(@alignCast(ctx));
        const res = self.backing.rawRemap(memory, alignment, new_len, ra) orelse return null;
        // A successful remap conceptually frees the old backing and
        // allocates the new one: count both sides so the balance holds no
        // matter which growth path (alloc vs remap) ArrayList takes.
        self.allocs += 1;
        self.frees += 1;
        return res;
    }

    fn freeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ra: usize) void {
        const self: *CountAlloc = @ptrCast(@alignCast(ctx));
        self.frees += 1;
        self.backing.rawFree(memory, alignment, ra);
    }
};

test "NavAgent.setDestination on an empty mesh frees the empty path (no leak)" {
    const testing = std.testing;
    // Real caller path: scene/nav_layer.zig createAgent is allocator.create
    // + NavAgent.init + append, so this drives the exact Scene wiring
    // (Scene.createNavAgent) with the layer, not just the struct.
    const NavLayer = @import("../scene/nav_layer.zig").NavLayer;
    const Crowd = @import("crowd.zig").Crowd;
    const CrowdAgentParams = @import("crowd.zig").CrowdAgentParams;

    // Empty mesh: findClosestNode finds nothing, so findPath returns its
    // len-0 early-out on EVERY call. Per std semantics those are comptime
    // sentinels that never reach the vtable (and free() on len 0 is a
    // no-op), so the 64 failed queries must leave the counter exactly
    // balanced; any heap-backed empty dropped without freeing would show up
    // here as allocs > frees.
    var counter = CountAlloc.init(testing.allocator);
    const alloc = counter.allocator();

    // NOT counter-owned: NavMesh.init stores the slice as-is and deinit
    // frees it, but a len-0 free is a std no-op that would never reach the
    // counter — routing it through the counter would imbalance the test's
    // own bookkeeping (alloc counted, free elided).
    const empty_nodes = try testing.allocator.alloc(NavNode, 0);
    var nav = NavMesh.init(alloc, empty_nodes, BoundingBox.zero);
    defer nav.deinit();

    var layer = NavLayer{};
    const agent = try layer.createAgent(alloc, &nav, Vec3.new(0, 0, 0));
    var i: usize = 0;
    while (i < 64) : (i += 1) {
        try testing.expect(!try agent.setDestination(Vec3.new(100, 0, 100)));
    }
    try testing.expect(agent.arrived);
    try testing.expect(agent.target_pos == null);
    layer.deinit(alloc);

    // Same empty-result ownership in Crowd.setAgentDestination.
    var crowd = Crowd.init(alloc, &nav);
    const id = try crowd.addAgent(Vec3.new(0, 0, 0), CrowdAgentParams{});
    var j: usize = 0;
    while (j < 64) : (j += 1) {
        try testing.expect(!try crowd.setAgentDestination(id, Vec3.new(100, 0, 100)));
    }
    // Explicit teardown BEFORE the balance check (a defer would run past it).
    crowd.deinit();

    try testing.expectEqual(counter.allocs, counter.frees);
}
