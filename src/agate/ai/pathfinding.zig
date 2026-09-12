const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const NavMesh = @import("navmesh.zig").NavMesh;
const funnel = @import("funnel.zig");
const Portal = funnel.Portal;
const triArea2D = funnel.triArea2D;

pub const Pathfinding = struct {
    const AStarNode = struct {
        node_idx: u32,
        f_cost: f32,
    };

    fn compareNodes(context: void, a: AStarNode, b: AStarNode) std.math.Order {
        _ = context;
        if (a.f_cost < b.f_cost) return .lt;
        if (a.f_cost > b.f_cost) return .gt;
        return .eq;
    }

    const PriorityQueue = std.PriorityQueue(AStarNode, void, compareNodes);

    /// Computes the shortest smoothed 3D navigation path between start_pos and end_pos.
    /// Returns a list of 3D waypoints through the NavMesh.
    pub fn findPath(
        navmesh: *const NavMesh,
        start_pos: Vec3,
        end_pos: Vec3,
        allocator: std.mem.Allocator,
    ) ![]Vec3 {
        if (navmesh.nodes.len == 0) return try allocator.alloc(Vec3, 0);

        const start_node_opt = navmesh.findClosestNode(start_pos);
        const end_node_opt = navmesh.findClosestNode(end_pos);

        if (start_node_opt == null or end_node_opt == null) {
            return try allocator.alloc(Vec3, 0);
        }

        const start_node_idx = start_node_opt.?;
        const end_node_idx = end_node_opt.?;

        // Fast path: start and end are in the same triangle
        if (start_node_idx == end_node_idx) {
            const waypoints = try allocator.alloc(Vec3, 2);
            waypoints[0] = start_pos;
            waypoints[1] = end_pos;
            return waypoints;
        }

        const num_nodes = navmesh.nodes.len;

        // A* search allocations
        const g_costs = try allocator.alloc(f32, num_nodes);
        defer allocator.free(g_costs);
        @memset(g_costs, std.math.inf(f32));

        const came_from = try allocator.alloc(?u32, num_nodes);
        defer allocator.free(came_from);
        @memset(came_from, null);

        const closed = try allocator.alloc(bool, num_nodes);
        defer allocator.free(closed);
        @memset(closed, false);

        var open_set: PriorityQueue = .empty;
        defer open_set.deinit(allocator);

        g_costs[start_node_idx] = 0.0;
        const initial_h = navmesh.nodes[start_node_idx].centroid.sub(end_pos).length();
        try open_set.push(allocator, .{ .node_idx = start_node_idx, .f_cost = initial_h });

        var reached_goal = false;

        while (open_set.pop()) |current| {
            const curr_idx = current.node_idx;

            if (curr_idx == end_node_idx) {
                reached_goal = true;
                break;
            }

            if (closed[curr_idx]) continue;
            closed[curr_idx] = true;

            const curr_node = navmesh.nodes[curr_idx];
            const curr_g = g_costs[curr_idx];

            for (curr_node.neighbors) |neighbor_opt| {
                if (neighbor_opt) |nbr_idx| {
                    if (closed[nbr_idx]) continue;

                    const nbr_node = navmesh.nodes[nbr_idx];
                    const edge_dist = curr_node.centroid.sub(nbr_node.centroid).length();
                    const tentative_g = curr_g + edge_dist * nbr_node.cost;

                    if (tentative_g < g_costs[nbr_idx]) {
                        g_costs[nbr_idx] = tentative_g;
                        came_from[nbr_idx] = curr_idx;
                        const h = nbr_node.centroid.sub(end_pos).length();
                        try open_set.push(allocator, .{
                            .node_idx = nbr_idx,
                            .f_cost = tentative_g + h,
                        });
                    }
                }
            }
        }

        if (!reached_goal) {
            // No corridor exists between start and goal
            const waypoints = try allocator.alloc(Vec3, 1);
            waypoints[0] = start_pos;
            return waypoints;
        }

        // Reconstruct corridor of node indices
        var corridor_rev = std.ArrayList(u32).empty;
        defer corridor_rev.deinit(allocator);

        var curr_trace: ?u32 = end_node_idx;
        while (curr_trace) |idx| {
            try corridor_rev.append(allocator, idx);
            if (idx == start_node_idx) break;
            curr_trace = came_from[idx];
        }

        // Reverse corridor to [start_node ... end_node]
        const corridor_len = corridor_rev.items.len;
        const corridor = try allocator.alloc(u32, corridor_len);
        defer allocator.free(corridor);
        for (0..corridor_len) |ci| {
            corridor[ci] = corridor_rev.items[corridor_len - 1 - ci];
        }

        // Build portals sequence between consecutive nodes in corridor
        var portals = std.ArrayList(Portal).empty;
        defer portals.deinit(allocator);

        for (0..corridor_len - 1) |step_i| {
            const from_idx = corridor[step_i];
            const to_idx = corridor[step_i + 1];
            const from_node = navmesh.nodes[from_idx];
            const to_node = navmesh.nodes[to_idx];

            // Find shared edge
            var shared_p0: ?Vec3 = null;
            var shared_p1: ?Vec3 = null;

            for (0..3) |ei| {
                if (from_node.neighbors[ei] == to_idx) {
                    const edge = from_node.getEdge(ei);
                    shared_p0 = edge[0];
                    shared_p1 = edge[1];
                    break;
                }
            }

            if (shared_p0 != null and shared_p1 != null) {
                const p0 = shared_p0.?;
                const p1 = shared_p1.?;

                // Determine Left vs Right portal endpoint with respect to travel direction (from -> to)
                const side = triArea2D(from_node.centroid, to_node.centroid, p0);
                if (side > 0.0) {
                    try portals.append(allocator, .{ .left = p0, .right = p1 });
                } else {
                    try portals.append(allocator, .{ .left = p1, .right = p0 });
                }
            }
        }

        // Final goal portal (both left and right point to end_pos)
        try portals.append(allocator, .{ .left = end_pos, .right = end_pos });

        // String pulling via Funnel algorithm
        return funnel.stringPull(allocator, start_pos, end_pos, portals.items);
    }
};
