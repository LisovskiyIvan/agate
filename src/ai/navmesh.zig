const std = @import("std");
const math = @import("math");
const Vec2 = math.Vec2;
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;

pub const NavNode = struct {
    /// 3D vertices forming the triangle face (in CCW order viewed from above).
    vertices: [3]Vec3,
    /// Center of mass of the triangle.
    centroid: Vec3,
    /// Face normal.
    normal: Vec3,
    /// Indices of neighbor nodes sharing edges [0]->v0v1, [1]->v1v2, [2]->v2v0.
    /// null if the edge is an exterior boundary wall.
    neighbors: [3]?u32 = .{ null, null, null },
    /// Traversal cost multiplier (1.0 = normal surface).
    cost: f32 = 1.0,
    /// Custom user/engine flags (e.g. jumpable, water, hazard).
    flags: u32 = 0,

    pub fn getEdge(self: NavNode, edge_index: usize) [2]Vec3 {
        const edge_i0 = edge_index % 3;
        const edge_i1 = (edge_index + 1) % 3;
        return .{ self.vertices[edge_i0], self.vertices[edge_i1] };
    }

    inline fn edgeCross(p0: Vec2, p1: Vec2, p: Vec2) f32 {
        return (p1.x - p0.x) * (p.y - p0.y) - (p1.y - p0.y) * (p.x - p0.x);
    }

    /// Tests if a 2D horizontal point (XZ) lies inside this triangle using 2D cross products.
    pub fn containsPointXZ(self: NavNode, pt: Vec3) bool {
        // Fast 2D AABB early-out rejection (4 simple comparisons)
        const min_x = @min(self.vertices[0].x, @min(self.vertices[1].x, self.vertices[2].x));
        const max_x = @max(self.vertices[0].x, @max(self.vertices[1].x, self.vertices[2].x));
        if (pt.x < min_x - 1e-4 or pt.x > max_x + 1e-4) return false;

        const min_z = @min(self.vertices[0].z, @min(self.vertices[1].z, self.vertices[2].z));
        const max_z = @max(self.vertices[0].z, @max(self.vertices[1].z, self.vertices[2].z));
        if (pt.z < min_z - 1e-4 or pt.z > max_z + 1e-4) return false;

        const p = Vec2.new(pt.x, pt.z);
        const a = Vec2.new(self.vertices[0].x, self.vertices[0].z);
        const b = Vec2.new(self.vertices[1].x, self.vertices[1].z);
        const c = Vec2.new(self.vertices[2].x, self.vertices[2].z);

        const d1 = edgeCross(a, b, p);
        const d2 = edgeCross(b, c, p);
        const d3 = edgeCross(c, a, p);

        const has_neg = (d1 < -1e-4) or (d2 < -1e-4) or (d3 < -1e-4);
        const has_pos = (d1 > 1e-4) or (d2 > 1e-4) or (d3 > 1e-4);

        return !(has_neg and has_pos);
    }

    /// Interpolates the 3D elevation (Y) on the triangle plane at (pt.x, pt.z).
    pub fn getYAtXZ(self: NavNode, pt: Vec3) f32 {
        if (@abs(self.normal.y) < 1e-6) return self.centroid.y;
        // Plane equation: n.x*(x - a.x) + n.y*(y - a.y) + n.z*(z - a.z) = 0
        // y = a.y - (n.x*(x - a.x) + n.z*(z - a.z)) / n.y
        const a = self.vertices[0];
        return a.y - (self.normal.x * (pt.x - a.x) + self.normal.z * (pt.z - a.z)) / self.normal.y;
    }
};

pub const NavMesh = struct {
    allocator: std.mem.Allocator,
    nodes: []NavNode,
    bounds: BoundingBox,

    pub fn init(allocator: std.mem.Allocator, nodes: []NavNode, bounds: BoundingBox) NavMesh {
        return .{
            .allocator = allocator,
            .nodes = nodes,
            .bounds = bounds,
        };
    }

    pub fn deinit(self: *NavMesh) void {
        self.allocator.free(self.nodes);
        self.nodes = &.{};
    }

    /// Builds a NavMesh from indexed 3D triangle geometry.
    /// Filters out faces whose slope angle exceeds max_slope_rad (e.g. walls).
    /// Automatically connects adjacent faces sharing an edge to construct the navigation dual graph.
    pub fn buildFromTriangles(
        allocator: std.mem.Allocator,
        positions: []const [3]f32,
        indices: []const u32,
        max_slope_rad: f32,
    ) !NavMesh {
        const cos_max_slope = @cos(max_slope_rad);
        const tri_count = indices.len / 3;

        var temp_nodes = std.ArrayList(NavNode).empty;
        defer temp_nodes.deinit(allocator);

        var b_min = Vec3.new(std.math.inf(f32), std.math.inf(f32), std.math.inf(f32));
        var b_max = Vec3.new(-std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32));

        // Spatial edge key for pairing shared boundaries
        const EdgeKey = struct {
            p0: u64,
            p1: u64,

            fn make(v0: Vec3, v1: Vec3) @This() {
                // Quantize to 1mm to handle floating point precision
                const q0 = quantize(v0);
                const q1 = quantize(v1);
                if (q0 < q1) {
                    return .{ .p0 = q0, .p1 = q1 };
                } else {
                    return .{ .p0 = q1, .p1 = q0 };
                }
            }

            fn quantize(v: Vec3) u64 {
                const ix: i32 = @intFromFloat(@round(v.x * 1000.0));
                const iy: i32 = @intFromFloat(@round(v.y * 1000.0));
                const iz: i32 = @intFromFloat(@round(v.z * 1000.0));
                var h: u64 = 0xcbf29ce484222325;
                h = (h ^ @as(u32, @bitCast(ix))) *% 0x100000001b3;
                h = (h ^ @as(u32, @bitCast(iy))) *% 0x100000001b3;
                h = (h ^ @as(u32, @bitCast(iz))) *% 0x100000001b3;
                return h;
            }
        };

        const EdgeRef = struct {
            node_index: u32,
            edge_index: u8,
        };

        var edge_map = std.AutoHashMap(EdgeKey, EdgeRef).init(allocator);
        defer edge_map.deinit();

        for (0..tri_count) |t| {
            const v_idx0 = indices[t * 3 + 0];
            const v_idx1 = indices[t * 3 + 1];
            const v_idx2 = indices[t * 3 + 2];

            const v0 = Vec3.new(positions[v_idx0][0], positions[v_idx0][1], positions[v_idx0][2]);
            const v1 = Vec3.new(positions[v_idx1][0], positions[v_idx1][1], positions[v_idx1][2]);
            const v2 = Vec3.new(positions[v_idx2][0], positions[v_idx2][1], positions[v_idx2][2]);

            // Normal
            const e1 = v1.sub(v0);
            const e2 = v2.sub(v0);
            var normal = e1.cross(e2);
            const len_sq = normal.lengthSq();
            if (len_sq < 1e-8) continue; // Degenerate triangle
            normal = normal.scale(1.0 / @sqrt(len_sq));

            const final_v0 = v0;
            var final_v1 = v1;
            var final_v2 = v2;

            // Ensure upward pointing normal for walkable surfaces
            if (normal.y < 0.0) {
                normal = normal.scale(-1.0);
                final_v1 = v2;
                final_v2 = v1;
            }

            // Slope check: normal.y >= cos(max_slope)
            if (normal.y < cos_max_slope) continue;

            const centroid = Vec3.new(
                (final_v0.x + final_v1.x + final_v2.x) / 3.0,
                (final_v0.y + final_v1.y + final_v2.y) / 3.0,
                (final_v0.z + final_v1.z + final_v2.z) / 3.0,
            );

            b_min.x = @min(b_min.x, @min(final_v0.x, @min(final_v1.x, final_v2.x)));
            b_min.y = @min(b_min.y, @min(final_v0.y, @min(final_v1.y, final_v2.y)));
            b_min.z = @min(b_min.z, @min(final_v0.z, @min(final_v1.z, final_v2.z)));
            b_max.x = @max(b_max.x, @max(final_v0.x, @max(final_v1.x, final_v2.x)));
            b_max.y = @max(b_max.y, @max(final_v0.y, @max(final_v1.y, final_v2.y)));
            b_max.z = @max(b_max.z, @max(final_v0.z, @max(final_v1.z, final_v2.z)));

            const node_idx: u32 = @intCast(temp_nodes.items.len);
            try temp_nodes.append(allocator, .{
                .vertices = .{ final_v0, final_v1, final_v2 },
                .centroid = centroid,
                .normal = normal,
            });

            // Edge pairing
            const verts = [3]Vec3{ final_v0, final_v1, final_v2 };
            for (0..3) |ei| {
                const key = EdgeKey.make(verts[ei], verts[(ei + 1) % 3]);
                if (edge_map.get(key)) |other| {
                    temp_nodes.items[node_idx].neighbors[ei] = other.node_index;
                    temp_nodes.items[other.node_index].neighbors[other.edge_index] = node_idx;
                } else {
                    try edge_map.put(key, .{
                        .node_index = node_idx,
                        .edge_index = @intCast(ei),
                    });
                }
            }
        }

        const nodes = try allocator.dupe(NavNode, temp_nodes.items);
        const bounds = if (nodes.len > 0) BoundingBox.init(b_min, b_max) else BoundingBox.zero;
        return NavMesh.init(allocator, nodes, bounds);
    }

    /// Builds a planar navigation grid around static bounding-box obstacles.
    pub fn buildGrid(
        allocator: std.mem.Allocator,
        min_x: f32,
        max_x: f32,
        min_z: f32,
        max_z: f32,
        elevation_y: f32,
        subdiv_x: usize,
        subdiv_z: usize,
        obstacles: []const BoundingBox,
    ) !NavMesh {
        const sx = @max(subdiv_x, 1);
        const sz = @max(subdiv_z, 1);
        const step_x = (max_x - min_x) / @as(f32, @floatFromInt(sx));
        const step_z = (max_z - min_z) / @as(f32, @floatFromInt(sz));

        var positions = std.ArrayList([3]f32).empty;
        defer positions.deinit(allocator);
        var indices = std.ArrayList(u32).empty;
        defer indices.deinit(allocator);

        for (0..sz) |iz| {
            const z0 = min_z + @as(f32, @floatFromInt(iz)) * step_z;
            const z1 = z0 + step_z;
            for (0..sx) |ix| {
                const x0 = min_x + @as(f32, @floatFromInt(ix)) * step_x;
                const x1 = x0 + step_x;

                // Check if this quad intersects any obstacle (using slight inset so touching edge does not block)
                const eps = 0.05 * @min(step_x, step_z);
                const quad_box = BoundingBox.init(
                    Vec3.new(x0 + eps, elevation_y - 0.5, z0 + eps),
                    Vec3.new(x1 - eps, elevation_y + 0.5, z1 - eps),
                );

                var blocked = false;
                for (obstacles) |obs| {
                    if (obs.intersects(quad_box)) {
                        blocked = true;
                        break;
                    }
                }
                if (blocked) continue;

                // Quad vertices: v0=(x0, z0), v1=(x1, z0), v2=(x1, z1), v3=(x0, z1)
                const base_idx: u32 = @intCast(positions.items.len);
                try positions.append(allocator, .{ x0, elevation_y, z0 });
                try positions.append(allocator, .{ x1, elevation_y, z0 });
                try positions.append(allocator, .{ x1, elevation_y, z1 });
                try positions.append(allocator, .{ x0, elevation_y, z1 });

                // CCW Tri 1: 0, 1, 2
                try indices.append(allocator, base_idx + 0);
                try indices.append(allocator, base_idx + 1);
                try indices.append(allocator, base_idx + 2);

                // CCW Tri 2: 0, 2, 3
                try indices.append(allocator, base_idx + 0);
                try indices.append(allocator, base_idx + 2);
                try indices.append(allocator, base_idx + 3);
            }
        }

        return buildFromTriangles(allocator, positions.items, indices.items, std.math.pi * 0.4);
    }

    /// Finds the NavNode containing the given point in horizontal 2D XZ coordinates.
    pub fn findNode(self: *const NavMesh, pt: Vec3) ?u32 {
        for (self.nodes, 0..) |node, i| {
            if (node.containsPointXZ(pt)) {
                // If elevation difference is reasonable (< 3m), accept
                const y = node.getYAtXZ(pt);
                if (@abs(y - pt.y) < 3.0) {
                    return @intCast(i);
                }
            }
        }
        return null;
    }

    /// Finds the node closest to the given point, even if outside mesh boundaries.
    pub fn findClosestNode(self: *const NavMesh, pt: Vec3) ?u32 {
        if (self.findNode(pt)) |idx| return idx;
        if (self.nodes.len == 0) return null;

        var best_idx: u32 = 0;
        var best_dist_sq: f32 = std.math.inf(f32);

        for (self.nodes, 0..) |node, i| {
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

    /// Clamps point to the nearest surface point on the NavMesh.
    pub fn clampToMesh(self: *const NavMesh, pt: Vec3) Vec3 {
        if (self.findClosestNode(pt)) |node_idx| {
            const node = self.nodes[node_idx];
            const y = node.getYAtXZ(pt);
            return Vec3.new(pt.x, y, pt.z);
        }
        return pt;
    }
};
