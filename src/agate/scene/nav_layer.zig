const std = @import("std");

const math = @import("math");
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const ai_mod = @import("../ai.zig");
const NavMesh = ai_mod.NavMesh;
const NavAgent = ai_mod.NavAgent;

/// Navigation ownership: nav meshes (built from triangles or a grid) plus
/// the agents steering on them. Scene forwards the historical create*/
/// update methods to this layer.
pub const NavLayer = struct {
    meshes: std.ArrayListUnmanaged(*NavMesh) = .empty,
    agents: std.ArrayListUnmanaged(*NavAgent) = .empty,

    pub fn deinit(self: *NavLayer, allocator: std.mem.Allocator) void {
        for (self.agents.items) |agent| {
            agent.deinit();
            allocator.destroy(agent);
        }
        self.agents.deinit(allocator);

        for (self.meshes.items) |nm| {
            nm.deinit();
            allocator.destroy(nm);
        }
        self.meshes.deinit(allocator);
    }

    pub fn createMeshFromTriangles(
        self: *NavLayer,
        allocator: std.mem.Allocator,
        positions: []const [3]f32,
        indices: []const u32,
        max_slope_rad: f32,
    ) !*NavMesh {
        const ptr = try allocator.create(NavMesh);
        errdefer allocator.destroy(ptr);
        ptr.* = try NavMesh.buildFromTriangles(allocator, positions, indices, max_slope_rad);
        try self.meshes.append(allocator, ptr);
        return ptr;
    }

    pub fn createMeshGrid(
        self: *NavLayer,
        allocator: std.mem.Allocator,
        min_x: f32,
        max_x: f32,
        min_z: f32,
        max_z: f32,
        elevation_y: f32,
        subdiv_x: usize,
        subdiv_z: usize,
        obstacles: []const BoundingBox,
    ) !*NavMesh {
        const ptr = try allocator.create(NavMesh);
        errdefer allocator.destroy(ptr);
        ptr.* = try NavMesh.buildGrid(allocator, min_x, max_x, min_z, max_z, elevation_y, subdiv_x, subdiv_z, obstacles);
        try self.meshes.append(allocator, ptr);
        return ptr;
    }

    pub fn createAgent(self: *NavLayer, allocator: std.mem.Allocator, nav_mesh: *const NavMesh, start_pos: Vec3) !*NavAgent {
        const ptr = try allocator.create(NavAgent);
        errdefer allocator.destroy(ptr);
        ptr.* = NavAgent.init(allocator, nav_mesh, start_pos);
        try self.agents.append(allocator, ptr);
        return ptr;
    }

    pub fn updateAgents(self: *NavLayer, dt: f32) void {
        for (self.agents.items) |ag| {
            ag.update(dt);
        }
    }
};
