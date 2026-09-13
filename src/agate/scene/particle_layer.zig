const std = @import("std");

const particles = @import("../particles.zig");
const ParticleSystem = particles.ParticleSystem;
const Camera = @import("../camera.zig").Camera;
const passes = @import("../passes/mod.zig");
const stats_mod = @import("stats.zig");
const SceneStats = stats_mod.SceneStats;

/// CPU particle systems and their GPU billboard pass. Owns the system list
/// (created via Scene.createParticleSystem) and renders all live systems in
/// the main pass, accumulating billboard draw stats.
pub const ParticleLayer = struct {
    systems: std.ArrayListUnmanaged(*ParticleSystem) = .empty,

    pass: passes.ParticlePass,

    pub fn init() ParticleLayer {
        return .{ .pass = passes.ParticlePass.init() };
    }

    pub fn deinit(self: *ParticleLayer, allocator: std.mem.Allocator) void {
        for (self.systems.items) |ps| {
            ps.deinit();
            allocator.destroy(ps);
        }
        self.systems.deinit(allocator);
        self.pass.deinit();
    }

    pub fn create(self: *ParticleLayer, allocator: std.mem.Allocator, name: []const u8, capacity: usize) !*ParticleSystem {
        const ps = try ParticleSystem.init(allocator, name, capacity);
        try self.systems.append(allocator, ps);
        return ps;
    }

    pub fn update(self: *ParticleLayer, dt: f32) void {
        for (self.systems.items) |ps| {
            ps.update(dt);
        }
        // Compute-simulated systems integrate in one shared compute pass
        // after their bookkeeping (spawn-ring uploads) and before the frame's
        // render passes. No-op when the backend lacks compute or no system
        // uses `.compute` mode.
        self.pass.runComputeSimulations(self.systems.items, dt);
    }

    /// Renders all particle systems inside the main pass.
    pub fn render(self: *ParticleLayer, camera: Camera, aspect: f32, stats: *SceneStats) void {
        if (self.systems.items.len == 0) return;
        self.pass.render(self.systems.items, camera, aspect);
        for (self.systems.items) |ps| {
            if (ps.active_count > 0) {
                stats.main_draw_calls += 1;
                stats.draw_calls += 1;
                stats.triangles += 2 * @as(u32, @intCast(ps.active_count));
            }
        }
    }
};
