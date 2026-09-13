const std = @import("std");

const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const physics = @import("../physics.zig");
const PhysicsWorld = physics.PhysicsWorld;
const passes = @import("../passes/mod.zig");
const stats_mod = @import("stats.zig");
const SceneStats = stats_mod.SceneStats;

/// Physics integration for the scene: the optional physics world, its
/// per-frame step, and the 3D debug wireframe overlay (lazily created GPU
/// pass + a line buffer refilled each frame while visible).
pub const PhysicsIntegration = struct {
    world: ?PhysicsWorld = null,
    // Toggles the physics debug wireframe rendering in the main pass.
    show_debug: bool = false,
    debug_lines: std.ArrayListUnmanaged(physics.DebugLine) = .empty,
    // Lazily created on first use; renders physics debug wireframes in 3D.
    debug_pass: ?passes.DebugPass = null,

    pub fn deinit(self: *PhysicsIntegration, allocator: std.mem.Allocator) void {
        if (self.world) |*pw| {
            pw.deinit();
            self.world = null;
        }
        self.debug_lines.deinit(allocator);
        if (self.debug_pass) |*dp| dp.deinit();
        self.debug_pass = null;
    }

    /// Creates the world on first call (gravity optional) and returns it.
    pub fn enable(self: *PhysicsIntegration, allocator: std.mem.Allocator, gravity: ?Vec3) *PhysicsWorld {
        if (self.world == null) {
            self.world = PhysicsWorld.init(allocator);
        }
        if (gravity) |g| {
            self.world.?.gravity = g;
        }
        return &self.world.?;
    }

    pub fn getWorld(self: *PhysicsIntegration) ?*PhysicsWorld {
        if (self.world) |*pw| return pw;
        return null;
    }

    pub fn step(self: *PhysicsIntegration, dt: f32) void {
        if (self.world) |*pw| {
            pw.step(dt);
        }
    }

    /// Renders the physics debug wireframe (main pass, depth-tested, no
    /// depth write). No-op unless show_debug is set and a world exists; the
    /// GPU pass is created on first visible frame.
    pub fn renderDebug(self: *PhysicsIntegration, allocator: std.mem.Allocator, view_proj: Mat4, stats: *SceneStats) void {
        if (!self.show_debug) return;
        const pw = &(self.world orelse return);
        self.debug_lines.clearRetainingCapacity();
        pw.appendDebugLines(allocator, &self.debug_lines) catch {};
        if (self.debug_lines.items.len == 0) return;
        if (self.debug_pass == null) {
            self.debug_pass = passes.DebugPass.init(allocator) catch null;
        }
        if (self.debug_pass) |*dp| {
            dp.render(view_proj, self.debug_lines.items);
            stats.main_draw_calls += 1;
            stats.draw_calls += 1;
        }
    }
};
