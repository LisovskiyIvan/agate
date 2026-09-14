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
    // MSAA twin (pipeline sample count must match the main target); lazily
    // created on the first MSAA frame, recreated on count changes.
    debug_pass_msaa: ?passes.DebugPass = null,
    debug_pass_msaa_samples: i32 = 0,

    pub fn deinit(self: *PhysicsIntegration, allocator: std.mem.Allocator) void {
        if (self.world) |*pw| {
            pw.deinit();
            self.world = null;
        }
        self.debug_lines.deinit(allocator);
        if (self.debug_pass) |*dp| dp.deinit();
        self.debug_pass = null;
        if (self.debug_pass_msaa) |*dp| dp.deinit();
        self.debug_pass_msaa = null;
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
    /// GPU pass is created on first visible frame. `samples` is the
    /// effective main-target sample count (scene/msaa.zig).
    pub fn renderDebug(self: *PhysicsIntegration, allocator: std.mem.Allocator, view_proj: Mat4, samples: i32, stats: *SceneStats) void {
        if (!self.show_debug) return;
        const pw = &(self.world orelse return);
        self.debug_lines.clearRetainingCapacity();
        pw.appendDebugLines(allocator, &self.debug_lines) catch {};
        if (self.debug_lines.items.len == 0) return;

        const pass = self.debugPassFor(allocator, samples) orelse return;
        pass.render(view_proj, self.debug_lines.items);
        stats.main_draw_calls += 1;
        stats.draw_calls += 1;
    }

    /// Debug pass variant matching the target sample count.
    fn debugPassFor(self: *PhysicsIntegration, allocator: std.mem.Allocator, samples: i32) ?*passes.DebugPass {
        if (samples <= 1) {
            if (self.debug_pass == null) {
                self.debug_pass = passes.DebugPass.init(allocator) catch null;
            }
            return if (self.debug_pass) |*dp| dp else null;
        }
        if (self.debug_pass_msaa == null or self.debug_pass_msaa_samples != samples) {
            if (self.debug_pass_msaa) |*dp| dp.deinit();
            self.debug_pass_msaa = passes.DebugPass.initSampled(allocator, samples) catch null;
            self.debug_pass_msaa_samples = samples;
        }
        return if (self.debug_pass_msaa) |*dp| dp else null;
    }
};
