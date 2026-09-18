const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const physics = @import("../physics.zig");
const PhysicsWorld = physics.PhysicsWorld;
const passes = @import("../passes/mod.zig");
const gpu_thread = @import("../gpu_thread.zig");
const stats_mod = @import("stats.zig");
const SceneStats = stats_mod.SceneStats;

/// Physics integration for the scene: the optional physics world, its
/// per-frame step, and the 3D debug wireframe overlay (lazily created GPU
/// pass + a line buffer refilled each frame while visible).
///
/// Update||render ownership: the world is stepped on the update side, but
/// the draw NEVER touches it. `captureDebug` (prepare, context thread,
/// BEFORE the update/render unlock) snapshots `world.appendDebugLines` into
/// the render-owned `prepared_lines`; `uploadDebug` spends the frame's single
/// `sg.updateBuffer`; `renderDebugPrepared` draws the committed upload once
/// per view. A concurrent update may step/mutate the world while render
/// draws — the draw only sees the capture. `show_debug`/world presence are
/// snapshotted into `prepared_visible` for the same reason: no live reads
/// at draw time.
pub const PhysicsIntegration = struct {
    world: ?PhysicsWorld = null,
    // Toggles the physics debug wireframe rendering in the main pass.
    show_debug: bool = false,
    debug_lines: std.ArrayListUnmanaged(physics.DebugLine) = .empty,
    /// Render-owned capture of the last prepared frame's debug lines
    /// (retained capacity, rebuilt every prepare). The draw reads ONLY this
    /// + `prepared_visible` + the render-owned passes below.
    prepared_lines: std.ArrayListUnmanaged(physics.DebugLine) = .empty,
    /// Snapshotted visibility for the prepared frame (show_debug && world at
    /// capture time). Draw site selection reads this, never live fields.
    prepared_visible: bool = false,
    // Lazily created on first use; renders physics debug wireframes in 3D.
    debug_pass: ?passes.DebugPass = null,
    // MSAA twin (pipeline sample count must match the main target); lazily
    // created on the first MSAA frame, recreated on count changes.
    debug_pass_msaa: ?passes.DebugPass = null,

    pub fn deinit(self: *PhysicsIntegration, allocator: std.mem.Allocator) void {
        if (self.world) |*pw| {
            pw.deinit();
            self.world = null;
        }
        self.debug_lines.deinit(allocator);
        self.prepared_lines.deinit(allocator);
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

    /// Captures the debug wireframe into the render-owned `prepared_lines`.
    /// Runs in prepare (context thread, BEFORE the update/render unlock), so
    /// `world.appendDebugLines` NEVER runs inside render. Newest wins every
    /// prepare; OOM fail-closes to coherent-empty (never a partial half) and
    /// recovers on the next funded prepare. Pure CPU + allocator: headless
    /// tests exercise it fully with no `sg.*`.
    pub fn captureDebug(self: *PhysicsIntegration, allocator: std.mem.Allocator) void {
        if (!self.show_debug or self.world == null) {
            self.prepared_lines.clearRetainingCapacity();
            self.prepared_visible = false;
            return;
        }
        const pw = &self.world.?;
        // Build DIRECTLY into the render-owned capture: prepare and render
        // are sequential, so no consumer reads prepared_lines during this
        // prepare — no transactional scratch copy is needed. OOM mid-append
        // fail-closes to coherent-empty (the partial prefix is cleared, the
        // frame marked invisible). Steady-state appends never reallocate
        // (appendDebugLines reserves the exact count once up front), so warm
        // captures are allocation-free; retained capacity is reused and the
        // next funded prepare recovers.
        self.prepared_lines.clearRetainingCapacity();
        pw.appendDebugLines(allocator, &self.prepared_lines) catch {
            self.prepared_lines.clearRetainingCapacity();
            self.prepared_visible = false;
            return;
        };
        self.prepared_visible = true;
    }

    /// Spends the frame's single `sg.updateBuffer` for the captured lines.
    /// Runs in prepare (context thread), once per frame no matter how many
    /// PIP views render; every view then draws via renderDebugPrepared with
    /// no re-upload. `samples` is the effective main-target sample count
    /// (scene/msaa.zig). Headless-safe: the capture above still runs (tests
    /// read `prepared_lines`), but no pass is created and no `sg.*` fires
    /// without a context.
    pub fn uploadDebug(self: *PhysicsIntegration, allocator: std.mem.Allocator, samples: i32) void {
        if (!self.prepared_visible or self.prepared_lines.items.len == 0) return;
        if (!gpu_thread.isOnContextThread()) return;
        if (!sg.isvalid()) return;
        const pass = self.debugPassFor(allocator, samples) orelse return;
        _ = pass.upload(self.prepared_lines.items);
    }

    /// Upload-free draw of the prepared debug lines for one view. Reads ONLY
    /// the capture (`prepared_visible`/`prepared_lines`) + the render-owned
    /// passes — never `show_debug`, never the live world. Bumps the stats
    /// counters once per ACTUALLY ISSUED draw (same totals as the legacy
    /// per-view renderDebug: one upload in prepare, N draws in render; a
    /// no-op draw from an empty/invalid upload counts nothing).
    /// Headless-safe no-op (no pass creation, no `sg.*`, counters stay
    /// clean).
    pub fn renderDebugPrepared(self: *PhysicsIntegration, view_proj: Mat4, samples: i32, stats: *SceneStats) void {
        if (!self.prepared_visible or self.prepared_lines.items.len == 0) return;
        if (!sg.isvalid()) return;
        const pass = self.passForDraw(samples) orelse return;
        if (!pass.drawPrepared(view_proj)) return;
        stats.main_draw_calls += 1;
        stats.draw_calls += 1;
    }

    /// Renders the physics debug wireframe (main pass, depth-tested, no
    /// depth write). No-op unless show_debug is set and a world exists; the
    /// GPU pass is created on first visible frame. `samples` is the
    /// effective main-target sample count (scene/msaa.zig).
    ///
    /// Legacy single-view path (capture + upload + draw from live state):
    /// Scene.render no longer calls this — it captures in prepare and draws
    /// prepared per view. Kept for standalone/tooling callers.
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

    /// Non-creating pass lookup for the draw path: the upload in prepare
    /// already created the matching variant. Render never creates GPU
    /// objects as a side effect (and never without a context — callers
    /// return early on `!sg.isvalid()` first).
    /// Non-creating pass lookup for the draw path: the upload in prepare
    /// already created the matching variant. Render never creates GPU
    /// objects as a side effect (and never without a context — callers
    /// return early on `!sg.isvalid()` first).
    fn passForDraw(self: *PhysicsIntegration, samples: i32) ?*passes.DebugPass {
        if (samples <= 1) {
            return if (self.debug_pass) |*dp| dp else null;
        }
        const msaa = if (self.debug_pass_msaa) |*dp| dp else return null;
        if (msaa.sample_count != samples) return null;
        return msaa;
    }

    /// Debug pass variant matching the target sample count.
    fn debugPassFor(self: *PhysicsIntegration, allocator: std.mem.Allocator, samples: i32) ?*passes.DebugPass {
        if (samples <= 1) {
            if (self.debug_pass == null) {
                self.debug_pass = passes.DebugPass.init(allocator) catch null;
            }
            return if (self.debug_pass) |*dp| dp else null;
        }
        if (self.debug_pass_msaa == null or self.debug_pass_msaa.?.sample_count != samples) {
            if (self.debug_pass_msaa) |*dp| dp.deinit();
            self.debug_pass_msaa = passes.DebugPass.initSampled(allocator, samples) catch null;
        }
        return if (self.debug_pass_msaa) |*dp| dp else null;
    }
};
