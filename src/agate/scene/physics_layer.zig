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
///
/// Stage 1 splits the capture: `buildDebug` (game side, any non-pool thread
/// under update-vs-prepare exclusion) snapshots the same lines into the
/// game-owned `build_lines`/`build_visible`, stamped with the scene
/// `build_seq`; `latchDebug` (prepare, context thread) copies the build
/// into the render-owned `prepared_lines`/`prepared_visible` when the seq is
/// newer, else runs the historical live capture. Plain fields, no atomics;
/// GPU handles stay borrowed under the P3 epochs.
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
    /// Game-owned build capture (stage 1): written by `buildDebug` on the
    /// update side, consumed by `latchDebug` on the context side. Same
    /// shape as the prepared capture; freed in deinit.
    build_lines: std.ArrayListUnmanaged(physics.DebugLine) = .empty,
    /// Snapshotted visibility for the build frame.
    build_visible: bool = false,
    /// Scene `build_seq` stamped by the last `buildDebug` (0 = never).
    build_seq: u64 = 0,
    /// Last `build_seq` consumed by `latchDebug`.
    latched_seq: u64 = 0,
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
        self.build_lines.deinit(allocator);
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
    ///
    /// Inline fallback path: `Scene.prepareFrame` calls this when no fresh
    /// game-side build exists, so apps that never call `buildPreparedFrame`
    /// behave exactly as before. Kept callable by tooling as well.
    pub fn captureDebug(self: *PhysicsIntegration, allocator: std.mem.Allocator) void {
        captureInto(self, allocator, &self.prepared_lines, &self.prepared_visible);
    }

    /// Game-side CPU capture (stage 1): the exact `captureDebug` logic
    /// writing the game-owned `build_lines`/`build_visible`, stamped with
    /// the scene `build_seq`. Pure CPU + allocator; callable from any
    /// non-pool thread under update-vs-prepare exclusion. OOM fail-closes
    /// to coherent-empty (mirroring `captureDebug`); the seq still advances
    /// — the empty IS the new state — so the latch publishes it instead of
    /// a stale prior frame.
    pub fn buildDebug(self: *PhysicsIntegration, allocator: std.mem.Allocator, seq: u64) void {
        self.build_seq = seq;
        captureInto(self, allocator, &self.build_lines, &self.build_visible);
    }

    /// Context-side latch (stage 1): when a fresh build exists (`build_seq`
    /// newer than `latched_seq`), copies `build_lines`/`build_visible` into
    /// the render-owned `prepared_lines`/`prepared_visible` (reserve-once,
    /// OOM coherent-empty) and advances `latched_seq`. Otherwise runs the
    /// historical live capture, so a latch without a fresh build stays
    /// coherent. Draws keep reading the prepared capture only.
    pub fn latchDebug(self: *PhysicsIntegration, allocator: std.mem.Allocator) void {
        if (self.build_seq == self.latched_seq) {
            self.captureDebug(allocator);
            return;
        }
        self.latched_seq = self.build_seq;
        self.prepared_lines.ensureTotalCapacity(allocator, self.build_lines.items.len) catch {
            self.prepared_lines.clearRetainingCapacity();
            self.prepared_visible = false;
            return;
        };
        self.prepared_lines.clearRetainingCapacity();
        for (self.build_lines.items) |line| {
            self.prepared_lines.appendAssumeCapacity(line);
        }
        self.prepared_visible = self.build_visible;
    }

    /// Shared capture body: snapshots `world.appendDebugLines` into `out`
    /// with `visible` set. Builds DIRECTLY into the target: capture and its
    /// consumer are sequential, so no transactional scratch copy is needed.
    /// OOM mid-append fail-closes to coherent-empty (the partial prefix is
    /// cleared, the frame marked invisible). Steady-state appends never
    /// reallocate (appendDebugLines reserves the exact count once up front),
    /// so warm captures are allocation-free; retained capacity is reused and
    /// the next funded capture recovers.
    fn captureInto(self: *PhysicsIntegration, allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(physics.DebugLine), visible: *bool) void {
        if (!self.show_debug or self.world == null) {
            out.clearRetainingCapacity();
            visible.* = false;
            return;
        }
        const pw = &self.world.?;
        out.clearRetainingCapacity();
        pw.appendDebugLines(allocator, out) catch {
            out.clearRetainingCapacity();
            visible.* = false;
            return;
        };
        visible.* = true;
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

// --- Stage 1 build/latch tests (pure CPU + allocator, no sg.*). ---

const TestMesh = @import("../mesh.zig").Mesh;

fn makeDebugFixture(allocator: std.mem.Allocator) !struct {
    integ: PhysicsIntegration,
    mesh: *TestMesh,
} {
    var integ = PhysicsIntegration{};
    _ = integ.enable(allocator, null);
    const m = try allocator.create(TestMesh);
    m.* = .{ .name = "build_dbg", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 0 };
    _ = try integ.getWorld().?.createBody(m, .box, 0.0);
    integ.show_debug = true;
    return .{ .integ = integ, .mesh = m };
}

test "physics buildDebug+latchDebug isolates live mutations" {
    const t = std.testing;
    var fx = try makeDebugFixture(t.allocator);
    defer fx.integ.deinit(t.allocator);
    defer t.allocator.destroy(fx.mesh);
    var integ = &fx.integ;

    // Game-side build: fills the build capture, stamps the seq, leaves the
    // render-owned capture untouched.
    integ.buildDebug(t.allocator, 9);
    try t.expectEqual(@as(u64, 9), integ.build_seq);
    try t.expect(integ.build_visible);
    try t.expectEqual(@as(usize, 12), integ.build_lines.items.len);
    try t.expect(!integ.prepared_visible);
    try t.expectEqual(@as(usize, 0), integ.prepared_lines.items.len);

    // Live mutation after the build: the build capture stays frozen.
    const x0 = integ.build_lines.items[0].a.x;
    fx.mesh.position = Vec3.new(5, 0, 0);
    try t.expectEqual(x0, integ.build_lines.items[0].a.x);

    // Context-side latch publishes the build-time snapshot; stepping the
    // world afterwards cannot reach the render-owned capture.
    integ.latchDebug(t.allocator);
    try t.expectEqual(@as(u64, 9), integ.latched_seq);
    try t.expect(integ.prepared_visible);
    try t.expectEqual(@as(usize, 12), integ.prepared_lines.items.len);
    try t.expectApproxEqAbs(x0, integ.prepared_lines.items[0].a.x, 1e-4);
    integ.step(0.016);
    try t.expectApproxEqAbs(x0, integ.prepared_lines.items[0].a.x, 1e-4);
}

test "physics two builds before latch: newest wins, stale latch recaptures" {
    const t = std.testing;
    var fx = try makeDebugFixture(t.allocator);
    defer fx.integ.deinit(t.allocator);
    defer t.allocator.destroy(fx.mesh);
    var integ = &fx.integ;

    integ.buildDebug(t.allocator, 1);
    const x0 = integ.build_lines.items[0].a.x;
    fx.mesh.position = Vec3.new(5, 0, 0);
    integ.buildDebug(t.allocator, 2);
    try t.expectApproxEqAbs(x0 + 5.0, integ.build_lines.items[0].a.x, 1e-4);

    integ.latchDebug(t.allocator);
    try t.expectApproxEqAbs(x0 + 5.0, integ.prepared_lines.items[0].a.x, 1e-4);

    // Latch without a fresh build falls back to the live capture (old
    // behavior): the capture tracks live state, never a stale build.
    fx.mesh.position = Vec3.new(9, 0, 0);
    integ.latchDebug(t.allocator);
    try t.expectApproxEqAbs(x0 + 9.0, integ.prepared_lines.items[0].a.x, 1e-4);
}

test "physics buildDebug OOM fail-closes, latch publishes empty, then recovers" {
    const t = std.testing;
    var fx = try makeDebugFixture(t.allocator);
    defer fx.integ.deinit(t.allocator);
    defer t.allocator.destroy(fx.mesh);
    var integ = &fx.integ;

    integ.buildDebug(t.allocator, 1);
    integ.latchDebug(t.allocator);
    try t.expect(integ.prepared_visible);
    try t.expectEqual(@as(usize, 12), integ.prepared_lines.items.len);

    // Unfunded build fail-closes to coherent-empty (mirrors captureDebug),
    // but the seq still advances — the latch publishes the empty, never the
    // stale prior frame.
    var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    integ.build_lines.clearAndFree(t.allocator);
    integ.buildDebug(failing.allocator(), 2);
    try t.expect(!integ.build_visible);
    try t.expectEqual(@as(usize, 0), integ.build_lines.items.len);
    integ.latchDebug(t.allocator);
    try t.expect(!integ.prepared_visible);
    try t.expectEqual(@as(usize, 0), integ.prepared_lines.items.len);

    // Recovery: a funded build+latch publishes the full capture again.
    integ.buildDebug(t.allocator, 3);
    integ.latchDebug(t.allocator);
    try t.expect(integ.prepared_visible);
    try t.expectEqual(@as(usize, 12), integ.prepared_lines.items.len);
}
