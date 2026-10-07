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
///
/// Wave 32 freeze-then-latch (adopted concurrent-build path): the game
/// build additionally freezes the capture into the claimed draw slot
/// (`stageIntoSlot`) and the prepare latch consumes the slot copy
/// (`latchSlotDebug`) — never the shared staging store — so a game-thread
/// build colliding with the context-side latch cannot tear the record.
/// The shared store stays for the sequential flow (bit-identical) and
/// direct/tooling use.
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
    build_seq: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    /// Last `build_seq` consumed by `latchDebug`.
    latched_seq: usize = 0,
    // Lazily created on first use; renders physics debug wireframes in 3D.
    debug_pass: ?passes.DebugPass = null,
    // Variant pass (pipeline sample count must match the main target);
    // lazily created on the first non-base frame, recreated on shape
    // changes. Keyed by (sample_count, color_format).
    debug_pass_msaa: ?passes.DebugPass = null,
    // Render-owned upload-shape tag: the (samples, color_format) of the last
    // SUCCESSFUL `uploadDebug` spend of the frame's single
    // `sg.updateBuffer`. Cleared before every upload attempt, set only when
    // `DebugPass.upload` reports success. `renderDebugPrepared` draws
    // only on a tag match, so a failed allocation (parent falls back to
    // another shape) or any shape change can never draw stale buffers from
    // an earlier frame. Plain values, no GPU calls.
    prepared_upload_samples: i32 = 0,
    prepared_upload_format: sg.PixelFormat = .RGBA16F,
    prepared_upload_valid: bool = false,

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
        self.prepared_upload_valid = false;
        self.prepared_upload_samples = 0;
        self.prepared_upload_format = .RGBA16F;
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
    /// Direct layer capture (also used by layer fixtures/tooling):
    /// `latchDebug` falls back to this when no fresh producer build exists.
    /// Scene itself always runs producer buildDebug → stageIntoSlot →
    /// context latchSlotDebug. Kept callable by tooling as well.
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
    pub fn buildDebug(self: *PhysicsIntegration, allocator: std.mem.Allocator, seq: usize) void {
        captureInto(self, allocator, &self.build_lines, &self.build_visible);
        self.build_seq.store(seq, .release);
    }

    /// Context-side latch (stage 1): when a fresh build exists (`build_seq`
    /// newer than `latched_seq`), copies `build_lines`/`build_visible` into
    /// the render-owned `prepared_lines`/`prepared_visible` (reserve-once,
    /// OOM coherent-empty) and advances `latched_seq`. Otherwise runs the
    /// direct live capture, so a latch without a fresh build stays
    /// coherent (direct/tooling path). Draws keep reading the prepared capture only.
    ///
    /// Sequential/standalone path only since wave 32: the adopted
    /// concurrent-build path freezes into the claimed draw slot
    /// (`stageIntoSlot`) and latches from it (`latchSlotDebug`), so the
    /// prepare latch there never reads the shared
    /// `build_lines`/`build_visible` — a game-thread build colliding with
    /// the context-side latch cannot tear the record. This shared-store
    /// latch stays for apps on the sequential flow (bit-identical) and
    /// for direct/tooling use.
    pub fn latchDebug(self: *PhysicsIntegration, allocator: std.mem.Allocator) void {
        const fresh_seq = self.build_seq.load(.acquire);
        if (fresh_seq == self.latched_seq) {
            self.captureDebug(allocator);
            return;
        }
        self.latched_seq = fresh_seq;
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

    /// Freeze-then-latch slot stage (wave 32, adopted concurrent-build
    /// path): copies the just-captured `build_lines`/`build_visible` into
    /// the claimed slot's `physics_lines`/`physics_visible`. Runs
    /// producer-side inside `Scene.buildIntoClaimedSlot` right after
    /// `buildDebug`; the frozen copy rides the publish release edge
    /// (`build_slot`/`build_seq`) to the prepare latch, which consumes it
    /// via `latchSlotDebug` — never the shared staging store.
    /// Staged-wins on OOM (fail-closes the slot copy to coherent-empty +
    /// invisible, mirroring `captureInto`); the shared build capture keeps
    /// its own existing semantics for direct use.
    pub fn stageIntoSlot(
        self: *PhysicsIntegration,
        allocator: std.mem.Allocator,
        out: *std.ArrayListUnmanaged(physics.DebugLine),
        visible: *bool,
    ) void {
        out.ensureTotalCapacity(allocator, self.build_lines.items.len) catch {
            out.clearRetainingCapacity();
            visible.* = false;
            return;
        };
        out.clearRetainingCapacity();
        out.appendSliceAssumeCapacity(self.build_lines.items);
        visible.* = self.build_visible;
    }

    /// Context-side slot latch (wave 32, adopted concurrent-build path):
    /// copies the claimed slot's frozen `lines`/`visible` into the
    /// render-owned `prepared_lines`/`prepared_visible` (reserve-once, OOM
    /// coherent-empty) and consumes the pending build generation
    /// (`latched_seq` catches up to `build_seq`, acquire-loaded — so a
    /// later standalone latch runs the live capture, never a stale build).
    /// Reads only the slot payload + the seq word: never the shared
    /// `build_lines`/`build_visible`, never the live world.
    pub fn latchSlotDebug(
        self: *PhysicsIntegration,
        allocator: std.mem.Allocator,
        lines: []const physics.DebugLine,
        visible: bool,
    ) void {
        const fresh_seq = self.build_seq.load(.acquire);
        self.latched_seq = fresh_seq;
        self.prepared_lines.ensureTotalCapacity(allocator, lines.len) catch {
            self.prepared_lines.clearRetainingCapacity();
            self.prepared_visible = false;
            return;
        };
        self.prepared_lines.clearRetainingCapacity();
        self.prepared_lines.appendSliceAssumeCapacity(lines);
        self.prepared_visible = visible;
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
    /// no re-upload. `samples`/`color_format` pin the exact main-target
    /// shape (upload stage may create the matching pass variant).
    /// Headless-safe: the capture above still runs (tests
    /// read `prepared_lines`), but no pass is created and no `sg.*` fires
    /// without a context.
    pub fn uploadDebug(self: *PhysicsIntegration, allocator: std.mem.Allocator, samples: i32, color_format: sg.PixelFormat) void {
        self.clearUploadTag();
        if (!self.prepared_visible or self.prepared_lines.items.len == 0) return;
        if (!gpu_thread.isOnContextThread()) return;
        if (!sg.isvalid()) return;
        const pass = self.debugPassFor(allocator, samples, color_format) orelse return;
        if (!pass.upload(self.prepared_lines.items)) return;
        self.prepared_upload_samples = samples;
        self.prepared_upload_format = color_format;
        self.prepared_upload_valid = true;
    }

    /// Clears the upload-shape tag (no successful upload outstanding).
    /// Pure, no `sg.*` calls.
    fn clearUploadTag(self: *PhysicsIntegration) void {
        self.prepared_upload_valid = false;
        self.prepared_upload_samples = 0;
        self.prepared_upload_format = .RGBA16F;
    }

    /// True when the last successful upload matches the requested target
    /// shape. Pure (no `sg.*`), so headless tests pin the gate directly.
    fn uploadTagMatches(self: *const PhysicsIntegration, samples: i32, color_format: sg.PixelFormat) bool {
        if (!self.prepared_upload_valid) return false;
        if (self.prepared_upload_samples != samples) return false;
        if (self.prepared_upload_format != color_format) return false;
        return true;
    }

    /// Upload-free draw of the prepared debug lines for one view (explicit
    /// target shape). Reads ONLY the capture (`prepared_visible`/
    /// `prepared_lines`) + the render-owned passes. Bumps the stats counters
    /// once per ACTUALLY ISSUED draw (one upload in prepare, N draws in
    /// render; a no-op draw from an empty/invalid upload counts nothing).
    /// RENDER STAGE ONLY: never creates GPU objects, never uploads — the
    /// non-creating lookup hits only the variant the upload stage built.
    /// Draws only when the upload-shape tag matches the requested shape AND
    /// the looked-up pass carries that shape, so a failed allocation or any
    /// shape change draws nothing instead of stale buffers. Stats bump only
    /// on an actually issued draw. Headless-safe no-op (no pass creation,
    /// no `sg.*`, counters stay clean).
    pub fn renderDebugPrepared(self: *PhysicsIntegration, view_proj: Mat4, samples: i32, color_format: sg.PixelFormat, stats: *SceneStats) void {
        if (!self.prepared_visible or self.prepared_lines.items.len == 0) return;
        if (!self.uploadTagMatches(samples, color_format)) return;
        if (!sg.isvalid()) return;
        const pass = self.passForDraw(samples, color_format) orelse return;
        if (pass.sample_count != samples or pass.color_format != color_format) return;
        if (!pass.drawPrepared(view_proj)) return;
        stats.main_draw_calls += 1;
        stats.draw_calls += 1;
    }

    /// Renders the physics debug wireframe (main pass, depth-tested, no
    /// depth write). No-op unless show_debug is set and a world exists; the
    /// GPU pass is created on first visible frame. `samples`/`color_format`
    /// pin the exact main-target shape.
    ///
    /// Single-view path (capture + upload + draw from live state):
    /// Scene.render no longer calls this — it captures in prepare and draws
    /// prepared per view. Kept for standalone/tooling callers.
    pub fn renderDebug(self: *PhysicsIntegration, allocator: std.mem.Allocator, view_proj: Mat4, samples: i32, color_format: sg.PixelFormat, stats: *SceneStats) void {
        if (!self.show_debug) return;
        const pw = &(self.world orelse return);
        self.debug_lines.clearRetainingCapacity();
        pw.appendDebugLines(allocator, &self.debug_lines) catch {};
        if (self.debug_lines.items.len == 0) return;

        const pass = self.debugPassFor(allocator, samples, color_format) orelse return;
        pass.render(view_proj, self.debug_lines.items);
        stats.main_draw_calls += 1;
        stats.draw_calls += 1;
    }

    /// Non-creating pass lookup for an explicit target shape. The single
    /// twin slot serves every non-base shape; the base shape stays on the
    /// base pass. Render stage only: miss (null or shape drift) means the
    /// upload stage never built this shape — draw nothing.
    fn passForDraw(self: *PhysicsIntegration, samples: i32, color_format: sg.PixelFormat) ?*passes.DebugPass {
        if (self.debug_pass) |*dp| {
            if (dp.sample_count == samples and dp.color_format == color_format) return dp;
        } else return null;
        const twin = if (self.debug_pass_msaa) |*dp| dp else return null;
        if (twin.sample_count != samples) return null;
        if (twin.color_format != color_format) return null;
        return twin;
    }

    /// Debug pass variant matching the exact target shape (sample count +
    /// color format). UPLOAD STAGE ONLY: creates the variant on demand via
    /// `DebugPass.init`. The first created pass becomes the base; other
    /// shapes use the single twin slot. Null on allocation failure — the
    /// caller leaves the upload tag invalid and the draw stays silent.
    fn debugPassFor(self: *PhysicsIntegration, allocator: std.mem.Allocator, samples: i32, color_format: sg.PixelFormat) ?*passes.DebugPass {
        if (self.debug_pass) |*dp| {
            if (dp.sample_count == samples and dp.color_format == color_format) return dp;
        } else {
            self.debug_pass = passes.DebugPass.init(allocator, samples, color_format) catch null;
            return if (self.debug_pass) |*dp| dp else null;
        }
        if (self.debug_pass_msaa == null or self.debug_pass_msaa.?.sample_count != samples or self.debug_pass_msaa.?.color_format != color_format) {
            if (self.debug_pass_msaa) |*dp| dp.deinit();
            self.debug_pass_msaa = passes.DebugPass.init(allocator, samples, color_format) catch null;
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
    try t.expectEqual(@as(u64, 9), integ.build_seq.load(.acquire));
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

test "physics stageIntoSlot+latchSlotDebug freezes the build generation" {
    const t = std.testing;
    var fx = try makeDebugFixture(t.allocator);
    defer fx.integ.deinit(t.allocator);
    defer t.allocator.destroy(fx.mesh);
    var integ = &fx.integ;

    // Game-side build + slot freeze (wave 32 adopted path).
    integ.buildDebug(t.allocator, 9);
    var slot_lines: std.ArrayListUnmanaged(physics.DebugLine) = .empty;
    defer slot_lines.deinit(t.allocator);
    var slot_visible = false;
    integ.stageIntoSlot(t.allocator, &slot_lines, &slot_visible);
    try t.expect(slot_visible);
    try t.expectEqual(@as(usize, 12), slot_lines.items.len);
    const x0 = slot_lines.items[0].a.x;

    // Live + shared-store mutation after the freeze: the slot copy is immune.
    fx.mesh.position = Vec3.new(5, 0, 0);
    integ.show_debug = false;
    integ.build_lines.items[0].a.x += 100.0;
    integ.build_visible = false;
    try t.expect(slot_visible);
    try t.expectApproxEqAbs(x0, slot_lines.items[0].a.x, 1e-4);

    // Slot latch publishes the frozen generation (never the shared store)
    // and consumes the pending layer generation.
    integ.latchSlotDebug(t.allocator, slot_lines.items, slot_visible);
    try t.expectEqual(@as(u64, 9), integ.latched_seq);
    try t.expect(integ.prepared_visible);
    try t.expectEqual(@as(usize, 12), integ.prepared_lines.items.len);
    try t.expectApproxEqAbs(x0, integ.prepared_lines.items[0].a.x, 1e-4);

    // With the generation consumed, a shared-store latch falls back to the
    // live capture — hidden now, so coherent-empty, never a stale build.
    integ.latchDebug(t.allocator);
    try t.expect(!integ.prepared_visible);
    try t.expectEqual(@as(usize, 0), integ.prepared_lines.items.len);
}

test "physics stageIntoSlot newest wins; OOM fail-closes, slot latch publishes empty" {
    const t = std.testing;
    var fx = try makeDebugFixture(t.allocator);
    defer fx.integ.deinit(t.allocator);
    defer t.allocator.destroy(fx.mesh);
    var integ = &fx.integ;

    // Two builds before the freeze: newest wins (single store recomputed).
    integ.buildDebug(t.allocator, 1);
    const x0 = integ.build_lines.items[0].a.x;
    fx.mesh.position = Vec3.new(5, 0, 0);
    integ.buildDebug(t.allocator, 2);
    var slot_lines: std.ArrayListUnmanaged(physics.DebugLine) = .empty;
    defer slot_lines.deinit(t.allocator);
    var slot_visible = false;
    integ.stageIntoSlot(t.allocator, &slot_lines, &slot_visible);
    try t.expect(slot_visible);
    try t.expectApproxEqAbs(x0 + 5.0, slot_lines.items[0].a.x, 1e-4);

    // Unfunded freeze fail-closes the slot copy to coherent-empty +
    // invisible (staged-wins: the latch below publishes the empty).
    var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    slot_lines.clearAndFree(t.allocator);
    integ.stageIntoSlot(failing.allocator(), &slot_lines, &slot_visible);
    try t.expect(!slot_visible);
    try t.expectEqual(@as(usize, 0), slot_lines.items.len);
    integ.latchSlotDebug(t.allocator, slot_lines.items, slot_visible);
    try t.expect(!integ.prepared_visible);
    try t.expectEqual(@as(usize, 0), integ.prepared_lines.items.len);

    // Recovery: a funded freeze+latch publishes the full capture again.
    integ.stageIntoSlot(t.allocator, &slot_lines, &slot_visible);
    integ.latchSlotDebug(t.allocator, slot_lines.items, slot_visible);
    try t.expect(integ.prepared_visible);
    try t.expectEqual(@as(usize, 12), integ.prepared_lines.items.len);
    try t.expectApproxEqAbs(x0 + 5.0, integ.prepared_lines.items[0].a.x, 1e-4);
}

test "physics upload tag matches only the tagged shape" {
    const t = std.testing;
    var integ = PhysicsIntegration{};
    try t.expect(!integ.uploadTagMatches(1, .BGRA8));
    integ.prepared_upload_valid = true;
    integ.prepared_upload_samples = 4;
    integ.prepared_upload_format = .RGBA16F;
    try t.expect(integ.uploadTagMatches(4, .RGBA16F));
    try t.expect(!integ.uploadTagMatches(1, .RGBA16F));
    try t.expect(!integ.uploadTagMatches(4, .BGRA8));
    try t.expect(!integ.uploadTagMatches(2, .RGBA16F));
    integ.clearUploadTag();
    try t.expect(!integ.uploadTagMatches(4, .RGBA16F));
}

test "physics upload clears tag headless; target draw stays silent" {
    const t = std.testing;
    var integ = PhysicsIntegration{};
    defer integ.prepared_lines.deinit(t.allocator);
    // Stale tag from an earlier shape: any upload attempt clears it first,
    // even headless where no `sg.*` may fire and no pass may be created.
    integ.prepared_upload_valid = true;
    integ.prepared_upload_samples = 4;
    integ.prepared_upload_format = .RGBA16F;
    integ.uploadDebug(t.allocator, 4, .RGBA16F);
    try t.expect(!integ.prepared_upload_valid);
    try t.expect(integ.debug_pass == null);
    try t.expect(integ.debug_pass_msaa == null);
    // Draw with no tag (and no context) issues nothing, on either entry.
    var stats = stats_mod.SceneStats{};
    integ.renderDebugPrepared(Mat4.identity, 4, .RGBA16F, &stats);
    try t.expectEqual(stats_mod.SceneStats{}, stats);
    integ.renderDebugPrepared(Mat4.identity, 1, .BGRA8, &stats);
    try t.expectEqual(stats_mod.SceneStats{}, stats);
    // Upload shares the clear-first discipline headless.
    integ.prepared_upload_valid = true;
    integ.uploadDebug(t.allocator, 1, .BGRA8);
    try t.expect(!integ.prepared_upload_valid);
    try t.expect(integ.debug_pass == null);
}
