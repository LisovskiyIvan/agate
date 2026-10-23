//! Chaos runner: deterministic seeded stress host over the PUBLIC API only.
//!
//! Skeleton: `runtime_worker.zig` (Runtime facade, spawnWorker game loop,
//! renderFrame context side). Seeded scenario mix on the game thread (all
//! mutations game-side, per the documented threaded contract):
//!
//!   - create/destroy meshes (MeshBuilder shapes + occasional raw
//!     uploadGeometry path), rigid bodies on chaos meshes (destroyMesh drops
//!     the body), particle systems (one compute-mode when available),
//!     decals on live targets, camera / clustered-light churn;
//!   - threading toggles: `scene.threading.physics` inline<->dedicated and
//!     `runtime.setProducerExclusion` on/off (requested game-side, applied
//!     context-side — exclusion is context-thread-only by contract);
//!   - forced reuse/skip windows: production paused for random stretches so
//!     the context reuses/skips (exercises renderReuse + transient rewrites);
//!   - zone-churn bursts: rapid registry adds/removes while the context
//!     prepares (the engine-level essence of sandbox zone teleports).
//!
//! `--heavy-create` adds the H7 hammer: mesh-creation bursts while
//! production is paused (reuse frames with fresh pending creations) with a
//! compute-mode particle system emitting across the reuse windows.
//!
//! Modes:
//!   `--headless` — CPU-only staged path, no sg, fast CI mode. Own main
//!     loop: worker game thread + serial context loop until `--ops` reached,
//!     then quiesce + serial drain + verdict.
//!   windowed (default) — full GPU path through sokol App: flush / render /
//!     reuse on a real context. `--frames N` auto-quits (drain runs inside
//!     the last frame callbacks, context still alive).
//!
//! Verdict line at exit:
//!   `chaos: verdict=PASS seed=.. ops=.. destroys=.. reuses=.. kicks=.. live_allocs=..`
//! FAIL on: any panic, log errors > 0, begins != finishes + cancels,
//! pendingRetires not draining to 0 after quiesce + serial drain, tracked
//! allocator live allocations not returning to baseline after scene.deinit,
//! retainedCount ever exceeding pending_cap (capped_drops > 0 is fine when
//! counted — that is the point; uncounted leaks trip log_errors instead).
//!
//! Run: `zig build run-chaos -- --headless --ops 20000 --seed 1`

const std = @import("std");
const z = @import("agate");
const sokol = @import("sokol");
const sapp = sokol.app;
const slog = sokol.log;

// --- config ----------------------------------------------------------------

var cfg_seed: u64 = 1;
var cfg_ops: u64 = 20000;
var cfg_frames: u32 = 600;
var cfg_headless = false;
var cfg_heavy_create = false;

// --- observability ---------------------------------------------------------

var log_errors: std.atomic.Value(u32) = std.atomic.Value(u32).init(0);

pub const std_options: std.Options = .{ .logFn = chaosLogFn };

fn chaosLogFn(
    comptime level: std.log.Level,
    comptime scope: @EnumLiteral(),
    comptime format: []const u8,
    args: anytype,
) void {
    if (@backingInt(level) <= @backingInt(std.log.Level.err)) {
        _ = log_errors.fetchAdd(1, .monotonic);
    }
    std.log.defaultLog(level, scope, format, args);
}

fn rtLogger(tag: [*c]const u8, level: u32, item: u32, msg: [*c]const u8, line: u32, file: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    if (level <= 1) _ = log_errors.fetchAdd(1, .monotonic);
    slog.func(tag, level, item, msg, line, file, user_data);
}

// --- counting allocator (live-count + high-water wrapper) ------------------

const CountingAllocator = struct {
    parent: std.mem.Allocator,
    live_count: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    live_bytes: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    high_count: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    high_bytes: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

    fn allocator(self: *CountingAllocator) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &vtable };
    }

    fn noteAlloc(self: *CountingAllocator, len: usize) void {
        const n = self.live_count.fetchAdd(1, .monotonic) + 1;
        const b = self.live_bytes.fetchAdd(len, .monotonic) + len;
        self.bumpHigh(n, b);
    }

    fn noteFree(self: *CountingAllocator, len: usize) void {
        _ = self.live_count.fetchSub(1, .monotonic);
        _ = self.live_bytes.fetchSub(len, .monotonic);
    }

    fn bumpHigh(self: *CountingAllocator, n: usize, b: usize) void {
        var cur = self.high_count.load(.monotonic);
        while (n > cur) {
            cur = self.high_count.cmpxchgWeak(cur, n, .monotonic, .monotonic) orelse break;
        }
        var curb = self.high_bytes.load(.monotonic);
        while (b > curb) {
            curb = self.high_bytes.cmpxchgWeak(curb, b, .monotonic, .monotonic) orelse break;
        }
    }

    const vtable: std.mem.Allocator.VTable = .{
        .alloc = allocFn,
        .resize = resizeFn,
        .remap = remapFn,
        .free = freeFn,
    };

    fn allocFn(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        const ptr = self.parent.vtable.alloc(self.parent.ptr, len, alignment, ret_addr) orelse return null;
        self.noteAlloc(len);
        return ptr;
    }

    fn resizeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        if (!self.parent.vtable.resize(self.parent.ptr, memory, alignment, new_len, ret_addr)) return false;
        if (new_len > memory.len) {
            _ = self.live_bytes.fetchAdd(new_len - memory.len, .monotonic);
        } else {
            _ = self.live_bytes.fetchSub(memory.len - new_len, .monotonic);
        }
        const b = self.live_bytes.load(.monotonic);
        const n = self.live_count.load(.monotonic);
        self.bumpHigh(n, b);
        return true;
    }

    fn remapFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        const ptr = self.parent.vtable.remap(self.parent.ptr, memory, alignment, new_len, ret_addr) orelse return null;
        if (new_len > memory.len) {
            _ = self.live_bytes.fetchAdd(new_len - memory.len, .monotonic);
        } else {
            _ = self.live_bytes.fetchSub(memory.len - new_len, .monotonic);
        }
        return ptr;
    }

    fn freeFn(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *CountingAllocator = @ptrCast(@alignCast(ctx));
        self.noteFree(memory.len);
        self.parent.vtable.free(self.parent.ptr, memory, alignment, ret_addr);
    }
};

var gpa = std.heap.DebugAllocator(.{ .thread_safe = true }){};
var counting = CountingAllocator{ .parent = undefined };

// --- shared chaos state -----------------------------------------------------

const max_chaos_meshes: usize = 40;
const max_systems: usize = 4;

var scene: z.Scene = undefined;
var runtime: z.Runtime = z.Runtime.init();
var chaos: Chaos = .{};

/// Game-thread-owned scenario state. Only the `std.atomic` fields are
/// touched from the context thread (observability + stop/exclusion
/// requests); everything else is game-thread-only.
const Chaos = struct {
    rng: std.Random.DefaultPrng = std.Random.DefaultPrng.init(1),
    meshes: std.ArrayListUnmanaged(*z.Mesh) = .empty,
    mesh_serial: u64 = 0,
    systems: [max_systems]?*z.ParticleSystem = @splat(null),
    n_systems: usize = 0,
    compute_armed: bool = false,
    pause_ticks: u32 = 0,
    heavy_ticks: u32 = 0,
    sim_time: f32 = 0,
    last_ns: u64 = 0,
    extra_cams: u32 = 0,
    extra_lights: u32 = 0,
    physics_dedicated: bool = false,
    next_physics_toggle: u64 = 400,
    next_exclusion_toggle: u64 = 600,
    next_pause: u64 = 800,
    next_heavy: u64 = 1000,
    // Cross-thread: written game, read context.
    ops: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    destroys: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    creates: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    builds_obs: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
    done: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    want_exclusion: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
};

/// Static name pool: MeshBuilder borrows the name slice (owns_name=false),
/// so chaos names must outlive the meshes. 256 rotating slots are enough —
/// live meshes may share a slot's text, but chaos never looks meshes up by
/// name, and no allocation means no lifetime to track.
var name_bufs: [256][24]u8 = undefined;

fn chaosName(serial: u64) []const u8 {
    const slot = serial % name_bufs.len;
    return std.fmt.bufPrint(&name_bufs[slot], "chaos_{d}", .{serial}) catch "chaos_x";
}

fn initNamePool() void {
    var i: u64 = 0;
    while (i < name_bufs.len) : (i += 1) {
        _ = std.fmt.bufPrint(&name_bufs[i], "chaos_{d}", .{i}) catch {};
    }
}

// --- game-side scenario ops --------------------------------------------------

fn countOp() void {
    _ = chaos.ops.fetchAdd(1, .monotonic);
}

fn createChaosMesh() void {
    if (chaos.meshes.items.len >= max_chaos_meshes) {
        destroyRandomMesh();
        if (chaos.meshes.items.len >= max_chaos_meshes) return;
    }
    const r = chaos.rng.random();
    const serial = chaos.mesh_serial;
    chaos.mesh_serial += 1;
    const name = chaosName(serial);
    const shape = r.intRangeAtMost(u8, 0, 9);
    const mesh: ?*z.Mesh = switch (shape) {
        0...3 => z.MeshBuilder.createBox(&scene, name, .{ .size = 0.3 + r.floatNorm(f32) * 0.2 + 0.6 }) catch null,
        4...5 => z.MeshBuilder.createSphere(&scene, name, .{ .diameter = 0.8, .segments = 8 }) catch null,
        6 => z.MeshBuilder.createPlane(&scene, name, .{}) catch null,
        7 => z.MeshBuilder.createCylinder(&scene, name, .{ .tessellation = 8 }) catch null,
        8 => createRawTriangle(name) catch null,
        else => z.MeshBuilder.createTorus(&scene, name, .{ .tessellation = 6 }) catch null,
    };
    countOp();
    if (mesh) |m| {
        m.position = z.Vec3.new(r.floatNorm(f32) * 4.0, 0.5 + r.floatNorm(f32), r.floatNorm(f32) * 4.0);
        chaos.meshes.append(scene.allocator, m) catch {
            scene.destroyMesh(m);
            _ = chaos.destroys.fetchAdd(1, .monotonic);
            return;
        };
        _ = chaos.creates.fetchAdd(1, .monotonic);
        // Occasionally pin a rigid body to the fresh mesh (outside the
        // physics window: ops run before Scene.update in the tick).
        if (r.intRangeAtMost(u8, 0, 3) == 0) {
            _ = scene.createRigidBody(m, .box, 1.0) catch {};
            countOp();
        }
    }
}

/// Raw uploadGeometry path (hand-built triangle): exercises the deferred
/// pending-mesh creation flow directly, the H7 corner (a) material.
fn createRawTriangle(name: []const u8) !*z.Mesh {
    const alloc = scene.allocator;
    const verts = try alloc.alloc(z.Vertex, 3);
    const positions: [3][3]f32 = .{ .{ -0.5, 0, 0 }, .{ 0.5, 0, 0 }, .{ 0, 1, 0 } };
    for (verts, 0..) |*v, i| {
        v.* = .{
            .position = positions[i],
            .normal = .{ 0, 1, 0 },
            .color = .{ 1, 1, 1, 1 },
            .uv = .{ 0, 0 },
        };
    }
    const idx = try alloc.alloc(u32, 3);
    errdefer alloc.free(idx);
    idx[0] = 0;
    idx[1] = 1;
    idx[2] = 2;
    var data = z.GeometryData{
        .vertices = verts,
        .indices = idx,
        .bounds = z.BoundingBox.init(z.Vec3.new(-0.5, 0, 0), z.Vec3.new(0.5, 1, 0)),
    };
    defer data.deinit(alloc);
    return z.mesh.uploadGeometry(&scene, name, data);
}

fn destroyRandomMesh() void {
    if (chaos.meshes.items.len == 0) return;
    const r = chaos.rng.random();
    const idx = r.intRangeAtMost(usize, 0, chaos.meshes.items.len - 1);
    const m = chaos.meshes.swapRemove(idx);
    scene.destroyMesh(m);
    _ = chaos.destroys.fetchAdd(1, .monotonic);
    countOp();
}

fn churnParticles() void {
    const r = chaos.rng.random();
    if (chaos.n_systems < max_systems and r.intRangeAtMost(u8, 0, 2) == 0) {
        const serial = chaos.mesh_serial;
        chaos.mesh_serial += 1;
        const ps = scene.createParticleSystem(chaosName(serial), 64) catch {
            countOp();
            return;
        };
        ps.is_emitting = true;
        ps.emit_rate = 60.0;
        // First system tries compute mode: across reuse windows the
        // render-time compute path re-runs while the rewrite must exclude
        // the state clear — the H7 corner (b) hammer.
        if (!chaos.compute_armed) {
            chaos.compute_armed = true;
            ps.setSimulationMode(.compute) catch {};
        }
        chaos.systems[chaos.n_systems] = ps;
        chaos.n_systems += 1;
        _ = chaos.creates.fetchAdd(1, .monotonic);
    } else if (chaos.n_systems > 0) {
        const i = r.intRangeAtMost(usize, 0, chaos.n_systems - 1);
        if (chaos.systems[i]) |ps| ps.is_emitting = !ps.is_emitting;
    }
    countOp();
}

fn churnDecal() void {
    if (chaos.meshes.items.len == 0) {
        countOp();
        return;
    }
    const r = chaos.rng.random();
    const target = chaos.meshes.items[r.intRangeAtMost(usize, 0, chaos.meshes.items.len - 1)];
    const serial = chaos.mesh_serial;
    chaos.mesh_serial += 1;
    const d = z.MeshBuilder.createDecal(&scene, chaosName(serial), target, .{
        .position = z.Vec3.new(0, 0.5, 0),
        .normal = z.Vec3.new(0, 1, 0),
        .size = z.Vec3.new(0.6, 0.6, 0.6),
    }) catch {
        countOp();
        return;
    };
    if (chaos.meshes.items.len >= max_chaos_meshes) {
        scene.destroyMesh(d);
        _ = chaos.destroys.fetchAdd(1, .monotonic);
    } else {
        chaos.meshes.append(scene.allocator, d) catch {
            scene.destroyMesh(d);
            _ = chaos.destroys.fetchAdd(1, .monotonic);
            countOp();
            return;
        };
        _ = chaos.creates.fetchAdd(1, .monotonic);
    }
    countOp();
}

fn churnCameraLight() void {
    const r = chaos.rng.random();
    if (r.boolean()) {
        if (chaos.extra_cams < 3) {
            const serial = chaos.mesh_serial;
            chaos.mesh_serial += 1;
            const cam = z.Camera{ .arc_rotate = z.ArcRotateCamera.init(chaosName(serial), .{
                .alpha = r.floatNorm(f32),
                .beta = std.math.pi / 3.0,
                .radius = 9.0,
                .target = z.Vec3.zero,
            }) };
            if (scene.addCamera(.{ .name = chaosName(serial), .camera = cam })) |_| {
                chaos.extra_cams += 1;
            } else |_| {}
        } else if (scene.cameras.items.len > 1) {
            scene.removeCamera(scene.cameras.items.len - 1);
            chaos.extra_cams -= 1;
        }
    } else {
        if (chaos.extra_lights < 4) {
            if (scene.addClusteredPointLight(
                z.Vec3.new(r.floatNorm(f32) * 3.0, 3.0, r.floatNorm(f32) * 3.0),
                .{ .color = z.Color3.new(1.0, 0.85, 0.6), .intensity = 1.5, .radius = 7.0 },
            )) |_| {
                chaos.extra_lights += 1;
            } else |_| {}
        } else {
            scene.removeClusteredPointLight(0);
            chaos.extra_lights -= 1;
        }
    }
    countOp();
}

/// Zone-churn burst: rapid registry adds/removes in one tick while the
/// context may be preparing — the engine-level essence of sandbox zone
/// teleports (commit guards hold coherence; live list is never read by the
/// context, only frozen slots).
fn zoneChurnBurst() void {
    const r = chaos.rng.random();
    const n = r.intRangeAtMost(u8, 4, 10);
    var i: u8 = 0;
    while (i < n) : (i += 1) {
        if (r.boolean()) {
            createChaosMesh();
        } else {
            destroyRandomMesh();
        }
    }
}

fn maybeToggleThreading() void {
    const ops = chaos.ops.load(.monotonic);
    if (ops >= chaos.next_physics_toggle) {
        chaos.next_physics_toggle = ops + 400 + chaos.rng.random().intRangeAtMost(u64, 0, 300);
        chaos.physics_dedicated = !chaos.physics_dedicated;
        scene.threading.physics = if (chaos.physics_dedicated) .dedicated else .@"inline";
        countOp();
    }
    if (ops >= chaos.next_exclusion_toggle) {
        chaos.next_exclusion_toggle = ops + 500 + chaos.rng.random().intRangeAtMost(u64, 0, 400);
        chaos.want_exclusion.store(!chaos.want_exclusion.load(.monotonic), .monotonic);
        countOp();
    }
    if (ops >= chaos.next_pause) {
        const r = chaos.rng.random();
        chaos.next_pause = ops + 700 + r.intRangeAtMost(u64, 0, 500);
        chaos.pause_ticks = r.intRangeAtMost(u32, 15, 40);
        countOp();
    }
    if (cfg_heavy_create and ops >= chaos.next_heavy) {
        chaos.next_heavy = ops + 1000;
        chaos.heavy_ticks = 30;
        chaos.pause_ticks = @max(chaos.pause_ticks, 30);
        countOp();
    }
}

/// One seeded game tick: scenario ops, live-transform wiggle (documented
/// game-side mutation), then Scene.update. Production publish happens in
/// the caller (skipped during pause windows to force reuse/skip frames).
fn chaosTick(c: *Chaos) void {
    const now = z.jobs.monoNs();
    var dt: f32 = if (c.last_ns == 0) 1.0 / 60.0 else @floatCast(@as(f64, @floatFromInt(now -% c.last_ns)) / 1_000_000_000.0);
    c.last_ns = now;
    dt = @min(dt, 0.1);
    c.sim_time += dt;

    const r = c.rng.random();
    if (c.heavy_ticks > 0) {
        // H7 hammer: mesh-creation bursts while production is paused, so
        // reuse frames land with fresh pending creations in flight.
        c.heavy_ticks -= 1;
        var i: u8 = 0;
        const n = r.intRangeAtMost(u8, 3, 6);
        while (i < n) : (i += 1) createChaosMesh();
    } else {
        const n = r.intRangeAtMost(u8, 2, 6);
        var i: u8 = 0;
        while (i < n) : (i += 1) {
            const pick = r.intRangeAtMost(u8, 0, 99);
            switch (pick) {
                0...34 => createChaosMesh(),
                35...54 => destroyRandomMesh(),
                55...64 => churnParticles(),
                65...72 => churnDecal(),
                73...79 => churnCameraLight(),
                80...84 => zoneChurnBurst(),
                else => countOp(),
            }
        }
        maybeToggleThreading();
    }

    for (c.meshes.items) |m| {
        m.rotation.y += dt * 0.7;
        m.position.y += @sin(c.sim_time * 1.4 + m.position.x) * dt * 0.2;
    }

    scene.update(dt) catch |err| {
        _ = log_errors.fetchAdd(1, .monotonic);
        std.debug.print("chaos: scene update error: {s}\n", .{@errorName(err)});
    };
}

fn drainTick(_: *Chaos) void {
    scene.update(1.0 / 60.0) catch |err| {
        _ = log_errors.fetchAdd(1, .monotonic);
        std.debug.print("chaos: drain update error: {s}\n", .{@errorName(err)});
    };
}

/// Game loop body (worker thread): tick, then produce unless paused or
/// done. Mirrors Runtime.update's exclusion discipline manually so pauses
/// skip the publish without skipping the tick.
fn gameLoop() void {
    while (runtime.shouldRun() and !chaos.done.load(.acquire)) {
        scene.publishFrameSnapshot(800.0 / 600.0, 800, 600);
        const excl = runtime.producer_exclusion.load(.acquire);
        if (excl) runtime.gameLock();
        chaosTick(&chaos);
        var built = false;
        if (chaos.pause_ticks > 0) {
            chaos.pause_ticks -= 1;
        } else {
            built = runtime.produceBuild(&scene);
        }
        if (excl) runtime.gameUnlock();
        if (built) _ = chaos.builds_obs.fetchAdd(1, .monotonic);
    }
}

// --- scene setup (headless fixture vs windowed full init) ---------------------

/// Windowed full init (live sg context, like runtime_worker).
fn setupSceneWindowed() void {
    scene = z.Scene.init(counting.allocator());
    scene.active_camera = .{ .arc_rotate = z.ArcRotateCamera.init("cam", .{
        .alpha = std.math.pi / 4.0,
        .beta = std.math.pi / 3.0,
        .radius = 9.0,
        .target = z.Vec3.zero,
    }) };
    scene.clear_color = z.Color4.new(0.07, 0.09, 0.13, 1.0);
    _ = scene.createHemisphericLight("hemi", .{
        .direction = z.Vec3.new(0.5, 1.0, 0.3),
        .diffuse = z.Color3.white,
        .ground_color = z.Color3.new(0.18, 0.20, 0.26),
        .intensity = 1.0,
    });
    // One clustered light so the reuse transient-write replay has storage
    // binds to exercise even before churn adds more (same rationale as the
    // runtime-worker exclusion gate).
    _ = scene.addClusteredPointLight(z.Vec3.new(0.0, 4.0, 0.0), .{
        .color = z.Color3.new(1.0, 0.85, 0.6),
        .intensity = 2.0,
        .radius = 9.0,
    }) catch {};
    _ = scene.enablePhysics(null);
}

/// Headless CPU-only scene (no sg): the unit suite's fixture
/// (`z.testing.testScene`) plus the documented headless full-path pattern
/// (stage1Scene in scene/tests_util.zig): culling off, faked GPU-init flag
/// (GPU halves publish bounds/count without touching sg.*), shadows off so
/// prepareInto never runs against the fixture's unset shadow pass.
fn setupSceneHeadless() void {
    const alloc = counting.allocator();
    scene = z.testing.testScene(alloc);
    scene.enable_frustum_culling = false;
    scene.enable_occlusion_culling = false;
    scene.default_white_texture.view.id = 1;
    scene.shadows.enabled = false;
    const cam = z.Camera{ .arc_rotate = z.ArcRotateCamera.init("cam", .{
        .alpha = std.math.pi / 4.0,
        .beta = std.math.pi / 3.0,
        .radius = 9.0,
        .target = z.Vec3.zero,
    }) };
    _ = scene.addCamera(.{ .name = "cam", .camera = cam }) catch {};
    _ = scene.createHemisphericLight("hemi", .{
        .direction = z.Vec3.new(0.5, 1.0, 0.3),
        .diffuse = z.Color3.white,
        .ground_color = z.Color3.new(0.18, 0.20, 0.26),
        .intensity = 1.0,
    });
    _ = scene.addClusteredPointLight(z.Vec3.new(0.0, 4.0, 0.0), .{
        .color = z.Color3.new(1.0, 0.85, 0.6),
        .intensity = 2.0,
        .radius = 9.0,
    }) catch {};
    _ = scene.enablePhysics(null);
}

/// Fixture teardown (mirrors scene/tests_util.zig stage1Deinit + the lists
/// chaos grows). Only lists chaos actually allocated through are deinited;
/// never-touched GPU passes (shadows/sky/postfx/forward) stay unset, exactly
/// like the unit fixtures. `Scene.deinit` must NOT be called here (it
/// assumes full-init GPU resources).
fn deinitHeadlessScene() void {
    const alloc = counting.allocator();
    for (scene.particles.systems.items) |ps| {
        ps.deinit();
        alloc.destroy(ps);
    }
    scene.particles.systems.deinit(alloc);
    scene.particles.frame.deinit(alloc);
    scene.particles.build_frame.deinit(alloc);
    // NOTE: particles.pass stays unset (never touched headless) — no deinit.
    scene.physics.deinit(alloc);
    scene.meshes.deinit(alloc);
    scene.outline_meshes.deinit(alloc);
    scene.draws.deinit(alloc);
    scene.gpu_retire.deinit(alloc);
    scene.cameras.deinit(alloc);
    scene.lights.deinit(alloc);
    scene.ui_frame.deinit(alloc);
    scene.profiler.deinit();
}

// --- verdict -----------------------------------------------------------------

const Verdict = struct {
    pass: bool,
    ops: u64,
    destroys: u64,
    creates: u64,
    reuses: u64,
    kicks: u64,
    joins: u64,
    fallbacks: u64,
    begins: u64,
    finishes: u64,
    cancels: u64,
    builds: u64,
    skips: u64,
    pending: usize,
    capped: u64,
    dups: u64,
    live: usize,
    log_errs: u32,
    fail_why: []const u8,
};

fn computeVerdict() Verdict {
    const m = runtime.metricsSnapshot();
    const wc = scene.physics.workerCounters();
    const pending = scene.pendingRetires();
    const capped = scene.gpu_retire.cappedDropCount();
    const dups = scene.gpu_retire.duplicateDropCount();
    const live = counting.live_count.load(.monotonic);
    const errs = log_errors.load(.monotonic);
    var v = Verdict{
        .pass = true,
        .ops = chaos.ops.load(.monotonic),
        .destroys = chaos.destroys.load(.monotonic),
        .creates = chaos.creates.load(.monotonic),
        .reuses = m.reuses,
        .kicks = wc.kicks,
        .joins = wc.joins,
        .fallbacks = wc.fallbacks,
        .begins = m.begins,
        .finishes = m.finishes,
        .cancels = m.cancels,
        .builds = m.producer_builds,
        .skips = m.producer_skips,
        .pending = pending,
        .capped = capped,
        .dups = dups,
        .live = live,
        .log_errs = errs,
        .fail_why = "",
    };
    if (errs != 0) {
        v.pass = false;
        v.fail_why = "log_errors>0";
    } else if (m.begins != m.finishes + m.cancels) {
        v.pass = false;
        v.fail_why = "begins!=finishes+cancels";
    } else if (pending != 0) {
        v.pass = false;
        v.fail_why = "pendingRetires!=0";
    } else if (scene.gpu_retire.retainedCount() > scene.gpu_retire.pending_cap) {
        v.pass = false;
        v.fail_why = "retained>cap";
    }
    return v;
}

/// Fold the post-teardown allocator balance into the verdict: the counting
/// wrapper must be back at baseline (0) once the scene is torn down.
fn foldLiveBalance(v: *Verdict) void {
    v.live = counting.live_count.load(.monotonic);
    if (v.live != 0 and v.pass) {
        v.pass = false;
        v.fail_why = "live_allocs!=baseline";
    }
}

fn printVerdict(v: Verdict) void {
    std.debug.print(
        "chaos: verdict={s} seed={d} ops={d} destroys={d} creates={d} reuses={d} kicks={d} joins={d} fallbacks={d} live_allocs={d}\n" ++
            "chaos: builds={d} skips={d} begins={d} finishes={d} cancels={d} pending_retires={d} capped_drops={d} dup_drops={d} log_errors={d} fail={s}\n",
        .{
            if (v.pass) "PASS" else "FAIL", cfg_seed,   v.ops,       v.destroys, v.creates, v.reuses,
            v.kicks,                        v.joins,    v.fallbacks, v.live,     v.builds,  v.skips,
            v.begins,                       v.finishes, v.cancels,   v.pending,  v.capped,  v.dups,
            v.log_errs,                     v.fail_why,
        },
    );
}

/// Quiesce the worker, destroy leftover chaos meshes/cameras on this
/// thread, then serial produce+prepare cycles until pendingRetires drains
/// (proves epochs complete without relying on deinit's unconditional
/// drain). Destroys the chaos mesh ITEMS; the caller deinits the list
/// itself. Returns false if the drain did not converge.
fn quiesceAndDrain() bool {
    runtime.quiesce();
    // Tear down chaos-owned registry entries first so no new retires
    // appear mid-drain. Game thread is joined: this thread owns the scene.
    while (chaos.meshes.items.len > 0) {
        const m = chaos.meshes.swapRemove(chaos.meshes.items.len - 1);
        scene.destroyMesh(m);
    }
    while (scene.cameras.items.len > 1) scene.removeCamera(scene.cameras.items.len - 1);
    var i: u32 = 0;
    while (i < 300) : (i += 1) {
        scene.publishFrameSnapshot(800.0 / 600.0, 800, 600);
        _ = runtime.update(&scene, &chaos, drainTick);
        _ = runtime.renderFrame(&scene);
        if (scene.pendingRetires() == 0) break;
    }
    return scene.pendingRetires() == 0;
}

fn dumpSecondLine(tag: []const u8) void {
    const m = runtime.metricsSnapshot();
    const pend = scene.pendingRetires();
    std.debug.print(
        "chaos[{s}]: ops={d} destroys={d} creates={d} builds={d}/{d} begins={d} fin={d} cancels={d} reuses={d} busy={d} pending_retires={d} capped={d} live={d}/{d}\n",
        .{
            tag,
            chaos.ops.load(.monotonic),
            chaos.destroys.load(.monotonic),
            chaos.creates.load(.monotonic),
            m.producer_builds,
            chaos.builds_obs.load(.monotonic),
            m.begins,
            m.finishes,
            m.cancels,
            m.reuses,
            m.begin_busy,
            pend,
            scene.gpu_retire.cappedDropCount(),
            counting.live_count.load(.monotonic),
            counting.high_count.load(.monotonic),
        },
    );
}

// --- headless mode -----------------------------------------------------------

fn runHeadless() u8 {
    setupSceneHeadless();
    chaos.rng = std.Random.DefaultPrng.init(cfg_seed);
    const spawned = runtime.spawnWorker(gameLoop);
    std.debug.print("chaos: headless, worker {s}, seed={d} ops_target={d}\n", .{
        if (spawned) "spawned" else "spawn failed",
        cfg_seed,
        cfg_ops,
    });
    if (!spawned) {
        std.debug.print("chaos: worker spawn failed, FAIL\n", .{});
        return 1;
    }
    var last_dump = z.jobs.monoNs();
    while (chaos.ops.load(.acquire) < cfg_ops) {
        // Apply exclusion requests context-side (contract: context only).
        const want = chaos.want_exclusion.load(.acquire);
        if (want != runtime.producer_exclusion.load(.acquire)) runtime.setProducerExclusion(want);
        _ = runtime.renderFrame(&scene);
        const now = z.jobs.monoNs();
        if (now -% last_dump >= 1_000_000_000) {
            last_dump = now;
            dumpSecondLine("headless");
        }
    }
    chaos.done.store(true, .release);
    const drained = quiesceAndDrain();
    chaos.meshes.deinit(counting.allocator());
    var v = computeVerdict();
    if (!drained and v.pass) {
        v.pass = false;
        v.fail_why = "drain-stall";
        v.pending = scene.pendingRetires();
    }
    deinitHeadlessScene();
    runtime.deinit();
    // Balance check after teardown: the counting wrapper must be back at
    // baseline (0).
    foldLiveBalance(&v);
    printVerdict(v);
    const leak = gpa.deinit();
    if (v.pass and leak == .leak) {
        std.debug.print("chaos: verdict=FAIL(gpa leak)\n", .{});
        return 1;
    }
    return if (v.pass) 0 else 1;
}

// --- windowed mode (sokol App, full GPU path) --------------------------------

var frame_count: u32 = 0;
var prepared_frames: u32 = 0;
var reused_frames: u32 = 0;
var skipped_frames: u32 = 0;
var busy_frames: u32 = 0;
var worker_running = false;
var draining = false;
var drain_iters: u32 = 0;
var verdict_pass = false;
var last_dump_ms: u64 = 0;

var win_w: std.atomic.Value(u32) = std.atomic.Value(u32).init(800);
var win_h: std.atomic.Value(u32) = std.atomic.Value(u32).init(600);

fn appInitCb(_: *z.App) void {
    setupSceneWindowed();
    chaos.rng = std.Random.DefaultPrng.init(cfg_seed);
    worker_running = runtime.spawnWorker(gameLoop);
    std.debug.print("chaos: windowed, worker {s}, seed={d} frames={d} heavy_create={}\n", .{
        if (worker_running) "spawned" else "spawn failed (inline serial fallback)",
        cfg_seed,
        cfg_frames,
        cfg_heavy_create,
    });
    last_dump_ms = z.jobs.monoNs();
}

fn appFrameCb(_: *z.App) void {
    // Apply exclusion requests context-side.
    const want = chaos.want_exclusion.load(.acquire);
    if (want != runtime.producer_exclusion.load(.acquire)) runtime.setProducerExclusion(want);

    if (!worker_running) {
        scene.publishFrameSnapshot(800.0 / 600.0, 800, 600);
        const excl = runtime.producer_exclusion.load(.acquire);
        if (excl) runtime.gameLock();
        chaosTick(&chaos);
        if (chaos.pause_ticks > 0) {
            chaos.pause_ticks -= 1;
        } else {
            _ = runtime.produceBuild(&scene);
        }
        if (excl) runtime.gameUnlock();
    }

    switch (runtime.renderFrame(&scene)) {
        .prepared => prepared_frames += 1,
        .reused => reused_frames += 1,
        .skipped => skipped_frames += 1,
        .busy => busy_frames += 1,
    }
    frame_count += 1;

    const now = z.jobs.monoNs();
    if (now -% last_dump_ms >= 1_000_000_000) {
        last_dump_ms = now;
        dumpSecondLine("windowed");
    }

    if (!draining and cfg_frames != 0 and frame_count >= cfg_frames) {
        // Quiesce inside the frame callback (context alive), then serial
        // drain; quit when converged or bounded.
        draining = true;
        chaos.done.store(true, .release);
        runtime.quiesce();
        while (chaos.meshes.items.len > 0) {
            const m = chaos.meshes.swapRemove(chaos.meshes.items.len - 1);
            scene.destroyMesh(m);
        }
        std.debug.print("chaos: draining...\n", .{});
    }
    if (draining) {
        scene.publishFrameSnapshot(800.0 / 600.0, 800, 600);
        _ = runtime.update(&scene, &chaos, drainTick);
        _ = runtime.renderFrame(&scene);
        drain_iters += 1;
        if (scene.pendingRetires() == 0 or drain_iters >= 300) {
            var v = computeVerdict();
            if (scene.pendingRetires() != 0 and v.pass) {
                v.pass = false;
                v.fail_why = "drain-stall";
            }
            // Reuses observed during drain count too; fold drain frames in.
            // The live balance folds in after scene.deinit (cleanup);
            // stash the pre-teardown verdict for it.
            drain_verdict = v;
            chaos.meshes.deinit(counting.allocator());
            sapp.quit();
        }
    }
}

var drain_verdict: ?Verdict = null;

fn appCleanupCb(_: *z.App) void {
    // The live-allocator balance folds in only after scene.deinit below;
    // every path stashes a pre-teardown verdict first.
    if (drain_verdict == null) {
        if (worker_running) {
            // Early quit (ESC): still quiesce + drain bounded for a fair verdict.
            chaos.done.store(true, .release);
            const drained = quiesceAndDrain();
            chaos.meshes.deinit(counting.allocator());
            var v = computeVerdict();
            if (!drained and v.pass) {
                v.pass = false;
                v.fail_why = "drain-stall";
            }
            drain_verdict = v;
        } else {
            // Inline serial fallback (spawn failed): no worker to join.
            chaos.meshes.deinit(counting.allocator());
            drain_verdict = computeVerdict();
        }
    }
    var v = drain_verdict.?;
    std.debug.print(
        "chaos: frames={d} prepared={d} reused={d} skipped={d} busy={d}\n",
        .{ frame_count, prepared_frames, reused_frames, skipped_frames, busy_frames },
    );
    scene.deinit();
    runtime.deinit();
    foldLiveBalance(&v);
    printVerdict(v);
    verdict_pass = v.pass;
    const leak = gpa.deinit();
    if (leak == .leak) {
        std.debug.print("chaos: gpa leak\n", .{});
        verdict_pass = false;
    }
    if (!verdict_pass) std.process.exit(1);
}

fn appEventCb(_: *z.App, ev: [*c]const sapp.Event) void {
    switch (ev.*.type) {
        .RESIZED => {
            win_w.store(@intCast(@max(1, sapp.width())), .monotonic);
            win_h.store(@intCast(@max(1, sapp.height())), .monotonic);
        },
        .KEY_DOWN => switch (ev.*.key_code) {
            .ESCAPE => sapp.quit(),
            else => {},
        },
        else => {},
    }
}

fn runWindowed() void {
    var app = z.App.init(counting.allocator(), .{
        .title = "agate chaos-runner (seeded stress host)",
        .width = 800,
        .height = 600,
        .sample_count = 1,
        .advanced = .{
            .gl_major = 4,
            .gl_minor = 3,
            .logger = rtLogger,
            .shader_pool_size = 128,
        },
    }, .{
        .init = appInitCb,
        .frame = appFrameCb,
        .cleanup = appCleanupCb,
        .event = appEventCb,
    });
    app.run();
}

// --- args + main -------------------------------------------------------------

fn parseArgs(allocator: std.mem.Allocator, args: std.process.Args) void {
    var it = std.process.Args.Iterator.initAllocator(args, allocator) catch return;
    defer it.deinit();
    _ = it.next();
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--headless")) {
            cfg_headless = true;
        } else if (std.mem.eql(u8, arg, "--heavy-create")) {
            cfg_heavy_create = true;
        } else if (std.mem.eql(u8, arg, "--seed")) {
            if (it.next()) |n| cfg_seed = std.fmt.parseInt(u64, n, 10) catch cfg_seed;
        } else if (std.mem.startsWith(u8, arg, "--seed=")) {
            cfg_seed = std.fmt.parseInt(u64, arg["--seed=".len..], 10) catch cfg_seed;
        } else if (std.mem.eql(u8, arg, "--ops")) {
            if (it.next()) |n| cfg_ops = std.fmt.parseInt(u64, n, 10) catch cfg_ops;
        } else if (std.mem.startsWith(u8, arg, "--ops=")) {
            cfg_ops = std.fmt.parseInt(u64, arg["--ops=".len..], 10) catch cfg_ops;
        } else if (std.mem.eql(u8, arg, "--frames")) {
            if (it.next()) |n| cfg_frames = std.fmt.parseInt(u32, n, 10) catch cfg_frames;
        } else if (std.mem.startsWith(u8, arg, "--frames=")) {
            cfg_frames = std.fmt.parseInt(u32, arg["--frames=".len..], 10) catch cfg_frames;
        }
    }
    if (cfg_ops == 0) cfg_ops = 20000;
    if (cfg_frames == 0) cfg_frames = 600;
}

pub fn main(minimal: std.process.Init.Minimal) void {
    counting.parent = gpa.allocator();
    initNamePool();
    parseArgs(counting.allocator(), minimal.args);
    if (cfg_headless) {
        std.process.exit(runHeadless());
    } else {
        runWindowed();
    }
}
