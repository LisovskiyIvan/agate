//! Tests for configured dedicated physics stepping
//! (`ThreadingOptions.physics`, fork-join inside `Scene.update`).
//!
//! - determinism methodology: inline-vs-inline bit identity first, then
//!   inline-vs-dedicated bitwise comparison of body mesh transforms;
//! - default config spawns no thread and counts nothing;
//! - forced spawn failure degrades to correct inline with a fallback bump;
//! - lifecycle stress (create/remove between updates, deinit after update);
//! - parallel-window tripwire panics on a game-thread mutation (fork-based:
//!   no panic-capture helper exists in the repo, so the child panics and
//!   the parent asserts abnormal termination; POSIX-only, skipped on
//!   Windows/wasm/single-threaded).

const std = @import("std");
const builtin = @import("builtin");
const math = @import("math");
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const Mesh = @import("../mesh.zig").Mesh;
const Scene = @import("../scene.zig").Scene;
const testScene = @import("../testing.zig").testScene;
const PhysicsWorld = @import("../physics.zig").PhysicsWorld;
const ThreadingOptions = @import("../threading.zig").ThreadingOptions;

const stack_boxes = 8;
const fixed_dt: f32 = 1.0 / 60.0;
const determinism_steps: usize = 120;

fn boxMesh(alloc: std.mem.Allocator, name: []const u8, pos: Vec3) !*Mesh {
    const m = try alloc.create(Mesh);
    m.* = .{
        .name = name,
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = pos,
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.5, -0.5, -0.5), Vec3.new(0.5, 0.5, 0.5)),
    };
    return m;
}

/// Fallen-stack scene: `stack_boxes` dynamic boxes dropped from staggered
/// heights over the default ground plane. Meshes are caller-owned (NOT in
/// `scene.meshes`: bodies hold raw mesh pointers, exactly like the
/// physics-only `Ragdoll`/`RaycastVehicle` bare mode).
fn makeStackScene(alloc: std.mem.Allocator, scene: *Scene, meshes: *std.ArrayListUnmanaged(*Mesh)) !void {
    scene.* = testScene(alloc);
    _ = scene.enablePhysics(null);
    var i: usize = 0;
    while (i < stack_boxes) : (i += 1) {
        const fi: f32 = @floatFromInt(i);
        const m = try boxMesh(alloc, "tbox", Vec3.new(fi * 0.35 - 1.2, 2.0 + fi * 1.15, 0.25 * @sin(fi)));
        errdefer alloc.destroy(m);
        try meshes.append(alloc, m);
        _ = try scene.createRigidBody(m, .box, 1.0);
    }
}

fn deinitStackScene(scene: *Scene, meshes: *std.ArrayListUnmanaged(*Mesh)) void {
    const alloc = scene.allocator;
    scene.physics.deinit(alloc);
    for (meshes.items) |m| alloc.destroy(m);
    meshes.deinit(alloc);
    scene.lights.deinit(alloc);
    scene.cameras.deinit(alloc);
    scene.draws.deinit(alloc);
    scene.gpu_retire.deinit(alloc);
    scene.profiler.deinit();
}

/// Deinit for scenes that created particle systems through `Scene` but do
/// not run the full `Scene.deinit` (testScene's GPU passes are undefined).
fn deinitParticleScene(scene: *Scene) void {
    const alloc = scene.allocator;
    const sim = scene.sim_allocator;
    for (scene.particles.systems.items) |ps| {
        ps.deinit();
        sim.destroy(ps);
    }
    scene.particles.systems.deinit(sim);
    scene.particles.frame.deinit(sim);
    scene.particles.build_frame.deinit(sim);
    scene.physics.deinit(alloc);
    scene.lights.deinit(alloc);
    scene.cameras.deinit(alloc);
    scene.draws.deinit(alloc);
    scene.gpu_retire.deinit(alloc);
    scene.profiler.deinit();
}

/// Snapshots every body mesh transform as raw bytes for bitwise comparison.
fn snapshotTransforms(alloc: std.mem.Allocator, scene: *Scene) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(alloc);
    const pw = scene.physics.getWorld() orelse return out.toOwnedSlice(alloc);
    for (pw.bodies.items) |b| {
        try out.appendSlice(alloc, std.mem.asBytes(&b.mesh.position));
        try out.appendSlice(alloc, std.mem.asBytes(&b.mesh.rotation));
    }
    return out.toOwnedSlice(alloc);
}

fn runUpdates(scene: *Scene, n: usize) !void {
    var i: usize = 0;
    while (i < n) : (i += 1) try scene.update(fixed_dt);
}

test "threading: inline run-to-run bit identity, then inline vs dedicated bitwise equal" {
    var gpa = std.heap.DebugAllocator(.{ .thread_safe = true }){};
    defer std.testing.expect(gpa.deinit() == .ok) catch {};
    const alloc = gpa.allocator();

    var meshes_a: std.ArrayListUnmanaged(*Mesh) = .empty;
    var scene_a: Scene = undefined;
    try makeStackScene(alloc, &scene_a, &meshes_a);
    defer deinitStackScene(&scene_a, &meshes_a);
    try runUpdates(&scene_a, determinism_steps);
    const snap_a = try snapshotTransforms(alloc, &scene_a);
    defer alloc.free(snap_a);

    // Methodology first: the identical inline scene must reproduce itself
    // bit-for-bit. If this fails, physics is nondeterministic and every
    // comparison below can only use an epsilon (reported honestly).
    var meshes_b: std.ArrayListUnmanaged(*Mesh) = .empty;
    var scene_b: Scene = undefined;
    try makeStackScene(alloc, &scene_b, &meshes_b);
    defer deinitStackScene(&scene_b, &meshes_b);
    try runUpdates(&scene_b, determinism_steps);
    const snap_b = try snapshotTransforms(alloc, &scene_b);
    defer alloc.free(snap_b);
    try std.testing.expect(snap_a.len > 0 and snap_a.len == snap_b.len);
    try std.testing.expectEqualSlices(u8, snap_a, snap_b);

    // Same construction, dedicated worker: bitwise identical final state.
    var meshes_c: std.ArrayListUnmanaged(*Mesh) = .empty;
    var scene_c: Scene = undefined;
    try makeStackScene(alloc, &scene_c, &meshes_c);
    defer deinitStackScene(&scene_c, &meshes_c);
    scene_c.threading.physics = .dedicated;
    try runUpdates(&scene_c, determinism_steps);
    const snap_c = try snapshotTransforms(alloc, &scene_c);
    defer alloc.free(snap_c);
    try std.testing.expectEqualSlices(u8, snap_a, snap_c);

    // The worker actually ran: kicks == joins == step count.
    const counters = scene_c.physics.workerCounters();
    try std.testing.expectEqual(@as(u64, determinism_steps), counters.kicks);
    try std.testing.expectEqual(@as(u64, determinism_steps), counters.joins);
    try std.testing.expectEqual(@as(u64, 0), counters.fallbacks);
}

test "threading: default config spawns no thread and counts nothing" {
    var gpa = std.heap.DebugAllocator(.{ .thread_safe = true }){};
    defer std.testing.expect(gpa.deinit() == .ok) catch {};
    const alloc = gpa.allocator();

    var meshes: std.ArrayListUnmanaged(*Mesh) = .empty;
    var scene: Scene = undefined;
    try makeStackScene(alloc, &scene, &meshes);
    defer deinitStackScene(&scene, &meshes);

    try std.testing.expectEqual(ThreadingOptions.PhysicsMode.@"inline", scene.threading.physics);
    try runUpdates(&scene, 10);
    try std.testing.expect(scene.physics.worker.thread == null);
    const counters = scene.physics.workerCounters();
    try std.testing.expectEqual(@as(u64, 0), counters.kicks);
    try std.testing.expectEqual(@as(u64, 0), counters.joins);
    try std.testing.expectEqual(@as(u64, 0), counters.fallbacks);
}

test "threading: forced spawn failure degrades to correct inline" {
    var gpa = std.heap.DebugAllocator(.{ .thread_safe = true }){};
    defer std.testing.expect(gpa.deinit() == .ok) catch {};
    const alloc = gpa.allocator();

    var meshes: std.ArrayListUnmanaged(*Mesh) = .empty;
    var scene: Scene = undefined;
    try makeStackScene(alloc, &scene, &meshes);
    defer deinitStackScene(&scene, &meshes);
    scene.threading.physics = .dedicated;
    scene.physics.worker.test_force_spawn_fail = true;
    try runUpdates(&scene, 30);
    const snap_fallback = try snapshotTransforms(alloc, &scene);
    defer alloc.free(snap_fallback);

    var meshes_ref: std.ArrayListUnmanaged(*Mesh) = .empty;
    var scene_ref: Scene = undefined;
    try makeStackScene(alloc, &scene_ref, &meshes_ref);
    defer deinitStackScene(&scene_ref, &meshes_ref);
    try runUpdates(&scene_ref, 30);
    const snap_ref = try snapshotTransforms(alloc, &scene_ref);
    defer alloc.free(snap_ref);
    try std.testing.expectEqualSlices(u8, snap_ref, snap_fallback);

    const counters = scene.physics.workerCounters();
    try std.testing.expectEqual(@as(u64, 0), counters.kicks);
    try std.testing.expectEqual(@as(u64, 0), counters.joins);
    try std.testing.expect(counters.fallbacks > 0);
    try std.testing.expect(scene.physics.worker.thread == null);
    try std.testing.expect(scene.physics.worker.degraded);
}

test "threading: lifecycle stress with create/remove between updates" {
    var gpa = std.heap.DebugAllocator(.{ .thread_safe = true }){};
    defer std.testing.expect(gpa.deinit() == .ok) catch {};
    const alloc = gpa.allocator();

    var meshes: std.ArrayListUnmanaged(*Mesh) = .empty;
    var scene: Scene = undefined;
    try makeStackScene(alloc, &scene, &meshes);
    // deinit runs immediately after the final update below (no idle
    // frames in between); cleanup is manual like the other tests here.
    scene.threading.physics = .dedicated;

    var extra: std.ArrayListUnmanaged(*Mesh) = .empty;
    defer {
        // Any body still alive at teardown is removed first so no world
        // holds a dangling mesh pointer; meshes free through the GPA.
        const pw = scene.physics.getWorld();
        if (pw) |w| {
            for (extra.items) |m| if (w.findBody(m)) |b| w.removeBody(b);
        }
        // Bodies created from `meshes` die with the world; free meshes.
        for (extra.items) |m| alloc.destroy(m);
        extra.deinit(alloc);
        deinitStackScene(&scene, &meshes);
    }

    var i: usize = 0;
    while (i < 60) : (i += 1) {
        try scene.update(fixed_dt);
        // Game-thread structural churn between updates (window closed):
        // create a body every 7th update, remove an extra every 11th.
        if (i % 7 == 0) {
            const m = try boxMesh(alloc, "extra", Vec3.new(0, 6, 0));
            try extra.append(alloc, m);
            _ = try scene.createRigidBody(m, .box, 1.0);
        }
        if (i % 11 == 5 and extra.items.len > 0) {
            const m = extra.pop() orelse unreachable;
            if (scene.physics.getWorld()) |w| if (w.findBody(m)) |b| w.removeBody(b);
            alloc.destroy(m);
        }
        // Runtime toggling mid-stream: park and re-arm the worker.
        if (i == 20) scene.threading.physics = .@"inline";
        if (i == 30) scene.threading.physics = .dedicated;
    }
    const counters = scene.physics.workerCounters();
    try std.testing.expect(counters.kicks > 0);
    try std.testing.expectEqual(counters.kicks, counters.joins);
    try std.testing.expectEqual(@as(u64, 0), counters.fallbacks);
}

test "threading: dedicated with tiny dt and worldless updates keep inline semantics" {
    var gpa = std.heap.DebugAllocator(.{ .thread_safe = true }){};
    defer std.testing.expect(gpa.deinit() == .ok) catch {};
    const alloc = gpa.allocator();

    // Worldless: no kick, physics time stores 0, no thread spawns.
    var bare: Scene = testScene(alloc);
    defer {
        bare.physics.deinit(alloc);
        bare.lights.deinit(alloc);
        bare.cameras.deinit(alloc);
        bare.draws.deinit(alloc);
        bare.gpu_retire.deinit(alloc);
        bare.profiler.deinit();
    }
    bare.threading.physics = .dedicated;
    try bare.update(fixed_dt);
    try std.testing.expect(bare.physics.worker.thread == null);
    try std.testing.expectEqual(@as(f32, 0), bare.pending_physics_ms.load(.acquire));
    const bare_counters = bare.physics.workerCounters();
    try std.testing.expectEqual(@as(u64, 0), bare_counters.kicks);

    // Tiny dt: inline early-return, no thread roundtrip.
    var meshes: std.ArrayListUnmanaged(*Mesh) = .empty;
    var scene: Scene = undefined;
    try makeStackScene(alloc, &scene, &meshes);
    defer deinitStackScene(&scene, &meshes);
    scene.threading.physics = .dedicated;
    try scene.update(0.0);
    try std.testing.expect(scene.physics.worker.thread == null);
    try std.testing.expectEqual(@as(u64, 0), scene.physics.workerCounters().kicks);
}

// Fork-based panic test: no panic-capture helper exists in the repo, so
// the child opens the parallel window and calls a guarded mutator (must
// panic); the parent asserts the child did NOT exit cleanly.
extern "c" fn fork() c_int;
extern "c" fn waitpid(pid: c_int, status: *c_int, options: c_int) c_int;
extern "c" fn _exit(status: c_int) noreturn;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn dup2(oldfd: c_int, newfd: c_int) c_int;
extern "c" fn close(fd: c_int) c_int;

test "threading: parallel-window tripwire panics on game-thread mutation" {
    if (builtin.os.tag == .windows or builtin.cpu.arch.isWasm() or builtin.single_threaded) {
        return error.SkipZigTest;
    }
    // The tripwire is deliberately compiled out in ReleaseFast/Small; the
    // panic expectation below only holds in Debug and ReleaseSafe.
    if (builtin.mode == .ReleaseFast or builtin.mode == .ReleaseSmall) {
        return error.SkipZigTest;
    }
    const pid = fork();
    if (pid == 0) {
        // Child: minimal world, open the window, touch it. A panic here
        // kills the child abnormally (expected); reaching _exit(0) means
        // the tripwire did NOT fire (failure). Silence the expected stack
        // trace: the --listen runner prints it as a failed command even
        // though the parent test passes.
        const devnull = open("/dev/null", 1); // O_WRONLY
        if (devnull >= 0) {
            _ = dup2(devnull, 2);
            _ = close(devnull);
        }
        var gpa = std.heap.DebugAllocator(.{}){};
        const alloc = gpa.allocator();
        var world = PhysicsWorld.init(alloc);
        var m = Mesh{
            .name = "tripwire",
            .vertex_buffer = .{},
            .index_buffer = .{},
            .index_count = 0,
            .local_bounding_box = BoundingBox.init(Vec3.new(-0.5, -0.5, -0.5), Vec3.new(0.5, 0.5, 0.5)),
        };
        world.beginParallelWindow();
        _ = world.createBody(&m, .box, 1.0) catch {};
        _exit(0);
    }
    if (pid < 0) return error.TestUnexpectedResult;
    var status: c_int = 0;
    _ = waitpid(pid, &status, 0);
    const exited_cleanly = (status & 0x7f) == 0 and (status >> 8) == 0;
    try std.testing.expect(!exited_cleanly);

    // Same call with the window closed succeeds (no false positive).
    var gpa = std.heap.DebugAllocator(.{}){};
    defer std.testing.expect(gpa.deinit() == .ok) catch {};
    const alloc = gpa.allocator();
    var world = PhysicsWorld.init(alloc);
    defer world.deinit();
    var m = Mesh{
        .name = "tripwire-ok",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.5, -0.5, -0.5), Vec3.new(0.5, 0.5, 0.5)),
    };
    const body = try world.createBody(&m, .box, 1.0);
    world.removeBody(body);
}

test "threading: physics worker shutdown is idempotent and stays inline" {
    if (builtin.single_threaded or builtin.cpu.arch.isWasm()) return;
    var gpa = std.heap.DebugAllocator(.{ .thread_safe = true }){};
    defer std.testing.expect(gpa.deinit() == .ok) catch {};
    const alloc = gpa.allocator();

    var scene: Scene = testScene(alloc);
    _ = scene.enablePhysics(null);
    scene.threading.physics = .dedicated;
    defer deinitParticleScene(&scene);

    try scene.update(fixed_dt);
    try std.testing.expect(scene.physics.worker.thread != null);

    scene.physics.worker.shutdown();
    // The first shutdown destroys the handoff primitives; the second must
    // not lock them again (it used to be UB/EINVAL on POSIX).
    scene.physics.worker.shutdown();
    try std.testing.expect(scene.physics.worker.thread == null);

    // A later update degrades to the inline step; it must never re-spawn a
    // worker on the destroyed handoff lot.
    try scene.update(fixed_dt);
    try std.testing.expect(scene.physics.worker.thread == null);
    try std.testing.expect(scene.physics.workerCounters().fallbacks >= 1);
}

test "threading: particle error still joins the dedicated physics worker" {
    if (builtin.single_threaded or builtin.cpu.arch.isWasm()) return;
    var gpa = std.heap.DebugAllocator(.{ .thread_safe = true }){};
    defer std.testing.expect(gpa.deinit() == .ok) catch {};
    const alloc = gpa.allocator();

    var scene: Scene = testScene(alloc);
    _ = scene.enablePhysics(null);
    scene.threading.physics = .dedicated;
    const ps = try scene.createParticleSystem("err_gpu", 4);
    // GPU-mode systems hard-error on local-space emitters; this is the
    // deterministic error path that used to return before joinPhysics().
    ps.simulation_mode = .gpu;
    ps.local_space = true;
    defer deinitParticleScene(&scene);

    try std.testing.expectError(error.LocalSpaceNeedsCpu, scene.update(fixed_dt));
    const after_error = scene.physics.workerCounters();
    try std.testing.expectEqual(@as(u64, 1), after_error.kicks);
    try std.testing.expectEqual(@as(u64, 1), after_error.joins);

    // The next update re-kicks the worker. Without the error-path join this
    // trips the kick/join pairing assert (Debug/ReleaseSafe) or overwrites
    // an in-flight step (ReleaseFast), so this second call is the regression.
    try std.testing.expectError(error.LocalSpaceNeedsCpu, scene.update(fixed_dt));
    const after_second = scene.physics.workerCounters();
    try std.testing.expectEqual(@as(u64, 2), after_second.kicks);
    try std.testing.expectEqual(@as(u64, 2), after_second.joins);
}
