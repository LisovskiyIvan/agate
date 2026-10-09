//! Runtime worker-thread smoke: the threaded frame lifecycle on a real GPU.
//!
//! The other examples (and the sandbox demo) drive the SERIAL staged path
//! (`buildPreparedFrame -> beginStagedPrepare -> finishStagedPrepare ->
//! render`) inline on the context thread. This example is the missing live
//! coverage for the lock-free THREADED path (`API.md` "Runtime facade"):
//!
//!   - init (context thread): build the scene, then `Runtime.spawnWorker`
//!     starts the game loop on a worker thread;
//!   - game thread: `while (rt.shouldRun())` -> tick (mutate live meshes +
//!     `scene.update(dt)`) -> `rt.update` (simulate + producer build,
//!     lock-free by default) -> `publishFrameSnapshot` before each build;
//!   - context thread (sokol frame callback): just `rt.renderFrame` and
//!     count `prepared/reused/skipped/busy`;
//!   - cleanup: `rt.deinit()` joins the worker BEFORE `scene.deinit()`.
//!
//! If thread spawn fails (single-threaded build, wasm), the host degrades
//! to the documented inline serial form: the frame callback runs the same
//! `rt.update` itself — same facade, same algorithms, one thread.
//!
//! Verdict (exit code): the producer must have published builds, the
//! context must have prepared and presented frames, and the log must stay
//! error-free. `--frames N` (default 240) auto-quits; ESC quits.
//!
//! Run: `zig build run-runtime-worker -- --frames 240`
//! Expected: `verdict=PASS producer_builds>0 prepared>0 log_errors=0`.

const std = @import("std");
const z = @import("agate");
const sokol = @import("sokol");
const sapp = sokol.app;
const slog = sokol.log;

// --- host state ------------------------------------------------------------

var gpa = std.heap.DebugAllocator(.{ .thread_safe = true }){};

var scene: z.Scene = undefined;
var runtime: z.Runtime = z.Runtime.init();
var sim: Sim = .{};

var frame_limit: u32 = 240;
var frame_count: u32 = 0;
var prepared_frames: u32 = 0;
var reused_frames: u32 = 0;
var skipped_frames: u32 = 0;
var busy_frames: u32 = 0;
var log_errors: u32 = 0;
var verdict_pass = false;
var worker_running = false;

/// Window size cache: the context thread writes it on init/resize, the
/// game thread reads it for `publishFrameSnapshot` (relaxed atomics — a
/// stale aspect for one tick is harmless, the mailbox is newest-wins).
var win_w: std.atomic.Value(u32) = std.atomic.Value(u32).init(800);
var win_h: std.atomic.Value(u32) = std.atomic.Value(u32).init(600);

const ring_meshes = 12;

/// Game-thread tick context. Everything in here is owned by the producer
/// thread; the context thread never touches it (it sees only frozen slot
/// payloads and the published snapshot).
const Sim = struct {
    meshes: [ring_meshes]*z.Mesh = undefined,
    last_ns: u64 = 0,
    sim_time: f32 = 0,
};

fn simTick(s: *Sim) void {
    const now = z.jobs.monoNs();
    var dt: f32 = if (s.last_ns == 0) 1.0 / 60.0 else @floatCast(@as(f64, @floatFromInt(now -% s.last_ns)) / 1_000_000_000.0);
    s.last_ns = now;
    dt = @min(dt, 0.1);
    s.sim_time += dt;

    // Live mesh mutation on the game thread: transforms only, exactly the
    // documented threaded contract (the staged build freezes them into the
    // slot; the context never reads live meshes).
    for (s.meshes, 0..) |m, i| {
        const fi: f32 = @floatFromInt(i);
        m.rotation.x = s.sim_time * (0.3 + 0.07 * fi);
        m.rotation.y = s.sim_time * (0.5 + 0.11 * fi);
        m.position.y = @sin(s.sim_time * 1.4 + fi * 0.5) * 0.35;
    }

    scene.update(dt) catch |err| {
        log_errors += 1;
        std.debug.print("runtime-worker: scene update error: {s}\n", .{@errorName(err)});
    };
}

/// The game loop body: runs on the worker thread. `rt.update` = tick +
/// producer build (lock-free default). A fresh snapshot precedes every
/// build so the queue builder sees a camera.
fn gameLoop() void {
    while (runtime.shouldRun()) {
        const w = win_w.load(.monotonic);
        const h = win_h.load(.monotonic);
        scene.publishFrameSnapshot(@as(f32, @floatFromInt(w)) / @as(f32, @floatFromInt(h)), @intCast(w), @intCast(h));
        _ = runtime.update(&scene, &sim, simTick);
    }
}

// --- sokol app callbacks (context thread) ----------------------------------

fn appInitCb(_: *z.App) void {
    const allocator = gpa.allocator();

    scene = z.Scene.init(allocator);
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

    // Ring of boxes on two radii: enough draws to prove the queues flow,
    // cheap enough to stay smooth everywhere.
    var created: usize = 0;
    while (created < ring_meshes) : (created += 1) {
        const fi: f32 = @floatFromInt(created);
        const outer = created % 2 == 0;
        const radius: f32 = if (outer) 4.2 else 2.4;
        const ang = fi * (2.0 * std.math.pi / @as(f32, @floatFromInt(ring_meshes)));
        const box = z.MeshBuilder.createBox(&scene, "box", .{ .size = if (outer) 1.1 else 0.7 }) catch |err| {
            std.debug.panic("runtime-worker: box creation failed: {}", .{err});
        };
        box.position = z.Vec3.new(@cos(ang) * radius, 0, @sin(ang) * radius);
        sim.meshes[created] = box;
    }

    // Spawn the game worker. Single-threaded/wasm builds return false: the
    // frame callback then runs the same facade inline (serial fallback).
    sim.last_ns = 0;
    worker_running = runtime.spawnWorker(gameLoop);
    std.debug.print("runtime-worker: game worker {s}\n", .{if (worker_running) "spawned (threaded staged path)" else "spawn failed (inline serial fallback)"});
}

fn appFrameCb(_: *z.App) void {
    frame_count += 1;

    if (!worker_running) {
        // Documented serial degradation: same facade, same tick, one thread.
        _ = runtime.update(&scene, &sim, simTick);
    }

    // The whole context-side frame is this one call: bounded begin, finish,
    // render — or reuse/skip when no fresh build is ready.
    switch (runtime.renderFrame(&scene)) {
        .prepared => prepared_frames += 1,
        .reused => reused_frames += 1,
        .skipped => skipped_frames += 1,
        .busy => busy_frames += 1,
    }

    if (frame_limit != 0 and frame_count >= frame_limit) sapp.quit();
}

fn appCleanupCb(_: *z.App) void {
    // Join the game thread BEFORE tearing down the scene it writes to.
    runtime.deinit();

    const m = runtime.metrics;
    verdict_pass = worker_running and
        m.producer_builds > 0 and
        m.begins > 0 and
        m.finishes > 0 and
        prepared_frames > 0 and
        log_errors == 0;

    std.debug.print(
        "runtime-worker: frames={} prepared={} reused={} skipped={} busy={}\n" ++
            "runtime-worker: producer_builds={} producer_skips={} begins={} begin_empty={} finishes={} reuses={}\n" ++
            "verdict={s} log_errors={}\n",
        .{
            frame_count,      prepared_frames,
            reused_frames,    skipped_frames,
            busy_frames,      m.producer_builds,
            m.producer_skips, m.begins,
            m.begin_empty,    m.finishes,
            m.reuses,         if (verdict_pass) "PASS" else "FAIL",
            log_errors,
        },
    );

    scene.deinit();
    _ = gpa.deinit();
    if (!verdict_pass) std.process.exit(1);
}

fn appEventCb(_: *z.App, ev: [*c]const sapp.Event) void {
    switch (ev.*.type) {
        .RESIZED => {
            const w: u32 = @intCast(@max(1, sapp.width()));
            const h: u32 = @intCast(@max(1, sapp.height()));
            win_w.store(w, .monotonic);
            win_h.store(h, .monotonic);
        },
        .KEY_DOWN => switch (ev.*.key_code) {
            .ESCAPE => sapp.quit(),
            else => {},
        },
        else => {},
    }
}

fn rtLogger(tag: [*c]const u8, level: u32, item: u32, msg: [*c]const u8, line: u32, file: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    if (level <= 1) log_errors += 1;
    slog.func(tag, level, item, msg, line, file, user_data);
}

fn parseArgs(allocator: std.mem.Allocator, args: std.process.Args) void {
    var it = std.process.Args.Iterator.initAllocator(args, allocator) catch return;
    defer it.deinit();
    _ = it.next();
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--frames")) {
            if (it.next()) |n| frame_limit = std.fmt.parseInt(u32, n, 10) catch 240;
        }
    }
    if (frame_limit == 0) frame_limit = 240;
}

pub fn main(minimal: std.process.Init.Minimal) void {
    parseArgs(gpa.allocator(), minimal.args);
    var app = z.App.init(gpa.allocator(), .{
        .title = "agate runtime-worker (threaded staged frame lifecycle)",
        .width = 800,
        .height = 600,
        .sample_count = 1,
        .gl_major = 4,
        .gl_minor = 3,
        .logger = rtLogger,
        .shader_pool_size = 128,
    }, .{
        .init = appInitCb,
        .frame = appFrameCb,
        .cleanup = appCleanupCb,
        .event = appEventCb,
    });
    app.run();
}
