const std = @import("std");
const sokol = @import("sokol");
const sapp = sokol.app;
const sg = sokol.gfx;
const slog = sokol.log;
const sglue = sokol.glue;
const z = @import("agate");

var gpa = std.heap.DebugAllocator(.{ .thread_safe = true }){};
var scene: z.Scene = undefined;
var box: *z.Mesh = undefined;
var camera: z.ArcRotateCamera = undefined;

// CLI: --frames N quits after N rendered frames (0 = run until closed);
// --particles <cpu|gpu> adds a demo particle system with that
// simulation mode (default: none). --msaa N requests MSAA for the offscreen
// main target (valid: 1/2/4, clamped per scene/msaa.zig; enabling it turns
// the post chain on, because MSAA applies only to the offscreen path, and
// suppresses SSAO/SSR/DoF, which need a depth texture sokol cannot resolve
// from an MSAA target). --stats prints a compact one-line frame-metrics
// summary (phase timings, draws, texture-upload tally) every 120 frames.
// --frames is useful for headless smokes:
// `agate --frames 120 --msaa 4` must complete without sokol validation
// errors (the classic MSAA failure is a pipeline/attachment sample-count
// mismatch).
var frame_limit: u32 = 0;
var frame_count: u32 = 0;
var particle_mode: ?z.SimulationMode = null;
var profile_mode: bool = false;
/// --stats gate: frame-metrics summary every `stats_interval` frames.
/// Off by default so normal runs see no stdout change.
/// Reads context-owned stats AFTER render; the update side never writes
/// stats (only pending_update_ms), so no lock is needed here even though
/// the game thread may already run the next update.
var show_stats: bool = false;
const stats_interval: u32 = 120;
var stats_tick: u32 = 0;
/// Actual update||render split. The game thread owns simulation
/// (Scene.update + input consumption + recordUpdateTime staging), the sapp
/// (context) thread owns prepare + render. Choreography lives in the
/// engine-owned `z.Runtime` facade (see runtime.zig): the game side holds
/// the phase mutex across its whole tick (`simulate` + producer build),
/// the context side holds it across the staged begin ONLY, then releases
/// before finish + render so the next update overlaps the draw. This is
/// sound because `finishStagedPrepare` consumes only the claimed slot plus
/// context-owned state and render reads ONLY render-owned captures
/// (prepared draws/UI/debug/sky/particle frames + frame_snapshot +
/// borrowed GPU handles under P3 epochs) — see Scene docs.
/// sg.* calls only happen on the sapp thread; the update phase is free of
/// them. The Scene allocator is thread-safe (GPA .thread_safe = true):
/// prepare/render lazy caches and jobs workers allocate from it.
var threaded: bool = true;
var runtime: z.Runtime = z.Runtime.init();
var quit_requested = std.atomic.Value(bool).init(false);
var particles: *z.ParticleSystem = undefined;
var msaa_samples: i32 = 1;

fn parseArgs(allocator: std.mem.Allocator, args: std.process.Args) void {
    var it = std.process.Args.Iterator.initAllocator(args, allocator) catch return;
    defer it.deinit();
    _ = it.next(); // program name
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--frames")) {
            if (it.next()) |n| {
                frame_limit = std.fmt.parseInt(u32, n, 10) catch 0;
            }
        } else if (std.mem.eql(u8, arg, "--no-threads")) {
            threaded = false;
        } else if (std.mem.eql(u8, arg, "--particles")) {
            const mode = it.next() orelse break;
            if (std.mem.eql(u8, mode, "cpu")) {
                particle_mode = .cpu;
            } else if (std.mem.eql(u8, mode, "gpu")) {
                particle_mode = .gpu;
            }
        } else if (std.mem.eql(u8, arg, "--msaa")) {
            const n = it.next() orelse break;
            msaa_samples = std.fmt.parseInt(i32, n, 10) catch 1;
        } else if (std.mem.eql(u8, arg, "--stats")) {
            show_stats = true;
        } else if (std.mem.eql(u8, arg, "--profile")) {
            profile_mode = true;
        }
    }
}

/// Milliseconds elapsed since a `sokol.time.now()` tick. Cheap, no
/// allocation; used for the SceneStats phase timings.
fn msSince(t0: u64) f32 {
    return @floatCast(sokol.time.ms(sokol.time.now() -% t0));
}

/// Compact one-line frame metrics, printed every `stats_interval` frames
/// when `--stats` is passed. update/prepare are measured around
/// Scene.update (staged via recordUpdateTime) / staged begin+finish; shadow/main/
/// post are timed inside Scene.render. Runs AFTER render on the context
/// thread: stats is context-owned and the update side never writes it, so
/// the game thread running the next update concurrently cannot race this
/// read (it only stages pending_update_ms + mutates live sim state, both
/// disjoint from stats).
fn printFrameStats() void {
    stats_tick += 1;
    if (stats_tick % stats_interval != 0) return;
    const s = scene.stats;
    std.debug.print("stats: update={d:.2}ms prepare={d:.2}ms shadow={d:.2}ms main={d:.2}ms post={d:.2}ms draws={d} tris={d} up_tex={d} up_bytes={d}\n", .{
        s.update_ms,
        s.prepare_ms,
        s.shadow_ms,
        s.main_ms,
        s.post_ms,
        s.draw_calls,
        s.triangles,
        s.uploaded_textures_frame,
        s.uploaded_bytes_frame,
    });
}

export fn init() callconv(.c) void {
    // This callback runs on the sokol-app thread: the only thread allowed
    // to touch sg.*. Engine paths use the marker to defer off-thread GPU
    // work (mesh destroy/create, particle buffer creation) to render start.
    z.gpu_thread.markContextThread();
    sg.setup(.{
        .environment = sglue.environment(),
        .logger = .{ .func = slog.func },
        // Scene subsystems (forward + DS twins + shadow/skybox/particles +
        // postfx) create well over the 128-pipeline sokol default.
        .pipeline_pool_size = 256,
        .shader_pool_size = 128,
    });

    // Worker pool for data-parallel systems (particles CPU integration).
    // Null on failure: everything degrades to serial execution by design.
    z.jobs.global = z.jobs.Pool.init(gpa.allocator(), z.jobs.Pool.recommendedWorkerCount()) catch null;
    sokol.time.setup();
    const allocator = gpa.allocator();
    scene = z.Scene.init(allocator);

    // MSAA applies to the offscreen main target, so the post chain must own
    // PASS 2. Depth-consuming effects (SSAO/SSR/DoF) are suppressed by the
    // engine (warned once) while MSAA is active.
    if (msaa_samples > 1) {
        scene.msaa_sample_count = msaa_samples;
        scene.post_process.enabled = true;
    }

    // Babylon.js style: настройка орбитальной камеры
    camera = z.ArcRotateCamera.init("MainCamera", .{
        .alpha = std.math.pi / 4.0,
        .beta = std.math.pi / 3.0,
        .radius = 5.5,
        .target = z.Vec3.zero,
    });
    scene.active_camera = .{ .arc_rotate = camera };

    // Babylon.js style: свет HemisphericLight (небо + земля)
    _ = scene.createHemisphericLight("hemiLight", .{
        .direction = z.Vec3.new(0.6, 1.0, 0.4),
        .diffuse = z.Color3.white,
        .ground_color = z.Color3.new(0.2, 0.22, 0.28),
        .intensity = 1.0,
    });

    // Babylon.js style: создание меша через MeshBuilder
    box = z.MeshBuilder.createBox(&scene, "box", .{
        .size = 2.0,
    }) catch |err| {
        std.debug.panic("Failed to create box: {}", .{err});
    };

    if (particle_mode) |mode| {
        particles = scene.createParticleSystem("demo", 256) catch |err| {
            std.debug.panic("Failed to create particle system: {}", .{err});
        };
        particles.simulation_mode = mode;
        particles.emit_rate = 120.0;
        particles.is_emitting = true;
        particles.gravity = z.Vec3.new(0.0, -1.5, 0.0);
        particles.drag = 0.8;
        particles.emitter_box_max = z.Vec3.new(0.2, 0.2, 0.2);
        particles.direction_max = z.Vec3.new(0.5, 2.0, 0.5);
        particles.speed_max = 3.0;
        particles.size_start = 0.1;
        particles.size_end = 0.02;
    }

    // Stage 3: spawn the game thread — simulation moves there; the sapp
    // thread keeps windowing + render. Spawn failure degrades to
    // single-threaded (frame() runs simulate inline).
    if (profile_mode) {
        scene.startProfiling();
        std.log.info("Profiling started via --profile CLI flag", .{});
    }

    if (threaded) {
        if (!runtime.spawnWorker(gameLoop)) threaded = false;
    }
}

/// Stage 3 seam: input events are produced in the sapp callback and
/// consumed by the game side (gameLoop when threaded). The ring crosses
/// the thread boundary; same-thread it is a plain FIFO.
const AppEvent = union(enum) {
    key_down: sapp.Keycode,
    mouse_down,
};
var input_ring: z.jobs.SpscRing(AppEvent, 64) = .{};
/// Push results are otherwise silent: count dropped events and rate-limit
/// the log (every 64th drop) so a flooded ring stays visible without
/// spamming. MOUSE_MOVE is intentionally never queued — simulate() consumes
/// discrete presses only, and queuing per-move events would flood the 64
/// slots with stale positions.
var input_dropped: u64 = 0;

fn pushInput(ev: AppEvent) void {
    if (!input_ring.push(ev)) {
        input_dropped += 1;
        if (input_dropped % 64 == 1) {
            std.debug.print("input ring full, dropped {} events\n", .{input_dropped});
        }
    }
}

/// One simulation step on the game side: input consumption + Scene.update
/// + demo state. Callers hold the runtime phase mutex (game side) across
/// the whole tick — see gameLoop / the single-threaded frame path.
fn simulate(dt_sec: f32) void {
    // Game-side input consumption: events were produced on the sapp
    // thread (ring), applied here where the simulation state lives.
    while (input_ring.pop()) |ev| {
        switch (ev) {
            .mouse_down => cycleClearColor(),
            .key_down => |key| switch (key) {
                .SPACE => cycleClearColor(),
                .ESCAPE => quit_requested.store(true, .release),
                else => {},
            },
        }
    }

    // Вращаем куб (60fps-normalized speed, preserved from the demo's
    // original frame-based pacing).
    const dt_norm: f32 = dt_sec * 60.0;
    box.rotation.x += 0.8 * dt_norm;
    box.rotation.y += 1.6 * dt_norm;

    const t_update = sokol.time.now();
    scene.update(dt_sec) catch |err| {
        std.debug.panic("scene update failed: {s} (particle mode: {s})", .{
            @errorName(err),
            if (particle_mode) |m| @tagName(m) else "off",
        });
    };
    // Producer build at the update boundary (CPU-only, sg-free): the
    // engine facade stages instance matrices + particle/physics captures
    // for the prepare latch via Runtime.update (claim -> build -> stageUi
    // -> publish, right after this body). Kept out of here so the simple
    // path owns the ordering in one place.
    // Stage the update tick WITHOUT touching stats (context-owned; render
    // may read it concurrently): the staged begin transfers it next frame.
    scene.recordUpdateTime(msSince(t_update));
}

/// Simple-path tick body for Runtime.update: simulation WITHOUT the build
/// (the facade appends produceBuild under the same game-side hold).
fn gameTick(ctx: *TickCtx) void {
    simulate(ctx.dt_sec);
}

const TickCtx = struct {
    dt_sec: f32,
};

fn gameLoop() void {
    var last = sokol.time.now();
    while (runtime.shouldRun()) {
        const now = sokol.time.now();
        const dt_sec: f32 = @floatCast(sokol.time.ms(now -% last) / 1000.0);
        last = now;

        // Simple-path producer tick: simulate + build under game-side
        // exclusion, overlapping the context's finish + render.
        var tick = TickCtx{ .dt_sec = dt_sec };
        _ = runtime.update(&scene, &tick, gameTick);

        // Pace the simulation thread (~1 kHz): leaves cores free and
        // keeps dt magnitudes sane for the demo's float32 state.
        z.jobs.sleepNs(1_000_000);
    }
}

export fn frame() callconv(.c) void {
    if (quit_requested.load(.acquire)) sapp.quit();

    if (threaded) {
        // Simple-path context frame: bounded begin, then finish + render,
        // reuse, or skip. Live reads stay under exclusion and mutex
        // acquisition is bounded. prepare_ms is written inside renderFrame on the
        // context thread (begin + finish minus acquisition wait), same as
        // render — no cross-thread stats write.
        _ = runtime.renderFrame(&scene);

        if (show_stats) printFrameStats();
    } else {
        // Single-threaded: SAME two calls inline (mutex uncontended) —
        // identical ordering to the worker path above.
        var tick = TickCtx{ .dt_sec = @floatCast(sapp.frameDuration()) };
        _ = runtime.update(&scene, &tick, gameTick);
        _ = runtime.renderFrame(&scene);
        if (show_stats) printFrameStats();
    }

    if (frame_limit != 0) {
        frame_count += 1;
        if (frame_count >= frame_limit) sapp.quit();
    }
}

export fn cleanup() callconv(.c) void {
    // Stop the game thread before any state it touches dies (idempotent
    // with any earlier quiesce; exactly one join total).
    runtime.deinit();
    if (z.jobs.global) |pool| {
        pool.deinit();
        z.jobs.global = null;
    }
    if (scene.isProfiling()) {
        scene.stopProfiling();
        scene.saveProfileReports("profile") catch |err| {
            std.log.err("Failed to save profile: {s}", .{@errorName(err)});
        };
        std.log.info("Saved profile reports to profile.html, profile.md, profile.json", .{});
    }
    scene.deinit();
    _ = gpa.deinit();
    sg.shutdown();
}

var color_toggle: usize = 0;
const bg_colors = [_]z.Color4{
    z.Color4.new(0.12, 0.14, 0.18, 1.0),
    z.Color4.new(0.25, 0.12, 0.14, 1.0),
    z.Color4.new(0.12, 0.22, 0.16, 1.0),
    z.Color4.new(0.14, 0.18, 0.28, 1.0),
};

fn cycleClearColor() void {
    color_toggle += 1;
    scene.clear_color = bg_colors[color_toggle % bg_colors.len];
}

export fn event(ev: [*c]const sapp.Event) callconv(.c) void {
    switch (ev.*.type) {
        .MOUSE_DOWN => pushInput(.mouse_down),
        .KEY_DOWN => switch (ev.*.key_code) {
            .F8 => {
                // Window callbacks run on the sapp (context) thread: no
                // render runs concurrently here. The lock excludes the
                // update side (profiler reports + memory capture read live
                // state); the Profiler itself is context-owned and workers
                // never touch it.
                runtime.mutex.lock();
                defer runtime.mutex.unlock();
                if (scene.isProfiling()) {
                    scene.stopProfiling();
                    scene.saveProfileReports("profile") catch |err| {
                        std.log.err("Failed to save profile: {s}", .{@errorName(err)});
                        return;
                    };
                    std.log.info("Profiling stopped. Reports written to profile.html, profile.md, profile.json", .{});
                } else {
                    scene.startProfiling();
                    std.log.info("Profiling started... Press F8 again to stop and save reports.", .{});
                }
            },
            .SPACE, .ESCAPE => pushInput(.{ .key_down = ev.*.key_code }),
            else => {},
        },
        else => {},
    }
}

pub fn main(minimal: std.process.Init.Minimal) void {
    parseArgs(gpa.allocator(), minimal.args);
    sapp.run(.{
        .init_cb = init,
        .frame_cb = frame,
        .cleanup_cb = cleanup,
        .event_cb = event,
        .window_title = "agate (Babylon.js-style 3D in Zig)",
        .width = 800,
        .height = 600,
        .sample_count = 1,
        // Desktop GL floor for the cluster storage buffers (Linux); Metal
        // and D3D11 ignore this field.
        .gl = .{ .major_version = 4, .minor_version = 3 },
        .wgpu_gpu_timing_enabled = z.gpu_timing.isEnabled(),
        .logger = .{ .func = slog.func },
    });
}
