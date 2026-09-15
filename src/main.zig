const std = @import("std");
const sokol = @import("sokol");
const sapp = sokol.app;
const sg = sokol.gfx;
const slog = sokol.log;
const sglue = sokol.glue;
const z = @import("agate");

var gpa = std.heap.DebugAllocator(.{}){};
var scene: z.Scene = undefined;
var box: *z.Mesh = undefined;
var camera: z.ArcRotateCamera = undefined;

// CLI: --frames N quits after N rendered frames (0 = run until closed);
// --particles <cpu|gpu> adds a demo particle system with that
// simulation mode (default: none). --msaa N requests MSAA for the offscreen
// main target (valid: 1/2/4, clamped per scene/msaa.zig; enabling it turns
// the post chain on, because MSAA applies only to the offscreen path, and
// suppresses SSAO/SSR/DoF, which need a depth texture sokol cannot resolve
// from an MSAA target). --frames is useful for headless smokes:
// `agate --frames 120 --msaa 4` must complete without sokol validation
// errors (the classic MSAA failure is a pipeline/attachment sample-count
// mismatch).
var frame_limit: u32 = 0;
var frame_count: u32 = 0;
var particle_mode: ?z.SimulationMode = null;
/// Stage 3: real thread split. The game thread owns simulation
/// (Scene.update + input consumption), the sapp thread owns windowing +
/// render. Coarse phase ownership: one mutex, held by whichever side is
/// inside its phase, so update and render never overlap. sg.* calls only
/// happen on the sapp thread (render + the deferred flushes at render
/// start); the update phase is free of them.
var threaded: bool = true;
var phase_mutex: z.jobs.Mutex = .{};
var game_running = std.atomic.Value(bool).init(false);
var quit_requested = std.atomic.Value(bool).init(false);
var game_thread: ?std.Thread = null;
var particles: *z.ParticleSystem = undefined;
var msaa_samples: i32 = 1;

fn parseArgs(args: std.process.Args) void {
    var it = std.process.Args.Iterator.init(args);
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
        }
    }
}

export fn init() callconv(.c) void {
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
    if (threaded) {
        game_running.store(true, .release);
        game_thread = std.Thread.spawn(.{}, gameLoop, .{}) catch |err| blk: {
            game_running.store(false, .release);
            std.debug.print("game thread spawn failed ({s}); running single-threaded\n", .{@errorName(err)});
            break :blk null;
        };
        if (game_thread == null) threaded = false;
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

/// One simulation step on the game side: input consumption + Scene.update
/// + demo state. Callers own phase ownership (the mutex when threaded).
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

    scene.update(dt_sec) catch |err| {
        std.debug.panic("scene update failed: {s} (particle mode: {s})", .{
            @errorName(err),
            if (particle_mode) |m| @tagName(m) else "off",
        });
    };
}

fn gameLoop() void {
    var last = sokol.time.now();
    while (game_running.load(.acquire)) {
        const now = sokol.time.now();
        const dt_sec: f32 = @floatCast(sokol.time.ms(now -% last) / 1000.0);
        last = now;

        phase_mutex.lock();
        if (game_running.load(.acquire)) simulate(dt_sec);
        phase_mutex.unlock();

        // Pace the simulation thread (~1 kHz): leaves cores free and
        // keeps dt magnitudes sane for the demo's float32 state.
        const ts = std.c.timespec{ .sec = 0, .nsec = 1_000_000 };
        var rem: std.c.timespec = undefined;
        _ = std.c.nanosleep(&ts, &rem);
    }
}

export fn frame() callconv(.c) void {
    if (quit_requested.load(.acquire)) sapp.quit();

    if (threaded) {
        // Render consumes the newest state under phase ownership; a long
        // update on the game thread delays this frame but cannot race it.
        phase_mutex.lock();
        scene.render();
        phase_mutex.unlock();
    } else {
        simulate(@floatCast(sapp.frameDuration()));
        scene.render();
    }

    if (frame_limit != 0) {
        frame_count += 1;
        if (frame_count >= frame_limit) sapp.quit();
    }
}

export fn cleanup() callconv(.c) void {
    // Stage 3: stop the game thread before any state it touches dies.
    if (game_thread) |t| {
        game_running.store(false, .release);
        t.join();
        game_thread = null;
    }
    if (z.jobs.global) |pool| {
        pool.deinit();
        z.jobs.global = null;
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
        .MOUSE_DOWN => _ = input_ring.push(.mouse_down),
        .KEY_DOWN => switch (ev.*.key_code) {
            .SPACE, .ESCAPE => _ = input_ring.push(.{ .key_down = ev.*.key_code }),
            else => {},
        },
        else => {},
    }
}

pub fn main(minimal: std.process.Init.Minimal) void {
    parseArgs(minimal.args);
    sapp.run(.{
        .init_cb = init,
        .frame_cb = frame,
        .cleanup_cb = cleanup,
        .event_cb = event,
        .window_title = "agate (Babylon.js-style 3D in Zig)",
        .width = 800,
        .height = 600,
        .sample_count = 1,
        .logger = .{ .func = slog.func },
    });
}
