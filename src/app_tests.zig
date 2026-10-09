//! Public lifecycle tests for `agate.App` (sibling of `app.zig`).
//!
//! All tests are CPU/headless: no window, no GPU. Tests that mark the GPU
//! owner thread reset it via the existing `resetContextThreadForTest` seam
//! so no test pollutes the next one.

const std = @import("std");
const sokol = @import("sokol");
const sapp = sokol.app;
const sg = sokol.gfx;

const app_mod = @import("app.zig");
const App = app_mod.App;
const AppConfig = app_mod.AppConfig;
const Scene = @import("scene.zig").Scene;
const Runtime = @import("runtime.zig").Runtime;
const ArcRotateCamera = @import("camera/arc_rotate.zig").ArcRotateCamera;
const gpu_thread = @import("gpu_thread.zig");

fn deinitCpuScene(scene: *Scene) void {
    // Mirrors the CPU-only fixture teardown: only allocator-backed domains
    // the fixture actually initializes (same subset as `runtime_tests`).
    const alloc = scene.allocator;
    scene.lights.deinit(alloc);
    scene.cameras.deinit(alloc);
    scene.draws.deinit(alloc);
    scene.gpu_retire.deinit(alloc);
    scene.profiler.deinit();
}

test "App headless run releases its active lifecycle before a subsequent run" {
    defer gpu_thread.resetContextThreadForTest();
    var app = App.init(std.testing.allocator, .{ .headless = true }, .{});
    app.run();
    try std.testing.expectEqual(@as(u64, 1), app.frameCount());
    try std.testing.expect(!app.running);
    app.runHeadless(2);
    try std.testing.expectEqual(@as(u64, 3), app.frameCount());
    try std.testing.expect(!app.running);
}

/// Non-default config exercising every mapped flag and pool size.
pub fn configForMappingTest() AppConfig {
    return .{
        .title = "Desc Test",
        .width = 1220,
        .height = 640,
        .sample_count = 4,
        .high_dpi = false,
        .disable_vsync = false,
        .srgb_backbuffer = true,
        .gl_major = 4,
        .gl_minor = 3,
        .wgpu_gpu_timing_enabled = false,
        .metal_disable_display_sync = false,
        .buffer_pool_size = 2048,
        .image_pool_size = 128,
        .sampler_pool_size = 64,
        .view_pool_size = 128,
        .pipeline_pool_size = 128,
        .shader_pool_size = 32,
        .uniform_buffer_size = 8 * 1024 * 1024,
    };
}

/// Full flag/pool mapping expectations plus backend-floor-safe defaults
/// (GL 4.3, sRGB off, pools at soak size).
pub fn expectMappings(sapp_desc: sapp.Desc, sg_desc: sg.Desc, def_sapp: sapp.Desc, def_sg: sg.Desc) !void {
    try std.testing.expectEqualStrings("Desc Test", std.mem.span(sapp_desc.window_title));
    try std.testing.expectEqual(@as(i32, 1220), sapp_desc.width);
    try std.testing.expectEqual(@as(i32, 640), sapp_desc.height);
    try std.testing.expectEqual(@as(i32, 4), sapp_desc.sample_count);
    try std.testing.expect(sapp_desc.srgb);
    try std.testing.expect(!sapp_desc.high_dpi);
    try std.testing.expect(!sapp_desc.disable_vsync);
    try std.testing.expect(!sapp_desc.wgpu_gpu_timing_enabled);
    try std.testing.expectEqual(@as(i32, 4), sapp_desc.gl.major_version);
    try std.testing.expectEqual(@as(i32, 3), sapp_desc.gl.minor_version);
    try std.testing.expect(!sapp_desc.metal.disable_display_sync);

    try std.testing.expectEqual(@as(i32, 2048), sg_desc.buffer_pool_size);
    try std.testing.expectEqual(@as(i32, 128), sg_desc.image_pool_size);
    try std.testing.expectEqual(@as(i32, 64), sg_desc.sampler_pool_size);
    try std.testing.expectEqual(@as(i32, 128), sg_desc.view_pool_size);
    try std.testing.expectEqual(@as(i32, 128), sg_desc.pipeline_pool_size);
    try std.testing.expectEqual(@as(i32, 32), sg_desc.shader_pool_size);
    try std.testing.expectEqual(@as(i32, 8 * 1024 * 1024), sg_desc.uniform_buffer_size);

    try std.testing.expect(!def_sapp.srgb);
    try std.testing.expectEqual(@as(i32, 4), def_sapp.gl.major_version);
    try std.testing.expectEqual(@as(i32, 3), def_sapp.gl.minor_version);
    try std.testing.expectEqual(@as(i32, 4096), def_sg.buffer_pool_size);
}

test "App descriptor mapping covers every flag and pool" {
    const alloc = std.testing.allocator;
    const app = App.init(alloc, configForMappingTest(), .{});
    const def = App.init(alloc, .{}, .{});
    try expectMappings(app.sappDesc(), app.sgDesc(.{}), def.sappDesc(), def.sgDesc(.{}));
}

test "App headless lifecycle: init, frame stepping, event dispatch, cleanup" {
    const alloc = std.testing.allocator;
    defer gpu_thread.resetContextThreadForTest();

    const State = struct {
        initialized: bool = false,
        frames_run: usize = 0,
        cleaned_up: bool = false,
        events_received: usize = 0,
    };

    var state = State{};

    const TestApp = struct {
        fn onInit(app: *App) void {
            const s = app.getUserData(State);
            s.initialized = app.running and app.is_headless;
        }

        fn onFrame(app: *App) void {
            const s = app.getUserData(State);
            s.frames_run += 1;
        }

        fn onEvent(app: *App, ev: [*c]const sapp.Event) void {
            const s = app.getUserData(State);
            s.events_received += 1;
            _ = ev;
        }

        fn onCleanup(app: *App) void {
            const s = app.getUserData(State);
            s.cleaned_up = true;
        }
    };

    var app = App.init(alloc, .{
        .title = "Headless Test App",
        .width = 800,
        .height = 600,
        .headless = true,
    }, .{
        .init = TestApp.onInit,
        .frame = TestApp.onFrame,
        .cleanup = TestApp.onCleanup,
        .event = TestApp.onEvent,
    });
    app.user_data = &state;

    // Manual stepping (instead of runHeadless) so an event can be injected
    // mid-run and resize bookkeeping asserted alongside the lifecycle.
    app.onInit();
    var i: usize = 0;
    while (i < 5) : (i += 1) {
        if (i == 2) {
            var ev: sapp.Event = .{
                .type = .RESIZED,
                .window_width = 800,
                .window_height = 600,
                .framebuffer_width = 800,
                .framebuffer_height = 600,
            };
            app.onEvent(&ev);
        }
        app.onFrame();
    }
    app.onCleanup();

    try std.testing.expect(state.initialized);
    try std.testing.expectEqual(@as(usize, 5), state.frames_run);
    try std.testing.expectEqual(@as(u64, 5), app.frameCount());
    try std.testing.expectEqual(@as(usize, 1), state.events_received);
    try std.testing.expectEqual(@as(i32, 800), app.width());
    try std.testing.expectEqual(@as(i32, 600), app.height());
    // Headless timing: fixed 1/60 step per frame.
    try std.testing.expectEqual(@as(f32, 1.0 / 60.0), app.deltaTime());
    try std.testing.expectEqual(5.0 * @as(f64, app.deltaTime()), app.time());
    try std.testing.expect(state.cleaned_up);
    try std.testing.expect(!app.running);
}

test "App staged frame integration headless: skipped without scene" {
    const alloc = std.testing.allocator;
    defer gpu_thread.resetContextThreadForTest();

    var app = App.init(alloc, .{
        .title = "App Staged Frame Test",
        .width = 640,
        .height = 480,
        .headless = true,
    }, .{});

    _ = app.initRuntime();
    try std.testing.expect(app.runtime != null);

    // Context thread is marked during onInit
    app.onInit();
    try std.testing.expect(gpu_thread.isOnContextThread());

    // Step frame without scene: should return .skipped safely without crash
    const res = app.renderStagedFrame();
    try std.testing.expectEqual(@import("runtime.zig").FrameResult.skipped, res);

    app.onCleanup();

    // Owned runtime is deinited and detached by cleanup.
    try std.testing.expect(app.runtime == null);
    try std.testing.expect(app.own_runtime == null);
}

var join_probe_rt: ?*Runtime = null;
var join_probe_entered = std.atomic.Value(bool).init(false);
var cleanup_saw_joined: bool = false;

fn spinUntilStopped() void {
    join_probe_entered.store(true, .release);
    while (join_probe_rt.?.shouldRun()) {
        std.atomic.spinLoopHint();
    }
}

fn probeCleanup(app: *App) void {
    if (app.own_runtime) |*rt| {
        cleanup_saw_joined = (rt.thread == null);
    }
}

test "App owned runtime: worker joins before user cleanup, then deinits" {
    const alloc = std.testing.allocator;
    defer gpu_thread.resetContextThreadForTest();
    defer join_probe_rt = null;
    defer join_probe_entered.store(false, .release);

    cleanup_saw_joined = false;
    var app = App.init(alloc, .{
        .title = "App Worker Cleanup Test",
        .headless = true,
    }, .{
        .cleanup = probeCleanup,
    });

    const rt = app.initRuntime();
    join_probe_rt = rt;
    const spawned = rt.spawnWorker(spinUntilStopped);
    if (!spawned) {
        // Single-threaded/wasm builds cannot spawn: cleanup must still
        // detach the (worker-free) owned runtime without crashing.
        join_probe_rt = null;
    }

    app.onInit();
    app.onFrame();
    app.onCleanup();
    join_probe_rt = null;

    if (spawned) {
        try std.testing.expect(join_probe_entered.load(.acquire));
        try std.testing.expect(cleanup_saw_joined);
    }
    try std.testing.expect(app.runtime == null);
    try std.testing.expect(app.own_runtime == null);
}

test "App repeated initRuntime returns the existing instance" {
    const alloc = std.testing.allocator;
    defer gpu_thread.resetContextThreadForTest();

    var app = App.init(alloc, .{ .headless = true }, .{});
    const r1 = app.initRuntime();
    const r2 = app.initRuntime();
    try std.testing.expect(r1 == r2);

    app.onInit();
    app.onCleanup();
    try std.testing.expect(app.runtime == null);
    try std.testing.expect(app.own_runtime == null);

    // Second cleanup is idempotent.
    app.onCleanup();
}

test "App cleanup never touches borrowed runtime or scene" {
    const alloc = std.testing.allocator;
    defer gpu_thread.resetContextThreadForTest();

    var borrowed_rt = Runtime.init();
    defer borrowed_rt.deinit();

    var sc = @import("testing.zig").testScene(alloc);
    defer deinitCpuScene(&sc);

    var app = App.init(alloc, .{ .headless = true }, .{});
    app.runtime = &borrowed_rt;
    app.scene = &sc;

    app.onInit();
    app.onCleanup();

    try std.testing.expect(app.runtime == &borrowed_rt);
    try std.testing.expect(app.scene == &sc);
    try std.testing.expect(app.own_runtime == null);
    try std.testing.expect(app.own_scene == null);
}

var swallow_seen: usize = 0;

fn swallowEvent(app: *App, ev: [*c]const sapp.Event) void {
    _ = app;
    _ = ev;
    swallow_seen += 1;
}

test "App event: default forwards to scene" {
    const alloc = std.testing.allocator;
    defer gpu_thread.resetContextThreadForTest();

    var sc = @import("testing.zig").testScene(alloc);
    defer deinitCpuScene(&sc);
    sc.active_camera = .{ .arc_rotate = ArcRotateCamera.init("t", .{}) };

    var app = App.init(alloc, .{ .headless = true }, .{});
    app.scene = &sc;

    app.onInit();
    var ev: sapp.Event = .{
        .type = .MOUSE_DOWN,
        .mouse_button = .LEFT,
        .mouse_x = 10,
        .mouse_y = 10,
    };
    app.onEvent(&ev);
    try std.testing.expect(sc.active_camera.?.arc_rotate.is_dragging);
    app.onCleanup();

    // Borrowed scene survives cleanup with its state intact.
    try std.testing.expect(app.scene == &sc);
    try std.testing.expect(sc.active_camera.?.arc_rotate.is_dragging);
}

test "App event: callback owns routing, scene untouched when swallowed" {
    const alloc = std.testing.allocator;
    defer gpu_thread.resetContextThreadForTest();

    swallow_seen = 0;
    var sc = @import("testing.zig").testScene(alloc);
    defer deinitCpuScene(&sc);
    sc.active_camera = .{ .arc_rotate = ArcRotateCamera.init("t", .{}) };

    var app = App.init(alloc, .{ .headless = true }, .{ .event = swallowEvent });
    app.scene = &sc;

    app.onInit();
    var ev: sapp.Event = .{
        .type = .MOUSE_DOWN,
        .mouse_button = .LEFT,
        .mouse_x = 10,
        .mouse_y = 10,
    };
    app.onEvent(&ev);
    try std.testing.expectEqual(@as(usize, 1), swallow_seen);
    try std.testing.expect(!sc.active_camera.?.arc_rotate.is_dragging);
    app.onCleanup();
}

test "App resize tracks framebuffer px, logical divides DPI" {
    const alloc = std.testing.allocator;

    var app = App.init(alloc, .{ .headless = true }, .{});
    app.dpi_scale = 2.0;

    var ev: sapp.Event = .{
        .type = .RESIZED,
        .window_width = 800,
        .window_height = 600,
        .framebuffer_width = 1600,
        .framebuffer_height = 1200,
    };
    app.onEvent(&ev);

    try std.testing.expectEqual(@as(i32, 1600), app.width());
    try std.testing.expectEqual(@as(i32, 1200), app.height());
    try std.testing.expectEqual(@as(i32, 1600), app.framebufferWidth());
    try std.testing.expectEqual(@as(i32, 1200), app.framebufferHeight());
    try std.testing.expectEqual(@as(i32, 800), app.logicalWidth());
    try std.testing.expectEqual(@as(i32, 600), app.logicalHeight());

    // Non-resize events never touch dimensions.
    var mv: sapp.Event = .{
        .type = .MOUSE_MOVE,
        .window_width = 1,
        .window_height = 1,
        .framebuffer_width = 1,
        .framebuffer_height = 1,
    };
    app.onEvent(&mv);
    try std.testing.expectEqual(@as(i32, 1600), app.width());
    try std.testing.expectEqual(@as(i32, 1200), app.height());
}
