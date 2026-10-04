//! Finite GPU timestamp/lifecycle gate; native Metal or browser WebGPU.
//! Exercises multiple shadow/post passes, compute dispatch, toggles and
//! shutdown/re-setup with timing still enabled. Unsupported is explicit.
const std = @import("std");
const builtin = @import("builtin");
const z = @import("agate");
const sokol = @import("sokol");
const sg = sokol.gfx;
const sapp = sokol.app;
const timing = z.gpu_timing;

const frame_limit = 270;
const phases = [_]timing.Pass{ .shadow, .main, .post };
const Counts = struct {
    unique: [4]u32 = @splat(0),
    positive: [4]u32 = @splat(0),
    last_index: [4]u32 = @splat(0),
};
var waves: [3]Counts = @splat(.{});
var wave: usize = 0;
var frames: u32 = 0;
var failures: u32 = 0;
var log_errors: u32 = 0;
var off: bool = false;
var device_timestamps: bool = true;
var caps: timing.Capabilities = .{};
var scene: z.Scene = undefined;
var box: *z.Mesh = undefined;
var particles: ?*z.ParticleSystem = null;
var compute_dispatches: u64 = 0;
var sokol_live_allocations: usize = 0;
var rt: z.Runtime = z.Runtime.init();

extern "c" fn malloc(size: usize) ?*anyopaque;
extern "c" fn free(ptr: ?*anyopaque) void;

fn sokolAlloc(size: usize, _: ?*anyopaque) callconv(.c) ?*anyopaque {
    const ptr = malloc(size) orelse return null;
    sokol_live_allocations += 1;
    return ptr;
}

fn sokolFree(ptr: ?*anyopaque, _: ?*anyopaque) callconv(.c) void {
    if (ptr != null) {
        if (sokol_live_allocations == 0) {
            fail("allocator free without an owned allocation", .{});
        } else sokol_live_allocations -= 1;
    }
    free(ptr);
}

fn fail(comptime fmt: []const u8, args: anytype) void {
    failures += 1;
    std.debug.print("gpu-timing: FAIL " ++ fmt ++ "\n", args);
}

fn logger(tag: [*c]const u8, level: u32, item: u32, msg: [*c]const u8, line: u32, file: [*c]const u8, data: ?*anyopaque) callconv(.c) void {
    if (level <= 1) log_errors += 1;
    sokol.log.func(tag, level, item, msg, line, file, data);
}

fn setupScene() void {
    sg.setup(.{
        .environment = sokol.glue.environment(),
        .logger = .{ .func = logger },
        .pipeline_pool_size = 256,
        .shader_pool_size = 128,
        .allocator = .{ .alloc_fn = sokolAlloc, .free_fn = sokolFree },
    });
    caps = timing.capabilities();
    scene = z.Scene.init(std.heap.c_allocator);
    scene.active_camera = .{ .arc_rotate = z.ArcRotateCamera.init("camera", .{
        .alpha = 0.7,
        .beta = 1.1,
        .radius = 7,
        .target = z.Vec3.zero,
    }) };
    _ = scene.createDirectionalLight("sun", .{ .direction = z.Vec3.new(0.4, 1, 0.3) }) catch @panic("sun creation failed");
    _ = scene.createHemisphericLight("sky", .{ .intensity = 0.6 });
    box = z.MeshBuilder.createBox(&scene, "box", .{ .size = 2 }) catch @panic("box creation failed");
    scene.post_process.enabled = true;
    scene.post_process.ssr_enabled = false;
    particles = null;
    if (z.compute.supported()) {
        const ps = scene.createParticleSystem("compute", 1024) catch @panic("particle creation failed");
        ps.setSimulationMode(.compute) catch @panic("compute selection failed");
        ps.emit_rate = 900;
        ps.is_emitting = true;
        ps.gravity = z.Vec3.new(0, -0.5, 0);
        particles = ps;
    }
    std.debug.print("gpu-timing: setup wave={} backend={s} frame_supported={} pass_supported={} scope={s} compute={}\n", .{
        wave, @tagName(sg.queryBackend()), caps.frame, caps.passes, @tagName(caps.frame_scope), particles != null,
    });
    checkUnavailable();
}

fn deinitScene() void {
    if (particles) |ps| compute_dispatches += ps.computeDispatchCount();
    particles = null;
    scene.deinit();
    // Deliberately do NOT disable timings: shutdown owns backend resources.
    sg.shutdown();
    if (sokol_live_allocations != 0) fail("allocator still owns {} allocations after shutdown", .{sokol_live_allocations});
    checkUnavailable();
    if (sg.queryGpuFrameMs() != -1 or sg.queryGpuFrameIndex() != 0) {
        fail("raw frame query survived shutdown", .{});
    }
}

fn checkUnavailable() void {
    if (timing.pollFrameSample() != null) fail("unexpected frame sample at frame {}", .{frames});
    for (phases) |phase| {
        if (timing.pollPassSample(phase) != null) fail("unexpected {s} sample at frame {}", .{ @tagName(phase), frames });
    }
}

fn observe(sample: ?timing.Sample, column: usize, supported: bool) void {
    const s = sample orelse return;
    if (!supported) fail("sample on unsupported timer {}", .{column});
    if (s.ms < 0 or !std.math.isFinite(s.ms) or s.frame_index == 0) {
        fail("invalid sample {}: {} ms index {}", .{ column, s.ms, s.frame_index });
        return;
    }
    const counts = &waves[wave];
    if (s.frame_index == counts.last_index[column]) return;
    if (s.frame_index < counts.last_index[column]) fail("sample order regressed", .{});
    counts.last_index[column] = s.frame_index;
    counts.unique[column] += 1;
    if (s.ms > 0) counts.positive[column] += 1;
    if (counts.unique[column] == 1) {
        std.debug.print("gpu-timing: sample wave={} timer={} submission={} ms={d:.4}\n", .{ wave, column, s.frame_index, s.ms });
    }
}

export fn init() callconv(.c) void {
    z.gpu_thread.markContextThread();
    sokol.time.setup();
    timing.setEnabled(false);
    setupScene();
}

export fn frame() callconv(.c) void {
    frames += 1;
    if (!off) {
        if (frames == 13 or frames == 110) timing.setEnabled(true);
    }
    if (frames == 110) wave = 1;
    box.rotation.y += 0.015;
    scene.update(1.0 / 60.0) catch |err| fail("update: {s}", .{@errorName(err)});
    // Staged producer build -> claim -> render. Every frame must carry a
    // fresh FULL build; a null begin after a successful build (or a busy
    // begin here — single-threaded, uncontended) fails loudly instead of
    // silently skipping timer frames. Reuse only covers the failure path.
    if (!scene.buildPreparedFrame()) {
        fail("producer saturated at frame {}", .{frames});
    }
    const begun = rt.beginPrepare(&scene);
    if (begun.claim) |c| {
        rt.finishPrepare(&scene, c);
        scene.render();
    } else {
        if (begun.busy) fail("begin busy at frame {}", .{frames}) else fail("fresh build not claimable at frame {}", .{frames});
        if (!rt.reuseIfConsumable(&scene)) fail("no consumable frame at {}", .{frames});
    }
    if (scene.stats.draw_calls == 0 or scene.stats.triangles == 0) fail("no real draws", .{});
    if (!timing.isEnabled()) {
        checkUnavailable();
    } else {
        observe(timing.pollFrameSample(), 0, caps.frame);
        for (phases, 1..) |phase, i| observe(timing.pollPassSample(phase), i, caps.passes);
    }
    // Do not yield to the browser event loop between submit and teardown:
    // the just-submitted WebGPU map callback is still pending here.
    if (!off and frames == 93) {
        timing.setEnabled(false);
        checkUnavailable();
    }
    if (frames == 189) {
        deinitScene();
        wave = 2;
        setupScene();
    }
    if (frames == frame_limit) sapp.quit();
}

export fn cleanup() callconv(.c) void {
    const had_compute = particles != null;
    deinitScene();
    if (had_compute and compute_dispatches == 0) fail("compute not exercised", .{});
    for (waves, 0..) |counts, i| {
        std.debug.print("gpu-timing: wave={} unique={any} positive={any}\n", .{ i, counts.unique, counts.positive });
        for (counts.unique, 0..) |n, column| {
            const supported = if (column == 0) caps.frame else caps.passes;
            if (!off and supported and (n < 2 or counts.positive[column] == 0)) {
                fail("wave {} timer {} not measured", .{ i, column });
            }
            if ((off or !supported) and n != 0) fail("unexpected measurements", .{});
        }
    }
    const ok = failures == 0 and log_errors == 0;
    std.debug.print("gpu-timing: frames={} compute_dispatches={} failures={} log_errors={} sokol_live_allocations={} verdict={s}\n", .{
        frames, compute_dispatches, failures, log_errors, sokol_live_allocations, if (ok) "PASS" else "FAIL",
    });
    if (!ok and !builtin.cpu.arch.isWasm()) std.process.exit(1);
}

extern fn emscripten_run_script_int(script: [*:0]const u8) c_int;

fn envFlag(name: [*:0]const u8) bool {
    const value = std.c.getenv(name) orelse return false;
    return timing.parseEnvFlag(std.mem.span(value));
}

pub fn startApp() void {
    off = envFlag("AGATE_GPU_TIMING_SMOKE_OFF");
    device_timestamps = !envFlag("AGATE_GPU_TIMING_SMOKE_NO_DEVICE_TIMESTAMPS");
    if (builtin.cpu.arch.isWasm()) {
        off = emscripten_run_script_int("new URLSearchParams(location.search).has('timings-off')") != 0;
        device_timestamps = emscripten_run_script_int("!new URLSearchParams(location.search).has('no-device-timestamps')") != 0;
    }
    sapp.run(.{
        .init_cb = init,
        .frame_cb = frame,
        .cleanup_cb = cleanup,
        .window_title = "Agate GPU timing gate",
        .width = 640,
        .height = 480,
        .sample_count = 1,
        .gl = .{ .major_version = 4, .minor_version = 3 },
        .wgpu_gpu_timing_enabled = device_timestamps,
        .logger = .{ .func = logger },
    });
}

pub fn main() void {
    startApp();
}
