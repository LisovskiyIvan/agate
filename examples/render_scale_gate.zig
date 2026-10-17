//! Render-scale + clustered-point-light pixel gate (headless Chromium/WebGPU).
//!
//! The fixture is deliberately smaller than the TAA gate: one neutral wall, a
//! fixed camera, and at most ONE point light placed on the right half of the
//! view. At `render_scale < 1` the main pass draws into a smaller HDR target,
//! so the clustered tile grid and `clustered_viewport` must be built from the
//! TARGET pixels, not the window. Before that fix the light was projected
//! into full-window tile rows that a half-res fragment never sampled, and the
//! hotspot vanished.
//!
//! The gate captures the held frame from three legs:
//!   lit05   scale 0.5 + light   expected realized target 480x300, bright ROI
//!   lit10   scale 1.0 + light   same scene, full-res control
//!   dark05  scale 0.5, no light negative control (black wall)
//!
//! `hold=1` freezes after the capture frame: frame() issues no update, no
//! prepare, no render and no commit, so two CDP captures must be byte-exact.
//! The gate then sets `window.rsgFinish=1`; the page quits itself and runs the
//! normal cleanup, so teardown/leak proof is real, never claimed from a
//! killed target.
//!
//! Knobs (native `AGATE_RSG_*` / web query): frames, rscale, light=0,
//! hold. Fixed dt keeps the same frame number identical across legs.

const std = @import("std");
const builtin = @import("builtin");
const z = @import("agate");
const sokol = @import("sokol");
const sapp = sokol.app;
const slog = sokol.log;
const sg = sokol.gfx;
const env_mod = @import("env.zig");
const envFlag = env_mod.envFlag;
const envF32 = env_mod.envF32;

var gpa = std.heap.DebugAllocator(.{ .thread_safe = true }){};
var scene: z.Scene = undefined;

var frame_limit: u32 = 8;
var frames: u32 = 0;
var failures: u32 = 0;
var log_errors: u32 = 0;
var sokol_live_allocations: usize = 0;
var render_scale: f32 = 0.5;
var point_light_on: bool = true;
var hold_final: bool = false;
var held: bool = false;
var held_ticks: u32 = 0;
var camera: z.ArcRotateCamera = undefined;

extern "c" fn malloc(size: usize) ?*anyopaque;
extern "c" fn free(ptr: ?*anyopaque) void;

fn sokolAlloc(size: usize, _: ?*anyopaque) callconv(.c) ?*anyopaque {
    const ptr = malloc(size) orelse return null;
    sokol_live_allocations += 1;
    return ptr;
}

fn sokolFree(ptr: ?*anyopaque, _: ?*anyopaque) callconv(.c) void {
    if (ptr != null) {
        if (sokol_live_allocations == 0) fail("sokol free without ownership", .{}) else sokol_live_allocations -= 1;
    }
    free(ptr);
}

fn allocator() std.mem.Allocator {
    return if (builtin.cpu.arch.isWasm()) std.heap.c_allocator else gpa.allocator();
}

fn fail(comptime fmt: []const u8, args: anytype) void {
    failures += 1;
    std.debug.print("render-scale-gate FAIL: " ++ fmt ++ "\n", args);
}

fn envU32(name: [*:0]const u8, fallback: u32) u32 {
    const value = std.c.getenv(name) orelse return fallback;
    return std.fmt.parseInt(u32, std.mem.span(value), 10) catch fallback;
}

fn qLogger(tag: [*c]const u8, level: u32, item: u32, msg: [*c]const u8, line: u32, file: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    if (level <= 1) {
        log_errors += 1;
        fail("sokol log: {s}", .{msg});
    }
    slog.func(tag, level, item, msg, line, file, user_data);
}

fn setupScene() void {
    // Fixed, symmetric framing: the wall covers the whole viewport and the
    // camera never moves, so the light's screen position is a pure function
    // of its world position at both render scales.
    camera = z.ArcRotateCamera.init("rsg_cam", .{
        .alpha = std.math.pi / 2.0,
        .beta = std.math.pi / 2.0,
        .radius = 8.0,
        .target = z.Vec3.zero,
        .fov_deg = 60.0,
    });
    scene.active_camera = .{ .arc_rotate = camera };
    scene.clear_color = z.Color4.new(0.004, 0.005, 0.007, 1.0);
    // Scene.init installs a default white hemispheric light. The gate needs
    // the point light to be the ONLY illumination, or the "off" leg is just
    // a dimmer copy of the same wall instead of a black negative control.
    _ = scene.createHemisphericLight("off", .{
        .diffuse = z.Color3.black,
        .ground_color = z.Color3.black,
        .intensity = 0.0,
    });
    // The default 1x1 environment cube is a dark blue-grey, but it is sampled
    // as linear radiance, so it still lifts an unlit wall to ~0.33 sRGB.
    // The gate needs a true black floor: zero the IBL weight (the cube stays
    // bound, the shader takes the ibl_intensity <= 0.001 path).
    scene.sky.ibl_intensity = 0.0;

    const wall = z.MeshBuilder.createPlane(&scene, "wall", .{ .width = 24.0, .height = 14.0 }) catch @panic("render-scale-gate: wall failed");
    wall.position = z.Vec3.zero;
    const mat = scene.createPBRMaterial("wall_pbr") catch @panic("render-scale-gate: material failed");
    mat.albedo_color = z.Color3.new(0.82, 0.82, 0.82);
    mat.metallic = 0.0;
    mat.roughness = 0.95;
    wall.setPBRMaterial(mat);

    if (point_light_on) {
        // Right-of-centre, in front of the wall: its projected hotspot is
        // around x~0.68 of the canvas. A full-window tile grid at scale 0.5
        // never lists that tile for the half-res fragment, so the wall goes
        // dark — the exact regression this gate exists for.
        _ = scene.createPointLight("key", .{
            .position = z.Vec3.new(2.6, 0.4, 0.8),
            .color = z.Color3.new(1.0, 0.96, 0.9),
            .intensity = 8.0,
            .range = 1.6,
        }) catch @panic("render-scale-gate: point light failed");
    }

    scene.shadows.enabled = false;
    scene.ssao.enabled = false;
    scene.post_process.enabled = true;
    scene.post_process.exposure = 1.0;
    scene.post_process.tonemapping = .aces;
    scene.post_process.bloom.enabled = false;
    scene.post_process.dof.enabled = false;
    scene.post_process.shafts.enabled = false;
    scene.post_process.ssgi.enabled = false;
    scene.post_process.ssr.enabled = false;
    scene.post_process.fxaa.enabled = false;
    scene.post_process.vignette.enabled = false;
    scene.post_process.taa.enabled = false;
    scene.post_process.fog.enabled = false;
    scene.post_process.contact_shadows.enabled = false;
    scene.post_process.motion_blur.enabled = false;
    scene.post_process.render_scale = render_scale;
}

fn applyTick() void {
    scene.active_camera = .{ .arc_rotate = camera };
    scene.update(1.0 / 60.0) catch |err| fail("update: {s}", .{@errorName(err)});
    const w: i32 = @max(2, sapp.width());
    const h: i32 = @max(2, sapp.height());
    scene.publishFrameSnapshot(@as(f32, @floatFromInt(w)) / @as(f32, @floatFromInt(h)), w, h);
    if (!scene.buildPreparedFrame()) {
        fail("producer claim unavailable", .{});
        return;
    }
    const claim = scene.beginStagedPrepare() orelse {
        fail("staged claim unavailable", .{});
        return;
    };
    scene.finishStagedPrepare(claim);
    scene.render();
}

fn logTarget(label: []const u8, f: u32) void {
    // Realized target AND tile grid are the two structural inputs the
    // clustered lookup depends on; logging both makes a window-sized grid
    // visible even if a pixel check happened to pass for another reason.
    std.debug.print("render-scale-gate: {s} frame={} target={}x{} tiles={}x{} win={}x{} draws={} main={} rscale={d:.2} lit={}\n", .{
        label,
        f,
        scene.postfx.renderWidth(),
        scene.postfx.renderHeight(),
        scene.clustered.tiles_x,
        scene.clustered.tiles_y,
        sapp.width(),
        sapp.height(),
        scene.stats.draw_calls,
        scene.stats.main_draw_calls,
        render_scale,
        point_light_on,
    });
}

export fn init() callconv(.c) void {
    z.gpu_thread.markContextThread();
    sokol.time.setup();
    sg.setup(.{
        .environment = sokol.glue.environment(),
        .allocator = .{ .alloc_fn = sokolAlloc, .free_fn = sokolFree },
        .logger = .{ .func = qLogger },
        .pipeline_pool_size = 128,
        .shader_pool_size = 64,
        .view_pool_size = 128,
    });
    std.debug.print("render-scale-gate: setup backend={s} scale={d:.2} lit={} frames={}\n", .{
        @tagName(sg.queryBackend()), render_scale, point_light_on, frame_limit,
    });
    scene = z.Scene.init(allocator());
    setupScene();
}

export fn frame() callconv(.c) void {
    if (held) {
        held_ticks += 1;
        if (builtin.cpu.arch.isWasm() and held_ticks % 30 == 0) {
            const finish = emscripten_run_script_int("(()=>{return (window.rsgFinish===1)?1:0;})()");
            if (finish != 0) {
                std.debug.print("render-scale-gate: finish requested at held tick {}\n", .{held_ticks});
                sapp.quit();
            }
        }
        return;
    }
    frames += 1;
    const f = frames;
    applyTick();
    if (f == 1) logTarget("frame1", f);
    if (frame_limit > 0 and f >= frame_limit) {
        logTarget("capture", f);
        if (hold_final) {
            held = true;
            std.debug.print("render-scale-gate: held frame={}\n", .{f});
            return;
        }
        sapp.quit();
    }
}

export fn cleanup() callconv(.c) void {
    scene.deinit();
    sg.shutdown();
    if (sokol_live_allocations != 0) fail("sokol allocations after shutdown: {}", .{sokol_live_allocations});
    if (log_errors != 0) fail("sokol log errors: {}", .{log_errors});
    std.debug.print("render-scale-gate: teardown log_errors={} sokol_live_allocations={}\n", .{ log_errors, sokol_live_allocations });
    std.debug.print("render-scale-gate: verdict={s} failures={} frames={}\n", .{ if (failures == 0) "PASS" else "FAIL", failures, frames });
    if (failures != 0 and !builtin.cpu.arch.isWasm()) std.process.exit(1);
}

export fn event(ev: [*c]const sapp.Event) callconv(.c) void {
    if (ev.*.type == .KEY_DOWN and ev.*.key_code == .ESCAPE) sapp.quit();
}

extern fn emscripten_run_script_int(script: [*:0]const u8) c_int;

pub fn startApp() void {
    if (builtin.cpu.arch.isWasm()) {
        frame_limit = @intCast(@max(0, emscripten_run_script_int("(()=>{const n=parseInt(new URLSearchParams(location.search).get('frames')||'8',10);return isNaN(n)?0:n;})()")));
        render_scale = @as(f32, @floatFromInt(@max(1, emscripten_run_script_int("(()=>{const n=parseFloat(new URLSearchParams(location.search).get('rscale')||'0.5');return isNaN(n)?50:Math.round(n*100);})()")))) / 100.0;
        point_light_on = emscripten_run_script_int("(()=>{return new URLSearchParams(location.search).get('light')==='0'?0:1;})()") != 0;
        hold_final = emscripten_run_script_int("new URLSearchParams(location.search).has('hold')") != 0;
    } else {
        frame_limit = envU32("AGATE_RSG_FRAMES", 0);
        render_scale = envF32("AGATE_RSG_RSCALE", 0.5);
        point_light_on = !envFlag("AGATE_RSG_NO_LIGHT");
        hold_final = envFlag("AGATE_RSG_HOLD");
    }
    render_scale = @min(@max(render_scale, z.postprocess.min_render_scale), 1.0);
    sapp.run(.{
        .init_cb = init,
        .frame_cb = frame,
        .cleanup_cb = cleanup,
        .event_cb = event,
        .window_title = "Agate render-scale + clustered point gate",
        .width = 960,
        .height = 600,
        .sample_count = 1,
        .logger = .{ .func = qLogger },
    });
}

pub fn main() void {
    startApp();
}
