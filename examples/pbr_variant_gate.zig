//! PBR variant parity fixture: rigid vs skinned vs instanced.
//!
//! Renders ONE box at the exact same transform/material/lighting in three
//! legs (`variant=rigid|skinned|instanced`). The floor, sky/IBL, directional
//! sun, point/spot shadows, area light and clustered extra are identical, so
//! the three held frames should be pixel-identical apart from compiler
//! rounding. This is the pixel gate for the shared PBR fragment body: a
//! dedup mistake that breaks only the skinned or instanced wiring cannot
//! hide behind a matching reference rendered with the same broken shader.
//!
//! The box uses clearcoat + sheen + anisotropy (no textures) so the shared
//! lobe tail, clustered loop, legacy point/spot loops, area lights, IBL and
//! CSM/spot/point shadow code all execute. `hold=1` freezes the capture
//! frame; the page quits itself via `window.pvgFinish=1` and must report a
//! clean teardown.
//!
//! Knobs (native `AGATE_PVG_*` / web query): frames, variant, hold.

const std = @import("std");
const builtin = @import("builtin");
const z = @import("agate");
const sokol = @import("sokol");
const sapp = sokol.app;
const slog = sokol.log;
const sg = sokol.gfx;
const env_mod = @import("env.zig");
const envFlag = env_mod.envFlag;

var gpa = std.heap.DebugAllocator(.{ .thread_safe = true }){};
var scene: z.Scene = undefined;

const Variant = enum { rigid, skinned, instanced };

var frame_limit: u32 = 10;
var frames: u32 = 0;
var failures: u32 = 0;
var log_errors: u32 = 0;
var sokol_live_allocations: usize = 0;
var variant: Variant = .rigid;
var hold_final: bool = false;
var held: bool = false;
var held_ticks: u32 = 0;
var camera: z.ArcRotateCamera = undefined;
var skeleton: ?*z.Skeleton = null;

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
    std.debug.print("pbr-variant-gate FAIL: " ++ fmt ++ "\n", args);
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

fn variantName(v: Variant) []const u8 {
    return @tagName(v);
}

fn pbrMaterial(name: []const u8) *z.PBRMaterial {
    const mat = scene.createPBRMaterial(name) catch @panic("pbr-variant-gate: material failed");
    // Deliberately non-trivial: every shared lobe tail branch executes.
    mat.albedo_color = z.Color3.new(0.36, 0.52, 0.74);
    mat.metallic = 0.35;
    mat.roughness = 0.34;
    mat.clearcoat = .{ .intensity = 0.7, .roughness = 0.16, .color = z.Color3.new(1.0, 0.95, 0.85) };
    mat.sheen = .{ .intensity = 0.45, .roughness = 0.4, .color = z.Color3.new(0.9, 0.45, 0.3) };
    mat.anisotropy = .{ .intensity = 0.5, .rotation = 0.6 };
    return mat;
}

fn setupScene() void {
    camera = z.ArcRotateCamera.init("pvg_cam", .{
        .alpha = std.math.pi / 2.0,
        .beta = 1.22,
        .radius = 7.0,
        .target = z.Vec3.new(0, 1.0, 0),
        .fov_deg = 42.0,
    });
    scene.active_camera = .{ .arc_rotate = camera };
    scene.clear_color = z.Color4.new(0.012, 0.014, 0.02, 1.0);

    // Shared floor: shadow receiver for CSM/spot/point, never variant-specific.
    const floor = z.MeshBuilder.createGround(&scene, "floor", .{ .width = 24.0, .height = 24.0, .subdivisions = 1 }) catch @panic("pbr-variant-gate: floor failed");
    const floor_mat = scene.createPBRMaterial("floor_pbr") catch @panic("pbr-variant-gate: floor material failed");
    floor_mat.albedo_color = z.Color3.new(0.16, 0.17, 0.2);
    floor_mat.metallic = 0.0;
    floor_mat.roughness = 0.9;
    floor.setPBRMaterial(floor_mat);
    floor.receive_shadows = true;
    floor.cast_shadows = false;

    const mat = pbrMaterial("box_pbr");
    if (variant == .rigid) {
        const box = z.MeshBuilder.createBox(&scene, "box", .{ .size = 2.0 }) catch @panic("pbr-variant-gate: box failed");
        box.position = z.Vec3.new(0, 1.0, 0);
        box.setPBRMaterial(mat);
        box.receive_shadows = true;
        box.cast_shadows = true;
    } else if (variant == .skinned) {
        const skel = z.Skeleton.init(allocator(), 1) catch @panic("pbr-variant-gate: skeleton failed");
        scene.skeletons.append(allocator(), skel) catch @panic("pbr-variant-gate: skeleton registration failed");
        skeleton = skel;
        const box = z.MeshBuilder.createBox(&scene, "box_skinned", .{ .size = 2.0 }) catch @panic("pbr-variant-gate: skinned box failed");
        box.position = z.Vec3.new(0, 1.0, 0);
        box.skeleton = skel;
        // The primitive's default skin is joint 0 / weight 1; one identity
        // bone makes the skinned transform exactly the rigid one.
        skel.update();
        box.setPBRMaterial(mat);
        box.receive_shadows = true;
        box.cast_shadows = true;
    } else {
        const source = z.MeshBuilder.createBox(&scene, "box_inst_source", .{ .size = 2.0 }) catch @panic("pbr-variant-gate: instance source failed");
        source.position = z.Vec3.new(0, 1.0, 0);
        source.setPBRMaterial(mat);
        source.is_visible = false; // source with instances is queue-excluded anyway
        source.receive_shadows = true;
        source.cast_shadows = true;
        const inst = source.createInstance(&scene, "box_inst") catch @panic("pbr-variant-gate: instance failed");
        inst.position = z.Vec3.new(0, 1.0, 0);
        inst.receive_shadows = true;
        inst.cast_shadows = true;
    }

    // Lighting: every shared forward branch gets a live input.
    _ = scene.createHemisphericLight("sky", .{
        .direction = z.Vec3.new(0, 1, 0),
        .diffuse = z.Color3.new(0.28, 0.34, 0.46),
        .ground_color = z.Color3.new(0.05, 0.055, 0.07),
        .intensity = 0.55,
    });
    _ = scene.createDirectionalLight("sun", .{
        .direction = z.Vec3.new(-0.35, 1.0, 0.55),
        .diffuse = z.Color3.new(1.0, 0.94, 0.84),
        .intensity = 2.6,
    }) catch @panic("pbr-variant-gate: sun failed");
    _ = scene.createPointLight("point", .{
        .position = z.Vec3.new(3.0, 3.6, 2.2),
        .color = z.Color3.new(0.45, 0.75, 1.0),
        .intensity = 26.0,
        .range = 14.0,
        .shadow = .{ .cast_shadows = true, .shadow_bias = 0.0025, .shadow_normal_bias = 0.004 },
    }) catch @panic("pbr-variant-gate: point failed");
    _ = scene.createSpotLight("spot", .{
        .position = z.Vec3.new(-3.0, 4.5, 2.0),
        .direction = z.Vec3.new(0.35, -1.0, -0.25),
        .color = z.Color3.new(1.0, 0.55, 0.3),
        .intensity = 40.0,
        .range = 18.0,
        .inner_angle_deg = 24.0,
        .outer_angle_deg = 48.0,
        .shadow = .{ .cast_shadows = true, .shadow_bias = 0.002, .shadow_normal_bias = 0.003 },
    }) catch @panic("pbr-variant-gate: spot failed");
    _ = scene.addAreaLight("area", .{
        .center = z.Vec3.new(0.0, 3.0, 3.0),
        .right = z.Vec3.new(1.2, 0, 0),
        .up = z.Vec3.new(0, 0.0, 1.2),
        .color = z.Color3.new(0.55, 1.0, 0.65),
        .intensity = 5.0,
    }) catch @panic("pbr-variant-gate: area failed");
    _ = scene.addClusteredPointLight(z.Vec3.new(2.0, 1.6, -1.8), .{
        .color = z.Color3.new(1.0, 0.35, 0.6),
        .intensity = 18.0,
        .radius = 7.0,
    }) catch @panic("pbr-variant-gate: clustered failed");

    scene.shadows.enabled = true;
    // IBL branch active; the default cube is the shared environment.
    scene.sky.ibl_intensity = 0.8;
    scene.ssao.enabled = false;
    scene.post_process.enabled = true;
    scene.post_process.exposure = 1.0;
    scene.post_process.tonemapping = .aces;
    scene.post_process.render_scale = 1.0;
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
}

fn applyTick() void {
    scene.active_camera = .{ .arc_rotate = camera };
    if (skeleton) |skel| skel.update();
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
    std.debug.print("pbr-variant-gate: {s} frame={} variant={s} target={}x{} draws={} main={} shadow={}\n", .{
        label,
        f,
        variantName(variant),
        scene.postfx.renderWidth(),
        scene.postfx.renderHeight(),
        scene.stats.draw_calls,
        scene.stats.main_draw_calls,
        scene.stats.shadow_draw_calls,
    });
}

export fn init() callconv(.c) void {
    z.gpu_thread.markContextThread();
    sokol.time.setup();
    sg.setup(.{
        .environment = sokol.glue.environment(),
        .allocator = .{ .alloc_fn = sokolAlloc, .free_fn = sokolFree },
        .logger = .{ .func = qLogger },
        .pipeline_pool_size = 256,
        .shader_pool_size = 128,
        .view_pool_size = 256,
    });
    std.debug.print("pbr-variant-gate: setup backend={s} variant={s} frames={}\n", .{
        @tagName(sg.queryBackend()), variantName(variant), frame_limit,
    });
    scene = z.Scene.init(allocator());
    setupScene();
}

export fn frame() callconv(.c) void {
    if (held) {
        held_ticks += 1;
        if (builtin.cpu.arch.isWasm() and held_ticks % 30 == 0) {
            const finish = emscripten_run_script_int("(()=>{return (window.pvgFinish===1)?1:0;})()");
            if (finish != 0) {
                std.debug.print("pbr-variant-gate: finish requested at held tick {}\n", .{held_ticks});
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
            std.debug.print("pbr-variant-gate: held frame={}\n", .{f});
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
    std.debug.print("pbr-variant-gate: teardown log_errors={} sokol_live_allocations={}\n", .{ log_errors, sokol_live_allocations });
    std.debug.print("pbr-variant-gate: verdict={s} failures={} frames={}\n", .{ if (failures == 0) "PASS" else "FAIL", failures, frames });
    if (failures != 0 and !builtin.cpu.arch.isWasm()) std.process.exit(1);
}

export fn event(ev: [*c]const sapp.Event) callconv(.c) void {
    if (ev.*.type == .KEY_DOWN and ev.*.key_code == .ESCAPE) sapp.quit();
}

extern fn emscripten_run_script_int(script: [*:0]const u8) c_int;

pub fn startApp() void {
    if (builtin.cpu.arch.isWasm()) {
        frame_limit = @intCast(@max(0, emscripten_run_script_int("(()=>{const n=parseInt(new URLSearchParams(location.search).get('frames')||'10',10);return isNaN(n)?0:n;})()")));
        const v = emscripten_run_script_int("(()=>{const q=(new URLSearchParams(location.search).get('variant')||'rigid');return q==='skinned'?1:(q==='instanced'?2:0);})()");
        variant = switch (v) {
            1 => .skinned,
            2 => .instanced,
            else => .rigid,
        };
        hold_final = emscripten_run_script_int("new URLSearchParams(location.search).has('hold')") != 0;
    } else {
        frame_limit = envU32("AGATE_PVG_FRAMES", 0);
        if (std.c.getenv("AGATE_PVG_VARIANT")) |raw| {
            const s = std.mem.span(raw);
            variant = if (std.mem.eql(u8, s, "skinned")) .skinned else if (std.mem.eql(u8, s, "instanced")) .instanced else .rigid;
        }
        hold_final = envFlag("AGATE_PVG_HOLD");
    }
    sapp.run(.{
        .init_cb = init,
        .frame_cb = frame,
        .cleanup_cb = cleanup,
        .event_cb = event,
        .window_title = "Agate PBR variant parity fixture",
        .width = 960,
        .height = 600,
        .sample_count = 1,
        .logger = .{ .func = qLogger },
    });
}

pub fn main() void {
    startApp();
}
