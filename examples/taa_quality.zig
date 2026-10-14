//! TAA pixel-quality temporal regression fixture (graphics-roadmap C.3 slice).
//!
//! Bounded subset: rigid opaque foreground + transparent glass over a
//! structured background, plus knob-gated skinned, instanced, and particle
//! motion classes (default OFF so the base 10 gate legs stay bit-identical).
//!
//! Fixture: high-contrast checker back wall, a white rigid box oscillating
//! in x (fixed-tick sinusoid, max velocity at the capture frame), and a
//! semi-transparent glass pane on a counter-phase path. The glass writes no
//! depth/velocity of its own, so only the velocity-pass reactive marker can
//! shed its history — the reactiveOFF leg is the negative control.
//!
//! Determinism: finite gate uses fixed dt=1/60, so the same frame number is
//! the same scene state on every leg. TAA legs run warm history; the
//! reference leg disables TAA for the exact same-frame instantaneous image.
//! Web `hold=1` FREEZES after the Nth frame: frame() issues no update, no
//! prepare, no render and no commit — the swapchain keeps presenting the Nth
//! encoded frame, so every CDP capture is that actual motion frame (never a
//! stationary-convergence tail). The gate captures twice, requires byte-exact
//! equality, then sets `window.tacaqFinish=1`; the page quits itself and runs
//! the normal cleanup, so the gate requires verdict=PASS with log_errors=0
//! and zero live allocations AFTER capture before closing the target.
//!
//! Knobs (native AGATE_TAAQ_* / web query): frames, taa (1/0), reactive
//! (1/0), render-scale/rscale (web fixed-point hundredths, e.g. 0.67),
//! cut (camera-cut frame, 0=none: the camera snaps +0.35 alpha once, an
//! ACTUAL view change both test and matched reference share), reset
//! (explicit resetTaa frame, 0=none), hold (freeze at limit; web gate uses
//! it, native via AGATE_TAAQ_HOLD=1).
//! Class knobs (default OFF — base 10 legs bit-identical without them):
//! skin/AGATE_TAAQ_SKINNED (skinned box + 2 animated bones crossing the ROI
//! sweep band) and inst/AGATE_TAAQ_INSTANCED (32-box InstancedMesh batch
//! with per-instance motion inside the ROI band). Both ride the same fixed
//! dt=1/60 trajectory: same frame number = same pose/matrix state.
//! Staged frame protocol mirrors q0-interior/hdr-showcase (serial path).

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

var frame_limit: u32 = 0;
var frames: u32 = 0;
var failures: u32 = 0;
var log_errors: u32 = 0;
var sokol_live_allocations: usize = 0;
var want_taa: bool = true;
var want_reactive: bool = true;
var render_scale: f32 = 1.0;
var cut_frame: u32 = 0;
var reset_frame: u32 = 0;
var hold_final: bool = false;
var held: bool = false;
var held_ticks: u32 = 0;
var cut_applied: bool = false;
var want_skinned: bool = false;
var want_instanced: bool = false;
var want_particles: bool = false;
var skin_mesh: ?*z.Mesh = null;
var skin_skel: ?*z.Skeleton = null;
var inst_source: ?*z.Mesh = null;
const INST_N: usize = 32;
const INST_COLS: usize = 8;
var inst_items: [INST_N]*z.InstancedMesh = undefined;
var inst_name_pool: [INST_N][16]u8 = undefined;
var camera: z.ArcRotateCamera = undefined;
var hero: *z.Mesh = undefined;
var glass: *z.Mesh = undefined;

/// Motion period in frames (fixed-tick sinusoid). Capture defaults to frame
/// 150 = phase 3*pi: hero crosses center at MAX velocity, so the held frame
/// carries genuine disocclusion with fully warm history behind it.
const MOTION_PERIOD: f32 = 100.0;

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
    std.debug.print("taa-quality FAIL: " ++ fmt ++ "\n", args);
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

fn pbr(name: []const u8, albedo: [3]f32, metallic: f32, roughness: f32) *z.PBRMaterial {
    const m = scene.createPBRMaterial(name) catch @panic("taa-quality: material failed");
    m.albedo_color = z.Color3.new(albedo[0], albedo[1], albedo[2]);
    m.metallic = metallic;
    m.roughness = roughness;
    return m;
}

/// Fixed-tick hero x at frame f: amplitude 1.1, max velocity at capture.
fn heroX(f: u32) f32 {
    const phase = 2.0 * std.math.pi * @as(f32, @floatFromInt(f)) / MOTION_PERIOD;
    return 1.1 * @sin(phase);
}

fn glassX(f: u32) f32 {
    const phase = 2.0 * std.math.pi * @as(f32, @floatFromInt(f)) / MOTION_PERIOD + 0.7;
    return -0.9 * @sin(phase);
}

/// Knob-gated skinned-box sweep: same period, phase-shifted so it crosses
/// the ROI band while the hero is elsewhere (capture f=150 -> x~-0.82).
fn skinX(f: u32) f32 {
    const phase = 2.0 * std.math.pi * @as(f32, @floatFromInt(f)) / MOTION_PERIOD + 2.0;
    return 0.9 * @sin(phase);
}

/// Deterministic bone offsets, pure functions of the frame number.
fn skinBoneA(f: u32) f32 {
    return 0.12 * @sin(2.0 * std.math.pi * @as(f32, @floatFromInt(f)) / 50.0);
}

fn skinBoneB(f: u32) f32 {
    return 0.10 * @cos(2.0 * std.math.pi * @as(f32, @floatFromInt(f)) / 50.0);
}

/// Per-instance deterministic pose inside the ROI band (z=-0.6 plane).
fn instPos(i: usize, f: u32) z.Vec3 {
    const col = i % INST_COLS;
    const row = i / INST_COLS;
    const base_x = -1.05 + (@as(f32, @floatFromInt(col)) + 0.5) * (2.1 / 8.0);
    const base_y = 0.55 + @as(f32, @floatFromInt(row)) * 0.32;
    const phase = 2.0 * std.math.pi * @as(f32, @floatFromInt(f)) / MOTION_PERIOD + @as(f32, @floatFromInt(i)) * 0.7;
    return z.Vec3.new(
        base_x + 0.25 * @sin(phase),
        base_y + 0.08 * @cos(phase + 0.4),
        -0.6,
    );
}

fn setupScene() void {
    camera = z.ArcRotateCamera.init("taaq_cam", .{
        // View from +Z: glass (+0.5), hero (-0.6), then checker (-2.5).
        .alpha = std.math.pi / 2.0,
        .beta = std.math.pi / 2.75,
        .radius = 6.2,
        .target = z.Vec3.new(0.0, 1.1, 0.0),
    });
    scene.active_camera = .{ .arc_rotate = camera };
    scene.clear_color = z.Color4.new(0.015, 0.016, 0.02, 1.0);

    _ = scene.createDirectionalLight("sun", .{
        .direction = z.Vec3.new(0.3, 1.0, 0.6),
        .diffuse = z.Color3.new(1.0, 0.95, 0.88),
        .intensity = 2.2,
    }) catch @panic("taa-quality: sun failed");
    _ = scene.createHemisphericLight("sky", .{
        .direction = z.Vec3.new(0.0, 1.0, 0.0),
        .diffuse = z.Color3.new(0.5, 0.55, 0.6),
        .ground_color = z.Color3.new(0.08, 0.08, 0.08),
        .intensity = 0.5,
    });
    scene.shadows.enabled = false;

    // Structured background: full-bleed high-contrast checker wall.
    const checker = z.Texture.createCheckerboard(allocator(), 512, 512, 16, .{ 12, 12, 14, 255 }, .{ 235, 232, 225, 255 }) catch @panic("taa-quality: checker failed");
    const bg = z.MeshBuilder.createPlane(&scene, "bg_wall", .{ .width = 14.0, .height = 8.0 }) catch @panic("taa-quality: wall failed");
    bg.position = z.Vec3.new(0.0, 2.0, -2.5);
    const bg_mat = pbr("bg_pbr", .{ 1, 1, 1 }, 0.0, 0.9);
    bg_mat.albedo_texture = checker;
    bg.setPBRMaterial(bg_mat);
    const floor = z.MeshBuilder.createGround(&scene, "floor", .{ .width = 14.0, .height = 10.0, .subdivisions = 1 }) catch @panic("taa-quality: floor failed");
    floor.position.z = 1.0;
    floor.setPBRMaterial(pbr("floor_pbr", .{ 0.03, 0.032, 0.038 }, 0.0, 0.9));

    // Rigid high-contrast foreground: near-white box over the checker.
    hero = z.MeshBuilder.createBox(&scene, "hero", .{ .width = 1.0, .height = 1.7, .depth = 0.5 }) catch @panic("taa-quality: hero failed");
    hero.position = z.Vec3.new(heroX(0), 1.0, -0.6);
    hero.setPBRMaterial(pbr("hero_pbr", .{ 0.93, 0.92, 0.9 }, 0.0, 0.5));

    // Moving transparent glass: velocity-less blended pixels, the true-marker
    // exercise. Alpha 0.35 over the checker/hero sweep path.
    glass = z.MeshBuilder.createPlane(&scene, "glass_pane", .{ .width = 2.2, .height = 2.3 }) catch @panic("taa-quality: glass failed");
    glass.position = z.Vec3.new(glassX(0), 1.1, 0.5);
    glass.cast_shadows = false;
    // Strong red tint + high alpha: an unshed glass ghost must carry a large
    // color residual against the neutral checker so the negative control
    // (reactive=0) separates from the resolve beyond the noise floor.
    const glass_mat = pbr("glass_pbr", .{ 0.9, 0.22, 0.16 }, 0.0, 0.08);
    glass_mat.alpha_mode = .blend;
    glass_mat.alpha = 0.7;
    glass_mat.double_sided = true;
    glass.setPBRMaterial(glass_mat);

    // Knob-gated classes (default OFF): same wiring as hdr-showcase finite
    // fixtures — skeleton registered on the scene, instances via
    // source.createInstance. Pure frame functions drive poses (see applyTick).
    if (want_skinned) {
        const skel = z.Skeleton.init(allocator(), 2) catch @panic("taa-quality: skeleton failed");
        scene.skeletons.append(allocator(), skel) catch @panic("taa-quality: skeleton registration failed");
        skin_skel = skel;
        const sm = z.MeshBuilder.createBox(&scene, "taaq_skin", .{ .size = 0.8 }) catch @panic("taa-quality: skinned mesh failed");
        sm.position = z.Vec3.new(skinX(0), 1.0, -0.6);
        sm.skeleton = skel;
        sm.culling_strategy = .always_render;
        sm.setPBRMaterial(pbr("skin_pbr", .{ 0.75, 0.45, 0.3 }, 0.0, 0.6));
        skin_mesh = sm;
    }
    if (want_instanced) {
        const source = z.MeshBuilder.createBox(&scene, "taaq_inst_source", .{ .size = 0.16 }) catch @panic("taa-quality: instance source failed");
        source.position = z.Vec3.new(0.0, 0.18, -0.6);
        source.culling_strategy = .always_render;
        source.setPBRMaterial(pbr("inst_pbr", .{ 0.3, 0.55, 0.7 }, 0.0, 0.5));
        inst_source = source;
        for (0..INST_N) |i| {
            const name = std.fmt.bufPrint(&inst_name_pool[i], "taaq_i{}", .{i}) catch @panic("taa-quality: instance name failed");
            const inst = source.createInstance(&scene, name) catch @panic("taa-quality: instance failed");
            inst.position = instPos(i, 0);
            inst_items[i] = inst;
        }
    }
    if (want_particles) {
        const dust = scene.createParticleSystem("taaq_dust", 64) catch @panic("taa-quality: dust failed");
        dust.emitter_position = z.Vec3.new(0.0, 1.1, -0.1);
        dust.emitter_box_min = z.Vec3.new(-1.0, -0.4, -0.3);
        dust.emitter_box_max = z.Vec3.new(1.0, 0.4, 0.3);
        dust.emit_rate = 20.0;
        dust.lifetime_min = 2.0;
        dust.lifetime_max = 3.5;
        dust.speed_min = 0.05;
        dust.speed_max = 0.22;
        dust.size_start = 0.06;
        dust.size_end = 0.02;
        dust.color_start = z.Color4.new(1.0, 0.85, 0.3, 0.8);
        dust.color_end = z.Color4.new(0.9, 0.3, 0.1, 0.0);
        dust.blend_mode = .alpha_blend;
        dust.start();
    }

    // Minimal post chain: TAA under test, everything else off so the gate
    // measures temporal resolve only (no DoF/bloom/shafts/SSGI/SSR blur).
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
    scene.post_process.taa.enabled = want_taa;
    scene.post_process.taa.reactive.enabled = want_reactive;
    scene.post_process.render_scale = render_scale;
    scene.ssao.enabled = false;
    scene.post_process.fog.enabled = false;
}

fn applyTick(f: u32) void {
    hero.position.x = heroX(f);
    glass.position.x = glassX(f);
    if (skin_mesh) |sm| {
        sm.position.x = skinX(f);
        if (skin_skel) |sk| {
            sk.bones[0].local_position.y = skinBoneA(f);
            if (sk.bones.len > 1) sk.bones[1].local_position.x = skinBoneB(f);
            sk.update();
        }
    }
    if (inst_source != null) {
        for (0..INST_N) |i| inst_items[i].position = instPos(i, f);
    }
    scene.active_camera = .{ .arc_rotate = camera };
    scene.update(1.0 / 60.0) catch |err| fail("update: {s}", .{@errorName(err)});
    const w: i32 = @max(2, sapp.width());
    const h: i32 = @max(2, sapp.height());
    scene.publishFrameSnapshot(@as(f32, @floatFromInt(w)) / @as(f32, @floatFromInt(h)), w, h);
    if (!scene.buildPreparedFrame()) {
        fail("producer claim unavailable at frame {}", .{f});
        return;
    }
    const claim = scene.beginStagedPrepare() orelse {
        fail("staged claim unavailable at frame {}", .{f});
        return;
    };
    scene.finishStagedPrepare(claim);
    scene.render();
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
    std.debug.print("taa-quality: setup backend={s} taa={} reactive={} scale={d:.2} frames={} cut={} reset={} skin={} inst={} part={}\n", .{
        @tagName(sg.queryBackend()), want_taa, want_reactive, render_scale, frame_limit, cut_frame, reset_frame, want_skinned, want_instanced, want_particles,
    });
    scene = z.Scene.init(allocator());
    setupScene();
}

export fn frame() callconv(.c) void {
    // Held tail (web gate): FROZEN — no update, no prepare, no render, no
    // commit. The swapchain keeps presenting the Nth encoded frame, so every
    // CDP capture is that actual motion frame (a re-render here would advance
    // TAA ping-pong/jitter/history and measure stationary convergence).
    // Controlled finish only: the gate sets window.tacaqFinish, the page
    // quits itself and runs the normal cleanup (teardown proof included).
    if (held) {
        held_ticks += 1;
        if (held_ticks % 60 == 0) std.debug.print("taa-quality: held tick={} frame={}\n", .{ held_ticks, frames });
        if (builtin.cpu.arch.isWasm() and held_ticks % 30 == 0) {
            const finish = emscripten_run_script_int("(()=>{return (window.tacaqFinish===1)?1:0;})()");
            if (finish != 0) {
                std.debug.print("taa-quality: finish requested at held tick {}\n", .{held_ticks});
                sapp.quit();
            }
        }
        return;
    }
    frames += 1;
    const f = frames;
    // Actual camera change at the cut frame (one-time +0.35 alpha snap, kept
    // afterwards): history is genuinely invalid, identically on the test leg
    // and its matched reference. The composite flag rides the same frame.
    if (cut_frame != 0 and f >= cut_frame and !cut_applied) {
        camera.alpha += 0.35;
        cut_applied = true;
    }
    if (cut_frame != 0 and f == cut_frame) scene.post_process.taa.camera_cut = true;
    if (cut_frame != 0 and f == cut_frame + 1) scene.post_process.taa.camera_cut = false;
    if (reset_frame != 0 and f == reset_frame) scene.resetTaa();
    applyTick(f);
    if (f == 1) {
        if (scene.stats.draw_calls == 0) fail("no draw calls at frame 1", .{});
        std.debug.print("taa-quality: frame1 draws={} main_draws={} post_draws={} target={}x{}\n", .{
            scene.stats.draw_calls, scene.stats.main_draw_calls, scene.stats.post_draw_calls, scene.postfx.renderWidth(), scene.postfx.renderHeight(),
        });
    }
    if (f == 30) {
        const vp = &scene.postfx.velocity_pass;
        const vel_ok = vp.width == scene.postfx.renderWidth() and vp.height == scene.postfx.renderHeight();
        const markers = z.postprocess.taa.taaReactiveMarkersActive(want_taa, scene.post_process);
        const skin_x: f32 = if (skin_mesh) |sm| sm.position.x else -99.0;
        const inst0_x: f32 = if (inst_source != null) inst_items[0].position.x else -99.0;
        std.debug.print("taa-quality: activity frame=30 draws={} main_draws={} post_draws={} velocity_sized={} markers_active={} hero_x={d:.3} glass_x={d:.3} skin_x={d:.3} inst0_x={d:.3}\n", .{
            scene.stats.draw_calls, scene.stats.main_draw_calls, scene.stats.post_draw_calls, vel_ok, markers, hero.position.x, glass.position.x, skin_x, inst0_x,
        });
        if (!vel_ok and want_taa) fail("velocity target unsized at frame 30", .{});
    }
    if (frame_limit > 0 and f >= frame_limit) {
        const markers = z.postprocess.taa.taaReactiveMarkersActive(want_taa, scene.post_process);
        const reset_pending = scene.taa_reset_requested.load(.acquire);
        const skin_x: f32 = if (skin_mesh) |sm| sm.position.x else -99.0;
        const inst0_x: f32 = if (inst_source != null) inst_items[0].position.x else -99.0;
        std.debug.print("taa-quality: capture frame={} draws={} main_draws={} post_draws={} markers_active={} taa_frame={} reset_pending={} cut_applied={} target={}x{} hero_x={d:.3} glass_x={d:.3} skin_x={d:.3} inst0_x={d:.3}\n", .{
            f,                          scene.stats.draw_calls,      scene.stats.main_draw_calls, scene.stats.post_draw_calls, markers, scene.postfx.taa_frame, reset_pending, cut_applied,
            scene.postfx.renderWidth(), scene.postfx.renderHeight(), hero.position.x,             glass.position.x,            skin_x,  inst0_x,
        });
        if (hold_final) {
            held = true;
            std.debug.print("taa-quality: held frame={}\n", .{f});
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
    std.debug.print("taa-quality: teardown log_errors={} sokol_live_allocations={}\n", .{ log_errors, sokol_live_allocations });
    std.debug.print("taa-quality: verdict={s} failures={} frames={}\n", .{ if (failures == 0) "PASS" else "FAIL", failures, frames });
    if (failures != 0 and !builtin.cpu.arch.isWasm()) std.process.exit(1);
}

export fn event(ev: [*c]const sapp.Event) callconv(.c) void {
    if (ev.*.type == .KEY_DOWN and ev.*.key_code == .ESCAPE) sapp.quit();
}

extern fn emscripten_run_script_int(script: [*:0]const u8) c_int;

pub fn startApp() void {
    if (builtin.cpu.arch.isWasm()) {
        frame_limit = @intCast(@max(0, emscripten_run_script_int("(()=>{const n=parseInt(new URLSearchParams(location.search).get('frames')||'150',10);return isNaN(n)?0:n;})()")));
        want_taa = emscripten_run_script_int("(()=>{return new URLSearchParams(location.search).get('taa')==='0'?0:1;})()") != 0;
        want_reactive = emscripten_run_script_int("(()=>{return new URLSearchParams(location.search).get('reactive')==='0'?0:1;})()") != 0;
        // Fixed-point hundredths: emscripten_run_script_int truncates floats,
        // so rscale=0.67 arrives as 67 (unambiguous; 0.67 -> 0 is the old bug).
        render_scale = @as(f32, @floatFromInt(@max(1, emscripten_run_script_int("(()=>{const n=parseFloat(new URLSearchParams(location.search).get('rscale')||'1');return isNaN(n)?100:Math.round(n*100);})()")))) / 100.0;
        cut_frame = @intCast(@max(0, emscripten_run_script_int("(()=>{const n=parseInt(new URLSearchParams(location.search).get('cut')||'0',10);return isNaN(n)?0:n;})()")));
        reset_frame = @intCast(@max(0, emscripten_run_script_int("(()=>{const n=parseInt(new URLSearchParams(location.search).get('reset')||'0',10);return isNaN(n)?0:n;})()")));
        hold_final = emscripten_run_script_int("new URLSearchParams(location.search).has('hold')") != 0;
        want_skinned = emscripten_run_script_int("(()=>{return new URLSearchParams(location.search).get('skin')==='1'?1:0;})()") != 0;
        want_instanced = emscripten_run_script_int("(()=>{return new URLSearchParams(location.search).get('inst')==='1'?1:0;})()") != 0;
        want_particles = emscripten_run_script_int("(()=>{return new URLSearchParams(location.search).get('part')==='1'?1:0;})()") != 0;
    } else {
        frame_limit = envU32("AGATE_TAAQ_FRAMES", 0);
        want_taa = !envFlag("AGATE_TAAQ_NO_TAA");
        want_reactive = !envFlag("AGATE_TAAQ_NO_REACTIVE");
        render_scale = envF32("AGATE_TAAQ_RENDER_SCALE", 1.0);
        cut_frame = envU32("AGATE_TAAQ_CUT", 0);
        reset_frame = envU32("AGATE_TAAQ_RESET", 0);
        hold_final = envFlag("AGATE_TAAQ_HOLD");
        want_skinned = envFlag("AGATE_TAAQ_SKINNED");
        want_instanced = envFlag("AGATE_TAAQ_INSTANCED");
        want_particles = envFlag("AGATE_TAAQ_PARTICLES");
    }
    render_scale = @min(@max(render_scale, z.postprocess.min_render_scale), 1.0);
    sapp.run(.{
        .init_cb = init,
        .frame_cb = frame,
        .cleanup_cb = cleanup,
        .event_cb = event,
        .window_title = "Agate TAA quality fixture (C.3 slice)",
        .width = 960,
        .height = 600,
        .sample_count = 1,
        .logger = .{ .func = qLogger },
    });
}

pub fn main() void {
    startApp();
}
