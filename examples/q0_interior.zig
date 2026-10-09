//! Q0 calibration interior (fill-bound A/B workload, graphics-roadmap Q0/Q4).
//!
//! A Cornell-style room — floor, three walls, ceiling, a tall colored box,
//! a white sphere and a mirror sphere — lit by a warm ceiling area panel
//! plus a sun through the open side (shadows + volumetric shafts). The
//! fullscreen post chain runs at maximum per-pixel weight on purpose:
//! SSR (32 steps, depth pyramid), DoF, shafts, a 7-mip bloom pyramid,
//! local tonemapping and TAA. That makes the frame FILL-BOUND — the class
//! of workload the hdr-showcase A/B could not separate from harness noise
//! (MEASUREMENTS.md wave 3: that scene is draw-bound at 102 draws).
//!
//! Frame flow is the serial staged path (build -> begin -> finish ->
//! render) exactly like the other examples; the shared three-phase
//! render-scale bench (`gpu_phase_bench.zig`) drives the A/B:
//!
//!   AGATE_Q0_FRAMES=540 AGATE_Q0_TIMINGS=1 ./zig-out/bin/q0-interior
//!   (phases 1.0 -> 0.75 -> 0.66 x 180 frames; median is canonical)
//!
//! No licensed assets yet (Q0 proper calls for them): the checker floor
//! and solid albedo panels carry the sharpness/ghosting story for now.
//! Run: `zig build run-q0 -- --frames 540` (or the env knobs above).

const std = @import("std");
const z = @import("agate");
const sokol = @import("sokol");
const sapp = sokol.app;
const slog = sokol.log;
const sg = sokol.gfx;
const timing = z.gpu_timing;
const bench_mod = @import("gpu_phase_bench.zig");

var gpa = std.heap.DebugAllocator(.{ .thread_safe = true }){};
var scene: z.Scene = undefined;

var bench: bench_mod.Bench = .{};
var frame_limit: u32 = 0;
var frames: u32 = 0;
var log_errors: u32 = 0;
var failures: u32 = 0;
var render_scale: f32 = 1.0;
var camera: z.ArcRotateCamera = undefined;

fn fail(comptime fmt: []const u8, args: anytype) void {
    failures += 1;
    std.debug.print("q0-interior FAIL: " ++ fmt ++ "\n", args);
}

fn envU32(name: [*:0]const u8, fallback: u32) u32 {
    const value = std.c.getenv(name) orelse return fallback;
    return std.fmt.parseInt(u32, std.mem.span(value), 10) catch fallback;
}

fn envF32(name: [*:0]const u8, fallback: f32) f32 {
    const value = std.c.getenv(name) orelse return fallback;
    return std.fmt.parseFloat(f32, std.mem.span(value)) catch fallback;
}

fn envFlag(name: [*:0]const u8) bool {
    const value = std.c.getenv(name) orelse return false;
    const s = std.mem.span(value);
    return std.mem.eql(u8, s, "1") or std.mem.eql(u8, s, "true");
}

fn qLogger(tag: [*c]const u8, level: u32, item: u32, msg: [*c]const u8, line: u32, file: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    if (level <= 1) {
        log_errors += 1;
        fail("sokol log: {s}", .{msg});
    }
    slog.func(tag, level, item, msg, line, file, user_data);
}

// ---- scene -----------------------------------------------------------------

fn pbr(name: []const u8, albedo: [3]f32, metallic: f32, roughness: f32) *z.PBRMaterial {
    const m = scene.createPBRMaterial(name) catch @panic("q0: material failed");
    m.albedo_color = z.Color3.new(albedo[0], albedo[1], albedo[2]);
    m.metallic = metallic;
    m.roughness = roughness;
    return m;
}

fn wall(name: []const u8, w: f32, h: f32) *z.Mesh {
    const m = z.MeshBuilder.createPlane(&scene, name, .{ .width = w, .height = h, .subdivisions_x = 2, .subdivisions_y = 2 }) catch @panic("q0: wall failed");
    m.cast_shadows = true;
    m.receive_shadows = true;
    return m;
}

fn setupScene() void {
    scene.active_camera = .{ .arc_rotate = camera };
    scene.clear_color = z.Color4.new(0.02, 0.02, 0.025, 1.0);

    _ = scene.createHemisphericLight("hemi", .{
        .direction = z.Vec3.new(0.1, 1.0, 0.2),
        .diffuse = z.Color3.white,
        .ground_color = z.Color3.new(0.12, 0.1, 0.09),
        .intensity = 0.18,
    });

    // Sun through the open front side: drives CSM + volumetric shafts.
    _ = scene.createDirectionalLight("sun", .{
        .direction = z.Vec3.new(0.25, 1.4, 1.0),
        .diffuse = z.Color3.new(1.0, 0.92, 0.8),
        .intensity = 2.6,
    }) catch @panic("q0: sun failed");
    scene.shadows.enabled = true;

    // Warm ceiling area panel + emissive quad (bloom picks the emitter up).
    _ = scene.addAreaLight("area_panel", .{
        .center = z.Vec3.new(0.0, 3.85, -0.8),
        .right = z.Vec3.new(1.4, 0.0, 0.0),
        .up = z.Vec3.new(0.0, 0.0, 1.0),
        .color = z.Color3.new(1.0, 0.72, 0.45),
        .intensity = 9.0,
        .is_enabled = true,
    }) catch null;
    const panel = z.MeshBuilder.createPlane(&scene, "area_panel_quad", .{ .width = 2.8, .height = 2.0 }) catch @panic("q0: panel failed");
    panel.position = z.Vec3.new(0.0, 3.9, -0.8);
    panel.rotation = z.Vec3.new(90.0, 0.0, 0.0);
    panel.cast_shadows = false;
    const panel_mat = pbr("area_panel_pbr", .{ 1.0, 0.8, 0.55 }, 0.0, 0.9);
    panel_mat.emissive_color = z.Color3.new(2.4, 1.4, 0.6);
    panel_mat.double_sided = true;
    panel.setPBRMaterial(panel_mat);

    // Room shell: floor + back/left/right walls + ceiling (open front).
    const floor = z.MeshBuilder.createGround(&scene, "floor", .{ .width = 6.0, .height = 6.0, .subdivisions = 8 }) catch @panic("q0: floor failed");
    const checker = z.Texture.createCheckerboard(gpa.allocator(), 512, 512, 32, .{ 26, 26, 30, 255 }, .{ 208, 203, 196, 255 }) catch @panic("q0: checker failed");
    const floor_mat = pbr("floor_pbr", .{ 1, 1, 1 }, 0.0, 0.55);
    floor_mat.albedo_texture = checker;
    floor_mat.roughness = 0.5;
    floor.setPBRMaterial(floor_mat);

    const back = wall("wall_back", 6.0, 4.0);
    back.position = z.Vec3.new(0.0, 2.0, -3.0);
    back.setPBRMaterial(pbr("wall_back_pbr", .{ 0.62, 0.64, 0.66 }, 0.0, 0.85));
    const left = wall("wall_left", 6.0, 4.0);
    left.position = z.Vec3.new(-3.0, 2.0, 0.0);
    left.rotation.y = 90.0;
    left.setPBRMaterial(pbr("wall_left_pbr", .{ 0.72, 0.36, 0.30 }, 0.0, 0.85));
    const right = wall("wall_right", 6.0, 4.0);
    right.position = z.Vec3.new(3.0, 2.0, 0.0);
    right.rotation.y = -90.0;
    right.setPBRMaterial(pbr("wall_right_pbr", .{ 0.34, 0.40, 0.55 }, 0.0, 0.85));
    const ceiling = wall("ceiling", 6.0, 6.0);
    ceiling.position = z.Vec3.new(0.0, 4.0, 0.0);
    ceiling.rotation.x = 90.0;
    ceiling.cast_shadows = false;
    ceiling.setPBRMaterial(pbr("ceiling_pbr", .{ 0.9, 0.88, 0.85 }, 0.0, 0.9));

    // Content: tall red box, white sphere, mirror sphere (SSR showpiece).
    const tall = z.MeshBuilder.createBox(&scene, "tall_box", .{ .width = 1.5, .height = 2.4, .depth = 1.5 }) catch @panic("q0: box failed");
    tall.position = z.Vec3.new(-1.7, 1.2, -1.3);
    tall.setPBRMaterial(pbr("tall_pbr", .{ 0.78, 0.22, 0.18 }, 0.0, 0.7));
    const ball = z.MeshBuilder.createSphere(&scene, "white_sphere", .{ .diameter = 1.6, .segments = 48 }) catch @panic("q0: sphere failed");
    ball.position = z.Vec3.new(1.5, 0.8, -1.1);
    ball.setPBRMaterial(pbr("ball_pbr", .{ 0.95, 0.93, 0.9 }, 0.0, 0.45));
    const mirror = z.MeshBuilder.createSphere(&scene, "mirror_sphere", .{ .diameter = 1.1, .segments = 48 }) catch @panic("q0: mirror failed");
    mirror.position = z.Vec3.new(0.1, 0.55, 1.0);
    const mirror_mat = pbr("mirror_pbr", .{ 0.98, 0.98, 0.98 }, 1.0, 0.04);
    mirror.setPBRMaterial(mirror_mat);

    // Fill-bound by design: the heaviest legal per-pixel chain.
    scene.post_process.enabled = true;
    scene.post_process.exposure = 1.1;
    scene.post_process.tonemapping = .agx;
    scene.post_process.bloom_enabled = true;
    scene.post_process.bloom_threshold = 1.0;
    scene.post_process.bloom_intensity = 0.55;
    scene.post_process.bloom_pyramid_mips = 7;
    scene.post_process.ssr_enabled = true;
    scene.post_process.ssr_intensity = 0.6;
    scene.post_process.ssr_steps = 32;
    scene.post_process.depth_pyramid_enabled = true;
    scene.post_process.dof_enabled = true;
    scene.post_process.dof_focus_distance = 4.6;
    scene.post_process.dof_focus_range = 2.2;
    scene.post_process.dof_max_blur = 5.0;
    scene.post_process.shaft_enabled = true;
    scene.post_process.shaft_intensity = 0.5;
    scene.post_process.shaft_steps = 32;
    scene.post_process.local_tonemapping_enabled = true;
    scene.post_process.taa_enabled = true;
    scene.post_process.taa_camera_cut = true;
    scene.post_process.fog_enabled = false;
    // Single-scale runs keep the env selection via overrideScale; full
    // 3-phase A/B runs start at the bench schedule's phase 0 (which a
    // reversed order sets to 0.66 — the env default must not clobber it).
    if (frame_limit < bench.scales.len * bench.phase_frames) bench.overrideScale(render_scale);
    scene.post_process.render_scale = bench.currentScale();
}

// ---- sokol app --------------------------------------------------------------

export fn init() callconv(.c) void {
    z.gpu_thread.markContextThread();
    sokol.time.setup();
    sg.setup(.{
        .environment = sokol.glue.environment(),
        .logger = .{ .func = qLogger },
        .pipeline_pool_size = 256,
        .shader_pool_size = 128,
        .view_pool_size = 256,
    });
    timing.setEnabled(envFlag("AGATE_Q0_TIMINGS") or envFlag("AGATE_GPU_TIMINGS"));
    camera = z.ArcRotateCamera.init("q0_cam", .{
        .alpha = -std.math.pi / 2.0,
        .beta = std.math.pi / 2.75,
        .radius = 5.4,
        .target = z.Vec3.new(0.0, 1.2, -0.4),
    });
    scene = z.Scene.init(gpa.allocator());
    setupScene();
}

export fn frame() callconv(.c) void {
    frames += 1;
    const f = frames;
    const gate = frame_limit > 0;
    defer if (gate and f >= frame_limit) sapp.quit();
    if (gate) {
        if (bench.beginPhase(f, frame_limit)) |finished| bench_mod.Bench.printPhase(finished);
        if (bench.phase_started) {
            scene.post_process.render_scale = bench.currentScale();
            scene.resetTaa();
        }
    }
    // Deterministic motion under the gate: same jitter/animation frame for
    // cross-run compare; slow drift keeps TAA/SSR/DoF genuinely working.
    const dt: f32 = if (gate) 1.0 / 60.0 else @floatCast(sapp.frameDuration());
    if (f == 2) scene.post_process.taa_camera_cut = false;
    camera.alpha += 0.045 * dt;
    camera.beta = std.math.pi / 2.75 + @sin(@as(f32, @floatFromInt(f)) * 0.008) * 0.12;
    scene.active_camera = .{ .arc_rotate = camera };

    scene.update(dt) catch |err| fail("update: {s}", .{@errorName(err)});
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
    // GPU sample poll must live HERE: render writes stats.gpu_frame_* at
    // its end and the next staged begin resets `stats`.
    if (timing.isEnabled()) bench.recordSample(scene.stats.gpu_frame_ms, scene.stats.gpu_frame_submit);
    if (f == 2 or bench.phase_started) {
        bench.noteTargetSize(scene.postfx.renderWidth(), scene.postfx.renderHeight());
        std.debug.print("q0-interior: render target {}x{} scale={d:.2} swapchain {}x{}\n", .{
            scene.postfx.renderWidth(),      scene.postfx.renderHeight(),
            scene.post_process.render_scale, sapp.width(),
            sapp.height(),
        });
    }
    if (f == 30) {
        if (scene.stats.draw_calls == 0) fail("no draw calls at frame 30", .{});
    }
}

export fn cleanup() callconv(.c) void {
    if (timing.isEnabled()) bench_mod.Bench.printPhase(bench.report(bench.index));
    scene.deinit();
    sg.shutdown();
    if (log_errors != 0) fail("sokol log errors: {}", .{log_errors});
    std.debug.print("q0-interior: verdict={s} failures={} frames={}\n", .{ if (failures == 0) "PASS" else "FAIL", failures, frames });
    _ = gpa.deinit();
    if (failures != 0) std.process.exit(1);
}

export fn event(ev: [*c]const sapp.Event) callconv(.c) void {
    bench.noteEvent(ev.*);
    if (ev.*.type == .KEY_DOWN and ev.*.key_code == .ESCAPE) sapp.quit();
}

pub fn main() void {
    frame_limit = envU32("AGATE_Q0_FRAMES", 0);
    render_scale = envF32("AGATE_Q0_RENDER_SCALE", 1.0);
    bench.phase_frames = @max(30, envU32("AGATE_Q0_PHASE_FRAMES", 180));
    if (envFlag("AGATE_Q0_PHASES_REVERSED")) bench.scales = .{ 0.66, 0.75, 1.0 };
    render_scale = @min(@max(render_scale, z.postprocess.min_render_scale), 1.0);
    sapp.run(.{
        .init_cb = init,
        .frame_cb = frame,
        .cleanup_cb = cleanup,
        .event_cb = event,
        .window_title = "Agate Q0 calibration interior (fill-bound)",
        // Small enough to fit any desktop work area: a window the WM has
        // to resize mid-run poisons phase medians (target-size mixing).
        .width = 720,
        .height = 540,
        .sample_count = 1,
        .logger = .{ .func = qLogger },
    });
}
