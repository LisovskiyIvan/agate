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
var ab: bench_mod.AbBench = .{};
/// Interleaved A/B mode (wave 5, opt-in via AGATE_Q0_AB=ssgi): alternates
/// SSGI on (arm A, scene default) / off (arm B) in blocks instead of the
/// back-to-back phase walk. SSGI is a composite-shader bool with no
/// target rebuild, so it is safe to toggle per block; a render_scale pair
/// would realloc targets every block and must stay in the phase bench.
var ab_mode: bool = false;
var frame_limit: u32 = 0;
var frames: u32 = 0;
var log_errors: u32 = 0;
var failures: u32 = 0;
var render_scale: f32 = 1.0;
var camera: z.ArcRotateCamera = undefined;
var still_camera: bool = false;

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
        .direction = z.Vec3.new(0.25, 0.55, 1.0),
        .diffuse = z.Color3.new(1.0, 0.92, 0.8),
        .intensity = 2.6,
    }) catch @panic("q0: sun failed");
    scene.shadows.enabled = true;

    // Spot PCSS calibration light: a small shadow-casting spot over the
    // content cluster. source_radius > 0 runs the spot PCSS path (blocker
    // search + variable penumbra); AGATE_Q0_NO_SPOT_PCSS=1 falls back to
    // the legacy fixed PCF bit-identically (0.0 disables).
    const spot_pcss_off = envFlag("AGATE_Q0_NO_SPOT_PCSS");
    _ = scene.createSpotLight("q0_key_spot", .{
        .position = z.Vec3.new(0.6, 3.6, 0.6),
        .direction = z.Vec3.new(-0.2, -1.0, -0.4),
        .color = z.Color3.new(1.0, 0.85, 0.7),
        .intensity = 6.0,
        .range = 12.0,
        .inner_angle_deg = 18.0,
        .outer_angle_deg = 32.0,
        .cast_shadows = true,
        .source_radius = if (spot_pcss_off) 0.0 else 0.02,
    }) catch @panic("q0: spot failed");

    // Point PCSS calibration light: a small shadow-casting point near the
    // content cluster. source_radius > 0 runs the border-correct point PCSS
    // path (world-space disk taps, per-tap face projection + cross-face
    // normalization, then blocker search + variable penumbra);
    // AGATE_Q0_NO_POINT_PCSS=1 falls back to the legacy
    // fixed PCF bit-identically (0.0 disables).
    const point_pcss_off = envFlag("AGATE_Q0_NO_POINT_PCSS");
    _ = scene.createPointLight("q0_key_point", .{
        .position = z.Vec3.new(-1.2, 2.6, 1.2),
        .color = z.Color3.new(1.0, 0.8, 0.6),
        .intensity = 8.0,
        .range = 10.0,
        .cast_shadows = true,
        .source_radius = if (point_pcss_off) 0.0 else 0.02,
    }) catch @panic("q0: point failed");

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
    ceiling.rotation.x = -90.0;
    ceiling.cast_shadows = true;
    const ceiling_mat = pbr("ceiling_pbr", .{ 0.9, 0.88, 0.85 }, 0.0, 0.9);
    ceiling_mat.double_sided = true;
    ceiling.setPBRMaterial(ceiling_mat);

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

    // Dust motes in the sun shaft (TAA reactive-marker live path, C.3 v2):
    // a slow alpha-blended CPU emitter whose billboards write NO depth and
    // carry NO velocity, so their pixels stamp the velocity-pass reactive
    // marker when taa_reactive is on. AGATE_Q0_NO_PARTICLES=1 isolates the
    // particle-free baseline bit-identically (no draws, no markers).
    if (!envFlag("AGATE_Q0_NO_PARTICLES")) {
        const dust = scene.createParticleSystem("q0_dust", 256) catch @panic("q0: dust failed");
        dust.emitter_position = z.Vec3.new(0.0, 1.8, 0.2);
        dust.emitter_box_min = z.Vec3.new(-1.4, -0.9, -1.2);
        dust.emitter_box_max = z.Vec3.new(1.4, 0.9, 1.2);
        dust.emit_rate = 24.0;
        dust.lifetime_min = 2.0;
        dust.lifetime_max = 3.5;
        dust.speed_min = 0.05;
        dust.speed_max = 0.22;
        dust.size_start = 0.06;
        dust.size_end = 0.02;
        dust.color_start = z.Color4.new(1.0, 0.9, 0.75, 0.35);
        dust.color_end = z.Color4.new(1.0, 0.85, 0.65, 0.0);
        dust.blend_mode = .alpha_blend;
        dust.start();
    }

    // Tinted glass pane (TAA reactive-marker glass half, C.3 v2): a blended
    // PBR quad standing in the open-front view axis, so the orbiting camera
    // sweeps it across the background every frame — its pixels carry the
    // background depth+velocity with no signal of their own, and only the
    // velocity-pass glass marker can shed their history. AGATE_Q0_NO_GLASS=1
    // isolates the glass-free baseline bit-identically (no draw, and the
    // marker loop over the empty transparent queue issues nothing).
    if (!envFlag("AGATE_Q0_NO_GLASS")) {
        const pane = z.MeshBuilder.createPlane(&scene, "q0_glass_pane", .{ .width = 1.4, .height = 2.0 }) catch @panic("q0: glass failed");
        pane.position = z.Vec3.new(-0.5, 1.5, 0.6);
        pane.rotation = z.Vec3.new(0.0, -18.0, 0.0);
        pane.cast_shadows = false;
        const glass_mat = pbr("q0_glass_pbr", .{ 0.65, 0.85, 0.95 }, 0.0, 0.08);
        glass_mat.alpha_mode = .blend;
        glass_mat.alpha = 0.35;
        glass_mat.double_sided = true;
        pane.setPBRMaterial(glass_mat);
    }

    // Fill-bound by design: the heaviest legal per-pixel chain.
    scene.post_process.enabled = true;
    scene.post_process.exposure = 1.1;
    scene.post_process.tonemapping = .agx;
    scene.post_process.bloom_enabled = true;
    scene.post_process.bloom_threshold = 1.0;
    scene.post_process.bloom_intensity = 0.55;
    scene.post_process.bloom_pyramid_mips = 7;
    scene.post_process.ssr_enabled = !envFlag("AGATE_Q0_NO_SSR");
    scene.post_process.ssr_intensity = 0.6;
    scene.post_process.ssr_steps = 32;
    scene.post_process.depth_pyramid_enabled = true;
    scene.post_process.dof_enabled = !envFlag("AGATE_Q0_NO_DOF");
    scene.post_process.dof_focus_distance = 4.6;
    scene.post_process.dof_focus_range = 2.2;
    scene.post_process.dof_max_blur = 5.0;
    // Lighting isolation: shafts off leaves the actual sunlit opening
    // footprint intact; DoF off + a still camera make shadow A/Bs readable.
    scene.post_process.shaft_enabled = !envFlag("AGATE_Q0_NO_SHAFTS");
    scene.post_process.shaft_intensity = 0.5;
    scene.post_process.shaft_steps = 32;
    scene.post_process.local_tonemapping_enabled = true;
    // SSGI v1 color bleed: the calibration scene carries it by default —
    // AGATE_Q0_NO_SSGI=1 isolates the pre-SSGI baseline.
    scene.post_process.ssgi_enabled = !envFlag("AGATE_Q0_NO_SSGI");
    scene.post_process.ssgi_intensity = 0.55;
    scene.post_process.ssgi_radius = 1.6;
    scene.post_process.ssgi_steps = 16;
    scene.post_process.taa_enabled = true;
    // TAA reactive treatment for velocity-less content (particles + glass
    // via the v2 velocity-alpha marker, v1 heuristic underneath for the
    // documented gaps): on in the calibration scene —
    // AGATE_Q0_NO_REACTIVE=1 isolates the pre-reactive baseline.
    scene.post_process.taa_reactive_enabled = !envFlag("AGATE_Q0_NO_REACTIVE");
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
        if (ab_mode) {
            if (ab.beginBlock(f, frame_limit)) |ev| {
                scene.post_process.ssgi_enabled = ev.arm == .A;
                if (ev.index > 0) scene.resetTaa();
                std.debug.print("q0-interior ab: block {} arm={s} ssgi={}\n", .{ ev.index, @tagName(ev.arm), scene.post_process.ssgi_enabled });
            }
        } else {
            if (bench.beginPhase(f, frame_limit)) |finished| bench_mod.Bench.printPhase(finished);
            if (bench.phase_started) {
                scene.post_process.render_scale = bench.currentScale();
                scene.resetTaa();
            }
        }
    }
    // Deterministic motion under the gate: same jitter/animation frame for
    // cross-run compare; slow drift keeps TAA/SSR/DoF genuinely working.
    const dt: f32 = if (gate) 1.0 / 60.0 else @floatCast(sapp.frameDuration());
    if (!still_camera) {
        camera.alpha += 0.045 * dt;
        camera.beta = std.math.pi / 2.75 + @sin(@as(f32, @floatFromInt(f)) * 0.008) * 0.12;
    }
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
    if (timing.isEnabled()) {
        if (ab_mode) {
            ab.recordSample(f, scene.stats.gpu_frame_ms, scene.stats.gpu_frame_submit);
        } else bench.recordSample(scene.stats.gpu_frame_ms, scene.stats.gpu_frame_submit);
    }
    if (ab_mode) {
        // Every-frame note (cheap), outside the timing gate: a mid-block
        // window settle marks the run UNSTABLE even if it settles back
        // before the next boundary.
        ab.noteTargetSize(scene.postfx.renderWidth(), scene.postfx.renderHeight());
    }
    if (f == 2 or bench.phase_started or ab.block_started) {
        if (ab_mode) {
            ab.noteTargetSize(scene.postfx.renderWidth(), scene.postfx.renderHeight());
        } else bench.noteTargetSize(scene.postfx.renderWidth(), scene.postfx.renderHeight());
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
    if (timing.isEnabled()) {
        if (ab_mode) {
            const s = ab.finalSummary("ssgi_on", "ssgi_off");
            ab.printBlocks();
            bench_mod.AbBench.printSummary(s);
        } else bench_mod.Bench.printPhase(bench.report(bench.index));
    }
    scene.deinit();
    sg.shutdown();
    if (log_errors != 0) fail("sokol log errors: {}", .{log_errors});
    std.debug.print("q0-interior: verdict={s} failures={} frames={}\n", .{ if (failures == 0) "PASS" else "FAIL", failures, frames });
    _ = gpa.deinit();
    if (failures != 0) std.process.exit(1);
}

export fn event(ev: [*c]const sapp.Event) callconv(.c) void {
    bench.noteEvent(ev.*);
    ab.noteEvent(ev.*);
    if (ev.*.type == .KEY_DOWN and ev.*.key_code == .ESCAPE) sapp.quit();
}

pub fn main() void {
    frame_limit = envU32("AGATE_Q0_FRAMES", 0);
    still_camera = envFlag("AGATE_Q0_STILL");
    render_scale = envF32("AGATE_Q0_RENDER_SCALE", 1.0);
    bench.phase_frames = @max(30, envU32("AGATE_Q0_PHASE_FRAMES", 180));
    if (envFlag("AGATE_Q0_PHASES_REVERSED")) bench.scales = .{ 0.66, 0.75, 1.0 };
    // Opt-in interleaved A/B: AGATE_Q0_AB=ssgi alternates SSGI on/off in
    // A/B blocks (default 2 settle blocks + 8 measured blocks x 60 frames
    // = 600 frames, one full rotated A B B A B A A B cycle, when
    // AGATE_Q0_FRAMES is unset). AGATE_Q0_AB_BLOCK overrides the block
    // size (clamped to the bench scratch). The phase bench is bypassed in
    // this mode; everything else keeps its meaning.
    if (std.c.getenv("AGATE_Q0_AB")) |sel| {
        if (std.mem.eql(u8, std.mem.span(sel), "ssgi")) ab_mode = true;
    }
    if (ab_mode) {
        ab.block_frames = @min(@max(envU32("AGATE_Q0_AB_BLOCK", 60), 16), bench_mod.ab_max_block_frames);
        if (frame_limit == 0) frame_limit = ab.block_frames * (8 + ab.settle_blocks);
    }
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
