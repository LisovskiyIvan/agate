//! Fresh modern HDR showcase + finite live gate (native Metal / browser WebGPU).
//!
//! Dark studio interior: PBR metallic/roughness ramp orbs, a polished chrome
//! pillar and torus, HDR emissive rails/cards at radiance 1/4/16, dark floor,
//! soft warm sun with shadows + teal sky ambient. Bloom lifts the rails
//! without clipping the main subject. Always-linear HDR RGBA16F engine path:
//! `scene.post_process.enabled` is only the effects master (output stays HDR
//! either way); exposure is manual, tonemapping is the existing ACES curve.
//!
//! Modes: interactive by default (never quits). Finite gate with
//! `AGATE_HDR_FRAMES=240` (native) or `?frames=240` (web): 240 frames with
//! live checks (RGBA16F targets, TAA history format, exposure/master/resize/
//! camera-cut phases, draws every frame even with the master off).
//! `AGATE_HDR_MSAA` / `?msaa=4` (default 1), `AGATE_HDR_SRGB=1` / `?srgb=1`
//! single-encode proof via `sapp_desc.srgb`.
//!
//! Keys: B bloom toggle, E exposure cycle, SPACE effects master, ESC quit.
//! Staged frame protocol, serial on the context thread (no worker):
//! update -> publishFrameSnapshot -> buildPreparedFrame ->
//! beginStagedPrepare -> finishStagedPrepare -> render.
const std = @import("std");
const builtin = @import("builtin");
const z = @import("agate");
const sokol = @import("sokol");
const sg = sokol.gfx;
const sapp = sokol.app;
const timing = z.gpu_timing;

const exposures = [_]f32{ 1.0, 0.6, 1.6, 2.5 };

var frame_limit: u32 = 0;
var msaa_count: i32 = 1;
var want_srgb: bool = false;
var frames: u32 = 0;
var checks: u32 = 0;
var failures: u32 = 0;
var log_errors: u32 = 0;
var skips: u32 = 0;
var exposure_idx: usize = 0;
var scene: z.Scene = undefined;
var hero: *z.Mesh = undefined;
var orbs: [5]*z.Mesh = undefined;
var sokol_live_allocations: usize = 0;

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
    std.debug.print("hdr-showcase: FAIL " ++ fmt ++ "\n", args);
}

fn check(name: []const u8, ok: bool) void {
    checks += 1;
    if (!ok) fail("check {s} failed at frame {}", .{ name, frames });
}

fn logger(tag: [*c]const u8, level: u32, item: u32, msg: [*c]const u8, line: u32, file: [*c]const u8, data: ?*anyopaque) callconv(.c) void {
    if (level <= 1) log_errors += 1;
    sokol.log.func(tag, level, item, msg, line, file, data);
}

fn pbrMat(name: []const u8, albedo: z.Color3, metallic: f32, roughness: f32) *z.PBRMaterial {
    const m = scene.createPBRMaterial(name) catch @panic("hdr-showcase: pbr material failed");
    m.albedo_color = albedo;
    m.metallic = metallic;
    m.roughness = roughness;
    return m;
}

fn emissiveMat(name: []const u8, radiance: z.Color3) *z.PBRMaterial {
    const m = scene.createPBRMaterial(name) catch @panic("hdr-showcase: emissive material failed");
    m.unlit = true;
    m.albedo_color = z.Color3.black;
    m.emissive_color = radiance;
    return m;
}

fn setupScene() void {
    sg.setup(.{
        .environment = sokol.glue.environment(),
        .logger = .{ .func = logger },
        .pipeline_pool_size = 256,
        .shader_pool_size = 128,
        .allocator = .{ .alloc_fn = sokolAlloc, .free_fn = sokolFree },
    });
    scene = z.Scene.init(std.heap.c_allocator);
    scene.msaa_sample_count = msaa_count;
    scene.clear_color = z.Color4.new(0.012, 0.02, 0.028, 1.0);
    scene.active_camera = .{ .arc_rotate = z.ArcRotateCamera.init("hdr_cam", .{
        .alpha = 0.85,
        .beta = 1.12,
        .radius = 10.5,
        .target = z.Vec3.new(0, 1.4, 0),
    }) };

    // Soft warm sun (shadow caster) + teal sky ambient.
    _ = scene.createDirectionalLight("sun", .{
        .direction = z.Vec3.new(0.45, 1.0, 0.35),
        .diffuse = z.Color3.new(1.0, 0.72, 0.5),
        .intensity = 2.4,
    }) catch @panic("hdr-showcase: sun failed");
    _ = scene.createHemisphericLight("sky", .{
        .direction = z.Vec3.new(0.2, 1.0, 0.1),
        .diffuse = z.Color3.new(0.4, 0.62, 0.68),
        .ground_color = z.Color3.new(0.10, 0.08, 0.06),
        .intensity = 0.55,
    });
    scene.shadows.enabled = true;

    // Dark polished floor + back wall (studio interior shell).
    const floor = z.MeshBuilder.createGround(&scene, "floor", .{ .width = 44.0, .height = 44.0, .subdivisions = 1 }) catch @panic("hdr-showcase: floor failed");
    floor.setPBRMaterial(pbrMat("floor_mat", z.Color3.new(0.025, 0.03, 0.035), 0.55, 0.28));
    const wall = z.MeshBuilder.createPlane(&scene, "wall", .{ .width = 44.0, .height = 14.0 }) catch @panic("hdr-showcase: wall failed");
    wall.position = z.Vec3.new(0, 7.0, -8.0);
    wall.setPBRMaterial(pbrMat("wall_mat", z.Color3.new(0.02, 0.05, 0.06), 0.0, 0.9));

    // Metallic ramp orbs (roughness fixed): 0 .. 1 across the row.
    const orb_names = [_][]const u8{ "orb0", "orb1", "orb2", "orb3", "orb4" };
    for (0..5) |i| {
        const orb = z.MeshBuilder.createSphere(&scene, orb_names[i], .{ .diameter = 1.1, .segments = 32 }) catch @panic("hdr-showcase: orb failed");
        const f = @as(f32, @floatFromInt(i)) / 4.0;
        orb.position = z.Vec3.new(-4.4 + 2.2 * @as(f32, @floatFromInt(i)), 0.56, -1.5);
        orb.setPBRMaterial(pbrMat(orb_names[i], z.Color3.new(0.75, 0.72, 0.70), f, 0.32));
        orbs[i] = orb;
    }

    // Polished chrome pillar + hero torus (amber tint, near-mirror).
    const pillar = z.MeshBuilder.createCylinder(&scene, "pillar", .{ .height = 2.6, .diameter = 0.9, .tessellation = 32 }) catch @panic("hdr-showcase: pillar failed");
    pillar.position = z.Vec3.new(5.4, 1.3, 1.8);
    pillar.setPBRMaterial(pbrMat("chrome_mat", z.Color3.new(0.9, 0.92, 0.95), 1.0, 0.08));
    hero = z.MeshBuilder.createTorus(&scene, "hero", .{ .diameter = 1.8, .thickness = 0.5, .tessellation = 48 }) catch @panic("hdr-showcase: hero failed");
    hero.position = z.Vec3.new(-5.2, 1.5, 1.6);
    hero.setPBRMaterial(pbrMat("hero_mat", z.Color3.new(0.9, 0.75, 0.55), 1.0, 0.12));

    // HDR emissive rails/cards, radiance 1 / 4 / 16 (amber/cyan/magenta).
    const rail_lo = z.MeshBuilder.createBox(&scene, "rail_lo", .{ .width = 10.0, .height = 0.1, .depth = 0.1 }) catch @panic("hdr-showcase: rail failed");
    rail_lo.position = z.Vec3.new(0, 0.1, 3.2);
    rail_lo.setPBRMaterial(emissiveMat("cyan_mat", z.Color3.new(0.3, 3.2, 4.0)));
    const rail_hi = z.MeshBuilder.createBox(&scene, "rail_hi", .{ .width = 10.0, .height = 0.1, .depth = 0.1 }) catch @panic("hdr-showcase: rail failed");
    rail_hi.position = z.Vec3.new(0, 0.1, -5.2);
    rail_hi.setPBRMaterial(emissiveMat("amber_mat", z.Color3.new(16.0, 8.8, 2.9)));
    const card = z.MeshBuilder.createPlane(&scene, "card", .{ .width = 2.4, .height = 1.5 }) catch @panic("hdr-showcase: card failed");
    card.position = z.Vec3.new(-3.5, 3.2, -7.9);
    card.setPBRMaterial(emissiveMat("magenta_mat", z.Color3.new(1.0, 0.15, 0.9)));

    // Manual exposure, existing ACES; bloom catches only HDR rails.
    scene.post_process.enabled = true;
    scene.post_process.exposure = exposures[0];
    scene.post_process.tonemapping = .aces;
    scene.post_process.bloom_enabled = true;
    scene.post_process.bloom_threshold = 1.0;
    scene.post_process.bloom_intensity = 0.7;
    scene.post_process.bloom_pyramid_mips = 5;
    scene.post_process.taa_enabled = true;
    scene.post_process.taa_camera_cut = true;
    scene.post_process.fog_enabled = false;
    std.debug.print("hdr-showcase: setup backend={s} msaa={} srgb={} env_color_fmt={s} swapchain_fmt={s} hdr_caps_sample={} hdr_caps_filter={} hdr_caps_render={} hdr_caps_blend={} hdr_caps_msaa={}\n", .{
        @tagName(sg.queryBackend()),
        msaa_count,
        want_srgb,
        @tagName(sg.queryDesc().environment.defaults.color_format),
        @tagName(sapp.colorFormat()),
        z.postprocess.hdr.queryCapabilities().sample,
        z.postprocess.hdr.queryCapabilities().filter,
        z.postprocess.hdr.queryCapabilities().render,
        z.postprocess.hdr.queryCapabilities().blend,
        z.postprocess.hdr.queryCapabilities().msaa,
    });
}

export fn init() callconv(.c) void {
    z.gpu_thread.markContextThread();
    sokol.time.setup();
    timing.setEnabled(false);
    setupScene();
}

export fn frame() callconv(.c) void {
    frames += 1;
    const f = frames;
    const gate = frame_limit > 0;
    defer if (gate and f >= frame_limit) sapp.quit();
    // Finite gate uses a fixed step so UNORM/sRGB captures compare exactly
    // (same jitter/animation frame); interactive mode keeps live timing.
    const dt: f32 = if (gate) 1.0 / 60.0 else @floatCast(sapp.frameDuration());

    if (gate) {
        if (f == 2) scene.post_process.taa_camera_cut = false;
        if (f == 61) {
            exposure_idx = 2;
            scene.post_process.exposure = exposures[exposure_idx];
        }
        if (f == 91) scene.post_process.enabled = false;
        if (f == 111) scene.post_process.enabled = true;
        if (f == 160) scene.post_process.taa_camera_cut = true;
        if (f == 161) scene.post_process.taa_camera_cut = false;
        if (f == 130) {
            scene.resizeOffscreen(320, 180);
            const pp = &scene.postfx.postprocess_pass;
            check("resize-shape", pp.width == 320 and pp.height == 180);
            check("resize-targets", pp.targetsValid());
            check("resize-format", pp.color_format == .RGBA16F);
        }
    }

    // Gentle life: hero spin + slow camera drift (fixed attractive framing).
    hero.rotation.y += 0.35 * dt;
    if (scene.active_camera != null and scene.active_camera.? == .arc_rotate) {
        scene.active_camera.?.arc_rotate.alpha += 0.04 * dt;
    }

    scene.update(dt) catch |err| fail("update: {s}", .{@errorName(err)});
    const w: i32 = @max(2, sapp.width());
    const h: i32 = @max(2, sapp.height());
    const aspect: f32 = @as(f32, @floatFromInt(w)) / @as(f32, @floatFromInt(h));
    scene.publishFrameSnapshot(aspect, w, h);
    if (!scene.buildPreparedFrame()) {
        skips += 1;
        fail("producer claim unavailable at frame {}", .{f});
        return;
    }
    const claim = scene.beginStagedPrepare();
    if (claim) |cl| {
        scene.finishStagedPrepare(cl);
    } else {
        skips += 1;
        fail("staged claim unavailable after successful build at frame {}", .{f});
        return;
    }
    scene.render();

    const pp = &scene.postfx.postprocess_pass;
    check("hdr-target", pp.color_format == .RGBA16F);
    check("targets-valid", pp.targetsValid());
    check("draws", scene.stats.draw_calls > 0);
    if (scene.post_process.enabled and scene.post_process.taa_enabled and scene.postfx.main_samples == 1) {
        check("taa-history-available", pp.taaAvailable());
        check("taa-history-hdr", pp.taa_format == .RGBA16F);
    }
    if (gate) {
        if (f == 1) check("initial-cut", scene.post_process.taa_camera_cut);
        if (f == 61) check("exposure-61", scene.post_process.exposure == exposures[2]);
        if (f == 91) check("master-off", !scene.post_process.enabled);
        if (f == 111) check("master-on", scene.post_process.enabled);
        if (f == 130) check("resize-reverted", pp.width == w and pp.height == h);
        if (f == 160) check("cut-160", scene.post_process.taa_camera_cut);
        if (f == 161) check("cut-cleared", !scene.post_process.taa_camera_cut);
    }
    if (f % 60 == 0) {
        std.debug.print("hdr-showcase: frame {} draws={} exposure={d:.2} master={} bloom={} samples={} srgb={} size={}x{}\n", .{
            f, scene.stats.draw_calls, scene.post_process.exposure, scene.post_process.enabled, scene.post_process.bloom_enabled, scene.postfx.main_samples, want_srgb, w, h,
        });
    }
}

export fn cleanup() callconv(.c) void {
    scene.deinit();
    sg.shutdown();
    if (sokol_live_allocations != 0) fail("allocator still owns {} allocations after shutdown", .{sokol_live_allocations});
    if (log_errors != 0) fail("sokol log errors {}", .{log_errors});
    if (checks == 0 or skips != 0) fail("staged frame coverage: checks={} skips={}", .{ checks, skips });
    const ok = failures == 0;
    std.debug.print("hdr-showcase: frames={} checks={} failures={} log_errors={} sokol_live_allocations={} verdict={s}\n", .{
        frames, checks, failures, log_errors, sokol_live_allocations, if (ok) "PASS" else "FAIL",
    });
    if (!ok and !builtin.cpu.arch.isWasm()) std.process.exit(1);
}

export fn event(ev: [*c]const sapp.Event) callconv(.c) void {
    switch (ev.*.type) {
        .KEY_DOWN => switch (ev.*.key_code) {
            .ESCAPE => sapp.quit(),
            .B => scene.post_process.bloom_enabled = !scene.post_process.bloom_enabled,
            .SPACE => scene.post_process.enabled = !scene.post_process.enabled,
            .E => {
                exposure_idx = (exposure_idx + 1) % exposures.len;
                scene.post_process.exposure = exposures[exposure_idx];
            },
            else => {},
        },
        else => {},
    }
}

extern fn emscripten_run_script_int(script: [*:0]const u8) c_int;

fn envU32(name: [*:0]const u8, fallback: u32) u32 {
    const value = std.c.getenv(name) orelse return fallback;
    return std.fmt.parseInt(u32, std.mem.span(value), 10) catch fallback;
}

fn envI32(name: [*:0]const u8, fallback: i32) i32 {
    const value = std.c.getenv(name) orelse return fallback;
    return std.fmt.parseInt(i32, std.mem.span(value), 10) catch fallback;
}

fn envFlag(name: [*:0]const u8) bool {
    const value = std.c.getenv(name) orelse return false;
    const s = std.mem.span(value);
    return std.mem.eql(u8, s, "1") or std.mem.eql(u8, s, "true");
}

pub fn startApp() void {
    if (builtin.cpu.arch.isWasm()) {
        frame_limit = @intCast(@max(0, emscripten_run_script_int("(()=>{const n=parseInt(new URLSearchParams(location.search).get('frames')||'0',10);return isNaN(n)?0:n;})()")));
        msaa_count = emscripten_run_script_int("(()=>{const n=parseInt(new URLSearchParams(location.search).get('msaa')||'1',10);return isNaN(n)?1:n;})()");
        want_srgb = emscripten_run_script_int("new URLSearchParams(location.search).has('srgb')") != 0;
    } else {
        frame_limit = envU32("AGATE_HDR_FRAMES", 0);
        msaa_count = envI32("AGATE_HDR_MSAA", 1);
        want_srgb = envFlag("AGATE_HDR_SRGB");
    }
    msaa_count = std.math.clamp(msaa_count, 1, 4);
    sapp.run(.{
        .init_cb = init,
        .frame_cb = frame,
        .cleanup_cb = cleanup,
        .event_cb = event,
        .window_title = "Agate HDR showcase",
        .width = 960,
        .height = 600,
        .sample_count = 1,
        .gl = .{ .major_version = 4, .minor_version = 3 },
        .srgb = want_srgb,
        .logger = .{ .func = logger },
    });
}

pub fn main() void {
    startApp();
}
