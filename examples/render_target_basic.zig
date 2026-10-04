//! RTT capture/sample integration smoke (runnable sokol-app program).
//!
//! Proves the v1 render-target contract on a real GPU, end to end:
//!   1. color+depth target creation (1x capture + 4x MSAA resolve twin);
//!   2. per-frame scene capture (`RenderTarget.renderPrimaryView`) from a
//!      dedicated capture scene with plain materials only;
//!   3. sampling the capture in a LATER pass — a second display scene draws
//!      an unlit textured quad bound to the borrowed capture texture;
//!   4. opt-in `--refraction`: a real glass sphere (PBR transmission,
//!      engine-owned refraction target + shaders) in front of the borrowed
//!      background panel; `--ior` selects the index, `--freeze` stops all
//!      motion for deterministic cross-run compare;
//!   5. mid-run resize (old content survives failures; borrow re-pointed);
//!   6. creation-failure and failed-resize regression checks with teeth;
//!   7. finite auto-quit (`--frames`, default 120) and a validation-log
//!      verdict (sokol errors counted through a wrapping logger).
//!
//! Feedback safety by construction: the captured scene never references the
//! target's views (default materials only), and the display scene — the only
//! place the borrowed texture is bound — is never captured. Within-pass
//! sampling additionally fails closed inside the target (`pass_open`).
//!
//! Frame flow is single-threaded serial (context thread only):
//! update both scenes -> optional resize -> refresh display borrow (BEFORE
//! display prepare so the staged draw records pick it up) -> prepare both
//! -> capture (fresh content) -> MSAA resolve exercise
//! -> render display scene -> refraction/reuse assertions.
//! The capture scene is prepare-only (probe-style offscreen capture; a
//! repeated prepare discards the pending frame latest-wins, and epochs
//! self-close on the next begin, so no present is required).
//!
//! The smoke has no portable pixel readback: every capture must issue real
//! draws (per-frame draw/triangle deltas > 0 or the smoke fails), the
//! display present must draw the quad/strips/sphere, and the engine
//! refraction target must be valid after a refractive render. Rotating
//! geometry exercises fresh captures; `--freeze` enables cross-run pixel
//! comparisons through an OS window capture (see docs/render-target.md).
//!
//! Build wiring (owner: main thread):
//! ```zig
//! const rtt_smoke = b.addExecutable(.{
//!     .name = "rtt-smoke",
//!     .root_module = b.createModule(.{
//!         .root_source_file = b.path("examples/render_target_basic.zig"),
//!         .target = target,
//!         .optimize = optimize,
//!         .imports = &.{
//!             .{ .name = "sokol", .module = mod_sokol },
//!             .{ .name = "agate", .module = mod_agate },
//!         },
//!     }),
//! });
//! b.installArtifact(rtt_smoke);
//! const run_rtt = b.addRunArtifact(rtt_smoke);
//! if (b.args) |args| run_rtt.addArgs(args);
//! b.step("example-rtt", "Run the RTT capture/sample smoke (needs GPU/display)")
//!     .dependOn(&run_rtt.step);
//! ```
//! Run: `zig build example-rtt -- --frames 120 --refraction --ior 1.5`
//! Expected: `verdict=PASS` with `log_errors=0`.
//! Compare: `--freeze --refraction --ior 1.0` vs `--freeze --refraction
//! --ior 1.5` (deterministic scenes; capture this program's window only).

const std = @import("std");
const sokol = @import("sokol");
const sapp = sokol.app;
const sg = sokol.gfx;
const sglue = sokol.glue;
const slog = sokol.log;
const z = @import("agate");

var gpa = std.heap.DebugAllocator(.{ .thread_safe = true }){};

// --- smoke CLI (defaults keep the run finite and deterministic) ---
var frame_limit: u32 = 120;
var frame_count: u32 = 0;
var rtt_size: u32 = 256;
var rtt_msaa: i32 = 1;
var want_refraction: bool = false;
var glass_ior: f32 = 1.5;
var glass_thickness: f32 = 0.7;
var freeze_motion: bool = false;

// --- sokol validation-log verdict (0=panic, 1=error, 2=warning) ---
var log_errors: u64 = 0;
var log_warnings: u64 = 0;
fn rttLogger(tag: [*c]const u8, level: u32, item: u32, msg: [*c]const u8, line: u32, file: [*c]const u8, user_data: ?*anyopaque) callconv(.c) void {
    if (level <= 1) {
        log_errors += 1;
    } else if (level == 2) {
        log_warnings += 1;
    }
    slog.func(tag, level, item, msg, line, file, user_data);
}

// --- smoke state (all context-thread owned) ---
var capture_scene: z.Scene = undefined;
var display_scene: z.Scene = undefined;
var capture_box: *z.Mesh = undefined;
var display_mat: *z.StandardMaterial = undefined;
var checker_tex: z.Texture = undefined;
var checker_owned: bool = false;
var rtt: z.RenderTarget = .{};
var rtt_msaa4: z.RenderTarget = .{};
var smoke_failures: u64 = 0;
var captures_ok: u64 = 0;
var resize_done: bool = false;
var reuse_probed: bool = false;
var window_resize_pending: bool = false;

fn smokeFail(comptime fmt: []const u8, args: anytype) void {
    smoke_failures += 1;
    std.debug.print("rtt-smoke FAIL: " ++ fmt ++ "\n", args);
}

/// Staged prepare helper (test-only): requires a fresh FULL producer build,
/// then claims + finishes it. Preserves an already-prepared pending frame
/// (no rebuild). Returns false + records a failure when no fresh build can
/// be claimed/finished.
fn prepare(scn: *z.Scene) bool {
    if (!scn.buildPreparedFrame()) {
        smokeFail("staged build saturated at frame {}", .{frame_count});
        return false;
    }
    const claim = scn.beginStagedPrepare() orelse {
        smokeFail("fresh build not claimable at frame {}", .{frame_count});
        return false;
    };
    scn.finishStagedPrepare(claim);
    return true;
}

fn parseArgs(allocator: std.mem.Allocator, args: std.process.Args) void {
    var it = std.process.Args.Iterator.initAllocator(args, allocator) catch return;
    defer it.deinit();
    _ = it.next();
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--frames")) {
            if (it.next()) |n| frame_limit = std.fmt.parseInt(u32, n, 10) catch 120;
        } else if (std.mem.eql(u8, arg, "--rtt-size")) {
            if (it.next()) |n| rtt_size = std.fmt.parseInt(u32, n, 10) catch 256;
        } else if (std.mem.eql(u8, arg, "--rtt-msaa")) {
            if (it.next()) |n| rtt_msaa = std.fmt.parseInt(i32, n, 10) catch 1;
        } else if (std.mem.eql(u8, arg, "--refraction")) {
            want_refraction = true;
        } else if (std.mem.eql(u8, arg, "--ior")) {
            if (it.next()) |n| glass_ior = std.fmt.parseFloat(f32, n) catch 1.5;
        } else if (std.mem.eql(u8, arg, "--thickness")) {
            if (it.next()) |n| glass_thickness = std.fmt.parseFloat(f32, n) catch 0.7;
        } else if (std.mem.eql(u8, arg, "--freeze")) {
            freeze_motion = true;
        }
    }
    if (frame_limit == 0) frame_limit = 120;
    rtt_size = @max(32, rtt_size);
    glass_ior = std.math.clamp(glass_ior, 1.0, 3.0);
}

/// One tiny quad strip with UV0 != UV1 (mirrored U), uploaded straight from
/// stack arrays (`uploadGeometry` retains its own CPU mirrors). The two
/// strips share one checker texture but select different coords
/// (`tex_coord` 0 vs 1), so identical geometry shades visibly different —
/// the UV1 path proof. No core changes: plain mesh + material fields.
fn makeUvStrip(scene: *z.Scene, name: []const u8, cx: f32, cy: f32) *z.Mesh {
    var verts = [_]z.Vertex{
        .{ .position = .{ cx - 0.6, cy - 0.25, 0 }, .normal = .{ 0, 0, 1 }, .color = .{ 1, 1, 1, 1 }, .uv = .{ 0, 0 }, .uv1 = .{ 1, 0 } },
        .{ .position = .{ cx + 0.6, cy - 0.25, 0 }, .normal = .{ 0, 0, 1 }, .color = .{ 1, 1, 1, 1 }, .uv = .{ 1, 0 }, .uv1 = .{ 0, 0 } },
        .{ .position = .{ cx + 0.6, cy + 0.25, 0 }, .normal = .{ 0, 0, 1 }, .color = .{ 1, 1, 1, 1 }, .uv = .{ 1, 1 }, .uv1 = .{ 0, 1 } },
        .{ .position = .{ cx - 0.6, cy + 0.25, 0 }, .normal = .{ 0, 0, 1 }, .color = .{ 1, 1, 1, 1 }, .uv = .{ 0, 1 }, .uv1 = .{ 1, 1 } },
    };
    var idx = [_]u32{ 0, 1, 2, 0, 2, 3 };
    return z.uploadGeometry(scene, name, .{
        .vertices = &verts,
        .indices = &idx,
        .bounds = z.BoundingBox.init(
            z.Vec3.new(cx - 0.6, cy - 0.25, 0),
            z.Vec3.new(cx + 0.6, cy + 0.25, 0),
        ),
    }) catch |err| {
        std.debug.panic("rtt-smoke: uv strip failed: {}", .{err});
    };
}

export fn init() callconv(.c) void {
    z.gpu_thread.markContextThread();
    sg.setup(.{
        .environment = sglue.environment(),
        .logger = .{ .func = rttLogger },
        .pipeline_pool_size = 256,
        .shader_pool_size = 128,
    });
    sokol.time.setup();
    const allocator = gpa.allocator();

    // Regression checks with teeth: creation failures must fail loudly
    // with the documented error, never crash, never half-allocate.
    if (z.RenderTarget.create(.{ .width = 0, .height = 64 })) |_| {
        smokeFail("zero-width creation unexpectedly succeeded", .{});
    } else |err| {
        if (err != error.InvalidDimensions) smokeFail("zero-width creation gave {s}, want InvalidDimensions", .{@errorName(err)});
    }

    // --- capture scene: camera + light + box, DEFAULT materials only ---
    // (no RTT view is ever bound here: feedback impossible by construction).
    capture_scene = z.Scene.init(allocator);
    capture_scene.active_camera = .{ .arc_rotate = z.ArcRotateCamera.init("cap_cam", .{
        .alpha = std.math.pi / 4.0,
        .beta = std.math.pi / 3.0,
        .radius = 5.5,
        .target = z.Vec3.zero,
    }) };
    capture_scene.clear_color = z.Color4.new(0.10, 0.16, 0.30, 1.0);
    _ = capture_scene.createHemisphericLight("cap_hemi", .{
        .direction = z.Vec3.new(0.6, 1.0, 0.4),
        .diffuse = z.Color3.white,
        .ground_color = z.Color3.new(0.2, 0.22, 0.28),
        .intensity = 1.0,
    });
    capture_box = z.MeshBuilder.createBox(&capture_scene, "cap_box", .{ .size = 2.0 }) catch |err| {
        std.debug.panic("rtt-smoke: capture box failed: {}", .{err});
    };

    // --- display scene: borrowed-RTT background panel (never captured) ---
    display_scene = z.Scene.init(allocator);
    display_scene.active_camera = .{ .arc_rotate = z.ArcRotateCamera.init("show_cam", .{
        .alpha = std.math.pi / 4.0,
        .beta = std.math.pi / 3.0,
        .radius = 6.0,
        .target = z.Vec3.zero,
    }) };
    display_scene.clear_color = z.Color4.new(0.08, 0.08, 0.10, 1.0);
    _ = display_scene.createHemisphericLight("show_hemi", .{
        .direction = z.Vec3.new(0.6, 1.0, 0.4),
        .diffuse = z.Color3.white,
        .ground_color = z.Color3.new(0.2, 0.22, 0.28),
        .intensity = 1.0,
    });
    const quad = z.MeshBuilder.createPlane(&display_scene, "screen", .{ .width = 4.0, .height = 3.0 }) catch |err| {
        std.debug.panic("rtt-smoke: display quad failed: {}", .{err});
    };
    quad.position.z = -1.5;
    display_mat = display_scene.createStandardMaterial("screen_mat") catch |err| {
        std.debug.panic("rtt-smoke: display material failed: {}", .{err});
    };
    display_mat.unlit = true;
    display_mat.double_sided = true;
    quad.setStandardMaterial(display_mat);

    // --- UV1 proof strips: shared checker, tex_coord 0 vs 1 ---
    checker_tex = z.Texture.createCheckerboard(allocator, 128, 128, 16, .{ 30, 30, 30, 255 }, .{ 225, 225, 225, 255 }) catch |err| {
        std.debug.panic("rtt-smoke: checker failed: {}", .{err});
    };
    checker_owned = true;
    const strip_a = makeUvStrip(&display_scene, "uv0_strip", -1.1, -2.1);
    const mat_a = display_scene.createStandardMaterial("uv0_mat") catch |err| {
        std.debug.panic("rtt-smoke: uv0 material failed: {}", .{err});
    };
    mat_a.double_sided = true;
    mat_a.diffuse_texture = checker_tex;
    mat_a.diffuse_uv_transform = .{ .tex_coord = 0 };
    strip_a.setStandardMaterial(mat_a);
    const strip_b = makeUvStrip(&display_scene, "uv1_strip", 1.1, -2.1);
    const mat_b = display_scene.createPBRMaterial("uv1_mat") catch |err| {
        std.debug.panic("rtt-smoke: uv1 material failed: {}", .{err});
    };
    mat_b.double_sided = true;
    mat_b.albedo_texture = checker_tex;
    mat_b.albedo_uv_transform = .{ .tex_coord = 1 };
    mat_b.roughness = 0.9;
    strip_b.setPBRMaterial(mat_b);

    // --- opt-in glass: real PBR transmission sphere (engine capture) ---
    if (want_refraction) {
        const glass = z.MeshBuilder.createSphere(&display_scene, "glass", .{ .diameter = 1.5, .segments = 32 }) catch |err| {
            std.debug.panic("rtt-smoke: glass sphere failed: {}", .{err});
        };
        const glass_mat = display_scene.createPBRMaterial("glass_mat") catch |err| {
            std.debug.panic("rtt-smoke: glass material failed: {}", .{err});
        };
        glass_mat.alpha = 1.0;
        glass_mat.roughness = 0.05;
        glass_mat.transmission = .{
            .factor = 1.0,
            .refract = true,
            .ior = glass_ior,
            .thickness = glass_thickness,
        };
        glass.setPBRMaterial(glass_mat);
        std.debug.print("rtt-smoke: glass on (ior={d:.2})\n", .{glass_ior});
    }

    // --- targets: 1x capture + 4x MSAA resolve twin (small, cheap) ---
    rtt = z.RenderTarget.create(.{
        .width = rtt_size / 2,
        .height = rtt_size / 2,
        .sample_count = rtt_msaa,
    }) catch |err| {
        std.debug.panic("rtt-smoke: capture target failed: {s}", .{@errorName(err)});
    };
    // Known first content: an invalid-but-live borrow must never show
    // uninitialized memory on frame 0.
    if (!rtt.clear(z.Color4.new(0.10, 0.16, 0.30, 1.0), 1.0)) {
        smokeFail("initial capture-target clear failed", .{});
    }
    std.debug.print("rtt-smoke: capture target {}x{} samples={} est={} bytes\n", .{
        rtt.width, rtt.height, rtt.sample_count, rtt.estimatedBytes(),
    });
    rtt_msaa4 = z.RenderTarget.create(.{
        .width = 256,
        .height = 256,
        .sample_count = 4,
        .min_filter = .NEAREST,
        .mag_filter = .NEAREST,
    }) catch |err| {
        std.debug.panic("rtt-smoke: msaa target failed: {s}", .{@errorName(err)});
    };
    std.debug.print("rtt-smoke: msaa target {}x{} samples={} (degraded to 1x where unsupported)\n", .{
        rtt_msaa4.width, rtt_msaa4.height, rtt_msaa4.sample_count,
    });

    // Failed resize retains the old target verbatim (regression pin, live).
    const keep_w = rtt.width;
    const keep_img = rtt.color_image.id;
    if (rtt.resize(0, 64)) {
        smokeFail("zero-width resize unexpectedly succeeded", .{});
    } else if (rtt.width != keep_w or rtt.color_image.id != keep_img or !rtt.isValid()) {
        smokeFail("failed resize did not retain the old target", .{});
    }

    // A saved borrow (even an alternate view of the same image) must be
    // rejected before beginning a capture pass. No GPU validation error is
    // acceptable here: rejection belongs to our preflight, not the driver.
    var borrow = rtt.asTexture();
    borrow.deinit();
    if (sg.queryImageState(rtt.color_image) != .VALID or sg.queryViewState(rtt.sampleView()) != .VALID) {
        smokeFail("borrow deinit destroyed target handles", .{});
    }
    const alias = sg.makeView(.{ .texture = .{ .image = borrow.image } });
    defer sg.destroyView(alias);
    if (sg.queryViewState(alias) != .VALID) smokeFail("alternate sampling view creation failed", .{});
    borrow.view = alias;
    display_mat.diffuse_texture = borrow;
    const saved_material = capture_box.material;
    capture_box.setStandardMaterial(display_mat);
    if (!prepare(&capture_scene)) smokeFail("feedback setup prepare failed", .{});
    if (rtt.renderPrimaryView(&capture_scene, .{})) |_| {
        smokeFail("alternate-view feedback capture unexpectedly succeeded", .{});
    } else |err| {
        if (err != error.FeedbackLoop) smokeFail("feedback capture gave {s}, want FeedbackLoop", .{@errorName(err)});
    }
    if (rtt.isCapturing()) smokeFail("rejected feedback left a pass open", .{});
    capture_box.material = saved_material;
    display_mat.diffuse_texture = null;
}

export fn frame() callconv(.c) void {
    const dt: f32 = @floatCast(sapp.frameDuration());

    // Simulate (rotating box => every capture differs; --freeze for
    // deterministic cross-run compare).
    if (!freeze_motion) {
        const dt_norm: f32 = dt * 60.0;
        capture_box.rotation.x += 0.8 * dt_norm;
        capture_box.rotation.y += 1.6 * dt_norm;
    }
    capture_scene.update(dt) catch |err| smokeFail("capture update: {s}", .{@errorName(err)});
    display_scene.update(dt) catch |err| smokeFail("display update: {s}", .{@errorName(err)});

    // Resize BEFORE capture and display prepare; published draws must never
    // retain a handle destroyed between prepare and render.
    if (!resize_done and frame_limit >= 2 and frame_count + 1 >= frame_limit / 2) {
        resize_done = true;
        const nw = @max(32, rtt_size / 3);
        if (nw == rtt.width) {
            smokeFail("resize test needs a distinct size ({} == {})", .{ nw, rtt.width });
        } else if (!rtt.resize(nw, nw)) {
            smokeFail("mid-run resize to {}x{} failed", .{ nw, nw });
        } else {
            std.debug.print("rtt-smoke: resized to {}x{} est={} bytes\n", .{ rtt.width, rtt.height, rtt.estimatedBytes() });
        }
        if (!rtt_msaa4.resize(128, 128)) smokeFail("msaa mid-run resize failed", .{});
        const sampler_desc = sg.querySamplerDesc(rtt_msaa4.sampleSampler());
        if (sampler_desc.min_filter != .NEAREST or sampler_desc.mag_filter != .NEAREST) {
            smokeFail("resize changed nearest sampler filters", .{});
        }
    }
    if (window_resize_pending) {
        window_resize_pending = false;
        const nw: u32 = @max(32, @as(u32, @intCast(@max(0, sapp.width()))) / 2);
        const nh: u32 = @max(32, @as(u32, @intCast(@max(0, sapp.height()))) / 2);
        if (nw != rtt.width or nh != rtt.height) {
            if (!rtt.resize(nw, nh)) smokeFail("window resize to {}x{} failed", .{ nw, nh });
        }
    }

    // Refresh the display borrow BEFORE display prepare: draw records bake
    // material textures at prepare, so a post-prepare refresh would lag a
    // frame (and a post-resize refresh would bind dead handles all frame).
    // Invalid target => null => default-texture fallback, never id-0 binds.
    if (rtt.isValid()) {
        display_mat.diffuse_texture = rtt.asTexture();
    } else {
        display_mat.diffuse_texture = null;
    }

    const previous_capture_frame = capture_scene.preparedDraws().frame_id;
    if (!prepare(&capture_scene)) smokeFail("capture prepare failed at frame {}", .{frame_count});
    if (!prepare(&display_scene)) smokeFail("display prepare failed at frame {}", .{frame_count});
    if (capture_scene.preparedDraws().frame_id != previous_capture_frame + 1) {
        smokeFail("capture did not consume a fresh producer frame {}", .{frame_count});
    }
    const cap_draws_before = capture_scene.stats.draw_calls;
    const cap_tris_before = capture_scene.stats.triangles;

    // Capture (between prepare and render, while draws are consumable).
    // Default capture slot is disjoint from primary, PIP and refraction.
    rtt.renderPrimaryView(&capture_scene, .{
        .clear_color = z.Color4.new(0.10, 0.16, 0.30, 1.0),
    }) catch |err| smokeFail("capture frame {}: {s}", .{ frame_count, @errorName(err) });
    // Visibility proof (no readback exists): the capture MUST have issued
    // real draws into the target — a log-0 run with zero draws is a fake.
    const cap_draws = capture_scene.stats.draw_calls - cap_draws_before;
    const cap_tris = capture_scene.stats.triangles - cap_tris_before;
    if (cap_draws == 0 or cap_tris == 0) {
        smokeFail("capture frame {} issued no draws ({} draws, {} tris)", .{ frame_count, cap_draws, cap_tris });
    } else {
        captures_ok += 1;
    }

    // 4x resolve exercise: a real clear through the MSAA+resolve pair.
    if (!rtt_msaa4.clear(z.Color4.new(0.20, 0.05, 0.05, 1.0), 1.0)) {
        smokeFail("msaa clear failed at frame {}", .{frame_count});
    }

    // Present the display scene (swapchain; the capture scene stays
    // offscreen — probe-style capture needs no present).
    const show_draws_before = display_scene.stats.draw_calls;
    display_scene.render();
    if (display_scene.stats.draw_calls == show_draws_before) {
        smokeFail("display render at frame {} drew nothing", .{frame_count});
    }

    // Engine refraction target must be live after a refractive render —
    // and must NOT exist without opt-in draws (no work without opt-in).
    if (want_refraction) {
        const rt = &display_scene.refraction.target;
        if (!rt.isValid() or rt.width == 0 or rt.height == 0) {
            smokeFail("refraction target invalid after refractive render at frame {}", .{frame_count});
        }
    } else if (display_scene.refraction.target.isValid()) {
        smokeFail("refraction target allocated with no refractive draws", .{});
    }

    // One reuse probe: re-present without re-preparing. Reuse must not
    // crash, must advance the streak, must leave stats restored, and must
    // keep the refraction target alive (capture is skipped under reuse).
    if (!reuse_probed and frame_limit > 12 and frame_count == 10) {
        reuse_probed = true;
        const draws_before_reuse = display_scene.stats.draw_calls;
        const streak_before = display_scene.reuseStreak();
        display_scene.renderReuse();
        if (display_scene.reuseStreak() != streak_before + 1) {
            smokeFail("renderReuse did not advance the streak", .{});
        }
        if (display_scene.stats.draw_calls != draws_before_reuse) {
            smokeFail("renderReuse did not restore stats", .{});
        }
        if (want_refraction and !display_scene.refraction.target.isValid()) {
            smokeFail("refraction target lost across renderReuse", .{});
        }
    }

    frame_count += 1;
    if (frame_count % 30 == 0) {
        std.debug.print("rtt-smoke: frame {}/{} captures={} sample_view={} refr_valid={}\n", .{
            frame_count,                               frame_limit, captures_ok, rtt.sampleView().id != 0,
            display_scene.refraction.target.isValid(),
        });
    }
    if (frame_count >= frame_limit) sapp.quit();
}

export fn cleanup() callconv(.c) void {
    const verdict_pass = smoke_failures == 0 and log_errors == 0;
    std.debug.print("rtt-smoke: frames={} captures_ok={} failures={} log_errors={} log_warnings={} verdict={s}\n", .{
        frame_count,                          captures_ok, smoke_failures, log_errors, log_warnings,
        if (verdict_pass) "PASS" else "FAIL",
    });
    // Borrow detach first: display materials reference the capture target
    // (and the checker); null them before any deinit so no destroy path can
    // ever see a borrowed handle as owned (materials never owned them, but
    // the detach makes the discipline structural, not conventional).
    // Scene keeps split registries (materials / pbr_materials).
    display_mat.diffuse_texture = null;
    for (display_scene.materials.items) |mat| mat.diffuse_texture = null;
    for (display_scene.pbr_materials.items) |mat| {
        mat.albedo_texture = null;
        mat.normal_texture = null;
        mat.metallic_roughness_texture = null;
        mat.emissive_texture = null;
        mat.occlusion_texture = null;
    }
    // Context lifecycle: scenes, then owned textures/targets, then context.
    capture_scene.deinit();
    display_scene.deinit();
    if (checker_owned) checker_tex.deinit();
    rtt.deinit();
    rtt_msaa4.deinit();
    _ = gpa.deinit();
    sg.shutdown();
    if (!verdict_pass) std.process.exit(1);
}

export fn event(ev: [*c]const sapp.Event) callconv(.c) void {
    switch (ev.*.type) {
        .RESIZED => window_resize_pending = true,
        .KEY_DOWN => switch (ev.*.key_code) {
            .ESCAPE => sapp.quit(),
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
        .window_title = "agate rtt-smoke (capture/sample)",
        .width = 800,
        .height = 600,
        .sample_count = 1,
        .logger = .{ .func = rttLogger },
    });
}
