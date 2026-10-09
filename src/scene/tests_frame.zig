//! Staged frame tests (part 1): snapshot handoff, staged finish, UI capture (P6). Split from scene/tests.zig.
const std = @import("std");
const sokol = @import("sokol");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color3 = math.Color3;
const Color4 = math.Color4;
const BoundingBox = math.BoundingBox;
const Ray = math.Ray;
const camera_mod = @import("../camera.zig");
const Camera = camera_mod.Camera;
const Mesh = @import("../mesh.zig").Mesh;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;
const ShaderMaterial = @import("../material.zig").ShaderMaterial;
const Texture = @import("../texture.zig").Texture;
const CubeTexture = @import("../texture.zig").CubeTexture;
const UICanvas = @import("../ui.zig").UICanvas;
const particles = @import("../particles.zig");
const ParticleSystem = particles.ParticleSystem;
const scene_mod = @import("../scene.zig");
const Scene = scene_mod.Scene;
const SceneStats = scene_mod.SceneStats;
const FrameDrawSlot = scene_mod.FrameDrawSlot;
const RenderMeshItem = scene_mod.RenderMeshItem;
const scene_lights = @import("light_rig.zig");
const gpu_thread = @import("../gpu_thread.zig");
const upload_meter = @import("../gpu_upload_meter.zig");
const postprocess = @import("../postprocess.zig");
const tu = @import("tests_util.zig");
const buildForTest = tu.buildForTest;
const finishForTest = tu.finishForTest;
const stageAndPrepareForTest = tu.stageAndPrepareForTest;
const p7CpuShadowPass = tu.p7CpuShadowPass;
const p7ShadowTotal = tu.p7ShadowTotal;
const p7FindByMeshIndex = tu.p7FindByMeshIndex;
const stage1FillInstances = tu.stage1FillInstances;
const stage1Scene = tu.stage1Scene;
const wave32PushTestSystem = tu.wave32PushTestSystem;
const wave32FreeTestSystems = tu.wave32FreeTestSystems;

test "publishFrameSnapshot and staged finish snapshot handoff" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);
    try std.testing.expect(!scene.frame_prepared);

    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expect(scene.frame_snapshot.has_camera);
    try std.testing.expectEqual(@as(i32, 1920), scene.frame_snapshot.screen_w);
    try std.testing.expectEqual(@as(i32, 1080), scene.frame_snapshot.screen_h);
}

test "saturated snapshot mailbox keeps the newest generation" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);

    // Staged newest-wins: three publishes without a consuming finish,
    // then staged begin/finish consumes the newest; stale slots drain so
    // the finish takes 300 — not an older published generation over
    // the newer staged build. packFrameSnapshot sets screen_w before
    // sets screen_w before the no-camera early-out, so no camera is needed.
    scene.publishFrameSnapshot(1.0, 100, 100);
    scene.publishFrameSnapshot(1.0, 200, 200);
    scene.publishFrameSnapshot(1.0, 300, 300);

    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(i32, 300), scene.frame_snapshot.screen_w);
    try std.testing.expectEqual(@as(i32, 300), scene.frame_snapshot.screen_h);
}

test "staged build stages an empty upload tally without an upload queue" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);

    // No uploads queue and no io runner on the fixture: the staged finish takes
    // the synchronous path and stages a zero tally for the stats publish.
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(usize, 0), scene.frame_uploads.count);
    try std.testing.expectEqual(@as(u64, 0), scene.frame_uploads.bytes);
    // Without GPU context there are no dynamic updates: zero metrics,
    // meter counter was reset at the start of staged finish.
    try std.testing.expectEqual(@as(u64, 0), scene.stats.updated_bytes_frame);
    try std.testing.expectEqual(@as(u64, 0), upload_meter.peek());
}

test "staged finish rebuilds outline snapshots without GPU" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);

    // No GPU context on the fixture (default textures zeroed): outline
    // capture still runs unconditionally after the (skipped) pre-stage, as
    // before P5 — snapshots clear and rebuild every staged finish (P7: into
    // the published slot, read via preparedDraws()).
    var mesh = Mesh{
        .name = "headless_outline",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(4, 0, 0),
    };
    try scene.outline_meshes.append(alloc, &mesh);

    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(usize, 1), scene.preparedDraws().outline_items.items.len);
    // Rebuild, not accumulate.
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(usize, 1), scene.preparedDraws().outline_items.items.len);
    // Clearing works: a hidden mesh rebuilds to empty.
    mesh.is_visible = false;
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(usize, 0), scene.preparedDraws().outline_items.items.len);
}

test "staged finish stages highlight snapshots without GPU" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);

    // Highlight entries borrow live meshes; the fixed-size layer needs no
    // deinit. The mesh must sit in scene.meshes (OOB referents are skipped
    // at stage — highlights have no latch patch stage).
    const m = try alloc.create(Mesh);
    m.* = @import("../testing.zig").testMesh("headless_highlight");
    m.index_count = 3;
    m.position = Vec3.new(4, 0, 0);
    try scene.meshes.append(alloc, m);
    defer {
        scene.destroyMesh(m);
        scene.meshes.deinit(alloc);
    }

    const id = try scene.addHighlightMesh(m, .{ .color = .{ 1.0, 0.0, 0.0, 1.0 }, .blur = 6.0, .intensity = 0.8 });
    try std.testing.expectEqual(@as(usize, 0), id);
    try std.testing.expectEqual(@as(usize, 1), scene.highlightCount());

    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(usize, 1), scene.preparedDraws().highlight_items.items.len);
    const staged = scene.preparedDraws().highlight_items.items[0];
    try std.testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 1.0 }, staged.color);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), staged.blur, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), staged.intensity, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), staged.model.m[12], 1e-4);
    // Rebuild, not accumulate.
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(usize, 1), scene.preparedDraws().highlight_items.items.len);
    // Clearing works: a hidden mesh rebuilds to empty.
    m.is_visible = false;
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(usize, 0), scene.preparedDraws().highlight_items.items.len);
    m.is_visible = true;
}

test "highlight staged items freeze the model (no live reads at render time)" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);

    const m = try alloc.create(Mesh);
    m.* = @import("../testing.zig").testMesh("frozen_highlight");
    m.index_count = 3;
    m.position = Vec3.new(4, 0, 0);
    try scene.meshes.append(alloc, m);
    defer {
        scene.destroyMesh(m);
        scene.meshes.deinit(alloc);
    }
    _ = try scene.addHighlightMesh(m, .{});

    try stageAndPrepareForTest(&scene);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), scene.preparedDraws().highlight_items.items[0].model.m[12], 1e-4);
    // Game-side mutation after prepare cannot tear the published front:
    // the render consumes this frozen model, never the live mesh.
    m.position = Vec3.new(9, 0, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), scene.preparedDraws().highlight_items.items[0].model.m[12], 1e-4);
    // The next prepare picks the new transform up (freshness preserved).
    try stageAndPrepareForTest(&scene);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), scene.preparedDraws().highlight_items.items[0].model.m[12], 1e-4);
}

test "destroyMesh drops highlight entries and the stage rebuilds empty" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.meshes.deinit(alloc);

    const m = try alloc.create(Mesh);
    m.* = @import("../testing.zig").testMesh("doomed_highlight");
    m.index_count = 3;
    try scene.meshes.append(alloc, m);
    _ = try scene.addHighlightMesh(m, .{});
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(usize, 1), scene.preparedDraws().highlight_items.items.len);

    // Off-context destroy (headless fixture: no context thread, so the
    // epoch-retired branch): the entry drops synchronously and the next
    // stage rebuilds to empty — fail-closed, never a dead draw.
    scene.destroyMesh(m);
    try std.testing.expectEqual(@as(usize, 0), scene.highlightCount());
    try std.testing.expect(scene.getHighlightMesh(0) == null);
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(usize, 0), scene.preparedDraws().highlight_items.items.len);
}

test "zero highlights is structurally bit-identical (no passes, no state)" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);

    // Fresh scene (and every load — highlights are transient, never
    // serialized): zero entries, zero staged items, gate closed.
    try std.testing.expectEqual(@as(usize, 0), scene.highlightCount());
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(usize, 0), scene.preparedDraws().highlight_items.items.len);
    try std.testing.expect(!postprocess.highlightActive(scene.post_process.enabled, scene.preparedDraws().highlight_items.items.len));
    // Post ON with zero items stays gated off too: renderChain skips the
    // whole PASS 2.85 block, so no mask/blur draws run and — by the lazy
    // target discipline (no resize in resizeAll/beginMainPass) — no
    // highlight RTs allocate (~12 MiB @1080p RGBA16F stays unsized).
    try std.testing.expect(!postprocess.highlightActive(true, 0));
    try std.testing.expect(postprocess.highlightActive(true, 1));
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, postprocess.highlightParams(false));
    // Headless fixture owns no highlight GPU state: no mask/blur pass can
    // run, no targets exist for the census (which gates on base size, so
    // the unsized pass honestly contributes zero VRAM).
    // (The OFF-path view discipline — renderChain feeds .{} blurred+mask
    // views when inactive — is structural in postfx_stack.zig; the
    // fixture's postprocess_pass stays `undefined` by design, so it is
    // not dereferenced here.)
    try std.testing.expectEqual(@as(u32, 0), scene.postfx.highlight_pass.mask_pipeline_u16.id);
    try std.testing.expectEqual(@as(u32, 0), scene.postfx.highlight_pass.blur_pipeline.id);
    try std.testing.expectEqual(@as(i32, 0), scene.postfx.highlight_pass.base_width);
    try std.testing.expectEqual(@as(i32, 0), scene.postfx.highlight_pass.base_height);
    try std.testing.expectEqual(@as(u32, 0), scene.postfx.highlight_pass.mask_image.id);
}

test "shaft defaults are off with zero GPU state (bit-identical)" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);

    // Fresh scene (and every load — shafts are transient, never
    // serialized): disabled by default, and the gate needs post + shaft
    // + live CSM data, so the default fixture keeps PASS 2.9 closed.
    try std.testing.expect(!scene.post_process.shaft_enabled);
    try std.testing.expect(!postprocess.shaftActive(scene.post_process.enabled, scene.post_process, true));
    try std.testing.expect(!postprocess.shaftActive(true, scene.post_process, true));
    try std.testing.expect(!postprocess.shaftActive(true, postprocess.PostProcessOptions{ .shaft_enabled = true }, false));
    try std.testing.expect(postprocess.shaftActive(true, postprocess.PostProcessOptions{ .shaft_enabled = true }, true));
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, postprocess.shaftParams(scene.post_process, false));
    // Headless fixture owns no shaft GPU state: no raymarch/blur pass
    // can run, no targets exist for the census (which gates on base
    // size, so the unsized pass honestly contributes zero VRAM).
    // (The OFF-path view discipline — renderChain feeds the resolved
    // scene placeholder when inactive — is structural in
    // postfx_stack.zig; the fixture's postprocess_pass stays `undefined`
    // by design, so it is not dereferenced here.)
    try std.testing.expectEqual(@as(u32, 0), scene.postfx.volumetric_pass.raymarch_pipeline.id);
    try std.testing.expectEqual(@as(u32, 0), scene.postfx.volumetric_pass.blur_pipeline.id);
    try std.testing.expectEqual(@as(i32, 0), scene.postfx.volumetric_pass.base_width);
    try std.testing.expectEqual(@as(i32, 0), scene.postfx.volumetric_pass.base_height);
    try std.testing.expectEqual(@as(u32, 0), scene.postfx.volumetric_pass.raymarch_image.id);
}

test "staged finish transfers update tick, preserves prepare_ms" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);

    // Game phase registered tick via recordUpdateTime (not in stats),
    // previous frame left counters and post_ms: reset on staged finish must
    // transfer staged update_ms, preserve prepare_ms, and zero the rest.
    // Direct mutation of stats.update_ms from update side is prohibited
    // (contract of recordUpdateTime).
    scene.recordUpdateTime(2.5);
    scene.stats.prepare_ms = 1.25;
    scene.stats.draw_calls = 41;
    scene.stats.triangles = 1000;
    scene.stats.post_ms = 3.0;

    try stageAndPrepareForTest(&scene);

    try std.testing.expectEqual(@as(f32, 2.5), scene.stats.update_ms);
    try std.testing.expectEqual(@as(f32, 1.25), scene.stats.prepare_ms);
    try std.testing.expectEqual(@as(u32, 0), scene.stats.draw_calls);
    try std.testing.expectEqual(@as(u32, 0), scene.stats.triangles);
    try std.testing.expectEqual(@as(f32, 0.0), scene.stats.post_ms);
    // Next frame prepare_ms is recorded by app after staged finish.
}

test "recordUpdateTime stages without touching stats (update||render disjoint)" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);

    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(f32, 0.0), scene.stats.update_ms);

    // Update side (game thread): touches only staged field; stats remain safe
    // for concurrent render reading.
    scene.recordUpdateTime(7.5);
    try std.testing.expectEqual(@as(f32, 7.5), scene.pending_update_ms.load(.monotonic));
    try std.testing.expectEqual(@as(f32, 0.0), scene.stats.update_ms);

    // Latest tick wins prior to prepare; prepare transfers it to stats.
    scene.recordUpdateTime(8.25);
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(f32, 8.25), scene.stats.update_ms);
}

test "async save/load report NoTaskRunner without an io runner" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);

    // The fixture has no io_runner: capture succeeds, dispatch fails with
    // NoTaskRunner (and frees the snapshot — testing.allocator verifies).
    try std.testing.expectError(error.NoTaskRunner, scene.saveStateFileAsync("no_runner.agsc"));
    try std.testing.expectError(error.NoTaskRunner, scene.loadStateFileAsync("no_runner.agsc"));
}

test "P6: staged finish captures UI into the render-owned frame" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.ui_frame.deinit(alloc);

    // Headless canvas (no GPU init): draws only fill CPU-side lists.
    scene.ui_canvas = UICanvas{
        .allocator = alloc,
        .font_texture = std.mem.zeroes(Texture),
    };
    defer {
        if (scene.ui_canvas) |*c| {
            c.vertices.deinit(alloc);
            c.indices.deinit(alloc);
        }
        scene.ui_canvas = null;
    }
    const canvas_ui = &scene.ui_canvas.?;
    canvas_ui.drawRect(10, 20, 30, 40, Color4.white);
    canvas_ui.drawText("ok", 0, 0, 16.0, Color4.white);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);

    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    // Frame owns copies of the full geometry; dims come from the scene
    // snapshot (not live sapp values); borrowed handle IDs are captured.
    try std.testing.expect(scene.ui_frame.has_capture);
    try std.testing.expectEqual(canvas_ui.vertices.items.len, scene.ui_frame.vertices.items.len);
    try std.testing.expectEqual(canvas_ui.indices.items.len, scene.ui_frame.indices.items.len);
    try std.testing.expectEqual(@as(f32, 1920.0), scene.ui_frame.screen_w);
    try std.testing.expectEqual(@as(f32, 1080.0), scene.ui_frame.screen_h);
    try std.testing.expectEqual(canvas_ui.pipeline.id, scene.ui_frame.pipeline.id);
    // Headless: staged but not drawable, zero meter bytes, stats clean.
    try std.testing.expect(!scene.ui_frame.gpu_ready);
    try std.testing.expectEqual(@as(u64, 0), upload_meter.peek());
    try std.testing.expectEqual(@as(u64, 0), scene.stats.updated_bytes_frame);
    // Upload-free draw of a never-uploaded frame: safe no-op.
    scene.ui_frame.drawPrepared();
    try std.testing.expectEqual(@as(u64, 0), upload_meter.peek());

    // Newest capture wins across prepares.
    canvas_ui.drawRect(1, 2, 3, 4, Color4.white);
    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(canvas_ui.vertices.items.len, scene.ui_frame.vertices.items.len);
}

test "P6: camera-less staged finish clears stale UI (no prior overlay)" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.ui_frame.deinit(alloc);

    scene.ui_canvas = UICanvas{
        .allocator = alloc,
        .font_texture = std.mem.zeroes(Texture),
    };
    defer {
        if (scene.ui_canvas) |*c| {
            c.vertices.deinit(alloc);
            c.indices.deinit(alloc);
        }
        scene.ui_canvas = null;
    }
    scene.ui_canvas.?.drawRect(0, 0, 10, 10, Color4.white);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.publishFrameSnapshot(1.0, 640, 480);
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.ui_frame.has_capture);

    // Camera lost: the next prepare must not redisplay the prior overlay.
    for (scene.cameras.items) |entry| {
        if (entry.owns_name) alloc.free(entry.name);
    }
    scene.cameras.clearRetainingCapacity();
    scene.active_camera = null;
    scene.active_camera_index = null;
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(!scene.ui_frame.has_capture);
    try std.testing.expectEqual(@as(usize, 0), scene.ui_frame.vertices.items.len);
    scene.ui_frame.drawPrepared();
    try std.testing.expectEqual(@as(u64, 0), upload_meter.peek());
}

test "P6: staged finish snapshots canvas presence apart from content" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.ui_frame.deinit(alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    // No canvas at all: presence false, content empty.
    scene.publishFrameSnapshot(1.0, 640, 480);
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(!scene.ui_frame.canvas_present);
    try std.testing.expect(!scene.ui_frame.has_capture);

    // Canvas exists but drew nothing: presence true (draw sites still
    // select the frame, preserving the phantom+1 counters), no capture.
    scene.ui_canvas = UICanvas{
        .allocator = alloc,
        .font_texture = std.mem.zeroes(Texture),
    };
    defer {
        if (scene.ui_canvas) |*c| {
            c.vertices.deinit(alloc);
            c.indices.deinit(alloc);
        }
        scene.ui_canvas = null;
    }
    scene.publishFrameSnapshot(1.0, 640, 480);
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.ui_frame.canvas_present);
    try std.testing.expect(!scene.ui_frame.has_capture);
    scene.ui_frame.drawPrepared();
    try std.testing.expectEqual(@as(u64, 0), upload_meter.peek());

    // Canvas removed again: presence follows prepare, never sticks.
    if (scene.ui_canvas) |*c| {
        c.vertices.deinit(alloc);
        c.indices.deinit(alloc);
    }
    scene.ui_canvas = null;
    scene.publishFrameSnapshot(1.0, 640, 480);
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(!scene.ui_frame.canvas_present);
}
