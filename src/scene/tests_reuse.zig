//! Staged frame tests (part 5): snapshot ownership, renderReuse, UI packets. Split from scene/tests.zig.
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

test "snapshot ownership repro: build must not touch consumed frame_snapshot" {
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.binned_source.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.sky.enabled = true;
    scene.sky.exposure = 2.0;

    // Frame A: publish + staged build + finish.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.frame_snapshot.has_camera);
    const eye_a = scene.frame_snapshot.primary_cam.eye;
    try std.testing.expectEqual(@as(i32, 800), scene.frame_snapshot.screen_w);
    try std.testing.expectEqual(@as(f32, 2.0), scene.frame_snapshot.sky_exposure);

    // Mutate live camera + environment, publish B, then build B.
    if (scene.active_camera) |*c| c.free.position = Vec3.new(10, 0, 0);
    if (scene.cameras.items.len > 0) scene.cameras.items[0].camera.free.position = Vec3.new(10, 0, 0);
    scene.sky.exposure = 9.0;
    scene.publishFrameSnapshot(16.0 / 9.0, 640, 480);
    try buildForTest(&scene);

    // The consumed render snapshot must still be A: the producer build owns
    // its own snapshot and never overwrites frame_snapshot (update may
    // overlap render).
    try std.testing.expectEqual(@as(i32, 800), scene.frame_snapshot.screen_w);
    try std.testing.expectEqual(@as(f32, 2.0), scene.frame_snapshot.sky_exposure);
    try std.testing.expectEqual(eye_a, scene.frame_snapshot.primary_cam.eye);
}

test "snapshot ownership: latch freezes B incl PIP/shadow; post-build C waits" {
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.binned_source.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam0 = Camera{ .free = camera_mod.FreeCamera.init("Main", .{}) };
    const cam1 = Camera{ .free = camera_mod.FreeCamera.init("Pip", .{}) };
    _ = try scene.addCamera(.{ .name = "Main", .camera = cam0 });
    _ = try scene.addCamera(.{ .name = "Pip", .camera = cam1 });
    scene.enable_multi_camera = true;
    scene.active_camera_index = 0;
    scene.shadows.enabled = true;
    scene.sky.enabled = true;
    scene.sky.exposure = 2.0;

    var mesh = Mesh{
        .name = "snap_b",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.zero,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    try scene.meshes.append(alloc, &mesh);

    // Prime A so the latch below is a B-vs-C comparison, not empty-vs-B.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try stageAndPrepareForTest(&scene);

    // Build B (frozen generation).
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try buildForTest(&scene);
    const eye_b = scene.build_snapshot.primary_cam.eye;
    try std.testing.expect(scene.build_snapshot.has_camera);
    try std.testing.expect(scene.build_snapshot.shadows_enabled);
    try std.testing.expect(scene.build_snapshot.enable_multi_camera);
    try std.testing.expectEqual(@as(usize, 2), scene.build_snapshot.camera_count);

    // Mutate + publish C AFTER the build, before the latch.
    mesh.position = Vec3.new(50, 0, 0);
    scene.sky.exposure = 9.0;
    scene.shadows.enabled = false;
    scene.cameras.items[1].enabled = false;
    scene.publishFrameSnapshot(16.0 / 9.0, 640, 480);

    // Latch: frame_snapshot must be exactly B, queues frozen at B.
    finishForTest(&scene);
    try std.testing.expectEqual(@as(i32, 800), scene.frame_snapshot.screen_w);
    try std.testing.expectEqual(@as(f32, 2.0), scene.frame_snapshot.sky_exposure);
    try std.testing.expect(scene.frame_snapshot.shadows_enabled);
    try std.testing.expect(scene.frame_snapshot.enable_multi_camera);
    try std.testing.expectEqual(@as(usize, 2), scene.frame_snapshot.camera_count);
    try std.testing.expectEqual(eye_b, scene.frame_snapshot.primary_cam.eye);
    const draws = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 1), draws.primary.items.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), draws.primary.items.items[0].model.m[12], 1e-4);
    try std.testing.expectEqual(@as(usize, 1), draws.views[1].items.items.len);
    try std.testing.expect(p7ShadowTotal(draws) > 0);

    // C waited in the mailbox: the next staged build sees it.
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(i32, 640), scene.frame_snapshot.screen_w);
    try std.testing.expectEqual(@as(f32, 9.0), scene.frame_snapshot.sky_exposure);
    try std.testing.expect(!scene.frame_snapshot.shadows_enabled);
    try std.testing.expectEqual(@as(usize, 0), scene.preparedDraws().shadow.items.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), scene.preparedDraws().primary.items.items[0].model.m[12], 1e-4);
}

test "snapshot ownership: multi-build newest wins; no-publish refresh; removal; staged" {
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.binned_source.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.sky.enabled = true;
    scene.sky.exposure = 1.0;
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try stageAndPrepareForTest(&scene);

    // Two builds before one latch: newest wins.
    scene.sky.exposure = 2.0;
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try buildForTest(&scene);
    scene.sky.exposure = 5.0;
    scene.publishFrameSnapshot(16.0 / 9.0, 640, 480);
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(f32, 5.0), scene.build_snapshot.sky_exposure);
    finishForTest(&scene);
    try std.testing.expectEqual(@as(f32, 5.0), scene.frame_snapshot.sky_exposure);
    try std.testing.expectEqual(@as(i32, 640), scene.frame_snapshot.screen_w);

    // No-publish build packs fresh live state (works without publish).
    scene.sky.exposure = 7.0;
    if (scene.active_camera) |*c| c.free.position = Vec3.new(3, 0, 0);
    if (scene.cameras.items.len > 0) scene.cameras.items[0].camera.free.position = Vec3.new(3, 0, 0);
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(f32, 7.0), scene.build_snapshot.sky_exposure);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), scene.build_snapshot.primary_cam.eye.x, 1e-4);
    finishForTest(&scene);
    try std.testing.expectEqual(@as(f32, 7.0), scene.frame_snapshot.sky_exposure);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), scene.frame_snapshot.primary_cam.eye.x, 1e-4);

    // Camera removal: fresh pack is camera-less, latch copies it coherently.
    for (scene.cameras.items) |entry| {
        if (entry.owns_name) alloc.free(entry.name);
    }
    scene.cameras.clearRetainingCapacity();
    scene.active_camera = null;
    scene.active_camera_index = null;
    try buildForTest(&scene);
    try std.testing.expect(!scene.build_snapshot.has_camera);
    finishForTest(&scene);
    try std.testing.expect(!scene.frame_snapshot.has_camera);
    try std.testing.expectEqual(@as(usize, 0), scene.preparedDraws().primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.preparedDraws().shadow.items.items.len);

    // Staged: publish + fresh build takes the published generation.
    const cam2 = Camera{ .free = camera_mod.FreeCamera.init("Cam2", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam2", .camera = cam2 });
    scene.sky.exposure = 4.0;
    scene.publishFrameSnapshot(16.0 / 9.0, 320, 240);
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.frame_snapshot.has_camera);
    try std.testing.expectEqual(@as(i32, 320), scene.frame_snapshot.screen_w);
    try std.testing.expectEqual(@as(f32, 4.0), scene.frame_snapshot.sky_exposure);
}

test "snapshot ownership: producer shadow freezes on snapshot, ignores live toggle" {
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.binned_source.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    var mesh = Mesh{
        .name = "snap_shadow",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.zero,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    try scene.meshes.append(alloc, &mesh);

    // Publish with shadows enabled, then toggle live OFF before the build
    // (no new publish): the producer must still build the snapshot gen.
    scene.shadows.enabled = true;
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.shadows.enabled = false;
    try buildForTest(&scene);
    try std.testing.expect(scene.build_snapshot.shadows_enabled);
    finishForTest(&scene);
    try std.testing.expect(scene.frame_snapshot.shadows_enabled);
    try std.testing.expect(p7ShadowTotal(scene.preparedDraws()) > 0);

    // Staged with live-off publish: the fresh build disables shadows
    // and the payload is empty.
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(!scene.frame_snapshot.shadows_enabled);
    try std.testing.expectEqual(@as(usize, 0), scene.preparedDraws().shadow.items.items.len);
}

test "snapshot ownership: producer staging uses published eye, not live mutation" {
    const InstancedMesh = @import("../mesh.zig").InstancedMesh;
    const material_mod = @import("../material.zig");
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.binned_source.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);
    scene.shadows.enabled = false;

    var blend_mat = material_mod.PBRMaterial.init("snap_eye_blend");
    blend_mat.alpha_mode = .blend;
    var src = Mesh{
        .name = "snap_eye_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var mem: [2]InstancedMesh = .{
        .{ .name = "t0", .source_mesh = &src, .position = Vec3.new(-10, 0, 0) },
        .{ .name = "t1", .source_mesh = &src, .position = Vec3.new(10, 0, 0) },
    };
    var ptrs = [_]*InstancedMesh{ &mem[0], &mem[1] };
    var parent = Mesh{
        .name = "snap_eye_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .material = .{ .pbr = &blend_mat },
        .instances = .{ .items = &ptrs, .capacity = 2 },
    };
    try scene.meshes.append(alloc, &parent);

    // Publish with the eye on the left, then move live to the right BEFORE
    // the build (no new publish): the build must freeze on the published eye.
    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{ .position = Vec3.new(-100, 0, 0) }) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    var cube_a = std.mem.zeroes(CubeTexture);
    cube_a.view.id = 77;
    scene.sky.setSkybox(cube_a);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    if (scene.active_camera) |*c| c.free.position = Vec3.new(100, 0, 0);
    if (scene.cameras.items.len > 0) scene.cameras.items[0].camera.free.position = Vec3.new(100, 0, 0);
    var cube_b = std.mem.zeroes(CubeTexture);
    cube_b.view.id = 88;
    scene.sky.setSkybox(cube_b);
    try buildForTest(&scene);

    // Frozen generation: published eye/sky, not the live mutation.
    try std.testing.expectApproxEqAbs(@as(f32, -100.0), scene.build_snapshot.primary_cam.eye.x, 1e-4);
    try std.testing.expectEqual(@as(u32, 77), scene.build_snapshot.sky_texture.?.view.id);
    // Staging sorted farthest-from-published-eye first (+10, then the source
    // mesh entry at x=0, then -10).
    const scratch = scene.draws.slots[scene.build_slot.load(.monotonic)].primary.instance_matrices.items;
    try std.testing.expectEqual(@as(usize, 3), scratch.len);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), scratch[0].m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), scratch[1].m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -10.0), scratch[2].m[12], 1e-4);
    finishForTest(&scene);
    try std.testing.expectApproxEqAbs(@as(f32, -100.0), scene.frame_snapshot.primary_cam.eye.x, 1e-4);
    try std.testing.expectEqual(@as(u32, 77), scene.frame_snapshot.sky_texture.?.view.id);
}

test "renderReuse re-draws the consumed front without a prepare" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);

    // Camera-less headless fixture: prepare publishes a consumable front
    // (frame_id != 0), render consumes it via the no-camera early return
    // (no sg.* headless).
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expect(scene.draws.slots[scene.draws.front].frame_id != 0);
    const front0 = scene.draws.front;
    const slot_id0 = scene.preparedDraws().frame_id;

    scene.render();
    try std.testing.expect(!scene.frame_prepared);
    try std.testing.expectEqual(front0, scene.draws.front);

    // Seed the already-recorded stats: reuse must leave them untouched, and
    // must not advance any frame/epoch identity.
    const fid = scene.frame_id;
    const bseq = scene.build_seq.load(.monotonic);
    const epoch = scene.retire_epoch;
    const completed = scene.gpu_retire.lastCompleted();
    scene.stats.draw_calls = 41;
    scene.stats.triangles = 1000;

    scene.renderReuse();
    try std.testing.expect(!scene.frame_prepared);
    try std.testing.expect(!scene.rendering_reuse);
    try std.testing.expectEqual(front0, scene.draws.front);
    try std.testing.expectEqual(slot_id0, scene.preparedDraws().frame_id);
    try std.testing.expectEqual(fid, scene.frame_id);
    try std.testing.expectEqual(bseq, scene.build_seq.load(.monotonic));
    try std.testing.expectEqual(epoch, scene.retire_epoch);
    try std.testing.expectEqual(completed, scene.gpu_retire.lastCompleted());
    try std.testing.expectEqual(@as(u32, 41), scene.stats.draw_calls);
    try std.testing.expectEqual(@as(u32, 1000), scene.stats.triangles);

    // Repeated reuse is stable: same front, same identities, same stats.
    scene.renderReuse();
    try std.testing.expect(!scene.frame_prepared);
    try std.testing.expectEqual(front0, scene.draws.front);
    try std.testing.expectEqual(slot_id0, scene.preparedDraws().frame_id);
    try std.testing.expectEqual(fid, scene.frame_id);
    try std.testing.expectEqual(epoch, scene.retire_epoch);
    try std.testing.expectEqual(completed, scene.gpu_retire.lastCompleted());
    try std.testing.expectEqual(@as(u32, 41), scene.stats.draw_calls);
}

test "pending prepared frame is consumed by render, not renderReuse" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);

    // Fresh prepare leaves a pending frame: the reuse contract requires
    // frame_prepared == false (debug assert), so a pending frame must go
    // through render(). This pins the documented pending-frame behavior
    // without tripping the assert (which traps in Debug/ReleaseSafe).
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    const pending_front = scene.draws.front;
    const pending_slot = scene.preparedDraws().frame_id;
    try std.testing.expect(pending_slot != 0);
    scene.render();
    try std.testing.expect(!scene.frame_prepared);
    try std.testing.expectEqual(pending_front, scene.draws.front);
    try std.testing.expectEqual(pending_slot, scene.preparedDraws().frame_id);
}

test "renderReuse records the re-presented frame with the consumed stats" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    sokol.time.setup();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.profiler.deinit();
    scene.profiler.start();
    defer scene.profiler.stop();

    // Camera-less headless fixture: the no-camera render early-returns
    // before the profiler tail, so prepare+render records nothing here —
    // the full sg draw path (where render() itself records) cannot run
    // headless. The renderReuse wrapper path below is camera-independent,
    // so this exercises the real new recording code end to end (no helper
    // indirection needed).
    try stageAndPrepareForTest(&scene);
    scene.render();
    try std.testing.expect(!scene.frame_prepared);
    try std.testing.expectEqual(@as(usize, 0), scene.profiler.frames.items.len);
    try std.testing.expectEqual(@as(u64, 0), scene.profiler_frame_seq);

    // Seed representative counters for the consumed frame (headless
    // no-camera yields zeros); the reuse record must repeat them exactly.
    scene.stats.draw_calls = 41;
    scene.stats.triangles = 1000;
    const consumed = scene.stats;

    scene.renderReuse();
    try std.testing.expectEqual(@as(usize, 1), scene.profiler.frames.items.len);
    try std.testing.expectEqual(@as(u64, 1), scene.profiler_frame_seq);
    try std.testing.expectEqual(@as(u64, 1), scene.profiler.frames.items[0].frame_index);
    try std.testing.expectEqual(consumed.draw_calls, scene.profiler.frames.items[0].draw_calls);
    try std.testing.expectEqual(consumed.triangles, scene.profiler.frames.items[0].triangles);
    try std.testing.expectEqual(consumed.draw_calls, scene.stats.draw_calls);

    // Second reuse: one more record, unique monotonic frame_index, no dup.
    scene.renderReuse();
    try std.testing.expectEqual(@as(usize, 2), scene.profiler.frames.items.len);
    try std.testing.expectEqual(@as(u64, 2), scene.profiler_frame_seq);
    try std.testing.expectEqual(@as(u64, 2), scene.profiler.frames.items[1].frame_index);
    try std.testing.expect(scene.profiler.frames.items[1].frame_index != scene.profiler.frames.items[0].frame_index);
    try std.testing.expectEqual(consumed.draw_calls, scene.profiler.frames.items[1].draw_calls);
    try std.testing.expectEqual(consumed.triangles, scene.profiler.frames.items[1].triangles);
}

// Upload-intent gate (slice 2, c): a reuse streak applies NOTHING — no
// retire flush, no upload completion, no staged-UI consumption, no frame
// advance. Only a successful prepare drains intents.
test "reuse applies no retire flush and no upload application" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);

    // Publish + consume one frame so reuse has a front to re-present.
    try stageAndPrepareForTest(&scene);
    scene.render();
    try std.testing.expect(scene.hasConsumableFrame());

    // One retire intent + one staged UI intent pending.
    const m = try alloc.create(Mesh);
    m.* = @import("../testing.zig").testMesh("reuse_gate_probe");
    scene.gpu_retire.retireMesh(alloc, m);
    try std.testing.expectEqual(@as(usize, 1), scene.pendingRetires());
    const completed = scene.gpu_retire.lastCompleted();
    scene.ui_frame.has_capture = true;
    scene.ui_frame.needs_upload = true;

    scene.renderReuse();
    // Nothing drained, nothing consumed, nothing advanced.
    try std.testing.expectEqual(@as(usize, 1), scene.pendingRetires());
    try std.testing.expectEqual(completed, scene.gpu_retire.lastCompleted());
    try std.testing.expect(scene.ui_frame.has_capture);
    try std.testing.expect(scene.ui_frame.needs_upload);
    try std.testing.expectEqual(@as(u64, 1), scene.reuseStreak());

    scene.renderReuse();
    try std.testing.expectEqual(@as(usize, 1), scene.pendingRetires());
    try std.testing.expectEqual(@as(u64, 2), scene.reuseStreak());

    // The next successful prepare ends the streak and flushes the retire.
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(u64, 0), scene.reuseStreak());
    try std.testing.expectEqual(@as(usize, 0), scene.pendingRetires());
}

test "prepare resets the reuse streak, reuse bumps it" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);

    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(u64, 0), scene.reuseStreak());
    scene.render();
    try std.testing.expectEqual(@as(u64, 0), scene.reuseStreak());
    scene.renderReuse();
    scene.renderReuse();
    scene.renderReuse();
    try std.testing.expectEqual(@as(u64, 3), scene.reuseStreak());
    try std.testing.expectEqual(@as(usize, 0), scene.pendingRetires());
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(u64, 0), scene.reuseStreak());
}

// Game-side UI CPU packet (slice 2, b): stage records canvas geometry into
// the back slot (sg-free, zero meter bytes); the latch consumes the staged
// bytes even when the canvas is mutated afterwards; a second prepare
// without a new stage falls back to the legacy canvas read.
test "ui packet staged build freezes bytes; post-stage canvas mutation invisible" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    _ = upload_meter.takeAndReset();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
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
    const canvas = &scene.ui_canvas.?;
    canvas.pipeline = .{ .id = 7 };
    canvas.vertex_buffer = .{ .id = 21 };
    canvas.index_buffer = .{ .id = 23 };
    canvas.drawRect(10, 20, 30, 40, Color4.white);
    canvas.drawText("packet", 0, 0, 16.0, Color4.white);
    canvas.drawLine(0, 0, 5, 5, 2.0, Color4.white);
    const staged_verts = canvas.vertices.items.len;
    const staged_idx = canvas.indices.items.len;
    try std.testing.expect(staged_verts > 0 and staged_idx > 0);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);

    // Full producer build then UI stage into the claimed slot (build resets
    // the slot, so stageUi runs after build), then publish. No GPU touched.
    var uic = scene.tryClaimBuildSlot() orelse std.debug.panic("{s}", .{"ui build saturated"});
    uic.build();
    uic.stageUi();
    uic.publish();
    const staged_slot = scene.build_slot.load(.monotonic);
    const back = scene.draws.slotAt(staged_slot);
    try std.testing.expect(back.ui_packet.valid);
    try std.testing.expect(back.ui_packet.canvas_present);
    try std.testing.expectEqual(staged_verts, back.ui_vertices.items.len);
    try std.testing.expectEqual(staged_idx, back.ui_indices.items.len);
    try std.testing.expectEqual(@as(u32, 7), back.ui_packet.handles.pipeline.id);
    try std.testing.expectEqual(@as(u32, 21), back.ui_packet.handles.vertex_buffer.id);
    try std.testing.expectEqual(@as(u32, 23), back.ui_packet.handles.index_buffer.id);
    try std.testing.expectEqual(@as(u64, 0), upload_meter.peek());

    // Mutate the canvas AFTER the stage (geometry AND handles): the latched
    // frame must still see the frozen staged bytes.
    canvas.begin();
    canvas.drawRect(1, 2, 3, 4, Color4.white);
    canvas.pipeline = .{ .id = 70 };
    canvas.vertex_buffer = .{ .id = 210 };
    canvas.index_buffer = .{ .id = 230 };
    try std.testing.expect(canvas.vertices.items.len != staged_verts);

    const UIVertex = @import("../ui.zig").UIVertex;
    const want_verts = try alloc.dupe(UIVertex, back.ui_vertices.items);
    defer alloc.free(want_verts);
    const want_idx = try alloc.dupe(u16, back.ui_indices.items);
    defer alloc.free(want_idx);

    // Consume WITHOUT rebuild: post-build mutation must not leak.
    finishForTest(&scene);

    try std.testing.expectEqual(@as(u64, 1), scene.uiPacketLatchedCount());
    try std.testing.expect(scene.ui_frame.has_capture);
    try std.testing.expectEqual(staged_verts, scene.ui_frame.vertices.items.len);
    try std.testing.expectEqual(staged_idx, scene.ui_frame.indices.items.len);
    try std.testing.expectEqualSlices(UIVertex, want_verts, scene.ui_frame.vertices.items[0..staged_verts]);
    try std.testing.expectEqualSlices(u16, want_idx, scene.ui_frame.indices.items[0..staged_idx]);
    try std.testing.expectEqual(@as(f32, 1920.0), scene.ui_frame.screen_w);
    try std.testing.expectEqual(@as(f32, 1080.0), scene.ui_frame.screen_h);
    // Staged handles win over the post-stage canvas mutation above.
    try std.testing.expectEqual(@as(u32, 7), scene.ui_frame.pipeline.id);
    try std.testing.expectEqual(@as(u32, 21), scene.ui_frame.vertex_buffer.id);
    try std.testing.expectEqual(@as(u32, 23), scene.ui_frame.index_buffer.id);
    // Headless: staged but not drawable, zero meter bytes.
    try std.testing.expect(scene.ui_frame.needs_upload);
    try std.testing.expect(!scene.ui_frame.gpu_ready);
    try std.testing.expectEqual(@as(u64, 0), upload_meter.peek());

    // Fresh build picks the mutated canvas up: newest wins, count advances.
    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(u64, 2), scene.uiPacketLatchedCount());
    try std.testing.expectEqual(canvas.vertices.items.len, scene.ui_frame.vertices.items.len);
    try std.testing.expect(canvas.vertices.items.len != staged_verts);
}

test "ui packet build always stages live canvas geometry" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
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

    // Every producer build stages the live canvas into the claimed slot.
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(u64, 1), scene.uiPacketLatchedCount());
    try std.testing.expect(scene.ui_frame.has_capture);
    try std.testing.expect(scene.ui_frame.canvas_present);
    try std.testing.expectEqual(scene.ui_canvas.?.vertices.items.len, scene.ui_frame.vertices.items.len);
    try std.testing.expectEqual(@as(f32, 640.0), scene.ui_frame.screen_w);
}

test "ui packet stage OOM fail-closes coherent-empty" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
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

    // Warm the build path first so retained capacity funds the build halves;
    // the failing allocator below then targets the UI slot copy specifically.
    scene.publishFrameSnapshot(1.0, 640, 480);
    try stageAndPrepareForTest(&scene);
    const warm_latched = scene.uiPacketLatchedCount();
    try std.testing.expectEqual(@as(u64, 1), warm_latched);
    try std.testing.expect(scene.ui_frame.has_capture);

    // Refuse the UI slot copy: the packet is marked invalid (never partial).
    scene.publishFrameSnapshot(1.0, 640, 480);
    var failing = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 0 });
    const real_alloc = scene.allocator;
    scene.allocator = failing.allocator();
    var oom = scene.tryClaimBuildSlot() orelse std.debug.panic("{s}", .{"oom claim saturated"});
    oom.build();
    oom.stageUi();
    oom.publish();
    scene.allocator = real_alloc;
    try std.testing.expect(!scene.draws.slotAt(scene.build_slot.load(.monotonic)).ui_packet.valid);

    // Coherent-empty: no partial geometry latched, counter unstamped.
    finishForTest(&scene);
    try std.testing.expectEqual(warm_latched, scene.uiPacketLatchedCount());
    try std.testing.expect(!scene.ui_frame.has_capture);
    try std.testing.expectEqual(@as(usize, 0), scene.ui_frame.vertices.items.len);

    // Recovery: a funded build stages and latches again.
    scene.publishFrameSnapshot(1.0, 640, 480);
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.ui_frame.has_capture);
    try std.testing.expectEqual(scene.ui_canvas.?.vertices.items.len, scene.ui_frame.vertices.items.len);
}
