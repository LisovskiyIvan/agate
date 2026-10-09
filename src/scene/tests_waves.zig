//! Staged frame tests (part 6): waves 26-39, slice6, auto-exposure. Split from scene/tests.zig.
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

test "wave26: 3-slot prepare rotation visits every slot, newest wins" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.binned_source.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    scene.enable_frustum_culling = false;
    scene.enable_occlusion_culling = false;
    scene.default_white_texture.view.id = 1;
    p7CpuShadowPass(&scene, alloc);

    var mesh = Mesh{
        .name = "rot_mesh",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
    };
    try scene.meshes.append(alloc, &mesh);
    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    // Four build/publish round-trips with no consumer lag: the rotation must
    // cycle 1, 2, 0, 1 (no stall, no slot reused before its turn), each
    // publish carries the building frame's identities, and moving the mesh
    // between prepares lands newest-wins in the next front.
    var fronts: [4]usize = undefined;
    var f: usize = 0;
    while (f < 4) : (f += 1) {
        mesh.position = Vec3.new(@floatFromInt(f * 10), 0, 0);
        scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);
        try buildForTest(&scene);
        const want = scene.build_slot.load(.monotonic);
        finishForTest(&scene);
        try std.testing.expect(scene.frame_prepared);
        try std.testing.expectEqual(want, scene.draws.front);
        fronts[f] = scene.draws.front;
        const d = scene.preparedDraws();
        try std.testing.expectEqual(scene.frame_id, d.frame_id);
        try std.testing.expectEqual(scene.retire_epoch, d.retire_epoch);
        try std.testing.expectEqual(@as(usize, 1), d.primary.items.items.len);
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(f * 10)), p7FindByMeshIndex(d.primary.items.items, 0).?.model.m[12], 1e-4);
    }
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 0, 1 }, &fronts);
    // After three prepares every slot holds a distinct published frame (no
    // slot was overwritten before its turn); the fourth reuses slot 1.
    try std.testing.expect(scene.draws.slots[0].frame_id != scene.draws.slots[1].frame_id);
    try std.testing.expect(scene.draws.slots[1].frame_id != scene.draws.slots[2].frame_id);
    try std.testing.expect(scene.draws.slots[0].frame_id != scene.draws.slots[2].frame_id);
}

test "wave26: staged records + UI packet latch correctly across all three slots" {
    const InstancedMesh = @import("../mesh.zig").InstancedMesh;
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
    defer scene.ui_frame.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "rot_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 4;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    var parent = Mesh{
        .name = "rot_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

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

    // Three full build+stage+latch frames: each build must land on a
    // different slot, each latch must consume exactly that frame's staged
    // records and UI packet (never a stale slot's).
    var build_slots: [3]usize = undefined;
    var frame: usize = 0;
    while (frame < 3) : (frame += 1) {
        mem[0].position = Vec3.new(@floatFromInt(frame * 10), 0, 0);
        const canvas = &scene.ui_canvas.?;
        canvas.begin();
        var r: usize = 0;
        while (r <= frame) : (r += 1) {
            canvas.drawRect(@floatFromInt(r * 10), 0, 10, 10, Color4.white);
        }
        const staged_verts = canvas.vertices.items.len;

        // Single staged claim per frame: publish then full build + UI stage
        // into the same claimed slot (build resets, so stageUi runs after).
        scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
        {
            var __uic = scene.tryClaimBuildSlot() orelse std.debug.panic("{s}", .{"stageUi: saturated"});
            __uic.build();
            __uic.stageUi();
            __uic.publish();
        }
        build_slots[frame] = scene.build_slot.load(.monotonic);
        finishForTest(&scene);

        try std.testing.expectEqual(scene.build_seq.load(.monotonic), scene.last_latched_seq.load(.monotonic));
        try std.testing.expectEqual(@as(u64, frame + 1), scene.uiPacketLatchedCount());
        try std.testing.expectEqual(scene.build_slot.load(.monotonic), scene.draws.front);
        try std.testing.expectEqual(scene.frame_id, scene.preparedDraws().frame_id);
        try std.testing.expectEqual(scene.retire_epoch, scene.preparedDraws().retire_epoch);
        // Staged records: this frame's instances, latched from the slot
        // (the latch mirrors; live meshes are committed by the NEXT build,
        // so live lags the mirror by exactly one frame).
        const mirror = scene.preparedDraws().staged_instances.items[0];
        try std.testing.expectEqual(@as(u32, 5), mirror.count);
        try std.testing.expectEqual(scene.frame_id, mirror.staged_frame);
        if (frame == 0) {
            try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
            try std.testing.expectEqual(std.math.maxInt(u64), parent.instance_render.staged_frame);
        } else {
            try std.testing.expectEqual(@as(u32, 5), parent.instance_render.count);
            try std.testing.expectEqual(scene.frame_id - 1, parent.instance_render.staged_frame);
        }
        // UI packet: this frame's staged bytes, not a stale slot's.
        try std.testing.expectEqual(staged_verts, scene.ui_frame.vertices.items.len);
        try std.testing.expect(scene.ui_frame.has_capture);
    }
    // The three builds rotated through three distinct slots (1, 2, 0).
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 0 }, &build_slots);
}

test "wave26: render pins/unpins the front; prepare rotates under a held pin" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.profiler.deinit();

    // Camera-less headless fixture: prepare publishes, render consumes via
    // the no-camera early return (no sg.* headless).
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    const front0 = scene.draws.front;

    // Render holds the pin only for the draw: no pins leak afterwards.
    scene.render();
    try std.testing.expect(!scene.frame_prepared);
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());
    try std.testing.expectEqual(front0, scene.draws.front);
    const consumed_id = scene.draws.slots[front0].frame_id;

    // A pinned presented frame is never a build target: the staged build
    // claims a different slot and leaves the pinned one untouched.
    try scene.draws.pin(front0);
    try stageAndPrepareForTest(&scene);
    const want = scene.draws.front;
    try std.testing.expect(want != front0);
    try std.testing.expectEqual(want, scene.draws.front);
    try std.testing.expectEqual(consumed_id, scene.draws.slots[front0].frame_id);
    try std.testing.expect(scene.draws.isPinned(front0));
    try scene.draws.unpin(front0);
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());

    // Memory census covers all three slots the same way (retained
    // capacities, not lengths): slot count is the rotation depth and the
    // byte total matches the live census helper exactly.
    const mem_snap = try scene.profiler.captureMemorySnapshot(&scene);
    try std.testing.expectEqual(@as(usize, 3), mem_snap.prepared_draws_slots);
    try std.testing.expectEqual(scene.draws.cpuBytes(), mem_snap.prepared_draws_cpu_bytes);
}

// ---- Wave 27: slot-owned frame snapshot. ----
//
// prepare/render/reuse/the UI latch read the consumed slot's staged
// `FrameDrawSlot.snapshot` — never the live `Scene.frame_snapshot` — so a
// game-side mutation after staging cannot tear the in-flight frame.

test "wave27: build stages the slot snapshot equal to the published generation" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.sky.enabled = true;
    scene.sky.exposure = 2.0;

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try buildForTest(&scene);

    // The claim slot stages the exact build generation (plain value copy).
    const staged = &scene.draws.slots[scene.build_slot.load(.monotonic)].snapshot;
    try std.testing.expectEqual(scene.build_snapshot.has_camera, staged.has_camera);
    try std.testing.expectEqual(scene.build_snapshot.screen_w, staged.screen_w);
    try std.testing.expectEqual(scene.build_snapshot.screen_h, staged.screen_h);
    try std.testing.expectEqual(@as(i32, 800), staged.screen_w);
    try std.testing.expectEqual(@as(i32, 600), staged.screen_h);
    try std.testing.expectEqual(scene.build_snapshot.sky_exposure, staged.sky_exposure);
    try std.testing.expectEqual(scene.build_snapshot.primary_cam.eye, staged.primary_cam.eye);
    try std.testing.expectEqual(scene.build_snapshot.camera_count, staged.camera_count);
    try std.testing.expectEqual(scene.build_snapshot.clear_color, staged.clear_color);
    // The live consumed copy is untouched by the build (update may overlap
    // render): still the zero fixture state.
    try std.testing.expect(!scene.frame_snapshot.has_camera);
    try std.testing.expectEqual(@as(i32, 0), scene.frame_snapshot.screen_w);
}

test "wave27: post-build live mutation does not change the prepared frame" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.sky.enabled = true;
    scene.sky.exposure = 2.0;

    // Generation B: publish + build (frozen).
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(i32, 800), scene.draws.slots[scene.build_slot.load(.monotonic)].snapshot.screen_w);

    // Live mutations AFTER the build: both working copies plus a newer
    // mailbox publication C (stays queued, like the B-vs-C latch test).
    scene.build_snapshot.sky_exposure = 9.0;
    scene.build_snapshot.screen_w = 111;
    scene.build_snapshot.screen_h = 111;
    scene.frame_snapshot.screen_w = 222;
    scene.sky.exposure = 5.0;
    scene.publishFrameSnapshot(16.0 / 9.0, 640, 480);

    // Latch: the prepared frame is exactly B — staged wins over every
    // post-build mutation. The compat mirror follows the staged copy.
    finishForTest(&scene);
    const front = scene.preparedDraws();
    try std.testing.expectEqual(@as(i32, 800), front.snapshot.screen_w);
    try std.testing.expectEqual(@as(i32, 600), front.snapshot.screen_h);
    try std.testing.expectEqual(@as(f32, 2.0), front.snapshot.sky_exposure);
    try std.testing.expectEqual(@as(i32, 800), scene.frame_snapshot.screen_w);
    try std.testing.expectEqual(@as(f32, 2.0), scene.frame_snapshot.sky_exposure);

    // C waited in the mailbox: the next staged build sees it.
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(i32, 640), scene.preparedDraws().snapshot.screen_w);
    try std.testing.expectEqual(@as(f32, 5.0), scene.preparedDraws().snapshot.sky_exposure);
}

test "wave27: staged build freezes the slot snapshot; render+reuse present it" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);

    // Camera-less headless fixture: dims latch before the no-camera
    // early-out; render consumes via the no-camera return (no sg.*).
    scene.publishFrameSnapshot(1.0, 640, 480);
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    const front0 = scene.draws.front;
    try std.testing.expectEqual(@as(i32, 640), scene.preparedDraws().snapshot.screen_w);
    try std.testing.expectEqual(@as(i32, 480), scene.preparedDraws().snapshot.screen_h);
    try std.testing.expect(!scene.preparedDraws().snapshot.has_camera);

    // Concurrent-update simulation: hammer the live working copy. Render
    // must still take the STAGED no-camera early return (a live read would
    // walk into the sg main-pass path headless).
    scene.frame_snapshot.has_camera = true;
    scene.frame_snapshot.screen_w = 999;
    scene.frame_snapshot.screen_h = 999;
    scene.render();
    try std.testing.expect(!scene.frame_prepared);
    try std.testing.expectEqual(front0, scene.draws.front);
    try std.testing.expectEqual(@as(i32, 640), scene.preparedDraws().snapshot.screen_w);

    // Reuse re-presents the same staged snapshot, untouched by the live
    // mutation above.
    scene.renderReuse();
    try std.testing.expectEqual(@as(u64, 1), scene.reuseStreak());
    try std.testing.expectEqual(front0, scene.draws.front);
    try std.testing.expectEqual(@as(i32, 640), scene.preparedDraws().snapshot.screen_w);
    try std.testing.expectEqual(@as(i32, 480), scene.preparedDraws().snapshot.screen_h);
    try std.testing.expect(!scene.preparedDraws().snapshot.has_camera);
}

test "wave27: UI latch reads the staged snapshot dims" {
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

    // Generation B through the staged claim path; the UI stage lands after
    // the build into the same claimed slot (the build reset would wipe an
    // earlier packet).
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    {
        var uic = scene.tryClaimBuildSlot() orelse std.debug.panic("{s}", .{"ui claim saturated"});
        uic.build();
        uic.stageUi();
        uic.publish();
    }
    // Post-build live mutation of the working copy: the latch must still
    // capture the staged dims.
    scene.build_snapshot.screen_w = 111;
    scene.build_snapshot.screen_h = 222;
    finishForTest(&scene);

    try std.testing.expectEqual(@as(u64, 1), scene.uiPacketLatchedCount());
    try std.testing.expectEqual(@as(f32, 800.0), scene.ui_frame.screen_w);
    try std.testing.expectEqual(@as(f32, 600.0), scene.ui_frame.screen_h);
    try std.testing.expectEqual(@as(i32, 800), scene.preparedDraws().snapshot.screen_w);
}

test "wave27: 3-slot rotation keeps per-slot snapshots distinct" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);

    // Camera-less: dims latch before the no-camera early-out, so the
    // rotation is observable without any GPU init.
    var fronts: [3]usize = undefined;
    const widths = [_]i32{ 111, 222, 333 };
    for (widths, 0..) |w, i| {
        scene.publishFrameSnapshot(1.0, w, w);
        try stageAndPrepareForTest(&scene);
        fronts[i] = scene.draws.front;
        try std.testing.expectEqual(w, scene.preparedDraws().snapshot.screen_w);
    }
    // Three prepares rotated through three distinct slots (1, 2, 0), each
    // retaining its own generation — no cross-slot copy, no wipe.
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 0 }, &fronts);
    for (fronts, widths) |f, w| {
        try std.testing.expectEqual(w, scene.draws.slots[f].snapshot.screen_w);
    }
}

// ---- Wave 29: concurrent-build claim flow (sequential proof; the phase
// mutex is still held — the lease-level stress test in frame_draws.zig
// proves slot-payload concurrency across threads). ----

test "wave29: claim build+stageUi+publish latches the claimed slot, counters sane" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.profiler.deinit();

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    var claim = scene.tryClaimBuildSlot().?;
    const slot = claim.slot;
    // The handoff is not committed until publish: filling alone moves
    // nothing global, and the slot is held (no longer the back index).
    try std.testing.expectEqual(@as(u64, 0), scene.build_seq.load(.monotonic));
    claim.build();
    try std.testing.expectEqual(@as(u64, 0), scene.build_seq.load(.monotonic));
    claim.stageUi(); // no canvas: stages absence (presence false, no geometry)
    try std.testing.expect(!scene.draws.slotAt(slot).ui_packet.canvas_present);
    claim.publish();
    try std.testing.expectEqual(@as(u64, 1), scene.build_seq.load(.monotonic));
    try std.testing.expectEqual(slot, scene.build_slot.load(.monotonic));
    try std.testing.expect(scene.build_slot.load(.monotonic) < scene.draws.slots.len);

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);

    try std.testing.expectEqual(scene.build_seq.load(.monotonic), scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(scene.build_slot.load(.monotonic), scene.draws.front);
    try std.testing.expectEqual(scene.frame_id, scene.preparedDraws().frame_id);
    // The staged absence packet latched presence (no geometry => latched count stays zero).
    try std.testing.expectEqual(@as(u64, 0), scene.uiPacketLatchedCount());
    // Counter sanity: a clean claim flow counts nothing.
    try std.testing.expectEqual(@as(u64, 0), scene.draws.saturation_skips);
    try std.testing.expectEqual(@as(u64, 0), scene.draws.publish_refusals);
    try std.testing.expectEqual(@as(u64, 0), scene.draws.pin_denials);
    try std.testing.expectEqual(@as(u64, 0), scene.draws.unpin_denials);
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());
}

test "wave29: cancelled claim commits nothing, rotation unaffected" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.profiler.deinit();

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    var dropped = scene.tryClaimBuildSlot().?;
    const dropped_slot = dropped.slot;
    dropped.build();
    dropped.cancel();
    // Nothing committed: no fresh build for prepare, rotation intact.
    try std.testing.expectEqual(@as(u64, 0), scene.build_seq.load(.monotonic));
    var funded = scene.tryClaimBuildSlot().?;
    // The dropped slot was released: the next claim reuses the same index.
    try std.testing.expectEqual(dropped_slot, funded.slot);
    funded.build();
    funded.publish();
    try std.testing.expectEqual(@as(u64, 1), scene.build_seq.load(.monotonic));
    try std.testing.expectEqual(funded.slot, scene.build_slot.load(.monotonic));

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try std.testing.expectEqual(@as(u64, 1), scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(scene.build_slot.load(.monotonic), scene.draws.front);
    try std.testing.expectEqual(@as(u64, 0), scene.draws.saturation_skips);
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());
}

test "wave38: concurrent_yield_ns publish keeps handoff semantics, default 0" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.profiler.deinit();

    // Default OFF: the phase-locked path never sets the knob.
    try std.testing.expectEqual(@as(u64, 0), scene.concurrent_yield_ns);
    // Enabled: claims still succeed and commit exactly one generation
    // each; the park fires only while a previous build is unconsumed
    // (backpressure) and holds no lease, so the latch below proceeds.
    scene.concurrent_yield_ns = 1_000;
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    var claim = scene.tryClaimBuildSlot().?;
    const slot = claim.slot;
    claim.build();
    claim.publish();
    try std.testing.expectEqual(@as(u64, 1), scene.build_seq.load(.monotonic));
    try std.testing.expectEqual(slot, scene.build_slot.load(.monotonic));
    // Second claim while the first build is still unlatched: backpressure
    // park runs first (tiny: 1µs), the claim itself is unaffected.
    var claim2 = scene.tryClaimBuildSlot().?;
    claim2.build();
    claim2.publish();
    try std.testing.expectEqual(@as(u64, 2), scene.build_seq.load(.monotonic));
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try std.testing.expectEqual(scene.build_seq.load(.monotonic), scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(scene.build_slot.load(.monotonic), scene.draws.front);
    try std.testing.expectEqual(@as(u64, 0), scene.draws.saturation_skips);
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());
}

test "wave29: build path never begins/completes retire epochs" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.profiler.deinit();

    // Camera-less headless fixture: prepare opens an epoch, the no-camera
    // render return closes it (same pairing the epoch tests pin).
    const cur0 = scene.gpu_retire.current();
    const done0 = scene.gpu_retire.lastCompleted();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    var claim = scene.tryClaimBuildSlot().?;
    claim.build();
    claim.stageUi();
    claim.publish();
    // The whole game-side claim flow leaves the epoch pairing untouched.
    try std.testing.expectEqual(cur0, scene.gpu_retire.current());
    try std.testing.expectEqual(done0, scene.gpu_retire.lastCompleted());

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try std.testing.expectEqual(cur0 + 1, scene.gpu_retire.current());
    scene.render();
    try std.testing.expectEqual(scene.gpu_retire.current(), scene.gpu_retire.lastCompleted());
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());
}

// ---- Wave 30: concurrent-build handoff edge (atomic seq words). ----
//
// A producer thread runs the adoption-edge claim flow (claim -> stageUi ->
// publish: the build commits the generation (`build_slot` then `build_seq`)
// release-stores `build_slot` then `build_seq`) while a consumer thread runs
// the latch half of the edge (acquire-reads `build_seq`,
// stamps `last_latched_seq`). No full build/prepare
// runs here: the remaining live touches are still phase-excluded (see the
// adoption checklist in scene/frame_draws.zig) — this proves exactly the
// atomic handoff: the consumer never observes a seq older than one already
// seen (no stale read after publish), and the join-time counters are exact
// (every publish latched, newest wins, no saturation drop uncongested).
test "wave30: concurrent claim/stageUi/publish vs latch — no stale seq, exact counters" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.profiler.deinit();

    const total_publishes: u64 = 5000;

    const Ctx = struct {
        scene: *Scene,
        total: u64,
        skipped: u64 = 0,
        done: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        // Consumer-side observations (consumer thread writes; the test
        // thread reads after join).
        max_build_seen: u64 = 0,
        max_ui_seen: u64 = 0,
        latches: u64 = 0,
    };

    const Producer = struct {
        fn run(c: *Ctx) void {
            var seq: u64 = 1;
            while (seq <= c.total) {
                var claim = while (c.scene.tryClaimBuildSlot()) |cl| break cl else {
                    c.skipped += 1;
                    std.atomic.spinLoopHint();
                    continue;
                };
                // Full staged claim: build then stageUi (absence packet, no
                // canvas) before publish — UI rides the owned full slot.
                claim.build();
                claim.stageUi();
                // Release edge: `build_slot` then `build_seq`.
                claim.publish();
                seq += 1;
            }
            c.done.store(true, .release);
        }
    };

    const Consumer = struct {
        fn latchOnce(c: *Ctx) void {
            // Acquire: pairs with the publish release-stores — a fresh
            // generation implies the staged payload is visible. Track the
            // owned full-slot packet generation (h.seq) with a fresh build
            // counter, and prove stage-byte happens-before by reading the
            // staged slot payload itself, not just the counter word.
            const b = c.scene.build_seq.load(.acquire);
            // Never stale: generations are strictly increasing by
            // single-producer construction, so an older read after a newer
            // one would be a torn handoff.
            if (b < c.max_build_seen) unreachable;
            if (b > c.max_build_seen) c.max_build_seen = b;
            // Context-side stamp only (the producer never touches this word).
            c.scene.last_latched_seq.store(b, .monotonic);
            const s = c.scene.build_slot.load(.acquire);
            const pkt = c.scene.draws.slotAtConst(s);
            if (pkt.build_seq != b) return; // unpublished slot yet; counter-only latch
            if (b < c.max_ui_seen) unreachable;
            if (b > c.max_ui_seen) c.max_ui_seen = b;
            c.latches += 1;
        }

        fn run(c: *Ctx) void {
            while (!c.done.load(.acquire)) latchOnce(c);
            // The `done` acquire synchronizes with the producer's release
            // after its last publish, so this final latch deterministically
            // observes the tail generation.
            latchOnce(c);
        }
    };

    var ctx = Ctx{ .scene = &scene, .total = total_publishes };
    const prod = try std.Thread.spawn(.{}, Producer.run, .{&ctx});
    const cons = try std.Thread.spawn(.{}, Consumer.run, .{&ctx});
    prod.join();
    cons.join();

    // Uncongested rotation (no consumer pins held): the producer always had
    // a free slot — zero skips, every publish landed exactly once.
    try std.testing.expectEqual(@as(u64, 0), ctx.skipped);
    try std.testing.expectEqual(@as(u64, 0), scene.draws.saturation_skips);
    try std.testing.expect(ctx.latches > 0);
    // Exact counter arithmetic against the join-time totals: N publishes
    // committed generations 1..N on both edges, and the consumer's last
    // latch stamped exactly the tail (nothing lost, nothing duplicated —
    // the seqs ARE the counters, no separate tally to drift).
    try std.testing.expectEqual(total_publishes, scene.build_seq.load(.monotonic));
    try std.testing.expectEqual(total_publishes, scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(total_publishes, ctx.max_build_seen);
    try std.testing.expectEqual(total_publishes, ctx.max_ui_seen);
    // The published slot index rode the same edge: it names a real slot and
    // the tail handoff is still held (released, never wedged).
    try std.testing.expect(scene.build_slot.load(.monotonic) < scene.draws.slots.len);
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());
}

// ---- Wave 32: particle/physics freeze-then-latch (slot payload). ----

test "wave32: adopted build freezes particle+physics into the slot; prepare latches the slot copy" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.profiler.deinit();
    defer scene.meshes.deinit(alloc);
    defer wave32FreeTestSystems(&scene);

    const ps = try wave32PushTestSystem(&scene, 4);
    ps.active_count = 2;
    ps.instance_buffer = .{ .id = 11 };

    // Standalone body mesh (not registered: the build stages zero scene
    // meshes while the debug capture still observes the world).
    const m = try alloc.create(Mesh);
    m.* = @import("../testing.zig").testMesh("box");
    defer alloc.destroy(m);
    _ = try scene.createRigidBody(m, .box, 1.0);
    scene.physics.show_debug = true;
    defer scene.physics.deinit(alloc);

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(u64, 1), scene.build_seq.load(.monotonic));
    const slot = scene.build_slot.load(.monotonic);
    const frozen = scene.draws.slotAtConst(slot);
    // The claimed slot froze both captures by value.
    try std.testing.expectEqual(@as(usize, 1), frozen.particle_draws.items.len);
    try std.testing.expectEqual(@as(usize, 2), frozen.particle_draws.items[0].active_count);
    try std.testing.expectEqual(@as(u32, 11), frozen.particle_draws.items[0].instance_buffer.id);
    try std.testing.expect(frozen.physics_visible);
    try std.testing.expectEqual(@as(usize, 12), frozen.physics_lines.items.len);
    const x0 = frozen.physics_lines.items[0].a.x;

    // Mutate every live field past recognition AFTER the build: the frozen
    // slot copies (and the shared build frames) stay at build time.
    ps.active_count = 1;
    ps.instance_buffer = .{ .id = 99 };
    m.position = Vec3.new(5, 0, 0);
    scene.physics.show_debug = false;
    try std.testing.expectEqual(@as(usize, 2), frozen.particle_draws.items[0].active_count);
    try std.testing.expectEqual(@as(u32, 11), frozen.particle_draws.items[0].instance_buffer.id);
    try std.testing.expect(frozen.physics_visible);
    try std.testing.expectApproxEqAbs(x0, frozen.physics_lines.items[0].a.x, 1e-4);

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try std.testing.expectEqual(scene.build_seq.load(.monotonic), scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(slot, scene.draws.front);

    // The latch published the frozen generation: identical to the shared
    // build frames (sequential bit-identical), immune to the live mutation.
    try std.testing.expectEqual(@as(usize, 1), scene.particles.frame.items.len);
    try std.testing.expectEqual(@as(usize, 2), scene.particles.frame.items[0].active_count);
    try std.testing.expectEqual(@as(u32, 11), scene.particles.frame.items[0].instance_buffer.id);
    try std.testing.expectEqual(scene.particles.build_frame.items[0].active_count, scene.particles.frame.items[0].active_count);
    try std.testing.expectEqual(
        scene.particles.build_frame.items[0].instance_buffer.id,
        scene.particles.frame.items[0].instance_buffer.id,
    );
    try std.testing.expect(scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 12), scene.physics.prepared_lines.items.len);
    try std.testing.expectApproxEqAbs(x0, scene.physics.prepared_lines.items[0].a.x, 1e-4);
    // Both layer generations were consumed by the staged latch.
    try std.testing.expectEqual(scene.particles.build_seq.load(.acquire), scene.particles.latched_seq);
    try std.testing.expectEqual(scene.physics.build_seq.load(.acquire), scene.physics.latched_seq);
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());
}

test "wave32: staged build freezes particle+physics slot copies" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.profiler.deinit();
    defer scene.meshes.deinit(alloc);
    defer wave32FreeTestSystems(&scene);

    const ps = try wave32PushTestSystem(&scene, 4);
    ps.active_count = 2;
    ps.instance_buffer = .{ .id = 11 };

    const m = try alloc.create(Mesh);
    m.* = @import("../testing.zig").testMesh("box");
    defer alloc.destroy(m);
    _ = try scene.createRigidBody(m, .box, 1.0);
    scene.physics.show_debug = true;
    defer scene.physics.deinit(alloc);

    // Staged build freezes the particle/physics copies into the claimed
    // slot; the finish latches exactly those slot copies (never live).
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(u64, 1), scene.build_seq.load(.monotonic));
    try std.testing.expectEqual(scene.build_seq.load(.monotonic), scene.last_latched_seq.load(.monotonic));
    const front = scene.draws.slotAtConst(scene.draws.front);
    try std.testing.expectEqual(@as(usize, 1), front.particle_draws.items.len);
    try std.testing.expectEqual(@as(usize, 2), front.particle_draws.items[0].active_count);
    try std.testing.expect(front.physics_visible);
    try std.testing.expectEqual(@as(usize, 12), front.physics_lines.items.len);
    // Prepared captures mirror the frozen slot copies.
    try std.testing.expectEqual(@as(usize, 1), scene.particles.frame.items.len);
    try std.testing.expectEqual(@as(usize, 2), scene.particles.frame.items[0].active_count);
    try std.testing.expect(scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 12), scene.physics.prepared_lines.items.len);
}

// Concurrent stress: a producer thread runs the adopted claim flow
// (claim -> build, freezing particle/physics into the claimed slot ->
// publish) while the test thread runs the prepare-side slot latch
// (stable-pair claim -> validate the FROZEN payload -> latchSlot ->
// publish). The per-generation canary: particle `active_count`/`instance_
// buffer.id` encode the generation parity/value, physics visibility encodes
// the same parity — any torn cross-generation mix (the wave-32 hazard on
// the old shared stores) fails the coherence check deterministically:
// consecutive generations always differ in parity. Latest-wins drops are
// legal (a moved-on handoff is skipped, counted) but latched generations
// are strictly increasing with the exact tail; an uncongested rotation
// never starves the producer (zero claimBack skips).
test "wave32: concurrent claim/build/publish vs slot-latch — no torn records, exact tail" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.profiler.deinit();
    defer scene.meshes.deinit(alloc);
    defer wave32FreeTestSystems(&scene);

    const ps = try wave32PushTestSystem(&scene, 4);
    const m = try alloc.create(Mesh);
    m.* = @import("../testing.zig").testMesh("box");
    defer alloc.destroy(m);
    _ = try scene.createRigidBody(m, .box, 1.0);
    defer scene.physics.deinit(alloc);

    const total_gens: u64 = 5000;

    const Ctx = struct {
        scene: *Scene,
        sys: *ParticleSystem,
        total: u64,
        skipped: u64 = 0,
        /// `claimSlot` refusals (`SlotBusy` — the handoff slot is mid-build):
        /// each one bumps `saturation_skips` exactly once, counted here.
        busy_misses: u64 = 0,
        /// Stable-pair races (a newer publish landed between our handoff
        /// load and our claim): counted latest-wins drops that bump NO
        /// global counter (`cancelClaim` is silent by design).
        moved_on: u64 = 0,
        latched: u64 = 0,
        max_seen: u64 = 0,
        done: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

        /// The no-torn-record assertion, applied to the frozen slot payload
        /// BEFORE the latch consumes it and to the latched copies after:
        /// every field of the generation decodes to the same `gen`.
        fn checkCoherent(draw_count: usize, draw_id: u32, n_lines: usize, visible: bool, gen: u64) void {
            // Particle canary: count (gen%2)+1, id gen.
            if (draw_count != @as(usize, @intCast((gen % 2) + 1))) unreachable;
            if (draw_id != @as(u32, @intCast(gen))) unreachable;
            // Physics canary: visible on even gens, 12 box lines or empty.
            if (visible != (gen % 2 == 0)) unreachable;
            if (n_lines != (if (visible) @as(usize, 12) else @as(usize, 0))) unreachable;
            // Cross-payload pairing: count==1 exactly on visible gens.
            if ((draw_count == 1) != visible) unreachable;
        }
    };

    const Producer = struct {
        fn run(c: *Ctx) void {
            var gen: u64 = 1;
            while (gen <= c.total) {
                // Encode the generation into live state (producer-only
                // writes; the consumer never reads live state, only slots).
                c.sys.active_count = @as(usize, @intCast((gen % 2) + 1));
                c.sys.instance_buffer = .{ .id = @as(u32, @intCast(gen)) };
                c.scene.physics.show_debug = (gen % 2 == 0);
                var claim = while (c.scene.tryClaimBuildSlot()) |cl| break cl else {
                    c.skipped += 1;
                    std.atomic.spinLoopHint();
                    continue;
                };
                claim.build();
                claim.publish();
                gen += 1;
            }
            c.done.store(true, .release);
        }
    };

    var ctx = Ctx{ .scene = &scene, .sys = ps, .total = total_gens };
    const prod = try std.Thread.spawn(.{}, Producer.run, .{&ctx});

    // Consumer = the prepare-side slot latch (test thread): only consume
    // when the handoff pair is stable across our claim — a moved-on
    // generation is a counted latest-wins miss, never a stale latch. The
    // consumed slot is released with `cancelClaim`, never published: the
    // front flip stays context-owned under exclusion in the real flow, and
    // the game-side build core resolves its commit slot through the lease
    // (`frontIndex`) — flipping it here would still move the commit target
    // outside the handoff edge under test. The handoff edge under test
    // (claim/build/publish vs claimSlot/validate/latch) is fully exercised
    // without the flip.
    while (!ctx.done.load(.acquire) or scene.last_latched_seq.load(.monotonic) != total_gens) {
        const b = scene.build_seq.load(.acquire);
        if (b == 0) {
            std.atomic.spinLoopHint();
            continue;
        }
        if (b == scene.last_latched_seq.load(.monotonic)) {
            std.atomic.spinLoopHint();
            continue;
        }
        if (b < ctx.max_seen) unreachable; // stale read after publish
        const s = scene.build_slot.load(.acquire);
        scene.draws.claimSlot(s) catch {
            ctx.busy_misses += 1;
            std.atomic.spinLoopHint();
            continue;
        };
        if (scene.build_seq.load(.acquire) != b or scene.build_slot.load(.acquire) != s) {
            scene.draws.cancelClaim(s) catch {};
            ctx.moved_on += 1;
            std.atomic.spinLoopHint();
            continue;
        }
        // The claim is held and the pair is stable: no producer can be
        // writing this slot — validate the FROZEN record before consuming.
        const back = scene.draws.slotAt(s);
        if (back.particle_draws.items.len != 1) unreachable;
        const frozen_draw = back.particle_draws.items[0];
        Ctx.checkCoherent(
            frozen_draw.active_count,
            frozen_draw.instance_buffer.id,
            back.physics_lines.items.len,
            back.physics_visible,
            b,
        );
        scene.particles.latchSlotFrame(alloc, back.particle_draws.items);
        scene.physics.latchSlotDebug(alloc, back.physics_lines.items, back.physics_visible);
        // The latched copies carry the same generation, whole.
        const got = scene.particles.frame.items;
        if (got.len != 1) unreachable;
        Ctx.checkCoherent(
            got[0].active_count,
            got[0].instance_buffer.id,
            scene.physics.prepared_lines.items.len,
            scene.physics.prepared_visible,
            b,
        );
        // Release the consumed slot back to the rotation (never publish:
        // the front flip stays context-owned — see the loop comment). The
        // release cannot fail: the claim is ours and unpinned by
        // construction (no pins are used in this test).
        scene.draws.cancelClaim(s) catch unreachable;
        scene.last_latched_seq.store(b, .monotonic);
        ctx.max_seen = b;
        ctx.latched += 1;
    }
    prod.join();

    // Contract accounting: the producer never starved (3 slots always leave
    // a free one with at most one consumer claim held), every global
    // saturation count is exactly one of our counted paths (producer
    // null-claims plus consumer SlotBusy refusals — the silent moved-on
    // drops bump nothing by design), generations only moved forward, and
    // the tail generation landed whole.
    try std.testing.expectEqual(@as(u64, 0), ctx.skipped);
    try std.testing.expectEqual(ctx.skipped + ctx.busy_misses, scene.draws.saturation_skips);
    try std.testing.expectEqual(@as(u64, 0), scene.draws.publish_refusals);
    try std.testing.expectEqual(@as(u64, 0), scene.draws.pin_denials);
    try std.testing.expect(ctx.latched > 0);
    try std.testing.expectEqual(total_gens, scene.build_seq.load(.monotonic));
    try std.testing.expectEqual(total_gens, scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(total_gens, ctx.max_seen);
    // Tail payload is exactly the last generation (never a stale mix).
    try std.testing.expectEqual(@as(usize, 1), scene.particles.frame.items.len);
    const tail_id: u32 = @intCast(total_gens);
    try std.testing.expectEqual(tail_id, scene.particles.frame.items[0].instance_buffer.id);
    const tail_count: usize = @intCast((total_gens % 2) + 1);
    try std.testing.expectEqual(tail_count, scene.particles.frame.items[0].active_count);
    try std.testing.expectEqual(total_gens % 2 == 0, scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());
}
// discipline, retire-on-remove. Headless: no sg.* below (buffers stay
// deferred without a context; the main test thread is context-marked by the
// earlier gpu_thread tests, so the flush paths take their real branches).

// ---- Wave 31: the staged finish holds its working slot as a lease claim. ----
//
// the staged begin resolves its slot via the lease claim,
// `claimSlot(build_slot)` (latch) at the top and releases it at every exit
// (`tryPublish` on success, `cancelClaim`/early return on contention). These
// tests prove the three properties that motivated the slice: prepares make
// progress under a concurrent pinned front, every exit pairs its claim
// (no leaked WRITING/pins, contention degrades to counted skips), and a
// concurrent game claim/stageUi can never target prepare's slot.

test "wave31: prepare publishes under a concurrent pinned front (stress), no leaks" {
    const alloc = std.testing.allocator;
    const LeaseError = @import("frame_draws.zig").LeaseError;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.profiler.deinit();

    const total: usize = 500;

    const Ctx = struct {
        scene: *Scene,
        stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        reads: u64 = 0,
    };

    const Pinner = struct {
        fn run(c: *Ctx) void {
            // Consumer-side only: frontIndex/pin/pinned-payload-read/unpin.
            // A pinned slot is never written by prepare (it builds only its
            // claimed slot), so these reads need no further locking; every
            // index word goes through the lease mutex.
            var held: ?usize = null;
            while (!c.stop.load(.acquire)) {
                if (held) |h| {
                    c.scene.draws.unpin(h) catch unreachable;
                    held = null;
                }
                const f = c.scene.draws.frontIndex();
                c.scene.draws.pin(f) catch |e| switch (e) {
                    LeaseError.AlreadyPinned, LeaseError.SlotBusy => continue,
                    else => unreachable,
                };
                held = f;
                // No-overwrite proof: the pinned pair must be stable for the
                // whole hold (prepare stamps both words before publish and
                // never rewrites a pinned slot afterwards).
                const s = c.scene.draws.slotAtConst(f);
                const fid0 = s.frame_id;
                const ep0 = s.retire_epoch;
                var spins: usize = 0;
                while (spins < 100) : (spins += 1) {
                    std.atomic.spinLoopHint();
                    const s2 = c.scene.draws.slotAtConst(f);
                    if (s2.frame_id != fid0 or s2.retire_epoch != ep0) unreachable;
                    c.reads += 1;
                }
            }
            if (held) |h| c.scene.draws.unpin(h) catch unreachable;
        }
    };

    const sat0 = scene.draws.saturation_skips;
    var ctx = Ctx{ .scene = &scene };
    const t = try std.Thread.spawn(.{}, Pinner.run, .{&ctx});
    var ok: usize = 0;
    var i: usize = 0;
    while (i < total) : (i += 1) {
        // Per-call success is the front flip (a contended prepare would
        // keep a stale `frame_prepared` from an earlier pending frame, so
        // the flag alone cannot tell them apart — the flip can).
        const f0 = scene.draws.front;
        try stageAndPrepareForTest(&scene);
        if (scene.draws.front != f0) ok += 1;
        try std.testing.expect(scene.frame_prepared);
    }
    ctx.stop.store(true, .release);
    t.join();

    // At most one pin held at a time: the 3-slot rotation always has a free
    // slot, so every prepare succeeds with zero saturation and zero
    // refusals, and the pinned presents were never overwritten (the pinner
    // would have hit `unreachable` on any rewrite).
    try std.testing.expectEqual(total, ok);
    try std.testing.expectEqual(sat0, scene.draws.saturation_skips);
    try std.testing.expectEqual(@as(u64, 0), scene.draws.publish_refusals);
    try std.testing.expect(ctx.reads > 0);
    try std.testing.expect(scene.hasConsumableFrame());
    for (0..scene.draws.slots.len) |idx| {
        try std.testing.expect(scene.draws.slots[idx].frame_id != 0);
        try std.testing.expect(!scene.draws.writing[idx]);
        try std.testing.expect(!scene.draws.pinned[idx]);
    }
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());
}

test "wave31: prepare claim accounting — no leaks across exits incl. early returns" {
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

    const expectClean = struct {
        fn run(s: *Scene) !void {
            try std.testing.expectEqual(@as(usize, 0), s.draws.pinsHeld());
            for (0..s.draws.slots.len) |idx| {
                try std.testing.expect(!s.draws.writing[idx]);
                try std.testing.expect(!s.draws.pinned[idx]);
            }
        }
    }.run;

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    // Baseline: the staged begin/finish exits pair every claim.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try buildForTest(&scene);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(scene.build_seq.load(.monotonic), scene.last_latched_seq.load(.monotonic));
    try expectClean(&scene);

    // (a) Latch-path contention: a pending build whose handoff slot is held
    // for writing (a concurrent game reclaim mid-fill for the NEXT build)
    // degrades to a counted skip — nothing consumed, the build stays fresh,
    // the slot payload is untouched, no side effect ran (the claim sits
    // before the retire/stats/frame prologue).
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try buildForTest(&scene);
    const pending_slot = scene.build_slot.load(.monotonic);
    const pending_seq = scene.build_seq.load(.monotonic);
    const latched_before = scene.last_latched_seq.load(.monotonic);
    try std.testing.expect(pending_seq != latched_before);
    const snap_w = scene.draws.slots[pending_slot].snapshot.screen_w;
    try scene.draws.claimSlot(pending_slot);
    const sat0 = scene.draws.saturation_skips;
    const front0 = scene.draws.front;
    const fid0 = scene.frame_id;
    // Contended begin degrades to a counted null skip — nothing consumed,
    // the build stays fresh, no side effect ran.
    try std.testing.expect(scene.beginStagedPrepare() == null);
    // Skip discards nothing: the earlier pending frame stays consumable.
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(front0, scene.draws.front);
    try std.testing.expectEqual(latched_before, scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(pending_seq, scene.build_seq.load(.monotonic));
    try std.testing.expectEqual(sat0 + 1, scene.draws.saturation_skips);
    try std.testing.expectEqual(fid0, scene.frame_id);
    try std.testing.expectEqual(snap_w, scene.draws.slots[pending_slot].snapshot.screen_w);
    // Release and retry: the still-pending build latches exactly once
    // (no rebuild — the fresh generation must not be superseded).
    try scene.draws.cancelClaim(pending_slot);
    finishForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(pending_slot, scene.draws.front);
    try std.testing.expectEqual(pending_seq, scene.last_latched_seq.load(.monotonic));
    try expectClean(&scene);

    // (b) Latch-path pinned handoff: a presented frame is never overwritten —
    // PinnedSlot refusal, counted, build stays fresh; unpin restores.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try buildForTest(&scene);
    const pinned_slot = scene.build_slot.load(.monotonic);
    const pinned_seq = scene.build_seq.load(.monotonic);
    try std.testing.expect(pinned_seq != scene.last_latched_seq.load(.monotonic));
    try scene.draws.pin(pinned_slot);
    const ref0 = scene.draws.publish_refusals;
    // Pinned handoff refuses the staged begin (counted); the build stays
    // fresh and nothing is consumed.
    try std.testing.expect(scene.beginStagedPrepare() == null);
    // Skip discards nothing: the earlier pending frame stays consumable.
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(ref0 + 1, scene.draws.publish_refusals);
    try std.testing.expectEqual(pinned_seq, scene.build_seq.load(.monotonic));
    try std.testing.expect(pinned_seq != scene.last_latched_seq.load(.monotonic));
    try scene.draws.unpin(pinned_slot);
    // Unpin restores: the still-pending build latches exactly (no rebuild).
    finishForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(pinned_slot, scene.draws.front);
    try expectClean(&scene);

    // (c) Fallback saturation: every non-front slot held degrades to a
    // counted skip, never a wedge; releasing restores the rotation.
    // Saturated build refuses (false, counted) and the staged begin stays
    // null: the earlier pending frame stays consumable.
    try scene.draws.pin(scene.draws.front);
    const h1 = scene.draws.claimBack().?;
    const h2 = scene.draws.claimBack().?;
    try std.testing.expect(scene.draws.claimBack() == null);
    const sat1 = scene.draws.saturation_skips;
    const front1 = scene.draws.front;
    try std.testing.expect(!scene.buildPreparedFrame());
    try std.testing.expect(scene.beginStagedPrepare() == null);
    // Skip discards nothing: the earlier pending frame stays consumable.
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(front1, scene.draws.front);
    try std.testing.expectEqual(sat1 + 1, scene.draws.saturation_skips);
    try scene.draws.cancelClaim(h1);
    try scene.draws.cancelClaim(h2);
    try scene.draws.unpin(front1);
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    try expectClean(&scene);
}

test "wave31: skip with nothing pending leaves nothing consumable; render drops the present" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.profiler.deinit();

    // Fresh scene: saturate every non-front slot. The staged build
    // refuses (false, counted) and the staged begin stays null. With no
    // earlier pending frame the skip keeps `frame_prepared` false and the
    // front untouched — and nothing is consumable yet.
    try scene.draws.pin(scene.draws.front);
    const g1 = scene.draws.claimBack().?;
    const g2 = scene.draws.claimBack().?;
    const sat0 = scene.draws.saturation_skips;
    try std.testing.expect(!scene.buildPreparedFrame());
    try std.testing.expect(scene.beginStagedPrepare() == null);
    try std.testing.expect(!scene.frame_prepared);
    try std.testing.expectEqual(@as(usize, 0), scene.draws.front);
    try std.testing.expectEqual(sat0 + 1, scene.draws.saturation_skips);
    try std.testing.expect(!scene.hasConsumableFrame());

    // `render` on the skipped state drops the present (no staged frame)
    // guard (no pin, no epoch, no stats record) instead of mislabeling the
    // stale front — then the rotation recovers cleanly once released.
    scene.render();
    try std.testing.expect(!scene.frame_prepared);
    try std.testing.expectEqual(@as(usize, 0), scene.draws.front);
    // The saturation pin is still held (released below): the guard path
    // must neither pin nor unpin anything, so the count is unchanged at 1.
    try std.testing.expectEqual(@as(usize, 1), scene.draws.pinsHeld());
    try scene.draws.cancelClaim(g1);
    try scene.draws.cancelClaim(g2);
    try scene.draws.unpin(0);
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expect(scene.hasConsumableFrame());
    // The pending frame consumes through the camera-less render path.
    scene.render();
    try std.testing.expect(!scene.frame_prepared);
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());
}

test "wave31: concurrent game claim can never target prepare's slot (steering + refusal)" {
    const alloc = std.testing.allocator;
    const LeaseError = @import("frame_draws.zig").LeaseError;
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
    var regular = Mesh{
        .name = "w31_reg",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.zero,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    try scene.meshes.append(alloc, &regular);

    // Prime one staged frame so the rotation is warm.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try stageAndPrepareForTest(&scene);
    const front0 = scene.draws.front;

    // A claim taken exactly the way the staged begin takes it.
    const held = scene.draws.claimBack().?;
    try std.testing.expect(held != front0);

    // The game-side claim steers to the OTHER free slot — never the held
    // one — then runs the full adopted flow (build + stageUi after the
    // build, which resets + publish).
    var game = scene.tryClaimBuildSlot().?;
    try std.testing.expect(game.slot != held);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    game.build();
    game.stageUi();
    const game_slot = game.slot;
    game.publish();
    try std.testing.expectEqual(game_slot, scene.build_slot.load(.monotonic));

    // The held slot is untouched by all of the above: no reset, no UI bytes.
    try std.testing.expectEqual(@as(u64, 0), scene.draws.slots[held].frame_id);
    try std.testing.expectEqual(@as(usize, 0), scene.draws.slots[held].ui_vertices.items.len);
    try std.testing.expect(!scene.draws.slots[held].ui_packet.valid);

    // A consumer pin on the prepare-held slot is refused and counted
    // (SlotBusy): the presenting side can never reclaim it mid-prepare.
    const den0 = scene.draws.pin_denials;
    try std.testing.expectError(LeaseError.SlotBusy, scene.draws.pin(held));
    try std.testing.expectEqual(den0 + 1, scene.draws.pin_denials);

    // A second full claim steers away too: claimBack skips WRITING slots,
    // so it never targets the held one either.
    try std.testing.expect(scene.build_slot.load(.monotonic) != held);
    {
        var __uic = scene.tryClaimBuildSlot() orelse std.debug.panic("{s}", .{"stageUi: saturated"});
        __uic.build();
        __uic.stageUi();
        __uic.publish();
    }
    try std.testing.expect(!scene.draws.slots[held].ui_packet.valid);

    // Release the simulated prepare hold, then run the real prepare: it
    // latches exactly the published handoff slot and pairs every claim.
    try scene.draws.cancelClaim(held);
    finishForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(game_slot, scene.draws.front);
    try std.testing.expectEqual(scene.build_seq.load(.monotonic), scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());
    for (0..scene.draws.slots.len) |idx| try std.testing.expect(!scene.draws.writing[idx]);
}

test "wave31: concurrent stageUi/publish vs real prepare — exact skip counters, no tears" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.profiler.deinit();

    const total_claims: u64 = 2000;
    const total_prepares: usize = 2000;

    const Ctx = struct {
        scene: *Scene,
        claims: u64,
        skipped: u64 = 0, // game-side null-claims (saturation)
        published: u64 = 0,
        cancelled: u64 = 0,
        done: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    };

    const Producer = struct {
        fn run(c: *Ctx) void {
            // Game-side concurrent flow (full staged claims: reserve, build,
            // stage, hand off — or cancel every 16th reservation to
            // exercise `cancelClaim` under prepare pressure). The fixture
            // owns no live content (no meshes, no canvas, no systems), so
            // the build's live reads race nothing the prepare latch
            // mutates; the handoff meets under the lease mutex and the
            // seq words are the release edge.
            var n: u64 = 0;
            while (n < c.claims) {
                // A null-claim is a skip, not a reservation: it must not
                // consume the claim budget below.
                var claim = c.scene.tryClaimBuildSlot() orelse {
                    c.skipped += 1;
                    std.atomic.spinLoopHint();
                    continue;
                };
                if (n % 16 == 7) {
                    claim.cancel();
                    c.cancelled += 1;
                } else {
                    claim.build();
                    claim.stageUi();
                    claim.publish();
                    c.published += 1;
                }
                n += 1;
            }
            c.done.store(true, .release);
        }
    };

    const sat0 = scene.draws.saturation_skips;
    var ctx = Ctx{ .scene = &scene, .claims = total_claims };
    const prod = try std.Thread.spawn(.{}, Producer.run, .{&ctx});
    var prepared_ok: usize = 0;
    var prepare_skips: u64 = 0;
    var i: usize = 0;
    while (i < total_prepares) {
        // Per-iteration success is the front flip. A null begin is either
        // counted contention (bumped the skip counter: consume the
        // iteration) or a no-fresh-build miss (uncounted: retry without
        // consuming an iteration, so the exact equation below only ever
        // sees counted outcomes). A contended skip keeps the earlier
        // pending frame (discards nothing), so the flag alone cannot count
        // skips — while every genuine finish flips to a new front.
        const f0 = scene.draws.front;
        const sat_b = scene.draws.saturation_skips;
        if (scene.beginStagedPrepare()) |claim| {
            scene.finishStagedPrepare(claim);
            i += 1;
            // Every genuine finish flips to a slot that was not the
            // front (claims never target the presented slot).
            try std.testing.expect(scene.draws.front != f0);
            prepared_ok += 1;
        } else if (scene.draws.saturation_skips != sat_b) {
            i += 1;
            prepare_skips += 1;
        } else if (ctx.done.load(.acquire) and
            scene.build_seq.load(.acquire) == scene.last_latched_seq.load(.monotonic))
        {
            // Producer joined-out with nothing fresh pending: no future
            // iteration could flip or count, so stop instead of spinning.
            break;
        } else {
            std.atomic.spinLoopHint();
        }
    }
    prod.join();
    // Drain: consume any build published after the last prepare. No
    // contention is left (the producer is joined), so each drain finish
    // flips — and no rebuild runs, so the published generation count is
    // untouched.
    var drain: usize = 0;
    while (scene.last_latched_seq.load(.monotonic) != scene.build_seq.load(.monotonic) and drain < 10) : (drain += 1) {
        const f0 = scene.draws.front;
        finishForTest(&scene);
        try std.testing.expect(scene.draws.front != f0);
    }
    try std.testing.expectEqual(scene.build_seq.load(.monotonic), scene.last_latched_seq.load(.monotonic));

    // Exact skip accounting: every game-side null-claim and every contended
    // begin bumps `saturation_skips` exactly once — the sum must match,
    // proving no silent interference in either direction (a missed release
    // would wedge the drain or leak WRITING below instead).
    try std.testing.expectEqual(ctx.skipped + prepare_skips, scene.draws.saturation_skips - sat0);
    // Every non-cancelled reservation published exactly once (the seqs ARE
    // the counters: one stage + one generation per publish, cancels commit
    // nothing).
    try std.testing.expectEqual(ctx.published + ctx.cancelled, total_claims);
    try std.testing.expectEqual(ctx.published, scene.build_seq.load(.monotonic));
    // No pin/refusal activity anywhere in this flow (nothing is presented
    // mid-loop, nothing publishes onto a pin).
    try std.testing.expectEqual(@as(u64, 0), scene.draws.publish_refusals);
    try std.testing.expectEqual(@as(u64, 0), scene.draws.pin_denials);
    // Clean teardown: no pins held, nothing left WRITING, consumable front.
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());
    for (0..scene.draws.slots.len) |idx| try std.testing.expect(!scene.draws.writing[idx]);
    try std.testing.expect(scene.hasConsumableFrame());
}

// ---- Wave 31 (second slice): build_stats as a slot payload. ----
//
// The game-side queue build writes its counters directly into the claimed
// slot; the staged finish merges that exact slot payload into context stats.
// There is no producer-shared stats accumulator for prepare to race.

test "wave31b: build stages slot build_stats; post-build accumulation never leaks into the merge" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
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
    var m0 = Mesh{
        .name = "bstats_0",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.zero,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    try scene.meshes.append(alloc, &m0);

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    var claim = scene.tryClaimBuildSlot().?;
    claim.build();

    // Staging point: queue counters land directly in the claimed slot.
    const slot = claim.slot;
    // The fixture is meaningful: the queue build counted the mesh and the
    // slot generation identifies the exact producer build.
    const staged = scene.draws.slots[slot].build_stats;
    try std.testing.expect(staged.total_meshes > 0);
    try std.testing.expectEqual(claim.seq, scene.draws.slots[slot].build_seq);
    claim.publish();

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try std.testing.expectEqual(scene.build_seq.load(.monotonic), scene.last_latched_seq.load(.monotonic));

    // Merge point: stats carry exactly the staged copy, not the live field.
    try std.testing.expectEqual(staged.total_meshes, scene.stats.total_meshes);
    try std.testing.expectEqual(staged.rendered_meshes, scene.stats.rendered_meshes);
    try std.testing.expectEqual(staged.culled_meshes, scene.stats.culled_meshes);
    try std.testing.expectEqual(staged.occluded_meshes, scene.stats.occluded_meshes);
    try std.testing.expectEqual(staged.occluders_count, scene.stats.occluders_count);
    try std.testing.expectEqual(staged.occluder_triangles, scene.stats.occluder_triangles);
    // The consumed slot's counters are cleared after merge.
    try std.testing.expectEqual(SceneStats{}, scene.draws.slots[scene.draws.front].build_stats);
}

test "wave31b: slot reset zeroes staged build_stats; reuse never resurfaces stale stats" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
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

    // Reset semantics: a nonzero staged copy never survives `reset`.
    scene.draws.slots[1].build_stats.total_meshes = 7;
    scene.draws.slots[1].build_stats.occluders_count = 9;
    scene.draws.slots[1].reset();
    try std.testing.expectEqual(SceneStats{}, scene.draws.slots[1].build_stats);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    var m0 = Mesh{
        .name = "bstats_reuse",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.zero,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    try scene.meshes.append(alloc, &m0);

    // Round 1 (meshes present): the latch merges nonzero staged stats.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try buildForTest(&scene);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try std.testing.expect(scene.stats.total_meshes > 0);

    // Round 2 (mesh list emptied, slots reused): the reused slot stages a
    // fresh zero copy — no stale round-1 stats resurface in the merge.
    scene.meshes.clearRetainingCapacity();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try buildForTest(&scene);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try std.testing.expectEqual(@as(u32, 0), scene.stats.total_meshes);
    try std.testing.expectEqual(@as(u32, 0), scene.stats.rendered_meshes);
    try std.testing.expectEqual(@as(u32, 0), scene.stats.culled_meshes);
    try std.testing.expectEqual(@as(u32, 0), scene.stats.occluded_meshes);
    try std.testing.expectEqual(@as(u32, 0), scene.stats.occluders_count);
    try std.testing.expectEqual(@as(u32, 0), scene.stats.occluder_triangles);
}

test "wave39: split prepare consumes its claimed generation, not a newer publication" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.profiler.deinit();

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    var first = scene.tryClaimBuildSlot().?;
    first.build();
    first.stageUi();
    first.publish();

    // Begin claims generation 1 and performs the live-touching prelude. The
    // next producer build is then allowed to publish before generation 1's
    // finish, exactly matching the sandbox unlock boundary.
    const first_prepare = scene.beginStagedPrepare().?;
    try std.testing.expectEqual(first.seq, first_prepare.build_seq);
    try std.testing.expect(scene.prepare_claim_active);

    scene.publishFrameSnapshot(16.0 / 9.0, 1024, 768);
    var second = scene.tryClaimBuildSlot().?;
    second.build();
    second.stageUi();
    second.publish();
    try std.testing.expect(second.seq > first.seq);

    scene.finishStagedPrepare(first_prepare);
    try std.testing.expectEqual(first.seq, scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(second.seq, scene.build_seq.load(.monotonic));
    try std.testing.expectEqual(first.slot, scene.draws.front);
    try std.testing.expect(!scene.prepare_claim_active);

    const second_prepare = scene.beginStagedPrepare().?;
    try std.testing.expectEqual(second.seq, second_prepare.build_seq);
    scene.finishStagedPrepare(second_prepare);
    try std.testing.expectEqual(second.seq, scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(second.slot, scene.draws.front);
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());
    try std.testing.expect(!scene.prepare_claim_active);
}

test "wave39: cancelled split prepare releases its slot and closes its epoch" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.profiler.deinit();

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    var build = scene.tryClaimBuildSlot().?;
    build.build();
    build.stageUi();
    build.publish();

    const claim = scene.beginStagedPrepare().?;
    scene.cancelStagedPrepare(claim);
    try std.testing.expect(!scene.prepare_claim_active);
    try std.testing.expect(!scene.draws.writing[claim.back_idx]);
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());
    try std.testing.expectEqual(scene.gpu_retire.current(), scene.gpu_retire.lastCompleted());
    try std.testing.expectEqual(@as(u64, 0), scene.last_latched_seq.load(.monotonic));
}

test "wave39: no fresh staged build begins null, front stays reusable" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.profiler.deinit();

    // No fresh producer build: staged begin returns null and touches
    // nothing (no stats/epoch/live reads/uploads); the consumed front stays
    // available for reuse.
    const epoch0 = scene.retire_epoch;
    const stats0 = scene.stats;
    try std.testing.expect(scene.beginStagedPrepare() == null);
    try std.testing.expectEqual(@as(u64, 0), scene.build_seq.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 0), scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(epoch0, scene.retire_epoch);
    try std.testing.expect(!scene.frame_prepared);

    // A full producer build then latches the exact generation.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    var full = scene.tryClaimBuildSlot().?;
    full.build();
    full.stageUi();
    full.publish();
    const ui_seq = scene.build_seq.load(.monotonic);
    const claim = scene.beginStagedPrepare().?;
    try std.testing.expect(claim.build_seq == ui_seq);
    scene.finishStagedPrepare(claim);
    try std.testing.expectEqual(full.seq, scene.last_latched_seq.load(.monotonic));
    _ = stats0;
}

test "wave39: staged build consumes the exact generation; empty stays null" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.profiler.deinit();

    // Full staged build consumes the exact generation; afterwards no
    // fresh build remains so staged begin stays null (no live reads).
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    var full_b = scene.tryClaimBuildSlot().?;
    full_b.build();
    full_b.stageUi();
    full_b.publish();
    const ui_seq = scene.build_seq.load(.monotonic);
    finishForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(ui_seq, scene.last_latched_seq.load(.monotonic));

    // Nothing pending: staged begin stays null, pins released.
    try std.testing.expect(scene.beginStagedPrepare() == null);
    try std.testing.expectEqual(ui_seq, scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());
}

test "wave39: cancel restores the handoff; a newer publish keeps newest-wins" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.profiler.deinit();

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    var build = scene.tryClaimBuildSlot().?;
    build.build();
    build.stageUi();
    build.publish();

    const claim = scene.beginStagedPrepare().?;
    scene.cancelStagedPrepare(claim);

    // The handoff was restored unchanged: re-begin resolves the SAME
    // generation and can be consumed to completion.
    try std.testing.expect(scene.draws.handoff != null);
    const retry = scene.beginStagedPrepare().?;
    try std.testing.expectEqual(claim.build_seq, retry.build_seq);
    try std.testing.expectEqual(claim.back_idx, retry.back_idx);
    scene.finishStagedPrepare(retry);
    try std.testing.expectEqual(build.seq, scene.last_latched_seq.load(.monotonic));

    // Newest-wins negative: cancel AFTER a newer publish must not restore
    // the stale handoff over the newer pending pair. Begin claims the mid
    // build's handoff, the producer then publishes a third build on the
    // remaining slot, and the cancel observes the changed pair.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    var mid = scene.tryClaimBuildSlot().?;
    mid.build();
    mid.stageUi();
    mid.publish();
    const mid_claim = scene.beginStagedPrepare().?;
    try std.testing.expectEqual(mid.seq, mid_claim.build_seq);

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    var newer = scene.tryClaimBuildSlot().?;
    try std.testing.expect(newer.slot != mid_claim.back_idx);
    newer.build();
    newer.publish();
    const newer_seq = scene.build_seq.load(.monotonic);

    scene.cancelStagedPrepare(mid_claim);
    const resumed = scene.beginStagedPrepare().?;
    try std.testing.expectEqual(newer_seq, resumed.build_seq);
    scene.finishStagedPrepare(resumed);
    try std.testing.expectEqual(newer_seq, scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());
}

test "wave39: pinned handoff refuses the staged begin, counted, stamp untouched" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.profiler.deinit();

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    var build = scene.tryClaimBuildSlot().?;
    build.build();
    build.stageUi();
    build.publish();
    const seq = scene.build_seq.load(.monotonic);

    try scene.draws.pin(scene.build_slot.load(.monotonic));
    const ref0 = scene.draws.publish_refusals;
    try std.testing.expect(scene.beginStagedPrepare() == null);
    try std.testing.expectEqual(ref0 + 1, scene.draws.publish_refusals);
    try std.testing.expectEqual(seq, scene.build_seq.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 0), scene.last_latched_seq.load(.monotonic));

    // Unpin restores: the same generation latches exactly once.
    try scene.draws.unpin(scene.build_slot.load(.monotonic));
    const claim = scene.beginStagedPrepare().?;
    try std.testing.expectEqual(seq, claim.build_seq);
    scene.finishStagedPrepare(claim);
    try std.testing.expectEqual(seq, scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());
}

test "slice6: staged prepare consumes frozen trail packet despite live mutation" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.profiler.deinit();
    defer scene.trails.deinit(alloc);
    defer {
        for (scene.meshes.items) |m| {
            m.deinit(alloc);
            alloc.destroy(m);
        }
        scene.meshes.deinit(alloc);
    }

    const TrailMesh = @import("../mesh/trail.zig").TrailMesh;
    const Vertex = @import("../mesh/types.zig").Vertex;
    // Manual fixture (no TrailMesh.init: it issues sg.* when a previous
    // test already marked this thread as the context thread).
    const mesh_ptr = try alloc.create(Mesh);
    mesh_ptr.* = .{ .name = "pkt_trail", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 0 };
    errdefer alloc.destroy(mesh_ptr);
    try scene.meshes.append(alloc, mesh_ptr);
    const verts = try alloc.alloc(Vertex, 4);
    errdefer alloc.free(verts);
    for (verts) |*v| v.* = std.mem.zeroes(Vertex);
    const idx = try alloc.alloc(u16, 6);
    errdefer alloc.free(idx);
    const tm = try alloc.create(TrailMesh);
    errdefer alloc.destroy(tm);
    tm.* = .{
        .allocator = alloc,
        .scene = &scene,
        .mesh = mesh_ptr,
        .options = .{},
        .vertices = verts,
        .indices = idx,
        .gpu_dirty = true,
        .buffers_pending = false,
        .pending_vertex_count = 2,
        .pending_index_count = 6,
        .pending_min_pt = Vec3.new(1, 2, 3),
        .pending_max_pt = Vec3.new(4, 5, 6),
    };
    errdefer tm.deinit();
    try scene.trails.meshes.append(alloc, tm);
    // Stage two verts + one quad (6 indices) as the frozen generation.
    tm.vertices[0].position = .{ 1, 2, 3 };
    tm.vertices[1].position = .{ 4, 5, 6 };
    tm.indices[0] = 0;
    tm.indices[1] = 1;
    tm.indices[2] = 0;
    tm.indices[3] = 1;
    tm.indices[4] = 0;
    tm.indices[5] = 1;
    tm.pending_vertex_count = 2;
    tm.pending_index_count = 6;
    tm.pending_min_pt = Vec3.new(1, 2, 3);
    tm.pending_max_pt = Vec3.new(4, 5, 6);
    tm.gpu_dirty = true;

    // Context-thread work starts here (creation above stayed deferred while
    // off-context, so no sg.* ran headless).
    gpu_thread.markContextThread();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    var build = scene.tryClaimBuildSlot().?;
    build.build();
    build.publish();
    const slot = &scene.draws.slots[build.slot];
    try std.testing.expectEqual(@as(usize, 1), slot.trail_uploads.items.len);
    try std.testing.expectEqual(@as(usize, 2), slot.trail_uploads.items[0].vert_count);

    // Live mutation after the freeze: different counts, verts, bounds.
    tm.pending_vertex_count = 1;
    tm.pending_index_count = 0;
    tm.vertices[0].position = .{ 99, 99, 99 };
    tm.pending_min_pt = Vec3.new(99, 99, 99);
    tm.pending_max_pt = Vec3.new(100, 100, 100);

    const claim = scene.beginStagedPrepare().?;
    // Begin flushes headless: the packet records undelivered WITHOUT
    // touching live state (phase 2 ownership — no flag clears, no scalar
    // publishes, no handle installs on the context path). The frozen
    // bytes stay intact despite the live mutation above.
    try std.testing.expect(!slot.trail_uploads.items[0].delivered);
    try std.testing.expectEqual(@as(u32, 0), tm.mesh.index_count);
    try std.testing.expect(!tm.gpu_dirty);
    try std.testing.expectEqual([3]f32{ 1, 2, 3 }, slot.trail_verts.items[0].position);
    scene.finishStagedPrepare(claim);
    try std.testing.expect(scene.frame_prepared);

    // Next build commits the undelivered outcome game-side: the flags
    // re-arm for retry (headless never lands), scalars stay unpublished.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    var build2 = scene.tryClaimBuildSlot().?;
    // Simulate the live-context delivery of the frozen generation for the
    // commit below: a real flush would have set this after uploading the
    // frozen bytes (headless-safe part of the test is the commit path).
    slot.trail_uploads.items[0].delivered = true;
    build2.build();
    // The frozen scalars published (commit path): index count + bounds
    // from the packet, never the mutated live pending values.
    try std.testing.expectEqual(@as(u32, 6), tm.mesh.index_count);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), tm.mesh.local_bounding_box.min.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), tm.mesh.local_bounding_box.max.x, 1e-6);
    build2.publish();
}

test "auto-exposure: Scene.update automatic luminance metering and temporal lifecycle" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);

    scene.meshes = .empty;
    scene.outline_meshes = .empty;
    scene.cameras = .empty;
    scene.draws = .{};
    scene.frame_handoff = .{};
    scene.light_handoff = .{};
    scene.uploads = null;
    scene.particles.systems = .empty;
    scene.particles.frame = .empty;
    scene.physics.world = null;
    scene.physics.debug_lines = .empty;
    scene.physics.prepared_lines = .empty;
    scene.physics.build_lines = .empty;
    scene.build_seq.store(0, .monotonic);
    scene.last_latched_seq.store(0, .monotonic);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    // Enable auto-exposure with calibrated middle-gray 0.18
    var post = postprocess.PostProcessOptions{
        .enabled = true,
        .auto_exposure_enabled = true,
        .auto_exposure_key_value = 0.18,
        .auto_exposure_speed_up = 3.0,
        .auto_exposure_speed_down = 3.0,
        .auto_exposure_min = 0.01,
        .auto_exposure_max = 16.0,
    };
    scene.lights.hemi.intensity = 0.0;
    scene.setPostProcess(post);

    // Initial adapted exposure is 1.0
    try std.testing.expectEqual(@as(f32, 1.0), scene.getAdaptedExposure());

    // 1. Moderate sun: sun intensity = 1.0
    // First update tick performs instant snap (no history yet)
    _ = try scene.createDirectionalLight("Sun", .{ .direction = Vec3.new(0, -1, 0), .diffuse = Color3.white, .intensity = 1.0 });
    try scene.update(0.016);
    const exp1 = scene.getAdaptedExposure();
    try std.testing.expect(exp1 > 0.5 and exp1 < 2.0);

    // 2. Bright sun jumps to 20.0: scene becomes much brighter
    scene.lights.directional.?.intensity = 20.0;
    // Over several updates with dt=0.1s, exposure adapts downward smoothly
    var prev_exp = exp1;
    for (0..10) |_| {
        try scene.update(0.1);
        const cur_exp = scene.getAdaptedExposure();
        try std.testing.expect(cur_exp < prev_exp);
        prev_exp = cur_exp;
    }
    // After adaptation, exposure is significantly lower than initial
    try std.testing.expect(scene.getAdaptedExposure() < 0.15);

    // 3. Camera cut test: when cut flag is set, adaptation snaps instantly in 1 tick
    scene.post_process.auto_exposure_camera_cut = true;
    // Set sun to pitch black
    scene.lights.directional.?.intensity = 0.0;
    try scene.update(0.016);
    // Instant snap to near max exposure (pitch black scene)
    try std.testing.expect(scene.getAdaptedExposure() > 10.0);
    // Camera cut flag was cleared after consumption
    try std.testing.expect(!scene.post_process.auto_exposure_camera_cut);

    // 4. Disabling auto-exposure resets exposure to 1.0
    post.auto_exposure_enabled = false;
    scene.setPostProcess(post);
    try std.testing.expectEqual(@as(f32, 1.0), scene.getAdaptedExposure());
}
