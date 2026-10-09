//! Staged frame tests (part 2): P7 triple-buffer slots, mailbox saturation, debug capture, sky snapshot, worker churn. Split from scene/tests.zig.
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

// ---- P7 triple-buffered prepared draws. ----

test "P7: slots alternate, newest wins, front intact while building back" {
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

    const skel = try Skeleton.init(alloc, 1);
    defer skel.deinit();
    skel.bones[0].local_position = Vec3.new(1, 0, 0);
    skel.update();

    var hook_mat = ShaderMaterial{ .name = "p7_hook" };
    var mesh_a = Mesh{
        .name = "p7_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(4, 0, 0),
    };
    var mesh_s = Mesh{
        .name = "p7_s",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(1, 0, 0),
        .skeleton = skel,
    };
    var mesh_h = Mesh{
        .name = "p7_h",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(2, 0, 0),
        .material = .{ .shader_material = &hook_mat },
    };
    try scene.meshes.append(alloc, &mesh_a);
    try scene.meshes.append(alloc, &mesh_s);
    try scene.meshes.append(alloc, &mesh_h);
    try scene.outline_meshes.append(alloc, &mesh_a);

    // PIP enabled so a views[i] output is part of every frame (and of the
    // zero-alloc proof below): the second camera sees all three meshes.
    scene.enable_multi_camera = true;
    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    const cam2 = Camera{ .free = camera_mod.FreeCamera.init("Cam2", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    _ = try scene.addCamera(.{ .name = "Cam2", .camera = cam2 });
    scene.active_camera_index = 0;
    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);

    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.frame_prepared);
    const front0 = scene.draws.front;
    const d0 = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 3), d0.primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), d0.primary.skin_storage.items.len);
    try std.testing.expectEqual(@as(usize, 1), d0.primary.shader_storage.items.len);
    try std.testing.expectEqual(@as(usize, 3), d0.views[1].items.items.len);
    try std.testing.expectEqual(@as(usize, 1), d0.outline_items.items.len);
    try std.testing.expectEqual(@as(usize, 3), p7ShadowTotal(d0));
    try std.testing.expectEqual(@as(usize, 3), d0.shadow.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), d0.shadow.skins.items.len);
    try std.testing.expectEqual(scene.frame_id, d0.frame_id);
    try std.testing.expectEqual(scene.retire_epoch, d0.retire_epoch);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), d0.outline_items.items[0].model.m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), p7FindByMeshIndex(d0.primary.items.items, 0).?.model.m[12], 1e-4);

    // Front intact while building back: move mesh_a, build a claimed slot
    // via the single staged protocol (claim.build), and prove the published
    // front is untouched while the claimed back sees the new state.
    mesh_a.position = Vec3.new(9, 0, 0);
    var scratch = scene.tryClaimBuildSlot() orelse std.debug.panic("{s}", .{"scratch claim saturated"});
    scratch.build();
    const back_idx = scratch.slot;
    const front_still = &scene.draws.slots[front0];
    try std.testing.expectEqual(@as(usize, 3), front_still.primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), front_still.primary.skin_storage.items.len);
    try std.testing.expectEqual(@as(usize, 1), front_still.primary.shader_storage.items.len);
    try std.testing.expectEqual(@as(usize, 1), front_still.outline_items.items.len);
    try std.testing.expectEqual(@as(usize, 3), front_still.shadow.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), front_still.shadow.skins.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), front_still.outline_items.items[0].model.m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), p7FindByMeshIndex(front_still.primary.items.items, 0).?.model.m[12], 1e-4);
    const back_built = &scene.draws.slots[back_idx];
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), p7FindByMeshIndex(back_built.primary.items.items, 0).?.model.m[12], 1e-4);
    // The staged build freezes the full payload (outline included) at the new pose.
    try std.testing.expectEqual(@as(usize, 1), back_built.outline_items.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), back_built.outline_items.items[0].model.m[12], 1e-4);
    scratch.publish();

    // Warmup: consume the claimed build (no rebuild: frozen pose wins), then
    // keep warming until every slot has been built once as back.
    finishForTest(&scene);
    try std.testing.expectEqual(back_idx, scene.draws.front);
    const d1 = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 3), d1.primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), d1.primary.skin_storage.items.len);
    try std.testing.expectEqual(@as(usize, 1), d1.primary.shader_storage.items.len);
    try std.testing.expectEqual(@as(usize, 3), d1.views[1].items.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), d1.outline_items.items[0].model.m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), p7FindByMeshIndex(d1.primary.items.items, 0).?.model.m[12], 1e-4);
    try std.testing.expectEqual(@as(usize, 3), p7ShadowTotal(d1));
    try std.testing.expectEqual(@as(usize, 1), d1.shadow.skins.items.len);
    // The discarded front is retained as scratch, not cleared or copied.
    const old_slot = &scene.draws.slots[front0];
    try std.testing.expectEqual(@as(usize, 3), old_slot.primary.items.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), old_slot.outline_items.items[0].model.m[12], 1e-4);

    // Second warmup prepare: with three slots rotating, one full prepare
    // warms exactly one slot as back — the third slot is still cold.
    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);
    {
        try buildForTest(&scene);
        const want = scene.build_slot.load(.monotonic);
        finishForTest(&scene);
        try std.testing.expectEqual(want, scene.draws.front);
        try std.testing.expectApproxEqAbs(@as(f32, 9.0), scene.preparedDraws().outline_items.items[0].model.m[12], 1e-4);
    }

    // Zero-alloc proof: all three slots are warm now (prepare#1, warmup#1,
    // warmup#2 built one slot each as back), so three further prepares — one
    // per slot as back — must not touch the allocator at all. The wrapper refuses ANY fresh
    // alloc (fail_index=0, flagged in has_induced_failure) and ANY
    // resize/remap (resize_fail_index=0); a refused growth always degrades
    // the frame (dropped items/counts), so the flag plus the expected
    // counts + selected poses under allocation refusal together prove zero
    // allocator traffic. Same wrapper feeds
    // scene.allocator and the shadow pass allocator.
    // Scope (narrow): this fixture exercises 3 regular meshes
    // (plain/skinned/hook), the serial cull path (no worker pool), one
    // outline item, shadow bin+snapshot, and primary+PIP view builds — with
    // no canvas, no instancing, no LOD/morphs. NOT covered by this proof:
    // parallel_scratch growth, instanced staging/buffer growth, UI capture
    // growth, or any parallel/GPU path.
    var round: usize = 0;
    while (round < 3) : (round += 1) {
        var refusing = std.testing.FailingAllocator.init(alloc, .{
            .fail_index = 0,
            .resize_fail_index = 0,
        });
        scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);
        // Warm slots prove zero allocator traffic on BOTH halves: the
        // refusing wrapper covers the staged build and the finish.
        const saved_alloc = scene.allocator;
        scene.allocator = refusing.allocator();
        scene.shadows.pass.allocator = refusing.allocator();
        // Rotation-agnostic expectation: the staged claim slot becomes front.
        try buildForTest(&scene);
        const want_front = scene.build_slot.load(.monotonic);
        finishForTest(&scene);
        scene.allocator = saved_alloc;
        scene.shadows.pass.allocator = saved_alloc;
        try std.testing.expect(!refusing.has_induced_failure);
        try std.testing.expectEqual(want_front, scene.draws.front);
        const dz = scene.preparedDraws();
        try std.testing.expectEqual(@as(usize, 3), dz.primary.items.items.len);
        try std.testing.expectEqual(@as(usize, 1), dz.primary.skin_storage.items.len);
        try std.testing.expectEqual(@as(usize, 1), dz.primary.shader_storage.items.len);
        try std.testing.expectEqual(@as(usize, 3), dz.views[1].items.items.len);
        try std.testing.expectEqual(@as(usize, 1), dz.outline_items.items.len);
        try std.testing.expectApproxEqAbs(@as(f32, 9.0), dz.outline_items.items[0].model.m[12], 1e-4);
        try std.testing.expectApproxEqAbs(@as(f32, 9.0), p7FindByMeshIndex(dz.primary.items.items, 0).?.model.m[12], 1e-4);
        try std.testing.expectEqual(@as(usize, 3), dz.shadow.items.items.len);
        try std.testing.expectEqual(@as(usize, 1), dz.shadow.skins.items.len);
        try std.testing.expectEqual(@as(usize, 3), p7ShadowTotal(dz));
        try std.testing.expectEqual(scene.frame_id, dz.frame_id);
        try std.testing.expectEqual(scene.retire_epoch, dz.retire_epoch);
    }
}

test "P7: main and PIP view slots stay isolated" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.binned_source.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    scene.enable_frustum_culling = false;
    scene.enable_occlusion_culling = false;
    scene.enable_multi_camera = true;
    scene.default_white_texture.view.id = 1;
    p7CpuShadowPass(&scene, alloc);

    var mesh_a = Mesh{
        .name = "pip_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .layer_mask = 0b01,
    };
    var mesh_b = Mesh{
        .name = "pip_b",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(5, 0, 0),
        .layer_mask = 0b10,
    };
    try scene.meshes.append(alloc, &mesh_a);
    try scene.meshes.append(alloc, &mesh_b);

    const cam0 = Camera{ .free = camera_mod.FreeCamera.init("Main", .{}) };
    const cam1 = Camera{ .free = camera_mod.FreeCamera.init("Pip", .{}) };
    const cam2 = Camera{ .free = camera_mod.FreeCamera.init("Off", .{}) };
    _ = try scene.addCamera(.{ .name = "Main", .camera = cam0 });
    _ = try scene.addCamera(.{ .name = "Pip", .camera = cam1 });
    _ = try scene.addCamera(.{ .name = "Off", .camera = cam2 });
    scene.cameras.items[1].culling_mask = 0b10;
    scene.cameras.items[2].enabled = false;
    scene.active_camera_index = 0;
    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);

    try stageAndPrepareForTest(&scene);
    const draws = scene.preparedDraws();
    // Primary (all-mask active camera) sees both meshes.
    try std.testing.expectEqual(@as(usize, 2), draws.primary.items.items.len);
    // PIP slot 1 (mask 0b10) sees only mesh_b; slot 2 (disabled) is
    // reset-empty so no prior frame can resurface through it.
    try std.testing.expectEqual(@as(usize, 1), draws.views[1].items.items.len);
    try std.testing.expectEqual(@as(u32, 1), draws.views[1].items.items[0].mesh_index);
    try std.testing.expectEqual(@as(usize, 0), draws.views[2].items.items.len);
    // Shadow ignores view masks: both meshes binned in the same slot.
    try std.testing.expectEqual(@as(usize, 2), p7ShadowTotal(draws));

    // Disable the PIP view: the next publish clears its slot instead of
    // resurfacing the mesh_b frame above.
    scene.cameras.items[1].enabled = false;
    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);
    try stageAndPrepareForTest(&scene);
    const draws2 = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 0), draws2.views[1].items.items.len);
    try std.testing.expectEqual(@as(usize, 2), draws2.primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 2), p7ShadowTotal(draws2));
}

test "P7: repeated staged builds win newest, no duplicate outline/UI capture" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.ui_frame.deinit(alloc);
    scene.enable_frustum_culling = false;
    scene.enable_occlusion_culling = false;

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

    var mesh = Mesh{
        .name = "rep_outline",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(4, 0, 0),
    };
    try scene.outline_meshes.append(alloc, &mesh);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);

    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(usize, 1), scene.preparedDraws().outline_items.items.len);
    const ui_n = canvas_ui.vertices.items.len;
    try std.testing.expectEqual(ui_n, scene.ui_frame.vertices.items.len);

    // More UI + moved outline, then a fresh staged build with no new camera
    // publish (build packs fresh live state): newest wins, nothing accumulates.
    canvas_ui.drawRect(1, 2, 3, 4, Color4.white);
    mesh.position = Vec3.new(7, 0, 0);
    try buildForTest(&scene);
    const want_rep = scene.build_slot.load(.monotonic);
    finishForTest(&scene);
    try std.testing.expectEqual(want_rep, scene.draws.front);
    try std.testing.expectEqual(@as(usize, 1), scene.preparedDraws().outline_items.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), scene.preparedDraws().outline_items.items[0].model.m[12], 1e-4);
    try std.testing.expectEqual(canvas_ui.vertices.items.len, scene.ui_frame.vertices.items.len);
    try std.testing.expect(canvas_ui.vertices.items.len > ui_n);
}

test "P7: no-camera/headless staged builds clear coherently, epochs consumed" {
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
        .name = "clr_mesh",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
    };
    try scene.meshes.append(alloc, &mesh);
    try scene.outline_meshes.append(alloc, &mesh);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.publishFrameSnapshot(1.0, 640, 480);
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(usize, 1), scene.preparedDraws().primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), p7ShadowTotal(scene.preparedDraws()));
    const epoch1 = scene.retire_epoch;

    // Camera lost: views + shadow publish coherent-empty (no resurface of
    // the frame above); outline stays unconditional (existing semantics).
    for (scene.cameras.items) |entry| {
        if (entry.owns_name) alloc.free(entry.name);
    }
    scene.cameras.clearRetainingCapacity();
    scene.active_camera = null;
    scene.active_camera_index = null;
    try stageAndPrepareForTest(&scene);
    const cleared = scene.preparedDraws();
    try std.testing.expect(!scene.frame_snapshot.has_camera);
    try std.testing.expectEqual(@as(usize, 0), cleared.primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 0), cleared.views[0].items.items.len);
    try std.testing.expectEqual(@as(usize, 0), cleared.shadow.items.items.len);
    try std.testing.expectEqual(@as(usize, 0), cleared.shadow.skins.items.len);
    for (cleared.shadow.bin.counts) |c| try std.testing.expectEqual(@as(usize, 0), c);
    try std.testing.expectEqual(@as(usize, 1), cleared.outline_items.items.len);
    // Repeated prepare discarded the pending frame: begin auto-closed its
    // epoch before the flush tore down anything it borrowed.
    try std.testing.expectEqual(epoch1 + 1, scene.retire_epoch);
    try std.testing.expectEqual(epoch1, scene.gpu_retire.lastCompleted());

    // No-camera render still completes the current epoch (headless-safe:
    // the early return runs before any sg.*).
    scene.render();
    try std.testing.expect(!scene.frame_prepared);
    try std.testing.expectEqual(scene.retire_epoch, scene.gpu_retire.lastCompleted());

    // Camera back but shadows disabled: primary rebuilds, shadow stays empty.
    const cam2 = Camera{ .free = camera_mod.FreeCamera.init("Cam2", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam2", .camera = cam2 });
    scene.shadows.enabled = false;
    scene.publishFrameSnapshot(1.0, 640, 480);
    try stageAndPrepareForTest(&scene);
    const noshadow = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 1), noshadow.primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 0), noshadow.shadow.items.items.len);
    for (noshadow.shadow.bin.counts) |c| try std.testing.expectEqual(@as(usize, 0), c);
}

test "P7: allocator-failure staged slot stays coherent and recovers" {
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

    const skel = try Skeleton.init(alloc, 1);
    defer skel.deinit();
    skel.bones[0].local_position = Vec3.new(1, 0, 0);
    skel.update();

    var mesh_a = Mesh{
        .name = "oom_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(4, 0, 0),
    };
    var mesh_s = Mesh{
        .name = "oom_s",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .skeleton = skel,
    };
    try scene.meshes.append(alloc, &mesh_a);
    try scene.meshes.append(alloc, &mesh_s);
    try scene.outline_meshes.append(alloc, &mesh_a);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.publishFrameSnapshot(1.0, 640, 480);
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(usize, 2), scene.preparedDraws().primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 2), p7ShadowTotal(scene.preparedDraws()));

    // Failing back build: everything fallible drops (existing per-item /
    // coherent-empty OOM semantics — never a full-frame transaction abort),
    // and the publish stays index-coherent: no skin/shader/order index
    // escapes its side store, no bin range escapes the items.
    const real_alloc = scene.allocator;
    var failing = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 0 });
    scene.allocator = failing.allocator();
    scene.shadows.pass.allocator = failing.allocator();
    scene.publishFrameSnapshot(1.0, 640, 480);
    try stageAndPrepareForTest(&scene);
    scene.allocator = real_alloc;
    scene.shadows.pass.allocator = real_alloc;
    const oom = scene.preparedDraws();
    for (oom.primary.items.items) |it| {
        if (it.skin_index) |s| try std.testing.expect(s < oom.primary.skin_storage.items.len);
        if (it.shader_index) |s| try std.testing.expect(s < oom.primary.shader_storage.items.len);
    }
    for (oom.primary.transparent.items) |it| {
        if (it.skin_index) |s| try std.testing.expect(s < oom.primary.skin_storage.items.len);
        if (it.shader_index) |s| try std.testing.expect(s < oom.primary.shader_storage.items.len);
    }
    for (oom.outline_items.items) |it| {
        if (it.skin_index) |s| try std.testing.expect(s < oom.outline_skins.items.len);
    }
    for (oom.primary.transparent_order.items) |e| {
        switch (e.kind) {
            .regular => try std.testing.expect(e.index < oom.primary.transparent.items.len),
            .instanced => try std.testing.expect(e.index < oom.primary.transparent_instanced.items.len),
        }
    }
    for (oom.shadow.bin.counts, oom.shadow.bin.offsets) |c, o| {
        try std.testing.expect(o + c <= oom.shadow.items.items.len);
    }
    for (oom.shadow.items.items) |it| {
        if (it.skin_index) |s| try std.testing.expect(s < oom.shadow.skins.items.len);
    }

    // Recovery with the working allocator: no stale items, full frame back.
    mesh_a.position = Vec3.new(6, 0, 0);
    scene.publishFrameSnapshot(1.0, 640, 480);
    try stageAndPrepareForTest(&scene);
    const rec = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 2), rec.primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 1), rec.primary.skin_storage.items.len);
    try std.testing.expectEqual(@as(usize, 1), rec.outline_items.items.len);
    try std.testing.expectEqual(@as(usize, 2), p7ShadowTotal(rec));
    try std.testing.expectEqual(@as(usize, 1), rec.shadow.skins.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), rec.outline_items.items[0].model.m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), p7FindByMeshIndex(rec.primary.items.items, 0).?.model.m[12], 1e-4);
}

// ---- Actual update||render boundary: render-owned captures. ----

test "saturated frame mailbox drops instead of overwriting the consumed snapshot" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);

    // Consumed snapshot (screen dims are set before the no-camera
    // early-out, so no camera is needed for this ownership proof).
    scene.publishFrameSnapshot(1.0, 111, 111);
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(i32, 111), scene.frame_snapshot.screen_w);

    // Force the last-unclaimable case: hold BOTH slots in WRITING (a lagging
    // holder). releasePublished only drains PUBLISHED slots, so the claim
    // still fails — the tick must DROP, never overwrite the consumed
    // snapshot under a concurrent render.
    const a = scene.frame_handoff.claim().?;
    const b = scene.frame_handoff.claim().?;
    try std.testing.expect(scene.frame_handoff.claim() == null);
    scene.publishFrameSnapshot(1.0, 999, 999);
    try std.testing.expectEqual(@as(i32, 111), scene.frame_snapshot.screen_w);

    // Finish the held claims as stale publishes; the next real tick drains
    // them and the newest wins (no resurfacing of 222/333).
    scene.frame_handoff.slot(a).* = scene.packFrameSnapshot(1.0, 222, 222);
    scene.frame_handoff.publish(a);
    scene.frame_handoff.slot(b).* = scene.packFrameSnapshot(1.0, 333, 333);
    scene.frame_handoff.publish(b);
    scene.publishFrameSnapshot(1.0, 444, 444);
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(@as(i32, 444), scene.frame_snapshot.screen_w);
}

test "debug capture: off/world-empty stay empty, capture immutable, OOM coherent" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.physics.deinit(alloc);

    // No world + off: empty, invisible.
    scene.physics.captureDebug(alloc);
    try std.testing.expect(!scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 0), scene.physics.prepared_lines.items.len);

    _ = scene.physics.enable(alloc, null);
    const m = try alloc.create(Mesh);
    defer alloc.destroy(m);
    m.* = @import("../testing.zig").testMesh("dbg_cap");
    _ = try scene.physics.getWorld().?.createBody(m, .box, 0.0);

    // World present but off: still empty (draw site selects on the capture).
    scene.physics.captureDebug(alloc);
    try std.testing.expect(!scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 0), scene.physics.prepared_lines.items.len);

    scene.physics.show_debug = true;
    scene.physics.captureDebug(alloc);
    try std.testing.expect(scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 12), scene.physics.prepared_lines.items.len);

    // Immutable under later mutation without recapture (the update side may
    // step/move while render draws the capture).
    const x0 = scene.physics.prepared_lines.items[0].a.x;
    m.position = Vec3.new(5, 0, 0);
    scene.physics.captureDebug(alloc);
    try std.testing.expectEqual(@as(usize, 12), scene.physics.prepared_lines.items.len);
    try std.testing.expectApproxEqAbs(x0 + 5.0, scene.physics.prepared_lines.items[0].a.x, 1e-4);
    scene.physics.step(0.016);
    try std.testing.expectApproxEqAbs(x0 + 5.0, scene.physics.prepared_lines.items[0].a.x, 1e-4);

    // Headless upload/draw: safe no-ops, stats clean, no pass created.
    var stats = SceneStats{};
    scene.physics.uploadDebug(alloc, 1, .RGBA16F);
    try std.testing.expect(scene.physics.debug_pass == null);
    scene.physics.renderDebugPrepared(Mat4.identity, 1, .RGBA16F, &stats);
    try std.testing.expectEqual(@as(u32, 0), stats.draw_calls);
    try std.testing.expectEqual(@as(u32, 0), stats.main_draw_calls);

    // Non-empty capture + present-but-empty upload: still no count (the
    // counters gate on the issued-draw report, never on visibility alone).
    scene.physics.debug_pass = @import("../passes/debug_pass.zig").DebugPass{ .allocator = alloc };
    try std.testing.expect(scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 12), scene.physics.prepared_lines.items.len);
    scene.physics.renderDebugPrepared(Mat4.identity, 1, .RGBA16F, &stats);
    try std.testing.expectEqual(@as(u32, 0), stats.draw_calls);
    try std.testing.expectEqual(@as(u32, 0), stats.main_draw_calls);
    // Pull the fake pass back out: DebugPass.deinit issues sg.destroy*
    // (context-only, traps headless) — headless teardown is staging-only,
    // so the fixture deinit below must see null here.
    var fake_pass = scene.physics.debug_pass.?;
    scene.physics.debug_pass = null;
    fake_pass.staging.deinit(alloc);

    // Off clears (no stale overlay).
    scene.physics.show_debug = false;
    scene.physics.captureDebug(alloc);
    try std.testing.expect(!scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 0), scene.physics.prepared_lines.items.len);
    // Retained capacity survives the clear (no per-frame churn).
    try std.testing.expect(scene.physics.prepared_lines.capacity >= 12);
}

test "debug capture OOM fail-closes coherent-empty then recovers" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.physics.deinit(alloc);

    _ = scene.physics.enable(alloc, null);
    const m = try alloc.create(Mesh);
    defer alloc.destroy(m);
    m.* = @import("../testing.zig").testMesh("dbg_oom");
    _ = try scene.physics.getWorld().?.createBody(m, .box, 0.0);
    scene.physics.show_debug = true;

    // Unfunded capture on empty capacity: the upfront reserve fails ->
    // coherent-empty (never a partial half).
    var failing_fresh = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 0 });
    scene.physics.captureDebug(failing_fresh.allocator());
    try std.testing.expect(!scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 0), scene.physics.prepared_lines.items.len);

    // Funded prime (12 lines, retained capacity high-water).
    scene.physics.captureDebug(alloc);
    try std.testing.expect(scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 12), scene.physics.prepared_lines.items.len);

    // Force growth past retained capacity: a second body needs 24 lines,
    // so the reserve MUST allocate -> the failing allocator fails ->
    // coherent-empty with capacity retained (no shrink, no partial).
    const m2 = try alloc.create(Mesh);
    defer alloc.destroy(m2);
    m2.* = @import("../testing.zig").testMesh("dbg_oom2");
    _ = try scene.physics.getWorld().?.createBody(m2, .box, 0.0);
    var failing_grow = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 0 });
    scene.physics.captureDebug(failing_grow.allocator());
    try std.testing.expect(!scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 0), scene.physics.prepared_lines.items.len);
    try std.testing.expect(scene.physics.prepared_lines.capacity >= 12);

    // Recovery funds the full 24.
    scene.physics.captureDebug(alloc);
    try std.testing.expect(scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 24), scene.physics.prepared_lines.items.len);

    // Warm capture refuses fresh allocs entirely: steady-state captures
    // are allocation-free (no per-frame churn).
    var refusing = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 0, .resize_fail_index = 0 });
    scene.physics.captureDebug(refusing.allocator());
    try std.testing.expect(!refusing.has_induced_failure);
    try std.testing.expect(scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 24), scene.physics.prepared_lines.items.len);
}

test "sky snapshot freezes enabled/texture/exposure/defaults at prepare" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    // Fake cube: plain-data handles, no GPU involved.
    var cube = std.mem.zeroes(CubeTexture);
    cube.view.id = 77;
    cube.sampler.id = 78;
    scene.sky.setSkybox(cube);
    scene.sky.exposure = 2.0;
    scene.sky.ibl_intensity = 0.5;

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.frame_snapshot.sky_enabled);
    try std.testing.expectEqual(@as(f32, 2.0), scene.frame_snapshot.sky_exposure);
    try std.testing.expectEqual(@as(f32, 0.5), scene.frame_snapshot.ibl_intensity);
    try std.testing.expectEqual(@as(u32, 77), scene.frame_snapshot.sky_texture.?.view.id);

    // Mutate live AFTER prepare: the consumed snapshot stays frozen
    // (a concurrent update cannot change the in-flight packet). Covers
    // every render-owned sky/default field: enabled, exposure, the sky
    // cube, and all three default copies. (Sampler ids, not view ids: the
    // fixture has no GPU init, and a nonzero default-white view id would
    // open the GPU queue-build path with an uninitialized shadow pass.)
    scene.sky.enabled = false;
    scene.sky.exposure = 9.0;
    scene.sky.texture = null;
    scene.default_white_texture.sampler.id = 5;
    scene.default_normal_texture.sampler.id = 6;
    scene.default_cube_texture.sampler.id = 7;
    try std.testing.expect(scene.frame_snapshot.sky_enabled);
    try std.testing.expectEqual(@as(f32, 2.0), scene.frame_snapshot.sky_exposure);
    try std.testing.expectEqual(@as(u32, 77), scene.frame_snapshot.sky_texture.?.view.id);
    try std.testing.expectEqual(@as(u32, 0), scene.frame_snapshot.default_white.sampler.id);
    try std.testing.expectEqual(@as(u32, 0), scene.frame_snapshot.default_normal.sampler.id);
    try std.testing.expectEqual(@as(u32, 0), scene.frame_snapshot.default_cube.sampler.id);

    // Headless prepared draw: safe no-op, stats clean (the fixture pass is
    // undefined, but the sg.isvalid() short-circuit returns first).
    var stats = SceneStats{};
    scene.sky.renderPrepared(
        scene.frame_snapshot.sky_enabled,
        scene.frame_snapshot.primary_cam.camera,
        scene.frame_snapshot.primary_cam.aspect,
        scene.frame_snapshot.sky_texture orelse scene.frame_snapshot.default_cube,
        scene.frame_snapshot.sky_exposure,
        1,
        .RGBA16F,
        &stats,
    );
    try std.testing.expectEqual(@as(u32, 0), stats.draw_calls);

    // A disabled snapshot draws nothing even with the live layer enabled.
    scene.sky.enabled = true;
    scene.sky.renderPrepared(
        false,
        scene.frame_snapshot.primary_cam.camera,
        scene.frame_snapshot.primary_cam.aspect,
        scene.frame_snapshot.sky_texture orelse scene.frame_snapshot.default_cube,
        scene.frame_snapshot.sky_exposure,
        1,
        .RGBA16F,
        &stats,
    );
    try std.testing.expectEqual(@as(u32, 0), stats.draw_calls);

    // pack/publish/prepare round-trip: the NEXT publish captures the newest
    // live values (nothing stale), proving each stage reads live state at
    // prepare time and freezes it after.
    scene.sky.enabled = true;
    scene.sky.exposure = 4.0;
    var cube2 = std.mem.zeroes(CubeTexture);
    cube2.view.id = 88;
    scene.sky.setSkybox(cube2);
    scene.default_normal_texture.sampler.id = 66;
    scene.default_cube_texture.sampler.id = 67;
    // Direct pack reads live.
    const repacked = scene.packFrameSnapshot(16.0 / 9.0, 800, 600);
    try std.testing.expect(repacked.sky_enabled);
    try std.testing.expectEqual(@as(f32, 4.0), repacked.sky_exposure);
    try std.testing.expectEqual(@as(u32, 88), repacked.sky_texture.?.view.id);
    try std.testing.expectEqual(@as(u32, 66), repacked.default_normal.sampler.id);
    try std.testing.expectEqual(@as(u32, 67), repacked.default_cube.sampler.id);
    // Publish + prepare consume the newest mailbox frame.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.frame_snapshot.sky_enabled);
    try std.testing.expectEqual(@as(f32, 4.0), scene.frame_snapshot.sky_exposure);
    try std.testing.expectEqual(@as(u32, 88), scene.frame_snapshot.sky_texture.?.view.id);
    try std.testing.expectEqual(@as(u32, 5), scene.frame_snapshot.default_white.sampler.id);
    try std.testing.expectEqual(@as(u32, 66), scene.frame_snapshot.default_normal.sampler.id);
    try std.testing.expectEqual(@as(u32, 67), scene.frame_snapshot.default_cube.sampler.id);
}

test "worker churn after prepare cannot mutate consumed snapshot/timings" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.physics.deinit(alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.sky.enabled = true;
    scene.sky.exposure = 1.5;
    scene.recordUpdateTime(3.0);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(scene.frame_snapshot.has_camera);
    try std.testing.expectEqual(@as(f32, 3.0), scene.stats.update_ms);

    const snap_w = scene.frame_snapshot.screen_w;
    const snap_exp = scene.frame_snapshot.sky_exposure;
    const snap_clear_r = scene.frame_snapshot.clear_color.r;

    // Worker = the next update tick hammering live state. Spawned AFTER
    // prepare (a happens-before edge — deterministic, no sleeps/timing).
    // It mutates ONLY update-owned words: live sky fields, the staged tick,
    // clear color. Snapshot/stats are never touched by it (disjoint fields).
    const Worker = struct {
        scene: *Scene,
        iters: usize,
        fn run(self: @This()) void {
            var i: usize = 0;
            while (i < self.iters) : (i += 1) {
                self.scene.sky.exposure = 9.0;
                self.scene.sky.enabled = false;
                self.scene.recordUpdateTime(99.0);
                self.scene.clear_color = Color4.new(1, 0, 0, 1);
            }
        }
    };
    const t = try std.Thread.spawn(.{}, Worker.run, .{Worker{ .scene = &scene, .iters = 500 }});
    t.join();

    // Consumed render state frozen despite the churn.
    try std.testing.expectEqual(snap_w, scene.frame_snapshot.screen_w);
    try std.testing.expectEqual(snap_exp, scene.frame_snapshot.sky_exposure);
    try std.testing.expect(scene.frame_snapshot.sky_enabled);
    try std.testing.expectEqual(snap_clear_r, scene.frame_snapshot.clear_color.r);
    try std.testing.expectEqual(@as(f32, 3.0), scene.stats.update_ms);
    try std.testing.expectEqual(@as(f32, 99.0), scene.pending_update_ms.load(.monotonic));

    // Next prepare picks up the newest live state (newest wins).
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try stageAndPrepareForTest(&scene);
    try std.testing.expect(!scene.frame_snapshot.sky_enabled);
    try std.testing.expectEqual(@as(f32, 9.0), scene.frame_snapshot.sky_exposure);
    try std.testing.expectEqual(@as(f32, 99.0), scene.stats.update_ms);
}

// ---- Stage 1 producer-build handoff: buildPreparedFrame + prepare latch. ---
//
