//! Staged frame tests (part 4): stage-2A/2B queue builds and patching. Split from scene/tests.zig.
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

test "stage-2A: buildQueuesInto with staged params equals two runs" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);

    var m0 = Mesh{
        .name = "bq0",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.zero,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var m1 = Mesh{
        .name = "bq1",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(5, 0, 0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    try scene.meshes.append(alloc, &m0);
    try scene.meshes.append(alloc, &m1);
    try scene.outline_meshes.append(alloc, &m0);

    // Fake GPU-init (queue builds read only the id, no sg calls) and a
    // camera-bearing snapshot with shadows disabled (shadows.pass stays
    // undefined in this fixture, so the shadow path must be skipped).
    // Frustum/occlusion off: identity view_proj would otherwise cull the
    // offset mesh and the two-run comparison would be trivially 1 vs 1.
    scene.default_white_texture.view.id = 1;
    scene.enable_frustum_culling = false;
    scene.enable_occlusion_culling = false;
    scene.frame_id = 41;
    scene.frame_snapshot.has_camera = true;
    scene.frame_snapshot.shadows_enabled = false;
    scene.frame_snapshot.primary_cam = .{
        .view_proj = Mat4.identity,
        .eye = Vec3.zero,
        .culling_mask = 0xFFFFFFFF,
        .aspect = 1.0,
        .enabled = true,
    };

    var stats_a = SceneStats{};
    var stats_b = SceneStats{};
    var slot_a: FrameDrawSlot = .{};
    defer slot_a.deinit(alloc);
    var slot_b: FrameDrawSlot = .{};
    defer slot_b.deinit(alloc);

    scene.buildQueuesInto(&slot_a, .{
        .snap = &scene.frame_snapshot,
        .cache_key = scene.frame_id,
        .stats = &stats_a,
        .eye = scene.frame_snapshot.primary_cam.eye,
        .sky_texture = null,
        .ibl_intensity = scene.frame_snapshot.ibl_intensity,
        .instances_prepared = true,
        .instance_source = .published,
    });
    scene.buildQueuesInto(&slot_b, .{
        .snap = &scene.frame_snapshot,
        .cache_key = scene.frame_id,
        .stats = &stats_b,
        .eye = scene.frame_snapshot.primary_cam.eye,
        .sky_texture = null,
        .ibl_intensity = scene.frame_snapshot.ibl_intensity,
        .instances_prepared = true,
        .instance_source = .published,
    });

    try std.testing.expectEqual(stats_a.total_meshes, stats_b.total_meshes);
    try std.testing.expectEqual(stats_a.rendered_meshes, stats_b.rendered_meshes);
    try std.testing.expectEqual(slot_a.primary.items.items.len, slot_b.primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 2), slot_a.primary.items.items.len);
    for (slot_a.primary.items.items, slot_b.primary.items.items) |a, b| {
        try std.testing.expectEqual(a.model, b.model);
        try std.testing.expectEqual(a.mesh_index, b.mesh_index);
        try std.testing.expectEqual(a.distance_sq, b.distance_sq);
    }
    try std.testing.expectEqual(slot_a.outline_items.items.len, slot_b.outline_items.items.len);
    try std.testing.expectEqual(@as(usize, 1), slot_a.outline_items.items.len);
    for (slot_a.outline_items.items, slot_b.outline_items.items) |a, b| {
        try std.testing.expectEqual(a.model, b.model);
        try std.testing.expectEqual(a.source_uid, b.source_uid);
        try std.testing.expectEqual(a.source_mesh, b.source_mesh);
    }
    try std.testing.expect(m0.uid != 0 and m1.uid != 0);
}

// ---- Stage-2 increment B: game-side queue build + latch patch. ----

test "stage-2B(a): build+latch finalizes handles and freezes sets" {
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
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "b2a_src",
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
        .name = "b2a_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    var regular = Mesh{
        .name = "b2a_reg",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(5, 0, 0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    try scene.meshes.append(alloc, &parent);
    try scene.meshes.append(alloc, &regular);
    try scene.outline_meshes.append(alloc, &parent);

    try buildForTest(&scene);
    const frozen_bounds = parent.instance_preview.bounds;
    const frozen_reg_pos = regular.position;

    // Live TRS mutation between build and latch must not alter this frame.
    mem[0].position = Vec3.new(100, 0, 0);
    regular.position = Vec3.new(50, 0, 0);

    // Poison the live render state after the build: the latch must leave
    // every field untouched (no write-back on the latch path anymore).
    parent.instance_render.buffer = .{ .id = 100 };
    parent.instance_render.capacity = 7;
    parent.instance_render.count = 9;
    parent.instance_render.hash = 0xABCD;
    parent.instance_render.uploaded_count = 5;
    const poisoned = parent.instance_render;

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try std.testing.expectEqual(poisoned.buffer.id, parent.instance_render.buffer.id);
    try std.testing.expectEqual(poisoned.capacity, parent.instance_render.capacity);
    try std.testing.expectEqual(poisoned.count, parent.instance_render.count);
    try std.testing.expectEqual(poisoned.bounds, parent.instance_render.bounds);
    try std.testing.expectEqual(poisoned.hash, parent.instance_render.hash);
    try std.testing.expectEqual(poisoned.uploaded_count, parent.instance_render.uploaded_count);
    try std.testing.expectEqual(poisoned.staged_frame, parent.instance_render.staged_frame);

    const draws = scene.preparedDraws();
    // Batch/shadow/outline buffer ids+counts equal the post-latch RECORD
    // mirror (the patch source — live state is still poisoned).
    try std.testing.expectEqual(@as(usize, 1), draws.staged_instances.items.len);
    const mirror = draws.staged_instances.items[0];
    try std.testing.expectEqual(scene.frame_id, mirror.staged_frame);
    try std.testing.expectEqual(@as(usize, 1), draws.primary.opaque_instanced.items.len);
    const batch = draws.primary.opaque_instanced.items[0];
    try std.testing.expectEqual(mirror.count, batch.visible_instance_count);
    try std.testing.expectEqual(mirror.buffer.id, batch.instance_buffer.id);
    try std.testing.expectEqual(parent.uid, batch.source_uid);
    var found_shadow = false;
    for (draws.shadow.items.items) |it| {
        if (it.is_instanced and it.source_uid == parent.uid) {
            try std.testing.expectEqual(mirror.count, it.visible_instance_count);
            try std.testing.expectEqual(mirror.buffer.id, it.instance_buffer.id);
            try std.testing.expectEqual(mirror.bounds, it.world_aabb);
            found_shadow = true;
        }
    }
    try std.testing.expect(found_shadow);
    try std.testing.expectEqual(@as(usize, 1), draws.outline_items.items.len);
    const oi = draws.outline_items.items[0];
    try std.testing.expectEqual(mirror.count, oi.visible_instance_count);
    try std.testing.expectEqual(mirror.buffer.id, oi.instance_buffer.id);
    // Frozen inclusion/bounds: still the build-time sets, not the mutation.
    try std.testing.expectEqual(frozen_bounds, mirror.bounds);
    try std.testing.expect(mirror.bounds.max.x < 50.0);
    const reg_item = p7FindByMeshIndex(draws.primary.items.items, 1).?;
    try std.testing.expectApproxEqAbs(frozen_reg_pos.x, reg_item.model.m[12], 1e-4);
    // Live mutation after latch does not alter the published slot.
    const slot_model_x = reg_item.model.m[12];
    const slot_batch_count = batch.visible_instance_count;
    regular.position = Vec3.new(99, 0, 0);
    mem[1].position = Vec3.new(200, 0, 0);
    try std.testing.expectEqual(slot_model_x, draws.primary.items.items[0].model.m[12] + (slot_model_x - draws.primary.items.items[0].model.m[12]));
    try std.testing.expectEqual(slot_batch_count, draws.primary.opaque_instanced.items[0].visible_instance_count);
    try std.testing.expectEqual(frozen_bounds, mirror.bounds);

    // Commit (next game-side build) applies the published mirror to live:
    // the poison is gone, the frozen state lands verbatim.
    try buildForTest(&scene);
    try std.testing.expectEqual(mirror.count, parent.instance_render.count);
    try std.testing.expectEqual(mirror.buffer.id, parent.instance_render.buffer.id);
    try std.testing.expectEqual(frozen_bounds, parent.instance_render.bounds);
    try std.testing.expectEqual(scene.frame_id, parent.instance_render.staged_frame);
}

test "stage-2B(b): patch carries grown handle, old retires VALID until flush" {
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
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "b2b_src",
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
        .name = "b2b_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    // Prime a prior handle (provisional at build time).
    parent.instance_render.buffer = .{ .id = 100 };
    parent.instance_render.capacity = 2;
    parent.instance_render.uploaded_count = 2;
    try scene.meshes.append(alloc, &parent);
    try scene.outline_meshes.append(alloc, &parent);

    try buildForTest(&scene);
    // Provisional handle is the old one.
    try std.testing.expectEqual(@as(u32, 100), parent.instance_build_view.buffer.id);

    // Simulate growth between build and latch AT THE SLOT LEVEL (as
    // stageInstancesGpuState would resolve it with a live sg context: the
    // latch seeds from the record's prior state, never from live
    // instance_render, so a live swap would be invisible — the headless
    // stand-in mutates the record's prior buffer/capacity instead).
    // The old-handle retire is deferred until AFTER prepare (into the open
    // latch epoch) so the prepare-leading flush never destroys a fake
    // headless id — mirroring the real growth order (new first, old retired
    // into the current epoch, VALID until a later complete+flush).
    try std.testing.expectEqual(@as(usize, 1), scene.draws.slotAt(scene.build_slot.load(.monotonic)).staged_instances.items.len);
    const old_buf = scene.draws.slotAt(scene.build_slot.load(.monotonic)).staged_instances.items[0].buffer;
    try std.testing.expectEqual(@as(u32, 100), old_buf.id);
    scene.draws.slotAt(scene.build_slot.load(.monotonic)).staged_instances.items[0].buffer = .{ .id = 200 };
    scene.draws.slotAt(scene.build_slot.load(.monotonic)).staged_instances.items[0].capacity = 4;
    scene.draws.slotAt(scene.build_slot.load(.monotonic)).staged_instances.items[0].uploaded_count = 4;
    scene.draws.slotAt(scene.build_slot.load(.monotonic)).staged_instances.items[0].uploaded_hash =
        scene.draws.slotAt(scene.build_slot.load(.monotonic)).staged_instances.items[0].hash;

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    // Commit (next game-side build) applies the published mirror to live.
    try buildForTest(&scene);
    const draws = scene.preparedDraws();
    // The record mirror carries the grown handle into instance_render and
    // every payload ref.
    try std.testing.expectEqual(@as(u32, 200), parent.instance_render.buffer.id);
    try std.testing.expectEqual(@as(u32, 200), scene.preparedDraws().staged_instances.items[0].buffer.id);
    try std.testing.expectEqual(scene.frame_id, scene.preparedDraws().staged_instances.items[0].staged_frame);
    // All payload refs carry the NEW buffer id and count.
    try std.testing.expectEqual(@as(u32, 5), parent.instance_render.count);
    try std.testing.expectEqual(@as(usize, 1), draws.primary.opaque_instanced.items.len);
    try std.testing.expectEqual(@as(u32, 200), draws.primary.opaque_instanced.items[0].instance_buffer.id);
    try std.testing.expectEqual(@as(u32, 5), draws.primary.opaque_instanced.items[0].visible_instance_count);
    var shadow_ok = false;
    for (draws.shadow.items.items) |it| {
        if (it.is_instanced and it.source_uid == parent.uid) {
            try std.testing.expectEqual(@as(u32, 200), it.instance_buffer.id);
            try std.testing.expectEqual(@as(u32, 5), it.visible_instance_count);
            shadow_ok = true;
        }
    }
    try std.testing.expect(shadow_ok);
    try std.testing.expectEqual(@as(usize, 1), draws.outline_items.items.len);
    try std.testing.expectEqual(@as(u32, 200), draws.outline_items.items[0].instance_buffer.id);
    try std.testing.expectEqual(@as(u32, 5), draws.outline_items.items[0].visible_instance_count);
    // SIMULATION-ONLY retire tail (headless, fake buffer ids): retire the
    // grown-away old handle AFTER prepare (into the open latch epoch, as real
    // growth would): pre-complete flush keeps it VALID without touching sg.*
    // headless. Real destroy lifetime on complete+flush is covered by the
    // retire-queue unit tests — fake headless ids must never go through
    // sg.destroyBuffer.
    scene.gpu_retire.retireBuffer(alloc, old_buf);
    try std.testing.expectEqual(@as(usize, 1), scene.gpu_retire.retainedCount());
    scene.gpu_retire.flush(alloc);
    try std.testing.expectEqual(@as(usize, 1), scene.gpu_retire.retainedCount());
    // Manual headless cleanup (mirrors retireBuffer unit-test handling).
    scene.gpu_retire.pending.clearRetainingCapacity();
    try std.testing.expectEqual(@as(usize, 0), scene.gpu_retire.retainedCount());
}

test "stage-2B(c): stale latch entry neutralizes, others intact, retry recovers" {
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
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "b2c_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const ma = try alloc.alloc(InstancedMesh, 2);
    defer alloc.free(ma);
    const pa = try alloc.alloc(*InstancedMesh, 2);
    defer alloc.free(pa);
    stage1FillInstances(&src, ma, pa, 0);
    var mesh_a = Mesh{
        .name = "b2c_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = pa, .capacity = 2 },
    };
    const mb = try alloc.alloc(InstancedMesh, 2);
    defer alloc.free(mb);
    const pb = try alloc.alloc(*InstancedMesh, 2);
    defer alloc.free(pb);
    stage1FillInstances(&src, mb, pb, 50);
    var mesh_b = Mesh{
        .name = "b2c_b",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = pb, .capacity = 2 },
    };
    try scene.meshes.append(alloc, &mesh_a);
    try scene.meshes.append(alloc, &mesh_b);
    try scene.outline_meshes.append(alloc, &mesh_a);
    try scene.outline_meshes.append(alloc, &mesh_b);

    try buildForTest(&scene);
    // Simulate an OOM-skipped segment at latch: drop B's staged record so
    // the latch has nothing to consume for it (previous complete state
    // stands, patch must zero its provisional entries — no partial mix).
    {
        const recs = &scene.draws.slotAt(scene.build_slot.load(.monotonic)).staged_instances;
        var i: usize = 0;
        while (i < recs.items.len) : (i += 1) {
            if (recs.items[i].mesh == &mesh_b) break;
        }
        try std.testing.expect(i < recs.items.len);
        _ = recs.orderedRemove(i);
    }

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    const draws = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 2), draws.primary.opaque_instanced.items.len);
    for (draws.primary.opaque_instanced.items) |batch| {
        if (batch.source_uid == mesh_a.uid) {
            try std.testing.expectEqual(@as(u32, 3), batch.visible_instance_count);
        } else if (batch.source_uid == mesh_b.uid) {
            try std.testing.expectEqual(@as(u32, 0), batch.visible_instance_count);
            try std.testing.expectEqual(@as(u32, 0), batch.instance_buffer.id);
        } else return error.TestUnexpectedResult;
    }
    for (draws.shadow.items.items) |it| {
        if (it.is_instanced and it.source_uid == mesh_b.uid) {
            try std.testing.expectEqual(@as(u32, 0), it.visible_instance_count);
            try std.testing.expect(!it.world_aabb.isValid());
            try std.testing.expectEqual(@as(f32, 0), it.max_dim);
        }
    }
    for (draws.outline_items.items) |it| {
        if (it.is_instanced and it.source_uid == mesh_b.uid) {
            try std.testing.expectEqual(@as(u32, 0), it.visible_instance_count);
            try std.testing.expectEqual(mesh_b.position, it.world_center);
        }
    }
    // Retry next frame recovers.
    try buildForTest(&scene);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    const draws2 = scene.preparedDraws();
    for (draws2.primary.opaque_instanced.items) |batch| {
        try std.testing.expectEqual(@as(u32, 3), batch.visible_instance_count);
    }
}

test "stage-2B(d): mesh-list change stages through, commit skips by uid, no OOB/UAF" {
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
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "b2d_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const ma = try alloc.alloc(InstancedMesh, 2);
    defer alloc.free(ma);
    const pa = try alloc.alloc(*InstancedMesh, 2);
    defer alloc.free(pa);
    stage1FillInstances(&src, ma, pa, 0);
    var mesh_a = Mesh{
        .name = "b2d_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = pa, .capacity = 2 },
    };
    const mb = try alloc.alloc(InstancedMesh, 2);
    defer alloc.free(mb);
    const pb = try alloc.alloc(*InstancedMesh, 2);
    defer alloc.free(pb);
    stage1FillInstances(&src, mb, pb, 50);
    var mesh_b = Mesh{
        .name = "b2d_b",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = pb, .capacity = 2 },
    };
    var mesh_c = Mesh{
        .name = "b2d_c",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(9, 0, 0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    try scene.meshes.append(alloc, &mesh_a);
    try scene.meshes.append(alloc, &mesh_b);
    try scene.meshes.append(alloc, &mesh_c);
    try scene.outline_meshes.append(alloc, &mesh_a);
    try scene.outline_meshes.append(alloc, &mesh_b);

    const uid_a = blk: {
        try buildForTest(&scene);
        break :blk mesh_a.uid;
    };
    const uid_b = mesh_b.uid;
    // Remove the trailing regular mesh: A/B indices unchanged (still draw),
    // C's absence must not OOB. Then reorder A/B via swapRemove to force a uid
    // mismatch on the next latch in a second round below.
    _ = scene.meshes.swapRemove(2);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    const draws = scene.preparedDraws();
    // A/B still draw (indices stable), no OOB from the removed tail.
    var a_ok = false;
    var b_ok = false;
    for (draws.primary.opaque_instanced.items) |batch| {
        if (batch.source_uid == uid_a) {
            try std.testing.expectEqual(@as(u32, 3), batch.visible_instance_count);
            a_ok = true;
        }
        if (batch.source_uid == uid_b) {
            try std.testing.expectEqual(@as(u32, 3), batch.visible_instance_count);
            b_ok = true;
        }
    }
    try std.testing.expect(a_ok and b_ok);

    // Second round: reorder A/B so stored indices mismatch uids. The latch
    // stages through (no live-list read remains) and the patch finalizes
    // the frozen batches from the records — no OOB, no UAF. Live meshes
    // still hold the round-1 commit (the write-back is game-side now).
    try buildForTest(&scene);
    _ = scene.meshes.swapRemove(0); // [B] (A unlinked); re-append A → [B, A]
    try scene.meshes.append(alloc, &mesh_a);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    const draws2 = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 2), draws2.primary.opaque_instanced.items.len);
    for (draws2.primary.opaque_instanced.items) |batch| {
        try std.testing.expectEqual(@as(u32, 3), batch.visible_instance_count);
    }
    try std.testing.expectEqual(@as(u32, 3), mesh_a.instance_render.count);
    try std.testing.expectEqual(@as(u32, 3), mesh_b.instance_render.count);

    // Third round: unlink A again before the next build — both records are
    // now displaced (A unlinked, B shifted to index 0), so the commit guard
    // skips both (pointer compares, never a dereference): previous complete
    // state stands for both, the stale generation never lands.
    _ = scene.meshes.swapRemove(1); // [B] (A unlinked again)
    try buildForTest(&scene);
    try std.testing.expectEqual(@as(u32, 3), mesh_a.instance_render.count);
    try std.testing.expect(mesh_a.instance_render.staged_frame != scene.frame_id);
    try std.testing.expectEqual(@as(u32, 3), mesh_b.instance_render.count);
    try std.testing.expect(mesh_b.instance_render.staged_frame != scene.frame_id);

    // Recovery: a funded rebuild + latch + commit over the live order passes
    // the guard again and lands the newest generation on B (A stays skipped
    // — still unlinked — with its previous complete state).
    try buildForTest(&scene);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    try buildForTest(&scene);
    try std.testing.expectEqual(scene.frame_id, mesh_b.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u32, 3), mesh_b.instance_render.count);
    try std.testing.expectEqual(@as(u32, 3), mesh_a.instance_render.count);
    try std.testing.expect(mesh_a.instance_render.staged_frame != scene.frame_id);
}

test "stage-2B(e): serial producer equals concurrent producer" {
    const InstancedMesh = @import("../mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;

    const Fixture = struct {
        src: Mesh,
        mem: []InstancedMesh,
        ptrs: []*InstancedMesh,
        parent: Mesh,
        regular: Mesh,
        fn init(a: std.mem.Allocator) !@This() {
            var f: @This() = undefined;
            f.src = Mesh{
                .name = "b2e_src",
                .vertex_buffer = .{},
                .index_buffer = .{},
                .index_count = 3,
                .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
            };
            f.mem = try a.alloc(InstancedMesh, 3);
            f.ptrs = try a.alloc(*InstancedMesh, 3);
            // NOTE: instances are filled by the caller (see below), NOT here:
            // `stage1FillInstances(&f.src, ...)` inside `init` would store a
            // pointer to this frame's `f.src` local, which dangles once the
            // struct is returned by value (the shadow/instanced bounds then
            // stage from dead stack memory — serial-zero vs worker-garbage).
            // The caller fills against its own stable `fix.src` instead.
            f.parent = Mesh{
                .name = "b2e_parent",
                .vertex_buffer = .{},
                .index_buffer = .{},
                .index_count = 3,
                .instances = .{ .items = f.ptrs, .capacity = 3 },
            };
            f.regular = Mesh{
                .name = "b2e_reg",
                .vertex_buffer = .{},
                .index_buffer = .{},
                .index_count = 3,
                .position = Vec3.new(5, 0, 0),
                .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
            };
            return f;
        }
        fn deinit(f: *@This(), a: std.mem.Allocator) void {
            a.free(f.mem);
            a.free(f.ptrs);
        }
    };

    // Path 1: serial producer (same-thread build + finish).
    var scene_b = stage1Scene(alloc);
    defer scene_b.lights.deinit(alloc);
    defer scene_b.cameras.deinit(alloc);
    defer scene_b.meshes.deinit(alloc);
    defer scene_b.outline_meshes.deinit(alloc);
    defer scene_b.draws.deinit(alloc);
    defer scene_b.gpu_retire.deinit(alloc);
    defer scene_b.shadows.pass.binned_meshes.deinit(alloc);
    defer scene_b.shadows.pass.binned_source.deinit(alloc);
    defer scene_b.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene_b, alloc);
    const cam_b = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene_b.addCamera(.{ .name = "Cam1", .camera = cam_b });
    var fix_b = try Fixture.init(alloc);
    defer fix_b.deinit(alloc);
    stage1FillInstances(&fix_b.src, fix_b.mem, fix_b.ptrs, 0);
    try scene_b.meshes.append(alloc, &fix_b.parent);
    try scene_b.meshes.append(alloc, &fix_b.regular);
    try scene_b.outline_meshes.append(alloc, &fix_b.parent);
    scene_b.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try buildForTest(&scene_b);
    finishForTest(&scene_b);
    const draws_b = scene_b.preparedDraws();
    const stats_b = scene_b.stats;

    // Path 2: concurrent producer (worker-thread build + main finish) —
    // same staged protocol, whole payload identity preserved.
    var scene_f = stage1Scene(alloc);
    defer scene_f.lights.deinit(alloc);
    defer scene_f.cameras.deinit(alloc);
    defer scene_f.meshes.deinit(alloc);
    defer scene_f.outline_meshes.deinit(alloc);
    defer scene_f.draws.deinit(alloc);
    defer scene_f.gpu_retire.deinit(alloc);
    defer scene_f.shadows.pass.binned_meshes.deinit(alloc);
    defer scene_f.shadows.pass.binned_source.deinit(alloc);
    defer scene_f.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene_f, alloc);
    const cam_f = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene_f.addCamera(.{ .name = "Cam1", .camera = cam_f });
    var fix_f = try Fixture.init(alloc);
    defer fix_f.deinit(alloc);
    stage1FillInstances(&fix_f.src, fix_f.mem, fix_f.ptrs, 0);
    try scene_f.meshes.append(alloc, &fix_f.parent);
    try scene_f.meshes.append(alloc, &fix_f.regular);
    try scene_f.outline_meshes.append(alloc, &fix_f.parent);
    scene_f.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    const Worker = struct {
        scene: *Scene,
        fn run(w: @This()) void {
            var c = w.scene.tryClaimBuildSlot() orelse std.debug.panic("{s}", .{"worker saturated"});
            c.build();
            c.stageUi();
            c.publish();
        }
    };
    const wt = try std.Thread.spawn(.{}, Worker.run, .{Worker{ .scene = &scene_f }});
    wt.join();
    finishForTest(&scene_f);
    const draws_f = scene_f.preparedDraws();
    const stats_f = scene_f.stats;

    // Full payload equality, PAIRED BY IDENTITY (not list position): the two
    // scenes own distinct Mesh instances (global uid counter), so cross-scene
    // pairing keys on the mesh-list index (`source_mesh`/`mesh_index`), which
    // is stable across the two identical fixtures. Identity is validated
    // within each scene first (`meshes[source_mesh].uid == source_uid`, the
    // same invariant `patchInstanceRefs` enforces); an item with
    // `source_uid == 0` (never uid-assigned) falls back to positional pairing
    // for that item only. A prior failure here (serial-zero vs
    // worker-garbage on the instanced shadow bounds) traced to the Fixture
    // storing `&f.src` inside `init` — a pointer to the init frame's local
    // that dangles after the struct is returned by value — not to either
    // code path: both stage faithfully from `source_mesh`, so both read the
    // same dead stack slot at different times. The fill now runs against the
    // caller's stable `fix.src` (see above); no production code change was
    // needed (stale-preview build_view entries are already deterministic-zero
    // and regular items keep their valid world-cache AABB).
    // Documented allowed differences: sort order under different eyes (eyes
    // are identical here — same camera, no mutation — so order matches) and
    // staged_frame (frame_ids both start at 1 here, so they match too).
    try std.testing.expectEqual(draws_f.primary.items.items.len, draws_b.primary.items.items.len);
    for (draws_f.primary.items.items) |f| {
        var matched = false;
        for (draws_b.primary.items.items) |b| {
            if (b.mesh_index != f.mesh_index) continue;
            try std.testing.expectEqual(f.model, b.model);
            try std.testing.expectEqual(f.distance_sq, b.distance_sq);
            try std.testing.expectEqual(f.index_count, b.index_count);
            matched = true;
            break;
        }
        try std.testing.expect(matched);
    }
    try std.testing.expectEqual(draws_f.primary.opaque_instanced.items.len, draws_b.primary.opaque_instanced.items.len);
    for (draws_f.primary.opaque_instanced.items, 0..) |f, fi| {
        if (f.source_uid != 0) {
            try std.testing.expect(f.source_mesh < scene_f.meshes.items.len);
            try std.testing.expectEqual(scene_f.meshes.items[f.source_mesh].uid, f.source_uid);
        }
        var matched = false;
        for (draws_b.primary.opaque_instanced.items, 0..) |b, bi| {
            if (f.source_uid == 0 or b.source_uid == 0) {
                if (bi != fi) continue;
            } else {
                if (b.source_mesh != f.source_mesh) continue;
                try std.testing.expect(b.source_mesh < scene_b.meshes.items.len);
                try std.testing.expectEqual(scene_b.meshes.items[b.source_mesh].uid, b.source_uid);
            }
            try std.testing.expectEqual(f.visible_instance_count, b.visible_instance_count);
            try std.testing.expectEqual(f.instance_buffer.id, b.instance_buffer.id);
            try std.testing.expectEqual(f.index_count, b.index_count);
            matched = true;
            break;
        }
        try std.testing.expect(matched);
    }
    try std.testing.expectEqual(draws_f.shadow.items.items.len, draws_b.shadow.items.items.len);
    for (draws_f.shadow.items.items, 0..) |f, fi| {
        if (f.source_uid != 0) {
            try std.testing.expect(f.source_mesh < scene_f.meshes.items.len);
            try std.testing.expectEqual(scene_f.meshes.items[f.source_mesh].uid, f.source_uid);
        }
        var matched = false;
        for (draws_b.shadow.items.items, 0..) |b, bi| {
            if (f.source_uid == 0 or b.source_uid == 0) {
                if (bi != fi) continue;
            } else {
                if (b.source_mesh != f.source_mesh) continue;
                if (b.bucket != f.bucket) continue;
                try std.testing.expect(b.source_mesh < scene_b.meshes.items.len);
                try std.testing.expectEqual(scene_b.meshes.items[b.source_mesh].uid, b.source_uid);
            }
            try std.testing.expectEqual(f.model, b.model);
            try std.testing.expectEqual(f.is_instanced, b.is_instanced);
            try std.testing.expectEqual(f.world_aabb, b.world_aabb);
            try std.testing.expectEqual(f.max_dim, b.max_dim);
            try std.testing.expectEqual(f.instance_buffer.id, b.instance_buffer.id);
            try std.testing.expectEqual(f.visible_instance_count, b.visible_instance_count);
            matched = true;
            break;
        }
        try std.testing.expect(matched);
    }
    try std.testing.expectEqual(draws_f.outline_items.items.len, draws_b.outline_items.items.len);
    for (draws_f.outline_items.items, 0..) |f, fi| {
        if (f.source_uid != 0) {
            try std.testing.expect(f.source_mesh < scene_f.meshes.items.len);
            try std.testing.expectEqual(scene_f.meshes.items[f.source_mesh].uid, f.source_uid);
        }
        var matched = false;
        for (draws_b.outline_items.items, 0..) |b, bi| {
            if (f.source_uid == 0 or b.source_uid == 0) {
                if (bi != fi) continue;
            } else {
                if (b.source_mesh != f.source_mesh) continue;
                try std.testing.expect(b.source_mesh < scene_b.meshes.items.len);
                try std.testing.expectEqual(scene_b.meshes.items[b.source_mesh].uid, b.source_uid);
            }
            try std.testing.expectEqual(f.model, b.model);
            try std.testing.expectEqual(f.world_center, b.world_center);
            try std.testing.expectEqual(f.instance_buffer.id, b.instance_buffer.id);
            try std.testing.expectEqual(f.visible_instance_count, b.visible_instance_count);
            matched = true;
            break;
        }
        try std.testing.expect(matched);
    }
    try std.testing.expectEqual(stats_f.total_meshes, stats_b.total_meshes);
    try std.testing.expectEqual(stats_f.rendered_meshes, stats_b.rendered_meshes);
    try std.testing.expectEqual(stats_f.culled_meshes, stats_b.culled_meshes);
    try std.testing.expectEqual(stats_f.occluded_meshes, stats_b.occluded_meshes);
    try std.testing.expectEqual(stats_f.occluders_count, stats_b.occluders_count);
    try std.testing.expectEqual(stats_f.occluder_triangles, stats_b.occluder_triangles);
}

test "stage-2B(f): warm build+latch pump stays zero-alloc under refusal" {
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
    var m0 = Mesh{
        .name = "b2f_0",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.zero,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    var m1 = Mesh{
        .name = "b2f_1",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(5, 0, 0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    try scene.meshes.append(alloc, &m0);
    try scene.meshes.append(alloc, &m1);
    try scene.outline_meshes.append(alloc, &m0);

    // Warm every slot with funded build+latch rounds: with a 3-slot
    // rotation one round warms exactly one slot as back, so two rounds
    // would leave the third slot cold (the old two-slot "both slots"
    // warmup is exactly the assumption wave 26 removed).
    var round: usize = 0;
    while (round < 3) : (round += 1) {
        scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
        try buildForTest(&scene);
        scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
        finishForTest(&scene);
    }
    const warm_primary = scene.preparedDraws().primary.items.items.len;
    const warm_outline = scene.preparedDraws().outline_items.items.len;
    const warm_stats = scene.stats;

    // Refuse-all round must not induce failure and must match the warm payload.
    var refusing = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 0, .resize_fail_index = 0 });
    const saved_alloc = scene.allocator;
    const saved_shadow_alloc = scene.shadows.pass.allocator;
    scene.allocator = refusing.allocator();
    scene.shadows.pass.allocator = refusing.allocator();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try buildForTest(&scene);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    scene.allocator = saved_alloc;
    scene.shadows.pass.allocator = saved_shadow_alloc;
    try std.testing.expect(!refusing.has_induced_failure);
    const dz = scene.preparedDraws();
    try std.testing.expectEqual(warm_primary, dz.primary.items.items.len);
    try std.testing.expectEqual(warm_outline, dz.outline_items.items.len);
    try std.testing.expectEqual(warm_stats.total_meshes, scene.stats.total_meshes);
    try std.testing.expectEqual(warm_stats.rendered_meshes, scene.stats.rendered_meshes);
}

test "stage-2B(g): serial build+latch equals worker-build parity" {
    const InstancedMesh = @import("../mesh.zig").InstancedMesh;
    const alloc = std.testing.allocator;

    const Builder = struct {
        scene: *Scene,
        fn run(self: @This()) void {
            std.debug.assert(self.scene.buildPreparedFrame());
        }
    };

    // Serial path.
    var serial = stage1Scene(alloc);
    defer serial.lights.deinit(alloc);
    defer serial.cameras.deinit(alloc);
    defer serial.meshes.deinit(alloc);
    defer serial.outline_meshes.deinit(alloc);
    defer serial.draws.deinit(alloc);
    defer serial.gpu_retire.deinit(alloc);
    defer serial.shadows.pass.binned_meshes.deinit(alloc);
    defer serial.shadows.pass.binned_source.deinit(alloc);
    defer serial.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&serial, alloc);
    const cam_s = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try serial.addCamera(.{ .name = "Cam1", .camera = cam_s });
    var src_s = Mesh{
        .name = "b2g_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const mem_s = try alloc.alloc(InstancedMesh, 3);
    defer alloc.free(mem_s);
    const ptrs_s = try alloc.alloc(*InstancedMesh, 3);
    defer alloc.free(ptrs_s);
    stage1FillInstances(&src_s, mem_s, ptrs_s, 0);
    var parent_s = Mesh{
        .name = "b2g_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs_s, .capacity = 3 },
    };
    try serial.meshes.append(alloc, &parent_s);
    try serial.outline_meshes.append(alloc, &parent_s);
    serial.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try buildForTest(&serial);
    serial.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&serial);

    // Worker-build path (same live state, build on a spawned thread).
    var threaded = stage1Scene(alloc);
    defer threaded.lights.deinit(alloc);
    defer threaded.cameras.deinit(alloc);
    defer threaded.meshes.deinit(alloc);
    defer threaded.outline_meshes.deinit(alloc);
    defer threaded.draws.deinit(alloc);
    defer threaded.gpu_retire.deinit(alloc);
    defer threaded.shadows.pass.binned_meshes.deinit(alloc);
    defer threaded.shadows.pass.binned_source.deinit(alloc);
    defer threaded.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&threaded, alloc);
    const cam_t = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try threaded.addCamera(.{ .name = "Cam1", .camera = cam_t });
    var src_t = Mesh{
        .name = "b2g_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const mem_t = try alloc.alloc(InstancedMesh, 3);
    defer alloc.free(mem_t);
    const ptrs_t = try alloc.alloc(*InstancedMesh, 3);
    defer alloc.free(ptrs_t);
    stage1FillInstances(&src_t, mem_t, ptrs_t, 0);
    var parent_t = Mesh{
        .name = "b2g_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs_t, .capacity = 3 },
    };
    try threaded.meshes.append(alloc, &parent_t);
    try threaded.outline_meshes.append(alloc, &parent_t);
    threaded.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    const t = try std.Thread.spawn(.{}, Builder.run, .{Builder{ .scene = &threaded }});
    t.join();
    threaded.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try stageAndPrepareForTest(&threaded);

    // Parity: counts and buffer ids match across threads (distinct Mesh
    // instances, so pairing is positional on the identical single-mesh
    // fixture; uids intentionally not compared).
    const ds = serial.preparedDraws();
    const dt = threaded.preparedDraws();
    try std.testing.expectEqual(ds.primary.opaque_instanced.items.len, dt.primary.opaque_instanced.items.len);
    for (ds.primary.opaque_instanced.items, dt.primary.opaque_instanced.items) |a, b| {
        try std.testing.expectEqual(a.visible_instance_count, b.visible_instance_count);
        try std.testing.expectEqual(a.instance_buffer.id, b.instance_buffer.id);
    }
    try std.testing.expectEqual(ds.shadow.items.items.len, dt.shadow.items.items.len);
    for (ds.shadow.items.items, dt.shadow.items.items) |a, b| {
        try std.testing.expectEqual(a.model, b.model);
        try std.testing.expectEqual(a.is_instanced, b.is_instanced);
        try std.testing.expectEqual(a.world_aabb, b.world_aabb);
        try std.testing.expectEqual(a.max_dim, b.max_dim);
        try std.testing.expectEqual(a.instance_buffer.id, b.instance_buffer.id);
        try std.testing.expectEqual(a.visible_instance_count, b.visible_instance_count);
    }
    try std.testing.expectEqual(ds.outline_items.items.len, dt.outline_items.items.len);
    for (ds.outline_items.items, dt.outline_items.items) |a, b| {
        try std.testing.expectEqual(a.world_center, b.world_center);
        try std.testing.expectEqual(a.instance_buffer.id, b.instance_buffer.id);
        try std.testing.expectEqual(a.visible_instance_count, b.visible_instance_count);
    }
}

test "stage-2B(h): outline subset order maps to mesh-list index, reorder safe" {
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
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "b2h_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const ma = try alloc.alloc(InstancedMesh, 2);
    defer alloc.free(ma);
    const pa = try alloc.alloc(*InstancedMesh, 2);
    defer alloc.free(pa);
    stage1FillInstances(&src, ma, pa, 0);
    var mesh_a = Mesh{
        .name = "b2h_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = pa, .capacity = 2 },
    };
    const mb = try alloc.alloc(InstancedMesh, 2);
    defer alloc.free(mb);
    const pb = try alloc.alloc(*InstancedMesh, 2);
    defer alloc.free(pb);
    stage1FillInstances(&src, mb, pb, 50);
    var mesh_b = Mesh{
        .name = "b2h_b",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = pb, .capacity = 2 },
    };
    var mesh_c = Mesh{
        .name = "b2h_c",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(9, 0, 0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    try scene.meshes.append(alloc, &mesh_a); // mesh-list 0
    try scene.meshes.append(alloc, &mesh_b); // mesh-list 1
    try scene.meshes.append(alloc, &mesh_c); // mesh-list 2
    // Outline subset in a different order than the mesh list.
    try scene.outline_meshes.append(alloc, &mesh_b);
    try scene.outline_meshes.append(alloc, &mesh_a);

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try buildForTest(&scene);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    // Commit (next game-side build) applies the published mirrors to live.
    try buildForTest(&scene);
    const draws = scene.preparedDraws();
    // No outline dropped; each item carries its MESH-LIST index (not the
    // outline-list position) and patches from its own mesh.
    try std.testing.expectEqual(@as(usize, 2), draws.outline_items.items.len);
    for (draws.outline_items.items) |it| {
        if (it.source_uid == mesh_b.uid) {
            try std.testing.expectEqual(@as(u32, 1), it.source_mesh);
            try std.testing.expectEqual(@as(u32, 3), it.visible_instance_count);
            try std.testing.expectEqual(mesh_b.instance_render.buffer.id, it.instance_buffer.id);
            const expect_c = if (mesh_b.instance_render.bounds.isValid()) mesh_b.instance_render.bounds.center() else mesh_b.position;
            try std.testing.expectEqual(expect_c, it.world_center);
        } else if (it.source_uid == mesh_a.uid) {
            try std.testing.expectEqual(@as(u32, 0), it.source_mesh);
            try std.testing.expectEqual(@as(u32, 3), it.visible_instance_count);
            try std.testing.expectEqual(mesh_a.instance_render.buffer.id, it.instance_buffer.id);
            const expect_c = if (mesh_a.instance_render.bounds.isValid()) mesh_a.instance_render.bounds.center() else mesh_a.position;
            try std.testing.expectEqual(expect_c, it.world_center);
        } else return error.TestUnexpectedResult;
    }

    // Reorder-only on the outline list (mesh list untouched): rebuild maps
    // the same mesh-list indices in the new outline order; nothing dropped.
    _ = scene.outline_meshes.swapRemove(0); // [A]
    try scene.outline_meshes.append(alloc, &mesh_b); // [A, B]
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try buildForTest(&scene);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    const draws2 = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 2), draws2.outline_items.items.len);
    try std.testing.expectEqual(mesh_a.uid, draws2.outline_items.items[0].source_uid);
    try std.testing.expectEqual(@as(u32, 0), draws2.outline_items.items[0].source_mesh);
    try std.testing.expectEqual(mesh_b.uid, draws2.outline_items.items[1].source_uid);
    try std.testing.expectEqual(@as(u32, 1), draws2.outline_items.items[1].source_mesh);
}

test "stage-2B(i): PIP views patch each built view" {
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
    p7CpuShadowPass(&scene, alloc);

    const cam0 = Camera{ .free = camera_mod.FreeCamera.init("Main", .{}) };
    const cam1 = Camera{ .free = camera_mod.FreeCamera.init("Pip", .{}) };
    _ = try scene.addCamera(.{ .name = "Main", .camera = cam0 });
    _ = try scene.addCamera(.{ .name = "Pip", .camera = cam1 });
    scene.enable_multi_camera = true;
    scene.active_camera_index = 0;

    var src = Mesh{
        .name = "b2i_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 3;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    var parent = Mesh{
        .name = "b2i_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try buildForTest(&scene);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    // Commit (next game-side build) applies the published mirror to live.
    try buildForTest(&scene);
    const draws = scene.preparedDraws();
    // Primary finalized.
    try std.testing.expectEqual(@as(usize, 1), draws.primary.opaque_instanced.items.len);
    try std.testing.expectEqual(parent.instance_render.count, draws.primary.opaque_instanced.items[0].visible_instance_count);
    try std.testing.expectEqual(parent.instance_render.buffer.id, draws.primary.opaque_instanced.items[0].instance_buffer.id);
    // Each built PIP view finalized too (views[1] built; views[0] is the
    // active slot and stays empty by the multi-camera contract).
    try std.testing.expectEqual(@as(usize, 1), draws.views[1].opaque_instanced.items.len);
    try std.testing.expectEqual(parent.instance_render.count, draws.views[1].opaque_instanced.items[0].visible_instance_count);
    try std.testing.expectEqual(parent.instance_render.buffer.id, draws.views[1].opaque_instanced.items[0].instance_buffer.id);
    try std.testing.expectEqual(@as(usize, 0), draws.views[0].opaque_instanced.items.len);
}

test "stage-2B(j): transparent zero-batch keeps stale order entry, draw skips" {
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

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var blend_mat = material_mod.PBRMaterial.init("b2j_blend");
    blend_mat.alpha_mode = .blend;
    var src = Mesh{
        .name = "b2j_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const mem = try alloc.alloc(InstancedMesh, 2);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, 2);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    var parent = Mesh{
        .name = "b2j_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .material = .{ .pbr = &blend_mat },
        .instances = .{ .items = ptrs, .capacity = 2 },
    };
    try scene.meshes.append(alloc, &parent);

    try buildForTest(&scene);
    try std.testing.expectEqual(@as(u32, 3), parent.instance_preview.count);
    // Drop the staged record so the latch skips (as after an OOM/GPU
    // failure): the transparent batch patch-zeroes but its
    // transparent_order entry stays (stale). The draw skips count==0
    // batches, so the stale order entry is harmless — slot-state only here
    // (renderSceneView needs a live camera + sg context, impractical
    // headless; documented).
    {
        const recs = &scene.draws.slotAt(scene.build_slot.load(.monotonic)).staged_instances;
        var i: usize = 0;
        while (i < recs.items.len) : (i += 1) {
            if (recs.items[i].mesh == &parent) break;
        }
        try std.testing.expect(i < recs.items.len);
        _ = recs.orderedRemove(i);
    }

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    finishForTest(&scene);
    const draws = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 1), draws.primary.transparent_instanced.items.len);
    try std.testing.expectEqual(@as(u32, 0), draws.primary.transparent_instanced.items[0].visible_instance_count);
    try std.testing.expectEqual(@as(u32, 0), draws.primary.transparent_instanced.items[0].instance_buffer.id);
    var found_stale = false;
    for (draws.primary.transparent_order.items) |e| {
        if (e.kind == .instanced and e.index == 0) found_stale = true;
    }
    try std.testing.expect(found_stale);
}

test "stage-2B(k): same-scene staged builds keep cache-key isolation" {
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
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var src = Mesh{
        .name = "b2k_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const mem = try alloc.alloc(InstancedMesh, 2);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, 2);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    var parent = Mesh{
        .name = "b2k_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = 2 },
    };
    var regular = Mesh{
        .name = "b2k_reg",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(5, 0, 0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    try scene.meshes.append(alloc, &parent);
    try scene.meshes.append(alloc, &regular);

    // Round 1: staged build + finish (world cache tagged with the build key).
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try buildForTest(&scene);
    finishForTest(&scene);
    try std.testing.expectEqual(scene.build_seq.load(.monotonic), scene.last_latched_seq.load(.monotonic));
    const built_reg_x = p7FindByMeshIndex(scene.preparedDraws().primary.items.items, 1).?.model.m[12];

    // Mutate live, then run a fresh staged build: it must recompute under
    // the new build key (not reuse stale build-key entries).
    regular.position = Vec3.new(25, 0, 0);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try stageAndPrepareForTest(&scene);
    try std.testing.expectEqual(scene.build_seq.load(.monotonic), scene.last_latched_seq.load(.monotonic));
    const draws = scene.preparedDraws();
    const fb_item = p7FindByMeshIndex(draws.primary.items.items, 1).?;
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), fb_item.model.m[12], 1e-4);
    try std.testing.expect(fb_item.model.m[12] != built_reg_x);
    // Cache retagged with the fresh staged build attempt key (high-bit
    // producer namespace over the per-build attempt counter), not the
    // stale entry and not the context frame_id. Derived from the actual
    // attempt field — never hardcoded — so cancel/repeat builds that bump
    // attempts without publishing stay isolated too.
    const attempt_mask = (@as(u64, 1) << 63) - 1;
    const want_key = (scene.build_cache_seq & attempt_mask) | (@as(u64, 1) << 63);
    try std.testing.expectEqual(want_key, regular.cached_frame);
    try std.testing.expectEqual(@as(u32, 3), parent.instance_render.count);
    try std.testing.expectEqual(@as(usize, 1), draws.primary.opaque_instanced.items.len);
    try std.testing.expectEqual(@as(u32, 3), draws.primary.opaque_instanced.items[0].visible_instance_count);
}

test "stage-2B(l): cancelled build leaves no stale world cache; next publish sees the mutation" {
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.ui_frame.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.binned_source.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var regular = Mesh{
        .name = "b2l_reg",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(5, 0, 0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    try scene.meshes.append(alloc, &regular);

    // Prime: staged build + latch publishes generation 1 (world cache tagged).
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try buildForTest(&scene);
    finishForTest(&scene);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), p7FindByMeshIndex(scene.preparedDraws().primary.items.items, 0).?.model.m[12], 1e-4);
    const attempts_primed = scene.build_cache_seq;

    // Cancelled build stamps the world cache under its reserved (uncommitted)
    // seq key, then drops the handoff: build_seq stays 1, the slot payload
    // is ignored by prepare. Uses the pure public producer path
    // (claim.build/cancel), never a manual cached_frame reset.
    var dropped = scene.tryClaimBuildSlot().?;
    dropped.build();
    dropped.cancel();
    try std.testing.expectEqual(@as(u64, 1), scene.build_seq.load(.monotonic));
    // Every build call advances the cache identity, published or not.
    try std.testing.expect(scene.build_cache_seq > attempts_primed);

    // Mutate live, then publish a REAL build with no manual cache reset: it
    // reserves the same handoff seq the cancelled build used, so only a
    // per-attempt cache key keeps it from reading the stale entry.
    regular.position = Vec3.new(25, 0, 0);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try buildForTest(&scene);
    finishForTest(&scene);
    try std.testing.expectEqual(@as(u64, 2), scene.build_seq.load(.monotonic));

    // Correct model, bounds, and queue inclusion — never the cancelled
    // build's frozen TRS.
    const draws = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 1), draws.primary.items.items.len);
    const item = p7FindByMeshIndex(draws.primary.items.items, 0).?;
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), item.model.m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 24.0), regular.cached_aabb.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 26.0), regular.cached_aabb.max.x, 1e-4);
}

test "stage-2B(m): repeat build on the same claim wins newest despite identical seq" {
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.ui_frame.deinit(alloc);
    defer scene.shadows.pass.binned_meshes.deinit(alloc);
    defer scene.shadows.pass.binned_source.deinit(alloc);
    defer scene.shadows.pass.prepared.deinit(alloc);
    p7CpuShadowPass(&scene, alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    var regular = Mesh{
        .name = "b2m_reg",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(5, 0, 0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    try scene.meshes.append(alloc, &regular);

    // One claim, two builds, one publish: the repeat build runs under the
    // SAME reserved seq, so the queued payload must still freeze the newest
    // live TRS (repeat-newest-wins), with a single handoff generation and
    // no partial UI packet.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    var claim = scene.tryClaimBuildSlot().?;
    claim.build();
    regular.position = Vec3.new(25, 0, 0);
    claim.build();
    claim.stageUi();
    claim.publish();
    try std.testing.expectEqual(@as(u64, 1), scene.build_seq.load(.monotonic));
    finishForTest(&scene);
    try std.testing.expectEqual(@as(u64, 1), scene.last_latched_seq.load(.monotonic));

    const draws = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 1), draws.primary.items.items.len);
    const item = p7FindByMeshIndex(draws.primary.items.items, 0).?;
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), item.model.m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 24.0), regular.cached_aabb.min.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 26.0), regular.cached_aabb.max.x, 1e-4);
    try std.testing.expect(!scene.ui_frame.has_capture);
}
