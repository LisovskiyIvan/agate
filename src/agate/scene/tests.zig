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

test "Scene camera switching and cycling" {
    const ally = std.testing.allocator;
    var scene: Scene = undefined;
    scene.allocator = ally;
    scene.cameras = .empty;
    scene.active_camera_index = null;
    scene.active_camera = null;
    scene.active_camera_owned_name = null;
    scene.enable_multi_camera = true;

    defer {
        for (scene.cameras.items) |entry| {
            if (entry.owns_name) ally.free(entry.name);
        }
        scene.cameras.deinit(ally);
    }

    const cam1 = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    const cam2 = Camera{ .arc_rotate = camera_mod.ArcRotateCamera.init("Cam2", .{}) };
    const cam3 = Camera{ .fly = camera_mod.FlyCamera.init("Cam3", .{}) };

    const idx0 = try scene.addCamera(.{ .name = "Cam1", .camera = cam1 });
    try std.testing.expectEqual(@as(usize, 0), idx0);
    try std.testing.expectEqual(@as(usize, 0), scene.getActiveCameraIndex().?);
    try std.testing.expectEqualStrings("Cam1", scene.getActiveCameraName().?);

    _ = try scene.addCamera(.{ .name = "Cam2", .camera = cam2 });
    _ = try scene.addCamera(.{ .name = "Cam3", .camera = cam3 });
    try std.testing.expectEqual(@as(usize, 3), scene.getCameraCount());

    // Cycle next: 0 -> 1 -> 2 -> 0
    scene.nextCamera();
    try std.testing.expectEqual(@as(usize, 1), scene.getActiveCameraIndex().?);
    try std.testing.expectEqualStrings("Cam2", scene.getActiveCameraName().?);

    scene.nextCamera();
    try std.testing.expectEqual(@as(usize, 2), scene.getActiveCameraIndex().?);
    try std.testing.expectEqualStrings("Cam3", scene.getActiveCameraName().?);

    scene.nextCamera();
    try std.testing.expectEqual(@as(usize, 0), scene.getActiveCameraIndex().?);

    // Cycle prev: 0 -> 2 -> 1
    scene.prevCamera();
    try std.testing.expectEqual(@as(usize, 2), scene.getActiveCameraIndex().?);

    // Switch by name
    try std.testing.expect(scene.switchCameraByName("Cam2"));
    try std.testing.expectEqual(@as(usize, 1), scene.getActiveCameraIndex().?);
    try std.testing.expect(!scene.switchCameraByName("NonExistent"));
}

test "updateLights packs point lights into the frame payload" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);

    _ = try scene.createPointLight("probe", .{
        .position = Vec3.zero,
        .intensity = 10.0,
        .range = 10.0,
    });

    // One point light packed into slot 0 (the light sits at the pack eye,
    // so its score is maximal). Hysteresis fades a newly seen light in
    // over fade_time (0.25 s), so the first pack carries a partial factor.
    // The pack travels through the mailbox: render-side consumption goes
    // via takeLatest, exactly what the render pass does.
    scene.updateLights(0.016);
    var pack_out: scene_lights.LightRig.FramePack = undefined;
    try std.testing.expect(scene.light_handoff.takeLatest(&pack_out));
    // First sight: packed with a partial enter-fade factor (intensity
    // lane = intensity * factor < full intensity).
    try std.testing.expectEqual(@as(f32, 1.0), pack_out.counts[0]);
    try std.testing.expect(pack_out.point_color_int[0][3] < 10.0);
    var frame: usize = 0;
    while (frame < 32) : (frame += 1) {
        scene.updateLights(0.016);
        // Consume per frame like render does: a fresh pack lands each
        // update, so the mailbox never lags.
        _ = scene.light_handoff.takeLatest(&pack_out);
    }
    // Fully faded in (0.512 s >= fade_time): the intensity lane carries
    // the light's exact intensity, and the incumbent keeps its slot.
    try std.testing.expectEqual(@as(f32, 1.0), pack_out.counts[0]);
    try std.testing.expectEqual(@as(f32, 10.0), pack_out.point_color_int[0][3]);
    // A second take without a new publish consumes nothing (last pack
    // stands on the render side) — the PIP multi-render pattern.
    try std.testing.expect(!scene.light_handoff.takeLatest(&pack_out));
}

test "publishFrameSnapshot and prepareFrame snapshot handoff" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);
    try std.testing.expect(!scene.frame_prepared);

    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expect(scene.frame_snapshot.has_camera);
    try std.testing.expectEqual(@as(i32, 1920), scene.frame_snapshot.screen_w);
    try std.testing.expectEqual(@as(i32, 1080), scene.frame_snapshot.screen_h);
}

test "area lights: Scene API, cap, enable, snapshot round-trip" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);

    // Empty scene: zero count, null get, zeroed pack lanes (zero lights
    // render bit-identically to today).
    try std.testing.expectEqual(@as(usize, 0), scene.areaLightCount());
    try std.testing.expect(scene.getAreaLight(0) == null);
    scene.updateLights(0.016);
    var pack: scene_lights.LightRig.FramePack = undefined;
    try std.testing.expect(scene.light_handoff.takeLatest(&pack));
    for (0..2) |i| {
        try std.testing.expectEqual([4]f32{ 0, 0, 0, 0 }, pack.area_center_int[i]);
        try std.testing.expectEqual([4]f32{ 0, 0, 0, 0 }, pack.area_right[i]);
        try std.testing.expectEqual([4]f32{ 0, 0, 0, 0 }, pack.area_up[i]);
        try std.testing.expectEqual([4]f32{ 0, 0, 0, 0 }, pack.area_color[i]);
    }

    const a0 = try scene.addAreaLight("key", .{
        .center = Vec3.new(0.0, 3.0, 0.0),
        .right = Vec3.new(2.0, 0.0, 0.0),
        .up = Vec3.new(0.0, 1.0, 0.0),
        .color = Color3.new(1.0, 0.0, 0.0),
        .intensity = 2.0,
    });
    _ = try scene.addAreaLight("rim", .{});
    try std.testing.expectEqual(@as(usize, 2), scene.areaLightCount());
    try std.testing.expect(scene.getAreaLight(0) == a0);
    try std.testing.expect(scene.getAreaLight(2) == null);
    // Hard cap: third light errors, count unchanged.
    try std.testing.expectError(error.TooManyAreaLights, scene.addAreaLight("third", .{}));
    try std.testing.expectEqual(@as(usize, 2), scene.areaLightCount());

    // Disable: lane zeroes out; re-enable restores it (per-light enable).
    a0.is_enabled = false;
    scene.updateLights(0.016);
    _ = scene.light_handoff.takeLatest(&pack);
    try std.testing.expectEqual([4]f32{ 0, 0, 0, 0 }, pack.area_center_int[0]);
    a0.is_enabled = true;

    // Update path carries the lanes into the frame snapshot (needs a
    // camera: packFrameSnapshot early-outs without one).
    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.updateLights(0.016);
    const snap = scene.packFrameSnapshot(1.0, 640, 480);
    try std.testing.expect(snap.has_camera);
    try std.testing.expectEqual([4]f32{ 0.0, 3.0, 0.0, 2.0 }, snap.light_pack.area_center_int[0]);
    try std.testing.expectEqual([4]f32{ 2.0, 0.0, 0.0, 0.0 }, snap.light_pack.area_right[0]);
    try std.testing.expectEqual([4]f32{ 0.0, 1.0, 0.0, 0.0 }, snap.light_pack.area_up[0]);
    try std.testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 0.0 }, snap.light_pack.area_color[0]);

    // Removal is order-preserving; out-of-range is a no-op. Area lights
    // are session-local (like directional fills): the serialization
    // SceneState carries no area field, so save/load never persists them.
    scene.removeAreaLight(7);
    try std.testing.expectEqual(@as(usize, 2), scene.areaLightCount());
    scene.removeAreaLight(0);
    try std.testing.expectEqual(@as(usize, 1), scene.areaLightCount());
    try std.testing.expectEqualStrings("rim", scene.getAreaLight(0).?.name);
}

test "clustered lights: Scene API, cap, enable, snapshot round-trip" {
    const alloc = std.testing.allocator;
    const cluster_lights = @import("../lights.zig");
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);

    // Empty scene: zero count, null get, zeroed staged lanes (zero lights
    // render bit-identically to today).
    try std.testing.expectEqual(@as(usize, 0), scene.clusteredPointLightCount());
    try std.testing.expect(scene.getClusteredPointLight(0) == null);
    scene.updateLights(0.016);
    var pack: scene_lights.LightRig.FramePack = undefined;
    try std.testing.expect(scene.light_handoff.takeLatest(&pack));
    try std.testing.expectEqual(@as(usize, 0), pack.clustered_count);
    try std.testing.expectEqual([4]f32{ 0, 0, 0, 0 }, pack.clustered_pos_range[0]);
    try std.testing.expectEqual([4]f32{ 0, 0, 0, 0 }, pack.clustered_color_int[0]);

    const idx0 = try scene.addClusteredPointLight(Vec3.new(1.0, 2.0, 3.0), .{
        .color = Color3.new(1.0, 0.0, 0.0),
        .intensity = 2.0,
        .radius = 7.0,
    });
    const idx1 = try scene.addClusteredPointLight(Vec3.zero, .{});
    try std.testing.expectEqual(@as(usize, 0), idx0);
    try std.testing.expectEqual(@as(usize, 1), idx1);
    try std.testing.expectEqual(@as(usize, 2), scene.clusteredPointLightCount());
    try std.testing.expect(scene.getClusteredPointLight(0).?.position.x == 1.0);
    try std.testing.expect(scene.getClusteredPointLight(2) == null);

    // Disable: lane zeroes out on the next staged pack; re-enable restores.
    scene.getClusteredPointLight(0).?.is_enabled = false;
    scene.updateLights(0.016);
    _ = scene.light_handoff.takeLatest(&pack);
    try std.testing.expectEqual(@as(usize, 2), pack.clustered_count);
    try std.testing.expectEqual([4]f32{ 0, 0, 0, 0 }, pack.clustered_pos_range[0]);
    scene.getClusteredPointLight(0).?.is_enabled = true;

    // Update path carries the lanes into the frame snapshot (needs a
    // camera: packFrameSnapshot early-outs without one).
    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.updateLights(0.016);
    var snap = scene.packFrameSnapshot(1.0, 640, 480);
    try std.testing.expect(snap.has_camera);
    try std.testing.expectEqual(@as(usize, 2), snap.light_pack.clustered_count);
    try std.testing.expectEqual([4]f32{ 1.0, 2.0, 3.0, 7.0 }, snap.light_pack.clustered_pos_range[0]);
    try std.testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 2.0 }, snap.light_pack.clustered_color_int[0]);

    // Update ordering: a move stages through updateLights, so the snapshot
    // WITHOUT an update still shows the old position (documented 1-frame
    // lag), and the snapshot AFTER the update shows the new one.
    scene.getClusteredPointLight(0).?.position = Vec3.new(9.0, 9.0, 9.0);
    snap = scene.packFrameSnapshot(1.0, 640, 480);
    try std.testing.expectEqual([4]f32{ 1.0, 2.0, 3.0, 7.0 }, snap.light_pack.clustered_pos_range[0]);
    scene.updateLights(0.016);
    snap = scene.packFrameSnapshot(1.0, 640, 480);
    try std.testing.expectEqual([4]f32{ 9.0, 9.0, 9.0, 7.0 }, snap.light_pack.clustered_pos_range[0]);

    // Hard cap: past 64 lights the add errors and the count is unchanged.
    var k: usize = 2;
    while (k < cluster_lights.max_clustered_lights) : (k += 1) {
        _ = try scene.addClusteredPointLight(Vec3.zero, .{});
    }
    try std.testing.expectEqual(cluster_lights.max_clustered_lights, scene.clusteredPointLightCount());
    try std.testing.expectError(error.TooManyClusteredLights, scene.addClusteredPointLight(Vec3.zero, .{}));
    try std.testing.expectEqual(cluster_lights.max_clustered_lights, scene.clusteredPointLightCount());

    // Removal is order-preserving; out-of-range is a no-op (and never
    // retires — see the retire test below).
    scene.removeClusteredPointLight(7_000);
    try std.testing.expectEqual(cluster_lights.max_clustered_lights, scene.clusteredPointLightCount());
    scene.removeClusteredPointLight(0);
    try std.testing.expectEqual(cluster_lights.max_clustered_lights - 1, scene.clusteredPointLightCount());

    // Session-local (like directional fills and area lights): the
    // serialization SceneState carries no clustered field, so save/load
    // never persists the pool.
    try std.testing.expect(!@hasField(@import("../serialization.zig").SceneState, "clustered_lights"));
    try std.testing.expect(!@hasField(@import("../serialization.zig").SceneState, "clustered"));
}

test "clustered removal retires live tile buffers; out-of-range never retires" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    // The retire queue only owns the pending list here (fake buffer ids,
    // never through sg.*): free the list directly instead of draining.
    defer scene.gpu_retire.pending.deinit(alloc);

    _ = try scene.addClusteredPointLight(Vec3.zero, .{});
    _ = try scene.addClusteredPointLight(Vec3.zero, .{});
    // Simulate a previously uploaded generation (headless: borrowed ids,
    // never created or destroyed through sg here).
    scene.clustered.light_buffer = .{ .id = 41 };
    scene.clustered.header_buffer = .{ .id = 42 };
    scene.clustered.index_buffer = .{ .id = 43 };
    scene.clustered.gpu_live = true;

    // Out-of-range removal is a no-op: count unchanged, nothing retired.
    scene.removeClusteredPointLight(9);
    try std.testing.expectEqual(@as(usize, 2), scene.clusteredPointLightCount());
    try std.testing.expectEqual(@as(usize, 0), scene.gpu_retire.retainedCount());

    // Real removal retires all three live tile buffers (epoch-stamped
    // appends, no sg.*) and zeroes the handles for the next rebuild.
    scene.removeClusteredPointLight(0);
    try std.testing.expectEqual(@as(usize, 1), scene.clusteredPointLightCount());
    try std.testing.expectEqual(@as(usize, 3), scene.gpu_retire.retainedCount());
    try std.testing.expectEqual(@as(u32, 0), scene.clustered.light_buffer.id);
    try std.testing.expectEqual(@as(u32, 0), scene.clustered.header_buffer.id);
    try std.testing.expectEqual(@as(u32, 0), scene.clustered.index_buffer.id);
    try std.testing.expect(!scene.clustered.gpu_live);
}

test "saturated frame mailbox keeps the newest snapshot" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);

    // Reproducer for the fallback-overwrite bug: three publishes without a
    // consuming prepareFrame saturate the 2-slot mailbox. The third publish
    // must drain the stale slots and land, so prepareFrame takes 300 — not
    // an older published frame over the newer fallback. packFrameSnapshot
    // sets screen_w before the no-camera early-out, so no camera is needed.
    scene.publishFrameSnapshot(1.0, 100, 100);
    scene.publishFrameSnapshot(1.0, 200, 200);
    scene.publishFrameSnapshot(1.0, 300, 300);

    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(i32, 300), scene.frame_snapshot.screen_w);
    try std.testing.expectEqual(@as(i32, 300), scene.frame_snapshot.screen_h);
}

test "saturated light mailbox keeps the newest pack" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);

    // Same saturation shape for lights: fill both slots with a sentinel,
    // then update through the saturated mailbox. updateLights must drain the
    // stale sentinels and publish the fresh pack, so takeLatest never
    // resurfaces one.
    var sentinel = std.mem.zeroes(scene_lights.LightRig.FramePack);
    sentinel.counts[0] = 7.0;
    for (0..2) |_| {
        const i = scene.light_handoff.claim().?;
        scene.light_handoff.slot(i).* = sentinel;
        scene.light_handoff.publish(i);
    }
    try std.testing.expect(scene.light_handoff.claim() == null);

    scene.updateLights(0.016);
    var pack_out: scene_lights.LightRig.FramePack = undefined;
    try std.testing.expect(scene.light_handoff.takeLatest(&pack_out));
    // Fresh pack from the light-less rig: zero lights, not the sentinel.
    try std.testing.expectEqual(@as(f32, 0.0), pack_out.counts[0]);
    try std.testing.expect(!scene.light_handoff.takeLatest(&pack_out));
}

test "destroyMesh removes the physics body" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.physics.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);

    const m = try alloc.create(Mesh);
    m.* = @import("../testing.zig").testMesh("body_mesh");
    try scene.meshes.append(alloc, m);

    _ = try scene.createRigidBody(m, .box, 1.0);
    try std.testing.expect(scene.getRigidBody(m) != null);

    scene.destroyMesh(m);
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.physics.getWorld().?.bodies.items.len);
}

test "destroyMesh orphans children and detaches bone links" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);

    const parent = try alloc.create(Mesh);
    parent.* = @import("../testing.zig").testMesh("parent");
    try scene.meshes.append(alloc, parent);
    const child = try alloc.create(Mesh);
    child.* = @import("../testing.zig").testMesh("child");
    child.parent = parent;
    try scene.meshes.append(alloc, child);
    const attached = try alloc.create(Mesh);
    attached.* = @import("../testing.zig").testMesh("attached");
    attached.attachToBone(parent, 2);
    try scene.meshes.append(alloc, attached);

    scene.destroyMesh(parent);
    try std.testing.expect(child.parent == null);
    try std.testing.expect(attached.attach_bone == null);
    // No cascade: the orphans stay alive under their own transform.
    try std.testing.expectEqual(@as(usize, 2), scene.meshes.items.len);

    scene.destroyMesh(child);
    scene.destroyMesh(attached);
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
}

test "destroyMesh removes LOD entries preserving order" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);

    const parent = try alloc.create(Mesh);
    parent.* = @import("../testing.zig").testMesh("lod_parent");
    try scene.meshes.append(alloc, parent);
    const c1 = try alloc.create(Mesh);
    c1.* = @import("../testing.zig").testMesh("lod_c1");
    try scene.meshes.append(alloc, c1);
    const c2 = try alloc.create(Mesh);
    c2.* = @import("../testing.zig").testMesh("lod_c2");
    try scene.meshes.append(alloc, c2);
    const c3 = try alloc.create(Mesh);
    c3.* = @import("../testing.zig").testMesh("lod_c3");
    try scene.meshes.append(alloc, c3);

    try parent.addLODLevel(alloc, 10.0, c1);
    try parent.addLODLevel(alloc, 20.0, c2);
    try parent.addLODLevel(alloc, 30.0, c3);

    scene.destroyMesh(c2);
    try std.testing.expectEqual(@as(usize, 2), parent.lod_levels.items.len);
    try std.testing.expectEqual(@as(f32, 10.0), parent.lod_levels.items[0].distance);
    try std.testing.expect(parent.lod_levels.items[0].mesh.? == c1);
    try std.testing.expectEqual(@as(f32, 30.0), parent.lod_levels.items[1].distance);
    try std.testing.expect(parent.lod_levels.items[1].mesh.? == c3);

    scene.destroyMesh(c1);
    scene.destroyMesh(c3);
    scene.destroyMesh(parent);
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
}

test "destroyMesh drops decal instances and frees material" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.decals.deinit();
    defer scene.meshes.deinit(alloc);
    defer scene.pbr_materials.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);

    // Manual instance (no projection/GPU): the manager only stores the
    // mesh/material pair, so removal + material teardown is fully
    // exercisable without projecting any decal geometry.
    const dm = scene.getOrCreateDecalManager(8);
    const dm_mesh = try alloc.create(Mesh);
    dm_mesh.* = @import("../testing.zig").testMesh("decal_mesh");
    try scene.meshes.append(alloc, dm_mesh);
    const mat = try scene.createPBRMaterial("decal_mat");
    try dm.instances.append(alloc, .{
        .mesh = dm_mesh,
        .material = mat,
        .base_color = Color3.white,
        .lifetime = 0.0,
        .fade_duration = 1.0,
    });

    scene.destroyMesh(dm_mesh);
    try std.testing.expectEqual(@as(usize, 0), dm.instances.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.pbr_materials.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
}

test "pending destroy overflow drains on flush" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);

    // Drive the allocation-free spillover directly: the off-context OOM
    // enqueue itself needs fault injection, but the drain path (flush and
    // deinit share it) is covered here with a buffer-free mesh, so no sg.*
    // call is involved. Запись помечена завершённым epoch, чтобы flush её
    // забрал — правило epoch относится и к overflow.
    const m = try alloc.create(Mesh);
    m.* = @import("../testing.zig").testMesh("overflow_mesh");
    const e = scene.gpu_retire.begin();
    scene.gpu_retire.overflow[0] = .{ .kind = .mesh, .mesh = m, .epoch = e };
    scene.gpu_retire.overflow_len = 1;
    scene.gpu_retire.complete(e);

    scene.flushPendingGpuUploads();
    try std.testing.expectEqual(@as(usize, 0), scene.gpu_retire.retainedCount());
}

test "destroyMesh вне контекста: ретенция + flush на контекстном потоке" {
    const alloc = std.testing.allocator;
    // Маркер идемпотентен: главный поток тестов уже помечен gpu_thread-тестами
    // (идут раньше по реестру), воркер ниже всё равно чужой. Без маркера тест
    // был бы синхронным и бессмысленным.
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);

    const m = try alloc.create(Mesh);
    m.* = @import("../testing.zig").testMesh("offctx_mesh");
    try scene.meshes.append(alloc, m);

    // Воркер — не-контекстный поток: destroyMesh обязан только отвязать меш
    // (физика/иерархия/LOD зачищены) и положить его в ретенцию, без sg.*.
    const Job = struct {
        scene: *Scene,
        mesh: *Mesh,
        fn run(j: @This()) void {
            j.scene.destroyMesh(j.mesh);
        }
    };
    const t = try std.Thread.spawn(.{}, Job.run, .{Job{ .scene = &scene, .mesh = m }});
    t.join();
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
    try std.testing.expectEqual(@as(usize, 1), scene.gpu_retire.retainedCount());

    // Конец кадра + flush на контекстном потоке: ретенция освобождена.
    scene.gpu_retire.complete(scene.gpu_retire.current());
    scene.flushPendingGpuUploads();
    try std.testing.expectEqual(@as(usize, 0), scene.gpu_retire.retainedCount());
}

test "prepareFrame stages an empty upload tally without an upload queue" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);

    // No uploads queue and no io runner on the fixture: prepareFrame takes
    // the synchronous path and stages a zero tally for the stats publish.
    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(usize, 0), scene.frame_uploads.count);
    try std.testing.expectEqual(@as(u64, 0), scene.frame_uploads.bytes);
    // Без GPU-контекста динамических апдейтов нет: метрика нулевая,
    // счётчик meter сброшен в начале prepareFrame.
    try std.testing.expectEqual(@as(u64, 0), scene.stats.updated_bytes_frame);
    try std.testing.expectEqual(@as(u64, 0), upload_meter.peek());
}

test "prepareFrame rebuilds outline snapshots without GPU" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);

    // No GPU context on the fixture (default textures zeroed): outline
    // capture still runs unconditionally after the (skipped) pre-stage, as
    // before P5 — snapshots clear and rebuild every prepareFrame (P7: into
    // the published slot, read via preparedDraws()).
    var mesh = Mesh{
        .name = "headless_outline",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(4, 0, 0),
    };
    try scene.outline_meshes.append(alloc, &mesh);

    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(usize, 1), scene.preparedDraws().outline_items.items.len);
    // Rebuild, not accumulate.
    scene.prepareFrame();
    try std.testing.expectEqual(@as(usize, 1), scene.preparedDraws().outline_items.items.len);
    // Clearing works: a hidden mesh rebuilds to empty.
    mesh.is_visible = false;
    scene.prepareFrame();
    try std.testing.expectEqual(@as(usize, 0), scene.preparedDraws().outline_items.items.len);
}

test "prepareFrame transfers staged update tick, preserves prepare_ms" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);

    // Игровая фаза сложила тик через recordUpdateTime (НЕ в stats),
    // прошлый кадр оставил счётчики и post_ms: сброс prepareFrame обязан
    // перенести staged update_ms и сохранить prepare_ms, остальное обнулить.
    // Прямая запись stats.update_ms с update-стороны запрещена (контракт
    // recordUpdateTime): этот тест пишет только staged поле + prepare_ms.
    scene.recordUpdateTime(2.5);
    scene.stats.prepare_ms = 1.25;
    scene.stats.draw_calls = 41;
    scene.stats.triangles = 1000;
    scene.stats.post_ms = 3.0;

    scene.prepareFrame();

    try std.testing.expectEqual(@as(f32, 2.5), scene.stats.update_ms);
    try std.testing.expectEqual(@as(f32, 1.25), scene.stats.prepare_ms);
    try std.testing.expectEqual(@as(u32, 0), scene.stats.draw_calls);
    try std.testing.expectEqual(@as(u32, 0), scene.stats.triangles);
    try std.testing.expectEqual(@as(f32, 0.0), scene.stats.post_ms);
    // prepare_ms следующего кадра app перезапишет поверх после prepareFrame
    // (frame() в main) — handoff не мешает новому замеру.
}

test "recordUpdateTime stages without touching stats (update||render disjoint)" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);

    scene.prepareFrame();
    try std.testing.expectEqual(@as(f32, 0.0), scene.stats.update_ms);

    // Update-сторона (game thread): только staged поле, stats не тронуты —
    // render может читать stats конкурентно.
    scene.recordUpdateTime(7.5);
    try std.testing.expectEqual(@as(f32, 7.5), scene.pending_update_ms);
    try std.testing.expectEqual(@as(f32, 0.0), scene.stats.update_ms);

    // Последний тик wins до prepare; prepare переносит его в stats.
    scene.recordUpdateTime(8.25);
    scene.prepareFrame();
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

test "P6: prepareFrame captures UI into the render-owned frame" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
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

    scene.prepareFrame();
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
    scene.prepareFrame();
    try std.testing.expectEqual(canvas_ui.vertices.items.len, scene.ui_frame.vertices.items.len);
}

test "P6: camera-less prepare clears stale UI (no prior overlay)" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
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
    scene.prepareFrame();
    try std.testing.expect(scene.ui_frame.has_capture);

    // Camera lost: the next prepare must not redisplay the prior overlay.
    for (scene.cameras.items) |entry| {
        if (entry.owns_name) alloc.free(entry.name);
    }
    scene.cameras.clearRetainingCapacity();
    scene.active_camera = null;
    scene.active_camera_index = null;
    scene.prepareFrame();
    try std.testing.expect(!scene.ui_frame.has_capture);
    try std.testing.expectEqual(@as(usize, 0), scene.ui_frame.vertices.items.len);
    scene.ui_frame.drawPrepared();
    try std.testing.expectEqual(@as(u64, 0), upload_meter.peek());
}

test "P6: prepare snapshots canvas presence apart from content" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.ui_frame.deinit(alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    // No canvas at all: presence false, content empty.
    scene.publishFrameSnapshot(1.0, 640, 480);
    scene.prepareFrame();
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
    scene.prepareFrame();
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
    scene.prepareFrame();
    try std.testing.expect(!scene.ui_frame.canvas_present);
}

// ---- P7 triple-buffered prepared draws. ----

// Headless full-path integration runs prepareFrame with a faked GPU-init
// flag (default_white_texture.view.id != 0) plus a CPU-only shadow pass
// (allocator + empty scratch/payload, zero GPU handles). No sg.* fires on
// the prepare path for plain/skinned/hook meshes: instance staging skips
// non-instanced meshes (and guards the rest with sg.isvalid), shadow
// prepareInto and the view-queue builds are pure CPU snapshots, and UI
// capture is sg-guarded (P6). Render never runs headless.
fn p7CpuShadowPass(scene: *Scene, alloc: std.mem.Allocator) void {
    // CPU-only stand-in: prepareInto/binMeshes never touch GPU handles
    // headless (render never runs), so every handle field is safely zero and
    // only allocator + scratch/payload carry state. Full literal — no
    // undefined fields left unread.
    scene.shadows.pass = .{
        .allocator = alloc,
        .image = .{},
        .attachment_view = .{},
        .texture_view = .{},
        .sampler = .{},
        .depth_sampler = .{},
        .spot_image = .{},
        .spot_attachment_view = .{},
        .spot_texture_view = .{},
        .spot_needs_clear = false,
        .point_image = .{},
        .point_attachment_view = .{},
        .point_texture_view = .{},
        .point_needs_clear = false,
        .pipeline_u16 = .{},
        .pipeline_u32 = .{},
        .inst_pipeline_u16 = .{},
        .inst_pipeline_u32 = .{},
        .skinned_pipeline_u16 = .{},
        .skinned_pipeline_u32 = .{},
        .shadow_shader = .{},
        .inst_shader = .{},
        .skinned_shader = .{},
        .binned_meshes = .empty,
        .prepared = .{},
    };
}

fn p7ShadowTotal(draws: *const FrameDrawSlot) usize {
    var total: usize = 0;
    for (draws.shadow.bin.counts) |c| total += c;
    return total;
}

fn p7FindByMeshIndex(items: []const RenderMeshItem, idx: u32) ?RenderMeshItem {
    for (items) |it| {
        if (it.mesh_index == idx) return it;
    }
    return null;
}

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

    scene.prepareFrame();
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

    // Front intact while building back: move mesh_a, build the OTHER slot
    // directly, and prove the published front is untouched (lists, skins,
    // shader, outline, shadow ranges) while the back sees the new state.
    mesh_a.position = Vec3.new(9, 0, 0);
    // Slot-count agnostic: the scratch is whatever backIndex() reports, not
    // `1 - front` (that two-slot math is exactly what wave 26 removed).
    const back_idx = scene.draws.backIndex();
    // New frame id for the manual back build (as prepareFrame would bump):
    // the world-matrix cache keys on it, and the published front snapshot
    // must stay at the old pose regardless.
    scene.frame_id +%= 1;
    scene.prepareViewQueues(&scene.draws.slots[back_idx].primary, scene.frame_snapshot.primary_cam, null, 1.0, scene.frame_id, &scene.stats, true, .published);
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
    // View builds never touch outline: the scratch back holds none.
    try std.testing.expectEqual(@as(usize, 0), back_built.outline_items.items.len);

    // Warmup: the other slots are still cold (the manual build above only ran
    // the primary view queues, never outline/shadow/UI), so run full prepares
    // until every slot has been built once as back — the refusal proof below
    // needs all three slots warm.
    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);
    scene.prepareFrame();
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
        const want = scene.draws.backIndex();
        scene.prepareFrame();
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
        // Rotation-agnostic expectation: the publish lands on the back index
        // captured before the build, whatever the slot count.
        const want_front = scene.draws.backIndex();
        const saved_alloc = scene.allocator;
        scene.allocator = refusing.allocator();
        scene.shadows.pass.allocator = refusing.allocator();
        scene.prepareFrame();
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

    scene.prepareFrame();
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
    scene.prepareFrame();
    const draws2 = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 0), draws2.views[1].items.items.len);
    try std.testing.expectEqual(@as(usize, 2), draws2.primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 2), p7ShadowTotal(draws2));
}

test "P7: repeated prepare wins newest, no duplicate outline/UI capture" {
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

    scene.prepareFrame();
    try std.testing.expectEqual(@as(usize, 1), scene.preparedDraws().outline_items.items.len);
    const ui_n = canvas_ui.vertices.items.len;
    try std.testing.expectEqual(ui_n, scene.ui_frame.vertices.items.len);

    // More UI + moved outline, then a repeated prepare with no new camera
    // publish (fallback snapshot path): newest wins, nothing accumulates.
    canvas_ui.drawRect(1, 2, 3, 4, Color4.white);
    mesh.position = Vec3.new(7, 0, 0);
    const want_rep = scene.draws.backIndex();
    scene.prepareFrame();
    try std.testing.expectEqual(want_rep, scene.draws.front);
    try std.testing.expectEqual(@as(usize, 1), scene.preparedDraws().outline_items.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), scene.preparedDraws().outline_items.items[0].model.m[12], 1e-4);
    try std.testing.expectEqual(canvas_ui.vertices.items.len, scene.ui_frame.vertices.items.len);
    try std.testing.expect(canvas_ui.vertices.items.len > ui_n);
}

test "P7: no-camera/headless and disabled shadows clear coherently, epochs consumed" {
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
    scene.prepareFrame();
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
    scene.prepareFrame();
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
    scene.prepareFrame();
    const noshadow = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 1), noshadow.primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 0), noshadow.shadow.items.items.len);
    for (noshadow.shadow.bin.counts) |c| try std.testing.expectEqual(@as(usize, 0), c);
}

test "P7: allocator-failure back stays coherent and recovers without stale items" {
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
    scene.prepareFrame();
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
    scene.prepareFrame();
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
    scene.prepareFrame();
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

    // Consumed snapshot (screen dims are set before the no-camera
    // early-out, so no camera is needed for this ownership proof).
    scene.publishFrameSnapshot(1.0, 111, 111);
    scene.prepareFrame();
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
    scene.prepareFrame();
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
    scene.physics.uploadDebug(alloc, 1);
    try std.testing.expect(scene.physics.debug_pass == null);
    scene.physics.renderDebugPrepared(Mat4.identity, 1, &stats);
    try std.testing.expectEqual(@as(u32, 0), stats.draw_calls);
    try std.testing.expectEqual(@as(u32, 0), stats.main_draw_calls);

    // Non-empty capture + present-but-empty upload: still no count (the
    // counters gate on the issued-draw report, never on visibility alone).
    scene.physics.debug_pass = @import("../passes/debug_pass.zig").DebugPass{ .allocator = alloc };
    try std.testing.expect(scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 12), scene.physics.prepared_lines.items.len);
    scene.physics.renderDebugPrepared(Mat4.identity, 1, &stats);
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
    scene.prepareFrame();
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
    scene.prepareFrame();
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
    defer scene.physics.deinit(alloc);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.sky.enabled = true;
    scene.sky.exposure = 1.5;
    scene.recordUpdateTime(3.0);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
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
    try std.testing.expectEqual(@as(f32, 99.0), scene.pending_update_ms);

    // Next prepare picks up the newest live state (newest wins).
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expect(!scene.frame_snapshot.sky_enabled);
    try std.testing.expectEqual(@as(f32, 9.0), scene.frame_snapshot.sky_exposure);
    try std.testing.expectEqual(@as(f32, 99.0), scene.stats.update_ms);
}

// ---- Stage 1 producer-build handoff: buildPreparedFrame + prepare latch. ---
//
// Headless full-path pattern (mirrors the P7 tests): faked GPU-init flag
// (default_white_texture.view.id != 0) + CPU-only shadow pass + disabled
// culling. sg.isvalid() stays false, so the GPU halves publish bounds/count
// without touching sg.*. Render never runs headless past no-camera.

fn stage1FillInstances(src: *Mesh, mem: []@import("../mesh.zig").InstancedMesh, ptrs: []*@import("../mesh.zig").InstancedMesh, x0: f32) void {
    for (mem, 0..) |*inst, i| {
        const fi: f32 = @floatFromInt(i);
        inst.* = .{ .name = "s1", .source_mesh = src, .position = Vec3.new(x0 + fi * 2.0, 0, 0) };
        ptrs[i] = inst;
    }
}

fn stage1Scene(alloc: std.mem.Allocator) Scene {
    var scene = @import("../testing.zig").testScene(alloc);
    scene.enable_frustum_culling = false;
    scene.enable_occlusion_culling = false;
    scene.default_white_texture.view.id = 1;
    return scene;
}

test "stage1: worker build + main latch publishes previews; post-build mutation invisible" {
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
        .name = "s1_src",
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
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    // Victim: heap mesh (destroyMesh frees it via the retire queue) with
    // heap instances (Mesh.deinit destroys them at flush).
    const victim = try alloc.create(Mesh);
    victim.* = @import("../testing.zig").testMesh("s1_victim");
    victim.local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1));
    for (0..2) |i| {
        const inst = try alloc.create(InstancedMesh);
        inst.* = .{ .name = "v", .source_mesh = victim, .position = Vec3.new(@floatFromInt(i), 0, 0) };
        try victim.instances.append(alloc, inst);
    }
    try scene.meshes.append(alloc, victim);

    // Mark the context on main: the worker below is off-context, so its
    // destroyMesh must take the retire-queue path (never sg.* off-thread).
    gpu_thread.markContextThread();

    // Worker = the game side: build, then destroy the victim inside the
    // build→latch window (sequential in the worker — no data race; the
    // spawn/join edges carry the happens-before). Main only joins.
    const Builder = struct {
        scene: *Scene,
        victim: *Mesh,
        fn run(self: @This()) void {
            self.scene.buildPreparedFrame();
            self.scene.destroyMesh(self.victim);
        }
    };
    const t = try std.Thread.spawn(.{}, Builder.run, .{Builder{ .scene = &scene, .victim = victim }});
    t.join();

    // Off-context destroy unlinked the victim into the retire queue; the
    // build recorded previews for both meshes (victim included).
    try std.testing.expectEqual(@as(u64, 1), scene.build_seq.load(.monotonic));
    try std.testing.expectEqual(@as(usize, 1), scene.meshes.items.len);
    try std.testing.expectEqual(@as(usize, 1), scene.gpu_retire.retainedCount());
    try std.testing.expectEqual(@as(u64, 1), parent.instance_preview.build_seq);
    try std.testing.expectEqual(@as(u32, 4), parent.instance_preview.count);
    try std.testing.expectEqual(@as(u64, 1), victim.instance_preview.build_seq);

    // Live mutation AFTER the build: the published record mirror must
    // reflect the build, not the mutation — and the latch must not have
    // touched the live mesh at all (write-back is a game-side commit now).
    mem[0].position = Vec3.new(100, 0, 0);
    const preview_bounds = parent.instance_preview.bounds;
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
    try std.testing.expectEqual(std.math.maxInt(u64), parent.instance_render.staged_frame);

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(u64, 1), scene.last_latched_seq.load(.monotonic));
    // Latch published the mirrors; live meshes still untouched.
    const latched = scene.preparedDraws().staged_instances.items;
    try std.testing.expectEqual(@as(usize, 2), latched.len);
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
    try std.testing.expectEqual(std.math.maxInt(u64), parent.instance_render.staged_frame);
    // Stage-2B: queues froze at build time (parent + victim = 2 batches);
    // the latch stages whatever the slot owns (no live-list guard remains),
    // so both batches finalize from their records — including the destroyed
    // victim's staged data. The game-side commit below skips the victim by
    // identity instead (no UAF), and the surviving parent lands live.
    const draws = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 2), draws.primary.opaque_instanced.items.len);
    var found_parent: usize = 0;
    var found_victim: usize = 0;
    for (draws.primary.opaque_instanced.items) |b| {
        if (b.source_uid == parent.uid) {
            try std.testing.expectEqual(@as(u32, 4), b.visible_instance_count);
            found_parent += 1;
        } else {
            try std.testing.expectEqual(@as(u32, 2), b.visible_instance_count);
            found_victim += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 1), found_parent);
    try std.testing.expectEqual(@as(usize, 1), found_victim);

    // Commit (next game-side build) applies the published mirrors to live:
    // parent lands, the unlinked victim is skipped by the identity guard.
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(u32, 4), parent.instance_render.count);
    try std.testing.expectEqual(preview_bounds, parent.instance_render.bounds);
    try std.testing.expectEqual(scene.frame_id, parent.instance_render.staged_frame);
    // The CPU preview carries the matrix-bytes hash for the GPU dedup gate;
    // headless (no sg context) the GPU half is skipped, so instance_render
    // keeps the stale hash (0) by design — bounds/count still publish.
    try std.testing.expect(parent.instance_preview.hash != 0);
    try std.testing.expectEqual(@as(u64, 0), parent.instance_render.hash);
    // A 100-unit move would have shifted the bounds; the commit kept build
    // time (max.x well under the mutated span).
    try std.testing.expect(parent.instance_render.bounds.max.x < 50.0);
}

test "stage1: two builds before latch, newest wins" {
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
        .name = "s1_src",
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
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    scene.buildPreparedFrame();
    const first_bounds = parent.instance_preview.bounds;
    // Mutate, rebuild: the single preview store is recomputed in place.
    mem[2].position = Vec3.new(40, 0, 0);
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(u64, 2), scene.build_seq.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 2), parent.instance_preview.build_seq);
    try std.testing.expect(parent.instance_preview.bounds.max.x > first_bounds.max.x + 10.0);

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(u64, 2), scene.last_latched_seq.load(.monotonic));
    // Commit (next game-side build) applies the newest publish to live.
    scene.buildPreparedFrame();
    try std.testing.expectEqual(parent.instance_preview.bounds, parent.instance_render.bounds);
    try std.testing.expect(parent.instance_render.bounds.max.x > first_bounds.max.x + 10.0);
}

test "stage1: no build runs the inline fallback with identical counts/bounds" {
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
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 5;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    mem[4].is_visible = false;
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    // Latch path first: the outcome lands in the published record mirror
    // (live meshes stay untouched — no write-back on the latch path).
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(u64, 1), scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
    const mirror = scene.preparedDraws().staged_instances.items[0];
    try std.testing.expectEqual(@as(u32, 4), mirror.count);
    const latched_bounds = mirror.bounds;
    const latched_hash = mirror.uploaded_hash;

    // Same live state, no fresh build: the inline fallback must publish the
    // identical counts/bounds/hash into the live mesh (only staged_frame
    // advances past the mirror's frame).
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(u64, 1), scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(mirror.count, parent.instance_render.count);
    try std.testing.expectEqual(latched_bounds, parent.instance_render.bounds);
    try std.testing.expectEqual(latched_hash, parent.instance_render.hash);
}

test "stage1: fallback after latch is not clobbered by the later commit" {
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
        .name = "s1_src",
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
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    // Latch (frame 1): outcome sits in the slot mirror, live untouched.
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);

    // Fallback (frame 2, no fresh build): the inline path publishes live
    // directly — the pending frame-1 mirrors must not leak back in later.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(u32, 3), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 2), parent.instance_render.staged_frame);

    // Next build commits: the front slot is the fallback's (no records), so
    // this is a no-op — live keeps the frame-2 state, never regresses to
    // the stale frame-1 mirror.
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(u32, 3), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 2), parent.instance_render.staged_frame);

    // A fresh latch + commit republishes identically.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(u32, 3), parent.instance_render.count);
    try std.testing.expectEqual(@as(u64, 3), parent.instance_render.staged_frame);
}

test "stage1: serial same-thread build+latch parity" {
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
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 6;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 10);
    mem[1].is_visible = false;
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    scene.buildPreparedFrame();
    try std.testing.expectEqual(scene.draws.backIndex(), scene.build_slot.load(.monotonic));
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();

    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(scene.build_seq.load(.monotonic), scene.last_latched_seq.load(.monotonic));
    // Commit (next game-side build) applies the publish to live.
    scene.buildPreparedFrame();
    try std.testing.expectEqual(parent.instance_preview.count, parent.instance_render.count);
    try std.testing.expectEqual(@as(u32, 5), parent.instance_render.count);
    try std.testing.expectEqual(parent.instance_preview.bounds, parent.instance_render.bounds);
    try std.testing.expectEqual(scene.frame_id, parent.instance_render.staged_frame);
    try std.testing.expectEqual(scene.frame_id, scene.preparedDraws().frame_id);
    try std.testing.expectEqual(scene.retire_epoch, scene.preparedDraws().retire_epoch);
}

test "stage1: OOM build advances nothing, latch keeps previous, then recovers" {
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
        .name = "s1_src",
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
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    // Prime: funded build + latch publishes the complete state; the next
    // build commits it to the live mesh.
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    scene.buildPreparedFrame();
    const primed_bounds = parent.instance_render.bounds;
    const primed_hash = parent.instance_render.hash;
    try std.testing.expectEqual(@as(u32, 4), parent.instance_render.count);

    // Mutate live, then build unfunded: drop ALL scratch capacity so the
    // segment really allocates, and refuse the first alloc. The preview
    // must not advance (scene seq still does — the latch will skip it).
    mem[0].position = Vec3.new(100, 0, 0);
    for (&scene.draws.slots) |*slot| slot.primary.instance_matrices.clearAndFree(alloc);
    const real_alloc = scene.allocator;
    var failing = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 0 });
    scene.allocator = failing.allocator();
    scene.buildPreparedFrame();
    scene.allocator = real_alloc;
    try std.testing.expectEqual(@as(u64, 3), scene.build_seq.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 2), parent.instance_preview.build_seq);
    try std.testing.expect(failing.has_induced_failure);

    // Latch: the stale mesh is skipped — previous complete state stands,
    // no partial publish.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(u64, 3), scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 4), parent.instance_render.count);
    try std.testing.expectEqual(primed_bounds, parent.instance_render.bounds);
    try std.testing.expectEqual(primed_hash, parent.instance_render.hash);

    // Recovery: a funded build + latch + commit publishes the mutated state.
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(u64, 5), parent.instance_preview.build_seq);
    try std.testing.expect(parent.instance_render.bounds.max.x > primed_bounds.max.x + 10.0);
}

test "stage1: particle build on worker + latch on main freezes the frame" {
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.particles.systems.deinit(alloc);
    defer scene.particles.frame.deinit(alloc);
    defer scene.particles.build_frame.deinit(alloc);

    // CPU-only system with fake borrowed handle ids (never deinited via
    // ps.deinit — that would issue sg.destroy* on the fake ids; members are
    // freed manually below).
    const parts = try alloc.alloc(particles.Particle, 4);
    defer alloc.free(parts);
    const insts = try alloc.alloc(particles.ParticleInstanceData, 4);
    defer alloc.free(insts);
    const scratch = try alloc.alloc(u8, 4);
    defer alloc.free(scratch);
    var ps = ParticleSystem{
        .name = "s1",
        .allocator = alloc,
        .particles = parts,
        .instances = insts,
        .alive_scratch = scratch,
        .capacity = 4,
        .instance_buffer = .{ .id = 11 },
        .prng = std.Random.DefaultPrng.init(42),
    };
    ps.active_count = 3;
    try scene.particles.systems.append(alloc, &ps);

    // Worker = the game side: build only. Main only joins, then prepares.
    const Builder = struct {
        scene: *Scene,
        fn run(self: @This()) void {
            self.scene.buildPreparedFrame();
        }
    };
    const t = try std.Thread.spawn(.{}, Builder.run, .{Builder{ .scene = &scene }});
    t.join();
    try std.testing.expectEqual(@as(u64, 1), scene.particles.build_seq.load(.acquire));
    try std.testing.expectEqual(@as(usize, 1), scene.particles.build_frame.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.particles.frame.items.len);

    // Live mutation after the build must not reach the render-owned frame.
    ps.active_count = 1;
    ps.instance_buffer = .{ .id = 99 };
    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(u64, 1), scene.particles.latched_seq);
    try std.testing.expectEqual(@as(usize, 1), scene.particles.frame.items.len);
    try std.testing.expectEqual(@as(usize, 3), scene.particles.frame.items[0].active_count);
    try std.testing.expectEqual(@as(u32, 11), scene.particles.frame.items[0].instance_buffer.id);
    // renderPrepared keeps reading `frame` only (headless no-op here).
    try std.testing.expectEqual(@as(usize, 1), scene.particles.build_frame.items.len);
    ps.active_count = 4;
    try std.testing.expectEqual(@as(usize, 3), scene.particles.frame.items[0].active_count);
}

test "stage1: compute particle capture borrows the baked buffer; retire takes all five" {
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.particles.systems.deinit(alloc);
    defer scene.particles.frame.deinit(alloc);
    defer scene.particles.build_frame.deinit(alloc);

    // Compute-mode system with fake borrowed handle ids (never deinited via
    // ps.deinit — that would issue sg.destroy* on the fake ids; members are
    // freed manually below, same idiom as the cpu test above).
    const parts = try alloc.alloc(particles.Particle, 4);
    defer alloc.free(parts);
    const insts = try alloc.alloc(particles.ParticleInstanceData, 4);
    defer alloc.free(insts);
    const scratch = try alloc.alloc(u8, 4);
    defer alloc.free(scratch);
    const staging = try alloc.alloc(particles.GpuParticleSlot, 4);
    defer alloc.free(staging);
    var ps = ParticleSystem{
        .name = "s1_compute",
        .allocator = alloc,
        .particles = parts,
        .instances = insts,
        .alive_scratch = scratch,
        .capacity = 4,
        .instance_buffer = .{ .id = 11 },
        .gpu_slot_buffer = .{ .id = 12 },
        .compute_state_buffer = .{ .id = 21 },
        .compute_spawn_buffer = .{ .id = 22 },
        .compute_draw_buffer = .{ .id = 23 },
        .compute_staging = staging,
        .prng = std.Random.DefaultPrng.init(42),
    };
    ps.simulation_mode = .compute;
    ps.active_count = 3;
    ps.compute_high_water = 3;
    try scene.particles.systems.append(alloc, &ps);

    // Capture borrows the baked draw buffer (not the instance/slot ones) and
    // snapshots the compute mode.
    scene.particles.captureFrame(alloc);
    try std.testing.expectEqual(@as(usize, 1), scene.particles.frame.items.len);
    const draw = scene.particles.frame.items[0];
    try std.testing.expectEqual(particles.SimulationMode.compute, draw.simulation_mode);
    try std.testing.expectEqual(@as(u32, 23), draw.compute_draw_buffer.id);
    try std.testing.expectEqual(@as(u32, 23), draw.drawBuffer().id);

    // Teardown: all five particle buffers retire through the queue from any
    // thread (fake ids: retire into the open epoch like prepareFrame would,
    // so the pre-complete flush keeps them without sg.*, then manual cleanup
    // — mirrors the retireBuffer unit-test handling).
    _ = scene.gpu_retire.begin();
    var out = [_]sokol.gfx.Buffer{.{}} ** 8;
    const n = ps.takeGpuBuffersForRetire(&out);
    try std.testing.expectEqual(@as(usize, 5), n);
    for (out[0..n]) |buf| scene.gpu_retire.retireBuffer(alloc, buf);
    try std.testing.expectEqual(@as(usize, 5), scene.gpu_retire.retainedCount());
    try std.testing.expectEqual(@as(u64, 0), scene.gpu_retire.duplicateDropCount());
    scene.gpu_retire.flush(alloc);
    try std.testing.expectEqual(@as(usize, 5), scene.gpu_retire.retainedCount());
    scene.gpu_retire.pending.clearRetainingCapacity();
    try std.testing.expectEqual(@as(usize, 0), scene.gpu_retire.retainedCount());
    // Handles zeroed: the layer deinit path (ps.deinit) would now skip every
    // buffer destroy; a second take finds nothing (no double-retire).
    try std.testing.expectEqual(@as(usize, 0), ps.takeGpuBuffersForRetire(&out));
    try std.testing.expectEqual(@as(u32, 0), ps.compute_draw_buffer.id);
}

test "stage1: physics build on worker + latch on main freezes the capture" {
    const alloc = std.testing.allocator;
    var scene = stage1Scene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.outline_meshes.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.physics.deinit(alloc);

    _ = scene.physics.enable(alloc, null);
    var pmesh = @import("../testing.zig").testMesh("s1_phys");
    _ = try scene.physics.getWorld().?.createBody(&pmesh, .box, 0.0);
    scene.physics.show_debug = true;

    const Builder = struct {
        scene: *Scene,
        fn run(self: @This()) void {
            self.scene.buildPreparedFrame();
        }
    };
    const t = try std.Thread.spawn(.{}, Builder.run, .{Builder{ .scene = &scene }});
    t.join();
    try std.testing.expectEqual(@as(u64, 1), scene.physics.build_seq.load(.acquire));
    try std.testing.expect(scene.physics.build_visible);
    try std.testing.expectEqual(@as(usize, 12), scene.physics.build_lines.items.len);
    try std.testing.expect(!scene.physics.prepared_visible);

    // Live mutation after the build must not reach the prepared capture.
    const x0 = scene.physics.build_lines.items[0].a.x;
    pmesh.position = Vec3.new(5, 0, 0);
    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(u64, 1), scene.physics.latched_seq);
    try std.testing.expect(scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 12), scene.physics.prepared_lines.items.len);
    try std.testing.expectApproxEqAbs(x0, scene.physics.prepared_lines.items[0].a.x, 1e-4);
    scene.physics.step(0.016);
    try std.testing.expectApproxEqAbs(x0, scene.physics.prepared_lines.items[0].a.x, 1e-4);
}

test "stage1: instances cleared between build and latch take the regular path" {
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
        .name = "s1_src",
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
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    // Build while instanced: the preview records 3 (the build never
    // publishes — instance_render stays empty until a latch or fallback).
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(u32, 3), parent.instance_preview.count);
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);

    // Drop all instances before the latch: stage-2B freezes queues at build
    // time, and the latch stages whatever the slot owns (no live emptiness
    // re-check remains) — so this frame still draws the staged 3. The
    // live mesh is untouched by the latch (still empty), and the game-side
    // commit skips the emptied mesh, so the previous complete live state
    // stands. The regular-path switch takes effect on the next build+latch,
    // not this one.
    parent.instances.clearRetainingCapacity();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(u64, 1), scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
    const draws = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 1), draws.primary.opaque_instanced.items.len);
    try std.testing.expectEqual(@as(u32, 3), draws.primary.opaque_instanced.items[0].visible_instance_count);
    try std.testing.expectEqual(@as(usize, 0), draws.primary.items.items.len);

    // Next build: the commit skips the emptied mesh (live stays empty) and
    // the rebuild reroutes to the regular path; the following latch draws
    // the regular item with no instanced payload.
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    const draws2 = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 1), draws2.primary.items.items.len);
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
}

test "stage1: latch consumes slot records when live previews are cleared" {
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
        .name = "s1_src",
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
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    // Build freezes the slot-owned records; wiping the live previews after
    // that must not matter — the latch reads records + scratch only.
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(u32, 3), parent.instance_preview.count);
    try std.testing.expectEqual(@as(usize, 1), scene.draws.backSlot().staged_instances.items.len);
    parent.instance_preview = .{};
    try std.testing.expectEqual(@as(u64, 0), parent.instance_preview.build_seq);

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(u64, 1), scene.last_latched_seq.load(.monotonic));
    // Post-latch state mirrors into the slot record (the patch source) even
    // though the live previews were wiped; live meshes stay untouched until
    // the game-side commit.
    try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
    try std.testing.expectEqual(std.math.maxInt(u64), parent.instance_render.staged_frame);
    const draws = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 1), draws.primary.opaque_instanced.items.len);
    try std.testing.expectEqual(@as(u32, 3), draws.primary.opaque_instanced.items[0].visible_instance_count);
    try std.testing.expectEqual(@as(usize, 1), draws.staged_instances.items.len);
    try std.testing.expectEqual(scene.frame_id, draws.staged_instances.items[0].staged_frame);
    try std.testing.expectEqual(@as(u32, 3), draws.staged_instances.items[0].count);

    // Commit (next game-side build) applies the published mirror to live.
    // (The rebuild recomputes the wiped preview from unchanged live TRS.)
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(u32, 3), parent.instance_render.count);
    try std.testing.expectEqual(scene.frame_id, parent.instance_render.staged_frame);
    try std.testing.expect(parent.instance_render.bounds.isValid());
    try std.testing.expectEqual(draws.staged_instances.items[0].buffer.id, parent.instance_render.buffer.id);
}

test "stage1: failed latch keeps previous complete state, patch fail-closes" {
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
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const mem = try alloc.alloc(InstancedMesh, 4);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, 4);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        // Two live now; grown to four for the failing round below.
        .instances = .{ .items = ptrs[0..2], .capacity = 4 },
    };
    try scene.meshes.append(alloc, &parent);
    try scene.outline_meshes.append(alloc, &parent);

    // Round 1: funded build + latch publishes the complete state; the next
    // build commits it to the live mesh.
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(u32, 2), parent.instance_render.count);
    try std.testing.expect(parent.instance_render.bounds.isValid());

    // Distinctive prior values in every preserved field (headless buffers
    // are id 0, so a fake id proves the buffer slot is untouched too).
    parent.instance_render.buffer = .{ .id = 100 };
    parent.instance_render.capacity = 7;
    parent.instance_render.hash = 0xABCD;
    parent.instance_render.uploaded_count = 5;
    const prior = parent.instance_render;

    // Round 2: grow to 4 instances (a live GPU would need growth here),
    // build, then truncate the back-slot scratch before the latch so the
    // record's slice is out of range — the latch fail-closes through the
    // exact same path a GPU-half failure takes (failRecord + continue, live
    // untouched — ST2-C covers the makeBuffer-failure detection itself on a
    // real GPU).
    parent.instances.items = ptrs[0..4];
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(u32, 4), parent.instance_preview.count);
    scene.draws.backSlot().primary.instance_matrices.clearRetainingCapacity();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();

    // Previous COMPLETE state kept in every field; staged_frame NOT advanced
    // to the current frame (residual readers can tell nothing new published).
    try std.testing.expectEqual(prior.buffer.id, parent.instance_render.buffer.id);
    try std.testing.expectEqual(prior.capacity, parent.instance_render.capacity);
    try std.testing.expectEqual(prior.count, parent.instance_render.count);
    try std.testing.expectEqual(prior.bounds, parent.instance_render.bounds);
    try std.testing.expectEqual(prior.hash, parent.instance_render.hash);
    try std.testing.expectEqual(prior.uploaded_count, parent.instance_render.uploaded_count);
    try std.testing.expectEqual(prior.staged_frame, parent.instance_render.staged_frame);
    try std.testing.expect(parent.instance_render.staged_frame != scene.frame_id);

    // The record carries the not-published state.
    const recs = scene.preparedDraws().staged_instances.items;
    try std.testing.expectEqual(@as(usize, 1), recs.len);
    try std.testing.expectEqual(std.math.maxInt(u64), recs[0].staged_frame);
    try std.testing.expectEqual(@as(u32, 0), recs[0].buffer.id);
    try std.testing.expectEqual(@as(u32, 0), recs[0].count);

    // Payload fail-closed: the frozen batch/shadow/outline entries for the
    // mesh go invisible (count 0, zeroed handles).
    const draws = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 1), draws.primary.opaque_instanced.items.len);
    try std.testing.expectEqual(@as(u32, 0), draws.primary.opaque_instanced.items[0].visible_instance_count);
    try std.testing.expectEqual(@as(u32, 0), draws.primary.opaque_instanced.items[0].instance_buffer.id);
    var shadow_found = false;
    for (draws.shadow.items.items) |it| {
        if (it.is_instanced and it.source_uid == parent.uid) {
            try std.testing.expectEqual(@as(u32, 0), it.visible_instance_count);
            try std.testing.expectEqual(@as(u32, 0), it.instance_buffer.id);
            try std.testing.expect(!it.world_aabb.isValid());
            try std.testing.expectEqual(@as(f32, 0), it.max_dim);
            shadow_found = true;
        }
    }
    try std.testing.expect(shadow_found);
    var outline_found = false;
    for (draws.outline_items.items) |it| {
        if (it.is_instanced and it.source_uid == parent.uid) {
            try std.testing.expectEqual(@as(u32, 0), it.visible_instance_count);
            try std.testing.expectEqual(@as(u32, 0), it.instance_buffer.id);
            outline_found = true;
        }
    }
    try std.testing.expect(outline_found);
    // No retirement happened.
    try std.testing.expectEqual(@as(usize, 0), scene.gpu_retire.retainedCount());

    // Recovery: restore the scratch with a funded rebuild, latch, and commit
    // publishes the grown state.
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(u32, 4), parent.instance_render.count);
    try std.testing.expectEqual(scene.frame_id, parent.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u32, 4), scene.preparedDraws().primary.opaque_instanced.items[0].visible_instance_count);
}

test "stage1: mesh reorder + post-build add stages through, commit skips, recovers next" {
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
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const mem_a = try alloc.alloc(InstancedMesh, 3);
    defer alloc.free(mem_a);
    const ptrs_a = try alloc.alloc(*InstancedMesh, 3);
    defer alloc.free(ptrs_a);
    stage1FillInstances(&src, mem_a, ptrs_a, 0);
    var mesh_a = Mesh{
        .name = "s1_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs_a, .capacity = 3 },
    };
    const mem_b = try alloc.alloc(InstancedMesh, 2);
    defer alloc.free(mem_b);
    const ptrs_b = try alloc.alloc(*InstancedMesh, 2);
    defer alloc.free(ptrs_b);
    stage1FillInstances(&src, mem_b, ptrs_b, 50);
    var mesh_b = Mesh{
        .name = "s1_b",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs_b, .capacity = 2 },
    };
    try scene.meshes.append(alloc, &mesh_a);
    try scene.meshes.append(alloc, &mesh_b);

    // Scratch is concatenated [A0 A1 A2 B0 B1]: A.lo=0, B.lo=3.
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(usize, 0), mesh_a.instance_preview.scratch_lo);
    try std.testing.expectEqual(@as(usize, 3), mesh_b.instance_preview.scratch_lo);

    // Reorder (swapRemove unlinks A) and add a fresh mesh C whose preview
    // predates every build (build_seq 0).
    _ = scene.meshes.swapRemove(0);
    const mem_c = try alloc.alloc(InstancedMesh, 2);
    defer alloc.free(mem_c);
    const ptrs_c = try alloc.alloc(*InstancedMesh, 2);
    defer alloc.free(ptrs_c);
    stage1FillInstances(&src, mem_c, ptrs_c, 200);
    var mesh_c = Mesh{
        .name = "s1_c",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs_c, .capacity = 2 },
    };
    try scene.meshes.append(alloc, &mesh_c);

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    // Slot-owned latch: the reorder is invisible here (no live-list read
    // remains) — both records publish from slot data and the patch finalizes
    // the frozen batches from them. Live meshes stay exactly as built
    // (never-staged): the write-back is a game-side commit now.
    try std.testing.expectEqual(@as(u32, 0), mesh_b.instance_render.count);
    try std.testing.expectEqual(std.math.maxInt(u64), mesh_b.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u32, 0), mesh_a.instance_render.count);
    try std.testing.expectEqual(std.math.maxInt(u64), mesh_a.instance_render.staged_frame);
    const latched_records = scene.preparedDraws().staged_instances.items;
    try std.testing.expectEqual(@as(usize, 2), latched_records.len);
    for (latched_records) |rec| {
        try std.testing.expectEqual(scene.frame_id, rec.staged_frame);
    }
    // C (never built) is skipped: no publish, no queue batch.
    try std.testing.expectEqual(@as(u64, 0), mesh_c.instance_preview.build_seq);
    try std.testing.expectEqual(@as(u32, 0), mesh_c.instance_render.count);
    try std.testing.expectEqual(std.math.maxInt(u64), mesh_c.instance_render.staged_frame);
    // Stage-2B: the mesh list MUST NOT be mutated between build and latch
    // (swapRemove + append here); queues froze at build (A+B) and both
    // entries finalize from their records — no OOB, no UAF. The NEXT build's
    // commit runs the identity guard instead: both displaced records skip,
    // so neither live mesh publishes the stale generation.
    const draws = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 2), draws.primary.opaque_instanced.items.len);
    for (draws.primary.opaque_instanced.items) |b| {
        try std.testing.expect(b.visible_instance_count == 3 or b.visible_instance_count == 2);
    }
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(u32, 0), mesh_a.instance_render.count);
    try std.testing.expectEqual(std.math.maxInt(u64), mesh_a.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u32, 0), mesh_b.instance_render.count);
    try std.testing.expectEqual(std.math.maxInt(u64), mesh_b.instance_render.staged_frame);

    // Recovery: a funded build + latch + commit publishes the live list
    // (B + C).
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(u32, 2), mesh_b.instance_render.count);
    try std.testing.expectEqual(@as(u32, 2), mesh_c.instance_render.count);
    try std.testing.expectEqual(scene.frame_id, mesh_b.instance_render.staged_frame);
    const draws_r = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 2), draws_r.primary.opaque_instanced.items.len);
    for (draws_r.primary.opaque_instanced.items) |b| {
        try std.testing.expectEqual(@as(u32, 2), b.visible_instance_count);
    }
}

test "stage1: truncated scratch between build and latch is skipped safely" {
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
        .name = "s1_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    const n: usize = 2;
    const mem = try alloc.alloc(InstancedMesh, n);
    defer alloc.free(mem);
    const ptrs = try alloc.alloc(*InstancedMesh, n);
    defer alloc.free(ptrs);
    stage1FillInstances(&src, mem, ptrs, 0);
    var parent = Mesh{
        .name = "s1_parent",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .instances = .{ .items = ptrs, .capacity = n },
    };
    try scene.meshes.append(alloc, &parent);

    // Prime: funded build + latch publishes the complete state; the next
    // build commits it to the live mesh.
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    scene.buildPreparedFrame();
    const primed_bounds = parent.instance_render.bounds;
    try std.testing.expectEqual(@as(u32, 2), parent.instance_render.count);

    // Mutate, rebuild, then truncate the scratch before the latch (a
    // contract violation the latch must survive): the out-of-range slice is
    // skipped via the bounds check and the previous state stands.
    mem[0].position = Vec3.new(100, 0, 0);
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(u64, 3), parent.instance_preview.build_seq);
    scene.draws.backSlot().primary.instance_matrices.clearRetainingCapacity();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(u64, 3), scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(@as(u32, 2), parent.instance_render.count);
    try std.testing.expectEqual(primed_bounds, parent.instance_render.bounds);

    // Recovery: a funded build + latch + commit publishes the mutated state.
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    scene.buildPreparedFrame();
    try std.testing.expect(parent.instance_render.bounds.max.x > primed_bounds.max.x + 10.0);
}

test "stage-2A: buildQueuesInto with fallback params equals two runs" {
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

    scene.buildPreparedFrame();
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
    scene.prepareFrame();
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
    scene.buildPreparedFrame();
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

    scene.buildPreparedFrame();
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
    try std.testing.expectEqual(@as(usize, 1), scene.draws.backSlot().staged_instances.items.len);
    const old_buf = scene.draws.backSlot().staged_instances.items[0].buffer;
    try std.testing.expectEqual(@as(u32, 100), old_buf.id);
    scene.draws.backSlot().staged_instances.items[0].buffer = .{ .id = 200 };
    scene.draws.backSlot().staged_instances.items[0].capacity = 4;
    scene.draws.backSlot().staged_instances.items[0].uploaded_count = 4;
    scene.draws.backSlot().staged_instances.items[0].uploaded_hash =
        scene.draws.backSlot().staged_instances.items[0].hash;

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    // Commit (next game-side build) applies the published mirror to live.
    scene.buildPreparedFrame();
    const draws = scene.preparedDraws();
    // The record mirror carries the grown handle into instance_render and
    // every payload ref.
    try std.testing.expectEqual(@as(u32, 200), parent.instance_render.buffer.id);
    try std.testing.expectEqual(@as(u32, 200), scene.preparedDraws().staged_instances.items[0].buffer.id);
    try std.testing.expectEqual(scene.frame_id, scene.preparedDraws().staged_instances.items[0].staged_frame);
    // All payload refs carry the NEW buffer id and count.
    try std.testing.expectEqual(@as(u32, 4), parent.instance_render.count);
    try std.testing.expectEqual(@as(usize, 1), draws.primary.opaque_instanced.items.len);
    try std.testing.expectEqual(@as(u32, 200), draws.primary.opaque_instanced.items[0].instance_buffer.id);
    try std.testing.expectEqual(@as(u32, 4), draws.primary.opaque_instanced.items[0].visible_instance_count);
    var shadow_ok = false;
    for (draws.shadow.items.items) |it| {
        if (it.is_instanced and it.source_uid == parent.uid) {
            try std.testing.expectEqual(@as(u32, 200), it.instance_buffer.id);
            try std.testing.expectEqual(@as(u32, 4), it.visible_instance_count);
            shadow_ok = true;
        }
    }
    try std.testing.expect(shadow_ok);
    try std.testing.expectEqual(@as(usize, 1), draws.outline_items.items.len);
    try std.testing.expectEqual(@as(u32, 200), draws.outline_items.items[0].instance_buffer.id);
    try std.testing.expectEqual(@as(u32, 4), draws.outline_items.items[0].visible_instance_count);
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

    scene.buildPreparedFrame();
    // Simulate an OOM-skipped segment at latch: drop B's staged record so
    // the latch has nothing to consume for it (previous complete state
    // stands, patch must zero its provisional entries — no partial mix).
    {
        const recs = &scene.draws.backSlot().staged_instances;
        var i: usize = 0;
        while (i < recs.items.len) : (i += 1) {
            if (recs.items[i].mesh == &mesh_b) break;
        }
        try std.testing.expect(i < recs.items.len);
        _ = recs.orderedRemove(i);
    }

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    const draws = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 2), draws.primary.opaque_instanced.items.len);
    for (draws.primary.opaque_instanced.items) |batch| {
        if (batch.source_uid == mesh_a.uid) {
            try std.testing.expectEqual(@as(u32, 2), batch.visible_instance_count);
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
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    const draws2 = scene.preparedDraws();
    for (draws2.primary.opaque_instanced.items) |batch| {
        try std.testing.expectEqual(@as(u32, 2), batch.visible_instance_count);
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
        scene.buildPreparedFrame();
        break :blk mesh_a.uid;
    };
    const uid_b = mesh_b.uid;
    // Remove the trailing regular mesh: A/B indices unchanged (still draw),
    // C's absence must not OOB. Then reorder A/B via swapRemove to force a uid
    // mismatch on the next latch in a second round below.
    _ = scene.meshes.swapRemove(2);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    const draws = scene.preparedDraws();
    // A/B still draw (indices stable), no OOB from the removed tail.
    var a_ok = false;
    var b_ok = false;
    for (draws.primary.opaque_instanced.items) |batch| {
        if (batch.source_uid == uid_a) {
            try std.testing.expectEqual(@as(u32, 2), batch.visible_instance_count);
            a_ok = true;
        }
        if (batch.source_uid == uid_b) {
            try std.testing.expectEqual(@as(u32, 2), batch.visible_instance_count);
            b_ok = true;
        }
    }
    try std.testing.expect(a_ok and b_ok);

    // Second round: reorder A/B so stored indices mismatch uids. The latch
    // stages through (no live-list read remains) and the patch finalizes
    // the frozen batches from the records — no OOB, no UAF. Live meshes
    // still hold the round-1 commit (the write-back is game-side now).
    scene.buildPreparedFrame();
    _ = scene.meshes.swapRemove(0); // [B] (A unlinked); re-append A → [B, A]
    try scene.meshes.append(alloc, &mesh_a);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    const draws2 = scene.preparedDraws();
    try std.testing.expectEqual(@as(usize, 2), draws2.primary.opaque_instanced.items.len);
    for (draws2.primary.opaque_instanced.items) |batch| {
        try std.testing.expectEqual(@as(u32, 2), batch.visible_instance_count);
    }
    try std.testing.expectEqual(@as(u32, 2), mesh_a.instance_render.count);
    try std.testing.expectEqual(@as(u32, 2), mesh_b.instance_render.count);

    // Third round: unlink A again before the next build — both records are
    // now displaced (A unlinked, B shifted to index 0), so the commit guard
    // skips both (pointer compares, never a dereference): previous complete
    // state stands for both, the stale generation never lands.
    _ = scene.meshes.swapRemove(1); // [B] (A unlinked again)
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(u32, 2), mesh_a.instance_render.count);
    try std.testing.expect(mesh_a.instance_render.staged_frame != scene.frame_id);
    try std.testing.expectEqual(@as(u32, 2), mesh_b.instance_render.count);
    try std.testing.expect(mesh_b.instance_render.staged_frame != scene.frame_id);

    // Recovery: a funded rebuild + latch + commit over the live order passes
    // the guard again and lands the newest generation on B (A stays skipped
    // — still unlinked — with its previous complete state).
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    scene.buildPreparedFrame();
    try std.testing.expectEqual(scene.frame_id, mesh_b.instance_render.staged_frame);
    try std.testing.expectEqual(@as(u32, 2), mesh_b.instance_render.count);
    try std.testing.expectEqual(@as(u32, 2), mesh_a.instance_render.count);
    try std.testing.expect(mesh_a.instance_render.staged_frame != scene.frame_id);
}

test "stage-2B(e): fallback equivalence with build+latch" {
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
            // stage from dead stack memory — fallback-zero vs build-garbage).
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

    // Path 1: build+latch.
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
    scene_b.buildPreparedFrame();
    scene_b.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene_b.prepareFrame();
    const draws_b = scene_b.preparedDraws();
    const stats_b = scene_b.stats;

    // Path 2: fallback (never built).
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
    scene_f.prepareFrame();
    const draws_f = scene_f.preparedDraws();
    const stats_f = scene_f.stats;

    // Full payload equality, PAIRED BY IDENTITY (not list position): the two
    // scenes own distinct Mesh instances (global uid counter), so cross-scene
    // pairing keys on the mesh-list index (`source_mesh`/`mesh_index`), which
    // is stable across the two identical fixtures. Identity is validated
    // within each scene first (`meshes[source_mesh].uid == source_uid`, the
    // same invariant `patchInstanceRefs` enforces); an item with
    // `source_uid == 0` (never uid-assigned) falls back to positional pairing
    // for that item only. A prior failure here (fallback-zero vs
    // build-garbage on the instanced shadow bounds) traced to the Fixture
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
        scene.buildPreparedFrame();
        scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
        scene.prepareFrame();
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
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
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
            self.scene.buildPreparedFrame();
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
    serial.buildPreparedFrame();
    serial.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    serial.prepareFrame();

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
    threaded.prepareFrame();

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
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    // Commit (next game-side build) applies the published mirrors to live.
    scene.buildPreparedFrame();
    const draws = scene.preparedDraws();
    // No outline dropped; each item carries its MESH-LIST index (not the
    // outline-list position) and patches from its own mesh.
    try std.testing.expectEqual(@as(usize, 2), draws.outline_items.items.len);
    for (draws.outline_items.items) |it| {
        if (it.source_uid == mesh_b.uid) {
            try std.testing.expectEqual(@as(u32, 1), it.source_mesh);
            try std.testing.expectEqual(@as(u32, 2), it.visible_instance_count);
            try std.testing.expectEqual(mesh_b.instance_render.buffer.id, it.instance_buffer.id);
            const expect_c = if (mesh_b.instance_render.bounds.isValid()) mesh_b.instance_render.bounds.center() else mesh_b.position;
            try std.testing.expectEqual(expect_c, it.world_center);
        } else if (it.source_uid == mesh_a.uid) {
            try std.testing.expectEqual(@as(u32, 0), it.source_mesh);
            try std.testing.expectEqual(@as(u32, 2), it.visible_instance_count);
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
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
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
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    // Commit (next game-side build) applies the published mirror to live.
    scene.buildPreparedFrame();
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

    var blend_mat = material_mod.StandardMaterial.init("b2j_blend");
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
        .material = .{ .standard = &blend_mat },
        .instances = .{ .items = ptrs, .capacity = 2 },
    };
    try scene.meshes.append(alloc, &parent);

    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(u32, 2), parent.instance_preview.count);
    // Drop the staged record so the latch skips (as after an OOM/GPU
    // failure): the transparent batch patch-zeroes but its
    // transparent_order entry stays (stale). The draw skips count==0
    // batches, so the stale order entry is harmless — slot-state only here
    // (renderSceneView needs a live camera + sg context, impractical
    // headless; documented).
    {
        const recs = &scene.draws.backSlot().staged_instances;
        var i: usize = 0;
        while (i < recs.items.len) : (i += 1) {
            if (recs.items[i].mesh == &parent) break;
        }
        try std.testing.expect(i < recs.items.len);
        _ = recs.orderedRemove(i);
    }

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
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

test "stage-2B(k): same-scene build-then-fallback cache-key isolation" {
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

    // Round 1: build+latch (world cache tagged with the build key).
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expectEqual(scene.build_seq.load(.monotonic), scene.last_latched_seq.load(.monotonic));
    const built_reg_x = p7FindByMeshIndex(scene.preparedDraws().primary.items.items, 1).?.model.m[12];

    // Mutate live, then suppress the build: fallback must recompute under the
    // frame_id key (not reuse stale build-key entries).
    regular.position = Vec3.new(25, 0, 0);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame(); // have_build == false → inline fallback
    try std.testing.expectEqual(scene.build_seq.load(.monotonic), scene.last_latched_seq.load(.monotonic));
    const draws = scene.preparedDraws();
    const fb_item = p7FindByMeshIndex(draws.primary.items.items, 1).?;
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), fb_item.model.m[12], 1e-4);
    try std.testing.expect(fb_item.model.m[12] != built_reg_x);
    // Cache retagged with the context frame_id (not the build high-bit key).
    try std.testing.expectEqual(scene.frame_id, regular.cached_frame);
    try std.testing.expectEqual(@as(u32, 2), parent.instance_render.count);
    try std.testing.expectEqual(@as(usize, 1), draws.primary.opaque_instanced.items.len);
    try std.testing.expectEqual(@as(u32, 2), draws.primary.opaque_instanced.items[0].visible_instance_count);
}

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

    // Frame A: publish + latch (fallback, no build).
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expect(scene.frame_snapshot.has_camera);
    const eye_a = scene.frame_snapshot.primary_cam.eye;
    try std.testing.expectEqual(@as(i32, 800), scene.frame_snapshot.screen_w);
    try std.testing.expectEqual(@as(f32, 2.0), scene.frame_snapshot.sky_exposure);

    // Mutate live camera + environment, publish B, then build B.
    if (scene.active_camera) |*c| c.free.position = Vec3.new(10, 0, 0);
    if (scene.cameras.items.len > 0) scene.cameras.items[0].camera.free.position = Vec3.new(10, 0, 0);
    scene.sky.exposure = 9.0;
    scene.publishFrameSnapshot(16.0 / 9.0, 640, 480);
    scene.buildPreparedFrame();

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
    scene.prepareFrame();

    // Build B (frozen generation).
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.buildPreparedFrame();
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
    scene.prepareFrame();
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

    // C waited: the next fallback (no build) sees it.
    scene.prepareFrame();
    try std.testing.expectEqual(@as(i32, 640), scene.frame_snapshot.screen_w);
    try std.testing.expectEqual(@as(f32, 9.0), scene.frame_snapshot.sky_exposure);
    try std.testing.expect(!scene.frame_snapshot.shadows_enabled);
    try std.testing.expectEqual(@as(usize, 0), scene.preparedDraws().shadow.items.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), scene.preparedDraws().primary.items.items[0].model.m[12], 1e-4);
}

test "snapshot ownership: multi-build newest wins; no-publish refresh; removal; fallback" {
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
    scene.prepareFrame();

    // Two builds before one latch: newest wins.
    scene.sky.exposure = 2.0;
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.buildPreparedFrame();
    scene.sky.exposure = 5.0;
    scene.publishFrameSnapshot(16.0 / 9.0, 640, 480);
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(f32, 5.0), scene.build_snapshot.sky_exposure);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(f32, 5.0), scene.frame_snapshot.sky_exposure);
    try std.testing.expectEqual(@as(i32, 640), scene.frame_snapshot.screen_w);

    // No-publish build packs fresh live state (works without publish).
    scene.sky.exposure = 7.0;
    if (scene.active_camera) |*c| c.free.position = Vec3.new(3, 0, 0);
    if (scene.cameras.items.len > 0) scene.cameras.items[0].camera.free.position = Vec3.new(3, 0, 0);
    scene.buildPreparedFrame();
    try std.testing.expectEqual(@as(f32, 7.0), scene.build_snapshot.sky_exposure);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), scene.build_snapshot.primary_cam.eye.x, 1e-4);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(f32, 7.0), scene.frame_snapshot.sky_exposure);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), scene.frame_snapshot.primary_cam.eye.x, 1e-4);

    // Camera removal: fresh pack is camera-less, latch copies it coherently.
    for (scene.cameras.items) |entry| {
        if (entry.owns_name) alloc.free(entry.name);
    }
    scene.cameras.clearRetainingCapacity();
    scene.active_camera = null;
    scene.active_camera_index = null;
    scene.buildPreparedFrame();
    try std.testing.expect(!scene.build_snapshot.has_camera);
    scene.prepareFrame();
    try std.testing.expect(!scene.frame_snapshot.has_camera);
    try std.testing.expectEqual(@as(usize, 0), scene.preparedDraws().primary.items.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.preparedDraws().shadow.items.items.len);

    // Fallback unchanged: publish + prepare without a build takes the publish.
    const cam2 = Camera{ .free = camera_mod.FreeCamera.init("Cam2", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam2", .camera = cam2 });
    scene.sky.exposure = 4.0;
    scene.publishFrameSnapshot(16.0 / 9.0, 320, 240);
    scene.prepareFrame();
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
    scene.buildPreparedFrame();
    try std.testing.expect(scene.build_snapshot.shadows_enabled);
    scene.prepareFrame();
    try std.testing.expect(scene.frame_snapshot.shadows_enabled);
    try std.testing.expect(p7ShadowTotal(scene.preparedDraws()) > 0);

    // Fallback keeps the historical live gate: no build, live still off, so
    // the fresh pack disables shadows and the payload is empty.
    scene.prepareFrame();
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

    var blend_mat = material_mod.StandardMaterial.init("snap_eye_blend");
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
        .material = .{ .standard = &blend_mat },
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
    scene.buildPreparedFrame();

    // Frozen generation: published eye/sky, not the live mutation.
    try std.testing.expectApproxEqAbs(@as(f32, -100.0), scene.build_snapshot.primary_cam.eye.x, 1e-4);
    try std.testing.expectEqual(@as(u32, 77), scene.build_snapshot.sky_texture.?.view.id);
    // Staging sorted farthest-from-published-eye first (+10 before -10).
    const scratch = scene.draws.slots[scene.build_slot.load(.monotonic)].primary.instance_matrices.items;
    try std.testing.expectEqual(@as(usize, 2), scratch.len);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), scratch[0].m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -10.0), scratch[1].m[12], 1e-4);
    scene.prepareFrame();
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
    scene.prepareFrame();
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
    scene.prepareFrame();
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
    scene.prepareFrame();
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
    scene.prepareFrame();
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
    scene.prepareFrame();
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

    scene.prepareFrame();
    try std.testing.expectEqual(@as(u64, 0), scene.reuseStreak());
    scene.render();
    try std.testing.expectEqual(@as(u64, 0), scene.reuseStreak());
    scene.renderReuse();
    scene.renderReuse();
    scene.renderReuse();
    try std.testing.expectEqual(@as(u64, 3), scene.reuseStreak());
    try std.testing.expectEqual(@as(usize, 0), scene.pendingRetires());
    scene.prepareFrame();
    try std.testing.expectEqual(@as(u64, 0), scene.reuseStreak());
}

// Game-side UI CPU packet (slice 2, b): stage records canvas geometry into
// the back slot (sg-free, zero meter bytes); the latch consumes the staged
// bytes even when the canvas is mutated afterwards; a second prepare
// without a new stage falls back to the legacy canvas read.
test "ui packet stage-latch equals legacy, staged bytes win over later canvas mutation" {
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

    // Game-side stage: CPU copies into the back slot, no GPU touched.
    scene.stageUiPacket();
    try std.testing.expectEqual(@as(u64, 1), scene.ui_packet_seq.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 0), scene.last_latched_ui_seq.load(.monotonic));
    const back = scene.draws.backSlot();
    try std.testing.expect(back.ui_packet.valid);
    try std.testing.expect(back.ui_packet.canvas_present);
    try std.testing.expectEqual(staged_verts, back.ui_vertices.items.len);
    try std.testing.expectEqual(staged_idx, back.ui_indices.items.len);
    try std.testing.expectEqual(@as(u32, 7), back.ui_packet.handles.pipeline.id);
    try std.testing.expectEqual(@as(u32, 21), back.ui_packet.handles.vertex_buffer.id);
    try std.testing.expectEqual(@as(u32, 23), back.ui_packet.handles.index_buffer.id);
    try std.testing.expectEqual(@as(u64, 0), upload_meter.peek());

    // Mutate the canvas AFTER the stage (geometry AND handles): the latch
    // must still see staged.
    canvas.begin();
    canvas.drawRect(1, 2, 3, 4, Color4.white);
    canvas.pipeline = .{ .id = 70 };
    canvas.vertex_buffer = .{ .id = 210 };
    canvas.index_buffer = .{ .id = 230 };
    try std.testing.expect(canvas.vertices.items.len != staged_verts);

    // Keep a copy of the staged bytes: the fallback prepare below resets
    // the back slot (the latch must have consumed the packet first).
    const UIVertex = @import("../ui.zig").UIVertex;
    const want_verts = try alloc.dupe(UIVertex, back.ui_vertices.items);
    defer alloc.free(want_verts);
    const want_idx = try alloc.dupe(u16, back.ui_indices.items);
    defer alloc.free(want_idx);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);
    scene.prepareFrame();

    try std.testing.expectEqual(@as(u64, 1), scene.last_latched_ui_seq.load(.monotonic));
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

    // Second prepare without a new stage: no double-consume — the legacy
    // canvas read runs and now reflects the MUTATED canvas.
    scene.publishFrameSnapshot(16.0 / 9.0, 1920, 1080);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(u64, 1), scene.last_latched_ui_seq.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 1), scene.uiPacketLatchedCount());
    try std.testing.expectEqual(canvas.vertices.items.len, scene.ui_frame.vertices.items.len);
    try std.testing.expect(canvas.vertices.items.len != staged_verts);
}

test "ui packet unused keeps the legacy path bit-identical" {
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

    // No stageUiPacket call: seqs stay zero, the legacy canvas read runs.
    scene.prepareFrame();
    try std.testing.expectEqual(@as(u64, 0), scene.ui_packet_seq.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 0), scene.last_latched_ui_seq.load(.monotonic));
    try std.testing.expect(scene.ui_frame.has_capture);
    try std.testing.expectEqual(scene.ui_canvas.?.vertices.items.len, scene.ui_frame.vertices.items.len);
    try std.testing.expectEqual(@as(f32, 640.0), scene.ui_frame.screen_w);
}

test "ui packet stage OOM degrades to the legacy canvas read" {
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

    // Refuse the slot copy: the packet is marked invalid (never partial),
    // but the seq still advances so the latch degrades exactly once.
    var failing = std.testing.FailingAllocator.init(alloc, .{ .fail_index = 0 });
    const real_alloc = scene.allocator;
    scene.allocator = failing.allocator();
    scene.stageUiPacket();
    scene.allocator = real_alloc;
    try std.testing.expectEqual(@as(u64, 1), scene.ui_packet_seq.load(.monotonic));
    try std.testing.expect(!scene.draws.backSlot().ui_packet.valid);

    // The latch falls back to the legacy canvas read (content intact there):
    // the frame still captures, and the seq is stamped consumed (no retry).
    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });
    scene.publishFrameSnapshot(1.0, 640, 480);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(u64, 1), scene.last_latched_ui_seq.load(.monotonic));
    try std.testing.expect(scene.ui_frame.has_capture);
    try std.testing.expectEqual(scene.ui_canvas.?.vertices.items.len, scene.ui_frame.vertices.items.len);
}

test "reflection probes: add/remove/capacity/dirty bookkeeping" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    // testScene leaves the retire queue undefined: zero it explicitly (the
    // removal path retires through it, even for empty pre-capture targets).
    scene.gpu_retire = .{};
    defer scene.gpu_retire.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 0), scene.reflectionProbeCount());
    try std.testing.expect(scene.getReflectionProbe(0) == null);

    const a = try scene.addReflectionProbe(Vec3.new(1, 0, 0), .{});
    try std.testing.expectEqual(@as(usize, 0), a);
    _ = try scene.addReflectionProbe(Vec3.zero, .{ .enabled = false });
    _ = try scene.addReflectionProbe(Vec3.zero, .{});
    _ = try scene.addReflectionProbe(Vec3.zero, .{});
    try std.testing.expectEqual(@as(usize, 4), scene.reflectionProbeCount());
    // Past the cap: hard error, never a silent clamp or replacement.
    try std.testing.expectError(error.TooManyReflectionProbes, scene.addReflectionProbe(Vec3.zero, .{}));
    try std.testing.expectEqual(@as(usize, 4), scene.reflectionProbeCount());
    // Fresh probes all start dirty.
    try std.testing.expectEqual(@as(usize, 4), scene.probeDirtyCount());

    // Live state is mutable through the accessor; OOB access is null.
    const p = scene.getReflectionProbe(0).?;
    try std.testing.expectEqual(Vec3.new(1, 0, 0), p.position);
    p.intensity = 0.25;
    try std.testing.expectEqual(@as(f32, 0.25), scene.getReflectionProbe(0).?.intensity);
    try std.testing.expect(scene.getReflectionProbe(9) == null);

    // On-demand capture requests are pure bookkeeping (the GPU work runs in
    // render, which needs a real context and is covered there).
    scene.probes.notifyCaptured(0); // simulate one landed capture
    try std.testing.expectEqual(@as(usize, 3), scene.probeDirtyCount());
    scene.captureReflectionProbe(0);
    try std.testing.expectEqual(@as(usize, 4), scene.probeDirtyCount());
    scene.captureDirtyReflectionProbes();
    try std.testing.expectEqual(@as(usize, 4), scene.probeDirtyCount());
    scene.captureReflectionProbe(99); // OOB: no-op.

    // Removal retires the target (here empty: uniform no-op-destroy entry)
    // and shifts higher indices down, order-preserving.
    scene.removeReflectionProbe(0);
    try std.testing.expectEqual(@as(usize, 3), scene.reflectionProbeCount());
    try std.testing.expectEqual(@as(usize, 1), scene.gpu_retire.retainedCount());
    scene.removeReflectionProbe(99); // OOB: no-op.
    try std.testing.expectEqual(@as(usize, 3), scene.reflectionProbeCount());
    // Drain through the epoch discipline: complete the open epoch, flush.
    scene.gpu_retire.complete(scene.gpu_retire.current());
    scene.gpu_retire.flush(alloc);
    try std.testing.expectEqual(@as(usize, 0), scene.gpu_retire.retainedCount());
}

test "reflection probe state packs into the frame snapshot" {
    const scene_probes = @import("probe_layer.zig");
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    scene.gpu_retire = .{};
    defer scene.gpu_retire.deinit(alloc);
    // testScene leaves non-tested subsystems undefined: zero everything
    // the prepare path below can observe (no luck-with-zeroed-stack).
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
    scene.ui_packet_seq.store(0, .monotonic);
    scene.last_latched_ui_seq.store(0, .monotonic);

    const cam = Camera{ .free = camera_mod.FreeCamera.init("Cam1", .{}) };
    _ = try scene.addCamera(.{ .name = "Cam1", .camera = cam });

    // No probes: empty pack (every draw takes the legacy path).
    var snap = scene.packFrameSnapshot(1.0, 640, 480);
    try std.testing.expect(snap.has_camera);
    try std.testing.expectEqual(@as(usize, 0), snap.probe_pack.count);

    _ = try scene.addReflectionProbe(Vec3.new(1, 2, 3), .{ .radius = 4.0 });
    _ = try scene.addReflectionProbe(Vec3.zero, .{ .enabled = false });
    snap = scene.packFrameSnapshot(1.0, 640, 480);
    try std.testing.expectEqual(@as(usize, 2), snap.probe_pack.count);
    try std.testing.expectEqual(Vec3.new(1, 2, 3), snap.probe_pack.entries[0].position);
    try std.testing.expectEqual(@as(f32, 4.0), snap.probe_pack.entries[0].radius);
    try std.testing.expect(snap.probe_pack.entries[0].enabled);
    try std.testing.expect(!snap.probe_pack.entries[1].enabled);
    // Nothing captured headlessly: selection skips everything (disabled-
    // probe path leaves the draw state untouched — legacy fallback).
    try std.testing.expect(!snap.probe_pack.entries[0].captured);
    try std.testing.expect(scene_probes.selectProbe(snap.probe_pack.entries[0..snap.probe_pack.count], Vec3.new(1, 2, 3)) == null);

    // The pack rides the prepare path into the consumed snapshot (the same
    // copy renderReuse later re-presents without capturing): publish, then
    // prepare latches it verbatim.
    scene.publishFrameSnapshot(1.0, 640, 480);
    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(@as(usize, 2), scene.frame_snapshot.probe_pack.count);
    try std.testing.expectEqual(Vec3.new(1, 2, 3), scene.frame_snapshot.probe_pack.entries[0].position);

    // The snapshot also round-trips the mailbox by value (newest wins, like
    // the light pack): claim, publish the packed generation, take it back.
    // What renderReuse re-presents is exactly what was packed.
    const slot_i = scene.frame_handoff.claim().?;
    scene.frame_handoff.slot(slot_i).* = snap;
    scene.frame_handoff.publish(slot_i);
    var snap_out: scene_mod.SceneFrameSnapshot = undefined;
    try std.testing.expect(scene.frame_handoff.takeLatest(&snap_out));
    try std.testing.expectEqual(@as(usize, 2), snap_out.probe_pack.count);
    try std.testing.expectEqual(@as(f32, 4.0), snap_out.probe_pack.entries[0].radius);
}

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
        const want = scene.draws.backIndex();
        scene.prepareFrame();
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

        scene.buildPreparedFrame();
        build_slots[frame] = scene.build_slot.load(.monotonic);
        scene.stageUiPacket();
        scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
        scene.prepareFrame();

        try std.testing.expectEqual(scene.build_seq.load(.monotonic), scene.last_latched_seq.load(.monotonic));
        try std.testing.expectEqual(@as(u64, frame + 1), scene.uiPacketLatchedCount());
        try std.testing.expectEqual(scene.build_slot.load(.monotonic), scene.draws.front);
        try std.testing.expectEqual(scene.frame_id, scene.preparedDraws().frame_id);
        try std.testing.expectEqual(scene.retire_epoch, scene.preparedDraws().retire_epoch);
        // Staged records: this frame's instances, latched from the slot
        // (the latch mirrors; live meshes are committed by the NEXT build,
        // so live lags the mirror by exactly one frame).
        const mirror = scene.preparedDraws().staged_instances.items[0];
        try std.testing.expectEqual(@as(u32, 4), mirror.count);
        try std.testing.expectEqual(scene.frame_id, mirror.staged_frame);
        if (frame == 0) {
            try std.testing.expectEqual(@as(u32, 0), parent.instance_render.count);
            try std.testing.expectEqual(std.math.maxInt(u64), parent.instance_render.staged_frame);
        } else {
            try std.testing.expectEqual(@as(u32, 4), parent.instance_render.count);
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
    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    const front0 = scene.draws.front;

    // Render holds the pin only for the draw: no pins leak afterwards.
    scene.render();
    try std.testing.expect(!scene.frame_prepared);
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());
    try std.testing.expectEqual(front0, scene.draws.front);
    const consumed_id = scene.draws.slots[front0].frame_id;

    // A pinned presented frame is never a build target: the next prepare
    // publishes a different slot and leaves the pinned one untouched.
    try scene.draws.pin(front0);
    const want = scene.draws.backIndex();
    try std.testing.expect(want != front0);
    scene.prepareFrame();
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
    scene.buildPreparedFrame();

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
    scene.buildPreparedFrame();
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
    scene.prepareFrame();
    const front = scene.preparedDraws();
    try std.testing.expectEqual(@as(i32, 800), front.snapshot.screen_w);
    try std.testing.expectEqual(@as(i32, 600), front.snapshot.screen_h);
    try std.testing.expectEqual(@as(f32, 2.0), front.snapshot.sky_exposure);
    try std.testing.expectEqual(@as(i32, 800), scene.frame_snapshot.screen_w);
    try std.testing.expectEqual(@as(f32, 2.0), scene.frame_snapshot.sky_exposure);

    // C waited in the mailbox: the next fallback (no build) sees it.
    scene.prepareFrame();
    try std.testing.expectEqual(@as(i32, 640), scene.preparedDraws().snapshot.screen_w);
    try std.testing.expectEqual(@as(f32, 5.0), scene.preparedDraws().snapshot.sky_exposure);
}

test "wave27: fallback stages the slot snapshot; render+reuse present it" {
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
    scene.prepareFrame();
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

    // Generation B through the build path; the UI stage lands after the
    // build (the build reset would wipe an earlier packet).
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.buildPreparedFrame();
    scene.stageUiPacket();
    // Post-build live mutation of the working copy: the latch must still
    // capture the staged dims.
    scene.build_snapshot.screen_w = 111;
    scene.build_snapshot.screen_h = 222;
    scene.prepareFrame();

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
        scene.prepareFrame();
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

test "wave28: Scene 3D-GUI panel add/remove/count/dirty" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.gui3d.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);

    const a = try scene.addUi3dPanel("hud", Vec3.new(0, 1, 5), .{});
    const b = try scene.addUi3dPanel("map", Vec3.new(3, 1, 5), .{ .canvas_width = 256, .canvas_height = 256 });
    try std.testing.expectEqual(@as(usize, 0), a);
    try std.testing.expectEqual(@as(usize, 1), b);
    try std.testing.expectEqual(@as(usize, 2), scene.ui3dPanelCount());
    try std.testing.expectEqualStrings("map", scene.getUi3dPanel(1).?.name);
    try std.testing.expectEqualStrings("hud", scene.getUi3dPanelByName("hud").?.name);
    try std.testing.expect(scene.getUi3dPanelByName("missing") == null);
    try std.testing.expect(scene.getUi3dPanel(7) == null);

    // Fresh panels start dirty (scheduled for on-demand capture).
    try std.testing.expectEqual(@as(usize, 2), scene.ui3dDirtyCount());
    scene.markUi3dPanelDirty(0);
    scene.markAllUi3dPanelsDirty();
    try std.testing.expectEqual(@as(usize, 2), scene.ui3dDirtyCount());

    // Cap: 4 max, hard error past it.
    _ = try scene.addUi3dPanel("c", Vec3.zero, .{});
    _ = try scene.addUi3dPanel("d", Vec3.zero, .{});
    try std.testing.expectError(error.TooManyUi3dPanels, scene.addUi3dPanel("e", Vec3.zero, .{}));
    try std.testing.expectEqual(@as(usize, 4), scene.ui3dPanelCount());

    // Removal retires through the epoch queue and keeps index order.
    scene.removeUi3dPanel(0);
    try std.testing.expectEqual(@as(usize, 3), scene.ui3dPanelCount());
    try std.testing.expectEqualStrings("map", scene.getUi3dPanel(0).?.name);
    try std.testing.expectEqual(@as(usize, 1), scene.gpu_retire.retainedCount());
    // Out-of-range removal is a no-op (never retires).
    scene.removeUi3dPanel(42);
    try std.testing.expectEqual(@as(usize, 3), scene.ui3dPanelCount());
    try std.testing.expectEqual(@as(usize, 1), scene.gpu_retire.retainedCount());
}

test "wave28: Scene pickUi3dPanel uses the staged snapshot camera" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.gui3d.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);

    // No staged camera: clean miss (never a fallback ray into the scene).
    try std.testing.expect(scene.pickUi3dPanel(400, 300) == null);

    // Stage a snapshot camera headlessly: identity view-proj over 800x600,
    // so the screen-center ray is origin +Z by construction.
    const front = scene.draws.front;
    scene.draws.slots[front].snapshot.has_camera = true;
    scene.draws.slots[front].snapshot.screen_w = 800;
    scene.draws.slots[front].snapshot.screen_h = 600;
    scene.draws.slots[front].snapshot.primary_cam.view_proj = Mat4.identity;

    // Still nothing: no panels exist.
    try std.testing.expect(scene.pickUi3dPanel(400, 300) == null);

    _ = try scene.addUi3dPanel("hud", Vec3.new(0, 0, 5), .{
        .width = 2.0,
        .height = 2.0,
        .yaw_deg = 180.0,
        .canvas_width = 512,
        .canvas_height = 256,
    });
    // Uncaptured panel: invisible to picking until the first capture lands.
    try std.testing.expect(scene.pickUi3dPanel(400, 300) == null);
    scene.gui3d.notifyCaptured(0);

    // Screen center hits the quad center: panel 0, mid-canvas pixels.
    const hit = scene.pickUi3dPanel(400, 300).?;
    try std.testing.expectEqual(@as(usize, 0), hit.panel_index);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), hit.u, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), hit.v, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 256.0), hit.canvas_x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 128.0), hit.canvas_y, 1e-4);

    // The app routes the pick result into the canvas input state itself.
    const panel = scene.getUi3dPanel(hit.panel_index).?;
    panel.injectPointer(hit.canvas_x, hit.canvas_y, true);
    try std.testing.expect(panel.canvas.?.mouse_down);
    panel.injectRelease();
    try std.testing.expect(!panel.canvas.?.mouse_down);

    // Off-panel cursor: clean miss. Disabled panel: skipped entirely.
    try std.testing.expect(scene.pickUi3dPanel(-100, 300) == null);
    panel.enabled = false;
    try std.testing.expect(scene.pickUi3dPanel(-100, 300) == null);
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
    claim.stageUi(); // no canvas: stages the absence packet, bumps the seq
    try std.testing.expectEqual(@as(u64, 1), scene.ui_packet_seq.load(.monotonic));
    claim.publish();
    try std.testing.expectEqual(@as(u64, 1), scene.build_seq.load(.monotonic));
    try std.testing.expectEqual(slot, scene.build_slot.load(.monotonic));
    try std.testing.expectEqual(scene.draws.backIndex(), scene.build_slot.load(.monotonic));

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();

    try std.testing.expectEqual(scene.build_seq.load(.monotonic), scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(scene.build_slot.load(.monotonic), scene.draws.front);
    try std.testing.expectEqual(scene.frame_id, scene.preparedDraws().frame_id);
    // The staged absence packet was consumed (seq stamped, not re-latched).
    try std.testing.expectEqual(scene.ui_packet_seq.load(.monotonic), scene.last_latched_ui_seq.load(.monotonic));
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
    scene.prepareFrame();
    try std.testing.expectEqual(@as(u64, 1), scene.last_latched_seq.load(.monotonic));
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
    scene.prepareFrame();
    try std.testing.expectEqual(cur0 + 1, scene.gpu_retire.current());
    scene.render();
    try std.testing.expectEqual(scene.gpu_retire.current(), scene.gpu_retire.lastCompleted());
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());
}

// ---- Wave 30: concurrent-build handoff edge (atomic seq words). ----
//
// A producer thread runs the adoption-edge claim flow (claim -> stageUi ->
// publish: the UI stage release-bumps `ui_packet_seq`, the publish
// release-stores `build_slot` then `build_seq`) while a consumer thread runs
// the latch half of the edge (acquire-reads `build_seq`/`ui_packet_seq`,
// stamps `last_latched_seq`/`last_latched_ui_seq`). No full build/prepare
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
                // No canvas: stages the absence packet and release-bumps
                // `ui_packet_seq` (one stage per publish, like the
                // build+stage claim flow).
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
            // Acquire: pairs with the publish/stage release-stores — a fresh
            // generation implies the staged payload is visible.
            const b = c.scene.build_seq.load(.acquire);
            // Never stale: generations are strictly increasing by
            // single-producer construction, so an older read after a newer
            // one would be a torn handoff.
            if (b < c.max_build_seen) unreachable;
            if (b > c.max_build_seen) c.max_build_seen = b;
            // Context-side stamp only (the producer never touches this word).
            c.scene.last_latched_seq.store(b, .monotonic);
            const u = c.scene.ui_packet_seq.load(.acquire);
            if (u < c.max_ui_seen) unreachable;
            if (u > c.max_ui_seen) c.max_ui_seen = u;
            c.scene.last_latched_ui_seq.store(u, .monotonic);
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
    try std.testing.expectEqual(total_publishes, scene.ui_packet_seq.load(.monotonic));
    try std.testing.expectEqual(total_publishes, scene.last_latched_ui_seq.load(.monotonic));
    try std.testing.expectEqual(total_publishes, ctx.max_build_seen);
    try std.testing.expectEqual(total_publishes, ctx.max_ui_seen);
    // The published slot index rode the same edge: it names a real slot and
    // the tail handoff is still held (released, never wedged).
    try std.testing.expect(scene.build_slot.load(.monotonic) < scene.draws.slots.len);
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());
}

// ---- Wave 32: particle/physics freeze-then-latch (slot payload). ----

// CPU-only particle system for scene-level freeze/latch tests: built with
// the shared headless helper (no sg.* anywhere — the init defers buffer
// creation off-context, and here construction itself is plain CPU allocs),
// appended to the layer list by pointer. Teardown is manual (free + list
// deinit): `ParticleLayer.deinit` would run `ps.deinit()` (sg destroys for
// the canary ids below), so scene tests that stage canary handle ids must
// not call it.
fn wave32PushTestSystem(scene: *Scene, capacity: usize) !*ParticleSystem {
    const psys_mod = @import("../particles/system.zig");
    const ps = try scene.allocator.create(ParticleSystem);
    errdefer scene.allocator.destroy(ps);
    ps.* = try psys_mod.makeTestSystem(scene.allocator, capacity);
    errdefer psys_mod.freeTestSystem(ps);
    try scene.particles.systems.append(scene.allocator, ps);
    return ps;
}

fn wave32FreeTestSystems(scene: *Scene) void {
    const psys_mod = @import("../particles/system.zig");
    for (scene.particles.systems.items) |ps| {
        psys_mod.freeTestSystem(ps);
        scene.allocator.destroy(ps);
    }
    scene.particles.systems.deinit(scene.allocator);
    scene.particles.frame.deinit(scene.allocator);
    scene.particles.build_frame.deinit(scene.allocator);
}

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
    scene.buildPreparedFrame();
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
    scene.prepareFrame();
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
    // Both layer generations were consumed (a later fallback runs live).
    try std.testing.expectEqual(scene.particles.build_seq.load(.acquire), scene.particles.latched_seq);
    try std.testing.expectEqual(scene.physics.build_seq.load(.acquire), scene.physics.latched_seq);
    try std.testing.expectEqual(@as(usize, 0), scene.draws.pinsHeld());
}

test "wave32: fallback (no build) still captures particles+physics live" {
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

    // No build: prepare runs the historical inline captures unchanged.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(u64, 0), scene.build_seq.load(.monotonic));
    try std.testing.expectEqual(@as(usize, 1), scene.particles.frame.items.len);
    try std.testing.expectEqual(@as(usize, 2), scene.particles.frame.items[0].active_count);
    try std.testing.expect(scene.physics.prepared_visible);
    try std.testing.expectEqual(@as(usize, 12), scene.physics.prepared_lines.items.len);
    // The slots staged nothing: a skipped path never resurfaces a prior frame.
    const front = scene.draws.slotAtConst(scene.draws.front);
    try std.testing.expectEqual(@as(usize, 0), front.particle_draws.items.len);
    try std.testing.expectEqual(@as(usize, 0), front.physics_lines.items.len);
    try std.testing.expect(!front.physics_visible);
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
    // the game-side build core reads the plain `front` word (commit over
    // the front slot) — flipping it here would race that read outside the
    // lease. The handoff edge under test (claim/build/publish vs
    // claimSlot/validate/latch) is fully exercised without the flip.
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

fn freeSoftbodyFixture(alloc: std.mem.Allocator, scene: *Scene) void {
    // Bodies first (solver + staging CPU only; meshes stay registered),
    // then meshes, materials, and the retire queue.
    scene.softbodies.deinit(alloc);
    for (scene.meshes.items) |m| {
        m.deinit(alloc);
        alloc.destroy(m);
    }
    scene.meshes.deinit(alloc);
    for (scene.materials.items) |m| alloc.destroy(m);
    scene.materials.deinit(alloc);
    scene.gpu_retire.deinit(alloc);
    scene.outline_meshes.deinit(alloc);
    scene.lights.deinit(alloc);
}

test "softbody add/get/count/cap/validation/remove errors" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer freeSoftbodyFixture(alloc, &scene);

    const b0 = try scene.addSoftBodyCloth("cloth_a", .{ .width = 4, .height = 4 });
    try std.testing.expectEqual(@as(usize, 1), scene.softBodyCount());
    try std.testing.expectEqual(b0, scene.getSoftBody(0).?);
    try std.testing.expectEqual(b0, scene.getSoftBodyByName("cloth_a").?);
    try std.testing.expect(scene.getSoftBody(7) == null);
    try std.testing.expect(scene.getSoftBodyByName("nope") == null);

    // Invalid options never register a body.
    try std.testing.expectError(error.InvalidOptions, scene.addSoftBodyCloth("bad", .{ .width = 1 }));
    try std.testing.expectEqual(@as(usize, 1), scene.softBodyCount());

    _ = try scene.addSoftBodyCloth("cloth_b", .{ .width = 3, .height = 3 });
    _ = try scene.addSoftBodyCloth("cloth_c", .{ .width = 3, .height = 3 });
    _ = try scene.addSoftBodyCloth("cloth_d", .{ .width = 3, .height = 3 });
    try std.testing.expectEqual(@as(usize, 4), scene.softBodyCount());
    try std.testing.expectError(error.TooManySoftBodies, scene.addSoftBodyCloth("cloth_e", .{ .width = 3, .height = 3 }));
    try std.testing.expectError(error.UnknownSoftBody, scene.removeSoftBodyCloth(9));
    try std.testing.expectEqual(@as(usize, 4), scene.softBodyCount());
}

test "softbody mesh coupling: vertex layout matches the solver grid" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer freeSoftbodyFixture(alloc, &scene);

    const body = try scene.addSoftBodyCloth("weave", .{ .width = 5, .height = 4, .spacing = 0.5 });
    try std.testing.expectEqual(@as(usize, 20), body.vertices.len);
    try std.testing.expectEqual(@as(u32, 20), body.mesh.vertex_count);
    try std.testing.expectEqual(@as(usize, 4 * 3 * 6), body.indices.len);
    try std.testing.expectEqual(@as(u32, 72), body.mesh.index_count);
    for (body.indices) |idx| try std.testing.expect(idx < 20);
    // Rest pose == solver positions; uv spans the unit square.
    for (body.vertices, body.cloth.pos) |v, p| try std.testing.expectEqual(p.toArray(), v.position);
    try std.testing.expectEqual([2]f32{ 0.0, 0.0 }, body.vertices[0].uv);
    try std.testing.expectEqual([2]f32{ 1.0, 1.0 }, body.vertices[19].uv);
    // Double-sided standard material; mesh registered in the scene.
    const is_standard = switch (body.mesh.material.?) {
        .standard => true,
        else => false,
    };
    try std.testing.expect(is_standard);
    try std.testing.expect(body.mesh.material.?.standard.double_sided);
    try std.testing.expectEqual(@as(usize, 1), scene.meshes.items.len);
    try std.testing.expect(body.mesh.local_bounding_box.isValid());
}

test "softbody upload flagged once per changed frame, cleared by flush" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer freeSoftbodyFixture(alloc, &scene);
    _ = upload_meter.takeAndReset();

    const body = try scene.addSoftBodyCloth("flag", .{ .width = 4, .height = 4 });
    try std.testing.expect(!body.upload_pending);
    scene.updateSoftBodies(1.0 / 60.0);
    try std.testing.expect(body.upload_pending);
    // Second changed frame: still exactly one pending flag, never queued.
    scene.updateSoftBodies(1.0 / 60.0);
    try std.testing.expect(body.upload_pending);
    // Staged vertices track the solver.
    for (body.vertices, body.cloth.pos) |v, p| try std.testing.expectEqual(p.toArray(), v.position);
    // Headless flush clears the flag with no sg.* and no meter bytes.
    scene.flushPendingGpuUploads();
    try std.testing.expect(!body.upload_pending);
    try std.testing.expectEqual(@as(u64, 0), upload_meter.peek());
    // Zero dt steps nothing and flags nothing.
    scene.updateSoftBodies(0.0);
    try std.testing.expect(!body.upload_pending);
    _ = upload_meter.takeAndReset();
}

test "softbody remove retires the mesh (retire-safe, epoch-queued)" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer freeSoftbodyFixture(alloc, &scene);

    const body = try scene.addSoftBodyCloth("bye", .{ .width = 4, .height = 4 });
    const mesh = body.mesh;
    try scene.removeSoftBodyCloth(0);
    try std.testing.expectEqual(@as(usize, 0), scene.softBodyCount());
    // Mesh unlinked from the registry but alive in the retire queue.
    for (scene.meshes.items) |m| try std.testing.expect(m != mesh);
    try std.testing.expectEqual(@as(usize, 1), scene.gpu_retire.retainedCount());
    // Headless flush completes the teardown (bufferless mesh: no sg.*).
    scene.flushPendingGpuUploads();
    try std.testing.expectEqual(@as(usize, 0), scene.gpu_retire.retainedCount());
}

test "softbody remove from a worker retires without sg.* (update||render)" {
    const alloc = std.testing.allocator;
    gpu_thread.markContextThread();
    var scene = @import("../testing.zig").testScene(alloc);
    defer freeSoftbodyFixture(alloc, &scene);

    _ = try scene.addSoftBodyCloth("worker", .{ .width = 4, .height = 4 });
    const Job = struct {
        scene: *Scene,
        fn run(j: @This()) void {
            j.scene.removeSoftBodyCloth(0) catch unreachable;
        }
    };
    const t = try std.Thread.spawn(.{}, Job.run, .{Job{ .scene = &scene }});
    t.join();
    try std.testing.expectEqual(@as(usize, 0), scene.softBodyCount());
    try std.testing.expectEqual(@as(usize, 1), scene.gpu_retire.retainedCount());
    scene.gpu_retire.complete(scene.gpu_retire.current());
    scene.flushPendingGpuUploads();
    try std.testing.expectEqual(@as(usize, 0), scene.gpu_retire.retainedCount());
}

test "softbody destroyMesh drops the bound body (referent cleanup)" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer freeSoftbodyFixture(alloc, &scene);

    const body = try scene.addSoftBodyCloth("doomed", .{ .width = 4, .height = 4 });
    const mesh = body.mesh;
    scene.destroyMesh(mesh);
    try std.testing.expectEqual(@as(usize, 0), scene.softBodyCount());
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
}

test "softbody disabled body pauses: no step, no upload flag" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer freeSoftbodyFixture(alloc, &scene);

    const body = try scene.addSoftBodyCloth("paused", .{ .width = 4, .height = 4 });
    body.cloth.enabled = false;
    const h = body.cloth.hashState();
    scene.updateSoftBodies(1.0 / 60.0);
    try std.testing.expectEqual(h, body.cloth.hashState());
    try std.testing.expect(!body.upload_pending);
    body.cloth.enabled = true;
    scene.updateSoftBodies(1.0 / 60.0);
    try std.testing.expect(body.upload_pending);
}

// ---- Wave 31: prepareFrame holds its working slot as a lease claim. ----
//
// prepare resolves its slot via `claimBack` (fallback) or
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
        scene.prepareFrame();
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

    // Baseline: the fallback and latch exits pair every claim.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(scene.build_seq.load(.monotonic), scene.last_latched_seq.load(.monotonic));
    try expectClean(&scene);

    // (a) Latch-path contention: a pending build whose handoff slot is held
    // for writing (a concurrent game reclaim mid-fill for the NEXT build)
    // degrades to a counted skip — nothing consumed, the build stays fresh,
    // the slot payload is untouched, no side effect ran (the claim sits
    // before the retire/stats/frame prologue).
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.buildPreparedFrame();
    const pending_slot = scene.build_slot.load(.monotonic);
    const pending_seq = scene.build_seq.load(.monotonic);
    const latched_before = scene.last_latched_seq.load(.monotonic);
    try std.testing.expect(pending_seq != latched_before);
    const snap_w = scene.draws.slots[pending_slot].snapshot.screen_w;
    try scene.draws.claimSlot(pending_slot);
    const sat0 = scene.draws.saturation_skips;
    const front0 = scene.draws.front;
    const fid0 = scene.frame_id;
    scene.prepareFrame();
    // Skip discards nothing: the earlier pending frame stays consumable.
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(front0, scene.draws.front);
    try std.testing.expectEqual(latched_before, scene.last_latched_seq.load(.monotonic));
    try std.testing.expectEqual(pending_seq, scene.build_seq.load(.monotonic));
    try std.testing.expectEqual(sat0 + 1, scene.draws.saturation_skips);
    try std.testing.expectEqual(fid0, scene.frame_id);
    try std.testing.expectEqual(snap_w, scene.draws.slots[pending_slot].snapshot.screen_w);
    // Release and retry: the pending build latches exactly once.
    try scene.draws.cancelClaim(pending_slot);
    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(pending_slot, scene.draws.front);
    try std.testing.expectEqual(pending_seq, scene.last_latched_seq.load(.monotonic));
    try expectClean(&scene);

    // (b) Latch-path pinned handoff: a presented frame is never overwritten —
    // PinnedSlot refusal, counted, build stays fresh; unpin restores.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.buildPreparedFrame();
    const pinned_slot = scene.build_slot.load(.monotonic);
    const pinned_seq = scene.build_seq.load(.monotonic);
    try std.testing.expect(pinned_seq != scene.last_latched_seq.load(.monotonic));
    try scene.draws.pin(pinned_slot);
    const ref0 = scene.draws.publish_refusals;
    scene.prepareFrame();
    // Skip discards nothing: the earlier pending frame stays consumable.
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(ref0 + 1, scene.draws.publish_refusals);
    try std.testing.expectEqual(pinned_seq, scene.build_seq.load(.monotonic));
    try std.testing.expect(pinned_seq != scene.last_latched_seq.load(.monotonic));
    try scene.draws.unpin(pinned_slot);
    scene.prepareFrame();
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(pinned_slot, scene.draws.front);
    try expectClean(&scene);

    // (c) Fallback saturation: every non-front slot held degrades to a
    // counted skip, never a wedge; releasing restores the rotation.
    try scene.draws.pin(scene.draws.front);
    const h1 = scene.draws.claimBack().?;
    const h2 = scene.draws.claimBack().?;
    try std.testing.expect(scene.draws.claimBack() == null);
    const sat1 = scene.draws.saturation_skips;
    const front1 = scene.draws.front;
    scene.prepareFrame();
    // Skip discards nothing: the earlier pending frame stays consumable.
    try std.testing.expect(scene.frame_prepared);
    try std.testing.expectEqual(front1, scene.draws.front);
    try std.testing.expectEqual(sat1 + 1, scene.draws.saturation_skips);
    try scene.draws.cancelClaim(h1);
    try scene.draws.cancelClaim(h2);
    try scene.draws.unpin(front1);
    scene.prepareFrame();
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

    // Fresh scene: saturate every non-front slot, then prepare. With no
    // earlier pending frame the skip keeps `frame_prepared` false and the
    // front untouched — and nothing is consumable yet.
    try scene.draws.pin(scene.draws.front);
    const g1 = scene.draws.claimBack().?;
    const g2 = scene.draws.claimBack().?;
    const sat0 = scene.draws.saturation_skips;
    scene.prepareFrame();
    try std.testing.expect(!scene.frame_prepared);
    try std.testing.expectEqual(@as(usize, 0), scene.draws.front);
    try std.testing.expectEqual(sat0 + 1, scene.draws.saturation_skips);
    try std.testing.expect(!scene.hasConsumableFrame());

    // `render` on the skipped state drops the present via its fallback
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
    scene.prepareFrame();
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

    // Prime one fallback frame so the rotation is warm.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    const front0 = scene.draws.front;

    // A claim taken exactly the way the fallback prepare takes it.
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

    // The legacy stage path steers away too: it targets the unlocked back
    // index, which skips WRITING slots — never the held one.
    try std.testing.expect(scene.draws.backIndex() != held);
    scene.stageUiPacket();
    try std.testing.expect(!scene.draws.slots[held].ui_packet.valid);

    // Release the simulated prepare hold, then run the real prepare: it
    // latches exactly the published handoff slot and pairs every claim.
    try scene.draws.cancelClaim(held);
    scene.prepareFrame();
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
    };

    const Producer = struct {
        fn run(c: *Ctx) void {
            // Game-side concurrent flow (UI-only claims, the wave-30 shape):
            // reserve, stage, hand off — or cancel every 16th reservation to
            // exercise `cancelClaim` under prepare pressure. Never touches
            // live scene state: the claimed slot payload is producer-owned
            // (no canvas, so no canvas reads either) and the seq words are
            // the release edge. Full `build()` calls stay out: the live
            // mesh/snapshot reads they need are still phase-excluded (the
            // documented remainder, not this slice).
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
                    claim.stageUi();
                    claim.publish();
                    c.published += 1;
                }
                n += 1;
            }
        }
    };

    const sat0 = scene.draws.saturation_skips;
    var ctx = Ctx{ .scene = &scene, .claims = total_claims };
    const prod = try std.Thread.spawn(.{}, Producer.run, .{&ctx});
    var prepared_ok: usize = 0;
    var i: usize = 0;
    while (i < total_prepares) : (i += 1) {
        // Per-call success is the front flip: a contended prepare keeps a
        // stale `frame_prepared` from an earlier pending frame (the skip
        // discards nothing), so the flag alone cannot count skips — while
        // every genuine prepare flips to a slot that was not the front.
        const f0 = scene.draws.front;
        scene.prepareFrame();
        if (scene.draws.front != f0) prepared_ok += 1;
    }
    prod.join();
    // Drain: consume any build published after the last prepare. No
    // contention is left (the producer is joined), so every drain prepare
    // flips unless nothing is pending.
    var drain: usize = 0;
    while (scene.last_latched_seq.load(.monotonic) != scene.build_seq.load(.monotonic) and drain < 10) : (drain += 1) {
        const f0 = scene.draws.front;
        scene.prepareFrame();
        try std.testing.expect(scene.draws.front != f0);
    }
    try std.testing.expectEqual(scene.build_seq.load(.monotonic), scene.last_latched_seq.load(.monotonic));

    // Exact skip accounting: every game-side null-claim and every contended
    // prepare (fallback null-claim and latch SlotBusy alike) bumps
    // `saturation_skips` exactly once — the sum must match, proving no
    // silent interference in either direction (a missed release would wedge
    // the drain or leak WRITING below instead).
    const prepare_skips = total_prepares - prepared_ok;
    try std.testing.expectEqual(ctx.skipped + prepare_skips, scene.draws.saturation_skips - sat0);
    // Every non-cancelled reservation published exactly once (the seqs ARE
    // the counters: one stage + one generation per publish, cancels commit
    // nothing).
    try std.testing.expectEqual(ctx.published + ctx.cancelled, total_claims);
    try std.testing.expectEqual(ctx.published, scene.build_seq.load(.monotonic));
    try std.testing.expectEqual(ctx.published, scene.ui_packet_seq.load(.monotonic));
    try std.testing.expectEqual(ctx.published, scene.last_latched_ui_seq.load(.monotonic));
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
// The game-side queue build accumulates into the live `Scene.build_stats`
// accumulator; the build freezes a plain copy into the claimed slot's
// `build_stats` (staged-wins over post-build accumulation, same precedent
// as the slot snapshot); `prepareFrame` merges the SLOT copy into `stats`
// instead of reading the shared field. Sequential behavior is bit-identical
// (the stage-2B equivalence tests below the wave-29 block prove the merged
// values unchanged); these tests pin the staging points.

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

    // Staging point: the claimed slot freezes exactly what the build
    // accumulated (whole-struct equality, plain copy).
    const slot = claim.slot;
    try std.testing.expectEqual(scene.build_stats, scene.draws.slots[slot].build_stats);
    // The fixture is meaningful: the queue build counted the mesh.
    const staged = scene.draws.slots[slot].build_stats;
    try std.testing.expect(staged.total_meshes > 0);

    // Post-build game-side accumulation AFTER the freeze: staged wins, so
    // none of this may reach the prepared frame's merge — including the
    // assign-merge fields (occluders_* use `=`, not `+=`).
    scene.build_stats.total_meshes += 1000;
    scene.build_stats.rendered_meshes += 1000;
    scene.build_stats.culled_meshes += 1000;
    scene.build_stats.occluded_meshes += 1000;
    scene.build_stats.occluders_count = 12345;
    scene.build_stats.occluder_triangles = 67890;
    claim.publish();

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expectEqual(scene.build_seq.load(.monotonic), scene.last_latched_seq.load(.monotonic));

    // Merge point: stats carry exactly the staged copy, not the live field.
    try std.testing.expectEqual(staged.total_meshes, scene.stats.total_meshes);
    try std.testing.expectEqual(staged.rendered_meshes, scene.stats.rendered_meshes);
    try std.testing.expectEqual(staged.culled_meshes, scene.stats.culled_meshes);
    try std.testing.expectEqual(staged.occluded_meshes, scene.stats.occluded_meshes);
    try std.testing.expectEqual(staged.occluders_count, scene.stats.occluders_count);
    try std.testing.expectEqual(staged.occluder_triangles, scene.stats.occluder_triangles);
    // Both copies are cleared for the next tick (same reset semantics).
    try std.testing.expectEqual(SceneStats{}, scene.build_stats);
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
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expect(scene.stats.total_meshes > 0);

    // Round 2 (mesh list emptied, slots reused): the reused slot stages a
    // fresh zero copy — no stale round-1 stats resurface in the merge.
    scene.meshes.clearRetainingCapacity();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.buildPreparedFrame();
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    scene.prepareFrame();
    try std.testing.expectEqual(@as(u32, 0), scene.stats.total_meshes);
    try std.testing.expectEqual(@as(u32, 0), scene.stats.rendered_meshes);
    try std.testing.expectEqual(@as(u32, 0), scene.stats.culled_meshes);
    try std.testing.expectEqual(@as(u32, 0), scene.stats.occluded_meshes);
    try std.testing.expectEqual(@as(u32, 0), scene.stats.occluders_count);
    try std.testing.expectEqual(@as(u32, 0), scene.stats.occluder_triangles);
}

test "Scene tag queries and tag-filtered raycasting" {
    const alloc = std.testing.allocator;
    var scene: Scene = undefined;
    scene.allocator = alloc;
    scene.meshes = .empty;
    scene.frame_id = 0;
    defer scene.meshes.deinit(alloc);

    var m1 = Mesh{
        .name = "orc_grunt",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(0, 0, 5),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, 4), Vec3.new(1, 1, 6)),
    };
    defer m1.deinit(alloc);
    _ = try m1.addTags(alloc, "enemy, orc, melee");
    try scene.meshes.append(alloc, &m1);

    var m2 = Mesh{
        .name = "orc_boss",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(10, 0, 5),
        .local_bounding_box = BoundingBox.init(Vec3.new(9, -1, 4), Vec3.new(11, 1, 6)),
    };
    defer m2.deinit(alloc);
    _ = try m2.addTags(alloc, "enemy, orc, boss, elite");
    try scene.meshes.append(alloc, &m2);

    var m3 = Mesh{
        .name = "player_hero",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .position = Vec3.new(0, 0, -5),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -6), Vec3.new(1, 1, -4)),
    };
    defer m3.deinit(alloc);
    _ = try m3.addTags(alloc, "player, hero");
    try scene.meshes.append(alloc, &m3);

    // Test getMeshByName
    try std.testing.expectEqual(&m1, scene.getMeshByName("orc_grunt"));
    try std.testing.expectEqual(&m2, scene.getMeshByName("orc_boss"));
    try std.testing.expect(scene.getMeshByName("nonexistent") == null);

    // Test countMeshesByTag
    try std.testing.expectEqual(@as(usize, 2), scene.countMeshesByTag("enemy"));
    try std.testing.expectEqual(@as(usize, 1), scene.countMeshesByTag("boss"));
    try std.testing.expectEqual(@as(usize, 1), scene.countMeshesByTag("player"));
    try std.testing.expectEqual(@as(usize, 0), scene.countMeshesByTag("dragon"));

    // Test getMeshesByTag
    var enemies = try scene.getMeshesByTag(alloc, "enemy");
    defer enemies.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 2), enemies.items.len);

    // Test getMeshesByQuery
    var boss_enemies = try scene.getMeshesByQuery(alloc, "enemy && boss");
    defer boss_enemies.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 1), boss_enemies.items.len);
    try std.testing.expectEqual(&m2, boss_enemies.items[0]);

    var non_boss = try scene.getMeshesByQuery(alloc, "enemy && !boss");
    defer non_boss.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 1), non_boss.items.len);
    try std.testing.expectEqual(&m1, non_boss.items[0]);

    // Test findFirstMesh
    try std.testing.expectEqual(&m2, scene.findFirstMeshByTag("boss"));
    try std.testing.expectEqual(&m3, scene.findFirstMeshByQuery("hero"));

    // Test pickWithRayTag: ray looking along +Z at (0, 0, 5) hits m1
    const ray = Ray.new(Vec3.new(0, 0, 0), Vec3.new(0, 0, 1));
    const hit_any_enemy = scene.pickWithRayTag(ray, "enemy");
    try std.testing.expect(hit_any_enemy.hit);
    try std.testing.expectEqual(&m1, hit_any_enemy.picked_mesh.?);

    // Query for boss along the same ray should NOT hit m1 because m1 lacks "boss" tag
    const hit_boss = scene.pickWithRayTag(ray, "boss");
    try std.testing.expect(!hit_boss.hit);

    // Query for hero along the same ray should NOT hit
    const hit_hero = scene.pickWithRayTag(ray, "player");
    try std.testing.expect(!hit_hero.hit);
}
