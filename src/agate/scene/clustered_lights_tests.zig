const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Color3 = math.Color3;
const camera_mod = @import("../camera.zig");
const Camera = camera_mod.Camera;
const scene_mod = @import("../scene.zig");
const Scene = scene_mod.Scene;
const scene_lights = @import("light_rig.zig");
const cluster_lights = @import("../lights.zig");

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

test "Scene exposes clustered spot pool: stages into FramePack, cap at 32, order-preserving removal" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);

    var pack: scene_lights.LightRig.FramePack = undefined;
    scene.updateLights(0.016);
    try std.testing.expect(scene.light_handoff.takeLatest(&pack));
    try std.testing.expectEqual(@as(usize, 0), pack.clustered_spot_count);

    const idx0 = try scene.addClusteredSpotLight(Vec3.new(1.0, 2.0, 3.0), .{
        .direction = Vec3.new(0, -1, 0),
        .color = Color3.new(1.0, 0.0, 0.0),
        .intensity = 2.0,
        .range = 15.0,
        .inner_angle_deg = 20.0,
        .outer_angle_deg = 40.0,
    });
    const idx1 = try scene.addClusteredSpotLight(Vec3.zero, .{});
    try std.testing.expectEqual(@as(usize, 0), idx0);
    try std.testing.expectEqual(@as(usize, 1), idx1);
    try std.testing.expectEqual(@as(usize, 2), scene.clusteredSpotLightCount());
    try std.testing.expect(scene.getClusteredSpotLight(0).?.position.x == 1.0);
    try std.testing.expect(scene.getClusteredSpotLight(2) == null);

    scene.getClusteredSpotLight(0).?.is_enabled = false;
    scene.updateLights(0.016);
    _ = scene.light_handoff.takeLatest(&pack);
    try std.testing.expectEqual(@as(usize, 2), pack.clustered_spot_count);
    try std.testing.expectEqual([4]f32{ 0, 0, 0, 0 }, pack.clustered_spot_pos_range[0]);
    scene.getClusteredSpotLight(0).?.is_enabled = true;

    // Hard cap: past 32 lights the add errors and the count is unchanged.
    var k: usize = 2;
    while (k < cluster_lights.max_clustered_spots) : (k += 1) {
        _ = try scene.addClusteredSpotLight(Vec3.zero, .{});
    }
    try std.testing.expectEqual(cluster_lights.max_clustered_spots, scene.clusteredSpotLightCount());
    try std.testing.expectError(error.TooManyClusteredLights, scene.addClusteredSpotLight(Vec3.zero, .{}));
    try std.testing.expectEqual(cluster_lights.max_clustered_spots, scene.clusteredSpotLightCount());

    scene.removeClusteredSpotLight(7_000); // out of range
    try std.testing.expectEqual(cluster_lights.max_clustered_spots, scene.clusteredSpotLightCount());
    scene.removeClusteredSpotLight(0);
    try std.testing.expectEqual(cluster_lights.max_clustered_spots - 1, scene.clusteredSpotLightCount());
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
    // never created or destroyed through sg here). Slot 0 = primary.
    scene.clustered.slots[0].light_buffer = .{ .id = 41 };
    scene.clustered.slots[0].header_buffer = .{ .id = 42 };
    scene.clustered.slots[0].index_buffer = .{ .id = 43 };
    scene.clustered.slots[0].live = true;

    // Out-of-range removal is a no-op: count unchanged, nothing retired.
    scene.removeClusteredPointLight(9);
    try std.testing.expectEqual(@as(usize, 2), scene.clusteredPointLightCount());
    try std.testing.expectEqual(@as(usize, 0), scene.gpu_retire.retainedCount());

    // Real removal retires all three live tile buffers (epoch-stamped
    // appends, no sg.*) and zeroes the handles for the next rebuild.
    scene.removeClusteredPointLight(0);
    try std.testing.expectEqual(@as(usize, 1), scene.clusteredPointLightCount());
    try std.testing.expectEqual(@as(usize, 3), scene.gpu_retire.retainedCount());
    try std.testing.expectEqual(@as(u32, 0), scene.clustered.slots[0].light_buffer.id);
    try std.testing.expectEqual(@as(u32, 0), scene.clustered.slots[0].header_buffer.id);
    try std.testing.expectEqual(@as(u32, 0), scene.clustered.slots[0].index_buffer.id);
    try std.testing.expect(!scene.clustered.isLive(0));
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
