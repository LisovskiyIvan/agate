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

const sokol = @import("sokol");
const sg = sokol.gfx;
const Mat4 = math.Mat4;
const cl = @import("clustered_lights.zig");
const ClusteredGpuCache = cl.ClusteredGpuCache;
const TileGrid = cl.TileGrid;
const ViewRect = cl.ViewRect;
const tilesForViewport = cl.tilesForViewport;
const tileNdcRect = cl.tileNdcRect;
const buildTileLists = cl.buildTileLists;
const buildTileListsFromLights = cl.buildTileListsFromLights;
const ClusterLightGpu = cl.ClusterLightGpu;
const ClusterTileGpu = cl.ClusterTileGpu;
const MAX_VIEW_SLOTS = cl.MAX_VIEW_SLOTS;
const RTT_VIEW_SLOT = cl.RTT_VIEW_SLOT;
const REFRACTION_VIEW_SLOT = cl.REFRACTION_VIEW_SLOT;
const snapshot = @import("snapshot.zig");

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

test "rebuildCpuFromLights builds tiles with spots and shadow-casting lights" {
    const alloc = std.testing.allocator;
    const scene_clustered = @import("clustered_lights.zig");
    var cache = scene_clustered.ClusteredGpuCache{};
    defer {
        cache.cpu_lights.deinit(alloc);
        cache.cpu_headers.deinit(alloc);
        cache.cpu_indices.deinit(alloc);
    }

    const lights_arr = [_]scene_clustered.ClusterLightGpu{
        .{
            .pos_range = .{ 0.75, 0.75, 0.0, 5.0 },
            .color_int = .{ 1.0, 0.0, 0.0, 2.0 },
            .dir_inner = .{ 0, 0, 0, -2.0 },
            .spot_params = .{ -2.0, 1.0, 0.0, 0.002 }, // Point caster
        },
        .{
            .pos_range = .{ -0.75, -0.75, 0.0, 5.0 },
            .color_int = .{ 0.0, 1.0, 0.0, 2.0 },
            .dir_inner = .{ 0, -1, 0, 0.9 }, // Spot caster
            .spot_params = .{ 0.7, 2.0, 1.0, 0.003 },
        },
    };

    const full = scene_clustered.ViewRect{ .x = 0, .y = 0, .w = 128, .h = 128 };
    try cache.rebuildCpuFromLights(alloc, &lights_arr, math.Mat4.identity, 128, 128, full, 0);

    try std.testing.expectEqual(@as(usize, 2), cache.staged_count);
    try std.testing.expectEqual(@as(usize, 2), cache.cpu_lights.items.len);
    try std.testing.expectEqual(@as(f32, 1.0), cache.cpu_lights.items[0].spot_params[1]); // point shadow tag
    try std.testing.expectEqual(@as(f32, 2.0), cache.cpu_lights.items[1].spot_params[1]); // spot shadow tag
    try std.testing.expect(cache.cpu_indices.items.len > 0);
}

test "growBytes rounds up to pow2 with a 64 B floor" {
    try std.testing.expectEqual(@as(usize, 64), ClusteredGpuCache.growBytes(0));
    try std.testing.expectEqual(@as(usize, 64), ClusteredGpuCache.growBytes(1));
    try std.testing.expectEqual(@as(usize, 64), ClusteredGpuCache.growBytes(64));
    try std.testing.expectEqual(@as(usize, 128), ClusteredGpuCache.growBytes(65));
    try std.testing.expectEqual(@as(usize, 256), ClusteredGpuCache.growBytes(200));
    try std.testing.expectEqual(@as(usize, 4096), ClusteredGpuCache.growBytes(2132));
    try std.testing.expectEqual(@as(usize, 8192), ClusteredGpuCache.growBytes(6240));
}

test "tilesForViewport is ceiling division with empty-viewport zero" {
    try std.testing.expectEqual(TileGrid{ .x = 0, .y = 0 }, tilesForViewport(0, 480));
    try std.testing.expectEqual(TileGrid{ .x = 0, .y = 0 }, tilesForViewport(640, 0));
    try std.testing.expectEqual(TileGrid{ .x = 0, .y = 0 }, tilesForViewport(-3, 480));
    try std.testing.expectEqual(TileGrid{ .x = 1, .y = 1 }, tilesForViewport(64, 64));
    try std.testing.expectEqual(TileGrid{ .x = 2, .y = 2 }, tilesForViewport(128, 65));
    try std.testing.expectEqual(TileGrid{ .x = 30, .y = 17 }, tilesForViewport(1920, 1080));
}

test "tileNdcRect covers the full NDC box bottom-left first" {
    const full = ViewRect{ .x = 0, .y = 0, .w = 128, .h = 128 };
    // 2x2 grid over a 128px window: tile (0,0) is bottom-left
    // [-1,0]x[-1,0], (1,1) top-right.
    try std.testing.expectEqual([4]f32{ -1, 0, -1, 0 }, tileNdcRect(0, 0, full, 128, 128));
    try std.testing.expectEqual([4]f32{ 0, 1, 0, 1 }, tileNdcRect(1, 1, full, 128, 128));
    try std.testing.expectEqual([4]f32{ -1, 0, 0, 1 }, tileNdcRect(0, 1, full, 128, 128));
    // Edge tiles meet exactly at the shared border (no gaps/overlaps).
    const wide = ViewRect{ .x = 0, .y = 0, .w = 256, .h = 64 };
    const left = tileNdcRect(0, 0, wide, 256, 64);
    const right = tileNdcRect(1, 0, wide, 256, 64);
    try std.testing.expectEqual(left[1], right[0]);
    // Sub-viewport: a left-half view maps its own pixels to full NDC.
    const half = ViewRect{ .x = 0, .y = 0, .w = 64, .h = 128 };
    try std.testing.expectEqual([4]f32{ -1, 1, -1, 0 }, tileNdcRect(0, 0, half, 128, 128));
    // Degenerate view rect: zero rect (planes go no-constraint upstream).
    try std.testing.expectEqual([4]f32{ 0, 0, 0, 0 }, tileNdcRect(0, 0, .{}, 128, 128));
}

test "buildTileLists assigns center lights, excludes off-screen and dead lanes" {
    // Identity view-projection: world == NDC (w = 1 everywhere).
    const vp = Mat4.identity;
    const full = ViewRect{ .x = 0, .y = 0, .w = 128, .h = 128 };
    const tx: u32 = 2;
    const ty: u32 = 2;
    var headers = [_][2]u32{.{ 0, 0 }} ** 4;
    var indices = [_]u32{0} ** 16;
    const pos = [_][4]f32{
        .{ 0.75, 0.75, 0.0, 0.5 }, // small light deep inside tile (1,1)
        .{ 5.0, 5.0, 0.0, 0.5 }, // off-screen entirely: excluded
        .{ -0.75, -0.75, 0.0, 0.5 }, // small light deep inside tile (0,0)
    };
    const col = [_][4]f32{
        .{ 1, 0, 0, 2.0 },
        .{ 0, 1, 0, 2.0 },
        .{ 0, 0, 1, 2.0 },
    };
    const n = buildTileLists(&pos, &col, 3, vp, tx, ty, full, 128, 128, &headers, &indices);
    // Light 0 only in tile (1,1) [id 3], light 2 only in tile (0,0) [id 0].
    try std.testing.expectEqual(@as(u32, 1), headers[3][1]);
    try std.testing.expectEqual(@as(u32, 1), headers[0][1]);
    try std.testing.expectEqual(@as(u32, 0), headers[1][1]);
    try std.testing.expectEqual(@as(u32, 0), headers[2][1]);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expectEqual(@as(u32, 2), indices[headers[0][0]]);
    try std.testing.expectEqual(@as(u32, 0), indices[headers[3][0]]);

    // Disabled lane (intensity 0) and degenerate radius pack as absent.
    const col_off = [_][4]f32{
        .{ 1, 0, 0, 0.0 },
        .{ 0, 1, 0, 2.0 },
        .{ 0, 0, 1, 2.0 },
    };
    const pos_flat = [_][4]f32{
        .{ 0.75, 0.75, 0.0, 0.0 },
        .{ 5.0, 5.0, 0.0, 0.5 },
        .{ -0.75, -0.75, 0.0, 0.5 },
    };
    const n2 = buildTileLists(&pos_flat, &col_off, 3, vp, tx, ty, full, 128, 128, &headers, &indices);
    try std.testing.expectEqual(@as(usize, 1), n2);
    try std.testing.expectEqual(@as(u32, 0), headers[3][1]);
    try std.testing.expectEqual(@as(u32, 2), indices[headers[0][0]]);
}

test "buildTileLists perspective: front lights assign, behind lights exclude unless huge" {
    const proj = Mat4.perspective(90.0, 1.0, 0.1, 100.0);
    const view = Mat4.lookAt(Vec3.zero, Vec3.new(0, 0, -1), Vec3.up);
    const vp = Mat4.mul(proj, view);
    const full = ViewRect{ .x = 0, .y = 0, .w = 128, .h = 128 };
    var headers = [_][2]u32{.{ 0, 0 }} ** 4;
    var indices = [_]u32{0} ** 8;
    // Light 5 units ahead: center tile column (x=0 spans tiles (0,*) and (1,*)).
    const pos = [_][4]f32{
        .{ 0, 0, -5, 1.0 },
        .{ 0, 0, 5, 1.0 }, // behind the eye, small: excluded everywhere
        .{ 0, 0, 5, 100.0 }, // behind but enormous: over-included (documented)
    };
    const col = [_][4]f32{
        .{ 1, 1, 1, 1.0 },
        .{ 1, 1, 1, 1.0 },
        .{ 1, 1, 1, 1.0 },
    };
    const n = buildTileLists(&pos, &col, 3, vp, 2, 2, full, 128, 128, &headers, &indices);
    var front_hits: u32 = 0;
    var behind_small_hits: u32 = 0;
    var behind_huge_hits: u32 = 0;
    for (headers) |h| {
        for (indices[h[0] .. h[0] + h[1]]) |li| {
            if (li == 0) front_hits += 1;
            if (li == 1) behind_small_hits += 1;
            if (li == 2) behind_huge_hits += 1;
        }
    }
    try std.testing.expect(front_hits > 0);
    try std.testing.expectEqual(@as(u32, 0), behind_small_hits);
    try std.testing.expect(behind_huge_hits > 0);
    try std.testing.expectEqual(@as(usize, front_hits + behind_huge_hits), n);
}

test "buildTileLists follows each view's own camera (PIP views independent)" {
    const proj = Mat4.perspective(90.0, 1.0, 0.1, 100.0);
    const full = ViewRect{ .x = 0, .y = 0, .w = 128, .h = 128 };
    const pos = [_][4]f32{.{ 0, 0, -5, 1.0 }};
    const col = [_][4]f32{.{ 1, 1, 1, 1.0 }};
    var headers = [_][2]u32{.{ 0, 0 }} ** 4;
    var indices = [_]u32{0} ** 4;
    // Camera A faces the light: some tile lists it.
    const vp_a = Mat4.mul(proj, Mat4.lookAt(Vec3.zero, Vec3.new(0, 0, -1), Vec3.up));
    const na = buildTileLists(&pos, &col, 1, vp_a, 2, 2, full, 128, 128, &headers, &indices);
    try std.testing.expect(na > 0);
    // Camera B faces away (light fully behind it): nothing lists it, so a
    // PIP view never inherits another view's tile lists.
    const vp_b = Mat4.mul(proj, Mat4.lookAt(Vec3.zero, Vec3.new(0, 0, 1), Vec3.up));
    const nb = buildTileLists(&pos, &col, 1, vp_b, 2, 2, full, 128, 128, &headers, &indices);
    try std.testing.expectEqual(@as(usize, 0), nb);
    for (headers) |h| try std.testing.expectEqual(@as(u32, 0), h[1]);
}

test "buildTileLists is empty-safe and deterministic" {
    var headers = [_][2]u32{.{ 9, 9 }} ** 1;
    var indices = [_]u32{0} ** 1;
    const full = ViewRect{ .x = 0, .y = 0, .w = 64, .h = 64 };
    // Empty viewport: nothing written, headers zeroed.
    const n0 = buildTileLists(&.{}, &.{}, 0, Mat4.identity, 0, 0, .{}, 0, 0, &.{}, &.{});
    try std.testing.expectEqual(@as(usize, 0), n0);
    // Zero lights on a live grid: headers zeroed, no indices.
    const n1 = buildTileLists(&.{}, &.{}, 0, Mat4.identity, 1, 1, full, 64, 64, &headers, &indices);
    try std.testing.expectEqual(@as(usize, 0), n1);
    try std.testing.expectEqual([2]u32{ 0, 0 }, headers[0]);
}

test "gpu mirror structs match the std430 shader layout" {
    // ClusterLightGpu == struct { vec4; vec4; vec4; vec4 }: 64 bytes, 16-aligned lanes.
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(ClusterLightGpu));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(ClusterLightGpu, "pos_range"));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(ClusterLightGpu, "color_int"));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(ClusterLightGpu, "dir_inner"));
    try std.testing.expectEqual(@as(usize, 48), @offsetOf(ClusterLightGpu, "spot_params"));
    // ClusterTileGpu == uvec2 (offset, count): 8 bytes, 4-aligned lanes.
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(ClusterTileGpu));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(ClusterTileGpu, "offset"));
    try std.testing.expectEqual(@as(usize, 4), @offsetOf(ClusterTileGpu, "count"));
    // Index entries are plain u32 (one per tile-list slot).
    try std.testing.expectEqual(@as(usize, 4), @sizeOf(u32));
    // The header reinterpret in rebuildCpu (ClusterTileGpu <-> [2]u32).
    try std.testing.expectEqual(@sizeOf(ClusterTileGpu), @sizeOf([2]u32));
}

test "rebuildCpu is headless-safe and stages exact-fit scratch" {
    const alloc = std.testing.allocator;
    var cache = ClusteredGpuCache{};
    defer {
        cache.cpu_lights.deinit(alloc);
        cache.cpu_headers.deinit(alloc);
        cache.cpu_indices.deinit(alloc);
    }
    const pos = [_][4]f32{.{ 0.75, 0.75, 0.0, 5.0 }};
    const col = [_][4]f32{.{ 1, 0, 0, 2.0 }};
    const full = ViewRect{ .x = 0, .y = 0, .w = 128, .h = 128 };
    try cache.rebuildCpu(alloc, &pos, &col, 1, Mat4.identity, 128, 128, full);
    try std.testing.expectEqual(@as(u32, 2), cache.tiles_x);
    try std.testing.expectEqual(@as(u32, 2), cache.tiles_y);
    try std.testing.expectEqual(@as(usize, 1), cache.staged_count);
    try std.testing.expectEqual(@as(usize, 1), cache.cpu_lights.items.len);
    try std.testing.expectEqual(@as(usize, 4), cache.cpu_headers.items.len);
    try std.testing.expect(!cache.isLive(0));
    // No context headless: upload fails closed for every slot, handles
    // stay zero.
    const FakeRetire = struct {
        calls: u32 = 0,
        pub fn retireBuffer(self: *@This(), allocator: std.mem.Allocator, buf: sg.Buffer) void {
            _ = allocator;
            _ = buf;
            self.calls += 1;
        }
    };
    var fake = FakeRetire{};
    try std.testing.expect(!sg.isvalid());
    var s: usize = 0;
    while (s < MAX_VIEW_SLOTS) : (s += 1) {
        try std.testing.expect(!cache.upload(alloc, &fake, s));
        try std.testing.expectEqual(@as(u32, 0), cache.slots[s].light_buffer.id);
        try std.testing.expect(!cache.isLive(s));
    }
    // Dummy fallback binds (zero views headless — the draw never runs
    // without a context; the count uniform gates the shader path).
    const views = cache.bindingViews();
    try std.testing.expectEqual(@as(u32, 0), views.lights.id);

    // Empty pool: zeroed headers, empty indices, still headless-safe.
    try cache.rebuildCpu(alloc, &.{}, &.{}, 0, Mat4.identity, 128, 128, full);
    try std.testing.expectEqual(@as(usize, 0), cache.cpu_lights.items.len);
    try std.testing.expectEqual(@as(usize, 0), cache.cpu_indices.items.len);
    for (cache.cpu_headers.items) |h| {
        try std.testing.expectEqual(@as(u32, 0), h.count);
    }
}

test "retireBuffers moves live buffers into the retire queue and zeroes handles" {
    const alloc = std.testing.allocator;
    const FakeRetire = struct {
        calls: u32 = 0,
        pub fn retireBuffer(self: *@This(), allocator: std.mem.Allocator, buf: sg.Buffer) void {
            _ = allocator;
            _ = buf;
            self.calls += 1;
        }
    };
    var cache = ClusteredGpuCache{};
    cache.slots[0] = .{
        .light_buffer = .{ .id = 7 },
        .header_buffer = .{ .id = 8 },
        .index_buffer = .{ .id = 9 },
        .light_view = .{ .id = 70 },
        .live = true,
        .light_cap = 64,
        .header_cap = 32,
        .index_cap = 128,
    };
    // A second live slot retires too (multi-view frames allocate 1+N).
    cache.slots[2].index_buffer = .{ .id = 11 };
    cache.slots[2].live = true;
    var fake = FakeRetire{};
    cache.retireBuffers(alloc, &fake);
    // All live storage buffers retire (views are context-owned and die
    // with the buffers' generation; the next upload recreates them).
    // Untouched slots hold zero ids and are skipped by retireBuffers.
    try std.testing.expectEqual(@as(u32, 4), fake.calls);
    try std.testing.expectEqual(@as(u32, 0), cache.slots[0].light_buffer.id);
    try std.testing.expectEqual(@as(u32, 0), cache.slots[0].header_buffer.id);
    try std.testing.expectEqual(@as(u32, 0), cache.slots[0].index_buffer.id);
    try std.testing.expectEqual(@as(u32, 0), cache.slots[0].light_view.id);
    try std.testing.expect(!cache.isLive(0));
    try std.testing.expect(!cache.isLive(2));
    try std.testing.expectEqual(@as(usize, 0), cache.slots[0].light_cap);
}

test "view slots clamp defensively and cover every camera" {
    // All cameras fit; capture slots are disjoint from them and each other.
    try std.testing.expectEqual(snapshot.MAX_CAMERAS + 2, MAX_VIEW_SLOTS);
    try std.testing.expect(REFRACTION_VIEW_SLOT >= snapshot.MAX_CAMERAS);
    try std.testing.expect(RTT_VIEW_SLOT > REFRACTION_VIEW_SLOT);
    try std.testing.expect(RTT_VIEW_SLOT < MAX_VIEW_SLOTS);
    try std.testing.expectEqual(@as(usize, 0), ClusteredGpuCache.clampSlot(0));
    try std.testing.expectEqual(@as(usize, 3), ClusteredGpuCache.clampSlot(3));
    try std.testing.expectEqual(MAX_VIEW_SLOTS - 1, ClusteredGpuCache.clampSlot(MAX_VIEW_SLOTS - 1));
    try std.testing.expectEqual(MAX_VIEW_SLOTS - 1, ClusteredGpuCache.clampSlot(MAX_VIEW_SLOTS));
    try std.testing.expectEqual(MAX_VIEW_SLOTS - 1, ClusteredGpuCache.clampSlot(std.math.maxInt(usize)));

    // Out-of-range reads never trap: they observe the last slot.
    var cache = ClusteredGpuCache{};
    cache.slots[MAX_VIEW_SLOTS - 1].live = true;
    try std.testing.expect(cache.isLive(std.math.maxInt(usize)));
    const views = cache.bindingViewsForSlot(std.math.maxInt(usize));
    // Live but viewless headless: real (zero-id) views, never the dummy
    // path confusion — liveness and view handles stay consistent.
    try std.testing.expectEqual(cache.slots[MAX_VIEW_SLOTS - 1].light_view.id, views.lights.id);
}

test "rebuildCpuForSlot clears only its own slot's liveness" {
    const alloc = std.testing.allocator;
    var cache = ClusteredGpuCache{};
    defer {
        cache.cpu_lights.deinit(alloc);
        cache.cpu_headers.deinit(alloc);
        cache.cpu_indices.deinit(alloc);
    }
    const pos = [_][4]f32{.{ 0.75, 0.75, 0.0, 5.0 }};
    const col = [_][4]f32{.{ 1, 0, 0, 2.0 }};
    const full = ViewRect{ .x = 0, .y = 0, .w = 128, .h = 128 };
    // Simulate two uploaded views (headless: liveness flags only, no
    // sg.*): rebuilding the secondary must not clear the primary.
    cache.slots[0].live = true;
    cache.slots[1].live = true;
    try cache.rebuildCpuForSlot(alloc, &pos, &col, 1, Mat4.identity, 128, 128, full, 1);
    try std.testing.expect(cache.isLive(0));
    try std.testing.expect(!cache.isLive(1));
    // The legacy single-arg rebuild targets the primary slot only.
    cache.slots[1].live = true;
    try cache.rebuildCpu(alloc, &pos, &col, 1, Mat4.identity, 128, 128, full);
    try std.testing.expect(!cache.isLive(0));
    try std.testing.expect(cache.isLive(1));
}

test "bindingViewsForSlot falls back to the shared dummy per slot" {
    var cache = ClusteredGpuCache{};
    // Fresh cache: every slot binds the (zero, headless) dummy.
    var s: usize = 0;
    while (s < MAX_VIEW_SLOTS) : (s += 1) {
        const v = cache.bindingViewsForSlot(s);
        try std.testing.expectEqual(@as(u32, 0), v.lights.id);
        try std.testing.expectEqual(@as(u32, 0), v.tiles.id);
        try std.testing.expectEqual(@as(u32, 0), v.indices.id);
    }
    // A live slot with real views binds them; a dead slot still binds
    // the dummy — a failing secondary falls back to legacy while the
    // primary keeps clustered.
    cache.slots[0].live = true;
    cache.slots[0].light_view = .{ .id = 21 };
    cache.slots[0].header_view = .{ .id = 22 };
    cache.slots[0].index_view = .{ .id = 23 };
    const primary = cache.bindingViews();
    try std.testing.expectEqual(@as(u32, 21), primary.lights.id);
    try std.testing.expectEqual(@as(u32, 22), primary.tiles.id);
    try std.testing.expectEqual(@as(u32, 23), primary.indices.id);
    const secondary = cache.bindingViewsForSlot(1);
    try std.testing.expectEqual(cache.dummy_view.id, secondary.lights.id);
}
