//! Tests for `serialization.zig` (moved from `serialization.zig` inline blocks).
const std = @import("std");
const jobs = @import("jobs.zig");
const math = @import("math");
const Vec3 = math.Vec3;
const format_mod = @import("serialization/format.zig");
const props_mod = @import("serialization/props.zig");
const writer_mod = @import("serialization/writer.zig");
const reader_mod = @import("serialization/reader.zig");
const CameraModule = @import("camera.zig");
const MaterialModule = @import("material.zig");
const Camera = CameraModule.Camera;
const TargetCamera = CameraModule.TargetCamera;
const FlyCamera = CameraModule.FlyCamera;
const PostProcessOptions = @import("postprocess.zig").PostProcessOptions;
const Writer = format_mod.Writer;
const writePostProcess = format_mod.writePostProcess;
const testScene = @import("testing.zig").testScene;
const testMesh = @import("testing.zig").testMesh;
const ser = @import("serialization.zig");
const MAGIC = ser.MAGIC;
const VERSION = ser.VERSION;
const MeshEntry = ser.MeshEntry;
const PointEntry = ser.PointEntry;
const SpotEntry = ser.SpotEntry;
const SceneState = ser.SceneState;
const capture = ser.capture;
const serializeAlloc = ser.serializeAlloc;
const saveFileAsync = ser.saveFileAsync;
const restore = ser.restore;
const deserializeAlloc = ser.deserializeAlloc;
const loadFileAsync = ser.loadFileAsync;

fn dupeStr(allocator: std.mem.Allocator, s: []const u8) ![]u8 {
    return allocator.dupe(u8, s);
}

fn makeFullState(allocator: std.mem.Allocator) !SceneState {
    var s = SceneState{};
    errdefer s.deinit(allocator);

    // Each list is built in its own labelled block so the construction-time
    // errdefers disarm on `break` (normal exit) and only fire on error.
    s.meshes = blk: {
        const meshes = try allocator.alloc(MeshEntry, 2);
        errdefer allocator.free(meshes);
        meshes[0] = .{
            .id = 101,
            .name = try dupeStr(allocator, "box"),
            .parent_name = try dupeStr(allocator, "sphere"),
            .position = .{ 1.0, 2.0, 3.0 },
            .rotation = .{ 10.0, 20.0, 30.0 },
            .scaling = .{ 1.0, 1.0, 1.0 },
            .is_visible = true,
            .cast_shadows = false,
            .receive_shadows = true,
            .material = .{ .standard = .{ .diffuse = .{ 0.5, 0.25, 0.125 }, .alpha = 0.75, .alpha_mode = 1 } },
        };
        errdefer meshes[0].deinit(allocator);
        meshes[1] = .{
            .id = 102,
            .name = try dupeStr(allocator, "sphere"),
            .parent_name = "",
            .position = .{ -4.0, 0.5, 8.0 },
            .rotation = .{ 0.0, 90.0, 0.0 },
            .scaling = .{ 2.0, 2.0, 2.0 },
            .is_visible = false,
            .cast_shadows = true,
            .receive_shadows = false,
            .material = .{ .pbr = .{
                .albedo = .{ 0.1, 0.2, 0.3 },
                .metallic = 0.9,
                .roughness = 0.15,
                .emissive = .{ 1.0, 0.5, 0.0 },
                .alpha = 1.0,
                .alpha_mode = 0,
            } },
        };
        break :blk meshes;
    };

    s.hemi = .{
        .name = try dupeStr(allocator, "hemi"),
        .direction = .{ 0.5, 1.0, 0.3 },
        .diffuse = .{ 1.0, 1.0, 1.0 },
        .ground = .{ 0.2, 0.25, 0.3 },
        .intensity = 0.8,
    };
    s.directional = .{
        .name = try dupeStr(allocator, "sun"),
        .direction = .{ 0.0, -1.0, 0.0 },
        .diffuse = .{ 1.0, 0.9, 0.8 },
        .intensity = 2.5,
    };

    s.point_lights = blk: {
        const points = try allocator.alloc(PointEntry, 2);
        errdefer allocator.free(points);
        points[0] = .{
            .name = try dupeStr(allocator, "lamp"),
            .position = .{ 3.0, 3.0, 3.0 },
            .diffuse = .{ 1.0, 0.0, 0.0 },
            .intensity = 1.5,
            .range = 12.0,
        };
        errdefer points[0].deinit(allocator);
        points[1] = .{
            .name = try dupeStr(allocator, "fill"),
            .position = .{ -3.0, 1.0, 0.0 },
            .diffuse = .{ 0.0, 1.0, 0.0 },
            .intensity = 0.5,
            .range = 5.0,
        };
        break :blk points;
    };

    s.spot_lights = blk: {
        const spots = try allocator.alloc(SpotEntry, 1);
        errdefer allocator.free(spots);
        spots[0] = .{
            .name = try dupeStr(allocator, "head"),
            .position = .{ 0.0, 5.0, 0.0 },
            .direction = .{ 0.0, -1.0, 0.0 },
            .diffuse = .{ 0.0, 0.0, 1.0 },
            .intensity = 3.0,
            .range = 20.0,
            .inner_deg = 10.0,
            .outer_deg = 25.0,
        };
        break :blk spots;
    };

    s.camera = .{ .arc_rotate = .{
        .name = try dupeStr(allocator, "orbit"),
        .alpha = 0.7,
        .beta = 1.1,
        .radius = 9.0,
        .target = .{ 1.0, 2.0, 3.0 },
        .fov_deg = 55.0,
        .near = 0.5,
        .far = 500.0,
    } };

    s.render = .{
        .skybox_enabled = true,
        .skybox_exposure = 1.25,
        .shadows_enabled = false,
        .shadow_softness = 2.5,
        .ibl_intensity = 0.6,
    };
    s.postprocess.enabled = true;
    s.postprocess.exposure = 1.1;
    s.postprocess.tonemapping = .reinhard;
    s.postprocess.bloom_threshold = 0.9;
    s.postprocess.fog_color = .{ 0.1, 0.2, 0.3 };
    s.postprocess.temperature = 0.5;
    s.postprocess.tint = -0.25;

    try s.setGameProperty(allocator, "quest_stage", "3");
    try s.setGameProperty(allocator, "difficulty", "hard");

    return s;
}

fn expectStatesEqual(a: *const SceneState, b: *const SceneState) !void {
    try std.testing.expectEqual(a.meshes.len, b.meshes.len);
    for (a.meshes, b.meshes) |*am, *bm| {
        try std.testing.expectEqual(am.id, bm.id);
        try std.testing.expectEqualStrings(am.name, bm.name);
        try std.testing.expectEqualStrings(am.parent_name, bm.parent_name);
        try std.testing.expectEqual(am.position, bm.position);
        try std.testing.expectEqual(am.rotation, bm.rotation);
        try std.testing.expectEqual(am.scaling, bm.scaling);
        try std.testing.expectEqual(am.is_visible, bm.is_visible);
        try std.testing.expectEqual(am.cast_shadows, bm.cast_shadows);
        try std.testing.expectEqual(am.receive_shadows, bm.receive_shadows);
        try std.testing.expectEqual(std.meta.activeTag(am.material), std.meta.activeTag(bm.material));
        switch (am.material) {
            .standard => |*s| {
                try std.testing.expectEqual(s.diffuse, bm.material.standard.diffuse);
                try std.testing.expectEqual(s.alpha, bm.material.standard.alpha);
                try std.testing.expectEqual(s.alpha_mode, bm.material.standard.alpha_mode);
                try std.testing.expectEqual(s.alpha_cutoff, bm.material.standard.alpha_cutoff);
                try std.testing.expectEqual(s.double_sided, bm.material.standard.double_sided);
            },
            .pbr => |*p| {
                try std.testing.expectEqual(p.albedo, bm.material.pbr.albedo);
                try std.testing.expectEqual(p.metallic, bm.material.pbr.metallic);
                try std.testing.expectEqual(p.roughness, bm.material.pbr.roughness);
                try std.testing.expectEqual(p.emissive, bm.material.pbr.emissive);
                try std.testing.expectEqual(p.alpha, bm.material.pbr.alpha);
                try std.testing.expectEqual(p.alpha_mode, bm.material.pbr.alpha_mode);
                try std.testing.expectEqual(p.alpha_cutoff, bm.material.pbr.alpha_cutoff);
                try std.testing.expectEqual(p.double_sided, bm.material.pbr.double_sided);
            },
        }
    }
    try std.testing.expectEqualStrings(a.hemi.name, b.hemi.name);
    try std.testing.expectEqual(a.hemi.direction, b.hemi.direction);
    try std.testing.expectEqual(a.hemi.diffuse, b.hemi.diffuse);
    try std.testing.expectEqual(a.hemi.ground, b.hemi.ground);
    try std.testing.expectEqual(a.hemi.intensity, b.hemi.intensity);
    try std.testing.expectEqual(a.directional != null, b.directional != null);
    if (a.directional) |*ad| {
        const bd = b.directional.?;
        try std.testing.expectEqualStrings(ad.name, bd.name);
        try std.testing.expectEqual(ad.direction, bd.direction);
        try std.testing.expectEqual(ad.diffuse, bd.diffuse);
        try std.testing.expectEqual(ad.intensity, bd.intensity);
    }
    try std.testing.expectEqual(a.point_lights.len, b.point_lights.len);
    for (a.point_lights, b.point_lights) |*ap, *bp| {
        try std.testing.expectEqualStrings(ap.name, bp.name);
        try std.testing.expectEqual(ap.position, bp.position);
        try std.testing.expectEqual(ap.diffuse, bp.diffuse);
        try std.testing.expectEqual(ap.intensity, bp.intensity);
        try std.testing.expectEqual(ap.range, bp.range);
    }
    try std.testing.expectEqual(a.spot_lights.len, b.spot_lights.len);
    for (a.spot_lights, b.spot_lights) |*as, *bs| {
        try std.testing.expectEqualStrings(as.name, bs.name);
        try std.testing.expectEqual(as.position, bs.position);
        try std.testing.expectEqual(as.direction, bs.direction);
        try std.testing.expectEqual(as.diffuse, bs.diffuse);
        try std.testing.expectEqual(as.intensity, bs.intensity);
        try std.testing.expectEqual(as.range, bs.range);
        try std.testing.expectEqual(as.inner_deg, bs.inner_deg);
        try std.testing.expectEqual(as.outer_deg, bs.outer_deg);
    }
    try std.testing.expectEqual(std.meta.activeTag(a.camera), std.meta.activeTag(b.camera));
    switch (a.camera) {
        .none => {},
        .arc_rotate => |*c| {
            const o = b.camera.arc_rotate;
            try std.testing.expectEqualStrings(c.name, o.name);
            try std.testing.expectEqual(c.alpha, o.alpha);
            try std.testing.expectEqual(c.beta, o.beta);
            try std.testing.expectEqual(c.radius, o.radius);
            try std.testing.expectEqual(c.target, o.target);
            try std.testing.expectEqual(c.fov_deg, o.fov_deg);
            try std.testing.expectEqual(c.near, o.near);
            try std.testing.expectEqual(c.far, o.far);
        },
        .free => |*c| {
            const o = b.camera.free;
            try std.testing.expectEqualStrings(c.name, o.name);
            try std.testing.expectEqual(c.position, o.position);
            try std.testing.expectEqual(c.rotation, o.rotation);
            try std.testing.expectEqual(c.fov_deg, o.fov_deg);
            try std.testing.expectEqual(c.near, o.near);
            try std.testing.expectEqual(c.far, o.far);
            try std.testing.expectEqual(c.speed, o.speed);
            try std.testing.expectEqual(c.angular_sensitivity, o.angular_sensitivity);
        },
        .follow => |*c| {
            const o = b.camera.follow;
            try std.testing.expectEqualStrings(c.name, o.name);
            try std.testing.expectEqual(c.position, o.position);
            try std.testing.expectEqual(c.target_position, o.target_position);
            try std.testing.expectEqual(c.radius, o.radius);
            try std.testing.expectEqual(c.height_offset, o.height_offset);
            try std.testing.expectEqual(c.rotation_offset_deg, o.rotation_offset_deg);
            try std.testing.expectEqual(c.fov_deg, o.fov_deg);
            try std.testing.expectEqual(c.near, o.near);
            try std.testing.expectEqual(c.far, o.far);
            try std.testing.expectEqual(c.lerp_speed, o.lerp_speed);
        },
        .target => |*c| {
            const o = b.camera.target;
            try std.testing.expectEqualStrings(c.name, o.name);
            try std.testing.expectEqual(c.position, o.position);
            try std.testing.expectEqual(c.target, o.target);
            try std.testing.expectEqual(c.up, o.up);
            try std.testing.expectEqual(c.fov_deg, o.fov_deg);
            try std.testing.expectEqual(c.near, o.near);
            try std.testing.expectEqual(c.far, o.far);
            try std.testing.expectEqual(c.smoothing, o.smoothing);
        },
        .fly => |*c| {
            const o = b.camera.fly;
            try std.testing.expectEqualStrings(c.name, o.name);
            try std.testing.expectEqual(c.position, o.position);
            try std.testing.expectEqual(c.rotation, o.rotation);
            try std.testing.expectEqual(c.fov_deg, o.fov_deg);
            try std.testing.expectEqual(c.near, o.near);
            try std.testing.expectEqual(c.far, o.far);
            try std.testing.expectEqual(c.speed, o.speed);
            try std.testing.expectEqual(c.boost_multiplier, o.boost_multiplier);
            try std.testing.expectEqual(c.angular_sensitivity, o.angular_sensitivity);
            try std.testing.expectEqual(c.roll_speed_deg, o.roll_speed_deg);
        },
    }
    try std.testing.expectEqual(a.render.skybox_enabled, b.render.skybox_enabled);
    try std.testing.expectEqual(a.render.skybox_exposure, b.render.skybox_exposure);
    try std.testing.expectEqual(a.render.shadows_enabled, b.render.shadows_enabled);
    try std.testing.expectEqual(a.render.shadow_softness, b.render.shadow_softness);
    try std.testing.expectEqual(a.render.ibl_intensity, b.render.ibl_intensity);
    try std.testing.expectEqual(a.postprocess, b.postprocess);
    try std.testing.expectEqual(a.game_properties.len, b.game_properties.len);
    for (a.game_properties, b.game_properties) |ap, bp| {
        try std.testing.expectEqualStrings(ap.key, bp.key);
        try std.testing.expectEqualStrings(ap.value, bp.value);
    }
}

fn roundTripState(alloc: std.mem.Allocator, original: *const SceneState) !SceneState {
    const bytes = try serializeAlloc(alloc, original);
    defer alloc.free(bytes);
    return deserializeAlloc(alloc, bytes);
}

test "serialization round-trip full state" {
    const alloc = std.testing.allocator;
    var original = try makeFullState(alloc);
    defer original.deinit(alloc);

    const bytes = try serializeAlloc(alloc, &original);
    defer alloc.free(bytes);

    var parsed = try deserializeAlloc(alloc, bytes);
    defer parsed.deinit(alloc);

    try expectStatesEqual(&original, &parsed);

    // Deterministic encoding: re-serializing must give identical bytes.
    const bytes2 = try serializeAlloc(alloc, &parsed);
    defer alloc.free(bytes2);
    try std.testing.expectEqualSlices(u8, bytes, bytes2);
}

test "serialization empty state round-trips" {
    const alloc = std.testing.allocator;
    var original = SceneState{};
    defer original.deinit(alloc);

    const bytes = try serializeAlloc(alloc, &original);
    defer alloc.free(bytes);

    var parsed = try deserializeAlloc(alloc, bytes);
    defer parsed.deinit(alloc);
    try expectStatesEqual(&original, &parsed);
}

test "serialization rejects bad magic" {
    const alloc = std.testing.allocator;
    var original = try makeFullState(alloc);
    defer original.deinit(alloc);
    const bytes = try serializeAlloc(alloc, &original);
    defer alloc.free(bytes);

    var bad = try alloc.dupe(u8, bytes);
    defer alloc.free(bad);
    bad[0] = 'X';
    try std.testing.expectError(error.BadMagic, deserializeAlloc(alloc, bad));
}

test "serialization rejects truncated input" {
    const alloc = std.testing.allocator;
    var original = try makeFullState(alloc);
    defer original.deinit(alloc);
    const bytes = try serializeAlloc(alloc, &original);
    defer alloc.free(bytes);

    // Empty, shorter-than-magic, magic-only, version-cut, and last-byte-cut.
    for ([_]usize{ 0, 3, 4, 7, bytes.len - 1 }) |n| {
        try std.testing.expectError(error.Truncated, deserializeAlloc(alloc, bytes[0..n]));
    }
}

test "serialization rejects unsupported version" {
    const alloc = std.testing.allocator;
    var original = try makeFullState(alloc);
    defer original.deinit(alloc);
    const bytes = try serializeAlloc(alloc, &original);
    defer alloc.free(bytes);

    // Current VERSION must parse; anything else (incl. v1) is rejected.
    var ok = try deserializeAlloc(alloc, bytes);
    ok.deinit(alloc);

    var bad = try alloc.dupe(u8, bytes);
    defer alloc.free(bad);
    std.mem.writeInt(u32, bad[4..8], 1, .little);
    try std.testing.expectError(error.UnsupportedVersion, deserializeAlloc(alloc, bad));
    std.mem.writeInt(u32, bad[4..8], 0, .little);
    try std.testing.expectError(error.UnsupportedVersion, deserializeAlloc(alloc, bad));
    std.mem.writeInt(u32, bad[4..8], VERSION + 1, .little);
    try std.testing.expectError(error.UnsupportedVersion, deserializeAlloc(alloc, bad));
}

test "serialization rejects huge counts and strings" {
    const alloc = std.testing.allocator;
    var buf: std.ArrayListUnmanaged(u8) = .empty;
    defer buf.deinit(alloc);
    try buf.appendSlice(alloc, MAGIC[0..]);
    var vb: [4]u8 = undefined;
    std.mem.writeInt(u32, &vb, VERSION, .little);
    try buf.appendSlice(alloc, &vb);
    // mesh_count = 0xFFFFFFFF: must be TooLarge, not OOM/hang.
    std.mem.writeInt(u32, &vb, 0xFFFFFFFF, .little);
    try buf.appendSlice(alloc, &vb);
    try std.testing.expectError(error.TooLarge, deserializeAlloc(alloc, buf.items));

    // mesh_count = 1 with a gigantic name length: TooLarge as well.
    buf.clearRetainingCapacity();
    try buf.appendSlice(alloc, MAGIC[0..]);
    std.mem.writeInt(u32, &vb, VERSION, .little);
    try buf.appendSlice(alloc, &vb);
    std.mem.writeInt(u32, &vb, 1, .little);
    try buf.appendSlice(alloc, &vb);
    var id_b: [8]u8 = [_]u8{0} ** 8;
    try buf.appendSlice(alloc, &id_b);
    std.mem.writeInt(u32, &vb, 0x0FFFFFFF, .little);
    try buf.appendSlice(alloc, &vb);
    try std.testing.expectError(error.TooLarge, deserializeAlloc(alloc, buf.items));

    // Declared length fits the caps but exceeds the buffer: Truncated.
    buf.clearRetainingCapacity();
    try buf.appendSlice(alloc, MAGIC[0..]);
    std.mem.writeInt(u32, &vb, VERSION, .little);
    try buf.appendSlice(alloc, &vb);
    std.mem.writeInt(u32, &vb, 1, .little);
    try buf.appendSlice(alloc, &vb);
    try buf.appendSlice(alloc, &id_b);
    std.mem.writeInt(u32, &vb, 64, .little);
    try buf.appendSlice(alloc, &vb);
    try std.testing.expectError(error.Truncated, deserializeAlloc(alloc, buf.items));
}

test "capture maps null material to pbr default" {
    const alloc = std.testing.allocator;
    var scene = testScene(alloc);

    var mesh = testMesh("plain");
    mesh.position = Vec3.new(1.0, 2.0, 3.0);
    // material stays null.
    try scene.meshes.append(alloc, &mesh);
    defer scene.meshes.deinit(alloc);

    var state = try capture(alloc, &scene);
    defer state.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 1), state.meshes.len);
    try std.testing.expectEqualStrings("plain", state.meshes[0].name);
    try std.testing.expectEqual([3]f32{ 1.0, 2.0, 3.0 }, state.meshes[0].position);
    try std.testing.expect(state.meshes[0].material == .pbr);
    try std.testing.expectEqual([3]f32{ 1.0, 1.0, 1.0 }, state.meshes[0].material.pbr.albedo);
    try std.testing.expectEqual(@as(f32, 1.0), state.meshes[0].material.pbr.alpha);
    try std.testing.expectEqual(@as(f32, 0.0), state.meshes[0].material.pbr.metallic);
    try std.testing.expectEqual(@as(f32, 0.5), state.meshes[0].material.pbr.roughness);
    try std.testing.expect(state.camera == .none);
    try std.testing.expect(state.directional == null);
}

test "restore applies by name and ignores missing" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var scene = testScene(alloc);

    var pbr_mat = MaterialModule.PBRMaterial.init("shared");
    var mesh = testMesh("box");
    mesh.material = .{ .pbr = &pbr_mat };
    try scene.meshes.append(alloc, &mesh);

    var state = SceneState{};
    const entries = try alloc.alloc(MeshEntry, 2);
    entries[0] = .{
        .name = try alloc.dupe(u8, "box"),
        .position = .{ 7.0, 8.0, 9.0 },
        .rotation = .{ 1.0, 2.0, 3.0 },
        .scaling = .{ 3.0, 3.0, 3.0 },
        .is_visible = false,
        .cast_shadows = false,
        .receive_shadows = false,
        .material = .{ .pbr = .{ .albedo = .{ 0.9, 0.1, 0.1 }, .alpha = 0.5, .alpha_mode = 1 } },
    };
    entries[1] = .{
        .name = try alloc.dupe(u8, "ghost-missing"),
        .position = .{ 99.0, 99.0, 99.0 },
        .material = .{ .pbr = .{} },
    };
    state.meshes = entries;
    state.hemi = .{
        .name = try alloc.dupe(u8, "hemi2"),
        .direction = .{ 0.0, 1.0, 0.0 },
        .diffuse = .{ 0.5, 0.5, 0.5 },
        .ground = .{ 0.1, 0.1, 0.1 },
        .intensity = 0.3,
    };
    const pl = try alloc.alloc(PointEntry, 1);
    pl[0] = .{
        .name = try alloc.dupe(u8, "lamp"),
        .position = .{ 1.0, 1.0, 1.0 },
        .diffuse = .{ 1.0, 1.0, 1.0 },
        .intensity = 2.0,
        .range = 11.0,
    };
    state.point_lights = pl;
    state.camera = .{ .free = .{
        .name = try alloc.dupe(u8, "fly"),
        .position = .{ 5.0, 5.0, 5.0 },
        .rotation = .{ 10.0, 20.0, 0.0 },
        .fov_deg = 70.0,
        .near = 0.2,
        .far = 200.0,
        .speed = 9.0,
        .angular_sensitivity = 0.5,
    } };
    state.render.shadows_enabled = false;

    restore(&scene, &state);

    // Matched mesh updated in place (same kind mutates the shared material).
    try std.testing.expectEqual(Vec3.new(7.0, 8.0, 9.0), mesh.position);
    try std.testing.expectEqual(Vec3.new(3.0, 3.0, 3.0), mesh.scaling);
    try std.testing.expect(!mesh.is_visible);
    try std.testing.expectEqual(@as(f32, 0.5), pbr_mat.alpha);
    try std.testing.expect(pbr_mat.alpha_mode == .blend);
    // Missing mesh ignored: no crash, light recreated, camera applied.
    try std.testing.expectEqual(@as(usize, 1), scene.lights.point_lights.items.len);
    try std.testing.expectEqualStrings("lamp", scene.lights.point_lights.items[0].name);
    try std.testing.expectEqual(@as(f32, 2.0), scene.lights.point_lights.items[0].intensity);
    try std.testing.expectEqual(@as(f32, 0.3), scene.lights.hemi.intensity);
    try std.testing.expect(scene.active_camera != null);
    try std.testing.expect(scene.active_camera.? == .free);
    try std.testing.expectEqual(@as(f32, 70.0), scene.active_camera.?.free.fov_deg);
    try std.testing.expect(!scene.shadows.enabled);
    // Camera union import is exercised (keeps the Camera symbol referenced).
    const _cam: ?Camera = scene.active_camera;
    try std.testing.expect(_cam != null);
}

test "serialization target camera round-trips exactly" {
    const alloc = std.testing.allocator;
    var original = SceneState{};
    original.camera = .{ .target = .{
        .name = try dupeStr(alloc, "watcher"),
        .position = .{ 1.0, 2.0, 5.0 },
        .target = .{ 0.0, 1.0, 0.0 },
        .up = .{ 0.0, 1.0, 0.0 },
        .fov_deg = 50.0,
        .near = 0.5,
        .far = 200.0,
        .smoothing = 4.0,
    } };
    defer original.deinit(alloc);

    var parsed = try roundTripState(alloc, &original);
    defer parsed.deinit(alloc);
    try expectStatesEqual(&original, &parsed);
}

test "serialization fly camera round-trips exactly" {
    const alloc = std.testing.allocator;
    var original = SceneState{};
    original.camera = .{ .fly = .{
        .name = try dupeStr(alloc, "pilot"),
        .position = .{ -3.0, 1.5, 7.0 },
        .rotation = .{ 10.0, 45.0, 30.0 },
        .fov_deg = 70.0,
        .near = 0.2,
        .far = 300.0,
        .speed = 9.0,
        .boost_multiplier = 3.0,
        .angular_sensitivity = 0.4,
        .roll_speed_deg = 120.0,
    } };
    defer original.deinit(alloc);

    var parsed = try roundTripState(alloc, &original);
    defer parsed.deinit(alloc);
    try expectStatesEqual(&original, &parsed);
}

test "serialization cutout standard material round-trips cutoff and double-sided" {
    const alloc = std.testing.allocator;
    var original = SceneState{};
    defer original.deinit(alloc);
    const meshes = try alloc.alloc(MeshEntry, 1);
    meshes[0] = .{
        .name = try dupeStr(alloc, "fence"),
        .material = .{ .standard = .{
            .diffuse = .{ 0.8, 0.7, 0.6 },
            .alpha = 0.9,
            .alpha_mode = 2,
            .alpha_cutoff = 0.2,
            .double_sided = true,
        } },
    };
    original.meshes = meshes;

    var parsed = try roundTripState(alloc, &original);
    defer parsed.deinit(alloc);
    try expectStatesEqual(&original, &parsed);
    try std.testing.expectEqual(@as(u8, 2), parsed.meshes[0].material.standard.alpha_mode);
    try std.testing.expectEqual(@as(f32, 0.2), parsed.meshes[0].material.standard.alpha_cutoff);
    try std.testing.expect(parsed.meshes[0].material.standard.double_sided);
}

test "serialization cutout pbr material round-trips cutoff and double-sided" {
    const alloc = std.testing.allocator;
    var original = SceneState{};
    defer original.deinit(alloc);
    const meshes = try alloc.alloc(MeshEntry, 1);
    meshes[0] = .{
        .name = try dupeStr(alloc, "grate"),
        .material = .{ .pbr = .{
            .albedo = .{ 0.3, 0.4, 0.5 },
            .metallic = 0.1,
            .roughness = 0.8,
            .emissive = .{ 0.0, 0.0, 0.0 },
            .alpha = 1.0,
            .alpha_mode = 2,
            .alpha_cutoff = 0.2,
            .double_sided = true,
        } },
    };
    original.meshes = meshes;

    var parsed = try roundTripState(alloc, &original);
    defer parsed.deinit(alloc);
    try expectStatesEqual(&original, &parsed);
    try std.testing.expectEqual(@as(u8, 2), parsed.meshes[0].material.pbr.alpha_mode);
    try std.testing.expectEqual(@as(f32, 0.2), parsed.meshes[0].material.pbr.alpha_cutoff);
    try std.testing.expect(parsed.meshes[0].material.pbr.double_sided);
}

test "serialization opaque and blend modes round-trip" {
    const alloc = std.testing.allocator;
    var original = SceneState{};
    defer original.deinit(alloc);
    const meshes = try alloc.alloc(MeshEntry, 2);
    meshes[0] = .{
        .name = try dupeStr(alloc, "solid"),
        .material = .{ .standard = .{ .diffuse = .{ 1.0, 0.0, 0.0 }, .alpha = 1.0, .alpha_mode = 0 } },
    };
    meshes[1] = .{
        .name = try dupeStr(alloc, "glass"),
        .material = .{ .pbr = .{
            .albedo = .{ 0.9, 0.9, 1.0 },
            .metallic = 0.0,
            .roughness = 0.1,
            .emissive = .{ 0.0, 0.0, 0.0 },
            .alpha = 0.35,
            .alpha_mode = 1,
        } },
    };
    original.meshes = meshes;

    var parsed = try roundTripState(alloc, &original);
    defer parsed.deinit(alloc);
    try expectStatesEqual(&original, &parsed);
    try std.testing.expectEqual(@as(u8, 0), parsed.meshes[0].material.standard.alpha_mode);
    try std.testing.expectEqual(@as(u8, 1), parsed.meshes[1].material.pbr.alpha_mode);
}

test "capture and restore target and fly cameras" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var scene = testScene(alloc);
    scene.active_camera = .{ .target = TargetCamera.init("watcher", .{
        .position = Vec3.new(0.0, 0.0, 5.0),
        .target = Vec3.zero,
        .fov_deg = 50.0,
        .near = 0.5,
        .far = 200.0,
        .smoothing = 4.0,
    }) };

    var state = try capture(alloc, &scene);
    try std.testing.expect(state.camera == .target);
    try std.testing.expectEqual(@as(f32, 4.0), state.camera.target.smoothing);

    restore(&scene, &state);
    try std.testing.expect(scene.active_camera != null);
    try std.testing.expect(scene.active_camera.? == .target);
    try std.testing.expectEqual(@as(f32, 50.0), scene.active_camera.?.target.fov_deg);
    try std.testing.expectEqual(@as(f32, 4.0), scene.active_camera.?.target.smoothing);

    scene.active_camera = .{ .fly = FlyCamera.init("pilot", .{
        .position = Vec3.new(1.0, 2.0, 3.0),
        .rotation = Vec3.new(10.0, 20.0, 30.0),
        .speed = 9.0,
        .boost_multiplier = 3.0,
        .angular_sensitivity = 0.4,
        .roll_speed_deg = 120.0,
    }) };
    var state2 = try capture(alloc, &scene);
    try std.testing.expect(state2.camera == .fly);
    restore(&scene, &state2);
    try std.testing.expect(scene.active_camera.? == .fly);
    try std.testing.expectEqual(@as(f32, 3.0), scene.active_camera.?.fly.boost_multiplier);
    try std.testing.expectEqual(@as(f32, 120.0), scene.active_camera.?.fly.roll_speed_deg);
}

test "capture and restore cutout material fields" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var scene = testScene(alloc);

    var pbr_mat = MaterialModule.PBRMaterial.init("cut");
    pbr_mat.alpha_mode = .cutout;
    pbr_mat.alpha_cutoff = 0.2;
    pbr_mat.double_sided = true;
    var mesh = testMesh("fence");
    mesh.material = .{ .pbr = &pbr_mat };
    try scene.meshes.append(alloc, &mesh);

    var state = try capture(alloc, &scene);
    try std.testing.expectEqual(@as(u8, 2), state.meshes[0].material.pbr.alpha_mode);
    try std.testing.expectEqual(@as(f32, 0.2), state.meshes[0].material.pbr.alpha_cutoff);
    try std.testing.expect(state.meshes[0].material.pbr.double_sided);

    // Mutate live, restore must bring the snapshot values back.
    pbr_mat.alpha_cutoff = 0.9;
    pbr_mat.double_sided = false;
    pbr_mat.alpha_mode = .@"opaque";
    restore(&scene, &state);
    try std.testing.expect(pbr_mat.alpha_mode == .cutout);
    try std.testing.expectEqual(@as(f32, 0.2), pbr_mat.alpha_cutoff);
    try std.testing.expect(pbr_mat.double_sided);
}

test "restore converts legacy standard material entry to pbr material" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var scene = testScene(alloc);

    var mesh = testMesh("legacy_mesh");
    try scene.meshes.append(alloc, &mesh);

    var state = SceneState{};
    const entries = try alloc.alloc(MeshEntry, 1);
    entries[0] = .{
        .name = try alloc.dupe(u8, "legacy_mesh"),
        .material = .{ .standard = .{
            .diffuse = .{ 0.8, 0.4, 0.2 },
            .alpha = 0.8,
            .alpha_mode = 2,
            .alpha_cutoff = 0.3,
            .double_sided = true,
        } },
    };
    state.meshes = entries;

    restore(&scene, &state);

    try std.testing.expect(mesh.material != null);
    try std.testing.expect(mesh.material.? == .pbr);
    const pbr = mesh.material.?.pbr;
    try std.testing.expectEqual(@as(f32, 0.8), pbr.albedo_color.r);
    try std.testing.expectEqual(@as(f32, 0.4), pbr.albedo_color.g);
    try std.testing.expectEqual(@as(f32, 0.2), pbr.albedo_color.b);
    try std.testing.expectEqual(@as(f32, 0.0), pbr.metallic);
    try std.testing.expectEqual(@as(f32, 0.5), pbr.roughness);
    try std.testing.expectEqual(@as(f32, 0.8), pbr.alpha);
    try std.testing.expect(pbr.alpha_mode == .cutout);
    try std.testing.expectEqual(@as(f32, 0.3), pbr.alpha_cutoff);
    try std.testing.expect(pbr.double_sided);
}

test "saveFileAsync and loadFileAsync round-trip with TaskRunner" {
    const alloc = std.testing.allocator;
    const runner = try jobs.TaskRunner.init(alloc, 2);
    defer runner.deinit();

    const original = try makeFullState(alloc);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const test_path = try std.fmt.allocPrint(alloc, ".zig-cache/tmp/{s}/test_async_scene.bin", .{tmp.sub_path});
    defer alloc.free(test_path);

    const save_task = try saveFileAsync(alloc, runner, original, test_path);
    defer save_task.deinit();

    // Poll until complete
    var waited: usize = 0;
    while (!save_task.isDone() and waited < 10_000_000) : (waited += 1) {
        std.atomic.spinLoopHint();
    }
    try std.testing.expect(save_task.isDone());
    try std.testing.expect(save_task.isSuccess());
    try std.testing.expect(save_task.bytes_written > 0);

    const load_task = try loadFileAsync(alloc, runner, test_path);
    defer load_task.deinit();

    waited = 0;
    while (!load_task.isDone() and waited < 10_000_000) : (waited += 1) {
        std.atomic.spinLoopHint();
    }
    try std.testing.expect(load_task.isDone());
    try std.testing.expect(load_task.isSuccess());
    try std.testing.expect(load_task.result != null);
    try std.testing.expectEqual(@as(usize, 2), load_task.result.?.meshes.len);
}

test "loadFileAsync reports failure for non-existent file" {
    const alloc = std.testing.allocator;
    const runner = try jobs.TaskRunner.init(alloc, 1);
    defer runner.deinit();

    const load_task = try loadFileAsync(alloc, runner, "non_existent_file_12345.bin");
    defer load_task.deinit();

    var waited: usize = 0;
    while (!load_task.isDone() and waited < 10_000_000) : (waited += 1) {
        std.atomic.spinLoopHint();
    }
    try std.testing.expect(load_task.isDone());
    try std.testing.expect(!load_task.isSuccess());
    try std.testing.expect(load_task.err_name != null);
}

test "serialization rejects version 2" {
    const alloc = std.testing.allocator;
    var w = Writer{ .alloc = alloc };
    defer w.buf.deinit(alloc);

    try w.bytes(MAGIC[0..]);
    try w.u32le(2); // version 2: rejected, not parsed as v3

    // 1 mesh in v2 layout (no id/parent_name prefix)
    try w.u32le(1);
    try w.str("v2_cube");
    try w.vec3(.{ 1.0, 2.0, 3.0 });
    try w.vec3(.{ 0.0, 45.0, 0.0 });
    try w.vec3(.{ 1.0, 1.0, 1.0 });
    try w.byte(1 | 2 | 4); // visible, cast, receive
    try w.byte(0); // standard material
    try w.vec3(.{ 0.8, 0.8, 0.8 }); // diffuse
    try w.f32le(1.0); // alpha
    try w.byte(0); // opaque
    try w.f32le(0.5); // cutoff
    try w.bool8(false); // double sided

    // hemi light
    try w.str("hemi");
    try w.vec3(.{ 0.0, 1.0, 0.0 });
    try w.vec3(.{ 1.0, 1.0, 1.0 });
    try w.vec3(.{ 0.2, 0.2, 0.2 });
    try w.f32le(1.0);

    // directional light
    try w.byte(0);

    // point lights
    try w.u32le(0);

    // spot lights
    try w.u32le(0);

    // camera kind: 0 none
    try w.byte(0);

    // render
    try w.bool8(false); // skybox
    try w.f32le(1.0);
    try w.bool8(true); // shadows
    try w.f32le(1.5);
    try w.f32le(1.0); // ibl

    // postprocess
    const pp = PostProcessOptions{};
    try writePostProcess(&w, &pp);

    try std.testing.expectError(error.UnsupportedVersion, deserializeAlloc(alloc, w.buf.items));
}

test "capture and restore mesh hierarchy and entity id" {
    const alloc = std.testing.allocator;
    var scene = testScene(alloc);
    defer {
        for (scene.pbr_materials.items) |m| alloc.destroy(m);
        scene.pbr_materials.deinit(alloc);
    }

    var mat_parent = MaterialModule.PBRMaterial.init("mat_parent");
    var parent = testMesh("root_node");
    parent.id = 1001;
    parent.material = .{ .pbr = &mat_parent };
    parent.position = Vec3.new(10.0, 0.0, 0.0);
    try scene.meshes.append(alloc, &parent);

    var mat_child = MaterialModule.PBRMaterial.init("mat_child");
    var child = testMesh("child_node");
    child.id = 1002;
    child.material = .{ .pbr = &mat_child };
    child.parent = &parent;
    child.position = Vec3.new(2.0, 3.0, 4.0);
    try scene.meshes.append(alloc, &child);
    defer scene.meshes.deinit(alloc);

    // Capture
    var snap = try capture(alloc, &scene);
    defer snap.deinit(alloc);

    try std.testing.expectEqual(@as(u64, 1001), snap.meshes[0].id);
    try std.testing.expectEqualStrings("", snap.meshes[0].parent_name);
    try std.testing.expectEqual(@as(u64, 1002), snap.meshes[1].id);
    try std.testing.expectEqualStrings("root_node", snap.meshes[1].parent_name);

    // Roundtrip via binary
    const bytes = try serializeAlloc(alloc, &snap);
    defer alloc.free(bytes);

    var loaded = try deserializeAlloc(alloc, bytes);
    defer loaded.deinit(alloc);

    // Reset scene hierarchy and IDs to ensure restore re-links them
    child.parent = null;
    parent.id = 0;
    child.id = 0;

    restore(&scene, &loaded);

    try std.testing.expectEqual(@as(u64, 1001), parent.id);
    try std.testing.expectEqual(@as(u64, 1002), child.id);
    try std.testing.expect(child.parent == &parent);
}

test "game properties set and get roundtrip" {
    const alloc = std.testing.allocator;
    var state = SceneState{};
    defer state.deinit(alloc);

    try std.testing.expect(state.getGameProperty("level") == null);

    try state.setGameProperty(alloc, "level", "dungeon_01");
    try state.setGameProperty(alloc, "player_hp", "100");
    try std.testing.expectEqualStrings("dungeon_01", state.getGameProperty("level").?);
    try std.testing.expectEqualStrings("100", state.getGameProperty("player_hp").?);

    // Mutate existing property in place
    try state.setGameProperty(alloc, "player_hp", "85");
    try std.testing.expectEqualStrings("85", state.getGameProperty("player_hp").?);
    try std.testing.expectEqual(@as(usize, 2), state.game_properties.len);

    // Roundtrip serialize/deserialize
    const bytes = try serializeAlloc(alloc, &state);
    defer alloc.free(bytes);

    var loaded = try deserializeAlloc(alloc, bytes);
    defer loaded.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 2), loaded.game_properties.len);
    try std.testing.expectEqualStrings("dungeon_01", loaded.getGameProperty("level").?);
    try std.testing.expectEqualStrings("85", loaded.getGameProperty("player_hp").?);
}

test "serialization v3 split preserves byte-identical output" {
    // Split-guard: the move into serialization/* must not change a single
    // byte. Builds the same representative v3 snapshot the pre-split code
    // produced (len 466, fnv1a64 0xc250f36ab4430269, magic AGSC, version 3,
    // mesh_count 2), then requires header invariants, a golden checksum,
    // deterministic re-encoding, and field-by-field equivalence.
    const alloc = std.testing.allocator;
    var original = SceneState{};
    defer original.deinit(alloc);

    const meshes = try alloc.alloc(MeshEntry, 2);
    meshes[0] = .{
        .id = 101,
        .name = try dupeStr(alloc, "box"),
        .parent_name = try dupeStr(alloc, "sphere"),
        .position = .{ 1.0, 2.0, 3.0 },
        .rotation = .{ 10.0, 20.0, 30.0 },
        .scaling = .{ 1.0, 1.0, 1.0 },
        .is_visible = true,
        .cast_shadows = false,
        .receive_shadows = true,
        .material = .{ .standard = .{ .diffuse = .{ 0.5, 0.25, 0.125 }, .alpha = 0.75, .alpha_mode = 1 } },
    };
    meshes[1] = .{
        .id = 102,
        .name = try dupeStr(alloc, "sphere"),
        .position = .{ -4.0, 0.5, 8.0 },
        .rotation = .{ 0.0, 90.0, 0.0 },
        .scaling = .{ 2.0, 2.0, 2.0 },
        .is_visible = false,
        .cast_shadows = true,
        .receive_shadows = false,
        .material = .{ .pbr = .{
            .albedo = .{ 0.1, 0.2, 0.3 },
            .metallic = 0.9,
            .roughness = 0.15,
            .emissive = .{ 1.0, 0.5, 0.0 },
            .alpha = 1.0,
            .alpha_mode = 0,
        } },
    };
    original.meshes = meshes;
    original.hemi = .{
        .name = try dupeStr(alloc, "hemi"),
        .direction = .{ 0.5, 1.0, 0.3 },
        .diffuse = .{ 1.0, 1.0, 1.0 },
        .ground = .{ 0.2, 0.25, 0.3 },
        .intensity = 0.8,
    };
    original.camera = .{ .arc_rotate = .{
        .name = try dupeStr(alloc, "orbit"),
        .alpha = 0.7,
        .beta = 1.1,
        .radius = 9.0,
        .target = .{ 1.0, 2.0, 3.0 },
        .fov_deg = 55.0,
        .near = 0.5,
        .far = 500.0,
    } };
    original.postprocess.enabled = true;
    original.postprocess.exposure = 1.1;
    try original.setGameProperty(alloc, "quest_stage", "3");
    try original.setGameProperty(alloc, "difficulty", "hard");

    const bytes = try serializeAlloc(alloc, &original);
    defer alloc.free(bytes);

    // Header invariants: magic + v3 + mesh_count 2.
    try std.testing.expectEqualSlices(u8, MAGIC[0..], bytes[0..4]);
    try std.testing.expectEqual(VERSION, std.mem.readInt(u32, bytes[4..8], .little));
    try std.testing.expectEqual(@as(u32, 3), std.mem.readInt(u32, bytes[4..8], .little));
    try std.testing.expectEqual(@as(u32, 2), std.mem.readInt(u32, bytes[8..12], .little));

    // Golden checksum/length from the pre-split encoder (/tmp/serial_golden_pre.bin).
    try std.testing.expectEqual(@as(usize, 466), bytes.len);
    var h: u64 = 14695981039346656037;
    for (bytes) |b| {
        h ^= b;
        h = h *% 1099511628211;
    }
    try std.testing.expectEqual(@as(u64, 0xc250f36ab4430269), h);

    // Round-trip: field equivalence plus deterministic re-encoding.
    var parsed = try deserializeAlloc(alloc, bytes);
    defer parsed.deinit(alloc);
    try expectStatesEqual(&original, &parsed);
    const bytes2 = try serializeAlloc(alloc, &parsed);
    defer alloc.free(bytes2);
    try std.testing.expectEqualSlices(u8, bytes, bytes2);
}
