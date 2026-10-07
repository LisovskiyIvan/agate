//! Bytes -> scene: binary parsing, snapshot restore, file + async load.
//!
//! deserializeAlloc() strictly validates magic, version (3 only; v2 rejected), counts,
//! string lengths and offset arithmetic; any short read or invalid
//! discriminant reports Truncated. restore() applies a snapshot to a live
//! scene (meshes match by id-then-name, hierarchy re-linked in a second
//! pass). loadFile and the AsyncLoadTask/loadFileAsync path only move bytes
//! from disk.

const std = @import("std");
const jobs = @import("../jobs.zig");
const math = @import("math");
const Vec3 = math.Vec3;
const Color3 = math.Color3;
const format = @import("format.zig");
const props = @import("props.zig");

const SceneModule = @import("../scene.zig");
const Scene = SceneModule.Scene;
const CameraModule = @import("../camera.zig");
const TargetCamera = CameraModule.TargetCamera;
const FlyCamera = CameraModule.FlyCamera;
const Lights = @import("../lights.zig");
const HemisphericLight = Lights.HemisphericLight;
const Mesh = @import("../mesh.zig").Mesh;

const SceneState = props.SceneState;
const MeshEntry = props.MeshEntry;
const PointEntry = props.PointEntry;
const SpotEntry = props.SpotEntry;
const DirectionalEntry = props.DirectionalEntry;
const CameraEntry = props.CameraEntry;
const ArcRotateEntry = props.ArcRotateEntry;
const FreeEntry = props.FreeEntry;
const FollowEntry = props.FollowEntry;
const TargetEntry = props.TargetEntry;
const FlyEntry = props.FlyEntry;
const GameProperty = props.GameProperty;
const Reader = format.Reader;
const MAGIC = format.MAGIC;
const MAX_FILE_BYTES = format.MAX_FILE_BYTES;
const readPostProcess = format.readPostProcess;
const findMesh = props.findMesh;
const findMeshByIdOrName = props.findMeshByIdOrName;
const restoreMeshMaterial = props.restoreMeshMaterial;

/// Applies a snapshot to a live scene. Meshes match by id (if non-zero) or by name,
/// unknown names are ignored. Parent-child hierarchy is reconstructed.
/// Point/spot lights are destroyed and recreated; the directional light is
/// replaced via createDirectionalLight. Follow-camera target_mesh resets to null
/// (target_position is kept).
pub fn restore(scene: *Scene, state: *const SceneState) void {
    for (state.meshes) |*entry| {
        const mesh = findMeshByIdOrName(scene, entry.id, entry.name) orelse continue;
        mesh.position = Vec3.new(entry.position[0], entry.position[1], entry.position[2]);
        mesh.rotation = Vec3.new(entry.rotation[0], entry.rotation[1], entry.rotation[2]);
        mesh.scaling = Vec3.new(entry.scaling[0], entry.scaling[1], entry.scaling[2]);
        mesh.is_visible = entry.is_visible;
        mesh.cast_shadows = entry.cast_shadows;
        mesh.receive_shadows = entry.receive_shadows;
        if (entry.id != 0) mesh.id = entry.id;
        restoreMeshMaterial(scene, mesh, &entry.material);
    }

    // Second pass: restore parent links across meshes
    for (state.meshes) |*entry| {
        const mesh = findMeshByIdOrName(scene, entry.id, entry.name) orelse continue;
        if (entry.parent_name.len == 0) {
            mesh.parent = null;
        } else if (findMesh(scene, entry.parent_name)) |parent_mesh| {
            mesh.parent = parent_mesh;
        }
    }

    scene.lights.hemi = HemisphericLight.init(scene.lights.hemi.name, .{
        .direction = Vec3.new(state.hemi.direction[0], state.hemi.direction[1], state.hemi.direction[2]),
        .diffuse = Color3.new(state.hemi.diffuse[0], state.hemi.diffuse[1], state.hemi.diffuse[2]),
        .ground_color = Color3.new(state.hemi.ground[0], state.hemi.ground[1], state.hemi.ground[2]),
        .intensity = state.hemi.intensity,
    });

    if (state.directional) |*d| {
        // createDirectionalLight destroys the previous sun internally.
        const owned: ?[]u8 = scene.allocator.dupe(u8, d.name) catch null;
        if (scene.createDirectionalLight(owned orelse d.name, .{
            .direction = Vec3.new(d.direction[0], d.direction[1], d.direction[2]),
            .diffuse = Color3.new(d.diffuse[0], d.diffuse[1], d.diffuse[2]),
            .intensity = d.intensity,
        })) |dl| {
            if (owned != null) dl.owns_name = true;
        } else |_| {
            if (owned) |o| scene.allocator.free(o);
        }
    } else if (scene.lights.directional) |old| {
        if (old.owns_name) scene.allocator.free(old.name);
        scene.allocator.destroy(old);
        scene.lights.directional = null;
    }

    for (scene.lights.point_lights.items) |pl| {
        if (pl.owns_name) scene.allocator.free(pl.name);
        scene.allocator.destroy(pl);
    }
    scene.lights.point_lights.clearRetainingCapacity();
    for (state.point_lights) |*p| {
        const owned: ?[]u8 = scene.allocator.dupe(u8, p.name) catch null;
        if (scene.createPointLight(owned orelse p.name, .{
            .position = Vec3.new(p.position[0], p.position[1], p.position[2]),
            .color = Color3.new(p.diffuse[0], p.diffuse[1], p.diffuse[2]),
            .intensity = p.intensity,
            .range = p.range,
        })) |pl| {
            if (owned != null) pl.owns_name = true;
        } else |_| {
            if (owned) |o| scene.allocator.free(o);
        }
    }

    for (scene.lights.spot_lights.items) |sl| {
        if (sl.owns_name) scene.allocator.free(sl.name);
        scene.allocator.destroy(sl);
    }
    scene.lights.spot_lights.clearRetainingCapacity();
    for (state.spot_lights) |*s| {
        const owned: ?[]u8 = scene.allocator.dupe(u8, s.name) catch null;
        if (scene.createSpotLight(owned orelse s.name, .{
            .position = Vec3.new(s.position[0], s.position[1], s.position[2]),
            .direction = Vec3.new(s.direction[0], s.direction[1], s.direction[2]),
            .color = Color3.new(s.diffuse[0], s.diffuse[1], s.diffuse[2]),
            .intensity = s.intensity,
            .range = s.range,
            .inner_angle_deg = s.inner_deg,
            .outer_angle_deg = s.outer_deg,
        })) |sl| {
            if (owned != null) sl.owns_name = true;
        } else |_| {
            if (owned) |o| scene.allocator.free(o);
        }
    }

    switch (state.camera) {
        .none => scene.setActiveCamera(null, null),
        .arc_rotate => |*c| {
            const owned: ?[]u8 = scene.allocator.dupe(u8, c.name) catch null;
            scene.setActiveCamera(.{ .arc_rotate = .{
                .name = owned orelse c.name,
                .alpha = c.alpha,
                .beta = c.beta,
                .radius = c.radius,
                .target = Vec3.new(c.target[0], c.target[1], c.target[2]),
                .fov_deg = c.fov_deg,
                .near = c.near,
                .far = c.far,
            } }, owned);
        },
        .free => |*c| {
            const owned: ?[]u8 = scene.allocator.dupe(u8, c.name) catch null;
            scene.setActiveCamera(.{ .free = .{
                .name = owned orelse c.name,
                .position = Vec3.new(c.position[0], c.position[1], c.position[2]),
                .rotation = Vec3.new(c.rotation[0], c.rotation[1], c.rotation[2]),
                .fov_deg = c.fov_deg,
                .near = c.near,
                .far = c.far,
                .speed = c.speed,
                .angular_sensitivity = c.angular_sensitivity,
            } }, owned);
        },
        .follow => |*c| {
            const owned: ?[]u8 = scene.allocator.dupe(u8, c.name) catch null;
            scene.setActiveCamera(.{ .follow = .{
                .name = owned orelse c.name,
                .target_mesh = null,
                .target_position = Vec3.new(c.target_position[0], c.target_position[1], c.target_position[2]),
                .position = Vec3.new(c.position[0], c.position[1], c.position[2]),
                .radius = c.radius,
                .height_offset = c.height_offset,
                .rotation_offset_deg = c.rotation_offset_deg,
                .fov_deg = c.fov_deg,
                .near = c.near,
                .far = c.far,
                .lerp_speed = c.lerp_speed,
            } }, owned);
        },
        .target => |*c| {
            const owned: ?[]u8 = scene.allocator.dupe(u8, c.name) catch null;
            scene.setActiveCamera(.{ .target = TargetCamera.init(owned orelse c.name, .{
                .position = Vec3.new(c.position[0], c.position[1], c.position[2]),
                .target = Vec3.new(c.target[0], c.target[1], c.target[2]),
                .up = Vec3.new(c.up[0], c.up[1], c.up[2]),
                .fov_deg = c.fov_deg,
                .near = c.near,
                .far = c.far,
                .smoothing = c.smoothing,
            }) }, owned);
        },
        .fly => |*c| {
            const owned: ?[]u8 = scene.allocator.dupe(u8, c.name) catch null;
            scene.setActiveCamera(.{ .fly = FlyCamera.init(owned orelse c.name, .{
                .position = Vec3.new(c.position[0], c.position[1], c.position[2]),
                .rotation = Vec3.new(c.rotation[0], c.rotation[1], c.rotation[2]),
                .fov_deg = c.fov_deg,
                .near = c.near,
                .far = c.far,
                .speed = c.speed,
                .boost_multiplier = c.boost_multiplier,
                .angular_sensitivity = c.angular_sensitivity,
                .roll_speed_deg = c.roll_speed_deg,
            }) }, owned);
        },
    }

    scene.sky.enabled = state.render.skybox_enabled;
    scene.sky.exposure = state.render.skybox_exposure;
    scene.shadows.enabled = state.render.shadows_enabled;
    scene.shadows.softness = state.render.shadow_softness;
    scene.sky.ibl_intensity = state.render.ibl_intensity;
    scene.post_process = state.postprocess;
}

fn readMeshes(allocator: std.mem.Allocator, r: *Reader) ![]MeshEntry {
    const count = try r.readCount();
    var list: std.ArrayListUnmanaged(MeshEntry) = .empty;
    errdefer {
        for (list.items) |*m| m.deinit(allocator);
        list.deinit(allocator);
    }
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        var entry = MeshEntry{};
        entry.id = try r.readU64();
        entry.name = try r.readString(allocator);
        errdefer entry.deinit(allocator);
        entry.parent_name = try r.readString(allocator);
        entry.position = try r.readVec3();
        entry.rotation = try r.readVec3();
        entry.scaling = try r.readVec3();
        const flags = try r.readU8();
        if (flags & ~@as(u8, 7) != 0) return error.Truncated;
        entry.is_visible = flags & 1 != 0;
        entry.cast_shadows = flags & 2 != 0;
        entry.receive_shadows = flags & 4 != 0;
        switch (try r.readU8()) {
            0 => entry.material = .{ .standard = .{
                .diffuse = try r.readVec3(),
                .alpha = try r.readF32(),
                .alpha_mode = try r.readU8(),
                .alpha_cutoff = try r.readF32(),
                .double_sided = try r.readBool(),
            } },
            1 => entry.material = .{ .pbr = .{
                .albedo = try r.readVec3(),
                .metallic = try r.readF32(),
                .roughness = try r.readF32(),
                .emissive = try r.readVec3(),
                .alpha = try r.readF32(),
                .alpha_mode = try r.readU8(),
                .alpha_cutoff = try r.readF32(),
                .double_sided = try r.readBool(),
            } },
            else => return error.Truncated,
        }
        if (entry.material == .standard and entry.material.standard.alpha_mode > 2) return error.Truncated;
        if (entry.material == .pbr and entry.material.pbr.alpha_mode > 2) return error.Truncated;
        try list.append(allocator, entry);
    }
    return list.toOwnedSlice(allocator);
}

fn readPoints(allocator: std.mem.Allocator, r: *Reader) ![]PointEntry {
    const count = try r.readCount();
    var list: std.ArrayListUnmanaged(PointEntry) = .empty;
    errdefer {
        for (list.items) |*p| p.deinit(allocator);
        list.deinit(allocator);
    }
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        var entry = PointEntry{};
        entry.name = try r.readString(allocator);
        errdefer entry.deinit(allocator);
        entry.position = try r.readVec3();
        entry.diffuse = try r.readVec3();
        entry.intensity = try r.readF32();
        entry.range = try r.readF32();
        try list.append(allocator, entry);
    }
    return list.toOwnedSlice(allocator);
}

fn readSpots(allocator: std.mem.Allocator, r: *Reader) ![]SpotEntry {
    const count = try r.readCount();
    var list: std.ArrayListUnmanaged(SpotEntry) = .empty;
    errdefer {
        for (list.items) |*s| s.deinit(allocator);
        list.deinit(allocator);
    }
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        var entry = SpotEntry{};
        entry.name = try r.readString(allocator);
        errdefer entry.deinit(allocator);
        entry.position = try r.readVec3();
        entry.direction = try r.readVec3();
        entry.diffuse = try r.readVec3();
        entry.intensity = try r.readF32();
        entry.range = try r.readF32();
        entry.inner_deg = try r.readF32();
        entry.outer_deg = try r.readF32();
        try list.append(allocator, entry);
    }
    return list.toOwnedSlice(allocator);
}

fn readDirectional(allocator: std.mem.Allocator, r: *Reader) !DirectionalEntry {
    var d = DirectionalEntry{};
    errdefer d.deinit(allocator);
    d.name = try r.readString(allocator);
    d.direction = try r.readVec3();
    d.diffuse = try r.readVec3();
    d.intensity = try r.readF32();
    return d;
}

fn readCamera(allocator: std.mem.Allocator, r: *Reader) !CameraEntry {
    switch (try r.readU8()) {
        0 => return .none,
        1 => {
            var c = ArcRotateEntry{};
            errdefer if (c.name.len > 0) allocator.free(c.name);
            c.name = try r.readString(allocator);
            c.alpha = try r.readF32();
            c.beta = try r.readF32();
            c.radius = try r.readF32();
            c.target = try r.readVec3();
            c.fov_deg = try r.readF32();
            c.near = try r.readF32();
            c.far = try r.readF32();
            return .{ .arc_rotate = c };
        },
        2 => {
            var c = FreeEntry{};
            errdefer if (c.name.len > 0) allocator.free(c.name);
            c.name = try r.readString(allocator);
            c.position = try r.readVec3();
            c.rotation = try r.readVec3();
            c.fov_deg = try r.readF32();
            c.near = try r.readF32();
            c.far = try r.readF32();
            c.speed = try r.readF32();
            c.angular_sensitivity = try r.readF32();
            return .{ .free = c };
        },
        3 => {
            var c = FollowEntry{};
            errdefer if (c.name.len > 0) allocator.free(c.name);
            c.name = try r.readString(allocator);
            c.position = try r.readVec3();
            c.target_position = try r.readVec3();
            c.radius = try r.readF32();
            c.height_offset = try r.readF32();
            c.rotation_offset_deg = try r.readF32();
            c.fov_deg = try r.readF32();
            c.near = try r.readF32();
            c.far = try r.readF32();
            c.lerp_speed = try r.readF32();
            return .{ .follow = c };
        },
        4 => {
            var c = TargetEntry{};
            errdefer if (c.name.len > 0) allocator.free(c.name);
            c.name = try r.readString(allocator);
            c.position = try r.readVec3();
            c.target = try r.readVec3();
            c.up = try r.readVec3();
            c.fov_deg = try r.readF32();
            c.near = try r.readF32();
            c.far = try r.readF32();
            c.smoothing = try r.readF32();
            return .{ .target = c };
        },
        5 => {
            var c = FlyEntry{};
            errdefer if (c.name.len > 0) allocator.free(c.name);
            c.name = try r.readString(allocator);
            c.position = try r.readVec3();
            c.rotation = try r.readVec3();
            c.fov_deg = try r.readF32();
            c.near = try r.readF32();
            c.far = try r.readF32();
            c.speed = try r.readF32();
            c.boost_multiplier = try r.readF32();
            c.angular_sensitivity = try r.readF32();
            c.roll_speed_deg = try r.readF32();
            return .{ .fly = c };
        },
        else => return error.Truncated,
    }
}

/// Parses a snapshot from bytes. Strictly validates magic, version, counts,
/// string lengths and offset arithmetic (overflow-safe via std.math.add/cast,
/// capped via MAX_ENTRIES/MAX_STRING_BYTES); any short read or invalid
/// discriminant (material/camera/present/bool/tonemapping) reports Truncated.
/// On error nothing leaks (per-section errdefer + state errdefer).
pub fn deserializeAlloc(allocator: std.mem.Allocator, bytes: []const u8) !SceneState {
    if (bytes.len < MAGIC.len) return error.Truncated;
    if (!std.mem.eql(u8, bytes[0..MAGIC.len], MAGIC[0..])) return error.BadMagic;
    var r = Reader{ .bytes = bytes, .pos = MAGIC.len };

    const version = try r.readU32();
    if (version != 3) return error.UnsupportedVersion;

    var state = SceneState{};
    errdefer state.deinit(allocator);

    state.meshes = try readMeshes(allocator, &r);

    state.hemi.name = try r.readString(allocator);
    state.hemi.direction = try r.readVec3();
    state.hemi.diffuse = try r.readVec3();
    state.hemi.ground = try r.readVec3();
    state.hemi.intensity = try r.readF32();

    switch (try r.readU8()) {
        0 => state.directional = null,
        1 => state.directional = try readDirectional(allocator, &r),
        else => return error.Truncated,
    }

    state.point_lights = try readPoints(allocator, &r);
    state.spot_lights = try readSpots(allocator, &r);
    state.camera = try readCamera(allocator, &r);

    state.render.skybox_enabled = try r.readBool();
    state.render.skybox_exposure = try r.readF32();
    state.render.shadows_enabled = try r.readBool();
    state.render.shadow_softness = try r.readF32();
    state.render.ibl_intensity = try r.readF32();

    state.postprocess = try readPostProcess(&r);

    {
        const count = try r.readCount();
        var props_list: std.ArrayListUnmanaged(GameProperty) = .empty;
        errdefer {
            for (props_list.items) |*p| p.deinit(allocator);
            props_list.deinit(allocator);
        }
        var i: u32 = 0;
        while (i < count) : (i += 1) {
            var prop = GameProperty{};
            prop.key = try r.readString(allocator);
            errdefer prop.deinit(allocator);
            prop.value = try r.readString(allocator);
            try props_list.append(allocator, prop);
        }
        state.game_properties = try props_list.toOwnedSlice(allocator);
    }

    return state;
}

/// Reads a whole file and parses it with deserializeAlloc.
pub fn loadFile(allocator: std.mem.Allocator, path: []const u8) !SceneState {
    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const len = try file.length(io);
    if (len > MAX_FILE_BYTES) return error.TooLarge;
    const n = std.math.cast(usize, len) orelse return error.TooLarge;
    const bytes = try allocator.alloc(u8, n);
    defer allocator.free(bytes);
    const read = try file.readPositionalAll(io, bytes, 0);
    if (read < bytes.len) return error.Truncated;
    return deserializeAlloc(allocator, bytes);
}

pub const AsyncLoadTask = struct {
    pub const State = enum(u8) {
        pending = 0,
        reading = 1,
        deserializing = 2,
        completed = 3,
        failed = 4,
    };

    allocator: std.mem.Allocator,
    path: []u8,
    state: std.atomic.Value(State) = std.atomic.Value(State).init(.pending),
    result: ?SceneState = null,
    err_name: ?[:0]const u8 = null,

    pub fn isDone(self: *const AsyncLoadTask) bool {
        const s = self.state.load(.acquire);
        return s == .completed or s == .failed;
    }

    pub fn isSuccess(self: *const AsyncLoadTask) bool {
        return self.state.load(.acquire) == .completed;
    }

    pub fn deinit(self: *AsyncLoadTask) void {
        if (self.result) |*r| r.deinit(self.allocator);
        self.allocator.free(self.path);
        self.allocator.destroy(self);
    }
};

fn runLoadTask(ctx: *anyopaque) void {
    const task: *AsyncLoadTask = @ptrCast(@alignCast(ctx));
    task.state.store(.reading, .release);

    const io = std.Io.Threaded.global_single_threaded.io();
    const file = std.Io.Dir.cwd().openFile(io, task.path, .{}) catch |err| {
        task.err_name = @errorName(err);
        task.state.store(.failed, .release);
        return;
    };
    defer file.close(io);

    const len = file.length(io) catch |err| {
        task.err_name = @errorName(err);
        task.state.store(.failed, .release);
        return;
    };
    if (len > MAX_FILE_BYTES) {
        task.err_name = @errorName(error.TooLarge);
        task.state.store(.failed, .release);
        return;
    }
    const n = std.math.cast(usize, len) orelse {
        task.err_name = @errorName(error.TooLarge);
        task.state.store(.failed, .release);
        return;
    };
    const bytes = task.allocator.alloc(u8, n) catch |err| {
        task.err_name = @errorName(err);
        task.state.store(.failed, .release);
        return;
    };
    defer task.allocator.free(bytes);

    const read = file.readPositionalAll(io, bytes, 0) catch |err| {
        task.err_name = @errorName(err);
        task.state.store(.failed, .release);
        return;
    };
    if (read < bytes.len) {
        task.err_name = @errorName(error.Truncated);
        task.state.store(.failed, .release);
        return;
    }

    task.state.store(.deserializing, .release);
    const scene_state = deserializeAlloc(task.allocator, bytes) catch |err| {
        task.err_name = @errorName(err);
        task.state.store(.failed, .release);
        return;
    };

    task.result = scene_state;
    task.state.store(.completed, .release);
}

/// Dispatches reading and deserializing a scene state file on a background
/// task thread. The caller polls `task.isDone()`, uses `task.result` on success,
/// and must call `task.deinit()` when finished.
pub fn loadFileAsync(allocator: std.mem.Allocator, runner: *jobs.TaskRunner, path: []const u8) !*AsyncLoadTask {
    const task = try allocator.create(AsyncLoadTask);
    errdefer allocator.destroy(task);

    const owned_path = try allocator.dupe(u8, path);
    errdefer allocator.free(owned_path);

    task.* = .{
        .allocator = allocator,
        .path = owned_path,
        .state = std.atomic.Value(AsyncLoadTask.State).init(.pending),
        .result = null,
        .err_name = null,
    };

    runner.post(task, runLoadTask);
    return task;
}
