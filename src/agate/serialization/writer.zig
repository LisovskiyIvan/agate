//! Scene -> bytes: snapshot capture, binary encoding, file + async save.
//!
//! capture() builds the POD SceneState from a live Scene; serializeAlloc()
//! encodes it with format.Writer in the exact layout the facade documents
//! (v3 writer; v2 files remain readable via reader.zig). saveFile and the
//! AsyncSaveTask/saveFileAsync path only move those bytes to disk.

const std = @import("std");
const jobs = @import("../jobs.zig");
const format = @import("format.zig");
const props = @import("props.zig");

const SceneModule = @import("../scene.zig");
const Scene = SceneModule.Scene;

const SceneState = props.SceneState;
const MeshEntry = props.MeshEntry;
const PointEntry = props.PointEntry;
const SpotEntry = props.SpotEntry;
const MaterialEntry = props.MaterialEntry;
const Writer = format.Writer;
const MAGIC = format.MAGIC;
const VERSION = format.VERSION;
const writePostProcess = format.writePostProcess;
const alphaModeToU8 = props.alphaModeToU8;
const PBRMaterial = @import("../material.zig").PBRMaterial;

/// POD snapshot of a live scene. Meshes without a material snapshot as the
/// standard default (white, alpha 1, opaque).
pub fn capture(allocator: std.mem.Allocator, scene: *const Scene) !SceneState {
    var state = SceneState{};
    errdefer state.deinit(allocator);

    var meshes: std.ArrayListUnmanaged(MeshEntry) = .empty;
    errdefer {
        for (meshes.items) |*m| m.deinit(allocator);
        meshes.deinit(allocator);
    }
    for (scene.meshes.items) |mesh| {
        const name = try allocator.dupe(u8, mesh.name);
        errdefer allocator.free(name);
        const parent_name = if (mesh.parent) |p| try allocator.dupe(u8, p.name) else "";
        errdefer if (parent_name.len > 0) allocator.free(parent_name);
        const material: MaterialEntry = if (mesh.material) |mat| blk: {
            // PBR keeps its own entry shape (metallic / roughness / emissive).
            if (mat == .pbr) {
                const p = mat.pbr;
                break :blk .{ .pbr = .{
                    .albedo = .{ p.albedo_color.r, p.albedo_color.g, p.albedo_color.b },
                    .metallic = p.metallic,
                    .roughness = p.roughness,
                    .emissive = .{ p.emissive_color.r, p.emissive_color.g, p.emissive_color.b },
                    .alpha = p.alpha,
                    .alpha_mode = alphaModeToU8(p.alpha_mode),
                    .alpha_cutoff = p.alpha_cutoff,
                    .double_sided = p.double_sided,
                } };
            }
            if (mat == .standard) {
                const s = mat.standard;
                break :blk .{ .pbr = .{
                    .albedo = .{ s.diffuse_color.r, s.diffuse_color.g, s.diffuse_color.b },
                    .metallic = 0.0,
                    .roughness = PBRMaterial.roughnessFromSpecularPower(s.specular_power),
                    .emissive = .{ s.emissive_color.r, s.emissive_color.g, s.emissive_color.b },
                    .alpha = s.alpha,
                    .alpha_mode = alphaModeToU8(s.alpha_mode),
                    .alpha_cutoff = s.alpha_cutoff,
                    .double_sided = s.double_sided,
                } };
            }
            // All other variants (e.g. shader materials) capture as PBR matte equivalent.
            const base = mat.baseColor3();
            break :blk .{ .pbr = .{
                .albedo = .{ base.r, base.g, base.b },
                .metallic = 0.0,
                .roughness = 0.5,
                .emissive = .{ 0.0, 0.0, 0.0 },
                .alpha = mat.alpha(),
                .alpha_mode = alphaModeToU8(mat.alphaMode()),
                .alpha_cutoff = mat.alphaCutoff(),
                .double_sided = mat.isDoubleSided(),
            } };
        } else .{ .pbr = .{} };
        try meshes.append(allocator, .{
            .id = mesh.id,
            .name = name,
            .parent_name = parent_name,
            .position = .{ mesh.position.x, mesh.position.y, mesh.position.z },
            .rotation = .{ mesh.rotation.x, mesh.rotation.y, mesh.rotation.z },
            .scaling = .{ mesh.scaling.x, mesh.scaling.y, mesh.scaling.z },
            .is_visible = mesh.is_visible,
            .cast_shadows = mesh.cast_shadows,
            .receive_shadows = mesh.receive_shadows,
            .material = material,
        });
    }
    state.meshes = try meshes.toOwnedSlice(allocator);

    state.hemi = .{
        .name = try allocator.dupe(u8, scene.lights.hemi.name),
        .direction = .{ scene.lights.hemi.direction.x, scene.lights.hemi.direction.y, scene.lights.hemi.direction.z },
        .diffuse = .{ scene.lights.hemi.diffuse.r, scene.lights.hemi.diffuse.g, scene.lights.hemi.diffuse.b },
        .ground = .{ scene.lights.hemi.ground_color.r, scene.lights.hemi.ground_color.g, scene.lights.hemi.ground_color.b },
        .intensity = scene.lights.hemi.intensity,
    };

    if (scene.lights.directional) |dl| {
        state.directional = .{
            .name = try allocator.dupe(u8, dl.name),
            .direction = .{ dl.direction.x, dl.direction.y, dl.direction.z },
            .diffuse = .{ dl.diffuse.r, dl.diffuse.g, dl.diffuse.b },
            .intensity = dl.intensity,
        };
    }

    var points: std.ArrayListUnmanaged(PointEntry) = .empty;
    errdefer {
        for (points.items) |*p| p.deinit(allocator);
        points.deinit(allocator);
    }
    for (scene.lights.point_lights.items) |pl| {
        const name = try allocator.dupe(u8, pl.name);
        errdefer allocator.free(name);
        try points.append(allocator, .{
            .name = name,
            .position = .{ pl.position.x, pl.position.y, pl.position.z },
            .diffuse = .{ pl.color.r, pl.color.g, pl.color.b },
            .intensity = pl.intensity,
            .range = pl.range,
        });
    }
    state.point_lights = try points.toOwnedSlice(allocator);

    var spots: std.ArrayListUnmanaged(SpotEntry) = .empty;
    errdefer {
        for (spots.items) |*s| s.deinit(allocator);
        spots.deinit(allocator);
    }
    for (scene.lights.spot_lights.items) |sl| {
        const name = try allocator.dupe(u8, sl.name);
        errdefer allocator.free(name);
        try spots.append(allocator, .{
            .name = name,
            .position = .{ sl.position.x, sl.position.y, sl.position.z },
            .direction = .{ sl.direction.x, sl.direction.y, sl.direction.z },
            .diffuse = .{ sl.color.r, sl.color.g, sl.color.b },
            .intensity = sl.intensity,
            .range = sl.range,
            .inner_deg = sl.inner_angle_deg,
            .outer_deg = sl.outer_angle_deg,
        });
    }
    state.spot_lights = try spots.toOwnedSlice(allocator);

    if (scene.active_camera) |cam| {
        state.camera = switch (cam) {
            .arc_rotate => |c| .{ .arc_rotate = .{
                .name = try allocator.dupe(u8, c.name),
                .alpha = c.alpha,
                .beta = c.beta,
                .radius = c.radius,
                .target = .{ c.target.x, c.target.y, c.target.z },
                .fov_deg = c.fov_deg,
                .near = c.near,
                .far = c.far,
            } },
            .free => |c| .{ .free = .{
                .name = try allocator.dupe(u8, c.name),
                .position = .{ c.position.x, c.position.y, c.position.z },
                .rotation = .{ c.rotation.x, c.rotation.y, c.rotation.z },
                .fov_deg = c.fov_deg,
                .near = c.near,
                .far = c.far,
                .speed = c.speed,
                .angular_sensitivity = c.angular_sensitivity,
            } },
            .follow => |c| .{ .follow = .{
                .name = try allocator.dupe(u8, c.name),
                .position = .{ c.position.x, c.position.y, c.position.z },
                .target_position = .{ c.target_position.x, c.target_position.y, c.target_position.z },
                .radius = c.radius,
                .height_offset = c.height_offset,
                .rotation_offset_deg = c.rotation_offset_deg,
                .fov_deg = c.fov_deg,
                .near = c.near,
                .far = c.far,
                .lerp_speed = c.lerp_speed,
            } },
            .target => |c| .{ .target = .{
                .name = try allocator.dupe(u8, c.name),
                .position = .{ c.position.x, c.position.y, c.position.z },
                .target = .{ c.target.x, c.target.y, c.target.z },
                .up = .{ c.up.x, c.up.y, c.up.z },
                .fov_deg = c.fov_deg,
                .near = c.near,
                .far = c.far,
                .smoothing = c.smoothing,
            } },
            .fly => |c| .{ .fly = .{
                .name = try allocator.dupe(u8, c.name),
                .position = .{ c.position.x, c.position.y, c.position.z },
                .rotation = .{ c.rotation.x, c.rotation.y, c.rotation.z },
                .fov_deg = c.fov_deg,
                .near = c.near,
                .far = c.far,
                .speed = c.speed,
                .boost_multiplier = c.boost_multiplier,
                .angular_sensitivity = c.angular_sensitivity,
                .roll_speed_deg = c.roll_speed_deg,
            } },
        };
    }

    state.render = .{
        .skybox_enabled = scene.sky.enabled,
        .skybox_exposure = scene.sky.exposure,
        .shadows_enabled = scene.shadows.enabled,
        .shadow_softness = scene.shadows.softness,
        .ibl_intensity = scene.sky.ibl_intensity,
    };
    state.postprocess = scene.post_process;

    return state;
}

/// Serializes a snapshot into a freshly allocated byte buffer (little-endian
/// layout documented at the top of the facade). Fails with TooLarge when a
/// slice/string length does not fit into u32.
pub fn serializeAlloc(allocator: std.mem.Allocator, state: *const SceneState) ![]u8 {
    var w = Writer{ .alloc = allocator };
    errdefer w.buf.deinit(allocator);

    try w.bytes(MAGIC[0..]);
    try w.u32le(VERSION);

    try w.u32le(std.math.cast(u32, state.meshes.len) orelse return error.TooLarge);
    for (state.meshes) |*m| {
        try w.u64le(m.id);
        try w.str(m.name);
        try w.str(m.parent_name);
        try w.vec3(m.position);
        try w.vec3(m.rotation);
        try w.vec3(m.scaling);
        var flags: u8 = 0;
        if (m.is_visible) flags |= 1;
        if (m.cast_shadows) flags |= 2;
        if (m.receive_shadows) flags |= 4;
        try w.byte(flags);
        switch (m.material) {
            .standard => |*s| {
                try w.byte(0);
                try w.vec3(s.diffuse);
                try w.f32le(s.alpha);
                try w.byte(s.alpha_mode);
                try w.f32le(s.alpha_cutoff);
                try w.bool8(s.double_sided);
            },
            .pbr => |*p| {
                try w.byte(1);
                try w.vec3(p.albedo);
                try w.f32le(p.metallic);
                try w.f32le(p.roughness);
                try w.vec3(p.emissive);
                try w.f32le(p.alpha);
                try w.byte(p.alpha_mode);
                try w.f32le(p.alpha_cutoff);
                try w.bool8(p.double_sided);
            },
        }
    }

    try w.str(state.hemi.name);
    try w.vec3(state.hemi.direction);
    try w.vec3(state.hemi.diffuse);
    try w.vec3(state.hemi.ground);
    try w.f32le(state.hemi.intensity);

    if (state.directional) |*d| {
        try w.byte(1);
        try w.str(d.name);
        try w.vec3(d.direction);
        try w.vec3(d.diffuse);
        try w.f32le(d.intensity);
    } else {
        try w.byte(0);
    }

    try w.u32le(std.math.cast(u32, state.point_lights.len) orelse return error.TooLarge);
    for (state.point_lights) |*p| {
        try w.str(p.name);
        try w.vec3(p.position);
        try w.vec3(p.diffuse);
        try w.f32le(p.intensity);
        try w.f32le(p.range);
    }

    try w.u32le(std.math.cast(u32, state.spot_lights.len) orelse return error.TooLarge);
    for (state.spot_lights) |*s| {
        try w.str(s.name);
        try w.vec3(s.position);
        try w.vec3(s.direction);
        try w.vec3(s.diffuse);
        try w.f32le(s.intensity);
        try w.f32le(s.range);
        try w.f32le(s.inner_deg);
        try w.f32le(s.outer_deg);
    }

    switch (state.camera) {
        .none => try w.byte(0),
        .arc_rotate => |*c| {
            try w.byte(1);
            try w.str(c.name);
            try w.f32le(c.alpha);
            try w.f32le(c.beta);
            try w.f32le(c.radius);
            try w.vec3(c.target);
            try w.f32le(c.fov_deg);
            try w.f32le(c.near);
            try w.f32le(c.far);
        },
        .free => |*c| {
            try w.byte(2);
            try w.str(c.name);
            try w.vec3(c.position);
            try w.vec3(c.rotation);
            try w.f32le(c.fov_deg);
            try w.f32le(c.near);
            try w.f32le(c.far);
            try w.f32le(c.speed);
            try w.f32le(c.angular_sensitivity);
        },
        .follow => |*c| {
            try w.byte(3);
            try w.str(c.name);
            try w.vec3(c.position);
            try w.vec3(c.target_position);
            try w.f32le(c.radius);
            try w.f32le(c.height_offset);
            try w.f32le(c.rotation_offset_deg);
            try w.f32le(c.fov_deg);
            try w.f32le(c.near);
            try w.f32le(c.far);
            try w.f32le(c.lerp_speed);
        },
        .target => |*c| {
            try w.byte(4);
            try w.str(c.name);
            try w.vec3(c.position);
            try w.vec3(c.target);
            try w.vec3(c.up);
            try w.f32le(c.fov_deg);
            try w.f32le(c.near);
            try w.f32le(c.far);
            try w.f32le(c.smoothing);
        },
        .fly => |*c| {
            try w.byte(5);
            try w.str(c.name);
            try w.vec3(c.position);
            try w.vec3(c.rotation);
            try w.f32le(c.fov_deg);
            try w.f32le(c.near);
            try w.f32le(c.far);
            try w.f32le(c.speed);
            try w.f32le(c.boost_multiplier);
            try w.f32le(c.angular_sensitivity);
            try w.f32le(c.roll_speed_deg);
        },
    }

    try w.bool8(state.render.skybox_enabled);
    try w.f32le(state.render.skybox_exposure);
    try w.bool8(state.render.shadows_enabled);
    try w.f32le(state.render.shadow_softness);
    try w.f32le(state.render.ibl_intensity);

    try writePostProcess(&w, &state.postprocess);

    try w.u32le(std.math.cast(u32, state.game_properties.len) orelse return error.TooLarge);
    for (state.game_properties) |gp| {
        try w.str(gp.key);
        try w.str(gp.value);
    }

    return w.buf.toOwnedSlice(allocator);
}

/// Writes serializeAlloc output to path (created/truncated).
pub fn saveFile(allocator: std.mem.Allocator, state: *const SceneState, path: []const u8) !void {
    const bytes = try serializeAlloc(allocator, state);
    defer allocator.free(bytes);
    const io = std.Io.Threaded.global_single_threaded.io();
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
}

pub const AsyncSaveTask = struct {
    pub const State = enum(u8) {
        pending = 0,
        serializing = 1,
        writing = 2,
        completed = 3,
        failed = 4,
    };

    allocator: std.mem.Allocator,
    path: []u8,
    scene_state: SceneState,
    state: std.atomic.Value(State) = std.atomic.Value(State).init(.pending),
    bytes_written: usize = 0,
    err_name: ?[:0]const u8 = null,

    pub fn isDone(self: *const AsyncSaveTask) bool {
        const s = self.state.load(.acquire);
        return s == .completed or s == .failed;
    }

    pub fn isSuccess(self: *const AsyncSaveTask) bool {
        return self.state.load(.acquire) == .completed;
    }

    pub fn deinit(self: *AsyncSaveTask) void {
        self.scene_state.deinit(self.allocator);
        self.allocator.free(self.path);
        self.allocator.destroy(self);
    }
};

fn runSaveTask(ctx: *anyopaque) void {
    const task: *AsyncSaveTask = @ptrCast(@alignCast(ctx));
    task.state.store(.serializing, .release);

    const bytes = serializeAlloc(task.allocator, &task.scene_state) catch |err| {
        task.err_name = @errorName(err);
        task.state.store(.failed, .release);
        return;
    };
    defer task.allocator.free(bytes);

    task.state.store(.writing, .release);
    const io = std.Io.Threaded.global_single_threaded.io();
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = task.path, .data = bytes }) catch |err| {
        task.err_name = @errorName(err);
        task.state.store(.failed, .release);
        return;
    };

    task.bytes_written = bytes.len;
    task.state.store(.completed, .release);
}

/// Dispatches serializing and writing the scene state to path on a background
/// task thread. Ownership of `scene_state` transfers into the returned task.
/// The caller polls `task.isDone()` and must call `task.deinit()` when finished.
pub fn saveFileAsync(allocator: std.mem.Allocator, runner: *jobs.TaskRunner, scene_state: SceneState, path: []const u8) !*AsyncSaveTask {
    const task = try allocator.create(AsyncSaveTask);
    errdefer allocator.destroy(task);

    const owned_path = try allocator.dupe(u8, path);
    errdefer allocator.free(owned_path);

    task.* = .{
        .allocator = allocator,
        .path = owned_path,
        .scene_state = scene_state,
        .state = std.atomic.Value(AsyncSaveTask.State).init(.pending),
        .bytes_written = 0,
        .err_name = null,
    };

    runner.post(task, runSaveTask);
    return task;
}
