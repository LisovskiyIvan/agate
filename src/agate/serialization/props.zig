//! Snapshot types plus entity-ID / hierarchy / custom-property helpers.
//!
//! All strings/slices in SceneState are owned (allocator.dupe) and released
//! by SceneState.deinit. Meshes carry the stable entity `id` (u64) and
//! `parent_name` for hierarchy persistence; the tail game-property table
//! stores custom key-value pairs. Material kind mapping (AlphaMode <-> u8)
//! and mesh lookup / material-restore helpers live here so writer.zig and
//! reader.zig share them without importing each other.

const std = @import("std");
const math = @import("math");
const Color3 = math.Color3;
const SceneModule = @import("../scene.zig");
const Scene = SceneModule.Scene;
const PostProcessOptions = @import("../postprocess.zig").PostProcessOptions;
const MaterialModule = @import("../material.zig");
const StandardMaterial = MaterialModule.StandardMaterial;
const PBRMaterial = MaterialModule.PBRMaterial;
const AlphaMode = MaterialModule.AlphaMode;
const Mesh = @import("../mesh.zig").Mesh;

pub const StandardEntry = struct {
    diffuse: [3]f32 = .{ 1.0, 1.0, 1.0 },
    alpha: f32 = 1.0,
    alpha_mode: u8 = 0, // 0 opaque, 1 blend, 2 cutout
    alpha_cutoff: f32 = 0.5,
    double_sided: bool = false,
};

pub const PbrEntry = struct {
    albedo: [3]f32 = .{ 1.0, 1.0, 1.0 },
    metallic: f32 = 0.0,
    roughness: f32 = 0.5,
    emissive: [3]f32 = .{ 0.0, 0.0, 0.0 },
    alpha: f32 = 1.0,
    alpha_mode: u8 = 0, // 0 opaque, 1 blend, 2 cutout
    alpha_cutoff: f32 = 0.5,
    double_sided: bool = false,
};

pub const MaterialEntry = union(enum) {
    standard: StandardEntry,
    pbr: PbrEntry,
};

pub const MeshEntry = struct {
    id: u64 = 0,
    name: []const u8 = "",
    parent_name: []const u8 = "",
    position: [3]f32 = .{ 0.0, 0.0, 0.0 },
    rotation: [3]f32 = .{ 0.0, 0.0, 0.0 },
    scaling: [3]f32 = .{ 1.0, 1.0, 1.0 },
    is_visible: bool = true,
    cast_shadows: bool = true,
    receive_shadows: bool = true,
    material: MaterialEntry = .{ .pbr = .{} },

    pub fn deinit(self: *MeshEntry, allocator: std.mem.Allocator) void {
        if (self.name.len > 0) allocator.free(self.name);
        if (self.parent_name.len > 0) allocator.free(self.parent_name);
    }
};

pub const HemiEntry = struct {
    name: []const u8 = "",
    direction: [3]f32 = .{ 0.0, 1.0, 0.0 },
    diffuse: [3]f32 = .{ 1.0, 1.0, 1.0 },
    ground: [3]f32 = .{ 0.2, 0.25, 0.3 },
    intensity: f32 = 1.0,

    pub fn deinit(self: *HemiEntry, allocator: std.mem.Allocator) void {
        if (self.name.len > 0) allocator.free(self.name);
    }
};

pub const DirectionalEntry = struct {
    name: []const u8 = "",
    direction: [3]f32 = .{ 0.5, 1.0, 0.5 },
    diffuse: [3]f32 = .{ 1.0, 1.0, 1.0 },
    intensity: f32 = 1.0,

    pub fn deinit(self: *DirectionalEntry, allocator: std.mem.Allocator) void {
        if (self.name.len > 0) allocator.free(self.name);
    }
};

pub const PointEntry = struct {
    name: []const u8 = "",
    position: [3]f32 = .{ 0.0, 0.0, 0.0 },
    diffuse: [3]f32 = .{ 1.0, 1.0, 1.0 },
    intensity: f32 = 1.0,
    range: f32 = 10.0,

    pub fn deinit(self: *PointEntry, allocator: std.mem.Allocator) void {
        if (self.name.len > 0) allocator.free(self.name);
    }
};

pub const SpotEntry = struct {
    name: []const u8 = "",
    position: [3]f32 = .{ 0.0, 0.0, 0.0 },
    direction: [3]f32 = .{ 0.0, -1.0, 0.0 },
    diffuse: [3]f32 = .{ 1.0, 1.0, 1.0 },
    intensity: f32 = 1.0,
    range: f32 = 15.0,
    inner_deg: f32 = 15.0,
    outer_deg: f32 = 30.0,

    pub fn deinit(self: *SpotEntry, allocator: std.mem.Allocator) void {
        if (self.name.len > 0) allocator.free(self.name);
    }
};

pub const ArcRotateEntry = struct {
    name: []const u8 = "",
    alpha: f32 = 0.0,
    beta: f32 = std.math.pi / 3.0,
    radius: f32 = 5.0,
    target: [3]f32 = .{ 0.0, 0.0, 0.0 },
    fov_deg: f32 = 60.0,
    near: f32 = 0.1,
    far: f32 = 100.0,
};

pub const FreeEntry = struct {
    name: []const u8 = "",
    position: [3]f32 = .{ 0.0, 0.0, 0.0 },
    rotation: [3]f32 = .{ 0.0, 0.0, 0.0 },
    fov_deg: f32 = 60.0,
    near: f32 = 0.1,
    far: f32 = 100.0,
    speed: f32 = 6.0,
    angular_sensitivity: f32 = 0.25,
};

pub const FollowEntry = struct {
    name: []const u8 = "",
    position: [3]f32 = .{ 0.0, 0.0, 0.0 },
    target_position: [3]f32 = .{ 0.0, 0.0, 0.0 },
    radius: f32 = 5.0,
    height_offset: f32 = 2.0,
    rotation_offset_deg: f32 = 0.0,
    fov_deg: f32 = 60.0,
    near: f32 = 0.1,
    far: f32 = 100.0,
    lerp_speed: f32 = 8.0,
};

pub const TargetEntry = struct {
    name: []const u8 = "",
    position: [3]f32 = .{ 0.0, 0.0, 0.0 },
    target: [3]f32 = .{ 0.0, 0.0, 0.0 },
    up: [3]f32 = .{ 0.0, 1.0, 0.0 },
    fov_deg: f32 = 60.0,
    near: f32 = 0.1,
    far: f32 = 100.0,
    smoothing: f32 = 8.0,
};

pub const FlyEntry = struct {
    name: []const u8 = "",
    position: [3]f32 = .{ 0.0, 0.0, 0.0 },
    rotation: [3]f32 = .{ 0.0, 0.0, 0.0 },
    fov_deg: f32 = 60.0,
    near: f32 = 0.1,
    far: f32 = 100.0,
    speed: f32 = 6.0,
    boost_multiplier: f32 = 4.0,
    angular_sensitivity: f32 = 0.25,
    roll_speed_deg: f32 = 90.0,
};

/// Camera snapshot, one variant per Camera union type (follow target_mesh
/// link is dropped, target_position is kept; target desired goals are
/// transient and not stored).
pub const CameraEntry = union(enum) {
    none,
    arc_rotate: ArcRotateEntry,
    free: FreeEntry,
    follow: FollowEntry,
    target: TargetEntry,
    fly: FlyEntry,

    pub fn deinit(self: *CameraEntry, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .none => {},
            .arc_rotate => |*c| if (c.name.len > 0) allocator.free(c.name),
            .free => |*c| if (c.name.len > 0) allocator.free(c.name),
            .follow => |*c| if (c.name.len > 0) allocator.free(c.name),
            .target => |*c| if (c.name.len > 0) allocator.free(c.name),
            .fly => |*c| if (c.name.len > 0) allocator.free(c.name),
        }
    }
};

pub const RenderEntry = struct {
    skybox_enabled: bool = false,
    skybox_exposure: f32 = 1.0,
    shadows_enabled: bool = true,
    shadow_softness: f32 = 1.5,
    ibl_intensity: f32 = 1.0,
};

pub const GameProperty = struct {
    key: []const u8 = "",
    value: []const u8 = "",

    pub fn deinit(self: *GameProperty, allocator: std.mem.Allocator) void {
        if (self.key.len > 0) allocator.free(self.key);
        if (self.value.len > 0) allocator.free(self.value);
    }
};

pub const SceneState = struct {
    meshes: []MeshEntry = &.{},
    hemi: HemiEntry = .{},
    directional: ?DirectionalEntry = null,
    point_lights: []PointEntry = &.{},
    spot_lights: []SpotEntry = &.{},
    camera: CameraEntry = .none,
    render: RenderEntry = .{},
    postprocess: PostProcessOptions = .{},
    game_properties: []GameProperty = &.{},

    pub fn getGameProperty(self: *const SceneState, key: []const u8) ?[]const u8 {
        for (self.game_properties) |p| {
            if (std.mem.eql(u8, p.key, key)) return p.value;
        }
        return null;
    }

    pub fn setGameProperty(self: *SceneState, allocator: std.mem.Allocator, key: []const u8, value: []const u8) !void {
        for (self.game_properties) |*p| {
            if (std.mem.eql(u8, p.key, key)) {
                if (p.value.len > 0) allocator.free(p.value);
                p.value = try allocator.dupe(u8, value);
                return;
            }
        }
        const old_len = self.game_properties.len;
        const new_slice = try allocator.alloc(GameProperty, old_len + 1);
        if (old_len > 0) {
            @memcpy(new_slice[0..old_len], self.game_properties);
            allocator.free(self.game_properties);
        }
        new_slice[old_len] = .{
            .key = try allocator.dupe(u8, key),
            .value = try allocator.dupe(u8, value),
        };
        self.game_properties = new_slice;
    }

    pub fn deinit(self: *SceneState, allocator: std.mem.Allocator) void {
        for (self.meshes) |*m| m.deinit(allocator);
        if (self.meshes.len > 0) allocator.free(self.meshes);
        self.hemi.deinit(allocator);
        if (self.directional) |*d| d.deinit(allocator);
        for (self.point_lights) |*p| p.deinit(allocator);
        if (self.point_lights.len > 0) allocator.free(self.point_lights);
        for (self.spot_lights) |*s| s.deinit(allocator);
        if (self.spot_lights.len > 0) allocator.free(self.spot_lights);
        self.camera.deinit(allocator);
        for (self.game_properties) |*gp| gp.deinit(allocator);
        if (self.game_properties.len > 0) allocator.free(self.game_properties);
        self.* = .{};
    }
};

pub fn alphaModeToU8(mode: AlphaMode) u8 {
    return switch (mode) {
        .@"opaque" => 0,
        .blend => 1,
        .cutout => 2,
    };
}

pub fn alphaModeFromU8(v: u8) AlphaMode {
    return switch (v) {
        0 => .@"opaque",
        1 => .blend,
        2 => .cutout,
        else => .@"opaque",
    };
}

pub fn findMesh(scene: *Scene, name: []const u8) ?*Mesh {
    for (scene.meshes.items) |m| {
        if (std.mem.eql(u8, m.name, name)) return m;
    }
    return null;
}

pub fn findMeshByIdOrName(scene: *Scene, id: u64, name: []const u8) ?*Mesh {
    if (id != 0) {
        for (scene.meshes.items) |m| {
            if (m.id == id) return m;
        }
    }
    return findMesh(scene, name);
}

/// Applies one snapshot material to a live mesh. Mutates the existing
/// material in place when the kind matches (note: shared materials change
/// for every mesh using them); otherwise allocates a fresh scene material
/// (silently keeps the old one on OOM).
pub fn restoreMeshMaterial(scene: *Scene, mesh: *Mesh, src: *const MaterialEntry) void {
    switch (src.*) {
        .standard => |*s| {
            // Legacy standard format restores into PBR matte equivalent
            const pm: *PBRMaterial = blk: {
                if (mesh.material) |m| {
                    if (m == .pbr) break :blk m.pbr;
                }
                const mat = scene.createPBRMaterial(mesh.name) catch return;
                break :blk mat;
            };
            pm.albedo_color = Color3.new(s.diffuse[0], s.diffuse[1], s.diffuse[2]);
            pm.metallic = 0.0;
            pm.roughness = 0.5;
            pm.emissive_color = Color3.new(0.0, 0.0, 0.0);
            pm.alpha = s.alpha;
            pm.alpha_mode = alphaModeFromU8(s.alpha_mode);
            pm.alpha_cutoff = s.alpha_cutoff;
            pm.double_sided = s.double_sided;
            mesh.material = .{ .pbr = pm };
        },
        .pbr => |*p| {
            const pm: *PBRMaterial = blk: {
                if (mesh.material) |m| {
                    if (m == .pbr) break :blk m.pbr;
                }
                const mat = scene.createPBRMaterial(mesh.name) catch return;
                break :blk mat;
            };
            pm.albedo_color = Color3.new(p.albedo[0], p.albedo[1], p.albedo[2]);
            pm.metallic = p.metallic;
            pm.roughness = p.roughness;
            pm.emissive_color = Color3.new(p.emissive[0], p.emissive[1], p.emissive[2]);
            pm.alpha = p.alpha;
            pm.alpha_mode = alphaModeFromU8(p.alpha_mode);
            pm.alpha_cutoff = p.alpha_cutoff;
            pm.double_sided = p.double_sided;
            mesh.material = .{ .pbr = pm };
        },
    }
}
