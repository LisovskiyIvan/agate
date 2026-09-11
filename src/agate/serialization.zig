//! Compact binary snapshot of a scene (".babylon"-style, but binary).
//!
//! Only what is listed below is captured; everything else is intentionally
//! left out (see NOT SERIALIZED). All strings/slices in SceneState are owned
//! (allocator.dupe) and released by SceneState.deinit.
//!
//! Binary layout (all integers little-endian, f32 as IEEE754 LE bits,
//! bool as u8 0/1, string as u32 length + raw bytes, no padding):
//!   magic[4] = "AGSC", version u32,
//!   mesh_count u32, meshes[],
//!     per mesh: name str, position[3]f32, rotation[3]f32 (euler deg),
//!       scaling[3]f32, flags u8 (bit0 visible, bit1 cast, bit2 receive),
//!       material_kind u8 (0 standard, 1 pbr),
//!       standard: diffuse[3]f32, alpha f32, alpha_mode u8 (0 opaque, 1 blend,
//!         2 cutout), alpha_cutoff f32, double_sided u8,
//!       pbr: albedo[3]f32, metallic f32, roughness f32, emissive[3]f32,
//!         alpha f32, alpha_mode u8 (0 opaque, 1 blend, 2 cutout),
//!         alpha_cutoff f32, double_sided u8,
//!   hemi: name str, direction[3]f32, diffuse[3]f32, ground[3]f32,
//!     intensity f32,
//!   directional_present u8, [direction[3]f32, diffuse[3]f32, intensity f32],
//!   point_count u32, per point: name str, position[3]f32, diffuse[3]f32,
//!     intensity f32, range f32,
//!   spot_count u32, per spot: name str, position[3]f32, direction[3]f32,
//!     diffuse[3]f32, intensity f32, range f32, inner_deg f32, outer_deg f32,
//!   camera_kind u8 (0 none, 1 arc_rotate, 2 free, 3 follow, 4 target, 5 fly),
//!     arc_rotate: name str, alpha f32, beta f32, radius f32, target[3]f32,
//!       fov f32, near f32, far f32,
//!     free: name str, position[3]f32, rotation[3]f32 (euler deg), fov f32,
//!       near f32, far f32, speed f32, angular_sensitivity f32,
//!     follow: name str, position[3]f32, target_position[3]f32, radius f32,
//!       height_offset f32, rotation_offset_deg f32, fov f32, near f32,
//!       far f32, lerp_speed f32,
//!     target: name str, position[3]f32, target[3]f32, up[3]f32, fov f32,
//!       near f32, far f32, smoothing f32,
//!     fly: name str, position[3]f32, rotation[3]f32 (euler deg), fov f32,
//!       near f32, far f32, speed f32, boost_multiplier f32,
//!       angular_sensitivity f32, roll_speed_deg f32,
//!   (v2 layout: camera kinds appended as 4 target, 5 fly; material tails
//!   extended with alpha_cutoff + double_sided. v1 bytes are NOT readable:
//!   any version != 2 reports UnsupportedVersion.)
//!   render: skybox_enabled u8, skybox_exposure f32, shadows_enabled u8,
//!     shadow_softness f32, ibl_intensity f32,
//!   postprocess (PostProcessConfig field order): enabled u8, exposure f32,
//!     tonemapping u32, bloom_enabled u8, bloom_threshold f32,
//!     bloom_intensity f32, bloom_radius f32, vignette_enabled u8,
//!     vignette_intensity f32, vignette_radius f32, saturation f32,
//!     contrast f32, chromatic_aberration f32, fxaa_enabled u8, fog_enabled u8,
//!     fog_density f32, fog_height_falloff f32, fog_start_distance f32,
//!     fog_color[3]f32, fog_sun_scattering f32, ssr_enabled u8,
//!     ssr_intensity f32, ssr_max_distance f32, ssr_thickness f32,
//!     sharpen_enabled u8, sharpen_amount f32, grain_enabled u8,
//!     grain_intensity f32, temperature f32, tint f32.
//!
//! NOT SERIALIZED (by design): geometry (vertices/indices), textures and
//! cube maps (skybox/IBL contents), skeletons/animations, morph targets,
//! physics bodies, particles, UI, instanced meshes, parent links, bone
//! attachments, follow-camera target link (target_position is kept),
//! material sharing topology (values are per-mesh), SSAO config, shadow
//! tuning beyond softness, pipeline/GPU handles. Meshes are matched by name
//! on restore; geometry is referenced, never stored.

const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Color3 = math.Color3;
const SceneModule = @import("scene.zig");
const Scene = SceneModule.Scene;
const PostProcessConfig = @import("postprocess.zig").PostProcessConfig;
const TonemappingType = @import("postprocess.zig").TonemappingType;
const CameraModule = @import("camera.zig");
const Camera = CameraModule.Camera;
const TargetCamera = CameraModule.TargetCamera;
const FlyCamera = CameraModule.FlyCamera;
const Lights = @import("lights.zig");
const HemisphericLight = Lights.HemisphericLight;
const MaterialModule = @import("material.zig");
const StandardMaterial = MaterialModule.StandardMaterial;
const PBRMaterial = MaterialModule.PBRMaterial;
const AlphaMode = MaterialModule.AlphaMode;
const Mesh = @import("mesh.zig").Mesh;

pub const MAGIC: [4]u8 = .{ 'A', 'G', 'S', 'C' };
/// v2: adds target/fly cameras (kinds 4/5) and material alpha cutout mode
/// (alpha_mode 2) with alpha_cutoff + double_sided tails. v1 is rejected.
pub const VERSION: u32 = 2;

/// Hard caps for untrusted input. Counts above max_entries and strings above
/// max_string_bytes report TooLarge instead of driving wild allocations.
pub const MAX_ENTRIES: u32 = 1_000_000;
pub const MAX_STRING_BYTES: u32 = 8 * 1024 * 1024;
pub const MAX_FILE_BYTES: u64 = 256 * 1024 * 1024;

pub const DecodeError = error{
    BadMagic,
    UnsupportedVersion,
    Truncated,
    TooLarge,
};

// ---------------------------------------------------------------------------
// Snapshot types (all strings/slices owned, see deinit)
// ---------------------------------------------------------------------------

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
    name: []const u8 = "",
    position: [3]f32 = .{ 0.0, 0.0, 0.0 },
    rotation: [3]f32 = .{ 0.0, 0.0, 0.0 },
    scaling: [3]f32 = .{ 1.0, 1.0, 1.0 },
    is_visible: bool = true,
    cast_shadows: bool = true,
    receive_shadows: bool = true,
    material: MaterialEntry = .{ .standard = .{} },

    pub fn deinit(self: *MeshEntry, allocator: std.mem.Allocator) void {
        if (self.name.len > 0) allocator.free(self.name);
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

pub const SceneState = struct {
    meshes: []MeshEntry = &.{},
    hemi: HemiEntry = .{},
    directional: ?DirectionalEntry = null,
    point_lights: []PointEntry = &.{},
    spot_lights: []SpotEntry = &.{},
    camera: CameraEntry = .none,
    render: RenderEntry = .{},
    postprocess: PostProcessConfig = .{},

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
        self.* = .{};
    }
};

// ---------------------------------------------------------------------------
// capture / restore
// ---------------------------------------------------------------------------

fn alphaModeToU8(mode: AlphaMode) u8 {
    return switch (mode) {
        .@"opaque" => 0,
        .blend => 1,
        .cutout => 2,
    };
}

fn alphaModeFromU8(v: u8) AlphaMode {
    return switch (v) {
        0 => .@"opaque",
        1 => .blend,
        2 => .cutout,
        else => .@"opaque",
    };
}

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
        const material: MaterialEntry = if (mesh.material) |mat| switch (mat) {
            .standard => |s| .{ .standard = .{
                .diffuse = .{ s.diffuse_color.r, s.diffuse_color.g, s.diffuse_color.b },
                .alpha = s.alpha,
                .alpha_mode = alphaModeToU8(s.alpha_mode),
                .alpha_cutoff = s.alpha_cutoff,
                .double_sided = s.double_sided,
            } },
            .pbr => |p| .{ .pbr = .{
                .albedo = .{ p.albedo_color.r, p.albedo_color.g, p.albedo_color.b },
                .metallic = p.metallic,
                .roughness = p.roughness,
                .emissive = .{ p.emissive_color.r, p.emissive_color.g, p.emissive_color.b },
                .alpha = p.alpha,
                .alpha_mode = alphaModeToU8(p.alpha_mode),
                .alpha_cutoff = p.alpha_cutoff,
                .double_sided = p.double_sided,
            } },
        } else .{ .standard = .{} };
        try meshes.append(allocator, .{
            .name = name,
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
        .name = try allocator.dupe(u8, scene.light.name),
        .direction = .{ scene.light.direction.x, scene.light.direction.y, scene.light.direction.z },
        .diffuse = .{ scene.light.diffuse.r, scene.light.diffuse.g, scene.light.diffuse.b },
        .ground = .{ scene.light.ground_color.r, scene.light.ground_color.g, scene.light.ground_color.b },
        .intensity = scene.light.intensity,
    };

    if (scene.directional_light) |dl| {
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
    for (scene.point_lights.items) |pl| {
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
    for (scene.spot_lights.items) |sl| {
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
        .skybox_enabled = scene.skybox_enabled,
        .skybox_exposure = scene.skybox_exposure,
        .shadows_enabled = scene.enable_shadows,
        .shadow_softness = scene.shadow_softness,
        .ibl_intensity = scene.ibl_intensity,
    };
    state.postprocess = scene.post_process;

    return state;
}

fn findMesh(scene: *Scene, name: []const u8) ?*Mesh {
    for (scene.meshes.items) |m| {
        if (std.mem.eql(u8, m.name, name)) return m;
    }
    return null;
}

/// Applies one snapshot material to a live mesh. Mutates the existing
/// material in place when the kind matches (note: shared materials change
/// for every mesh using them); otherwise allocates a fresh scene material
/// (silently keeps the old one on OOM).
fn restoreMeshMaterial(scene: *Scene, mesh: *Mesh, src: *const MaterialEntry) void {
    switch (src.*) {
        .standard => |*s| {
            const sm: *StandardMaterial = blk: {
                if (mesh.material) |m| {
                    if (m == .standard) break :blk m.standard;
                }
                const owned: ?[]u8 = scene.allocator.dupe(u8, mesh.name) catch null;
                const mat = scene.createStandardMaterial(owned orelse mesh.name) catch {
                    if (owned) |o| scene.allocator.free(o);
                    return;
                };
                break :blk mat;
            };
            sm.diffuse_color = Color3.new(s.diffuse[0], s.diffuse[1], s.diffuse[2]);
            sm.alpha = s.alpha;
            sm.alpha_mode = alphaModeFromU8(s.alpha_mode);
            sm.alpha_cutoff = s.alpha_cutoff;
            sm.double_sided = s.double_sided;
            mesh.material = .{ .standard = sm };
        },
        .pbr => |*p| {
            const pm: *PBRMaterial = blk: {
                if (mesh.material) |m| {
                    if (m == .pbr) break :blk m.pbr;
                }
                const owned: ?[]u8 = scene.allocator.dupe(u8, mesh.name) catch null;
                const mat = scene.createPBRMaterial(owned orelse mesh.name) catch {
                    if (owned) |o| scene.allocator.free(o);
                    return;
                };
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

/// Applies a snapshot to a live scene. Meshes match by name, unknown names
/// are ignored. Point/spot lights are destroyed and recreated; the
/// directional light is replaced via createDirectionalLight (or removed when
/// the snapshot has none; note the scene API destroys the old sun before
/// allocating, so OOM there loses the sun). Follow-camera target_mesh resets
/// to null (target_position is kept). Never fails and never crashes on count
/// mismatches; OOM during point/spot/material recreation silently keeps the
/// previous object.
pub fn restore(scene: *Scene, state: *const SceneState) void {
    for (state.meshes) |*entry| {
        const mesh = findMesh(scene, entry.name) orelse continue;
        mesh.position = Vec3.new(entry.position[0], entry.position[1], entry.position[2]);
        mesh.rotation = Vec3.new(entry.rotation[0], entry.rotation[1], entry.rotation[2]);
        mesh.scaling = Vec3.new(entry.scaling[0], entry.scaling[1], entry.scaling[2]);
        mesh.is_visible = entry.is_visible;
        mesh.cast_shadows = entry.cast_shadows;
        mesh.receive_shadows = entry.receive_shadows;
        restoreMeshMaterial(scene, mesh, &entry.material);
    }

    {
        const owned: ?[]u8 = scene.allocator.dupe(u8, state.hemi.name) catch null;
        scene.light = HemisphericLight.init(owned orelse state.hemi.name, .{
            .direction = Vec3.new(state.hemi.direction[0], state.hemi.direction[1], state.hemi.direction[2]),
            .diffuse = Color3.new(state.hemi.diffuse[0], state.hemi.diffuse[1], state.hemi.diffuse[2]),
            .ground_color = Color3.new(state.hemi.ground[0], state.hemi.ground[1], state.hemi.ground[2]),
            .intensity = state.hemi.intensity,
        });
    }

    if (state.directional) |*d| {
        // createDirectionalLight destroys the previous sun internally.
        const owned: ?[]u8 = scene.allocator.dupe(u8, d.name) catch null;
        _ = scene.createDirectionalLight(owned orelse d.name, .{
            .direction = Vec3.new(d.direction[0], d.direction[1], d.direction[2]),
            .diffuse = Color3.new(d.diffuse[0], d.diffuse[1], d.diffuse[2]),
            .intensity = d.intensity,
        }) catch {
            if (owned) |o| scene.allocator.free(o);
        };
    } else if (scene.directional_light) |old| {
        scene.allocator.destroy(old);
        scene.directional_light = null;
    }

    for (scene.point_lights.items) |pl| scene.allocator.destroy(pl);
    scene.point_lights.clearRetainingCapacity();
    for (state.point_lights) |*p| {
        const owned: ?[]u8 = scene.allocator.dupe(u8, p.name) catch null;
        _ = scene.createPointLight(owned orelse p.name, .{
            .position = Vec3.new(p.position[0], p.position[1], p.position[2]),
            .color = Color3.new(p.diffuse[0], p.diffuse[1], p.diffuse[2]),
            .intensity = p.intensity,
            .range = p.range,
        }) catch {
            if (owned) |o| scene.allocator.free(o);
        };
    }

    for (scene.spot_lights.items) |sl| scene.allocator.destroy(sl);
    scene.spot_lights.clearRetainingCapacity();
    for (state.spot_lights) |*s| {
        const owned: ?[]u8 = scene.allocator.dupe(u8, s.name) catch null;
        _ = scene.createSpotLight(owned orelse s.name, .{
            .position = Vec3.new(s.position[0], s.position[1], s.position[2]),
            .direction = Vec3.new(s.direction[0], s.direction[1], s.direction[2]),
            .color = Color3.new(s.diffuse[0], s.diffuse[1], s.diffuse[2]),
            .intensity = s.intensity,
            .range = s.range,
            .inner_angle_deg = s.inner_deg,
            .outer_angle_deg = s.outer_deg,
        }) catch {
            if (owned) |o| scene.allocator.free(o);
        };
    }

    switch (state.camera) {
        .none => scene.active_camera = null,
        .arc_rotate => |*c| {
            const owned: ?[]u8 = scene.allocator.dupe(u8, c.name) catch null;
            scene.active_camera = .{ .arc_rotate = .{
                .name = owned orelse c.name,
                .alpha = c.alpha,
                .beta = c.beta,
                .radius = c.radius,
                .target = Vec3.new(c.target[0], c.target[1], c.target[2]),
                .fov_deg = c.fov_deg,
                .near = c.near,
                .far = c.far,
            } };
        },
        .free => |*c| {
            const owned: ?[]u8 = scene.allocator.dupe(u8, c.name) catch null;
            scene.active_camera = .{ .free = .{
                .name = owned orelse c.name,
                .position = Vec3.new(c.position[0], c.position[1], c.position[2]),
                .rotation = Vec3.new(c.rotation[0], c.rotation[1], c.rotation[2]),
                .fov_deg = c.fov_deg,
                .near = c.near,
                .far = c.far,
                .speed = c.speed,
                .angular_sensitivity = c.angular_sensitivity,
            } };
        },
        .follow => |*c| {
            const owned: ?[]u8 = scene.allocator.dupe(u8, c.name) catch null;
            scene.active_camera = .{ .follow = .{
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
            } };
        },
        .target => |*c| {
            const owned: ?[]u8 = scene.allocator.dupe(u8, c.name) catch null;
            scene.active_camera = .{ .target = TargetCamera.init(owned orelse c.name, .{
                .position = Vec3.new(c.position[0], c.position[1], c.position[2]),
                .target = Vec3.new(c.target[0], c.target[1], c.target[2]),
                .up = Vec3.new(c.up[0], c.up[1], c.up[2]),
                .fov_deg = c.fov_deg,
                .near = c.near,
                .far = c.far,
                .smoothing = c.smoothing,
            }) };
        },
        .fly => |*c| {
            const owned: ?[]u8 = scene.allocator.dupe(u8, c.name) catch null;
            scene.active_camera = .{ .fly = FlyCamera.init(owned orelse c.name, .{
                .position = Vec3.new(c.position[0], c.position[1], c.position[2]),
                .rotation = Vec3.new(c.rotation[0], c.rotation[1], c.rotation[2]),
                .fov_deg = c.fov_deg,
                .near = c.near,
                .far = c.far,
                .speed = c.speed,
                .boost_multiplier = c.boost_multiplier,
                .angular_sensitivity = c.angular_sensitivity,
                .roll_speed_deg = c.roll_speed_deg,
            }) };
        },
    }

    scene.skybox_enabled = state.render.skybox_enabled;
    scene.skybox_exposure = state.render.skybox_exposure;
    scene.enable_shadows = state.render.shadows_enabled;
    scene.shadow_softness = state.render.shadow_softness;
    scene.ibl_intensity = state.render.ibl_intensity;
    scene.post_process = state.postprocess;
}

// ---------------------------------------------------------------------------
// Binary codec
// ---------------------------------------------------------------------------

const Writer = struct {
    alloc: std.mem.Allocator,
    buf: std.ArrayListUnmanaged(u8) = .empty,

    fn bytes(self: *Writer, data: []const u8) !void {
        try self.buf.appendSlice(self.alloc, data);
    }

    fn byte(self: *Writer, v: u8) !void {
        try self.buf.append(self.alloc, v);
    }

    fn u32le(self: *Writer, v: u32) !void {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, v, .little);
        try self.bytes(&b);
    }

    fn f32le(self: *Writer, v: f32) !void {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, @as(u32, @bitCast(v)), .little);
        try self.bytes(&b);
    }

    fn bool8(self: *Writer, v: bool) !void {
        try self.byte(if (v) 1 else 0);
    }

    fn vec3(self: *Writer, v: [3]f32) !void {
        try self.f32le(v[0]);
        try self.f32le(v[1]);
        try self.f32le(v[2]);
    }

    fn str(self: *Writer, s: []const u8) !void {
        const len = std.math.cast(u32, s.len) orelse return error.TooLarge;
        try self.u32le(len);
        try self.bytes(s);
    }
};

const Reader = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn readU8(self: *Reader) DecodeError!u8 {
        if (self.pos >= self.bytes.len) return error.Truncated;
        const v = self.bytes[self.pos];
        self.pos += 1;
        return v;
    }

    fn readU32(self: *Reader) DecodeError!u32 {
        const raw = try self.readRaw(4);
        return std.mem.readInt(u32, raw[0..4], .little);
    }

    fn readF32(self: *Reader) DecodeError!f32 {
        const raw = try self.readRaw(4);
        return @as(f32, @bitCast(std.mem.readInt(u32, raw[0..4], .little)));
    }

    fn readBool(self: *Reader) DecodeError!bool {
        return switch (try self.readU8()) {
            0 => false,
            1 => true,
            else => error.Truncated,
        };
    }

    fn readRaw(self: *Reader, n: u32) DecodeError![]const u8 {
        const len = std.math.cast(usize, n) orelse return error.TooLarge;
        const end = std.math.add(usize, self.pos, len) catch return error.TooLarge;
        if (end > self.bytes.len) return error.Truncated;
        const out = self.bytes[self.pos..end];
        self.pos = end;
        return out;
    }

    /// Validated element count for a variable-length list.
    fn readCount(self: *Reader) DecodeError!u32 {
        const n = try self.readU32();
        if (n > MAX_ENTRIES) return error.TooLarge;
        return n;
    }

    fn readString(self: *Reader, allocator: std.mem.Allocator) (DecodeError || std.mem.Allocator.Error)![]u8 {
        const n = try self.readU32();
        if (n > MAX_STRING_BYTES) return error.TooLarge;
        const raw = try self.readRaw(n);
        return allocator.dupe(u8, raw);
    }

    fn readVec3(self: *Reader) DecodeError![3]f32 {
        return .{ try self.readF32(), try self.readF32(), try self.readF32() };
    }
};

fn writePostProcess(w: *Writer, pp: *const PostProcessConfig) !void {
    try w.bool8(pp.enabled);
    try w.f32le(pp.exposure);
    try w.u32le(@intFromEnum(pp.tonemapping));
    try w.bool8(pp.bloom_enabled);
    try w.f32le(pp.bloom_threshold);
    try w.f32le(pp.bloom_intensity);
    try w.f32le(pp.bloom_radius);
    try w.bool8(pp.vignette_enabled);
    try w.f32le(pp.vignette_intensity);
    try w.f32le(pp.vignette_radius);
    try w.f32le(pp.saturation);
    try w.f32le(pp.contrast);
    try w.f32le(pp.chromatic_aberration);
    try w.bool8(pp.fxaa_enabled);
    try w.bool8(pp.fog_enabled);
    try w.f32le(pp.fog_density);
    try w.f32le(pp.fog_height_falloff);
    try w.f32le(pp.fog_start_distance);
    try w.vec3(pp.fog_color);
    try w.f32le(pp.fog_sun_scattering);
    try w.bool8(pp.ssr_enabled);
    try w.f32le(pp.ssr_intensity);
    try w.f32le(pp.ssr_max_distance);
    try w.f32le(pp.ssr_thickness);
    try w.bool8(pp.sharpen_enabled);
    try w.f32le(pp.sharpen_amount);
    try w.bool8(pp.grain_enabled);
    try w.f32le(pp.grain_intensity);
    try w.f32le(pp.temperature);
    try w.f32le(pp.tint);
}

fn readPostProcess(r: *Reader) DecodeError!PostProcessConfig {
    var pp = PostProcessConfig{};
    pp.enabled = try r.readBool();
    pp.exposure = try r.readF32();
    pp.tonemapping = switch (try r.readU32()) {
        0 => .none,
        1 => .aces,
        2 => .reinhard,
        else => return error.Truncated,
    };
    pp.bloom_enabled = try r.readBool();
    pp.bloom_threshold = try r.readF32();
    pp.bloom_intensity = try r.readF32();
    pp.bloom_radius = try r.readF32();
    pp.vignette_enabled = try r.readBool();
    pp.vignette_intensity = try r.readF32();
    pp.vignette_radius = try r.readF32();
    pp.saturation = try r.readF32();
    pp.contrast = try r.readF32();
    pp.chromatic_aberration = try r.readF32();
    pp.fxaa_enabled = try r.readBool();
    pp.fog_enabled = try r.readBool();
    pp.fog_density = try r.readF32();
    pp.fog_height_falloff = try r.readF32();
    pp.fog_start_distance = try r.readF32();
    pp.fog_color = try r.readVec3();
    pp.fog_sun_scattering = try r.readF32();
    pp.ssr_enabled = try r.readBool();
    pp.ssr_intensity = try r.readF32();
    pp.ssr_max_distance = try r.readF32();
    pp.ssr_thickness = try r.readF32();
    pp.sharpen_enabled = try r.readBool();
    pp.sharpen_amount = try r.readF32();
    pp.grain_enabled = try r.readBool();
    pp.grain_intensity = try r.readF32();
    pp.temperature = try r.readF32();
    pp.tint = try r.readF32();
    return pp;
}

/// Serializes a snapshot into a freshly allocated byte buffer (little-endian
/// layout documented at the top of this file). Fails with TooLarge when a
/// slice/string length does not fit into u32.
pub fn serializeAlloc(allocator: std.mem.Allocator, state: *const SceneState) ![]u8 {
    var w = Writer{ .alloc = allocator };
    errdefer w.buf.deinit(allocator);

    try w.bytes(MAGIC[0..]);
    try w.u32le(VERSION);

    try w.u32le(std.math.cast(u32, state.meshes.len) orelse return error.TooLarge);
    for (state.meshes) |*m| {
        try w.str(m.name);
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

    return w.buf.toOwnedSlice(allocator);
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
        entry.name = try r.readString(allocator);
        errdefer entry.deinit(allocator);
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
    if (version != VERSION) return error.UnsupportedVersion;

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

    return state;
}

// ---------------------------------------------------------------------------
// File helpers (std.Io, same pattern as AudioClip.fromWavFile)
// ---------------------------------------------------------------------------

/// Writes serializeAlloc output to path (created/truncated).
pub fn saveFile(allocator: std.mem.Allocator, state: *const SceneState, path: []const u8) !void {
    const bytes = try serializeAlloc(allocator, state);
    defer allocator.free(bytes);
    const io = std.Io.Threaded.global_single_threaded.io();
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
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

// ---------------------------------------------------------------------------
// GPU-free tests
// ---------------------------------------------------------------------------

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
            .name = try dupeStr(allocator, "box"),
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
            .name = try dupeStr(allocator, "sphere"),
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

    return s;
}

fn expectStatesEqual(a: *const SceneState, b: *const SceneState) !void {
    try std.testing.expectEqual(a.meshes.len, b.meshes.len);
    for (a.meshes, b.meshes) |*am, *bm| {
        try std.testing.expectEqualStrings(am.name, bm.name);
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

    // Current VERSION must parse; anything else (incl. legacy v1) is rejected.
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
    std.mem.writeInt(u32, &vb, 64, .little);
    try buf.appendSlice(alloc, &vb);
    try std.testing.expectError(error.Truncated, deserializeAlloc(alloc, buf.items));
}

test "capture maps null material to standard default" {
    const alloc = std.testing.allocator;
    var scene: Scene = std.mem.zeroes(Scene);
    scene.allocator = alloc;

    var mesh = std.mem.zeroes(Mesh);
    mesh.name = "plain";
    mesh.position = Vec3.new(1.0, 2.0, 3.0);
    // material stays null.
    try scene.meshes.append(alloc, &mesh);
    defer scene.meshes.deinit(alloc);

    var state = try capture(alloc, &scene);
    defer state.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 1), state.meshes.len);
    try std.testing.expectEqualStrings("plain", state.meshes[0].name);
    try std.testing.expectEqual([3]f32{ 1.0, 2.0, 3.0 }, state.meshes[0].position);
    try std.testing.expect(state.meshes[0].material == .standard);
    try std.testing.expectEqual([3]f32{ 1.0, 1.0, 1.0 }, state.meshes[0].material.standard.diffuse);
    try std.testing.expectEqual(@as(f32, 1.0), state.meshes[0].material.standard.alpha);
    try std.testing.expect(state.camera == .none);
    try std.testing.expect(state.directional == null);
}

test "restore applies by name and ignores missing" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var scene: Scene = std.mem.zeroes(Scene);
    scene.allocator = alloc;

    var std_mat = StandardMaterial.init("shared");
    var mesh = std.mem.zeroes(Mesh);
    mesh.name = "box";
    mesh.material = .{ .standard = &std_mat };
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
        .material = .{ .standard = .{ .diffuse = .{ 0.9, 0.1, 0.1 }, .alpha = 0.5, .alpha_mode = 1 } },
    };
    entries[1] = .{
        .name = try alloc.dupe(u8, "ghost-missing"),
        .position = .{ 99.0, 99.0, 99.0 },
        .material = .{ .standard = .{} },
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
    try std.testing.expectEqual(@as(f32, 0.5), std_mat.alpha);
    try std.testing.expect(std_mat.alpha_mode == .blend);
    // Missing mesh ignored: no crash, light recreated, camera applied.
    try std.testing.expectEqual(@as(usize, 1), scene.point_lights.items.len);
    try std.testing.expectEqualStrings("lamp", scene.point_lights.items[0].name);
    try std.testing.expectEqual(@as(f32, 2.0), scene.point_lights.items[0].intensity);
    try std.testing.expectEqual(@as(f32, 0.3), scene.light.intensity);
    try std.testing.expect(scene.active_camera != null);
    try std.testing.expect(scene.active_camera.? == .free);
    try std.testing.expectEqual(@as(f32, 70.0), scene.active_camera.?.free.fov_deg);
    try std.testing.expect(!scene.enable_shadows);
    // Camera union import is exercised (keeps the Camera symbol referenced).
    const _cam: ?Camera = scene.active_camera;
    try std.testing.expect(_cam != null);
}

fn roundTripState(alloc: std.mem.Allocator, original: *const SceneState) !SceneState {
    const bytes = try serializeAlloc(alloc, original);
    defer alloc.free(bytes);
    return deserializeAlloc(alloc, bytes);
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

    var scene: Scene = std.mem.zeroes(Scene);
    scene.allocator = alloc;
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

    var scene: Scene = std.mem.zeroes(Scene);
    scene.allocator = alloc;

    var std_mat = StandardMaterial.init("cut");
    std_mat.alpha_mode = .cutout;
    std_mat.alpha_cutoff = 0.2;
    std_mat.double_sided = true;
    var mesh = std.mem.zeroes(Mesh);
    mesh.name = "fence";
    mesh.material = .{ .standard = &std_mat };
    try scene.meshes.append(alloc, &mesh);

    var state = try capture(alloc, &scene);
    try std.testing.expectEqual(@as(u8, 2), state.meshes[0].material.standard.alpha_mode);
    try std.testing.expectEqual(@as(f32, 0.2), state.meshes[0].material.standard.alpha_cutoff);
    try std.testing.expect(state.meshes[0].material.standard.double_sided);

    // Mutate live, restore must bring the snapshot values back.
    std_mat.alpha_cutoff = 0.9;
    std_mat.double_sided = false;
    std_mat.alpha_mode = .@"opaque";
    restore(&scene, &state);
    try std.testing.expect(std_mat.alpha_mode == .cutout);
    try std.testing.expectEqual(@as(f32, 0.2), std_mat.alpha_cutoff);
    try std.testing.expect(std_mat.double_sided);
}
