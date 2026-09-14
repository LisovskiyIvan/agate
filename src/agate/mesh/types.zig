const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const BoundingBox = math.BoundingBox;

const Mesh = @import("mesh.zig").Mesh;

pub const Vertex = extern struct {
    position: [3]f32,
    normal: [3]f32,
    color: [4]f32,
    uv: [2]f32,
    tangent: [4]f32 = .{ 1.0, 0.0, 0.0, 1.0 },
    joints: [4]f32 = .{ 0.0, 0.0, 0.0, 0.0 },
    weights: [4]f32 = .{ 1.0, 0.0, 0.0, 0.0 },
};

pub const SkinJointWeight = extern struct {
    joints: [4]f32 = .{ 0.0, 0.0, 0.0, 0.0 },
    weights: [4]f32 = .{ 1.0, 0.0, 0.0, 0.0 },
};

pub const CullingStrategy = enum {
    frustum,
    occlusion,
    always_render,
};

pub const LODLevel = struct {
    distance: f32,
    mesh: ?*Mesh,
};

/// Maximum morph targets (blend shapes) per mesh.
/// glTF files with more targets load only the first MAX_MORPH_TARGETS;
/// extras are dropped at import (documented loader behavior).
/// CPU mode blends with dirty tracking and rewrites the dynamic vertex
/// buffer; GPU mode blends in the vertex shader from the packed delta
/// texture (see mesh/morph_gpu.zig). The 8-target cap holds for both.
pub const MAX_MORPH_TARGETS: usize = 8;

/// Where morph deltas are blended. .cpu is the default and the historical
/// behavior (applyMorphs rewrites the vertex buffer); .gpu keeps a static
/// base-pose vertex buffer and lets standard/pbr/skinned_pbr shaders blend
/// deltas from an RGBA32F delta texture (opt-in, forward draw paths only).
/// Drawing a .gpu mesh without an uploaded delta texture panics
/// (morph_gpu.vsUniforms) — the engine never silently renders base pose.
pub const MorphMode = enum { cpu, gpu };

/// One glTF morph target: per-vertex deltas added to the base attributes,
/// scaled by the matching entry of Mesh.morph_weights.
/// All slices are owned (scene allocator) and freed in Mesh.deinit.
/// Any slice may be empty when the glTF target omits that attribute
/// (tangents are often absent); empty means "no delta".
pub const MorphTarget = struct {
    position_deltas: [][3]f32 = &.{},
    normal_deltas: [][3]f32 = &.{},
    /// Vec3 deltas applied to tangent xyz; tangent w is preserved.
    tangent_deltas: [][3]f32 = &.{},
};

pub const InstancedMesh = struct {
    name: []const u8,
    position: Vec3 = Vec3.zero,
    rotation: Vec3 = Vec3.zero, // Euler angles in degrees
    scaling: Vec3 = Vec3.one,
    is_visible: bool = true,
    cast_shadows: bool = true,
    receive_shadows: bool = true,
    culling_strategy: CullingStrategy = .frustum,
    layer_mask: u32 = 0xFFFFFFFF,
    source_mesh: *Mesh,

    cached_world_matrix: Mat4 = Mat4.identity,
    cached_bounding_box: BoundingBox = BoundingBox.zero,
    last_position: Vec3 = Vec3.new(std.math.nan(f32), 0, 0),
    last_rotation: Vec3 = Vec3.zero,
    last_scaling: Vec3 = Vec3.zero,
    dirty: bool = true,

    pub fn markDirty(self: *InstancedMesh) void {
        self.dirty = true;
    }

    pub fn computeWorldMatrix(self: InstancedMesh) Mat4 {
        const trs = Mat4.fromRotationTranslationScale(self.position, self.rotation, self.scaling);
        return Mat4.mul(trs, self.source_mesh.base_matrix);
    }

    pub fn updateCachedTransforms(self: *InstancedMesh) void {
        const moved = self.dirty or
            !self.position.eql(self.last_position) or
            !self.rotation.eql(self.last_rotation) or
            !self.scaling.eql(self.last_scaling);
        if (moved) {
            self.cached_world_matrix = self.computeWorldMatrix();
            self.cached_bounding_box = self.source_mesh.local_bounding_box.transform(self.cached_world_matrix);
            self.last_position = self.position;
            self.last_rotation = self.rotation;
            self.last_scaling = self.scaling;
            self.dirty = false;
        }
    }

    pub fn getWorldMatrix(self: *InstancedMesh) Mat4 {
        self.updateCachedTransforms();
        return self.cached_world_matrix;
    }

    pub fn getWorldBoundingBox(self: *InstancedMesh) BoundingBox {
        self.updateCachedTransforms();
        return self.cached_bounding_box;
    }
};

pub const BoneAttachment = struct {
    host_mesh: *Mesh,
    bone_index: usize,
    offset_matrix: Mat4 = Mat4.identity,
};

/// CPU-side geometry produced by pure builder helpers and loaders.
pub const GeometryData = struct {
    vertices: []Vertex,
    indices: []u32,
    bounds: BoundingBox,

    pub fn deinit(self: *GeometryData, allocator: std.mem.Allocator) void {
        allocator.free(self.vertices);
        allocator.free(self.indices);
    }
};
