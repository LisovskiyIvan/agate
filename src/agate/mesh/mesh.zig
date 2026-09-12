const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const BoundingBox = math.BoundingBox;

const types = @import("types.zig");
const Vertex = types.Vertex;
const CullingStrategy = types.CullingStrategy;
const MAX_MORPH_TARGETS = types.MAX_MORPH_TARGETS;
const MorphTarget = types.MorphTarget;
const InstancedMesh = types.InstancedMesh;
const BoneAttachment = types.BoneAttachment;
const GeometryData = types.GeometryData;
const LODLevel = types.LODLevel;

const Material = @import("../material.zig").Material;
const StandardMaterial = @import("../material.zig").StandardMaterial;
const PBRMaterial = @import("../material.zig").PBRMaterial;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;
const Scene = @import("../scene.zig").Scene;

pub const Mesh = struct {
    name: []const u8,
    owns_name: bool = false,
    position: Vec3 = Vec3.zero,
    rotation: Vec3 = Vec3.zero, // Euler angles in degrees
    scaling: Vec3 = Vec3.one,

    vertex_buffer: sg.Buffer,
    index_buffer: sg.Buffer,
    index_count: u32,
    index_type: sg.IndexType = .UINT16,
    material: ?Material = null,
    parent: ?*Mesh = null,
    skeleton: ?*Skeleton = null,
    attach_bone: ?BoneAttachment = null,
    base_matrix: Mat4 = Mat4.identity,

    // Culling, Shadows & Visibility
    is_visible: bool = true,
    cast_shadows: bool = true,
    receive_shadows: bool = true,
    culling_strategy: CullingStrategy = .frustum,
    local_bounding_box: BoundingBox = BoundingBox.zero,

    // Optional CPU-side geometry retained for physics collider creation
    // (convex hull / triangle mesh shapes). Owned by the scene allocator.
    cpu_positions: []Vec3 = &.{},
    cpu_indices: []u32 = &.{},

    // Morph targets (blend shapes), CPU-blended with dirty tracking.
    // morph_targets/morph_weights/morph_base/morph_staging are owned by the
    // scene allocator; Mesh.deinit frees them.
    morph_targets: []MorphTarget = &.{},
    morph_weights: []f32 = &.{},
    /// Base (unmorphed) vertices: the blend source. See retainMorphBase.
    morph_base: []Vertex = &.{},
    /// Current blended result; uploaded to vertex_buffer when dirty.
    /// Without a GPU buffer (unit tests, id == 0) this is the output.
    morph_staging: []Vertex = &.{},
    morph_dirty: bool = false,

    // Instancing support
    instances: std.ArrayListUnmanaged(*InstancedMesh) = .empty,
    instance_buffer: sg.Buffer = .{},
    instance_buffer_capacity: usize = 0,
    visible_instance_count: u32 = 0,
    // Per-frame transform cache (Scene.worldMatrixCached fills these once per render()).
    cached_matrix: Mat4 = Mat4.identity,
    cached_aabb: BoundingBox = BoundingBox.zero,
    cached_frame: u64 = std.math.maxInt(u64),
    // Instance buffer upload dedup: skip sg.updateBuffer when data unchanged.
    instance_hash: u64 = 0,
    instance_uploaded_count: usize = 0,

    // Level of Detail (LOD)
    lod_levels: std.ArrayListUnmanaged(LODLevel) = .empty,
    is_lod_child: bool = false,

    pub fn createInstance(self: *Mesh, scene: *Scene, name: []const u8) !*InstancedMesh {
        const inst = try scene.allocator.create(InstancedMesh);
        inst.* = .{
            .name = name,
            .source_mesh = self,
        };
        try self.instances.append(scene.allocator, inst);
        return inst;
    }

    pub fn setStandardMaterial(self: *Mesh, mat: *StandardMaterial) void {
        self.material = .{ .standard = mat };
    }

    pub fn setPBRMaterial(self: *Mesh, mat: *PBRMaterial) void {
        self.material = .{ .pbr = mat };
    }

    /// Attaches this mesh to a specific bone socket of host_mesh.
    pub fn attachToBone(self: *Mesh, host_mesh: *Mesh, bone_index: usize) void {
        self.attach_bone = .{
            .host_mesh = host_mesh,
            .bone_index = bone_index,
            .offset_matrix = Mat4.identity,
        };
    }

    /// Attaches this mesh to a bone identified by name on host_mesh.
    pub fn attachToBoneByName(self: *Mesh, host_mesh: *Mesh, bone_name: []const u8) !void {
        const skel = host_mesh.skeleton orelse return error.NoSkeletonOnMesh;
        const idx = skel.findBoneIndex(bone_name) orelse return error.BoneNotFound;
        self.attachToBone(host_mesh, idx);
    }

    /// Detaches this mesh from its bone socket.
    pub fn detachFromBone(self: *Mesh) void {
        self.attach_bone = null;
    }

    /// Adds an LOD level. Levels can be added in any order; they will be kept
    /// sorted by distance ascending. Setting lod_mesh to null means culling
    /// the mesh when distance >= distance.
    pub fn addLODLevel(self: *Mesh, allocator: std.mem.Allocator, distance: f32, lod_mesh: ?*Mesh) !void {
        if (lod_mesh) |lm| {
            lm.is_lod_child = true;
        }
        var insert_idx: usize = self.lod_levels.items.len;
        for (self.lod_levels.items, 0..) |lvl, i| {
            if (distance < lvl.distance) {
                insert_idx = i;
                break;
            }
        }
        try self.lod_levels.insert(allocator, insert_idx, .{
            .distance = distance,
            .mesh = lod_mesh,
        });
    }

    /// Selects the appropriate active LOD mesh for a given distance from camera.
    /// Returns self if distance is below the first LOD threshold,
    /// or the LOD mesh for the matched distance bracket,
    /// or null if the matched bracket specifies a culled mesh (null).
    pub fn getLOD(self: *const Mesh, distance: f32) ?*Mesh {
        if (self.lod_levels.items.len == 0) return @constCast(self);
        var active: ?*Mesh = @constCast(self);
        for (self.lod_levels.items) |lvl| {
            if (distance >= lvl.distance) {
                active = lvl.mesh;
            } else {
                break;
            }
        }
        return active;
    }

    /// Selects the active LOD mesh based on camera distance to this mesh's center.
    pub fn getLODForCamera(self: *const Mesh, camera_pos: Vec3) ?*Mesh {
        const center = if (self.cached_aabb.isValid()) self.cached_aabb.center() else self.position;
        const dist = center.distance(camera_pos);
        return self.getLOD(dist);
    }

    pub fn getWorldMatrix(self: Mesh) Mat4 {
        const trs = Mat4.fromRotationTranslationScale(self.position, self.rotation, self.scaling);
        const local = Mat4.mul(trs, self.base_matrix);
        if (self.attach_bone) |att| {
            if (att.host_mesh.skeleton) |skel| {
                const host_mat = att.host_mesh.getWorldMatrix();
                const bone_mat = skel.getBoneWorldMatrix(att.bone_index, host_mat);
                const with_offset = Mat4.mul(bone_mat, att.offset_matrix);
                return Mat4.mul(with_offset, local);
            }
        }
        if (self.parent) |p| {
            return Mat4.mul(p.getWorldMatrix(), local);
        }
        return local;
    }

    pub fn getWorldBoundingBox(self: Mesh) BoundingBox {
        return self.local_bounding_box.transform(self.getWorldMatrix());
    }

    /// Retains a CPU copy of the geometry so physics colliders (convex hull,
    /// triangle mesh) can be built from it later. Owned by the allocator.
    pub fn retainCpuGeometry(self: *Mesh, allocator: std.mem.Allocator, vertices: []const Vertex, indices: []const u16) !void {
        const positions = try allocator.alloc(Vec3, vertices.len);
        errdefer allocator.free(positions);
        for (vertices, 0..) |v, i| {
            positions[i] = @bitCast(v.position);
        }

        const index_u32 = try allocator.alloc(u32, indices.len);
        var i: usize = 0;
        const widen_tail = indices.len & ~@as(usize, 7);
        while (i < widen_tail) : (i += 8) {
            const narrow: @Vector(8, u16) = indices[i..][0..8].*;
            const wide: @Vector(8, u32) = @as(@Vector(8, u32), narrow);
            index_u32[i..][0..8].* = wide;
        }
        while (i < indices.len) : (i += 1) {
            index_u32[i] = indices[i];
        }

        self.cpu_positions = positions;
        self.cpu_indices = index_u32;
    }

    /// Same as retainCpuGeometry but for sources that are already 32-bit indexed.
    pub fn retainCpuGeometryU32(self: *Mesh, allocator: std.mem.Allocator, vertices: []const Vertex, indices: []const u32) !void {
        const positions = try allocator.alloc(Vec3, vertices.len);
        errdefer allocator.free(positions);
        for (vertices, 0..) |v, i| {
            positions[i] = @bitCast(v.position);
        }

        const index_copy = try allocator.alloc(u32, indices.len);
        @memcpy(index_copy, indices);

        self.cpu_positions = positions;
        self.cpu_indices = index_copy;
    }

    pub fn hasMorphTargets(self: *const Mesh) bool {
        return self.morph_targets.len > 0;
    }

    /// Sets one morph weight, clamped to [0, 1]. Out-of-range indices are
    /// ignored (never crash on animation data). Marks the mesh dirty.
    pub fn setMorphWeight(self: *Mesh, index: usize, weight: f32) void {
        if (index >= self.morph_weights.len) return;
        self.morph_weights[index] = std.math.clamp(weight, 0.0, 1.0);
        self.morph_dirty = true;
    }

    /// Sets all morph weights (up to min(dst, src)), each clamped to
    /// [0, 1]. Extra inputs are ignored; missing inputs keep their value.
    /// No-op (stays clean) when the mesh has no weights. Marks dirty.
    pub fn setMorphWeights(self: *Mesh, weights: []const f32) void {
        if (self.morph_weights.len == 0) return;
        const n = @min(self.morph_weights.len, weights.len);
        for (0..n) |i| {
            self.morph_weights[i] = std.math.clamp(weights[i], 0.0, 1.0);
        }
        self.morph_dirty = true;
    }

    /// Retains a CPU copy of the base vertices for morph blending.
    /// Base and staging start identical; dirty is cleared.
    /// Owned by the allocator; previous copies are freed.
    pub fn retainMorphBase(self: *Mesh, allocator: std.mem.Allocator, vertices: []const Vertex) !void {
        const base = try allocator.dupe(Vertex, vertices);
        errdefer allocator.free(base);
        const staging = try allocator.dupe(Vertex, vertices);
        errdefer allocator.free(staging);
        if (self.morph_base.len > 0) allocator.free(self.morph_base);
        if (self.morph_staging.len > 0) allocator.free(self.morph_staging);
        self.morph_base = base;
        self.morph_staging = staging;
        self.morph_dirty = false;
    }

    /// CPU blend: staging = base + Σ weight_i * delta_i (positions, normals,
    /// tangent xyz; colors/uv/joints/weights and tangent w untouched).
    /// Optimized single-pass per-vertex SIMD accumulation over active targets.
    /// Normals are NOT renormalized: keeps the blend exact and cheap;
    /// shaders consume them as-is. No-op when dirty == false.
    /// Uploads via sg.updateBuffer only when the vertex buffer exists (id != 0).
    pub fn applyMorphs(self: *Mesh) void {
        if (!self.morph_dirty) return;
        self.morph_dirty = false;
        if (self.morph_base.len == 0 or self.morph_staging.len == 0) return;
        const n = @min(self.morph_base.len, self.morph_staging.len);

        const target_count = @min(self.morph_targets.len, self.morph_weights.len);
        const ActiveTarget = struct {
            w: f32,
            pos: [][3]f32,
            norm: [][3]f32,
            tan: [][3]f32,
        };
        var active: [MAX_MORPH_TARGETS]ActiveTarget = undefined;
        var active_count: usize = 0;
        for (0..target_count) |t| {
            const w = self.morph_weights[t];
            if (w == 0.0) continue;
            const mt = &self.morph_targets[t];
            if (mt.position_deltas.len == 0 and mt.normal_deltas.len == 0 and mt.tangent_deltas.len == 0) continue;
            active[active_count] = .{
                .w = w,
                .pos = mt.position_deltas,
                .norm = mt.normal_deltas,
                .tan = mt.tangent_deltas,
            };
            active_count += 1;
        }

        if (active_count == 0) {
            @memcpy(self.morph_staging[0..n], self.morph_base[0..n]);
        } else {
            const targets = active[0..active_count];
            for (0..n) |i| {
                var v = self.morph_base[i];
                var p: @Vector(4, f32) = .{ v.position[0], v.position[1], v.position[2], 0.0 };
                var norm: @Vector(4, f32) = .{ v.normal[0], v.normal[1], v.normal[2], 0.0 };
                var tan: @Vector(4, f32) = .{ v.tangent[0], v.tangent[1], v.tangent[2], 0.0 };

                for (targets) |t| {
                    const w_vec: @Vector(4, f32) = @splat(t.w);
                    if (i < t.pos.len) {
                        const d = t.pos[i];
                        const d_vec: @Vector(4, f32) = .{ d[0], d[1], d[2], 0.0 };
                        p += w_vec * d_vec;
                    }
                    if (i < t.norm.len) {
                        const d = t.norm[i];
                        const d_vec: @Vector(4, f32) = .{ d[0], d[1], d[2], 0.0 };
                        norm += w_vec * d_vec;
                    }
                    if (i < t.tan.len) {
                        const d = t.tan[i];
                        const d_vec: @Vector(4, f32) = .{ d[0], d[1], d[2], 0.0 };
                        tan += w_vec * d_vec;
                    }
                }

                v.position = .{ p[0], p[1], p[2] };
                v.normal = .{ norm[0], norm[1], norm[2] };
                v.tangent[0] = tan[0];
                v.tangent[1] = tan[1];
                v.tangent[2] = tan[2];
                self.morph_staging[i] = v;
            }
        }

        if (self.vertex_buffer.id != 0) {
            sg.updateBuffer(self.vertex_buffer, sg.asRange(self.morph_staging[0..n]));
        }
    }

    pub fn deinit(self: *Mesh, allocator: std.mem.Allocator) void {
        if (self.vertex_buffer.id != 0) {
            sg.destroyBuffer(self.vertex_buffer);
        }
        if (self.index_buffer.id != 0) {
            sg.destroyBuffer(self.index_buffer);
        }
        if (self.instance_buffer.id != 0) {
            sg.destroyBuffer(self.instance_buffer);
        }
        for (self.instances.items) |inst| {
            allocator.destroy(inst);
        }
        self.instances.deinit(allocator);
        self.lod_levels.deinit(allocator);
        for (self.morph_targets) |*mt| {
            if (mt.position_deltas.len > 0) allocator.free(mt.position_deltas);
            if (mt.normal_deltas.len > 0) allocator.free(mt.normal_deltas);
            if (mt.tangent_deltas.len > 0) allocator.free(mt.tangent_deltas);
        }
        if (self.morph_targets.len > 0) allocator.free(self.morph_targets);
        if (self.morph_weights.len > 0) allocator.free(self.morph_weights);
        if (self.morph_base.len > 0) allocator.free(self.morph_base);
        if (self.morph_staging.len > 0) allocator.free(self.morph_staging);
        if (self.cpu_positions.len > 0) {
            allocator.free(self.cpu_positions);
        }
        if (self.cpu_indices.len > 0) {
            allocator.free(self.cpu_indices);
        }
        if (self.owns_name and self.name.len > 0) {
            allocator.free(self.name);
        }
    }
};

// Uploads CPU-side geometry to sokol buffers, narrowing indices to 16-bit
// when the vertex count allows it (matching the existing builders).
pub fn uploadGeometry(scene: *Scene, name: []const u8, data: GeometryData) !*Mesh {
    const vbuf = sg.makeBuffer(.{
        .data = sg.asRange(data.vertices),
    });

    const mesh = try scene.allocator.create(Mesh);
    errdefer scene.allocator.destroy(mesh);

    if (data.vertices.len <= std.math.maxInt(u16)) {
        // CPU copy first: it needs the original u32 indices.
        try mesh.retainCpuGeometryU32(scene.allocator, data.vertices, data.indices);
        const cpu_positions = mesh.cpu_positions;
        const cpu_indices = mesh.cpu_indices;
        // Narrow the u32 source buffer in place instead of allocating a temp
        // indices16 array.
        const wide = data.indices;
        const narrow_ptr: [*]u16 = @ptrCast(wide.ptr);
        const indices16 = narrow_ptr[0..wide.len];
        var k: usize = 0;
        const narrow_tail = wide.len & ~@as(usize, 7);
        while (k < narrow_tail) : (k += 8) {
            const v: @Vector(8, u32) = wide[k..][0..8].*;
            const n: @Vector(8, u16) = @truncate(v);
            (narrow_ptr + k)[0..8].* = n;
        }
        while (k < wide.len) : (k += 1) {
            indices16[k] = @intCast(wide[k]);
        }
        const ibuf = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(indices16),
        });
        mesh.* = .{
            .name = name,
            .vertex_buffer = vbuf,
            .index_buffer = ibuf,
            .index_count = @intCast(data.indices.len),
            .local_bounding_box = data.bounds,
            .cpu_positions = cpu_positions,
            .cpu_indices = cpu_indices,
        };
    } else {
        const ibuf = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(data.indices),
        });
        mesh.* = .{
            .name = name,
            .vertex_buffer = vbuf,
            .index_buffer = ibuf,
            .index_count = @intCast(data.indices.len),
            .index_type = .UINT32,
            .local_bounding_box = data.bounds,
        };
        try mesh.retainCpuGeometryU32(scene.allocator, data.vertices, data.indices);
    }

    try scene.meshes.append(scene.allocator, mesh);
    return mesh;
}
