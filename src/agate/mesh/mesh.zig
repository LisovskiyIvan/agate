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
const MorphMode = types.MorphMode;
const InstancedMesh = types.InstancedMesh;
const BoneAttachment = types.BoneAttachment;
const GeometryData = types.GeometryData;
const LODLevel = types.LODLevel;
const SkinJointWeight = types.SkinJointWeight;

const Material = @import("../material.zig").Material;
const StandardMaterial = @import("../material.zig").StandardMaterial;
const PBRMaterial = @import("../material.zig").PBRMaterial;
const Skeleton = @import("../animation/skeleton.zig").Skeleton;
const Scene = @import("../scene.zig").Scene;
const gpu_thread = @import("../gpu_thread.zig");
const morph_gpu = @import("morph_gpu.zig");

pub const Mesh = struct {
    id: u64 = 0,
    name: []const u8,
    owns_name: bool = false,
    position: Vec3 = Vec3.zero,
    rotation: Vec3 = Vec3.zero, // Euler angles in degrees
    scaling: Vec3 = Vec3.one,

    vertex_buffer: sg.Buffer,
    index_buffer: sg.Buffer,
    vertex_count: u32 = 0,
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
    is_occluder: bool = false,
    layer_mask: u32 = 0xFFFFFFFF,
    local_bounding_box: BoundingBox = BoundingBox.zero,

    // Optional CPU-side geometry retained for physics collider creation
    // (convex hull / triangle mesh shapes) and decal projection. Owned by the scene allocator.
    cpu_positions: []Vec3 = &.{},
    cpu_indices: []u32 = &.{},
    cpu_skin: []SkinJointWeight = &.{},

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
    morph_upload_needed: bool = false,
    /// Where deltas are blended: .cpu rewrites the vertex buffer from
    /// morph_staging (default, historical behavior); .gpu keeps a static
    /// base-pose vertex buffer and the vertex shader blends from the delta
    /// texture below (see mesh/morph_gpu.zig).
    morph_mode: MorphMode = .cpu,
    /// GPU-mode resources: RGBA32F delta strip (image + texture view),
    /// created by morph_gpu.uploadMorphDeltas. Destroyed in deinit.
    morph_delta_image: sg.Image = .{},
    morph_delta_view: sg.View = .{},
    morph_tex_width: u32 = 0,
    morph_tex_height: u32 = 0,

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
    instance_uploaded_frame: u64 = std.math.maxInt(u64),

    // Level of Detail (LOD)
    lod_levels: std.ArrayListUnmanaged(LODLevel) = .empty,
    is_lod_child: bool = false,

    // Decal
    is_decal: bool = false,

    /// Deferred GPU upload: true while the vertex/index buffers still need
    /// creating on the context thread (see uploadGeometry). `pending_vertices`
    /// holds the owned vertex copy the finish step builds the buffers from;
    /// indices come from the regular cpu_indices mirror. Set at creation when
    /// the caller is off-context (or no sg context exists yet); cleared by
    /// finishGpuUpload. Mesh.deinit frees the retained copies either way, so
    /// a mesh that dies pending never leaks.
    gpu_pending: bool = false,
    pending_vertices: []Vertex = &.{},
    /// Set by the glTF loader for off-context CPU-morph meshes: finishGpuUpload
    /// then creates the same EMPTY dynamic vertex buffer the immediate loader
    /// path builds (first frame's applyMorphs fills it) instead of a static
    /// buffer. One-shot: cleared together with gpu_pending on success.
    pending_dynamic_update: bool = false,
    /// Set by the glTF loader for off-context GPU-morph meshes instead of
    /// calling morph_gpu.uploadMorphDeltas inline (an sg.* call). finishGpuUpload
    /// performs the upload on the context thread and clears this only on
    /// success, so a failure/OOM retries on the next flush.
    morph_upload_pending: bool = false,
    /// One-shot guard: logs the first morph-delta upload failure so a
    /// permanent condition (e.g. missing RGBA32F support) is visible without
    /// spamming every flush; the retry itself continues.
    morph_upload_failure_logged: bool = false,

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

    /// Retains CPU skin joints and weights for skinned decal projection.
    pub fn retainCpuSkin(self: *Mesh, allocator: std.mem.Allocator, vertices: []const Vertex) !void {
        const skin = try allocator.alloc(SkinJointWeight, vertices.len);
        errdefer allocator.free(skin);
        for (vertices, 0..) |v, i| {
            skin[i] = .{
                .joints = v.joints,
                .weights = v.weights,
            };
        }
        if (self.cpu_skin.len > 0) allocator.free(self.cpu_skin);
        self.cpu_skin = skin;
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
    /// GPU mode returns immediately: deltas blend in the vertex shader from
    /// the delta texture, the vertex buffer keeps the static base pose, and
    /// per-frame CPU cost is only the weights the draw path reads.
    pub fn applyMorphs(self: *Mesh) void {
        if (self.morph_mode == .gpu) return;
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

        self.morph_upload_needed = true;
    }

    /// Pushes update-staged morph vertex data into the GPU vertex buffer.
    /// Called at render start on the context thread (flushPendingGpuUploads pattern).
    pub fn flushGpuUploads(self: *Mesh) void {
        if (!self.morph_upload_needed) return;
        self.morph_upload_needed = false;
        const n = @min(self.morph_base.len, self.morph_staging.len);
        if (n > 0 and self.vertex_buffer.id != 0) {
            sg.updateBuffer(self.vertex_buffer, sg.asRange(self.morph_staging[0..n]));
        }
    }

    /// Completes a deferred GPU upload: creates the vertex/index buffers from
    /// the retained pending_vertices/cpu_indices and frees the temporary
    /// vertex copy. Runs on the sg-context thread via
    /// Scene.flushPendingGpuUploads at render start — never from the update
    /// phase. No-op unless gpu_pending; when no sg context is valid yet the
    /// mesh stays pending and the next flush retries (tests exercise meshes
    /// in that state without ever touching sg.*).
    pub fn finishGpuUpload(self: *Mesh, allocator: std.mem.Allocator) void {
        if (!self.gpu_pending) return;
        if (!sg.isvalid()) return;
        // CPU-morph glTF meshes created off-context need the same empty
        // dynamic vertex buffer the immediate loader path builds (the first
        // frame's applyMorphs fills it); everything else uploads static.
        const vbuf = if (self.pending_dynamic_update)
            sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = self.pending_vertices.len * @sizeOf(Vertex),
            })
        else
            sg.makeBuffer(.{
                .data = sg.asRange(self.pending_vertices),
            });
        if (vbuf.id == 0) return;
        if (self.index_type == .UINT16) {
            const indices16 = allocator.alloc(u16, self.cpu_indices.len) catch return;
            defer allocator.free(indices16);
            for (self.cpu_indices, 0..) |idx, k| {
                indices16[k] = @intCast(idx);
            }
            const ibuf = sg.makeBuffer(.{
                .usage = .{ .index_buffer = true },
                .data = sg.asRange(indices16),
            });
            if (ibuf.id == 0) {
                sg.destroyBuffer(vbuf);
                return;
            }
            self.index_buffer = ibuf;
        } else {
            const ibuf = sg.makeBuffer(.{
                .usage = .{ .index_buffer = true },
                .data = sg.asRange(self.cpu_indices),
            });
            if (ibuf.id == 0) {
                sg.destroyBuffer(vbuf);
                return;
            }
            self.index_buffer = ibuf;
        }
        // Deferred GPU-morph delta texture: upload on the context thread now
        // that the base buffers exist. On failure (OOM) or pool exhaustion
        // (dead view handle) tear the fresh buffers back down and keep every
        // pending flag, so the next flush retries the whole finish
        // atomically and the mesh stays skipped by the queue meanwhile.
        if (self.morph_upload_pending) {
            morph_gpu.uploadMorphDeltas(self, allocator) catch {
                self.logMorphUploadFailure();
                sg.destroyBuffer(vbuf);
                sg.destroyBuffer(self.index_buffer);
                self.index_buffer = .{};
                return;
            };
            if (self.morph_delta_view.id == 0) {
                self.logMorphUploadFailure();
                sg.destroyBuffer(vbuf);
                sg.destroyBuffer(self.index_buffer);
                self.index_buffer = .{};
                return;
            }
            self.morph_upload_pending = false;
        }
        self.vertex_buffer = vbuf;
        if (self.vertex_count == 0 and self.pending_vertices.len > 0) {
            self.vertex_count = @intCast(self.pending_vertices.len);
        }
        self.gpu_pending = false;
        // CPU-morph deferred creation can land after this frame's applyMorphs
        // already ran (or before it ever will): upload the retained base pose
        // so frame 1 cannot draw an all-zero dynamic buffer. Staging equals
        // the base pose until applyMorphs blends the first time.
        if (self.pending_dynamic_update) self.morph_upload_needed = true;
        self.pending_dynamic_update = false;
        if (self.pending_vertices.len > 0) {
            allocator.free(self.pending_vertices);
            self.pending_vertices = &.{};
        }
    }

    /// Returns total estimated GPU memory in bytes for this mesh's vertex/index buffers
    /// and morph delta textures.
    pub fn getGpuMemoryBytes(self: *const Mesh) usize {
        var total: usize = 0;
        if (self.vertex_buffer.id != 0) {
            total += @as(usize, self.vertex_count) * @sizeOf(Vertex);
        }
        if (self.index_buffer.id != 0) {
            const idx_size: usize = if (self.index_type == .UINT16) 2 else 4;
            total += @as(usize, self.index_count) * idx_size;
        }
        if (self.morph_delta_image.id != 0) {
            total += @as(usize, self.morph_tex_width) * @as(usize, self.morph_tex_height) * 16;
        }
        if (self.instance_buffer.id != 0) {
            total += self.instance_buffer_capacity * @sizeOf(Mat4);
        }
        return total;
    }

    /// Returns total CPU heap memory in bytes retained by this mesh.
    pub fn getCpuMemoryBytes(self: *const Mesh) usize {
        var total: usize = @sizeOf(Mesh);
        if (self.owns_name) total += self.name.len;
        total += self.cpu_positions.len * @sizeOf(Vec3);
        total += self.cpu_indices.len * @sizeOf(u32);
        total += self.cpu_skin.len * @sizeOf(SkinJointWeight);
        total += self.pending_vertices.len * @sizeOf(Vertex);
        total += self.morph_weights.len * @sizeOf(f32);
        total += self.morph_base.len * @sizeOf(Vertex);
        total += self.morph_staging.len * @sizeOf(Vertex);
        total += self.morph_targets.len * @sizeOf(MorphTarget);
        for (self.morph_targets) |mt| {
            total += mt.position_deltas.len * @sizeOf([3]f32);
            total += mt.normal_deltas.len * @sizeOf([3]f32);
            total += mt.tangent_deltas.len * @sizeOf([3]f32);
        }
        return total;
    }

    /// Logs the first morph-delta failure only (permanent conditions such as
    /// missing RGBA32F would otherwise log on every flush).
    fn logMorphUploadFailure(self: *Mesh) void {
        if (self.morph_upload_failure_logged) return;
        self.morph_upload_failure_logged = true;
        std.log.err("mesh '{s}': morph delta texture unavailable, retrying on each flush", .{self.name});
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
        if (self.morph_delta_view.id != 0) {
            sg.destroyView(self.morph_delta_view);
        }
        if (self.morph_delta_image.id != 0) {
            sg.destroyImage(self.morph_delta_image);
        }
        if (self.cpu_positions.len > 0) {
            allocator.free(self.cpu_positions);
        }
        if (self.cpu_indices.len > 0) {
            allocator.free(self.cpu_indices);
        }
        if (self.cpu_skin.len > 0) {
            allocator.free(self.cpu_skin);
        }
        if (self.pending_vertices.len > 0) {
            allocator.free(self.pending_vertices);
            self.pending_vertices = &.{};
        }
        if (self.owns_name and self.name.len > 0) {
            allocator.free(self.name);
        }
    }
};

// Uploads CPU-side geometry to sokol buffers, narrowing indices to 16-bit
// when the vertex count allows it (matching the existing builders).
// Callers on a non-context thread (input-driven decal stamping, drag-box
// creation on the game thread) must not touch sg.*: they take the deferred
// path below (CPU-only + gpu_pending), and Scene.flushPendingGpuUploads
// finishes the buffers on the context thread. prepareFrame() flushes before
// queue building, so a pending mesh always has buffers before it can be
// queued for drawing. The deferred path also covers "no valid sg context
// yet", so context-less tests never crash inside makeBuffer.
//
// The immediate path fails with error.GpuBufferAllocationFailed when the
// sokol buffer pool is exhausted (sg.makeBuffer returns id == 0) instead of
// publishing a mesh with dead handles; nothing is appended and no GPU/CPU
// state leaks.
pub fn uploadGeometry(scene: *Scene, name: []const u8, data: GeometryData) !*Mesh {
    if (!gpu_thread.isOnContextThread() or !sg.isvalid()) {
        const mesh = try scene.allocator.create(Mesh);
        errdefer scene.allocator.destroy(mesh);
        mesh.* = .{
            .name = name,
            .vertex_buffer = .{},
            .index_buffer = .{},
            .vertex_count = @intCast(data.vertices.len),
            .index_count = @intCast(data.indices.len),
            .index_type = if (data.vertices.len <= std.math.maxInt(u16)) .UINT16 else .UINT32,
            .local_bounding_box = data.bounds,
            .gpu_pending = true,
        };
        errdefer {
            if (mesh.pending_vertices.len > 0) scene.allocator.free(mesh.pending_vertices);
            if (mesh.cpu_positions.len > 0) scene.allocator.free(mesh.cpu_positions);
            if (mesh.cpu_indices.len > 0) scene.allocator.free(mesh.cpu_indices);
        }
        // Same CPU mirrors as the immediate path (physics colliders and
        // decal projection read them); indices double as the finish-step
        // source, vertices are kept in pending_vertices for buffer creation.
        try mesh.retainCpuGeometryU32(scene.allocator, data.vertices, data.indices);
        mesh.pending_vertices = try scene.allocator.dupe(Vertex, data.vertices);
        try scene.meshes.append(scene.allocator, mesh);
        return mesh;
    }

    // Immediate path: default-initialize the Mesh first so the single
    // errdefer below cleans whatever exists on any failure (pool-exhausted
    // buffers, CPU mirror OOM, scene append OOM). owns_name stays false:
    // the name slice is caller-owned until scene.meshes adopts the mesh.
    const index_type: sg.IndexType = if (data.vertices.len <= std.math.maxInt(u16)) .UINT16 else .UINT32;
    const mesh = try scene.allocator.create(Mesh);
    mesh.* = .{
        .name = name,
        .vertex_buffer = .{},
        .index_buffer = .{},
        .vertex_count = @intCast(data.vertices.len),
        .index_count = @intCast(data.indices.len),
        .index_type = index_type,
        .local_bounding_box = data.bounds,
    };
    errdefer {
        if (mesh.vertex_buffer.id != 0) sg.destroyBuffer(mesh.vertex_buffer);
        if (mesh.index_buffer.id != 0) sg.destroyBuffer(mesh.index_buffer);
        if (mesh.cpu_positions.len > 0) scene.allocator.free(mesh.cpu_positions);
        if (mesh.cpu_indices.len > 0) scene.allocator.free(mesh.cpu_indices);
        if (mesh.pending_vertices.len > 0) scene.allocator.free(mesh.pending_vertices);
        scene.allocator.destroy(mesh);
    }

    const vbuf = sg.makeBuffer(.{
        .data = sg.asRange(data.vertices),
    });
    if (vbuf.id == 0) return error.GpuBufferAllocationFailed;
    mesh.vertex_buffer = vbuf;

    if (index_type == .UINT16) {
        // CPU copy first: it needs the original u32 indices.
        try mesh.retainCpuGeometryU32(scene.allocator, data.vertices, data.indices);
        const indices16 = try scene.allocator.alloc(u16, data.indices.len);
        defer scene.allocator.free(indices16);
        for (data.indices, 0..) |idx, k| {
            indices16[k] = @intCast(idx);
        }
        const ibuf = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(indices16),
        });
        if (ibuf.id == 0) return error.GpuBufferAllocationFailed;
        mesh.index_buffer = ibuf;
    } else {
        const ibuf = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(data.indices),
        });
        if (ibuf.id == 0) return error.GpuBufferAllocationFailed;
        mesh.index_buffer = ibuf;
        try mesh.retainCpuGeometryU32(scene.allocator, data.vertices, data.indices);
    }

    try scene.meshes.append(scene.allocator, mesh);
    return mesh;
}

test "finishGpuUpload with pending_dynamic_update stays pending without sg context" {
    const ally = std.testing.allocator;
    var m: Mesh = .{
        .name = "pending_dynamic",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
    };
    m.gpu_pending = true;
    m.pending_dynamic_update = true;
    m.pending_vertices = try ally.dupe(Vertex, &[_]Vertex{
        std.mem.zeroes(Vertex),
        std.mem.zeroes(Vertex),
        std.mem.zeroes(Vertex),
    });
    m.cpu_indices = try ally.dupe(u32, &[_]u32{ 0, 1, 2 });
    defer m.deinit(ally);
    // No sg context in tests: must stay pending without touching sg.* and
    // keep the retained vertex copy for the later context-thread finish.
    m.finishGpuUpload(ally);
    try std.testing.expect(m.gpu_pending);
    try std.testing.expect(m.pending_dynamic_update);
    try std.testing.expectEqual(@as(usize, 3), m.pending_vertices.len);
    try std.testing.expectEqual(@as(u32, 0), m.vertex_buffer.id);
}

test "finishGpuUpload with morph_upload_pending survives without sg context" {
    const ally = std.testing.allocator;
    var m: Mesh = .{
        .name = "pending_morph",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
    };
    m.gpu_pending = true;
    m.morph_upload_pending = true;
    m.pending_vertices = try ally.dupe(Vertex, &[_]Vertex{
        std.mem.zeroes(Vertex),
        std.mem.zeroes(Vertex),
        std.mem.zeroes(Vertex),
    });
    m.cpu_indices = try ally.dupe(u32, &[_]u32{ 0, 1, 2 });
    defer m.deinit(ally);
    // No sg context: the delta-texture upload must not run; the flag stays
    // set so the context-thread flush retries it.
    m.finishGpuUpload(ally);
    try std.testing.expect(m.gpu_pending);
    try std.testing.expect(m.morph_upload_pending);
    try std.testing.expectEqual(@as(usize, 3), m.pending_vertices.len);
}

test "uploadGeometry defers without sg context and finish preserves pending" {
    const testScene = @import("../testing.zig").testScene;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var scene = testScene(alloc);
    var verts = [_]Vertex{
        std.mem.zeroes(Vertex),
        std.mem.zeroes(Vertex),
        std.mem.zeroes(Vertex),
    };
    var idx = [_]u32{ 0, 1, 2 };
    const m = try uploadGeometry(&scene, "deferred", .{
        .vertices = &verts,
        .indices = &idx,
        .bounds = BoundingBox.zero,
    });
    try std.testing.expect(m.gpu_pending);
    try std.testing.expectEqual(@as(usize, 3), m.pending_vertices.len);
    try std.testing.expectEqual(@as(usize, 3), m.cpu_positions.len);
    try std.testing.expectEqual(@as(usize, 3), m.cpu_indices.len);
    // Still no sg context: the finish attempt must keep everything pending.
    m.finishGpuUpload(alloc);
    try std.testing.expect(m.gpu_pending);
    try std.testing.expectEqual(@as(usize, 3), m.pending_vertices.len);
}
