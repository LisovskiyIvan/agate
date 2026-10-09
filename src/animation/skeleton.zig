const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Quat = math.Quat;
const Mat4 = math.Mat4;

pub const MAX_BONES: usize = 64;

pub const Bone = struct {
    name: []const u8 = "",
    parent_index: ?usize = null,

    // Current local TRS
    local_position: Vec3 = Vec3.zero,
    local_rotation: Quat = Quat.identity,
    local_scale: Vec3 = Vec3.one,

    // Bind pose local TRS
    bind_position: Vec3 = Vec3.zero,
    bind_rotation: Quat = Quat.identity,
    bind_scale: Vec3 = Vec3.one,

    // Inverse bind matrix (from glTF skin)
    inverse_bind_matrix: Mat4 = Mat4.identity,

    // Computed bone model transform (relative to armature/mesh)
    model_matrix: Mat4 = Mat4.identity,
};

pub const Skeleton = struct {
    allocator: std.mem.Allocator,
    name: []const u8 = "",
    bones: []Bone,
    root_transform: Mat4 = Mat4.identity,
    /// Double-buffered skin matrices: writers fill (1 - render_slot),
    /// then publish by updating render_slot with release semantics.
    skin_slots: [2][MAX_BONES]Mat4 = [_][MAX_BONES]Mat4{[_]Mat4{Mat4.identity} ** MAX_BONES} ** 2,
    render_slot: std.atomic.Value(u8) = std.atomic.Value(u8).init(0),
    /// Simulation-thread view of the latest computed skin matrices (preserves backward compatibility)
    skin_matrices: [MAX_BONES]Mat4 = [_]Mat4{Mat4.identity} ** MAX_BONES,
    /// Previous PRESENTED frame's skin matrices for velocity calculation.
    /// Written only by Skeleton.commitPresentedSkin (game side, once per
    /// published front slot) — never by update() — so sim update rate and
    /// cancelled/repeated builds cannot advance it.
    prev_skin_matrices: [MAX_BONES]Mat4 = [_]Mat4{Mat4.identity} ** MAX_BONES,
    /// Front-slot frame_id that produced prev_skin_matrices (maxInt = never).
    vel_presented_frame: u64 = std.math.maxInt(u64),

    pub fn init(allocator: std.mem.Allocator, bone_count: usize) !*Skeleton {
        const skel = try allocator.create(Skeleton);
        const bones = try allocator.alloc(Bone, bone_count);
        for (bones) |*b| {
            b.* = .{};
        }
        skel.* = .{
            .allocator = allocator,
            .bones = bones,
        };
        return skel;
    }

    pub fn deinit(self: *Skeleton) void {
        for (self.bones) |b| {
            if (b.name.len > 0) {
                self.allocator.free(b.name);
            }
        }
        if (self.name.len > 0) {
            self.allocator.free(self.name);
        }
        self.allocator.free(self.bones);
        self.allocator.destroy(self);
    }

    pub fn resetToBindPose(self: *Skeleton) void {
        for (self.bones) |*b| {
            b.local_position = b.bind_position;
            b.local_rotation = b.bind_rotation;
            b.local_scale = b.bind_scale;
        }
        self.update();
        self.skin_slots[0] = self.skin_matrices;
        self.skin_slots[1] = self.skin_matrices;
        self.prev_skin_matrices = self.skin_matrices;
        self.vel_presented_frame = std.math.maxInt(u64);
    }

    pub fn update(self: *Skeleton) void {
        var computed = [_]bool{false} ** MAX_BONES;
        const count = @min(self.bones.len, MAX_BONES);
        for (0..count) |i| {
            self.computeBoneMatrix(i, &computed);
        }
        const write_slot: u8 = 1 - (self.render_slot.load(.monotonic) & 1);
        for (0..count) |i| {
            const m = self.bones[i].model_matrix.mul(self.bones[i].inverse_bind_matrix);
            self.skin_slots[write_slot][i] = m;
            self.skin_matrices[i] = m;
        }
        self.render_slot.store(write_slot, .release);
    }

    /// Read-side: returns a pointer to the currently published skin matrices slot.
    /// Thread-safe against concurrent simulation updates (which write to the alternate slot).
    pub fn getRenderSkinMatrices(self: *const Skeleton) *const [MAX_BONES]Mat4 {
        const slot = self.render_slot.load(.acquire) & 1;
        return &self.skin_slots[slot];
    }

    pub fn getPrevSkinMatrices(self: *const Skeleton) *const [MAX_BONES]Mat4 {
        // Unpresented skeletons report zero motion: the cull pairs this
        // with the current slot on first presentation (see cullNonInstancedMesh).
        if (self.vel_presented_frame == std.math.maxInt(u64)) return self.getRenderSkinMatrices();
        return &self.prev_skin_matrices;
    }

    /// Presented-frame commit (game side, once per published front slot):
    /// freezes the queue's skin copy as this skeleton's previous frame.
    /// The caller resolves ownership (mesh.skeleton); repeated commits of
    /// the same frame are idempotent.
    pub fn commitPresentedSkin(self: *Skeleton, presented: *const [MAX_BONES]Mat4, frame_id: u64) void {
        self.prev_skin_matrices = presented.*;
        self.vel_presented_frame = frame_id;
    }

    fn computeBoneMatrix(self: *Skeleton, index: usize, computed: []bool) void {
        if (index >= MAX_BONES or computed[index]) return;
        const b = &self.bones[index];
        const local = Mat4.fromQuatTranslationScale(b.local_position, b.local_rotation, b.local_scale);
        if (b.parent_index) |pi| {
            if (pi < self.bones.len and pi != index) {
                self.computeBoneMatrix(pi, computed);
                b.model_matrix = self.bones[pi].model_matrix.mul(local);
            } else {
                b.model_matrix = self.root_transform.mul(local);
            }
        } else {
            b.model_matrix = self.root_transform.mul(local);
        }
        computed[index] = true;
    }

    pub fn findBoneIndex(self: *const Skeleton, name: []const u8) ?usize {
        for (self.bones, 0..) |b, i| {
            if (std.mem.eql(u8, b.name, name)) return i;
        }
        return null;
    }

    /// Gets the bone's model-space matrix relative to skeleton root
    pub fn getBoneModelMatrix(self: *const Skeleton, bone_index: usize) Mat4 {
        if (bone_index >= self.bones.len) return Mat4.identity;
        return self.bones[bone_index].model_matrix;
    }

    /// Computes the bone's full world-space transform matrix given the host mesh's world matrix
    pub fn getBoneWorldMatrix(self: *const Skeleton, bone_index: usize, host_world_matrix: Mat4) Mat4 {
        if (bone_index >= self.bones.len) return host_world_matrix;
        return host_world_matrix.mul(self.bones[bone_index].model_matrix);
    }

    /// Computes the bone's 3D position in world space given the host mesh's world matrix
    pub fn getBoneWorldPosition(self: *const Skeleton, bone_index: usize, host_world_matrix: Mat4) Vec3 {
        return self.getBoneWorldMatrix(bone_index, host_world_matrix).getTranslation();
    }
};
