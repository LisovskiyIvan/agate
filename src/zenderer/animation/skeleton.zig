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
    skin_matrices: [MAX_BONES]Mat4 = [_]Mat4{Mat4.identity} ** MAX_BONES,

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
    }

    pub fn update(self: *Skeleton) void {
        var computed = [_]bool{false} ** MAX_BONES;
        const count = @min(self.bones.len, MAX_BONES);
        for (0..count) |i| {
            self.computeBoneMatrix(i, &computed);
        }
        for (0..count) |i| {
            self.skin_matrices[i] = self.bones[i].model_matrix.mul(self.bones[i].inverse_bind_matrix);
        }
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
};

test "Skeleton bind pose identity skin matrices" {
    const ally = std.testing.allocator;
    const skel = try Skeleton.init(ally, 2);
    defer skel.deinit();

    skel.bones[0].local_position = Vec3.new(0, 1, 0);
    skel.bones[0].bind_position = skel.bones[0].local_position;
    skel.bones[0].inverse_bind_matrix = Mat4.translation(Vec3.new(0, -1, 0));

    skel.bones[1].parent_index = 0;
    skel.bones[1].local_position = Vec3.new(0, 2, 0);
    skel.bones[1].bind_position = skel.bones[1].local_position;
    skel.bones[1].inverse_bind_matrix = Mat4.translation(Vec3.new(0, -3, 0));

    skel.update();

    for (0..16) |i| {
        try std.testing.expectApproxEqAbs(Mat4.identity.m[i], skel.skin_matrices[0].m[i], 1e-4);
        try std.testing.expectApproxEqAbs(Mat4.identity.m[i], skel.skin_matrices[1].m[i], 1e-4);
    }
}
