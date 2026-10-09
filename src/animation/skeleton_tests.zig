const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const skeleton = @import("skeleton.zig");
const Skeleton = skeleton.Skeleton;

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

test "Skeleton bone socket and world transform queries" {
    const ally = std.testing.allocator;
    const skel = try Skeleton.init(ally, 2);
    defer skel.deinit();

    skel.bones[0].local_position = Vec3.new(1, 0, 0);
    skel.bones[1].parent_index = 0;
    skel.bones[1].local_position = Vec3.new(0, 5, 0);
    skel.update();

    // Bone 1 model position should be (1, 5, 0)
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), skel.bones[1].model_matrix.m[12], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), skel.bones[1].model_matrix.m[13], 1e-4);

    // Host mesh placed at (10, 20, 30)
    const host_world = Mat4.translation(Vec3.new(10, 20, 30));
    const bone_pos = skel.getBoneWorldPosition(1, host_world);

    try std.testing.expectApproxEqAbs(@as(f32, 11.0), bone_pos.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), bone_pos.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 30.0), bone_pos.z, 1e-4);
}

test "Skeleton double-buffered skin matrices are published safely" {
    const ally = std.testing.allocator;
    const skel = try Skeleton.init(ally, 1);
    defer skel.deinit();

    // Initial state: slot 0
    skel.bones[0].local_position = Vec3.new(1, 0, 0);
    skel.update();

    const read1 = skel.getRenderSkinMatrices();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), read1[0].m[12], 1e-4);

    // Second update: writes into slot 1, publishes slot 1
    skel.bones[0].local_position = Vec3.new(5, 0, 0);
    skel.update();

    const read2 = skel.getRenderSkinMatrices();
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), read2[0].m[12], 1e-4);
    // read1 was in slot 0, so slot 1 is a different address in skin_slots
    try std.testing.expect(read1 != read2);
}
