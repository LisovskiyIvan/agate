const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Mesh = @import("mesh.zig").Mesh;
const pm = @import("physics_mesh.zig");
const createBoxMesh = pm.createBoxMesh;
const createSphereMesh = pm.createSphereMesh;
const createCapsuleMesh = pm.createCapsuleMesh;
const freeMesh = pm.freeMesh;
const freeMeshes = pm.freeMeshes;

test "bare meshes carry position and correct bounds" {
    const pos = Vec3.new(1.0, 2.0, 3.0);

    const box = try createBoxMesh(std.testing.allocator, null, "box", Vec3.new(2.0, 4.0, 6.0), pos);
    defer freeMesh(std.testing.allocator, null, box);
    try std.testing.expectEqual(pos, box.position);
    try std.testing.expectEqual(Vec3.new(-1.0, -2.0, -3.0), box.local_bounding_box.min);
    try std.testing.expectEqual(Vec3.new(1.0, 2.0, 3.0), box.local_bounding_box.max);

    const sphere = try createSphereMesh(std.testing.allocator, null, "sphere", 0.5, pos);
    defer freeMesh(std.testing.allocator, null, sphere);
    try std.testing.expectEqual(pos, sphere.position);
    try std.testing.expectEqual(Vec3.new(-0.25, -0.25, -0.25), sphere.local_bounding_box.min);
    try std.testing.expectEqual(Vec3.new(0.25, 0.25, 0.25), sphere.local_bounding_box.max);

    const capsule = try createCapsuleMesh(std.testing.allocator, null, "capsule", 0.06, 0.32, pos);
    defer freeMesh(std.testing.allocator, null, capsule);
    try std.testing.expectEqual(pos, capsule.position);
    try std.testing.expectEqual(Vec3.new(-0.06, -0.16, -0.06), capsule.local_bounding_box.min);
    try std.testing.expectEqual(Vec3.new(0.06, 0.16, 0.06), capsule.local_bounding_box.max);
}

test "freeMeshes frees a bare mesh list" {
    var meshes: std.ArrayListUnmanaged(*Mesh) = .empty;
    defer meshes.deinit(std.testing.allocator);

    try meshes.append(std.testing.allocator, try createBoxMesh(
        std.testing.allocator,
        null,
        "a",
        Vec3.new(1.0, 1.0, 1.0),
        Vec3.zero,
    ));
    try meshes.append(std.testing.allocator, try createSphereMesh(
        std.testing.allocator,
        null,
        "b",
        1.0,
        Vec3.zero,
    ));
    try std.testing.expectEqual(@as(usize, 2), meshes.items.len);
    freeMeshes(std.testing.allocator, null, meshes.items);
    meshes.clearRetainingCapacity();
    try std.testing.expectEqual(@as(usize, 0), meshes.items.len);
}
