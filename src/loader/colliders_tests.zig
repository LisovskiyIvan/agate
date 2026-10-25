//! Unit tests for glTF / scene physics collider detection and parsing.
const std = @import("std");
const colliders = @import("colliders.zig");
const ColliderType = @import("../physics.zig").ColliderType;

test "colliders: parseJsonCollider direct collider fields" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "collider": "box",
        \\  "mass": 4.5,
        \\  "friction": 0.35,
        \\  "restitution": 0.15,
        \\  "is_sensor": false
        \\}
    ;
    const spec = colliders.parseJsonCollider(allocator, json);
    try std.testing.expect(spec != null);
    const s = spec.?;
    try std.testing.expectEqual(ColliderType.box, s.collider_type);
    try std.testing.expectApproxEqAbs(@as(f32, 4.5), s.mass, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.35), s.friction, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.15), s.restitution, 1e-4);
    try std.testing.expect(!s.is_sensor);
    try std.testing.expect(!s.is_collision_proxy);
}

test "colliders: parseJsonCollider nested physics object" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "physics": {
        \\    "shape": "sphere",
        \\    "type": "dynamic",
        \\    "mass": 2.0,
        \\    "friction": 0.5,
        \\    "restitution": 0.8,
        \\    "is_trigger": true
        \\  }
        \\}
    ;
    const spec = colliders.parseJsonCollider(allocator, json);
    try std.testing.expect(spec != null);
    const s = spec.?;
    try std.testing.expectEqual(ColliderType.sphere, s.collider_type);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), s.mass, 1e-4);
    try std.testing.expect(s.is_sensor);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), s.restitution, 1e-4);
}

test "colliders: parseJsonCollider OMI_physics_body style" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "motion": {
        \\    "type": "static",
        \\    "mass": 0.0
        \\  },
        \\  "collider": {
        \\    "shape": "trimesh"
        \\  }
        \\}
    ;
    const spec = colliders.parseJsonCollider(allocator, json);
    try std.testing.expect(spec != null);
    const s = spec.?;
    try std.testing.expectEqual(ColliderType.mesh, s.collider_type);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), s.mass, 1e-4);
}

test "colliders: parseJsonCollider convex hull and proxy flag" {
    const allocator = std.testing.allocator;
    const json =
        \\{
        \\  "shape": "convex",
        \\  "mass": 10.0,
        \\  "is_collision_proxy": true
        \\}
    ;
    const spec = colliders.parseJsonCollider(allocator, json);
    try std.testing.expect(spec != null);
    const s = spec.?;
    try std.testing.expectEqual(ColliderType.hull, s.collider_type);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), s.mass, 1e-4);
    try std.testing.expect(s.is_collision_proxy);
}

test "colliders: parseJsonCollider invalid json or non-collider returns null" {
    const allocator = std.testing.allocator;
    try std.testing.expect(colliders.parseJsonCollider(allocator, "") == null);
    try std.testing.expect(colliders.parseJsonCollider(allocator, "{") == null);
    try std.testing.expect(colliders.parseJsonCollider(allocator, "{\"unrelated\": 123}") == null);
}

test "colliders: parseNameCollider unreal conventions" {
    const ucx = colliders.parseNameCollider("UCX_Barrel_01");
    try std.testing.expect(ucx != null);
    try std.testing.expectEqual(ColliderType.hull, ucx.?.collider_type);
    try std.testing.expect(ucx.?.is_collision_proxy);
    try std.testing.expectEqual(@as(f32, 0.0), ucx.?.mass);

    const ubx = colliders.parseNameCollider("UBX_Crate_Big");
    try std.testing.expect(ubx != null);
    try std.testing.expectEqual(ColliderType.box, ubx.?.collider_type);
    try std.testing.expect(ubx.?.is_collision_proxy);

    const usp = colliders.parseNameCollider("USP_Sphere_Target");
    try std.testing.expect(usp != null);
    try std.testing.expectEqual(ColliderType.sphere, usp.?.collider_type);
    try std.testing.expect(usp.?.is_collision_proxy);

    const ucp = colliders.parseNameCollider("UCP_Pillar_High");
    try std.testing.expect(ucp != null);
    try std.testing.expectEqual(ColliderType.capsule, ucp.?.collider_type);
    try std.testing.expect(ucp.?.is_collision_proxy);
}

test "colliders: parseNameCollider col prefix and suffix conventions" {
    const col_box = colliders.parseNameCollider("col_box_wall");
    try std.testing.expect(col_box != null);
    try std.testing.expectEqual(ColliderType.box, col_box.?.collider_type);
    try std.testing.expect(col_box.?.is_collision_proxy);

    const col_hull = colliders.parseNameCollider("col_hull_rock");
    try std.testing.expect(col_hull != null);
    try std.testing.expectEqual(ColliderType.hull, col_hull.?.collider_type);
    try std.testing.expect(col_hull.?.is_collision_proxy);

    const col_mesh = colliders.parseNameCollider("col_mesh_terrain");
    try std.testing.expect(col_mesh != null);
    try std.testing.expectEqual(ColliderType.mesh, col_mesh.?.collider_type);
    try std.testing.expect(col_mesh.?.is_collision_proxy);

    const suff_col = colliders.parseNameCollider("Castle_Wall_col");
    try std.testing.expect(suff_col != null);
    try std.testing.expectEqual(ColliderType.mesh, suff_col.?.collider_type);
    try std.testing.expect(suff_col.?.is_collision_proxy);

    const suff_box_col = colliders.parseNameCollider("Crate_box_collider");
    try std.testing.expect(suff_box_col != null);
    try std.testing.expectEqual(ColliderType.box, suff_box_col.?.collider_type);
    try std.testing.expect(suff_box_col.?.is_collision_proxy);

    const plain = colliders.parseNameCollider("Hero_Character_Mesh");
    try std.testing.expect(plain == null);
}
