//! Shared CPU-side mesh helpers for physics-driven helpers (ragdoll, vehicle).
//!
//! Internal module: not re-exported from `root.zig`. It owns the lifecycle of
//! the meshes attached to physics bodies:
//!   - creation, either as visible scene meshes (`MeshBuilder`, GPU buffers)
//!     when `scene` is non-null, or as bare physics-only `Mesh` structs
//!     (bounds + transform, no GPU work) when `scene` is null;
//!   - unlinking from `scene.meshes` before freeing (the scene would otherwise
//!     keep a dangling pointer and double-free in `Scene.deinit`);
//!   - copying the last solver-synced body transforms into the meshes;
//!   - the safe teardown order: joints, then bodies, then meshes.
//! All functions take explicit parameters and keep no global state.
//! Size conventions (same as the former per-helper copies):
//!   - box: full extents in `size`;
//!   - sphere: diameter in `size.x` (or the `diameter` parameter);
//!   - capsule: `(radius, total height)` in `size.x`, `size.y`.

const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const physics = @import("physics.zig");
const PhysicsWorld = physics.PhysicsWorld;
const RigidBody = physics.RigidBody;
const JointId = physics.JointId;
const mesh_mod = @import("mesh.zig");
const Mesh = mesh_mod.Mesh;
const MeshBuilder = mesh_mod.MeshBuilder;
const Scene = @import("scene.zig").Scene;

pub const MeshKind = enum { box, sphere, capsule };

/// Frees one body mesh. Scene meshes are unlinked from `scene.meshes` first;
/// bare physics-only meshes (created with `scene == null`) are just destroyed.
pub fn freeMesh(allocator: std.mem.Allocator, scene: ?*Scene, mesh: *Mesh) void {
    if (scene) |sc| {
        // Unlink from the scene first: the scene owns its mesh list and would
        // otherwise keep a dangling pointer (double free in Scene.deinit).
        var i: usize = 0;
        while (i < sc.meshes.items.len) {
            if (sc.meshes.items[i] == mesh) {
                _ = sc.meshes.swapRemove(i);
                break;
            }
            i += 1;
        }
        mesh.deinit(sc.allocator);
        sc.allocator.destroy(mesh);
    } else {
        // Bare physics-only mesh: no GPU buffers, no CPU geometry.
        allocator.destroy(mesh);
    }
}

/// Frees every mesh in the slice (unlinking scene meshes first).
pub fn freeMeshes(allocator: std.mem.Allocator, scene: ?*Scene, meshes: []*Mesh) void {
    for (meshes) |mesh| {
        freeMesh(allocator, scene, mesh);
    }
}

/// Creates a body mesh of the given kind. See the module docs for the `size`
/// conventions per kind. The mesh transform is set to `pos`.
pub fn createMesh(
    allocator: std.mem.Allocator,
    scene: ?*Scene,
    name: []const u8,
    kind: MeshKind,
    size: Vec3,
    pos: Vec3,
) !*Mesh {
    if (scene) |sc| {
        const mesh = switch (kind) {
            .box => try MeshBuilder.createBox(sc, name, .{
                .width = size.x,
                .height = size.y,
                .depth = size.z,
            }),
            .sphere => try MeshBuilder.createSphere(sc, name, .{
                .diameter = size.x,
                .segments = 18,
            }),
            .capsule => try MeshBuilder.createCapsule(sc, name, .{
                .radius = size.x,
                .height = size.y,
                .tessellation = 12,
                .cap_subdivisions = 4,
            }),
        };
        mesh.position = pos;
        return mesh;
    }
    const mesh = try allocator.create(Mesh);
    const bounds = switch (kind) {
        .box => BoundingBox.init(size.scale(-0.5), size.scale(0.5)),
        .sphere => BoundingBox.init(
            Vec3.new(-size.x * 0.5, -size.x * 0.5, -size.x * 0.5),
            Vec3.new(size.x * 0.5, size.x * 0.5, size.x * 0.5),
        ),
        .capsule => BoundingBox.init(
            Vec3.new(-size.x, -size.y * 0.5, -size.x),
            Vec3.new(size.x, size.y * 0.5, size.x),
        ),
    };
    mesh.* = .{
        .name = name,
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = pos,
        .local_bounding_box = bounds,
    };
    return mesh;
}

/// Creates a box body mesh with full `size` extents.
pub fn createBoxMesh(
    allocator: std.mem.Allocator,
    scene: ?*Scene,
    name: []const u8,
    size: Vec3,
    pos: Vec3,
) !*Mesh {
    return createMesh(allocator, scene, name, .box, size, pos);
}

/// Creates a sphere body mesh with the given `diameter`.
pub fn createSphereMesh(
    allocator: std.mem.Allocator,
    scene: ?*Scene,
    name: []const u8,
    diameter: f32,
    pos: Vec3,
) !*Mesh {
    return createMesh(allocator, scene, name, .sphere, Vec3.new(diameter, 0.0, 0.0), pos);
}

/// Creates a capsule body mesh with `radius` and total `height`.
pub fn createCapsuleMesh(
    allocator: std.mem.Allocator,
    scene: ?*Scene,
    name: []const u8,
    radius: f32,
    height: f32,
    pos: Vec3,
) !*Mesh {
    return createMesh(allocator, scene, name, .capsule, Vec3.new(radius, height, 0.0), pos);
}

/// Copies the last solver-synced body transforms into the body meshes.
/// Normally `step` already does this; use it to refresh meshes after
/// teleporting bodies or before rendering without stepping.
pub fn syncMeshes(bodies: []*RigidBody) void {
    for (bodies) |body| {
        body.mesh.position = body.last_pos;
        body.mesh.rotation = body.last_rot;
    }
}

/// Tears down helper state in the safe order: destroy joints, remove bodies
/// from the world, free meshes (unlinking scene meshes first), then deinit
/// the lists. Call before destroying the scene and the world.
pub fn teardown(
    allocator: std.mem.Allocator,
    world: *PhysicsWorld,
    scene: ?*Scene,
    joints: *std.ArrayListUnmanaged(JointId),
    bodies: *std.ArrayListUnmanaged(*RigidBody),
    meshes: *std.ArrayListUnmanaged(*Mesh),
) void {
    for (joints.items) |jid| {
        if (world.isJointValid(jid)) world.destroyJoint(jid);
    }
    joints.deinit(allocator);
    for (bodies.items) |body| {
        world.removeBody(body);
    }
    bodies.deinit(allocator);
    freeMeshes(allocator, scene, meshes.items);
    meshes.deinit(allocator);
}
