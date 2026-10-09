//! Ready-made ragdoll helper on top of Box3D.
//!
//! Builds an 11-body humanoid (pelvis, chest, head, upper/lower arms and
//! legs) linked by spherical joints (waist, neck, shoulders, hips) and hinge
//! joints (elbows, knees), mirroring the manual ragdoll in the sandbox demo.
//! Meshes are optional: pass a scene to get visible capsule/box/sphere meshes,
//! or null for a physics-only ragdoll (unit tests, headless servers).
//! The caller owns the `PhysicsWorld`; this helper never copies it.

const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Quat = math.Quat;
const physics = @import("physics.zig");
const PhysicsWorld = physics.PhysicsWorld;
const RigidBody = physics.RigidBody;
const ColliderType = physics.ColliderType;
const JointId = physics.JointId;
const mesh_mod = @import("mesh.zig");
const Mesh = mesh_mod.Mesh;
const physics_mesh = @import("physics_mesh.zig");
const MeshKind = physics_mesh.MeshKind;
const Scene = @import("scene.zig").Scene;

/// One of the 11 ragdoll bodies, in creation order.
pub const RagdollPart = enum(u8) {
    pelvis = 0,
    chest,
    head,
    upper_arm_l,
    upper_arm_r,
    lower_arm_l,
    lower_arm_r,
    upper_leg_l,
    upper_leg_r,
    lower_leg_l,
    lower_leg_r,
};

/// Number of bodies created by `Ragdoll.init`.
pub const part_count: usize = 11;
/// Number of joints created by `Ragdoll.init` (waist, neck, 2x shoulder,
/// elbow, hip, knee).
pub const joint_count: usize = 10;

pub const RagdollOptions = struct {
    /// Ground point under the feet; part offsets are added on top of it.
    position: Vec3 = Vec3.zero,
    /// Uniform scale applied to every size and offset (masses are unchanged).
    scale: f32 = 1.0,
    pelvis_size: Vec3 = Vec3.new(0.34, 0.18, 0.20),
    pelvis_mass: f32 = 2.5,
    chest_size: Vec3 = Vec3.new(0.40, 0.50, 0.24),
    chest_mass: f32 = 4.0,
    head_radius: f32 = 0.14,
    head_mass: f32 = 1.0,
    upper_arm_radius: f32 = 0.06,
    upper_arm_length: f32 = 0.32, // total height including caps
    upper_arm_mass: f32 = 0.7,
    lower_arm_radius: f32 = 0.055,
    lower_arm_length: f32 = 0.30,
    lower_arm_mass: f32 = 0.5,
    upper_leg_radius: f32 = 0.09,
    upper_leg_length: f32 = 0.46,
    upper_leg_mass: f32 = 1.2,
    lower_leg_radius: f32 = 0.07,
    lower_leg_length: f32 = 0.44,
    lower_leg_mass: f32 = 0.8,
    friction: f32 = 0.6,
    restitution: f32 = 0.1,
};

const PartSpec = struct {
    name: []const u8,
    kind: MeshKind,
    // Box: full extents. Sphere: diameter in x. Capsule: (radius, total height).
    size: Vec3,
    // Unit-space offset from the spawn position (multiplied by scale).
    offset: Vec3,
    mass: f32,
    collider: ColliderType,
};

pub const Ragdoll = struct {
    allocator: std.mem.Allocator,
    scene: ?*Scene,
    bodies: std.ArrayListUnmanaged(*RigidBody) = .empty,
    joints: std.ArrayListUnmanaged(JointId) = .empty,
    meshes: std.ArrayListUnmanaged(*Mesh) = .empty,
    home_positions: std.ArrayListUnmanaged(Vec3) = .empty,
    home_rotations: std.ArrayListUnmanaged(Vec3) = .empty,

    /// Builds the bodies, joints and optional meshes. On error any partial
    /// state is torn down, so the caller only deinits on success.
    /// Meshes are created through the scene (and unlinked from it in deinit)
    /// when `scene` is non-null, otherwise as bare physics-only meshes.
    /// Deinit the ragdoll before the scene and before the world.
    pub fn init(
        allocator: std.mem.Allocator,
        world: *PhysicsWorld,
        scene: ?*Scene,
        options: RagdollOptions,
    ) !Ragdoll {
        var self = Ragdoll{
            .allocator = allocator,
            .scene = scene,
        };
        errdefer self.deinit(world);

        const s = options.scale;
        const specs = [part_count]PartSpec{
            .{ .name = "ragdoll_pelvis", .kind = .box, .size = options.pelvis_size, .offset = Vec3.new(0.0, 0.99, 0.0), .mass = options.pelvis_mass, .collider = .box },
            .{ .name = "ragdoll_chest", .kind = .box, .size = options.chest_size, .offset = Vec3.new(0.0, 1.33, 0.0), .mass = options.chest_mass, .collider = .box },
            .{ .name = "ragdoll_head", .kind = .sphere, .size = Vec3.new(options.head_radius * 2.0, 0.0, 0.0), .offset = Vec3.new(0.0, 1.70, 0.0), .mass = options.head_mass, .collider = .sphere },
            .{ .name = "ragdoll_upper_arm_l", .kind = .capsule, .size = Vec3.new(options.upper_arm_radius, options.upper_arm_length, 0.0), .offset = Vec3.new(0.26, 1.34, 0.0), .mass = options.upper_arm_mass, .collider = .capsule },
            .{ .name = "ragdoll_upper_arm_r", .kind = .capsule, .size = Vec3.new(options.upper_arm_radius, options.upper_arm_length, 0.0), .offset = Vec3.new(-0.26, 1.34, 0.0), .mass = options.upper_arm_mass, .collider = .capsule },
            .{ .name = "ragdoll_lower_arm_l", .kind = .capsule, .size = Vec3.new(options.lower_arm_radius, options.lower_arm_length, 0.0), .offset = Vec3.new(0.26, 1.03, 0.0), .mass = options.lower_arm_mass, .collider = .capsule },
            .{ .name = "ragdoll_lower_arm_r", .kind = .capsule, .size = Vec3.new(options.lower_arm_radius, options.lower_arm_length, 0.0), .offset = Vec3.new(-0.26, 1.03, 0.0), .mass = options.lower_arm_mass, .collider = .capsule },
            .{ .name = "ragdoll_upper_leg_l", .kind = .capsule, .size = Vec3.new(options.upper_leg_radius, options.upper_leg_length, 0.0), .offset = Vec3.new(0.11, 0.67, 0.0), .mass = options.upper_leg_mass, .collider = .capsule },
            .{ .name = "ragdoll_upper_leg_r", .kind = .capsule, .size = Vec3.new(options.upper_leg_radius, options.upper_leg_length, 0.0), .offset = Vec3.new(-0.11, 0.67, 0.0), .mass = options.upper_leg_mass, .collider = .capsule },
            .{ .name = "ragdoll_lower_leg_l", .kind = .capsule, .size = Vec3.new(options.lower_leg_radius, options.lower_leg_length, 0.0), .offset = Vec3.new(0.11, 0.22, 0.0), .mass = options.lower_leg_mass, .collider = .capsule },
            .{ .name = "ragdoll_lower_leg_r", .kind = .capsule, .size = Vec3.new(options.lower_leg_radius, options.lower_leg_length, 0.0), .offset = Vec3.new(-0.11, 0.22, 0.0), .mass = options.lower_leg_mass, .collider = .capsule },
        };

        for (specs) |spec| {
            const pos = options.position.add(spec.offset.scale(s));
            const scaled_size = spec.size.scale(s);
            const mesh = try physics_mesh.createMesh(allocator, scene, spec.name, spec.kind, scaled_size, pos);
            var mesh_tracked = false;
            defer if (!mesh_tracked) physics_mesh.freeMesh(allocator, scene, mesh);
            try self.meshes.append(allocator, mesh);
            mesh_tracked = true;

            const body = try world.createBody(mesh, spec.collider, spec.mass);
            var body_tracked = false;
            defer if (!body_tracked) world.removeBody(body);
            body.friction = options.friction;
            body.restitution = options.restitution;
            try self.bodies.append(allocator, body);
            body_tracked = true;

            try self.home_positions.append(allocator, pos);
            try self.home_rotations.append(allocator, Vec3.zero);
        }

        const at = struct {
            fn anchor(root: Vec3, x: f32, y: f32, scale: f32) Vec3 {
                return root.add(Vec3.new(x * scale, y * scale, 0.0));
            }
        }.anchor;
        const root = options.position;
        const hinge = Quat.fromEulerDeg(Vec3.new(0.0, 90.0, 0.0));
        const b = self.bodies.items;

        const ball_joints = [6]struct { a: usize, c: usize, x: f32, y: f32, cone: f32, twist: f32 }{
            .{ .a = 0, .c = 1, .x = 0.0, .y = 1.08, .cone = 0.35, .twist = 0.0 }, // waist
            .{ .a = 1, .c = 2, .x = 0.0, .y = 1.59, .cone = 0.45, .twist = 0.0 }, // neck
            .{ .a = 1, .c = 3, .x = 0.24, .y = 1.50, .cone = 0.8, .twist = 0.6 }, // shoulder L
            .{ .a = 1, .c = 4, .x = -0.24, .y = 1.50, .cone = 0.8, .twist = 0.6 }, // shoulder R
            .{ .a = 0, .c = 7, .x = 0.11, .y = 0.90, .cone = 0.6, .twist = 0.5 }, // hip L
            .{ .a = 0, .c = 8, .x = -0.11, .y = 0.90, .cone = 0.6, .twist = 0.5 }, // hip R
        };
        for (ball_joints) |j| {
            const jid = try world.createSphericalJointWorld(b[j.a], b[j.c], at(root, j.x, j.y, s), .{
                .enable_cone_limit = true,
                .cone_angle_rad = j.cone,
                .enable_twist_limit = j.twist > 0.0,
                .lower_twist_angle_rad = -j.twist,
                .upper_twist_angle_rad = j.twist,
            });
            var tracked = false;
            defer if (!tracked and world.isJointValid(jid)) world.destroyJoint(jid);
            try self.joints.append(allocator, jid);
            tracked = true;
        }

        const hinge_joints = [4]struct { a: usize, c: usize, x: f32, y: f32, lower: f32, upper: f32 }{
            .{ .a = 3, .c = 5, .x = 0.26, .y = 1.18, .lower = -2.4, .upper = 0.0 }, // elbow L
            .{ .a = 4, .c = 6, .x = -0.26, .y = 1.18, .lower = -2.4, .upper = 0.0 }, // elbow R
            .{ .a = 7, .c = 9, .x = 0.11, .y = 0.44, .lower = 0.0, .upper = 2.4 }, // knee L
            .{ .a = 8, .c = 10, .x = -0.11, .y = 0.44, .lower = 0.0, .upper = 2.4 }, // knee R
        };
        for (hinge_joints) |j| {
            const jid = try world.createRevoluteJointWorld(b[j.a], b[j.c], at(root, j.x, j.y, s), .{
                .frame_a = hinge,
                .frame_b = hinge,
                .enable_limit = true,
                .lower_angle_rad = j.lower,
                .upper_angle_rad = j.upper,
            });
            var tracked = false;
            defer if (!tracked and world.isJointValid(jid)) world.destroyJoint(jid);
            try self.joints.append(allocator, jid);
            tracked = true;
        }

        return self;
    }

    /// Returns the body for a named part.
    pub fn getPart(self: *Ragdoll, part: RagdollPart) *RigidBody {
        return self.bodies.items[@as(usize, @intFromEnum(part))];
    }

    /// Kicks every part with the same impulse (e.g. an explosion push).
    pub fn applyImpulse(self: *Ragdoll, impulse: Vec3) void {
        for (self.bodies.items) |body| {
            body.applyImpulse(impulse);
        }
    }

    /// Teleports every part back to its spawn pose with zero velocity.
    /// The change is picked up by the solver on the next `step`.
    pub fn reset(self: *Ragdoll) void {
        for (self.bodies.items, 0..) |body, i| {
            body.mesh.position = self.home_positions.items[i];
            body.mesh.rotation = self.home_rotations.items[i];
            body.velocity = Vec3.zero;
            body.angular_velocity = Vec3.zero;
        }
    }

    /// Copies the last solver-synced body transforms into the meshes.
    /// Normally `step` already does this; use it to refresh meshes after
    /// teleporting bodies or before rendering without stepping.
    pub fn syncMeshes(self: *Ragdoll) void {
        physics_mesh.syncMeshes(self.bodies.items);
    }

    /// Destroys joints, removes bodies from the world and frees meshes/lists.
    /// Scene meshes are unlinked from the scene first. Call before destroying
    /// the scene and the world.
    pub fn deinit(self: *Ragdoll, world: *PhysicsWorld) void {
        physics_mesh.teardown(
            self.allocator,
            world,
            self.scene,
            &self.joints,
            &self.bodies,
            &self.meshes,
        );
        self.home_positions.deinit(self.allocator);
        self.home_rotations.deinit(self.allocator);
    }
};
