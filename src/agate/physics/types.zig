//! Self-contained public option/event types for the physics modules.
//! Extracted verbatim from `physics.zig`; behavior unchanged.
const math = @import("math");
const Vec3 = math.Vec3;
const Quat = math.Quat;
const Mesh = @import("../mesh.zig").Mesh;
const c = @import("../c.zig").c;

pub const PickingInfo = struct {
    hit: bool = false,
    distance: f32 = 0.0,
    picked_point: Vec3 = Vec3.zero,
    picked_normal: Vec3 = Vec3.up,
    picked_mesh: ?*Mesh = null,
    /// Index into `Mesh.instances.items` when the hit landed on an
    /// instance of the picked mesh; null for plain (non-instanced) hits
    /// and for misses. Defaulted so existing literals stay valid.
    picked_instance: ?usize = null,
};

pub const ColliderType = enum {
    box,
    sphere,
    capsule,
    /// Convex hull computed from `Mesh.cpu_positions`.
    hull,
    /// Triangle mesh collider from `Mesh.cpu_positions` + `Mesh.cpu_indices`.
    /// Static bodies only (Box3D restriction).
    mesh,
    /// Terrain collider. Create via `PhysicsWorld.createHeightField`.
    heightfield,
};

pub const HeightFieldOptions = struct {
    /// x/z cell size and y height multiplier.
    scale: Vec3 = Vec3.new(1.0, 1.0, 1.0),
    clockwise_winding: bool = false,
};

/// Shape collision filter (Box3D b3Filter). Category/mask use bit sets;
/// two shapes collide when (catA & maskB) != 0 and (catB & maskA) != 0.
/// A non-zero group_index overrides the mask: negative never collides,
/// positive always collides.
pub const CollisionFilter = struct {
    category_bits: u64 = 0x0000000000000001,
    mask_bits: u64 = 0xFFFFFFFFFFFFFFFF,
    group_index: i32 = 0,
};

pub fn toB3Filter(f: CollisionFilter) c.b3Filter {
    return .{
        .categoryBits = f.category_bits,
        .maskBits = f.mask_bits,
        .groupIndex = f.group_index,
    };
}

pub fn sameFilter(a: CollisionFilter, b: CollisionFilter) bool {
    return a.category_bits == b.category_bits and
        a.mask_bits == b.mask_bits and
        a.group_index == b.group_index;
}

/// Optional body setup for `PhysicsWorld.createBodyWith`.
pub const BodyOptions = struct {
    filter: CollisionFilter = .{},
    is_sensor: bool = false,
    enable_sensor_events: bool = false,
    enable_contact_events: bool = false,
    enable_hit_events: bool = false,
};

/// Options for the `PhysicsWorld.add*Shape` compound helpers.
pub const ChildShapeOptions = struct {
    offset: Vec3 = Vec3.zero,
    friction: ?f32 = null,
    restitution: ?f32 = null,
    filter: ?CollisionFilter = null,
    is_sensor: ?bool = null,
    enable_sensor_events: ?bool = null,
    enable_contact_events: ?bool = null,
    enable_hit_events: ?bool = null,
};

pub const JointId = c.b3JointId;

pub const DistanceJointOptions = struct {
    collide_connected: bool = false,
    length: ?f32 = null, // if null, auto-computed from anchor distance
    enable_spring: bool = false,
    hertz: f32 = 4.0,
    damping_ratio: f32 = 0.5,
    min_length: ?f32 = null,
    max_length: ?f32 = null,
};

pub const SphericalJointOptions = struct {
    collide_connected: bool = false,
    enable_spring: bool = false,
    hertz: f32 = 4.0,
    damping_ratio: f32 = 0.5,
    enable_cone_limit: bool = false,
    cone_angle_rad: f32 = 0.0,
    enable_twist_limit: bool = false,
    lower_twist_angle_rad: f32 = 0.0,
    upper_twist_angle_rad: f32 = 0.0,
};

pub const RevoluteJointOptions = struct {
    collide_connected: bool = false,
    /// Joint frames. Rotation happens around frame A z-axis (world z when
    /// both bodies are unrotated and frames are identity).
    frame_a: Quat = Quat.identity,
    frame_b: Quat = Quat.identity,
    enable_spring: bool = false,
    hertz: f32 = 4.0,
    damping_ratio: f32 = 0.5,
    enable_limit: bool = false,
    lower_angle_rad: f32 = 0.0,
    upper_angle_rad: f32 = 0.0,
    enable_motor: bool = false,
    motor_speed_rad: f32 = 0.0,
    max_motor_torque: f32 = 0.0,
};

/// Options for `PhysicsWorld.createWheelJoint` (body A = chassis).
pub const WheelJointOptions = struct {
    collide_connected: bool = false,
    /// Joint frames. Suspension/steering act around frame A x-axis, the wheel
    /// spins around frame B z-axis.
    frame_a: Quat = Quat.identity,
    frame_b: Quat = Quat.identity,
    enable_suspension: bool = true,
    suspension_hertz: f32 = 7.0,
    suspension_damping: f32 = 0.7,
    enable_suspension_limit: bool = true,
    lower_suspension_limit: f32 = -0.25,
    upper_suspension_limit: f32 = 0.25,
    enable_spin_motor: bool = false,
    spin_speed_rad: f32 = 0.0,
    max_spin_torque: f32 = 0.0,
    enable_steering: bool = false,
    steering_hertz: f32 = 8.0,
    steering_damping: f32 = 1.0,
    target_steering_angle_rad: f32 = 0.0,
    max_steering_torque: f32 = 0.0,
    enable_steering_limit: bool = false,
    lower_steering_limit_rad: f32 = 0.0,
    upper_steering_limit_rad: f32 = 0.0,
};

/// Options for `PhysicsWorld.createMotorJoint`. A motor joint drives body B
/// with a velocity motor plus an optional pose spring. With zero spring and
/// force settings it is inert; set a linear velocity every frame to drag a
/// body toward a target (see SandboxScene drag interaction for the pattern).
pub const MotorJointOptions = struct {
    collide_connected: bool = false,
    linear_velocity: Vec3 = Vec3.zero,
    max_velocity_force: f32 = 0.0,
    angular_velocity_rad: Vec3 = Vec3.zero,
    max_velocity_torque: f32 = 0.0,
    linear_hertz: f32 = 0.0,
    linear_damping: f32 = 0.0,
    max_spring_force: f32 = 0.0,
    angular_hertz: f32 = 0.0,
    angular_damping: f32 = 0.0,
    max_spring_torque: f32 = 0.0,
};

/// Options for `PhysicsWorld.createWeldJoint`. Rigidly attaches two bodies
/// at their creation relative pose (zero hertz = maximum stiffness, springs
/// mimic soft-body behavior). Pair with `jointConstraintForce` polling to
/// implement breakable joints.
pub const WeldJointOptions = struct {
    collide_connected: bool = false,
    linear_hertz: f32 = 0.0,
    angular_hertz: f32 = 0.0,
    linear_damping: f32 = 0.0,
    angular_damping: f32 = 0.0,
};

/// Options for `PhysicsWorld.createParallelJoint`. Constrains the angle
/// between the z-axis of body A and the z-axis of body B with a spring.
/// Useful to keep a body upright.
pub const ParallelJointOptions = struct {
    collide_connected: bool = false,
    hertz: f32 = 4.0,
    damping_ratio: f32 = 0.7,
    max_torque: f32 = 0.0,
};

/// Options for `PhysicsWorld.createPrismaticJoint` (slider along frame A
/// x-axis, rotation locked). Translation is measured in meters from the
/// creation pose.
pub const PrismaticJointOptions = struct {
    collide_connected: bool = false,
    frame_a: Quat = Quat.identity,
    frame_b: Quat = Quat.identity,
    enable_spring: bool = false,
    hertz: f32 = 4.0,
    damping_ratio: f32 = 0.5,
    enable_limit: bool = false,
    lower_translation: f32 = 0.0,
    upper_translation: f32 = 0.0,
    enable_motor: bool = false,
    motor_speed: f32 = 0.0,
    max_motor_force: f32 = 0.0,
};

/// One CPU-side debug line segment (a line list for an external renderer or
/// a future debug pass). The default color is dynamic-body green.
pub const DebugLine = struct {
    a: Vec3,
    b: Vec3,
    color: [3]f32 = .{ 0.1, 0.9, 0.3 },
};
