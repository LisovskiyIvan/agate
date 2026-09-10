const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const Ray = math.Ray;
const RayHit = math.RayHit;
const Quat = math.Quat;
const Mat4 = math.Mat4;
const Mesh = @import("mesh.zig").Mesh;
const c = @import("c.zig").c;

pub const PickingInfo = struct {
    hit: bool = false,
    distance: f32 = 0.0,
    picked_point: Vec3 = Vec3.zero,
    picked_normal: Vec3 = Vec3.up,
    picked_mesh: ?*Mesh = null,
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

pub const PhysicsRayHit = struct {
    hit: bool = false,
    point: Vec3 = Vec3.zero,
    normal: Vec3 = Vec3.up,
    distance: f32 = 0.0,
    body: ?*RigidBody = null,
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

fn toB3Filter(f: CollisionFilter) c.b3Filter {
    return .{
        .categoryBits = f.category_bits,
        .maskBits = f.mask_bits,
        .groupIndex = f.group_index,
    };
}

fn sameFilter(a: CollisionFilter, b: CollisionFilter) bool {
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

/// Sensor overlap event. `began` reports begin/end of the overlap.
/// `visitor` is null when the other shape does not belong to a tracked body.
pub const SensorEvent = struct {
    sensor: ?*RigidBody,
    visitor: ?*RigidBody,
    began: bool,
};

/// Contact begin/end event between two shapes.
pub const ContactEvent = struct {
    a: ?*RigidBody,
    b: ?*RigidBody,
    began: bool,
};

/// Contact hit event, reported when the approach speed exceeds
/// `PhysicsWorld.hit_event_threshold`.
pub const ContactHitEvent = struct {
    a: ?*RigidBody,
    b: ?*RigidBody,
    point: Vec3 = Vec3.zero,
    normal: Vec3 = Vec3.up,
    approach_speed: f32 = 0.0,
};

/// An extra collision shape attached to a `RigidBody` (compound collider).
/// Dimensions are in the body's local (unscaled) units and are rebaked when
/// the mesh scale changes.
pub const ChildShape = struct {
    pub const Kind = union(enum) {
        box: Vec3, // half extents
        sphere: f32, // radius (center = offset)
        capsule: struct { half_height: f32, radius: f32 }, // Y axis around offset
        hull: []Vec3, // owned copy of local points
    };

    kind: Kind,
    offset: Vec3 = Vec3.zero,
    shape_id: c.b3ShapeId = .{ .index1 = 0, .world0 = 0, .generation = 0 },
    scaled_volume: f32 = 0.0,

    // Optional per-shape overrides (null = inherit the body value live).
    friction: ?f32 = null,
    restitution: ?f32 = null,
    filter: ?CollisionFilter = null,
    is_sensor: ?bool = null,
    sensor_events: ?bool = null,
    contact_events: ?bool = null,
    hit_events: ?bool = null,

    // Last applied resolved values (for change detection).
    last_friction: f32 = 0.0,
    last_restitution: f32 = 0.0,
    last_filter: CollisionFilter = .{},
    last_sensor: bool = false,
    last_sensor_events: bool = false,
    last_contact_events: bool = false,
    last_hit_events: bool = false,

    fn freeOwned(self: *ChildShape, allocator: std.mem.Allocator) void {
        if (self.kind == .hull) {
            allocator.free(self.kind.hull);
        }
    }

    fn resolvedFriction(self: *const ChildShape, body: *const RigidBody) f32 {
        return self.friction orelse body.friction;
    }

    fn resolvedRestitution(self: *const ChildShape, body: *const RigidBody) f32 {
        return self.restitution orelse body.restitution;
    }

    fn resolvedFilter(self: *const ChildShape, body: *const RigidBody) CollisionFilter {
        return self.filter orelse body.filter;
    }

    fn resolvedIsSensor(self: *const ChildShape, body: *const RigidBody) bool {
        return self.is_sensor orelse body.is_sensor;
    }

    fn resolvedSensorEvents(self: *const ChildShape, body: *const RigidBody) bool {
        return self.resolvedIsSensor(body) or (self.sensor_events orelse body.enable_sensor_events);
    }

    fn resolvedContactEvents(self: *const ChildShape, body: *const RigidBody) bool {
        return self.contact_events orelse body.enable_contact_events;
    }

    fn resolvedHitEvents(self: *const ChildShape, body: *const RigidBody) bool {
        return self.hit_events orelse body.enable_hit_events;
    }
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

/// Kinematic capsule character controller built on Box3D's mover API
/// (collide + plane solve + velocity clip). It is NOT a rigid body: it
/// slides around dynamic bodies without pushing them and is invisible to
/// sensors. `position` is the feet point (capsule bottom).
pub const CharacterController = struct {
    position: Vec3 = Vec3.zero,
    velocity: Vec3 = Vec3.zero,
    radius: f32 = 0.3,
    height: f32 = 1.5, // total capsule height
    move_speed: f32 = 4.5,
    jump_speed: f32 = 6.0,
    gravity: f32 = 18.0,
    slope_limit_cos: f32 = 0.7,
    is_grounded: bool = false,
    ground_normal: Vec3 = Vec3.up,

    const max_planes = 16;

    const PlaneCollector = struct {
        planes: []c.b3CollisionPlane,
        count: usize = 0,
    };

    pub fn init(position: Vec3, radius: f32, height: f32) CharacterController {
        return .{ .position = position, .radius = radius, .height = height };
    }

    /// Advances the controller. `wish_dir` is the desired horizontal move
    /// direction (y is ignored, longer than 1 is normalized). `jump_pressed`
    /// is an edge trigger consumed while grounded.
    pub fn move(self: *CharacterController, world: *PhysicsWorld, wish_dir: Vec3, jump_pressed: bool, dt: f32) void {
        if (dt <= 0.0) return;
        world.syncWorldParams();
        const h = @min(dt, 1.0 / 30.0);

        var plane_buf: [max_planes]c.b3CollisionPlane = undefined;
        var collector = PlaneCollector{ .planes = &plane_buf };
        var mover = c.b3Capsule{
            .center1 = .{ .x = 0.0, .y = self.radius, .z = 0.0 },
            .center2 = .{ .x = 0.0, .y = @max(self.height - self.radius, self.radius), .z = 0.0 },
            .radius = self.radius,
        };
        const filter = c.b3DefaultQueryFilter();
        c.b3World_CollideMover(world.world_id, toB3Pos(self.position), &mover, filter, &planeCollectFcn, &collector);

        self.is_grounded = false;
        self.ground_normal = Vec3.up;
        var best_up: f32 = self.slope_limit_cos;
        for (plane_buf[0..collector.count]) |cp| {
            const n = fromB3Vec(cp.plane.normal);
            if (n.y > best_up) {
                best_up = n.y;
                self.ground_normal = n;
                self.is_grounded = true;
            }
        }

        var vx = wish_dir.x;
        var vz = wish_dir.z;
        const wl = @sqrt(vx * vx + vz * vz);
        if (wl > 1.0) {
            vx /= wl;
            vz /= wl;
        }
        vx *= self.move_speed;
        vz *= self.move_speed;

        var vy = self.velocity.y;
        if (self.is_grounded) {
            if (jump_pressed) {
                vy = self.jump_speed;
                self.is_grounded = false;
            } else {
                vy = -2.0; // stick to the ground on slopes and ledges
            }
        } else {
            vy = @max(vy - self.gravity * h, -30.0);
        }
        self.velocity = Vec3.new(vx, vy, vz);

        const target = toB3Vec(Vec3.new(vx * h, vy * h, vz * h));
        const solved = c.b3SolvePlanes(target, &plane_buf, @intCast(collector.count));
        self.position = self.position.add(fromB3Vec(solved.delta));

        const clipped = c.b3ClipVector(toB3Vec(self.velocity), &plane_buf, @intCast(collector.count));
        self.velocity = fromB3Vec(clipped);
    }
};

fn planeCollectFcn(
    shape_id: c.b3ShapeId,
    planes: [*c]const c.b3PlaneResult,
    plane_count: c_int,
    context: ?*anyopaque,
) callconv(.c) bool {
    _ = shape_id;
    const ctx = context orelse return false;
    const collector: *CharacterController.PlaneCollector = @ptrCast(@alignCast(ctx));
    var i: c_int = 0;
    while (i < plane_count) : (i += 1) {
        if (collector.count >= collector.planes.len) return false;
        const pr = planes[@intCast(i)];
        collector.planes[collector.count] = .{
            .plane = pr.plane,
            .pushLimit = std.math.floatMax(f32),
            .push = 0.0,
            .clipVelocity = true,
        };
        collector.count += 1;
    }
    return true;
}

/// Options for `PhysicsWorld.createRope`.
pub const RopeOptions = struct {
    collider: ColliderType = .capsule,
    segment_mass: f32 = 0.4,
    restitution: f32 = 0.1,
    friction: f32 = 0.4,
    /// Joint max length = spacing * max_stretch. 1.0 means rigid links.
    max_stretch: f32 = 1.05,
    /// Optional joint min length factor (rigid chain links resist compression).
    min_compress: ?f32 = null,
    /// Bungee mode: springy joints instead of limited ones.
    spring: bool = false,
    spring_hertz: f32 = 3.0,
    spring_damping: f32 = 0.6,
    collide_connected: bool = false,
    /// Bodies the first/last segment is pinned to (null = free end).
    pin_start: ?*RigidBody = null,
    pin_end: ?*RigidBody = null,
    /// World-space anchor points on the pin bodies (null = rope endpoint).
    pin_start_anchor: ?Vec3 = null,
    pin_end_anchor: ?Vec3 = null,
    enable_sensor_events: bool = false,
    enable_contact_events: bool = false,
    enable_hit_events: bool = false,
};

/// A rope/chain: dynamic bodies built from caller meshes, linked by distance
/// joints, optionally pinned to other bodies at the ends. The rope owns its
/// segment bodies; pinned bodies stay external.
pub const Rope = struct {
    bodies: std.ArrayListUnmanaged(*RigidBody) = .empty,
    /// joints[i] links bodies[i] to bodies[i + 1].
    joints: std.ArrayListUnmanaged(JointId) = .empty,
    start_joint: ?JointId = null,
    end_joint: ?JointId = null,
    from: Vec3 = Vec3.zero,
    to: Vec3 = Vec3.zero,
    spacing: f32 = 0.5,
    max_stretch: f32 = 1.05,
    min_compress: ?f32 = null,
    spring: bool = false,
    spring_hertz: f32 = 3.0,
    spring_damping: f32 = 0.6,
    collide_connected: bool = false,
    pin_start: ?*RigidBody = null,
    pin_end: ?*RigidBody = null,
    pin_start_anchor: ?Vec3 = null,
    pin_end_anchor: ?Vec3 = null,
    start_rest: f32 = 0.001,
    end_rest: f32 = 0.001,

    pub fn deinit(self: *Rope, world: *PhysicsWorld) void {
        // Links may already be severed (detachAt) or invalidated by body
        // removal; destroying an invalid joint fails a Box3D assert.
        if (self.start_joint) |j| {
            if (c.b3Joint_IsValid(j)) world.destroyJoint(j);
        }
        if (self.end_joint) |j| {
            if (c.b3Joint_IsValid(j)) world.destroyJoint(j);
        }
        for (self.joints.items) |j| {
            if (c.b3Joint_IsValid(j)) world.destroyJoint(j);
        }
        self.joints.deinit(world.allocator);
        for (self.bodies.items) |b| {
            world.removeBody(b);
        }
        self.bodies.deinit(world.allocator);
        self.* = .{};
    }

    /// Severs the link between bodies[index] and bodies[index + 1].
    /// Out-of-range or already-severed links are ignored.
    pub fn detachAt(self: *Rope, world: *PhysicsWorld, index: usize) void {
        if (index >= self.joints.items.len) return;
        const j = self.joints.items[index];
        if (!c.b3Joint_IsValid(j)) return;
        world.destroyJoint(j);
    }

    /// Recreates every missing link/pin joint (after detachAt or reset).
    pub fn repair(self: *Rope, world: *PhysicsWorld) void {
        if (self.bodies.items.len == 0) return;
        if (self.pin_start) |pin| {
            if (self.start_joint == null or !c.b3Joint_IsValid(self.start_joint.?)) {
                const anchor = self.pin_start_anchor orelse self.from;
                self.start_joint = world.createDistanceJointWorld(
                    pin,
                    self.bodies.items[0],
                    anchor,
                    self.bodies.items[0].mesh.position,
                    .{ .length = self.start_rest },
                ) catch null;
            }
        }
        for (self.joints.items, 0..) |j, i| {
            if (!c.b3Joint_IsValid(j)) {
                self.joints.items[i] = world.createDistanceJoint(
                    self.bodies.items[i],
                    self.bodies.items[i + 1],
                    Vec3.zero,
                    Vec3.zero,
                    self.linkJointOptions(),
                ) catch continue;
            }
        }
        if (self.pin_end) |pin| {
            if (self.end_joint == null or !c.b3Joint_IsValid(self.end_joint.?)) {
                const anchor = self.pin_end_anchor orelse self.to;
                const last = self.bodies.items[self.bodies.items.len - 1];
                self.end_joint = world.createDistanceJointWorld(
                    last,
                    pin,
                    last.mesh.position,
                    anchor,
                    .{ .length = self.end_rest },
                ) catch null;
            }
        }
    }

    /// Lays the segments back on the creation line, zeroes velocities and
    /// repairs severed joints.
    pub fn reset(self: *Rope, world: *PhysicsWorld) void {
        const n = self.bodies.items.len;
        if (n == 0) return;
        for (self.bodies.items, 0..) |b, i| {
            const t = if (n > 1) @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n - 1)) else 0.0;
            b.mesh.position = Vec3.new(
                self.from.x + (self.to.x - self.from.x) * t,
                self.from.y + (self.to.y - self.from.y) * t,
                self.from.z + (self.to.z - self.from.z) * t,
            );
            b.mesh.rotation = Vec3.zero;
            b.velocity = Vec3.zero;
            b.angular_velocity = Vec3.zero;
        }
        self.repair(world);
    }

    fn linkJointOptions(self: *const Rope) DistanceJointOptions {
        return .{
            .length = self.spacing,
            .max_length = self.spacing * self.max_stretch,
            .min_length = if (self.min_compress) |m| self.spacing * m else null,
            .collide_connected = self.collide_connected,
            .enable_spring = self.spring,
            .hertz = self.spring_hertz,
            .damping_ratio = self.spring_damping,
        };
    }
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

// Fixed simulation rate Box3D integrates at; step() accumulates frame dt
// towards it (Fix Your Timestep) instead of feeding variable dt to the solver.
const step_h: f32 = 1.0 / 60.0;
// Our damping fields are ~per-1/60-step velocity retention factors;
// Box3D damping is per-second. 60x maps one to the other.
const damp_scale: f32 = 60.0;
const deg2rad: f32 = std.math.pi / 180.0;
const rad2deg: f32 = 180.0 / std.math.pi;
// Thin-shape guard: Box3D hulls dislike degenerate half-extents.
const min_half_extent: f32 = 0.01;

fn toB3Vec(v: Vec3) c.b3Vec3 {
    return .{ .x = v.x, .y = v.y, .z = v.z };
}

fn fromB3Vec(v: c.b3Vec3) Vec3 {
    return Vec3.new(v.x, v.y, v.z);
}

fn toB3Pos(v: Vec3) c.b3Pos {
    return .{ .x = @floatCast(v.x), .y = @floatCast(v.y), .z = @floatCast(v.z) };
}

fn fromB3Pos(p: c.b3Pos) Vec3 {
    return Vec3.new(@floatCast(p.x), @floatCast(p.y), @floatCast(p.z));
}

fn toB3Quat(q: Quat) c.b3Quat {
    return .{ .v = .{ .x = q.x, .y = q.y, .z = q.z }, .s = q.w };
}

fn fromB3Quat(q: c.b3Quat) Quat {
    return .{ .x = q.v.x, .y = q.v.y, .z = q.v.z, .w = q.s };
}

pub const RigidBody = struct {
    mesh: *Mesh,
    collider: ColliderType = .box,
    mass: f32 = 1.0,
    inv_mass: f32 = 1.0,

    // Live state, mirrored with the solver:
    // - solver -> fields on every stepped frame (pull)
    // - field writes are picked up and pushed on the next step()
    // So `body.velocity = ...` and `mesh.position = ...` keep working,
    // including teleport-style resets. Pushes canonicalize through the
    // solver (rotation re-extracted as Euler), keeping change detection stable.
    velocity: Vec3 = Vec3.zero, // m/s
    angular_velocity: Vec3 = Vec3.zero, // deg/s

    restitution: f32 = 0.6, // Bounciness [0..1]
    friction: f32 = 0.25,
    linear_damping: f32 = 0.015,
    angular_damping: f32 = 0.04,

    use_gravity: bool = true,
    is_grounded: bool = false,
    enabled: bool = true,

    // Collision filtering & events.
    filter: CollisionFilter = .{},
    is_sensor: bool = false,
    /// Enable overlap events for this shape. Required on BOTH the sensor and
    /// the visitor for sensor events to fire (Box3D rule).
    enable_sensor_events: bool = false,
    enable_contact_events: bool = false,
    enable_hit_events: bool = false,

    // Base (unscaled) shape dims; live dims = base * mesh.scaling.
    base_extents: Vec3 = Vec3.new(0.5, 0.5, 0.5),
    base_radius: f32 = 0.5,
    // Picking helpers (base dims, like before).
    box_extents: Vec3 = Vec3.new(0.5, 0.5, 0.5),
    sphere_radius: f32 = 0.5,

    body_id: c.b3BodyId = .{ .index1 = 0, .world0 = 0, .generation = 0 },
    shape_id: c.b3ShapeId = .{ .index1 = 0, .world0 = 0, .generation = 0 },

    // Geometry owned by us whose lifetime must cover the shape:
    // triangle mesh and height field shapes reference this data directly.
    // (Hull shape data is cloned into the Box3D world database.)
    mesh_data: ?*c.b3MeshData = null,
    heightfield_data: ?*c.b3HeightFieldData = null,

    // Extra collision shapes on this body (compound collider).
    child_shapes: std.ArrayListUnmanaged(ChildShape) = .empty,

    // Change-detection shadows; always equal the last solver-synced values.
    last_pos: Vec3 = Vec3.zero,
    last_rot: Vec3 = Vec3.zero,
    last_scale: Vec3 = Vec3.one,
    last_vel: Vec3 = Vec3.zero,
    last_angvel: Vec3 = Vec3.zero, // deg/s, canonical
    last_rest: f32 = 0.6,
    last_fric: f32 = 0.25,
    last_damp_l: f32 = 0.015,
    last_damp_a: f32 = 0.04,
    last_grav: bool = true,
    last_enabled: bool = true,
    last_mass: f32 = 1.0,
    last_filter: CollisionFilter = .{},
    last_sensor_events: bool = false,
    last_contact_events: bool = false,
    last_hit_events: bool = false,

    pub fn init(mesh: *Mesh, collider: ColliderType, mass: f32) RigidBody {
        const inv_m = if (mass > 0.0) 1.0 / mass else 0.0;
        const aabb = mesh.local_bounding_box;
        const ext = aabb.extents();
        const rad = @max(ext.x, @max(ext.y, ext.z));
        const base_ext = if (ext.lengthSq() > 1e-6) ext else Vec3.new(0.5, 0.5, 0.5);
        const base_rad = if (rad > 1e-4) rad else 0.5;

        return .{
            .mesh = mesh,
            .collider = collider,
            .mass = mass,
            .inv_mass = inv_m,
            .base_extents = base_ext,
            .base_radius = base_rad,
            .box_extents = base_ext,
            .sphere_radius = base_rad,
            .restitution = 0.6,
            .friction = 0.25,
            .linear_damping = 0.015,
            .angular_damping = 0.04,
            .last_pos = mesh.position,
            .last_rot = mesh.rotation,
            .last_scale = mesh.scaling,
            .last_mass = mass,
        };
    }

    fn shapeVolume(b: *const RigidBody, scale: Vec3) f32 {
        return switch (b.collider) {
            .box => 8.0 * (b.base_extents.x * scale.x) * (b.base_extents.y * scale.y) * (b.base_extents.z * scale.z),
            .sphere => blk: {
                const r = b.base_radius * scale.x;
                break :blk 4.1887902 * r * r * r;
            },
            .capsule => blk: {
                // Y-capsule: radius is the horizontal extent, not base_radius
                // (which stores the max extent for spheres).
                const r = @max(@min(b.base_extents.x, b.base_extents.z) * scale.x, min_half_extent);
                const hh = @max(0.0, b.base_extents.y * scale.y - r);
                const sph_vol = 4.1887902 * r * r * r;
                const cyl_vol = 6.2831853 * r * r * hh;
                break :blk sph_vol + cyl_vol;
            },
            .hull => blk: {
                const hull = c.b3Shape_GetHull(b.shape_id);
                break :blk if (hull != null) hull.*.volume else 0.0;
            },
            .mesh, .heightfield => 0.0,
        };
    }

    fn densityForMass(mass: f32, volume: f32) f32 {
        if (mass <= 0.0) return 0.0;
        return mass / @max(volume, 1e-6);
    }

    pub fn applyImpulse(self: *RigidBody, impulse: Vec3) void {
        if (self.mass <= 0.0) return;
        c.b3Body_ApplyLinearImpulseToCenter(self.body_id, toB3Vec(impulse), true);
        self.velocity = fromB3Vec(c.b3Body_GetLinearVelocity(self.body_id));
        self.last_vel = self.velocity;
        self.is_grounded = false;
    }

    pub fn applyTorqueImpulse(self: *RigidBody, torque_impulse: Vec3) void {
        if (self.mass <= 0.0) return;
        const m = c.b3Body_GetMass(self.body_id);
        if (m <= 0.0) return;
        const w = fromB3Vec(c.b3Body_GetAngularVelocity(self.body_id));
        c.b3Body_SetAngularVelocity(self.body_id, toB3Vec(w.add(torque_impulse.scale(deg2rad / m))));
        const got = fromB3Vec(c.b3Body_GetAngularVelocity(self.body_id)).scale(rad2deg);
        self.angular_velocity = got;
        self.last_angvel = got;
    }

    pub fn applyForce(self: *RigidBody, force: Vec3, dt: f32) void {
        if (self.mass <= 0.0) return;
        const m = @max(c.b3Body_GetMass(self.body_id), 1e-6);
        const v = fromB3Vec(c.b3Body_GetLinearVelocity(self.body_id)).add(force.scale(dt / m));
        c.b3Body_SetLinearVelocity(self.body_id, toB3Vec(v));
        c.b3Body_SetAwake(self.body_id, true);
        self.velocity = fromB3Vec(c.b3Body_GetLinearVelocity(self.body_id));
        self.last_vel = self.velocity;
    }

    pub fn setMass(self: *RigidBody, mass: f32) void {
        self.mass = mass;
        self.inv_mass = if (mass > 0.0) 1.0 / mass else 0.0;
        // Density/type sync happens in pushBody() on the next step().
    }

    /// Transforms a point in world space into the body's local space.
    pub fn worldToLocal(self: *const RigidBody, world_point: Vec3) Vec3 {
        const q = Quat.fromEulerDeg(self.mesh.rotation);
        const rel = world_point.sub(self.mesh.position);
        return q.conjugate().rotateVec(rel);
    }

    /// Transforms a point in the body's local space into world space.
    pub fn localToWorld(self: *const RigidBody, local_point: Vec3) Vec3 {
        const q = Quat.fromEulerDeg(self.mesh.rotation);
        return self.mesh.position.add(q.rotateVec(local_point));
    }
};

/// One CPU-side debug line segment (a line list for an external renderer or
/// a future debug pass). The default color is dynamic-body green.
pub const DebugLine = struct {
    a: Vec3,
    b: Vec3,
    color: [3]f32 = .{ 0.1, 0.9, 0.3 },
};

// Debug wireframe colors: dynamic bodies green, static/kinematic
// (mass <= 0) white, sensors yellow. The sensor tint wins over the other two.
const debug_dynamic_color: [3]f32 = .{ 0.1, 0.9, 0.3 };
const debug_static_color: [3]f32 = .{ 0.9, 0.9, 0.9 };
const debug_sensor_color: [3]f32 = .{ 0.95, 0.8, 0.15 };

// Default segments per debug circle (sphere great circles, capsule cap
// rings); override per world with PhysicsWorld.debug_circle_segments.
const default_debug_circle_segments: usize = 24;
const min_debug_circle_segments: usize = 4;
// Unit-circle samples precomputed on the stack per appendDebugLines call.
// Larger counts use per-segment trig instead (same angles, same points).
const debug_unit_stack_max: usize = 128;

// 12 edges of a box, shared by the scaled-box and world-AABB wireframes.
const debug_box_edges: [12][2]usize = .{
    .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 0 },
    .{ 4, 5 }, .{ 5, 6 }, .{ 6, 7 }, .{ 7, 4 },
    .{ 0, 4 }, .{ 1, 5 }, .{ 2, 6 }, .{ 3, 7 },
};

const DebugCirclePlane = enum { xy, xz, yz };

// One unit-circle sample shared by every debug ring in a single
// appendDebugLines call; scaled by the shape radius when emitting.
const DebugUnit = struct { c: f32, s: f32 };

// Ring source for the circle helpers: the precomputed unit table when the
// segment count fits the stack buffer, otherwise direct per-segment trig.
// Both evaluate the same angles, so the emitted points match exactly.
const DebugRings = struct {
    segs: usize,
    unit: ?[]const DebugUnit,
};

// Local Y-capsule dims in body units, mirroring RigidBody.shapeVolume:
// radius is the horizontal extent, half height is the leftover Y extent.
fn debugCapsuleRadius(base_extents: Vec3) f32 {
    return @min(base_extents.x, base_extents.z);
}

fn debugCapsuleHalfHeight(base_extents: Vec3) f32 {
    return @max(0.0, base_extents.y - @min(base_extents.x, base_extents.z));
}

// Appends the 12 edges of the box (center, half extents) in body units,
// transformed to world space by the mesh world matrix (which carries the
// body scale, so callers pass unscaled dims). Allocation-free: the caller
// reserves debugLineCount() entries up front.
fn appendDebugBoxLines(
    out: *std.ArrayListUnmanaged(DebugLine),
    wm: Mat4,
    center: Vec3,
    half_extents: Vec3,
    color: [3]f32,
) void {
    const hx = half_extents.x;
    const hy = half_extents.y;
    const hz = half_extents.z;
    const corners = [8]Vec3{
        Vec3.new(center.x - hx, center.y - hy, center.z - hz),
        Vec3.new(center.x + hx, center.y - hy, center.z - hz),
        Vec3.new(center.x + hx, center.y + hy, center.z - hz),
        Vec3.new(center.x - hx, center.y + hy, center.z - hz),
        Vec3.new(center.x - hx, center.y - hy, center.z + hz),
        Vec3.new(center.x + hx, center.y - hy, center.z + hz),
        Vec3.new(center.x + hx, center.y + hy, center.z + hz),
        Vec3.new(center.x - hx, center.y + hy, center.z + hz),
    };
    for (debug_box_edges) |e| {
        out.appendAssumeCapacity(.{
            .a = wm.transformPoint(corners[e[0]]),
            .b = wm.transformPoint(corners[e[1]]),
            .color = color,
        });
    }
}

fn debugCirclePoint(center: Vec3, radius: f32, plane: DebugCirclePlane, angle: f32) Vec3 {
    const cx = @cos(angle) * radius;
    const sx = @sin(angle) * radius;
    return switch (plane) {
        .xy => Vec3.new(center.x + cx, center.y + sx, center.z),
        .xz => Vec3.new(center.x + cx, center.y, center.z + sx),
        .yz => Vec3.new(center.x, center.y + cx, center.z + sx),
    };
}

// Same math as debugCirclePoint but from a precomputed unit sample, so
// shared ring endpoints reuse one cos/sin pair instead of recomputing it.
fn debugCirclePointUnit(center: Vec3, radius: f32, plane: DebugCirclePlane, u: DebugUnit) Vec3 {
    const cx = u.c * radius;
    const sx = u.s * radius;
    return switch (plane) {
        .xy => Vec3.new(center.x + cx, center.y + sx, center.z),
        .xz => Vec3.new(center.x + cx, center.y, center.z + sx),
        .yz => Vec3.new(center.x, center.y + cx, center.z + sx),
    };
}

// Appends one ring of segs lines in the given local plane, transformed to
// world space by the mesh world matrix. Allocation-free (see above).
fn appendDebugCircleLines(
    out: *std.ArrayListUnmanaged(DebugLine),
    wm: Mat4,
    center: Vec3,
    radius: f32,
    plane: DebugCirclePlane,
    color: [3]f32,
    rings: DebugRings,
) void {
    if (rings.unit) |unit| {
        var i: usize = 0;
        while (i < rings.segs) : (i += 1) {
            out.appendAssumeCapacity(.{
                .a = wm.transformPoint(debugCirclePointUnit(center, radius, plane, unit[i])),
                .b = wm.transformPoint(debugCirclePointUnit(center, radius, plane, unit[i + 1])),
                .color = color,
            });
        }
        return;
    }
    var i: usize = 0;
    while (i < rings.segs) : (i += 1) {
        const t0 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(rings.segs)) * 2.0 * std.math.pi;
        const t1 = @as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(rings.segs)) * 2.0 * std.math.pi;
        out.appendAssumeCapacity(.{
            .a = wm.transformPoint(debugCirclePoint(center, radius, plane, t0)),
            .b = wm.transformPoint(debugCirclePoint(center, radius, plane, t1)),
            .color = color,
        });
    }
}

// 3 orthogonal great circles: 3 * segs lines.
fn appendDebugSphereLines(
    out: *std.ArrayListUnmanaged(DebugLine),
    wm: Mat4,
    center: Vec3,
    radius: f32,
    color: [3]f32,
    rings: DebugRings,
) void {
    appendDebugCircleLines(out, wm, center, radius, .xy, color, rings);
    appendDebugCircleLines(out, wm, center, radius, .xz, color, rings);
    appendDebugCircleLines(out, wm, center, radius, .yz, color, rings);
}

// 4 side lines joining the capsule cap rings at the cardinal points. Kept
// on the exact k * pi/2 angles (not the unit table) so points stay
// bit-identical to the historical output.
fn appendDebugCapsuleSides(
    out: *std.ArrayListUnmanaged(DebugLine),
    wm: Mat4,
    center: Vec3,
    radius: f32,
    half_height: f32,
    color: [3]f32,
) void {
    var k: usize = 0;
    while (k < 4) : (k += 1) {
        const t = @as(f32, @floatFromInt(k)) * 0.5 * std.math.pi;
        const x = @cos(t) * radius;
        const z = @sin(t) * radius;
        out.appendAssumeCapacity(.{
            .a = wm.transformPoint(Vec3.new(center.x + x, center.y + half_height, center.z + z)),
            .b = wm.transformPoint(Vec3.new(center.x + x, center.y - half_height, center.z + z)),
            .color = color,
        });
    }
}

// Y-capsule: top/bottom cap rings plus 4 side lines joining them at the
// cardinal points: 2 * segs + 4 lines.
fn appendDebugCapsuleLines(
    out: *std.ArrayListUnmanaged(DebugLine),
    wm: Mat4,
    center: Vec3,
    radius: f32,
    half_height: f32,
    color: [3]f32,
    rings: DebugRings,
) void {
    const top = Vec3.new(center.x, center.y + half_height, center.z);
    const bottom = Vec3.new(center.x, center.y - half_height, center.z);
    appendDebugCircleLines(out, wm, top, radius, .xz, color, rings);
    appendDebugCircleLines(out, wm, bottom, radius, .xz, color, rings);
    appendDebugCapsuleSides(out, wm, center, radius, half_height, color);
}

// Wireframe box of a shape's world AABB (already world space, no transform).
// Invalid shapes are skipped.
fn appendDebugAabbLines(
    out: *std.ArrayListUnmanaged(DebugLine),
    shape_id: c.b3ShapeId,
    color: [3]f32,
) void {
    if (shape_id.index1 == 0) return;
    const aabb = c.b3Shape_GetAABB(shape_id);
    const lo = fromB3Vec(aabb.lowerBound);
    const hi = fromB3Vec(aabb.upperBound);
    const corners = [8]Vec3{
        Vec3.new(lo.x, lo.y, lo.z),
        Vec3.new(hi.x, lo.y, lo.z),
        Vec3.new(hi.x, hi.y, lo.z),
        Vec3.new(lo.x, hi.y, lo.z),
        Vec3.new(lo.x, lo.y, hi.z),
        Vec3.new(hi.x, lo.y, hi.z),
        Vec3.new(hi.x, hi.y, hi.z),
        Vec3.new(lo.x, hi.y, hi.z),
    };
    for (debug_box_edges) |e| {
        out.appendAssumeCapacity(.{ .a = corners[e[0]], .b = corners[e[1]], .color = color });
    }
}

fn primaryDebugLineCount(collider: ColliderType, segs: usize) usize {
    return switch (collider) {
        .box => 12,
        .sphere => 3 * segs,
        .capsule => 2 * segs + 4,
        .hull, .mesh, .heightfield => 12,
    };
}

fn childDebugLineCount(kind: ChildShape.Kind, segs: usize) usize {
    return switch (kind) {
        .box => 12,
        .sphere => 3 * segs,
        .capsule => 2 * segs + 4,
        // No cheap wireframe; skipped gracefully.
        .hull => 0,
    };
}

pub const PhysicsWorld = struct {
    allocator: std.mem.Allocator,
    gravity: Vec3 = Vec3.new(0.0, -9.81, 0.0),
    ground_y: ?f32 = -1.2,
    substeps: u32 = 4,
    /// Minimum approach speed (m/s) for contact hit events.
    hit_event_threshold: f32 = 1.0,
    /// Debug wireframe density: segments per circle ring (sphere great
    /// circles, capsule cap rings). Clamped to a minimum of 4. Default 24
    /// keeps the historical counts (sphere 72 lines, capsule 52 lines).
    debug_circle_segments: usize = default_debug_circle_segments,
    bodies: std.ArrayListUnmanaged(*RigidBody) = .empty,
    joints: std.ArrayListUnmanaged(c.b3JointId) = .empty,

    // Events collected from the most recent step() call.
    sensor_events: std.ArrayListUnmanaged(SensorEvent) = .empty,
    contact_events: std.ArrayListUnmanaged(ContactEvent) = .empty,
    contact_hit_events: std.ArrayListUnmanaged(ContactHitEvent) = .empty,

    world_id: c.b3WorldId,
    ground_body: ?c.b3BodyId = null,
    acc: f32 = 0.0,
    last_gravity: Vec3 = Vec3.new(0.0, -9.81, 0.0),
    last_ground_y: ?f32 = null,
    last_hit_threshold: f32 = 1.0,

    pub fn init(allocator: std.mem.Allocator) PhysicsWorld {
        var wdef = c.b3DefaultWorldDef();
        wdef.gravity = toB3Vec(Vec3.new(0.0, -9.81, 0.0));
        return .{
            .allocator = allocator,
            .world_id = c.b3CreateWorld(&wdef),
        };
    }

    pub fn deinit(self: *PhysicsWorld) void {
        for (self.joints.items) |j| {
            if (c.b3Joint_IsValid(j)) {
                c.b3DestroyJoint(j, false);
            }
        }
        self.joints.deinit(self.allocator);
        for (self.bodies.items) |b| {
            if (b.mesh_data) |md| c.b3DestroyMesh(md);
            if (b.heightfield_data) |hf| c.b3DestroyHeightField(hf);
            for (b.child_shapes.items) |*ch| {
                ch.freeOwned(self.allocator);
            }
            b.child_shapes.deinit(self.allocator);
            self.allocator.destroy(b);
        }
        self.bodies.deinit(self.allocator);
        self.sensor_events.deinit(self.allocator);
        self.contact_events.deinit(self.allocator);
        self.contact_hit_events.deinit(self.allocator);
        c.b3DestroyWorld(self.world_id);
    }

    pub fn createBody(self: *PhysicsWorld, mesh: *Mesh, collider: ColliderType, mass: f32) !*RigidBody {
        return self.createBodyWith(mesh, collider, mass, .{});
    }

    /// Creates a body with collision filter, sensor and event options applied
    /// before the shape is built.
    pub fn createBodyWith(
        self: *PhysicsWorld,
        mesh: *Mesh,
        collider: ColliderType,
        mass: f32,
        options: BodyOptions,
    ) !*RigidBody {
        const body = try self.createBodyRaw(mesh, collider, mass);
        errdefer self.removeBody(body);

        body.filter = options.filter;
        body.is_sensor = options.is_sensor;
        body.enable_sensor_events = options.enable_sensor_events;
        body.enable_contact_events = options.enable_contact_events;
        body.enable_hit_events = options.enable_hit_events;
        body.last_filter = options.filter;
        body.last_sensor_events = options.is_sensor or options.enable_sensor_events;
        body.last_contact_events = options.enable_contact_events;
        body.last_hit_events = options.enable_hit_events;

        body.shape_id = try self.createShapeOnBody(body);
        return body;
    }

    fn createBodyRaw(self: *PhysicsWorld, mesh: *Mesh, collider: ColliderType, mass: f32) !*RigidBody {
        const body = try self.allocator.create(RigidBody);
        body.* = RigidBody.init(mesh, collider, mass);
        errdefer self.allocator.destroy(body);

        var bdef = c.b3DefaultBodyDef();
        bdef.type = if (mass > 0.0) c.b3_dynamicBody else c.b3_staticBody;
        bdef.position = toB3Pos(mesh.position);
        bdef.rotation = toB3Quat(Quat.fromEulerDeg(mesh.rotation));
        bdef.linearDamping = body.linear_damping * damp_scale;
        bdef.angularDamping = body.angular_damping * damp_scale;
        body.body_id = c.b3CreateBody(self.world_id, &bdef);
        c.b3Body_SetGravityScale(body.body_id, if (body.use_gravity) 1.0 else 0.0);

        try self.bodies.append(self.allocator, body);
        return body;
    }

    /// Creates a static terrain body from a regular height grid.
    /// `heights` is row-major: index = row * count_x + column, where column
    /// maps to local x and row maps to local z (matching Box3D's convention).
    pub fn createHeightField(
        self: *PhysicsWorld,
        mesh: *Mesh,
        heights: []const f32,
        count_x: u32,
        count_z: u32,
        options: HeightFieldOptions,
    ) !*RigidBody {
        if (count_x < 2 or count_z < 2 or heights.len != @as(usize, count_x) * count_z) {
            return error.InvalidHeightFieldDimensions;
        }
        if (options.scale.x <= 0.0 or options.scale.y <= 0.0 or options.scale.z <= 0.0) {
            return error.InvalidHeightFieldScale;
        }

        const body = try self.createBodyRaw(mesh, .heightfield, 0.0);
        errdefer self.removeBody(body);

        var min_h = heights[0];
        var max_h = heights[0];
        for (heights) |h| {
            min_h = @min(min_h, h);
            max_h = @max(max_h, h);
        }

        var def = std.mem.zeroes(c.b3HeightFieldDef);
        def.heights = @constCast(heights.ptr);
        def.materialIndices = null;
        def.scale = toB3Vec(options.scale);
        def.countX = @intCast(count_x);
        def.countZ = @intCast(count_z);
        def.globalMinimumHeight = min_h;
        def.globalMaximumHeight = max_h;
        def.clockwiseWinding = options.clockwise_winding;

        const heightfield = c.b3CreateHeightField(&def);
        if (heightfield == null) return error.PhysicsShapeCreationFailed;
        body.heightfield_data = heightfield;

        var sdef = c.b3DefaultShapeDef();
        sdef.density = 0.0;
        sdef.baseMaterial.friction = body.friction;
        sdef.baseMaterial.restitution = body.restitution;
        sdef.filter = toB3Filter(body.filter);
        body.shape_id = c.b3CreateHeightFieldShape(body.body_id, &sdef, heightfield);
        if (body.shape_id.index1 == 0) return error.PhysicsShapeCreationFailed;
        return body;
    }

    /// Builds a rope/chain from caller-provided meshes laid out from the first
    /// to the last position. Each mesh becomes a dynamic segment body linked
    /// to its neighbors by distance joints; the ends can be pinned to other
    /// bodies. Segment bodies are owned by the rope (see Rope.deinit).
    pub fn createRope(self: *PhysicsWorld, segment_meshes: []*Mesh, options: RopeOptions) !Rope {
        if (segment_meshes.len == 0) return error.EmptyRope;

        var rope = Rope{
            .from = segment_meshes[0].position,
            .to = segment_meshes[segment_meshes.len - 1].position,
            .spacing = 0.5,
            .max_stretch = options.max_stretch,
            .min_compress = options.min_compress,
            .spring = options.spring,
            .spring_hertz = options.spring_hertz,
            .spring_damping = options.spring_damping,
            .collide_connected = options.collide_connected,
            .pin_start = options.pin_start,
            .pin_end = options.pin_end,
            .pin_start_anchor = options.pin_start_anchor,
            .pin_end_anchor = options.pin_end_anchor,
        };
        errdefer rope.deinit(self);

        if (segment_meshes.len > 1) {
            const span = segment_meshes[0].position.sub(segment_meshes[segment_meshes.len - 1].position).length();
            rope.spacing = @max(span / @as(f32, @floatFromInt(segment_meshes.len - 1)), 0.05);
        }

        for (segment_meshes) |mesh| {
            const body = try self.createBody(mesh, options.collider, options.segment_mass);
            errdefer self.removeBody(body);
            body.restitution = options.restitution;
            body.friction = options.friction;
            body.enable_sensor_events = options.enable_sensor_events;
            body.enable_contact_events = options.enable_contact_events;
            body.enable_hit_events = options.enable_hit_events;
            try rope.bodies.append(self.allocator, body);
        }

        for (0..rope.bodies.items.len -| 1) |i| {
            const jid = try self.createDistanceJoint(
                rope.bodies.items[i],
                rope.bodies.items[i + 1],
                Vec3.zero,
                Vec3.zero,
                rope.linkJointOptions(),
            );
            try rope.joints.append(self.allocator, jid);
        }

        if (options.pin_start) |pin| {
            const anchor = options.pin_start_anchor orelse rope.from;
            rope.start_rest = @max(anchor.sub(rope.bodies.items[0].mesh.position).length(), 0.001);
            rope.start_joint = try self.createDistanceJointWorld(
                pin,
                rope.bodies.items[0],
                anchor,
                rope.bodies.items[0].mesh.position,
                .{ .length = rope.start_rest },
            );
        }
        if (options.pin_end) |pin| {
            const last = rope.bodies.items[rope.bodies.items.len - 1];
            const anchor = options.pin_end_anchor orelse rope.to;
            rope.end_rest = @max(last.mesh.position.sub(anchor).length(), 0.001);
            rope.end_joint = try self.createDistanceJointWorld(
                last,
                pin,
                last.mesh.position,
                anchor,
                .{ .length = rope.end_rest },
            );
        }

        return rope;
    }

    /// Attaches an extra box shape to a body (compound collider).
    /// Dimensions are in the body's local (unscaled) units.
    /// Density is redistributed across all shapes to preserve the body mass.
    pub fn addBoxShape(self: *PhysicsWorld, body: *RigidBody, half_extents: Vec3, options: ChildShapeOptions) !void {
        try self.attachChild(body, .{
            .kind = .{ .box = half_extents },
            .offset = options.offset,
            .friction = options.friction,
            .restitution = options.restitution,
            .filter = options.filter,
            .is_sensor = options.is_sensor,
            .sensor_events = options.enable_sensor_events,
            .contact_events = options.enable_contact_events,
            .hit_events = options.enable_hit_events,
        });
    }

    /// Attaches an extra sphere shape to a body (compound collider).
    /// The center comes from `options.offset`.
    pub fn addSphereShape(self: *PhysicsWorld, body: *RigidBody, radius: f32, options: ChildShapeOptions) !void {
        try self.attachChild(body, .{
            .kind = .{ .sphere = radius },
            .offset = options.offset,
            .friction = options.friction,
            .restitution = options.restitution,
            .filter = options.filter,
            .is_sensor = options.is_sensor,
            .sensor_events = options.enable_sensor_events,
            .contact_events = options.enable_contact_events,
            .hit_events = options.enable_hit_events,
        });
    }

    /// Attaches an extra Y-axis capsule shape to a body (compound collider).
    /// `half_height` is the distance from the midpoint (`options.offset`) to
    /// each cap sphere center.
    pub fn addCapsuleShape(self: *PhysicsWorld, body: *RigidBody, half_height: f32, radius: f32, options: ChildShapeOptions) !void {
        try self.attachChild(body, .{
            .kind = .{ .capsule = .{ .half_height = half_height, .radius = radius } },
            .offset = options.offset,
            .friction = options.friction,
            .restitution = options.restitution,
            .filter = options.filter,
            .is_sensor = options.is_sensor,
            .sensor_events = options.enable_sensor_events,
            .contact_events = options.enable_contact_events,
            .hit_events = options.enable_hit_events,
        });
    }

    /// Attaches an extra convex hull shape to a body (compound collider),
    /// built from local points shifted by `options.offset`.
    pub fn addHullShape(self: *PhysicsWorld, body: *RigidBody, points: []const Vec3, options: ChildShapeOptions) !void {
        if (points.len < 4) return error.MissingCollisionGeometry;
        const owned = try self.allocator.dupe(Vec3, points);
        errdefer self.allocator.free(owned);
        try self.attachChild(body, .{
            .kind = .{ .hull = owned },
            .offset = options.offset,
            .friction = options.friction,
            .restitution = options.restitution,
            .filter = options.filter,
            .is_sensor = options.is_sensor,
            .sensor_events = options.enable_sensor_events,
            .contact_events = options.enable_contact_events,
            .hit_events = options.enable_hit_events,
        });
    }

    fn attachChild(self: *PhysicsWorld, body: *RigidBody, child: ChildShape) !void {
        var pending = child;
        errdefer pending.freeOwned(self.allocator);
        pending.shape_id = try self.buildChildShape(body, &pending);
        errdefer c.b3DestroyShape(pending.shape_id, false);
        self.syncChildState(body, &pending);
        try body.child_shapes.append(self.allocator, pending);
        self.syncShapeDensities(body);
    }

    /// Applies the resolved material/filter/event state to a child shape and
    /// records it for change detection.
    fn syncChildState(_: *PhysicsWorld, body: *RigidBody, child: *ChildShape) void {
        const f = child.resolvedFriction(body);
        if (f != child.last_friction) {
            c.b3Shape_SetFriction(child.shape_id, f);
            child.last_friction = f;
        }
        const r = child.resolvedRestitution(body);
        if (r != child.last_restitution) {
            c.b3Shape_SetRestitution(child.shape_id, r);
            child.last_restitution = r;
        }
        const flt = child.resolvedFilter(body);
        if (!sameFilter(flt, child.last_filter)) {
            c.b3Shape_SetFilter(child.shape_id, toB3Filter(flt), true);
            child.last_filter = flt;
        }
        const se = child.resolvedSensorEvents(body);
        if (se != child.last_sensor_events) {
            c.b3Shape_EnableSensorEvents(child.shape_id, se);
            child.last_sensor_events = se;
        }
        const ce = child.resolvedContactEvents(body);
        if (ce != child.last_contact_events) {
            c.b3Shape_EnableContactEvents(child.shape_id, ce);
            child.last_contact_events = ce;
        }
        const he = child.resolvedHitEvents(body);
        if (he != child.last_hit_events) {
            c.b3Shape_EnableHitEvents(child.shape_id, he);
            child.last_hit_events = he;
        }
    }

    fn buildChildShape(self: *PhysicsWorld, body: *RigidBody, child: *ChildShape) !c.b3ShapeId {
        const scale = body.mesh.scaling;
        const off = Vec3.new(child.offset.x * scale.x, child.offset.y * scale.y, child.offset.z * scale.z);

        var sdef = c.b3DefaultShapeDef();
        sdef.baseMaterial.friction = child.resolvedFriction(body);
        sdef.baseMaterial.restitution = child.resolvedRestitution(body);
        sdef.filter = toB3Filter(child.resolvedFilter(body));
        sdef.isSensor = child.resolvedIsSensor(body);
        sdef.enableSensorEvents = child.resolvedSensorEvents(body);
        sdef.enableContactEvents = child.resolvedContactEvents(body);
        sdef.enableHitEvents = child.resolvedHitEvents(body);

        switch (child.kind) {
            .box => |he| {
                const hx = @max(he.x * scale.x, min_half_extent);
                const hy = @max(he.y * scale.y, min_half_extent);
                const hz = @max(he.z * scale.z, min_half_extent);
                var hull = c.b3MakeOffsetBoxHull(hx, hy, hz, toB3Vec(off));
                child.scaled_volume = 8.0 * hx * hy * hz;
                const shape = c.b3CreateHullShape(body.body_id, &sdef, &hull.base);
                if (shape.index1 == 0) return error.PhysicsShapeCreationFailed;
                return shape;
            },
            .sphere => |r| {
                const rs = @max(r * scale.x, min_half_extent);
                const sph = c.b3Sphere{ .center = toB3Vec(off), .radius = rs };
                child.scaled_volume = 4.1887902 * rs * rs * rs;
                const shape = c.b3CreateSphereShape(body.body_id, &sdef, &sph);
                if (shape.index1 == 0) return error.PhysicsShapeCreationFailed;
                return shape;
            },
            .capsule => |cp| {
                const r = @max(cp.radius * scale.x, min_half_extent);
                const hh = @max(cp.half_height * scale.y, 0.0);
                const cap = c.b3Capsule{
                    .center1 = toB3Vec(off.sub(Vec3.new(0.0, hh, 0.0))),
                    .center2 = toB3Vec(off.add(Vec3.new(0.0, hh, 0.0))),
                    .radius = r,
                };
                child.scaled_volume = 4.1887902 * r * r * r + 6.2831853 * r * r * hh;
                const shape = c.b3CreateCapsuleShape(body.body_id, &sdef, &cap);
                if (shape.index1 == 0) return error.PhysicsShapeCreationFailed;
                return shape;
            },
            .hull => |pts| {
                const points = try self.allocator.alloc(c.b3Vec3, pts.len);
                defer self.allocator.free(points);
                for (pts, 0..) |p, i| {
                    points[i] = toB3Vec(Vec3.new(
                        (p.x + child.offset.x) * scale.x,
                        (p.y + child.offset.y) * scale.y,
                        (p.z + child.offset.z) * scale.z,
                    ));
                }
                const hull = c.b3CreateHull(points.ptr, @intCast(points.len), @intCast(points.len));
                if (hull == null) return error.PhysicsShapeCreationFailed;
                defer c.b3DestroyHull(hull);
                const shape = c.b3CreateHullShape(body.body_id, &sdef, hull);
                if (shape.index1 == 0) return error.PhysicsShapeCreationFailed;
                const baked = c.b3Shape_GetHull(shape);
                child.scaled_volume = if (baked != null) baked.*.volume else 0.0;
                return shape;
            },
        }
    }

    /// Redistributes density across the primary shape and all child shapes so
    /// the body keeps its configured mass. Sensor children stay massless.
    fn syncShapeDensities(self: *PhysicsWorld, body: *RigidBody) void {
        _ = self;
        var total = RigidBody.shapeVolume(body, body.mesh.scaling);
        for (body.child_shapes.items) |*ch| {
            if (!ch.resolvedIsSensor(body)) total += ch.scaled_volume;
        }
        const density = RigidBody.densityForMass(body.mass, total);
        c.b3Shape_SetDensity(body.shape_id, density, true);
        for (body.child_shapes.items) |ch| {
            const d = if (ch.resolvedIsSensor(body)) 0.0 else density;
            c.b3Shape_SetDensity(ch.shape_id, d, true);
        }
    }

    /// Casts a ray through the physics world and returns the closest hit.
    /// Unlike mesh picking this also hits hull, mesh and height-field colliders.
    pub fn raycast(self: *PhysicsWorld, origin: Vec3, direction: Vec3, max_distance: f32) PhysicsRayHit {
        return self.raycastWithFilter(origin, direction, max_distance, .{});
    }

    /// Same as raycast but only accepts shapes matching the filter mask.
    pub fn raycastWithFilter(
        self: *PhysicsWorld,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
        filter: CollisionFilter,
    ) PhysicsRayHit {
        if (max_distance <= 0.0) return .{};
        const len = direction.length();
        if (len < 1e-6) return .{};

        const translation = direction.scale(max_distance / len);
        var query_filter = c.b3DefaultQueryFilter();
        query_filter.categoryBits = filter.category_bits;
        query_filter.maskBits = filter.mask_bits;
        const result = c.b3World_CastRayClosest(self.world_id, toB3Pos(origin), toB3Vec(translation), query_filter);
        if (!result.hit) return .{};

        return .{
            .hit = true,
            .point = fromB3Pos(result.point),
            .normal = fromB3Vec(result.normal),
            .distance = max_distance * result.fraction,
            .body = self.findBodyByShape(result.shapeId),
        };
    }

    /// Maps a `CollisionFilter` onto a Box3D query filter, exactly like
    /// `raycastWithFilter` does (category/mask bits; recorder id/name stay default).
    fn toB3QueryFilter(f: CollisionFilter) c.b3QueryFilter {
        var q = c.b3DefaultQueryFilter();
        q.categoryBits = f.category_bits;
        q.maskBits = f.mask_bits;
        return q;
    }

    /// Shared context for the overlap-query callbacks below.
    const OverlapCollectCtx = struct {
        world: *PhysicsWorld,
        results: *std.ArrayListUnmanaged(*RigidBody),
        /// When set (queryPoint), only accept shapes whose world AABB contains this point.
        point: ?Vec3 = null,
        /// Last reported body: Box3D often reports a compound body's shapes
        /// back to back, so re-check it before the full body scan.
        last: ?*RigidBody = null,
        /// Set when `results.append` runs out of memory; the query is aborted
        /// (callback returns false) and the caller converts this to `error.OutOfMemory`.
        oom: bool = false,
    };

    /// `b3OverlapResultFcn`: maps each reported shape to its body, skips
    /// untracked shapes (e.g. the internal ground plane) and `!enabled`
    /// bodies, and appends each body at most once (compound bodies report
    /// one shape per child).
    fn overlapCollectFcn(shape_id: c.b3ShapeId, context: ?*anyopaque) callconv(.c) bool {
        const ctx_ptr = context orelse return true;
        const ctx: *OverlapCollectCtx = @ptrCast(@alignCast(ctx_ptr));
        // Fast path for consecutive shapes of one compound body; the
        // ownsShape check returns the same body findBodyByShape would.
        var cached: ?*RigidBody = null;
        if (ctx.last) |last| {
            if (ctx.world.ownsShape(last, shape_id)) cached = last;
        }
        const body = cached orelse ctx.world.findBodyByShape(shape_id) orelse return true;
        ctx.last = body;
        if (!body.enabled) return true;
        if (ctx.point) |p| {
            const aabb = c.b3Shape_GetAABB(shape_id);
            if (p.x < aabb.lowerBound.x or p.x > aabb.upperBound.x or
                p.y < aabb.lowerBound.y or p.y > aabb.upperBound.y or
                p.z < aabb.lowerBound.z or p.z > aabb.upperBound.z) return true;
        }
        // Fast path: a repeat report lands at the end of the list, which the
        // scan below would find anyway; same ordering and contents.
        if (ctx.results.items.len > 0 and ctx.results.items[ctx.results.items.len - 1] == body) return true;
        for (ctx.results.items) |b| {
            if (b == body) return true;
        }
        ctx.results.append(ctx.world.allocator, body) catch {
            ctx.oom = true;
            return false;
        };
        return true;
    }

    /// Appends every enabled body with a shape potentially overlapping the
    /// box `[min, max]` (broadphase `b3World_OverlapAABB`). Each body is
    /// appended at most once. Results are APPENDED, not cleared.
    pub fn queryAABB(
        self: *PhysicsWorld,
        min: Vec3,
        max: Vec3,
        results: *std.ArrayListUnmanaged(*RigidBody),
    ) !void {
        return self.queryAABBWithFilter(min, max, .{}, results);
    }

    /// Same as `queryAABB` but only accepts shapes matching the filter mask
    /// (mapped exactly like `raycastWithFilter`).
    pub fn queryAABBWithFilter(
        self: *PhysicsWorld,
        min: Vec3,
        max: Vec3,
        filter: CollisionFilter,
        results: *std.ArrayListUnmanaged(*RigidBody),
    ) !void {
        const aabb = c.b3AABB{
            .lowerBound = toB3Vec(Vec3.new(@min(min.x, max.x), @min(min.y, max.y), @min(min.z, max.z))),
            .upperBound = toB3Vec(Vec3.new(@max(min.x, max.x), @max(min.y, max.y), @max(min.z, max.z))),
        };
        var ctx = OverlapCollectCtx{ .world = self, .results = results };
        _ = c.b3World_OverlapAABB(self.world_id, aabb, toB3QueryFilter(filter), &overlapCollectFcn, &ctx);
        if (ctx.oom) return error.OutOfMemory;
    }

    /// Appends every enabled body overlapping the sphere (`center`, `radius`)
    /// via an exact `b3World_OverlapShape` query. The sphere proxy is a single
    /// point with a non-zero radius (see `b3ShapeCastInput` docs), so this is
    /// precise for all collider types — no AABB approximation. Each body is
    /// appended at most once. Results are APPENDED, not cleared.
    pub fn querySphere(
        self: *PhysicsWorld,
        center: Vec3,
        radius: f32,
        results: *std.ArrayListUnmanaged(*RigidBody),
    ) !void {
        return self.querySphereWithFilter(center, radius, .{}, results);
    }

    /// Same as `querySphere` but only accepts shapes matching the filter mask
    /// (mapped exactly like `raycastWithFilter`).
    pub fn querySphereWithFilter(
        self: *PhysicsWorld,
        center: Vec3,
        radius: f32,
        filter: CollisionFilter,
        results: *std.ArrayListUnmanaged(*RigidBody),
    ) !void {
        if (!(radius > 0.0)) return;
        // Proxy points are relative to `origin`, so a sphere is one
        // origin-centered point plus the radius.
        var point = c.b3Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
        var proxy = c.b3ShapeProxy{ .points = &point, .count = 1, .radius = radius };
        var ctx = OverlapCollectCtx{ .world = self, .results = results };
        _ = c.b3World_OverlapShape(self.world_id, toB3Pos(center), &proxy, toB3QueryFilter(filter), &overlapCollectFcn, &ctx);
        if (ctx.oom) return error.OutOfMemory;
    }

    /// Appends every enabled body whose shape world AABB (`b3Shape_GetAABB`)
    /// contains `point`. Broadphase is a zero-extent `b3World_OverlapAABB`
    /// query; the per-shape AABB check is the precise test, so rotated/thin
    /// shapes report AABB containment, not exact surface containment. Each
    /// body is appended at most once. Results are APPENDED, not cleared.
    pub fn queryPoint(
        self: *PhysicsWorld,
        point: Vec3,
        results: *std.ArrayListUnmanaged(*RigidBody),
    ) !void {
        return self.queryPointWithFilter(point, .{}, results);
    }

    /// Same as `queryPoint` but only accepts shapes matching the filter mask
    /// (mapped exactly like `raycastWithFilter`).
    pub fn queryPointWithFilter(
        self: *PhysicsWorld,
        point: Vec3,
        filter: CollisionFilter,
        results: *std.ArrayListUnmanaged(*RigidBody),
    ) !void {
        const p = toB3Vec(point);
        const aabb = c.b3AABB{ .lowerBound = p, .upperBound = p };
        var ctx = OverlapCollectCtx{ .world = self, .results = results, .point = point };
        _ = c.b3World_OverlapAABB(self.world_id, aabb, toB3QueryFilter(filter), &overlapCollectFcn, &ctx);
        if (ctx.oom) return error.OutOfMemory;
    }

    const SphereCastCtx = struct {
        world: *PhysicsWorld,
        travel: f32,
        best_fraction: f32 = std.math.floatMax(f32),
        body: ?*RigidBody = null,
        point: Vec3 = Vec3.zero,
        normal: Vec3 = Vec3.up,
        /// Last reported body (same consecutive-shape fast path as above).
        last: ?*RigidBody = null,
    };

    /// `b3CastResultFcn`: keeps the closest accepted hit, scanning all shapes
    /// (returns 1.0 to continue). Untracked shapes and `!enabled` bodies are
    /// ignored (return -1.0).
    fn sphereCastCollectFcn(
        shape_id: c.b3ShapeId,
        point: c.b3Pos,
        normal: c.b3Vec3,
        fraction: f32,
        _: u64,
        _: c_int,
        _: c_int,
        context: ?*anyopaque,
    ) callconv(.c) f32 {
        const ctx_ptr = context orelse return 1.0;
        const ctx: *SphereCastCtx = @ptrCast(@alignCast(ctx_ptr));
        var cached: ?*RigidBody = null;
        if (ctx.last) |last| {
            if (ctx.world.ownsShape(last, shape_id)) cached = last;
        }
        const body = cached orelse ctx.world.findBodyByShape(shape_id) orelse return -1.0;
        ctx.last = body;
        if (!body.enabled) return -1.0;
        if (fraction < ctx.best_fraction) {
            ctx.best_fraction = fraction;
            ctx.body = body;
            ctx.point = fromB3Pos(point);
            ctx.normal = fromB3Vec(normal);
        }
        return 1.0;
    }

    /// Sweeps a sphere (`origin`, `radius`) along `translation` and returns
    /// the closest hit, or null on a miss. Implemented with `b3World_CastShape`
    /// and a single-point sphere proxy (same construction as `querySphere`).
    pub fn spherecast(self: *PhysicsWorld, origin: Vec3, radius: f32, translation: Vec3) ?PhysicsRayHit {
        return self.spherecastWithFilter(origin, radius, translation, .{});
    }

    /// Same as `spherecast` but only accepts shapes matching the filter mask
    /// (mapped exactly like `raycastWithFilter`).
    pub fn spherecastWithFilter(
        self: *PhysicsWorld,
        origin: Vec3,
        radius: f32,
        translation: Vec3,
        filter: CollisionFilter,
    ) ?PhysicsRayHit {
        const travel = translation.length();
        if (!(radius > 0.0) or travel < 1e-6) return null;
        var point = c.b3Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
        var proxy = c.b3ShapeProxy{ .points = &point, .count = 1, .radius = radius };
        var ctx = SphereCastCtx{ .world = self, .travel = travel };
        _ = c.b3World_CastShape(self.world_id, toB3Pos(origin), &proxy, toB3Vec(translation), toB3QueryFilter(filter), &sphereCastCollectFcn, &ctx);
        const body = ctx.body orelse return null;
        return .{
            .hit = true,
            .point = ctx.point,
            .normal = ctx.normal,
            .distance = travel * ctx.best_fraction,
            .body = body,
        };
    }

    pub fn findBodyByShape(self: *PhysicsWorld, shape_id: c.b3ShapeId) ?*RigidBody {
        for (self.bodies.items) |b| {
            if (self.ownsShape(b, shape_id)) {
                return b;
            }
        }
        return null;
    }

    /// True when the shape is the body's primary shape or one of its children.
    pub fn ownsShape(_: *PhysicsWorld, body: *RigidBody, shape_id: c.b3ShapeId) bool {
        if (body.shape_id.index1 == shape_id.index1 and body.shape_id.generation == shape_id.generation) {
            return true;
        }
        for (body.child_shapes.items) |ch| {
            if (ch.shape_id.index1 == shape_id.index1 and ch.shape_id.generation == shape_id.generation) {
                return true;
            }
        }
        return false;
    }

    /// Events are collected inside step() and cleared at the start of each
    /// call, so they describe the most recent frame only.
    pub fn clearEvents(self: *PhysicsWorld) void {
        self.sensor_events.clearRetainingCapacity();
        self.contact_events.clearRetainingCapacity();
        self.contact_hit_events.clearRetainingCapacity();
    }

    fn drainEvents(self: *PhysicsWorld) void {
        const sensor = c.b3World_GetSensorEvents(self.world_id);
        var i: i32 = 0;
        while (i < sensor.beginCount) : (i += 1) {
            const ev = sensor.beginEvents[@intCast(i)];
            self.sensor_events.append(self.allocator, .{
                .sensor = self.findBodyByShape(ev.sensorShapeId),
                .visitor = self.findBodyByShape(ev.visitorShapeId),
                .began = true,
            }) catch {};
        }
        i = 0;
        while (i < sensor.endCount) : (i += 1) {
            const ev = sensor.endEvents[@intCast(i)];
            self.sensor_events.append(self.allocator, .{
                .sensor = self.findBodyByShape(ev.sensorShapeId),
                .visitor = self.findBodyByShape(ev.visitorShapeId),
                .began = false,
            }) catch {};
        }

        const contacts = c.b3World_GetContactEvents(self.world_id);
        i = 0;
        while (i < contacts.beginCount) : (i += 1) {
            const ev = contacts.beginEvents[@intCast(i)];
            self.contact_events.append(self.allocator, .{
                .a = self.findBodyByShape(ev.shapeIdA),
                .b = self.findBodyByShape(ev.shapeIdB),
                .began = true,
            }) catch {};
        }
        i = 0;
        while (i < contacts.endCount) : (i += 1) {
            const ev = contacts.endEvents[@intCast(i)];
            self.contact_events.append(self.allocator, .{
                .a = self.findBodyByShape(ev.shapeIdA),
                .b = self.findBodyByShape(ev.shapeIdB),
                .began = false,
            }) catch {};
        }
        i = 0;
        while (i < contacts.hitCount) : (i += 1) {
            const ev = contacts.hitEvents[@intCast(i)];
            self.contact_hit_events.append(self.allocator, .{
                .a = self.findBodyByShape(ev.shapeIdA),
                .b = self.findBodyByShape(ev.shapeIdB),
                .point = fromB3Pos(ev.point),
                .normal = fromB3Vec(ev.normal),
                .approach_speed = ev.approachSpeed,
            }) catch {};
        }
    }

    pub fn removeBody(self: *PhysicsWorld, body: *RigidBody) void {
        for (self.bodies.items, 0..) |b, idx| {
            if (b == body) {
                c.b3DestroyBody(b.body_id);
                if (b.mesh_data) |md| c.b3DestroyMesh(md);
                if (b.heightfield_data) |hf| c.b3DestroyHeightField(hf);
                for (b.child_shapes.items) |*ch| {
                    ch.freeOwned(self.allocator);
                }
                b.child_shapes.deinit(self.allocator);
                _ = self.bodies.swapRemove(idx);
                self.allocator.destroy(body);

                // Prune any joints that Box3D destroyed when body was destroyed
                var ji: usize = 0;
                while (ji < self.joints.items.len) {
                    if (!c.b3Joint_IsValid(self.joints.items[ji])) {
                        _ = self.joints.swapRemove(ji);
                    } else {
                        ji += 1;
                    }
                }
                return;
            }
        }
    }

    pub fn findBody(self: *PhysicsWorld, mesh: *const Mesh) ?*RigidBody {
        for (self.bodies.items) |b| {
            if (b.mesh == mesh) return b;
        }
        return null;
    }

    pub fn step(self: *PhysicsWorld, dt: f32) void {
        self.clearEvents();
        if (dt <= 0.0001) return;
        self.syncWorldParams();
        if (self.bodies.items.len == 0) return;

        for (self.bodies.items) |b| {
            self.pushBody(b);
        }

        self.acc += @min(dt, 0.1);
        var n: u32 = 0;
        while (self.acc >= step_h and n < 4) : (n += 1) {
            c.b3World_Step(self.world_id, step_h, @intCast(self.substeps));
            self.drainEvents();
            self.acc -= step_h;
        }
        if (n == 4) self.acc = 0.0; // drop backlog instead of spiraling
        if (n == 0) return;

        for (self.bodies.items) |b| {
            self.pullBody(b);
        }
    }

    fn syncWorldParams(self: *PhysicsWorld) void {
        if (self.hit_event_threshold != self.last_hit_threshold) {
            c.b3World_SetHitEventThreshold(self.world_id, self.hit_event_threshold);
            self.last_hit_threshold = self.hit_event_threshold;
        }
        if (self.gravity.x != self.last_gravity.x or self.gravity.y != self.last_gravity.y or self.gravity.z != self.last_gravity.z) {
            c.b3World_SetGravity(self.world_id, toB3Vec(self.gravity));
            self.last_gravity = self.gravity;
        }
        if (self.ground_y == self.last_ground_y) return;
        if (self.ground_body) |gb| {
            c.b3DestroyBody(gb);
            self.ground_body = null;
        }
        if (self.ground_y) |gy| {
            var bdef = c.b3DefaultBodyDef();
            bdef.type = c.b3_staticBody;
            bdef.position = toB3Pos(Vec3.new(0.0, gy - 1.0, 0.0));
            const gb = c.b3CreateBody(self.world_id, &bdef);
            var sdef = c.b3DefaultShapeDef();
            sdef.density = 0.0;
            sdef.baseMaterial.friction = 1.0; // neutral under geometric-mean mixing
            sdef.baseMaterial.restitution = 0.0; // neutral under max mixing
            var hull = c.b3MakeBoxHull(500.0, 1.0, 500.0);
            _ = c.b3CreateHullShape(gb, &sdef, &hull.base);
            self.ground_body = gb;
        }
        self.last_ground_y = self.ground_y;
    }

    fn scaledPoints(self: *PhysicsWorld, positions: []const Vec3, scale: Vec3) ![]c.b3Vec3 {
        const points = try self.allocator.alloc(c.b3Vec3, positions.len);
        for (positions, 0..) |p, i| {
            points[i] = toB3Vec(Vec3.new(p.x * scale.x, p.y * scale.y, p.z * scale.z));
        }
        return points;
    }

    fn buildMeshData(self: *PhysicsWorld, mesh: *const Mesh) !*c.b3MeshData {
        const positions = try self.allocator.alloc(c.b3Vec3, mesh.cpu_positions.len);
        defer self.allocator.free(positions);
        for (mesh.cpu_positions, 0..) |p, i| {
            positions[i] = toB3Vec(p);
        }

        const indices = try self.allocator.alloc(i32, mesh.cpu_indices.len);
        defer self.allocator.free(indices);
        for (mesh.cpu_indices, 0..) |ix, i| {
            indices[i] = @intCast(ix);
        }

        var def = std.mem.zeroes(c.b3MeshDef);
        def.vertices = positions.ptr;
        def.stride = 0;
        def.indices = indices.ptr;
        def.materialIndices = null;
        def.weldTolerance = 1e-4;
        def.vertexCount = @intCast(positions.len);
        def.triangleCount = @intCast(indices.len / 3);
        def.weldVertices = true;
        def.useMedianSplit = false;
        def.identifyEdges = false;
        def.clockWiseWinding = false;

        const data = c.b3CreateMesh(&def, null, 0);
        if (data == null) return error.PhysicsShapeCreationFailed;
        return data;
    }

    fn createShapeOnBody(self: *PhysicsWorld, b: *RigidBody) !c.b3ShapeId {
        var sdef = c.b3DefaultShapeDef();
        sdef.baseMaterial.friction = b.friction;
        sdef.baseMaterial.restitution = b.restitution;
        sdef.filter = toB3Filter(b.filter);
        sdef.isSensor = b.is_sensor;
        sdef.enableSensorEvents = b.is_sensor or b.enable_sensor_events;
        sdef.enableContactEvents = b.enable_contact_events;
        sdef.enableHitEvents = b.enable_hit_events;
        const scale = b.mesh.scaling;

        switch (b.collider) {
            .box => {
                sdef.density = RigidBody.densityForMass(b.mass, RigidBody.shapeVolume(b, scale));
                const hx = @max(b.base_extents.x * scale.x, min_half_extent);
                const hy = @max(b.base_extents.y * scale.y, min_half_extent);
                const hz = @max(b.base_extents.z * scale.z, min_half_extent);
                var hull = c.b3MakeBoxHull(hx, hy, hz);
                return c.b3CreateHullShape(b.body_id, &sdef, &hull.base);
            },
            .sphere => {
                sdef.density = RigidBody.densityForMass(b.mass, RigidBody.shapeVolume(b, scale));
                const sph = c.b3Sphere{
                    .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
                    .radius = @max(b.base_radius * scale.x, min_half_extent),
                };
                return c.b3CreateSphereShape(b.body_id, &sdef, &sph);
            },
            .capsule => {
                sdef.density = RigidBody.densityForMass(b.mass, RigidBody.shapeVolume(b, scale));
                // Y-capsule: radius comes from the horizontal extents.
                const r = @max(@min(b.base_extents.x, b.base_extents.z) * scale.x, min_half_extent);
                const hh = @max(0.0, b.base_extents.y * scale.y - r);
                const cap = c.b3Capsule{
                    .center1 = .{ .x = 0.0, .y = -hh, .z = 0.0 },
                    .center2 = .{ .x = 0.0, .y = hh, .z = 0.0 },
                    .radius = r,
                };
                return c.b3CreateCapsuleShape(b.body_id, &sdef, &cap);
            },
            .hull => {
                if (b.mesh.cpu_positions.len < 4) return error.MissingCollisionGeometry;
                const points = try self.scaledPoints(b.mesh.cpu_positions, scale);
                defer self.allocator.free(points);

                const hull = c.b3CreateHull(points.ptr, @intCast(points.len), @intCast(points.len));
                if (hull == null) return error.PhysicsShapeCreationFailed;
                defer c.b3DestroyHull(hull);

                // Hull shape data is cloned into the world hull database,
                // so the temporary hull can be released right away.
                sdef.density = RigidBody.densityForMass(b.mass, hull.*.volume);
                const shape = c.b3CreateHullShape(b.body_id, &sdef, hull);
                if (shape.index1 == 0) return error.PhysicsShapeCreationFailed;
                return shape;
            },
            .mesh => {
                if (b.mesh.cpu_positions.len < 3 or b.mesh.cpu_indices.len < 3) return error.MissingCollisionGeometry;
                if (b.mass > 0.0) return error.StaticColliderRequiresZeroMass;

                const data = if (b.mesh_data) |existing|
                    existing
                else blk: {
                    const built = try self.buildMeshData(b.mesh);
                    b.mesh_data = built;
                    break :blk built;
                };
                sdef.density = 0.0;
                const shape = c.b3CreateMeshShape(b.body_id, &sdef, data, toB3Vec(scale));
                if (shape.index1 == 0) return error.PhysicsShapeCreationFailed;
                return shape;
            },
            .heightfield => return error.UseCreateHeightField,
        }
    }

    /// Replaces a body's shapes after a scale change. Each replacement is built
    /// before the old shape is destroyed, so failures keep the old collider.
    fn recreateShape(self: *PhysicsWorld, b: *RigidBody) void {
        const new_shape = self.createShapeOnBody(b) catch return;
        c.b3DestroyShape(b.shape_id, false);
        b.shape_id = new_shape;
        for (b.child_shapes.items) |*ch| {
            const new_child = self.buildChildShape(b, ch) catch continue;
            c.b3DestroyShape(ch.shape_id, false);
            ch.shape_id = new_child;
        }
        self.syncShapeDensities(b);
    }

    fn transformMoved(b: *RigidBody) bool {
        const m = b.mesh;
        return m.position.x != b.last_pos.x or m.position.y != b.last_pos.y or m.position.z != b.last_pos.z or
            m.rotation.x != b.last_rot.x or m.rotation.y != b.last_rot.y or m.rotation.z != b.last_rot.z;
    }

    fn pushBody(self: *PhysicsWorld, b: *RigidBody) void {
        if (b.enabled != b.last_enabled) {
            if (b.enabled) c.b3Body_Enable(b.body_id) else c.b3Body_Disable(b.body_id);
            b.last_enabled = b.enabled;
        }
        if (!b.enabled) return;

        const m = b.mesh;
        const is_static = b.mass <= 0.0;

        if (b.mass != b.last_mass) {
            if ((b.last_mass <= 0.0) != is_static) {
                c.b3Body_SetType(b.body_id, if (is_static) c.b3_staticBody else c.b3_dynamicBody);
            }
            if (!is_static) {
                self.syncShapeDensities(b);
            }
            b.inv_mass = if (b.mass > 0.0) 1.0 / b.mass else 0.0;
            b.last_mass = b.mass;
        }

        if (m.scaling.x != b.last_scale.x or m.scaling.y != b.last_scale.y or m.scaling.z != b.last_scale.z) {
            self.recreateShape(b);
            b.last_scale = m.scaling;
        }

        if (transformMoved(b)) {
            c.b3Body_SetTransform(b.body_id, toB3Pos(m.position), toB3Quat(Quat.fromEulerDeg(m.rotation)));
            if (!is_static) c.b3Body_SetAwake(b.body_id, true);
            // Canonicalize through the solver so detection stays stable
            // (Euler quaternions re-extract to equivalent, not identical, angles).
            b.last_pos = fromB3Pos(c.b3Body_GetPosition(b.body_id));
            m.position = b.last_pos;
            const e = fromB3Quat(c.b3Body_GetRotation(b.body_id)).toEulerDeg();
            m.rotation = e;
            b.last_rot = e;
        }

        if (b.restitution != b.last_rest) {
            c.b3Shape_SetRestitution(b.shape_id, b.restitution);
            for (b.child_shapes.items) |*ch| {
                self.syncChildState(b, ch);
            }
            b.last_rest = b.restitution;
        }
        if (b.friction != b.last_fric) {
            c.b3Shape_SetFriction(b.shape_id, b.friction);
            for (b.child_shapes.items) |*ch| {
                self.syncChildState(b, ch);
            }
            b.last_fric = b.friction;
        }
        if (!sameFilter(b.filter, b.last_filter)) {
            // invokeContacts=true re-evaluates existing contacts immediately,
            // so a mask change starts/stops collisions on the next step.
            c.b3Shape_SetFilter(b.shape_id, toB3Filter(b.filter), true);
            for (b.child_shapes.items) |*ch| {
                self.syncChildState(b, ch);
            }
            b.last_filter = b.filter;
        }

        const want_sensor_events = b.is_sensor or b.enable_sensor_events;
        if (want_sensor_events != b.last_sensor_events) {
            c.b3Shape_EnableSensorEvents(b.shape_id, want_sensor_events);
            for (b.child_shapes.items) |*ch| {
                self.syncChildState(b, ch);
            }
            b.last_sensor_events = want_sensor_events;
        }
        if (b.enable_contact_events != b.last_contact_events) {
            c.b3Shape_EnableContactEvents(b.shape_id, b.enable_contact_events);
            for (b.child_shapes.items) |*ch| {
                self.syncChildState(b, ch);
            }
            b.last_contact_events = b.enable_contact_events;
        }
        if (b.enable_hit_events != b.last_hit_events) {
            c.b3Shape_EnableHitEvents(b.shape_id, b.enable_hit_events);
            for (b.child_shapes.items) |*ch| {
                self.syncChildState(b, ch);
            }
            b.last_hit_events = b.enable_hit_events;
        }
        if (b.linear_damping != b.last_damp_l) {
            c.b3Body_SetLinearDamping(b.body_id, b.linear_damping * damp_scale);
            b.last_damp_l = b.linear_damping;
        }
        if (b.angular_damping != b.last_damp_a) {
            c.b3Body_SetAngularDamping(b.body_id, b.angular_damping * damp_scale);
            b.last_damp_a = b.angular_damping;
        }
        if (b.use_gravity != b.last_grav) {
            c.b3Body_SetGravityScale(b.body_id, if (b.use_gravity) 1.0 else 0.0);
            b.last_grav = b.use_gravity;
        }

        if (is_static) return;
        if (b.velocity.x != b.last_vel.x or b.velocity.y != b.last_vel.y or b.velocity.z != b.last_vel.z) {
            c.b3Body_SetLinearVelocity(b.body_id, toB3Vec(b.velocity));
            c.b3Body_SetAwake(b.body_id, true);
            b.velocity = fromB3Vec(c.b3Body_GetLinearVelocity(b.body_id));
            b.last_vel = b.velocity;
        }
        const want_rad = b.angular_velocity.scale(deg2rad);
        if (want_rad.x != b.last_angvel.x * deg2rad or want_rad.y != b.last_angvel.y * deg2rad or want_rad.z != b.last_angvel.z * deg2rad) {
            c.b3Body_SetAngularVelocity(b.body_id, toB3Vec(want_rad));
            c.b3Body_SetAwake(b.body_id, true);
            const got = fromB3Vec(c.b3Body_GetAngularVelocity(b.body_id)).scale(rad2deg);
            b.angular_velocity = got;
            b.last_angvel = got;
        }
    }

    fn pullBody(self: *PhysicsWorld, b: *RigidBody) void {
        if (!b.enabled or b.mass <= 0.0) {
            b.is_grounded = false;
            return;
        }
        const m = b.mesh;
        const p = fromB3Pos(c.b3Body_GetPosition(b.body_id));
        m.position = p;
        b.last_pos = p;
        const e = fromB3Quat(c.b3Body_GetRotation(b.body_id)).toEulerDeg();
        m.rotation = e;
        b.last_rot = e;
        const v = fromB3Vec(c.b3Body_GetLinearVelocity(b.body_id));
        b.velocity = v;
        b.last_vel = v;
        const w = fromB3Vec(c.b3Body_GetAngularVelocity(b.body_id)).scale(rad2deg);
        b.angular_velocity = w;
        b.last_angvel = w;
        b.is_grounded = self.computeGrounded(b);
    }

    fn computeGrounded(self: *PhysicsWorld, b: *RigidBody) bool {
        var buf: [8]c.b3ContactData = undefined;
        const n = c.b3Body_GetContactData(b.body_id, &buf, 8);
        var i: i32 = 0;
        while (i < n) : (i += 1) {
            const cd = buf[@intCast(i)];
            var mi: i32 = 0;
            while (mi < cd.manifoldCount) : (mi += 1) {
                const man = cd.manifolds[@intCast(mi)];
                if (man.pointCount == 0) continue;
                // Manifold normal points from shape A to shape B.
                const own_a = self.ownsShape(b, cd.shapeIdA);
                const up: f32 = if (own_a) -man.normal.y else man.normal.y;
                if (up > 0.7) {
                    var pi: i32 = 0;
                    while (pi < man.pointCount) : (pi += 1) {
                        if (man.points[@intCast(pi)].separation <= 0.01) return true;
                    }
                }
            }
        }
        return false;
    }

    pub fn createDistanceJoint(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        options: DistanceJointOptions,
    ) !JointId {
        var def = c.b3DefaultDistanceJointDef();
        def.base.bodyIdA = body_a.body_id;
        def.base.bodyIdB = body_b.body_id;
        def.base.collideConnected = options.collide_connected;
        def.base.localFrameA.p = toB3Vec(local_anchor_a);
        def.base.localFrameA.q = .{ .v = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .s = 1.0 };
        def.base.localFrameB.p = toB3Vec(local_anchor_b);
        def.base.localFrameB.q = .{ .v = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .s = 1.0 };

        const rest_len = if (options.length) |l|
            @max(l, 0.001)
        else blk: {
            const wa = body_a.localToWorld(local_anchor_a);
            const wb = body_b.localToWorld(local_anchor_b);
            break :blk @max(wa.sub(wb).length(), 0.001);
        };
        def.length = rest_len;
        def.enableSpring = options.enable_spring;
        def.hertz = options.hertz;
        def.dampingRatio = options.damping_ratio;
        if (options.min_length) |min_l| def.minLength = min_l;
        if (options.max_length) |max_l| {
            def.maxLength = max_l;
            def.enableLimit = true;
        }

        const jid = c.b3CreateDistanceJoint(self.world_id, &def);
        try self.joints.append(self.allocator, jid);
        return jid;
    }

    pub fn createDistanceJointWorld(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        world_anchor_a: Vec3,
        world_anchor_b: Vec3,
        options: DistanceJointOptions,
    ) !JointId {
        return self.createDistanceJoint(
            body_a,
            body_b,
            body_a.worldToLocal(world_anchor_a),
            body_b.worldToLocal(world_anchor_b),
            options,
        );
    }

    pub fn createSphericalJoint(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        options: SphericalJointOptions,
    ) !JointId {
        var def = c.b3DefaultSphericalJointDef();
        def.base.bodyIdA = body_a.body_id;
        def.base.bodyIdB = body_b.body_id;
        def.base.collideConnected = options.collide_connected;
        def.base.localFrameA.p = toB3Vec(local_anchor_a);
        def.base.localFrameA.q = .{ .v = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .s = 1.0 };
        def.base.localFrameB.p = toB3Vec(local_anchor_b);
        def.base.localFrameB.q = .{ .v = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .s = 1.0 };
        def.enableSpring = options.enable_spring;
        def.hertz = options.hertz;
        def.dampingRatio = options.damping_ratio;
        def.enableConeLimit = options.enable_cone_limit;
        def.coneAngle = options.cone_angle_rad;
        def.enableTwistLimit = options.enable_twist_limit;
        def.lowerTwistAngle = options.lower_twist_angle_rad;
        def.upperTwistAngle = options.upper_twist_angle_rad;

        const jid = c.b3CreateSphericalJoint(self.world_id, &def);
        try self.joints.append(self.allocator, jid);
        return jid;
    }

    pub fn createSphericalJointWorld(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        world_anchor: Vec3,
        options: SphericalJointOptions,
    ) !JointId {
        return self.createSphericalJoint(
            body_a,
            body_b,
            body_a.worldToLocal(world_anchor),
            body_b.worldToLocal(world_anchor),
            options,
        );
    }

    /// Creates a hinge (revolute) joint. Relative rotation happens around the
    /// z-axis of the joint frame, which is the world z-axis when both bodies
    /// are unrotated at creation time.
    pub fn createRevoluteJoint(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        options: RevoluteJointOptions,
    ) !JointId {
        var def = c.b3DefaultRevoluteJointDef();
        def.base.bodyIdA = body_a.body_id;
        def.base.bodyIdB = body_b.body_id;
        def.base.collideConnected = options.collide_connected;
        def.base.localFrameA.p = toB3Vec(local_anchor_a);
        def.base.localFrameA.q = toB3Quat(options.frame_a);
        def.base.localFrameB.p = toB3Vec(local_anchor_b);
        def.base.localFrameB.q = toB3Quat(options.frame_b);
        def.enableSpring = options.enable_spring;
        def.hertz = options.hertz;
        def.dampingRatio = options.damping_ratio;
        def.enableLimit = options.enable_limit;
        def.lowerAngle = options.lower_angle_rad;
        def.upperAngle = options.upper_angle_rad;
        def.enableMotor = options.enable_motor;
        def.motorSpeed = options.motor_speed_rad;
        def.maxMotorTorque = options.max_motor_torque;

        const jid = c.b3CreateRevoluteJoint(self.world_id, &def);
        try self.joints.append(self.allocator, jid);
        return jid;
    }

    /// Creates a hinge joint from a shared world-space pivot point.
    pub fn createRevoluteJointWorld(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        world_anchor: Vec3,
        options: RevoluteJointOptions,
    ) !JointId {
        return self.createRevoluteJoint(
            body_a,
            body_b,
            body_a.worldToLocal(world_anchor),
            body_b.worldToLocal(world_anchor),
            options,
        );
    }

    /// Enables/disables a hinge motor and sets its target speed (rad/s) and
    /// torque budget. Wakes the bodies so the change applies immediately.
    pub fn setRevoluteMotor(
        self: *PhysicsWorld,
        joint_id: JointId,
        enabled: bool,
        motor_speed_rad: f32,
        max_motor_torque: f32,
    ) void {
        _ = self;
        c.b3RevoluteJoint_EnableMotor(joint_id, enabled);
        c.b3RevoluteJoint_SetMotorSpeed(joint_id, motor_speed_rad);
        c.b3RevoluteJoint_SetMaxMotorTorque(joint_id, max_motor_torque);
        c.b3Joint_WakeBodies(joint_id);
    }

    /// Enables and sets the hinge angular limits (radians, [-0.99*pi .. 0.99*pi]).
    pub fn setRevoluteLimits(
        self: *PhysicsWorld,
        joint_id: JointId,
        lower_angle_rad: f32,
        upper_angle_rad: f32,
    ) void {
        _ = self;
        c.b3RevoluteJoint_SetLimits(joint_id, lower_angle_rad, upper_angle_rad);
        c.b3RevoluteJoint_EnableLimit(joint_id, true);
        c.b3Joint_WakeBodies(joint_id);
    }

    /// Current hinge angle (radians), relative to the reference angle at creation.
    pub fn revoluteAngleRad(self: *PhysicsWorld, joint_id: JointId) f32 {
        _ = self;
        return c.b3RevoluteJoint_GetAngle(joint_id);
    }

    /// Current hinge angle in degrees.
    pub fn revoluteAngleDeg(self: *PhysicsWorld, joint_id: JointId) f32 {
        _ = self;
        return c.b3RevoluteJoint_GetAngle(joint_id) * rad2deg;
    }

    /// Creates a wheel joint (body A = chassis, body B = wheel). The wheel
    /// spins around the z-axis of frame B and suspends/steers around the
    /// x-axis of frame A. Suspension travel is measured from the creation
    /// pose, so build the vehicle at rest ride height.
    pub fn createWheelJoint(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        options: WheelJointOptions,
    ) !JointId {
        var def = c.b3DefaultWheelJointDef();
        def.base.bodyIdA = body_a.body_id;
        def.base.bodyIdB = body_b.body_id;
        def.base.collideConnected = options.collide_connected;
        def.base.localFrameA.p = toB3Vec(local_anchor_a);
        def.base.localFrameA.q = toB3Quat(options.frame_a);
        def.base.localFrameB.p = toB3Vec(local_anchor_b);
        def.base.localFrameB.q = toB3Quat(options.frame_b);
        def.enableSuspensionSpring = options.enable_suspension;
        def.suspensionHertz = options.suspension_hertz;
        def.suspensionDampingRatio = options.suspension_damping;
        def.enableSuspensionLimit = options.enable_suspension_limit;
        def.lowerSuspensionLimit = options.lower_suspension_limit;
        def.upperSuspensionLimit = options.upper_suspension_limit;
        def.enableSpinMotor = options.enable_spin_motor;
        def.spinSpeed = options.spin_speed_rad;
        def.maxSpinTorque = options.max_spin_torque;
        def.enableSteering = options.enable_steering;
        def.steeringHertz = options.steering_hertz;
        def.steeringDampingRatio = options.steering_damping;
        def.targetSteeringAngle = options.target_steering_angle_rad;
        def.maxSteeringTorque = options.max_steering_torque;
        def.enableSteeringLimit = options.enable_steering_limit;
        def.lowerSteeringLimit = options.lower_steering_limit_rad;
        def.upperSteeringLimit = options.upper_steering_limit_rad;

        const jid = c.b3CreateWheelJoint(self.world_id, &def);
        try self.joints.append(self.allocator, jid);
        return jid;
    }

    /// Creates a wheel joint from a shared world-space anchor point.
    pub fn createWheelJointWorld(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        world_anchor: Vec3,
        options: WheelJointOptions,
    ) !JointId {
        return self.createWheelJoint(
            body_a,
            body_b,
            body_a.worldToLocal(world_anchor),
            body_b.worldToLocal(world_anchor),
            options,
        );
    }

    /// Enables/disables the spin motor and sets target speed (rad/s) and
    /// torque budget. Negative speed drives forward when the wheel axle is
    /// +Z and forward is +X. Wakes the bodies so it applies immediately.
    pub fn setWheelSpin(
        self: *PhysicsWorld,
        joint_id: JointId,
        enabled: bool,
        spin_speed_rad: f32,
        max_spin_torque: f32,
    ) void {
        _ = self;
        c.b3WheelJoint_EnableSpinMotor(joint_id, enabled);
        c.b3WheelJoint_SetSpinMotorSpeed(joint_id, spin_speed_rad);
        c.b3WheelJoint_SetMaxSpinTorque(joint_id, max_spin_torque);
        c.b3Joint_WakeBodies(joint_id);
    }

    /// Enables/disables steering and sets the target angle (radians).
    pub fn setWheelSteering(
        self: *PhysicsWorld,
        joint_id: JointId,
        enabled: bool,
        target_angle_rad: f32,
        max_steering_torque: f32,
    ) void {
        _ = self;
        c.b3WheelJoint_EnableSteering(joint_id, enabled);
        c.b3WheelJoint_SetTargetSteeringAngle(joint_id, target_angle_rad);
        c.b3WheelJoint_SetMaxSteeringTorque(joint_id, max_steering_torque);
        c.b3Joint_WakeBodies(joint_id);
    }

    /// Current wheel spin speed (rad/s), relative between wheel and chassis.
    pub fn wheelSpinSpeed(self: *PhysicsWorld, joint_id: JointId) f32 {
        _ = self;
        return c.b3WheelJoint_GetSpinSpeed(joint_id);
    }

    /// Current steering angle (radians).
    pub fn wheelSteeringAngle(self: *PhysicsWorld, joint_id: JointId) f32 {
        _ = self;
        return c.b3WheelJoint_GetSteeringAngle(joint_id);
    }

    /// Creates a prismatic (slider) joint. Body B translates along frame A
    /// x-axis with rotation locked. Build it in the rest pose: limits and the
    /// reported translation are relative to creation.
    pub fn createPrismaticJoint(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        options: PrismaticJointOptions,
    ) !JointId {
        var def = c.b3DefaultPrismaticJointDef();
        def.base.bodyIdA = body_a.body_id;
        def.base.bodyIdB = body_b.body_id;
        def.base.collideConnected = options.collide_connected;
        def.base.localFrameA.p = toB3Vec(local_anchor_a);
        def.base.localFrameA.q = toB3Quat(options.frame_a);
        def.base.localFrameB.p = toB3Vec(local_anchor_b);
        def.base.localFrameB.q = toB3Quat(options.frame_b);
        def.enableSpring = options.enable_spring;
        def.hertz = options.hertz;
        def.dampingRatio = options.damping_ratio;
        def.enableLimit = options.enable_limit;
        def.lowerTranslation = options.lower_translation;
        def.upperTranslation = options.upper_translation;
        def.enableMotor = options.enable_motor;
        def.motorSpeed = options.motor_speed;
        def.maxMotorForce = options.max_motor_force;

        const jid = c.b3CreatePrismaticJoint(self.world_id, &def);
        try self.joints.append(self.allocator, jid);
        return jid;
    }

    /// Creates a slider joint from a shared world-space anchor point.
    pub fn createPrismaticJointWorld(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        world_anchor: Vec3,
        options: PrismaticJointOptions,
    ) !JointId {
        return self.createPrismaticJoint(
            body_a,
            body_b,
            body_a.worldToLocal(world_anchor),
            body_b.worldToLocal(world_anchor),
            options,
        );
    }

    /// Enables/disables the slider motor and sets target speed (m/s along
    /// frame A x-axis) and force budget. Wakes the bodies.
    pub fn setPrismaticMotor(
        self: *PhysicsWorld,
        joint_id: JointId,
        enabled: bool,
        motor_speed: f32,
        max_motor_force: f32,
    ) void {
        _ = self;
        c.b3PrismaticJoint_EnableMotor(joint_id, enabled);
        c.b3PrismaticJoint_SetMotorSpeed(joint_id, motor_speed);
        c.b3PrismaticJoint_SetMaxMotorForce(joint_id, max_motor_force);
        c.b3Joint_WakeBodies(joint_id);
    }

    /// Enables and sets the slider travel limits (meters, from rest pose).
    pub fn setPrismaticLimits(
        self: *PhysicsWorld,
        joint_id: JointId,
        lower_translation: f32,
        upper_translation: f32,
    ) void {
        _ = self;
        c.b3PrismaticJoint_SetLimits(joint_id, lower_translation, upper_translation);
        c.b3PrismaticJoint_EnableLimit(joint_id, true);
        c.b3Joint_WakeBodies(joint_id);
    }

    /// Current slider translation (meters) relative to the creation pose.
    pub fn prismaticTranslation(self: *PhysicsWorld, joint_id: JointId) f32 {
        _ = self;
        return c.b3PrismaticJoint_GetTranslation(joint_id);
    }

    /// Current slider speed (m/s).
    pub fn prismaticSpeed(self: *PhysicsWorld, joint_id: JointId) f32 {
        _ = self;
        return c.b3PrismaticJoint_GetSpeed(joint_id);
    }

    /// Creates a motor joint driving body B with a velocity motor and an
    /// optional pose spring. With no spring/force configured the joint is
    /// inert until per-frame velocity targets are set.
    pub fn createMotorJoint(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        options: MotorJointOptions,
    ) !JointId {
        var def = c.b3DefaultMotorJointDef();
        def.base.bodyIdA = body_a.body_id;
        def.base.bodyIdB = body_b.body_id;
        def.base.collideConnected = options.collide_connected;
        def.base.localFrameA.p = toB3Vec(local_anchor_a);
        def.base.localFrameA.q = .{ .v = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .s = 1.0 };
        def.base.localFrameB.p = toB3Vec(local_anchor_b);
        def.base.localFrameB.q = .{ .v = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .s = 1.0 };
        def.linearVelocity = toB3Vec(options.linear_velocity);
        def.maxVelocityForce = options.max_velocity_force;
        def.angularVelocity = toB3Vec(options.angular_velocity_rad);
        def.maxVelocityTorque = options.max_velocity_torque;
        def.linearHertz = options.linear_hertz;
        def.linearDampingRatio = options.linear_damping;
        def.maxSpringForce = options.max_spring_force;
        def.angularHertz = options.angular_hertz;
        def.angularDampingRatio = options.angular_damping;
        def.maxSpringTorque = options.max_spring_torque;

        const jid = c.b3CreateMotorJoint(self.world_id, &def);
        try self.joints.append(self.allocator, jid);
        return jid;
    }

    /// Creates a motor joint from a shared world-space anchor point.
    pub fn createMotorJointWorld(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        world_anchor: Vec3,
        options: MotorJointOptions,
    ) !JointId {
        return self.createMotorJoint(
            body_a,
            body_b,
            body_a.worldToLocal(world_anchor),
            body_b.worldToLocal(world_anchor),
            options,
        );
    }

    /// Sets the motor linear velocity target (m/s). Wakes the bodies.
    pub fn setMotorLinearVelocity(self: *PhysicsWorld, joint_id: JointId, velocity: Vec3) void {
        _ = self;
        c.b3MotorJoint_SetLinearVelocity(joint_id, toB3Vec(velocity));
        c.b3Joint_WakeBodies(joint_id);
    }

    /// Sets the motor angular velocity target (rad/s). Wakes the bodies.
    pub fn setMotorAngularVelocity(self: *PhysicsWorld, joint_id: JointId, velocity_rad: Vec3) void {
        _ = self;
        c.b3MotorJoint_SetAngularVelocity(joint_id, toB3Vec(velocity_rad));
        c.b3Joint_WakeBodies(joint_id);
    }

    /// Sets the linear motor force budget (N). Wakes the bodies.
    pub fn setMotorMaxVelocityForce(self: *PhysicsWorld, joint_id: JointId, max_force: f32) void {
        _ = self;
        c.b3MotorJoint_SetMaxVelocityForce(joint_id, max_force);
        c.b3Joint_WakeBodies(joint_id);
    }

    /// Sets the angular motor torque budget (N*m). Wakes the bodies.
    pub fn setMotorMaxVelocityTorque(self: *PhysicsWorld, joint_id: JointId, max_torque: f32) void {
        _ = self;
        c.b3MotorJoint_SetMaxVelocityTorque(joint_id, max_torque);
        c.b3Joint_WakeBodies(joint_id);
    }

    /// Creates a weld joint holding the creation relative pose of two bodies.
    pub fn createWeldJoint(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        options: WeldJointOptions,
    ) !JointId {
        var def = c.b3DefaultWeldJointDef();
        def.base.bodyIdA = body_a.body_id;
        def.base.bodyIdB = body_b.body_id;
        def.base.collideConnected = options.collide_connected;
        def.base.localFrameA.p = toB3Vec(local_anchor_a);
        def.base.localFrameA.q = .{ .v = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .s = 1.0 };
        def.base.localFrameB.p = toB3Vec(local_anchor_b);
        def.base.localFrameB.q = .{ .v = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .s = 1.0 };
        def.linearHertz = options.linear_hertz;
        def.angularHertz = options.angular_hertz;
        def.linearDampingRatio = options.linear_damping;
        def.angularDampingRatio = options.angular_damping;

        const jid = c.b3CreateWeldJoint(self.world_id, &def);
        try self.joints.append(self.allocator, jid);
        return jid;
    }

    /// Creates a weld joint from a shared world-space anchor point.
    pub fn createWeldJointWorld(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        world_anchor: Vec3,
        options: WeldJointOptions,
    ) !JointId {
        return self.createWeldJoint(
            body_a,
            body_b,
            body_a.worldToLocal(world_anchor),
            body_b.worldToLocal(world_anchor),
            options,
        );
    }

    /// Current joint constraint force (N). Poll it to implement breakable
    /// joints: destroy the joint once the load exceeds a threshold.
    pub fn jointConstraintForce(self: *PhysicsWorld, joint_id: JointId) Vec3 {
        _ = self;
        return fromB3Vec(c.b3Joint_GetConstraintForce(joint_id));
    }

    /// Current joint constraint torque (N*m).
    pub fn jointConstraintTorque(self: *PhysicsWorld, joint_id: JointId) Vec3 {
        _ = self;
        return fromB3Vec(c.b3Joint_GetConstraintTorque(joint_id));
    }

    /// Creates a parallel joint: a spring pulling the z-axis of body B
    /// parallel to the z-axis of body A. Anchor points only define the
    /// joint frames, not a position constraint.
    pub fn createParallelJoint(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        options: ParallelJointOptions,
    ) !JointId {
        var def = c.b3DefaultParallelJointDef();
        def.base.bodyIdA = body_a.body_id;
        def.base.bodyIdB = body_b.body_id;
        def.base.collideConnected = options.collide_connected;
        def.base.localFrameA.p = toB3Vec(local_anchor_a);
        def.base.localFrameA.q = .{ .v = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .s = 1.0 };
        def.base.localFrameB.p = toB3Vec(local_anchor_b);
        def.base.localFrameB.q = .{ .v = .{ .x = 0.0, .y = 0.0, .z = 0.0 }, .s = 1.0 };
        def.hertz = options.hertz;
        def.dampingRatio = options.damping_ratio;
        def.maxTorque = options.max_torque;

        const jid = c.b3CreateParallelJoint(self.world_id, &def);
        try self.joints.append(self.allocator, jid);
        return jid;
    }

    /// Creates a parallel joint from a shared world-space anchor point.
    pub fn createParallelJointWorld(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        world_anchor: Vec3,
        options: ParallelJointOptions,
    ) !JointId {
        return self.createParallelJoint(
            body_a,
            body_b,
            body_a.worldToLocal(world_anchor),
            body_b.worldToLocal(world_anchor),
            options,
        );
    }

    /// Retunes the parallel spring at runtime. Wakes the bodies.
    pub fn setParallelSpring(
        self: *PhysicsWorld,
        joint_id: JointId,
        hertz: f32,
        damping_ratio: f32,
        max_torque: f32,
    ) void {
        _ = self;
        c.b3ParallelJoint_SetSpringHertz(joint_id, hertz);
        c.b3ParallelJoint_SetSpringDampingRatio(joint_id, damping_ratio);
        c.b3ParallelJoint_SetMaxTorque(joint_id, max_torque);
        c.b3Joint_WakeBodies(joint_id);
    }

    /// True when the joint still exists in the world.
    pub fn isJointValid(_: *PhysicsWorld, joint_id: JointId) bool {
        return c.b3Joint_IsValid(joint_id);
    }

    pub fn destroyJoint(self: *PhysicsWorld, joint_id: JointId) void {
        c.b3DestroyJoint(joint_id, true);
        for (self.joints.items, 0..) |j, idx| {
            if (j.index1 == joint_id.index1 and j.generation == joint_id.generation) {
                _ = self.joints.swapRemove(idx);
                break;
            }
        }
    }

    pub fn applyExplosion(
        self: *PhysicsWorld,
        epicenter: Vec3,
        radius: f32,
        max_impulse: f32,
        upward_modifier: f32,
    ) void {
        if (radius <= 0.0 or max_impulse <= 0.0) return;
        for (self.bodies.items) |b| {
            if (!b.enabled or b.mass <= 0.0) continue;
            const diff = b.mesh.position.sub(epicenter);
            const dist = diff.length();
            if (dist >= radius) continue;

            // Quadratic smooth falloff [0..1]
            const norm_dist = dist / radius;
            const factor = (1.0 - norm_dist) * (1.0 - norm_dist);

            // Upward biased impulse vector so debris lifts into the air
            var dir = if (dist > 1e-4) diff.scale(1.0 / dist) else Vec3.up;
            dir.y += upward_modifier;
            const impulse_dir = dir.normalize();

            b.applyImpulse(impulse_dir.scale(max_impulse * factor));

            // Tumbling torque impulse around perpendicular axes
            const torque = Vec3.new(
                (dir.z - dir.y) * 0.7,
                (dir.x + dir.z) * 0.4,
                (dir.y - dir.x) * 0.7,
            ).scale(max_impulse * factor * 0.8);
            b.applyTorqueImpulse(torque);
        }
    }

    /// Clamped ring segment count used by the debug wireframe helpers.
    pub fn debugCircleSegments(self: *const PhysicsWorld) usize {
        return @max(min_debug_circle_segments, self.debug_circle_segments);
    }

    /// Builds a CPU-side wireframe for every enabled body and appends it to
    /// `out` (never cleared) for an external/debug line renderer. Local
    /// wireframes live in body units; the mesh world matrix
    /// (`Mesh.getWorldMatrix`, carrying scale/rotation/translation) moves
    /// them to world space, so uniform scales land exactly on the collider
    /// dims (non-uniform scales approximate circles as ellipses).
    /// Hull/mesh/heightfield colliders fall back to the shape's world AABB
    /// (`b3Shape_GetAABB`); hull child shapes are skipped. Colors: dynamic
    /// green, static/kinematic (mass <= 0) white, sensors yellow (sensor
    /// wins). No global state; allocation failures surface as
    /// `error.OutOfMemory`.
    /// Line counts: box 12, sphere 3 * debugCircleSegments() (72 at the
    /// default 24), capsule 2 * debugCircleSegments() + 4 (52 at default),
    /// hull/mesh/heightfield 12 (AABB box). Capacity for the exact count is
    /// reserved once up front, so steady-state appends never reallocate;
    /// circle trig is computed once per call into a shared unit table.
    pub fn appendDebugLines(self: *PhysicsWorld, allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(DebugLine)) !void {
        // Reserve once; every helper below uses appendAssumeCapacity.
        try out.ensureUnusedCapacity(allocator, self.debugLineCount());
        const segs = self.debugCircleSegments();
        var rings = DebugRings{ .segs = segs, .unit = null };
        // segs + 1 samples so consecutive segments share endpoints.
        var unit_buf: [debug_unit_stack_max + 1]DebugUnit = undefined;
        if (segs <= debug_unit_stack_max) {
            var j: usize = 0;
            while (j <= segs) : (j += 1) {
                const t = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(segs)) * 2.0 * std.math.pi;
                unit_buf[j] = .{ .c = @cos(t), .s = @sin(t) };
            }
            rings.unit = unit_buf[0 .. segs + 1];
        }
        for (self.bodies.items) |body| {
            if (!body.enabled) continue;
            const color: [3]f32 = if (body.is_sensor)
                debug_sensor_color
            else if (body.mass <= 0.0)
                debug_static_color
            else
                debug_dynamic_color;
            const wm = body.mesh.getWorldMatrix();
            switch (body.collider) {
                .box => appendDebugBoxLines(out, wm, Vec3.zero, body.base_extents, color),
                .sphere => appendDebugSphereLines(out, wm, Vec3.zero, body.base_radius, color, rings),
                .capsule => appendDebugCapsuleLines(
                    out,
                    wm,
                    Vec3.zero,
                    debugCapsuleRadius(body.base_extents),
                    debugCapsuleHalfHeight(body.base_extents),
                    color,
                    rings,
                ),
                .hull, .mesh, .heightfield => appendDebugAabbLines(out, body.shape_id, color),
            }
            for (body.child_shapes.items) |child| {
                switch (child.kind) {
                    .box => |he| appendDebugBoxLines(out, wm, child.offset, he, color),
                    .sphere => |r| appendDebugSphereLines(out, wm, child.offset, r, color, rings),
                    .capsule => |cp| appendDebugCapsuleLines(out, wm, child.offset, cp.radius, cp.half_height, color, rings),
                    .hull => {},
                }
            }
        }
    }

    /// Exact line count `appendDebugLines` would add (no allocation).
    pub fn debugLineCount(self: *PhysicsWorld) usize {
        const segs = self.debugCircleSegments();
        var n: usize = 0;
        for (self.bodies.items) |body| {
            if (!body.enabled) continue;
            n += primaryDebugLineCount(body.collider, segs);
            for (body.child_shapes.items) |child| {
                n += childDebugLineCount(child.kind, segs);
            }
        }
        return n;
    }
};

test "PhysicsWorld ground collision and step" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var m = Mesh{
        .name = "dummy",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0, 2.0, 0),
    };
    const body = try pw.createBody(&m, .sphere, 2.0);
    body.restitution = 0.5;

    // Simulate free fall
    pw.step(0.1);
    try std.testing.expect(body.velocity.y < 0.0);
    try std.testing.expect(m.position.y < 2.0);

    // Simulate until it lands on ground (-1.2)
    var step_i: usize = 0;
    while (step_i < 120) : (step_i += 1) {
        pw.step(0.016);
    }
    // Ground is at -1.2, bottom of sphere is position.y - radius
    const bottom_y = m.position.y - body.sphere_radius;
    try std.testing.expect(bottom_y >= -1.25);
}

test "PhysicsWorld capsule collision and landing" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var m = Mesh{
        .name = "capsule_dummy",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0, 3.0, 0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.5, -1.0, -0.5), Vec3.new(0.5, 1.0, 0.5)),
    };
    const cap_body = try pw.createBody(&m, .capsule, 1.5);
    cap_body.restitution = 0.2;

    var step_i: usize = 0;
    while (step_i < 150) : (step_i += 1) {
        pw.step(0.016);
    }
    // Ground is at -1.2, bottom of vertical capsule (half-height 1.0) is position.y - 1.0
    try std.testing.expect(m.position.y >= -0.25);
}

test "PhysicsWorld distance joint suspension" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var anchor_mesh = Mesh{
        .name = "anchor",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 5.0, 0.0),
    };
    const anchor = try pw.createBody(&anchor_mesh, .box, 0.0); // static

    var bob_mesh = Mesh{
        .name = "bob",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 3.0, 0.0),
    };
    const bob = try pw.createBody(&bob_mesh, .sphere, 2.0); // dynamic

    const jid = try pw.createDistanceJoint(anchor, bob, Vec3.zero, Vec3.zero, .{ .length = 2.0 });
    try std.testing.expect(jid.index1 > 0);

    // Step physics under gravity
    var step_i: usize = 0;
    while (step_i < 60) : (step_i += 1) {
        pw.step(0.016);
    }

    // Bob should hang around y = 3.0 (length 2.0 from anchor at 5.0), not fall to the ground (-1.2)
    try std.testing.expect(bob_mesh.position.y >= 2.8 and bob_mesh.position.y <= 3.2);

    // Destroy joint
    pw.destroyJoint(jid);
    try std.testing.expectEqual(@as(usize, 0), pw.joints.items.len);
}

test "PhysicsWorld radial explosion impulse" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var close_mesh = Mesh{
        .name = "close",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(1.0, 0.0, 0.0),
    };
    const close_body = try pw.createBody(&close_mesh, .box, 1.0);

    var far_mesh = Mesh{
        .name = "far",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(20.0, 0.0, 0.0),
    };
    const far_body = try pw.createBody(&far_mesh, .box, 1.0);

    // Epicenter at origin, radius 5.0
    pw.applyExplosion(Vec3.zero, 5.0, 50.0, 0.5);

    // Close body must receive linear velocity and upward boost
    try std.testing.expect(close_body.velocity.x > 5.0);
    try std.testing.expect(close_body.velocity.y > 1.0);

    // Far body (at distance 20 > 5) should have zero velocity
    try std.testing.expectEqual(@as(f32, 0.0), far_body.velocity.x);
    try std.testing.expectEqual(@as(f32, 0.0), far_body.velocity.y);
}

test "PhysicsWorld revolute motor spins hinged bar" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var anchor_mesh = Mesh{
        .name = "hinge_anchor",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 3.0, 0.0),
    };
    const anchor = try pw.createBody(&anchor_mesh, .box, 0.0); // static

    var rotor_mesh = Mesh{
        .name = "hinge_rotor",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 3.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1.0, -0.125, -0.125), Vec3.new(1.0, 0.125, 0.125)),
    };
    const rotor = try pw.createBody(&rotor_mesh, .box, 2.0);

    const jid = try pw.createRevoluteJointWorld(anchor, rotor, Vec3.new(0.0, 3.0, 0.0), .{
        .enable_motor = true,
        .motor_speed_rad = 3.0,
        .max_motor_torque = 200.0,
    });
    try std.testing.expect(jid.index1 > 0);

    var step_i: usize = 0;
    while (step_i < 30) : (step_i += 1) {
        pw.step(0.016);
    }

    // The bar should be spinning around the hinge axis (z) near motor speed.
    // angular_velocity is stored in deg/s (3 rad/s ~= 172 deg/s).
    try std.testing.expect(@abs(rotor.angular_velocity.z) > 60.0);
    try std.testing.expect(@abs(pw.revoluteAngleRad(jid)) > 0.2);
}

test "PhysicsWorld revolute limits constrain hinge angle" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var anchor_mesh = Mesh{
        .name = "limit_anchor",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 3.0, 0.0),
    };
    const anchor = try pw.createBody(&anchor_mesh, .box, 0.0);

    var rotor_mesh = Mesh{
        .name = "limit_rotor",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 3.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1.0, -0.125, -0.125), Vec3.new(1.0, 0.125, 0.125)),
    };
    const rotor = try pw.createBody(&rotor_mesh, .box, 2.0);

    const jid = try pw.createRevoluteJointWorld(anchor, rotor, Vec3.new(0.0, 3.0, 0.0), .{
        .enable_limit = true,
        .lower_angle_rad = -0.5,
        .upper_angle_rad = 0.5,
        .enable_motor = true,
        .motor_speed_rad = 3.0,
        .max_motor_torque = 30.0,
    });

    // Drive the motor into the upper limit and hold it there.
    var step_i: usize = 0;
    while (step_i < 120) : (step_i += 1) {
        pw.step(0.016);
    }

    const angle = pw.revoluteAngleRad(jid);
    try std.testing.expect(angle >= 0.35 and angle <= 0.6);

    // Runtime direction control should push the bar back toward the lower limit.
    pw.setRevoluteMotor(jid, true, -3.0, 30.0);
    step_i = 0;
    while (step_i < 120) : (step_i += 1) {
        pw.step(0.016);
    }
    const reversed = pw.revoluteAngleRad(jid);
    try std.testing.expect(reversed <= -0.35 and reversed >= -0.6);
}

test "PhysicsWorld convex hull collider from mesh points" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    // Octahedron: |x| + |y| + |z| = 1, bottom vertex at y = -1.
    var positions = [_]Vec3{
        Vec3.new(-1.0, 0.0, 0.0),
        Vec3.new(1.0, 0.0, 0.0),
        Vec3.new(0.0, -1.0, 0.0),
        Vec3.new(0.0, 1.0, 0.0),
        Vec3.new(0.0, 0.0, -1.0),
        Vec3.new(0.0, 0.0, 1.0),
    };
    var indices = [_]u32{ 0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 0, 2, 5, 2, 1, 5, 1, 3, 5, 3, 0, 5 };

    var mesh = Mesh{
        .name = "octa",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 2.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1.0, -1.0, -1.0), Vec3.new(1.0, 1.0, 1.0)),
        .cpu_positions = &positions,
        .cpu_indices = &indices,
    };
    const body = try pw.createBody(&mesh, .hull, 2.0);
    try std.testing.expect(c.b3Shape_GetHull(body.shape_id) != null);

    var step_i: usize = 0;
    while (step_i < 180) : (step_i += 1) {
        pw.step(0.016);
    }

    // Bottom vertex should rest on the ground plane at y = -1.2.
    try std.testing.expect(mesh.position.y > -0.35 and mesh.position.y < 0.15);
}

test "PhysicsWorld static mesh collider" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    // A 10x10 flat quad (2 triangles) at y = 0.
    var positions = [_]Vec3{
        Vec3.new(-5.0, 0.0, -5.0),
        Vec3.new(5.0, 0.0, -5.0),
        Vec3.new(5.0, 0.0, 5.0),
        Vec3.new(-5.0, 0.0, 5.0),
    };
    var indices = [_]u32{ 0, 1, 2, 0, 2, 3 };

    var floor_mesh = Mesh{
        .name = "tri_floor",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .local_bounding_box = BoundingBox.init(Vec3.new(-5.0, 0.0, -5.0), Vec3.new(5.0, 0.0, 5.0)),
        .cpu_positions = &positions,
        .cpu_indices = &indices,
    };
    _ = try pw.createBody(&floor_mesh, .mesh, 0.0);

    var ball_mesh = Mesh{
        .name = "mesh_ball",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 2.0, 0.0),
    };
    const ball = try pw.createBody(&ball_mesh, .sphere, 1.0);
    try std.testing.expectEqual(@as(f32, 0.5), ball.sphere_radius);

    var step_i: usize = 0;
    while (step_i < 180) : (step_i += 1) {
        pw.step(0.016);
    }

    // Sphere radius is 0.5, so it should rest around y = 0.5, well above ground.
    try std.testing.expect(ball_mesh.position.y > 0.3 and ball_mesh.position.y < 0.8);

    // Dynamic mesh colliders are rejected.
    var dyn_mesh = Mesh{
        .name = "dyn_tri",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .cpu_positions = &positions,
        .cpu_indices = &indices,
    };
    try std.testing.expectError(error.StaticColliderRequiresZeroMass, pw.createBody(&dyn_mesh, .mesh, 1.0));
}

test "PhysicsWorld height field terrain" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    // Flat plateau at height 0.5, 5x5 grid, 1 unit cells.
    var heights = [_]f32{0.5} ** 25;
    var terrain_mesh = Mesh{
        .name = "terrain",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
    };
    _ = try pw.createHeightField(&terrain_mesh, &heights, 5, 5, .{
        .scale = Vec3.new(1.0, 1.0, 1.0),
    });

    var ball_mesh = Mesh{
        .name = "terrain_ball",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(2.0, 3.0, 2.0),
    };
    _ = try pw.createBody(&ball_mesh, .sphere, 1.0);

    var step_i: usize = 0;
    while (step_i < 180) : (step_i += 1) {
        pw.step(0.016);
    }

    // Ball radius 0.5 resting on a 0.5 high surface: center around y = 1.0.
    try std.testing.expect(ball_mesh.position.y > 0.7 and ball_mesh.position.y < 1.4);

    // Invalid dimensions are rejected.
    try std.testing.expectError(error.InvalidHeightFieldDimensions, pw.createHeightField(&terrain_mesh, &heights, 5, 6));
}

test "PhysicsWorld raycast hits hulls and height fields" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var heights = [_]f32{0.0} ** 25;
    var terrain_mesh = Mesh{
        .name = "ray_terrain",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
    };
    _ = try pw.createHeightField(&terrain_mesh, &heights, 5, 5, .{});

    var ball_mesh = Mesh{
        .name = "ray_ball",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(2.0, 2.0, 2.0),
    };
    const ball = try pw.createBody(&ball_mesh, .sphere, 1.0);

    // Downward ray from above the ball.
    const hit = pw.raycast(Vec3.new(2.0, 6.0, 2.0), Vec3.new(0.0, -1.0, 0.0), 20.0);
    try std.testing.expect(hit.hit);
    try std.testing.expect(hit.body == ball);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), hit.point.y, 0.05);
    try std.testing.expectApproxEqAbs(@as(f32, 3.5), hit.distance, 0.05);

    // Ray into the sky misses.
    const miss = pw.raycast(Vec3.new(2.0, 6.0, 2.0), Vec3.new(0.0, 1.0, 0.0), 20.0);
    try std.testing.expect(!miss.hit);

    // Ray above the ball still hits the height field below it.
    const floor_hit = pw.raycast(Vec3.new(4.5, 6.0, 4.5), Vec3.new(0.0, -1.0, 0.0), 20.0);
    try std.testing.expect(floor_hit.hit);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), floor_hit.point.y, 0.1);

    // A layer-filtered ray can skip the ball and hit the terrain instead.
    ball.filter = .{ .category_bits = 0b10, .mask_bits = 0b01 };
    pw.step(0.016); // applies the filter through change detection
    const filtered = pw.raycastWithFilter(
        Vec3.new(2.0, 6.0, 2.0),
        Vec3.new(0.0, -1.0, 0.0),
        20.0,
        .{ .category_bits = 0b01, .mask_bits = 0b01 },
    );
    try std.testing.expect(filtered.hit);
    try std.testing.expect(filtered.body != ball);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), filtered.point.y, 0.1);
}

test "PhysicsWorld collision filters disable and enable contacts" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var platform_mesh = Mesh{
        .name = "filter_platform",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .local_bounding_box = BoundingBox.init(Vec3.new(-2.0, -0.1, -2.0), Vec3.new(2.0, 0.1, 2.0)),
    };
    const platform = try pw.createBody(&platform_mesh, .box, 0.0);
    platform.filter = .{ .category_bits = 0b10, .mask_bits = 0b01 }; // accepts layer 1 only

    var orb_mesh = Mesh{
        .name = "filter_orb",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 2.0, 0.0),
    };
    const orb = try pw.createBody(&orb_mesh, .sphere, 1.0);
    orb.filter = .{ .category_bits = 0b100, .mask_bits = 0xFFFFFFFFFFFFFFFF };

    // Layer 4 is not in the platform mask: the orb falls through to the ground.
    var step_i: usize = 0;
    while (step_i < 90) : (step_i += 1) {
        pw.step(0.016);
    }
    try std.testing.expect(orb_mesh.position.y < -0.4);

    // Add the orb layer to the platform mask, reset, and land on the platform.
    platform.filter.mask_bits = 0b110;
    orb_mesh.position = Vec3.new(0.0, 2.0, 0.0);
    orb.velocity = Vec3.zero;
    orb.angular_velocity = Vec3.zero;
    step_i = 0;
    while (step_i < 150) : (step_i += 1) {
        pw.step(0.016);
    }
    try std.testing.expect(orb_mesh.position.y > 0.3 and orb_mesh.position.y < 0.9);
}

test "PhysicsWorld sensor overlap events" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var sensor_mesh = Mesh{
        .name = "sensor_pad",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 1.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1.0, -1.0, -1.0), Vec3.new(1.0, 1.0, 1.0)),
    };
    const sensor = try pw.createBodyWith(&sensor_mesh, .box, 0.0, .{ .is_sensor = true });

    var orb_mesh = Mesh{
        .name = "sensor_orb",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 4.0, 0.0),
    };
    // Visitors must also enable sensor events; set after creation to exercise
    // the runtime change-detection path.
    const orb = try pw.createBody(&orb_mesh, .sphere, 1.0);
    orb.enable_sensor_events = true;

    var saw_begin = false;
    var saw_end = false;
    var step_i: usize = 0;
    while (step_i < 240) : (step_i += 1) {
        pw.step(0.016);
        for (pw.sensor_events.items) |ev| {
            if (ev.sensor == sensor and ev.visitor == orb) {
                if (ev.began) saw_begin = true else saw_end = true;
            }
        }
    }
    try std.testing.expect(saw_begin);
    try std.testing.expect(saw_end);
}

test "PhysicsWorld contact and hit events" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var platform_mesh = Mesh{
        .name = "event_platform",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .local_bounding_box = BoundingBox.init(Vec3.new(-2.0, -0.1, -2.0), Vec3.new(2.0, 0.1, 2.0)),
    };
    const platform = try pw.createBody(&platform_mesh, .box, 0.0);
    platform.enable_contact_events = true;
    platform.enable_hit_events = true;

    var orb_mesh = Mesh{
        .name = "event_orb",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 3.0, 0.0),
    };
    const orb = try pw.createBody(&orb_mesh, .sphere, 1.0);

    var saw_begin_contact = false;
    var saw_hit = false;
    var step_i: usize = 0;
    while (step_i < 240) : (step_i += 1) {
        pw.step(0.016);
        for (pw.contact_events.items) |ev| {
            if (ev.began and (ev.a == orb or ev.b == orb)) {
                saw_begin_contact = true;
            }
        }
        for (pw.contact_hit_events.items) |ev| {
            if ((ev.a == orb or ev.b == orb) and ev.approach_speed >= pw.hit_event_threshold) {
                saw_hit = true;
            }
        }
    }
    try std.testing.expect(saw_begin_contact);
    try std.testing.expect(saw_hit);

    // Teleport the orb away to force the end-of-contact event.
    orb_mesh.position = Vec3.new(20.0, 3.0, 0.0);
    orb.velocity = Vec3.zero;
    var saw_contact_end = false;
    step_i = 0;
    while (step_i < 30) : (step_i += 1) {
        pw.step(0.016);
        for (pw.contact_events.items) |ev| {
            if (!ev.began and (ev.a == orb or ev.b == orb)) {
                saw_contact_end = true;
            }
        }
    }
    try std.testing.expect(saw_contact_end);
}

test "PhysicsWorld compound body rests on an extra box shape" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var base_mesh = Mesh{
        .name = "compound_base",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .local_bounding_box = BoundingBox.init(Vec3.new(-2.0, -0.5, -2.0), Vec3.new(2.0, 0.5, 2.0)),
    };
    const base = try pw.createBody(&base_mesh, .box, 0.0);
    // Child occupies y 0.5..1.5 above the base.
    try pw.addBoxShape(base, Vec3.new(1.0, 0.5, 1.0), .{ .offset = Vec3.new(0.0, 1.0, 0.0) });
    try std.testing.expectEqual(@as(usize, 1), base.child_shapes.items.len);

    var orb_mesh = Mesh{
        .name = "compound_orb",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 5.0, 0.0),
    };
    _ = try pw.createBody(&orb_mesh, .sphere, 1.0);

    var step_i: usize = 0;
    while (step_i < 180) : (step_i += 1) {
        pw.step(0.016);
    }

    // Sphere radius 0.5 resting on child top (y = 1.5): center around 2.0.
    try std.testing.expect(orb_mesh.position.y > 1.7 and orb_mesh.position.y < 2.3);
}

test "PhysicsWorld compound body preserves total mass" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var car_mesh = Mesh{
        .name = "compound_car",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 2.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1.0, -0.25, -0.5), Vec3.new(1.0, 0.25, 0.5)),
    };
    const car = try pw.createBody(&car_mesh, .box, 3.0);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), c.b3Body_GetMass(car.body_id), 0.05);

    // Attach 4 wheel spheres; mass must stay at 3.0.
    try pw.addSphereShape(car, 0.3, .{ .offset = Vec3.new(-0.7, -0.35, 0.5) });
    try pw.addSphereShape(car, 0.3, .{ .offset = Vec3.new(0.7, -0.35, 0.5) });
    try pw.addSphereShape(car, 0.3, .{ .offset = Vec3.new(-0.7, -0.35, -0.5) });
    try pw.addSphereShape(car, 0.3, .{ .offset = Vec3.new(0.7, -0.35, -0.5) });
    try std.testing.expectEqual(@as(usize, 4), car.child_shapes.items.len);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), c.b3Body_GetMass(car.body_id), 0.05);

    // Scaling the mesh rebuilds every shape and keeps the mass.
    car_mesh.scaling = Vec3.new(2.0, 2.0, 2.0);
    var step_i: usize = 0;
    while (step_i < 60) : (step_i += 1) {
        pw.step(0.016);
    }
    try std.testing.expect(car.child_shapes.items.len == 4);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), c.b3Body_GetMass(car.body_id), 0.1);
}

test "CharacterController falls and lands on the ground" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var cc = CharacterController.init(Vec3.new(0.0, 2.0, 0.0), 0.3, 1.5);
    var i: usize = 0;
    while (i < 180) : (i += 1) {
        pw.step(0.016);
        cc.move(&pw, Vec3.zero, false, 0.016);
    }

    try std.testing.expect(cc.is_grounded);
    try std.testing.expectApproxEqAbs(@as(f32, -1.2), cc.position.y, 0.12);
}

test "CharacterController walks into a wall and stays grounded" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var wall_mesh = Mesh{
        .name = "cc_wall",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(2.0, 0.3, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.1, -1.5, -2.0), Vec3.new(0.1, 1.5, 2.0)),
    };
    _ = try pw.createBody(&wall_mesh, .box, 0.0);

    var cc = CharacterController.init(Vec3.new(0.0, -1.0, 0.0), 0.3, 1.5);
    cc.move_speed = 4.5;
    var i: usize = 0;
    while (i < 120) : (i += 1) {
        pw.step(0.016);
        cc.move(&pw, Vec3.new(1.0, 0.0, 0.0), false, 0.016);
    }

    // Wall face at x = 1.9, capsule radius 0.3: rests near x = 1.6.
    try std.testing.expect(cc.position.x > 1.0 and cc.position.x < 1.8);
    try std.testing.expect(cc.is_grounded);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cc.position.z, 0.05);
}

test "CharacterController jumps and lands back" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var cc = CharacterController.init(Vec3.new(0.0, 0.5, 0.0), 0.3, 1.5);
    var i: usize = 0;
    while (i < 90) : (i += 1) {
        pw.step(0.016);
        cc.move(&pw, Vec3.zero, false, 0.016);
    }
    try std.testing.expect(cc.is_grounded);

    var max_y = cc.position.y;
    i = 0;
    while (i < 40) : (i += 1) {
        pw.step(0.016);
        cc.move(&pw, Vec3.zero, i == 0, 0.016);
        max_y = @max(max_y, cc.position.y);
    }
    // Jump fired: feet rose well above the rest height.
    try std.testing.expect(max_y > -0.5);

    i = 0;
    while (i < 150) : (i += 1) {
        pw.step(0.016);
        cc.move(&pw, Vec3.zero, false, 0.016);
    }
    try std.testing.expect(cc.is_grounded);
    try std.testing.expectApproxEqAbs(@as(f32, -1.2), cc.position.y, 0.12);
}

test "PhysicsWorld rope hangs from a pin" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var anchor_mesh = Mesh{
        .name = "rope_anchor",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 5.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.25, -0.25, -0.25), Vec3.new(0.25, 0.25, 0.25)),
    };
    const anchor = try pw.createBody(&anchor_mesh, .box, 0.0);

    var seg_storage: [6]Mesh = undefined;
    var seg_ptrs: [6]*Mesh = undefined;
    for (0..6) |i| {
        seg_storage[i] = .{
            .name = "rope_seg",
            .vertex_buffer = .{},
            .index_buffer = .{},
            .index_count = 0,
            .position = Vec3.new(0.0, 4.5 - @as(f32, @floatFromInt(i)) * 0.4, 0.0),
            .local_bounding_box = BoundingBox.init(Vec3.new(-0.1, -0.1, -0.1), Vec3.new(0.1, 0.1, 0.1)),
        };
        seg_ptrs[i] = &seg_storage[i];
    }
    var rope = try pw.createRope(&seg_ptrs, .{ .collider = .box, .pin_start = anchor });
    defer rope.deinit(&pw);
    try std.testing.expectEqual(@as(usize, 6), rope.bodies.items.len);
    try std.testing.expectEqual(@as(usize, 5), rope.joints.items.len);

    var i: usize = 0;
    while (i < 150) : (i += 1) {
        pw.step(0.016);
    }

    // Free end hangs below the anchor instead of falling to the ground.
    try std.testing.expect(seg_storage[5].position.y < 2.8);
    try std.testing.expect(seg_storage[5].position.y > 0.5);
    for (rope.joints.items) |j| {
        try std.testing.expect(c.b3Joint_IsValid(j));
    }
}

test "PhysicsWorld rope bridge sags but holds" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var anchor_a_mesh = Mesh{
        .name = "bridge_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(-2.5, 3.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.25, -0.25, -0.25), Vec3.new(0.25, 0.25, 0.25)),
    };
    const anchor_a = try pw.createBody(&anchor_a_mesh, .box, 0.0);
    var anchor_b_mesh = Mesh{
        .name = "bridge_b",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(2.5, 3.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.25, -0.25, -0.25), Vec3.new(0.25, 0.25, 0.25)),
    };
    const anchor_b = try pw.createBody(&anchor_b_mesh, .box, 0.0);

    // Rope laid wider than the anchor span so it must sag.
    var seg_storage: [10]Mesh = undefined;
    var seg_ptrs: [10]*Mesh = undefined;
    for (0..10) |i| {
        const t = @as(f32, @floatFromInt(i)) / 9.0;
        seg_storage[i] = .{
            .name = "bridge_seg",
            .vertex_buffer = .{},
            .index_buffer = .{},
            .index_count = 0,
            .position = Vec3.new(-3.0 + 6.0 * t, 3.0, 0.0),
            .local_bounding_box = BoundingBox.init(Vec3.new(-0.1, -0.1, -0.1), Vec3.new(0.1, 0.1, 0.1)),
        };
        seg_ptrs[i] = &seg_storage[i];
    }
    var rope = try pw.createRope(&seg_ptrs, .{
        .collider = .box,
        .pin_start = anchor_a,
        .pin_end = anchor_b,
    });
    defer rope.deinit(&pw);

    var i: usize = 0;
    while (i < 240) : (i += 1) {
        pw.step(0.016);
    }

    // Middle sags clearly below the anchor line but stays well above ground.
    try std.testing.expect(seg_storage[4].position.y < 2.4);
    try std.testing.expect(seg_storage[5].position.y < 2.4);
    try std.testing.expect(seg_storage[4].position.y > 0.5);
    for (rope.joints.items) |j| {
        try std.testing.expect(c.b3Joint_IsValid(j));
    }
}

test "PhysicsWorld rope detach drops the tail and repair restores it" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var anchor_mesh = Mesh{
        .name = "cut_anchor",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 5.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.25, -0.25, -0.25), Vec3.new(0.25, 0.25, 0.25)),
    };
    const anchor = try pw.createBody(&anchor_mesh, .box, 0.0);

    var seg_storage: [6]Mesh = undefined;
    var seg_ptrs: [6]*Mesh = undefined;
    for (0..6) |i| {
        seg_storage[i] = .{
            .name = "cut_seg",
            .vertex_buffer = .{},
            .index_buffer = .{},
            .index_count = 0,
            .position = Vec3.new(0.0, 4.5 - @as(f32, @floatFromInt(i)) * 0.4, 0.0),
            .local_bounding_box = BoundingBox.init(Vec3.new(-0.1, -0.1, -0.1), Vec3.new(0.1, 0.1, 0.1)),
        };
        seg_ptrs[i] = &seg_storage[i];
    }
    var rope = try pw.createRope(&seg_ptrs, .{ .collider = .box, .pin_start = anchor });
    defer rope.deinit(&pw);

    var i: usize = 0;
    while (i < 90) : (i += 1) {
        pw.step(0.016);
    }

    rope.detachAt(&pw, 2);
    try std.testing.expect(!c.b3Joint_IsValid(rope.joints.items[2]));
    i = 0;
    while (i < 90) : (i += 1) {
        pw.step(0.016);
    }

    // Severed tail fell to the ground while the top still hangs.
    try std.testing.expect(seg_storage[5].position.y < 0.5);
    try std.testing.expect(seg_storage[0].position.y > 4.0);

    rope.reset(&pw);
    i = 0;
    while (i < 90) : (i += 1) {
        pw.step(0.016);
    }
    try std.testing.expect(c.b3Joint_IsValid(rope.joints.items[2]));
    try std.testing.expect(seg_storage[5].position.y > 1.8 and seg_storage[5].position.y < 2.8);
}

test "Rope deinit releases lists after a severed link" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var anchor_mesh = Mesh{
        .name = "deinit_anchor",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 5.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.25, -0.25, -0.25), Vec3.new(0.25, 0.25, 0.25)),
    };
    const anchor = try pw.createBody(&anchor_mesh, .box, 0.0);

    var seg_storage: [6]Mesh = undefined;
    var seg_ptrs: [6]*Mesh = undefined;
    for (0..6) |i| {
        seg_storage[i] = .{
            .name = "deinit_seg",
            .vertex_buffer = .{},
            .index_buffer = .{},
            .index_count = 0,
            .position = Vec3.new(0.0, 4.5 - @as(f32, @floatFromInt(i)) * 0.4, 0.0),
            .local_bounding_box = BoundingBox.init(Vec3.new(-0.1, -0.1, -0.1), Vec3.new(0.1, 0.1, 0.1)),
        };
        seg_ptrs[i] = &seg_storage[i];
    }
    var rope = try pw.createRope(&seg_ptrs, .{ .collider = .box, .pin_start = anchor });
    rope.detachAt(&pw, 2);
    // The severed link stays in rope.joints; deinit must skip it instead of
    // destroying the stale id a second time.
    rope.deinit(&pw);
    try std.testing.expectEqual(@as(usize, 0), rope.bodies.items.len);
    try std.testing.expectEqual(@as(usize, 0), rope.joints.items.len);
}

fn makeWheelTestRig(pw: *PhysicsWorld) !struct {
    chassis: *RigidBody,
    wheels: [4]*RigidBody,
    joints: [4]JointId,
} {
    var chassis_mesh = Mesh{
        .name = "wheel_chassis",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 0.5, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1.0, -0.25, -0.5), Vec3.new(1.0, 0.25, 0.5)),
    };
    const chassis = try pw.createBody(&chassis_mesh, .box, 4.0);

    // Suspension acts along frame A x-axis: rotate it onto world -Y.
    const frame_a = Quat.fromEulerDeg(Vec3.new(0.0, 0.0, -90.0));

    var wheels: [4]*RigidBody = undefined;
    var joints: [4]JointId = undefined;
    const wheel_offsets = [4]Vec3{
        Vec3.new(-0.7, 0.0, 0.4),
        Vec3.new(0.7, 0.0, 0.4),
        Vec3.new(-0.7, 0.0, -0.4),
        Vec3.new(0.7, 0.0, -0.4),
    };
    var wheel_meshes: [4]Mesh = undefined;
    for (0..4) |i| {
        wheel_meshes[i] = .{
            .name = "wheel_test",
            .vertex_buffer = .{},
            .index_buffer = .{},
            .index_count = 0,
            .position = wheel_offsets[i],
        };
        wheels[i] = try pw.createBody(&wheel_meshes[i], .sphere, 0.8);
        joints[i] = try pw.createWheelJointWorld(chassis, wheels[i], wheel_offsets[i], .{
            .frame_a = frame_a,
            .suspension_hertz = 8.0,
            .suspension_damping = 0.7,
            .lower_suspension_limit = -0.2,
            .upper_suspension_limit = 0.2,
        });
    }
    return .{ .chassis = chassis, .wheels = wheels, .joints = joints };
}

test "PhysicsWorld wheel suspension holds the chassis" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    const rig = try makeWheelTestRig(&pw);
    _ = rig.wheels;

    var i: usize = 0;
    while (i < 180) : (i += 1) {
        pw.step(0.016);
    }

    // Wheels (r = 0.5) rest on the ground plane at y = -1.2.
    const wheel_y = rig.wheels[0].mesh.position.y;
    try std.testing.expect(wheel_y > -1.0 and wheel_y < -0.5);
    // Chassis hangs ~0.5 above the wheels on its suspension.
    const chassis_y = rig.chassis.mesh.position.y;
    try std.testing.expect(chassis_y > wheel_y + 0.2 and chassis_y < wheel_y + 0.9);
}

test "PhysicsWorld wheel spin motor spins up" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var chassis_mesh = Mesh{
        .name = "spin_chassis",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 2.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1.0, -0.25, -0.5), Vec3.new(1.0, 0.25, 0.5)),
    };
    const chassis = try pw.createBody(&chassis_mesh, .box, 0.0);

    var wheel_mesh = Mesh{
        .name = "spin_wheel",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.7, 1.5, 0.0),
    };
    const wheel = try pw.createBody(&wheel_mesh, .sphere, 1.0);

    const jid = try pw.createWheelJointWorld(chassis, wheel, Vec3.new(0.7, 1.5, 0.0), .{
        .frame_a = Quat.fromEulerDeg(Vec3.new(0.0, 0.0, -90.0)),
    });
    pw.setWheelSpin(jid, true, 10.0, 50.0);

    var i: usize = 0;
    while (i < 90) : (i += 1) {
        pw.step(0.016);
    }
    try std.testing.expect(pw.wheelSpinSpeed(jid) > 5.0);
}

test "PhysicsWorld wheel steering reaches the target angle" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var chassis_mesh = Mesh{
        .name = "steer_chassis",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 2.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1.0, -0.25, -0.5), Vec3.new(1.0, 0.25, 0.5)),
    };
    const chassis = try pw.createBody(&chassis_mesh, .box, 0.0);

    var wheel_mesh = Mesh{
        .name = "steer_wheel",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.7, 1.5, 0.0),
    };
    const wheel = try pw.createBody(&wheel_mesh, .sphere, 1.0);

    const jid = try pw.createWheelJointWorld(chassis, wheel, Vec3.new(0.7, 1.5, 0.0), .{
        .frame_a = Quat.fromEulerDeg(Vec3.new(0.0, 0.0, -90.0)),
        .enable_steering = true,
        .target_steering_angle_rad = 0.5,
        .max_steering_torque = 30.0,
    });

    var i: usize = 0;
    while (i < 120) : (i += 1) {
        pw.step(0.016);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), pw.wheelSteeringAngle(jid), 0.15);
}

test "PhysicsWorld revolute joint rotates around a custom frame axis" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var anchor_mesh = Mesh{
        .name = "frame_anchor",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 3.0, 0.0),
    };
    const anchor = try pw.createBody(&anchor_mesh, .box, 0.0);

    var rotor_mesh = Mesh{
        .name = "frame_rotor",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 3.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1.0, -0.125, -0.125), Vec3.new(1.0, 0.125, 0.125)),
    };
    const rotor = try pw.createBody(&rotor_mesh, .box, 2.0);

    // Rotate the hinge axis from Z onto X.
    const frame = Quat.fromEulerDeg(Vec3.new(0.0, 90.0, 0.0));
    const jid = try pw.createRevoluteJointWorld(anchor, rotor, Vec3.new(0.0, 3.0, 0.0), .{
        .frame_a = frame,
        .frame_b = frame,
        .enable_motor = true,
        .motor_speed_rad = 3.0,
        .max_motor_torque = 200.0,
    });

    var i: usize = 0;
    while (i < 30) : (i += 1) {
        pw.step(0.016);
    }

    // Spin happens around X now (deg/s), not Z.
    try std.testing.expect(@abs(rotor.angular_velocity.x) > 60.0);
    try std.testing.expect(@abs(rotor.angular_velocity.z) < 20.0);
    try std.testing.expect(@abs(pw.revoluteAngleRad(jid)) > 0.2);
}

test "PhysicsWorld spherical twist limits are plumbed through" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var anchor_mesh = Mesh{
        .name = "twist_anchor",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 3.0, 0.0),
    };
    const anchor = try pw.createBody(&anchor_mesh, .box, 0.0);

    var ball_mesh = Mesh{
        .name = "twist_ball",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 2.0, 0.0),
    };
    const ball = try pw.createBody(&ball_mesh, .sphere, 1.0);

    const jid = try pw.createSphericalJointWorld(anchor, ball, Vec3.new(0.0, 2.5, 0.0), .{
        .enable_cone_limit = true,
        .cone_angle_rad = 0.6,
        .enable_twist_limit = true,
        .lower_twist_angle_rad = -0.3,
        .upper_twist_angle_rad = 0.3,
    });

    try std.testing.expect(c.b3SphericalJoint_IsConeLimitEnabled(jid));
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), c.b3SphericalJoint_GetConeLimit(jid), 1e-4);
    try std.testing.expect(c.b3SphericalJoint_IsTwistLimitEnabled(jid));
    try std.testing.expectApproxEqAbs(@as(f32, -0.3), c.b3SphericalJoint_GetLowerTwistLimit(jid), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), c.b3SphericalJoint_GetUpperTwistLimit(jid), 1e-4);

    // Settles hanging straight down without exploding.
    var i: usize = 0;
    while (i < 90) : (i += 1) {
        pw.step(0.016);
    }
    try std.testing.expect(ball_mesh.position.y > 1.0 and ball_mesh.position.y < 2.2);
}

test "PhysicsWorld prismatic motor slides along the frame axis" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var anchor_mesh = Mesh{
        .name = "prism_anchor",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 3.0, 0.0),
    };
    const anchor = try pw.createBody(&anchor_mesh, .box, 0.0);

    var slider_mesh = Mesh{
        .name = "prism_slider",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 3.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.25, -0.25, -0.25), Vec3.new(0.25, 0.25, 0.25)),
    };
    const slider = try pw.createBody(&slider_mesh, .box, 1.0);

    // Slide axis = frame A x-axis rotated onto world -Y (down positive).
    const jid = try pw.createPrismaticJointWorld(anchor, slider, Vec3.new(0.0, 3.0, 0.0), .{
        .frame_a = Quat.fromEulerDeg(Vec3.new(0.0, 0.0, -90.0)),
        .enable_motor = true,
        .motor_speed = 2.0,
        .max_motor_force = 100.0,
    });

    var i: usize = 0;
    while (i < 60) : (i += 1) {
        pw.step(0.016);
    }

    // ~2 m/s for ~1 s straight down, no sideways drift, no rotation.
    try std.testing.expect(slider_mesh.position.y < 1.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), slider_mesh.position.x, 0.05);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), slider_mesh.position.z, 0.05);
    try std.testing.expect(pw.prismaticTranslation(jid) > 1.0);
}

test "PhysicsWorld prismatic limits clamp travel" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var anchor_mesh = Mesh{
        .name = "prism_lim_anchor",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 3.0, 0.0),
    };
    const anchor = try pw.createBody(&anchor_mesh, .box, 0.0);

    var slider_mesh = Mesh{
        .name = "prism_lim_slider",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 3.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.25, -0.25, -0.25), Vec3.new(0.25, 0.25, 0.25)),
    };
    const slider = try pw.createBody(&slider_mesh, .box, 1.0);

    const jid = try pw.createPrismaticJointWorld(anchor, slider, Vec3.new(0.0, 3.0, 0.0), .{
        .frame_a = Quat.fromEulerDeg(Vec3.new(0.0, 0.0, -90.0)),
        .enable_limit = true,
        .lower_translation = -0.5,
        .upper_translation = 0.5,
        .enable_motor = true,
        .motor_speed = 2.0,
        .max_motor_force = 100.0,
    });

    var i: usize = 0;
    while (i < 180) : (i += 1) {
        pw.step(0.016);
    }

    const travel = pw.prismaticTranslation(jid);
    try std.testing.expect(travel >= -0.05 and travel <= 0.6);
    // Clamped near the lower... upper end instead of running away.
    try std.testing.expect(slider_mesh.position.y > 2.0 and slider_mesh.position.y < 3.2);
}

test "PhysicsWorld motor joint velocity drive lifts a body" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var anchor_mesh = Mesh{
        .name = "motor_anchor",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 0.0, 0.0),
    };
    const anchor = try pw.createBody(&anchor_mesh, .box, 0.0);

    var box_mesh = Mesh{
        .name = "motor_box",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 0.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.25, -0.25, -0.25), Vec3.new(0.25, 0.25, 0.25)),
    };
    const cargo = try pw.createBody(&box_mesh, .box, 2.0);

    const jid = try pw.createMotorJointWorld(anchor, cargo, Vec3.new(0.0, 0.0, 0.0), .{
        .max_velocity_force = 500.0,
    });
    pw.setMotorLinearVelocity(jid, Vec3.new(0.0, 3.0, 0.0));

    var i: usize = 0;
    while (i < 60) : (i += 1) {
        pw.step(0.016);
    }

    try std.testing.expect(box_mesh.position.y > 1.0);
    try std.testing.expect(cargo.velocity.y > 1.0);
}

test "PhysicsWorld motor joint drag pattern reaches a target" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var anchor_mesh = Mesh{
        .name = "drag_anchor",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 2.0, 0.0),
    };
    const anchor = try pw.createBody(&anchor_mesh, .box, 0.0);

    var box_mesh = Mesh{
        .name = "drag_box",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 2.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.25, -0.25, -0.25), Vec3.new(0.25, 0.25, 0.25)),
    };
    const cargo = try pw.createBody(&box_mesh, .box, 2.0);

    const jid = try pw.createMotorJointWorld(anchor, cargo, Vec3.new(0.0, 2.0, 0.0), .{
        .max_velocity_force = 400.0,
        .max_velocity_torque = 40.0,
    });

    // Per-frame velocity servo toward a fixed target, like mouse dragging.
    const target = Vec3.new(3.0, 4.0, -1.0);
    var i: usize = 0;
    while (i < 120) : (i += 1) {
        const to_target = target.sub(box_mesh.position);
        var vel = to_target.scale(6.0);
        if (vel.length() > 12.0) vel = vel.normalize().scale(12.0);
        pw.setMotorLinearVelocity(jid, vel);
        pw.setMotorAngularVelocity(jid, Vec3.zero);
        pw.step(0.016);
    }

    try std.testing.expect(box_mesh.position.sub(target).length() < 0.6);
}

test "PhysicsWorld weld joint holds a body against gravity" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var anchor_mesh = Mesh{
        .name = "weld_anchor",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 3.0, 0.0),
    };
    const anchor = try pw.createBody(&anchor_mesh, .box, 0.0);

    var box_mesh = Mesh{
        .name = "weld_box",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 2.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.25, -0.25, -0.25), Vec3.new(0.25, 0.25, 0.25)),
    };
    const cargo = try pw.createBody(&box_mesh, .box, 2.0);

    const jid = try pw.createWeldJointWorld(anchor, cargo, Vec3.new(0.0, 2.5, 0.0), .{});

    var i: usize = 0;
    while (i < 90) : (i += 1) {
        pw.step(0.016);
    }

    // Rigid weld: the box hangs where it was built instead of falling.
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), box_mesh.position.y, 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), box_mesh.position.x, 0.05);

    // Constraint force carries the 2 kg load (~19.6 N).
    const load = pw.jointConstraintForce(jid).length();
    try std.testing.expect(load > 5.0 and load < 60.0);

    // Breaking the weld drops the box to the ground.
    pw.destroyJoint(jid);
    i = 0;
    while (i < 120) : (i += 1) {
        pw.step(0.016);
    }
    try std.testing.expect(box_mesh.position.y < -0.5);
}

test "PhysicsWorld child shape overrides survive body changes" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var base_mesh = Mesh{
        .name = "override_base",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 2.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.5, -0.5, -0.5), Vec3.new(0.5, 0.5, 0.5)),
    };
    const body = try pw.createBody(&base_mesh, .box, 2.0);
    try pw.addBoxShape(body, Vec3.new(0.2, 0.2, 0.2), .{
        .offset = Vec3.new(0.7, 0.0, 0.0),
        .friction = 1.0,
        .restitution = 0.9,
    });

    body.friction = 0.9;
    body.restitution = 0.2;
    pw.step(0.016);

    const ch = &body.child_shapes.items[0];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), c.b3Shape_GetFriction(ch.shape_id), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), c.b3Shape_GetRestitution(ch.shape_id), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), c.b3Shape_GetFriction(body.shape_id), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), c.b3Shape_GetRestitution(body.shape_id), 1e-4);
}

test "PhysicsWorld sensor child reports overlaps" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var base_mesh = Mesh{
        .name = "sensor_child_base",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 1.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.25, -0.25, -0.25), Vec3.new(0.25, 0.25, 0.25)),
    };
    const base = try pw.createBody(&base_mesh, .box, 0.0);
    try pw.addBoxShape(base, Vec3.new(1.0, 0.5, 1.0), .{
        .offset = Vec3.new(0.0, 1.5, 0.0),
        .is_sensor = true,
    });
    try std.testing.expect(c.b3Shape_IsSensor(base.child_shapes.items[0].shape_id));

    var orb_mesh = Mesh{
        .name = "sensor_child_orb",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 4.5, 0.0),
    };
    const orb = try pw.createBodyWith(&orb_mesh, .sphere, 1.0, .{ .enable_sensor_events = true });

    var saw_begin = false;
    var i: usize = 0;
    while (i < 180) : (i += 1) {
        pw.step(0.016);
        for (pw.sensor_events.items) |ev| {
            if (ev.began and ev.sensor == base and ev.visitor == orb) saw_begin = true;
        }
    }
    try std.testing.expect(saw_begin);
}

test "PhysicsWorld parallel joint keeps a body upright" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var anchor_mesh = Mesh{
        .name = "parallel_anchor",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 0.0, 0.0),
    };
    const anchor = try pw.createBody(&anchor_mesh, .box, 0.0);

    var weeble_mesh = Mesh{
        .name = "weeble",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(2.0, 2.0, 0.0),
        .rotation = Vec3.new(20.0, 0.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.2, -1.0, -0.2), Vec3.new(0.2, 1.0, 0.2)),
    };
    const weeble = try pw.createBody(&weeble_mesh, .box, 2.0);
    _ = try pw.createParallelJointWorld(anchor, weeble, Vec3.new(2.0, 2.0, 0.0), .{
        .hertz = 3.0,
        .damping_ratio = 0.7,
        .max_torque = 300.0,
    });

    var free_mesh = Mesh{
        .name = "free_topple",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(-2.0, 2.0, 0.0),
        .rotation = Vec3.new(20.0, 0.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.2, -1.0, -0.2), Vec3.new(0.2, 1.0, 0.2)),
    };
    _ = try pw.createBody(&free_mesh, .box, 2.0);

    var i: usize = 0;
    while (i < 240) : (i += 1) {
        pw.step(0.016);
    }

    // Spring joint stands back up; the free box topples onto its side.
    try std.testing.expect(@abs(weeble_mesh.rotation.x) < 10.0);
    try std.testing.expect(@abs(free_mesh.rotation.x) > 45.0);
}

test "PhysicsWorld parallel spring retunes at runtime" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var anchor_mesh = Mesh{
        .name = "retune_anchor",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 0.0, 0.0),
    };
    const anchor = try pw.createBody(&anchor_mesh, .box, 0.0);

    var tilt_mesh = Mesh{
        .name = "retune_tilt",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(2.0, 3.0, 0.0),
        .rotation = Vec3.new(25.0, 0.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.2, -1.0, -0.2), Vec3.new(0.2, 1.0, 0.2)),
    };
    const tilt = try pw.createBody(&tilt_mesh, .box, 2.0);
    const jid = try pw.createParallelJointWorld(anchor, tilt, Vec3.new(2.0, 3.0, 0.0), .{
        .max_torque = 0.0,
    });

    var i: usize = 0;
    while (i < 60) : (i += 1) {
        pw.step(0.016);
    }
    // Dead spring: still tilted.
    try std.testing.expect(@abs(tilt_mesh.rotation.x) > 12.0);

    pw.setParallelSpring(jid, 4.0, 0.7, 300.0);
    i = 0;
    while (i < 180) : (i += 1) {
        pw.step(0.016);
    }
    try std.testing.expect(@abs(tilt_mesh.rotation.x) < 10.0);
}

test "PhysicsWorld queryAABB finds bodies in box" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var mesh_a = Mesh{
        .name = "qa_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 0.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1.0, -1.0, -1.0), Vec3.new(1.0, 1.0, 1.0)),
    };
    const body_a = try pw.createBody(&mesh_a, .box, 0.0);

    var mesh_b = Mesh{
        .name = "qa_b",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(10.0, 0.0, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1.0, -1.0, -1.0), Vec3.new(1.0, 1.0, 1.0)),
    };
    const body_b = try pw.createBody(&mesh_b, .box, 0.0);

    var results: std.ArrayListUnmanaged(*RigidBody) = .empty;
    defer results.deinit(std.testing.allocator);

    // Box around A only.
    try pw.queryAABB(Vec3.new(-2.0, -2.0, -2.0), Vec3.new(2.0, 2.0, 2.0), &results);
    try std.testing.expectEqual(@as(usize, 1), results.items.len);
    try std.testing.expect(results.items[0] == body_a);

    // Box around B only (results accumulate across calls).
    results.clearRetainingCapacity();
    try pw.queryAABB(Vec3.new(8.0, -2.0, -2.0), Vec3.new(12.0, 2.0, 2.0), &results);
    try std.testing.expectEqual(@as(usize, 1), results.items.len);
    try std.testing.expect(results.items[0] == body_b);

    // Huge box finds both.
    results.clearRetainingCapacity();
    try pw.queryAABB(Vec3.new(-20.0, -20.0, -20.0), Vec3.new(20.0, 20.0, 20.0), &results);
    try std.testing.expectEqual(@as(usize, 2), results.items.len);

    // Disabled bodies are excluded even before the next step pushes the flag.
    body_b.enabled = false;
    results.clearRetainingCapacity();
    try pw.queryAABB(Vec3.new(-20.0, -20.0, -20.0), Vec3.new(20.0, 20.0, 20.0), &results);
    try std.testing.expectEqual(@as(usize, 1), results.items.len);
    try std.testing.expect(results.items[0] == body_a);
    body_b.enabled = true;

    // A compound body (2 shapes) is still returned exactly once.
    try pw.addBoxShape(body_a, Vec3.new(0.5, 0.5, 0.5), .{ .offset = Vec3.new(0.0, 1.5, 0.0) });
    results.clearRetainingCapacity();
    try pw.queryAABB(Vec3.new(-20.0, -20.0, -20.0), Vec3.new(20.0, 20.0, 20.0), &results);
    try std.testing.expectEqual(@as(usize, 2), results.items.len);
}

test "PhysicsWorld querySphere finds bodies by distance" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var mesh = Mesh{
        .name = "qs_ball",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.zero,
    };
    const body = try pw.createBody(&mesh, .sphere, 0.0);

    var results: std.ArrayListUnmanaged(*RigidBody) = .empty;
    defer results.deinit(std.testing.allocator);

    // Body surface (r=0.5) at distance 2 is well inside query radius 3.
    try pw.querySphere(Vec3.new(2.0, 0.0, 0.0), 3.0, &results);
    try std.testing.expectEqual(@as(usize, 1), results.items.len);
    try std.testing.expect(results.items[0] == body);

    // Same body is far outside a radius-3 query centered 10 units away.
    results.clearRetainingCapacity();
    try pw.querySphere(Vec3.new(10.0, 0.0, 0.0), 3.0, &results);
    try std.testing.expectEqual(@as(usize, 0), results.items.len);
}

test "PhysicsWorld queryPoint finds bodies by containment" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var mesh = Mesh{
        .name = "qp_box",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.zero,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1.0, -1.0, -1.0), Vec3.new(1.0, 1.0, 1.0)),
    };
    const body = try pw.createBody(&mesh, .box, 0.0);

    var results: std.ArrayListUnmanaged(*RigidBody) = .empty;
    defer results.deinit(std.testing.allocator);

    try pw.queryPoint(Vec3.new(0.5, 0.0, 0.0), &results);
    try std.testing.expectEqual(@as(usize, 1), results.items.len);
    try std.testing.expect(results.items[0] == body);

    results.clearRetainingCapacity();
    try pw.queryPoint(Vec3.new(5.0, 0.0, 0.0), &results);
    try std.testing.expectEqual(@as(usize, 0), results.items.len);
}

test "PhysicsWorld spatial queries honor collision filters" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var mesh_a = Mesh{
        .name = "qf_a",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 0.0, 0.0),
    };
    _ = try pw.createBody(&mesh_a, .box, 0.0); // default: category 1

    var mesh_b = Mesh{
        .name = "qf_b",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(5.0, 0.0, 0.0),
    };
    _ = try pw.createBodyWith(&mesh_b, .box, 0.0, .{ .filter = .{ .category_bits = 0b10 } });
    pw.step(0.016); // push the baked-in filters through change detection

    var results: std.ArrayListUnmanaged(*RigidBody) = .empty;
    defer results.deinit(std.testing.allocator);

    // Unfiltered: both bodies.
    try pw.queryAABB(Vec3.new(-20.0, -20.0, -20.0), Vec3.new(20.0, 20.0, 20.0), &results);
    try std.testing.expectEqual(@as(usize, 2), results.items.len);

    // Layer-1-only query mask excludes the layer-2 body.
    results.clearRetainingCapacity();
    try pw.queryAABBWithFilter(
        Vec3.new(-20.0, -20.0, -20.0),
        Vec3.new(20.0, 20.0, 20.0),
        .{ .category_bits = 0b01, .mask_bits = 0b01 },
        &results,
    );
    try std.testing.expectEqual(@as(usize, 1), results.items.len);
    try std.testing.expect(results.items[0].mesh == &mesh_a);

    // Same exclusion applies to sphere and point queries.
    results.clearRetainingCapacity();
    try pw.querySphereWithFilter(Vec3.new(5.0, 0.0, 0.0), 3.0, .{ .category_bits = 0b01, .mask_bits = 0b01 }, &results);
    try std.testing.expectEqual(@as(usize, 0), results.items.len);

    results.clearRetainingCapacity();
    try pw.queryPointWithFilter(Vec3.new(5.0, 0.0, 0.0), .{ .category_bits = 0b01, .mask_bits = 0b01 }, &results);
    try std.testing.expectEqual(@as(usize, 0), results.items.len);
}

test "PhysicsWorld spherecast hits and misses" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var mesh = Mesh{
        .name = "sc_ball",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.zero,
    };
    const body = try pw.createBody(&mesh, .sphere, 0.0); // radius 0.5

    // Cast sphere (r=0.25) from x=5 toward -x: contact at x=0.75, travel 4.25.
    const hit = pw.spherecast(Vec3.new(5.0, 0.0, 0.0), 0.25, Vec3.new(-10.0, 0.0, 0.0));
    try std.testing.expect(hit != null);
    try std.testing.expect(hit.?.body == body);
    try std.testing.expectApproxEqAbs(@as(f32, 4.25), hit.?.distance, 0.05);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), hit.?.point.x, 0.05);

    // Parallel lane misses.
    try std.testing.expect(pw.spherecast(Vec3.new(5.0, 0.0, 0.0), 0.25, Vec3.new(0.0, 10.0, 0.0)) == null);
}

test "PhysicsWorld debug box lines match the collider AABB" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var m = Mesh{
        .name = "dbg_box",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(1.0, 2.0, 3.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.5, -1.0, -0.25), Vec3.new(0.5, 1.0, 0.25)),
    };
    _ = try pw.createBody(&m, .box, 0.0);

    var lines: std.ArrayListUnmanaged(DebugLine) = .empty;
    defer lines.deinit(std.testing.allocator);
    try pw.appendDebugLines(std.testing.allocator, &lines);
    try std.testing.expectEqual(@as(usize, 12), lines.items.len);
    try std.testing.expectEqual(pw.debugLineCount(), lines.items.len);

    const he = pw.bodies.items[0].base_extents;
    for (lines.items) |ln| {
        // Static bodies draw white.
        try std.testing.expectEqual([3]f32{ 0.9, 0.9, 0.9 }, ln.color);
        const pts = [2]Vec3{ ln.a, ln.b };
        for (pts) |p| {
            try std.testing.expect(@abs(p.x - m.position.x) <= he.x + 1e-4);
            try std.testing.expect(@abs(p.y - m.position.y) <= he.y + 1e-4);
            try std.testing.expect(@abs(p.z - m.position.z) <= he.z + 1e-4);
        }
    }
}

test "PhysicsWorld debug sensor lines use the sensor color" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var m = Mesh{
        .name = "dbg_sensor",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
    };
    _ = try pw.createBodyWith(&m, .box, 0.0, .{ .is_sensor = true });

    var lines: std.ArrayListUnmanaged(DebugLine) = .empty;
    defer lines.deinit(std.testing.allocator);
    try pw.appendDebugLines(std.testing.allocator, &lines);
    try std.testing.expectEqual(@as(usize, 12), lines.items.len);
    for (lines.items) |ln| {
        try std.testing.expectEqual([3]f32{ 0.95, 0.8, 0.15 }, ln.color);
    }

    // Lines accumulate across calls; the list is never cleared.
    try pw.appendDebugLines(std.testing.allocator, &lines);
    try std.testing.expectEqual(@as(usize, 24), lines.items.len);
}

test "PhysicsWorld debug lines include compound children" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var m = Mesh{
        .name = "dbg_compound",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
    };
    const body = try pw.createBody(&m, .box, 1.0);
    const base_count = pw.debugLineCount();
    try std.testing.expectEqual(@as(usize, 12), base_count);

    try pw.addBoxShape(body, Vec3.new(0.2, 0.2, 0.2), .{});
    try std.testing.expect(pw.debugLineCount() > base_count);
    try std.testing.expectEqual(base_count + 12, pw.debugLineCount());

    var lines: std.ArrayListUnmanaged(DebugLine) = .empty;
    defer lines.deinit(std.testing.allocator);
    try pw.appendDebugLines(std.testing.allocator, &lines);
    try std.testing.expectEqual(pw.debugLineCount(), lines.items.len);
}

test "PhysicsWorld debug lines skip disabled bodies" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var m = Mesh{
        .name = "dbg_disabled",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
    };
    const body = try pw.createBody(&m, .sphere, 1.0);
    body.enabled = false;

    var lines: std.ArrayListUnmanaged(DebugLine) = .empty;
    defer lines.deinit(std.testing.allocator);
    try pw.appendDebugLines(std.testing.allocator, &lines);
    try std.testing.expectEqual(@as(usize, 0), lines.items.len);
    try std.testing.expectEqual(@as(usize, 0), pw.debugLineCount());
}

test "PhysicsWorld debug sphere lines trace the surface" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var m = Mesh{
        .name = "dbg_sphere",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(2.0, 3.0, 4.0),
    };
    const body = try pw.createBody(&m, .sphere, 1.0);

    var lines: std.ArrayListUnmanaged(DebugLine) = .empty;
    defer lines.deinit(std.testing.allocator);
    try pw.appendDebugLines(std.testing.allocator, &lines);
    try std.testing.expect(lines.items.len > 12);

    const r = body.base_radius;
    for (lines.items) |ln| {
        // Dynamic bodies draw green.
        try std.testing.expectEqual([3]f32{ 0.1, 0.9, 0.3 }, ln.color);
        const pts = [2]Vec3{ ln.a, ln.b };
        for (pts) |p| {
            try std.testing.expect(p.sub(m.position).length() <= r + 1e-3);
        }
    }
}

test "PhysicsWorld debug default ring counts stay 72 and 52" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var sm = Mesh{
        .name = "dbg_def_sphere",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(2.0, 3.0, 4.0),
    };
    const sbody = try pw.createBody(&sm, .sphere, 1.0);
    try std.testing.expectEqual(@as(usize, 24), pw.debugCircleSegments());
    try std.testing.expectEqual(@as(usize, 72), pw.debugLineCount());

    var lines: std.ArrayListUnmanaged(DebugLine) = .empty;
    defer lines.deinit(std.testing.allocator);
    try pw.appendDebugLines(std.testing.allocator, &lines);
    try std.testing.expectEqual(@as(usize, 72), lines.items.len);

    // First xy ring starts at angle 0: exact golden points prove the shared
    // unit table matches the historical per-segment trig.
    const r = sbody.base_radius;
    try std.testing.expectEqual(Vec3.new(sm.position.x + r, sm.position.y, sm.position.z), lines.items[0].a);
    const t1 = @as(f32, 1) / @as(f32, 24) * 2.0 * std.math.pi;
    try std.testing.expectEqual(
        Vec3.new(sm.position.x + @cos(t1) * r, sm.position.y + @sin(t1) * r, sm.position.z),
        lines.items[0].b,
    );

    var pw2 = PhysicsWorld.init(std.testing.allocator);
    defer pw2.deinit();
    var cm = Mesh{
        .name = "dbg_def_capsule",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.5, -1.0, -0.5), Vec3.new(0.5, 1.0, 0.5)),
    };
    _ = try pw2.createBody(&cm, .capsule, 1.0);
    try std.testing.expectEqual(@as(usize, 52), pw2.debugLineCount());
}

test "PhysicsWorld debug circle segments are configurable" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();
    pw.debug_circle_segments = 8;

    var sm = Mesh{
        .name = "dbg_cfg_sphere",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
    };
    _ = try pw.createBody(&sm, .sphere, 1.0);
    try std.testing.expectEqual(@as(usize, 24), pw.debugLineCount());

    var lines: std.ArrayListUnmanaged(DebugLine) = .empty;
    defer lines.deinit(std.testing.allocator);
    try pw.appendDebugLines(std.testing.allocator, &lines);
    try std.testing.expectEqual(@as(usize, 24), lines.items.len);

    // Below-minimum settings clamp to 4 segments (sphere: 12 lines).
    pw.debug_circle_segments = 0;
    try std.testing.expectEqual(@as(usize, 4), pw.debugCircleSegments());
    try std.testing.expectEqual(@as(usize, 12), pw.debugLineCount());
    lines.clearRetainingCapacity();
    try pw.appendDebugLines(std.testing.allocator, &lines);
    try std.testing.expectEqual(@as(usize, 12), lines.items.len);
}

test "PhysicsWorld debug capsule segments are configurable" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();
    pw.debug_circle_segments = 8;

    var cm = Mesh{
        .name = "dbg_cfg_capsule",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .local_bounding_box = BoundingBox.init(Vec3.new(-0.5, -1.0, -0.5), Vec3.new(0.5, 1.0, 0.5)),
    };
    _ = try pw.createBody(&cm, .capsule, 1.0);
    try std.testing.expectEqual(@as(usize, 20), pw.debugLineCount());

    var lines: std.ArrayListUnmanaged(DebugLine) = .empty;
    defer lines.deinit(std.testing.allocator);
    try pw.appendDebugLines(std.testing.allocator, &lines);
    try std.testing.expectEqual(@as(usize, 20), lines.items.len);
}

test "PhysicsWorld debug lines reserve once and repeat identically" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var m = Mesh{
        .name = "dbg_reserve",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(1.0, 2.0, 3.0),
    };
    _ = try pw.createBody(&m, .sphere, 1.0);
    const n = pw.debugLineCount();

    var lines: std.ArrayListUnmanaged(DebugLine) = .empty;
    defer lines.deinit(std.testing.allocator);
    try lines.ensureTotalCapacity(std.testing.allocator, n);
    const cap = lines.capacity;
    try pw.appendDebugLines(std.testing.allocator, &lines);
    try std.testing.expectEqual(n, lines.items.len);
    // No growth: the call reserved everything it needed up front.
    try std.testing.expectEqual(cap, lines.capacity);

    // Same world state appends byte-identical lines.
    try pw.appendDebugLines(std.testing.allocator, &lines);
    try std.testing.expectEqual(2 * n, lines.items.len);
    try std.testing.expectEqualSlices(DebugLine, lines.items[0..n], lines.items[n .. 2 * n]);
}
