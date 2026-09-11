//! Rigid bodies and their directly attached shapes/events.
//! Extracted verbatim from `physics.zig`; behavior unchanged.
//! `ChildShape` and `RigidBody` live together: they reference each other
//! (`child_shapes` field vs `resolved*` helpers), so splitting them further
//! would only create a mutual import.
const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Quat = math.Quat;
const Mesh = @import("../mesh.zig").Mesh;
const c = @import("../c.zig").c;
const convert = @import("convert.zig");
const types = @import("types.zig");
const ColliderType = types.ColliderType;
const CollisionFilter = types.CollisionFilter;
const toB3Vec = convert.toB3Vec;
const fromB3Vec = convert.fromB3Vec;
const deg2rad = convert.deg2rad;
const rad2deg = convert.rad2deg;
const min_half_extent = convert.min_half_extent;

pub const PhysicsRayHit = struct {
    hit: bool = false,
    point: Vec3 = Vec3.zero,
    normal: Vec3 = Vec3.up,
    distance: f32 = 0.0,
    body: ?*RigidBody = null,
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

    pub fn freeOwned(self: *ChildShape, allocator: std.mem.Allocator) void {
        if (self.kind == .hull) {
            allocator.free(self.kind.hull);
        }
    }

    pub fn resolvedFriction(self: *const ChildShape, body: *const RigidBody) f32 {
        return self.friction orelse body.friction;
    }

    pub fn resolvedRestitution(self: *const ChildShape, body: *const RigidBody) f32 {
        return self.restitution orelse body.restitution;
    }

    pub fn resolvedFilter(self: *const ChildShape, body: *const RigidBody) CollisionFilter {
        return self.filter orelse body.filter;
    }

    pub fn resolvedIsSensor(self: *const ChildShape, body: *const RigidBody) bool {
        return self.is_sensor orelse body.is_sensor;
    }

    pub fn resolvedSensorEvents(self: *const ChildShape, body: *const RigidBody) bool {
        return self.resolvedIsSensor(body) or (self.sensor_events orelse body.enable_sensor_events);
    }

    pub fn resolvedContactEvents(self: *const ChildShape, body: *const RigidBody) bool {
        return self.contact_events orelse body.enable_contact_events;
    }

    pub fn resolvedHitEvents(self: *const ChildShape, body: *const RigidBody) bool {
        return self.hit_events orelse body.enable_hit_events;
    }
};

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

    pub fn shapeVolume(b: *const RigidBody, scale: Vec3) f32 {
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

    pub fn densityForMass(mass: f32, volume: f32) f32 {
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
