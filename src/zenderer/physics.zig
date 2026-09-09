const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const Ray = math.Ray;
const RayHit = math.RayHit;
const Quat = math.Quat;
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

    // Base (unscaled) shape dims; live dims = base * mesh.scaling.
    base_extents: Vec3 = Vec3.new(0.5, 0.5, 0.5),
    base_radius: f32 = 0.5,
    // Picking helpers (base dims, like before).
    box_extents: Vec3 = Vec3.new(0.5, 0.5, 0.5),
    sphere_radius: f32 = 0.5,

    body_id: c.b3BodyId = .{ .index1 = 0, .world0 = 0, .generation = 0 },
    shape_id: c.b3ShapeId = .{ .index1 = 0, .world0 = 0, .generation = 0 },

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
};

pub const PhysicsWorld = struct {
    allocator: std.mem.Allocator,
    gravity: Vec3 = Vec3.new(0.0, -9.81, 0.0),
    ground_y: ?f32 = -1.2,
    substeps: u32 = 4,
    bodies: std.ArrayListUnmanaged(*RigidBody) = .empty,

    world_id: c.b3WorldId,
    ground_body: ?c.b3BodyId = null,
    acc: f32 = 0.0,
    last_gravity: Vec3 = Vec3.new(0.0, -9.81, 0.0),
    last_ground_y: ?f32 = null,

    pub fn init(allocator: std.mem.Allocator) PhysicsWorld {
        var wdef = c.b3DefaultWorldDef();
        wdef.gravity = toB3Vec(Vec3.new(0.0, -9.81, 0.0));
        return .{
            .allocator = allocator,
            .world_id = c.b3CreateWorld(&wdef),
        };
    }

    pub fn deinit(self: *PhysicsWorld) void {
        for (self.bodies.items) |b| {
            self.allocator.destroy(b);
        }
        self.bodies.deinit(self.allocator);
        c.b3DestroyWorld(self.world_id);
    }

    pub fn createBody(self: *PhysicsWorld, mesh: *Mesh, collider: ColliderType, mass: f32) !*RigidBody {
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
        body.shape_id = createShapeOnBody(body);
        c.b3Body_SetGravityScale(body.body_id, if (body.use_gravity) 1.0 else 0.0);

        try self.bodies.append(self.allocator, body);
        return body;
    }

    pub fn removeBody(self: *PhysicsWorld, body: *RigidBody) void {
        for (self.bodies.items, 0..) |b, idx| {
            if (b == body) {
                c.b3DestroyBody(b.body_id);
                _ = self.bodies.swapRemove(idx);
                self.allocator.destroy(body);
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
            self.acc -= step_h;
        }
        if (n == 4) self.acc = 0.0; // drop backlog instead of spiraling
        if (n == 0) return;

        for (self.bodies.items) |b| {
            self.pullBody(b);
        }
    }

    fn syncWorldParams(self: *PhysicsWorld) void {
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

    fn createShapeOnBody(b: *RigidBody) c.b3ShapeId {
        var sdef = c.b3DefaultShapeDef();
        sdef.density = RigidBody.densityForMass(b.mass, b.shapeVolume(b.mesh.scaling));
        sdef.baseMaterial.friction = b.friction;
        sdef.baseMaterial.restitution = b.restitution;
        return switch (b.collider) {
            .box => blk: {
                const hx = @max(b.base_extents.x * b.mesh.scaling.x, min_half_extent);
                const hy = @max(b.base_extents.y * b.mesh.scaling.y, min_half_extent);
                const hz = @max(b.base_extents.z * b.mesh.scaling.z, min_half_extent);
                var hull = c.b3MakeBoxHull(hx, hy, hz);
                break :blk c.b3CreateHullShape(b.body_id, &sdef, &hull.base);
            },
            .sphere => blk: {
                const sph = c.b3Sphere{
                    .center = .{ .x = 0.0, .y = 0.0, .z = 0.0 },
                    .radius = @max(b.base_radius * b.mesh.scaling.x, min_half_extent),
                };
                break :blk c.b3CreateSphereShape(b.body_id, &sdef, &sph);
            },
        };
    }

    fn rebuildShape(b: *RigidBody) void {
        // Mass is preserved: density is recomputed for the new volume.
        c.b3DestroyShape(b.shape_id, false);
        b.shape_id = createShapeOnBody(b);
    }

    fn transformMoved(b: *RigidBody) bool {
        const m = b.mesh;
        return m.position.x != b.last_pos.x or m.position.y != b.last_pos.y or m.position.z != b.last_pos.z or
            m.rotation.x != b.last_rot.x or m.rotation.y != b.last_rot.y or m.rotation.z != b.last_rot.z;
    }

    fn pushBody(self: *PhysicsWorld, b: *RigidBody) void {
        _ = self;
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
                c.b3Shape_SetDensity(b.shape_id, RigidBody.densityForMass(b.mass, b.shapeVolume(m.scaling)), true);
            }
            b.inv_mass = if (b.mass > 0.0) 1.0 / b.mass else 0.0;
            b.last_mass = b.mass;
        }

        if (m.scaling.x != b.last_scale.x or m.scaling.y != b.last_scale.y or m.scaling.z != b.last_scale.z) {
            rebuildShape(b);
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
            b.last_rest = b.restitution;
        }
        if (b.friction != b.last_fric) {
            c.b3Shape_SetFriction(b.shape_id, b.friction);
            b.last_fric = b.friction;
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

    fn computeGrounded(_: *PhysicsWorld, b: *RigidBody) bool {
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
                const own_a = cd.shapeIdA.index1 == b.shape_id.index1 and cd.shapeIdA.generation == b.shape_id.generation;
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
