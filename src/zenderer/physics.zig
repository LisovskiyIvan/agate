const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const BoundingBox = math.BoundingBox;
const Ray = math.Ray;
const RayHit = math.RayHit;
const Mesh = @import("mesh.zig").Mesh;

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

pub const RigidBody = struct {
    mesh: *Mesh,
    collider: ColliderType = .box,
    mass: f32 = 1.0,
    inv_mass: f32 = 1.0,

    velocity: Vec3 = Vec3.zero,
    angular_velocity: Vec3 = Vec3.zero, // deg/s

    restitution: f32 = 0.6, // Bounciness [0..1]
    friction: f32 = 0.25,
    linear_damping: f32 = 0.015,
    angular_damping: f32 = 0.04,

    use_gravity: bool = true,
    is_grounded: bool = false,
    enabled: bool = true,

    box_extents: Vec3 = Vec3.new(0.5, 0.5, 0.5),
    sphere_radius: f32 = 0.5,

    pub fn init(mesh: *Mesh, collider: ColliderType, mass: f32) RigidBody {
        const inv_m = if (mass > 0.0) 1.0 / mass else 0.0;
        const aabb = mesh.local_bounding_box;
        const ext = aabb.extents();
        const rad = @max(ext.x, @max(ext.y, ext.z));

        return .{
            .mesh = mesh,
            .collider = collider,
            .mass = mass,
            .inv_mass = inv_m,
            .box_extents = if (ext.lengthSq() > 1e-6) ext else Vec3.new(0.5, 0.5, 0.5),
            .sphere_radius = if (rad > 1e-4) rad else 0.5,
        };
    }

    pub fn applyImpulse(self: *RigidBody, impulse: Vec3) void {
        if (self.inv_mass == 0.0) return;
        self.velocity = self.velocity.add(impulse.scale(self.inv_mass));
        self.is_grounded = false;
    }

    pub fn applyTorqueImpulse(self: *RigidBody, torque_impulse: Vec3) void {
        if (self.inv_mass == 0.0) return;
        self.angular_velocity = self.angular_velocity.add(torque_impulse.scale(self.inv_mass));
    }

    pub fn applyForce(self: *RigidBody, force: Vec3, dt: f32) void {
        if (self.inv_mass == 0.0) return;
        self.velocity = self.velocity.add(force.scale(dt * self.inv_mass));
    }

    pub fn setMass(self: *RigidBody, mass: f32) void {
        self.mass = mass;
        self.inv_mass = if (mass > 0.0) 1.0 / mass else 0.0;
    }
};

pub const PhysicsWorld = struct {
    allocator: std.mem.Allocator,
    gravity: Vec3 = Vec3.new(0.0, -9.81, 0.0),
    ground_y: ?f32 = -1.2,
    substeps: u32 = 2,
    bodies: std.ArrayListUnmanaged(*RigidBody) = .empty,

    pub fn init(allocator: std.mem.Allocator) PhysicsWorld {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *PhysicsWorld) void {
        for (self.bodies.items) |b| {
            self.allocator.destroy(b);
        }
        self.bodies.deinit(self.allocator);
    }

    pub fn createBody(self: *PhysicsWorld, mesh: *Mesh, collider: ColliderType, mass: f32) !*RigidBody {
        const body = try self.allocator.create(RigidBody);
        body.* = RigidBody.init(mesh, collider, mass);
        try self.bodies.append(self.allocator, body);
        return body;
    }

    pub fn removeBody(self: *PhysicsWorld, body: *RigidBody) void {
        for (self.bodies.items, 0..) |b, idx| {
            if (b == body) {
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
        if (dt <= 0.0001 or self.bodies.items.len == 0) return;
        const clamped_dt = @min(dt, 0.05); // Avoid spiral of death
        const substep_dt = clamped_dt / @as(f32, @floatFromInt(self.substeps));

        var sub: u32 = 0;
        while (sub < self.substeps) : (sub += 1) {
            self.internalSubstep(substep_dt);
        }
    }

    fn internalSubstep(self: *PhysicsWorld, dt: f32) void {
        // 1. Integrate forces and positions
        for (self.bodies.items) |b| {
            if (!b.enabled or b.inv_mass == 0.0) continue;

            if (b.use_gravity) {
                b.velocity = b.velocity.add(self.gravity.scale(dt));
            }

            // Damping
            const lin_damp = std.math.clamp(1.0 - b.linear_damping * dt * 60.0, 0.0, 1.0);
            const ang_damp = std.math.clamp(1.0 - b.angular_damping * dt * 60.0, 0.0, 1.0);
            b.velocity = b.velocity.scale(lin_damp);
            b.angular_velocity = b.angular_velocity.scale(ang_damp);

            b.mesh.position = b.mesh.position.add(b.velocity.scale(dt));
            b.mesh.rotation = b.mesh.rotation.add(b.angular_velocity.scale(dt));

            // 2. Ground collision
            if (self.ground_y) |gy| {
                var bottom_y = b.mesh.position.y;
                if (b.collider == .sphere) {
                    bottom_y -= b.sphere_radius * b.mesh.scaling.y;
                } else {
                    bottom_y -= b.box_extents.y * b.mesh.scaling.y;
                }

                if (bottom_y < gy) {
                    const penetration = gy - bottom_y;
                    b.mesh.position.y += penetration;

                    if (b.velocity.y < 0.0) {
                        b.velocity.y = -b.velocity.y * b.restitution;

                        const f_factor = std.math.clamp(1.0 - b.friction, 0.0, 1.0);
                        b.velocity.x *= f_factor;
                        b.velocity.z *= f_factor;
                        b.angular_velocity.x *= f_factor;
                        b.angular_velocity.z *= f_factor;

                        if (@abs(b.velocity.y) < 0.25) {
                            b.velocity.y = 0.0;
                            b.is_grounded = true;
                        }
                    }
                } else {
                    b.is_grounded = false;
                }
            }
        }

        // 3. Pairwise dynamic body-body collision
        const count = self.bodies.items.len;
        var i: usize = 0;
        while (i < count) : (i += 1) {
            const a = self.bodies.items[i];
            if (!a.enabled) continue;

            var j = i + 1;
            while (j < count) : (j += 1) {
                const b = self.bodies.items[j];
                if (!b.enabled) continue;
                if (a.inv_mass == 0.0 and b.inv_mass == 0.0) continue;

                self.resolveCollision(a, b);
            }
        }
    }

    fn resolveCollision(_: *PhysicsWorld, a: *RigidBody, b: *RigidBody) void {
        const pos_a = a.mesh.position;
        const pos_b = b.mesh.position;
        const center_delta = pos_b.sub(pos_a);
        const center_dist_sq = center_delta.lengthSq();

        // Broadphase early rejection
        const bound_r_a = if (a.collider == .sphere) a.sphere_radius * a.mesh.scaling.x else a.box_extents.length() * a.mesh.scaling.x;
        const bound_r_b = if (b.collider == .sphere) b.sphere_radius * b.mesh.scaling.x else b.box_extents.length() * b.mesh.scaling.x;
        const max_dist = bound_r_a + bound_r_b;
        if (center_dist_sq > max_dist * max_dist or center_dist_sq < 1e-8) return;

        if (a.collider == .sphere and b.collider == .sphere) {
            // Sphere vs Sphere
            const r_a = a.sphere_radius * a.mesh.scaling.x;
            const r_b = b.sphere_radius * b.mesh.scaling.x;
            const min_dist = r_a + r_b;

            if (center_dist_sq < min_dist * min_dist) {
                const dist = @sqrt(center_dist_sq);
                const normal = center_delta.scale(1.0 / dist);
                const penetration = min_dist - dist;
                const total_inv = a.inv_mass + b.inv_mass;

                if (total_inv > 0.0) {
                    if (a.inv_mass > 0.0) a.mesh.position = a.mesh.position.sub(normal.scale(penetration * (a.inv_mass / total_inv)));
                    if (b.inv_mass > 0.0) b.mesh.position = b.mesh.position.add(normal.scale(penetration * (b.inv_mass / total_inv)));

                    const rel_vel = b.velocity.sub(a.velocity);
                    const v_norm = rel_vel.dot(normal);
                    if (v_norm < 0.0) {
                        const e = (a.restitution + b.restitution) * 0.5;
                        const j_mag = -(1.0 + e) * v_norm / total_inv;
                        const impulse = normal.scale(j_mag);
                        if (a.inv_mass > 0.0) a.velocity = a.velocity.sub(impulse.scale(a.inv_mass));
                        if (b.inv_mass > 0.0) b.velocity = b.velocity.add(impulse.scale(b.inv_mass));
                    }
                }
            }
        } else if (a.collider == .sphere or b.collider == .sphere) {
            // Sphere vs Box
            const s = if (a.collider == .sphere) a else b;
            const bx = if (a.collider == .sphere) b else a;

            const s_center = s.mesh.position;
            const s_radius = s.sphere_radius * s.mesh.scaling.x;
            const b_box = bx.mesh.getWorldBoundingBox();
            const closest = b_box.closestPoint(s_center);
            const delta = s_center.sub(closest);
            const dist_sq = delta.lengthSq();

            if (dist_sq < s_radius * s_radius and dist_sq > 1e-8) {
                const dist = @sqrt(dist_sq);
                const normal = delta.scale(1.0 / dist);
                const penetration = s_radius - dist;
                const total_inv = s.inv_mass + bx.inv_mass;

                if (total_inv > 0.0) {
                    if (s.inv_mass > 0.0) s.mesh.position = s.mesh.position.add(normal.scale(penetration * (s.inv_mass / total_inv)));
                    if (bx.inv_mass > 0.0) bx.mesh.position = bx.mesh.position.sub(normal.scale(penetration * (bx.inv_mass / total_inv)));

                    const rel_vel = s.velocity.sub(bx.velocity);
                    const v_norm = rel_vel.dot(normal);
                    if (v_norm < 0.0) {
                        const e = (s.restitution + bx.restitution) * 0.5;
                        const j_mag = -(1.0 + e) * v_norm / total_inv;
                        const impulse = normal.scale(j_mag);
                        if (s.inv_mass > 0.0) s.velocity = s.velocity.add(impulse.scale(s.inv_mass));
                        if (bx.inv_mass > 0.0) bx.velocity = bx.velocity.sub(impulse.scale(bx.inv_mass));
                    }
                }
            }
        } else {
            // Box vs Box (AABB)
            const box_a = a.mesh.getWorldBoundingBox();
            const box_b = b.mesh.getWorldBoundingBox();
            if (box_a.intersects(box_b)) {
                const overlap_x = @min(box_a.max.x, box_b.max.x) - @max(box_a.min.x, box_b.min.x);
                const overlap_y = @min(box_a.max.y, box_b.max.y) - @max(box_a.min.y, box_b.min.y);
                const overlap_z = @min(box_a.max.z, box_b.max.z) - @max(box_a.min.z, box_b.min.z);

                var min_overlap = overlap_x;
                var normal = Vec3.new(if (b.mesh.position.x > a.mesh.position.x) 1.0 else -1.0, 0.0, 0.0);

                if (overlap_y < min_overlap) {
                    min_overlap = overlap_y;
                    normal = Vec3.new(0.0, if (b.mesh.position.y > a.mesh.position.y) 1.0 else -1.0, 0.0);
                }
                if (overlap_z < min_overlap) {
                    min_overlap = overlap_z;
                    normal = Vec3.new(0.0, 0.0, if (b.mesh.position.z > a.mesh.position.z) 1.0 else -1.0);
                }

                const total_inv = a.inv_mass + b.inv_mass;
                if (total_inv > 0.0) {
                    if (a.inv_mass > 0.0) a.mesh.position = a.mesh.position.sub(normal.scale(min_overlap * (a.inv_mass / total_inv)));
                    if (b.inv_mass > 0.0) b.mesh.position = b.mesh.position.add(normal.scale(min_overlap * (b.inv_mass / total_inv)));

                    const rel_vel = b.velocity.sub(a.velocity);
                    const v_norm = rel_vel.dot(normal);
                    if (v_norm < 0.0) {
                        const e = (a.restitution + b.restitution) * 0.5;
                        const j_mag = -(1.0 + e) * v_norm / total_inv;
                        const impulse = normal.scale(j_mag);
                        if (a.inv_mass > 0.0) a.velocity = a.velocity.sub(impulse.scale(a.inv_mass));
                        if (b.inv_mass > 0.0) b.velocity = b.velocity.add(impulse.scale(b.inv_mass));
                    }
                }
            }
        }
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
