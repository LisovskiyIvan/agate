const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const Quat = math.Quat;
const Mesh = @import("../mesh.zig").Mesh;
const c = @import("../c.zig").c;

const physics = @import("../physics.zig");
const PhysicsWorld = physics.PhysicsWorld;
const RigidBody = physics.RigidBody;
const CharacterController = physics.CharacterController;
const Rope = physics.Rope;
const RopeOptions = physics.RopeOptions;
const DebugLine = physics.DebugLine;
const CollisionFilter = physics.CollisionFilter;
const SensorEvent = physics.SensorEvent;
const ContactEvent = physics.ContactEvent;
const ContactHitEvent = physics.ContactHitEvent;
const ChildShape = physics.ChildShape;
const ChildShapeOptions = physics.ChildShapeOptions;
const ColliderType = physics.ColliderType;
const JointId = physics.JointId;
const HeightFieldOptions = physics.HeightFieldOptions;
const DistanceJointOptions = physics.DistanceJointOptions;
const SphericalJointOptions = physics.SphericalJointOptions;
const RevoluteJointOptions = physics.RevoluteJointOptions;
const WheelJointOptions = physics.WheelJointOptions;
const MotorJointOptions = physics.MotorJointOptions;
const WeldJointOptions = physics.WeldJointOptions;
const ParallelJointOptions = physics.ParallelJointOptions;
const PrismaticJointOptions = physics.PrismaticJointOptions;

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

    // A 10x10 flat quad (2 triangles) at y = 0. CCW seen from +Y: Box3D
    // treats the opposite winding as back faces and ignores them, so the
    // ball would fall straight through.
    var positions = [_]Vec3{
        Vec3.new(-5.0, 0.0, -5.0),
        Vec3.new(5.0, 0.0, -5.0),
        Vec3.new(5.0, 0.0, 5.0),
        Vec3.new(-5.0, 0.0, 5.0),
    };
    var indices = [_]u32{ 0, 2, 1, 0, 3, 2 };

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
    try std.testing.expectError(error.InvalidHeightFieldDimensions, pw.createHeightField(&terrain_mesh, &heights, 5, 6, .{}));
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

    // Ray above the ball still hits the height field below it. The field
    // spans [0, count-1] per axis (Box3D corner-origin layout), so the ray
    // must target a point inside that footprint.
    const floor_hit = pw.raycast(Vec3.new(3.5, 6.0, 3.5), Vec3.new(0.0, -1.0, 0.0), 20.0);
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

fn makeWheelTestRig(
    pw: *PhysicsWorld,
    chassis_mesh: *Mesh,
    wheel_meshes: *[4]Mesh,
) !struct {
    chassis: *RigidBody,
    wheels: [4]*RigidBody,
    joints: [4]JointId,
} {
    // Mesh storage is caller-owned: bodies keep `mesh` pointers alive across
    // steps, so they must not point into this function's stack frame.
    chassis_mesh.* = .{
        .name = "wheel_chassis",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 0.5, 0.0),
        .local_bounding_box = BoundingBox.init(Vec3.new(-1.0, -0.25, -0.5), Vec3.new(1.0, 0.25, 0.5)),
    };
    const chassis = try pw.createBody(chassis_mesh, .box, 4.0);

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

    var chassis_mesh: Mesh = undefined;
    var wheel_meshes: [4]Mesh = undefined;
    const rig = try makeWheelTestRig(&pw, &chassis_mesh, &wheel_meshes);
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

test "PhysicsWorld continuous collision detection and bullet flags" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    // Default CCD state
    try std.testing.expect(pw.isContinuousEnabled());
    pw.enableContinuous(false);
    try std.testing.expect(!pw.isContinuousEnabled());
    pw.enableContinuous(true);
    try std.testing.expect(pw.isContinuousEnabled());

    var m = Mesh{
        .name = "bullet_mesh",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
        .position = Vec3.new(0.0, 5.0, 0.0),
    };
    const b = try pw.createBodyWith(&m, .sphere, 1.0, .{ .is_bullet = true });
    try std.testing.expect(b.isBullet());
    b.setBullet(false);
    try std.testing.expect(!b.isBullet());
    b.setBullet(true);
    try std.testing.expect(b.isBullet());
}

test "CharacterController push dynamic bodies and step climbing options" {
    var pw = PhysicsWorld.init(std.testing.allocator);
    defer pw.deinit();

    var ctrl = @import("character.zig").CharacterController.init(Vec3.new(0.0, 0.0, 0.0), 0.3, 1.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.35), ctrl.step_height, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), ctrl.push_strength, 1e-4);
    try std.testing.expect(ctrl.push_dynamic_bodies);

    // Mover advances without error
    ctrl.move(&pw, Vec3.new(1.0, 0.0, 0.0), false, 1.0 / 60.0);
    try std.testing.expect(ctrl.position.x > 0.0);
}
