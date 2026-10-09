//! Ready-made wheeled-vehicle helper on top of Box3D wheel joints.
//!
//! Builds a box chassis plus four sphere wheels linked by sprung wheel joints
//! with spin motors and front-axle steering, mirroring the manual car in the
//! sandbox demo (suspension along frame A x-axis rotated onto world -Y,
//! negative spin speed drives forward when forward is +X).
//! Meshes are optional: pass a scene to get visible box/sphere meshes, or
//! null for a physics-only vehicle (unit tests, headless servers).
//! The world pointer is retained for per-frame motor updates; the world
//! itself is still owned by the caller.

const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Quat = math.Quat;
const physics = @import("physics.zig");
const PhysicsWorld = physics.PhysicsWorld;
const RigidBody = physics.RigidBody;
const JointId = physics.JointId;
const mesh_mod = @import("mesh.zig");
const Mesh = mesh_mod.Mesh;
const physics_mesh = @import("physics_mesh.zig");
const Scene = @import("scene.zig").Scene;

/// Number of wheels. Indices 0..3 follow `VehicleOptions.wheel_offsets`.
pub const wheel_count: usize = 4;

pub const VehicleOptions = struct {
    /// Spawn position of the chassis center.
    position: Vec3 = Vec3.zero,
    chassis_size: Vec3 = Vec3.new(2.2, 0.5, 1.0),
    chassis_mass: f32 = 4.0,
    wheel_radius: f32 = 0.35,
    wheel_mass: f32 = 0.8,
    /// Wheel centers relative to the chassis center. Front axle is +X:
    /// wheels with a positive x offset steer when `steer_front` is set.
    wheel_offsets: [wheel_count]Vec3 = .{
        Vec3.new(-0.75, -0.35, 0.55),
        Vec3.new(0.75, -0.35, 0.55),
        Vec3.new(-0.75, -0.35, -0.55),
        Vec3.new(0.75, -0.35, -0.55),
    },
    /// Suspension spring stiffness.
    suspension_hertz: f32 = 8.0,
    /// Suspension damping ratio.
    suspension_damping: f32 = 0.7,
    /// Suspension travel in meters (limits become [-travel, +travel]).
    suspension_travel: f32 = 0.2,
    /// Target wheel spin speed (rad/s) at full throttle.
    drive_speed: f32 = 12.0,
    /// Spin motor torque budget.
    drive_torque: f32 = 30.0,
    /// Steering angle (radians) at full steering lock.
    max_steer_angle: f32 = 0.45,
    /// Steering servo torque budget.
    steer_torque: f32 = 25.0,
    /// Spin motor torque budget used to hold the wheels at zero speed while
    /// braking (scaled by the brake input 0..1).
    brake_torque: f32 = 40.0,
    drive_front: bool = true,
    drive_rear: bool = true,
    steer_front: bool = true,
    steer_rear: bool = false,
    chassis_friction: f32 = 0.6,
    chassis_restitution: f32 = 0.2,
    wheel_friction: f32 = 0.8,
    wheel_restitution: f32 = 0.1,
};

pub const RaycastVehicle = struct {
    allocator: std.mem.Allocator,
    world: *PhysicsWorld,
    scene: ?*Scene,
    /// Bodies in order: chassis first, then the four wheels.
    bodies: std.ArrayListUnmanaged(*RigidBody) = .empty,
    /// One wheel joint per wheel, in wheel order.
    joints: std.ArrayListUnmanaged(JointId) = .empty,
    meshes: std.ArrayListUnmanaged(*Mesh) = .empty,
    throttle: f32 = 0.0,
    steering: f32 = 0.0,
    brake: f32 = 0.0,
    drive_speed: f32 = 12.0,
    drive_torque: f32 = 30.0,
    max_steer_angle: f32 = 0.45,
    steer_torque: f32 = 25.0,
    brake_torque: f32 = 40.0,
    drive: [wheel_count]bool = .{true} ** wheel_count,
    steer: [wheel_count]bool = .{false} ** wheel_count,

    /// Builds the chassis, wheels, joints and optional meshes. On error any
    /// partial state is torn down, so the caller only deinits on success.
    /// Deinit the vehicle before the scene and before the world.
    pub fn init(
        allocator: std.mem.Allocator,
        world: *PhysicsWorld,
        scene: ?*Scene,
        options: VehicleOptions,
    ) !RaycastVehicle {
        var self = RaycastVehicle{
            .allocator = allocator,
            .world = world,
            .scene = scene,
            .drive_speed = options.drive_speed,
            .drive_torque = options.drive_torque,
            .max_steer_angle = options.max_steer_angle,
            .steer_torque = options.steer_torque,
            .brake_torque = options.brake_torque,
        };
        errdefer self.deinit(world);

        for (0..wheel_count) |i| {
            const is_front = options.wheel_offsets[i].x > 0.0;
            self.drive[i] = if (is_front) options.drive_front else options.drive_rear;
            self.steer[i] = if (is_front) options.steer_front else options.steer_rear;
        }

        // Chassis body.
        const chassis_mesh = try physics_mesh.createBoxMesh(
            allocator,
            scene,
            "vehicle_chassis",
            options.chassis_size,
            options.position,
        );
        var chassis_mesh_tracked = false;
        defer if (!chassis_mesh_tracked) physics_mesh.freeMesh(allocator, scene, chassis_mesh);
        try self.meshes.append(allocator, chassis_mesh);
        chassis_mesh_tracked = true;

        const chassis = try world.createBody(chassis_mesh, .box, options.chassis_mass);
        var chassis_tracked = false;
        defer if (!chassis_tracked) world.removeBody(chassis);
        chassis.friction = options.chassis_friction;
        chassis.restitution = options.chassis_restitution;
        try self.bodies.append(allocator, chassis);
        chassis_tracked = true;

        // Suspension acts along frame A x-axis: rotate it onto world -Y.
        const frame_a = Quat.fromEulerDeg(Vec3.new(0.0, 0.0, -90.0));
        const travel = @max(options.suspension_travel, 0.0);

        for (0..wheel_count) |i| {
            const wheel_pos = options.position.add(options.wheel_offsets[i]);
            const wheel_mesh = try physics_mesh.createSphereMesh(
                allocator,
                scene,
                wheelMeshName(i),
                options.wheel_radius * 2.0,
                wheel_pos,
            );
            var wheel_mesh_tracked = false;
            defer if (!wheel_mesh_tracked) physics_mesh.freeMesh(allocator, scene, wheel_mesh);
            try self.meshes.append(allocator, wheel_mesh);
            wheel_mesh_tracked = true;

            const wheel = try world.createBody(wheel_mesh, .sphere, options.wheel_mass);
            var wheel_tracked = false;
            defer if (!wheel_tracked) world.removeBody(wheel);
            wheel.friction = options.wheel_friction;
            wheel.restitution = options.wheel_restitution;
            try self.bodies.append(allocator, wheel);
            wheel_tracked = true;

            const jid = try world.createWheelJointWorld(chassis, wheel, wheel_pos, .{
                .frame_a = frame_a,
                .suspension_hertz = options.suspension_hertz,
                .suspension_damping = options.suspension_damping,
                .enable_suspension_limit = true,
                .lower_suspension_limit = -travel,
                .upper_suspension_limit = travel,
                .enable_spin_motor = false,
                .max_spin_torque = options.drive_torque,
                .enable_steering = self.steer[i],
                .steering_hertz = 8.0,
                .steering_damping = 1.0,
                .max_steering_torque = options.steer_torque,
                .enable_steering_limit = self.steer[i],
                .lower_steering_limit_rad = -options.max_steer_angle,
                .upper_steering_limit_rad = options.max_steer_angle,
            });
            var joint_tracked = false;
            defer if (!joint_tracked and world.isJointValid(jid)) world.destroyJoint(jid);
            try self.joints.append(allocator, jid);
            joint_tracked = true;
        }

        return self;
    }

    /// Chassis body (bodies[0]).
    pub fn chassisBody(self: *RaycastVehicle) *RigidBody {
        return self.bodies.items[0];
    }

    /// Wheel body by wheel index 0..3.
    pub fn wheelBody(self: *RaycastVehicle, index: usize) *RigidBody {
        return self.bodies.items[1 + index];
    }

    /// Wheel joint by wheel index 0..3.
    pub fn wheelJoint(self: *RaycastVehicle, index: usize) JointId {
        return self.joints.items[index];
    }

    /// Drive input, clamped to [-1, 1]. Positive drives forward (+X).
    pub fn setThrottle(self: *RaycastVehicle, value: f32) void {
        self.throttle = std.math.clamp(value, -1.0, 1.0);
    }

    /// Steering input, clamped to [-1, 1]. Positive steers left.
    pub fn setSteering(self: *RaycastVehicle, value: f32) void {
        self.steering = std.math.clamp(value, -1.0, 1.0);
    }

    /// Brake input, clamped to [0, 1].
    pub fn setBrake(self: *RaycastVehicle, value: f32) void {
        self.brake = std.math.clamp(value, 0.0, 1.0);
    }

    /// Applies the current throttle/steering/brake inputs to the wheel
    /// motors. Call once per frame before `world.step`.
    pub fn update(self: *RaycastVehicle, dt: f32) void {
        _ = dt;
        const world = self.world;
        for (0..wheel_count) |i| {
            const jid = self.joints.items[i];
            if (!world.isJointValid(jid)) continue;
            if (self.brake > 0.01) {
                // Hold the wheels at zero speed with a brake-scaled torque.
                world.setWheelSpin(jid, true, 0.0, self.brake_torque * self.brake);
            } else if (self.drive[i] and @abs(self.throttle) > 1e-4) {
                // Negative spin speed drives forward (+X), matching the
                // sandbox car convention.
                world.setWheelSpin(jid, true, -self.throttle * self.drive_speed, self.drive_torque);
            } else {
                world.setWheelSpin(jid, false, 0.0, 0.0);
            }
            if (self.steer[i]) {
                world.setWheelSteering(jid, true, self.steering * self.max_steer_angle, self.steer_torque);
            }
        }
    }

    /// Copies the last solver-synced body transforms into the meshes.
    pub fn syncMeshes(self: *RaycastVehicle) void {
        physics_mesh.syncMeshes(self.bodies.items);
    }

    /// Destroys joints, removes bodies from the world and frees meshes/lists.
    /// Scene meshes are unlinked from the scene first. Call before destroying
    /// the scene and the world.
    pub fn deinit(self: *RaycastVehicle, world: *PhysicsWorld) void {
        physics_mesh.teardown(
            self.allocator,
            world,
            self.scene,
            &self.joints,
            &self.bodies,
            &self.meshes,
        );
    }
};

fn wheelMeshName(index: usize) []const u8 {
    return switch (index) {
        0 => "vehicle_wheel_0",
        1 => "vehicle_wheel_1",
        2 => "vehicle_wheel_2",
        else => "vehicle_wheel_3",
    };
}
