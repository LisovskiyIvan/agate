//! Joint creation, tuning and destruction as free functions.
//! Extracted from `PhysicsWorld` methods in `physics.zig`; behavior unchanged.
//! Each function takes `world: anytype` (concretely `*PhysicsWorld`) so this
//! module never imports `physics.zig` (no import cycle).
const math = @import("math");
const Vec3 = math.Vec3;
const c = @import("../c.zig").c;
const convert = @import("convert.zig");
const types = @import("types.zig");
const body_mod = @import("body.zig");
const RigidBody = body_mod.RigidBody;
const JointId = types.JointId;
const DistanceJointOptions = types.DistanceJointOptions;
const SphericalJointOptions = types.SphericalJointOptions;
const RevoluteJointOptions = types.RevoluteJointOptions;
const WheelJointOptions = types.WheelJointOptions;
const PrismaticJointOptions = types.PrismaticJointOptions;
const MotorJointOptions = types.MotorJointOptions;
const WeldJointOptions = types.WeldJointOptions;
const ParallelJointOptions = types.ParallelJointOptions;
const toB3Vec = convert.toB3Vec;
const fromB3Vec = convert.fromB3Vec;
const toB3Quat = convert.toB3Quat;
const rad2deg = convert.rad2deg;

pub fn createDistanceJoint(
    world: anytype,
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

    const jid = c.b3CreateDistanceJoint(world.world_id, &def);
    try world.joints.append(world.allocator, jid);
    return jid;
}

pub fn createDistanceJointWorld(
    world: anytype,
    body_a: *RigidBody,
    body_b: *RigidBody,
    world_anchor_a: Vec3,
    world_anchor_b: Vec3,
    options: DistanceJointOptions,
) !JointId {
    return createDistanceJoint(
        world,
        body_a,
        body_b,
        body_a.worldToLocal(world_anchor_a),
        body_b.worldToLocal(world_anchor_b),
        options,
    );
}

pub fn createSphericalJoint(
    world: anytype,
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

    const jid = c.b3CreateSphericalJoint(world.world_id, &def);
    try world.joints.append(world.allocator, jid);
    return jid;
}

pub fn createSphericalJointWorld(
    world: anytype,
    body_a: *RigidBody,
    body_b: *RigidBody,
    world_anchor: Vec3,
    options: SphericalJointOptions,
) !JointId {
    return createSphericalJoint(
        world,
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
    world: anytype,
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

    const jid = c.b3CreateRevoluteJoint(world.world_id, &def);
    try world.joints.append(world.allocator, jid);
    return jid;
}

/// Creates a hinge joint from a shared world-space pivot point.
pub fn createRevoluteJointWorld(
    world: anytype,
    body_a: *RigidBody,
    body_b: *RigidBody,
    world_anchor: Vec3,
    options: RevoluteJointOptions,
) !JointId {
    return createRevoluteJoint(
        world,
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
    world: anytype,
    joint_id: JointId,
    enabled: bool,
    motor_speed_rad: f32,
    max_motor_torque: f32,
) void {
    _ = world;
    c.b3RevoluteJoint_EnableMotor(joint_id, enabled);
    c.b3RevoluteJoint_SetMotorSpeed(joint_id, motor_speed_rad);
    c.b3RevoluteJoint_SetMaxMotorTorque(joint_id, max_motor_torque);
    c.b3Joint_WakeBodies(joint_id);
}

/// Enables and sets the hinge angular limits (radians, [-0.99*pi .. 0.99*pi]).
pub fn setRevoluteLimits(
    world: anytype,
    joint_id: JointId,
    lower_angle_rad: f32,
    upper_angle_rad: f32,
) void {
    _ = world;
    c.b3RevoluteJoint_SetLimits(joint_id, lower_angle_rad, upper_angle_rad);
    c.b3RevoluteJoint_EnableLimit(joint_id, true);
    c.b3Joint_WakeBodies(joint_id);
}

/// Current hinge angle (radians), relative to the reference angle at creation.
pub fn revoluteAngleRad(world: anytype, joint_id: JointId) f32 {
    _ = world;
    return c.b3RevoluteJoint_GetAngle(joint_id);
}

/// Current hinge angle in degrees.
pub fn revoluteAngleDeg(world: anytype, joint_id: JointId) f32 {
    _ = world;
    return c.b3RevoluteJoint_GetAngle(joint_id) * rad2deg;
}

/// Creates a wheel joint (body A = chassis, body B = wheel). The wheel
/// spins around the z-axis of frame B and suspends/steers around the
/// x-axis of frame A. Suspension travel is measured from the creation
/// pose, so build the vehicle at rest ride height.
pub fn createWheelJoint(
    world: anytype,
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

    const jid = c.b3CreateWheelJoint(world.world_id, &def);
    try world.joints.append(world.allocator, jid);
    return jid;
}

/// Creates a wheel joint from a shared world-space anchor point.
pub fn createWheelJointWorld(
    world: anytype,
    body_a: *RigidBody,
    body_b: *RigidBody,
    world_anchor: Vec3,
    options: WheelJointOptions,
) !JointId {
    return createWheelJoint(
        world,
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
    world: anytype,
    joint_id: JointId,
    enabled: bool,
    spin_speed_rad: f32,
    max_spin_torque: f32,
) void {
    _ = world;
    c.b3WheelJoint_EnableSpinMotor(joint_id, enabled);
    c.b3WheelJoint_SetSpinMotorSpeed(joint_id, spin_speed_rad);
    c.b3WheelJoint_SetMaxSpinTorque(joint_id, max_spin_torque);
    c.b3Joint_WakeBodies(joint_id);
}

/// Enables/disables steering and sets the target angle (radians).
pub fn setWheelSteering(
    world: anytype,
    joint_id: JointId,
    enabled: bool,
    target_angle_rad: f32,
    max_steering_torque: f32,
) void {
    _ = world;
    c.b3WheelJoint_EnableSteering(joint_id, enabled);
    c.b3WheelJoint_SetTargetSteeringAngle(joint_id, target_angle_rad);
    c.b3WheelJoint_SetMaxSteeringTorque(joint_id, max_steering_torque);
    c.b3Joint_WakeBodies(joint_id);
}

/// Current wheel spin speed (rad/s), relative between wheel and chassis.
pub fn wheelSpinSpeed(world: anytype, joint_id: JointId) f32 {
    _ = world;
    return c.b3WheelJoint_GetSpinSpeed(joint_id);
}

/// Current steering angle (radians).
pub fn wheelSteeringAngle(world: anytype, joint_id: JointId) f32 {
    _ = world;
    return c.b3WheelJoint_GetSteeringAngle(joint_id);
}

/// Creates a prismatic (slider) joint. Body B translates along frame A
/// x-axis with rotation locked. Build it in the rest pose: limits and the
/// reported translation are relative to creation.
pub fn createPrismaticJoint(
    world: anytype,
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

    const jid = c.b3CreatePrismaticJoint(world.world_id, &def);
    try world.joints.append(world.allocator, jid);
    return jid;
}

/// Creates a slider joint from a shared world-space anchor point.
pub fn createPrismaticJointWorld(
    world: anytype,
    body_a: *RigidBody,
    body_b: *RigidBody,
    world_anchor: Vec3,
    options: PrismaticJointOptions,
) !JointId {
    return createPrismaticJoint(
        world,
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
    world: anytype,
    joint_id: JointId,
    enabled: bool,
    motor_speed: f32,
    max_motor_force: f32,
) void {
    _ = world;
    c.b3PrismaticJoint_EnableMotor(joint_id, enabled);
    c.b3PrismaticJoint_SetMotorSpeed(joint_id, motor_speed);
    c.b3PrismaticJoint_SetMaxMotorForce(joint_id, max_motor_force);
    c.b3Joint_WakeBodies(joint_id);
}

/// Enables and sets the slider travel limits (meters, from rest pose).
pub fn setPrismaticLimits(
    world: anytype,
    joint_id: JointId,
    lower_translation: f32,
    upper_translation: f32,
) void {
    _ = world;
    c.b3PrismaticJoint_SetLimits(joint_id, lower_translation, upper_translation);
    c.b3PrismaticJoint_EnableLimit(joint_id, true);
    c.b3Joint_WakeBodies(joint_id);
}

/// Current slider translation (meters) relative to the creation pose.
pub fn prismaticTranslation(world: anytype, joint_id: JointId) f32 {
    _ = world;
    return c.b3PrismaticJoint_GetTranslation(joint_id);
}

/// Current slider speed (m/s).
pub fn prismaticSpeed(world: anytype, joint_id: JointId) f32 {
    _ = world;
    return c.b3PrismaticJoint_GetSpeed(joint_id);
}

/// Creates a motor joint driving body B with a velocity motor and an
/// optional pose spring. With no spring/force configured the joint is
/// inert until per-frame velocity targets are set.
pub fn createMotorJoint(
    world: anytype,
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

    const jid = c.b3CreateMotorJoint(world.world_id, &def);
    try world.joints.append(world.allocator, jid);
    return jid;
}

/// Creates a motor joint from a shared world-space anchor point.
pub fn createMotorJointWorld(
    world: anytype,
    body_a: *RigidBody,
    body_b: *RigidBody,
    world_anchor: Vec3,
    options: MotorJointOptions,
) !JointId {
    return createMotorJoint(
        world,
        body_a,
        body_b,
        body_a.worldToLocal(world_anchor),
        body_b.worldToLocal(world_anchor),
        options,
    );
}

/// Sets the motor linear velocity target (m/s). Wakes the bodies.
pub fn setMotorLinearVelocity(world: anytype, joint_id: JointId, velocity: Vec3) void {
    _ = world;
    c.b3MotorJoint_SetLinearVelocity(joint_id, toB3Vec(velocity));
    c.b3Joint_WakeBodies(joint_id);
}

/// Sets the motor angular velocity target (rad/s). Wakes the bodies.
pub fn setMotorAngularVelocity(world: anytype, joint_id: JointId, velocity_rad: Vec3) void {
    _ = world;
    c.b3MotorJoint_SetAngularVelocity(joint_id, toB3Vec(velocity_rad));
    c.b3Joint_WakeBodies(joint_id);
}

/// Sets the linear motor force budget (N). Wakes the bodies.
pub fn setMotorMaxVelocityForce(world: anytype, joint_id: JointId, max_force: f32) void {
    _ = world;
    c.b3MotorJoint_SetMaxVelocityForce(joint_id, max_force);
    c.b3Joint_WakeBodies(joint_id);
}

/// Sets the angular motor torque budget (N*m). Wakes the bodies.
pub fn setMotorMaxVelocityTorque(world: anytype, joint_id: JointId, max_torque: f32) void {
    _ = world;
    c.b3MotorJoint_SetMaxVelocityTorque(joint_id, max_torque);
    c.b3Joint_WakeBodies(joint_id);
}

/// Creates a weld joint holding the creation relative pose of two bodies.
pub fn createWeldJoint(
    world: anytype,
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

    const jid = c.b3CreateWeldJoint(world.world_id, &def);
    try world.joints.append(world.allocator, jid);
    return jid;
}

/// Creates a weld joint from a shared world-space anchor point.
pub fn createWeldJointWorld(
    world: anytype,
    body_a: *RigidBody,
    body_b: *RigidBody,
    world_anchor: Vec3,
    options: WeldJointOptions,
) !JointId {
    return createWeldJoint(
        world,
        body_a,
        body_b,
        body_a.worldToLocal(world_anchor),
        body_b.worldToLocal(world_anchor),
        options,
    );
}

/// Current joint constraint force (N). Poll it to implement breakable
/// joints: destroy the joint once the load exceeds a threshold.
pub fn jointConstraintForce(world: anytype, joint_id: JointId) Vec3 {
    _ = world;
    return fromB3Vec(c.b3Joint_GetConstraintForce(joint_id));
}

/// Current joint constraint torque (N*m).
pub fn jointConstraintTorque(world: anytype, joint_id: JointId) Vec3 {
    _ = world;
    return fromB3Vec(c.b3Joint_GetConstraintTorque(joint_id));
}

/// Creates a parallel joint: a spring pulling the z-axis of body B
/// parallel to the z-axis of body A. Anchor points only define the
/// joint frames, not a position constraint.
pub fn createParallelJoint(
    world: anytype,
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

    const jid = c.b3CreateParallelJoint(world.world_id, &def);
    try world.joints.append(world.allocator, jid);
    return jid;
}

/// Creates a parallel joint from a shared world-space anchor point.
pub fn createParallelJointWorld(
    world: anytype,
    body_a: *RigidBody,
    body_b: *RigidBody,
    world_anchor: Vec3,
    options: ParallelJointOptions,
) !JointId {
    return createParallelJoint(
        world,
        body_a,
        body_b,
        body_a.worldToLocal(world_anchor),
        body_b.worldToLocal(world_anchor),
        options,
    );
}

/// Retunes the parallel spring at runtime. Wakes the bodies.
pub fn setParallelSpring(
    world: anytype,
    joint_id: JointId,
    hertz: f32,
    damping_ratio: f32,
    max_torque: f32,
) void {
    _ = world;
    c.b3ParallelJoint_SetSpringHertz(joint_id, hertz);
    c.b3ParallelJoint_SetSpringDampingRatio(joint_id, damping_ratio);
    c.b3ParallelJoint_SetMaxTorque(joint_id, max_torque);
    c.b3Joint_WakeBodies(joint_id);
}

/// True when the joint still exists in the world.
/// Takes no world: the original `_: *PhysicsWorld` parameter was unused.
pub fn isJointValid(joint_id: JointId) bool {
    return c.b3Joint_IsValid(joint_id);
}

pub fn destroyJoint(world: anytype, joint_id: JointId) void {
    c.b3DestroyJoint(joint_id, true);
    for (world.joints.items, 0..) |j, idx| {
        if (j.index1 == joint_id.index1 and j.generation == joint_id.generation) {
            _ = world.joints.swapRemove(idx);
            break;
        }
    }
}
