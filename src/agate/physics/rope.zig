const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const c = @import("../c.zig").c;
const body_mod = @import("body.zig");
const RigidBody = body_mod.RigidBody;
const types = @import("types.zig");
const ColliderType = types.ColliderType;
const JointId = types.JointId;
const DistanceJointOptions = types.DistanceJointOptions;

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

    pub fn deinit(self: *Rope, world: anytype) void {
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
    pub fn detachAt(self: *Rope, world: anytype, index: usize) void {
        if (index >= self.joints.items.len) return;
        const j = self.joints.items[index];
        if (!c.b3Joint_IsValid(j)) return;
        world.destroyJoint(j);
    }

    /// Recreates every missing link/pin joint (after detachAt or reset).
    pub fn repair(self: *Rope, world: anytype) void {
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
    pub fn reset(self: *Rope, world: anytype) void {
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

    pub fn linkJointOptions(self: *const Rope) DistanceJointOptions {
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
