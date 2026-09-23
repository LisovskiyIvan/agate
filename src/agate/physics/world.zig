//! `PhysicsWorld` core: the world type with its fields and methods.
//! The type lives in this neutral module (not in the `physics.zig` facade)
//! so the `physics/*` submodules can name `*PhysicsWorld` directly in their
//! signatures. `physics.zig` imports this module for re-export and
//! the sibling submodules (queries, joints, debug, events, rope) import it
//! back for the type. That file-level import cycle is fine in Zig because
//! analysis is lazy: the struct fields never depend on the siblings, only
//! method bodies do, so no comptime "depends on itself" loop exists.
const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Quat = math.Quat;
const Mesh = @import("../mesh.zig").Mesh;
const c = @import("../c.zig").c;

const convert = @import("convert.zig");
const types = @import("types.zig");
const body_mod = @import("body.zig");
const debug_geo = @import("debug_geo.zig");
const physics_queries = @import("queries.zig");
const physics_joints = @import("joints.zig");
const physics_debug = @import("debug.zig");
const physics_events = @import("events.zig");
const rope_mod = @import("rope.zig");
const audio = @import("../audio.zig");

// Private aliases to the extracted helpers (behavior unchanged).
const step_h = convert.step_h;
const damp_scale = convert.damp_scale;
const deg2rad = convert.deg2rad;
const rad2deg = convert.rad2deg;
const min_half_extent = convert.min_half_extent;
const toB3Vec = convert.toB3Vec;
const fromB3Vec = convert.fromB3Vec;
const toB3Pos = convert.toB3Pos;
const fromB3Pos = convert.fromB3Pos;
const toB3Quat = convert.toB3Quat;
const fromB3Quat = convert.fromB3Quat;
const toB3Filter = types.toB3Filter;
const sameFilter = types.sameFilter;
const default_debug_circle_segments = debug_geo.default_debug_circle_segments;

// Private aliases for the option/event types used in the API surface below.
const ColliderType = types.ColliderType;
const HeightFieldOptions = types.HeightFieldOptions;
const CollisionFilter = types.CollisionFilter;
const BodyOptions = types.BodyOptions;
const ChildShapeOptions = types.ChildShapeOptions;
const JointId = types.JointId;
const DistanceJointOptions = types.DistanceJointOptions;
const SphericalJointOptions = types.SphericalJointOptions;
const RevoluteJointOptions = types.RevoluteJointOptions;
const WheelJointOptions = types.WheelJointOptions;
const MotorJointOptions = types.MotorJointOptions;
const WeldJointOptions = types.WeldJointOptions;
const ParallelJointOptions = types.ParallelJointOptions;
const PrismaticJointOptions = types.PrismaticJointOptions;
const DebugLine = types.DebugLine;
const PhysicsRayHit = body_mod.PhysicsRayHit;
const SensorEvent = body_mod.SensorEvent;
const ContactEvent = body_mod.ContactEvent;
const ContactHitEvent = body_mod.ContactHitEvent;
const ChildShape = body_mod.ChildShape;
const RigidBody = body_mod.RigidBody;
const Rope = rope_mod.Rope;
const RopeOptions = rope_mod.RopeOptions;

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

    pub fn enableContinuous(self: *PhysicsWorld, flag: bool) void {
        c.b3World_EnableContinuous(self.world_id, flag);
    }

    pub fn isContinuousEnabled(self: *const PhysicsWorld) bool {
        return c.b3World_IsContinuousEnabled(self.world_id);
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

        if (options.is_bullet) {
            body.setBullet(true);
        }

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
        return physics_queries.raycast(self, origin, direction, max_distance);
    }

    pub fn raycastWithFilter(
        self: *PhysicsWorld,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
        filter: CollisionFilter,
    ) PhysicsRayHit {
        return physics_queries.raycastWithFilter(self, origin, direction, max_distance, filter);
    }

    /// Raycast adapter matching audio.RaycastFn for audio occlusion queries.
    pub fn audioRaycastAdapter(origin: Vec3, direction: Vec3, max_distance: f32, user_data: ?*anyopaque) bool {
        const self: *PhysicsWorld = @ptrCast(@alignCast(user_data orelse return false));
        const hit = self.raycast(origin, direction, max_distance);
        return hit.hit;
    }

    /// Evaluates audio occlusion between listener and emitter through physics colliders.
    pub fn evaluateAudioOcclusion(
        self: *PhysicsWorld,
        listener_pos: Vec3,
        emitter_pos: Vec3,
        config: audio.AudioOcclusionConfig,
    ) f32 {
        return audio.evaluateRaycastOcclusion(listener_pos, emitter_pos, config, audioRaycastAdapter, self);
    }

    pub fn queryAABB(
        self: *PhysicsWorld,
        min: Vec3,
        max: Vec3,
        results: *std.ArrayListUnmanaged(*RigidBody),
    ) !void {
        return physics_queries.queryAABB(self, min, max, results);
    }

    pub fn queryAABBWithFilter(
        self: *PhysicsWorld,
        min: Vec3,
        max: Vec3,
        filter: CollisionFilter,
        results: *std.ArrayListUnmanaged(*RigidBody),
    ) !void {
        return physics_queries.queryAABBWithFilter(self, min, max, filter, results);
    }

    pub fn querySphere(
        self: *PhysicsWorld,
        center: Vec3,
        radius: f32,
        results: *std.ArrayListUnmanaged(*RigidBody),
    ) !void {
        return physics_queries.querySphere(self, center, radius, results);
    }

    pub fn querySphereWithFilter(
        self: *PhysicsWorld,
        center: Vec3,
        radius: f32,
        filter: CollisionFilter,
        results: *std.ArrayListUnmanaged(*RigidBody),
    ) !void {
        return physics_queries.querySphereWithFilter(self, center, radius, filter, results);
    }

    pub fn queryPoint(
        self: *PhysicsWorld,
        point: Vec3,
        results: *std.ArrayListUnmanaged(*RigidBody),
    ) !void {
        return physics_queries.queryPoint(self, point, results);
    }

    pub fn queryPointWithFilter(
        self: *PhysicsWorld,
        point: Vec3,
        filter: CollisionFilter,
        results: *std.ArrayListUnmanaged(*RigidBody),
    ) !void {
        return physics_queries.queryPointWithFilter(self, point, filter, results);
    }

    pub fn spherecast(self: *PhysicsWorld, origin: Vec3, radius: f32, translation: Vec3) ?PhysicsRayHit {
        return physics_queries.spherecast(self, origin, radius, translation);
    }

    pub fn spherecastWithFilter(
        self: *PhysicsWorld,
        origin: Vec3,
        radius: f32,
        translation: Vec3,
        filter: CollisionFilter,
    ) ?PhysicsRayHit {
        return physics_queries.spherecastWithFilter(self, origin, radius, translation, filter);
    }

    pub fn findBodyByShape(self: *PhysicsWorld, shape_id: c.b3ShapeId) ?*RigidBody {
        return physics_queries.findBodyByShape(self, shape_id);
    }

    pub fn ownsShape(_: *PhysicsWorld, body: *RigidBody, shape_id: c.b3ShapeId) bool {
        return physics_queries.ownsShape(body, shape_id);
    }

    pub fn clearEvents(self: *PhysicsWorld) void {
        return physics_events.clearEvents(self);
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
            physics_events.drainEvents(self);
            self.acc -= step_h;
        }
        if (n == 4) self.acc = 0.0; // drop backlog instead of spiraling
        if (n == 0) return;

        for (self.bodies.items) |b| {
            self.pullBody(b);
        }
    }

    pub fn syncWorldParams(self: *PhysicsWorld) void {
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
        const is_awake = c.b3Body_IsAwake(b.body_id);
        if (!is_awake and !b.was_awake) {
            return;
        }
        b.was_awake = is_awake;

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
        return physics_joints.createDistanceJoint(self, body_a, body_b, local_anchor_a, local_anchor_b, options);
    }

    pub fn createDistanceJointWorld(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        world_anchor_a: Vec3,
        world_anchor_b: Vec3,
        options: DistanceJointOptions,
    ) !JointId {
        return physics_joints.createDistanceJointWorld(self, body_a, body_b, world_anchor_a, world_anchor_b, options);
    }

    pub fn createSphericalJoint(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        options: SphericalJointOptions,
    ) !JointId {
        return physics_joints.createSphericalJoint(self, body_a, body_b, local_anchor_a, local_anchor_b, options);
    }

    pub fn createSphericalJointWorld(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        world_anchor: Vec3,
        options: SphericalJointOptions,
    ) !JointId {
        return physics_joints.createSphericalJointWorld(self, body_a, body_b, world_anchor, options);
    }

    pub fn createRevoluteJoint(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        options: RevoluteJointOptions,
    ) !JointId {
        return physics_joints.createRevoluteJoint(self, body_a, body_b, local_anchor_a, local_anchor_b, options);
    }

    pub fn createRevoluteJointWorld(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        world_anchor: Vec3,
        options: RevoluteJointOptions,
    ) !JointId {
        return physics_joints.createRevoluteJointWorld(self, body_a, body_b, world_anchor, options);
    }

    pub fn setRevoluteMotor(
        self: *PhysicsWorld,
        joint_id: JointId,
        enabled: bool,
        motor_speed_rad: f32,
        max_motor_torque: f32,
    ) void {
        return physics_joints.setRevoluteMotor(self, joint_id, enabled, motor_speed_rad, max_motor_torque);
    }

    pub fn setRevoluteLimits(
        self: *PhysicsWorld,
        joint_id: JointId,
        lower_angle_rad: f32,
        upper_angle_rad: f32,
    ) void {
        return physics_joints.setRevoluteLimits(self, joint_id, lower_angle_rad, upper_angle_rad);
    }

    pub fn revoluteAngleRad(self: *PhysicsWorld, joint_id: JointId) f32 {
        return physics_joints.revoluteAngleRad(self, joint_id);
    }

    pub fn revoluteAngleDeg(self: *PhysicsWorld, joint_id: JointId) f32 {
        return physics_joints.revoluteAngleDeg(self, joint_id);
    }

    pub fn createWheelJoint(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        options: WheelJointOptions,
    ) !JointId {
        return physics_joints.createWheelJoint(self, body_a, body_b, local_anchor_a, local_anchor_b, options);
    }

    pub fn createWheelJointWorld(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        world_anchor: Vec3,
        options: WheelJointOptions,
    ) !JointId {
        return physics_joints.createWheelJointWorld(self, body_a, body_b, world_anchor, options);
    }

    pub fn setWheelSpin(
        self: *PhysicsWorld,
        joint_id: JointId,
        enabled: bool,
        spin_speed_rad: f32,
        max_spin_torque: f32,
    ) void {
        return physics_joints.setWheelSpin(self, joint_id, enabled, spin_speed_rad, max_spin_torque);
    }

    pub fn setWheelSteering(
        self: *PhysicsWorld,
        joint_id: JointId,
        enabled: bool,
        target_angle_rad: f32,
        max_steering_torque: f32,
    ) void {
        return physics_joints.setWheelSteering(self, joint_id, enabled, target_angle_rad, max_steering_torque);
    }

    pub fn wheelSpinSpeed(self: *PhysicsWorld, joint_id: JointId) f32 {
        return physics_joints.wheelSpinSpeed(self, joint_id);
    }

    pub fn wheelSteeringAngle(self: *PhysicsWorld, joint_id: JointId) f32 {
        return physics_joints.wheelSteeringAngle(self, joint_id);
    }

    pub fn createPrismaticJoint(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        options: PrismaticJointOptions,
    ) !JointId {
        return physics_joints.createPrismaticJoint(self, body_a, body_b, local_anchor_a, local_anchor_b, options);
    }

    pub fn createPrismaticJointWorld(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        world_anchor: Vec3,
        options: PrismaticJointOptions,
    ) !JointId {
        return physics_joints.createPrismaticJointWorld(self, body_a, body_b, world_anchor, options);
    }

    pub fn setPrismaticMotor(
        self: *PhysicsWorld,
        joint_id: JointId,
        enabled: bool,
        motor_speed: f32,
        max_motor_force: f32,
    ) void {
        return physics_joints.setPrismaticMotor(self, joint_id, enabled, motor_speed, max_motor_force);
    }

    pub fn setPrismaticLimits(
        self: *PhysicsWorld,
        joint_id: JointId,
        lower_translation: f32,
        upper_translation: f32,
    ) void {
        return physics_joints.setPrismaticLimits(self, joint_id, lower_translation, upper_translation);
    }

    pub fn prismaticTranslation(self: *PhysicsWorld, joint_id: JointId) f32 {
        return physics_joints.prismaticTranslation(self, joint_id);
    }

    pub fn prismaticSpeed(self: *PhysicsWorld, joint_id: JointId) f32 {
        return physics_joints.prismaticSpeed(self, joint_id);
    }

    pub fn createMotorJoint(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        options: MotorJointOptions,
    ) !JointId {
        return physics_joints.createMotorJoint(self, body_a, body_b, local_anchor_a, local_anchor_b, options);
    }

    pub fn createMotorJointWorld(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        world_anchor: Vec3,
        options: MotorJointOptions,
    ) !JointId {
        return physics_joints.createMotorJointWorld(self, body_a, body_b, world_anchor, options);
    }

    pub fn setMotorLinearVelocity(self: *PhysicsWorld, joint_id: JointId, velocity: Vec3) void {
        return physics_joints.setMotorLinearVelocity(self, joint_id, velocity);
    }

    pub fn setMotorAngularVelocity(self: *PhysicsWorld, joint_id: JointId, velocity_rad: Vec3) void {
        return physics_joints.setMotorAngularVelocity(self, joint_id, velocity_rad);
    }

    pub fn setMotorMaxVelocityForce(self: *PhysicsWorld, joint_id: JointId, max_force: f32) void {
        return physics_joints.setMotorMaxVelocityForce(self, joint_id, max_force);
    }

    pub fn setMotorMaxVelocityTorque(self: *PhysicsWorld, joint_id: JointId, max_torque: f32) void {
        return physics_joints.setMotorMaxVelocityTorque(self, joint_id, max_torque);
    }

    pub fn createWeldJoint(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        options: WeldJointOptions,
    ) !JointId {
        return physics_joints.createWeldJoint(self, body_a, body_b, local_anchor_a, local_anchor_b, options);
    }

    pub fn createWeldJointWorld(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        world_anchor: Vec3,
        options: WeldJointOptions,
    ) !JointId {
        return physics_joints.createWeldJointWorld(self, body_a, body_b, world_anchor, options);
    }

    pub fn jointConstraintForce(self: *PhysicsWorld, joint_id: JointId) Vec3 {
        return physics_joints.jointConstraintForce(self, joint_id);
    }

    pub fn jointConstraintTorque(self: *PhysicsWorld, joint_id: JointId) Vec3 {
        return physics_joints.jointConstraintTorque(self, joint_id);
    }

    pub fn createParallelJoint(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        local_anchor_a: Vec3,
        local_anchor_b: Vec3,
        options: ParallelJointOptions,
    ) !JointId {
        return physics_joints.createParallelJoint(self, body_a, body_b, local_anchor_a, local_anchor_b, options);
    }

    pub fn createParallelJointWorld(
        self: *PhysicsWorld,
        body_a: *RigidBody,
        body_b: *RigidBody,
        world_anchor: Vec3,
        options: ParallelJointOptions,
    ) !JointId {
        return physics_joints.createParallelJointWorld(self, body_a, body_b, world_anchor, options);
    }

    pub fn setParallelSpring(
        self: *PhysicsWorld,
        joint_id: JointId,
        hertz: f32,
        damping_ratio: f32,
        max_torque: f32,
    ) void {
        return physics_joints.setParallelSpring(self, joint_id, hertz, damping_ratio, max_torque);
    }

    pub fn isJointValid(_: *PhysicsWorld, joint_id: JointId) bool {
        return physics_joints.isJointValid(joint_id);
    }

    pub fn destroyJoint(self: *PhysicsWorld, joint_id: JointId) void {
        return physics_joints.destroyJoint(self, joint_id);
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

    pub fn debugCircleSegments(self: *const PhysicsWorld) usize {
        return physics_debug.debugCircleSegments(self);
    }

    pub fn appendDebugLines(self: *PhysicsWorld, allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(DebugLine)) !void {
        return physics_debug.appendDebugLines(self, allocator, out);
    }

    pub fn debugLineCount(self: *PhysicsWorld) usize {
        return physics_debug.debugLineCount(self);
    }
};
