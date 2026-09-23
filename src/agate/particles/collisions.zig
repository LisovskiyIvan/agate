//! Particle-vs-geometry collisions v1 (CPU-only): static sphere colliders +
//! an optional ground plane. Split out of `particles.zig` (facade).
//!
//! Every function takes the system as `anytype` (a `*ParticleSystem` from
//! `system.zig` in practice) so this module never imports `system.zig` or the
//! facade back — same discipline as `profiler/*` and `flow.zig`. `cpu.zig`
//! snapshots `activeCollisionCtx` once per tick (sibling import, no cycle:
//! this module only imports `types.zig`); the `.gpu` / `.compute` leaves call
//! it directly for their explicit-reject gates. Moved tests reach
//! `system.zig` helpers through block-scoped imports that exist only in test
//! builds.
//!
//! SPACE SEMANTICS: collider positions are interpreted in the system's stored
//! simulation coordinates — world units when `local_space == false`,
//! emitter-local units when `local_space == true`. This matches the flow
//! field (`flowUvForPosition`) and the integrator itself, which advances
//! gravity/velocity/position in stored coordinates and applies the emitter
//! world matrix only at instance-fill time. Consequence: in local_space mode
//! the colliders travel with the emitter (they are fixed in the emitter
//! frame); in world-space mode they are fixed in the world.
//!
//! MODES (`CollisionMode`, default `.none` = feature fully off): with
//! `.none`, `activeCollisionCtx` returns null and `updateCpu` executes the
//! legacy instruction stream bit-for-bit (one predictable branch per slot,
//! no extra FP ops, no memory traffic — same discipline as the disarmed flow
//! field). `.kill` marks a contacting particle dead (it compacts away like
//! any age death: it may fire sub-emitters, and its slot is recycled by
//! normal emission — a re-emitted particle is a fresh spawn, so a respawn
//! inside a collider simply dies on contact again). `.bounce` pushes the
//! particle out to the surface and reflects the into-surface velocity about
//! the contact normal with `restitution`, damping the tangential part by
//! `friction`.
//!
//! KNOBS: `restitution` in [0, 1] scales the reflected normal speed (0 =
//! dead stop, 1 = perfectly elastic); `friction` in [0, 1] is the fraction
//! of tangential velocity KEPT (1 = slick, 0 = full tangential stop — the
//! softbody `Cloth.friction` convention). Both clamp at use, so mid-run
//! knob writes can never inject NaN-scale energy.
//!
//! DETERMINISM: the response is a pure function of (position, velocity,
//! collider parameters) — no PRNG draws (unlike emission, which owns the
//! shared PRNG, and sub-emitters, which use the SplitMix-per-event hash
//! stream). Same input state + same dt gives identical results on any worker
//! count: every slot touches only its own index.
//!
//! SIMULATION-MODE GATE: collisions need per-particle live state, so only
//! `.cpu` can express them. Arming colliders (or a non-`.none` mode) on a
//! `.gpu`/`.compute` system is an explicit `error.CollisionNeedsCpu` — both
//! at enable time (the setters below) and at update time (`updateGpu` /
//! `updateCompute` re-check, covering a mode switch after arming) — never a
//! silent downgrade. Disarming (`.none`, `clearColliders`) never fails.
//!
//! v1 NON-GOALS (explicit): continuous collision detection (fast particles
//! may tunnel past thin colliders in one step — discrete resolve only),
//! iterative multi-contact solving (spheres resolve sequentially in index
//! order, then the ground, single pass), mesh collision, dynamic rigid-body
//! coupling, friction-mediated resting contact stabilization (a resting
//! particle micro-bounces; restitution 0 settles it in one step), GPU paths.
//!
//! PERFORMANCE: disabled = one null check per slot (see above). Armed = one
//! sphere loop (cap 8, disabled/degenerate entries skipped) + one ground
//! check per live particle; no allocations per frame (fixed inline storage,
//! view-only snapshot).

const std = @import("std");
const math = @import("math");

const types = @import("types.zig");

const Vec3 = math.Vec3;
const SimulationMode = types.SimulationMode;

/// Collision response of a ParticleSystem. Default `.none` keeps every
/// existing system bit-for-bit identical (see `activeCollisionCtx`).
pub const CollisionMode = enum {
    /// Feature fully off: no code path changes, no errors on any mode.
    none,
    /// A contacting particle dies (compacts away like an age death).
    kill,
    /// Positional push-out + velocity reflection (see module docs).
    bounce,
};

/// Errors surfaced by the collision setters. `CollisionNeedsCpu` is shared
/// by name with `types.UpdateError` (returned by `updateGpu` /
/// `updateCompute` on armed non-CPU systems); Zig unifies same-named errors
/// across sets, so callers can `catch` either spelling.
pub const CollisionError = error{
    /// Colliders / a non-`.none` mode need the CPU integrator; the system
    /// runs `.gpu`/`.compute`. Never a silent downgrade: reconfigure or
    /// disarm first.
    CollisionNeedsCpu,
    /// The fixed sphere array (`max_sphere_colliders`) is full.
    TooManyColliders,
    /// Degenerate parameters (non-positive/non-finite sphere radius,
    /// non-finite ground height).
    InvalidOptions,
};

/// Maximum static sphere colliders per system. Fixed inline storage (same
/// cap as the softbody cloth): attaching them allocates nothing, per-frame
/// or otherwise.
pub const max_sphere_colliders: usize = 8;
/// Maximum static axis-aligned box colliders per system (wave 43, v2). Fixed
/// inline storage: attaching allocates nothing.
pub const max_box_colliders: usize = 8;
/// Maximum static plane colliders per system (wave 43, v2). Fixed inline
/// storage: attaching allocates nothing.
pub const max_plane_colliders: usize = 4;

/// One static sphere collider, interpreted in stored simulation coordinates
/// (world, or emitter-local when `local_space` — see module docs).
pub const ParticleSphereCollider = struct {
    center: Vec3 = Vec3.zero,
    radius: f32 = 0.5,
    /// False skips the entry without removing it (numbering preserved).
    enabled: bool = true,
};

/// One static axis-aligned box collider, interpreted in stored simulation
/// coordinates. `half_extents` are distances from `center` along each axis.
pub const ParticleBoxCollider = struct {
    center: Vec3 = Vec3.zero,
    half_extents: Vec3 = Vec3.new(0.5, 0.5, 0.5),
    enabled: bool = true,
};

/// One static oriented plane collider, interpreted in stored simulation
/// coordinates. Particles penetrating the negative half-space ((pos - point).dot(normal) < 0)
/// bounce or die.
pub const ParticlePlaneCollider = struct {
    point: Vec3 = Vec3.zero,
    normal: Vec3 = Vec3.up,
    enabled: bool = true,
};

/// Read-only per-tick snapshot of the armed collision state, shared by all
/// Phase-A workers (all members are read-only; no locks needed). Null when
/// disarmed: mode `.none`, or mode armed but no geometry (zero spheres, boxes,
/// planes and no ground plane — a mode with nothing to hit costs nothing and
/// errors on no path).
pub const CollisionCtx = struct {
    /// Never `.none` (constructor guarantees).
    mode: CollisionMode,
    /// Live prefix of `collision_spheres` (count clamped to capacity).
    spheres: []const ParticleSphereCollider,
    /// Live prefix of `collision_boxes` (count clamped to capacity).
    boxes: []const ParticleBoxCollider,
    /// Live prefix of `collision_planes` (count clamped to capacity).
    planes: []const ParticlePlaneCollider,
    /// Clamped to [0, 1] at snapshot time.
    restitution: f32,
    /// Clamped to [0, 1] at snapshot time.
    friction: f32,
    /// Null = no ground plane.
    ground: ?f32,
};

/// Snapshots the armed collision state for one updateCpu tick, or null when
/// disarmed (see `CollisionCtx`). Pure and headless-safe; the `.gpu` /
/// `.compute` gates call this directly.
pub fn activeCollisionCtx(self: anytype) ?CollisionCtx {
    if (self.collision_mode == .none) return null;
    const s_count = @min(self.collision_sphere_count, self.collision_spheres.len);
    const b_count = @min(self.collision_box_count, self.collision_boxes.len);
    const p_count = @min(self.collision_plane_count, self.collision_planes.len);
    if (s_count == 0 and b_count == 0 and p_count == 0 and self.collision_ground == null) return null;
    return .{
        .mode = self.collision_mode,
        .spheres = self.collision_spheres[0..s_count],
        .boxes = self.collision_boxes[0..b_count],
        .planes = self.collision_planes[0..p_count],
        .restitution = std.math.clamp(self.collision_restitution, 0.0, 1.0),
        .friction = std.math.clamp(self.collision_friction, 0.0, 1.0),
        .ground = self.collision_ground,
    };
}

/// Adds a static sphere collider. Fixed inline storage (up to
/// `max_sphere_colliders`); never allocates. Validates before mutating: on
/// error the previous set is left untouched.
pub fn addSphereCollider(self: anytype, collider: ParticleSphereCollider) CollisionError!void {
    if (self.simulation_mode != .cpu) return error.CollisionNeedsCpu;
    if (!(collider.radius > 0.0) or !std.math.isFinite(collider.radius)) return error.InvalidOptions;
    if (self.collision_sphere_count >= max_sphere_colliders) return error.TooManyColliders;
    self.collision_spheres[self.collision_sphere_count] = collider;
    self.collision_sphere_count += 1;
}

/// Adds a static axis-aligned box collider. Fixed inline storage (up to
/// `max_box_colliders`); never allocates. Validates positive half-extents.
pub fn addBoxCollider(self: anytype, collider: ParticleBoxCollider) CollisionError!void {
    if (self.simulation_mode != .cpu) return error.CollisionNeedsCpu;
    if (!(collider.half_extents.x > 0.0) or !std.math.isFinite(collider.half_extents.x) or
        !(collider.half_extents.y > 0.0) or !std.math.isFinite(collider.half_extents.y) or
        !(collider.half_extents.z > 0.0) or !std.math.isFinite(collider.half_extents.z) or
        !std.math.isFinite(collider.center.x) or !std.math.isFinite(collider.center.y) or !std.math.isFinite(collider.center.z))
    {
        return error.InvalidOptions;
    }
    if (self.collision_box_count >= max_box_colliders) return error.TooManyColliders;
    self.collision_boxes[self.collision_box_count] = collider;
    self.collision_box_count += 1;
}

/// Adds a static oriented plane collider. Fixed inline storage (up to
/// `max_plane_colliders`); normal is normalized at add time.
pub fn addPlaneCollider(self: anytype, collider: ParticlePlaneCollider) CollisionError!void {
    if (self.simulation_mode != .cpu) return error.CollisionNeedsCpu;
    const n_sq = collider.normal.lengthSq();
    if (n_sq <= 1e-12 or !std.math.isFinite(n_sq) or
        !std.math.isFinite(collider.point.x) or !std.math.isFinite(collider.point.y) or !std.math.isFinite(collider.point.z))
    {
        return error.InvalidOptions;
    }
    if (self.collision_plane_count >= max_plane_colliders) return error.TooManyColliders;
    var norm_plane = collider;
    norm_plane.normal = collider.normal.scale(1.0 / @sqrt(n_sq));
    self.collision_planes[self.collision_plane_count] = norm_plane;
    self.collision_plane_count += 1;
}

/// Helper that extracts a Mesh's world bounding box and adds it as a box collider.
pub fn addMeshAabbCollider(self: anytype, mesh_obj: anytype) CollisionError!void {
    const aabb = mesh_obj.getWorldBoundingBox();
    if (!aabb.isValid()) return error.InvalidOptions;
    return addBoxCollider(self, .{
        .center = aabb.center(),
        .half_extents = aabb.extents(),
    });
}

/// Disarms sphere colliders.
pub fn clearSphereColliders(self: anytype) void {
    self.collision_sphere_count = 0;
}

/// Disarms box colliders.
pub fn clearBoxColliders(self: anytype) void {
    self.collision_box_count = 0;
}

/// Disarms plane colliders.
pub fn clearPlaneColliders(self: anytype) void {
    self.collision_plane_count = 0;
}

/// Disarms all geometry (spheres + boxes + planes + ground plane); response
/// knobs (mode, restitution, friction) are kept, mirroring `clearFlowMap`.
/// Never fails.
pub fn clearColliders(self: anytype) void {
    self.collision_sphere_count = 0;
    self.collision_box_count = 0;
    self.collision_plane_count = 0;
    self.collision_ground = null;
}

/// Selects the collision response. Enabling (anything but `.none`) on a
/// non-CPU system is an explicit error; selecting `.none` (disarm) always
/// succeeds on any mode.
pub fn setCollisionMode(self: anytype, mode: CollisionMode) CollisionError!void {
    if (mode != .none and self.simulation_mode != .cpu) return error.CollisionNeedsCpu;
    self.collision_mode = mode;
}

/// Arms the ground plane (particles collide at y == `height`). Validates
/// before mutating.
pub fn setGroundPlane(self: anytype, height: f32) CollisionError!void {
    if (self.simulation_mode != .cpu) return error.CollisionNeedsCpu;
    if (!std.math.isFinite(height)) return error.InvalidOptions;
    self.collision_ground = height;
}

/// Disarms the ground plane (other colliders kept). Never fails.
pub fn clearGroundPlane(self: anytype) void {
    self.collision_ground = null;
}

/// Outcome of one contact resolve: corrected position/velocity, or death.
pub const ContactResult = struct {
    pos: Vec3,
    vel: Vec3,
    killed: bool,
};

/// Resolves one particle-vs-sphere contact. Pure: no PRNG, no memory access
/// beyond the arguments. Contact is strict penetration (`dist < radius`;
/// exact touch is free). `.kill` dies; otherwise the position pushes out to
/// the surface and only into-surface velocity (`vn < 0`) reflects — a
/// particle pushed out with outward velocity keeps it. A particle exactly at
/// the center pushes out along +Y (the softbody fallback).
pub fn resolveSphereContact(
    pos: Vec3,
    vel: Vec3,
    center: Vec3,
    radius: f32,
    mode: CollisionMode,
    restitution: f32,
    friction: f32,
) ContactResult {
    const delta = pos.sub(center);
    const dist = delta.length();
    if (dist >= radius) return .{ .pos = pos, .vel = vel, .killed = false };
    if (mode == .kill) return .{ .pos = pos, .vel = vel, .killed = true };
    const n = if (dist > 1e-9) delta.scale(1.0 / dist) else Vec3.new(0.0, 1.0, 0.0);
    const pushed = center.add(n.scale(radius));
    const vn = vel.dot(n);
    if (vn >= 0.0) return .{ .pos = pushed, .vel = vel, .killed = false };
    const rest = std.math.clamp(restitution, 0.0, 1.0);
    const fric = std.math.clamp(friction, 0.0, 1.0);
    const tangential = vel.sub(n.scale(vn));
    return .{
        .pos = pushed,
        .vel = n.scale(-vn * rest).add(tangential.scale(fric)),
        .killed = false,
    };
}

/// Resolves one particle-vs-box contact. Pure: no PRNG, no allocations.
/// Contact is strict penetration (particle inside all three half-extents).
/// `.kill` dies; otherwise pushes out along the closest face normal and
/// reflects only inward velocity (with restitution and tangential friction).
pub fn resolveBoxContact(
    pos: Vec3,
    vel: Vec3,
    center: Vec3,
    half_extents: Vec3,
    mode: CollisionMode,
    restitution: f32,
    friction: f32,
) ContactResult {
    if (!(half_extents.x > 0.0 and half_extents.y > 0.0 and half_extents.z > 0.0)) {
        return .{ .pos = pos, .vel = vel, .killed = false };
    }
    const d = pos.sub(center);
    const dx = @abs(d.x) - half_extents.x;
    const dy = @abs(d.y) - half_extents.y;
    const dz = @abs(d.z) - half_extents.z;

    // Must be strictly inside the box on all 3 axes.
    if (dx >= 0.0 or dy >= 0.0 or dz >= 0.0) {
        return .{ .pos = pos, .vel = vel, .killed = false };
    }
    if (mode == .kill) return .{ .pos = pos, .vel = vel, .killed = true };

    // Closest face is the one with least penetration (max of negative dx, dy, dz).
    var n = Vec3.zero;
    var pushed = pos;
    if (dx >= dy and dx >= dz) {
        const sign: f32 = if (d.x >= 0.0) 1.0 else -1.0;
        n = Vec3.new(sign, 0.0, 0.0);
        pushed.x = center.x + sign * half_extents.x;
    } else if (dy >= dx and dy >= dz) {
        const sign: f32 = if (d.y >= 0.0) 1.0 else -1.0;
        n = Vec3.new(0.0, sign, 0.0);
        pushed.y = center.y + sign * half_extents.y;
    } else {
        const sign: f32 = if (d.z >= 0.0) 1.0 else -1.0;
        n = Vec3.new(0.0, 0.0, sign);
        pushed.z = center.z + sign * half_extents.z;
    }

    const vn = vel.dot(n);
    if (vn >= 0.0) return .{ .pos = pushed, .vel = vel, .killed = false };
    const rest = std.math.clamp(restitution, 0.0, 1.0);
    const fric = std.math.clamp(friction, 0.0, 1.0);
    const tangential = vel.sub(n.scale(vn));
    return .{
        .pos = pushed,
        .vel = n.scale(-vn * rest).add(tangential.scale(fric)),
        .killed = false,
    };
}

/// Resolves one particle-vs-plane contact. Pure: no PRNG, no allocations.
/// Plane defined by `point` and unit `normal`. Particle penetrates if
/// (pos - point).dot(normal) < 0.
pub fn resolvePlaneContact(
    pos: Vec3,
    vel: Vec3,
    point: Vec3,
    normal: Vec3,
    mode: CollisionMode,
    restitution: f32,
    friction: f32,
) ContactResult {
    const n_sq = normal.lengthSq();
    if (n_sq <= 1e-12 or !std.math.isFinite(n_sq)) return .{ .pos = pos, .vel = vel, .killed = false };
    const n = normal.scale(1.0 / @sqrt(n_sq));
    const dist = pos.sub(point).dot(n);
    if (dist >= 0.0) return .{ .pos = pos, .vel = vel, .killed = false };
    if (mode == .kill) return .{ .pos = pos, .vel = vel, .killed = true };
    const pushed = pos.sub(n.scale(dist));
    const vn = vel.dot(n);
    if (vn >= 0.0) return .{ .pos = pushed, .vel = vel, .killed = false };
    const rest = std.math.clamp(restitution, 0.0, 1.0);
    const fric = std.math.clamp(friction, 0.0, 1.0);
    const tangential = vel.sub(n.scale(vn));
    return .{
        .pos = pushed,
        .vel = n.scale(-vn * rest).add(tangential.scale(fric)),
        .killed = false,
    };
}

/// Resolves one particle-vs-ground-plane contact (plane normal +Y). Pure;
/// contact is strict (`pos.y < height`). `.kill` dies; otherwise y clamps
/// and only downward velocity reflects (`vy` scaled by restitution, `vx/vz`
/// damped by friction).
pub fn resolveGroundContact(
    pos: Vec3,
    vel: Vec3,
    height: f32,
    mode: CollisionMode,
    restitution: f32,
    friction: f32,
) ContactResult {
    if (pos.y >= height) return .{ .pos = pos, .vel = vel, .killed = false };
    if (mode == .kill) return .{ .pos = pos, .vel = vel, .killed = true };
    const rest = std.math.clamp(restitution, 0.0, 1.0);
    const fric = std.math.clamp(friction, 0.0, 1.0);
    var v = vel;
    if (v.y < 0.0) v = Vec3.new(v.x * fric, -v.y * rest, v.z * fric);
    return .{ .pos = Vec3.new(pos.x, height, pos.z), .vel = v, .killed = false };
}

/// Resolves a particle against the whole snapshot: spheres in index order,
/// then boxes in index order, then planes in index order, then the ground,
/// single pass (v2: no iteration). Pure per slot — the CPU integrator calls
/// this once per live particle per tick. Degenerate collider entries are skipped.
pub fn collideParticle(pos: Vec3, vel: Vec3, ctx: CollisionCtx) ContactResult {
    var p = pos;
    var v = vel;
    for (ctx.spheres) |s| {
        if (!s.enabled) continue;
        if (!(s.radius > 0.0) or !std.math.isFinite(s.radius)) continue;
        const r = resolveSphereContact(p, v, s.center, s.radius, ctx.mode, ctx.restitution, ctx.friction);
        if (r.killed) return .{ .pos = p, .vel = v, .killed = true };
        p = r.pos;
        v = r.vel;
    }
    for (ctx.boxes) |b| {
        if (!b.enabled) continue;
        if (!(b.half_extents.x > 0.0 and b.half_extents.y > 0.0 and b.half_extents.z > 0.0)) continue;
        const r = resolveBoxContact(p, v, b.center, b.half_extents, ctx.mode, ctx.restitution, ctx.friction);
        if (r.killed) return .{ .pos = p, .vel = v, .killed = true };
        p = r.pos;
        v = r.vel;
    }
    for (ctx.planes) |pl| {
        if (!pl.enabled) continue;
        const r = resolvePlaneContact(p, v, pl.point, pl.normal, ctx.mode, ctx.restitution, ctx.friction);
        if (r.killed) return .{ .pos = p, .vel = v, .killed = true };
        p = r.pos;
        v = r.vel;
    }
    if (ctx.ground) |h| {
        const r = resolveGroundContact(p, v, h, ctx.mode, ctx.restitution, ctx.friction);
        if (r.killed) return .{ .pos = p, .vel = v, .killed = true };
        p = r.pos;
        v = r.vel;
    }
    return .{ .pos = p, .vel = v, .killed = false };
}

// --- Collision tests (CPU-only; headless, deterministic) ---

test "collision defaults off and .none keeps the legacy path bit-identical" {
    const sys = @import("system.zig");
    const ParticleSystem = sys.ParticleSystem;
    const a = std.testing.allocator;
    var baseline = try sys.makeTestSystem(a, 64);
    defer sys.freeTestSystem(&baseline);
    var knobs = try sys.makeTestSystem(a, 64);
    defer sys.freeTestSystem(&knobs);
    for ([2]*ParticleSystem{ &baseline, &knobs }) |ps| {
        ps.gravity = Vec3.new(0.0, -3.0, 0.0);
        ps.emit_rate = 120.0;
        ps.is_emitting = true;
        ps.lifetime_min = 0.5;
        ps.lifetime_max = 1.0;
    }
    // Every collision knob touched, but mode stays .none: armed geometry
    // with the feature off must not perturb the stream, the integration, or
    // the instance fill (mirrors the flow "map unset" parity test).
    knobs.collision_restitution = 0.0;
    knobs.collision_friction = 0.0;
    try knobs.addSphereCollider(.{ .center = Vec3.new(0.0, 1.0, 0.0), .radius = 0.5 });
    try knobs.addSphereCollider(.{ .center = Vec3.zero, .radius = 2.0, .enabled = false });
    try knobs.setGroundPlane(5.0);
    try std.testing.expectEqual(CollisionMode.none, knobs.collision_mode);
    try std.testing.expect(activeCollisionCtx(&knobs) == null);

    var frame: usize = 0;
    while (frame < 60) : (frame += 1) {
        baseline.updateCpu(1.0 / 60.0);
        knobs.updateCpu(1.0 / 60.0);
        try std.testing.expectEqual(baseline.active_count, knobs.active_count);
        const live = baseline.active_count;
        try std.testing.expect(live > 0);
        try std.testing.expect(std.mem.eql(
            u8,
            std.mem.sliceAsBytes(baseline.particles[0..live]),
            std.mem.sliceAsBytes(knobs.particles[0..live]),
        ));
        try std.testing.expect(std.mem.eql(
            u8,
            std.mem.sliceAsBytes(baseline.instances[0..live]),
            std.mem.sliceAsBytes(knobs.instances[0..live]),
        ));
    }
}

test "sphere bounce reflects velocity about the contact normal" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.bounce);
    ps.collision_restitution = 1.0;
    ps.collision_friction = 1.0;
    try ps.addSphereCollider(.{ .center = Vec3.zero, .radius = 1.0 });
    // Head-on along -Z: (0,0,1.5) + (0,0,-1)*1 = (0,0,0.5), inside.
    sys.placeTestParticle(&ps, Vec3.new(0.0, 0.0, 1.5), Vec3.new(0.0, 0.0, -1.0));

    ps.updateCpu(1.0);
    try std.testing.expectEqual(@as(usize, 1), ps.active_count);
    // Pushed out to the surface, velocity mirrored exactly.
    try std.testing.expectEqual(Vec3.new(0.0, 0.0, 1.0), ps.particles[0].position);
    try std.testing.expectEqual(Vec3.new(0.0, 0.0, 1.0), ps.particles[0].velocity);
}

test "restitution 1 mirrors, restitution 0 stops the normal" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.bounce);
    ps.collision_friction = 1.0;
    try ps.addSphereCollider(.{ .center = Vec3.zero, .radius = 1.0 });

    ps.collision_restitution = 1.0;
    sys.placeTestParticle(&ps, Vec3.new(0.0, 0.0, 1.5), Vec3.new(0.0, 0.0, -1.0));
    ps.updateCpu(1.0);
    try std.testing.expectEqual(Vec3.new(0.0, 0.0, 1.0), ps.particles[0].velocity);

    // Same contact, dead-stop normal: the particle stays on the surface
    // with zero velocity (one-step settle).
    ps.collision_restitution = 0.0;
    sys.placeTestParticle(&ps, Vec3.new(0.0, 0.0, 1.5), Vec3.new(0.0, 0.0, -1.0));
    ps.updateCpu(1.0);
    try std.testing.expectEqual(Vec3.new(0.0, 0.0, 1.0), ps.particles[0].position);
    try std.testing.expectEqual(Vec3.zero, ps.particles[0].velocity);
}

test "friction damps only the tangential component" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.bounce);
    ps.collision_restitution = 1.0;
    try ps.addSphereCollider(.{ .center = Vec3.zero, .radius = 1.0 });
    // Oblique: (2,1.2,0) + (-2,-1,0)*1 = (0,0.2,0), normal +Y, vn = -1.
    // Reflected normal (+1) is friction-independent; tangential (-2,0,0)
    // scales by friction exactly.
    ps.collision_friction = 0.5;
    sys.placeTestParticle(&ps, Vec3.new(2.0, 1.2, 0.0), Vec3.new(-2.0, -1.0, 0.0));
    ps.updateCpu(1.0);
    try std.testing.expectEqual(Vec3.new(0.0, 1.0, 0.0), ps.particles[0].position);
    try std.testing.expectEqual(Vec3.new(-1.0, 1.0, 0.0), ps.particles[0].velocity);

    ps.collision_friction = 1.0;
    sys.placeTestParticle(&ps, Vec3.new(2.0, 1.2, 0.0), Vec3.new(-2.0, -1.0, 0.0));
    ps.updateCpu(1.0);
    try std.testing.expectEqual(Vec3.new(-2.0, 1.0, 0.0), ps.particles[0].velocity);

    // Full tangential stop: only the reflected normal survives.
    ps.collision_friction = 0.0;
    sys.placeTestParticle(&ps, Vec3.new(2.0, 1.2, 0.0), Vec3.new(-2.0, -1.0, 0.0));
    ps.updateCpu(1.0);
    try std.testing.expectEqual(Vec3.new(0.0, 1.0, 0.0), ps.particles[0].velocity);
}

test "kill mode destroys on sphere and ground contact, respawn is a fresh spawn" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.kill);
    try ps.addSphereCollider(.{ .center = Vec3.zero, .radius = 1.0 });

    // Placed particle (lifetime 10, so only the collision can kill it).
    sys.placeTestParticle(&ps, Vec3.zero, Vec3.zero);
    ps.updateCpu(1.0);
    try std.testing.expectEqual(@as(usize, 0), ps.active_count);

    // Ground kill: same proof, falling through y = 0.
    var ground = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ground);
    try ground.setCollisionMode(.kill);
    try ground.setGroundPlane(0.0);
    sys.placeTestParticle(&ground, Vec3.new(0.0, 0.5, 0.0), Vec3.new(0.0, -2.0, 0.0));
    ground.updateCpu(1.0);
    try std.testing.expectEqual(@as(usize, 0), ground.active_count);

    // Respawn is a normal fresh emission (age 0 at the emitter); the slot
    // is recycled like after any death.
    ps.burst(1);
    try std.testing.expectEqual(@as(usize, 1), ps.active_count);
    try std.testing.expectEqual(@as(f32, 0.0), ps.particles[0].age);
    ps.updateCpu(1.0);
    try std.testing.expectEqual(@as(usize, 0), ps.active_count);
}

test "ground bounce clamps above the plane and reflects" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.bounce);
    ps.collision_restitution = 0.5;
    ps.collision_friction = 0.5;
    try ps.setGroundPlane(0.0);
    // (0,2,0) + (2,-4,0)*1 = (2,-2,0) -> clamp y = 0, vy = +2, vx = 1.
    sys.placeTestParticle(&ps, Vec3.new(0.0, 2.0, 0.0), Vec3.new(2.0, -4.0, 0.0));

    ps.updateCpu(1.0);
    try std.testing.expectEqual(Vec3.new(2.0, 0.0, 0.0), ps.particles[0].position);
    try std.testing.expectEqual(Vec3.new(1.0, 2.0, 0.0), ps.particles[0].velocity);

    // Rising particles (vy > 0) pass clamped positions through untouched:
    // push-out only, no velocity change.
    sys.placeTestParticle(&ps, Vec3.new(0.0, -0.5, 0.0), Vec3.new(0.0, 3.0, 0.0));
    ps.updateCpu(0.0);
    try std.testing.expectEqual(Vec3.new(0.0, 0.0, 0.0), ps.particles[0].position);
    try std.testing.expectEqual(Vec3.new(0.0, 3.0, 0.0), ps.particles[0].velocity);
}

test "disabled collider is ignored" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.bounce);
    ps.collision_restitution = 1.0;
    try ps.addSphereCollider(.{ .center = Vec3.zero, .radius = 1.0, .enabled = false });
    try std.testing.expect(activeCollisionCtx(&ps) != null); // armed, but skipped
    sys.placeTestParticle(&ps, Vec3.new(0.0, 0.0, 1.5), Vec3.new(0.0, 0.0, -1.0));

    ps.updateCpu(1.0);
    // Straight through: no push-out, no reflection.
    try std.testing.expectEqual(Vec3.new(0.0, 0.0, 0.5), ps.particles[0].position);
    try std.testing.expectEqual(Vec3.new(0.0, 0.0, -1.0), ps.particles[0].velocity);
}

test "collider capacity and parameter validation" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    for (0..max_sphere_colliders) |i| {
        try ps.addSphereCollider(.{ .center = Vec3.new(@floatFromInt(i), 0.0, 0.0), .radius = 0.5 });
    }
    try std.testing.expectEqual(max_sphere_colliders, ps.collision_sphere_count);
    try std.testing.expectError(error.TooManyColliders, ps.addSphereCollider(.{}));
    try std.testing.expectEqual(max_sphere_colliders, ps.collision_sphere_count);

    var fresh = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&fresh);
    try std.testing.expectError(error.InvalidOptions, fresh.addSphereCollider(.{ .radius = 0.0 }));
    try std.testing.expectError(error.InvalidOptions, fresh.addSphereCollider(.{ .radius = -1.0 }));
    try std.testing.expectError(error.InvalidOptions, fresh.addSphereCollider(.{ .radius = std.math.inf(f32) }));
    try std.testing.expectError(error.InvalidOptions, fresh.addSphereCollider(.{ .radius = std.math.nan(f32) }));
    try std.testing.expectEqual(@as(usize, 0), fresh.collision_sphere_count);
    try std.testing.expectError(error.InvalidOptions, fresh.setGroundPlane(std.math.inf(f32)));
    try std.testing.expectEqual(@as(?f32, null), fresh.collision_ground);

    // Clearing disarms geometry but keeps the response knobs.
    try fresh.setCollisionMode(.bounce);
    try fresh.setGroundPlane(1.0);
    fresh.clearColliders();
    try std.testing.expectEqual(@as(usize, 0), fresh.collision_sphere_count);
    try std.testing.expectEqual(@as(?f32, null), fresh.collision_ground);
    try std.testing.expectEqual(CollisionMode.bounce, fresh.collision_mode);
    try std.testing.expect(activeCollisionCtx(&fresh) == null);
}

test "gpu and compute reject armed collisions, never downgrade" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.bounce);
    try ps.addSphereCollider(.{ .center = Vec3.zero, .radius = 1.0 });

    // .gpu: both update entry points error; the mode is never mutated.
    ps.simulation_mode = .gpu;
    try std.testing.expectError(error.CollisionNeedsCpu, ps.updateGpu(0.016));
    try std.testing.expectError(error.CollisionNeedsCpu, ps.update(0.016));
    try std.testing.expectEqual(SimulationMode.gpu, ps.simulation_mode);
    try std.testing.expectEqual(@as(usize, 0), ps.gpu_high_water);
    // Disarming restores the GPU path (disarm never fails, any mode).
    try ps.setCollisionMode(.none);
    try ps.update(0.016);

    // .compute: same explicit rejection (re-arm on CPU first: enabling
    // while .gpu is correctly rejected above).
    ps.simulation_mode = .cpu;
    try ps.setCollisionMode(.bounce);
    ps.simulation_mode = .compute;
    try std.testing.expectError(error.CollisionNeedsCpu, ps.updateCompute(0.016));
    try std.testing.expectError(error.CollisionNeedsCpu, ps.update(0.016));
    try std.testing.expectEqual(SimulationMode.compute, ps.simulation_mode);
    try std.testing.expectEqual(@as(usize, 0), ps.compute_high_water);
    try ps.setCollisionMode(.none);
    try ps.update(0.016);

    // Enable-time rejection: arming a non-CPU system fails immediately.
    var gpu = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&gpu);
    gpu.simulation_mode = .gpu;
    try std.testing.expectError(error.CollisionNeedsCpu, gpu.addSphereCollider(.{}));
    try std.testing.expectError(error.CollisionNeedsCpu, gpu.setCollisionMode(.bounce));
    try std.testing.expectError(error.CollisionNeedsCpu, gpu.setGroundPlane(0.0));
    try std.testing.expectEqual(@as(usize, 0), gpu.collision_sphere_count);
    try std.testing.expectEqual(CollisionMode.none, gpu.collision_mode);
    try std.testing.expectEqual(@as(?f32, null), gpu.collision_ground);
    // Disarm paths always succeed, even off-CPU.
    try gpu.setCollisionMode(.none);
    gpu.clearColliders();
    gpu.clearGroundPlane();
}

test "same state plus same dt gives identical results" {
    const sys = @import("system.zig");
    const ParticleSystem = sys.ParticleSystem;
    const a = std.testing.allocator;
    var run_a = try sys.makeTestSystem(a, 128);
    defer sys.freeTestSystem(&run_a);
    var run_b = try sys.makeTestSystem(a, 128);
    defer sys.freeTestSystem(&run_b);
    for ([2]*ParticleSystem{ &run_a, &run_b }) |ps| {
        ps.gravity = Vec3.new(0.0, -3.0, 0.0);
        ps.emit_rate = 120.0;
        ps.is_emitting = true;
        ps.lifetime_min = 0.5;
        ps.lifetime_max = 1.0;
        try ps.setCollisionMode(.bounce);
        ps.collision_restitution = 0.5;
        ps.collision_friction = 0.8;
        try ps.addSphereCollider(.{ .center = Vec3.new(0.0, 1.0, 0.0), .radius = 0.75 });
        try ps.setGroundPlane(-0.5);
    }

    var frame: usize = 0;
    while (frame < 60) : (frame += 1) {
        run_a.updateCpu(1.0 / 60.0);
        run_b.updateCpu(1.0 / 60.0);
        try std.testing.expectEqual(run_a.active_count, run_b.active_count);
        const live = run_a.active_count;
        try std.testing.expect(live > 0);
        try std.testing.expect(std.mem.eql(
            u8,
            std.mem.sliceAsBytes(run_a.particles[0..live]),
            std.mem.sliceAsBytes(run_b.particles[0..live]),
        ));
    }
    const hash_a = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(run_a.particles[0..run_a.active_count]));
    const hash_b = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(run_b.particles[0..run_b.active_count]));
    try std.testing.expectEqual(hash_a, hash_b);
}

test "kill absorption agrees across dyadic frame rates" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    const hashLive = struct {
        fn hashLive(ps: *sys.ParticleSystem) u64 {
            return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(ps.particles[0..ps.active_count]));
        }
    }.hashLive;
    // Exact-fp setup: unit velocity, dyadic dts, contact strictly inside.
    var run_a = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&run_a);
    var run_b = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&run_b);
    for ([2]*sys.ParticleSystem{ &run_a, &run_b }) |ps| {
        try ps.setCollisionMode(.kill);
        try ps.addSphereCollider(.{ .center = Vec3.new(1.0, 0.0, 0.0), .radius = 0.5 });
        sys.placeTestParticle(ps, Vec3.zero, Vec3.new(1.0, 0.0, 0.0));
    }
    // Same simulated instant (t = 0.5) via different stepping: exact in
    // binary fp (0.5, 0.25 + 0.25), so the live states are bit-identical.
    // x = 0.5 is exactly at touch distance (dist == radius): not contact.
    run_a.updateCpu(0.5);
    run_b.updateCpu(0.25);
    run_b.updateCpu(0.25);
    try std.testing.expectEqual(@as(usize, 1), run_a.active_count);
    try std.testing.expectEqual(@as(usize, 1), run_b.active_count);
    try std.testing.expectEqual(hashLive(&run_a), hashLive(&run_b));

    // Both schedules absorb the particle (discrete contact timing is
    // step-size dependent by nature — the kill event itself is what agrees).
    run_a.updateCpu(0.5);
    run_b.updateCpu(0.25);
    try std.testing.expectEqual(@as(usize, 0), run_a.active_count);
    try std.testing.expectEqual(@as(usize, 0), run_b.active_count);
    run_b.updateCpu(0.25); // empty tick is a no-op
    try std.testing.expectEqual(hashLive(&run_a), hashLive(&run_b));
}

test "collisions are worker-count invariant" {
    const sys = @import("system.zig");
    const ParticleSystem = sys.ParticleSystem;
    const jobs = @import("../jobs.zig");
    const a = std.testing.allocator;
    const pool = try jobs.Pool.init(a, 2);
    defer pool.deinit();

    var serial = try sys.makeTestSystem(a, 16_384);
    defer sys.freeTestSystem(&serial);
    var parallel = try sys.makeTestSystem(a, 16_384);
    defer sys.freeTestSystem(&parallel);
    parallel.thread_pool = pool;
    for ([2]*ParticleSystem{ &serial, &parallel }) |ps| {
        ps.gravity = Vec3.new(0.0, -3.0, 0.0);
        ps.emit_rate = 8000.0;
        ps.is_emitting = true;
        try ps.setCollisionMode(.bounce);
        ps.collision_restitution = 0.5;
        ps.collision_friction = 0.9;
        try ps.addSphereCollider(.{ .center = Vec3.new(0.0, 1.0, 0.0), .radius = 1.5 });
        try ps.setGroundPlane(-1.0);
    }

    var frame: usize = 0;
    while (frame < 40) : (frame += 1) {
        serial.updateCpu(1.0 / 60.0);
        parallel.updateCpu(1.0 / 60.0);
        try std.testing.expectEqual(serial.active_count, parallel.active_count);
        const live = serial.active_count;
        try std.testing.expect(std.mem.eql(
            u8,
            std.mem.sliceAsBytes(serial.particles[0..live]),
            std.mem.sliceAsBytes(parallel.particles[0..live]),
        ));
        try std.testing.expect(std.mem.eql(
            u8,
            std.mem.sliceAsBytes(serial.instances[0..live]),
            std.mem.sliceAsBytes(parallel.instances[0..live]),
        ));
    }
    // Past the pool threshold with real contacts: proves the armed path
    // actually forked (not just the inline fallback).
    try std.testing.expect(serial.active_count > jobs.Pool.min_len_for_workers);
}

test "armed collision scenario pins a golden hash" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeTestSystem(a, 32);
    defer sys.freeTestSystem(&ps);
    ps.gravity = Vec3.new(0.0, -2.0, 0.0);
    ps.emit_rate = 30.0;
    ps.is_emitting = true;
    try ps.setCollisionMode(.bounce);
    ps.collision_restitution = 0.5;
    ps.collision_friction = 0.5;
    try ps.addSphereCollider(.{ .center = Vec3.new(0.0, 1.0, 0.0), .radius = 0.5 });
    try ps.setGroundPlane(-0.5);

    var frame: usize = 0;
    while (frame < 30) : (frame += 1) ps.updateCpu(1.0 / 60.0);
    // Pinned regression values (seed 42 emission + collisions; regenerate
    // deliberately if the integrator changes — see the .none parity test,
    // which proves the default path is untouched by this feature).
    try std.testing.expectEqual(@as(u64, 15457552461678759088), std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(ps.particles[0..ps.active_count])));
    // 30 frames x 0.5 emissions/frame, lifetimes >= 1 s: nothing dies.
    try std.testing.expectEqual(@as(usize, 15), ps.active_count);
}

test "box bounce reflects velocity about the closest face normal" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.bounce);
    ps.collision_restitution = 1.0;
    ps.collision_friction = 1.0;
    try ps.addBoxCollider(.{
        .center = Vec3.zero,
        .half_extents = Vec3.new(1.0, 1.0, 1.0),
    });

    // Head-on along -X towards +X face: (1.5, 0, 0) + (-1, 0, 0)*1 = (0.5, 0, 0), inside.
    sys.placeTestParticle(&ps, Vec3.new(1.5, 0.0, 0.0), Vec3.new(-1.0, 0.0, 0.0));
    ps.updateCpu(1.0);
    try std.testing.expectEqual(@as(usize, 1), ps.active_count);
    try std.testing.expectEqual(Vec3.new(1.0, 0.0, 0.0), ps.particles[0].position);
    try std.testing.expectEqual(Vec3.new(1.0, 0.0, 0.0), ps.particles[0].velocity);
}

test "box bounce with restitution and friction" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.bounce);
    ps.collision_restitution = 0.5;
    ps.collision_friction = 0.25;
    try ps.addBoxCollider(.{
        .center = Vec3.zero,
        .half_extents = Vec3.new(2.0, 0.5, 2.0),
    });

    // Particle lands on top face (+Y): (0, 1.0, 0) + (4.0, -0.8, 0.0)*1 = (4.0, 0.2, 0.0).
    // dx = 4 - 2 = 2 (outside along X if x=4, so let's keep x inside box: x=0.5).
    // Start at (0.5, 1.0, 0.0), vel = (4.0, -0.8, 0.0).
    // After 1s: pos = (4.5, 0.2, 0) -> outside on X! So let's use small horizontal vel: (0.2, -0.8, 0.0).
    // After 1s: pos = (0.7, 0.2, 0.0).
    // d = (0.7, 0.2, 0.0).
    // dx = 0.7 - 2.0 = -1.3.
    // dy = 0.2 - 0.5 = -0.3.
    // dz = 0.0 - 2.0 = -2.0.
    // Closest face is +Y (dy = -0.3 is maximum negative value).
    sys.placeTestParticle(&ps, Vec3.new(0.5, 1.0, 0.0), Vec3.new(0.2, -0.8, 0.0));
    ps.updateCpu(1.0);
    try std.testing.expectEqual(@as(usize, 1), ps.active_count);
    // Pushed out to y = 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), ps.particles[0].position.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), ps.particles[0].position.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ps.particles[0].position.z, 1e-6);
    // Normal vel -0.8 reflected with rest 0.5 -> +0.4.
    // Tangential vel 0.2 scaled by fric 0.25 -> 0.05.
    try std.testing.expectApproxEqAbs(@as(f32, 0.05), ps.particles[0].velocity.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), ps.particles[0].velocity.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ps.particles[0].velocity.z, 1e-6);
}

test "plane bounce reflects velocity about plane normal" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.bounce);
    ps.collision_restitution = 1.0;
    ps.collision_friction = 1.0;
    // Plane passing through (0, 0, 0) with normal along +Z
    try ps.addPlaneCollider(.{
        .point = Vec3.zero,
        .normal = Vec3.new(0.0, 0.0, 1.0),
    });

    // Particle moves from +Z towards -Z: (0, 0, 0.5) + (0, 0, -1.0)*1 = (0, 0, -0.5), behind plane.
    sys.placeTestParticle(&ps, Vec3.new(0.0, 0.0, 0.5), Vec3.new(0.0, 0.0, -1.0));
    ps.updateCpu(1.0);
    try std.testing.expectEqual(@as(usize, 1), ps.active_count);
    // Pushed out to z = 0.0, velocity mirrored along +Z
    try std.testing.expectEqual(Vec3.zero, ps.particles[0].position);
    try std.testing.expectEqual(Vec3.new(0.0, 0.0, 1.0), ps.particles[0].velocity);
}

test "box and plane kill mode eliminates contacting particle" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.kill);
    try ps.addBoxCollider(.{
        .center = Vec3.new(2.0, 0.0, 0.0),
        .half_extents = Vec3.new(0.5, 0.5, 0.5),
    });
    // Particle moves into box
    sys.placeTestParticle(&ps, Vec3.new(1.0, 0.0, 0.0), Vec3.new(1.0, 0.0, 0.0));
    ps.updateCpu(0.8); // reaches 1.8, inside box
    try std.testing.expectEqual(@as(usize, 0), ps.active_count);
}

