//! Scene simulation/content-builder attach points: decals, particles,
//! trails, CSG/greased-line/simplify builders, nav, animations,
//! physics. Split out of `scene.zig` (facade).
//!
/// Anti-cycle rule (same as `audio/*`, `profiler/*`): every function takes
/// the scene as `anytype` (a `*Scene` from `core.zig` in practice) and this
/// module never imports `core.zig` or the `scene.zig` facade back.
/// Cross-leaf helpers consumed here are `pub` in their home module but are
/// deliberately NOT re-exported by the facade.
const math = @import("math");
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const particles = @import("../particles.zig");
const ParticleSystem = particles.ParticleSystem;
const UpdateError = particles.UpdateError;
const trail_mod = @import("../mesh/trail.zig");
const TrailMesh = trail_mod.TrailMesh;
const TrailOptions = trail_mod.TrailOptions;
const csg_mod = @import("../mesh/csg.zig");
const greased_mod = @import("../mesh/greased_line.zig");
const GreasedLineOptions = greased_mod.GreasedLineOptions;
const GreasedLineMesh = greased_mod.GreasedLineMesh;
const mesh_mesh = @import("../mesh/mesh.zig");
const mesh_mod = @import("../mesh.zig");
const Mesh = mesh_mod.Mesh;
const simplify_mod = @import("../mesh/simplify.zig");
const SimplifyOptions = simplify_mod.SimplifyOptions;
const LODLevelSpec = simplify_mod.LODLevelSpec;
const ai_mod = @import("../ai.zig");
const physics = @import("../physics.zig");
const PhysicsWorld = physics.PhysicsWorld;
const RigidBody = physics.RigidBody;
const ColliderType = physics.ColliderType;
const scene_animation = @import("animation_runtime.zig");
const DecalManager = @import("../mesh/decal.zig").DecalManager;

// ---- Decals / particles / trails / CSG / nav. ----

pub fn getOrCreateDecalManager(self: anytype, max_decals: usize) *DecalManager {
    return self.decals.getOrCreate(self, max_decals);
}

pub fn updateDecals(self: anytype, dt: f32) void {
    self.decals.update(dt);
}

pub fn createParticleSystem(self: anytype, name: []const u8, capacity: usize) !*ParticleSystem {
    return self.particles.create(self.allocator, name, capacity);
}

/// Explicit particle stepping: GPU simulation modes either run or return
/// an error (particles.UpdateError) — never a silent CPU downgrade.
pub fn updateParticles(self: anytype, dt: f32) particles.UpdateError!void {
    try self.particles.update(dt);
}

pub fn createTrailMesh(self: anytype, name: []const u8, options: TrailOptions) !*TrailMesh {
    return self.trails.create(self, self.allocator, name, options);
}

pub fn updateTrails(self: anytype, dt: f32) void {
    const cam_pos = if (self.active_camera) |cam| cam.getPosition() else Vec3.zero;
    self.trails.update(dt, cam_pos);
}

pub fn createCSGMesh(self: anytype, name: []const u8, csg_solid: *const csg_mod.CSG) !*Mesh {
    return csg_solid.toMesh(self, name);
}

pub fn createGreasedLine(self: anytype, name: []const u8, options: GreasedLineOptions) !*Mesh {
    var data = try greased_mod.buildGreasedLineData(self.allocator, options);
    defer data.deinit(self.allocator);
    return mesh_mesh.uploadGeometry(self, name, data);
}

pub fn createGreasedLineMesh(self: anytype, name: []const u8, options: GreasedLineOptions) !*GreasedLineMesh {
    const gl = try GreasedLineMesh.init(self, name, options);
    try self.greased_lines.append(self.allocator, gl);
    return gl;
}

pub fn simplifyMesh(self: anytype, name: []const u8, source_mesh: *Mesh, options: SimplifyOptions) !*Mesh {
    return simplify_mod.simplifyMesh(self.allocator, self, name, source_mesh, options);
}

pub fn generateLODLevels(self: anytype, source_mesh: *Mesh, specs: []const LODLevelSpec) !void {
    return simplify_mod.generateLODLevels(self.allocator, self, source_mesh, specs);
}

pub fn createNavMeshFromTriangles(
    self: anytype,
    positions: []const [3]f32,
    indices: []const u32,
    max_slope_rad: f32,
) !*ai_mod.NavMesh {
    return self.nav.createMeshFromTriangles(self.allocator, positions, indices, max_slope_rad);
}

pub fn createNavMeshGrid(
    self: anytype,
    min_x: f32,
    max_x: f32,
    min_z: f32,
    max_z: f32,
    elevation_y: f32,
    subdiv_x: usize,
    subdiv_z: usize,
    obstacles: []const BoundingBox,
) !*ai_mod.NavMesh {
    return self.nav.createMeshGrid(self.allocator, min_x, max_x, min_z, max_z, elevation_y, subdiv_x, subdiv_z, obstacles);
}

pub fn createNavAgent(self: anytype, nav_mesh: *const ai_mod.NavMesh, start_pos: Vec3) !*ai_mod.NavAgent {
    return self.nav.createAgent(self.allocator, nav_mesh, start_pos);
}

pub fn updateNavAgents(self: anytype, dt: f32) void {
    self.nav.updateAgents(dt);
}

// ---- Animation / physics / camera / picking / UI / projection. ----

pub fn updateAnimations(self: anytype, dt: f32) void {
    scene_animation.updateAnimations(self.animation_groups.items, self.skeletons.items, self.meshes.items, dt);
}

pub fn enablePhysics(self: anytype, gravity: ?Vec3) *PhysicsWorld {
    return self.physics.enable(self.allocator, gravity);
}

/// Finds the rigid body previously created for `mesh`, if any.
pub fn getRigidBody(self: anytype, mesh: *const Mesh) ?*RigidBody {
    if (self.physics.getWorld()) |pw| {
        return pw.findBody(mesh);
    }
    return null;
}

pub fn createRigidBody(self: anytype, mesh: *Mesh, collider: ColliderType, mass: f32) !*RigidBody {
    const pw = self.physics.getWorld() orelse self.physics.enable(self.allocator, null);
    return pw.createBody(mesh, collider, mass);
}

/// Creates a rigid body with collision filter / sensor / event options.
pub fn createRigidBodyWith(self: anytype, mesh: *Mesh, collider: ColliderType, mass: f32, options: physics.BodyOptions) !*RigidBody {
    const pw = self.physics.getWorld() orelse self.physics.enable(self.allocator, null);
    return pw.createBodyWith(mesh, collider, mass, options);
}

pub fn updatePhysics(self: anytype, dt: f32) void {
    self.physics.step(dt);
    if (self.physics.world) |*pw| {
        self.pending_physics_ms = pw.getProfile().step_ms;
    } else {
        self.pending_physics_ms = 0;
    }
}
