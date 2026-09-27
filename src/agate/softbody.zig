//! Soft-body simulation, wave 29 v1: position-based-dynamics (PBD) cloth
//! with mesh coupling.
//!
//! MORPH-AUDIT VERDICT (see module docs below): the engine's sanctioned
//! dynamic-upload path is game-side CPU staging (`Mesh.applyMorphs` writes
//! `morph_staging`, sets `morph_upload_needed`) + a context-thread flush
//! (`Mesh.flushGpuUploads` issues the frame's single `sg.updateBuffer` per
//! buffer and records the byte range in `gpu_upload_meter`), driven by
//! `Scene.flushPendingGpuUploads`. `TrailMesh` (mesh/trail.zig) is the
//! established pattern for a simulated scene entity that owns this
//! discipline itself: own CPU staging arrays + `gpu_dirty` flag staged on
//! update, deferred buffer creation + upload in `flushGpuUploads`, meter
//! accounting next to every `sg.updateBuffer`.
//!
//! Cloth follows the TrailMesh pattern rather than the morph fields
//! themselves: a cloth mesh has no morph targets, and routing deformation
//! through fake morph targets would abuse the animation binding (weights
//! slices, tombstoning in `destroyMesh`). The discipline is identical —
//! dynamic_update vertex buffer, update stages / flush uploads, metered —
//! so the audit's sanctioned path is reused without touching morphs.
//!
//! NON-GOALS (v1): self-collision, tearing, fluids, box3d/rigid-body
//! coupling, GPU simulation, cloth-cloth collision, strain-rate limiting
//! beyond iteration count, persistence (solver state is session-local like
//! particle systems and trail nodes; a saved cloth mesh restores as a
//! static mesh).
//!
//! STRAIN LIMITING (exact): none explicit. Stretch/shear resistance comes
//! from Jacobi-projecting every distance constraint `iterations` times per
//! substep (structural + shear always, skip-one bend optionally). Residual
//! strain after a step is therefore O(1/iterations) and load-dependent —
//! documented, not bounded. There is no continuous strain cap, no stiffness
//! matrix, no tearing threshold. Known PBD property, stated here so it is
//! not misread as a bug: anchored projections are dissipative — a uniform
//! kick to a pinned curtain is absorbed within a few steps (the pins give
//! the projection a momentum sink). Free sheets conserve kick velocity up
//! to `damping` (the impulse test pins this).
//!
//! THREADING: the solver is pure CPU (`Cloth` touches only owned slices —
//! headless-testable, deterministic). `SoftBodyLayer.update` runs game-side
//! inside `Scene.update`'s existing sequential ordering (after animations,
//! before particles); `SoftBody.flushGpuUploads` runs on the context thread
//! from `Scene.flushPendingGpuUploads`. No `sg.*` off the context thread.
//! With zero soft bodies every hook early-outs over an empty list, so
//! rendering and update stay bit-identical.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const Mesh = @import("mesh.zig").Mesh;
const Vertex = @import("mesh.zig").Vertex;
const computeNormals = @import("mesh.zig").computeNormals;
const StandardMaterial = @import("material.zig").StandardMaterial;
const gpu_thread = @import("gpu_thread.zig");
const upload_meter = @import("gpu_upload_meter.zig");
const jobs = @import("jobs.zig");

/// Maximum cloth grid resolution per axis (particle counts, not quads).
/// 64x64 = 4096 particles, the `u16` index fast path still applies.
pub const max_grid: usize = 64;
/// Maximum coupled cloth bodies per scene (parity with the probe/ui3d caps).
pub const max_bodies: usize = 4;
/// Maximum static sphere colliders per cloth.
pub const max_spheres: usize = 8;
/// Fixed-step accumulator safety: simulated seconds consumed per update at
/// most (0.25 s); leftover time past the per-update step cap is dropped so a
/// hitch can never spiral. Dropping is deterministic for a given dt stream.
pub const max_frame_dt: f32 = 0.25;
pub const max_steps_per_update: usize = 32;

pub const SoftBodyError = error{
    InvalidOptions,
    TooManySoftBodies,
    TooManyColliders,
    UnknownSoftBody,
    OutOfMemory,
};

/// Static sphere collider (world space). Cloth particles are projected out
/// positionally; pinned particles are never moved.
pub const SphereCollider = struct {
    center: Vec3 = Vec3.zero,
    radius: f32 = 0.5,
};

pub const ClothOptions = struct {
    /// Particle counts per axis (2..=64 each).
    width: usize = 16,
    height: usize = 16,
    /// Rest spacing between adjacent particles (> 0).
    spacing: f32 = 0.25,
    /// World position of the (0,0) particle (top-left of the curtain).
    origin: Vec3 = Vec3.zero,
    /// Per-particle mass in kg (> 0). Uniform over the grid.
    mass: f32 = 0.1,
    gravity: Vec3 = Vec3.new(0.0, -9.81, 0.0),
    /// Velocity retention loss per second (>= 0). 0 = no damping.
    damping: f32 = 0.02,
    /// Constraint projection passes per substep (>= 1).
    iterations: u32 = 5,
    /// PBD substeps per fixed step (>= 1).
    substeps: u32 = 2,
    /// Fixed simulation step in seconds (> 0, <= 1/5).
    fixed_dt: f32 = 1.0 / 60.0,
    /// Constant wind acceleration in units/s^2 (added to gravity).
    wind: Vec3 = Vec3.zero,
    /// Optional horizontal floor plane (clamps y >= floor_y).
    floor_y: ?f32 = null,
    /// Tangential velocity kept on contact (0 = full friction, 1 = slick).
    friction: f32 = 0.2,
    /// Skip-one bending constraints in addition to structural + shear.
    bend_constraints: bool = false,
    /// Pin the entire top row (y == 0) at creation: the hanging-curtain default.
    pin_top_row: bool = true,
    /// Maximum strain error (|dist - rest| / rest) before a constraint breaks.
    /// null = tearing disabled (infinite tensile strength).
    tear_strain: ?f32 = null,

    pub fn validate(self: ClothOptions) SoftBodyError!void {
        if (self.width < 2 or self.width > max_grid) return error.InvalidOptions;
        if (self.height < 2 or self.height > max_grid) return error.InvalidOptions;
        if (!(self.spacing > 0.0) or !std.math.isFinite(self.spacing)) return error.InvalidOptions;
        if (!(self.mass > 0.0) or !std.math.isFinite(self.mass)) return error.InvalidOptions;
        if (self.iterations == 0 or self.substeps == 0) return error.InvalidOptions;
        if (!(self.fixed_dt > 0.0) or self.fixed_dt > 0.2 or !std.math.isFinite(self.fixed_dt)) return error.InvalidOptions;
        if (!(self.damping >= 0.0) or !std.math.isFinite(self.damping)) return error.InvalidOptions;
        if (!(self.friction >= 0.0) or self.friction > 1.0 or !std.math.isFinite(self.friction)) return error.InvalidOptions;
        if (self.tear_strain) |ts| {
            if (!(ts > 0.0) or !std.math.isFinite(ts)) return error.InvalidOptions;
        }
    }
};

pub const Constraint = struct {
    a: u32,
    b: u32,
    rest: f32,
    active: bool = true,
};

/// Pure-CPU PBD cloth solver: no scene/mesh/sg imports, deterministic for a
/// given options + dt stream. Particles are Verlet-integrated; constraints
/// are Jacobi-projected; collisions are positional projections.
pub const Cloth = struct {
    allocator: std.mem.Allocator,
    width: usize,
    height: usize,
    pos: []Vec3,
    prev: []Vec3,
    inv_mass: []f32,
    constraints: []Constraint,
    spheres: [max_spheres]SphereCollider = undefined,
    sphere_count: usize = 0,
    mass: f32,
    gravity: Vec3,
    damping: f32,
    iterations: u32,
    substeps: u32,
    fixed_dt: f32,
    wind: Vec3,
    floor_y: ?f32,
    friction: f32,
    accumulator: f32 = 0.0,
    sim_time: f32 = 0.0,
    enabled: bool = true,
    tear_strain: ?f32 = null,
    torn_count: usize = 0,
    thread_pool: ?*jobs.Pool = null,

    pub fn init(allocator: std.mem.Allocator, options: ClothOptions) SoftBodyError!Cloth {
        try options.validate();
        const n = options.width * options.height;
        const pos = allocator.alloc(Vec3, n) catch return error.OutOfMemory;
        errdefer allocator.free(pos);
        const prev = allocator.alloc(Vec3, n) catch return error.OutOfMemory;
        errdefer allocator.free(prev);
        const inv_mass = allocator.alloc(f32, n) catch return error.OutOfMemory;
        errdefer allocator.free(inv_mass);

        for (0..options.height) |y| {
            for (0..options.width) |x| {
                const i = y * options.width + x;
                const p = Vec3.new(
                    options.origin.x + @as(f32, @floatFromInt(x)) * options.spacing,
                    options.origin.y - @as(f32, @floatFromInt(y)) * options.spacing,
                    options.origin.z,
                );
                pos[i] = p;
                prev[i] = p;
                inv_mass[i] = 1.0 / options.mass;
            }
        }

        // Structural (right + down) + shear (both diagonals); optional
        // skip-one bend (right-2 + down-2).
        var list: std.ArrayListUnmanaged(Constraint) = .empty;
        errdefer list.deinit(allocator);
        const at = struct {
            fn at(w: usize, x: usize, y: usize) u32 {
                return @intCast(y * w + x);
            }
        }.at;
        const rest1 = options.spacing;
        const rest_diag = options.spacing * @sqrt(2.0);
        for (0..options.height) |y| {
            for (0..options.width) |x| {
                if (x + 1 < options.width) {
                    list.append(allocator, .{ .a = at(options.width, x, y), .b = at(options.width, x + 1, y), .rest = rest1 }) catch return error.OutOfMemory;
                    if (options.bend_constraints and x + 2 < options.width) {
                        list.append(allocator, .{ .a = at(options.width, x, y), .b = at(options.width, x + 2, y), .rest = rest1 * 2.0 }) catch return error.OutOfMemory;
                    }
                }
                if (y + 1 < options.height) {
                    list.append(allocator, .{ .a = at(options.width, x, y), .b = at(options.width, x, y + 1), .rest = rest1 }) catch return error.OutOfMemory;
                    if (options.bend_constraints and y + 2 < options.height) {
                        list.append(allocator, .{ .a = at(options.width, x, y), .b = at(options.width, x, y + 2), .rest = rest1 * 2.0 }) catch return error.OutOfMemory;
                    }
                }
                if (x + 1 < options.width and y + 1 < options.height) {
                    list.append(allocator, .{ .a = at(options.width, x, y), .b = at(options.width, x + 1, y + 1), .rest = rest_diag }) catch return error.OutOfMemory;
                    list.append(allocator, .{ .a = at(options.width, x + 1, y), .b = at(options.width, x, y + 1), .rest = rest_diag }) catch return error.OutOfMemory;
                }
            }
        }

        var self = Cloth{
            .allocator = allocator,
            .width = options.width,
            .height = options.height,
            .pos = pos,
            .prev = prev,
            .inv_mass = inv_mass,
            .constraints = list.toOwnedSlice(allocator) catch return error.OutOfMemory,
            .mass = options.mass,
            .gravity = options.gravity,
            .damping = options.damping,
            .iterations = options.iterations,
            .substeps = options.substeps,
            .fixed_dt = options.fixed_dt,
            .wind = options.wind,
            .floor_y = options.floor_y,
            .friction = options.friction,
            .tear_strain = options.tear_strain,
        };
        if (options.pin_top_row) {
            for (0..options.width) |x| self.setPinned(x, 0, true);
        }
        return self;
    }

    pub fn deinit(self: *Cloth) void {
        self.allocator.free(self.constraints);
        self.allocator.free(self.inv_mass);
        self.allocator.free(self.prev);
        self.allocator.free(self.pos);
        self.* = undefined;
    }

    pub fn particleCount(self: *const Cloth) usize {
        return self.pos.len;
    }

    pub fn indexOf(self: *const Cloth, x: usize, y: usize) usize {
        return y * self.width + x;
    }

    pub fn isPinned(self: *const Cloth, x: usize, y: usize) bool {
        return self.inv_mass[self.indexOf(x, y)] == 0.0;
    }

    /// Pin (inv_mass = 0, velocity killed) or unpin (restores 1/mass) one
    /// particle. Pinned particles are skipped by integrate, solve and
    /// collide, so they never move.
    pub fn setPinned(self: *Cloth, x: usize, y: usize, pinned: bool) void {
        const i = self.indexOf(x, y);
        if (pinned) {
            self.inv_mass[i] = 0.0;
        } else {
            self.inv_mass[i] = 1.0 / self.mass;
        }
        self.prev[i] = self.pos[i];
    }

    /// Instant velocity change in units/s for every unpinned particle
    /// (Verlet: shifts `prev` by dv * substep_h so the velocity gain is
    /// exactly dv regardless of substep count).
    pub fn applyImpulse(self: *Cloth, dv: Vec3) void {
        const sh = self.fixed_dt / @as(f32, @floatFromInt(self.substeps));
        for (self.pos, self.prev, self.inv_mass) |_, *pr, w| {
            if (w == 0.0) continue;
            pr.* = pr.*.sub(dv.scale(sh));
        }
    }

    pub fn addSphere(self: *Cloth, collider: SphereCollider) SoftBodyError!void {
        if (!(collider.radius > 0.0) or !std.math.isFinite(collider.radius)) return error.InvalidOptions;
        if (self.sphere_count >= max_spheres) return error.TooManyColliders;
        self.spheres[self.sphere_count] = collider;
        self.sphere_count += 1;
    }

    pub fn clearSpheres(self: *Cloth) void {
        self.sphere_count = 0;
    }

    pub fn setFloor(self: *Cloth, floor_y: ?f32) void {
        self.floor_y = floor_y;
    }

    /// Deactivates a specific constraint by index. Returns true if severed.
    pub fn tearConstraint(self: *Cloth, index: usize) bool {
        if (index >= self.constraints.len) return false;
        if (!self.constraints[index].active) return false;
        self.constraints[index].active = false;
        self.torn_count += 1;
        return true;
    }

    /// Deactivates all constraints connecting to particle (x, y).
    pub fn tearAt(self: *Cloth, x: usize, y: usize) usize {
        if (x >= self.width or y >= self.height) return 0;
        const idx: u32 = @intCast(self.indexOf(x, y));
        var severed: usize = 0;
        for (self.constraints) |*c| {
            if (c.active and (c.a == idx or c.b == idx)) {
                c.active = false;
                self.torn_count += 1;
                severed += 1;
            }
        }
        return severed;
    }

    /// Cuts a vertical seam down the cloth across column `x`.
    pub fn tearSeam(self: *Cloth, x: usize) usize {
        if (x >= self.width) return 0;
        var severed: usize = 0;
        const w = self.width;
        for (0..self.height) |y| {
            const idx: u32 = @intCast(y * w + x);
            for (self.constraints) |*c| {
                if (c.active and (c.a == idx or c.b == idx)) {
                    const ax = c.a % w;
                    const bx = c.b % w;
                    if (ax != bx) {
                        c.active = false;
                        self.torn_count += 1;
                        severed += 1;
                    }
                }
            }
        }
        return severed;
    }

    /// Reactivates all constraints in the cloth.
    pub fn resetTears(self: *Cloth) void {
        for (self.constraints) |*c| {
            c.active = true;
        }
        self.torn_count = 0;
    }

    /// Advances the simulation by `dt` seconds through the fixed-step
    /// accumulator. Returns true when at least one fixed step ran (the mesh
    /// coupling uses this as the exactly-once upload flag). Disabled cloths
    /// and non-positive dt never step. Deterministic for a given dt stream:
    /// any two dt sequences that consume the same fixed-step count from the
    /// same state produce bit-identical states.
    pub fn step(self: *Cloth, dt: f32) bool {
        if (!self.enabled) return false;
        if (!(dt > 0.0) or !std.math.isFinite(dt)) return false;
        self.accumulator += @min(dt, max_frame_dt);
        var stepped = false;
        var n: usize = 0;
        while (self.accumulator >= self.fixed_dt and n < max_steps_per_update) : (n += 1) {
            self.stepFixed(self.fixed_dt);
            self.accumulator -= self.fixed_dt;
            stepped = true;
        }
        // Hitch guard: leftover past the per-update step cap is dropped
        // (never carried unboundedly); same dt stream, same result.
        if (self.accumulator >= self.fixed_dt) self.accumulator = 0.0;
        return stepped;
    }

    /// One fixed step: `substeps` x (integrate + iterations x solve +
    /// collide). Public for determinism tests; `step` is the normal entry.
    pub fn stepFixed(self: *Cloth, h: f32) void {
        const sh = h / @as(f32, @floatFromInt(self.substeps));
        var s: u32 = 0;
        while (s < self.substeps) : (s += 1) {
            self.integrate(sh);
            var it: u32 = 0;
            while (it < self.iterations) : (it += 1) {
                self.solveConstraints();
            }
            self.collide();
        }
        self.sim_time += h;
    }

    const IntegrateCtx = struct {
        cloth: *Cloth,
        accel_hh: Vec3,
        damp: f32,

        fn run(ctx: *IntegrateCtx, start: usize, end: usize) void {
            const pos = ctx.cloth.pos[start..end];
            const prev = ctx.cloth.prev[start..end];
            const inv_mass = ctx.cloth.inv_mass[start..end];
            const damp = ctx.damp;
            const accel_hh = ctx.accel_hh;
            for (pos, prev, inv_mass) |*p, *pr, w| {
                if (w == 0.0) continue;
                const vel = p.*.sub(pr.*).scale(damp);
                pr.* = p.*;
                p.* = p.*.add(vel).add(accel_hh);
            }
        }
    };

    fn integrate(self: *Cloth, h: f32) void {
        const accel = self.gravity.add(self.wind);
        const damp = @max(0.0, 1.0 - self.damping * h);
        const hh = h * h;
        const accel_hh = accel.scale(hh);
        var ctx = IntegrateCtx{ .cloth = self, .accel_hh = accel_hh, .damp = damp };
        const pool = self.thread_pool orelse jobs.global;
        if (pool) |p| {
            if (self.pos.len >= 256 and p.workerCount() > 0) {
                p.forkJoin(IntegrateCtx, &ctx, IntegrateCtx.run, self.pos.len);
                return;
            }
        }
        ctx.run(0, self.pos.len);
    }

    fn solveConstraints(self: *Cloth) void {
        for (self.constraints) |*c| {
            if (!c.active) continue;
            const wa = self.inv_mass[c.a];
            const wb = self.inv_mass[c.b];
            const w = wa + wb;
            if (w == 0.0) continue;
            const delta = self.pos[c.b].sub(self.pos[c.a]);
            const dist = delta.length();
            if (dist == 0.0) continue;
            const strain = @abs(dist - c.rest) / c.rest;
            if (self.tear_strain) |limit| {
                if (strain > limit) {
                    c.active = false;
                    self.torn_count += 1;
                    continue;
                }
            }
            const diff = (dist - c.rest) / dist;
            self.pos[c.a] = self.pos[c.a].add(delta.scale(diff * (wa / w)));
            self.pos[c.b] = self.pos[c.b].sub(delta.scale(diff * (wb / w)));
        }
    }

    const CollideCtx = struct {
        cloth: *Cloth,

        fn run(ctx: *CollideCtx, start: usize, end: usize) void {
            const c_ptr = ctx.cloth;
            const pos = c_ptr.pos[start..end];
            const prev = c_ptr.prev[start..end];
            const inv_mass = c_ptr.inv_mass[start..end];
            const spheres = c_ptr.spheres[0..c_ptr.sphere_count];
            const floor_y = c_ptr.floor_y;
            const friction = c_ptr.friction;

            for (pos, prev, inv_mass) |*p, *pr, w| {
                if (w == 0.0) continue;
                for (spheres) |s| {
                    const delta = p.*.sub(s.center);
                    const dist = delta.length();
                    if (dist < s.radius) {
                        const n = if (dist > 1e-9) delta.scale(1.0 / dist) else Vec3.new(0.0, 1.0, 0.0);
                        p.* = s.center.add(n.scale(s.radius));
                        const vel = p.*.sub(pr.*);
                        const vt = vel.sub(n.scale(vel.dot(n)));
                        pr.* = p.*.sub(vt.scale(friction));
                    }
                }
                if (floor_y) |fy| {
                    if (p.y < fy) {
                        p.y = fy;
                        const vel = p.*.sub(pr.*);
                        pr.* = Vec3.new(p.x - vel.x * friction, p.y, p.z - vel.z * friction);
                    }
                }
            }
        }
    };

    fn collide(self: *Cloth) void {
        if (self.sphere_count == 0 and self.floor_y == null) return;
        var ctx = CollideCtx{ .cloth = self };
        const pool = self.thread_pool orelse jobs.global;
        if (pool) |p| {
            if (self.pos.len >= 256 and p.workerCount() > 0) {
                p.forkJoin(CollideCtx, &ctx, CollideCtx.run, self.pos.len);
                return;
            }
        }
        ctx.run(0, self.pos.len);
    }

    /// Maximum relative rest-length violation over all active constraints
    /// (|dist - rest| / rest). Convergence tests assert this shrinks.
    pub fn maxStrainError(self: *const Cloth) f32 {
        var worst: f32 = 0.0;
        for (self.constraints) |c| {
            if (!c.active) continue;
            const dist = self.pos[c.a].distance(self.pos[c.b]);
            const err = @abs(dist - c.rest) / c.rest;
            if (err > worst) worst = err;
        }
        return worst;
    }

    /// Wyhash over the raw position bytes: determinism tests compare this.
    pub fn hashState(self: *const Cloth) u64 {
        return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(self.pos));
    }
};

/// Mesh-coupled cloth body: a `Cloth` solver plus the `Vertex` staging it
/// deforms and the `Mesh` that draws it. Staging discipline mirrors
/// TrailMesh: `update` (game side) writes vertices + sets `upload_pending`;
/// `flushGpuUploads` (context side) creates deferred buffers once, then
/// issues the frame's single vertex `sg.updateBuffer` + meter record.
pub const SoftBody = struct {
    allocator: std.mem.Allocator,
    cloth: Cloth,
    mesh: *Mesh,
    material: *StandardMaterial,
    vertices: []Vertex,
    indices: []u32,
    buffers_pending: bool = false,
    upload_pending: bool = false,

    fn deinitCpu(self: *SoftBody, allocator: std.mem.Allocator) void {
        self.cloth.deinit();
        allocator.free(self.indices);
        allocator.free(self.vertices);
        // Mesh + material stay owned by the scene registries (or the retire
        // queue after remove): never touched here. The name slice is owned
        // by the mesh (owns_name); the material name aliases it.
    }

    /// Rebuilds staging vertices from solver positions and recomputes
    /// normals + bounds. Game side only (no sg.*).
    pub fn writeVertices(self: *SoftBody) void {
        var min_pt = Vec3.new(std.math.inf(f32), std.math.inf(f32), std.math.inf(f32));
        var max_pt = Vec3.new(-std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32));
        for (self.vertices, self.cloth.pos) |*v, p| {
            v.position = p.toArray();
            min_pt = Vec3.new(@min(min_pt.x, p.x), @min(min_pt.y, p.y), @min(min_pt.z, p.z));
            max_pt = Vec3.new(@max(max_pt.x, p.x), @max(max_pt.y, p.y), @max(max_pt.z, p.z));
        }
        computeNormals(self.vertices, self.indices, null);
        self.mesh.local_bounding_box = BoundingBox.init(min_pt, max_pt);
        self.mesh.cached_aabb = self.mesh.local_bounding_box;
    }

    /// Context thread only (called from `Scene.flushPendingGpuUploads`).
    /// Headless-safe: with no sg context the pending flags still clear, so
    /// tests observe the flag discipline without any `sg.*` call.
    pub fn flushGpuUploads(self: *SoftBody) void {
        if (self.buffers_pending and sg.isvalid()) {
            const vb = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                .size = self.vertices.len * @sizeOf(Vertex),
            });
            // Cloth index buffers are always u32 (vertices cap at 4096, so
            // this wastes nothing material and keeps one upload path).
            const ib = sg.makeBuffer(.{
                .usage = .{ .index_buffer = true },
                .data = sg.asRange(self.indices),
            });
            if (vb.id != 0 and ib.id != 0) {
                self.mesh.vertex_buffer = vb;
                self.mesh.index_buffer = ib;
                self.buffers_pending = false;
                // Fresh buffers need full content: force the vertex upload.
                self.upload_pending = true;
            } else {
                if (vb.id != 0) sg.destroyBuffer(vb);
                if (ib.id != 0) sg.destroyBuffer(ib);
            }
        }
        if (!self.upload_pending) return;
        self.upload_pending = false;
        if (self.mesh.vertex_buffer.id != 0) {
            sg.updateBuffer(self.mesh.vertex_buffer, sg.asRange(self.vertices));
            upload_meter.record(self.vertices.len * @sizeOf(Vertex));
        }
    }
};

/// Scene-owned registry of coupled cloth bodies. The layer owns the
/// `SoftBody` structs + all CPU staging; meshes/materials live in the scene
/// registries (or the retire queue after removal). Scene passes itself as
/// `anytype` (TrailLayer pattern) so this module never imports scene.zig.
pub const SoftBodyLayer = struct {
    bodies: std.ArrayListUnmanaged(*SoftBody) = .empty,
    thread_pool: ?*jobs.Pool = null,

    pub fn deinit(self: *SoftBodyLayer, allocator: std.mem.Allocator) void {
        for (self.bodies.items) |b| {
            b.deinitCpu(allocator);
            allocator.destroy(b);
        }
        self.bodies.deinit(allocator);
    }

    pub fn count(self: *const SoftBodyLayer) usize {
        return self.bodies.items.len;
    }

    pub fn get(self: *SoftBodyLayer, index: usize) ?*SoftBody {
        if (index >= self.bodies.items.len) return null;
        return self.bodies.items[index];
    }

    /// Creates a cloth + its deformable mesh (textured, two-sided standard
    /// material) and registers both. Headless/off-context safe: GPU buffers
    /// defer to the first context-thread flush.
    pub fn create(self: *SoftBodyLayer, scene: anytype, name: []const u8, options: ClothOptions) SoftBodyError!*SoftBody {
        if (self.bodies.items.len >= max_bodies) return error.TooManySoftBodies;
        try options.validate();
        const allocator = scene.allocator;

        var cloth = try Cloth.init(allocator, options);
        errdefer cloth.deinit();

        const n = options.width * options.height;
        const vertices = allocator.alloc(Vertex, n) catch return error.OutOfMemory;
        errdefer allocator.free(vertices);
        const quads = (options.width - 1) * (options.height - 1);
        const indices = allocator.alloc(u32, quads * 6) catch return error.OutOfMemory;
        errdefer allocator.free(indices);

        // Grid topology: row-major, y-down (curtain in the XY plane facing
        // +Z). Winding (a, c, b) / (b, c, d) faces +Z.
        const at = struct {
            fn at(w: usize, x: usize, y: usize) u32 {
                return @intCast(y * w + x);
            }
        }.at;
        for (vertices, 0..) |*v, i| {
            const x = i % options.width;
            const y = i / options.width;
            v.* = .{
                .position = cloth.pos[i].toArray(),
                .normal = .{ 0.0, 0.0, 1.0 },
                .color = .{ 1.0, 1.0, 1.0, 1.0 },
                .uv = .{
                    @as(f32, @floatFromInt(x)) / @as(f32, @floatFromInt(options.width - 1)),
                    @as(f32, @floatFromInt(y)) / @as(f32, @floatFromInt(options.height - 1)),
                },
                .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
            };
        }
        var k: usize = 0;
        for (0..options.height - 1) |y| {
            for (0..options.width - 1) |x| {
                const a = at(options.width, x, y);
                const b = at(options.width, x + 1, y);
                const c = at(options.width, x, y + 1);
                const d = at(options.width, x + 1, y + 1);
                indices[k + 0] = a;
                indices[k + 1] = c;
                indices[k + 2] = b;
                indices[k + 3] = b;
                indices[k + 4] = c;
                indices[k + 5] = d;
                k += 6;
            }
        }

        // Name + mesh: manual cleanup until the mesh is adopted by the
        // scene registry; afterwards a single rollback errdefer owns the
        // teardown. No separate owned_name/mesh errdefers: they would
        // double-free against both the append-failure path below and the
        // rollback (mesh.deinit frees the name via owns_name).
        const owned_name = allocator.dupe(u8, name) catch return error.OutOfMemory;
        const mesh = allocator.create(Mesh) catch {
            allocator.free(owned_name);
            return error.OutOfMemory;
        };
        mesh.* = .{
            .name = owned_name,
            .owns_name = true,
            .vertex_buffer = .{},
            .index_buffer = .{},
            .vertex_count = @intCast(n),
            .index_count = @intCast(indices.len),
            .index_type = .UINT32,
            .local_bounding_box = clothBoundingBox(&cloth),
            .cast_shadows = true,
        };
        mesh.cached_aabb = mesh.local_bounding_box;
        scene.meshes.append(allocator, mesh) catch {
            allocator.free(owned_name);
            allocator.destroy(mesh);
            return error.OutOfMemory;
        };
        // Mesh + name now scene-owned; rollback preserves removeMesh
        // semantics (unlink only, no retire queue: never published).
        errdefer {
            _ = scene.removeMesh(mesh);
            mesh.deinit(allocator);
            allocator.destroy(mesh);
        }

        const mat = scene.createStandardMaterial(owned_name) catch return error.OutOfMemory;
        // Material name aliases the mesh-owned slice; unlink + free the
        // struct before the mesh rollback above frees the name. A fresh
        // material owns no texture, so no texture teardown here.
        errdefer {
            for (scene.materials.items, 0..) |m, i| {
                if (m == mat) {
                    _ = scene.materials.swapRemove(i);
                    break;
                }
            }
            allocator.destroy(mat);
        }
        mat.double_sided = true;
        mesh.material = .{ .standard = mat };

        const body = allocator.create(SoftBody) catch return error.OutOfMemory;
        body.* = .{
            .allocator = allocator,
            .cloth = cloth,
            .mesh = mesh,
            .material = mat,
            .vertices = vertices,
            .indices = indices,
            .buffers_pending = true,
            .upload_pending = false,
        };
        body.cloth.thread_pool = self.thread_pool orelse jobs.global;
        self.bodies.append(allocator, body) catch {
            allocator.destroy(body);
            return error.OutOfMemory;
        };
        return body;
    }

    /// Advances every enabled body and stages its vertex upload. Game side
    /// only (no sg.*). With zero bodies this is a no-op over an empty list.
    pub fn update(self: *SoftBodyLayer, dt: f32) void {
        for (self.bodies.items) |b| {
            if (b.cloth.step(dt)) {
                b.writeVertices();
                b.upload_pending = true;
            }
        }
    }

    /// Unlinks + frees the body at `index` and returns its mesh for the
    /// caller to retire. The mesh itself is never touched here.
    pub fn extractAt(self: *SoftBodyLayer, allocator: std.mem.Allocator, index: usize) SoftBodyError!*Mesh {
        if (index >= self.bodies.items.len) return error.UnknownSoftBody;
        const body = self.bodies.orderedRemove(index);
        const mesh = body.mesh;
        body.deinitCpu(allocator);
        allocator.destroy(body);
        return mesh;
    }

    /// Drops the body bound to `mesh` (if any), freeing its CPU side. Used
    /// by `Scene.destroyMesh` so a directly-destroyed cloth mesh never
    /// leaves a dangling body. The mesh itself is never touched here.
    pub fn removeForMesh(self: *SoftBodyLayer, allocator: std.mem.Allocator, mesh: *const Mesh) void {
        for (self.bodies.items, 0..) |b, i| {
            if (b.mesh == mesh) {
                const body = self.bodies.orderedRemove(i);
                body.deinitCpu(allocator);
                allocator.destroy(body);
                return;
            }
        }
    }
};

fn clothBoundingBox(cloth: *const Cloth) BoundingBox {
    var min_pt = Vec3.new(std.math.inf(f32), std.math.inf(f32), std.math.inf(f32));
    var max_pt = Vec3.new(-std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32));
    for (cloth.pos) |p| {
        min_pt = Vec3.new(@min(min_pt.x, p.x), @min(min_pt.y, p.y), @min(min_pt.z, p.z));
        max_pt = Vec3.new(@max(max_pt.x, p.x), @max(max_pt.y, p.y), @max(max_pt.z, p.z));
    }
    return BoundingBox.init(min_pt, max_pt);
}

// ---------------------------------------------------------------------------
// Headless solver tests: validation, constraint layout, pins, convergence,
// determinism, collisions, impulse, wind. Mesh-coupling + scene-API tests
// live in scene/tests.zig (that module already owns the Scene fixtures;
// defining them here would be an import cycle: scene.zig imports this one).
// ---------------------------------------------------------------------------

fn testOptions() ClothOptions {
    return .{
        .width = 6,
        .height = 6,
        .spacing = 0.25,
        .pin_top_row = true,
        .iterations = 8,
        .substeps = 2,
    };
}

test "cloth rejects invalid options" {
    const t = std.testing;
    var bad = testOptions();
    bad.width = 1;
    try t.expectError(error.InvalidOptions, Cloth.init(t.allocator, bad));
    bad = testOptions();
    bad.height = max_grid + 1;
    try t.expectError(error.InvalidOptions, Cloth.init(t.allocator, bad));
    bad = testOptions();
    bad.spacing = 0.0;
    try t.expectError(error.InvalidOptions, Cloth.init(t.allocator, bad));
    bad = testOptions();
    bad.mass = -1.0;
    try t.expectError(error.InvalidOptions, Cloth.init(t.allocator, bad));
    bad = testOptions();
    bad.iterations = 0;
    try t.expectError(error.InvalidOptions, Cloth.init(t.allocator, bad));
    bad = testOptions();
    bad.substeps = 0;
    try t.expectError(error.InvalidOptions, Cloth.init(t.allocator, bad));
    bad = testOptions();
    bad.fixed_dt = 0.0;
    try t.expectError(error.InvalidOptions, Cloth.init(t.allocator, bad));
    bad = testOptions();
    bad.friction = 2.0;
    try t.expectError(error.InvalidOptions, Cloth.init(t.allocator, bad));
}

test "cloth constraint counts: structural + shear, bend optional" {
    const t = std.testing;
    const w: usize = 6;
    const h: usize = 5;
    var c = try Cloth.init(t.allocator, .{ .width = w, .height = h, .pin_top_row = false });
    defer c.deinit();
    const structural = (w - 1) * h + w * (h - 1);
    const shear = 2 * (w - 1) * (h - 1);
    try t.expectEqual(structural + shear, c.constraints.len);

    var cb = try Cloth.init(t.allocator, .{ .width = w, .height = h, .pin_top_row = false, .bend_constraints = true });
    defer cb.deinit();
    const bend = (w - 2) * h + w * (h - 2);
    try t.expectEqual(structural + shear + bend, cb.constraints.len);
}

test "pinned particles never move under gravity" {
    const t = std.testing;
    var c = try Cloth.init(t.allocator, testOptions());
    defer c.deinit();
    var rest: [6]Vec3 = undefined;
    for (0..6) |x| rest[x] = c.pos[c.indexOf(x, 0)];
    for (0..120) |_| _ = c.step(1.0 / 60.0);
    for (0..6) |x| try t.expect(c.pos[c.indexOf(x, 0)].eql(rest[x]));
    // ...while the free sheet sags below its rest pose.
    try t.expect(c.pos[c.indexOf(3, 5)].y < -1.2);
}

test "rest-length error shrinks (convergence sanity)" {
    const t = std.testing;
    var c = try Cloth.init(t.allocator, testOptions());
    defer c.deinit();
    // Deterministic perturbation: yank the interior down. Error > 0 now.
    for (1..5) |y| {
        for (1..5) |x| {
            const i = c.indexOf(x, y);
            c.pos[i].y -= 0.35;
            c.prev[i] = c.pos[i];
        }
    }
    const before = c.maxStrainError();
    try t.expect(before > 0.05);
    for (0..60) |_| _ = c.step(1.0 / 60.0);
    const after = c.maxStrainError();
    try t.expect(after < before * 0.5);
}

test "two identical runs produce identical hashes (determinism)" {
    const t = std.testing;
    var a = try Cloth.init(t.allocator, testOptions());
    defer a.deinit();
    var b = try Cloth.init(t.allocator, testOptions());
    defer b.deinit();
    for (0..90) |_| {
        _ = a.step(1.0 / 60.0);
        _ = b.step(1.0 / 60.0);
    }
    try t.expectEqual(a.hashState(), b.hashState());
}

test "fixed-step accumulator is deterministic under variable dt" {
    const t = std.testing;
    var a = try Cloth.init(t.allocator, testOptions());
    defer a.deinit();
    var b = try Cloth.init(t.allocator, testOptions());
    defer b.deinit();
    var c = try Cloth.init(t.allocator, testOptions());
    defer c.deinit();
    for (0..60) |_| _ = a.step(1.0 / 60.0);
    for (0..30) |_| _ = b.step(1.0 / 30.0);
    for (0..120) |_| _ = c.step(1.0 / 120.0);
    try t.expectEqual(a.hashState(), b.hashState());
    try t.expectEqual(a.hashState(), c.hashState());
    // Non-positive dt never steps.
    const h = a.hashState();
    try t.expect(!a.step(0.0));
    try t.expect(!a.step(-0.1));
    try t.expectEqual(h, a.hashState());
}

test "sphere collider pushes particles out" {
    const t = std.testing;
    var c = try Cloth.init(t.allocator, .{
        .width = 4,
        .height = 4,
        .spacing = 0.25,
        .pin_top_row = false,
        .gravity = Vec3.zero,
    });
    defer c.deinit();
    // Park every particle at the sphere center, then step: all must exit.
    for (c.pos) |*p| p.* = Vec3.new(1.0, 2.0, 3.0);
    for (c.prev) |*p| p.* = Vec3.new(1.0, 2.0, 3.0);
    try c.addSphere(.{ .center = Vec3.new(1.0, 2.0, 3.0), .radius = 1.0 });
    for (0..10) |_| _ = c.step(1.0 / 60.0);
    for (c.pos) |p| {
        try t.expect(p.distance(Vec3.new(1.0, 2.0, 3.0)) >= 1.0 - 1e-4);
    }
    // Collider cap is enforced.
    for (0..max_spheres - 1) |_| try c.addSphere(.{});
    try t.expectError(error.TooManyColliders, c.addSphere(.{}));
    try t.expectError(error.InvalidOptions, c.addSphere(.{ .radius = 0.0 }));
}

test "floor plane clamps falling cloth" {
    const t = std.testing;
    var c = try Cloth.init(t.allocator, .{
        .width = 4,
        .height = 4,
        .spacing = 0.25,
        .origin = Vec3.new(0.0, 2.0, 0.0),
        .pin_top_row = false,
        .floor_y = 0.0,
    });
    defer c.deinit();
    for (0..240) |_| _ = c.step(1.0 / 60.0);
    for (c.pos) |p| try t.expect(p.y >= 0.0 - 1e-5);
}

test "applyImpulse lifts the sheet vs control" {
    const t = std.testing;
    // Free sheet, zero gravity, zero pins: the kick is a pure translation
    // (zero constraint strain), so no projection can eat it. NOTE: anchored
    // projections are dissipative — even one pinned corner absorbs a uniform
    // kick within a few steps. That is inherent PBD behavior with stiff
    // anchored constraints, not a bug (see header docs). Pin immunity
    // itself is pinned by the "pinned particles never move" test: the
    // impulse path skips inv_mass == 0 exactly like integrate/solve/collide.
    var a = try Cloth.init(t.allocator, .{
        .width = 4,
        .height = 4,
        .spacing = 0.25,
        .pin_top_row = false,
        .gravity = Vec3.zero,
    });
    defer a.deinit();
    var b = try Cloth.init(t.allocator, .{
        .width = 4,
        .height = 4,
        .spacing = 0.25,
        .pin_top_row = false,
        .gravity = Vec3.zero,
    });
    defer b.deinit();
    a.applyImpulse(Vec3.new(0.0, 5.0, 0.0));
    const pai = a.indexOf(1, 1);
    for (0..20) |_| {
        _ = a.step(1.0 / 60.0);
        _ = b.step(1.0 / 60.0);
    }
    // ~5 units/s * 1/3 s = ~1.67 rise (damping 0.02 shaves ~0.01%).
    try t.expect(a.pos[pai].y > b.pos[b.indexOf(1, 1)].y + 1.0);
}

test "disabled cloth pauses (step is a no-op)" {
    const t = std.testing;
    var c = try Cloth.init(t.allocator, testOptions());
    defer c.deinit();
    c.enabled = false;
    const h = c.hashState();
    try t.expect(!c.step(1.0 / 60.0));
    try t.expectEqual(h, c.hashState());
    try t.expectEqual(@as(f32, 0.0), c.sim_time);
}

test "pin/unpin round-trips mass" {
    const t = std.testing;
    var c = try Cloth.init(t.allocator, testOptions());
    defer c.deinit();
    try t.expect(c.isPinned(0, 0));
    c.setPinned(0, 0, false);
    try t.expect(!c.isPinned(0, 0));
    const at = c.indexOf(0, 0);
    try t.expectApproxEqAbs(@as(f32, 10.0), c.inv_mass[at], 1e-5); // mass 0.1
    c.setPinned(0, 0, true);
    try t.expect(c.isPinned(0, 0));
}

test "wind accelerates the sheet downwind vs control" {
    const t = std.testing;
    var a = try Cloth.init(t.allocator, .{
        .width = 4,
        .height = 4,
        .spacing = 0.25,
        .pin_top_row = false,
        .gravity = Vec3.zero,
        .wind = Vec3.new(2.0, 0.0, 0.0),
    });
    defer a.deinit();
    var b = try Cloth.init(t.allocator, .{
        .width = 4,
        .height = 4,
        .spacing = 0.25,
        .pin_top_row = false,
        .gravity = Vec3.zero,
    });
    defer b.deinit();
    for (0..30) |_| {
        _ = a.step(1.0 / 60.0);
        _ = b.step(1.0 / 60.0);
    }
    try t.expect(a.pos[a.indexOf(1, 1)].x > b.pos[b.indexOf(1, 1)].x + 0.05);
}

test "cloth tearing: tearConstraint, tearSeam, resetTears and auto-tearing" {
    const t = std.testing;
    var c = try Cloth.init(t.allocator, .{
        .width = 4,
        .height = 4,
        .spacing = 0.25,
        .pin_top_row = true,
        .tear_strain = 0.5,
    });
    defer c.deinit();

    try t.expectEqual(@as(usize, 0), c.torn_count);
    try t.expect(c.tearConstraint(0));
    try t.expectEqual(@as(usize, 1), c.torn_count);
    // Already severed
    try t.expect(!c.tearConstraint(0));

    // Seam cut
    const cut = c.tearSeam(1);
    try t.expect(cut > 0);
    try t.expectEqual(@as(usize, 1 + cut), c.torn_count);

    // Reset restores all
    c.resetTears();
    try t.expectEqual(@as(usize, 0), c.torn_count);

    // Simulate with excessive stretch to trigger auto-tearing
    c.pos[c.indexOf(1, 2)] = Vec3.new(10.0, -10.0, 0.0);
    _ = c.step(1.0 / 60.0);
    try t.expect(c.torn_count > 0);
}
