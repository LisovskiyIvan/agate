//! ParticleSystem owner: type, lifecycle and thin forwarders. Split out of
//! `particles.zig` (facade).
//!
//! This module owns the `ParticleSystem` type: the fields, the trivial
//! lifecycle (`init`/`deinit`/`start`/`stop`/`reset`), the small accessors
//! (`addSubEmitter`/`subEmitters`/`burst`/`resolveEmitterMatrix`) and thin
//! forwarders into the siblings below, so every call site keeps working
//! unchanged (same pattern as `profiler/core.zig` — the type cannot span
//! files in Zig, so cross-file methods live as free functions taking the
//! system as `anytype` and are reached through same-name forwarders here):
//!
//! - `types.zig` — modes, errors, slot/state layouts, pure math, bounds.
//! - `sampling.zig` — `sampleSpawn` (+ visual helpers).
//! - `cpu.zig` — `updateCpu` (three-phase integrate/compact/fill).
//! - `gpu.zig` — `updateGpu` (+ slot-ring staging).
//! - `flow.zig` — `setFlowMap`/`clearFlowMap`/`sampleFlow`.
//! - `collisions.zig` — CPU-only collisions (`CollisionMode`,
//!   `ParticleSphereCollider`, setters, pure contact resolvers).
//! - `subemitters.zig` — on-death child-spawn pass.
//! - `compute_mode.zig` — `.compute` staging/flush/dispatch/retire.
//!
//! Anti-cycle rule (same as `profiler/`, `ui/`): siblings take the system as
//! `anytype` and never import this module or the `particles.zig` facade back;
//! this module passes `self` straight through. `particles.zig` re-exports
//! `ParticleSystem` (plus `SubEmitter`/`SubEmitterTrigger`) under its
//! historical path. Cross-leaf helpers (`sampling.sampleSpawn`,
//! `gpu.pushGpuSlot`, `compute_mode.pushComputeSpawn`, `flow.activeFlowCtx`)
//! are `pub` in their home module for the sibling that needs them but are
//! deliberately NOT re-exported by the facade, so the public surface is
//! identical to the pre-split file.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Texture = @import("../texture.zig").Texture;
const Mesh = @import("../mesh.zig").Mesh;
const jobs = @import("../jobs.zig");
const gpu_thread = @import("../gpu_thread.zig");
const upload_meter = @import("../gpu_upload_meter.zig");
const pc_shd = @import("particle_compute_shader");

const types = @import("types.zig");
const sampling = @import("sampling.zig");
const cpu = @import("cpu.zig");
const gpu = @import("gpu.zig");
const flow = @import("flow.zig");
const collisions = @import("collisions.zig");
const compute_mode = @import("compute_mode.zig");

const Vec2 = math.Vec2;
const Vec3 = math.Vec3;
const Vec4 = math.Vec4;
const Color4 = math.Color4;
const Mat4 = math.Mat4;
const ParticleBlendMode = types.ParticleBlendMode;
const FlowSpace = types.FlowSpace;
const FlowWrap = types.FlowWrap;
const SimulationMode = types.SimulationMode;
const UpdateError = types.UpdateError;
const ComputeModeError = types.ComputeModeError;
const GpuParticleSlot = types.GpuParticleSlot;
const Particle = types.Particle;
const ParticleInstanceData = types.ParticleInstanceData;
const max_sub_emitters = types.max_sub_emitters;
const CollisionMode = collisions.CollisionMode;
const CollisionError = collisions.CollisionError;
const ParticleSphereCollider = collisions.ParticleSphereCollider;
const max_sphere_colliders = collisions.max_sphere_colliders;
const ParticleBoxCollider = collisions.ParticleBoxCollider;
const max_box_colliders = collisions.max_box_colliders;
const ParticlePlaneCollider = collisions.ParticlePlaneCollider;
const max_plane_colliders = collisions.max_plane_colliders;

/// Sub-emitter trigger point. Only `.on_death` exists today (Babylon.js
/// parity target): a parent particle spawns children in another system on the
/// tick its age reaches its lifetime.
pub const SubEmitterTrigger = enum {
    on_death,
};

/// One child-spawn rule evaluated when a particle of the owning system dies.
/// Off by default: a system with zero sub-emitters takes no extra PRNG draws
/// and follows the legacy update path bit-for-bit.
pub const SubEmitter = struct {
    /// Child system receiving the spawned particles. May be the owner itself;
    /// see the chain rule on `max_sub_emitter_depth`.
    system: *ParticleSystem,
    trigger: SubEmitterTrigger = .on_death,
    /// Spawn chance per dying particle, in [0, 1]. Evaluated from a
    /// deterministic per-death hash (seed + tick + death index), never from
    /// the shared PRNG, so attaching a sub-emitter never perturbs the parent
    /// emission stream and results are worker-count invariant.
    probability: f32 = 1.0,
    /// Children spawned per triggered death (bounded per tick by
    /// `max_sub_emitter_spawns_per_tick`).
    count: u32 = 1,
    /// child.velocity = parent.velocity * inherit_velocity +
    ///     sampled.velocity * (1 - inherit_velocity), clamped to [0, 1].
    /// 1.0 keeps the full parent velocity, 0.0 uses the child system's own
    /// sampled velocity.
    inherit_velocity: f32 = 0.5,
    /// True: children start at the death position (+ spawn_radius jitter).
    /// False: children keep their own system's sampled spawn position.
    inherit_position: bool = true,
    /// Uniform cube half-extent around the death position added to each
    /// child (>= 0; negative clamps to 0). Drawn from the per-death hash
    /// stream, so jitter is deterministic and worker-count invariant.
    spawn_radius: f32 = 0.0,
};

pub const ParticleSystem = struct {
    name: []const u8,
    allocator: std.mem.Allocator,
    particles: []Particle,
    instances: []ParticleInstanceData,
    /// Scratch survival flags for the parallel CPU integration (phase A
    /// writes them, the serial compaction phase consumes them). Allocated
    /// alongside `particles`; OOM surfaces from init, never at update time.
    alive_scratch: []u8 = &.{},
    /// Optional per-system override of the worker pool used by the CPU
    /// integration. Null falls back to `jobs.global`, then to serial
    /// execution — a scheduling detail, never a behavior change.
    thread_pool: ?*jobs.Pool = null,
    capacity: usize,
    active_count: usize = 0,
    /// Stage 3: update() stages CPU data and sets these flags; the sg
    /// uploads happen in flushGpuUploads on the render side (sg is
    /// single-context — the update phase must stay free of sg.* calls).
    instance_dirty: bool = false,
    gpu_flush_pending: bool = false,

    instance_buffer: sg.Buffer,
    /// True when `.instance_buffer` could not be created at construction
    /// time (off-context `createParticleSystem`): flushGpuUploads creates it
    /// on the render side once a context is available.
    instance_buffer_pending: bool = false,
    texture: ?Texture = null,
    blend_mode: ParticleBlendMode = .additive,

    // --- GPU simulation path (simulation_mode == .gpu) ---
    /// See `SimulationMode` for the exact feature matrix. Default .cpu keeps
    /// existing systems bit-for-bit identical.
    simulation_mode: SimulationMode = .cpu,
    /// Exponential drag coefficient (1/s), GPU-path only (see matrix note (1)).
    drag: f32 = 0.0,
    /// Epoch clock for the GPU path: seconds accumulated by updateGpu. Slot
    /// spawn times are offsets from this epoch so float32 magnitudes stay small.
    clock_seconds: f32 = 0.0,
    /// CPU mirror of the GPU slot ring; provisioned lazily on first GPU update
    /// (and from tests) so CPU-only systems pay no extra memory.
    gpu_slots: []GpuParticleSlot = &.{},
    /// Instance buffer holding `gpu_slots` (per-instance spawn records).
    gpu_slot_buffer: sg.Buffer = .{},
    /// Staged by update() on first `.gpu` use; the buffer itself is created in
    /// flushGpuUploads on the context thread, so the update phase stays free
    /// of sg.* calls. Cleared once the buffer exists.
    gpu_slot_buffer_pending: bool = false,
    /// Ring cursor: next slot to overwrite. Emission is strictly sequential,
    /// which keeps the CPU cursor and the upload ranges trivially in sync.
    gpu_write_cursor: usize = 0,
    /// High-water mark of written slots = GPU draw instance count. Slots below
    /// it may be dead; the shader culls them, so `active_count` (aliased to
    /// this in .gpu mode) is an upper bound — the exact live count lives on
    /// the GPU only.
    gpu_high_water: usize = 0,
    // Dirty bookkeeping for the once-per-frame sg.updateBuffer: emission writes
    // a contiguous cursor run per frame unless it wraps the ring, in which case
    // the whole [0, high_water) prefix is re-uploaded.
    gpu_dirty: bool = false,
    gpu_dirty_start: usize = 0,
    gpu_dirty_end: usize = 0,
    gpu_dirty_wrapped: bool = false,

    // --- Stateful compute simulation (simulation_mode == .compute; v1) ---
    // CPU side stages spawn records (shared `sampleSpawn`, same PRNG order as
    // .cpu/.gpu) into `compute_staging`; the prepare-boundary flush uploads
    // the [0, staged) prefix into the spawn storage buffer and dispatches the
    // compute pass, which claims the ring window
    // [stage_base, stage_base + staged) mod capacity. All fields default to
    // zero/empty: non-compute systems pay no memory and take no extra draws.
    /// Staged spawn records in emission order (capacity-sized when
    /// provisioned; `GpuParticleSlot` layout reused verbatim).
    compute_staging: []GpuParticleSlot = &.{},
    /// Ring cursor at the first staged emission (shader `addr.y`).
    compute_stage_base: usize = 0,
    /// Staged count (shader `addr.z`); always <= capacity (bulk eviction
    /// keeps the latest window when one frame stages more than capacity).
    compute_staged: usize = 0,
    /// Ring cursor: next slot to overwrite (wraps; overwrites oldest).
    compute_cursor: usize = 0,
    /// High-water mark of written slots = draw instance count (upper bound;
    /// exact live count lives on the GPU only). Aliased to `active_count`.
    compute_high_water: usize = 0,
    /// Frame dt accumulated across updates since the last flush; the dispatch
    /// consumes (and zeroes) it, so paused frames (no update) dispatch nothing.
    compute_dt_accum: f32 = 0.0,
    /// Staged by update() (sg objects are context-thread only); consumed by
    /// flushGpuUploads. Without a valid sg context the flags stay set and a
    /// later flush retries.
    compute_buffers_pending: bool = false,
    compute_flush_pending: bool = false,
    /// Staged by reset(): the next flush re-zeroes the GPU state buffer so
    /// the new epoch starts clean.
    compute_state_clear_pending: bool = false,
    /// Latched by the context-side flush on a backend without compute
    /// support: subsequent updates return error.ComputeUnsupported (never a
    /// silent fallback). Stays false headless (unknown until a context exists).
    compute_known_unsupported: bool = false,
    /// Compute dispatches issued since creation (monotonic, wrapped): the
    /// live proof that the `.compute` path actually ran, exposed via
    /// `computeDispatchCount()` for apps and GPU fixtures.
    compute_dispatches: u64 = 0,
    /// Support override (tests/apps): `false` forces
    /// error.ComputeUnsupported out of the mode-selecting API and `update`
    /// without needing a GPU context; `true` forces availability. Null (the
    /// default) means "ask the backend when a context is live, assume
    /// stageable while headless".
    compute_support_override: ?bool = null,
    /// GPU objects: created in flushGpuUploads on the context thread,
    /// destroyed in deinit/deinitComputeGpuObjects on the context thread,
    /// retired via takeGpuBuffersForRetire (buffers only) from any thread.
    compute_state_buffer: sg.Buffer = .{},
    compute_spawn_buffer: sg.Buffer = .{},
    /// Compute-baked draw instances (`ParticleInstanceData` layout): bound
    /// through the EXISTING cpu billboard pipeline, so no new render
    /// pipeline/shader exists for this mode.
    compute_draw_buffer: sg.Buffer = .{},
    compute_state_view: sg.View = .{},
    compute_spawn_view: sg.View = .{},
    compute_draw_view: sg.View = .{},
    compute_pipeline: sg.Pipeline = .{},
    compute_shader: sg.Shader = .{},

    // Emitter shape & origin
    emitter_position: Vec3 = Vec3.zero,
    emitter_box_min: Vec3 = Vec3.zero,
    emitter_box_max: Vec3 = Vec3.zero,

    // Emission rate
    emit_rate: f32 = 100.0,
    is_emitting: bool = false,
    emit_accumulator: f32 = 0.0,

    // Velocity & physics
    direction_min: Vec3 = Vec3.new(-0.2, 1.0, -0.2),
    direction_max: Vec3 = Vec3.new(0.2, 2.0, 0.2),
    speed_min: f32 = 1.0,
    speed_max: f32 = 2.0,
    gravity: Vec3 = Vec3.zero,

    // --- Flow-field texture (Babylon.js parity; CPU-only force field) ---
    // Off by default: with `flow_map == null` (or no CPU copy, or
    // `flow_strength == 0`) updateCpu skips sampling entirely — one cached
    // branch per particle, no memory traffic — and follows the legacy path
    // bit-for-bit. The CPU simulation NEVER reads GPU memory: sampling uses
    // only `flow_pixels`, an owned CPU-side RGBA8 copy installed by
    // setFlowMap (one alloc + memcpy, w*h*4 bytes; freed in deinit/
    // clearFlowMap/setFlowMap). Assigning `flow_map` directly without a CPU
    // copy leaves the field disarmed: documented, ignored, no crash.
    // Texel encoding (linear data, never sRGB-converted): R = dir.x,
    // G = dir.z in [0, 255] -> [-1, 1] via (v/255*2-1), B = per-texel
    // strength in [0, 1] via (v/255); A is ignored. Sampling is bilinear
    // (corner convention: uv (0,0)/(1,1) hit the corner texel centers
    // exactly) and integrates as acceleration:
    //   velocity += decoded_dir * tex_strength * flow_strength * dt.
    // `flow_map` itself is the optional GPU handle (future GPU-path
    // binding / debug views); the CPU path never dereferences it.
    flow_map: ?Texture = null,
    /// Owned CPU-side RGBA8 copy of the flow field (w*h*4 bytes). Empty
    /// means "no field" regardless of `flow_map`.
    flow_pixels: []u8 = &.{},
    flow_width: u32 = 0,
    flow_height: u32 = 0,
    /// Field acceleration scale in units/s^2. 0.0 disarms the field.
    flow_strength: f32 = 0.0,
    flow_space: FlowSpace = .world_xz,
    flow_wrap: FlowWrap = .repeat,
    /// UV scale applied per axis (uv = pos.xz * flow_scale + flow_scroll):
    /// 1 world unit covers `flow_scale` texture tiles.
    flow_scale: Vec2 = Vec2.one,
    flow_scroll: Vec2 = Vec2.zero,

    // Visual attributes over lifetime
    color_start: Color4 = Color4.new(1.0, 1.0, 1.0, 1.0),
    color_end: Color4 = Color4.new(1.0, 1.0, 1.0, 0.0),
    size_start: f32 = 0.2,
    size_end: f32 = 0.0,
    lifetime_min: f32 = 1.0,
    lifetime_max: f32 = 2.0,

    // Billboard rotation (degrees) and spin (degrees/second), randomized per particle.
    rotation_min: f32 = 0.0,
    rotation_max: f32 = 0.0,
    angular_velocity_min: f32 = 0.0,
    angular_velocity_max: f32 = 0.0,

    // Spritesheet animation: grid columns x rows, looped `spritesheet_loops`
    // times over each particle's life. 1x1 reproduces the old behavior exactly.
    spritesheet_columns: u32 = 1,
    spritesheet_rows: u32 = 1,
    spritesheet_loops: f32 = 1.0,

    // Local-space simulation: when true, particle positions are stored and
    // integrated in the emitter's local frame; the emitter world matrix is
    // applied only when filling instance data for rendering (see
    // resolveEmitterMatrix/localToWorld). `emitter_position` + box act as a
    // local offset in this mode. Default false = legacy world-space behavior.
    local_space: bool = false,
    emitter_mesh: ?*Mesh = null,

    // --- Sub-emitters (CPU on-death triggers; zero cost when unused) ---
    // Inline fixed storage: addSubEmitter never allocates, per-frame or
    // otherwise. updateCpu only records deaths when count > 0, so systems
    // without sub-emitters keep the legacy path bit-for-bit.
    sub_emitter_store: [max_sub_emitters]SubEmitter = undefined,
    sub_emitter_count: usize = 0,
    /// Seed for the per-death sub-emitter hash stream (probability rolls and
    /// spawn jitter). Fixed default keeps runs reproducible; override per
    /// system to decorrelate sibling systems sharing one child.
    sub_emitter_seed: u64 = 0x9E3779B97F4A7C15,
    /// updateCpu tick counter feeding the per-death hash. Reset by reset().
    sub_tick: u64 = 0,

    // --- Particle collisions (CPU-only: static spheres + ground plane) ---
    // Off by default: with `collision_mode == .none` updateCpu never
    // snapshots collision state (one cached null branch per particle) and
    // follows the legacy path bit-for-bit (see collisions.zig). Inline
    // fixed storage: addSphereCollider never allocates, per-frame or
    // otherwise. Colliders are interpreted in stored simulation coordinates
    // (world units when local_space == false, emitter-local units when
    // true — same rule as the flow field). A `.kill` death compacts away
    // like any age death (may fire sub-emitters; the slot is recycled by
    // normal emission, so a respawn inside a collider dies on contact
    // again).
    collision_mode: CollisionMode = .none,
    collision_spheres: [max_sphere_colliders]ParticleSphereCollider = undefined,
    collision_sphere_count: usize = 0,
    collision_boxes: [max_box_colliders]ParticleBoxCollider = undefined,
    collision_box_count: usize = 0,
    collision_planes: [max_plane_colliders]ParticlePlaneCollider = undefined,
    collision_plane_count: usize = 0,
    /// Normal-speed scale on bounce, clamped to [0, 1] at use (0 = dead
    /// stop, 1 = perfectly elastic). Direct field write, like gravity.
    collision_restitution: f32 = 0.5,
    /// Tangential-velocity fraction KEPT on bounce, clamped to [0, 1] at
    /// use (1 = slick, 0 = full tangential stop — the softbody convention).
    /// Direct field write, like gravity.
    collision_friction: f32 = 1.0,
    /// Ground plane height (particles collide at y == height), or null for
    /// no plane. Prefer setGroundPlane/clearGroundPlane (they validate and
    /// gate non-CPU modes); a direct write skips both.
    collision_ground: ?f32 = null,

    prng: std.Random.DefaultPrng,

    pub fn init(allocator: std.mem.Allocator, name: []const u8, capacity: usize) !*ParticleSystem {
        const ps = try allocator.create(ParticleSystem);
        errdefer allocator.destroy(ps);

        const particles = try allocator.alloc(Particle, capacity);
        errdefer allocator.free(particles);

        const instances = try allocator.alloc(ParticleInstanceData, capacity);
        errdefer allocator.free(instances);

        const alive_scratch = try allocator.alloc(u8, capacity);
        errdefer allocator.free(alive_scratch);

        // CPU-phase ownership is not GPU authorization without a live context.
        const deferred = !sg.isvalid() or !gpu_thread.isOnContextThread();
        const buf = if (deferred) sg.Buffer{} else sg.makeBuffer(.{
            .usage = .{ .vertex_buffer = true, .write_transient = true },
            .size = capacity * @sizeOf(ParticleInstanceData),
        });

        ps.* = .{
            .name = name,
            .allocator = allocator,
            .particles = particles,
            .instances = instances,
            .alive_scratch = alive_scratch,
            .capacity = capacity,
            .active_count = 0,
            .instance_buffer = buf,
            .instance_buffer_pending = deferred,
            .prng = std.Random.DefaultPrng.init(1337),
        };
        return ps;
    }

    pub fn deinit(self: *ParticleSystem) void {
        if (self.instance_buffer.id != 0) {
            sg.destroyBuffer(self.instance_buffer);
            self.instance_buffer = .{};
        }
        if (self.gpu_slot_buffer.id != 0) {
            sg.destroyBuffer(self.gpu_slot_buffer);
            self.gpu_slot_buffer = .{};
        }
        // Context-thread only (same contract as the buffers above): destroys
        // compute views/pipeline/shader (skipped when retired/never created)
        // and compute buffers not previously retired via takeGpuBuffersForRetire.
        self.deinitComputeGpuObjects();
        for ([_]*sg.Buffer{ &self.compute_state_buffer, &self.compute_spawn_buffer, &self.compute_draw_buffer }) |slot| {
            if (slot.*.id != 0) {
                sg.destroyBuffer(slot.*);
                slot.* = .{};
            }
        }
        if (self.compute_staging.len > 0) {
            self.allocator.free(self.compute_staging);
            self.compute_staging = &.{};
        }
        if (self.gpu_slots.len > 0) {
            self.allocator.free(self.gpu_slots);
            self.gpu_slots = &.{};
        }
        if (self.texture) |*t| {
            t.deinit();
            self.texture = null;
        }
        if (self.flow_map) |*t| {
            t.deinit();
            self.flow_map = null;
        }
        if (self.flow_pixels.len > 0) {
            self.allocator.free(self.flow_pixels);
            self.flow_pixels = &.{};
            self.flow_width = 0;
            self.flow_height = 0;
        }
        self.allocator.free(self.particles);
        self.allocator.free(self.instances);
        if (self.alive_scratch.len > 0) self.allocator.free(self.alive_scratch);
    }

    pub fn start(self: *ParticleSystem) void {
        self.is_emitting = true;
    }

    pub fn stop(self: *ParticleSystem) void {
        self.is_emitting = false;
    }

    pub fn reset(self: *ParticleSystem) void {
        self.active_count = 0;
        self.emit_accumulator = 0.0;
        // Sub-emitter hash stream restarts so a reset system behaves like a
        // fresh one; the sub-emitter configuration itself persists (like the
        // emission settings above).
        self.sub_tick = 0;
        // GPU ring: re-anchor the epoch and drop all slots. Old slot records
        // (if the buffer is not cleared) carry spawn times far ahead of the
        // new epoch, so they cull as unborn (t < 0) in the shader.
        self.clock_seconds = 0.0;
        self.gpu_write_cursor = 0;
        self.gpu_high_water = 0;
        self.gpu_dirty = false;
        self.gpu_dirty_wrapped = false;
        // Compute ring: same re-anchor (cursor/high-water/staging/dt); the
        // GPU state buffer is re-zeroed by the next flush (staged flag, so
        // reset stays sg-free like every other update-side mutator).
        self.compute_cursor = 0;
        self.compute_high_water = 0;
        self.compute_stage_base = 0;
        self.compute_staged = 0;
        self.compute_dt_accum = 0.0;
        self.compute_state_clear_pending = true;
    }

    pub fn emitOne(self: *ParticleSystem) void {
        // `.gpu` emits into the spawn-slot ring, `.compute` stages into the
        // compute spawn window; only the integration stage differs from the
        // CPU path (the sampler below is shared verbatim by all three).
        if (self.simulation_mode == .gpu) {
            gpu.emitGpuSlot(self);
            return;
        }
        if (self.simulation_mode == .compute) {
            compute_mode.emitComputeSlot(self);
            return;
        }
        if (self.active_count >= self.capacity) return;
        const sample = sampling.sampleSpawn(self, self.prng.random());

        self.particles[self.active_count] = .{
            .position = sample.position,
            .velocity = sample.velocity,
            .size = self.size_start,
            .size_end = self.size_end,
            .color = self.color_start,
            .color_end = self.color_end,
            .age = 0.0,
            .lifetime = sample.lifetime,
            .rotation = sample.rotation_deg,
            .angular_velocity = sample.angular_velocity,
            .sub_depth = 0,
        };
        self.active_count += 1;
    }

    /// Attaches an on-death child-spawn rule. Fixed inline storage (up to
    /// `max_sub_emitters`); never allocates. Asserts in Debug when full —
    /// size the constant, not the heap, if you need more.
    pub fn addSubEmitter(self: *ParticleSystem, sub: SubEmitter) void {
        std.debug.assert(self.sub_emitter_count < max_sub_emitters);
        self.sub_emitter_store[self.sub_emitter_count] = sub;
        self.sub_emitter_count += 1;
    }

    /// Attached sub-emitter rules (empty when unused).
    pub fn subEmitters(self: *const ParticleSystem) []const SubEmitter {
        return self.sub_emitter_store[0..self.sub_emitter_count];
    }

    /// Arms the flow field: takes ownership of `texture` and keeps an owned
    /// CPU-side copy of `pixels` (see flow.zig).
    pub fn setFlowMap(
        self: *ParticleSystem,
        texture: ?Texture,
        pixels: []const u8,
        width: u32,
        height: u32,
    ) !void {
        return flow.setFlowMap(self, texture, pixels, width, height);
    }

    /// Disarms the field (see flow.zig).
    pub fn clearFlowMap(self: *ParticleSystem) void {
        flow.clearFlowMap(self);
    }

    /// Flow acceleration at `pos` (see flow.zig).
    pub fn sampleFlow(self: *const ParticleSystem, pos: Vec3) Vec3 {
        return flow.sampleFlow(self, pos);
    }

    /// Adds a static sphere collider (see collisions.zig). Fixed inline
    /// storage; explicit errors, never a silent downgrade.
    pub fn addSphereCollider(self: *ParticleSystem, collider: ParticleSphereCollider) CollisionError!void {
        return collisions.addSphereCollider(self, collider);
    }

    /// Adds a static axis-aligned box collider (see collisions.zig).
    pub fn addBoxCollider(self: *ParticleSystem, collider: ParticleBoxCollider) CollisionError!void {
        return collisions.addBoxCollider(self, collider);
    }

    /// Adds a static oriented plane collider (see collisions.zig).
    pub fn addPlaneCollider(self: *ParticleSystem, collider: ParticlePlaneCollider) CollisionError!void {
        return collisions.addPlaneCollider(self, collider);
    }

    /// Adds a box collider matching the world bounding box of `mesh_obj`.
    pub fn addMeshAabbCollider(self: *ParticleSystem, mesh_obj: anytype) CollisionError!void {
        return collisions.addMeshAabbCollider(self, mesh_obj);
    }

    /// Disarms sphere colliders.
    pub fn clearSphereColliders(self: *ParticleSystem) void {
        collisions.clearSphereColliders(self);
    }

    /// Disarms box colliders.
    pub fn clearBoxColliders(self: *ParticleSystem) void {
        collisions.clearBoxColliders(self);
    }

    /// Disarms plane colliders.
    pub fn clearPlaneColliders(self: *ParticleSystem) void {
        collisions.clearPlaneColliders(self);
    }

    /// Disarms all collision geometry (spheres + boxes + planes + ground plane); response
    /// knobs kept. Never fails (see collisions.zig).
    pub fn clearColliders(self: *ParticleSystem) void {
        collisions.clearColliders(self);
    }

    /// Selects the collision response (see collisions.zig). Enabling on a
    /// non-CPU system is an explicit error; `.none` always succeeds.
    pub fn setCollisionMode(self: *ParticleSystem, mode: CollisionMode) CollisionError!void {
        return collisions.setCollisionMode(self, mode);
    }

    /// Arms the ground plane at `height` (see collisions.zig). Validates
    /// before mutating; non-CPU systems get an explicit error.
    pub fn setGroundPlane(self: *ParticleSystem, height: f32) CollisionError!void {
        return collisions.setGroundPlane(self, height);
    }

    /// Disarms the ground plane (other colliders kept). Never fails.
    pub fn clearGroundPlane(self: *ParticleSystem) void {
        collisions.clearGroundPlane(self);
    }

    pub fn burst(self: *ParticleSystem, count: usize) void {
        var n: usize = 0;
        while (n < count) : (n += 1) {
            // CPU keeps the old drop-when-full guard; the GPU rings always
            // accept (oldest slot is recycled).
            if (self.simulation_mode == .cpu and self.active_count >= self.capacity) break;
            self.emitOne();
        }
    }

    /// Emitter world matrix for local-space rendering, or null when world-space
    /// rendering applies (local_space == false or no emitter_mesh bound).
    /// With local_space == true but emitter_mesh == null the local coordinates
    /// pass through unchanged (identity).
    pub fn resolveEmitterMatrix(self: *const ParticleSystem) ?Mat4 {
        if (!self.local_space) return null;
        const mesh = self.emitter_mesh orelse return null;
        return mesh.getWorldMatrix();
    }

    /// CPU simulation step: three-phase parallel integrate / serial compact /
    /// parallel instance fill (see cpu.zig).
    pub fn updateCpu(self: *ParticleSystem, dt: f32) void {
        cpu.updateCpu(self, dt);
    }

    /// GPU-path frame step: advances the epoch clock and appends spawn slots
    /// to the ring — O(emitted), never O(particles) (see gpu.zig).
    pub fn updateGpu(self: *ParticleSystem, dt: f32) UpdateError!void {
        return gpu.updateGpu(self, dt);
    }

    /// Mode-selecting API for `.compute`: hard contract, never a silent
    /// fallback (see compute_mode.zig).
    pub fn setSimulationMode(self: *ParticleSystem, mode: SimulationMode) ComputeModeError!void {
        return compute_mode.setSimulationMode(self, mode);
    }

    /// Whether `.compute` updates may proceed (see compute_mode.zig).
    pub fn computeAvailable(self: *const ParticleSystem) bool {
        return compute_mode.computeAvailable(self);
    }

    /// Compute dispatches issued since creation (see compute_mode.zig).
    pub fn computeDispatchCount(self: *const ParticleSystem) u64 {
        return compute_mode.computeDispatchCount(self);
    }

    /// Compute-path frame step: stages spawn slots + frame dt (see
    /// compute_mode.zig).
    pub fn updateCompute(self: *ParticleSystem, dt: f32) UpdateError!void {
        return try compute_mode.updateCompute(self, dt);
    }

    /// Dispatch workgroup count covering the written-slot prefix (see
    /// compute_mode.zig).
    pub fn computeGroups(self: *const ParticleSystem) usize {
        return compute_mode.computeGroups(self);
    }

    /// Shader parameters for the next dispatch (see compute_mode.zig).
    pub fn buildComputeParams(self: *const ParticleSystem) pc_shd.CsParams {
        return compute_mode.buildComputeParams(self);
    }

    /// Destroys compute views/pipeline/shader (see compute_mode.zig).
    pub fn deinitComputeGpuObjects(self: *ParticleSystem) void {
        compute_mode.deinitComputeGpuObjects(self);
    }

    /// Collects live GPU buffers for retire-queue teardown (see
    /// compute_mode.zig).
    pub fn takeGpuBuffersForRetire(self: *ParticleSystem, out: []sg.Buffer) usize {
        return compute_mode.takeGpuBuffersForRetire(self, out);
    }

    /// Explicit mode dispatch: the CPU path always runs, `.gpu`/`.compute`
    /// run or return an error (`UpdateError`) — never a silent downgrade.
    pub fn update(self: *ParticleSystem, dt: f32) UpdateError!void {
        if (self.simulation_mode == .gpu) {
            // First use only stages the creation flag (sg.makeBuffer is a
            // context-thread call); flushGpuUploads creates the buffer.
            if (self.gpu_slot_buffer.id == 0) {
                self.gpu_slot_buffer_pending = true;
            }
            try self.updateGpu(dt);
            self.gpu_flush_pending = true;
            return;
        }
        // `.compute` mirrors the `.gpu` staging above (creation flag +
        // update + flush flag); the CPU particle array is never touched.
        if (self.simulation_mode == .compute) {
            if (self.compute_state_buffer.id == 0) {
                self.compute_buffers_pending = true;
            }
            try self.updateCompute(dt);
            self.compute_flush_pending = true;
            return;
        }
        self.updateCpu(dt);

        if (self.active_count > 0) {
            self.instance_dirty = true;
        }
    }

    /// Uploads staged instance data. Runs on the sg-context thread
    /// (Scene.render start) — the update phase only stages CPU data and
    /// sets the dirty flags, so simulation stays free of sg.* calls.
    pub fn flushGpuUploads(self: *ParticleSystem) void {
        // Deferred construction (off-context spawn): create the instance
        // buffer here, on the render side, and upload any staged data.
        if (self.instance_buffer_pending and sg.isvalid()) {
            self.instance_buffer = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .write_transient = true },
                .size = self.capacity * @sizeOf(ParticleInstanceData),
            });
            if (self.instance_buffer.id != 0) {
                self.instance_buffer_pending = false;
                if (self.active_count > 0) self.instance_dirty = true;
            }
        }
        if (self.instance_dirty) {
            self.instance_dirty = false;
            if (self.active_count > 0 and self.instance_buffer.id != 0) {
                sg.writeBufferTransient(.{
                    .dst = .{ .buffer = self.instance_buffer },
                    .src = .{ .data = sg.asRange(self.instances[0..self.active_count]) },
                });
                // Dynamic upload tracking: all active_count instances.
                upload_meter.record(self.active_count * @sizeOf(ParticleInstanceData));
            }
        }
        // Deferred first-use creation (staged by update): without a valid sg
        // context the flag stays set and a later flush retries.
        if (self.gpu_slot_buffer_pending and self.gpu_slot_buffer.id == 0 and sg.isvalid()) {
            self.gpu_slot_buffer = sg.makeBuffer(.{
                .usage = .{ .vertex_buffer = true, .write_transient = true },
                .size = self.capacity * @sizeOf(GpuParticleSlot),
            });
            if (self.gpu_slot_buffer.id != 0) self.gpu_slot_buffer_pending = false;
        }
        if (self.gpu_flush_pending) {
            self.gpu_flush_pending = false;
            gpu.flushGpuUpload(self);
        }
        // Stateful compute path (prepare boundary, context thread): creation,
        // spawn upload, state clear and dispatch. No-op unless the system runs
        // `.compute` (or holds staged compute work); headless-safe.
        if (self.simulation_mode == .compute) {
            compute_mode.flushComputeUploads(self);
        }
    }
};

// --- Shared test helpers (headless; pub for sibling-leaf test blocks) ---

/// Test-only constructors live here — next to their owner — because only
/// this module may name `ParticleSystem` at top level without an import
/// cycle (leaves take the system as `anytype`). Reached from moved tests
/// through block-scoped imports that exist only in test builds. Never
/// re-exported from the `particles.zig` facade.
pub fn makeTestSystem(allocator: std.mem.Allocator, capacity: usize) !ParticleSystem {
    const parts = try allocator.alloc(Particle, capacity);
    errdefer allocator.free(parts);
    const insts = try allocator.alloc(ParticleInstanceData, capacity);
    errdefer allocator.free(insts);
    const scratch = try allocator.alloc(u8, capacity);
    errdefer allocator.free(scratch);
    return ParticleSystem{
        .name = "test",
        .allocator = allocator,
        .particles = parts,
        .instances = insts,
        .alive_scratch = scratch,
        .capacity = capacity,
        .instance_buffer = .{},
        .prng = std.Random.DefaultPrng.init(42),
    };
}

pub fn freeTestSystem(ps: *ParticleSystem) void {
    if (ps.gpu_slots.len > 0) ps.allocator.free(ps.gpu_slots);
    if (ps.compute_staging.len > 0) ps.allocator.free(ps.compute_staging);
    if (ps.flow_pixels.len > 0) ps.allocator.free(ps.flow_pixels);
    ps.allocator.free(ps.particles);
    ps.allocator.free(ps.instances);
    if (ps.alive_scratch.len > 0) ps.allocator.free(ps.alive_scratch);
}

/// Quiescent test system: no emission, no gravity, zero sampled velocity, so
/// tests control the live set exactly (manual placement or burst) and deaths
/// come only from aging past a fixed lifetime.
pub fn makeQuiescentSystem(allocator: std.mem.Allocator, capacity: usize) !ParticleSystem {
    var ps = try makeTestSystem(allocator, capacity);
    ps.is_emitting = false;
    ps.emit_rate = 0.0;
    ps.gravity = Vec3.zero;
    ps.direction_min = Vec3.zero;
    ps.direction_max = Vec3.zero;
    ps.speed_min = 0.0;
    ps.speed_max = 0.0;
    ps.lifetime_min = 0.5;
    ps.lifetime_max = 0.5;
    return ps;
}

/// Compute-capable test system: emission on, deterministic sampler.
pub fn makeComputeSystem(allocator: std.mem.Allocator, capacity: usize) !ParticleSystem {
    var ps = try makeTestSystem(allocator, capacity);
    ps.simulation_mode = .compute;
    ps.is_emitting = true;
    ps.emit_rate = 60.0;
    ps.lifetime_min = 1.0;
    ps.lifetime_max = 1.0;
    return ps;
}

/// Places one known particle, bypassing the PRNG emission stream.
pub fn placeTestParticle(ps: *ParticleSystem, pos: Vec3, vel: Vec3) void {
    ps.particles[0] = .{
        .position = pos,
        .velocity = vel,
        .size = 0.2,
        .size_end = 0.0,
        .color = Color4.new(1.0, 1.0, 1.0, 1.0),
        .color_end = Color4.new(1.0, 1.0, 1.0, 0.0),
        .age = 0.0,
        .lifetime = 10.0,
    };
    ps.active_count = 1;
}
