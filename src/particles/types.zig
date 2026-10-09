//! Shared particle vocabulary: modes, errors, slot/state layouts and pure
//! trajectory math. Split out of `particles.zig` (facade).
//!
//! Leaf: no sibling imports (only `math`, plus `../compute.zig` for the
//! dispatch workgroup-size contract). Everything here is mode-independent:
//! the `.cpu`/`.gpu`/`.compute` contracts live on `SimulationMode` below,
//! and the `ParticleSystem` owner lives in `system.zig`. `particles.zig`
//! re-exports every public declaration unchanged.

const std = @import("std");
const math = @import("math");
const compute = @import("../compute.zig");

const Vec2 = math.Vec2;
const Vec3 = math.Vec3;
const Vec4 = math.Vec4;
const Color4 = math.Color4;

pub const ParticleBlendMode = enum {
    additive,
    alpha_blend,
};

/// Coordinate frame the flow-field texture (`ParticleSystem.flow_map`) is
/// sampled in. Only the XZ plane is used (y is never affected by the flow):
/// - `.world_xz`: uv = position.xz * flow_scale + flow_scroll.
/// - `.local_xz`: uv = (position - emitter_position).xz * flow_scale +
///   flow_scroll, i.e. the field travels with the emitter origin.
pub const FlowSpace = enum {
    world_xz,
    local_xz,
};

/// Out-of-bounds UV policy for flow-field sampling (after scale/scroll).
/// `.repeat` wraps with fract (negative-safe); `.clamp` clamps to [0, 1]
/// (edge texel stretches to infinity).
pub const FlowWrap = enum {
    repeat,
    clamp,
};

/// Simulation driver of a ParticleSystem. Default `.cpu` keeps every existing
/// system bit-for-bit identical; `.gpu` moves the integration into the vertex
/// shader (stateless: each particle is a fixed slot holding only spawn data,
/// position/fade/size are evaluated analytically from the age of the slot);
/// `.compute` (v1, OFF by default) integrates per-particle state on the GPU
/// in a compute pass (stateful: position/velocity/age persist in a storage
/// buffer across frames).
///
/// Feature matrix — the `.cpu`/`.gpu` table below is frozen: adding
/// `.compute` changed no `.cpu`/`.gpu` line, branch, or default. The
/// `.compute` contract lives in its own table further down (search
/// "COMPUTE MODE v1") so the two can never be silently merged.
///
/// | Feature                              | .cpu      | .gpu                         |
/// |--------------------------------------|-----------|------------------------------|
/// | gravity                              | yes       | yes                          |
/// | exponential drag (`drag`)            | no (1)    | yes                          |
/// | lifetime / burst / emit_rate         | yes       | yes                          |
/// | color & size start->end lerp         | yes       | yes                          |
/// | spritesheet grid + loops             | yes       | yes                          |
/// | rotation + angular velocity          | yes       | yes                          |
/// | additive / alpha blend               | yes       | yes                          |
/// | world-space emitter                  | yes       | yes                          |
/// | flow-field texture (`flow_map`)        | yes       | error.FlowMapNeedsCpu (5)    |
/// | sub-emitters (on-death spawn)        | yes       | parent never fires (4)       |
/// | local_space (moving emitter frame)   | yes       | error.LocalSpaceNeedsCpu (2) |
/// | collisions (spheres + ground plane)  | yes (6)   | error.CollisionNeedsCpu (6)  |
/// | noise, arbitrary forces              | (3)       | error (3)                    |
///
/// (1) `drag` is GPU-only by design: the CPU path integrates semi-implicit
///     Euler per frame while the GPU path uses the exact exponential form;
///     implementing drag on only one side keeps the two integration schemes
///     from being silently mixed in one system.
/// (2) local_space needs the per-particle emitter-transform history; only the
///     CPU path can express it.
/// (3) Not implemented on any path today (noise, arbitrary force fields).
///     Particle collisions (static spheres + ground plane, see
///     `collisions.zig`) ARE implemented on the CPU path and reject `.gpu`
///     with error.CollisionNeedsCpu when armed — never a silent downgrade.
/// (4) Deaths happen in the vertex shader, unobservable on CPU, so a `.gpu`
///     parent never fires its sub-emitters (silently not firing is correct
///     here: erroring would break the steady-state emission the ring was
///     built for). A `.gpu` child still accepts spawns into its slot ring;
///     being stateless it tracks no chain depth, so chains stop there.
/// (5) Flow sampling needs the CPU pixel copy plus per-particle UV mapping;
///     the GPU path has no flow-texture binding, so an armed flow field
///     (strength != 0 with a valid CPU copy) rejects `.gpu` with
///     error.FlowMapNeedsCpu — never a silent downgrade. An unset/empty/
///     zero-strength field costs nothing on either path and never errors.
/// (6) Static sphere colliders (fixed cap 8, `addSphereCollider`) plus an
///     optional ground plane (`setGroundPlane`), with `.kill`/`.bounce`
///     response (`setCollisionMode`, restitution + friction knobs).
///     Default `.none` is bit-identical to the pre-collision path;
///     armed collisions reject `.gpu`/`.compute` with
///     error.CollisionNeedsCpu — never a silent downgrade. Colliders live
///     in stored simulation coordinates (world, or emitter-local when
///     local_space).
///
/// Integration semantics differ by construction: `.cpu` advances with the
/// frame dt (semi-implicit Euler), `.gpu` evaluates the exact closed form
/// p(t) = p0 + v0*s + g*(t - s)/k with s = (1 - e^(-k*t)) / k (k = drag, and
/// s = t, (t - s)/k = t^2/2 when k = 0). The analytic form is pinned by
/// golden values in `analyticPosition` tests.
///
/// COMPUTE MODE v1 (`.compute`) — stateful GPU simulation, additive and OFF
/// by default. Chosen shape: a third `SimulationMode` variant on the existing
/// `ParticleSystem` (not a separate type), because the CPU-side spawn
/// semantics — `sampleSpawn` PRNG order, emitter box/direction/speed ranges,
/// lifetime/rotation sampling, burst/emit_rate accumulation — are shared
/// verbatim with `.cpu`/`.gpu` (identical seeds produce identical spawn
/// records in any mode), and the visual parameters (color gradient, size
/// curve, spritesheet, blend mode, texture) are reused unchanged: the compute
/// pass bakes them into `ParticleInstanceData` records, so the render path
/// binds the compute-written draw buffer through the EXISTING cpu billboard
/// pipeline bit-for-bit (same blending, same soft-particle behavior, same
/// texture plumbing; no new render pipeline or shader).
///
/// State layout (mirrors `CState` in shaders/particle_compute.glsl):
/// per-particle `ComputeParticleState` (6 x vec4 = 96 bytes: pos+age,
/// vel+life, rot0+angvel+seed, color start/end, size start/end) in ONE
/// storage buffer updated IN PLACE — no ping-pong: each dispatch invocation
/// touches only its own index, so no barrier is needed. Spawns staged by the
/// CPU update reuse the `GpuParticleSlot` record (80 bytes) in a CPU staging
/// array, uploaded at the prepare boundary into a spawn storage buffer, and
/// claimed by the shader ring window [base, base+count) mod capacity.
///
/// | Feature                              | .compute                                                    |
/// |--------------------------------------|---------------------------------------------------------------|
/// | gravity                              | yes (semi-implicit Euler per dispatch)                       |
/// | exponential drag (`drag`)            | yes (per-step exp damping; NOT the .gpu analytic form)       |
/// | lifetime / burst / emit_rate         | yes (CPU-staged, shared sampler)                              |
/// | color & size start->end lerp         | yes (baked per slot at respawn; mid-life edits hit new only) |
/// | spritesheet grid + loops             | yes (frame baked by compute from system uniforms)             |
/// | rotation + angular velocity          | yes (exact rot0 + w*age, baked)                               |
/// | additive / alpha blend               | yes (existing cpu pipeline)                                   |
/// | world-space emitter                  | yes                                                           |
/// | flow-field texture (`flow_map`)      | error.FlowMapNeedsCpu (same rule as .gpu)                     |
/// | sub-emitters (on-death spawn)        | parent never fires (deaths are GPU-side, unobservable)        |
/// | local_space (moving emitter frame)   | error.LocalSpaceNeedsCpu (same rule as .gpu)                  |
/// | collisions (+ noise, sorting)        | error.CollisionNeedsCpu when armed (CPU-only);                |
/// |                                      | noise/sorting not implemented                                 |
///
/// Space recycling: a wrapping CPU cursor overwrites the oldest slot (ring
/// semantics, same as `.gpu`). Consequence: no per-particle death events are
/// observable on CPU, a `.compute` parent never fires sub-emitters (a
/// `.compute` child still accepts spawns via `emitChild`), there is no
/// deterministic CPU replay (integration lives on the GPU), and over-capacity
/// emission recycles live slots instead of dropping.
///
/// Fallback contract: selecting `.compute` is a hard contract, never a silent
/// fallback. `setSimulationMode(.compute)` returns `error.ComputeUnsupported`
/// when the backend is known-unsupported (live context without compute, or a
/// test/app override), and `error.InvalidCapacity` for a zero-capacity
/// system. Headless selection (no sg context yet) stages the mode; the first
/// context-side flush latches support authoritatively, and the next `update`
/// returns `error.ComputeUnsupported` when the backend lacks compute. The
/// mode field is never mutated behind the caller's back.
///
/// Upload discipline: the update phase only stages CPU bytes and sets flags
/// (never sg.*); buffer/view/pipeline/shader creation, spawn uploads and the
/// compute dispatch all happen in `flushGpuUploads` on the context thread
/// (prepare boundary). Teardown: buffers retire through `GpuRetireQueue`
/// (see `takeGpuBuffersForRetire`); views/pipeline/shader destroy on the
/// context thread (`deinitComputeGpuObjects`, called by `deinit`).
///
/// v1 NON-GOALS (explicit): sub-emitters fired from compute deaths, flow
/// maps, particle sorting, collisions, deterministic CPU replay, multi-view /
/// PIP correctness beyond what the normal particle pass already does, and
/// sharing one compute pipeline across systems (each compute system owns its
/// pipeline+shader; created once, destroyed in `deinit`).
pub const SimulationMode = enum {
    cpu,
    gpu,
    compute,
};

/// Errors surfaced by `update`/`updateGpu`/`updateCompute`. A requested GPU
/// simulation mode is a hard contract: when the configuration cannot express
/// it, these errors propagate instead of a silent CPU downgrade. Fix the
/// configuration (features/mode).
pub const UpdateError = error{
    /// `local_space` emitters need per-particle history; only the CPU path
    /// can express them (see the feature matrix above).
    LocalSpaceNeedsCpu,
    /// An armed flow-field texture needs CPU sampling; the GPU paths have no
    /// flow-texture binding (see the feature matrix note (5)).
    FlowMapNeedsCpu,
    /// Armed particle collisions (a non-`.none` mode with sphere/ground
    /// geometry) need the CPU integrator; the GPU paths carry no collider
    /// state (see the feature matrix note (6)).
    CollisionNeedsCpu,
    /// First-frame GPU slot allocation failed; the GPU ring cannot run.
    OutOfMemory,
    /// `.compute` selected on a backend without compute support (latched by
    /// the context-side flush, or forced by the support override). Never a
    /// silent fallback to `.cpu`/`.gpu`.
    ComputeUnsupported,
    /// `.compute` selected on a zero-capacity system (no slots to simulate).
    InvalidCapacity,
};

/// Errors surfaced by the mode-selecting API `setSimulationMode` (subset of
/// `UpdateError` relevant at selection time).
pub const ComputeModeError = error{
    ComputeUnsupported,
    InvalidCapacity,
};

/// One stateful compute particle: integrated GPU state consumed by the
/// compute program `particle_compute` in shaders/particle_compute.glsl.
/// Layout mirrors the `CState` struct there (six vec4, 6 * 16 = 96 bytes);
/// vec4-only fields keep std430 offsets trivially 16-aligned.
pub const ComputeParticleState = extern struct {
    /// xyz = position, w = age in seconds.
    pos_age: [4]f32,
    /// xyz = velocity, w = lifetime in seconds (<= 0 = never written = dead).
    vel_life: [4]f32,
    /// x = rotation start (radians), y = angular velocity (radians/second),
    /// z = seed, w unused.
    rot_seed: [4]f32,
    /// Baked visual endpoints (persist per slot so mid-life parameter edits
    /// affect only later spawns — same rule as the `.gpu` slot ring).
    color_start: [4]f32,
    color_end: [4]f32,
    /// x = size start, y = size end, zw unused.
    size_size: [4]f32,
};

comptime {
    if (@sizeOf(ComputeParticleState) != 6 * 16) @compileError("ComputeParticleState layout drifted from particle_compute.glsl");
}

/// Dispatch workgroup width. Must equal `layout(local_size_x = ...) in;` in
/// shaders/particle_compute.glsl (pinned by the test below, mirroring the
/// `compute.default_workgroup_size` contract).
pub const compute_workgroup_size: usize = 64;

comptime {
    if (compute_workgroup_size != compute.default_workgroup_size) @compileError("compute_workgroup_size drifted from compute.default_workgroup_size");
}

/// One GPU particle slot: fixed-size spawn record consumed by the vertex
/// shader (program `particle_gpu` in shaders/particle.glsl). Layout mirrors
/// the five FLOAT4 instance attributes declared there; 5 * 16 = 80 bytes.
pub const GpuParticleSlot = extern struct {
    /// xyz = spawn position (world space), w = spawn time in seconds since
    /// the system's epoch clock (`clock_seconds`). Epoch-relative times keep
    /// float32 magnitudes small: f32 has a 24-bit mantissa, so the spawn-time
    /// resolution degrades to ~0.00024 s after one hour of session time;
    /// reset() re-anchors the clock for long-lived systems.
    spawn_pos_time: [4]f32,
    /// xyz = initial velocity, w = lifetime in seconds (>= 0.0001).
    velocity_lifetime: [4]f32,
    color_start: [4]f32,
    color_end: [4]f32,
    /// x = size start, y = size end, z = rotation start (radians),
    /// w = angular velocity (radians/second).
    size_rotation: [4]f32,
};

comptime {
    // Vertex-shader ABI (particle_gpu program, five FLOAT4 attributes).
    // Compile-time guarantee instead of a runtime test that only some
    // build configurations ever run.
    if (@sizeOf(GpuParticleSlot) != 5 * 16) @compileError("GpuParticleSlot layout drifted from particle.glsl");
}

/// Age of a slot relative to the render clock. `alive` is false for unborn
/// (spawn in the future) and dead (t >= 1) slots; the shader collapses those
/// into a degenerate off-screen triangle.
pub const SlotAge = struct {
    t: f32,
    alive: bool,
};

/// Normalized age of a particle slot. Mirrors the shader expression
/// `t = (time - spawn_time) / max(lifetime, 1.0e-4)` so CPU tests pin the
/// exact branching the GLSL performs (unborn -> t=0/dead, dead -> t clamped 1).
pub fn slotAge(now: f32, spawn_time: f32, lifetime: f32) SlotAge {
    const life: f32 = if (lifetime > 0.0001) lifetime else 0.0001;
    const age = now - spawn_time;
    if (age < 0.0) return .{ .t = 0.0, .alive = false };
    const t = age / life;
    if (t >= 1.0) return .{ .t = 1.0, .alive = false };
    return .{ .t = t, .alive = true };
}

/// Integration spans shared by the Zig and GLSL analytic trajectories:
/// position = spawn + velocity * s + gravity * s2. With drag k > 0 the
/// velocity ODE dv/dt = g - k*v has the closed form v(t) = v0*e^(-kt) +
/// (g/k)*(1 - e^(-kt)); integrating gives s = (1 - e^(-kt)) / k and
/// s2 = (t - s) / k. For k = 0 these degenerate to s = t, s2 = t^2 / 2, i.e.
/// the classic p = p0 + v0*t + 0.5*g*t^2. The k threshold (1e-6) avoids the
/// catastrophic cancellation of (1 - e^(-kt))/k for near-zero drag.
pub const DragSpans = struct { s: f32, s2: f32 };

pub fn analyticDragSpans(drag: f32, time: f32) DragSpans {
    if (drag > 1.0e-6) {
        const s = (1.0 - @exp(-drag * time)) / drag;
        return .{ .s = s, .s2 = (time - s) / drag };
    }
    return .{ .s = time, .s2 = 0.5 * time * time };
}

/// Analytic particle position after `time` seconds. GLSL duplicate lives in
/// shaders/particle.glsl (program particle_gpu); both use exp(), so expect
/// ~1e-6 relative deviation between backends (libm expf vs GPU intrinsic) —
/// the golden tests below pin the Zig side to 1e-4 absolute tolerance.
pub fn analyticPosition(spawn: Vec3, velocity: Vec3, gravity: Vec3, drag: f32, time: f32) Vec3 {
    const spans = analyticDragSpans(drag, time);
    return spawn.add(velocity.scale(spans.s)).add(gravity.scale(spans.s2));
}

pub const Particle = struct {
    position: Vec3,
    velocity: Vec3,
    size: f32,
    size_end: f32,
    color: Color4,
    color_end: Color4,
    age: f32,
    lifetime: f32,
    /// Z-rotation of the billboard in degrees, normalized to [0, 360).
    rotation: f32 = 0.0,
    /// Spin speed in degrees per second (integrated into rotation by update).
    angular_velocity: f32 = 0.0,
    /// Sub-emitter chain generation: 0 for normally emitted particles, set to
    /// parent + 1 for sub-emitter spawns. Deaths at
    /// `sub_depth >= max_sub_emitter_depth` spawn no children, which bounds
    /// cyclic graphs (A→B→A, self-emitters). Always 0 on systems without
    /// sub-emitters.
    sub_depth: u8 = 0,
};

pub const ParticleInstanceData = extern struct {
    pos_size: [4]f32,
    color: [4]f32,
    /// Spritesheet sub-rect: xy = UV offset, zw = UV scale (1/columns, 1/rows).
    /// Default (0,0,1,1) reproduces the old full-texture sampling bit-for-bit.
    uv_offset_scale: [4]f32 = .{ 0.0, 0.0, 1.0, 1.0 },
    /// x = billboard rotation in radians, yzw reserved (must stay 0).
    rotation_misc: [4]f32 = .{ 0.0, 0.0, 0.0, 0.0 },
};

/// Maximum sub-emitters per system. Fixed inline storage: attaching them
/// allocates nothing, per-frame or otherwise.
pub const max_sub_emitters: usize = 4;

/// Chain rule: a particle with `sub_depth >= max_sub_emitter_depth` spawns no
/// children on death. Every sub-emitter spawn sets
/// child.sub_depth = parent.sub_depth + 1, so a cyclic graph (A→B→A, or a
/// self-emitter) fires at most this many generations and cannot recurse
/// infinitely. Chains additionally stop at GPU-mode systems: a `.gpu` parent
/// never fires (deaths are shader-side and unobservable on CPU) and a `.gpu`
/// child accepts spawns but tracks no depth.
pub const max_sub_emitter_depth: u8 = 4;

/// Explosion bound: at most this many child particles are spawned from one
/// parent system in a single updateCpu tick, across all its sub-emitters.
/// Deaths beyond the recorded prefix still compact normally, they just do not
/// spawn.
pub const max_sub_emitter_spawns_per_tick: usize = 256;
