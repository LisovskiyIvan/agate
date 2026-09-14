const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Vec3 = math.Vec3;
const Vec4 = math.Vec4;
const Color4 = math.Color4;
const Mat4 = math.Mat4;
const Texture = @import("texture.zig").Texture;
const Mesh = @import("mesh.zig").Mesh;
const jobs = @import("jobs.zig");

pub const ParticleBlendMode = enum {
    additive,
    alpha_blend,
};

/// Simulation driver of a ParticleSystem. Default `.cpu` keeps every existing
/// system bit-for-bit identical; `.gpu` moves the integration into the vertex
/// shader (stateless: each particle is a fixed slot holding only spawn data,
/// position/fade/size are evaluated analytically from the age of the slot).
///
/// Feature matrix — `.gpu` can only express what its stateless state model
/// supports (closed-form function of age). Requesting an unexpressible
/// combination is an explicit `UpdateError` from `update`; the engine never
/// silently downgrades a requested GPU mode to the CPU path. Stateful
/// compute-simulated particles are NOT part of this API: build them on the
/// compute.zig primitives (supported/makePipeline/makeStorageView/groupCount)
/// over your own storage buffers.
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
/// | local_space (moving emitter frame)   | yes       | error.LocalSpaceNeedsCpu (2) |
/// | collisions, noise, arbitrary forces  | (3)       | error (3)                    |
///
/// (1) `drag` is GPU-only by design: the CPU path integrates semi-implicit
///     Euler per frame while the GPU path uses the exact exponential form;
///     implementing drag on only one side keeps the two integration schemes
///     from being silently mixed in one system.
/// (2) local_space needs the per-particle emitter-transform history; only the
///     CPU path can express it.
/// (3) Not implemented on any path today; features that need per-particle
///     historical state beyond the live record (collisions, force fields)
///     must stay CPU-only until the state layout grows. If such a flag is
///     ever added it must reject `.gpu` in update() with an error — never
///     downgrade.
///
/// Integration semantics differ by construction: `.cpu` advances with the
/// frame dt (semi-implicit Euler), `.gpu` evaluates the exact closed form
/// p(t) = p0 + v0*s + g*(t - s)/k with s = (1 - e^(-k*t)) / k (k = drag, and
/// s = t, (t - s)/k = t^2/2 when k = 0). The analytic form is pinned by
/// golden values in `analyticPosition` tests.
pub const SimulationMode = enum {
    cpu,
    gpu,
};

/// Errors surfaced by `update`/`updateGpu`. A requested GPU simulation mode
/// is a hard contract: when the configuration cannot express it, these errors
/// propagate instead of a silent CPU downgrade. Fix the configuration
/// (features/mode).
pub const UpdateError = error{
    /// `local_space` emitters need per-particle history; only the CPU path
    /// can express them (see the feature matrix above).
    LocalSpaceNeedsCpu,
    /// First-frame GPU slot allocation failed; the GPU ring cannot run.
    OutOfMemory,
};

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

/// Wraps a degree angle into [0, 360).
pub fn normalizeAngleDeg(angle: f32) f32 {
    var a = @mod(angle, 360.0);
    if (a < 0.0) a += 360.0;
    // @mod of negative multiples (e.g. -360) already yields +0.0; guard -0.0.
    if (a == 0.0) return 0.0;
    return a;
}

/// Degrees -> radians for the instance rotation field.
pub fn rotationToRadians(angle_deg: f32) f32 {
    return angle_deg * std.math.pi / 180.0;
}

/// Guards a zero grid dimension to 1 (avoids div-by-zero; 0 means "no sheet").
inline fn sanitizedGrid(columns: u32, rows: u32) struct { cols: u32, rows: u32 } {
    return .{
        .cols = if (columns == 0) 1 else columns,
        .rows = if (rows == 0) 1 else rows,
    };
}

/// Total frame count of a columns x rows sheet.
pub fn spritesheetFrameCount(columns: u32, rows: u32) u32 {
    const g = sanitizedGrid(columns, rows);
    return g.cols * g.rows;
}

/// Frame index for a particle age: floor(age_norm * loops * frames) % frames,
/// with age_norm = clamp(age / lifetime, 0, 1). At exactly age == lifetime the
/// index wraps to 0 when loops * frames is integral (particle dies there anyway).
pub fn spritesheetFrameForAge(age: f32, lifetime: f32, columns: u32, rows: u32, loops: f32) u32 {
    const frames = spritesheetFrameCount(columns, rows);
    if (frames <= 1) return 0;
    if (!(lifetime > 0.0)) return 0;
    const age_norm = std.math.clamp(age / lifetime, 0.0, 1.0);
    const pos = @floor(age_norm * loops * @as(f32, @floatFromInt(frames)));
    const wrapped = @mod(pos, @as(f32, @floatFromInt(frames)));
    return @intFromFloat(wrapped);
}

/// UV sub-rect for a frame as [offset_u, offset_v, scale_u, scale_v].
/// Frames run left-to-right, bottom-to-top in UV space (frame 0 = UV origin cell).
pub fn spritesheetUvRect(frame: u32, columns: u32, rows: u32) [4]f32 {
    const g = sanitizedGrid(columns, rows);
    const frames = g.cols * g.rows;
    const f = if (frames > 0) frame % frames else 0;
    const col = f % g.cols;
    const row = (f / g.cols) % g.rows;
    const su: f32 = 1.0 / @as(f32, @floatFromInt(g.cols));
    const sv: f32 = 1.0 / @as(f32, @floatFromInt(g.rows));
    return .{
        @as(f32, @floatFromInt(col)) * su,
        @as(f32, @floatFromInt(row)) * sv,
        su,
        sv,
    };
}

/// Transforms a local-space point to world space with an emitter matrix.
pub fn localToWorld(matrix: Mat4, point: Vec3) Vec3 {
    return matrix.transformPoint(point);
}

/// Approximate uniform scale of a TRS matrix: mean length of the basis columns.
/// Exact for uniform scales (factor 1 for identity); heuristic for non-uniform.
pub fn worldScaleFactor(matrix: Mat4) f32 {
    const sx = Vec3.new(matrix.m[0], matrix.m[1], matrix.m[2]).length();
    const sy = Vec3.new(matrix.m[4], matrix.m[5], matrix.m[6]).length();
    const sz = Vec3.new(matrix.m[8], matrix.m[9], matrix.m[10]).length();
    return (sx + sy + sz) / 3.0;
}

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

    instance_buffer: sg.Buffer,
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

        const buf = sg.makeBuffer(.{
            .usage = .{ .vertex_buffer = true, .dynamic_update = true },
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
        if (self.gpu_slots.len > 0) {
            self.allocator.free(self.gpu_slots);
            self.gpu_slots = &.{};
        }
        if (self.texture) |*t| {
            t.deinit();
            self.texture = null;
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
        // GPU ring: re-anchor the epoch and drop all slots. Old slot records
        // (if the buffer is not cleared) carry spawn times far ahead of the
        // new epoch, so they cull as unborn (t < 0) in the shader.
        self.clock_seconds = 0.0;
        self.gpu_write_cursor = 0;
        self.gpu_high_water = 0;
        self.gpu_dirty = false;
        self.gpu_dirty_wrapped = false;
    }

    inline fn randomRange(rnd: std.Random, min_val: f32, max_val: f32) f32 {
        return min_val + rnd.float(f32) * (max_val - min_val);
    }

    /// One sampled particle spawn, shared verbatim by both simulation paths so
    /// identical seeds produce identical particles regardless of mode. The PRNG
    /// call order (box xyz, direction xyz, speed, lifetime, rotation, angular
    /// velocity) is part of the CPU behaviour contract — do not reorder.
    const SpawnSample = struct {
        position: Vec3,
        velocity: Vec3,
        lifetime: f32,
        /// Degrees, normalized to [0, 360).
        rotation_deg: f32,
        /// Degrees/second.
        angular_velocity: f32,
    };

    fn sampleSpawn(self: *ParticleSystem, rnd: std.Random) SpawnSample {
        // In local_space mode this offset is stored verbatim (emitter-local);
        // the emitter world matrix is applied only at instance-fill time.
        const spawn_pos = Vec3.new(
            self.emitter_position.x + randomRange(rnd, self.emitter_box_min.x, self.emitter_box_max.x),
            self.emitter_position.y + randomRange(rnd, self.emitter_box_min.y, self.emitter_box_max.y),
            self.emitter_position.z + randomRange(rnd, self.emitter_box_min.z, self.emitter_box_max.z),
        );

        const dir = Vec3.new(
            randomRange(rnd, self.direction_min.x, self.direction_max.x),
            randomRange(rnd, self.direction_min.y, self.direction_max.y),
            randomRange(rnd, self.direction_min.z, self.direction_max.z),
        );
        const speed = randomRange(rnd, self.speed_min, self.speed_max);
        const dir_len = dir.length();
        const vel = if (dir_len > 0.0001) dir.scale(speed / dir_len) else Vec3.new(0, speed, 0);

        const lifetime = randomRange(rnd, self.lifetime_min, self.lifetime_max);

        return .{
            .position = spawn_pos,
            .velocity = vel,
            .lifetime = if (lifetime > 0.0001) lifetime else 0.0001,
            .rotation_deg = normalizeAngleDeg(randomRange(rnd, self.rotation_min, self.rotation_max)),
            .angular_velocity = randomRange(rnd, self.angular_velocity_min, self.angular_velocity_max),
        };
    }

    pub fn emitOne(self: *ParticleSystem) void {
        // `.gpu` emits into the spawn-slot ring; only the integration stage
        // differs from the CPU path.
        if (self.simulation_mode == .gpu) {
            self.emitGpuSlot();
            return;
        }
        if (self.active_count >= self.capacity) return;
        const sample = self.sampleSpawn(self.prng.random());

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
        };
        self.active_count += 1;
    }

    /// Writes the next ring slot for the GPU path. The ring overwrites the
    /// oldest slot instead of dropping the spawn (the CPU path drops when
    /// full) — with no live-count tracking on the CPU this is the only
    /// policy a stateless ring can afford, and it keeps steady emission
    /// allocation-free. Bookkeeping here feeds the per-frame upload ranges.
    fn emitGpuSlot(self: *ParticleSystem) void {
        // Slots are provisioned by updateGpu; emissions before the first
        // update (no ring yet) drop instead of allocating on the hot path.
        if (self.capacity == 0 or self.gpu_slots.len < self.capacity) return;
        const sample = self.sampleSpawn(self.prng.random());

        const cap = self.capacity;
        const c = self.gpu_write_cursor;
        if (!self.gpu_dirty) {
            self.gpu_dirty = true;
            self.gpu_dirty_start = c;
            self.gpu_dirty_wrapped = false;
        }
        self.gpu_slots[c] = .{
            .spawn_pos_time = .{ sample.position.x, sample.position.y, sample.position.z, self.clock_seconds },
            .velocity_lifetime = .{ sample.velocity.x, sample.velocity.y, sample.velocity.z, sample.lifetime },
            .color_start = self.color_start.toArray(),
            .color_end = self.color_end.toArray(),
            .size_rotation = .{
                self.size_start,
                self.size_end,
                rotationToRadians(sample.rotation_deg),
                rotationToRadians(sample.angular_velocity),
            },
        };
        self.gpu_write_cursor = (c + 1) % cap;
        if (self.gpu_write_cursor < c) self.gpu_dirty_wrapped = true;
        if (self.gpu_high_water < cap) self.gpu_high_water += 1;
        self.gpu_dirty_end = self.gpu_write_cursor;
        // Upper bound only (see gpu_high_water): scene stats and the render
        // skip-check key off active_count, so it must track the draw count.
        self.active_count = self.gpu_high_water;
    }

    pub fn burst(self: *ParticleSystem, count: usize) void {
        var n: usize = 0;
        while (n < count) : (n += 1) {
            // CPU keeps the old drop-when-full guard; the GPU ring always
            // accepts (oldest slot is recycled).
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

    /// CPU simulation step, structured as three phases so the heavy work is
    /// data-parallel while the results stay bit-identical to the legacy
    /// fused loop for any worker count:
    ///   A (parallel)  — age + integrate every live slot exactly once,
    ///                   recording survival in `alive_scratch`;
    ///   B (serial)    — legacy swap-compaction of dead slots;
    ///   C (parallel)  — fill render instance data for the compacted range.
    /// Each slot is touched only through its own index in A and C, so no
    /// locks are needed; emission (PRNG-driven) stays serial up front.
    pub fn updateCpu(self: *ParticleSystem, dt: f32) void {
        if (self.is_emitting and self.emit_rate > 0.0) {
            self.emit_accumulator += dt * self.emit_rate;
            while (self.emit_accumulator >= 1.0 and self.active_count < self.capacity) {
                self.emitOne();
                self.emit_accumulator -= 1.0;
            }
        }

        const emitter_matrix = self.resolveEmitterMatrix();
        const emitter_scale = if (emitter_matrix) |m| worldScaleFactor(m) else 1.0;
        const grav_dt = self.gravity.scale(dt);

        const pool = self.thread_pool orelse jobs.global;

        // Phase A: integrate. Each worker owns its index range exclusively.
        var ictx = IntegrateCtx{ .ps = self, .grav_dt = grav_dt, .dt = dt };
        jobs.parallelFor(pool, IntegrateCtx, &ictx, integrateRange, self.active_count);

        // Phase B: compact. Same swap-with-last recycling as the legacy
        // loop (dead slot at i replaced by the last integrated slot).
        const alive = self.alive_scratch;
        var i: usize = 0;
        while (i < self.active_count) {
            if (alive[i] == 0) {
                self.active_count -= 1;
                if (i < self.active_count) {
                    self.particles[i] = self.particles[self.active_count];
                    alive[i] = alive[self.active_count];
                    continue;
                } else {
                    break;
                }
            }
            i += 1;
        }

        // Phase C: render-data fill over the compacted range.
        var fctx = FillCtx{
            .ps = self,
            .emitter_matrix = emitter_matrix,
            .emitter_scale = emitter_scale,
        };
        jobs.parallelFor(pool, FillCtx, &fctx, fillRange, self.active_count);
    }

    /// Lazily provisions the slot ring. `false` means allocation failure;
    /// updateGpu surfaces that as error.OutOfMemory — no CPU fallback.
    fn provisionGpuSlots(self: *ParticleSystem) bool {
        if (self.gpu_slots.len == self.capacity) return true;
        const slots = self.allocator.alloc(GpuParticleSlot, self.capacity) catch return false;
        self.gpu_slots = slots;
        return true;
    }

    /// GPU-path frame step: advances the epoch clock and appends spawn slots
    /// to the ring — O(emitted), never O(particles). The simulation itself
    /// happens in the vertex shader (shaders/particle.glsl, program
    /// particle_gpu). Explicit support contract: `local_space` and slot
    /// allocation failure return errors, never a silent CPU downgrade.
    pub fn updateGpu(self: *ParticleSystem, dt: f32) UpdateError!void {
        if (self.local_space) return error.LocalSpaceNeedsCpu;
        if (!self.provisionGpuSlots()) return error.OutOfMemory;
        self.clock_seconds += dt;
        if (self.is_emitting and self.emit_rate > 0.0) {
            self.emit_accumulator += dt * self.emit_rate;
            // No capacity guard: the ring recycles the oldest slot, so a full
            // system never blocks the accumulator (CPU drops instead).
            while (self.emit_accumulator >= 1.0) {
                self.emitGpuSlot();
                self.emit_accumulator -= 1.0;
            }
        }
    }

    /// Dirty slot range to upload this frame, or null when nothing changed.
    /// Pure so tests can exercise the bookkeeping without a GPU context.
    fn gpuUploadRange(self: *const ParticleSystem) ?[]GpuParticleSlot {
        if (!self.gpu_dirty) return null;
        // A frame that wrapped the ring wrote two disjoint segments; a single
        // sg.updateBuffer can only cover one range from offset 0, so such
        // frames re-upload the whole [0, high_water) prefix. Beyond that
        // prefix nothing changed since the previous upload.
        if (self.gpu_dirty_wrapped) return self.gpu_slots[0..self.gpu_high_water];
        if (self.gpu_dirty_end <= self.gpu_dirty_start) return null;
        return self.gpu_slots[self.gpu_dirty_start..self.gpu_dirty_end];
    }

    fn flushGpuUpload(self: *ParticleSystem) void {
        if (self.gpu_slot_buffer.id == 0) return;
        if (self.gpuUploadRange()) |range| {
            sg.updateBuffer(self.gpu_slot_buffer, sg.asRange(range));
        }
        self.gpu_dirty = false;
        self.gpu_dirty_wrapped = false;
    }

    /// Explicit mode dispatch: the CPU path always runs, `.gpu` runs or
    /// returns an error (`UpdateError`) — never a silent downgrade.
    pub fn update(self: *ParticleSystem, dt: f32) UpdateError!void {
        if (self.simulation_mode == .gpu) {
            // Lazily create the GPU instance buffer on first update (the
            // sg context is required, which tests never have).
            if (self.gpu_slot_buffer.id == 0) {
                self.gpu_slot_buffer = sg.makeBuffer(.{
                    .usage = .{ .vertex_buffer = true, .dynamic_update = true },
                    .size = self.capacity * @sizeOf(GpuParticleSlot),
                });
            }
            try self.updateGpu(dt);
            self.flushGpuUpload();
            return;
        }
        self.updateCpu(dt);

        if (self.active_count > 0) {
            sg.updateBuffer(self.instance_buffer, sg.asRange(self.instances[0..self.active_count]));
        }
    }
};

/// Phase-A payload: per-slot integration. A slot touches only `particles[i]`
/// and `alive_scratch[i]`, so workers own disjoint ranges without locks.
const IntegrateCtx = struct {
    ps: *ParticleSystem,
    grav_dt: Vec3,
    dt: f32,
};

fn integrateRange(ctx: *IntegrateCtx, start: usize, end: usize) void {
    const ps = ctx.ps;
    for (start..end) |i| {
        const p = &ps.particles[i];
        p.age += ctx.dt;
        if (p.age >= p.lifetime) {
            ps.alive_scratch[i] = 0;
            continue;
        }
        ps.alive_scratch[i] = 1;
        // Physics update (hoisted gravity delta + scaled velocity).
        // In local_space mode gravity/velocity integrate in emitter-local
        // units; the world transform applies at instance-fill time.
        p.velocity = p.velocity.add(ctx.grav_dt);
        p.position = p.position.add(p.velocity.scale(ctx.dt));
        p.rotation = normalizeAngleDeg(p.rotation + p.angular_velocity * ctx.dt);
    }
}

/// Phase-C payload: render instance fill. Reads `particles[i]` (already
/// compacted), writes `instances[i]` — again index-exclusive.
const FillCtx = struct {
    ps: *ParticleSystem,
    emitter_matrix: ?Mat4,
    emitter_scale: f32,
};

fn fillRange(ctx: *FillCtx, start: usize, end: usize) void {
    const ps = ctx.ps;
    for (start..end) |i| {
        const p = &ps.particles[i];
        const t = p.age / p.lifetime;
        const current_size = p.size + (p.size_end - p.size) * t;
        const current_color = Color4.lerp(p.color, p.color_end, t);
        const frame = spritesheetFrameForAge(
            p.age,
            p.lifetime,
            ps.spritesheet_columns,
            ps.spritesheet_rows,
            ps.spritesheet_loops,
        );
        const uv = spritesheetUvRect(frame, ps.spritesheet_columns, ps.spritesheet_rows);

        var render_pos = p.position;
        var render_size = current_size;
        if (ctx.emitter_matrix) |m| {
            render_pos = localToWorld(m, p.position);
            render_size = current_size * ctx.emitter_scale;
        }

        ps.instances[i] = .{
            .pos_size = .{ render_pos.x, render_pos.y, render_pos.z, render_size },
            .color = current_color.toArray(),
            .uv_offset_scale = uv,
            .rotation_misc = .{ rotationToRadians(p.rotation), 0.0, 0.0, 0.0 },
        };
    }
}

test "two-phase CPU update is worker-count invariant" {
    const a = std.testing.allocator;
    const pool = try jobs.Pool.init(a, 2);
    defer pool.deinit();

    // Identical seed + config; the only difference is the execution path:
    // `serial` runs inline (no pool), `parallel` fork-joins once the live
    // count crosses jobs.Pool.min_len_for_workers.
    var serial = try makeTestSystem(a, 16_384);
    defer freeTestSystem(&serial);
    var parallel = try makeTestSystem(a, 16_384);
    defer freeTestSystem(&parallel);
    parallel.thread_pool = pool;

    for ([2]*ParticleSystem{ &serial, &parallel }) |ps| {
        ps.gravity = Vec3.new(0.0, -3.0, 0.0);
        ps.emit_rate = 8000.0;
        ps.is_emitting = true;
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
    // Past the pool threshold with real deaths: proves phase A/B/C actually
    // forked (not just the inline fallback both systems would share on a
    // small range).
    try std.testing.expect(serial.active_count > jobs.Pool.min_len_for_workers);
}

// --- GPU-free test helpers & tests (no sg.* calls below this line) ---

fn makeTestSystem(allocator: std.mem.Allocator, capacity: usize) !ParticleSystem {
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

fn freeTestSystem(ps: *ParticleSystem) void {
    if (ps.gpu_slots.len > 0) ps.allocator.free(ps.gpu_slots);
    ps.allocator.free(ps.particles);
    ps.allocator.free(ps.instances);
    if (ps.alive_scratch.len > 0) ps.allocator.free(ps.alive_scratch);
}

test "spritesheet frames across ages and loops" {
    // 2x2 sheet, lifetime 4s, single loop: one frame per second.
    try std.testing.expectEqual(@as(u32, 0), spritesheetFrameForAge(0.0, 4.0, 2, 2, 1.0));
    try std.testing.expectEqual(@as(u32, 1), spritesheetFrameForAge(1.0, 4.0, 2, 2, 1.0));
    try std.testing.expectEqual(@as(u32, 2), spritesheetFrameForAge(2.0, 4.0, 2, 2, 1.0));
    try std.testing.expectEqual(@as(u32, 3), spritesheetFrameForAge(3.0, 4.0, 2, 2, 1.0));
    // Double loop: age 1s (norm 0.25) -> floor(0.25*2*4) = 2.
    try std.testing.expectEqual(@as(u32, 2), spritesheetFrameForAge(1.0, 4.0, 2, 2, 2.0));
    try std.testing.expectEqual(@as(u32, 1), spritesheetFrameForAge(0.5, 4.0, 2, 2, 2.0));
    // UV rects run left-to-right, bottom-to-top.
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.5, 0.5 }, spritesheetUvRect(0, 2, 2));
    try std.testing.expectEqual([4]f32{ 0.5, 0.0, 0.5, 0.5 }, spritesheetUvRect(1, 2, 2));
    try std.testing.expectEqual([4]f32{ 0.0, 0.5, 0.5, 0.5 }, spritesheetUvRect(2, 2, 2));
    try std.testing.expectEqual([4]f32{ 0.5, 0.5, 0.5, 0.5 }, spritesheetUvRect(3, 2, 2));
}

test "spritesheet boundaries and 1x1 default" {
    // 1x1 (default) always yields frame 0 / full-texture UV.
    try std.testing.expectEqual(@as(u32, 0), spritesheetFrameForAge(0.0, 1.0, 1, 1, 1.0));
    try std.testing.expectEqual(@as(u32, 0), spritesheetFrameForAge(0.99, 1.0, 1, 1, 5.0));
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 1.0, 1.0 }, spritesheetUvRect(0, 1, 1));
    // Boundary: exactly age == lifetime wraps per spec (floor(loops*frames) % frames).
    try std.testing.expectEqual(@as(u32, 0), spritesheetFrameForAge(4.0, 4.0, 2, 2, 1.0));
    try std.testing.expectEqual(@as(u32, 3), spritesheetFrameForAge(3.999, 4.0, 2, 2, 1.0));
    // Zero grid dimensions are guarded to 1 (no div-by-zero).
    try std.testing.expectEqual(@as(u32, 0), spritesheetFrameForAge(0.5, 1.0, 0, 0, 1.0));
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 1.0, 1.0 }, spritesheetUvRect(7, 0, 0));
    try std.testing.expectEqual(@as(u32, 4), spritesheetFrameCount(2, 2));
    try std.testing.expectEqual(@as(u32, 1), spritesheetFrameCount(0, 0));
}

test "rotation integrates angular velocity" {
    var ps = try makeTestSystem(std.testing.allocator, 4);
    defer freeTestSystem(&ps);
    ps.direction_min = Vec3.zero;
    ps.direction_max = Vec3.zero;
    ps.speed_min = 0.0;
    ps.speed_max = 0.0;
    ps.gravity = Vec3.zero;
    ps.lifetime_min = 10.0;
    ps.lifetime_max = 10.0;
    ps.rotation_min = 0.0;
    ps.rotation_max = 0.0;
    ps.angular_velocity_min = 90.0;
    ps.angular_velocity_max = 90.0;
    ps.emitOne();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ps.particles[0].rotation, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 90.0), ps.particles[0].angular_velocity, 1e-5);
    ps.updateCpu(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 90.0), ps.particles[0].rotation, 1e-4);
    // Instance carries radians: 90deg = pi/2.
    try std.testing.expectApproxEqAbs(
        std.math.pi / 2.0,
        ps.instances[0].rotation_misc[0],
        1e-5,
    );
    // Wrap-around: 350deg + 20deg/s * 1s = 10deg.
    var ps2 = try makeTestSystem(std.testing.allocator, 4);
    defer freeTestSystem(&ps2);
    ps2.direction_min = Vec3.zero;
    ps2.direction_max = Vec3.zero;
    ps2.speed_min = 0.0;
    ps2.speed_max = 0.0;
    ps2.gravity = Vec3.zero;
    ps2.lifetime_min = 10.0;
    ps2.lifetime_max = 10.0;
    ps2.rotation_min = 350.0;
    ps2.rotation_max = 350.0;
    ps2.angular_velocity_min = 20.0;
    ps2.angular_velocity_max = 20.0;
    ps2.emitOne();
    ps2.updateCpu(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), ps2.particles[0].rotation, 1e-4);
}

test "angle normalization" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), normalizeAngleDeg(0.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), normalizeAngleDeg(360.0), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), normalizeAngleDeg(-360.0), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), normalizeAngleDeg(370.0), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 350.0), normalizeAngleDeg(-10.0), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 180.0), normalizeAngleDeg(540.0), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), rotationToRadians(0.0), 1e-6);
    try std.testing.expectApproxEqAbs(std.math.pi, rotationToRadians(180.0), 1e-5);
}

test "localToWorld helper with rotation and scale" {
    const m = Mat4.fromRotationTranslationScale(
        Vec3.new(10.0, 0.0, 0.0),
        Vec3.new(0.0, 0.0, 90.0),
        Vec3.new(2.0, 2.0, 2.0),
    );
    // (1,0,0) -> scaled (2,0,0) -> rotZ90 -> (0,2,0) -> translated (10,2,0).
    const w = localToWorld(m, Vec3.new(1.0, 0.0, 0.0));
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), w.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), w.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), w.z, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), worldScaleFactor(m), 1e-5);
    // Identity is exact.
    const id = Mat4.identity;
    const p = localToWorld(id, Vec3.new(1.0, 2.0, 3.0));
    try std.testing.expectEqual(Vec3.new(1.0, 2.0, 3.0), p);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), worldScaleFactor(id), 1e-6);
}

test "defaults keep world positions bit-identical" {
    var ps = try makeTestSystem(std.testing.allocator, 4);
    defer freeTestSystem(&ps);
    // Defaults: local_space=false, emitter_mesh=null, 1x1 sheet, zero rotation.
    try std.testing.expectEqual(false, ps.local_space);
    try std.testing.expectEqual(@as(?*Mesh, null), ps.emitter_mesh);
    ps.emitter_position = Vec3.new(1.0, 2.0, 3.0);
    ps.direction_min = Vec3.zero;
    ps.direction_max = Vec3.zero;
    ps.speed_min = 0.0;
    ps.speed_max = 0.0;
    ps.gravity = Vec3.zero;
    ps.lifetime_min = 10.0;
    ps.lifetime_max = 10.0;
    ps.emitOne();
    ps.updateCpu(0.0);
    const p = ps.particles[0];
    const inst = ps.instances[0];
    try std.testing.expectEqual(p.position.x, inst.pos_size[0]);
    try std.testing.expectEqual(p.position.y, inst.pos_size[1]);
    try std.testing.expectEqual(p.position.z, inst.pos_size[2]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 1.0, 1.0 }, inst.uv_offset_scale);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, inst.rotation_misc);
    try std.testing.expectEqual(@as(?Mat4, null), ps.resolveEmitterMatrix());
}

test "local_space spawn stays relative, instances follow emitter" {
    var ps = try makeTestSystem(std.testing.allocator, 4);
    defer freeTestSystem(&ps);
    // Buffer-free emitter mesh: local-space update only reads the transform.
    var emitter = Mesh{ .name = "emitter", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 0 };
    emitter.position = Vec3.new(5.0, 0.0, 0.0);
    emitter.rotation = Vec3.zero;
    emitter.scaling = Vec3.one;
    emitter.base_matrix = Mat4.identity;

    ps.local_space = true;
    ps.emitter_mesh = &emitter;
    ps.emitter_position = Vec3.new(1.0, 2.0, 3.0);
    ps.direction_min = Vec3.zero;
    ps.direction_max = Vec3.zero;
    ps.speed_min = 0.0;
    ps.speed_max = 0.0;
    ps.gravity = Vec3.zero;
    ps.lifetime_min = 10.0;
    ps.lifetime_max = 10.0;
    ps.emitOne();

    // Stored coordinates are emitter-local, not world.
    try std.testing.expectEqual(Vec3.new(1.0, 2.0, 3.0), ps.particles[0].position);

    ps.updateCpu(0.0);
    // Instance = world matrix applied: (1,2,3) + emitter offset (5,0,0).
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), ps.instances[0].pos_size[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), ps.instances[0].pos_size[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), ps.instances[0].pos_size[2], 1e-5);

    // Moving the emitter moves rendered instances, stored locals stay put.
    emitter.position = Vec3.new(10.0, 0.0, 0.0);
    ps.updateCpu(0.0);
    try std.testing.expectEqual(Vec3.new(1.0, 2.0, 3.0), ps.particles[0].position);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), ps.instances[0].pos_size[0], 1e-5);
}

// --- GPU-path tests: analytic math, ring bookkeeping and fallback (sg-free;
// the GLSL side of the formulas is exercised by sokol-shdc at build time) ---

test "gpu mode defaults off and slot layout matches shader attrs" {
    var ps = try makeTestSystem(std.testing.allocator, 4);
    defer freeTestSystem(&ps);
    try std.testing.expectEqual(SimulationMode.cpu, ps.simulation_mode);
    try std.testing.expectEqual(@as(f32, 0.0), ps.drag);
    try std.testing.expectEqual(@as(f32, 0.0), ps.clock_seconds);
    // Five FLOAT4 vertex attributes (see particle_gpu program in
    // shaders/particle.glsl): spawn_pos_time, velocity_lifetime, color_start,
    // color_end, size_rotation.
    try std.testing.expectEqual(@as(usize, 80), @sizeOf(GpuParticleSlot));
}

test "slot age gates the alive window" {
    // Unborn (spawn in the future): also culls stale slots after a reset.
    const unborn = slotAge(1.0, 2.0, 1.5);
    try std.testing.expectEqual(false, unborn.alive);
    try std.testing.expectEqual(@as(f32, 0.0), unborn.t);
    // Mid-life.
    const mid = slotAge(2.75, 2.0, 1.5);
    try std.testing.expectEqual(true, mid.alive);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mid.t, 1e-6);
    // Death at exactly t == 1.
    const dead = slotAge(3.5, 2.0, 1.5);
    try std.testing.expectEqual(false, dead.alive);
    try std.testing.expectEqual(@as(f32, 1.0), dead.t);
    // Degenerate lifetimes clamp so the shader division stays finite; a
    // zeroed slot (never written) is dead at any clock >= 1e-4 and unborn
    // exactly at the epoch.
    try std.testing.expectEqual(false, slotAge(0.5, 0.0, 0.0).alive);
    try std.testing.expectEqual(true, slotAge(0.0, 0.0, 0.0).alive);
}

test "analytic trajectory matches golden values" {
    const p0 = Vec3.new(1.0, 2.0, 3.0);
    const v0 = Vec3.new(1.0, 0.0, -1.0);
    const g = Vec3.new(0.0, -9.8, 0.0);

    // No drag: p = p0 + v0*t + 0.5*g*t^2 at t = 0.5 ->
    // (1.5, 2 - 1.225, 2.5).
    const free = analyticPosition(p0, v0, g, 0.0, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), free.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.775), free.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), free.z, 1e-6);

    // Zero elapsed time returns the spawn point exactly.
    try std.testing.expectEqual(p0, analyticPosition(p0, v0, g, 3.0, 0.0));

    // Drag k = 2 at t = 0.5: s = (1 - e^-1)/2 = 0.3160602794,
    // s2 = (0.5 - s)/2 = 0.0919698603:
    //   x = 1 + s         = 1.3160602794
    //   y = 2 - 9.8 * s2  = 1.0986953694
    //   z = 3 - s         = 2.6839397206
    // Tolerance 1e-4 covers the float32 rounding of the hand-computed
    // doubles; the GLSL duplicate may deviate by ~1e-6 relative on top
    // (documented on analyticPosition).
    const dragged = analyticPosition(p0, v0, g, 2.0, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.3160603), dragged.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0986954), dragged.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.6839397), dragged.z, 1e-4);

    // Drag below the cancellation threshold (1e-6) degrades to the free
    // trajectory without catastrophic cancellation.
    const tiny = analyticPosition(p0, v0, g, 1.0e-7, 0.5);
    try std.testing.expectApproxEqAbs(free.x, tiny.x, 1e-5);
    try std.testing.expectApproxEqAbs(free.y, tiny.y, 1e-5);
    try std.testing.expectApproxEqAbs(free.z, tiny.z, 1e-5);
}

test "gpu spawn sampling matches cpu with the same seed" {
    const a = std.testing.allocator;
    var cpu = try makeTestSystem(a, 4);
    defer freeTestSystem(&cpu);
    var gpu = try makeTestSystem(a, 4);
    defer freeTestSystem(&gpu);
    gpu.gpu_slots = try a.alloc(GpuParticleSlot, 4);
    gpu.simulation_mode = .gpu;
    cpu.emitOne();
    gpu.emitOne(); // ring write at slot 0, clock 0

    const p = cpu.particles[0];
    const s = gpu.gpu_slots[0];
    // Same seed + shared sampler => identical spawn attributes.
    try std.testing.expectEqual(p.position.x, s.spawn_pos_time[0]);
    try std.testing.expectEqual(p.position.y, s.spawn_pos_time[1]);
    try std.testing.expectEqual(p.position.z, s.spawn_pos_time[2]);
    try std.testing.expectEqual(@as(f32, 0.0), s.spawn_pos_time[3]);
    try std.testing.expectEqual(p.velocity.x, s.velocity_lifetime[0]);
    try std.testing.expectEqual(p.velocity.y, s.velocity_lifetime[1]);
    try std.testing.expectEqual(p.velocity.z, s.velocity_lifetime[2]);
    try std.testing.expectEqual(p.lifetime, s.velocity_lifetime[3]);
    try std.testing.expectEqual(cpu.color_start.toArray(), s.color_start);
    try std.testing.expectEqual(cpu.color_end.toArray(), s.color_end);
    try std.testing.expectEqual(p.size, s.size_rotation[0]);
    try std.testing.expectEqual(p.size_end, s.size_rotation[1]);
    // Rotation is stored normalized-degrees on CPU, radians on GPU.
    try std.testing.expectApproxEqAbs(rotationToRadians(p.rotation), s.size_rotation[2], 1e-6);
    try std.testing.expectApproxEqAbs(rotationToRadians(p.angular_velocity), s.size_rotation[3], 1e-6);
    try std.testing.expectEqual(@as(usize, 1), gpu.active_count);
    try std.testing.expectEqual(@as(usize, 1), gpu.gpu_high_water);
}

test "gpu ring wraps and recycles oldest slots" {
    const a = std.testing.allocator;
    var ps = try makeTestSystem(a, 4);
    defer freeTestSystem(&ps);
    ps.gpu_slots = try a.alloc(GpuParticleSlot, 4);
    ps.simulation_mode = .gpu;
    // Stamp each emission with a distinct epoch time: the spawn time tags
    // which emission a slot holds.
    var i: usize = 0;
    while (i < 6) : (i += 1) {
        ps.clock_seconds = @floatFromInt(i);
        ps.emitOne();
    }
    try std.testing.expectEqual(@as(usize, 2), ps.gpu_write_cursor);
    try std.testing.expectEqual(@as(usize, 4), ps.gpu_high_water);
    try std.testing.expectEqual(@as(usize, 4), ps.active_count);
    // Emissions 5 and 6 recycled slots 0 and 1.
    try std.testing.expectEqual(@as(f32, 4.0), ps.gpu_slots[0].spawn_pos_time[3]);
    try std.testing.expectEqual(@as(f32, 5.0), ps.gpu_slots[1].spawn_pos_time[3]);
    try std.testing.expectEqual(@as(f32, 2.0), ps.gpu_slots[2].spawn_pos_time[3]);
    try std.testing.expectEqual(@as(f32, 3.0), ps.gpu_slots[3].spawn_pos_time[3]);
    // The frame that crossed the ring end re-uploads the whole prefix.
    try std.testing.expectEqual(true, ps.gpu_dirty);
    try std.testing.expectEqual(true, ps.gpu_dirty_wrapped);
    try std.testing.expectEqual(@as(usize, 4), ps.gpuUploadRange().?.len);

    // Simulate the flush (the sg.updateBuffer part has no GPU in tests).
    ps.gpu_dirty = false;
    ps.gpu_dirty_wrapped = false;
    ps.clock_seconds = 6.0;
    ps.emitOne(); // slot 2
    try std.testing.expectEqual(false, ps.gpu_dirty_wrapped);
    const range = ps.gpuUploadRange().?;
    try std.testing.expectEqual(@as(usize, 1), range.len);
    try std.testing.expectEqual(@as(f32, 6.0), range[0].spawn_pos_time[3]);
    try std.testing.expectEqual(@as(usize, 2), ps.gpu_dirty_start);
    try std.testing.expectEqual(@as(usize, 3), ps.gpu_dirty_end);

    // Reset re-anchors the epoch and empties the ring.
    ps.reset();
    try std.testing.expectEqual(@as(usize, 0), ps.gpu_write_cursor);
    try std.testing.expectEqual(@as(usize, 0), ps.gpu_high_water);
    try std.testing.expectEqual(@as(usize, 0), ps.active_count);
    try std.testing.expectEqual(@as(f32, 0.0), ps.clock_seconds);
    try std.testing.expectEqual(false, ps.gpu_dirty);
}

test "gpu burst overflows the ring instead of dropping" {
    const a = std.testing.allocator;
    var ps = try makeTestSystem(a, 4);
    defer freeTestSystem(&ps);
    ps.gpu_slots = try a.alloc(GpuParticleSlot, 4);
    ps.simulation_mode = .gpu;
    ps.burst(6);
    // All six emissions landed (ring recycled the two oldest slots), while a
    // CPU burst caps at capacity.
    try std.testing.expectEqual(@as(usize, 2), ps.gpu_write_cursor);
    try std.testing.expectEqual(@as(usize, 4), ps.active_count);
    try std.testing.expectEqual(true, ps.gpu_dirty_wrapped);

    var cpu = try makeTestSystem(a, 4);
    defer freeTestSystem(&cpu);
    cpu.burst(6);
    try std.testing.expectEqual(@as(usize, 4), cpu.active_count);
}

test "gpu update does no per-particle cpu work" {
    const a = std.testing.allocator;
    var ps = try makeTestSystem(a, 4);
    defer freeTestSystem(&ps);
    ps.simulation_mode = .gpu;
    ps.is_emitting = true;
    ps.emit_rate = 60.0;
    ps.lifetime_min = 1.0;
    ps.lifetime_max = 1.0;
    // Sentinel in the CPU storage: the GPU frame step must never touch it.
    const sentinel = Particle{
        .position = Vec3.new(7.0, 8.0, 9.0),
        .velocity = Vec3.new(1.0, 2.0, 3.0),
        .size = 1.0,
        .size_end = 0.0,
        .color = Color4.new(1.0, 1.0, 1.0, 1.0),
        .color_end = Color4.new(0.0, 0.0, 0.0, 0.0),
        .age = 0.5,
        .lifetime = 2.0,
        .rotation = 45.0,
        .angular_velocity = 10.0,
    };
    ps.particles[0] = sentinel;
    try ps.updateGpu(1.0);
    try std.testing.expectEqual(sentinel.age, ps.particles[0].age);
    try std.testing.expectEqual(sentinel.position, ps.particles[0].position);
    // A full second at 60 p/s filled the 4-slot ring; the epoch clock advanced
    // exactly once and the accumulator drained to zero.
    try std.testing.expectEqual(@as(usize, 4), ps.gpu_high_water);
    try std.testing.expectEqual(@as(usize, 4), ps.active_count);
    try std.testing.expectEqual(@as(usize, 0), ps.gpu_write_cursor);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ps.clock_seconds, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ps.emit_accumulator, 1e-5);
}

test "gpu with local_space is an explicit error, not a downgrade" {
    const a = std.testing.allocator;
    var ps = try makeTestSystem(a, 4);
    defer freeTestSystem(&ps);
    ps.simulation_mode = .gpu;
    ps.local_space = true; // moving-emitter local frame needs historical state
    try std.testing.expectError(error.LocalSpaceNeedsCpu, ps.updateGpu(0.016));
    // The requested mode is never mutated behind the caller's back and the
    // GPU ring never engaged.
    try std.testing.expectEqual(SimulationMode.gpu, ps.simulation_mode);
    try std.testing.expectEqual(@as(usize, 0), ps.active_count);
    try std.testing.expectEqual(@as(usize, 0), ps.gpu_high_water);
}
