//! Stateless GPU simulation path (spawn-slot ring). Split out of
//! `particles.zig` (facade).
//!
//! Every function takes the system as `anytype` so this module never imports
//! `system.zig` or the facade back — same discipline as `profiler/*`.
//! `subemitters.zig` pushes child spawns via `pushGpuSlot` (sibling import,
//! no cycle). The analytic trajectory mirror (`analyticPosition`, `slotAge`)
//! lives in `types.zig`; the GLSL side is pinned by sokol-shdc at build time.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const upload_meter = @import("../gpu_upload_meter.zig");

const types = @import("types.zig");
const sampling = @import("sampling.zig");
const flow = @import("flow.zig");
const collisions = @import("collisions.zig");

const Vec3 = math.Vec3;
const Color4 = math.Color4;
const GpuParticleSlot = types.GpuParticleSlot;
const Particle = types.Particle;
const SimulationMode = types.SimulationMode;
const UpdateError = types.UpdateError;

// Local aliases so the moved bodies stay byte-identical.
const sampleSpawn = sampling.sampleSpawn;
const rotationToRadians = sampling.rotationToRadians;
const activeFlowCtx = flow.activeFlowCtx;
const activeCollisionCtx = collisions.activeCollisionCtx;
const SpawnSample = sampling.SpawnSample;

/// Writes the next ring slot for the GPU path. The ring overwrites the
/// oldest slot instead of dropping the spawn (the CPU path drops when
/// full) — with no live-count tracking on the CPU this is the only
/// policy a stateless ring can afford, and it keeps steady emission
/// allocation-free. Bookkeeping here feeds the per-frame upload ranges.
pub fn emitGpuSlot(self: anytype) void {
    // Slots are provisioned by updateGpu; emissions before the first
    // update (no ring yet) drop instead of allocating on the hot path.
    if (self.capacity == 0 or self.gpu_slots.len < self.capacity) return;
    pushGpuSlot(self, sampleSpawn(self, self.prng.random()));
}

/// Appends one sampled spawn to the GPU ring. Shared by emitGpuSlot and
/// sub-emitter child spawns so both take the same slot layout and upload
/// bookkeeping.
pub fn pushGpuSlot(self: anytype, sample: SpawnSample) void {
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

/// Lazily provisions the slot ring. `false` means allocation failure;
/// updateGpu surfaces that as error.OutOfMemory — no CPU fallback.
pub fn provisionGpuSlots(self: anytype) bool {
    if (self.gpu_slots.len == self.capacity) return true;
    const slots = self.allocator.alloc(GpuParticleSlot, self.capacity) catch return false;
    self.gpu_slots = slots;
    return true;
}

/// GPU-path frame step: advances the epoch clock and appends spawn slots
/// to the ring — O(emitted), never O(particles). The simulation itself
/// happens in the vertex shader (shaders/particle.glsl, program
/// particle_gpu). Explicit support contract: `local_space`, an armed flow
/// field, armed collisions, and slot allocation failure return errors,
/// never a silent CPU downgrade.
pub fn updateGpu(self: anytype, dt: f32) UpdateError!void {
    if (self.local_space) return error.LocalSpaceNeedsCpu;
    if (activeFlowCtx(self) != null) return error.FlowMapNeedsCpu;
    if (activeCollisionCtx(self) != null) return error.CollisionNeedsCpu;
    if (!provisionGpuSlots(self)) return error.OutOfMemory;
    self.clock_seconds += dt;
    if (self.is_emitting and self.emit_rate > 0.0) {
        self.emit_accumulator += dt * self.emit_rate;
        // No capacity guard: the ring recycles the oldest slot, so a full
        // system never blocks the accumulator (CPU drops instead).
        while (self.emit_accumulator >= 1.0) {
            emitGpuSlot(self);
            self.emit_accumulator -= 1.0;
        }
    }
}

/// Dirty slot range to upload this frame, or null when nothing changed.
/// Pure so tests can exercise the bookkeeping without a GPU context.
pub fn gpuUploadRange(self: anytype) ?[]GpuParticleSlot {
    if (!self.gpu_dirty) return null;
    // A frame that wrapped the ring wrote two disjoint segments; a single
    // sg.updateBuffer can only cover one range from offset 0, so such
    // frames re-upload the whole [0, high_water) prefix. Beyond that
    // prefix nothing changed since the previous upload.
    if (self.gpu_dirty_wrapped) return self.gpu_slots[0..self.gpu_high_water];
    if (self.gpu_dirty_end <= self.gpu_dirty_start) return null;
    return self.gpu_slots[self.gpu_dirty_start..self.gpu_dirty_end];
}

pub fn flushGpuUpload(self: anytype) void {
    if (self.gpu_slot_buffer.id == 0) return;
    if (gpuUploadRange(self)) |range| {
        sg.writeBufferTransient(.{
            .dst = .{ .buffer = self.gpu_slot_buffer },
            .src = .{ .data = sg.asRange(range) },
        });
        // Учёт динамики: весь переданный диапазон ушёл в GPU-буфер.
        upload_meter.record(range.len * @sizeOf(GpuParticleSlot));
    }
    self.gpu_dirty = false;
    self.gpu_dirty_wrapped = false;
}

// --- GPU-path tests: ring bookkeeping and fallback (sg-free) ---

test "gpu slot layout matches shader attrs" {
    const sys = @import("system.zig");
    var ps = try sys.makeTestSystem(std.testing.allocator, 4);
    defer sys.freeTestSystem(&ps);
    // Five FLOAT4 vertex attributes (see particle_gpu program in
    // shaders/particle.glsl): spawn_pos_time, velocity_lifetime, color_start,
    // color_end, size_rotation.
    try std.testing.expectEqual(@as(usize, 80), @sizeOf(GpuParticleSlot));
}

test "gpu spawn sampling matches cpu with the same seed" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var cpu = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&cpu);
    var gpu = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&gpu);
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
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&ps);
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
    try std.testing.expectEqual(@as(usize, 4), gpuUploadRange(&ps).?.len);

    // Simulate the flush (the sg.updateBuffer part has no GPU in tests).
    ps.gpu_dirty = false;
    ps.gpu_dirty_wrapped = false;
    ps.clock_seconds = 6.0;
    ps.emitOne(); // slot 2
    try std.testing.expectEqual(false, ps.gpu_dirty_wrapped);
    const range = gpuUploadRange(&ps).?;
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
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    ps.gpu_slots = try a.alloc(GpuParticleSlot, 4);
    ps.simulation_mode = .gpu;
    ps.burst(6);
    // All six emissions landed (ring recycled the two oldest slots), while a
    // CPU burst caps at capacity.
    try std.testing.expectEqual(@as(usize, 2), ps.gpu_write_cursor);
    try std.testing.expectEqual(@as(usize, 4), ps.active_count);
    try std.testing.expectEqual(true, ps.gpu_dirty_wrapped);

    var cpu = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&cpu);
    cpu.burst(6);
    try std.testing.expectEqual(@as(usize, 4), cpu.active_count);
}

test "gpu update does no per-particle cpu work" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&ps);
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
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    ps.simulation_mode = .gpu;
    ps.local_space = true; // moving-emitter local frame needs historical state
    try std.testing.expectError(error.LocalSpaceNeedsCpu, ps.updateGpu(0.016));
    // The requested mode is never mutated behind the caller's back and the
    // GPU ring never engaged.
    try std.testing.expectEqual(SimulationMode.gpu, ps.simulation_mode);
    try std.testing.expectEqual(@as(usize, 0), ps.active_count);
    try std.testing.expectEqual(@as(usize, 0), ps.gpu_high_water);
}

test "gpu update stages buffer creation without an sg context" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    ps.simulation_mode = .gpu;
    ps.is_emitting = true;
    try ps.update(0.016);
    // No sg context in tests: no buffer was created, but the creation was
    // staged (and the frame's upload flagged) instead of crashing.
    try std.testing.expectEqual(true, ps.gpu_slot_buffer_pending);
    try std.testing.expectEqual(@as(u32, 0), ps.gpu_slot_buffer.id);
    try std.testing.expectEqual(true, ps.gpu_flush_pending);
    // Flushing without a context is a safe no-op that keeps the staged flag.
    ps.flushGpuUploads();
    try std.testing.expectEqual(true, ps.gpu_slot_buffer_pending);
    try std.testing.expectEqual(@as(u32, 0), ps.gpu_slot_buffer.id);
}
