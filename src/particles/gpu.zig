//! Stateless GPU simulation path (spawn-slot ring).

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
        // Dynamic upload tracking: entire uploaded range went to GPU buffer.
        upload_meter.record(range.len * @sizeOf(GpuParticleSlot));
    }
    self.gpu_dirty = false;
    self.gpu_dirty_wrapped = false;
}
