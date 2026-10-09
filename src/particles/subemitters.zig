//! Sub-emitters: CPU on-death child-spawn rules.

const std = @import("std");
const math = @import("math");

const types = @import("types.zig");
const sampling = @import("sampling.zig");
const gpu = @import("gpu.zig");
const compute_mode = @import("compute_mode.zig");

const Vec3 = math.Vec3;
const Color4 = math.Color4;
const max_sub_emitter_depth = types.max_sub_emitter_depth;
const max_sub_emitter_spawns_per_tick = types.max_sub_emitter_spawns_per_tick;

// Local aliases so the moved bodies stay byte-identical.
const sampleSpawn = sampling.sampleSpawn;
const pushGpuSlot = gpu.pushGpuSlot;
const pushComputeSpawn = compute_mode.pushComputeSpawn;

/// SplitMix64 step: the per-death hash stream for sub-emitter rolls/jitter.
/// Hash-based (not PRNG-based) so evaluation order and worker count cannot
/// affect the outcome.
fn splitmix64(state: *u64) u64 {
    state.* +%= 0x9E3779B97F4A7C15;
    var z = state.*;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

/// Maps a 64-bit hash to f32 in [0, 1] (top 32 bits over 2^32; may round to
/// exactly 1.0 for the all-ones hash, which callers treat as "no fire").
fn hash01(h: u64) f32 {
    return @as(f32, @floatFromInt(h >> 32)) / 4294967296.0;
}

/// One avalanche of the sub-emitter seed, tick, death index, emitter index
/// and salt into a hash state. Distinct salts give independent streams for
/// the probability roll vs each jitter component.
fn subHash(seed: u64, tick: u64, death_idx: usize, emitter_idx: usize, salt: u64) u64 {
    var h: u64 = seed ^ (tick *% 0x9E3779B97F4A7C15) ^
        (@as(u64, @intCast(death_idx)) *% 0xBF58476D1CE4E5B9) ^
        (@as(u64, @intCast(emitter_idx)) *% 0x94D049BB133111EB) ^ (salt *% 0xDA942042E4DD58B5);
    h ^= h >> 30;
    h *%= 0xBF58476D1CE4E5B9;
    h ^= h >> 27;
    h *%= 0x94D049BB133111EB;
    h ^= h >> 31;
    return h;
}

/// Record of one death for the serial sub-emitter pass: copied out during
/// compaction so firing (which may append to this same system for
/// self-emitters) never runs while the array is being compacted.
pub const DeathEvent = struct {
    position: Vec3,
    velocity: Vec3,
    depth: u8,
};

/// Spawns one sub-emitter child. Lifetime/rotation/color come from the
/// child system's own sampler (PRNG order stays serial and
/// deterministic); position/velocity blend toward the dead parent per the
/// rule. Drops when full (CPU) or unprovisioned (GPU ring), same policy
/// as the matching emit path. A `.gpu` child accepts the spawn into its
/// slot ring (stateless: depth untracked, chains stop there).
pub fn emitChild(
    self: anytype,
    spawn_pos: Vec3,
    parent_velocity: Vec3,
    inherit_velocity: f32,
    inherit_position: bool,
    depth: u8,
) void {
    const sample = sampleSpawn(self, self.prng.random());
    const velocity = parent_velocity.scale(inherit_velocity).add(sample.velocity.scale(1.0 - inherit_velocity));
    if (self.simulation_mode == .gpu) {
        if (self.capacity == 0 or self.gpu_slots.len < self.capacity) return;
        pushGpuSlot(self, .{
            .position = if (inherit_position) spawn_pos else sample.position,
            .velocity = velocity,
            .lifetime = sample.lifetime,
            .rotation_deg = sample.rotation_deg,
            .angular_velocity = sample.angular_velocity,
        });
        return;
    }
    // A `.compute` child accepts the spawn into its staged spawn window
    // (stateless on CPU: depth untracked, chains stop there — same rule
    // as the `.gpu` child above).
    if (self.simulation_mode == .compute) {
        if (self.capacity == 0 or self.compute_staging.len < self.capacity) return;
        pushComputeSpawn(self, .{
            .position = if (inherit_position) spawn_pos else sample.position,
            .velocity = velocity,
            .lifetime = sample.lifetime,
            .rotation_deg = sample.rotation_deg,
            .angular_velocity = sample.angular_velocity,
        });
        return;
    }
    if (self.active_count >= self.capacity) return;
    self.particles[self.active_count] = .{
        .position = if (inherit_position) spawn_pos else sample.position,
        .velocity = velocity,
        .size = self.size_start,
        .size_end = self.size_end,
        .color = self.color_start,
        .color_end = self.color_end,
        .age = 0.0,
        .lifetime = sample.lifetime,
        .rotation = sample.rotation_deg,
        .angular_velocity = sample.angular_velocity,
        .sub_depth = depth,
    };
    self.active_count += 1;
}

/// Serial on-death pass over the deaths recorded by the compaction loop.
/// Runs between compaction (phase B) and instance fill (phase C) so
/// self-spawned children land in valid instance data the same tick.
/// Bounded: at most `max_sub_emitter_spawns_per_tick` children per call.
pub fn fireSubEmitters(self: anytype, deaths: []const DeathEvent) void {
    var spawned_total: usize = 0;
    var di: usize = 0;
    while (di < deaths.len) : (di += 1) {
        const d = deaths[di];
        // Chain rule: deep-enough deaths spawn nothing.
        if (d.depth >= max_sub_emitter_depth) continue;
        const child_depth = d.depth + 1;
        for (self.sub_emitter_store[0..self.sub_emitter_count], 0..) |sub, ei| {
            if (sub.trigger != .on_death) continue;
            const probability = std.math.clamp(sub.probability, 0.0, 1.0);
            if (probability <= 0.0) continue;
            if (probability < 1.0) {
                const roll = hash01(subHash(self.sub_emitter_seed, self.sub_tick, di, ei, 0));
                if (roll >= probability) continue;
            }
            const inherit_velocity = std.math.clamp(sub.inherit_velocity, 0.0, 1.0);
            const radius = @max(sub.spawn_radius, 0.0);
            var k: u32 = 0;
            while (k < sub.count) : (k += 1) {
                if (spawned_total >= max_sub_emitter_spawns_per_tick) return;
                var spawn_pos = d.position;
                if (sub.inherit_position and radius > 0.0) {
                    var js: u64 = subHash(self.sub_emitter_seed, self.sub_tick, di, ei, 1 + @as(u64, k));
                    spawn_pos = spawn_pos.add(Vec3.new(
                        (hash01(splitmix64(&js)) * 2.0 - 1.0) * radius,
                        (hash01(splitmix64(&js)) * 2.0 - 1.0) * radius,
                        (hash01(splitmix64(&js)) * 2.0 - 1.0) * radius,
                    ));
                }
                emitChild(sub.system, spawn_pos, d.velocity, inherit_velocity, sub.inherit_position, child_depth);
                spawned_total += 1;
            }
        }
    }
}
