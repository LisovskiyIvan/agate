//! Sub-emitters: CPU on-death child-spawn rules. Split out of `particles.zig`
//! (facade).
//!
//! `emitChild` / `fireSubEmitters` take the system as `anytype` so this module
//! never imports `system.zig` or the facade back — same discipline as
//! `profiler/*`. The `SubEmitter` rule type itself stays in `system.zig` next
//! to its owner (its `system` field pins it to `*ParticleSystem`); the chain
//! bounds live in `types.zig`. Child spawns reach the `.gpu` / `.compute`
//! rings via sibling imports (no cycle: neither sibling imports back).
//! Moved tests reach `system.zig` helpers through block-scoped imports that
//! exist only in test builds.

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

// --- Sub-emitter tests (CPU on-death triggers; headless, deterministic) ---

test "sub-emitter probability 0 never fires, 1 always fires" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var parent = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&parent);
    var child = try sys.makeQuiescentSystem(a, 8);
    defer sys.freeTestSystem(&child);
    child.lifetime_min = 10.0;
    child.lifetime_max = 10.0;

    parent.addSubEmitter(.{ .system = &child, .probability = 0.0, .count = 3 });
    parent.burst(2);
    parent.updateCpu(1.0);
    // Deaths compact away, the zero-probability rule spawns nothing.
    try std.testing.expectEqual(@as(usize, 0), parent.active_count);
    try std.testing.expectEqual(@as(usize, 0), child.active_count);

    parent.sub_emitter_store[0].probability = 1.0;
    parent.burst(2);
    parent.updateCpu(1.0);
    // Both deaths fire count=3 each.
    try std.testing.expectEqual(@as(usize, 0), parent.active_count);
    try std.testing.expectEqual(@as(usize, 6), child.active_count);
    // Children are fresh (age 0) with the child system's long lifetime.
    for (child.particles[0..child.active_count]) |p| {
        try std.testing.expectEqual(@as(f32, 0.0), p.age);
        try std.testing.expectEqual(@as(f32, 10.0), p.lifetime);
        try std.testing.expectEqual(@as(u8, 1), p.sub_depth);
    }
}

test "sub-emitter inherits death position and blended velocity" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var parent = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&parent);
    var child = try sys.makeQuiescentSystem(a, 8);
    defer sys.freeTestSystem(&child);
    child.lifetime_min = 10.0;
    child.lifetime_max = 10.0;
    child.emitter_position = Vec3.new(9.0, 9.0, 9.0);

    parent.addSubEmitter(.{
        .system = &child,
        .probability = 1.0,
        .count = 1,
        .inherit_velocity = 0.5,
        .inherit_position = true,
        .spawn_radius = 0.0,
    });
    // Hand-placed death: position/velocity known exactly. The death tick ages
    // past lifetime WITHOUT integrating position, so the death position is
    // the stored one verbatim.
    parent.particles[0] = .{
        .position = Vec3.new(1.0, 2.0, 3.0),
        .velocity = Vec3.new(4.0, 0.0, 0.0),
        .size = 0.2,
        .size_end = 0.0,
        .color = Color4.new(1.0, 1.0, 1.0, 1.0),
        .color_end = Color4.new(1.0, 1.0, 1.0, 0.0),
        .age = 0.0,
        .lifetime = 0.5,
    };
    parent.active_count = 1;
    parent.updateCpu(1.0);
    try std.testing.expectEqual(@as(usize, 1), child.active_count);
    // Death position inherited verbatim (radius 0); velocity blended
    // 0.5 * parent + 0.5 * sampled(0,0,0) = (2,0,0).
    try std.testing.expectEqual(Vec3.new(1.0, 2.0, 3.0), child.particles[0].position);
    try std.testing.expectEqual(Vec3.new(2.0, 0.0, 0.0), child.particles[0].velocity);

    // Full inheritance keeps the whole parent velocity.
    parent.sub_emitter_store[0].inherit_velocity = 1.0;
    parent.particles[0] = child.particles[0];
    parent.particles[0].age = 0.0;
    parent.particles[0].lifetime = 0.5;
    parent.particles[0].sub_depth = 0;
    parent.active_count = 1;
    child.active_count = 0;
    parent.updateCpu(1.0);
    try std.testing.expectEqual(Vec3.new(2.0, 0.0, 0.0), child.particles[0].velocity);

    // Zero inheritance uses the child system's own sampled velocity (0 here).
    parent.sub_emitter_store[0].inherit_velocity = 0.0;
    parent.particles[0].velocity = Vec3.new(4.0, 0.0, 0.0);
    parent.particles[0].age = 0.0;
    parent.active_count = 1;
    child.active_count = 0;
    parent.updateCpu(1.0);
    try std.testing.expectEqual(Vec3.zero, child.particles[0].velocity);

    // inherit_position=false keeps the child system's sampled spawn (9,9,9).
    parent.sub_emitter_store[0].inherit_position = false;
    parent.particles[0].age = 0.0;
    parent.active_count = 1;
    child.active_count = 0;
    parent.updateCpu(1.0);
    try std.testing.expectEqual(Vec3.new(9.0, 9.0, 9.0), child.particles[0].position);

    // Radius jitter stays inside the cube half-extent per component.
    parent.sub_emitter_store[0].inherit_position = true;
    parent.sub_emitter_store[0].spawn_radius = 2.0;
    parent.particles[0].age = 0.0;
    parent.active_count = 1;
    child.active_count = 0;
    parent.updateCpu(1.0);
    const jp = child.particles[0].position;
    try std.testing.expect(@abs(jp.x - 1.0) <= 2.0);
    try std.testing.expect(@abs(jp.y - 2.0) <= 2.0);
    try std.testing.expect(@abs(jp.z - 3.0) <= 2.0);
}

test "sub-emitter cycle A->B->A extinguishes at the depth bound" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var sys_a = try sys.makeQuiescentSystem(a, 32);
    defer sys.freeTestSystem(&sys_a);
    var sys_b = try sys.makeQuiescentSystem(a, 32);
    defer sys.freeTestSystem(&sys_b);
    sys_a.lifetime_min = 0.25;
    sys_a.lifetime_max = 0.25;
    sys_b.lifetime_min = 0.25;
    sys_b.lifetime_max = 0.25;
    sys_a.addSubEmitter(.{ .system = &sys_b, .probability = 1.0, .count = 2 });
    sys_b.addSubEmitter(.{ .system = &sys_a, .probability = 1.0, .count = 2 });

    sys_a.burst(1);
    var max_depth_seen: u8 = 0;
    var frame: usize = 0;
    while (frame < 30) : (frame += 1) {
        sys_a.updateCpu(0.5);
        sys_b.updateCpu(0.5);
        // Bounded throughout: never exceeds capacity, never hangs.
        try std.testing.expect(sys_a.active_count <= 32);
        try std.testing.expect(sys_b.active_count <= 32);
        for (sys_a.particles[0..sys_a.active_count]) |p| max_depth_seen = @max(max_depth_seen, p.sub_depth);
        for (sys_b.particles[0..sys_b.active_count]) |p| max_depth_seen = @max(max_depth_seen, p.sub_depth);
    }
    // The chain fired (generations beyond the seed existed) but the depth
    // rule extinguished it: depth-4 deaths spawn nothing, so both systems
    // drain instead of ping-ponging forever.
    try std.testing.expect(max_depth_seen > 0);
    try std.testing.expect(max_depth_seen <= max_sub_emitter_depth);
    try std.testing.expectEqual(@as(usize, 0), sys_a.active_count);
    try std.testing.expectEqual(@as(usize, 0), sys_b.active_count);
}

test "sub-emitter self-cycle terminates at the depth bound" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 64);
    defer sys.freeTestSystem(&ps);
    ps.lifetime_min = 0.25;
    ps.lifetime_max = 0.25;
    ps.addSubEmitter(.{ .system = &ps, .probability = 1.0, .count = 1 });

    ps.burst(1);
    var max_depth_seen: u8 = 0;
    var frame: usize = 0;
    while (frame < 15) : (frame += 1) {
        ps.updateCpu(0.5);
        try std.testing.expect(ps.active_count <= 64);
        for (ps.particles[0..ps.active_count]) |p| max_depth_seen = @max(max_depth_seen, p.sub_depth);
    }
    // One seed -> one depth-1 child -> ... -> depth-4 deaths spawn nothing.
    try std.testing.expectEqual(max_sub_emitter_depth, max_depth_seen);
    try std.testing.expectEqual(@as(usize, 0), ps.active_count);
}

test "sub-emitter plumbing is bit-identical when nothing fires" {
    const sys = @import("system.zig");
    const ParticleSystem = sys.ParticleSystem;
    const a = std.testing.allocator;
    var baseline = try sys.makeTestSystem(a, 64);
    defer sys.freeTestSystem(&baseline);
    var plumbed = try sys.makeTestSystem(a, 64);
    defer sys.freeTestSystem(&plumbed);
    var child = try sys.makeQuiescentSystem(a, 64);
    defer sys.freeTestSystem(&child);
    child.lifetime_min = 10.0;
    child.lifetime_max = 10.0;
    for ([2]*ParticleSystem{ &baseline, &plumbed }) |ps| {
        ps.gravity = Vec3.new(0.0, -3.0, 0.0);
        ps.emit_rate = 120.0;
        ps.is_emitting = true;
        ps.lifetime_min = 0.5;
        ps.lifetime_max = 1.0;
    }
    // Attached but probability 0: the record+roll path executes on every
    // death yet must perturb neither the parent stream nor the child.
    plumbed.addSubEmitter(.{ .system = &child, .probability = 0.0, .count = 2 });

    var frame: usize = 0;
    while (frame < 60) : (frame += 1) {
        baseline.updateCpu(1.0 / 60.0);
        plumbed.updateCpu(1.0 / 60.0);
        try std.testing.expectEqual(baseline.active_count, plumbed.active_count);
        const live = baseline.active_count;
        try std.testing.expect(live > 0);
        try std.testing.expect(std.mem.eql(
            u8,
            std.mem.sliceAsBytes(baseline.particles[0..live]),
            std.mem.sliceAsBytes(plumbed.particles[0..live]),
        ));
        try std.testing.expect(std.mem.eql(
            u8,
            std.mem.sliceAsBytes(baseline.instances[0..live]),
            std.mem.sliceAsBytes(plumbed.instances[0..live]),
        ));
    }
    // Deaths happened (the roll path ran) but the child stayed empty.
    try std.testing.expect(baseline.sub_tick > 0);
    try std.testing.expectEqual(plumbed.sub_tick, baseline.sub_tick);
    try std.testing.expectEqual(@as(usize, 0), child.active_count);
}

test "sub-emitter per-tick spawn bound" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var parent = try sys.makeQuiescentSystem(a, 1024);
    defer sys.freeTestSystem(&parent);
    var child = try sys.makeQuiescentSystem(a, 4096);
    defer sys.freeTestSystem(&child);
    child.lifetime_min = 10.0;
    child.lifetime_max = 10.0;
    // 512 deaths x count 4 = 2048 potential spawns, child capacity is not
    // the limiter (4096) — the per-tick bound is.
    parent.addSubEmitter(.{ .system = &child, .probability = 1.0, .count = 4 });
    parent.burst(512);
    try std.testing.expectEqual(@as(usize, 512), parent.active_count);
    parent.updateCpu(1.0);
    try std.testing.expectEqual(@as(usize, 0), parent.active_count);
    try std.testing.expectEqual(max_sub_emitter_spawns_per_tick, child.active_count);
}

test "sub-emitter spawns into a gpu child slot ring" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var parent = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&parent);
    var child = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&child);
    child.simulation_mode = .gpu;
    child.lifetime_min = 10.0;
    child.lifetime_max = 10.0;
    // Provision the ring without emitting (steady clock, no emission).
    try child.updateGpu(0.0);

    parent.addSubEmitter(.{ .system = &child, .probability = 1.0, .count = 1 });
    parent.particles[0] = .{
        .position = Vec3.new(1.0, 2.0, 3.0),
        .velocity = Vec3.new(4.0, 0.0, 0.0),
        .size = 0.2,
        .size_end = 0.0,
        .color = Color4.new(1.0, 1.0, 1.0, 1.0),
        .color_end = Color4.new(1.0, 1.0, 1.0, 0.0),
        .age = 0.0,
        .lifetime = 0.5,
    };
    parent.active_count = 1;
    parent.updateCpu(1.0);
    // One slot written at the death position with the inherited velocity
    // (0.5 * (4,0,0) + 0.5 * sampled(0,0,0) = (2,0,0)).
    try std.testing.expectEqual(@as(usize, 1), child.gpu_high_water);
    try std.testing.expectEqual(@as(f32, 1.0), child.gpu_slots[0].spawn_pos_time[0]);
    try std.testing.expectEqual(@as(f32, 2.0), child.gpu_slots[0].spawn_pos_time[1]);
    try std.testing.expectEqual(@as(f32, 3.0), child.gpu_slots[0].spawn_pos_time[2]);
    try std.testing.expectEqual(@as(f32, 2.0), child.gpu_slots[0].velocity_lifetime[0]);
    try std.testing.expectEqual(@as(f32, 0.0), child.gpu_slots[0].velocity_lifetime[1]);
    try std.testing.expectEqual(@as(f32, 10.0), child.gpu_slots[0].velocity_lifetime[3]);
}
