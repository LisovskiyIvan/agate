//! Tests for `compute_mode.zig` (moved verbatim from inline blocks;
//! v1, all headless — no sg.* below). Production keeps a
//! `test { _ = @import("compute_mode_tests.zig"); }` block (mesh.zig facade
//! pattern); the generated `agate/tests.zig` registry picks this file up
//! via `zig build update-tests`.
const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Vec3 = math.Vec3;
const Color4 = math.Color4;
const types = @import("types.zig");
const Particle = types.Particle;
const SimulationMode = types.SimulationMode;
const GpuParticleSlot = types.GpuParticleSlot;
const ComputeParticleState = types.ComputeParticleState;
const compute_workgroup_size = types.compute_workgroup_size;
const ParticleInstanceData = types.ParticleInstanceData;
const sampling = @import("sampling.zig");
const rotationToRadians = sampling.rotationToRadians;
const compute = @import("../compute.zig");
const pc_shd = @import("particle_compute_shader");

// --- Stateful compute mode tests (v1; all headless — no sg.* below) ---
//
// What these tests CANNOT verify (needs a live GPU context; the sandbox gets
// a `--test-*` fixture later): pipeline/shader creation, spawn upload bytes,
// the dispatch itself (integration, respawn, death culling), draw-buffer
// contents, and visual parity with `.cpu`/`.gpu`. Covered here instead:
// mode selection/gating, feature-matrix consistency, spawn staging/upload
// bookkeeping, buffer retire lifecycle, capacity/ring policy, and parameter
// validation.

test "compute workgroup layout matches the shader" {
    const sys = @import("system.zig");
    var ps = try sys.makeTestSystem(std.testing.allocator, 4);
    defer sys.freeTestSystem(&ps);
    // Six vec4 (see CState in shaders/particle_compute.glsl).
    try std.testing.expectEqual(@as(usize, 96), @sizeOf(ComputeParticleState));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(ComputeParticleState, "pos_age"));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(ComputeParticleState, "vel_life"));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(ComputeParticleState, "rot_seed"));
    try std.testing.expectEqual(@as(usize, 48), @offsetOf(ComputeParticleState, "color_start"));
    try std.testing.expectEqual(@as(usize, 64), @offsetOf(ComputeParticleState, "color_end"));
    try std.testing.expectEqual(@as(usize, 80), @offsetOf(ComputeParticleState, "size_size"));
    // Workgroup matches both the shader (`local_size_x = 64`) and the
    // compute.zig default (groupCount math below depends on it).
    try std.testing.expectEqual(compute.default_workgroup_size, compute_workgroup_size);
    // The draw buffer reuses the cpu instance stride exactly (same pipeline).
    try std.testing.expectEqual(@sizeOf(ParticleInstanceData), 4 * @sizeOf([4]f32));
    // Uniform block layout matches the shdc output byte-for-byte.
    try std.testing.expectEqual(@sizeOf(pc_shd.CsParams), @as(usize, 64));
}

test "compute mode selection gates explicitly, never silently" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&ps);

    // Headless selection (no sg context yet) stages the mode for a later
    // context-side latch — setup before init works.
    try ps.setSimulationMode(.compute);
    try std.testing.expectEqual(SimulationMode.compute, ps.simulation_mode);

    // Forced-unsupported override: explicit error, mode UNCHANGED (no silent
    // fallback to .cpu/.gpu, no partial mode flip).
    ps.simulation_mode = .cpu;
    ps.compute_support_override = false;
    try std.testing.expectError(error.ComputeUnsupported, ps.setSimulationMode(.compute));
    try std.testing.expectEqual(SimulationMode.cpu, ps.simulation_mode);
    // The same error surfaces from update (both entry points).
    ps.simulation_mode = .compute;
    try std.testing.expectError(error.ComputeUnsupported, ps.updateCompute(0.016));
    try std.testing.expectError(error.ComputeUnsupported, ps.update(0.016));
    // Nothing staged, ring untouched, no flush flagged.
    try std.testing.expectEqual(@as(usize, 0), ps.compute_staged);
    try std.testing.expectEqual(@as(usize, 0), ps.compute_high_water);
    try std.testing.expectEqual(@as(usize, 0), ps.active_count);
    try std.testing.expectEqual(false, ps.compute_flush_pending);

    // Zero capacity is a validation error, not a degenerate ring.
    var empty = try sys.makeTestSystem(a, 0);
    defer sys.freeTestSystem(&empty);
    try std.testing.expectError(error.InvalidCapacity, empty.setSimulationMode(.compute));
    try std.testing.expectEqual(SimulationMode.cpu, empty.simulation_mode);

    // Forced-available override restores the path.
    ps.compute_support_override = true;
    try ps.setSimulationMode(.compute);
    try ps.updateCompute(0.016);
    try std.testing.expect(ps.compute_staging.len == 4);

    // `.cpu`/`.gpu` selection never fails (frozen matrix: untouched).
    try ps.setSimulationMode(.cpu);
    try ps.setSimulationMode(.gpu);
    try std.testing.expectEqual(SimulationMode.gpu, ps.simulation_mode);
}

test "compute update rejects cpu-only features like gpu" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeComputeSystem(a, 4);
    defer sys.freeTestSystem(&ps);

    ps.local_space = true;
    try std.testing.expectError(error.LocalSpaceNeedsCpu, ps.updateCompute(0.016));
    try std.testing.expectError(error.LocalSpaceNeedsCpu, ps.update(0.016));
    try std.testing.expectEqual(SimulationMode.compute, ps.simulation_mode);
    try std.testing.expectEqual(@as(usize, 0), ps.compute_high_water);
    ps.local_space = false;

    try ps.updateCompute(0.0); // provisions staging
    var px = [_]u8{ 255, 128, 255, 255 };
    try ps.setFlowMap(null, &px, 1, 1);
    ps.flow_strength = 1.0;
    try std.testing.expectError(error.FlowMapNeedsCpu, ps.updateCompute(0.016));
    try std.testing.expectError(error.FlowMapNeedsCpu, ps.update(0.016));
    try std.testing.expectEqual(SimulationMode.compute, ps.simulation_mode);
    try std.testing.expectEqual(@as(usize, 0), ps.compute_high_water);
    ps.flow_strength = 0.0;
    try ps.update(0.016);
}

test "compute spawn staging shares the cpu sampler bit-for-bit" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var cpu = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&cpu);
    // Same defaults as cpu (mirrors the .gpu sampling test): identical seeds
    // must produce identical spawn records in any mode.
    var gpu = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&gpu);
    gpu.simulation_mode = .compute;
    gpu.emit_rate = 0.0; // manual emission only

    // Sentinel: the compute frame step must never touch the CPU arrays.
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
    gpu.particles[0] = sentinel;

    cpu.emitOne();
    try gpu.updateCompute(0.0); // provisions staging, no emission (rate 0)
    gpu.emitOne(); // staged record 0 at clock 0

    const p = cpu.particles[0];
    const s = gpu.compute_staging[0];
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
    try std.testing.expectApproxEqAbs(rotationToRadians(p.rotation), s.size_rotation[2], 1e-6);
    try std.testing.expectApproxEqAbs(rotationToRadians(p.angular_velocity), s.size_rotation[3], 1e-6);

    // Bookkeeping: one staged, cursor advanced, high-water + active alias.
    try std.testing.expectEqual(@as(usize, 1), gpu.compute_staged);
    try std.testing.expectEqual(@as(usize, 0), gpu.compute_stage_base);
    try std.testing.expectEqual(@as(usize, 1), gpu.compute_cursor);
    try std.testing.expectEqual(@as(usize, 1), gpu.compute_high_water);
    try std.testing.expectEqual(@as(usize, 1), gpu.active_count);
    // CPU arrays untouched; staging is sg-free (buffers stay zero headless).
    try std.testing.expectEqual(sentinel.age, gpu.particles[0].age);
    try std.testing.expectEqual(@as(u32, 0), gpu.compute_state_buffer.id);
}

test "compute update stages emission and flags the flush, headless-safe" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeComputeSystem(a, 8);
    defer sys.freeTestSystem(&ps);

    // One second at 60 p/s overflows the 8-slot ring: cursor wraps, staging
    // keeps the latest window via bulk eviction (see ring test for order).
    try ps.update(1.0);
    try std.testing.expectEqual(@as(usize, 8), ps.compute_high_water);
    try std.testing.expectEqual(@as(usize, 8), ps.active_count);
    try std.testing.expectEqual(@as(usize, 60 % 8), ps.compute_cursor);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ps.clock_seconds, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ps.compute_dt_accum, 1e-6);
    try std.testing.expectEqual(true, ps.compute_buffers_pending);
    try std.testing.expectEqual(true, ps.compute_flush_pending);
    // Headless flush: safe no-op, keeps every staged flag for a later retry
    // (upload discipline — no sg.* without a context).
    ps.flushGpuUploads();
    try std.testing.expectEqual(true, ps.compute_buffers_pending);
    try std.testing.expectEqual(true, ps.compute_flush_pending);
    try std.testing.expectEqual(@as(usize, 8), ps.compute_staged);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ps.compute_dt_accum, 1e-6);
}

test "compute ring recycles oldest; over-capacity frames keep the latest window" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeComputeSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    ps.emit_rate = 0.0;
    try ps.updateCompute(0.0); // provision

    // Stamp each emission with a distinct clock: the spawn time tags order.
    var i: usize = 0;
    while (i < 6) : (i += 1) {
        ps.clock_seconds = @floatFromInt(i);
        ps.emitOne();
    }
    try std.testing.expectEqual(@as(usize, 2), ps.compute_cursor);
    try std.testing.expectEqual(@as(usize, 4), ps.compute_high_water);
    try std.testing.expectEqual(@as(usize, 4), ps.active_count);
    // Six emissions into a 4-wide window: bulk eviction kept the latest four
    // (clocks 2..5) with the base advanced past the evicted pair.
    try std.testing.expectEqual(@as(usize, 4), ps.compute_staged);
    try std.testing.expectEqual(@as(usize, 2), ps.compute_stage_base);
    try std.testing.expectEqual(@as(f32, 2.0), ps.compute_staging[0].spawn_pos_time[3]);
    try std.testing.expectEqual(@as(f32, 3.0), ps.compute_staging[1].spawn_pos_time[3]);
    try std.testing.expectEqual(@as(f32, 4.0), ps.compute_staging[2].spawn_pos_time[3]);
    try std.testing.expectEqual(@as(f32, 5.0), ps.compute_staging[3].spawn_pos_time[3]);

    // Single-frame overflow via burst: same latest-window rule.
    var ps2 = try sys.makeComputeSystem(a, 4);
    defer sys.freeTestSystem(&ps2);
    ps2.emit_rate = 0.0;
    try ps2.updateCompute(0.0);
    ps2.burst(6);
    try std.testing.expectEqual(@as(usize, 2), ps2.compute_cursor);
    try std.testing.expectEqual(@as(usize, 4), ps2.compute_high_water);
    try std.testing.expectEqual(@as(usize, 4), ps2.compute_staged);

    // Reset re-anchors everything and stages a GPU state clear (consumed by
    // the next context flush; headless it stays staged).
    ps.reset();
    try std.testing.expectEqual(@as(usize, 0), ps.compute_cursor);
    try std.testing.expectEqual(@as(usize, 0), ps.compute_high_water);
    try std.testing.expectEqual(@as(usize, 0), ps.compute_staged);
    try std.testing.expectEqual(@as(usize, 0), ps.compute_stage_base);
    try std.testing.expectEqual(@as(usize, 0), ps.active_count);
    try std.testing.expectEqual(@as(f32, 0.0), ps.clock_seconds);
    try std.testing.expectEqual(@as(f32, 0.0), ps.compute_dt_accum);
    try std.testing.expectEqual(true, ps.compute_state_clear_pending);
}

test "compute dispatch groups and shader params are pure snapshots" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeComputeSystem(a, 130);
    defer sys.freeTestSystem(&ps);
    ps.emit_rate = 0.0;
    try ps.updateCompute(0.0);

    try std.testing.expectEqual(@as(usize, 0), ps.computeGroups());
    ps.burst(1);
    try std.testing.expectEqual(@as(usize, 1), ps.computeGroups());
    ps.burst(63);
    try std.testing.expectEqual(@as(usize, 1), ps.computeGroups());
    ps.burst(1); // 65 written -> 2 groups of 64
    try std.testing.expectEqual(@as(usize, 2), ps.computeGroups());
    try std.testing.expectEqual(ps.compute_high_water, ps.compute_staged);

    ps.gravity = Vec3.new(0.0, -9.8, 0.0);
    ps.drag = 1.5;
    ps.compute_dt_accum = 0.016;
    ps.spritesheet_columns = 4;
    ps.spritesheet_rows = 2;
    ps.spritesheet_loops = 3.0;
    const params = ps.buildComputeParams();
    try std.testing.expectEqual([4]f32{ 0.016, 1.5, 0.0, 0.0 }, params.dyn);
    try std.testing.expectEqual([4]f32{ 0.0, -9.8, 0.0, 0.0 }, params.grav);
    try std.testing.expectEqual([4]f32{ 4.0, 2.0, 3.0, 0.0 }, params.sheet);
    try std.testing.expectEqual([4]f32{ 130.0, 0.0, 65.0, 0.0 }, params.addr);
    try std.testing.expectEqual(@sizeOf(pc_shd.CsParams), @as(usize, 64));
}

test "compute takeGpuBuffersForRetire collects, zeroes, and never double-takes" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeComputeSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    // Fake borrowed ids (headless: never through sg.destroyBuffer here).
    ps.instance_buffer = .{ .id = 11 };
    ps.gpu_slot_buffer = .{ .id = 12 };
    ps.compute_state_buffer = .{ .id = 21 };
    ps.compute_spawn_buffer = .{ .id = 22 };
    ps.compute_draw_buffer = .{ .id = 23 };

    var out = [_]sg.Buffer{.{}} ** 8;
    const n = ps.takeGpuBuffersForRetire(&out);
    try std.testing.expectEqual(@as(usize, 5), n);
    try std.testing.expectEqual(@as(u32, 11), out[0].id);
    try std.testing.expectEqual(@as(u32, 12), out[1].id);
    try std.testing.expectEqual(@as(u32, 21), out[2].id);
    try std.testing.expectEqual(@as(u32, 22), out[3].id);
    try std.testing.expectEqual(@as(u32, 23), out[4].id);
    // Handles zeroed: deinit skips them, a second take finds nothing.
    try std.testing.expectEqual(@as(u32, 0), ps.instance_buffer.id);
    try std.testing.expectEqual(@as(u32, 0), ps.compute_state_buffer.id);
    try std.testing.expectEqual(@as(usize, 0), ps.takeGpuBuffersForRetire(&out));
    // Views/pipeline/shader with zero ids destroy as a safe no-op set
    // (guards only; no sg.* effect headless).
    ps.deinitComputeGpuObjects();
}

test "compute staging provision is idempotent and reset retains capacity" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeComputeSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.updateCompute(0.016);
    const first_ptr = ps.compute_staging.ptr;
    try ps.updateCompute(0.016); // no re-alloc
    try std.testing.expectEqual(first_ptr, ps.compute_staging.ptr);
    try std.testing.expectEqual(@as(usize, 4), ps.compute_staging.len);
    ps.emitOne();
    ps.reset();
    // Allocation retained (like the prepared-frame lists), content dropped.
    try std.testing.expectEqual(@as(usize, 4), ps.compute_staging.len);
    try std.testing.expectEqual(@as(usize, 0), ps.compute_staged);
}

test "cpu/gpu/compute full-system policies stay distinct" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var cpu = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&cpu);
    var gpu = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&gpu);
    gpu.simulation_mode = .gpu;
    gpu.gpu_slots = try a.alloc(GpuParticleSlot, 4);
    var cmp = try sys.makeComputeSystem(a, 4);
    defer sys.freeTestSystem(&cmp);
    cmp.emit_rate = 0.0;
    try cmp.updateCompute(0.0);

    cpu.burst(6);
    gpu.burst(6);
    cmp.burst(6);
    // CPU drops when full; both GPU rings recycle the oldest slot.
    try std.testing.expectEqual(@as(usize, 4), cpu.active_count);
    try std.testing.expectEqual(@as(usize, 4), gpu.active_count);
    try std.testing.expectEqual(@as(usize, 2), gpu.gpu_write_cursor);
    try std.testing.expectEqual(@as(usize, 4), cmp.active_count);
    try std.testing.expectEqual(@as(usize, 2), cmp.compute_cursor);
    try std.testing.expectEqual(@as(usize, 4), cmp.compute_staged);
}

test "sub-emitter spawns into a compute child staged window" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var parent = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&parent);
    var child = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&child);
    child.simulation_mode = .compute;
    child.lifetime_min = 10.0;
    child.lifetime_max = 10.0;
    try child.updateCompute(0.0); // provision staging, no emission

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
    // One record staged at the death position with the inherited velocity
    // (0.5 * (4,0,0) + 0.5 * sampled(0,0,0) = (2,0,0)).
    try std.testing.expectEqual(@as(usize, 1), child.compute_high_water);
    try std.testing.expectEqual(@as(f32, 1.0), child.compute_staging[0].spawn_pos_time[0]);
    try std.testing.expectEqual(@as(f32, 2.0), child.compute_staging[0].spawn_pos_time[1]);
    try std.testing.expectEqual(@as(f32, 3.0), child.compute_staging[0].spawn_pos_time[2]);
    try std.testing.expectEqual(@as(f32, 2.0), child.compute_staging[0].velocity_lifetime[0]);
    try std.testing.expectEqual(@as(f32, 10.0), child.compute_staging[0].velocity_lifetime[3]);
}
