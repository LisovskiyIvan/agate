const std = @import("std");
const math = @import("math");
const sys = @import("system.zig");
const types = @import("types.zig");
const sampling = @import("sampling.zig");
const gpu = @import("gpu.zig");

const Vec3 = math.Vec3;
const Color4 = math.Color4;
const GpuParticleSlot = types.GpuParticleSlot;
const Particle = types.Particle;
const SimulationMode = types.SimulationMode;
const gpuUploadRange = gpu.gpuUploadRange;
const rotationToRadians = sampling.rotationToRadians;

test "gpu slot layout matches shader attrs" {
    var ps = try sys.makeTestSystem(std.testing.allocator, 4);
    defer sys.freeTestSystem(&ps);
    try std.testing.expectEqual(@as(usize, 80), @sizeOf(GpuParticleSlot));
}

test "gpu spawn sampling matches cpu with the same seed" {
    const a = std.testing.allocator;
    var cpu = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&cpu);
    var gpu_sys = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&gpu_sys);
    gpu_sys.gpu_slots = try a.alloc(GpuParticleSlot, 4);
    gpu_sys.simulation_mode = .gpu;
    cpu.emitOne();
    gpu_sys.emitOne();

    const p = cpu.particles[0];
    const s = gpu_sys.gpu_slots[0];
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
    try std.testing.expectEqual(@as(usize, 1), gpu_sys.active_count);
    try std.testing.expectEqual(@as(usize, 1), gpu_sys.gpu_high_water);
}

test "gpu ring wraps and recycles oldest slots" {
    const a = std.testing.allocator;
    var ps = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    ps.gpu_slots = try a.alloc(GpuParticleSlot, 4);
    ps.simulation_mode = .gpu;
    var i: usize = 0;
    while (i < 6) : (i += 1) {
        ps.clock_seconds = @floatFromInt(i);
        ps.emitOne();
    }
    try std.testing.expectEqual(@as(usize, 2), ps.gpu_write_cursor);
    try std.testing.expectEqual(@as(usize, 4), ps.gpu_high_water);
    try std.testing.expectEqual(@as(usize, 4), ps.active_count);
    try std.testing.expectEqual(@as(f32, 4.0), ps.gpu_slots[0].spawn_pos_time[3]);
    try std.testing.expectEqual(@as(f32, 5.0), ps.gpu_slots[1].spawn_pos_time[3]);
    try std.testing.expectEqual(@as(f32, 2.0), ps.gpu_slots[2].spawn_pos_time[3]);
    try std.testing.expectEqual(@as(f32, 3.0), ps.gpu_slots[3].spawn_pos_time[3]);
    try std.testing.expectEqual(true, ps.gpu_dirty);
    try std.testing.expectEqual(true, ps.gpu_dirty_wrapped);
    try std.testing.expectEqual(@as(usize, 4), gpuUploadRange(&ps).?.len);

    ps.gpu_dirty = false;
    ps.gpu_dirty_wrapped = false;
    ps.clock_seconds = 6.0;
    ps.emitOne();
    try std.testing.expectEqual(false, ps.gpu_dirty_wrapped);
    const range = gpuUploadRange(&ps).?;
    try std.testing.expectEqual(@as(usize, 1), range.len);
    try std.testing.expectEqual(@as(f32, 6.0), range[0].spawn_pos_time[3]);
    try std.testing.expectEqual(@as(usize, 2), ps.gpu_dirty_start);
    try std.testing.expectEqual(@as(usize, 3), ps.gpu_dirty_end);

    ps.reset();
    try std.testing.expectEqual(@as(usize, 0), ps.gpu_write_cursor);
    try std.testing.expectEqual(@as(usize, 0), ps.gpu_high_water);
    try std.testing.expectEqual(@as(usize, 0), ps.active_count);
    try std.testing.expectEqual(@as(f32, 0.0), ps.clock_seconds);
    try std.testing.expectEqual(false, ps.gpu_dirty);
}

test "gpu burst overflows the ring instead of dropping" {
    const a = std.testing.allocator;
    var ps = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    ps.gpu_slots = try a.alloc(GpuParticleSlot, 4);
    ps.simulation_mode = .gpu;
    ps.burst(6);
    try std.testing.expectEqual(@as(usize, 2), ps.gpu_write_cursor);
    try std.testing.expectEqual(@as(usize, 4), ps.active_count);
    try std.testing.expectEqual(true, ps.gpu_dirty_wrapped);

    var cpu = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&cpu);
    cpu.burst(6);
    try std.testing.expectEqual(@as(usize, 4), cpu.active_count);
}

test "gpu update does no per-particle cpu work" {
    const a = std.testing.allocator;
    var ps = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    ps.simulation_mode = .gpu;
    ps.is_emitting = true;
    ps.emit_rate = 60.0;
    ps.lifetime_min = 1.0;
    ps.lifetime_max = 1.0;
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
    try std.testing.expectEqual(@as(usize, 4), ps.gpu_high_water);
    try std.testing.expectEqual(@as(usize, 4), ps.active_count);
    try std.testing.expectEqual(@as(usize, 0), ps.gpu_write_cursor);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ps.clock_seconds, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ps.emit_accumulator, 1e-5);
}

test "gpu with local_space is an explicit error, not a downgrade" {
    const a = std.testing.allocator;
    var ps = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    ps.simulation_mode = .gpu;
    ps.local_space = true;
    try std.testing.expectError(error.LocalSpaceNeedsCpu, ps.updateGpu(0.016));
    try std.testing.expectEqual(SimulationMode.gpu, ps.simulation_mode);
    try std.testing.expectEqual(@as(usize, 0), ps.active_count);
    try std.testing.expectEqual(@as(usize, 0), ps.gpu_high_water);
}

test "gpu update stages buffer creation without an sg context" {
    const a = std.testing.allocator;
    var ps = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    ps.simulation_mode = .gpu;
    ps.is_emitting = true;
    try ps.update(0.016);
    try std.testing.expectEqual(true, ps.gpu_slot_buffer_pending);
    try std.testing.expectEqual(@as(u32, 0), ps.gpu_slot_buffer.id);
    try std.testing.expectEqual(true, ps.gpu_flush_pending);
    ps.flushGpuUploads();
    try std.testing.expectEqual(true, ps.gpu_slot_buffer_pending);
    try std.testing.expectEqual(@as(u32, 0), ps.gpu_slot_buffer.id);
}
