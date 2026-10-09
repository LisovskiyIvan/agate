const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const particles = @import("../particles.zig");
const ParticleSystem = particles.ParticleSystem;
const Texture = @import("../texture.zig").Texture;
const ParticlePass = @import("particle_pass.zig").ParticlePass;

fn makePassTestSystem(allocator: std.mem.Allocator, capacity: usize) !ParticleSystem {
    const parts = try allocator.alloc(particles.Particle, capacity);
    errdefer allocator.free(parts);
    const insts = try allocator.alloc(particles.ParticleInstanceData, capacity);
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
        .instance_buffer = .{ .id = 11 },
        .prng = std.Random.DefaultPrng.init(42),
    };
}

fn freePassTestSystem(ps: *ParticleSystem) void {
    if (ps.gpu_slots.len > 0) ps.allocator.free(ps.gpu_slots);
    ps.allocator.free(ps.particles);
    ps.allocator.free(ps.instances);
    if (ps.alive_scratch.len > 0) ps.allocator.free(ps.alive_scratch);
}

test "ParticleDraw.fromSystem copies values, never live references" {
    const t = std.testing;
    const math_mod = @import("math");
    var ps = try makePassTestSystem(t.allocator, 4);
    defer freePassTestSystem(&ps);
    ps.active_count = 3;
    ps.simulation_mode = .gpu;
    ps.instance_buffer = .{ .id = 11 };
    ps.gpu_slot_buffer = .{ .id = 12 };
    ps.blend_mode = .alpha_blend;
    var tex: Texture = std.mem.zeroes(Texture);
    tex.view = .{ .id = 77 };
    ps.texture = tex;
    ps.clock_seconds = 1.5;
    ps.drag = 2.0;
    ps.gravity = math_mod.Vec3.new(1.0, -2.0, 3.0);
    ps.spritesheet_columns = 4;
    ps.spritesheet_rows = 2;
    ps.spritesheet_loops = 3.0;

    const draw = ParticlePass.ParticleDraw.fromSystem(&ps);
    try t.expectEqual(@as(usize, 3), draw.active_count);
    try t.expectEqual(particles.SimulationMode.gpu, draw.simulation_mode);
    try t.expectEqual(@as(u32, 11), draw.instance_buffer.id);
    try t.expectEqual(@as(u32, 12), draw.gpu_slot_buffer.id);
    try t.expectEqual(particles.ParticleBlendMode.alpha_blend, draw.blend_mode);
    try t.expect(draw.texture_view != null);
    try t.expectEqual(@as(u32, 77), draw.texture_view.?.id);
    try t.expectEqual(@as(f32, 1.5), draw.clock_seconds);
    try t.expectEqual(@as(f32, 2.0), draw.drag);
    try t.expectEqual(math_mod.Vec3.new(1.0, -2.0, 3.0), draw.gravity);
    try t.expectEqual(@as(u32, 4), draw.spritesheet_columns);
    try t.expectEqual(@as(u32, 2), draw.spritesheet_rows);
    try t.expectEqual(@as(f32, 3.0), draw.spritesheet_loops);

    // No texture -> null (pass substitutes its default dot at draw).
    ps.texture = null;
    try t.expectEqual(@as(?sg.View, null), ParticlePass.ParticleDraw.fromSystem(&ps).texture_view);

    // Mutating the live system leaves the earlier snapshot untouched.
    ps.active_count = 0;
    ps.clock_seconds = 9.0;
    try t.expectEqual(@as(usize, 3), draw.active_count);
    try t.expectEqual(@as(f32, 1.5), draw.clock_seconds);
}

test "ParticleDraw.drawBuffer follows the simulation mode" {
    const t = std.testing;
    var cpu = ParticlePass.ParticleDraw{
        .simulation_mode = .cpu,
        .instance_buffer = .{ .id = 11 },
        .gpu_slot_buffer = .{ .id = 12 },
    };
    try t.expectEqual(@as(u32, 11), cpu.drawBuffer().id);
    cpu.simulation_mode = .gpu;
    try t.expectEqual(@as(u32, 12), cpu.drawBuffer().id);
}

test "statsForDraws preserves the legacy count semantics" {
    const t = std.testing;
    // Zero-count draws contribute nothing (render skips them and the
    // stats loop only counts active_count > 0).
    const draws = [_]ParticlePass.ParticleDraw{
        .{ .active_count = 0 },
        .{ .active_count = 3 },
        .{ .active_count = 5 },
    };
    const s = ParticlePass.statsForDraws(&draws);
    try t.expectEqual(@as(u32, 2), s.draw_calls);
    try t.expectEqual(@as(u32, 2 * (3 + 5)), s.triangles);
    const empty: []const ParticlePass.ParticleDraw = &[_]ParticlePass.ParticleDraw{};
    try t.expectEqual(ParticlePass.DrawStats{}, ParticlePass.statsForDraws(empty));
}

test "ParticleDraw.compute mode binds the baked buffer, keeps cpu visuals" {
    const t = std.testing;
    const math_mod = @import("math");
    var ps = try makePassTestSystem(t.allocator, 4);
    defer freePassTestSystem(&ps);
    ps.simulation_mode = .compute;
    ps.active_count = 3;
    ps.instance_buffer = .{ .id = 11 };
    ps.gpu_slot_buffer = .{ .id = 12 };
    ps.compute_draw_buffer = .{ .id = 13 };
    ps.blend_mode = .alpha_blend;
    ps.color_start = math_mod.Color4.new(1.0, 0.0, 0.0, 1.0);
    ps.size_start = 0.5;

    const draw = ParticlePass.ParticleDraw.fromSystem(&ps);
    try t.expectEqual(particles.SimulationMode.compute, draw.simulation_mode);
    try t.expectEqual(@as(u32, 13), draw.compute_draw_buffer.id);
    // Mode-selected buffer: compute binds its baked instances (cpu-pipeline
    // stride), cpu/gpu selections unchanged.
    try t.expectEqual(@as(u32, 13), draw.drawBuffer().id);
    var cpu_draw = draw;
    cpu_draw.simulation_mode = .cpu;
    try t.expectEqual(@as(u32, 11), cpu_draw.drawBuffer().id);
    var gpu_draw = draw;
    gpu_draw.simulation_mode = .gpu;
    try t.expectEqual(@as(u32, 12), gpu_draw.drawBuffer().id);
    // Stats keep the count semantics in every mode.
    const s = ParticlePass.statsForDraws(&[_]ParticlePass.ParticleDraw{draw});
    try t.expectEqual(@as(u32, 1), s.draw_calls);
    try t.expectEqual(@as(u32, 2 * 3), s.triangles);
}
