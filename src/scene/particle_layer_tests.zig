const std = @import("std");
const math_mod = @import("math");
const passes = @import("../passes/mod.zig");
const particles = @import("../particles.zig");
const ParticleSystem = particles.ParticleSystem;
const particle_layer = @import("particle_layer.zig");
const ParticleLayer = particle_layer.ParticleLayer;
const ParticleDraw = particle_layer.ParticleDraw;
const stats_mod = @import("stats.zig");
const Camera = @import("../camera.zig").Camera;
const cam_mod = @import("../camera.zig");
const meter = @import("../gpu_upload_meter.zig");
const TestTexture = @import("../texture.zig").Texture;

fn makeLayerTestSystem(allocator: std.mem.Allocator, capacity: usize) !ParticleSystem {
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

fn freeLayerTestSystem(ps: *ParticleSystem) void {
    if (ps.gpu_slots.len > 0) ps.allocator.free(ps.gpu_slots);
    ps.allocator.free(ps.particles);
    ps.allocator.free(ps.instances);
    if (ps.alive_scratch.len > 0) ps.allocator.free(ps.alive_scratch);
}

test "particle ParticleDraw is a plain value record (no pointers)" {
    comptime {
        for (std.meta.fields(ParticleDraw)) |f| {
            switch (@typeInfo(f.type)) {
                .pointer => @compileError("ParticleDraw must stay a plain value record"),
                .optional => |o| {
                    switch (@typeInfo(o.child)) {
                        .pointer => @compileError("ParticleDraw optional must not wrap a pointer"),
                        else => {},
                    }
                },
                else => {},
            }
        }
    }
    try std.testing.expect(ParticleDraw == passes.ParticlePass.ParticleDraw);
}

test "particle captureFrame packet is immutable under live mutations" {
    const t = std.testing;
    var layer: ParticleLayer = .{ .pass = undefined };
    defer layer.systems.deinit(t.allocator);
    defer layer.frame.deinit(t.allocator);

    var cpu = try makeLayerTestSystem(t.allocator, 4);
    defer freeLayerTestSystem(&cpu);
    cpu.active_count = 3;
    cpu.blend_mode = .additive;
    var tex: TestTexture = std.mem.zeroes(TestTexture);
    tex.view = .{ .id = 77 };
    cpu.texture = tex;

    var gpu = try makeLayerTestSystem(t.allocator, 4);
    defer freeLayerTestSystem(&gpu);
    gpu.simulation_mode = .gpu;
    gpu.active_count = 2;
    gpu.instance_buffer = .{ .id = 21 };
    gpu.gpu_slot_buffer = .{ .id = 22 };
    gpu.blend_mode = .alpha_blend;
    gpu.clock_seconds = 1.5;
    gpu.drag = 2.0;
    gpu.gravity = math_mod.Vec3.new(1.0, -2.0, 3.0);
    gpu.spritesheet_columns = 4;
    gpu.spritesheet_rows = 2;
    gpu.spritesheet_loops = 3.0;

    try layer.systems.append(t.allocator, &cpu);
    try layer.systems.append(t.allocator, &gpu);
    layer.captureFrame(t.allocator);
    try t.expectEqual(@as(usize, 2), layer.frame.items.len);

    const cpu_draw = layer.frame.items[0];
    try t.expectEqual(@as(usize, 3), cpu_draw.active_count);
    try t.expectEqual(particles.SimulationMode.cpu, cpu_draw.simulation_mode);
    try t.expectEqual(@as(u32, 11), cpu_draw.instance_buffer.id);
    try t.expectEqual(particles.ParticleBlendMode.additive, cpu_draw.blend_mode);
    try t.expectEqual(@as(u32, 77), cpu_draw.texture_view.?.id);

    const gpu_draw = layer.frame.items[1];
    try t.expectEqual(@as(usize, 2), gpu_draw.active_count);
    try t.expectEqual(particles.SimulationMode.gpu, gpu_draw.simulation_mode);
    try t.expectEqual(@as(u32, 22), gpu_draw.gpu_slot_buffer.id);
    try t.expectEqual(@as(f32, 1.5), gpu_draw.clock_seconds);
    try t.expectEqual(@as(f32, 2.0), gpu_draw.drag);
    try t.expectEqual(math_mod.Vec3.new(1.0, -2.0, 3.0), gpu_draw.gravity);
    try t.expectEqual(@as(u32, 4), gpu_draw.spritesheet_columns);

    // Mutate live fields
    cpu.active_count = 1;
    cpu.instance_buffer = .{ .id = 99 };
    cpu.texture = null;
    gpu.active_count = 4;
    gpu.simulation_mode = .cpu;
    gpu.gpu_slot_buffer = .{ .id = 98 };
    gpu.texture = tex;
    gpu.clock_seconds = 9.0;
    gpu.gravity = math_mod.Vec3.zero;
    gpu.spritesheet_columns = 1;

    try t.expectEqual(@as(usize, 2), layer.frame.items.len);
    try t.expectEqual(@as(usize, 3), layer.frame.items[0].active_count);
    try t.expectEqual(@as(u32, 11), layer.frame.items[0].instance_buffer.id);
    try t.expectEqual(@as(u32, 77), layer.frame.items[0].texture_view.?.id);
    try t.expectEqual(particles.SimulationMode.gpu, layer.frame.items[1].simulation_mode);
    try t.expectEqual(@as(usize, 2), layer.frame.items[1].active_count);
    try t.expectEqual(@as(u32, 22), layer.frame.items[1].gpu_slot_buffer.id);
    try t.expectEqual(@as(f32, 1.5), layer.frame.items[1].clock_seconds);
    try t.expectEqual(math_mod.Vec3.new(1.0, -2.0, 3.0), layer.frame.items[1].gravity);
    try t.expectEqual(@as(u32, 4), layer.frame.items[1].spritesheet_columns);

    var stats = stats_mod.SceneStats{};
    const s = passes.ParticlePass.statsForDraws(layer.frame.items);
    stats.main_draw_calls += s.draw_calls;
    stats.draw_calls += s.draw_calls;
    stats.triangles += s.triangles;
    try t.expectEqual(@as(u32, 2), stats.draw_calls);
    try t.expectEqual(@as(u32, 2), stats.main_draw_calls);
    try t.expectEqual(@as(u32, 2 * (3 + 2)), stats.triangles);
}

test "particle captureFrame empty paths stay coherent" {
    const t = std.testing;
    var layer: ParticleLayer = .{ .pass = undefined };
    defer layer.systems.deinit(t.allocator);
    defer layer.frame.deinit(t.allocator);

    layer.captureFrame(t.allocator);
    try t.expectEqual(@as(usize, 0), layer.frame.items.len);

    var ps = try makeLayerTestSystem(t.allocator, 4);
    defer freeLayerTestSystem(&ps);
    ps.active_count = 0;
    try layer.systems.append(t.allocator, &ps);
    layer.captureFrame(t.allocator);
    try t.expectEqual(@as(usize, 0), layer.frame.items.len);

    ps.active_count = 2;
    layer.captureFrame(t.allocator);
    try t.expectEqual(@as(usize, 1), layer.frame.items.len);
    ps.active_count = 0;
    layer.captureFrame(t.allocator);
    try t.expectEqual(@as(usize, 0), layer.frame.items.len);
}

test "particle captureFrame reuses retained capacity" {
    const t = std.testing;
    var layer: ParticleLayer = .{ .pass = undefined };
    defer layer.systems.deinit(t.allocator);
    defer layer.frame.deinit(t.allocator);

    var a = try makeLayerTestSystem(t.allocator, 8);
    defer freeLayerTestSystem(&a);
    var b = try makeLayerTestSystem(t.allocator, 8);
    defer freeLayerTestSystem(&b);
    a.active_count = 5;
    b.active_count = 5;
    try layer.systems.append(t.allocator, &a);
    try layer.systems.append(t.allocator, &b);
    layer.captureFrame(t.allocator);
    try t.expectEqual(@as(usize, 2), layer.frame.items.len);
    const big_cap = layer.frame.capacity;
    try t.expect(big_cap >= 2);

    b.active_count = 0;
    layer.captureFrame(t.allocator);
    try t.expectEqual(@as(usize, 1), layer.frame.items.len);
    try t.expectEqual(big_cap, layer.frame.capacity);
}

test "particle captureFrame OOM fail-closes without stale records, then recovers" {
    const t = std.testing;
    var layer: ParticleLayer = .{ .pass = undefined };
    defer layer.systems.deinit(t.allocator);
    defer layer.frame.deinit(t.allocator);

    var ps = try makeLayerTestSystem(t.allocator, 4);
    defer freeLayerTestSystem(&ps);
    ps.active_count = 2;
    try layer.systems.append(t.allocator, &ps);

    layer.captureFrame(t.allocator);
    try t.expectEqual(@as(usize, 1), layer.frame.items.len);

    var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    layer.frame.clearAndFree(t.allocator);
    layer.captureFrame(failing.allocator());
    try t.expectEqual(@as(usize, 0), layer.frame.items.len);

    layer.captureFrame(t.allocator);
    try t.expectEqual(@as(usize, 1), layer.frame.items.len);
    try t.expectEqual(@as(usize, 2), layer.frame.items[0].active_count);
}

test "particle renderPrepared is a headless no-op over a nonempty frame" {
    const t = std.testing;
    _ = meter.takeAndReset();
    var layer: ParticleLayer = .{ .pass = undefined };
    defer layer.systems.deinit(t.allocator);
    defer layer.frame.deinit(t.allocator);

    var ps = try makeLayerTestSystem(t.allocator, 4);
    defer freeLayerTestSystem(&ps);
    ps.active_count = 3;
    try layer.systems.append(t.allocator, &ps);
    layer.captureFrame(t.allocator);
    try t.expectEqual(@as(usize, 1), layer.frame.items.len);

    const cam: Camera = .{ .free = cam_mod.FreeCamera.init("test", .{}) };
    var stats = stats_mod.SceneStats{};
    layer.renderPrepared(cam, 16.0 / 9.0, 1, .RGBA16F, &stats);
    try t.expectEqual(stats_mod.SceneStats{}, stats);
    try t.expectEqual(@as(u64, 0), meter.peek());

    layer.renderPrepared(cam, 16.0 / 9.0, 4, .RGBA16F, &stats);
    try t.expectEqual(stats_mod.SceneStats{}, stats);
    try t.expectEqual(@as(u64, 0), meter.peek());
    try t.expect(layer.pass_msaa == null);
}

test "particle buildCapture+latchFrame isolates live mutations" {
    const t = std.testing;
    var layer: ParticleLayer = .{ .pass = undefined };
    defer layer.systems.deinit(t.allocator);
    defer layer.frame.deinit(t.allocator);
    defer layer.build_frame.deinit(t.allocator);

    var ps = try makeLayerTestSystem(t.allocator, 4);
    defer freeLayerTestSystem(&ps);
    ps.active_count = 3;
    try layer.systems.append(t.allocator, &ps);

    layer.buildCapture(t.allocator, 9);
    try t.expectEqual(@as(usize, 9), layer.build_seq.load(.acquire));
    try t.expectEqual(@as(usize, 1), layer.build_frame.items.len);
    try t.expectEqual(@as(usize, 0), layer.frame.items.len);

    ps.active_count = 1;
    ps.instance_buffer = .{ .id = 99 };
    try t.expectEqual(@as(usize, 3), layer.build_frame.items[0].active_count);
    try t.expectEqual(@as(u32, 11), layer.build_frame.items[0].instance_buffer.id);

    layer.latchFrame(t.allocator);
    try t.expectEqual(@as(u64, 9), layer.latched_seq);
    try t.expectEqual(@as(usize, 1), layer.frame.items.len);
    try t.expectEqual(@as(usize, 3), layer.frame.items[0].active_count);
    try t.expectEqual(@as(u32, 11), layer.frame.items[0].instance_buffer.id);
    ps.active_count = 4;
    try t.expectEqual(@as(usize, 3), layer.frame.items[0].active_count);
}

test "particle two builds before latch: newest wins, no build reuses last" {
    const t = std.testing;
    var layer: ParticleLayer = .{ .pass = undefined };
    defer layer.systems.deinit(t.allocator);
    defer layer.frame.deinit(t.allocator);
    defer layer.build_frame.deinit(t.allocator);

    var ps = try makeLayerTestSystem(t.allocator, 4);
    defer freeLayerTestSystem(&ps);
    ps.active_count = 2;
    try layer.systems.append(t.allocator, &ps);

    layer.buildCapture(t.allocator, 1);
    ps.active_count = 3;
    layer.buildCapture(t.allocator, 2);
    try t.expectEqual(@as(usize, 1), layer.build_frame.items.len);
    try t.expectEqual(@as(usize, 3), layer.build_frame.items[0].active_count);

    layer.latchFrame(t.allocator);
    try t.expectEqual(@as(usize, 3), layer.frame.items[0].active_count);

    ps.active_count = 1;
    layer.latchFrame(t.allocator);
    try t.expectEqual(@as(usize, 1), layer.frame.items[0].active_count);
}

test "particle buildCapture OOM fail-closes the build frame, latch publishes empty" {
    const t = std.testing;
    var layer: ParticleLayer = .{ .pass = undefined };
    defer layer.systems.deinit(t.allocator);
    defer layer.frame.deinit(t.allocator);
    defer layer.build_frame.deinit(t.allocator);

    var ps = try makeLayerTestSystem(t.allocator, 4);
    defer freeLayerTestSystem(&ps);
    ps.active_count = 2;
    try layer.systems.append(t.allocator, &ps);

    layer.buildCapture(t.allocator, 1);
    layer.latchFrame(t.allocator);
    try t.expectEqual(@as(usize, 1), layer.frame.items.len);

    var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    layer.build_frame.clearAndFree(t.allocator);
    layer.buildCapture(failing.allocator(), 2);
    try t.expectEqual(@as(usize, 0), layer.build_frame.items.len);
    layer.latchFrame(t.allocator);
    try t.expectEqual(@as(usize, 0), layer.frame.items.len);

    layer.buildCapture(t.allocator, 3);
    layer.latchFrame(t.allocator);
    try t.expectEqual(@as(usize, 1), layer.frame.items.len);
    try t.expectEqual(@as(usize, 2), layer.frame.items[0].active_count);
}

test "particle stageIntoSlot+latchSlotFrame freezes the build generation" {
    const t = std.testing;
    var layer: ParticleLayer = .{ .pass = undefined };
    defer layer.systems.deinit(t.allocator);
    defer layer.frame.deinit(t.allocator);
    defer layer.build_frame.deinit(t.allocator);

    var ps = try makeLayerTestSystem(t.allocator, 4);
    defer freeLayerTestSystem(&ps);
    ps.active_count = 3;
    try layer.systems.append(t.allocator, &ps);

    layer.buildCapture(t.allocator, 9);
    var slot_draws: std.ArrayListUnmanaged(ParticleDraw) = .empty;
    defer slot_draws.deinit(t.allocator);
    layer.stageIntoSlot(t.allocator, &slot_draws);
    try t.expectEqual(@as(usize, 1), slot_draws.items.len);
    try t.expectEqual(@as(usize, 3), slot_draws.items[0].active_count);
    try t.expectEqual(@as(u32, 11), slot_draws.items[0].instance_buffer.id);

    ps.active_count = 1;
    ps.instance_buffer = .{ .id = 99 };
    layer.build_frame.items[0].active_count = 1;
    try t.expectEqual(@as(usize, 3), slot_draws.items[0].active_count);
    try t.expectEqual(@as(u32, 11), slot_draws.items[0].instance_buffer.id);

    layer.latchSlotFrame(t.allocator, slot_draws.items);
    try t.expectEqual(@as(u64, 9), layer.latched_seq);
    try t.expectEqual(@as(usize, 1), layer.frame.items.len);
    try t.expectEqual(@as(usize, 3), layer.frame.items[0].active_count);
    try t.expectEqual(@as(u32, 11), layer.frame.items[0].instance_buffer.id);

    layer.latchFrame(t.allocator);
    try t.expectEqual(@as(usize, 1), layer.frame.items[0].active_count);
}

test "particle stageIntoSlot newest wins; OOM fail-closes, slot latch publishes empty" {
    const t = std.testing;
    var layer: ParticleLayer = .{ .pass = undefined };
    defer layer.systems.deinit(t.allocator);
    defer layer.frame.deinit(t.allocator);
    defer layer.build_frame.deinit(t.allocator);

    var ps = try makeLayerTestSystem(t.allocator, 4);
    defer freeLayerTestSystem(&ps);
    ps.active_count = 2;
    try layer.systems.append(t.allocator, &ps);

    layer.buildCapture(t.allocator, 1);
    ps.active_count = 3;
    layer.buildCapture(t.allocator, 2);
    var slot_draws: std.ArrayListUnmanaged(ParticleDraw) = .empty;
    defer slot_draws.deinit(t.allocator);
    layer.stageIntoSlot(t.allocator, &slot_draws);
    try t.expectEqual(@as(usize, 3), slot_draws.items[0].active_count);

    var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    slot_draws.clearAndFree(t.allocator);
    layer.stageIntoSlot(failing.allocator(), &slot_draws);
    try t.expectEqual(@as(usize, 0), slot_draws.items.len);
    layer.latchSlotFrame(t.allocator, slot_draws.items);
    try t.expectEqual(@as(usize, 0), layer.frame.items.len);

    layer.stageIntoSlot(t.allocator, &slot_draws);
    layer.latchSlotFrame(t.allocator, slot_draws.items);
    try t.expectEqual(@as(usize, 1), layer.frame.items.len);
    try t.expectEqual(@as(usize, 3), layer.frame.items[0].active_count);
}
