const std = @import("std");
const math = @import("math");
const types = @import("types.zig");
const sys = @import("system.zig");

const Vec3 = math.Vec3;
const Color4 = math.Color4;
const max_sub_emitter_depth = types.max_sub_emitter_depth;
const max_sub_emitter_spawns_per_tick = types.max_sub_emitter_spawns_per_tick;

test "sub-emitter probability 0 never fires, 1 always fires" {
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
    try std.testing.expectEqual(@as(usize, 0), parent.active_count);
    try std.testing.expectEqual(@as(usize, 0), child.active_count);

    parent.sub_emitter_store[0].probability = 1.0;
    parent.burst(2);
    parent.updateCpu(1.0);
    try std.testing.expectEqual(@as(usize, 0), parent.active_count);
    try std.testing.expectEqual(@as(usize, 6), child.active_count);
    for (child.particles[0..child.active_count]) |p| {
        try std.testing.expectEqual(@as(f32, 0.0), p.age);
        try std.testing.expectEqual(@as(f32, 10.0), p.lifetime);
        try std.testing.expectEqual(@as(u8, 1), p.sub_depth);
    }
}

test "sub-emitter inherits death position and blended velocity" {
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
    try std.testing.expectEqual(Vec3.new(1.0, 2.0, 3.0), child.particles[0].position);
    try std.testing.expectEqual(Vec3.new(2.0, 0.0, 0.0), child.particles[0].velocity);

    parent.sub_emitter_store[0].inherit_velocity = 1.0;
    parent.particles[0] = child.particles[0];
    parent.particles[0].age = 0.0;
    parent.particles[0].lifetime = 0.5;
    parent.particles[0].sub_depth = 0;
    parent.active_count = 1;
    child.active_count = 0;
    parent.updateCpu(1.0);
    try std.testing.expectEqual(Vec3.new(2.0, 0.0, 0.0), child.particles[0].velocity);

    parent.sub_emitter_store[0].inherit_velocity = 0.0;
    parent.particles[0].velocity = Vec3.new(4.0, 0.0, 0.0);
    parent.particles[0].age = 0.0;
    parent.active_count = 1;
    child.active_count = 0;
    parent.updateCpu(1.0);
    try std.testing.expectEqual(Vec3.zero, child.particles[0].velocity);

    parent.sub_emitter_store[0].inherit_position = false;
    parent.particles[0].age = 0.0;
    parent.active_count = 1;
    child.active_count = 0;
    parent.updateCpu(1.0);
    try std.testing.expectEqual(Vec3.new(9.0, 9.0, 9.0), child.particles[0].position);

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
        try std.testing.expect(sys_a.active_count <= 32);
        try std.testing.expect(sys_b.active_count <= 32);
        for (sys_a.particles[0..sys_a.active_count]) |p| max_depth_seen = @max(max_depth_seen, p.sub_depth);
        for (sys_b.particles[0..sys_b.active_count]) |p| max_depth_seen = @max(max_depth_seen, p.sub_depth);
    }
    try std.testing.expect(max_depth_seen > 0);
    try std.testing.expect(max_depth_seen <= max_sub_emitter_depth);
    try std.testing.expectEqual(@as(usize, 0), sys_a.active_count);
    try std.testing.expectEqual(@as(usize, 0), sys_b.active_count);
}

test "sub-emitter self-cycle terminates at the depth bound" {
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
    try std.testing.expectEqual(max_sub_emitter_depth, max_depth_seen);
    try std.testing.expectEqual(@as(usize, 0), ps.active_count);
}

test "sub-emitter plumbing is bit-identical when nothing fires" {
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
    try std.testing.expect(baseline.sub_tick > 0);
    try std.testing.expectEqual(plumbed.sub_tick, baseline.sub_tick);
    try std.testing.expectEqual(@as(usize, 0), child.active_count);
}

test "sub-emitter per-tick spawn bound" {
    const a = std.testing.allocator;
    var parent = try sys.makeQuiescentSystem(a, 1024);
    defer sys.freeTestSystem(&parent);
    var child = try sys.makeQuiescentSystem(a, 4096);
    defer sys.freeTestSystem(&child);
    child.lifetime_min = 10.0;
    child.lifetime_max = 10.0;
    parent.addSubEmitter(.{ .system = &child, .probability = 1.0, .count = 4 });
    parent.burst(512);
    try std.testing.expectEqual(@as(usize, 512), parent.active_count);
    parent.updateCpu(1.0);
    try std.testing.expectEqual(@as(usize, 0), parent.active_count);
    try std.testing.expectEqual(max_sub_emitter_spawns_per_tick, child.active_count);
}

test "sub-emitter spawns into a gpu child slot ring" {
    const a = std.testing.allocator;
    var parent = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&parent);
    var child = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&child);
    child.simulation_mode = .gpu;
    child.lifetime_min = 10.0;
    child.lifetime_max = 10.0;
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
    try std.testing.expectEqual(@as(usize, 1), child.gpu_high_water);
    try std.testing.expectEqual(@as(f32, 1.0), child.gpu_slots[0].spawn_pos_time[0]);
    try std.testing.expectEqual(@as(f32, 2.0), child.gpu_slots[0].spawn_pos_time[1]);
    try std.testing.expectEqual(@as(f32, 3.0), child.gpu_slots[0].spawn_pos_time[2]);
    try std.testing.expectEqual(@as(f32, 2.0), child.gpu_slots[0].velocity_lifetime[0]);
    try std.testing.expectEqual(@as(f32, 0.0), child.gpu_slots[0].velocity_lifetime[1]);
    try std.testing.expectEqual(@as(f32, 10.0), child.gpu_slots[0].velocity_lifetime[3]);
}
