const std = @import("std");
const jobs = @import("../jobs.zig");
const math = @import("math");
const Vec3 = math.Vec3;
const sys = @import("system.zig");
const ParticleSystem = sys.ParticleSystem;

test "two-phase CPU update is worker-count invariant" {
    const a = std.testing.allocator;
    const pool = try jobs.Pool.init(a, 2);
    defer pool.deinit();

    // Identical seed + config; the only difference is the execution path:
    // `serial` runs inline (no pool), `parallel` fork-joins once the live
    // count crosses jobs.Pool.min_len_for_workers.
    var serial = try sys.makeTestSystem(a, 16_384);
    defer sys.freeTestSystem(&serial);
    var parallel = try sys.makeTestSystem(a, 16_384);
    defer sys.freeTestSystem(&parallel);
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
