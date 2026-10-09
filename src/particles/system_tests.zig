const std = @import("std");
const sg = @import("sokol").gfx;
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Mesh = @import("../mesh.zig").Mesh;
const gpu_thread = @import("../gpu_thread.zig");
const system_mod = @import("system.zig");
const ParticleSystem = system_mod.ParticleSystem;
const makeTestSystem = system_mod.makeTestSystem;
const freeTestSystem = system_mod.freeTestSystem;

fn expectDeferredInit() !void {
    const ps = try ParticleSystem.init(std.testing.allocator, "headless", 8);
    defer std.testing.allocator.destroy(ps);
    defer ps.deinit();
    try std.testing.expectEqual(sg.Buffer{}, ps.instance_buffer);
    try std.testing.expect(ps.instance_buffer_pending);
    ps.emitOne();
    try std.testing.expectEqual(@as(usize, 1), ps.active_count);
}

test "headless particles defer GPU creation on an unregistered CPU thread" {
    try std.testing.expect(!sg.isvalid());
    gpu_thread.resetContextThreadForTest();
    defer gpu_thread.resetContextThreadForTest();
    try std.testing.expect(gpu_thread.isOnContextThread());
    try expectDeferredInit();
}

test "headless particles defer GPU creation even on the marked owner" {
    try std.testing.expect(!sg.isvalid());
    gpu_thread.markContextThread();
    defer gpu_thread.resetContextThreadForTest();
    try std.testing.expect(gpu_thread.isOnContextThread());
    try expectDeferredInit();
}

test "defaults keep world positions bit-identical" {
    var ps = try makeTestSystem(std.testing.allocator, 4);
    defer freeTestSystem(&ps);
    // Defaults: local_space=false, emitter_mesh=null, 1x1 sheet, zero rotation.
    try std.testing.expectEqual(false, ps.local_space);
    try std.testing.expectEqual(@as(?*Mesh, null), ps.emitter_mesh);
    ps.emitter_position = Vec3.new(1.0, 2.0, 3.0);
    ps.direction_min = Vec3.zero;
    ps.direction_max = Vec3.zero;
    ps.speed_min = 0.0;
    ps.speed_max = 0.0;
    ps.gravity = Vec3.zero;
    ps.lifetime_min = 10.0;
    ps.lifetime_max = 10.0;
    ps.emitOne();
    ps.updateCpu(0.0);
    const p = ps.particles[0];
    const inst = ps.instances[0];
    try std.testing.expectEqual(p.position.x, inst.pos_size[0]);
    try std.testing.expectEqual(p.position.y, inst.pos_size[1]);
    try std.testing.expectEqual(p.position.z, inst.pos_size[2]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 1.0, 1.0 }, inst.uv_offset_scale);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, inst.rotation_misc);
    try std.testing.expectEqual(@as(?Mat4, null), ps.resolveEmitterMatrix());
}

test "local_space spawn stays relative, instances follow emitter" {
    var ps = try makeTestSystem(std.testing.allocator, 4);
    defer freeTestSystem(&ps);
    // Buffer-free emitter mesh: local-space update only reads the transform.
    var emitter = Mesh{ .name = "emitter", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 0 };
    emitter.position = Vec3.new(5.0, 0.0, 0.0);
    emitter.rotation = Vec3.zero;
    emitter.scaling = Vec3.one;
    emitter.base_matrix = Mat4.identity;

    ps.local_space = true;
    ps.emitter_mesh = &emitter;
    ps.emitter_position = Vec3.new(1.0, 2.0, 3.0);
    ps.direction_min = Vec3.zero;
    ps.direction_max = Vec3.zero;
    ps.speed_min = 0.0;
    ps.speed_max = 0.0;
    ps.gravity = Vec3.zero;
    ps.lifetime_min = 10.0;
    ps.lifetime_max = 10.0;
    ps.emitOne();

    // Stored coordinates are emitter-local, not world.
    try std.testing.expectEqual(Vec3.new(1.0, 2.0, 3.0), ps.particles[0].position);

    ps.updateCpu(0.0);
    // Instance = world matrix applied: (1,2,3) + emitter offset (5,0,0).
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), ps.instances[0].pos_size[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), ps.instances[0].pos_size[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), ps.instances[0].pos_size[2], 1e-5);

    // Moving the emitter moves rendered instances, stored locals stay put.
    emitter.position = Vec3.new(10.0, 0.0, 0.0);
    ps.updateCpu(0.0);
    try std.testing.expectEqual(Vec3.new(1.0, 2.0, 3.0), ps.particles[0].position);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), ps.instances[0].pos_size[0], 1e-5);
}
