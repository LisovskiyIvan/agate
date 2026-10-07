//! Tests for `collisions.zig` (moved from `collisions.zig` inline blocks).
const std = @import("std");
const math = @import("math");
const types = @import("types.zig");
const SimulationMode = types.SimulationMode;
const Vec3 = math.Vec3;
const jobs = @import("../jobs.zig");
const col = @import("collisions.zig");
const CollisionMode = col.CollisionMode;
const max_sphere_colliders = col.max_sphere_colliders;
const activeCollisionCtx = col.activeCollisionCtx;
const addSphereCollider = col.addSphereCollider;
const addBoxCollider = col.addBoxCollider;
const addPlaneCollider = col.addPlaneCollider;
const clearColliders = col.clearColliders;
const setCollisionMode = col.setCollisionMode;
const setGroundPlane = col.setGroundPlane;
const clearGroundPlane = col.clearGroundPlane;

test "collision defaults off and .none keeps the legacy path bit-identical" {
    const sys = @import("system.zig");
    const ParticleSystem = sys.ParticleSystem;
    const a = std.testing.allocator;
    var baseline = try sys.makeTestSystem(a, 64);
    defer sys.freeTestSystem(&baseline);
    var knobs = try sys.makeTestSystem(a, 64);
    defer sys.freeTestSystem(&knobs);
    for ([2]*ParticleSystem{ &baseline, &knobs }) |ps| {
        ps.gravity = Vec3.new(0.0, -3.0, 0.0);
        ps.emit_rate = 120.0;
        ps.is_emitting = true;
        ps.lifetime_min = 0.5;
        ps.lifetime_max = 1.0;
    }
    // Every collision knob touched, but mode stays .none: armed geometry
    // with the feature off must not perturb the stream, the integration, or
    // the instance fill (mirrors the flow "map unset" parity test).
    knobs.collision_restitution = 0.0;
    knobs.collision_friction = 0.0;
    try knobs.addSphereCollider(.{ .center = Vec3.new(0.0, 1.0, 0.0), .radius = 0.5 });
    try knobs.addSphereCollider(.{ .center = Vec3.zero, .radius = 2.0, .enabled = false });
    try knobs.setGroundPlane(5.0);
    try std.testing.expectEqual(CollisionMode.none, knobs.collision_mode);
    try std.testing.expect(activeCollisionCtx(&knobs) == null);

    var frame: usize = 0;
    while (frame < 60) : (frame += 1) {
        baseline.updateCpu(1.0 / 60.0);
        knobs.updateCpu(1.0 / 60.0);
        try std.testing.expectEqual(baseline.active_count, knobs.active_count);
        const live = baseline.active_count;
        try std.testing.expect(live > 0);
        try std.testing.expect(std.mem.eql(
            u8,
            std.mem.sliceAsBytes(baseline.particles[0..live]),
            std.mem.sliceAsBytes(knobs.particles[0..live]),
        ));
        try std.testing.expect(std.mem.eql(
            u8,
            std.mem.sliceAsBytes(baseline.instances[0..live]),
            std.mem.sliceAsBytes(knobs.instances[0..live]),
        ));
    }
}

test "sphere bounce reflects velocity about the contact normal" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.bounce);
    ps.collision_restitution = 1.0;
    ps.collision_friction = 1.0;
    try ps.addSphereCollider(.{ .center = Vec3.zero, .radius = 1.0 });
    // Head-on along -Z: (0,0,1.5) + (0,0,-1)*1 = (0,0,0.5), inside.
    sys.placeTestParticle(&ps, Vec3.new(0.0, 0.0, 1.5), Vec3.new(0.0, 0.0, -1.0));

    ps.updateCpu(1.0);
    try std.testing.expectEqual(@as(usize, 1), ps.active_count);
    // Pushed out to the surface, velocity mirrored exactly.
    try std.testing.expectEqual(Vec3.new(0.0, 0.0, 1.0), ps.particles[0].position);
    try std.testing.expectEqual(Vec3.new(0.0, 0.0, 1.0), ps.particles[0].velocity);
}

test "restitution 1 mirrors, restitution 0 stops the normal" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.bounce);
    ps.collision_friction = 1.0;
    try ps.addSphereCollider(.{ .center = Vec3.zero, .radius = 1.0 });

    ps.collision_restitution = 1.0;
    sys.placeTestParticle(&ps, Vec3.new(0.0, 0.0, 1.5), Vec3.new(0.0, 0.0, -1.0));
    ps.updateCpu(1.0);
    try std.testing.expectEqual(Vec3.new(0.0, 0.0, 1.0), ps.particles[0].velocity);

    // Same contact, dead-stop normal: the particle stays on the surface
    // with zero velocity (one-step settle).
    ps.collision_restitution = 0.0;
    sys.placeTestParticle(&ps, Vec3.new(0.0, 0.0, 1.5), Vec3.new(0.0, 0.0, -1.0));
    ps.updateCpu(1.0);
    try std.testing.expectEqual(Vec3.new(0.0, 0.0, 1.0), ps.particles[0].position);
    try std.testing.expectEqual(Vec3.zero, ps.particles[0].velocity);
}

test "friction damps only the tangential component" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.bounce);
    ps.collision_restitution = 1.0;
    try ps.addSphereCollider(.{ .center = Vec3.zero, .radius = 1.0 });
    // Oblique: (2,1.2,0) + (-2,-1,0)*1 = (0,0.2,0), normal +Y, vn = -1.
    // Reflected normal (+1) is friction-independent; tangential (-2,0,0)
    // scales by friction exactly.
    ps.collision_friction = 0.5;
    sys.placeTestParticle(&ps, Vec3.new(2.0, 1.2, 0.0), Vec3.new(-2.0, -1.0, 0.0));
    ps.updateCpu(1.0);
    try std.testing.expectEqual(Vec3.new(0.0, 1.0, 0.0), ps.particles[0].position);
    try std.testing.expectEqual(Vec3.new(-1.0, 1.0, 0.0), ps.particles[0].velocity);

    ps.collision_friction = 1.0;
    sys.placeTestParticle(&ps, Vec3.new(2.0, 1.2, 0.0), Vec3.new(-2.0, -1.0, 0.0));
    ps.updateCpu(1.0);
    try std.testing.expectEqual(Vec3.new(-2.0, 1.0, 0.0), ps.particles[0].velocity);

    // Full tangential stop: only the reflected normal survives.
    ps.collision_friction = 0.0;
    sys.placeTestParticle(&ps, Vec3.new(2.0, 1.2, 0.0), Vec3.new(-2.0, -1.0, 0.0));
    ps.updateCpu(1.0);
    try std.testing.expectEqual(Vec3.new(0.0, 1.0, 0.0), ps.particles[0].velocity);
}

test "kill mode destroys on sphere and ground contact, respawn is a fresh spawn" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.kill);
    try ps.addSphereCollider(.{ .center = Vec3.zero, .radius = 1.0 });

    // Placed particle (lifetime 10, so only the collision can kill it).
    sys.placeTestParticle(&ps, Vec3.zero, Vec3.zero);
    ps.updateCpu(1.0);
    try std.testing.expectEqual(@as(usize, 0), ps.active_count);

    // Ground kill: same proof, falling through y = 0.
    var ground = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ground);
    try ground.setCollisionMode(.kill);
    try ground.setGroundPlane(0.0);
    sys.placeTestParticle(&ground, Vec3.new(0.0, 0.5, 0.0), Vec3.new(0.0, -2.0, 0.0));
    ground.updateCpu(1.0);
    try std.testing.expectEqual(@as(usize, 0), ground.active_count);

    // Respawn is a normal fresh emission (age 0 at the emitter); the slot
    // is recycled like after any death.
    ps.burst(1);
    try std.testing.expectEqual(@as(usize, 1), ps.active_count);
    try std.testing.expectEqual(@as(f32, 0.0), ps.particles[0].age);
    ps.updateCpu(1.0);
    try std.testing.expectEqual(@as(usize, 0), ps.active_count);
}

test "ground bounce clamps above the plane and reflects" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.bounce);
    ps.collision_restitution = 0.5;
    ps.collision_friction = 0.5;
    try ps.setGroundPlane(0.0);
    // (0,2,0) + (2,-4,0)*1 = (2,-2,0) -> clamp y = 0, vy = +2, vx = 1.
    sys.placeTestParticle(&ps, Vec3.new(0.0, 2.0, 0.0), Vec3.new(2.0, -4.0, 0.0));

    ps.updateCpu(1.0);
    try std.testing.expectEqual(Vec3.new(2.0, 0.0, 0.0), ps.particles[0].position);
    try std.testing.expectEqual(Vec3.new(1.0, 2.0, 0.0), ps.particles[0].velocity);

    // Rising particles (vy > 0) pass clamped positions through untouched:
    // push-out only, no velocity change.
    sys.placeTestParticle(&ps, Vec3.new(0.0, -0.5, 0.0), Vec3.new(0.0, 3.0, 0.0));
    ps.updateCpu(0.0);
    try std.testing.expectEqual(Vec3.new(0.0, 0.0, 0.0), ps.particles[0].position);
    try std.testing.expectEqual(Vec3.new(0.0, 3.0, 0.0), ps.particles[0].velocity);
}

test "disabled collider is ignored" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.bounce);
    ps.collision_restitution = 1.0;
    try ps.addSphereCollider(.{ .center = Vec3.zero, .radius = 1.0, .enabled = false });
    try std.testing.expect(activeCollisionCtx(&ps) != null); // armed, but skipped
    sys.placeTestParticle(&ps, Vec3.new(0.0, 0.0, 1.5), Vec3.new(0.0, 0.0, -1.0));

    ps.updateCpu(1.0);
    // Straight through: no push-out, no reflection.
    try std.testing.expectEqual(Vec3.new(0.0, 0.0, 0.5), ps.particles[0].position);
    try std.testing.expectEqual(Vec3.new(0.0, 0.0, -1.0), ps.particles[0].velocity);
}

test "collider capacity and parameter validation" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    for (0..max_sphere_colliders) |i| {
        try ps.addSphereCollider(.{ .center = Vec3.new(@floatFromInt(i), 0.0, 0.0), .radius = 0.5 });
    }
    try std.testing.expectEqual(max_sphere_colliders, ps.collision_sphere_count);
    try std.testing.expectError(error.TooManyColliders, ps.addSphereCollider(.{}));
    try std.testing.expectEqual(max_sphere_colliders, ps.collision_sphere_count);

    var fresh = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&fresh);
    try std.testing.expectError(error.InvalidOptions, fresh.addSphereCollider(.{ .radius = 0.0 }));
    try std.testing.expectError(error.InvalidOptions, fresh.addSphereCollider(.{ .radius = -1.0 }));
    try std.testing.expectError(error.InvalidOptions, fresh.addSphereCollider(.{ .radius = std.math.inf(f32) }));
    try std.testing.expectError(error.InvalidOptions, fresh.addSphereCollider(.{ .radius = std.math.nan(f32) }));
    try std.testing.expectEqual(@as(usize, 0), fresh.collision_sphere_count);
    try std.testing.expectError(error.InvalidOptions, fresh.setGroundPlane(std.math.inf(f32)));
    try std.testing.expectEqual(@as(?f32, null), fresh.collision_ground);

    // Clearing disarms geometry but keeps the response knobs.
    try fresh.setCollisionMode(.bounce);
    try fresh.setGroundPlane(1.0);
    fresh.clearColliders();
    try std.testing.expectEqual(@as(usize, 0), fresh.collision_sphere_count);
    try std.testing.expectEqual(@as(?f32, null), fresh.collision_ground);
    try std.testing.expectEqual(CollisionMode.bounce, fresh.collision_mode);
    try std.testing.expect(activeCollisionCtx(&fresh) == null);
}

test "gpu and compute reject armed collisions, never downgrade" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.bounce);
    try ps.addSphereCollider(.{ .center = Vec3.zero, .radius = 1.0 });

    // .gpu: both update entry points error; the mode is never mutated.
    ps.simulation_mode = .gpu;
    try std.testing.expectError(error.CollisionNeedsCpu, ps.updateGpu(0.016));
    try std.testing.expectError(error.CollisionNeedsCpu, ps.update(0.016));
    try std.testing.expectEqual(SimulationMode.gpu, ps.simulation_mode);
    try std.testing.expectEqual(@as(usize, 0), ps.gpu_high_water);
    // Disarming restores the GPU path (disarm never fails, any mode).
    try ps.setCollisionMode(.none);
    try ps.update(0.016);

    // .compute: same explicit rejection (re-arm on CPU first: enabling
    // while .gpu is correctly rejected above).
    ps.simulation_mode = .cpu;
    try ps.setCollisionMode(.bounce);
    ps.simulation_mode = .compute;
    try std.testing.expectError(error.CollisionNeedsCpu, ps.updateCompute(0.016));
    try std.testing.expectError(error.CollisionNeedsCpu, ps.update(0.016));
    try std.testing.expectEqual(SimulationMode.compute, ps.simulation_mode);
    try std.testing.expectEqual(@as(usize, 0), ps.compute_high_water);
    try ps.setCollisionMode(.none);
    try ps.update(0.016);

    // Enable-time rejection: arming a non-CPU system fails immediately.
    var gpu = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&gpu);
    gpu.simulation_mode = .gpu;
    try std.testing.expectError(error.CollisionNeedsCpu, gpu.addSphereCollider(.{}));
    try std.testing.expectError(error.CollisionNeedsCpu, gpu.setCollisionMode(.bounce));
    try std.testing.expectError(error.CollisionNeedsCpu, gpu.setGroundPlane(0.0));
    try std.testing.expectEqual(@as(usize, 0), gpu.collision_sphere_count);
    try std.testing.expectEqual(CollisionMode.none, gpu.collision_mode);
    try std.testing.expectEqual(@as(?f32, null), gpu.collision_ground);
    // Disarm paths always succeed, even off-CPU.
    try gpu.setCollisionMode(.none);
    gpu.clearColliders();
    gpu.clearGroundPlane();
}

test "same state plus same dt gives identical results" {
    const sys = @import("system.zig");
    const ParticleSystem = sys.ParticleSystem;
    const a = std.testing.allocator;
    var run_a = try sys.makeTestSystem(a, 128);
    defer sys.freeTestSystem(&run_a);
    var run_b = try sys.makeTestSystem(a, 128);
    defer sys.freeTestSystem(&run_b);
    for ([2]*ParticleSystem{ &run_a, &run_b }) |ps| {
        ps.gravity = Vec3.new(0.0, -3.0, 0.0);
        ps.emit_rate = 120.0;
        ps.is_emitting = true;
        ps.lifetime_min = 0.5;
        ps.lifetime_max = 1.0;
        try ps.setCollisionMode(.bounce);
        ps.collision_restitution = 0.5;
        ps.collision_friction = 0.8;
        try ps.addSphereCollider(.{ .center = Vec3.new(0.0, 1.0, 0.0), .radius = 0.75 });
        try ps.setGroundPlane(-0.5);
    }

    var frame: usize = 0;
    while (frame < 60) : (frame += 1) {
        run_a.updateCpu(1.0 / 60.0);
        run_b.updateCpu(1.0 / 60.0);
        try std.testing.expectEqual(run_a.active_count, run_b.active_count);
        const live = run_a.active_count;
        try std.testing.expect(live > 0);
        try std.testing.expect(std.mem.eql(
            u8,
            std.mem.sliceAsBytes(run_a.particles[0..live]),
            std.mem.sliceAsBytes(run_b.particles[0..live]),
        ));
    }
    const hash_a = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(run_a.particles[0..run_a.active_count]));
    const hash_b = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(run_b.particles[0..run_b.active_count]));
    try std.testing.expectEqual(hash_a, hash_b);
}

test "kill absorption agrees across dyadic frame rates" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    const hashLive = struct {
        fn hashLive(ps: *sys.ParticleSystem) u64 {
            return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(ps.particles[0..ps.active_count]));
        }
    }.hashLive;
    // Exact-fp setup: unit velocity, dyadic dts, contact strictly inside.
    var run_a = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&run_a);
    var run_b = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&run_b);
    for ([2]*sys.ParticleSystem{ &run_a, &run_b }) |ps| {
        try ps.setCollisionMode(.kill);
        try ps.addSphereCollider(.{ .center = Vec3.new(1.0, 0.0, 0.0), .radius = 0.5 });
        sys.placeTestParticle(ps, Vec3.zero, Vec3.new(1.0, 0.0, 0.0));
    }
    // Same simulated instant (t = 0.5) via different stepping: exact in
    // binary fp (0.5, 0.25 + 0.25), so the live states are bit-identical.
    // x = 0.5 is exactly at touch distance (dist == radius): not contact.
    run_a.updateCpu(0.5);
    run_b.updateCpu(0.25);
    run_b.updateCpu(0.25);
    try std.testing.expectEqual(@as(usize, 1), run_a.active_count);
    try std.testing.expectEqual(@as(usize, 1), run_b.active_count);
    try std.testing.expectEqual(hashLive(&run_a), hashLive(&run_b));

    // Both schedules absorb the particle (discrete contact timing is
    // step-size dependent by nature — the kill event itself is what agrees).
    run_a.updateCpu(0.5);
    run_b.updateCpu(0.25);
    try std.testing.expectEqual(@as(usize, 0), run_a.active_count);
    try std.testing.expectEqual(@as(usize, 0), run_b.active_count);
    run_b.updateCpu(0.25); // empty tick is a no-op
    try std.testing.expectEqual(hashLive(&run_a), hashLive(&run_b));
}

test "collisions are worker-count invariant" {
    const sys = @import("system.zig");
    const ParticleSystem = sys.ParticleSystem;
    const a = std.testing.allocator;
    const pool = try jobs.Pool.init(a, 2);
    defer pool.deinit();

    var serial = try sys.makeTestSystem(a, 16_384);
    defer sys.freeTestSystem(&serial);
    var parallel = try sys.makeTestSystem(a, 16_384);
    defer sys.freeTestSystem(&parallel);
    parallel.thread_pool = pool;
    for ([2]*ParticleSystem{ &serial, &parallel }) |ps| {
        ps.gravity = Vec3.new(0.0, -3.0, 0.0);
        ps.emit_rate = 8000.0;
        ps.is_emitting = true;
        try ps.setCollisionMode(.bounce);
        ps.collision_restitution = 0.5;
        ps.collision_friction = 0.9;
        try ps.addSphereCollider(.{ .center = Vec3.new(0.0, 1.0, 0.0), .radius = 1.5 });
        try ps.setGroundPlane(-1.0);
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
    // Past the pool threshold with real contacts: proves the armed path
    // actually forked (not just the inline fallback).
    try std.testing.expect(serial.active_count > jobs.Pool.min_len_for_workers);
}

test "armed collision scenario pins a golden hash" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeTestSystem(a, 32);
    defer sys.freeTestSystem(&ps);
    ps.gravity = Vec3.new(0.0, -2.0, 0.0);
    ps.emit_rate = 30.0;
    ps.is_emitting = true;
    try ps.setCollisionMode(.bounce);
    ps.collision_restitution = 0.5;
    ps.collision_friction = 0.5;
    try ps.addSphereCollider(.{ .center = Vec3.new(0.0, 1.0, 0.0), .radius = 0.5 });
    try ps.setGroundPlane(-0.5);

    var frame: usize = 0;
    while (frame < 30) : (frame += 1) ps.updateCpu(1.0 / 60.0);
    // Pinned regression values (seed 42 emission + collisions; regenerate
    // deliberately if the integrator changes — see the .none parity test,
    // which proves the default path is untouched by this feature).
    // In ReleaseFast, LLVM fuses multiply-add into hardware FMA instructions.
    const want_hash: u64 = switch (@import("builtin").mode) {
        .ReleaseFast => 7238270927919988946,
        else => 15457552461678759088,
    };
    try std.testing.expectEqual(want_hash, std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(ps.particles[0..ps.active_count])));
    // 30 frames x 0.5 emissions/frame, lifetimes >= 1 s: nothing dies.
    try std.testing.expectEqual(@as(usize, 15), ps.active_count);
}

test "box bounce reflects velocity about the closest face normal" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.bounce);
    ps.collision_restitution = 1.0;
    ps.collision_friction = 1.0;
    try ps.addBoxCollider(.{
        .center = Vec3.zero,
        .half_extents = Vec3.new(1.0, 1.0, 1.0),
    });

    // Head-on along -X towards +X face: (1.5, 0, 0) + (-1, 0, 0)*1 = (0.5, 0, 0), inside.
    sys.placeTestParticle(&ps, Vec3.new(1.5, 0.0, 0.0), Vec3.new(-1.0, 0.0, 0.0));
    ps.updateCpu(1.0);
    try std.testing.expectEqual(@as(usize, 1), ps.active_count);
    try std.testing.expectEqual(Vec3.new(1.0, 0.0, 0.0), ps.particles[0].position);
    try std.testing.expectEqual(Vec3.new(1.0, 0.0, 0.0), ps.particles[0].velocity);
}

test "box bounce with restitution and friction" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.bounce);
    ps.collision_restitution = 0.5;
    ps.collision_friction = 0.25;
    try ps.addBoxCollider(.{
        .center = Vec3.zero,
        .half_extents = Vec3.new(2.0, 0.5, 2.0),
    });

    // Particle lands on top face (+Y): (0, 1.0, 0) + (4.0, -0.8, 0.0)*1 = (4.0, 0.2, 0.0).
    // dx = 4 - 2 = 2 (outside along X if x=4, so let's keep x inside box: x=0.5).
    // Start at (0.5, 1.0, 0.0), vel = (4.0, -0.8, 0.0).
    // After 1s: pos = (4.5, 0.2, 0) -> outside on X! So let's use small horizontal vel: (0.2, -0.8, 0.0).
    // After 1s: pos = (0.7, 0.2, 0.0).
    // d = (0.7, 0.2, 0.0).
    // dx = 0.7 - 2.0 = -1.3.
    // dy = 0.2 - 0.5 = -0.3.
    // dz = 0.0 - 2.0 = -2.0.
    // Closest face is +Y (dy = -0.3 is maximum negative value).
    sys.placeTestParticle(&ps, Vec3.new(0.5, 1.0, 0.0), Vec3.new(0.2, -0.8, 0.0));
    ps.updateCpu(1.0);
    try std.testing.expectEqual(@as(usize, 1), ps.active_count);
    // Pushed out to y = 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), ps.particles[0].position.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), ps.particles[0].position.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ps.particles[0].position.z, 1e-6);
    // Normal vel -0.8 reflected with rest 0.5 -> +0.4.
    // Tangential vel 0.2 scaled by fric 0.25 -> 0.05.
    try std.testing.expectApproxEqAbs(@as(f32, 0.05), ps.particles[0].velocity.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), ps.particles[0].velocity.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ps.particles[0].velocity.z, 1e-6);
}

test "plane bounce reflects velocity about plane normal" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.bounce);
    ps.collision_restitution = 1.0;
    ps.collision_friction = 1.0;
    // Plane passing through (0, 0, 0) with normal along +Z
    try ps.addPlaneCollider(.{
        .point = Vec3.zero,
        .normal = Vec3.new(0.0, 0.0, 1.0),
    });

    // Particle moves from +Z towards -Z: (0, 0, 0.5) + (0, 0, -1.0)*1 = (0, 0, -0.5), behind plane.
    sys.placeTestParticle(&ps, Vec3.new(0.0, 0.0, 0.5), Vec3.new(0.0, 0.0, -1.0));
    ps.updateCpu(1.0);
    try std.testing.expectEqual(@as(usize, 1), ps.active_count);
    // Pushed out to z = 0.0, velocity mirrored along +Z
    try std.testing.expectEqual(Vec3.zero, ps.particles[0].position);
    try std.testing.expectEqual(Vec3.new(0.0, 0.0, 1.0), ps.particles[0].velocity);
}

test "box and plane kill mode eliminates contacting particle" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.setCollisionMode(.kill);
    try ps.addBoxCollider(.{
        .center = Vec3.new(2.0, 0.0, 0.0),
        .half_extents = Vec3.new(0.5, 0.5, 0.5),
    });
    // Particle moves into box
    sys.placeTestParticle(&ps, Vec3.new(1.0, 0.0, 0.0), Vec3.new(1.0, 0.0, 0.0));
    ps.updateCpu(0.8); // reaches 1.8, inside box
    try std.testing.expectEqual(@as(usize, 0), ps.active_count);
}
