const std = @import("std");
const math = @import("math");
const Vec2 = math.Vec2;
const Vec3 = math.Vec3;

const types = @import("types.zig");
const SimulationMode = types.SimulationMode;
const Texture = @import("../texture.zig").Texture;

const sys = @import("system.zig");
const ParticleSystem = sys.ParticleSystem;

test "flow map unset keeps the legacy path bit-identical" {
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
    // Knobs touched but field disarmed (strength 0, no CPU copy): must not
    // perturb the stream, the integration, or the instance fill.
    knobs.flow_space = .local_xz;
    knobs.flow_wrap = .clamp;
    knobs.flow_scale = Vec2.new(0.5, 2.0);
    knobs.flow_scroll = Vec2.new(0.25, -1.0);

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

test "constant flow accelerates along the sampled direction" {
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    // 1x1 texel: R=255 -> +1 x, G=128 -> ~0 z, B=255 -> full strength.
    var px = [_]u8{ 255, 128, 255, 255 };
    try ps.setFlowMap(null, &px, 1, 1);
    ps.flow_strength = 2.0;
    sys.placeTestParticle(&ps, Vec3.zero, Vec3.zero);

    ps.updateCpu(1.0);
    // Semi-implicit Euler: v += flow * strength * dt, then p += v * dt.
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), ps.particles[0].velocity.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), ps.particles[0].position.x, 1e-4);
    // G=128 decodes to (128/255*2-1) ~= 0.00392, times strength 2.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0078431), ps.particles[0].velocity.z, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ps.particles[0].velocity.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ps.particles[0].position.y, 1e-6);
    // sampleFlow agrees with the integrated push (per-second acceleration).
    const acc = ps.sampleFlow(Vec3.zero);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), acc.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), acc.y, 1e-6);
}

test "flow strength 0 is a no-op" {
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    var px = [_]u8{ 255, 128, 255, 255 };
    try ps.setFlowMap(null, &px, 1, 1);
    ps.flow_strength = 0.0; // armed copy, disarmed strength
    sys.placeTestParticle(&ps, Vec3.zero, Vec3.new(1.0, 2.0, 3.0));

    ps.updateCpu(0.5);
    // Legacy path bit-for-bit: velocity untouched, position advances alone.
    try std.testing.expectEqual(Vec3.new(1.0, 2.0, 3.0), ps.particles[0].velocity);
    try std.testing.expectEqual(Vec3.new(0.5, 1.0, 1.5), ps.particles[0].position);
    try std.testing.expectEqual(Vec3.zero, ps.sampleFlow(Vec3.zero));
}

test "local_xz samples relative to the emitter origin" {
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    // 2x2: corner texel (0,0) pushes +X, every other texel pushes -X.
    var px = [_]u8{
        255, 128, 255, 255, 0, 128, 255, 255,
        0,   128, 255, 255, 0, 128, 255, 255,
    };
    try ps.setFlowMap(null, &px, 2, 2);
    ps.flow_strength = 1.0;
    ps.emitter_position = Vec3.new(0.5, 0.0, 0.5);

    // World: uv (0,0) lands exactly on the +X corner texel.
    ps.flow_space = .world_xz;
    const world = ps.sampleFlow(Vec3.zero);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), world.x, 1e-4);

    // Local: uv (-0.5,-0.5) wraps to (0.5,0.5), the average of (+1,-1,-1,-1).
    ps.flow_space = .local_xz;
    const local = ps.sampleFlow(Vec3.zero);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), local.x, 1e-4);

    try std.testing.expect(world.x != local.x);
}

test "flow out-of-bounds repeat wraps, clamp stretches the edge" {
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    // Columns differ (+X left, -X right), rows identical (v-independent).
    var px = [_]u8{
        255, 128, 255, 255, 0, 128, 255, 255,
        255, 128, 255, 255, 0, 128, 255, 255,
    };
    try ps.setFlowMap(null, &px, 2, 2);
    ps.flow_strength = 1.0;

    // Repeat (default): u = 1.25 wraps to 0.25, bit-identical to sampling
    // 0.25 directly (1.25 and 0.25 are exact binaries).
    ps.flow_wrap = .repeat;
    const wrapped = ps.sampleFlow(Vec3.new(1.25, 0.0, 0.0));
    const direct = ps.sampleFlow(Vec3.new(0.25, 0.0, 0.0));
    try std.testing.expectEqual(direct, wrapped);
    // u = 0.25 lerps 75% +X / 25% -X -> x = +0.5.
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), wrapped.x, 1e-4);
    // Negative wraps too: u = -0.25 -> 0.75 -> 25% +X / 75% -X -> x = -0.5.
    const neg = ps.sampleFlow(Vec3.new(-0.25, 0.0, 0.0));
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), neg.x, 1e-5);

    // Clamp: u = 1.25 clamps to 1.0, the -X edge texel exactly.
    ps.flow_wrap = .clamp;
    const edge = ps.sampleFlow(Vec3.new(1.25, 0.0, 0.0));
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), edge.x, 1e-5);
    try std.testing.expect(wrapped.x != edge.x);
}

test "flow without a CPU copy is ignored" {
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    // GPU handle assigned directly (borrowed fake, as in the layer tests),
    // but no CPU copy installed: the field stays disarmed, never crashes.
    var tex: Texture = std.mem.zeroes(Texture);
    tex.view = .{ .id = 77 };
    ps.flow_map = tex;
    ps.flow_strength = 5.0;
    sys.placeTestParticle(&ps, Vec3.zero, Vec3.new(1.0, 0.0, 0.0));

    ps.updateCpu(1.0);
    try std.testing.expectEqual(Vec3.new(1.0, 0.0, 0.0), ps.particles[0].velocity);
    try std.testing.expectEqual(Vec3.new(1.0, 0.0, 0.0), ps.particles[0].position);
    try std.testing.expectEqual(Vec3.zero, ps.sampleFlow(Vec3.zero));
    // The borrowed handle is test-owned: release the field without
    // destroying the fake GPU objects (mirrors freeTestSystem, which never
    // deinits textures).
    ps.flow_map = null;
}

test "flow map setter validates dimensions before mutating" {
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    var px = [_]u8{ 255, 128, 255, 255 };
    try ps.setFlowMap(null, &px, 1, 1);
    // Short buffer, zero dims, and overflow-scale dims all fail...
    try std.testing.expectError(error.InvalidDimensions, ps.setFlowMap(null, &px, 2, 2));
    try std.testing.expectError(error.InvalidDimensions, ps.setFlowMap(null, &.{}, 0, 1));
    try std.testing.expectError(error.InvalidDimensions, ps.setFlowMap(null, &px, 100000, 100000));
    // ...leaving the previous field untouched and still sampling.
    try std.testing.expectEqual(@as(usize, 4), ps.flow_pixels.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ps.sampleFlow(Vec3.zero).x, 1e-6);
    ps.flow_strength = 1.0;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ps.sampleFlow(Vec3.zero).x, 1e-4);
    // Clearing disarms (knobs kept) and frees the copy.
    ps.clearFlowMap();
    try std.testing.expectEqual(@as(usize, 0), ps.flow_pixels.len);
    try std.testing.expectEqual(Vec3.zero, ps.sampleFlow(Vec3.zero));
}

test "flow map on gpu is an explicit error, not a downgrade" {
    const a = std.testing.allocator;
    var ps = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    ps.simulation_mode = .gpu;
    try ps.updateGpu(0.0); // provisions the ring
    var px = [_]u8{ 255, 128, 255, 255 };
    try ps.setFlowMap(null, &px, 1, 1);
    ps.flow_strength = 1.0;

    try std.testing.expectError(error.FlowMapNeedsCpu, ps.updateGpu(0.016));
    try std.testing.expectError(error.FlowMapNeedsCpu, ps.update(0.016));
    // The requested mode is never mutated and the ring never advanced past
    // the provisioned state.
    try std.testing.expectEqual(SimulationMode.gpu, ps.simulation_mode);
    try std.testing.expectEqual(@as(usize, 0), ps.gpu_high_water);
    // Disarming restores the GPU path.
    ps.flow_strength = 0.0;
    try ps.update(0.016);
}
