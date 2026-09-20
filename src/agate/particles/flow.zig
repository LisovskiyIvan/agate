//! Flow-field texture: CPU-side sampling and arming. Split out of
//! `particles.zig` (facade).
//!
//! `setFlowMap` / `clearFlowMap` / `activeFlowCtx` / `sampleFlow` take the
//! system as `anytype` so this module never imports `system.zig` or the
//! facade back — same discipline as `profiler/*`. `cpu.zig` and the `.gpu` /
//! `.compute` leaves call `activeFlowCtx` directly (sibling import, no cycle:
//! this module only imports `types.zig`). Moved tests reach `system.zig`
//! helpers through block-scoped imports that exist only in test builds.

const std = @import("std");
const math = @import("math");

const types = @import("types.zig");
const Texture = @import("../texture.zig").Texture;

const Vec2 = math.Vec2;
const Vec3 = math.Vec3;
const FlowSpace = types.FlowSpace;
const FlowWrap = types.FlowWrap;
const SimulationMode = types.SimulationMode;

/// Arms the flow field: takes ownership of `texture` (may be null for a
/// CPU-only/headless field — the CPU path never dereferences the GPU
/// handle) and keeps an owned CPU-side copy of `pixels` (w*h*4 RGBA8
/// bytes). Cost: one alloc + memcpy, documented on the `flow_pixels`
/// field. Replaces any previous field (old texture destroyed, old copy
/// freed). Validates BEFORE mutating: on error.InvalidDimensions the
/// previous field (if any) is left untouched.
///
/// Sandbox recipe (procedural, main thread): fill an RGBA8 buffer
/// (R = dir.x, G = dir.z, B = strength), upload with
/// `Texture.initRaw(w, h, &buf, .{ .min_filter = .LINEAR, ... })`, then
/// `try ps.setFlowMap(tex, &buf, w, h)` — the engine dupes the CPU copy,
/// so the caller buffer may be transient. Keep data textures linear
/// (no sRGB conversion): the encoding is authored linear.
pub fn setFlowMap(
    self: anytype,
    texture: ?Texture,
    pixels: []const u8,
    width: u32,
    height: u32,
) !void {
    if (width == 0 or height == 0) return error.InvalidDimensions;
    const pixel_count = std.math.mul(u32, width, height) catch return error.InvalidDimensions;
    const expected = std.math.mul(u32, pixel_count, 4) catch return error.InvalidDimensions;
    if (pixels.len != @as(usize, expected)) return error.InvalidDimensions;
    const copy = try self.allocator.dupe(u8, pixels);
    if (self.flow_map) |*t| t.deinit();
    if (self.flow_pixels.len > 0) self.allocator.free(self.flow_pixels);
    self.flow_map = texture;
    self.flow_pixels = copy;
    self.flow_width = width;
    self.flow_height = height;
}

pub fn clearFlowMap(self: anytype) void {
    if (self.flow_map) |*t| {
        t.deinit();
        self.flow_map = null;
    }
    if (self.flow_pixels.len > 0) {
        self.allocator.free(self.flow_pixels);
        self.flow_pixels = &.{};
        self.flow_width = 0;
        self.flow_height = 0;
    }
}

/// Snapshots the armed flow field for one updateCpu tick, or null when
/// disarmed (strength == 0, no CPU copy, zero dims, or length mismatch —
/// the length check also makes a foreign/short `flow_pixels` slice safe
/// instead of OOB). The GPU `flow_map` handle is deliberately NOT
/// consulted: the CPU path must never read GPU memory.
pub fn activeFlowCtx(self: anytype) ?FlowCtx {
    if (self.flow_strength == 0.0) return null;
    if (self.flow_pixels.len == 0) return null;
    if (self.flow_width == 0 or self.flow_height == 0) return null;
    const pixel_count = std.math.mul(u32, self.flow_width, self.flow_height) catch return null;
    const expected = std.math.mul(u32, pixel_count, 4) catch return null;
    if (self.flow_pixels.len != @as(usize, expected)) return null;
    return .{
        .pixels = self.flow_pixels,
        .width = self.flow_width,
        .height = self.flow_height,
        .strength = self.flow_strength,
        .space = self.flow_space,
        .wrap = self.flow_wrap,
        .scale = self.flow_scale,
        .scroll = self.flow_scroll,
        .emitter = self.emitter_position,
    };
}

/// Flow acceleration at `pos` (decoded dir * tex strength * flow_strength,
/// y == 0). Zero when disarmed. Pure and headless-safe; tests pin the
/// wrap/clamp and space mapping through this entry point.
pub fn sampleFlow(self: anytype, pos: Vec3) Vec3 {
    const f = activeFlowCtx(self) orelse return Vec3.zero;
    const uv = flowUvForPosition(f.space, pos, f.emitter, f.scale, f.scroll);
    return sampleFlowPixels(f.pixels, f.width, f.height, uv, f.wrap).scale(f.strength);
}

/// Read-only per-tick snapshot of an armed flow field, shared by all Phase-A
/// workers (all members are read-only; no locks needed).
pub const FlowCtx = struct {
    pixels: []const u8,
    width: u32,
    height: u32,
    strength: f32,
    space: FlowSpace,
    wrap: FlowWrap,
    scale: Vec2,
    scroll: Vec2,
    emitter: Vec3,
};

/// Maps a simulation position to flow-texture UV. Uses the stored simulation
/// coordinates verbatim (world units when local_space == false, emitter-local
/// units when true); `.local_xz` additionally subtracts the emitter origin so
/// the field travels with the emitter.
pub fn flowUvForPosition(space: FlowSpace, pos: Vec3, emitter: Vec3, scale: Vec2, scroll: Vec2) Vec2 {
    const bx = switch (space) {
        .world_xz => pos.x,
        .local_xz => pos.x - emitter.x,
    };
    const bz = switch (space) {
        .world_xz => pos.z,
        .local_xz => pos.z - emitter.z,
    };
    return Vec2.new(bx * scale.x + scroll.x, bz * scale.y + scroll.y);
}

/// One bilinear RGB tap (channels as f32 in [0, 255]; alpha ignored).
/// Precondition: pixels.len >= w*h*4, x < w, y < h (callers clamp).
fn flowTexel(pixels: []const u8, width: u32, x: u32, y: u32) [3]f32 {
    const idx = (@as(usize, y) * @as(usize, width) + @as(usize, x)) * 4;
    return .{
        @floatFromInt(pixels[idx]),
        @floatFromInt(pixels[idx + 1]),
        @floatFromInt(pixels[idx + 2]),
    };
}

/// Decodes one (R, G, B) tap to a planar flow vector: R/G map [0, 255] ->
/// [-1, 1] as x/z, B maps [0, 255] -> [0, 1] as the per-texel strength
/// multiplier (y is always 0). Raw, never normalized: a zero texel yields a
/// zero vector (no NaN path).
fn decodeFlowTap(rgb: [3]f32) Vec3 {
    const k: f32 = 2.0 / 255.0;
    return Vec3.new(
        rgb[0] * k - 1.0,
        0.0,
        rgb[1] * k - 1.0,
    ).scale(rgb[2] / 255.0);
}

/// Bilinear flow sample with corner convention (uv (0,0)/(1,1) land exactly
/// on the corner texels; 1x1 fields are constant). Defense in depth: an
/// empty or short pixel slice yields zero instead of OOB (activeFlowCtx
/// already guarantees the shape, direct callers may not).
pub fn sampleFlowPixels(pixels: []const u8, width: u32, height: u32, uv: Vec2, wrap: FlowWrap) Vec3 {
    if (width == 0 or height == 0) return Vec3.zero;
    const pixel_count = std.math.mul(u32, width, height) catch return Vec3.zero;
    const expected = std.math.mul(u32, pixel_count, 4) catch return Vec3.zero;
    if (pixels.len != @as(usize, expected)) return Vec3.zero;

    var u = uv.x;
    var v = uv.y;
    switch (wrap) {
        .repeat => {
            u = u - @floor(u);
            v = v - @floor(v);
        },
        .clamp => {
            u = std.math.clamp(u, 0.0, 1.0);
            v = std.math.clamp(v, 0.0, 1.0);
        },
    }
    const wf: f32 = @floatFromInt(width);
    const hf: f32 = @floatFromInt(height);
    const fx = u * (wf - 1.0);
    const fy = v * (hf - 1.0);
    const x0: u32 = @intFromFloat(@floor(fx));
    const y0: u32 = @intFromFloat(@floor(fy));
    // fx in [0, w-1] by construction, so x0 <= w-1; same for y.
    const x1 = @min(x0 + 1, width - 1);
    const y1 = @min(y0 + 1, height - 1);
    const tx = fx - @floor(fx);
    const ty = fy - @floor(fy);

    const c00 = flowTexel(pixels, width, x0, y0);
    const c10 = flowTexel(pixels, width, x1, y0);
    const c01 = flowTexel(pixels, width, x0, y1);
    const c11 = flowTexel(pixels, width, x1, y1);
    var rgb: [3]f32 = undefined;
    for (0..3) |ch| {
        const top = c00[ch] + (c10[ch] - c00[ch]) * tx;
        const bot = c01[ch] + (c11[ch] - c01[ch]) * tx;
        rgb[ch] = top + (bot - top) * ty;
    }
    return decodeFlowTap(rgb);
}

// --- Flow-field texture tests (CPU sampling; headless, deterministic) ---

test "flow map unset keeps the legacy path bit-identical" {
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
    const sys = @import("system.zig");
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
    const sys = @import("system.zig");
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
    const sys = @import("system.zig");
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
    const sys = @import("system.zig");
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
    const sys = @import("system.zig");
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
    const sys = @import("system.zig");
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
    const sys = @import("system.zig");
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
