//! Prepared particle frame handoff: update/render isolation for particles.
//!
//! Ownership block:
//! - UPDATE (`update`, game side): advances the CPU simulation only —
//!   `ParticleSystem.update` stages instance/slot bytes and sets the dirty
//!   flags, never touching sg.* (see particles.zig: the init defers buffer
//!   creation off-context, update only stages, flushGpuUploads owns sg).
//! - PREPARE (`captureFrame`, context side, called by the integrator AFTER
//!   `flushGpuUploads` and BEFORE publish): copies one plain
//!   `ParticleDraw` value per live system out of `systems` into the retained
//!   `frame` list. sg-free (handle ids are copied as values, no sg.*
//!   calls). Capture-after-flush matters: the flush creates deferred
//!   buffers, so capturing earlier would snapshot stale zero ids.
//! - RENDER (`renderPrepared`, context side): draws ONLY `frame` through the
//!   render-owned pass (MSAA twin lazy-created here, disjoint from game
//!   state) plus snapshot-count stats. Never reads `systems`, never the
//!   live texture pointer — only the borrowed view id in the record.
//! - GPU handles in the frame are BORROWED: instance/gpu-slot buffers and
//!   texture views stay owned by their ParticleSystem (via this layer's
//!   `systems` list). The owner must live until context teardown; replacing
//!   or destroying a system/texture between captureFrame and renderPrepared
//!   without a recapture is a caller-obligation violation. No full CPU
//!   geometry copy: the instance/slot bytes already sit in the borrowed GPU
//!   buffers after the prepare flush.
//! - Failure coherence: the single reserve lands BEFORE any publish (OOM
//!   gives a coherent empty frame, never a stale half); retained capacity
//!   is reused across captures and freed in deinit.
//! - Lifetime audit (existing API, unchanged): `create` only calls
//!   `ParticleSystem.init` (off-context safe: defers sg buffer creation to
//!   flushGpuUploads) and appends — no new sg.* introduced here.
//!   `ParticleSystem.deinit` destroys sg buffers + the owned texture inline
//!   (Texture.deinit issues sg.destroy*), so `ParticleLayer.deinit` — like
//!   any direct `ps.deinit()` — is context-thread only per the existing API
//!   (no retire queue involved). Dynamic creation itself is safe from any
//!   thread (`Scene.createParticleSystem`, e.g. main.zig demo spawn carries
//!   no sg calls); only teardown needs the context. This layer adds no new
//!   lifetime operations beyond the retained `frame` list (plain values,
//!   freed in deinit with the given allocator).
//!
//! Headless note: `update`/`captureFrame`/stats are sg-free, so unit tests
//! below exercise the packet, the policy and the byte math purely on CPU
//! with fake borrowed handle ids. Real draws need the live harness (parent
//! lane). `render` (immediate, from live systems) is kept for concrete
//! standalone usage; both it and `renderPrepared` share the single sg
//! algorithm in `ParticlePass.drawRecord`.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const particles = @import("../particles.zig");
const ParticleSystem = particles.ParticleSystem;
const Camera = @import("../camera.zig").Camera;
const passes = @import("../passes/mod.zig");
const stats_mod = @import("stats.zig");
const SceneStats = stats_mod.SceneStats;

/// Prepared per-system draw record, re-exported here so the integrator and
/// core fixtures inspect a single type from either side:
/// `ParticleLayer.ParticleDraw` === `ParticlePass.ParticleDraw`.
/// PLAIN values only — no *ParticleSystem, no game-texture pointer.
pub const ParticleDraw = passes.ParticlePass.ParticleDraw;

/// CPU particle systems and their GPU billboard pass. Owns the system list
/// (created via Scene.createParticleSystem) plus the retained prepared
/// frame consumed by `renderPrepared`.
pub const ParticleLayer = struct {
    systems: std.ArrayListUnmanaged(*ParticleSystem) = .empty,

    /// Retained owning prepared frame: one plain `ParticleDraw` per live
    /// system, published by `captureFrame`, consumed by `renderPrepared`.
    frame: std.ArrayListUnmanaged(ParticleDraw) = .empty,

    pass: passes.ParticlePass,

    // MSAA twin pass (render pipeline sample counts must match the main
    // target). Lazily created on the first MSAA frame, recreated on count
    // changes. Compute simulation always runs through the 1x pass: compute
    // pipelines have no attachments and are sample-count independent.
    pass_msaa: ?passes.ParticlePass = null,

    pub fn init() ParticleLayer {
        return .{ .pass = passes.ParticlePass.init() };
    }

    pub fn deinit(self: *ParticleLayer, allocator: std.mem.Allocator) void {
        // Context-thread only (existing API): ps.deinit destroys sg buffers
        // and the owned texture inline. The retained frame holds plain
        // values (borrowed ids, never destroyed here).
        for (self.systems.items) |ps| {
            ps.deinit();
            allocator.destroy(ps);
        }
        self.systems.deinit(allocator);
        self.frame.deinit(allocator);
        self.pass.deinit();
        if (self.pass_msaa) |*p| p.deinit();
        self.pass_msaa = null;
    }

    pub fn create(self: *ParticleLayer, allocator: std.mem.Allocator, name: []const u8, capacity: usize) !*ParticleSystem {
        // No sg.* here: ParticleSystem.init defers buffer creation when
        // called off the context thread (see particles.zig).
        const ps = try ParticleSystem.init(allocator, name, capacity);
        try self.systems.append(allocator, ps);
        return ps;
    }

    pub fn update(self: *ParticleLayer, dt: f32) particles.UpdateError!void {
        // CPU simulation + dirty-flag staging only; the sg uploads happen in
        // flushGpuUploads on the render side.
        for (self.systems.items) |ps| {
            try ps.update(dt);
        }
    }

    /// Captures the prepared frame out of the live systems. sg-free: copies
    /// plain values only. Integrator protocol: call AFTER flushGpuUploads
    /// (deferred buffers exist by then) and BEFORE publish, on the context
    /// side. Zero systems — or no system with active_count > 0 — captures a
    /// coherent empty frame. OOM fail-closes to coherent-empty (no stale
    /// records); retained capacity is reused, never shrunk here.
    pub fn captureFrame(self: *ParticleLayer, allocator: std.mem.Allocator) void {
        if (self.systems.items.len == 0) {
            self.clearFrame();
            return;
        }
        // Single reserve BEFORE publish: either the whole frame lands or the
        // frame is coherent-empty (no old-record/new-record mix on OOM).
        self.frame.ensureTotalCapacity(allocator, self.systems.items.len) catch {
            self.clearFrame();
            return;
        };
        self.frame.clearRetainingCapacity();
        for (self.systems.items) |ps| {
            // Zero-count systems draw and count nothing under the legacy
            // semantics, so they occupy no frame slot; an all-empty layer
            // captures a coherent empty frame.
            if (ps.active_count == 0) continue;
            self.frame.append(allocator, ParticleDraw.fromSystem(ps)) catch {
                self.clearFrame();
                return;
            };
        }
    }

    /// Drops the frame CONTENT to coherent-empty, retaining capacity for the
    /// next capture.
    fn clearFrame(self: *ParticleLayer) void {
        self.frame.clearRetainingCapacity();
    }

    /// Renders the prepared frame inside the main pass. `samples` is the
    /// effective main-target sample count (scene/msaa.zig). Reads ONLY
    /// `frame` and the render-owned pass — never live systems. Upload-free:
    /// the prepare flush already moved every staged byte into the borrowed
    /// buffers. Stats come from the snapshot counts with the legacy
    /// semantics (active_count > 0 counts one draw call + two triangles per
    /// particle). Headless-safe: without an sg context this is a no-op that
    /// touches neither the pass (no passFor/MSAA creation) nor the stats —
    /// a nonempty frame over an undefined pass draws nothing. The immediate
    /// `render` below stays context-only by contrast.
    pub fn renderPrepared(self: *ParticleLayer, camera: Camera, aspect: f32, samples: i32, stats: *SceneStats) void {
        if (!sg.isvalid()) return;
        if (self.frame.items.len == 0) return;
        const pass = self.passFor(samples);
        pass.renderDraws(self.frame.items, camera, aspect);
        const s = passes.ParticlePass.statsForDraws(self.frame.items);
        stats.main_draw_calls += s.draw_calls;
        stats.draw_calls += s.draw_calls;
        stats.triangles += s.triangles;
    }

    /// Immediate render from the live systems (concrete standalone usage,
    /// e.g. tooling/tests with a context). The sg algorithm is shared with
    /// `renderPrepared` via `ParticlePass.drawRecord` — this wrapper only
    /// selects the source (live systems vs retained frame).
    pub fn render(self: *ParticleLayer, camera: Camera, aspect: f32, samples: i32, stats: *SceneStats) void {
        if (self.systems.items.len == 0) return;
        const pass = self.passFor(samples);
        pass.render(self.systems.items, camera, aspect);
        for (self.systems.items) |ps| {
            if (ps.active_count > 0) {
                stats.main_draw_calls += 1;
                stats.draw_calls += 1;
                stats.triangles += 2 * @as(u32, @intCast(ps.active_count));
            }
        }
    }

    /// Pass variant matching the target sample count.
    fn passFor(self: *ParticleLayer, samples: i32) *passes.ParticlePass {
        if (samples <= 1) return &self.pass;
        if (self.pass_msaa == null or self.pass_msaa.?.sample_count != samples) {
            if (self.pass_msaa) |*p| p.deinit();
            self.pass_msaa = passes.ParticlePass.initSampled(samples);
        }
        return &self.pass_msaa.?;
    }
};

// --- GPU-free frame tests (no sg.* calls below this line; fake borrowed
// handle ids only) ---

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
    // Re-export identity: one type from either side for fixtures.
    try std.testing.expect(ParticleDraw == passes.ParticlePass.ParticleDraw);
}

test "particle captureFrame packet is immutable under live mutations" {
    const t = std.testing;
    const math_mod = @import("math");
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

    // Mutate every live field past recognition, including count, mode,
    // borrowed handles, texture presence, clock and gravity.
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

    // Stats from the snapshot preserve the legacy semantics.
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

    // Zero systems -> coherent empty.
    layer.captureFrame(t.allocator);
    try t.expectEqual(@as(usize, 0), layer.frame.items.len);

    // Systems with only empty counts -> coherent empty (no stale slots).
    var ps = try makeLayerTestSystem(t.allocator, 4);
    defer freeLayerTestSystem(&ps);
    ps.active_count = 0;
    try layer.systems.append(t.allocator, &ps);
    layer.captureFrame(t.allocator);
    try t.expectEqual(@as(usize, 0), layer.frame.items.len);

    // Live then emptied -> empty again, never the prior records.
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

    // Smaller live set: length shrinks, capacity retained for the next frame.
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

    // Funded capture publishes.
    layer.captureFrame(t.allocator);
    try t.expectEqual(@as(usize, 1), layer.frame.items.len);

    // Unfunded capture fail-closes to coherent-empty (prior record gone).
    var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    // Drop the retained capacity so the reserve below really allocates.
    layer.frame.clearAndFree(t.allocator);
    layer.captureFrame(failing.allocator());
    try t.expectEqual(@as(usize, 0), layer.frame.items.len);

    // Recovery: a funded capture publishes the full frame again.
    layer.captureFrame(t.allocator);
    try t.expectEqual(@as(usize, 1), layer.frame.items.len);
    try t.expectEqual(@as(usize, 2), layer.frame.items[0].active_count);
}

test "particle renderPrepared is a headless no-op over a nonempty frame" {
    const t = std.testing;
    const cam_mod = @import("../camera.zig");
    const meter = @import("../gpu_upload_meter.zig");
    _ = meter.takeAndReset();
    // Undefined GPU pass on purpose: headless renderPrepared must return
    // before touching the pass (including the MSAA passFor path).
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
    layer.renderPrepared(cam, 16.0 / 9.0, 1, &stats);
    try t.expectEqual(stats_mod.SceneStats{}, stats);
    try t.expectEqual(@as(u64, 0), meter.peek());
    // MSAA sample count would create the twin pass on a live context;
    // headless it stays a no-op with zero stats.
    layer.renderPrepared(cam, 16.0 / 9.0, 4, &stats);
    try t.expectEqual(stats_mod.SceneStats{}, stats);
    try t.expectEqual(@as(u64, 0), meter.peek());
    try t.expect(layer.pass_msaa == null);
}
