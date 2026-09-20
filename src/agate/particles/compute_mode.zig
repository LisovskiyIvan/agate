//! Stateful compute simulation path (`.compute`, v1). Split out of
//! `particles.zig` (facade).
//!
//! Every function takes the system as `anytype` so this module never imports
//! `system.zig` or the facade back — same discipline as `profiler/*`.
//! `subemitters.zig` stages child spawns via `pushComputeSpawn` (sibling
//! import, no cycle). Upload discipline is unchanged: the update phase only
//! stages CPU bytes and sets flags (never `sg.*`); creation, spawn uploads
//! and dispatch happen in `flushComputeUploads` on the context thread.
//! Moved tests reach `system.zig` helpers through block-scoped imports that
//! exist only in test builds.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const gpu_thread = @import("../gpu_thread.zig");
const upload_meter = @import("../gpu_upload_meter.zig");
const compute = @import("../compute.zig");
const pc_shd = @import("particle_compute_shader");

const types = @import("types.zig");
const sampling = @import("sampling.zig");
const flow = @import("flow.zig");

const Vec3 = math.Vec3;
const GpuParticleSlot = types.GpuParticleSlot;
const ComputeParticleState = types.ComputeParticleState;
const compute_workgroup_size = types.compute_workgroup_size;
const Particle = types.Particle;
const SimulationMode = types.SimulationMode;
const UpdateError = types.UpdateError;
const ComputeModeError = types.ComputeModeError;

// Local aliases so the moved bodies stay byte-identical.
const sampleSpawn = sampling.sampleSpawn;
const rotationToRadians = sampling.rotationToRadians;
const activeFlowCtx = flow.activeFlowCtx;
const SpawnSample = sampling.SpawnSample;
const ParticleInstanceData = types.ParticleInstanceData;
const Color4 = math.Color4;

/// Mode-selecting API for `.compute`. Hard contract, never a silent
/// fallback: `error.InvalidCapacity` for zero-capacity systems,
/// `error.ComputeUnsupported` when the backend is known-unsupported (a
/// live context without compute support, or
/// `compute_support_override == false`). Headless selection (no sg
/// context yet) stages the mode; the first context-side flush latches
/// support authoritatively (see `flushComputeUploads`). Direct field
/// assignment (`ps.simulation_mode = .compute`) skips this validation —
/// same convention as `.gpu` today; prefer this function. `.cpu`/`.gpu`
/// selection never fails (frozen matrix).
pub fn setSimulationMode(self: anytype, mode: SimulationMode) ComputeModeError!void {
    if (mode == .compute) {
        if (self.capacity == 0) return error.InvalidCapacity;
        if (self.compute_support_override) |forced| {
            if (!forced) return error.ComputeUnsupported;
        } else if (sg.isvalid() and !compute.supported()) {
            self.compute_known_unsupported = true;
            return error.ComputeUnsupported;
        }
    }
    self.simulation_mode = mode;
}

/// Whether `.compute` updates may proceed. False only when unsupported is
/// LATCHED (context-side flush) or FORCED (support override) — never from
/// merely lacking a context: headless updates stage CPU-side bookkeeping
/// so tests and pre-context setup work without a GPU.
pub fn computeAvailable(self: anytype) bool {
    if (self.compute_support_override) |forced| return forced;
    return !self.compute_known_unsupported;
}

/// Compute dispatches issued since creation: > 0 proves the `.compute`
/// path ran on a live context (each dispatch is one prepare-boundary
/// flush with staged spawns or a pending dt). 0 on headless/unsupported.
pub fn computeDispatchCount(self: anytype) u64 {
    return self.compute_dispatches;
}

/// Lazily provisions the compute spawn staging. `false` means allocation
/// failure; updateCompute surfaces that as error.OutOfMemory — no CPU
/// fallback. Idempotent (retains capacity across frames, like the
/// prepared-frame lists).
pub fn provisionComputeStaging(self: anytype) bool {
    if (self.compute_staging.len == self.capacity) return true;
    if (self.capacity == 0) return false;
    const staging = self.allocator.alloc(GpuParticleSlot, self.capacity) catch return false;
    if (self.compute_staging.len > 0) self.allocator.free(self.compute_staging);
    self.compute_staging = staging;
    return true;
}

/// Stages one sampled spawn into the compute spawn window. Ring
/// semantics: the window cursor overwrites the oldest slot; when one
/// frame stages more than capacity, the oldest half of the staged window
/// is evicted in bulk (amortized O(1)) so the staged records always stay
/// the latest `min(emitted, capacity)` in emission order with
/// `stage_base` pointing at the first.
pub fn pushComputeSpawn(self: anytype, sample: SpawnSample) void {
    const cap = self.capacity;
    if (self.compute_staged == 0) self.compute_stage_base = self.compute_cursor;
    if (self.compute_staged >= cap) {
        const drop = @max(cap / 2, 1);
        std.mem.copyForwards(
            GpuParticleSlot,
            self.compute_staging[0 .. cap - drop],
            self.compute_staging[drop..cap],
        );
        self.compute_stage_base = (self.compute_stage_base + drop) % cap;
        self.compute_staged = cap - drop;
    }
    self.compute_staging[self.compute_staged] = .{
        .spawn_pos_time = .{ sample.position.x, sample.position.y, sample.position.z, self.clock_seconds },
        .velocity_lifetime = .{ sample.velocity.x, sample.velocity.y, sample.velocity.z, sample.lifetime },
        .color_start = self.color_start.toArray(),
        .color_end = self.color_end.toArray(),
        .size_rotation = .{
            self.size_start,
            self.size_end,
            rotationToRadians(sample.rotation_deg),
            rotationToRadians(sample.angular_velocity),
        },
    };
    self.compute_staged += 1;
    self.compute_cursor = (self.compute_cursor + 1) % cap;
    if (self.compute_high_water < cap) self.compute_high_water += 1;
    // Upper bound only (exact live count lives on the GPU): scene stats
    // and the render skip-check key off active_count, so it tracks the
    // draw count.
    self.active_count = self.compute_high_water;
}

/// Compute-path emission. Staging is provisioned by updateCompute;
/// emissions before provisioning drop instead of allocating on the hot
/// path (same rule as `emitGpuSlot`).
pub fn emitComputeSlot(self: anytype) void {
    if (self.capacity == 0 or self.compute_staging.len < self.capacity) return;
    pushComputeSpawn(self, sampleSpawn(self, self.prng.random()));
}

/// Compute-path frame step: advances the epoch clock, accumulates frame
/// dt for the dispatch, and stages spawn slots — O(emitted), never
/// O(particles). The integration itself happens in the compute pass
/// (shaders/particle_compute.glsl). Explicit support contract: CPU-only
/// feature combinations (`local_space`, armed flow field) and a
/// known-unsupported backend return errors, never a silent CPU downgrade.
/// A `.compute` parent never records deaths (they happen GPU-side), so it
/// never fires sub-emitters — the `.gpu` rule (matrix note (4)) applies.
pub fn updateCompute(self: anytype, dt: f32) UpdateError!void {
    if (self.local_space) return error.LocalSpaceNeedsCpu;
    if (activeFlowCtx(self) != null) return error.FlowMapNeedsCpu;
    if (!computeAvailable(self)) return error.ComputeUnsupported;
    if (self.capacity == 0) return error.InvalidCapacity;
    if (!provisionComputeStaging(self)) return error.OutOfMemory;
    self.clock_seconds += dt;
    self.compute_dt_accum += dt;
    if (self.is_emitting and self.emit_rate > 0.0) {
        self.emit_accumulator += dt * self.emit_rate;
        // No capacity guard: the ring recycles the oldest slot, so a full
        // system never blocks the accumulator (CPU drops instead).
        while (self.emit_accumulator >= 1.0) {
            emitComputeSlot(self);
            self.emit_accumulator -= 1.0;
        }
    }
}

/// Dispatch workgroup count covering the written-slot prefix. Pure so
/// tests pin the bookkeeping without a GPU context (0 slots -> 0 groups,
/// a legal no-op dispatch the flush skips).
pub fn computeGroups(self: anytype) usize {
    return compute.groupCount(self.compute_high_water, compute_workgroup_size);
}

/// Shader parameters for the next dispatch. Pure (no sg.*): snapshot of
/// the staged window (base/count), the accumulated dt, gravity, drag and
/// the spritesheet shape. Uses the generated `CsParams` type directly so
/// the block layout cannot drift from the shdc output.
pub fn buildComputeParams(self: anytype) pc_shd.CsParams {
    return .{
        .dyn = .{ self.compute_dt_accum, self.drag, 0.0, 0.0 },
        .grav = .{ self.gravity.x, self.gravity.y, self.gravity.z, 0.0 },
        .sheet = .{
            @floatFromInt(self.spritesheet_columns),
            @floatFromInt(self.spritesheet_rows),
            self.spritesheet_loops,
            0.0,
        },
        .addr = .{
            @floatFromInt(self.capacity),
            @floatFromInt(self.compute_stage_base),
            @floatFromInt(self.compute_staged),
            0.0,
        },
    };
}

/// Creates compute GPU objects on the context thread (called from
/// flushComputeUploads). Idempotent per object; leaves the pending flag
/// set when anything is still missing so a later flush retries. The
/// state buffer is zero-initialized once at creation (never-written
/// slots read life <= 0 = dead in the shader).
pub fn ensureComputeGpu(self: anytype) void {
    const cap = self.capacity;
    if (self.compute_state_buffer.id == 0) {
        const zeros = self.allocator.alloc(u8, cap * @sizeOf(ComputeParticleState)) catch return;
        defer self.allocator.free(zeros);
        @memset(zeros, 0);
        self.compute_state_buffer = sg.makeBuffer(.{
            .usage = .{ .storage_buffer = true, .dynamic_update = true },
            .data = sg.Range{ .ptr = zeros.ptr, .size = zeros.len },
        });
    }
    if (self.compute_spawn_buffer.id == 0) {
        self.compute_spawn_buffer = sg.makeBuffer(.{
            .usage = .{ .storage_buffer = true, .dynamic_update = true },
            .size = cap * @sizeOf(GpuParticleSlot),
        });
    }
    if (self.compute_draw_buffer.id == 0) {
        self.compute_draw_buffer = sg.makeBuffer(.{
            .usage = .{ .vertex_buffer = true, .storage_buffer = true, .dynamic_update = true },
            .size = cap * @sizeOf(ParticleInstanceData),
        });
    }
    if (self.compute_state_view.id == 0 and self.compute_state_buffer.id != 0) {
        self.compute_state_view = compute.makeStorageView(self.compute_state_buffer, "compute-particles-state");
    }
    if (self.compute_spawn_view.id == 0 and self.compute_spawn_buffer.id != 0) {
        self.compute_spawn_view = compute.makeStorageView(self.compute_spawn_buffer, "compute-particles-spawn");
    }
    if (self.compute_draw_view.id == 0 and self.compute_draw_buffer.id != 0) {
        self.compute_draw_view = compute.makeStorageView(self.compute_draw_buffer, "compute-particles-draw");
    }
    if (self.compute_shader.id == 0) {
        self.compute_shader = sg.makeShader(pc_shd.particleComputeShaderDesc(sg.queryBackend()));
    }
    if (self.compute_pipeline.id == 0 and self.compute_shader.id != 0) {
        self.compute_pipeline = compute.makePipeline(self.compute_shader, "compute-particles");
    }
    if (self.compute_state_buffer.id != 0 and self.compute_spawn_buffer.id != 0 and
        self.compute_draw_buffer.id != 0 and self.compute_state_view.id != 0 and
        self.compute_spawn_view.id != 0 and self.compute_draw_view.id != 0 and
        self.compute_shader.id != 0 and self.compute_pipeline.id != 0)
    {
        self.compute_buffers_pending = false;
    }
}

/// Re-zeroes the GPU state buffer (reset() path). Context thread only;
/// called from flushComputeUploads when staged.
pub fn clearComputeState(self: anytype) void {
    if (self.compute_state_buffer.id == 0) return;
    const zeros = self.allocator.alloc(u8, self.capacity * @sizeOf(ComputeParticleState)) catch return;
    defer self.allocator.free(zeros);
    @memset(zeros, 0);
    sg.updateBuffer(self.compute_state_buffer, sg.Range{ .ptr = zeros.ptr, .size = zeros.len });
    upload_meter.record(zeros.len);
}

/// Runs the compute dispatch for the staged window. Context thread only;
/// caller guarantees live pipeline/views and high_water > 0.
pub fn dispatchCompute(self: anytype, staged: usize) void {
    gpu_thread.assertOnContextThread();
    const params = buildComputeParams(self);
    std.debug.assert(params.addr[2] == @as(f32, @floatFromInt(staged)));
    sg.beginPass(.{ .compute = true, .label = "compute-particles" });
    sg.applyPipeline(self.compute_pipeline);
    var bind = sg.Bindings{};
    bind.views[pc_shd.VIEW_cs_state] = self.compute_state_view;
    bind.views[pc_shd.VIEW_cs_spawn] = self.compute_spawn_view;
    bind.views[pc_shd.VIEW_cs_draw] = self.compute_draw_view;
    sg.applyBindings(bind);
    sg.applyUniforms(pc_shd.UB_cs_params, sg.asRange(&params));
    sg.dispatch(@intCast(computeGroups(self)), 1, 1);
    sg.endPass();
    self.compute_dispatches +%= 1;
}

/// Prepare-boundary compute work: creation, spawn upload, state clear and
/// dispatch. Runs on the sg-context thread (called from
/// `flushGpuUploads`, itself called by `Scene.flushPendingGpuUploads` at
/// render start — the update phase stays free of sg.* calls).
/// Headless (`!sg.isvalid()`) is a safe no-op that keeps every staged
/// flag for a later retry. On a live context WITHOUT compute support the
/// pending flags are dropped and support is latched unsupported, so the
/// next `update` returns error.ComputeUnsupported — never a silent
/// fallback, never a retry spin.
pub fn flushComputeUploads(self: anytype) void {
    if (!sg.isvalid()) return;
    if (!compute.supported()) {
        self.compute_known_unsupported = true;
        self.compute_buffers_pending = false;
        self.compute_flush_pending = false;
        return;
    }
    if (self.compute_buffers_pending) ensureComputeGpu(self);
    if (self.compute_state_buffer.id == 0) return;
    if (self.compute_state_clear_pending) {
        self.compute_state_clear_pending = false;
        clearComputeState(self);
    }
    // Creation incomplete (a make* failed; the pending flag stays set
    // for a retry): keep everything staged, consume nothing.
    if (self.compute_pipeline.id == 0) return;
    const staged = self.compute_staged;
    if (staged > 0 and self.compute_spawn_buffer.id != 0) {
        sg.updateBuffer(self.compute_spawn_buffer, sg.asRange(self.compute_staging[0..staged]));
        upload_meter.record(staged * @sizeOf(GpuParticleSlot));
    }
    // Dispatch on staged spawns even without a frame update (a
    // sub-emitter child may hold spawns while its own update did not run;
    // dt 0 then integrates nothing but still applies respawns). Consumes
    // the window/dt/flag only here, after the upload above.
    if (self.compute_flush_pending or staged > 0) {
        if (self.compute_high_water > 0) dispatchCompute(self, staged);
        self.compute_flush_pending = false;
        self.compute_dt_accum = 0.0;
        self.compute_staged = 0;
    }
}

/// Destroys compute views/pipeline/shader. Context-thread only (same
/// contract as the buffer destroys in `deinit`); skips zeroed handles so
/// retired or never-created objects are safe. Buffers are NOT destroyed
/// here: `deinit` destroys them inline, or `takeGpuBuffersForRetire`
/// zeroes them first for queue teardown.
pub fn deinitComputeGpuObjects(self: anytype) void {
    for ([_]*sg.View{ &self.compute_state_view, &self.compute_spawn_view, &self.compute_draw_view }) |slot| {
        if (slot.*.id != 0) {
            sg.destroyView(slot.*);
            slot.* = .{};
        }
    }
    if (self.compute_pipeline.id != 0) {
        sg.destroyPipeline(self.compute_pipeline);
        self.compute_pipeline = .{};
    }
    if (self.compute_shader.id != 0) {
        sg.destroyShader(self.compute_shader);
        self.compute_shader = .{};
    }
}

/// Collects live GPU buffers into `out` for retire-queue teardown and
/// zeroes the handles (double-call safe; `deinit` skips zeroed buffers).
/// Any thread, sg-free: the queue owns the deferred destroy (epochs), so
/// this is the off-context teardown path — pair with `deinit` (or
/// `deinitComputeGpuObjects` when on-context) for the views/pipeline/
/// shader, which cannot travel through the buffer/mesh retire queue.
/// Returns the number of collected buffers (0 when nothing is live).
/// Covers all five particle buffers (instance, gpu slot, compute
/// state/spawn/draw): a system that switched modes mid-life retires
/// every leftover, never just the active mode's.
pub fn takeGpuBuffersForRetire(self: anytype, out: []sg.Buffer) usize {
    var n: usize = 0;
    for ([_]*sg.Buffer{
        &self.instance_buffer,
        &self.gpu_slot_buffer,
        &self.compute_state_buffer,
        &self.compute_spawn_buffer,
        &self.compute_draw_buffer,
    }) |slot| {
        if (slot.*.id == 0) continue;
        // A short `out` takes a prefix and leaves the rest live for a
        // second call — never zeroes a handle the caller cannot see.
        if (n >= out.len) break;
        out[n] = slot.*;
        n += 1;
        slot.* = .{};
    }
    return n;
}

// --- Stateful compute mode tests (v1; all headless — no sg.* below) ---
//
// What these tests CANNOT verify (needs a live GPU context; the sandbox gets
// a `--test-*` fixture later): pipeline/shader creation, spawn upload bytes,
// the dispatch itself (integration, respawn, death culling), draw-buffer
// contents, and visual parity with `.cpu`/`.gpu`. Covered here instead:
// mode selection/gating, feature-matrix consistency, spawn staging/upload
// bookkeeping, buffer retire lifecycle, capacity/ring policy, and parameter
// validation.

test "compute mode defaults off; state and workgroup layout pinned" {
    const sys = @import("system.zig");
    var ps = try sys.makeTestSystem(std.testing.allocator, 4);
    defer sys.freeTestSystem(&ps);
    // OFF by default: existing systems never opt in implicitly.
    try std.testing.expectEqual(SimulationMode.cpu, ps.simulation_mode);
    try std.testing.expectEqual(@as(usize, 0), ps.compute_staged);
    try std.testing.expectEqual(@as(usize, 0), ps.compute_high_water);
    try std.testing.expectEqual(false, ps.compute_flush_pending);
    try std.testing.expectEqual(false, ps.compute_known_unsupported);
    try std.testing.expectEqual(@as(?bool, null), ps.compute_support_override);
    // Six vec4 (see CState in shaders/particle_compute.glsl).
    try std.testing.expectEqual(@as(usize, 96), @sizeOf(ComputeParticleState));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(ComputeParticleState, "pos_age"));
    try std.testing.expectEqual(@as(usize, 16), @offsetOf(ComputeParticleState, "vel_life"));
    try std.testing.expectEqual(@as(usize, 32), @offsetOf(ComputeParticleState, "rot_seed"));
    try std.testing.expectEqual(@as(usize, 48), @offsetOf(ComputeParticleState, "color_start"));
    try std.testing.expectEqual(@as(usize, 64), @offsetOf(ComputeParticleState, "color_end"));
    try std.testing.expectEqual(@as(usize, 80), @offsetOf(ComputeParticleState, "size_size"));
    // Workgroup matches both the shader (`local_size_x = 64`) and the
    // compute.zig default (groupCount math below depends on it).
    try std.testing.expectEqual(compute.default_workgroup_size, compute_workgroup_size);
    // The draw buffer reuses the cpu instance stride exactly (same pipeline).
    try std.testing.expectEqual(@sizeOf(ParticleInstanceData), 4 * @sizeOf([4]f32));
    // Uniform block layout matches the shdc output byte-for-byte.
    try std.testing.expectEqual(@sizeOf(pc_shd.CsParams), @as(usize, 64));
}

test "compute mode selection gates explicitly, never silently" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&ps);

    // Headless selection (no sg context yet) stages the mode for a later
    // context-side latch — setup before init works.
    try ps.setSimulationMode(.compute);
    try std.testing.expectEqual(SimulationMode.compute, ps.simulation_mode);

    // Forced-unsupported override: explicit error, mode UNCHANGED (no silent
    // fallback to .cpu/.gpu, no partial mode flip).
    ps.simulation_mode = .cpu;
    ps.compute_support_override = false;
    try std.testing.expectError(error.ComputeUnsupported, ps.setSimulationMode(.compute));
    try std.testing.expectEqual(SimulationMode.cpu, ps.simulation_mode);
    // The same error surfaces from update (both entry points).
    ps.simulation_mode = .compute;
    try std.testing.expectError(error.ComputeUnsupported, ps.updateCompute(0.016));
    try std.testing.expectError(error.ComputeUnsupported, ps.update(0.016));
    // Nothing staged, ring untouched, no flush flagged.
    try std.testing.expectEqual(@as(usize, 0), ps.compute_staged);
    try std.testing.expectEqual(@as(usize, 0), ps.compute_high_water);
    try std.testing.expectEqual(@as(usize, 0), ps.active_count);
    try std.testing.expectEqual(false, ps.compute_flush_pending);

    // Zero capacity is a validation error, not a degenerate ring.
    var empty = try sys.makeTestSystem(a, 0);
    defer sys.freeTestSystem(&empty);
    try std.testing.expectError(error.InvalidCapacity, empty.setSimulationMode(.compute));
    try std.testing.expectEqual(SimulationMode.cpu, empty.simulation_mode);

    // Forced-available override restores the path.
    ps.compute_support_override = true;
    try ps.setSimulationMode(.compute);
    try ps.updateCompute(0.016);
    try std.testing.expect(ps.compute_staging.len == 4);

    // `.cpu`/`.gpu` selection never fails (frozen matrix: untouched).
    try ps.setSimulationMode(.cpu);
    try ps.setSimulationMode(.gpu);
    try std.testing.expectEqual(SimulationMode.gpu, ps.simulation_mode);
}

test "compute update rejects cpu-only features like gpu" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeComputeSystem(a, 4);
    defer sys.freeTestSystem(&ps);

    ps.local_space = true;
    try std.testing.expectError(error.LocalSpaceNeedsCpu, ps.updateCompute(0.016));
    try std.testing.expectError(error.LocalSpaceNeedsCpu, ps.update(0.016));
    try std.testing.expectEqual(SimulationMode.compute, ps.simulation_mode);
    try std.testing.expectEqual(@as(usize, 0), ps.compute_high_water);
    ps.local_space = false;

    try ps.updateCompute(0.0); // provisions staging
    var px = [_]u8{ 255, 128, 255, 255 };
    try ps.setFlowMap(null, &px, 1, 1);
    ps.flow_strength = 1.0;
    try std.testing.expectError(error.FlowMapNeedsCpu, ps.updateCompute(0.016));
    try std.testing.expectError(error.FlowMapNeedsCpu, ps.update(0.016));
    try std.testing.expectEqual(SimulationMode.compute, ps.simulation_mode);
    try std.testing.expectEqual(@as(usize, 0), ps.compute_high_water);
    ps.flow_strength = 0.0;
    try ps.update(0.016);
}

test "compute spawn staging shares the cpu sampler bit-for-bit" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var cpu = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&cpu);
    // Same defaults as cpu (mirrors the .gpu sampling test): identical seeds
    // must produce identical spawn records in any mode.
    var gpu = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&gpu);
    gpu.simulation_mode = .compute;
    gpu.emit_rate = 0.0; // manual emission only

    // Sentinel: the compute frame step must never touch the CPU arrays.
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
    gpu.particles[0] = sentinel;

    cpu.emitOne();
    try gpu.updateCompute(0.0); // provisions staging, no emission (rate 0)
    gpu.emitOne(); // staged record 0 at clock 0

    const p = cpu.particles[0];
    const s = gpu.compute_staging[0];
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

    // Bookkeeping: one staged, cursor advanced, high-water + active alias.
    try std.testing.expectEqual(@as(usize, 1), gpu.compute_staged);
    try std.testing.expectEqual(@as(usize, 0), gpu.compute_stage_base);
    try std.testing.expectEqual(@as(usize, 1), gpu.compute_cursor);
    try std.testing.expectEqual(@as(usize, 1), gpu.compute_high_water);
    try std.testing.expectEqual(@as(usize, 1), gpu.active_count);
    // CPU arrays untouched; staging is sg-free (buffers stay zero headless).
    try std.testing.expectEqual(sentinel.age, gpu.particles[0].age);
    try std.testing.expectEqual(@as(u32, 0), gpu.compute_state_buffer.id);
}

test "compute update stages emission and flags the flush, headless-safe" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeComputeSystem(a, 8);
    defer sys.freeTestSystem(&ps);

    // One second at 60 p/s overflows the 8-slot ring: cursor wraps, staging
    // keeps the latest window via bulk eviction (see ring test for order).
    try ps.update(1.0);
    try std.testing.expectEqual(@as(usize, 8), ps.compute_high_water);
    try std.testing.expectEqual(@as(usize, 8), ps.active_count);
    try std.testing.expectEqual(@as(usize, 60 % 8), ps.compute_cursor);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ps.clock_seconds, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ps.compute_dt_accum, 1e-6);
    try std.testing.expectEqual(true, ps.compute_buffers_pending);
    try std.testing.expectEqual(true, ps.compute_flush_pending);
    // Headless flush: safe no-op, keeps every staged flag for a later retry
    // (upload discipline — no sg.* without a context).
    ps.flushGpuUploads();
    try std.testing.expectEqual(true, ps.compute_buffers_pending);
    try std.testing.expectEqual(true, ps.compute_flush_pending);
    try std.testing.expectEqual(@as(usize, 8), ps.compute_staged);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ps.compute_dt_accum, 1e-6);
}

test "compute ring recycles oldest; over-capacity frames keep the latest window" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeComputeSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    ps.emit_rate = 0.0;
    try ps.updateCompute(0.0); // provision

    // Stamp each emission with a distinct clock: the spawn time tags order.
    var i: usize = 0;
    while (i < 6) : (i += 1) {
        ps.clock_seconds = @floatFromInt(i);
        ps.emitOne();
    }
    try std.testing.expectEqual(@as(usize, 2), ps.compute_cursor);
    try std.testing.expectEqual(@as(usize, 4), ps.compute_high_water);
    try std.testing.expectEqual(@as(usize, 4), ps.active_count);
    // Six emissions into a 4-wide window: bulk eviction kept the latest four
    // (clocks 2..5) with the base advanced past the evicted pair.
    try std.testing.expectEqual(@as(usize, 4), ps.compute_staged);
    try std.testing.expectEqual(@as(usize, 2), ps.compute_stage_base);
    try std.testing.expectEqual(@as(f32, 2.0), ps.compute_staging[0].spawn_pos_time[3]);
    try std.testing.expectEqual(@as(f32, 3.0), ps.compute_staging[1].spawn_pos_time[3]);
    try std.testing.expectEqual(@as(f32, 4.0), ps.compute_staging[2].spawn_pos_time[3]);
    try std.testing.expectEqual(@as(f32, 5.0), ps.compute_staging[3].spawn_pos_time[3]);

    // Single-frame overflow via burst: same latest-window rule.
    var ps2 = try sys.makeComputeSystem(a, 4);
    defer sys.freeTestSystem(&ps2);
    ps2.emit_rate = 0.0;
    try ps2.updateCompute(0.0);
    ps2.burst(6);
    try std.testing.expectEqual(@as(usize, 2), ps2.compute_cursor);
    try std.testing.expectEqual(@as(usize, 4), ps2.compute_high_water);
    try std.testing.expectEqual(@as(usize, 4), ps2.compute_staged);

    // Reset re-anchors everything and stages a GPU state clear (consumed by
    // the next context flush; headless it stays staged).
    ps.reset();
    try std.testing.expectEqual(@as(usize, 0), ps.compute_cursor);
    try std.testing.expectEqual(@as(usize, 0), ps.compute_high_water);
    try std.testing.expectEqual(@as(usize, 0), ps.compute_staged);
    try std.testing.expectEqual(@as(usize, 0), ps.compute_stage_base);
    try std.testing.expectEqual(@as(usize, 0), ps.active_count);
    try std.testing.expectEqual(@as(f32, 0.0), ps.clock_seconds);
    try std.testing.expectEqual(@as(f32, 0.0), ps.compute_dt_accum);
    try std.testing.expectEqual(true, ps.compute_state_clear_pending);
}

test "compute dispatch groups and shader params are pure snapshots" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeComputeSystem(a, 130);
    defer sys.freeTestSystem(&ps);
    ps.emit_rate = 0.0;
    try ps.updateCompute(0.0);

    try std.testing.expectEqual(@as(usize, 0), ps.computeGroups());
    ps.burst(1);
    try std.testing.expectEqual(@as(usize, 1), ps.computeGroups());
    ps.burst(63);
    try std.testing.expectEqual(@as(usize, 1), ps.computeGroups());
    ps.burst(1); // 65 written -> 2 groups of 64
    try std.testing.expectEqual(@as(usize, 2), ps.computeGroups());
    try std.testing.expectEqual(ps.compute_high_water, ps.compute_staged);

    ps.gravity = Vec3.new(0.0, -9.8, 0.0);
    ps.drag = 1.5;
    ps.compute_dt_accum = 0.016;
    ps.spritesheet_columns = 4;
    ps.spritesheet_rows = 2;
    ps.spritesheet_loops = 3.0;
    const params = ps.buildComputeParams();
    try std.testing.expectEqual([4]f32{ 0.016, 1.5, 0.0, 0.0 }, params.dyn);
    try std.testing.expectEqual([4]f32{ 0.0, -9.8, 0.0, 0.0 }, params.grav);
    try std.testing.expectEqual([4]f32{ 4.0, 2.0, 3.0, 0.0 }, params.sheet);
    try std.testing.expectEqual([4]f32{ 130.0, 0.0, 65.0, 0.0 }, params.addr);
    try std.testing.expectEqual(@sizeOf(pc_shd.CsParams), @as(usize, 64));
}

test "compute takeGpuBuffersForRetire collects, zeroes, and never double-takes" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeComputeSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    // Fake borrowed ids (headless: never through sg.destroyBuffer here).
    ps.instance_buffer = .{ .id = 11 };
    ps.gpu_slot_buffer = .{ .id = 12 };
    ps.compute_state_buffer = .{ .id = 21 };
    ps.compute_spawn_buffer = .{ .id = 22 };
    ps.compute_draw_buffer = .{ .id = 23 };

    var out = [_]sg.Buffer{.{}} ** 8;
    const n = ps.takeGpuBuffersForRetire(&out);
    try std.testing.expectEqual(@as(usize, 5), n);
    try std.testing.expectEqual(@as(u32, 11), out[0].id);
    try std.testing.expectEqual(@as(u32, 12), out[1].id);
    try std.testing.expectEqual(@as(u32, 21), out[2].id);
    try std.testing.expectEqual(@as(u32, 22), out[3].id);
    try std.testing.expectEqual(@as(u32, 23), out[4].id);
    // Handles zeroed: deinit skips them, a second take finds nothing.
    try std.testing.expectEqual(@as(u32, 0), ps.instance_buffer.id);
    try std.testing.expectEqual(@as(u32, 0), ps.compute_state_buffer.id);
    try std.testing.expectEqual(@as(usize, 0), ps.takeGpuBuffersForRetire(&out));
    // Views/pipeline/shader with zero ids destroy as a safe no-op set
    // (guards only; no sg.* effect headless).
    ps.deinitComputeGpuObjects();
}

test "compute staging provision is idempotent and reset retains capacity" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var ps = try sys.makeComputeSystem(a, 4);
    defer sys.freeTestSystem(&ps);
    try ps.updateCompute(0.016);
    const first_ptr = ps.compute_staging.ptr;
    try ps.updateCompute(0.016); // no re-alloc
    try std.testing.expectEqual(first_ptr, ps.compute_staging.ptr);
    try std.testing.expectEqual(@as(usize, 4), ps.compute_staging.len);
    ps.emitOne();
    ps.reset();
    // Allocation retained (like the prepared-frame lists), content dropped.
    try std.testing.expectEqual(@as(usize, 4), ps.compute_staging.len);
    try std.testing.expectEqual(@as(usize, 0), ps.compute_staged);
}

test "cpu/gpu/compute full-system policies stay distinct" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var cpu = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&cpu);
    var gpu = try sys.makeTestSystem(a, 4);
    defer sys.freeTestSystem(&gpu);
    gpu.simulation_mode = .gpu;
    gpu.gpu_slots = try a.alloc(GpuParticleSlot, 4);
    var cmp = try sys.makeComputeSystem(a, 4);
    defer sys.freeTestSystem(&cmp);
    cmp.emit_rate = 0.0;
    try cmp.updateCompute(0.0);

    cpu.burst(6);
    gpu.burst(6);
    cmp.burst(6);
    // CPU drops when full; both GPU rings recycle the oldest slot.
    try std.testing.expectEqual(@as(usize, 4), cpu.active_count);
    try std.testing.expectEqual(@as(usize, 4), gpu.active_count);
    try std.testing.expectEqual(@as(usize, 2), gpu.gpu_write_cursor);
    try std.testing.expectEqual(@as(usize, 4), cmp.active_count);
    try std.testing.expectEqual(@as(usize, 2), cmp.compute_cursor);
    try std.testing.expectEqual(@as(usize, 4), cmp.compute_staged);
}

test "sub-emitter spawns into a compute child staged window" {
    const sys = @import("system.zig");
    const a = std.testing.allocator;
    var parent = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&parent);
    var child = try sys.makeQuiescentSystem(a, 4);
    defer sys.freeTestSystem(&child);
    child.simulation_mode = .compute;
    child.lifetime_min = 10.0;
    child.lifetime_max = 10.0;
    try child.updateCompute(0.0); // provision staging, no emission

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
    // One record staged at the death position with the inherited velocity
    // (0.5 * (4,0,0) + 0.5 * sampled(0,0,0) = (2,0,0)).
    try std.testing.expectEqual(@as(usize, 1), child.compute_high_water);
    try std.testing.expectEqual(@as(f32, 1.0), child.compute_staging[0].spawn_pos_time[0]);
    try std.testing.expectEqual(@as(f32, 2.0), child.compute_staging[0].spawn_pos_time[1]);
    try std.testing.expectEqual(@as(f32, 3.0), child.compute_staging[0].spawn_pos_time[2]);
    try std.testing.expectEqual(@as(f32, 2.0), child.compute_staging[0].velocity_lifetime[0]);
    try std.testing.expectEqual(@as(f32, 10.0), child.compute_staging[0].velocity_lifetime[3]);
}
