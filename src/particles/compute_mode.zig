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
const collisions = @import("collisions.zig");

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
const activeCollisionCtx = collisions.activeCollisionCtx;
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
/// feature combinations (`local_space`, armed flow field, armed collisions)
/// and a
/// known-unsupported backend return errors, never a silent CPU downgrade.
/// A `.compute` parent never records deaths (they happen GPU-side), so it
/// never fires sub-emitters — the `.gpu` rule (matrix note (4)) applies.
pub fn updateCompute(self: anytype, dt: f32) UpdateError!void {
    if (self.local_space) return error.LocalSpaceNeedsCpu;
    if (activeFlowCtx(self) != null) return error.FlowMapNeedsCpu;
    if (activeCollisionCtx(self) != null) return error.CollisionNeedsCpu;
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
            .usage = .{ .storage_buffer = true },
            .data = sg.Range{ .ptr = zeros.ptr, .size = zeros.len },
        });
    }
    if (self.compute_spawn_buffer.id == 0) {
        self.compute_spawn_buffer = sg.makeBuffer(.{
            .usage = .{ .storage_buffer = true, .write_transient = true },
            .size = cap * @sizeOf(GpuParticleSlot),
        });
    }
    if (self.compute_draw_buffer.id == 0) {
        self.compute_draw_buffer = sg.makeBuffer(.{
            .usage = .{ .vertex_buffer = true, .storage_buffer = true },
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
    if (self.compute_state_view.id != 0) {
        sg.destroyView(self.compute_state_view);
        self.compute_state_view = .{};
    }
    sg.destroyBuffer(self.compute_state_buffer);
    self.compute_state_buffer = .{};
    ensureComputeGpu(self);
    upload_meter.record(self.capacity * @sizeOf(ComputeParticleState));
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

/// Quiesced/direct context-side compute work: creation, spawn upload, state
/// clear and dispatch through `flushGpuUploads`. Normal Scene frames instead
/// dispatch frozen slot packets and commit outcomes on the producer; update
/// stays free of sg.* calls.
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
    const will_dispatch = (self.compute_flush_pending or staged > 0) and self.compute_high_water > 0;
    if (self.compute_spawn_buffer.id != 0) {
        if (staged > 0) {
            sg.writeBufferTransient(.{
                .dst = .{ .buffer = self.compute_spawn_buffer },
                .src = .{ .data = sg.asRange(self.compute_staging[0..staged]) },
            });
            upload_meter.record(staged * @sizeOf(GpuParticleSlot));
        } else if (will_dispatch and self.compute_staging.len > 0) {
            sg.writeBufferTransient(.{
                .dst = .{ .buffer = self.compute_spawn_buffer },
                .src = .{ .data = sg.asRange(self.compute_staging[0..1]) },
            });
        }
    }
    // Dispatch on staged spawns even without a frame update (a
    // sub-emitter child may hold spawns while its own update did not run;
    // dt 0 then integrates nothing but still applies respawns). Consumes
    // the window/dt/flag only here, after the upload above.
    if (will_dispatch) {
        dispatchCompute(self, staged);
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

// Stateful compute mode tests (v1; all headless — no sg.* below):
// moved verbatim to `compute_mode_tests.zig`; the block below keeps them
// reachable from this module (mesh.zig facade pattern).
