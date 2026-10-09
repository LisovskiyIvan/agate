//! Prepared particle frame handoff: update/render isolation for particles.
//!
//! Ownership block:
//! - UPDATE (`update`, game side): advances the CPU simulation only —
//!   `ParticleSystem.update` stages instance/slot bytes and sets the dirty
//!   flags, never touching sg.* (see particles.zig: the init defers buffer
//!   creation off-context, update only stages, flushGpuUploads owns sg).
//! - BUILD (`buildCapture`, game side, stage 1): copies one plain
//!   `ParticleDraw` value per live system out of `systems` into the retained
//!   game-owned `build_frame` and stamps `build_seq`. sg-free (handle ids
//!   are copied as values, no sg.* calls), callable from any non-pool thread
//!   under update-vs-prepare exclusion. Capture-after-flush still matters
//!   for the latch below: the staged-begin flush creates deferred buffers, so a
//!   build from before the flush would snapshot stale zero ids — apps must
//!   build AFTER the sim mutations of the tick whose flush the staged begin
//!   will run (same ordering the direct capture path needs).
//! - PREPARE (context side, called by the staged begin AFTER
//!   `flushGpuUploads` and BEFORE publish): publishes the retained `frame`
//!   list. `latchFrame` copies `build_frame` → `frame` when `build_seq` is
//!   newer than `latched_seq` (reserve-once, OOM coherent-empty, same as
//!   the direct capture); otherwise it runs the direct live capture, so a
//!   latch without a fresh build stays coherent (direct/tooling path — Scene
//!   itself always builds via buildCapture → stageIntoSlot → latchSlotFrame).
//! - RENDER (`renderPrepared`, context side): draws ONLY `frame` through the
//!   render-owned pass (MSAA twin lazy-created here, disjoint from game
//!   state) plus snapshot-count stats. Never reads `systems`, never
//!   `build_frame`, never the live texture pointer — only the borrowed view
//!   id in the record.
//! - Wave 32 freeze-then-latch (adopted concurrent-build path): the game
//!   build additionally freezes the capture into the claimed draw slot
//!   (`stageIntoSlot`) and the prepare latch consumes the slot copy
//!   (`latchSlotFrame`) — never the shared `build_frame` — so a
//!   game-thread build colliding with the context-side latch cannot tear
//!   the record. The shared store below stays for the sequential flow
//!   (bit-identical) and direct/tooling use.
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
    /// system, published by `captureFrame` (direct layer path) or `latchFrame`
    /// (producer build path), consumed by `renderPrepared`.
    frame: std.ArrayListUnmanaged(ParticleDraw) = .empty,

    /// Game-owned build frame (stage 1): written by `buildCapture` on the
    /// update side, consumed by `latchFrame` on the context side. Plain
    /// values only, same shape as `frame`; freed in deinit.
    build_frame: std.ArrayListUnmanaged(ParticleDraw) = .empty,
    /// `Scene.build_seq` stamped by the last `buildCapture` (0 = never).
    build_seq: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
    /// Last `build_seq` consumed by `latchFrame`.
    latched_seq: usize = 0,

    pass: passes.ParticlePass,

    // Variant pass (render pipeline sample counts must match the main
    // target). Lazily created on the first non-base frame, recreated on
    // shape changes. Compute simulation always runs through the 1x pass:
    // compute pipelines have no attachments and are sample-count independent.
    pass_msaa: ?passes.ParticlePass = null,

    pub fn init() ParticleLayer {
        return .{ .pass = passes.ParticlePass.init(1, .RGBA16F) };
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
        self.build_frame.deinit(allocator);
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
    ///
    /// Direct layer capture (also used by layer fixtures/tooling):
    /// `latchFrame` falls back to this when no fresh producer build exists.
    /// Scene itself always runs producer buildCapture → stageIntoSlot →
    /// context latchSlotFrame.
    pub fn captureFrame(self: *ParticleLayer, allocator: std.mem.Allocator) void {
        captureInto(self.systems.items, allocator, &self.frame);
    }

    /// Game-side CPU capture (stage 1): the exact `captureFrame` logic
    /// writing the retained game-owned `build_frame` instead of `frame`,
    /// stamped with the scene `build_seq`. sg-free; callable from any
    /// non-pool thread under update-vs-prepare exclusion. OOM fail-closes
    /// `build_frame` to coherent-empty (mirroring `captureFrame`); the seq
    /// still advances — the empty IS the new state — so the latch publishes
    /// it instead of a stale prior frame.
    pub fn buildCapture(self: *ParticleLayer, allocator: std.mem.Allocator, seq: usize) void {
        captureInto(self.systems.items, allocator, &self.build_frame);
        self.build_seq.store(seq, .release);
    }

    /// Context-side latch (stage 1): when a fresh build exists (`build_seq`
    /// newer than `latched_seq`), copies `build_frame` → `frame`
    /// (reserve-once, OOM coherent-empty) and advances `latched_seq`.
    /// Otherwise runs the direct live capture, so a latch without a
    /// fresh build stays coherent (direct/tooling path). `renderPrepared` keeps reading `frame`
    /// only — never `build_frame`, never live systems.
    ///
    /// Sequential/standalone path only since wave 32: the adopted
    /// concurrent-build path freezes into the claimed draw slot
    /// (`stageIntoSlot`) and latches from it (`latchSlotFrame`), so the
    /// prepare latch there never reads the shared `build_frame` — a
    /// game-thread build colliding with the context-side latch cannot
    /// tear the record. This shared-store latch stays for apps on the
    /// sequential flow (bit-identical) and for direct/tooling use.
    pub fn latchFrame(self: *ParticleLayer, allocator: std.mem.Allocator) void {
        const fresh_seq = self.build_seq.load(.acquire);
        if (fresh_seq == self.latched_seq) {
            self.captureFrame(allocator);
            return;
        }
        self.latched_seq = fresh_seq;
        self.frame.ensureTotalCapacity(allocator, self.build_frame.items.len) catch {
            self.clearFrame();
            return;
        };
        self.frame.clearRetainingCapacity();
        for (self.build_frame.items) |draw| {
            self.frame.appendAssumeCapacity(draw);
        }
    }

    /// Freeze-then-latch slot stage (wave 32, adopted concurrent-build
    /// path): copies the just-captured `build_frame` into the claimed
    /// slot's `particle_draws`. Runs producer-side inside
    /// `Scene.buildIntoClaimedSlot` right after `buildCapture`; the frozen
    /// copy rides the publish release edge (`build_slot`/`build_seq`) to
    /// the prepare latch, which consumes it via `latchSlotFrame` — never
    /// the shared `build_frame`. Staged-wins on OOM (fail-closes the slot
    /// copy to coherent-empty, mirroring `captureInto`); the shared
    /// `build_frame` keeps its own existing semantics for direct use.
    pub fn stageIntoSlot(self: *ParticleLayer, allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(ParticleDraw)) void {
        out.ensureTotalCapacity(allocator, self.build_frame.items.len) catch {
            out.clearRetainingCapacity();
            return;
        };
        out.clearRetainingCapacity();
        out.appendSliceAssumeCapacity(self.build_frame.items);
    }

    /// Context-side slot latch (wave 32, adopted concurrent-build path):
    /// copies the claimed slot's frozen `draws` into `frame`
    /// (reserve-once, OOM coherent-empty) and consumes the pending build
    /// generation (`latched_seq` catches up to `build_seq`, acquire-loaded
    /// — so a later fallback latch runs the live capture, never a stale
    /// build). Reads only the slot payload + the seq word: never the
    /// shared `build_frame`, never live systems.
    pub fn latchSlotFrame(self: *ParticleLayer, allocator: std.mem.Allocator, draws: []const ParticleDraw) void {
        const fresh_seq = self.build_seq.load(.acquire);
        self.latched_seq = fresh_seq;
        self.frame.ensureTotalCapacity(allocator, draws.len) catch {
            self.clearFrame();
            return;
        };
        self.frame.clearRetainingCapacity();
        self.frame.appendSliceAssumeCapacity(draws);
    }

    /// Shared capture body: one plain `ParticleDraw` per live system with
    /// active_count > 0 into `out`. Single reserve BEFORE publish (either
    /// the whole frame lands or the frame is coherent-empty); retained
    /// capacity is reused, never shrunk here.
    fn captureInto(systems: []*ParticleSystem, allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(ParticleDraw)) void {
        if (systems.len == 0) {
            out.clearRetainingCapacity();
            return;
        }
        // Single reserve BEFORE publish: either the whole frame lands or the
        // frame is coherent-empty (no old-record/new-record mix on OOM).
        out.ensureTotalCapacity(allocator, systems.len) catch {
            out.clearRetainingCapacity();
            return;
        };
        out.clearRetainingCapacity();
        for (systems) |ps| {
            // Zero-count systems draw and count nothing under the count
            // semantics, so they occupy no frame slot; an all-empty layer
            // captures a coherent empty frame.
            if (ps.active_count == 0) continue;
            out.append(allocator, ParticleDraw.fromSystem(ps)) catch {
                out.clearRetainingCapacity();
                return;
            };
        }
    }

    /// Drops the frame CONTENT to coherent-empty, retaining capacity for the
    /// next capture.
    fn clearFrame(self: *ParticleLayer) void {
        self.frame.clearRetainingCapacity();
    }

    /// Renders the prepared frame inside the main pass. `samples`/
    /// `color_format` pin the exact main-target shape. Reads ONLY
    /// `frame` and the render-owned pass — never live systems. Upload-free:
    /// the prepare flush already moved every staged byte into the borrowed
    /// buffers. Stats come from the snapshot counts. Headless-safe: without
    /// an sg context this is a no-op that touches neither the pass (no
    /// passFor creation) nor the stats — a nonempty frame over an undefined
    /// pass draws nothing. The immediate `render` below stays context-only
    /// by contrast.
    pub fn renderPrepared(self: *ParticleLayer, camera: Camera, aspect: f32, samples: i32, color_format: sg.PixelFormat, stats: *SceneStats) void {
        if (!sg.isvalid()) return;
        if (self.frame.items.len == 0) return;
        const pass = self.passFor(samples, color_format);
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
    pub fn render(self: *ParticleLayer, camera: Camera, aspect: f32, samples: i32, color_format: sg.PixelFormat, stats: *SceneStats) void {
        if (self.systems.items.len == 0) return;
        const pass = self.passFor(samples, color_format);
        pass.render(self.systems.items, camera, aspect);
        for (self.systems.items) |ps| {
            if (ps.active_count > 0) {
                stats.main_draw_calls += 1;
                stats.draw_calls += 1;
                stats.triangles += 2 * @as(u32, @intCast(ps.active_count));
            }
        }
    }

    /// Pass variant matching the exact target shape (sample count + color
    /// format). The single twin slot serves every non-base shape, keyed by
    /// both; the base shape stays on the base pass.
    fn passFor(self: *ParticleLayer, samples: i32, color_format: sg.PixelFormat) *passes.ParticlePass {
        if (samples == self.pass.sample_count and color_format == self.pass.color_format) return &self.pass;
        if (self.pass_msaa == null or self.pass_msaa.?.sample_count != samples or self.pass_msaa.?.color_format != color_format) {
            if (self.pass_msaa) |*p| p.deinit();
            self.pass_msaa = passes.ParticlePass.init(samples, color_format);
        }
        return &self.pass_msaa.?;
    }
};
