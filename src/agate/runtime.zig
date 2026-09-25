//! Thin engine-owned frame lifecycle facade for threaded apps.
//!
//! Problem: every threaded app must reproduce the same choreography —
//! producer `claim -> build -> stageUi -> publish` on the game side,
//! `beginStagedPrepare` under producer exclusion on the context side,
//! `finishStagedPrepare` + `render` unlocked (they consume only slot-owned
//! and context-owned state), the `renderReuse` fallback when no fresh build
//! is ready, and worker start/stop ordering. Hand-rolling it per app drifts
//! (the Sandbox shipped a default path with ZERO exclusion around `begin`,
//! whose live reads — `flushPendingGpuUploads` over the mesh/particle CPU
//! arrays, the inline particle/physics captures, the snapshot pack, the UI
//! canvas read — race the game thread's `simulate` mutations; snapshots
//! freeze descriptors, never the upload bytes).
//!
//! This module owns that ordering. Two levels:
//! - Simple (`update` + `renderFrame`): the whole frame for normal apps —
//!   the agate demo runs on these two calls. Acquisition is bounded;
//!   live-state reads remain inside producer exclusion.
//! - Advanced (`gameLock`/`gameUnlock`, `produceBuild`, `beginExcluded*`,
//!   `finishPrepare`/`cancelPrepare`, `reuseIfConsumable`, `prepareSerial`,
//!   direct `mutex` access): the same primitives for instrumented hosts
//!   (Sandbox interleaves phase metrics, test hooks, and UI snapshot
//!   transfer). It is NOT a generic backend, ECS, event bus, or plugin
//!   system.
//!
//! Ownership contract (static, same as the engine handoff):
//! - The game side holds `mutex` across its whole tick (`simulate` +
//!   `produceBuild`). It may overlap `finishStagedPrepare` and `render`.
//! - The context side holds `mutex` across `beginExcluded*` ONLY (plus any
//!   host live-state reads the caller runs inside the `beginExcludedWith`
//!   callback: picked-name copies, memory-snapshot serve, profiler toggles,
//!   deferred report saves). `finish`, `cancel`, and `render` run unlocked.
//! - Context-side acquisition is ALWAYS bounded (`lock_wait_ns`, default 0
//!   = pure non-blocking try): a busy mutex yields `busy` (or a
//!   `tryRunLocked` skip). Scheduling and GPU work can still delay a present.
//!   `prepareSerial` is the deliberate
//!   exception — the legacy/single path blocks by contract (uncontended
//!   single-threaded, or the opt-out `--no-concurrent-build` diagnostic).
//! - `prepareSerial` (the `--no-concurrent-build` / `--no-threads` path)
//!   holds `mutex` across the FULL `prepareFrame`: its live-read fallback
//!   must stay inside the exclusion window, and only render overlaps.
//! - Only the context thread calls `begin`/`finish`/`cancel`/`prepareSerial`
//!   (the engine asserts this); only one game producer calls `produceBuild`.
//!
//! Host-owned responsibilities (NOT here): sokol window setup/shutdown, the
//! `simulate` body itself, the UI snapshot transfer policy, render calls and
//! their instrumentation hooks (advanced hosts), phase metrics, save/quit/
//! quiesce policy. The facade sequences the shared engine calls so the
//! order cannot drift; hosts keep their policy.

const std = @import("std");
const jobs = @import("jobs.zig");
const Scene = @import("scene.zig").Scene;

/// Result of a `beginExcluded*` call: the staged claim (null when no fresh
/// producer frame is ready — never a live-read fallback), whether the
/// phase mutex could not be acquired within the budget (`busy`, distinct
/// from "no fresh build": no begin was attempted, reuse/skip exactly like
/// the empty case but count it as contention, never as idle), plus honest
/// exclusion timings. `wait_ns` covers acquisition only; `held_ns` covers
/// the locked window (host locked work + `beginStagedPrepare`).
pub const BeginResult = struct {
    claim: ?Scene.PrepareClaim,
    busy: bool,
    wait_ns: u64,
    held_ns: u64,
};

/// Outcome of the simple `renderFrame`: a fresh frame was prepared and
/// presented (`prepared`), the last front was re-presented (`reused`), or
/// nothing was presented — no consumable frame yet (`skipped`), or the
/// phase mutex stayed busy past the budget (`busy`). `busy` still presents
/// when a front is consumable (non-blocking goal); it only reports WHY no
/// fresh prepare ran.
pub const FrameResult = enum {
    prepared,
    reused,
    skipped,
    busy,
};

/// Always-on choreography counters. Plain integers, no atomics: every
/// facade method runs on a single owner thread at a time (game side for
/// `producer_*`, context side for the rest), exactly like the engine stats
/// they mirror. Hosts ALSO bump these directly for the events the facade
/// does not mediate (legacy `--no-concurrent-build` reuse/skip branches
/// and serial prepares outside `prepareSerial`, which keep bespoke lock
/// handling with interleaved harness work): the counters stay truthful
/// because every site that performs the event names it here. Nothing is
/// hardcoded — every counter is bumped at the site it names. No vanity
/// counters: `begin_busy` exists because acquisition failure and "no fresh
/// build" need different contention accounting (see `BeginResult.busy`).
pub const Metrics = struct {
    producer_builds: u64 = 0,
    producer_skips: u64 = 0,
    begins: u64 = 0,
    begin_empty: u64 = 0,
    begin_busy: u64 = 0,
    finishes: u64 = 0,
    cancels: u64 = 0,
    reuses: u64 = 0,
    skipped_presents: u64 = 0,
    serial_prepares: u64 = 0,
};

pub const Runtime = struct {
    /// Phase mutex: update-vs-begin exclusion. Game holds it across the
    /// tick; context holds it across begin (staged) or the full prepare
    /// (serial). Never held across finish/render. Advanced hosts may hold
    /// it directly for bespoke windows (the legacy diagnostic path); the
    /// simple path never exposes it.
    mutex: jobs.Mutex = .{},
    running: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
    thread: ?std.Thread = null,
    metrics: Metrics = .{},
    /// Bounded context-side acquisition budget in nanoseconds (0 = pure
    /// non-blocking try). The game side always blocks: a tick must not be
    /// dropped. Hosts with a wait budget flag (Sandbox `--lock-wait-us`)
    /// publish it here; the staged path then honors it instead of
    /// stalling the present.
    lock_wait_ns: u64 = 0,

    pub fn init() Runtime {
        return .{
            .mutex = .{},
            .running = std.atomic.Value(bool).init(false),
            .thread = null,
            .metrics = .{},
            .lock_wait_ns = 0,
        };
    }

    /// Publish the context-side acquisition budget (e.g. from
    /// `--lock-wait-us`). Takes effect on the next `beginExcluded*` /
    /// `renderFrame`. Context thread (or pre-spawn init) only.
    pub fn setLockWaitNs(self: *Runtime, ns: u64) void {
        self.lock_wait_ns = ns;
    }

    // -- worker lifecycle (engine owns start/stop ordering) --

    /// Spawn the game worker running `entry` (a `fn () void`, usually the
    /// app's game loop, which must hold `gameLock` across its tick and exit
    /// when `shouldRun` goes false). Returns false on spawn failure with
    /// `running` left clear so the host degrades to single-threaded.
    pub fn spawnWorker(self: *Runtime, comptime entry: fn () void) bool {
        std.debug.assert(self.thread == null);
        self.running.store(true, .release);
        self.thread = std.Thread.spawn(.{}, entry, .{}) catch |err| {
            self.running.store(false, .release);
            std.debug.print("game thread spawn failed ({s}); running single-threaded\n", .{@errorName(err)});
            return false;
        };
        return true;
    }

    /// Idempotent quiesce: signal stop, join, clear the handle. After return
    /// no game-thread writer exists, so game-owned plain fields are safe to
    /// read on this thread. A lock read would NOT suffice: game-side writes
    /// happen under the phase mutex by design only for shared live state —
    /// joining is what removes the writer. Call with the phase mutex
    /// UNHELD (the worker may be parked acquiring it).
    pub fn quiesce(self: *Runtime) void {
        self.running.store(false, .release);
        if (self.thread) |t| {
            t.join();
            self.thread = null;
        }
    }

    pub fn shouldRun(self: *const Runtime) bool {
        return self.running.load(.acquire);
    }

    /// Join the worker before releasing the OS mutex. Call before Scene.deinit.
    pub fn deinit(self: *Runtime) void {
        self.quiesce();
        self.mutex.deinit();
    }

    // -- game-side exclusion (game thread) --

    /// Hold across the whole producer tick (`simulate` + `produceBuild`).
    /// Pairs with the context's `beginExcluded*`: while held, no begin runs;
    /// `finish`/`render` overlap freely. Advanced hosts only — the simple
    /// path uses `update`.
    pub fn gameLock(self: *Runtime) void {
        self.mutex.lock();
    }

    pub fn gameUnlock(self: *Runtime) void {
        self.mutex.unlock();
    }

    /// Producer build one-liner (game side, under `gameLock`, AFTER the
    /// tick's sim mutations and UI build): `tryClaimBuildSlot -> build ->
    /// stageUi -> publish`. True when a build published; false when every
    /// non-front slot was pinned/claimed (counted skip — the context reuses
    /// the last consumable front). `stageUi` runs AFTER `build` (the build
    /// resets the slot); UI-only ticks without `build` are still legal but
    /// staged-only handoffs never satisfy a staged begin (the serialized
    /// fallback consumes them instead).
    pub fn produceBuild(self: *Runtime, scene: *Scene) bool {
        const slot_claim = scene.tryClaimBuildSlot() orelse {
            self.metrics.producer_skips += 1;
            return false;
        };
        var claim = slot_claim;
        claim.build();
        claim.stageUi();
        claim.publish();
        self.metrics.producer_builds += 1;
        return true;
    }

    // -- simple path (normal apps: the agate demo runs on these two) --

    /// Simplest game-side tick: hold the phase mutex, run `tick(ctx)` (the
    /// simulate body WITHOUT the build), then the producer one-liner.
    /// Returns whether a build published. Single-threaded hosts call this
    /// inline (the mutex is uncontended; the lock still documents producer
    /// ownership for the same code path).
    pub fn update(self: *Runtime, scene: *Scene, ctx: anytype, comptime tick: fn (@TypeOf(ctx)) void) bool {
        self.gameLock();
        defer self.gameUnlock();
        tick(ctx);
        return self.produceBuild(scene);
    }

    /// Simplest context-side frame: bounded begin, then finish + render, or
    /// reuse, or skip. Acquisition uses `lock_wait_ns` (plus scheduling jitter);
    /// live reads are excluded from the producer, not eliminated.
    /// the first frames skip until the producer's first build is ready.
    /// Sets `scene.stats.prepare_ms` to the begin + finish cost MINUS the
    /// acquisition wait, so contention tallies never double-count the wait
    /// inside prepare.
    pub fn renderFrame(self: *Runtime, scene: *Scene) FrameResult {
        const t0 = jobs.monoNs();
        const begun = self.beginExcluded(scene);
        if (begun.claim) |c| {
            self.finishPrepare(scene, c);
            const elapsed_ns = jobs.monoNs() -% t0;
            const net_ns = if (elapsed_ns > begun.wait_ns) elapsed_ns - begun.wait_ns else 0;
            scene.stats.prepare_ms = @as(f32, @floatCast(@as(f64, @floatFromInt(net_ns)) / 1_000_000.0));
            scene.render();
            return .prepared;
        }
        if (self.reuseIfConsumable(scene)) return if (begun.busy) .busy else .reused;
        return if (begun.busy) .busy else .skipped;
    }

    // -- advanced context-side choreography (instrumented hosts) --

    /// Bounded acquisition for every context-side entry below: try the
    /// mutex for `lock_wait_ns`, else report busy. Zero budget is a pure
    /// non-blocking try.
    fn acquireContext(self: *Runtime) struct { acquired: bool, wait_ns: u64 } {
        const t0 = jobs.monoNs();
        if (self.mutex.tryLockWithin(self.lock_wait_ns)) {
            return .{ .acquired = true, .wait_ns = jobs.monoNs() -% t0 };
        }
        return .{ .acquired = false, .wait_ns = jobs.monoNs() -% t0 };
    }

    /// Non-blocking locked section for diagnostic frames (forced reuse):
    /// runs `work` only if the mutex is free RIGHT NOW, else skips it —
    /// the host reuses its previous snapshot/title. Never blocks the
    /// present just for UI snapshot reads.
    pub fn tryRunLocked(self: *Runtime, comptime work: fn () void) bool {
        if (!self.mutex.tryLock()) return false;
        work();
        self.mutex.unlock();
        return true;
    }

    /// Staged begin with the smallest safe exclusion boundary: bounded
    /// acquire, run NOTHING else, begin, unlock. `finishPrepare` + `render`
    /// stay unlocked by the caller. Null + `busy == false` means no fresh
    /// producer frame — never a live read; the caller reuses the last front
    /// or skips the present. Null + `busy == true` means the mutex stayed
    /// held past the budget: same reuse/skip, counted as contention.
    pub fn beginExcluded(self: *Runtime, scene: *Scene) BeginResult {
        const acq = self.acquireContext();
        if (!acq.acquired) {
            self.metrics.begin_busy += 1;
            return .{ .claim = null, .busy = true, .wait_ns = acq.wait_ns, .held_ns = 0 };
        }
        const t1 = jobs.monoNs();
        const claim = scene.beginStagedPrepare();
        const t2 = jobs.monoNs();
        self.mutex.unlock();
        if (claim != null) {
            self.metrics.begins += 1;
        } else {
            self.metrics.begin_empty += 1;
        }
        return .{ .claim = claim, .busy = false, .wait_ns = acq.wait_ns, .held_ns = t2 -% t1 };
    }

    /// Same as `beginExcluded`, but runs the host's live-state reads
    /// (`work`: picked-name copies, memory-snapshot serve, profiler toggle
    /// consume, deferred report save) inside the SAME exclusion window as
    /// the begin, so no torn read can slip between a snapshot copy and the
    /// begin that consumes it.
    pub fn beginExcludedWith(
        self: *Runtime,
        scene: *Scene,
        comptime work: fn () void,
    ) BeginResult {
        const acq = self.acquireContext();
        if (!acq.acquired) {
            self.metrics.begin_busy += 1;
            return .{ .claim = null, .busy = true, .wait_ns = acq.wait_ns, .held_ns = 0 };
        }
        const t1 = jobs.monoNs();
        work();
        const claim = scene.beginStagedPrepare();
        const t2 = jobs.monoNs();
        self.mutex.unlock();
        if (claim != null) {
            self.metrics.begins += 1;
        } else {
            self.metrics.begin_empty += 1;
        }
        return .{ .claim = claim, .busy = false, .wait_ns = acq.wait_ns, .held_ns = t2 -% t1 };
    }

    /// Complete a staged claim. May run with the producer UNLOCKED (only
    /// `have_build` claims are concurrency-safe here — the fallback claim
    /// reads live meshes through `finishPrepare`, so a fallback begin must
    /// never be released early; staged begins never take the fallback).
    pub fn finishPrepare(self: *Runtime, scene: *Scene, claim: Scene.PrepareClaim) void {
        std.debug.assert(claim.have_build);
        scene.finishStagedPrepare(claim);
        self.metrics.finishes += 1;
    }

    /// Drop a staged claim without publishing (error paths only — every
    /// successful begin MUST pair with exactly one finish or cancel).
    pub fn cancelPrepare(self: *Runtime, scene: *Scene, claim: Scene.PrepareClaim) void {
        scene.cancelStagedPrepare(claim);
        self.metrics.cancels += 1;
    }

    /// Non-blocking consumer fallback for frames with no fresh build:
    /// re-present the last front when consumable, else skip the present
    /// (first-frame race before any prepare published). True when a present
    /// happened.
    pub fn reuseIfConsumable(self: *Runtime, scene: *Scene) bool {
        if (!scene.hasConsumableFrame()) {
            self.metrics.skipped_presents += 1;
            return false;
        }
        scene.renderReuse();
        self.metrics.reuses += 1;
        return true;
    }

    /// Serial legacy prepare (`--no-concurrent-build` / `--no-threads`):
    /// the FULL `prepareFrame` under the mutex — its live-read fallback
    /// must stay inside the exclusion window. Released before render.
    /// In single-threaded mode the mutex is uncontended; the lock still
    /// documents "producer quiesced" (trivially true, no worker exists).
    pub fn prepareSerial(self: *Runtime, scene: *Scene) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        scene.prepareFrame();
        self.metrics.serial_prepares += 1;
    }
};

// -- focused unit tests (run via the generated tests.zig registry) --

const testing = std.testing;
const gpu_thread = @import("gpu_thread.zig");

fn testSceneOwned(alloc: std.mem.Allocator) Scene {
    return @import("testing.zig").testScene(alloc);
}

fn deinitTestScene(scene: *Scene) void {
    const alloc = scene.allocator;
    scene.lights.deinit(alloc);
    scene.cameras.deinit(alloc);
    scene.draws.deinit(alloc);
    scene.gpu_retire.deinit(alloc);
    scene.profiler.deinit();
}

/// Nullable-thread join guard: join exactly once on every path. An
/// `errdefer` join alone double-joins when an expect AFTER a successful
/// join fails; clearing the slot after the join keeps all exits single.
/// Worker loops below use bounded spins (never bare flag waits) so a join
/// on an unwinding path always terminates.
fn joinSlot(slot: *?std.Thread) void {
    if (slot.*) |t| {
        t.join();
        slot.* = null;
    }
}

test "runtime: produceBuild publishes one build; begin/finish consume it" {
    const alloc = testing.allocator;
    gpu_thread.markContextThread();
    var scene = testSceneOwned(alloc);
    defer deinitTestScene(&scene);
    var rt = Runtime.init();
    defer rt.deinit();

    // No fresh build: staged begin stays null (not busy — uncontended),
    // never a live fallback.
    const empty = rt.beginExcluded(&scene);
    try testing.expect(empty.claim == null);
    try testing.expect(!empty.busy);
    try testing.expectEqual(@as(u64, 1), rt.metrics.begin_empty);
    try testing.expect(!scene.hasConsumableFrame());

    // One producer tick, then the split consume.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try testing.expect(rt.produceBuild(&scene));
    try testing.expectEqual(@as(u64, 1), rt.metrics.producer_builds);

    const begun = rt.beginExcluded(&scene);
    try testing.expect(!begun.busy);
    const claim = begun.claim orelse return error.TestUnexpectedResult;
    try testing.expect(claim.have_build);
    try testing.expectEqual(@as(u64, 1), rt.metrics.begins);
    rt.finishPrepare(&scene, claim);
    try testing.expectEqual(@as(u64, 1), rt.metrics.finishes);
    try testing.expect(scene.hasConsumableFrame());
    try testing.expect(scene.frame_prepared);
}

var begin_window_probe: struct {
    ran: bool = false,
} = .{};

fn beginWindowProbeWork() void {
    // No-arg locked callback: records that it ran inside the window.
    begin_window_probe.ran = true;
}

test "runtime: beginExcludedWith runs locked work in the begin window" {
    const alloc = testing.allocator;
    gpu_thread.markContextThread();
    var scene = testSceneOwned(alloc);
    defer deinitTestScene(&scene);
    var rt = Runtime.init();
    defer rt.deinit();

    begin_window_probe.ran = false;
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try testing.expect(rt.produceBuild(&scene));
    const begun = rt.beginExcludedWith(&scene, beginWindowProbeWork);
    try testing.expect(begin_window_probe.ran);
    try testing.expect(!begun.busy);
    const claim = begun.claim orelse return error.TestUnexpectedResult;
    rt.finishPrepare(&scene, claim);
    try testing.expect(scene.hasConsumableFrame());
}

test "runtime: bounded acquisition reports busy instead of stalling" {
    const alloc = testing.allocator;
    gpu_thread.markContextThread();
    var scene = testSceneOwned(alloc);
    defer deinitTestScene(&scene);
    var rt = Runtime.init();
    defer rt.deinit();

    // Self-held mutex: even the owner cannot re-acquire (non-recursive),
    // so the bounded begin reports busy without blocking.
    rt.mutex.lock();
    const self_busy = rt.beginExcluded(&scene);
    rt.mutex.unlock();
    try testing.expect(self_busy.claim == null);
    try testing.expect(self_busy.busy);
    try testing.expectEqual(@as(u64, 1), rt.metrics.begin_busy);

    // Cross-thread: the worker holds the game lock past the budget while
    // the context (this thread) attempts the begin.
    const Holder = struct {
        rt: *Runtime,
        holding: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        release: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        fn run(self_: *@This()) void {
            self_.rt.gameLock();
            self_.holding.store(true, .release);
            var spins: u32 = 0;
            while (!self_.release.load(.acquire) and spins < 200_000) : (spins += 1) {
                jobs.sleepNs(10_000);
            }
            self_.rt.gameUnlock();
        }
    };
    var holder = Holder{ .rt = &rt };
    var thread: ?std.Thread = try std.Thread.spawn(.{}, Holder.run, .{&holder});
    errdefer joinSlot(&thread);
    var spins: u32 = 0;
    while (!holder.holding.load(.acquire) and spins < 100_000) : (spins += 1) {
        jobs.sleepNs(10_000);
    }
    try testing.expect(holder.holding.load(.acquire));

    rt.setLockWaitNs(2_000_000);
    const contended = rt.beginExcluded(&scene);
    holder.release.store(true, .release);
    try testing.expect(contended.claim == null);
    try testing.expect(contended.busy);
    try testing.expect(contended.wait_ns >= 2_000_000);
    try testing.expectEqual(@as(u64, 2), rt.metrics.begin_busy);
    joinSlot(&thread);

    // Budget restored to instant: an uncontended begin is not busy.
    rt.setLockWaitNs(0);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try testing.expect(rt.produceBuild(&scene));
    const free = rt.beginExcluded(&scene);
    try testing.expect(!free.busy);
    rt.finishPrepare(&scene, free.claim orelse return error.TestUnexpectedResult);
}

var try_run_probe_count: u32 = 0;

fn tryRunProbeWork() void {
    try_run_probe_count += 1;
}

test "runtime: tryRunLocked never blocks; skips when busy" {
    const alloc = testing.allocator;
    gpu_thread.markContextThread();
    var scene = testSceneOwned(alloc);
    defer deinitTestScene(&scene);
    var rt = Runtime.init();
    defer rt.deinit();

    try_run_probe_count = 0;
    try testing.expect(rt.tryRunLocked(tryRunProbeWork));
    try testing.expectEqual(@as(u32, 1), try_run_probe_count);

    rt.mutex.lock();
    try testing.expect(!rt.tryRunLocked(tryRunProbeWork));
    rt.mutex.unlock();
    // Skipped work leaves prior state untouched (hosts reuse it).
    try testing.expectEqual(@as(u32, 1), try_run_probe_count);
}

test "runtime: cancel releases the claim; begin works again" {
    const alloc = testing.allocator;
    gpu_thread.markContextThread();
    var scene = testSceneOwned(alloc);
    defer deinitTestScene(&scene);
    var rt = Runtime.init();
    defer rt.deinit();

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try testing.expect(rt.produceBuild(&scene));
    const seq = scene.build_seq.load(.monotonic);

    const begun = rt.beginExcluded(&scene);
    const claim = begun.claim orelse return error.TestUnexpectedResult;
    rt.cancelPrepare(&scene, claim);
    try testing.expectEqual(@as(u64, 1), rt.metrics.cancels);
    try testing.expect(!scene.prepare_claim_active);

    // The handoff was restored: re-begin resolves the same generation.
    const retry = rt.beginExcluded(&scene);
    const claim2 = retry.claim orelse return error.TestUnexpectedResult;
    try testing.expectEqual(seq, claim2.build_seq);
    rt.finishPrepare(&scene, claim2);
    try testing.expectEqual(seq, scene.last_latched_seq.load(.monotonic));
}

test "runtime: reuseIfConsumable skips before the first prepare, reuses after" {
    const alloc = testing.allocator;
    gpu_thread.markContextThread();
    var scene = testSceneOwned(alloc);
    defer deinitTestScene(&scene);
    var rt = Runtime.init();
    defer rt.deinit();

    try testing.expect(!rt.reuseIfConsumable(&scene));
    try testing.expectEqual(@as(u64, 1), rt.metrics.skipped_presents);

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try testing.expect(rt.produceBuild(&scene));
    const begun = rt.beginExcluded(&scene);
    rt.finishPrepare(&scene, begun.claim orelse return error.TestUnexpectedResult);
    // Consume the prepared frame (headless: no camera, so the inner render
    // early-outs after epoch completion — sg never touched), then the
    // non-blocking consumer re-presents the same front without a prepare.
    scene.render();
    try testing.expect(!scene.frame_prepared);
    try testing.expect(rt.reuseIfConsumable(&scene));
    try testing.expectEqual(@as(u64, 1), rt.metrics.reuses);
    try testing.expectEqual(@as(u64, 1), scene.reuseStreak());
}

test "runtime: simple update/renderFrame presents without manual locking" {
    const alloc = testing.allocator;
    gpu_thread.markContextThread();
    var scene = testSceneOwned(alloc);
    defer deinitTestScene(&scene);
    var rt = Runtime.init();
    defer rt.deinit();

    // Nothing published yet: skip until ready (first-frame behavior).
    try testing.expectEqual(FrameResult.skipped, rt.renderFrame(&scene));

    const Tick = struct {
        scene: *Scene,
        fn run(ctx: *@This()) void {
            ctx.scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
        }
    };
    var tick = Tick{ .scene = &scene };
    try testing.expect(rt.update(&scene, &tick, Tick.run));
    try testing.expectEqual(FrameResult.prepared, rt.renderFrame(&scene));
    try testing.expect(scene.hasConsumableFrame());
    // prepare_ms excludes acquisition wait (uncontended here: ~full cost).
    try testing.expect(scene.stats.prepare_ms >= 0);
}

test "runtime: begin window excludes the producer; finish does not" {
    // Deterministic exclusion proof by handshake (no timing assumptions):
    // while the begin callback holds the phase mutex, a controlled worker
    // attempt MUST fail; after the callback the worker is admitted; and
    // `finishPrepare` returns while the worker holds the mutex (it never
    // blocks on producer exclusion). Removing the exclusion (unlocked
    // callback) flips the worker's attempt to success and fails this test.
    const alloc = testing.allocator;
    gpu_thread.markContextThread();
    var scene = testSceneOwned(alloc);
    defer deinitTestScene(&scene);
    var rt = Runtime.init();
    defer rt.deinit();

    const Harness = struct {
        rt: *Runtime,
        in_callback: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        attempt_done: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        callback_done: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        holding: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        ready: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        // Worker-written, main-read after join (join is the edge):
        denied_during_callback: bool = false,
        admitted_after_callback: bool = false,
        finish_done: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        finished_while_held: bool = false,
        fn run(self_: *@This()) void {
            self_.ready.store(true, .release);
            var spins: u32 = 0;
            while (!self_.in_callback.load(.acquire) and spins < 200_000) : (spins += 1) {
                jobs.sleepNs(10_000);
            }
            if (!self_.in_callback.load(.acquire)) return; // setup failed; main unwinds
            // Controlled attempt DURING the callback's hold: must fail.
            if (self_.rt.mutex.tryLock()) {
                self_.rt.mutex.unlock();
                self_.denied_during_callback = false;
            } else {
                self_.denied_during_callback = true;
            }
            self_.attempt_done.store(true, .release);
            spins = 0;
            while (!self_.callback_done.load(.acquire) and spins < 200_000) : (spins += 1) {
                jobs.sleepNs(10_000);
            }
            if (!self_.callback_done.load(.acquire)) return;
            // After the callback: admitted, then hold across main's finish.
            self_.rt.mutex.lock();
            self_.admitted_after_callback = true;
            self_.holding.store(true, .release);
            const deadline = jobs.monoNs() + 10_000_000_000;
            while (!self_.finish_done.load(.acquire) and jobs.monoNs() < deadline) {
                jobs.sleepNs(50_000);
            }
            self_.finished_while_held = self_.finish_done.load(.acquire);
            self_.rt.mutex.unlock();
        }
    };

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try testing.expect(rt.produceBuild(&scene));
    const published = scene.build_seq.load(.monotonic);

    var harness = Harness{ .rt = &rt };
    var harness_view = ExclusionHarness{
        .rt = &rt,
        .in_callback = &harness.in_callback,
        .attempt_done = &harness.attempt_done,
        .callback_done = &harness.callback_done,
    };
    exclusion_harness_slot = &harness_view;
    defer exclusion_harness_slot = null;
    var thread: ?std.Thread = try std.Thread.spawn(.{}, Harness.run, .{&harness});
    errdefer joinSlot(&thread);
    var spins: u32 = 0;
    while (!harness.ready.load(.acquire) and spins < 100_000) : (spins += 1) {
        jobs.sleepNs(10_000);
    }
    try testing.expect(harness.ready.load(.acquire));

    const begun = rt.beginExcludedWith(&scene, exclusionWindowWork);
    exclusion_harness_slot = null;
    const claim = begun.claim orelse return error.TestUnexpectedResult;
    try testing.expect(!begun.busy);
    try testing.expect(exclusion_saw_locked);

    // The worker now holds the phase mutex; finish must return WITHOUT
    // waiting for its release (finish takes no producer exclusion).
    spins = 0;
    while (!harness.holding.load(.acquire) and spins < 200_000) : (spins += 1) {
        jobs.sleepNs(10_000);
    }
    try testing.expect(harness.holding.load(.acquire));
    rt.finishPrepare(&scene, claim);
    harness.finish_done.store(true, .release);
    joinSlot(&thread);

    try testing.expect(harness.denied_during_callback);
    try testing.expect(harness.admitted_after_callback);
    try testing.expect(harness.finished_while_held);
    try testing.expectEqual(published, scene.last_latched_seq.load(.monotonic));
}

/// File-scope slot for the exclusion test's no-arg window callback (tests
/// run sequentially; the slot is set only around its begin call). The
/// concrete harness type is test-local, so the slot erases to the four
/// fields the callback touches.
const ExclusionHarness = struct {
    rt: *Runtime,
    in_callback: *std.atomic.Value(bool),
    attempt_done: *std.atomic.Value(bool),
    callback_done: *std.atomic.Value(bool),
};
var exclusion_harness_slot: ?*const ExclusionHarness = null;
var exclusion_saw_locked: bool = false;

fn exclusionWindowWork() void {
    const h = exclusion_harness_slot orelse return;
    // The callback must OBSERVE the mutex held: even we (the holder) cannot
    // re-acquire a non-recursive mutex. A success here would prove the
    // window runs unlocked — record and release immediately either way.
    if (h.rt.mutex.tryLock()) {
        h.rt.mutex.unlock();
        exclusion_saw_locked = false;
    } else {
        exclusion_saw_locked = true;
    }
    h.in_callback.store(true, .release);
    var spins: u32 = 0;
    while (!h.attempt_done.load(.acquire) and spins < 200_000) : (spins += 1) {
        jobs.sleepNs(10_000);
    }
    h.callback_done.store(true, .release);
}

test "runtime: producer ticks under gameLock overlap finish safely" {
    // The fix's mechanism as a smoke test: a worker thread runs producer
    // ticks (snapshot publish + build, under gameLock) while the context
    // thread begins/finishes. Exclusion is by the phase mutex; the finish
    // stays unlocked so overlap is real. Latest-wins postconditions (no
    // per-generation accounting: a fast producer legitimately supersedes
    // pending builds before the consumer latches them):
    // - every latched generation is strictly newer than the last (no
    //   overwrite, no reorder, no re-latch);
    // - eventually the latest published generation is consumed;
    // - cross-thread assertions use the atomic handoff words only
    //   (`build_seq`/`last_latched_seq`); plain counters are main-side.
    const alloc = testing.allocator;
    gpu_thread.markContextThread();
    var scene = testSceneOwned(alloc);
    defer deinitTestScene(&scene);
    var rt = Runtime.init();
    defer rt.deinit();

    const Worker = struct {
        scene: *Scene,
        rt: *Runtime,
        ticks: u32,
        done: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        fn run(self_: *@This()) void {
            defer self_.done.store(true, .release);
            var i: u32 = 0;
            while (i < self_.ticks) : (i += 1) {
                self_.rt.gameLock();
                self_.scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
                _ = self_.rt.produceBuild(self_.scene);
                self_.scene.recordUpdateTime(0.5);
                self_.rt.gameUnlock();
                jobs.sleepNs(100_000);
            }
        }
    };
    const total_ticks: u32 = 20;
    var worker = Worker{ .scene = &scene, .rt = &rt, .ticks = total_ticks };
    var thread: ?std.Thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    errdefer joinSlot(&thread);

    // Context: latch whatever is pending; busy/empty begins just spin —
    // both are legal under latest-wins with a zero budget.
    var prev_latched: u64 = 0;
    var finished: u64 = 0;
    var spins: u32 = 0;
    while (spins < 200_000) : (spins += 1) {
        if (worker.done.load(.acquire)) break;
        const begun = rt.beginExcluded(&scene);
        if (begun.claim) |claim| {
            try testing.expect(claim.have_build);
            rt.finishPrepare(&scene, claim);
            finished += 1;
            const latched = scene.last_latched_seq.load(.monotonic);
            try testing.expect(latched > prev_latched);
            prev_latched = latched;
        } else {
            jobs.sleepNs(50_000);
        }
    }
    joinSlot(&thread);
    // Final drain after the join (uncontended: busies here would be a bug).
    var drains: u32 = 0;
    while (drains < 64) : (drains += 1) {
        const begun = rt.beginExcluded(&scene);
        if (begun.claim) |claim| {
            try testing.expect(!begun.busy);
            rt.finishPrepare(&scene, claim);
            finished += 1;
            const latched = scene.last_latched_seq.load(.monotonic);
            try testing.expect(latched > prev_latched);
            prev_latched = latched;
        } else break;
    }
    const published = scene.build_seq.load(.monotonic);
    try testing.expect(published >= 1);
    try testing.expect(published <= total_ticks);
    try testing.expectEqual(published, scene.last_latched_seq.load(.monotonic));
    try testing.expectEqual(finished, rt.metrics.finishes);
}
