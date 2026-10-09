//! Engine-owned frame lifecycle facade for threaded apps.
//!
//! Game side: `claim -> build -> stageUi -> publish` per tick (see
//! `produceBuild`), one producer only. Context side: `beginStagedPrepare`
//! (fresh FULL build only, null when none), `finishStagedPrepare` +
//! `render`, or `renderReuse` when no fresh build is ready. Worker
//! start/stop ordering lives here so apps cannot drift it.
//!
//! `producer_exclusion` (default false) is a diagnostic switch only: it
//! bounds the SAME frozen begin data with the phase mutex. It changes no
//! sources — the begin never reads live producer state either way.
//! `tryRunLocked` stays for diagnostic frames. Host-owned: window
//! setup/shutdown, the `simulate` body, UI policy, render hooks, phase
//! metrics, save/quit policy.

const std = @import("std");
const builtin = @import("builtin");
const jobs = @import("jobs.zig");
const Scene = @import("scene.zig").Scene;

/// Result of a `beginPrepare*` call: the staged claim (null when no fresh
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
/// they mirror. `begin_busy` exists because acquisition failure and "no
/// fresh build" need different contention accounting (see `BeginResult.busy`).
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
};

pub const Runtime = struct {
    /// Phase mutex: update-vs-begin exclusion. Game holds it across the
    /// tick; context holds it across the begin ONLY when the diagnostic
    /// exclusion is enabled. Never held across finish/render.
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
    /// Producer phase exclusion around the staged begin. DEFAULT FALSE:
    /// the begin consumes only frozen slot payloads + context-owned state,
    /// so no mutex is needed. `setProducerExclusion(true)` is a diagnostic
    /// switch: it bounds the SAME frozen begin with the mutex. It changes
    /// no data sources — only acquisition timing (`busy` + `wait_ns`).
    /// The game side keeps `gameLock` semantics for its own producers
    /// either way (simulate + produce still serialize; a single producer
    /// remains mandatory).
    producer_exclusion: bool = false,

    pub fn init() Runtime {
        return .{
            .mutex = .{},
            .running = std.atomic.Value(bool).init(false),
            .thread = null,
            .metrics = .{},
            .lock_wait_ns = 0,
        };
    }

    /// Switch the staged begin between unlocked (false, default) and the
    /// diagnostic exclusion window (true). Affects `beginPrepare*` /
    /// `renderFrame` on the next call. Context thread (or pre-spawn init)
    /// only.
    pub fn setProducerExclusion(self: *Runtime, excluded: bool) void {
        self.producer_exclusion = excluded;
    }

    /// Publish the context-side acquisition budget (e.g. from
    /// `--lock-wait-us`). Only meaningful with `setProducerExclusion(true)`;
    /// takes effect on the next `beginPrepare*` / `renderFrame`. Context
    /// thread (or pre-spawn init) only.
    pub fn setLockWaitNs(self: *Runtime, ns: u64) void {
        self.lock_wait_ns = ns;
    }

    // -- worker lifecycle (engine owns start/stop ordering) --

    /// Spawn the game worker running `entry` (a `fn () void`, usually the
    /// app's game loop, which must hold `gameLock` across its tick and exit
    /// when `shouldRun` goes false). Returns false on spawn failure with
    /// `running` left clear so the host degrades to single-threaded.
    pub fn spawnWorker(self: *Runtime, comptime entry: fn () void) bool {
        if (comptime builtin.single_threaded or builtin.cpu.arch.isWasm()) {
            return false;
        }
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
    /// Pairs with the context's `beginPrepare*` when the diagnostic
    /// exclusion is enabled; `finish`/`render` overlap freely. Advanced
    /// hosts only — the simple path uses `update`.
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
    /// the last consumable front).
    pub fn produceBuild(self: *Runtime, scene: *Scene) bool {
        return self.produceBuildWithHostBytes(scene, null);
    }

    /// Same as `produceBuild`, plus freezing `host_bytes` (when non-null)
    /// into the claimed slot between `stageUi` and `publish` — the
    /// producer-side half of the lock-free host pipe (`PrepareClaim.
    /// host_bytes`). Must run before `publish` (which releases the slot);
    /// staging after publish would race the context latch.
    pub fn produceBuildWithHostBytes(self: *Runtime, scene: *Scene, host_bytes: ?[]const u8) bool {
        const slot_claim = scene.tryClaimBuildSlot() orelse {
            self.metrics.producer_skips += 1;
            return false;
        };
        var claim = slot_claim;
        claim.build();
        claim.stageUi();
        if (host_bytes) |bytes| claim.stageHostBytes(bytes);
        claim.publish();
        self.metrics.producer_builds += 1;
        return true;
    }

    // -- simple path (normal apps: the agate demo runs on these two) --

    /// Simplest game-side tick: run `tick(ctx)` (the simulate body WITHOUT
    /// the build), then the producer one-liner. Holds the phase mutex only
    /// when the diagnostic exclusion is enabled. Returns whether a build
    /// published. Single-threaded hosts call this inline.
    pub fn update(self: *Runtime, scene: *Scene, ctx: anytype, comptime tick: fn (@TypeOf(ctx)) void) bool {
        if (self.producer_exclusion) {
            self.gameLock();
            defer self.gameUnlock();
            tick(ctx);
            return self.produceBuild(scene);
        }
        tick(ctx);
        return self.produceBuild(scene);
    }

    /// Simplest context-side frame: bounded begin, then finish + render, or
    /// reuse, or skip. The first frames skip until the producer's first
    /// build is ready. Sets `scene.stats.prepare_ms` to the begin + finish
    /// cost MINUS the acquisition wait, so contention tallies never
    /// double-count the wait inside prepare.
    pub fn renderFrame(self: *Runtime, scene: *Scene) FrameResult {
        const t0 = jobs.monoNs();
        const begun = self.beginPrepare(scene);
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

    /// Staged begin. DEFAULT (unlocked): no mutex is taken; the fresh FULL
    /// build already froze every payload into the slot. With
    /// `setProducerExclusion(true)`: bounded acquire, same begin, unlock.
    /// `finishPrepare` + `render` stay unlocked by the caller.
    /// Null + `busy == false` means no fresh producer frame; the caller
    /// reuses the last front or skips the present. Null + `busy == true`
    /// means the mutex stayed held past the budget: same reuse/skip,
    /// counted as contention (exclusion mode only).
    pub fn beginPrepare(self: *Runtime, scene: *Scene) BeginResult {
        if (!self.producer_exclusion) {
            const t1 = jobs.monoNs();
            const claim = scene.beginStagedPrepare();
            const t2 = jobs.monoNs();
            if (claim != null) {
                self.metrics.begins += 1;
            } else {
                self.metrics.begin_empty += 1;
            }
            return .{ .claim = claim, .busy = false, .wait_ns = 0, .held_ns = t2 -% t1 };
        }
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

    /// Same as `beginPrepare`, but runs the host's pre-begin work (`work`)
    /// in the same window. Unlocked by default: `work` must only read
    /// frozen bytes / context-owned state / atomics. With
    /// `setProducerExclusion(true)`: `work` runs inside the SAME exclusion
    /// window as the begin.
    pub fn beginPrepareWith(
        self: *Runtime,
        scene: *Scene,
        comptime work: fn () void,
    ) BeginResult {
        if (!self.producer_exclusion) {
            const t1 = jobs.monoNs();
            work();
            const claim = scene.beginStagedPrepare();
            const t2 = jobs.monoNs();
            if (claim != null) {
                self.metrics.begins += 1;
            } else {
                self.metrics.begin_empty += 1;
            }
            return .{ .claim = claim, .busy = false, .wait_ns = 0, .held_ns = t2 -% t1 };
        }
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

    /// Complete a staged claim. Runs with the producer UNLOCKED: the claim
    /// is a fresh FULL build, so only slot-owned + context-owned state is
    /// consumed.
    pub fn finishPrepare(self: *Runtime, scene: *Scene, claim: Scene.PrepareClaim) void {
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
};

// -- focused unit tests live in `runtime_tests.zig` (same directory,
// imported below so the test registry picks them up exactly once). --
