//! Tests for `runtime.zig` (moved verbatim from inline blocks; production code unchanged).
const std = @import("std");
const runtime_mod = @import("runtime.zig");
const Runtime = runtime_mod.Runtime;
const FrameResult = runtime_mod.FrameResult;
const Scene = @import("scene.zig").Scene;
const jobs = @import("jobs.zig");
const gpu_thread = @import("gpu_thread.zig");

const testing = std.testing;

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
    const empty = rt.beginPrepare(&scene);
    try testing.expect(empty.claim == null);
    try testing.expect(!empty.busy);
    try testing.expectEqual(@as(u64, 1), rt.metrics.begin_empty);
    try testing.expect(!scene.hasConsumableFrame());

    // One producer tick, then the split consume.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try testing.expect(rt.produceBuild(&scene));
    try testing.expectEqual(@as(u64, 1), rt.metrics.producer_builds);

    const begun = rt.beginPrepare(&scene);
    try testing.expect(!begun.busy);
    const claim = begun.claim orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u64, 1), rt.metrics.begins);
    rt.finishPrepare(&scene, claim);
    try testing.expectEqual(@as(u64, 1), rt.metrics.finishes);
    try testing.expect(scene.hasConsumableFrame());
    try testing.expect(scene.frame_prepared);
}

test "runtime: mismatched finish/cancel auto-recovers instead of wedging begins" {
    // Contract violations (double finish, stale token against an active
    // claim) must log and release the active claim — never wedge the
    // pipeline into reuse-forever like the historical assert path did in
    // ReleaseFast.
    const alloc = testing.allocator;
    gpu_thread.markContextThread();
    var scene = testSceneOwned(alloc);
    defer deinitTestScene(&scene);
    var rt = Runtime.init();
    defer rt.deinit();

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try testing.expect(rt.produceBuild(&scene));
    const begun = rt.beginPrepare(&scene);
    const claim = begun.claim orelse return error.TestUnexpectedResult;
    rt.finishPrepare(&scene, claim);

    // Double finish: stale claim, no active claim — logged no-op, no crash.
    rt.finishPrepare(&scene, claim);

    // Mismatched token against an ACTIVE claim: released defensively; the
    // slot lease is gone and no claim stays active.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try testing.expect(rt.produceBuild(&scene));
    const second = rt.beginPrepare(&scene);
    _ = second.claim orelse return error.TestUnexpectedResult;
    rt.finishPrepare(&scene, claim); // stale token vs active claim2
    try testing.expect(!scene.prepare_claim_active);

    // The pipeline stays live: a fresh build is still consumable.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try testing.expect(rt.produceBuild(&scene));
    const third = rt.beginPrepare(&scene);
    const claim3 = third.claim orelse return error.TestUnexpectedResult;
    rt.finishPrepare(&scene, claim3);
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

test "runtime: beginPrepareWith runs the pre-begin work in the same window" {
    const alloc = testing.allocator;
    gpu_thread.markContextThread();
    var scene = testSceneOwned(alloc);
    defer deinitTestScene(&scene);
    var rt = Runtime.init();
    defer rt.deinit();

    begin_window_probe.ran = false;
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try testing.expect(rt.produceBuild(&scene));
    const begun = rt.beginPrepareWith(&scene, beginWindowProbeWork);
    try testing.expect(begin_window_probe.ran);
    try testing.expect(!begun.busy);
    const claim = begun.claim orelse return error.TestUnexpectedResult;
    rt.finishPrepare(&scene, claim);
    try testing.expect(scene.hasConsumableFrame());
}

test "runtime: unlocked is the default; setProducerExclusion restores the guarded window" {
    // Default: staged begins run WITHOUT taking the phase mutex — even when
    // the game side holds it — and report zero acquisition wait.
    // setProducerExclusion(true) restores the exclusion: a held mutex
    // reports busy instead. Either way the begin consumes the SAME frozen
    // FULL build.
    const alloc = testing.allocator;
    gpu_thread.markContextThread();
    var scene = testSceneOwned(alloc);
    defer deinitTestScene(&scene);
    var rt = Runtime.init();
    defer rt.deinit();

    try testing.expect(!rt.producer_exclusion);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try testing.expect(rt.produceBuild(&scene));

    // Game side holds its lock across the whole tick; the unlocked begin
    // must still succeed (it would report busy under the exclusion).
    rt.gameLock();
    const unlocked = rt.beginPrepare(&scene);
    rt.gameUnlock();
    try testing.expect(!unlocked.busy);
    try testing.expectEqual(@as(u64, 0), unlocked.wait_ns);
    const claim = unlocked.claim orelse return error.TestUnexpectedResult;
    // The frozen host pipe rides the claim (empty: produceBuild stages no
    // host bytes) and finish consumes the staged build unlocked.
    try testing.expectEqual(@as(usize, 0), claim.host_bytes.len);
    rt.finishPrepare(&scene, claim);
    try testing.expect(scene.hasConsumableFrame());

    // Exclusion restored: the same held mutex now reports busy.
    rt.setProducerExclusion(true);
    rt.mutex.lock();
    const guarded = rt.beginPrepare(&scene);
    rt.mutex.unlock();
    try testing.expect(guarded.claim == null);
    try testing.expect(guarded.busy);
    // Back to the unlocked default.
    rt.setProducerExclusion(false);
    try testing.expect(!rt.producer_exclusion);
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try testing.expect(rt.produceBuild(&scene));
    const unlocked_again = rt.beginPrepare(&scene);
    rt.finishPrepare(&scene, unlocked_again.claim orelse return error.TestUnexpectedResult);
}

test "runtime: produceBuildWithHostBytes freezes the host pipe for the claim" {
    const alloc = testing.allocator;
    gpu_thread.markContextThread();
    var scene = testSceneOwned(alloc);
    defer deinitTestScene(&scene);
    var rt = Runtime.init();
    defer rt.deinit();

    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    const payload = [_]u8{ 3, 'f', 'o', 'o' };
    try testing.expect(rt.produceBuildWithHostBytes(&scene, &payload));
    try testing.expectEqual(@as(u64, 1), rt.metrics.producer_builds);

    const begun = rt.beginPrepare(&scene);
    const claim = begun.claim orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 4), claim.host_bytes.len);
    try testing.expectEqualSlices(u8, &payload, claim.host_bytes);
    rt.finishPrepare(&scene, claim);

    // Null host bytes: the claim carries an empty pipe, same as
    // produceBuild.
    scene.publishFrameSnapshot(16.0 / 9.0, 800, 600);
    try testing.expect(rt.produceBuild(&scene));
    const begun2 = rt.beginPrepare(&scene);
    const claim2 = begun2.claim orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(usize, 0), claim2.host_bytes.len);
    rt.finishPrepare(&scene, claim2);
}
test "runtime: bounded acquisition reports busy instead of stalling" {
    const alloc = testing.allocator;
    gpu_thread.markContextThread();
    var scene = testSceneOwned(alloc);
    defer deinitTestScene(&scene);
    var rt = Runtime.init();
    defer rt.deinit();
    rt.setProducerExclusion(true); // this test exercises the exclusion path

    // Self-held mutex: even the owner cannot re-acquire (non-recursive),
    // so the bounded begin reports busy without blocking.
    rt.mutex.lock();
    const self_busy = rt.beginPrepare(&scene);
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
    const contended = rt.beginPrepare(&scene);
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
    const free = rt.beginPrepare(&scene);
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

    const begun = rt.beginPrepare(&scene);
    const claim = begun.claim orelse return error.TestUnexpectedResult;
    rt.cancelPrepare(&scene, claim);
    try testing.expectEqual(@as(u64, 1), rt.metrics.cancels);
    try testing.expect(!scene.prepare_claim_active);

    // The handoff was restored: re-begin resolves the same generation.
    const retry = rt.beginPrepare(&scene);
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
    const begun = rt.beginPrepare(&scene);
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

test "runtime: exclusion-mode begin window excludes the producer; finish does not" {
    // Deterministic exclusion proof by handshake (no timing assumptions):
    // with `setProducerExclusion(true)` and the begin callback holding the
    // phase mutex, a controlled worker attempt MUST fail; after the callback
    // the worker is admitted; and `finishPrepare` returns while the worker
    // holds the mutex (it never blocks on producer exclusion). Removing the
    // exclusion (unlocked callback) flips the worker's attempt to success
    // and fails this test.
    const alloc = testing.allocator;
    gpu_thread.markContextThread();
    var scene = testSceneOwned(alloc);
    defer deinitTestScene(&scene);
    var rt = Runtime.init();
    defer rt.deinit();
    rt.setProducerExclusion(true);

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

    const begun = rt.beginPrepareWith(&scene, exclusionWindowWork);
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
        const begun = rt.beginPrepare(&scene);
        if (begun.claim) |claim| {
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
        const begun = rt.beginPrepare(&scene);
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
