const std = @import("std");
const jobs = @import("jobs.zig");

const Pool = jobs.Pool;
const parallelFor = jobs.parallelFor;
const TaskRunner = jobs.TaskRunner;
const Mutex = jobs.Mutex;
const SpscRing = jobs.SpscRing;
const monoNs = jobs.monoNs;
const sleepNs = jobs.sleepNs;

// --- tests ---

const CountCtx = struct {
    seen: []std.atomic.Value(u32),
    total: std.atomic.Value(usize),
    max_end: std.atomic.Value(usize),
};

fn countRange(ctx: *CountCtx, start: usize, end: usize) void {
    for (start..end) |i| {
        _ = ctx.seen[i].fetchAdd(1, .acq_rel);
    }
    _ = ctx.total.fetchAdd(end - start, .acq_rel);
    _ = ctx.max_end.fetchMax(end, .acq_rel);
}

test "forkJoin covers every index exactly once" {
    const a = std.testing.allocator;
    const pool = try Pool.init(a, 2);
    defer pool.deinit();

    const len = 10_003; // prime: forces a remainder chunk
    const seen = try a.alloc(std.atomic.Value(u32), len);
    defer a.free(seen);
    for (seen) |*s| s.* = std.atomic.Value(u32).init(0);

    var ctx = CountCtx{
        .seen = seen,
        .total = std.atomic.Value(usize).init(0),
        .max_end = std.atomic.Value(usize).init(0),
    };
    pool.forkJoin(CountCtx, &ctx, countRange, len);

    try std.testing.expectEqual(@as(usize, len), ctx.total.load(.acquire));
    try std.testing.expectEqual(len, ctx.max_end.load(.acquire));
    for (seen) |*s| try std.testing.expectEqual(@as(u32, 1), s.load(.acquire));
}

test "forkJoin is reusable across consecutive jobs" {
    const a = std.testing.allocator;
    const pool = try Pool.init(a, 2);
    defer pool.deinit();

    const len = 5_000;
    const seen = try a.alloc(std.atomic.Value(u32), len);
    defer a.free(seen);
    for (seen) |*s| s.* = std.atomic.Value(u32).init(0);
    var ctx = CountCtx{
        .seen = seen,
        .total = std.atomic.Value(usize).init(0),
        .max_end = std.atomic.Value(usize).init(0),
    };
    for (0..8) |_| pool.forkJoin(CountCtx, &ctx, countRange, len);
    try std.testing.expectEqual(@as(usize, 8 * len), ctx.total.load(.acquire));
    for (seen) |*s| try std.testing.expectEqual(@as(u32, 8), s.load(.acquire));
}

test "zero-worker pool and empty ranges stay correct" {
    const a = std.testing.allocator;
    const pool = try Pool.init(a, 0);
    defer pool.deinit();
    try std.testing.expectEqual(@as(usize, 0), pool.workerCount());

    const len = 100;
    const seen = try a.alloc(std.atomic.Value(u32), len);
    defer a.free(seen);
    for (seen) |*s| s.* = std.atomic.Value(u32).init(0);
    var ctx = CountCtx{
        .seen = seen,
        .total = std.atomic.Value(usize).init(0),
        .max_end = std.atomic.Value(usize).init(0),
    };
    pool.forkJoin(CountCtx, &ctx, countRange, len);
    try std.testing.expectEqual(@as(usize, len), ctx.total.load(.acquire));

    // Empty range: no chunks, no trampoline calls, no hangs.
    ctx.total = std.atomic.Value(usize).init(0);
    pool.forkJoin(CountCtx, &ctx, countRange, 0);
    try std.testing.expectEqual(@as(usize, 0), ctx.total.load(.acquire));
}

test "parallelFor falls back to inline below the threshold" {
    const a = std.testing.allocator;
    const pool = try Pool.init(a, 2);
    defer pool.deinit();

    // Small range: runs inline even with a pool.
    const small = try a.alloc(std.atomic.Value(u32), 64);
    defer a.free(small);
    for (small) |*s| s.* = std.atomic.Value(u32).init(0);
    var ctx = CountCtx{ .seen = small, .total = std.atomic.Value(usize).init(0), .max_end = std.atomic.Value(usize).init(0) };
    parallelFor(pool, CountCtx, &ctx, countRange, 64);
    try std.testing.expectEqual(@as(usize, 64), ctx.total.load(.acquire));

    // Null pool: large range still completes inline.
    const big = try a.alloc(std.atomic.Value(u32), Pool.min_len_for_workers * 2);
    defer a.free(big);
    for (big) |*s| s.* = std.atomic.Value(u32).init(0);
    var big_ctx = CountCtx{ .seen = big, .total = std.atomic.Value(usize).init(0), .max_end = std.atomic.Value(usize).init(0) };
    parallelFor(null, CountCtx, &big_ctx, countRange, Pool.min_len_for_workers * 2);
    try std.testing.expectEqual(@as(usize, Pool.min_len_for_workers * 2), big_ctx.total.load(.acquire));
}

test "init/deinit cycles leak nothing and restart cleanly" {
    const a = std.testing.allocator;
    const rounds = 4;
    for (0..rounds) |_| {
        const pool = try Pool.init(a, 2);
        defer pool.deinit();
        const len = Pool.min_len_for_workers;
        const seen = try a.alloc(std.atomic.Value(u32), len);
        defer a.free(seen);
        for (seen) |*s| s.* = std.atomic.Value(u32).init(0);
        var ctx = CountCtx{ .seen = seen, .total = std.atomic.Value(usize).init(0), .max_end = std.atomic.Value(usize).init(0) };
        pool.forkJoin(CountCtx, &ctx, countRange, len);
        try std.testing.expectEqual(@as(usize, len), ctx.total.load(.acquire));
    }
}

const Counters = struct {
    hits: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    freed: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
};

fn bumpTask(ctx: *anyopaque) void {
    const counters: *Counters = @ptrCast(@alignCast(ctx));
    _ = counters.hits.fetchAdd(1, .acq_rel);
}

test "TaskRunner runs every posted task exactly once" {
    const a = std.testing.allocator;
    const runner = try TaskRunner.init(a, 2);
    defer runner.deinit();

    var counters = Counters{};
    for (0..100) |_| runner.post(&counters, bumpTask);
    // Tasks may still be in flight; deinit joins them. Poll cheaply before
    // that to prove progress happens without any further posting.
    var waited: usize = 0;
    while (counters.hits.load(.acquire) < 100 and waited < 10_000_000) : (waited += 1) {
        std.atomic.spinLoopHint();
    }
    try std.testing.expectEqual(@as(u32, 100), counters.hits.load(.acquire));
}

test "TaskRunner deinit drains queued tasks" {
    const a = std.testing.allocator;
    const runner = try TaskRunner.init(a, 2);
    var counters = Counters{};
    // Posted but never waited for: shutdown must run them all before join.
    for (0..50) |_| runner.post(&counters, bumpTask);
    runner.deinit();
    // Join-on-shutdown implies a fully drained queue; hits would be short
    // otherwise.
    try std.testing.expectEqual(@as(u32, 50), counters.hits.load(.acquire));
}

test "zero-thread TaskRunner runs tasks inline on the poster" {
    const a = std.testing.allocator;
    const runner = try TaskRunner.init(a, 0);
    defer runner.deinit();
    var counters = Counters{};
    runner.post(&counters, bumpTask);
    // Inline: complete before post returns.
    try std.testing.expectEqual(@as(u32, 1), counters.hits.load(.acquire));
}

// --- SpscRing tests ---

test "SpscRing preserves order and capacity" {
    var ring = SpscRing(usize, 8){};
    for (0..8) |i| try testing_expect(ring.push(i));
    // Full: push rejected (drop-newest).
    try testing_expect(!ring.push(100));
    try testing_expectEqual(usize, 8, ring.len());
    for (0..8) |i| {
        try testing_expectEqual(usize, i, ring.pop().?);
    }
    try testing_expect(ring.pop() == null);
}

test "SpscRing wraps cleanly across many fills" {
    var ring = SpscRing(u32, 4){};
    var produced: u32 = 0;
    var consumed: u32 = 0;
    // Interleave pushes/pops far beyond one lap of the buffer.
    while (consumed < 1000) {
        var pushes: u32 = 0;
        while (pushes < 3) : (pushes += 1) {
            if (ring.push(produced)) produced += 1;
        }
        while (ring.pop()) |v| {
            try testing_expectEqual(u32, consumed, v);
            consumed += 1;
        }
    }
}

test "SpscRing crosses threads in order" {
    var ring = SpscRing(u64, 64){};
    const thread = try std.Thread.spawn(.{}, struct {
        fn run(r: *SpscRing(u64, 64)) void {
            var i: u64 = 1;
            while (i <= 10_000) : (i += 1) {
                while (!r.push(i)) std.atomic.spinLoopHint();
            }
        }
    }.run, .{&ring});
    defer thread.join();

    var expected: u64 = 1;
    while (expected <= 10_000) {
        if (ring.pop()) |v| {
            try testing_expectEqual(u64, expected, v);
            expected += 1;
        } else {
            std.atomic.spinLoopHint();
        }
    }
}

fn testing_expect(ok: bool) !void {
    if (!ok) return error.TestUnexpectedResult;
}

fn testing_expectEqual(comptime T: type, expected: T, actual: T) !void {
    if (expected != actual) return error.TestExpectedEqual;
}

test "Mutex serializes concurrent sections" {
    var m = Mutex{};
    var counter = std.atomic.Value(u32).init(0);
    const W = struct {
        fn run(mu: *Mutex, c: *std.atomic.Value(u32)) void {
            var i: usize = 0;
            while (i < 10_000) : (i += 1) {
                mu.lock();
                // Non-atomic read-modify-write inside the critical section:
                // lost updates would show up as a short count.
                const v = c.load(.monotonic);
                std.atomic.spinLoopHint();
                c.store(v + 1, .monotonic);
                mu.unlock();
            }
        }
    };
    const t1 = try std.Thread.spawn(.{}, W.run, .{ &m, &counter });
    const t2 = try std.Thread.spawn(.{}, W.run, .{ &m, &counter });
    t1.join();
    t2.join();
    try std.testing.expectEqual(@as(u32, 20_000), counter.load(.acquire));
}

test "Mutex tryLock probes ownership without blocking" {
    var m = Mutex{};
    // Free: tryLock acquires.
    try std.testing.expect(m.tryLock());
    m.unlock();

    // Held by a worker: tryLock reports contended (false) without waiting.
    // The worker signals via atomics; both waits are bounded spins, no sleeps.
    var locked = std.atomic.Value(bool).init(false);
    var release = std.atomic.Value(bool).init(false);
    const H = struct {
        fn run(mu: *Mutex, held: *std.atomic.Value(bool), rel: *std.atomic.Value(bool)) void {
            mu.lock();
            held.store(true, .release);
            while (!rel.load(.acquire)) std.atomic.spinLoopHint();
            mu.unlock();
        }
    };
    const t = try std.Thread.spawn(.{}, H.run, .{ &m, &locked, &release });
    var spins: usize = 0;
    while (!locked.load(.acquire)) : (spins += 1) {
        if (spins > 10_000_000) return error.TestUnexpectedResult;
        std.atomic.spinLoopHint();
    }
    try std.testing.expect(!m.tryLock());
    release.store(true, .release);
    t.join();

    // Released again: tryLock acquires.
    try std.testing.expect(m.tryLock());
    m.unlock();
}

test "Mutex tryLockWithin acquires a free lock immediately" {
    var m = Mutex{};
    // Zero budget on a free lock still acquires via the first probe.
    try std.testing.expect(m.tryLockWithin(0));
    m.unlock();

    const start = monoNs();
    try std.testing.expect(m.tryLockWithin(2_000_000));
    m.unlock();
    const elapsed = monoNs() - start;
    // Immediate: well under the budget plus generous scheduling slack.
    try std.testing.expect(elapsed <= 2_000_000 + 10_000_000);
}

test "Mutex tryLockWithin times out while the lock is held" {
    var m = Mutex{};
    var locked = std.atomic.Value(bool).init(false);
    var release = std.atomic.Value(bool).init(false);
    const H = struct {
        fn run(mu: *Mutex, held: *std.atomic.Value(bool), rel: *std.atomic.Value(bool)) void {
            mu.lock();
            held.store(true, .release);
            while (!rel.load(.acquire)) std.atomic.spinLoopHint();
            mu.unlock();
        }
    };
    const t = try std.Thread.spawn(.{}, H.run, .{ &m, &locked, &release });
    var spins: usize = 0;
    while (!locked.load(.acquire)) : (spins += 1) {
        if (spins > 10_000_000) return error.TestUnexpectedResult;
        std.atomic.spinLoopHint();
    }
    const timeout: u64 = 2_000_000; // 2ms
    const start = monoNs();
    try std.testing.expect(!m.tryLockWithin(timeout));
    const elapsed = monoNs() - start;
    // Waited out (near) the full budget, never more than one sleep step
    // over it plus generous scheduling slack.
    try std.testing.expect(elapsed >= 1_000_000);
    try std.testing.expect(elapsed <= timeout + 10_000_000);

    // True only after release.
    release.store(true, .release);
    t.join();
    try std.testing.expect(m.tryLockWithin(timeout));
    m.unlock();
}

test "Mutex tryLockWithin acquires after a mid-wait release" {
    var m = Mutex{};
    defer m.deinit();
    var held = std.atomic.Value(bool).init(false);
    var release_holder = std.atomic.Value(bool).init(false);
    // The holder releases ONLY after tryLockWithin reaches its contended
    // wait. Scheduling delays cannot turn this into an uncontended pass.
    const H = struct {
        fn run(mu: *Mutex, h: *std.atomic.Value(bool), release: *std.atomic.Value(bool)) void {
            mu.lock();
            defer mu.unlock();
            h.store(true, .release);
            while (!release.load(.acquire)) sleepNs(50_000);
        }
    };
    const t = try std.Thread.spawn(.{}, H.run, .{ &m, &held, &release_holder });
    defer {
        release_holder.store(true, .release);
        t.join();
    }
    const start = monoNs();
    while (!held.load(.acquire)) {
        if (monoNs() - start > 10_000_000_000) return error.TestUnexpectedResult;
        sleepNs(50_000);
    }
    const Observer = struct {
        threadlocal var release: *std.atomic.Value(bool) = undefined;
        fn observe() void {
            release.store(true, .release);
        }
    };
    Observer.release = &release_holder;
    jobs.test_wait_observer = Observer.observe;
    defer jobs.test_wait_observer = null;
    try std.testing.expect(m.tryLockWithin(10_000_000_000));
    defer m.unlock();
    try std.testing.expect(release_holder.load(.acquire));
}

test "Pool.forkJoin is multi-producer safe under concurrent callers" {
    const a = std.testing.allocator;
    const pool = try Pool.init(a, 2);
    defer pool.deinit();

    const Shared = struct {
        p: *Pool,
        total1: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
        total2: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),

        fn worker1(s: *@This()) void {
            for (0..20) |_| {
                const ItemCtx = struct {
                    c: *std.atomic.Value(usize),
                    fn run(ctx: *@This(), start: usize, end: usize) void {
                        _ = ctx.c.fetchAdd(end - start, .monotonic);
                    }
                };
                var ic = ItemCtx{ .c = &s.total1 };
                s.p.forkJoin(ItemCtx, &ic, ItemCtx.run, 500);
            }
        }

        fn worker2(s: *@This()) void {
            for (0..20) |_| {
                const ItemCtx = struct {
                    c: *std.atomic.Value(usize),
                    fn run(ctx: *@This(), start: usize, end: usize) void {
                        _ = ctx.c.fetchAdd(end - start, .monotonic);
                    }
                };
                var ic = ItemCtx{ .c = &s.total2 };
                s.p.forkJoin(ItemCtx, &ic, ItemCtx.run, 500);
            }
        }
    };

    var shared = Shared{ .p = pool };
    const t1 = try std.Thread.spawn(.{}, Shared.worker1, .{&shared});
    const t2 = try std.Thread.spawn(.{}, Shared.worker2, .{&shared});
    t1.join();
    t2.join();

    try std.testing.expectEqual(@as(usize, 20 * 500), shared.total1.load(.acquire));
    try std.testing.expectEqual(@as(usize, 20 * 500), shared.total2.load(.acquire));
}
