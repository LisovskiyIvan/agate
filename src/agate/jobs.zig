//! Fork-join data-parallel job pool.
//!
//! Executes `parallelFor`-style work over a disjoint index range: the calling
//! thread and N workers forage chunks of the range off an atomic cursor.
//! Jobs are pure CPU work — they must never call `sg.*` (sokol is
//! single-context, see REFACTOR.md), never allocate shared state, and never
//! nest a `parallelFor` inside a worker.
//!
//! Determinism contract: a job that writes only to indices it owns produces
//! bit-identical results for any worker count. Scheduling (which thread runs
//! which chunk) is never observable through results.
//!
//! Threading model: single-producer — only the thread that owns the pool
//! (the main thread) may call `forkJoin`; workers only consume. Workers park
//! on a pthread condition (Zig 0.16 std.Thread ships no public condvar and
//! the std.Io futexes are internal, so jobs.zig wraps std.c directly; the
//! engine links libc for sokol anyway). `pending` counts active participants
//! (workers + the calling thread); every participant decrements after it
//! stops foraging. The calling thread spins until `pending == 0`, which
//! proves every fired worker finished reading the stack-allocated job.

const std = @import("std");
const builtin = @import("builtin");

/// pthread mutex + condvar pair guarding the wake protocol. All parking
/// workers share one lot; the producer broadcasts on it.
const ParkingLot = struct {
    mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,
    cond: std.c.pthread_cond_t = std.c.PTHREAD_COND_INITIALIZER,

    fn lock(self: *ParkingLot) void {
        _ = std.c.pthread_mutex_lock(&self.mutex);
    }

    fn unlock(self: *ParkingLot) void {
        _ = std.c.pthread_mutex_unlock(&self.mutex);
    }

    /// Sleeps until `broadcast` fires. Must be called with the mutex held;
    /// atomically releases it while waiting, reacquires before returning.
    /// The predicate re-check belongs to the caller (`while` loop).
    fn sleep(self: *ParkingLot) void {
        _ = std.c.pthread_cond_wait(&self.cond, &self.mutex);
    }

    /// Wakes every parked worker. The predicate change must have happened
    /// under `lock`, otherwise the broadcast can be lost between a worker's
    /// check and its sleep.
    fn broadcast(self: *ParkingLot) void {
        _ = std.c.pthread_cond_broadcast(&self.cond);
    }
};

pub const Pool = struct {
    /// Below this length the generic `parallelFor` runs inline on the
    /// calling thread: waking the workers costs more than the chunks.
    pub const min_len_for_workers: usize = 4096;

    const Job = struct {
        ctx: *anyopaque,
        run: *const fn (ctx: *anyopaque, start: usize, end: usize) void,
        len: usize,
        chunk: usize,
        /// Next unclaimed chunk start; chunks are [c, min(c + chunk, len)).
        cursor: std.atomic.Value(usize),
        /// Participants still inside forage (workers + calling thread).
        pending: std.atomic.Value(usize),
    };

    allocator: std.mem.Allocator,
    workers: []std.Thread,
    /// Per-worker published job; null = nothing to do. Workers release it
    /// with an .acq_rel swap, so they always see a valid job or null.
    mailbox: []std.atomic.Value(?*Job),
    lot: ParkingLot = .{},
    /// Protected by `lot.mutex`; workers check it while holding the lock.
    quit: bool = false,

    /// Worker count for pool setup: one thread per core, capped so
    /// pathological many-core machines don't over-park, floor 1.
    pub fn recommendedWorkerCount() usize {
        const cpus = std.Thread.getCpuCount() catch return 1;
        return @min(@max(cpus -| 1, 1), 8);
    }

    /// Heap-allocates the pool so worker threads hold a stable pointer.
    /// `worker_count == 0` (or single-threaded builds) yields a valid pool
    /// whose forkJoin runs everything inline.
    pub fn init(allocator: std.mem.Allocator, worker_count: usize) !*Pool {
        const self = try allocator.create(Pool);
        errdefer allocator.destroy(self);
        self.* = .{
            .allocator = allocator,
            .workers = &.{},
            .mailbox = &.{},
        };
        const effective: usize = if (builtin.single_threaded) 0 else worker_count;
        if (effective == 0) return self;

        self.workers = try allocator.alloc(std.Thread, effective);
        errdefer allocator.free(self.workers);
        self.mailbox = try allocator.alloc(std.atomic.Value(?*Job), effective);
        errdefer allocator.free(self.mailbox);
        for (self.mailbox) |*m| m.* = std.atomic.Value(?*Job).init(null);

        var spawned: usize = 0;
        errdefer for (self.workers[0..spawned]) |t| t.join();
        for (0..effective) |i| {
            self.workers[i] = try std.Thread.spawn(.{}, workerMain, .{ self, i });
            spawned += 1;
        }
        return self;
    }

    /// Joins and frees everything. No forkJoin may be in flight; the pool
    /// must not be used afterwards.
    pub fn deinit(self: *Pool) void {
        if (self.workers.len > 0) {
            self.lot.lock();
            self.quit = true;
            self.lot.unlock();
            self.lot.broadcast();
            for (self.workers) |t| t.join();
        }
        const a = self.allocator;
        if (self.workers.len > 0) a.free(self.workers);
        if (self.mailbox.len > 0) a.free(self.mailbox);
        _ = std.c.pthread_mutex_destroy(&self.lot.mutex);
        _ = std.c.pthread_cond_destroy(&self.lot.cond);
        a.destroy(self);
    }

    pub fn workerCount(self: *const Pool) usize {
        return self.workers.len;
    }

    /// Splits [0, len) into chunks and runs `run(ctx, start, end)` over all
    /// of them, cooperating with the workers. Blocks until every chunk is
    /// finished and every fired worker has stopped reading the job.
    /// Single-producer: do not call concurrently from two threads.
    pub fn forkJoin(
        self: *Pool,
        comptime C: type,
        ctx: *C,
        comptime run: fn (ctx: *C, start: usize, end: usize) void,
        len: usize,
    ) void {
        if (len == 0 or self.workers.len == 0) {
            if (len > 0) run(ctx, 0, len);
            return;
        }
        const TypeErased = struct {
            fn trampoline(raw: *anyopaque, start: usize, end: usize) void {
                run(@as(*C, @ptrCast(@alignCast(raw))), start, end);
            }
        };
        // Enough chunks per participant to smooth out load imbalance, but
        // few enough that the atomic cursor is not the bottleneck.
        const chunk = @max(len / ((self.workers.len + 1) * 4), 1);
        var job = Job{
            .ctx = @ptrCast(ctx),
            .run = TypeErased.trampoline,
            .len = len,
            .chunk = chunk,
            .cursor = std.atomic.Value(usize).init(0),
            .pending = std.atomic.Value(usize).init(self.workers.len + 1),
        };
        // Publish before waking: the mailbox stores happen-before this
        // thread's wakeWorkers handshake, and workers only read the mailbox
        // after passing the same mutex, so their acquire loads always
        // observe a fully initialized job.
        for (self.mailbox) |*m| m.store(&job, .release);
        self.wakeWorkers();
        forage(&job);
        _ = job.pending.fetchSub(1, .acq_rel);
        while (job.pending.load(.acquire) != 0) std.atomic.spinLoopHint();
    }

    /// Mutex handshake with every parked worker: taking and releasing the
    /// lock proves no worker sits between its predicate check and its
    /// cond_wait, so the broadcast that follows cannot be lost.
    fn wakeWorkers(self: *Pool) void {
        self.lot.lock();
        self.lot.unlock();
        self.lot.broadcast();
    }

    fn forage(job: *Job) void {
        while (true) {
            const c = job.cursor.fetchAdd(job.chunk, .acq_rel);
            if (c >= job.len) return;
            job.run(job.ctx, c, @min(c + job.chunk, job.len));
        }
    }

    fn workerMain(self: *Pool, index: usize) void {
        while (true) {
            var exit = false;
            self.lot.lock();
            while (true) {
                // A published job exempts the worker from sleeping even
                // while it is cycling through the predicate checks.
                if (self.mailbox[index].load(.acquire) != null) break;
                if (self.quit) {
                    exit = true;
                    break;
                }
                self.lot.sleep();
            }
            self.lot.unlock();
            if (exit) return;
            const job = self.mailbox[index].swap(null, .acq_rel);
            if (job) |j| {
                forage(j);
                _ = j.pending.fetchSub(1, .acq_rel);
            }
        }
    }
};

/// Process-global pool for engine subsystems. Null (the default) means
/// serial execution: `parallelFor` runs everything on the calling thread.
/// Apps set it once at startup; tests use explicit pools instead so the
/// Zig test runner's concurrent tests never share it.
pub var global: ?*Pool = null;

/// Runs `run(ctx, 0, len)` either inline (null pool, tiny range, zero
/// workers) or fork-joined across the pool. Results are identical in both
/// modes by the determinism contract above.
pub fn parallelFor(
    pool: ?*Pool,
    comptime C: type,
    ctx: *C,
    comptime run: fn (ctx: *C, start: usize, end: usize) void,
    len: usize,
) void {
    if (pool == null or len < Pool.min_len_for_workers) {
        if (len > 0) run(ctx, 0, len);
        return;
    }
    pool.?.forkJoin(C, ctx, run, len);
}

/// Fire-and-forget background tasks on dedicated threads — the asset
/// loading half of the threading story (decode while frames render).
/// Deliberately independent of `Pool.forkJoin`: forkJoin spins its callers
/// until participants finish, so a long task (texture decode, compression)
/// must never share its threads.
///
/// `post` enqueues under the parking-lot mutex and wakes one idle worker
/// (broadcast — cheap at asset-scale rates). Tasks run to completion on
/// shutdown: `deinit` drains the queue before joining, so a posted task is
/// either done or running when deinit returns.
pub const TaskRunner = struct {
    const Task = struct {
        ctx: *anyopaque,
        run: *const fn (ctx: *anyopaque) void,
    };

    allocator: std.mem.Allocator,
    lot: ParkingLot = .{},
    threads: []std.Thread,
    queue: std.ArrayListUnmanaged(Task) = .empty,
    /// Guarded by `lot`.
    quit: bool = false,

    /// Heap-allocates so worker threads hold a stable pointer. Zero (or
    /// single-threaded builds) runs every posted task inline on the poster.
    pub fn init(allocator: std.mem.Allocator, thread_count: usize) !*TaskRunner {
        const self = try allocator.create(TaskRunner);
        errdefer allocator.destroy(self);
        self.* = .{
            .allocator = allocator,
            .threads = &.{},
        };
        const effective: usize = if (builtin.single_threaded) 0 else thread_count;
        if (effective == 0) return self;

        self.threads = try allocator.alloc(std.Thread, effective);
        errdefer allocator.free(self.threads);
        var spawned: usize = 0;
        errdefer for (self.threads[0..spawned]) |t| t.join();
        for (0..effective) |i| {
            self.threads[i] = try std.Thread.spawn(.{}, workerMain, .{self});
            spawned += 1;
        }
        return self;
    }

    pub fn deinit(self: *TaskRunner) void {
        if (self.threads.len > 0) {
            self.lot.lock();
            self.quit = true;
            self.lot.unlock();
            self.lot.broadcast();
            for (self.threads) |t| t.join();
        }
        self.queue.deinit(self.allocator);
        const a = self.allocator;
        if (self.threads.len > 0) a.free(self.threads);
        _ = std.c.pthread_mutex_destroy(&self.lot.mutex);
        _ = std.c.pthread_cond_destroy(&self.lot.cond);
        a.destroy(self);
    }

    /// Number of tasks not yet picked up. Tasks currently running are not
    /// counted; poll the task's own completion state for that.
    pub fn queuedCount(self: *TaskRunner) usize {
        self.lot.lock();
        defer self.lot.unlock();
        return self.queue.items.len;
    }

    /// Enqueues a task. `ctx` must outlive the run — tasks own their
    /// context and free it in `run`, or the poster observes completion via
    /// its own state and frees afterwards. On queue-allocation failure the
    /// task runs inline on the posting thread: scheduling fallback, never
    /// a dropped job.
    pub fn post(self: *TaskRunner, ctx: *anyopaque, run: *const fn (ctx: *anyopaque) void) void {
        self.lot.lock();
        if (self.threads.len == 0) {
            self.lot.unlock();
            run(ctx);
            return;
        }
        self.queue.append(self.allocator, .{ .ctx = ctx, .run = run }) catch {
            self.lot.unlock();
            run(ctx);
            return;
        };
        self.lot.unlock();
        self.lot.broadcast();
    }

    fn workerMain(self: *TaskRunner) void {
        while (true) {
            self.lot.lock();
            while (self.queue.items.len == 0) {
                if (self.quit) {
                    self.lot.unlock();
                    return;
                }
                self.lot.sleep();
            }
            // FIFO: asset requests decode roughly in request order, which
            // keeps later frames' textures arriving after earlier ones.
            const task = self.queue.orderedRemove(0);
            self.lot.unlock();
            task.run(task.ctx);
        }
    }
};

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

/// Lock-free single-producer / single-consumer ring buffer. The producer
/// owns `head`, the consumer owns `tail`; both indices are monotonic and
/// slot selection is `index % capacity`, so no ABA is possible.
///
/// `push` returns false when the ring is full (drop-newest policy — the
/// caller decides what a lost item means). `pop` returns null when empty.
/// Works same-thread too (plain reads then), which lets apps adopt the
/// queue before any thread actually crosses it. Stage 3 seam: sapp input
/// events are produced on the window thread and consumed by the update
/// side, whichever thread that ends up on.
pub fn SpscRing(comptime T: type, comptime capacity: usize) type {
    comptime std.debug.assert(capacity > 0 and (capacity & (capacity - 1)) == 0); // power of two
    return struct {
        const Self = @This();
        head: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        tail: std.atomic.Value(u64) = std.atomic.Value(u64).init(0),
        buf: [capacity]T = undefined,

        pub fn push(self: *Self, value: T) bool {
            const h = self.head.load(.monotonic);
            const t = self.tail.load(.acquire);
            if (h - t >= capacity) return false;
            self.buf[h % capacity] = value;
            self.head.store(h + 1, .release);
            return true;
        }

        pub fn pop(self: *Self) ?T {
            const t = self.tail.load(.monotonic);
            const h = self.head.load(.acquire);
            if (t >= h) return null;
            const v = self.buf[t % capacity];
            self.tail.store(t + 1, .release);
            return v;
        }

        /// Items available for the consumer.
        pub fn len(self: *Self) usize {
            return @intCast(self.head.load(.monotonic) - self.tail.load(.monotonic));
        }
    };
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
