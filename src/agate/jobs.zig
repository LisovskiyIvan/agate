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

const windows = std.os.windows;

// Windows API externs (kernel32) for thread synchronization
const win_sync = if (builtin.os.tag == .windows) struct {
    extern "kernel32" fn AcquireSRWLockExclusive(SRWLock: *windows.SRWLOCK) callconv(.winapi) void;
    extern "kernel32" fn ReleaseSRWLockExclusive(SRWLock: *windows.SRWLOCK) callconv(.winapi) void;
    extern "kernel32" fn TryAcquireSRWLockExclusive(SRWLock: *windows.SRWLOCK) callconv(.winapi) windows.BOOLEAN;
    extern "kernel32" fn SleepConditionVariableSRW(ConditionVariable: *windows.CONDITION_VARIABLE, SRWLock: *windows.SRWLOCK, dwMilliseconds: windows.DWORD, Flags: windows.ULONG) callconv(.winapi) windows.BOOL;
    extern "kernel32" fn WakeAllConditionVariable(ConditionVariable: *windows.CONDITION_VARIABLE) callconv(.winapi) void;
    extern "kernel32" fn Sleep(dwMilliseconds: windows.DWORD) callconv(.winapi) void;
    extern "kernel32" fn QueryPerformanceCounter(lpPerformanceCount: *windows.LARGE_INTEGER) callconv(.winapi) windows.BOOL;
    extern "kernel32" fn QueryPerformanceFrequency(lpFrequency: *windows.LARGE_INTEGER) callconv(.winapi) windows.BOOL;
} else struct {};

const INFINITE: u32 = 0xFFFF_FFFF;

/// Mutex + condvar pair guarding the wake protocol. All parking
/// workers share one lot; the producer broadcasts on it.
/// Uses native SRWLock / ConditionVariable on Windows, pthread on POSIX.
const ParkingLot = if (builtin.os.tag == .windows) struct {
    srw: windows.SRWLOCK = .{},
    cond: windows.CONDITION_VARIABLE = .{},

    fn lock(self: *ParkingLot) void {
        win_sync.AcquireSRWLockExclusive(&self.srw);
    }

    fn unlock(self: *ParkingLot) void {
        win_sync.ReleaseSRWLockExclusive(&self.srw);
    }

    fn sleep(self: *ParkingLot) void {
        _ = win_sync.SleepConditionVariableSRW(&self.cond, &self.srw, INFINITE, 0);
    }

    fn broadcast(self: *ParkingLot) void {
        win_sync.WakeAllConditionVariable(&self.cond);
    }

    fn deinit(_: *ParkingLot) void {}
} else struct {
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

    fn deinit(self: *ParkingLot) void {
        _ = std.c.pthread_mutex_destroy(&self.mutex);
        _ = std.c.pthread_cond_destroy(&self.cond);
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
    dispatch_mutex: Mutex = .{},
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
            .dispatch_mutex = .{},
        };
        errdefer {
            self.lot.deinit();
            self.dispatch_mutex.deinit();
        }
        const effective: usize = if (builtin.single_threaded) 0 else worker_count;
        if (effective == 0) return self;

        self.workers = try allocator.alloc(std.Thread, effective);
        errdefer allocator.free(self.workers);
        self.mailbox = try allocator.alloc(std.atomic.Value(?*Job), effective);
        errdefer allocator.free(self.mailbox);
        for (self.mailbox) |*m| m.* = std.atomic.Value(?*Job).init(null);

        var spawned: usize = 0;
        errdefer {
            self.lot.lock();
            self.quit = true;
            self.lot.unlock();
            self.lot.broadcast();
            for (self.workers[0..spawned]) |t| t.join();
        }
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
        self.lot.deinit();
        self.dispatch_mutex.deinit();
        a.destroy(self);
    }

    pub fn workerCount(self: *const Pool) usize {
        return self.workers.len;
    }

    /// Splits [0, len) into chunks and runs `run(ctx, start, end)` over all
    /// of them, cooperating with the workers. Blocks until every chunk is
    /// finished and every fired worker has stopped reading the job.
    /// Thread-safe: serializes multiple producers via dispatch_mutex.
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
        self.dispatch_mutex.lock();
        defer self.dispatch_mutex.unlock();
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

/// Unit-test handshake at the contended-wait boundary; compiled out of apps.
pub threadlocal var test_wait_observer: ?*const fn () void = null;

/// Blocking mutex for coarse phase ownership: the threaded game
/// loop holds it during Scene.update, the sapp thread during render, so
/// the two phases never overlap. Small critical sections only — this is
/// ownership, not a data-race bandage. Condition-variable parking lives
/// in ParkingLot (TaskRunner); this type is the bare lock.
/// Uses native SRWLock on Windows, pthread_mutex on POSIX.
pub const Mutex = if (builtin.os.tag == .windows) struct {
    srw: windows.SRWLOCK = .{},

    pub fn lock(self: *Mutex) void {
        win_sync.AcquireSRWLockExclusive(&self.srw);
    }

    pub fn tryLock(self: *Mutex) bool {
        return win_sync.TryAcquireSRWLockExclusive(&self.srw) != .FALSE;
    }

    pub fn tryLockWithin(self: *Mutex, timeout_ns: u64) bool {
        if (self.tryLock()) return true;
        if (timeout_ns == 0) return false;
        const start = monoNs();
        var slept: u64 = 0;
        while (true) {
            const now = monoNs();
            const clock_elapsed: u64 = if (now >= start) now - start else 0;
            const elapsed: u64 = @max(clock_elapsed, slept);
            if (elapsed >= timeout_ns) return false;
            const remaining = timeout_ns - elapsed;
            const step: u64 = @min(remaining, 50_000); // 50us park
            if (builtin.is_test) {
                if (test_wait_observer) |obs| obs();
            }
            sleepNs(step);
            slept += step;
            if (self.tryLock()) return true;
        }
    }

    pub fn unlock(self: *Mutex) void {
        win_sync.ReleaseSRWLockExclusive(&self.srw);
    }

    pub fn deinit(_: *Mutex) void {}
} else struct {
    mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,

    pub fn lock(self: *Mutex) void {
        _ = std.c.pthread_mutex_lock(&self.mutex);
    }

    /// Non-blocking ownership probe for the phase mutex; false means
    /// contended, caller falls back; no waiting.
    pub fn tryLock(self: *Mutex) bool {
        return std.c.pthread_mutex_trylock(&self.mutex) == .SUCCESS;
    }

    /// Bounded acquisition for the phase mutex: tries `tryLock` until
    /// `timeout_ns` elapses on the monotonic clock, parking briefly
    /// between attempts instead of spinning hot. A render consumer uses
    /// this when it prefers a fresh prepare but must stay bounded and
    /// fall back to frame reuse on timeout. Never unbounded, never spins
    /// hot. Total wait exceeds `timeout_ns` by at most one sleep step
    /// (plus scheduling jitter). Deliberately a sleep loop, not
    /// pthread_mutex_timedlock (darwin declaration issues); the sleep
    /// loop is the portable contract.
    pub fn tryLockWithin(self: *Mutex, timeout_ns: u64) bool {
        if (self.tryLock()) return true;
        if (timeout_ns == 0) return false;
        const start = monoNs();
        var slept: u64 = 0;
        while (true) {
            const now = monoNs();
            const clock_elapsed: u64 = if (now >= start) now - start else 0;
            // `slept` bounds the wait even if the clock ever stalls, so
            // termination never depends on clock progress alone.
            const elapsed: u64 = @max(clock_elapsed, slept);
            if (elapsed >= timeout_ns) return false;
            const remaining = timeout_ns - elapsed;
            const step: u64 = @min(remaining, 50_000); // 50us park
            if (builtin.is_test) {
                if (test_wait_observer) |obs| obs();
            }
            sleepNs(step);
            slept += step;
            if (self.tryLock()) return true;
        }
    }

    pub fn unlock(self: *Mutex) void {
        _ = std.c.pthread_mutex_unlock(&self.mutex);
    }

    pub fn deinit(self: *Mutex) void {
        _ = std.c.pthread_mutex_destroy(&self.mutex);
    }
};

/// Monotonic nanoseconds: QPC on Windows, clock_gettime(CLOCK.MONOTONIC) on POSIX.
pub fn monoNs() u64 {
    if (builtin.os.tag == .windows) {
        var count: windows.LARGE_INTEGER = undefined;
        var freq: windows.LARGE_INTEGER = undefined;
        if (win_sync.QueryPerformanceCounter(&count) == .FALSE) return 0;
        if (win_sync.QueryPerformanceFrequency(&freq) == .FALSE) return 0;
        const c: u64 = @bitCast(count);
        const f: u64 = @bitCast(freq);
        if (f == 0) return 0;
        const c_u128: u128 = c;
        const res = (c_u128 * 1_000_000_000) / f;
        return @intCast(@min(res, std.math.maxInt(u64)));
    } else {
        var ts: std.c.timespec = undefined;
        if (std.c.clock_gettime(.MONOTONIC, &ts) != 0) return 0;
        const sec: i64 = @intCast(ts.sec);
        const nsec: i64 = @intCast(ts.nsec);
        if (sec <= 0) return @intCast(@max(nsec, @as(i64, 0)));
        return @as(u64, @intCast(sec)) * 1_000_000_000 + @as(u64, @intCast(nsec));
    }
}

/// Parks the caller for `ns` nanoseconds: Sleep on Windows, nanosleep on POSIX.
pub fn sleepNs(ns: u64) void {
    if (builtin.os.tag == .windows) {
        win_sync.Sleep(@intCast(if (ns == 0) 0 else @max(1, ns / 1_000_000)));
    } else {
        const ts = std.c.timespec{
            .sec = @intCast(ns / 1_000_000_000),
            .nsec = @intCast(ns % 1_000_000_000),
        };
        _ = std.c.nanosleep(&ts, null);
    }
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
        errdefer {
            self.lot.deinit();
        }
        const effective: usize = if (builtin.single_threaded) 0 else thread_count;
        if (effective == 0) return self;

        self.threads = try allocator.alloc(std.Thread, effective);
        errdefer allocator.free(self.threads);
        var spawned: usize = 0;
        errdefer {
            self.lot.lock();
            self.quit = true;
            self.lot.unlock();
            self.lot.broadcast();
            for (self.threads[0..spawned]) |t| t.join();
        }
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
        self.lot.deinit();
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
