//! Dedicated physics-step worker (see `docs/threading.md`).
//!
//! A Scene-owned single worker thread that runs `PhysicsWorld.step(dt)`
//! while the game thread continues with the physics-independent phases
//! (fork-join inside `Scene.update`). Handoff is a mutex + condition pair
//! with an explicit idle/work/done/stop state machine: no per-frame
//! allocations, no busy-burn (the worker parks on the condvar between
//! kicks, the game thread parks on it in `join`).
//!
//! Zig 0.16 `std.Thread` ships no public mutex/condvar (same reason
//! `jobs.zig` wraps libc directly in its `ParkingLot`), so this file wraps
//! the same primitives the same way: SRWLock + CONDITION_VARIABLE on
//! Windows, pthread mutex + cond on POSIX. `jobs.TaskRunner` stays for
//! queued background work; this worker is a stricter single-slot
//! kick/join pair with pairing asserts.
//!
//! Ownership: only the scene-update (game) thread calls `kick`/`join`/
//! `shutdown`. The worker thread only runs `step` between
//! `beginParallelWindow`/`endParallelWindow` and never touches scene state.

const std = @import("std");
const builtin = @import("builtin");
const physics = @import("../physics.zig");
const PhysicsWorld = physics.PhysicsWorld;

const windows = std.os.windows;

const win_sync = if (builtin.os.tag == .windows) struct {
    extern "kernel32" fn AcquireSRWLockExclusive(SRWLock: *windows.SRWLOCK) callconv(.winapi) void;
    extern "kernel32" fn ReleaseSRWLockExclusive(SRWLock: *windows.SRWLOCK) callconv(.winapi) void;
    extern "kernel32" fn SleepConditionVariableSRW(ConditionVariable: *windows.CONDITION_VARIABLE, SRWLock: *windows.SRWLOCK, dwMilliseconds: windows.DWORD, Flags: windows.ULONG) callconv(.winapi) windows.BOOL;
    extern "kernel32" fn WakeAllConditionVariable(ConditionVariable: *windows.CONDITION_VARIABLE) callconv(.winapi) void;
} else struct {};

const INFINITE: u32 = 0xFFFF_FFFF;

/// Mutex + condvar handoff pair. Same shape as `jobs.zig`'s `ParkingLot`
/// (kept local: that type is private, and this worker needs signal
/// semantics for exactly one waiter rather than pool broadcast).
const HandoffLot = if (builtin.os.tag == .windows) struct {
    srw: windows.SRWLOCK = .{},
    cond: windows.CONDITION_VARIABLE = .{},

    fn lock(self: *HandoffLot) void {
        win_sync.AcquireSRWLockExclusive(&self.srw);
    }

    fn unlock(self: *HandoffLot) void {
        win_sync.ReleaseSRWLockExclusive(&self.srw);
    }

    fn sleep(self: *HandoffLot) void {
        _ = win_sync.SleepConditionVariableSRW(&self.cond, &self.srw, INFINITE, 0);
    }

    fn broadcast(self: *HandoffLot) void {
        win_sync.WakeAllConditionVariable(&self.cond);
    }

    fn deinit(_: *HandoffLot) void {}
} else struct {
    mutex: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER,
    cond: std.c.pthread_cond_t = std.c.PTHREAD_COND_INITIALIZER,

    fn lock(self: *HandoffLot) void {
        _ = std.c.pthread_mutex_lock(&self.mutex);
    }

    fn unlock(self: *HandoffLot) void {
        _ = std.c.pthread_mutex_unlock(&self.mutex);
    }

    /// Parks until `broadcast` fires. Must hold the mutex; the predicate
    /// re-check belongs to the caller (`while` loop).
    fn sleep(self: *HandoffLot) void {
        _ = std.c.pthread_cond_wait(&self.cond, &self.mutex);
    }

    /// Wakes the parked waiter. The state change must have happened under
    /// `lock`, otherwise the wakeup can be lost between check and sleep.
    fn broadcast(self: *HandoffLot) void {
        _ = std.c.pthread_cond_broadcast(&self.cond);
    }

    fn deinit(self: *HandoffLot) void {
        _ = std.c.pthread_mutex_destroy(&self.mutex);
        _ = std.c.pthread_cond_destroy(&self.cond);
    }
};

/// Always-on handoff counters. Plain u64, game-thread-read like
/// `Runtime.metrics`: `kick`/`join`/`shutdown` all run on the scene-update
/// thread, the worker never touches them.
pub const WorkerCounters = struct {
    kicks: u64 = 0,
    joins: u64 = 0,
    fallbacks: u64 = 0,
};

const WorkerState = enum {
    idle,
    work,
    done,
    stop,
};

/// Best-effort worker thread name ("agate-physics"). Zig 0.16
/// `SpawnConfig` carries no name, so the worker names itself after spawn;
/// failures are ignored (observability only, never correctness).
fn nameWorkerThread() void {
    if (builtin.os.tag == .windows or builtin.os.tag == .wasi) return;
    if (builtin.os.tag == .macos) {
        const c = struct {
            extern "c" fn pthread_setname_np(name: [*:0]const u8) c_int;
        };
        _ = c.pthread_setname_np("agate-physics");
    } else {
        const c = struct {
            extern "c" fn pthread_setname_np(thread: std.c.pthread_t, name: [*:0]const u8) c_int;
            extern "c" fn pthread_self() std.c.pthread_t;
        };
        _ = c.pthread_setname_np(c.pthread_self(), "agate-physics");
    }
}

pub const PhysicsWorker = struct {
    thread: ?std.Thread = null,
    lot: HandoffLot = .{},
    state: WorkerState = .idle,
    world: ?*PhysicsWorld = null,
    dt: f32 = 0,
    /// Permanent inline degradation after a spawn failure (mirrors
    /// `Runtime.spawnWorker`: fail once, stay single-threaded, count it).
    degraded: bool = false,
    /// Test hook: forces the spawn-failure path without spawning.
    test_force_spawn_fail: bool = false,
    counters: WorkerCounters = .{},

    /// Hands one step to the worker. Returns true when the worker owns the
    /// step (the caller MUST `join` before touching the world again);
    /// false when the caller must step inline instead (no world posted,
    /// fallback counted). Spawns lazily on first use; after a spawn
    /// failure degrades permanently to inline.
    pub fn kick(self: *PhysicsWorker, world: *PhysicsWorld, dt: f32) bool {
        if (comptime builtin.single_threaded or builtin.cpu.arch.isWasm()) {
            self.degraded = true;
            self.counters.fallbacks += 1;
            return false;
        }
        // Game-thread-only flag read outside the lock so a worker that was
        // shut down (and whose handoff primitives were destroyed) is never
        // re-locked by a later kick.
        if (self.degraded) {
            self.counters.fallbacks += 1;
            return false;
        }
        self.lot.lock();
        if (self.thread == null) {
            if (comptime builtin.is_test) {
                if (self.test_force_spawn_fail) {
                    self.degradeLocked("test hook forced physics worker spawn failure; running inline");
                    return false;
                }
            }
            // Spawn outside the handoff lock: pthread_create never reads
            // worker state, and holding the mutex across it would serialize
            // an unrelated syscall with the join path.
            self.lot.unlock();
            const spawned = std.Thread.spawn(.{}, run, .{self}) catch |err| {
                self.lot.lock();
                self.degradeLocked(null);
                std.debug.print("agate-physics worker spawn failed ({s}); running inline\n", .{@errorName(err)});
                return false;
            };
            self.lot.lock();
            self.thread = spawned;
        }
        // Pairing contract: every kick is consumed by exactly one join
        // before the next kick (frame_api upholds this; Debug asserts it).
        std.debug.assert(self.state == .idle);
        self.world = world;
        self.dt = dt;
        self.state = .work;
        self.counters.kicks += 1;
        self.lot.broadcast();
        self.lot.unlock();
        return true;
    }

    /// Waits for the kicked step to finish. No-op when no step is pending
    /// (never kicked, or already joined) and when no thread exists, so
    /// deinit-right-after-update and double shutdown stay safe.
    pub fn join(self: *PhysicsWorker) void {
        // Checked before the lock: shutdown() destroys the handoff primitives,
        // so a late join must not touch them.
        if (self.thread == null) return;
        self.lot.lock();
        defer self.lot.unlock();
        if (self.state == .idle) return;
        while (self.state != .done) self.lot.sleep();
        self.state = .idle;
        self.world = null;
        self.counters.joins += 1;
    }

    /// Parks the worker permanently and joins the thread. Drains an
    /// in-flight step first (defensive: `Scene.update` always joins before
    /// returning). Idempotent: a second call is a no-op that never touches
    /// the handoff primitives destroyed by the first call. After shutdown the
    /// worker degrades permanently to inline (no thread is ever re-spawned on
    /// the destroyed handoff lot).
    pub fn shutdown(self: *PhysicsWorker) void {
        if (self.thread == null) {
            self.degraded = true;
            return;
        }
        self.join();
        self.lot.lock();
        if (self.thread == null) {
            self.lot.unlock();
            self.degraded = true;
            return;
        }
        const t = self.thread.?;
        self.thread = null;
        self.state = .stop;
        self.degraded = true;
        self.lot.broadcast();
        self.lot.unlock();
        t.join();
        self.lot.deinit();
    }

    fn degradeLocked(self: *PhysicsWorker, msg: ?[]const u8) void {
        self.degraded = true;
        self.counters.fallbacks += 1;
        if (msg) |m| std.debug.print("{s}\n", .{m});
        self.lot.unlock();
    }

    fn run(self: *PhysicsWorker) void {
        nameWorkerThread();
        self.lot.lock();
        while (true) {
            while (self.state == .idle) self.lot.sleep();
            if (self.state == .stop) {
                self.lot.unlock();
                return;
            }
            const w = self.world orelse {
                // Unreachable under the kick/join pairing (kick always
                // posts a world with .work); fail closed to idle rather
                // than spinning so a logic bug can never busy-burn.
                self.state = .idle;
                continue;
            };
            const dt = self.dt;
            self.lot.unlock();
            w.beginParallelWindow();
            w.step(dt);
            w.endParallelWindow();
            self.lot.lock();
            self.state = .done;
            self.lot.broadcast();
            // Wait until the join consumes the result (done -> idle) or
            // shutdown stops us. Without this the worker would re-step
            // the same kick while the game thread reads the profile.
            while (self.state == .done) self.lot.sleep();
        }
    }
};
