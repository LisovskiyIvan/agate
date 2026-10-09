//! Async profiler report file writes over the scene `io_runner`.
//!
//! Problem: the sandbox's bounded profiler window held the phase mutex
//! ACROSS report file IO by design (rare save frames). Disk latency inside
//! an exclusion window stalls the producer — the window must shrink to
//! enqueue-only: capture (live-registry walk) under the lock, encode
//! (profiler is context-owned; the game never touches it) off-lock, and the
//! bytes handed to the existing async IO machinery.
//!
//! This module owns the last hop: a `ReportWriteTask` that OWNS its encoded
//! bytes (allocated by the profiler's `generateReportsAlloc`, never stack
//! state) plus its path, posted to a `jobs.TaskRunner` — in practice
//! `Scene.io_runner`, the dedicated 1-thread file-I/O runner that already
//! serves `saveStateFileAsync`/`loadStateFileAsync` (decode work stays on
//! `uploads.runner`, so long disk I/O can never starve texture decodes).
//!
//! Ownership/lifecycle (same precedent as `serialization.AsyncSaveTask`):
//! - `enqueueReportWrite` takes ownership of `data` on success; on failure
//!   (task-struct alloc or path dupe) `data` stays the caller's to free.
//! - The task runs to terminal state on the runner thread, then the POSTER
//!   polls `isDone()` and calls `deinit()` (which frees path + bytes +
//!   struct). The worker never frees: completion (`bytes_written`) and
//!   errors (`err_name`) must stay readable after the run, and the showcase
//!   already polls save/load tasks exactly this way from its update path.
//! - `TaskRunner.deinit` joins workers after the queue drains, so tasks
//!   posted before teardown always reach terminal state; the poster must
//!   still `deinit` them afterwards (see the sandbox cleanup drain).
//! - Zero-thread runners execute inline on the poster (same as every other
//!   `TaskRunner` user): the same code path, synchronous behavior.
//!
//! Leaf module: imports `jobs` only (plus std). No profiler/scene imports,
//! so the `profiler/*` anti-cycle rule holds.

const std = @import("std");
const jobs = @import("../jobs.zig");

/// One owned report file write: path + encoded bytes + completion state.
/// Allocated by `enqueueReportWrite`, freed by the poster's `deinit()` after
/// `isDone()`. Field writes before the terminal release store follow the
/// `PendingTexture` invariant (no writes after it: the poster may deinit on
/// observing the terminal state).
pub const ReportWriteTask = struct {
    pub const State = enum(u8) {
        pending = 0,
        writing = 1,
        completed = 2,
        failed = 3,
    };

    allocator: std.mem.Allocator,
    /// Owned dupe of the destination path.
    path: []u8,
    /// Owned encoded report bytes (from `generateReportsAlloc`).
    data: []u8,
    state: std.atomic.Value(State) = std.atomic.Value(State).init(.pending),
    bytes_written: usize = 0,
    err_name: ?[:0]const u8 = null,

    pub fn isDone(self: *const ReportWriteTask) bool {
        const s = self.state.load(.acquire);
        return s == .completed or s == .failed;
    }

    pub fn isSuccess(self: *const ReportWriteTask) bool {
        return self.state.load(.acquire) == .completed;
    }

    pub fn deinit(self: *ReportWriteTask) void {
        self.allocator.free(self.path);
        self.allocator.free(self.data);
        self.allocator.destroy(self);
    }
};

fn runWriteTask(ctx: *anyopaque) void {
    const task: *ReportWriteTask = @ptrCast(@alignCast(ctx));
    task.state.store(.writing, .release);
    const io = std.Io.Threaded.global_single_threaded.io();
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = task.path, .data = task.data }) catch |err| {
        task.err_name = @errorName(err);
        task.state.store(.failed, .release);
        return;
    };
    task.bytes_written = task.data.len;
    task.state.store(.completed, .release);
}

/// Enqueues one owned report write on `runner`. Takes ownership of `data`
/// on success (do NOT free it afterwards); on error `data` stays yours.
/// `path` is always duped (never borrowed).
pub fn enqueueReportWrite(
    allocator: std.mem.Allocator,
    runner: *jobs.TaskRunner,
    path: []const u8,
    data: []u8,
) !*ReportWriteTask {
    const task = try allocator.create(ReportWriteTask);
    errdefer allocator.destroy(task);
    const owned_path = try allocator.dupe(u8, path);
    errdefer allocator.free(owned_path);
    task.* = .{
        .allocator = allocator,
        .path = owned_path,
        .data = data,
        .state = std.atomic.Value(ReportWriteTask.State).init(.pending),
        .bytes_written = 0,
        .err_name = null,
    };
    runner.post(task, runWriteTask);
    return task;
}
