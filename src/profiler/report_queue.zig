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

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

/// Sleep helper (Zig 0.16 has no Thread.sleep).
fn testSleepNs(ns: u64) void {
    jobs.sleepNs(ns);
}

test "report queue writes owned bytes and frees the task" {
    const ally = std.testing.allocator;
    // Zero-thread runner: inline execution, same code path, sync behavior.
    const runner = try jobs.TaskRunner.init(ally, 0);
    defer runner.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_base = try std.fmt.allocPrint(ally, ".zig-cache/tmp/{s}/report_q", .{tmp.sub_path});
    defer ally.free(tmp_base);
    const out_path = try std.fmt.allocPrint(ally, "{s}.html", .{tmp_base});
    defer ally.free(out_path);

    const payload = try ally.dupe(u8, "<html>owned-bytes</html>");
    // Ownership moves into the queue here: no free of payload below.
    const task = try enqueueReportWrite(ally, runner, out_path, payload);
    try std.testing.expect(task.isDone());
    try std.testing.expect(task.isSuccess());
    try std.testing.expectEqual(payload.len, task.bytes_written);
    const n = task.bytes_written;

    const io = std.Io.Threaded.global_single_threaded.io();
    const back = try std.Io.Dir.cwd().readFileAlloc(io, out_path, ally, .unlimited);
    defer ally.free(back);
    try std.testing.expectEqualSlices(u8, "<html>owned-bytes</html>", back);
    try std.testing.expectEqual(back.len, n);
    task.deinit();
    // testing.allocator fails the test on any leak: payload + path + task
    // must all be gone exactly once (no leak, no double free).
}

test "report queue failure reports failed state and stays deinit-safe" {
    const ally = std.testing.allocator;
    const runner = try jobs.TaskRunner.init(ally, 0);
    defer runner.deinit();

    // Nonexistent directory: the write must fail, never hang or panic.
    const payload = try ally.dupe(u8, "doomed");
    const task = try enqueueReportWrite(ally, runner, ".zig-cache/tmp/definitely-not-here-xyz/report.html", payload);
    try std.testing.expect(task.isDone());
    try std.testing.expect(!task.isSuccess());
    try std.testing.expect(task.err_name != null);
    try std.testing.expectEqual(@as(usize, 0), task.bytes_written);
    task.deinit();
}

test "report queue enqueue failure keeps data ownership with the caller" {
    const ally = std.testing.allocator;
    const runner = try jobs.TaskRunner.init(ally, 0);
    defer runner.deinit();

    const payload = try ally.dupe(u8, "still-mine");
    // Failing allocator refuses the task struct: ownership must NOT move.
    const bad = std.testing.failing_allocator;
    const err = enqueueReportWrite(bad, runner, "x.html", payload);
    try std.testing.expectError(error.OutOfMemory, err);
    // Caller still owns the bytes: freeing here must be exactly-once clean.
    ally.free(payload);
}

test "report queue hands off across threads on a live runner" {
    const ally = std.testing.allocator;
    const runner = try jobs.TaskRunner.init(ally, 1);
    defer runner.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const tmp_base = try std.fmt.allocPrint(ally, ".zig-cache/tmp/{s}/report_q_mt", .{tmp.sub_path});
    defer ally.free(tmp_base);
    const out_path = try std.fmt.allocPrint(ally, "{s}.json", .{tmp_base});
    defer ally.free(out_path);

    const payload = try ally.dupe(u8, "{\"trace\":true}");
    const task = try enqueueReportWrite(ally, runner, out_path, payload);
    // Bounded cross-thread poll (never a bare flag wait): the runner owns
    // progress; 5 s is generous for a 15-byte file write.
    const deadline = jobs.monoNs() + 5_000_000_000;
    while (!task.isDone()) {
        if (jobs.monoNs() >= deadline) return error.TestUnexpectedResult;
        testSleepNs(500_000);
    }
    try std.testing.expect(task.isSuccess());
    const io = std.Io.Threaded.global_single_threaded.io();
    const back = try std.Io.Dir.cwd().readFileAlloc(io, out_path, ally, .unlimited);
    defer ally.free(back);
    try std.testing.expectEqualSlices(u8, "{\"trace\":true}", back);
    task.deinit();
}
