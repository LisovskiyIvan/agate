const std = @import("std");
const jobs = @import("../jobs.zig");
const report_queue = @import("report_queue.zig");
const enqueueReportWrite = report_queue.enqueueReportWrite;

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
