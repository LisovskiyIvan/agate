const std = @import("std");
const meter = @import("gpu_upload_meter.zig");

test "record accumulates, takeAndReset returns sum and clears" {
    _ = meter.takeAndReset();
    meter.record(100);
    meter.record(56);
    try std.testing.expectEqual(@as(u64, 156), meter.peek());
    try std.testing.expectEqual(@as(u64, 156), meter.takeAndReset());
    try std.testing.expectEqual(@as(u64, 0), meter.peek());
    try std.testing.expectEqual(@as(u64, 0), meter.takeAndReset());
}

test "parallel record calls from workers do not drop bytes" {
    _ = meter.takeAndReset();
    const Worker = struct {
        fn run(n: usize) void {
            var i: usize = 0;
            while (i < n) : (i += 1) meter.record(64);
        }
    };
    const thread_count = 4;
    const per_thread = 1000;
    var threads: [thread_count]std.Thread = undefined;
    for (&threads) |*t| t.* = try std.Thread.spawn(.{}, Worker.run, .{per_thread});
    for (&threads) |*t| t.join();
    try std.testing.expectEqual(@as(u64, thread_count * per_thread * 64), meter.takeAndReset());
}
