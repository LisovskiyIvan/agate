const std = @import("std");
const Profiler = @import("core.zig").Profiler;
const sokol = @import("sokol");
const SceneStats = @import("../scene/stats.zig").SceneStats;

fn testSleepMs(ms: u64) void {
    const ts = std.c.timespec{ .sec = @intCast(ms / 1000), .nsec = @intCast((ms % 1000) * 1_000_000) };
    var rem: std.c.timespec = undefined;
    _ = std.c.nanosleep(&ts, &rem);
}

test "Profiler analyzer detects bottlenecks" {
    sokol.time.setup();
    const ally = std.testing.allocator;
    var prof = Profiler.init(ally);
    defer prof.deinit();

    prof.start();
    var stats: SceneStats = .{
        .update_ms = 1.0,
        .prepare_ms = 0.5,
        .shadow_ms = 1.0,
        .main_ms = 45.0, // huge hitch in main pass
        .post_ms = 1.0,
        .draw_calls = 1200, // critical draw call count
        .triangles = 200000,
        .pipeline_switches = 150,
    };
    for (0..10) |i| {
        if (i > 0) testSleepMs(40);
        prof.recordFrame(i, &stats);
    }
    prof.stop();

    const summary = prof.summarize();
    try std.testing.expect(summary.interval_hitches_over_33ms > 0);

    const findings = try prof.analyze(null, ally);
    defer {
        for (findings) |*f| @constCast(f).deinit(ally);
        ally.free(findings);
    }

    var found_draw_calls = false;
    var found_main_pass = false;
    var found_hitch = false;

    for (findings) |f| {
        if (std.mem.indexOf(u8, f.title, "Draw Calls") != null or std.mem.indexOf(u8, f.title, "draw call") != null) found_draw_calls = true;
        if (std.mem.indexOf(u8, f.title, "Main Render Pass") != null) found_main_pass = true;
        if (std.mem.indexOf(u8, f.title, "Framerate drop") != null) found_hitch = true;
    }

    try std.testing.expect(found_draw_calls);
    try std.testing.expect(found_main_pass);
    try std.testing.expect(found_hitch);
}
