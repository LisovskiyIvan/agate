const std = @import("std");
const sokol = @import("sokol");
const Profiler = @import("core.zig").Profiler;
const SceneStats = @import("../scene/stats.zig").SceneStats;

/// Sleep helper for pacing-sensitive tests (Zig 0.16 has no Thread.sleep).
fn testSleepMs(ms: u64) void {
    const ts = std.c.timespec{ .sec = @intCast(ms / 1000), .nsec = @intCast((ms % 1000) * 1_000_000) };
    var rem: std.c.timespec = undefined;
    _ = std.c.nanosleep(&ts, &rem);
}

test "Profiler start, recordFrame, and summarize" {
    // App-owned monotonic clock: initialized once, single-threaded here.
    // (Runtime: main.zig init calls sokol.time.setup() before threads spawn;
    // Profiler.start must never re-initialize it.)
    sokol.time.setup();
    const ally = std.testing.allocator;
    var prof = Profiler.init(ally);
    defer prof.deinit();

    try std.testing.expect(!prof.isRecording());
    prof.start();
    try std.testing.expect(prof.isRecording());

    // Record synthetic frames
    var stats: SceneStats = .{
        .update_ms = 2.0,
        .prepare_ms = 1.0,
        .shadow_ms = 3.0,
        .main_ms = 8.0,
        .post_ms = 2.0,
        .draw_calls = 50,
        .triangles = 10000,
        .pipeline_switches = 5,
    };

    prof.recordFrame(1, &stats);
    stats.main_ms = 12.0;
    prof.recordFrame(2, &stats);
    stats.main_ms = 25.0; // spike frame
    prof.recordFrame(3, &stats);

    prof.stop();
    try std.testing.expect(!prof.isRecording());

    const summary = prof.summarize();
    try std.testing.expectEqual(@as(usize, 3), summary.frame_count);
    try std.testing.expect(summary.avg_frame_ms > 15.0);
    try std.testing.expect(summary.max_frame_ms >= 33.0);
    try std.testing.expectEqual(@as(u32, 50), summary.avg_draw_calls);

    // Analyze
    const findings = try prof.analyze(null, ally);
    defer {
        for (findings) |*f| @constCast(f).deinit(ally);
        ally.free(findings);
    }
    try std.testing.expect(findings.len > 0);
}

test "Profiler summarize tracks measured GPU frame time" {
    sokol.time.setup();
    const ally = std.testing.allocator;
    var prof = Profiler.init(ally);
    defer prof.deinit();

    prof.start();
    var stats: SceneStats = .{
        .update_ms = 1.0,
        .prepare_ms = 0.5,
        .shadow_ms = 1.0,
        .main_ms = 6.0,
        .post_ms = 1.0,
        .draw_calls = 10,
        .triangles = 1000,
        .pipeline_switches = 1,
    };
    // Disabled path: zeros with no submission id flow through without
    // touching CPU-submit stats.
    prof.recordFrame(0, &stats);
    stats.gpu_frame_ms = 2.0;
    stats.gpu_frame_submit = 31;
    prof.recordFrame(1, &stats);
    stats.gpu_frame_ms = 4.0;
    stats.gpu_frame_submit = 32;
    prof.recordFrame(2, &stats);
    prof.stop();

    try std.testing.expectEqual(@as(f32, 0), prof.frames.items[0].gpu_frame_ms);
    try std.testing.expectEqual(@as(u32, 0), prof.frames.items[0].gpu_frame_submit);
    try std.testing.expectEqual(@as(f32, 2.0), prof.frames.items[1].gpu_frame_ms);
    try std.testing.expectEqual(@as(f32, 4.0), prof.frames.items[2].gpu_frame_ms);

    const summary = prof.summarize();
    // Average covers the two DISTINCT available submissions, not the three
    // CPU frames: (2+4)/2 == 3.
    try std.testing.expectEqual(@as(f32, 3.0), summary.avg_gpu_frame_ms);
    try std.testing.expectEqual(@as(f32, 4.0), summary.max_gpu_frame_ms);
    try std.testing.expectEqual(@as(usize, 2), summary.gpu_frame_samples);
    // CPU-submit stats are untouched by the GPU field.
    try std.testing.expectEqual(@as(f32, 9.5), summary.avg_frame_ms);
}

test "Profiler summarize uses frame intervals for pacing" {
    sokol.time.setup();
    const ally = std.testing.allocator;
    var prof = Profiler.init(ally);
    defer prof.deinit();

    prof.start();
    var stats: SceneStats = .{
        .update_ms = 1.0,
        .prepare_ms = 0.5,
        .shadow_ms = 1.0,
        .main_ms = 6.0,
        .post_ms = 1.0,
        .draw_calls = 10,
        .triangles = 1000,
        .pipeline_switches = 1,
    };
    const n_frames = 6;
    for (0..n_frames) |i| {
        // Keep every interval comfortably above the 0.1ms fps-validity gate
        // so fps == 1/dt_s holds exactly for every record.
        testSleepMs(2);
        prof.recordFrame(i, &stats);
    }
    prof.stop();

    try std.testing.expectEqual(n_frames, prof.frames.items.len);

    // Each record's wall-clock fields derive from its own dt_s.
    for (prof.frames.items) |rec| {
        try std.testing.expect(rec.dt_s > 0);
        try std.testing.expectEqual(rec.dt_s * 1000.0, rec.frame_interval_ms);
        try std.testing.expect(rec.dt_s > 0.0001);
        try std.testing.expectEqual(@as(f32, 1.0) / rec.dt_s, rec.fps);
    }

    // Recompute pacing directly from the recorded dt_s values
    // (exact comparison, no wall-clock thresholds).
    const summary = prof.summarize();

    // CPU-submit average is untouched by pacing: 1+0.5+1+6+1 == 9.5.
    try std.testing.expectEqual(@as(f32, 9.5), summary.avg_frame_ms);

    var expected = try ally.alloc(f32, n_frames);
    defer ally.free(expected);
    var m: usize = 0;
    var sum: f64 = 0;
    var max_iv: f32 = 0;
    var eh16: u32 = 0;
    var eh33: u32 = 0;
    var eh50: u32 = 0;
    for (prof.frames.items) |rec| {
        if (rec.dt_s > 0) {
            const iv = rec.frame_interval_ms;
            expected[m] = iv;
            m += 1;
            sum += iv;
            max_iv = @max(max_iv, iv);
            if (iv > 50.0) eh50 += 1;
            if (iv > 33.33) eh33 += 1;
            if (iv > 16.67) eh16 += 1;
        }
    }
    try std.testing.expect(m > 0);
    const valid = expected[0..m];
    std.mem.sort(f32, valid, {}, struct {
        fn lessThan(_: void, a: f32, b: f32) bool {
            return a < b;
        }
    }.lessThan);

    const exp_avg: f32 = @floatCast(sum / @as(f64, @floatFromInt(m)));
    const exp_fps: f32 = if (exp_avg > 0.001) 1000.0 / exp_avg else 0;
    const p50_idx = @min(m - 1, @as(usize, @intFromFloat(@as(f32, @floatFromInt(m)) * 0.50)));
    const p99_idx = @min(m - 1, @as(usize, @intFromFloat(@as(f32, @floatFromInt(m)) * 0.99)));
    const c1 = @max(1, m / 100);
    var s1: f64 = 0;
    for (valid[m - c1 ..]) |t| s1 += t;
    const a1: f32 = @floatCast(s1 / @as(f64, @floatFromInt(c1)));
    const exp_1pct: f32 = if (a1 > 0.001) 1000.0 / a1 else 0;
    const c01 = @max(1, m / 1000);
    var s01: f64 = 0;
    for (valid[m - c01 ..]) |t| s01 += t;
    const a01: f32 = @floatCast(s01 / @as(f64, @floatFromInt(c01)));
    const exp_01pct: f32 = if (a01 > 0.001) 1000.0 / a01 else 0;

    try std.testing.expectEqual(exp_avg, summary.avg_interval_ms);
    try std.testing.expectEqual(exp_fps, summary.observed_avg_fps);
    try std.testing.expectEqual(valid[p50_idx], summary.p50_interval_ms);
    try std.testing.expectEqual(valid[p99_idx], summary.p99_interval_ms);
    try std.testing.expectEqual(max_iv, summary.max_interval_ms);
    try std.testing.expectEqual(exp_1pct, summary.observed_fps_1pct_low);
    try std.testing.expectEqual(exp_01pct, summary.observed_fps_01pct_low);
    try std.testing.expectEqual(eh16, summary.interval_hitches_over_16ms);
    try std.testing.expectEqual(eh33, summary.interval_hitches_over_33ms);
    try std.testing.expectEqual(eh50, summary.interval_hitches_over_50ms);
}
