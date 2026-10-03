//! Profiler session statistics. Split out of `profiler.zig` (facade).
//!
//! `summarize` takes the profiler as `anytype` (a `*const Profiler` from
//! `core.zig` in practice) so this module never imports `core.zig` or the
//! facade back — same discipline as `ui/*` taking a generic canvas.
//! `core.zig` owns the `Profiler` type and forwards `summarize` here.
//! Leaf: imports `types.zig` only.

const std = @import("std");

const types = @import("types.zig");

const FrameRecord = types.FrameRecord;
const SessionSummary = types.SessionSummary;

/// Computes statistical summary across all recorded frames.
/// CPU-submit fields come from `total_frame_ms`; pacing fields come from
/// wall-clock `frame_interval_ms` (frames with dt_s <= 0 are skipped).
pub fn summarize(self: anytype) SessionSummary {
    const n = self.frames.items.len;
    if (n == 0) return .{};

    var sum_time: f64 = 0;
    var sum_update: f64 = 0;
    var sum_physics: f64 = 0;
    var max_physics: f32 = 0;
    var sum_prepare: f64 = 0;
    var sum_shadow: f64 = 0;
    var sum_main: f64 = 0;
    var sum_post: f64 = 0;
    var sum_gpu: f64 = 0;
    var max_gpu: f32 = 0;
    var count_gpu: usize = 0;
    var last_gpu_submit: u32 = 0;
    var sum_gpu_shadow: f64 = 0;
    var max_gpu_shadow: f32 = 0;
    var count_gpu_shadow: usize = 0;
    var last_gpu_shadow_submit: u32 = 0;
    var sum_gpu_main: f64 = 0;
    var max_gpu_main: f32 = 0;
    var count_gpu_main: usize = 0;
    var last_gpu_main_submit: u32 = 0;
    var sum_gpu_post: f64 = 0;
    var max_gpu_post: f32 = 0;
    var count_gpu_post: usize = 0;
    var last_gpu_post_submit: u32 = 0;

    var sum_draw_calls: u64 = 0;
    var max_draw_calls: u32 = 0;
    var sum_triangles: u64 = 0;
    var max_triangles: u32 = 0;
    var sum_switches: u64 = 0;
    var sum_uploaded_bytes: usize = 0;
    var sum_updated_bytes: usize = 0;

    var hitches_16: u32 = 0;
    var hitches_33: u32 = 0;
    var hitches_50: u32 = 0;

    // Temporary slice to sort frame times for percentiles
    var times = self.allocator.alloc(f32, n) catch return .{};
    defer self.allocator.free(times);

    // Wall-clock intervals for real frame pacing (subset of frames).
    var intervals = self.allocator.alloc(f32, n) catch return .{};
    defer self.allocator.free(intervals);
    var interval_count: usize = 0;
    var sum_interval: f64 = 0;
    var max_interval: f32 = 0;
    var interval_hitches_16: u32 = 0;
    var interval_hitches_33: u32 = 0;
    var interval_hitches_50: u32 = 0;

    var min_ms: f32 = std.math.floatMax(f32);
    var max_ms: f32 = 0;

    for (self.frames.items, 0..) |frame, i| {
        const ms = frame.total_frame_ms;
        times[i] = ms;
        sum_time += ms;
        min_ms = @min(min_ms, ms);
        max_ms = @max(max_ms, ms);

        if (frame.dt_s > 0) {
            const iv_ms = frame.frame_interval_ms;
            intervals[interval_count] = iv_ms;
            interval_count += 1;
            sum_interval += iv_ms;
            max_interval = @max(max_interval, iv_ms);
            if (iv_ms > 50.0) interval_hitches_50 += 1;
            if (iv_ms > 33.33) interval_hitches_33 += 1;
            if (iv_ms > 16.67) interval_hitches_16 += 1;
        }

        sum_update += frame.update_ms;
        sum_physics += frame.physics_ms;
        max_physics = @max(max_physics, frame.physics_ms);
        sum_prepare += frame.prepare_ms;
        sum_shadow += frame.shadow_ms;
        sum_main += frame.main_ms;
        sum_post += frame.post_ms;
        // GPU channels: average AVAILABLE DISTINCT completed submissions
        // only — never divide by the CPU frame count. Availability is the
        // submission id (0 = absent), so valid quantized zeros (id != 0,
        // ms == 0) are real measurements. Repeated async polls of the same
        // submission surface the same id on consecutive CPU frames; count
        // each id once (consecutive-identity dedup suffices: reads are
        // ordered last-completed polls, never an interleaved set).
        if (frame.gpu_frame_submit != 0 and frame.gpu_frame_submit != last_gpu_submit) {
            last_gpu_submit = frame.gpu_frame_submit;
            sum_gpu += frame.gpu_frame_ms;
            max_gpu = @max(max_gpu, frame.gpu_frame_ms);
            count_gpu += 1;
        }
        if (frame.gpu_shadow_submit != 0 and frame.gpu_shadow_submit != last_gpu_shadow_submit) {
            last_gpu_shadow_submit = frame.gpu_shadow_submit;
            sum_gpu_shadow += frame.gpu_shadow_ms;
            max_gpu_shadow = @max(max_gpu_shadow, frame.gpu_shadow_ms);
            count_gpu_shadow += 1;
        }
        if (frame.gpu_main_submit != 0 and frame.gpu_main_submit != last_gpu_main_submit) {
            last_gpu_main_submit = frame.gpu_main_submit;
            sum_gpu_main += frame.gpu_main_ms;
            max_gpu_main = @max(max_gpu_main, frame.gpu_main_ms);
            count_gpu_main += 1;
        }
        if (frame.gpu_post_submit != 0 and frame.gpu_post_submit != last_gpu_post_submit) {
            last_gpu_post_submit = frame.gpu_post_submit;
            sum_gpu_post += frame.gpu_post_ms;
            max_gpu_post = @max(max_gpu_post, frame.gpu_post_ms);
            count_gpu_post += 1;
        }

        sum_draw_calls += frame.draw_calls;
        max_draw_calls = @max(max_draw_calls, frame.draw_calls);
        sum_triangles += frame.triangles;
        max_triangles = @max(max_triangles, frame.triangles);
        sum_switches += frame.pipeline_switches;
        sum_uploaded_bytes += frame.uploaded_bytes;
        sum_updated_bytes += frame.updated_bytes;

        if (ms > 50.0) {
            hitches_50 += 1;
        }
        if (ms > 33.33) {
            hitches_33 += 1;
        }
        if (ms > 16.67) {
            hitches_16 += 1;
        }
    }

    std.mem.sort(f32, times, {}, struct {
        fn lessThan(_: void, a: f32, b: f32) bool {
            return a < b;
        }
    }.lessThan);

    const avg_ms: f32 = @floatCast(sum_time / @as(f64, @floatFromInt(n)));
    const avg_fps: f32 = if (avg_ms > 0.001) 1000.0 / avg_ms else 0;

    // Percentiles
    const p50_idx = @min(n - 1, @as(usize, @intFromFloat(@as(f32, @floatFromInt(n)) * 0.50)));
    const p95_idx = @min(n - 1, @as(usize, @intFromFloat(@as(f32, @floatFromInt(n)) * 0.95)));
    const p99_idx = @min(n - 1, @as(usize, @intFromFloat(@as(f32, @floatFromInt(n)) * 0.99)));

    // 1% low: average of worst 1% frames
    const count_1pct = @max(1, n / 100);
    var sum_1pct: f64 = 0;
    for (times[n - count_1pct ..]) |t| sum_1pct += t;
    const avg_1pct_ms: f32 = @floatCast(sum_1pct / @as(f64, @floatFromInt(count_1pct)));
    const fps_1pct_low: f32 = if (avg_1pct_ms > 0.001) 1000.0 / avg_1pct_ms else 0;

    // 0.1% low: average of worst 0.1% frames
    const count_01pct = @max(1, n / 1000);
    var sum_01pct: f64 = 0;
    for (times[n - count_01pct ..]) |t| sum_01pct += t;
    const avg_01pct_ms: f32 = @floatCast(sum_01pct / @as(f64, @floatFromInt(count_01pct)));
    const fps_01pct_low: f32 = if (avg_01pct_ms > 0.001) 1000.0 / avg_01pct_ms else 0;

    // Wall-clock pacing from observed frame intervals.
    var avg_interval_ms: f32 = 0;
    var p50_interval_ms: f32 = 0;
    var p99_interval_ms: f32 = 0;
    var observed_avg_fps: f32 = 0;
    var observed_fps_1pct_low: f32 = 0;
    var observed_fps_01pct_low: f32 = 0;
    if (interval_count > 0) {
        const valid = intervals[0..interval_count];
        std.mem.sort(f32, valid, {}, struct {
            fn lessThan(_: void, a: f32, b: f32) bool {
                return a < b;
            }
        }.lessThan);
        const m = interval_count;
        avg_interval_ms = @floatCast(sum_interval / @as(f64, @floatFromInt(m)));
        observed_avg_fps = if (avg_interval_ms > 0.001) 1000.0 / avg_interval_ms else 0;
        const p50_iv_idx = @min(m - 1, @as(usize, @intFromFloat(@as(f32, @floatFromInt(m)) * 0.50)));
        const p99_iv_idx = @min(m - 1, @as(usize, @intFromFloat(@as(f32, @floatFromInt(m)) * 0.99)));
        p50_interval_ms = valid[p50_iv_idx];
        p99_interval_ms = valid[p99_iv_idx];

        const count_iv_1pct = @max(1, m / 100);
        var sum_iv_1pct: f64 = 0;
        for (valid[m - count_iv_1pct ..]) |t| sum_iv_1pct += t;
        const avg_iv_1pct_ms: f32 = @floatCast(sum_iv_1pct / @as(f64, @floatFromInt(count_iv_1pct)));
        observed_fps_1pct_low = if (avg_iv_1pct_ms > 0.001) 1000.0 / avg_iv_1pct_ms else 0;

        const count_iv_01pct = @max(1, m / 1000);
        var sum_iv_01pct: f64 = 0;
        for (valid[m - count_iv_01pct ..]) |t| sum_iv_01pct += t;
        const avg_iv_01pct_ms: f32 = @floatCast(sum_iv_01pct / @as(f64, @floatFromInt(count_iv_01pct)));
        observed_fps_01pct_low = if (avg_iv_01pct_ms > 0.001) 1000.0 / avg_iv_01pct_ms else 0;
    }

    const nf = @as(f64, @floatFromInt(n));
    return .{
        .frame_count = n,
        .total_time_ms = sum_time,
        .avg_fps = avg_fps,
        .fps_1pct_low = fps_1pct_low,
        .fps_01pct_low = fps_01pct_low,
        .min_frame_ms = min_ms,
        .avg_frame_ms = avg_ms,
        .max_frame_ms = max_ms,
        .p50_frame_ms = times[p50_idx],
        .p95_frame_ms = times[p95_idx],
        .p99_frame_ms = times[p99_idx],
        .avg_update_ms = @floatCast(sum_update / nf),
        .avg_physics_ms = @floatCast(sum_physics / nf),
        .max_physics_ms = max_physics,
        .avg_prepare_ms = @floatCast(sum_prepare / nf),
        .avg_shadow_ms = @floatCast(sum_shadow / nf),
        .avg_main_ms = @floatCast(sum_main / nf),
        .avg_post_ms = @floatCast(sum_post / nf),
        .avg_gpu_frame_ms = if (count_gpu > 0) @floatCast(sum_gpu / @as(f64, @floatFromInt(count_gpu))) else 0,
        .max_gpu_frame_ms = max_gpu,
        .gpu_frame_samples = count_gpu,
        .avg_gpu_shadow_ms = if (count_gpu_shadow > 0) @floatCast(sum_gpu_shadow / @as(f64, @floatFromInt(count_gpu_shadow))) else 0,
        .max_gpu_shadow_ms = max_gpu_shadow,
        .gpu_shadow_samples = count_gpu_shadow,
        .avg_gpu_main_ms = if (count_gpu_main > 0) @floatCast(sum_gpu_main / @as(f64, @floatFromInt(count_gpu_main))) else 0,
        .max_gpu_main_ms = max_gpu_main,
        .gpu_main_samples = count_gpu_main,
        .avg_gpu_post_ms = if (count_gpu_post > 0) @floatCast(sum_gpu_post / @as(f64, @floatFromInt(count_gpu_post))) else 0,
        .max_gpu_post_ms = max_gpu_post,
        .gpu_post_samples = count_gpu_post,
        .avg_draw_calls = @intCast(sum_draw_calls / n),
        .max_draw_calls = max_draw_calls,
        .avg_triangles = @intCast(sum_triangles / n),
        .max_triangles = max_triangles,
        .avg_pipeline_switches = @intCast(sum_switches / n),
        .total_uploaded_bytes = sum_uploaded_bytes,
        .total_updated_bytes = sum_updated_bytes,
        .hitches_over_16ms = hitches_16,
        .hitches_over_33ms = hitches_33,
        .hitches_over_50ms = hitches_50,
        .avg_interval_ms = avg_interval_ms,
        .p50_interval_ms = p50_interval_ms,
        .p99_interval_ms = p99_interval_ms,
        .max_interval_ms = max_interval,
        .observed_avg_fps = observed_avg_fps,
        .observed_fps_1pct_low = observed_fps_1pct_low,
        .observed_fps_01pct_low = observed_fps_01pct_low,
        .interval_hitches_over_16ms = interval_hitches_16,
        .interval_hitches_over_33ms = interval_hitches_33,
        .interval_hitches_over_50ms = interval_hitches_50,
    };
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

/// Sleep helper for pacing-sensitive tests (Zig 0.16 has no Thread.sleep).
fn testSleepMs(ms: u64) void {
    const ts = std.c.timespec{ .sec = @intCast(ms / 1000), .nsec = @intCast((ms % 1000) * 1_000_000) };
    var rem: std.c.timespec = undefined;
    _ = std.c.nanosleep(&ts, &rem);
}

test "Profiler start, recordFrame, and summarize" {
    const Profiler = @import("core.zig").Profiler;
    const sokol = @import("sokol");
    const SceneStats = @import("../scene/stats.zig").SceneStats;
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
    const Profiler = @import("core.zig").Profiler;
    const sokol = @import("sokol");
    const SceneStats = @import("../scene/stats.zig").SceneStats;
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
    const Profiler = @import("core.zig").Profiler;
    const sokol = @import("sokol");
    const SceneStats = @import("../scene/stats.zig").SceneStats;
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
