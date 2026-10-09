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
