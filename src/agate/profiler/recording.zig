//! Per-frame profiler recording. Split out of `profiler.zig` (facade).
//!
//! `recordFrame` takes the profiler as `anytype` (a `*Profiler` from
//! `core.zig` in practice) so this module never imports `core.zig` or the
//! facade back — same discipline as `ui/*` taking a generic canvas.
//! `core.zig` owns the `Profiler` type and forwards `recordFrame` here.

const std = @import("std");
const sokol = @import("sokol");

const types = @import("types.zig");
const stats_mod = @import("../scene/stats.zig");

const FrameRecord = types.FrameRecord;
const SceneStats = stats_mod.SceneStats;

/// Records metrics for the completed frame. Called at the end of Scene.render().
pub fn recordFrame(self: anytype, frame_id: u64, stats: *const SceneStats) void {
    if (!self.is_recording) return;

    const now = sokol.time.now();
    const dt_s: f32 = if (self.last_frame_ticks > 0)
        @floatCast(sokol.time.sec(sokol.time.diff(now, self.last_frame_ticks)))
    else
        0.0166;
    self.last_frame_ticks = now;

    const total_ms = stats.update_ms + stats.prepare_ms + stats.shadow_ms + stats.main_ms + stats.post_ms;
    // Observed wall-clock pacing: fps derives from the real frame interval.
    // total_ms is the CPU-submit sum (timers around sg submit calls), not wall/GPU time.
    const frame_interval_ms: f32 = dt_s * 1000.0;
    const fps: f32 = if (dt_s > 0.0001) 1.0 / dt_s else (if (total_ms > 0.001) 1000.0 / total_ms else 60.0);
    const rel_us: u64 = if (self.start_time_ticks > 0)
        @intFromFloat(sokol.time.us(sokol.time.diff(now, self.start_time_ticks)))
    else
        0;

    const rec: FrameRecord = .{
        .frame_index = frame_id,
        .timestamp_us = rel_us,
        .dt_s = dt_s,
        .fps = fps,
        .frame_interval_ms = frame_interval_ms,
        .total_frame_ms = total_ms,
        .update_ms = stats.update_ms,
        .prepare_ms = stats.prepare_ms,
        .shadow_ms = stats.shadow_ms,
        .main_ms = stats.main_ms,
        .post_ms = stats.post_ms,
        .gpu_frame_ms = stats.gpu_frame_ms,
        .gpu_shadow_ms = stats.gpu_shadow_ms,
        .gpu_main_ms = stats.gpu_main_ms,
        .gpu_post_ms = stats.gpu_post_ms,
        .draw_calls = stats.draw_calls,
        .triangles = stats.triangles,
        .pipeline_switches = stats.pipeline_switches,
        .rendered_meshes = stats.rendered_meshes,
        .culled_objects = stats.culled_meshes + stats.occluded_meshes,
        .uploaded_textures = stats.uploaded_textures_frame,
        .uploaded_bytes = @intCast(stats.uploaded_bytes_frame),
        .updated_bytes = @intCast(stats.updated_bytes_frame),
    };

    if (self.frames.items.len >= self.max_frames) {
        _ = self.frames.orderedRemove(0);
    }
    self.frames.append(self.allocator, rec) catch return;
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

test "Profiler переносит динамику буферов отдельно от текстур" {
    const Profiler = @import("core.zig").Profiler;
    sokol.time.setup();
    const ally = std.testing.allocator;
    var prof = Profiler.init(ally);
    defer prof.deinit();
    prof.start();
    defer prof.stop();

    // Кадр 1: только текстуры; кадр 2: только динамика — метрики не смешиваются.
    var stats: SceneStats = .{
        .update_ms = 1.0,
        .prepare_ms = 1.0,
        .shadow_ms = 1.0,
        .main_ms = 5.0,
        .post_ms = 1.0,
        .uploaded_bytes_frame = 1024,
        .updated_bytes_frame = 0,
    };
    prof.recordFrame(1, &stats);
    stats.uploaded_bytes_frame = 0;
    stats.updated_bytes_frame = 2048;
    prof.recordFrame(2, &stats);

    try std.testing.expectEqual(@as(usize, 1024), prof.frames.items[0].uploaded_bytes);
    try std.testing.expectEqual(@as(usize, 0), prof.frames.items[0].updated_bytes);
    try std.testing.expectEqual(@as(usize, 0), prof.frames.items[1].uploaded_bytes);
    try std.testing.expectEqual(@as(usize, 2048), prof.frames.items[1].updated_bytes);

    const summary = prof.summarize();
    try std.testing.expectEqual(@as(usize, 1024), summary.total_uploaded_bytes);
    try std.testing.expectEqual(@as(usize, 2048), summary.total_updated_bytes);
}

test "Profiler переносит измеренные per-pass GPU-времена" {
    const Profiler = @import("core.zig").Profiler;
    sokol.time.setup();
    const ally = std.testing.allocator;
    var prof = Profiler.init(ally);
    defer prof.deinit();
    prof.start();
    defer prof.stop();

    // Disabled path: zeros flow through without touching CPU-submit stats.
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
    prof.recordFrame(0, &stats);
    stats.gpu_shadow_ms = 0.5;
    stats.gpu_main_ms = 8.0;
    stats.gpu_post_ms = 1.0;
    stats.gpu_frame_ms = 9.5;
    prof.recordFrame(1, &stats);

    try std.testing.expectEqual(@as(f32, 0), prof.frames.items[0].gpu_shadow_ms);
    try std.testing.expectEqual(@as(f32, 0.5), prof.frames.items[1].gpu_shadow_ms);
    try std.testing.expectEqual(@as(f32, 8.0), prof.frames.items[1].gpu_main_ms);
    try std.testing.expectEqual(@as(f32, 1.0), prof.frames.items[1].gpu_post_ms);

    const summary = prof.summarize();
    try std.testing.expectEqual(@as(f32, 0.25), summary.avg_gpu_shadow_ms);
    try std.testing.expectEqual(@as(f32, 0.5), summary.max_gpu_shadow_ms);
    try std.testing.expectEqual(@as(f32, 4.0), summary.avg_gpu_main_ms);
    try std.testing.expectEqual(@as(f32, 8.0), summary.max_gpu_main_ms);
    try std.testing.expectEqual(@as(f32, 0.5), summary.avg_gpu_post_ms);
    try std.testing.expectEqual(@as(f32, 1.0), summary.max_gpu_post_ms);
    // CPU-submit stats are untouched by the GPU fields.
    try std.testing.expectEqual(@as(f32, 9.5), summary.avg_frame_ms);
}
