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

    const total_ms = stats.update_ms + stats.physics_ms + stats.prepare_ms + stats.shadow_ms + stats.main_ms + stats.post_ms;
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
        .physics_ms = stats.physics_ms,
        .prepare_ms = stats.prepare_ms,
        .shadow_ms = stats.shadow_ms,
        .main_ms = stats.main_ms,
        .post_ms = stats.post_ms,
        .gpu_frame_ms = stats.gpu_frame_ms,
        .gpu_frame_submit = stats.gpu_frame_submit,
        .gpu_frame_scope = stats.gpu_frame_scope,
        .gpu_shadow_ms = stats.gpu_shadow_ms,
        .gpu_shadow_submit = stats.gpu_shadow_submit,
        .gpu_main_ms = stats.gpu_main_ms,
        .gpu_main_submit = stats.gpu_main_submit,
        .gpu_post_ms = stats.gpu_post_ms,
        .gpu_post_submit = stats.gpu_post_submit,
        .draw_calls = stats.draw_calls,
        .triangles = stats.triangles,
        .pipeline_switches = stats.pipeline_switches,
        .rendered_meshes = stats.rendered_meshes,
        .culled_objects = stats.culled_meshes + stats.occluded_meshes,
        .uploaded_textures = stats.uploaded_textures_frame,
        .uploaded_bytes = @intCast(stats.uploaded_bytes_frame),
        .updated_bytes = @intCast(stats.updated_bytes_frame),
    };

    // O(1) ring-buffer insert: avoid O(N) orderedRemove memmove across thousands of frames.
    if (!self.wrapped) {
        if (self.frames.items.len < self.max_frames) {
            self.frames.append(self.allocator, rec) catch return;
            if (self.frames.items.len == self.max_frames) {
                self.ring_head = 0;
            }
        } else {
            self.wrapped = true;
            self.frames.items[self.ring_head] = rec;
            self.ring_head = (self.ring_head + 1) % self.max_frames;
        }
    } else {
        self.frames.items[self.ring_head] = rec;
        self.ring_head = (self.ring_head + 1) % self.max_frames;
    }
}
