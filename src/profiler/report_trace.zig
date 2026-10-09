//! Chrome-trace JSON generation. Split out of `report.zig` (facade).
//!
//! Owns `generateTraceJson`. Imports std + `types` + the `report_gates`
//! availability predicate; no Scene import, so the report graph stays acyclic.

const std = @import("std");
const types = @import("types.zig");
const gates = @import("report_gates.zig");

const FrameRecord = types.FrameRecord;

const hasGpuData = gates.hasGpuData;

/// Generates Chrome Trace Event JSON for chrome://tracing and ui.perfetto.dev.
pub fn generateTraceJson(frames: []const FrameRecord, allocator: std.mem.Allocator) ![]u8 {
    var buf = std.ArrayListUnmanaged(u8).empty;
    errdefer buf.deinit(allocator);

    // GPU events (cat "gpu", tid 2) carry last-completed GPU samples;
    // the disabled path (no submission ids) emits exactly the pre-GPU
    // event stream. NOTE: every GPU event below is anchored to the CPU
    // observation time (the CPU frame start `ts`) — NOT to an absolute
    // GPU timestamp or a synchronized GPU/CPU timeline. Durations are the
    // completed-sample values; repeats of the same submission id across
    // consecutive CPU frames (async re-poll before a newer completion)
    // emit once, including valid zero-duration measurements.
    const has_gpu = hasGpuData(frames);

    try buf.appendSlice(allocator, "{\n  \"traceEvents\": [\n");

    var last_trace_frame_submit: u32 = 0;
    var last_trace_shadow_submit: u32 = 0;
    var last_trace_main_submit: u32 = 0;
    var last_trace_post_submit: u32 = 0;

    for (frames, 0..) |f, i| {
        const is_last_frame = (i + 1 == frames.len);
        const ts = f.timestamp_us;
        const dur_us: u64 = @intFromFloat(f.total_frame_ms * 1000.0);

        // Whether this CPU frame introduces a NEW completed submission
        // per channel (dedup: the same id on consecutive frames is one
        // measurement observed again, not a new event).
        const frame_new = f.gpu_frame_submit != 0 and f.gpu_frame_submit != last_trace_frame_submit;
        const shadow_new = f.gpu_shadow_submit != 0 and f.gpu_shadow_submit != last_trace_shadow_submit;
        const main_new = f.gpu_main_submit != 0 and f.gpu_main_submit != last_trace_main_submit;
        const post_new = f.gpu_post_submit != 0 and f.gpu_post_submit != last_trace_post_submit;
        if (f.gpu_frame_submit != 0) last_trace_frame_submit = f.gpu_frame_submit;
        if (f.gpu_shadow_submit != 0) last_trace_shadow_submit = f.gpu_shadow_submit;
        if (f.gpu_main_submit != 0) last_trace_main_submit = f.gpu_main_submit;
        if (f.gpu_post_submit != 0) last_trace_post_submit = f.gpu_post_submit;
        const pass_new_count: usize =
            (if (shadow_new) @as(usize, 1) else 0) +
            (if (main_new) @as(usize, 1) else 0) +
            (if (post_new) @as(usize, 1) else 0);

        // Complete event for whole frame. dur covers the CPU-submit span;
        // fps/interval are observed wall-clock pacing.
        const frame_event = if (has_gpu) try std.fmt.allocPrint(allocator,
            \\    {{"name": "Frame #{d} (CPU submit)", "cat": "frame", "ph": "X", "ts": {d}, "dur": {d}, "pid": 1, "tid": 1, "args": {{"fps_wall": {d:.1}, "frame_interval_ms": {d:.3}, "cpu_submit_ms": {d:.3}, "gpu_frame_ms": {d:.3}, "gpu_frame_submit": {d}, "gpu_frame_scope": "{s}", "draw_calls": {d}, "triangles": {d}, "switches": {d}}}}},
        , .{ f.frame_index, ts, dur_us, f.fps, f.frame_interval_ms, f.total_frame_ms, f.gpu_frame_ms, f.gpu_frame_submit, @tagName(f.gpu_frame_scope), f.draw_calls, f.triangles, f.pipeline_switches }) else try std.fmt.allocPrint(allocator,
            \\    {{"name": "Frame #{d} (CPU submit)", "cat": "frame", "ph": "X", "ts": {d}, "dur": {d}, "pid": 1, "tid": 1, "args": {{"fps_wall": {d:.1}, "frame_interval_ms": {d:.3}, "cpu_submit_ms": {d:.3}, "draw_calls": {d}, "triangles": {d}, "switches": {d}}}}},
        , .{ f.frame_index, ts, dur_us, f.fps, f.frame_interval_ms, f.total_frame_ms, f.draw_calls, f.triangles, f.pipeline_switches });
        defer allocator.free(frame_event);
        try buf.appendSlice(allocator, frame_event);
        try buf.appendSlice(allocator, "\n");

        // Sub-phase slices inside the frame (CPU-submit times, not GPU execution).
        // All phases use "cpu": Shadow/Main/Post measure CPU timers around
        // sg submit calls, never GPU timestamps.
        var cur_ts = ts;
        const u_us: u64 = @intFromFloat(f.update_ms * 1000.0);
        const phys_us: u64 = @intFromFloat(f.physics_ms * 1000.0);
        const p_us: u64 = @intFromFloat(f.prepare_ms * 1000.0);
        const s_us: u64 = @intFromFloat(f.shadow_ms * 1000.0);
        const m_us: u64 = @intFromFloat(f.main_ms * 1000.0);
        const post_us: u64 = @intFromFloat(f.post_ms * 1000.0);

        // Update
        const u_ev = try std.fmt.allocPrint(allocator,
            \\    {{"name": "Update", "cat": "cpu", "ph": "X", "ts": {d}, "dur": {d}, "pid": 1, "tid": 1}},
        , .{ cur_ts, u_us });
        defer allocator.free(u_ev);
        try buf.appendSlice(allocator, u_ev);
        try buf.appendSlice(allocator, "\n");
        cur_ts += u_us;

        // Physics
        if (phys_us > 0) {
            const phys_ev = try std.fmt.allocPrint(allocator,
                \\    {{"name": "Physics (Box3D)", "cat": "cpu", "ph": "X", "ts": {d}, "dur": {d}, "pid": 1, "tid": 1}},
            , .{ cur_ts, phys_us });
            defer allocator.free(phys_ev);
            try buf.appendSlice(allocator, phys_ev);
            try buf.appendSlice(allocator, "\n");
            cur_ts += phys_us;
        }

        // Prepare
        const p_ev = try std.fmt.allocPrint(allocator,
            \\    {{"name": "Prepare", "cat": "cpu", "ph": "X", "ts": {d}, "dur": {d}, "pid": 1, "tid": 1}},
        , .{ cur_ts, p_us });
        defer allocator.free(p_ev);
        try buf.appendSlice(allocator, p_ev);
        try buf.appendSlice(allocator, "\n");
        cur_ts += p_us;

        // Shadow
        const s_ev = try std.fmt.allocPrint(allocator,
            \\    {{"name": "Shadow Pass (CPU submit)", "cat": "cpu", "ph": "X", "ts": {d}, "dur": {d}, "pid": 1, "tid": 1}},
        , .{ cur_ts, s_us });
        defer allocator.free(s_ev);
        try buf.appendSlice(allocator, s_ev);
        try buf.appendSlice(allocator, "\n");
        cur_ts += s_us;

        // Main
        const m_ev = try std.fmt.allocPrint(allocator,
            \\    {{"name": "Main Pass (CPU submit)", "cat": "cpu", "ph": "X", "ts": {d}, "dur": {d}, "pid": 1, "tid": 1}},
        , .{ cur_ts, m_us });
        defer allocator.free(m_ev);
        try buf.appendSlice(allocator, m_ev);
        try buf.appendSlice(allocator, "\n");
        cur_ts += m_us;

        // PostFX. A measured-GPU event may follow on tid 2 (emitted once
        // per DISTINCT completed submission, zero durations included);
        // the comma below accounts for them so the stream stays valid
        // JSON in every combination.
        const gpu_emit = frame_new;
        const pass_names = [_][]const u8{ "GPU Shadow (measured)", "GPU Main (measured)", "GPU PostFX (measured)" };
        const pass_keys = [_][]const u8{ "gpu_shadow_ms", "gpu_main_ms", "gpu_post_ms" };
        const pass_vals = [_]f32{ f.gpu_shadow_ms, f.gpu_main_ms, f.gpu_post_ms };
        const pass_submits = [_]u32{ f.gpu_shadow_submit, f.gpu_main_submit, f.gpu_post_submit };
        const pass_news = [_]bool{ shadow_new, main_new, post_new };
        const trailing_count: usize = (if (gpu_emit) @as(usize, 1) else 0) + pass_new_count;
        const post_comma = if (trailing_count > 0 or !is_last_frame) "," else "";
        const post_ev = try std.fmt.allocPrint(allocator,
            \\    {{"name": "PostFX (CPU submit)", "cat": "cpu", "ph": "X", "ts": {d}, "dur": {d}, "pid": 1, "tid": 1}}{s}
        , .{ cur_ts, post_us, post_comma });
        defer allocator.free(post_ev);
        try buf.appendSlice(allocator, post_ev);
        try buf.appendSlice(allocator, "\n");

        // Measured GPU frame sample: separate thread id and "gpu" category
        // so it can never be mistaken for a CPU-submit slice. ts is the
        // CPU frame start (observation time, not a GPU timestamp); the
        // value is the newly completed submission's duration, and args
        // carry the submission id plus the backend scope.
        if (gpu_emit) {
            const gpu_us: u64 = @intFromFloat(f.gpu_frame_ms * 1000.0);
            const gpu_comma = if (pass_new_count > 0 or !is_last_frame) "," else "";
            const gpu_ev = try std.fmt.allocPrint(allocator,
                \\    {{"name": "GPU Frame (measured)", "cat": "gpu", "ph": "X", "ts": {d}, "dur": {d}, "pid": 1, "tid": 2, "args": {{"gpu_frame_ms": {d:.3}, "gpu_frame_submit": {d}, "gpu_frame_scope": "{s}"}}}}{s}
            , .{ ts, gpu_us, f.gpu_frame_ms, f.gpu_frame_submit, @tagName(f.gpu_frame_scope), gpu_comma });
            defer allocator.free(gpu_ev);
            try buf.appendSlice(allocator, gpu_ev);
            try buf.appendSlice(allocator, "\n");
        }

        // Measured per-pass GPU samples (tid 2 alongside the frame event):
        // one event per phase with a NEW completed submission (zero
        // durations included), so the disabled path and frame-scope-only
        // sessions emit exactly the pre-per-pass stream. ts is the CPU
        // frame start (observation time); each value is that submission's
        // duration, and args carry its submission id. Channels are
        // independent — values from different submission ids are never
        // combined.
        var pass_done: usize = 0;
        for (pass_names, pass_keys, pass_vals, pass_submits, pass_news) |pname, pkey, pval, psubmit, pnew| {
            if (!pnew) continue;
            pass_done += 1;
            const pass_comma = if (is_last_frame and pass_done == pass_new_count) "" else ",";
            const pass_us: u64 = @intFromFloat(pval * 1000.0);
            const pass_ev = try std.fmt.allocPrint(allocator,
                \\    {{"name": "{s}", "cat": "gpu", "ph": "X", "ts": {d}, "dur": {d}, "pid": 1, "tid": 2, "args": {{"{s}": {d:.3}, "submit": {d}}}}}{s}
            , .{ pname, ts, pass_us, pkey, pval, psubmit, pass_comma });
            defer allocator.free(pass_ev);
            try buf.appendSlice(allocator, pass_ev);
            try buf.appendSlice(allocator, "\n");
        }
    }

    try buf.appendSlice(allocator, "  ],\n  \"displayTimeUnit\": \"ms\"\n}\n");
    return buf.toOwnedSlice(allocator);
}
