const std = @import("std");
const types = @import("types.zig");
const report = @import("report.zig");

const FrameRecord = types.FrameRecord;
const SessionSummary = types.SessionSummary;
const DiagnosticFinding = types.DiagnosticFinding;
const generateReportHtml = report.generateReportHtml;
const generateReportMd = report.generateReportMd;
const generateTraceJson = report.generateTraceJson;
const hasGpuData = report.hasGpuData;
const hasGpuPassData = report.hasGpuPassData;

test "Profiler report HTML, MD, and JSON generation" {
    const ally = std.testing.allocator;
    const frames = [_]FrameRecord{
        .{
            .frame_index = 1,
            .timestamp_us = 0,
            .dt_s = 0.011,
            .fps = 90.9,
            .frame_interval_ms = 11.0,
            .total_frame_ms = 11.0,
            .update_ms = 1.5,
            .prepare_ms = 0.5,
            .shadow_ms = 2.0,
            .main_ms = 6.0,
            .post_ms = 1.0,
            .draw_calls = 25,
            .triangles = 5000,
            .pipeline_switches = 2,
        },
        .{
            .frame_index = 2,
            .timestamp_us = 11000,
            .dt_s = 0.011,
            .fps = 90.9,
            .frame_interval_ms = 11.0,
            .total_frame_ms = 11.0,
            .update_ms = 1.5,
            .prepare_ms = 0.5,
            .shadow_ms = 2.0,
            .main_ms = 6.0,
            .post_ms = 1.0,
            .draw_calls = 25,
            .triangles = 5000,
            .pipeline_switches = 2,
        },
    };
    const summary: SessionSummary = .{
        .frame_count = 2,
        .total_time_ms = 22.0,
        .avg_fps = 90.9,
        .fps_1pct_low = 90.9,
        .fps_01pct_low = 90.9,
        .min_frame_ms = 11.0,
        .avg_frame_ms = 11.0,
        .max_frame_ms = 11.0,
        .p50_frame_ms = 11.0,
        .p95_frame_ms = 11.0,
        .p99_frame_ms = 11.0,
        .avg_update_ms = 1.5,
        .avg_prepare_ms = 0.5,
        .avg_shadow_ms = 2.0,
        .avg_main_ms = 6.0,
        .avg_post_ms = 1.0,
        .avg_draw_calls = 25,
        .max_draw_calls = 25,
        .avg_triangles = 5000,
        .max_triangles = 5000,
        .avg_pipeline_switches = 2,
        .avg_interval_ms = 11.0,
        .p50_interval_ms = 11.0,
        .p99_interval_ms = 11.0,
        .max_interval_ms = 11.0,
        .observed_avg_fps = 90.9,
        .observed_fps_1pct_low = 90.9,
        .observed_fps_01pct_low = 90.9,
    };
    const findings = [_]DiagnosticFinding{
        .{
            .severity = .good,
            .title = "All key metrics normal",
            .details = "No critical hitches, memory bloat, or excessive draw calls detected.",
            .recommendation = "The current scene configuration is performing within expected budgets.",
        },
    };

    // HTML
    const html = try generateReportHtml(&frames, summary, &findings, null, ally);
    defer ally.free(html);
    try std.testing.expect(std.mem.indexOf(u8, html, "<!DOCTYPE html>") != null);
    try std.testing.expect(std.mem.indexOf(u8, html, "Performance & Memory Profile Report") != null);

    // MD
    const md = try generateReportMd(&frames, summary, &findings, null, ally);
    defer ally.free(md);
    try std.testing.expect(std.mem.indexOf(u8, md, "# Agate Engine - Performance & Memory Profile Report") != null);

    // JSON
    const json = try generateTraceJson(&frames, ally);
    defer ally.free(json);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"traceEvents\":") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"Frame #1 (CPU submit)\"") != null);
    // Submit times must never be presented as GPU time.
    try std.testing.expect(std.mem.indexOf(u8, json, "\"cat\": \"gpu\"") == null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"cat\": \"cpu\"") != null);

    // Reports must label CPU-submit vs wall-clock pacing explicitly.
    try std.testing.expect(std.mem.indexOf(u8, html, "wall") != null);
    try std.testing.expect(std.mem.indexOf(u8, html, "CPU-submit") != null);
    try std.testing.expect(std.mem.indexOf(u8, md, "wall") != null);
    try std.testing.expect(std.mem.indexOf(u8, md, "CPU-submit") != null);

    // Disabled GPU path (all zeros): no GPU-named output anywhere — the
    // reports stay byte-identical to the pre-GPU generators.
    try std.testing.expect(!hasGpuData(&frames));
    try std.testing.expect(!hasGpuPassData(&frames));
    try std.testing.expect(std.mem.indexOf(u8, html, "gpu_frame_ms") == null);
    try std.testing.expect(std.mem.indexOf(u8, html, "GPU Frame (measured)") == null);
    try std.testing.expect(std.mem.indexOf(u8, html, "GPU Shadow (measured)") == null);
    try std.testing.expect(std.mem.indexOf(u8, html, "Per-pass avg") == null);
    try std.testing.expect(std.mem.indexOf(u8, md, "gpu_frame_ms") == null);
    try std.testing.expect(std.mem.indexOf(u8, md, "GPU frame") == null);
    try std.testing.expect(std.mem.indexOf(u8, md, "per-pass") == null);
    try std.testing.expect(std.mem.indexOf(u8, json, "gpu_frame_ms") == null);
    try std.testing.expect(std.mem.indexOf(u8, json, "GPU Shadow (measured)") == null);

    // Enabled path: measured GPU samples are exported under explicit GPU
    // names, never relabeled as CPU-submit phases.
    var gpu_frames = frames;
    gpu_frames[0].gpu_frame_ms = 2.5;
    gpu_frames[0].gpu_frame_submit = 51;
    gpu_frames[0].gpu_frame_scope = .command_buffer;
    // Last frame intentionally left without a sample: exercises the
    // no-GPU-event tail (PostFX of the final frame must carry no
    // trailing comma).
    try std.testing.expect(hasGpuData(&gpu_frames));

    var gpu_summary = summary;
    gpu_summary.avg_gpu_frame_ms = 1.25;
    gpu_summary.max_gpu_frame_ms = 2.5;

    const gpu_html = try generateReportHtml(&gpu_frames, gpu_summary, &findings, null, ally);
    defer ally.free(gpu_html);
    try std.testing.expect(std.mem.indexOf(u8, gpu_html, "GPU Frame (measured)") != null);
    try std.testing.expect(std.mem.indexOf(u8, gpu_html, "GPU (measured)") != null);

    const gpu_md = try generateReportMd(&gpu_frames, gpu_summary, &findings, null, ally);
    defer ally.free(gpu_md);
    try std.testing.expect(std.mem.indexOf(u8, gpu_md, "GPU frame") != null);
    try std.testing.expect(std.mem.indexOf(u8, gpu_md, "gpu_frame_ms") == null); // human prose, no raw field names

    const gpu_json = try generateTraceJson(&gpu_frames, ally);
    defer ally.free(gpu_json);
    try std.testing.expect(std.mem.indexOf(u8, gpu_json, "\"cat\": \"gpu\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, gpu_json, "\"name\": \"GPU Frame (measured)\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, gpu_json, "\"gpu_frame_ms\": 2.500") != null);
    // Trace args carry the submission identity and backend scope.
    try std.testing.expect(std.mem.indexOf(u8, gpu_json, "\"gpu_frame_submit\": 51") != null);
    try std.testing.expect(std.mem.indexOf(u8, gpu_json, "\"gpu_frame_scope\": \"command_buffer\"") != null);
    // CPU slices keep their submit labels; the stream has no trailing comma.
    try std.testing.expect(std.mem.indexOf(u8, gpu_json, "\"Main Pass (CPU submit)\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, gpu_json, ",\n  ]") == null);
    // Frame-scope-only session (no per-pass samples): no per-pass output anywhere.
    try std.testing.expect(!hasGpuPassData(&gpu_frames));
    try std.testing.expect(std.mem.indexOf(u8, gpu_html, "Per-pass avg") == null);
    try std.testing.expect(std.mem.indexOf(u8, gpu_html, "GPU shadow/main/post") == null);
    try std.testing.expect(std.mem.indexOf(u8, gpu_md, "per-pass") == null);
    try std.testing.expect(std.mem.indexOf(u8, gpu_json, "GPU Shadow (measured)") == null);

    // Per-pass session: frame + per-pass samples flow through
    // every export under explicit GPU names.
    var pass_frames = frames;
    pass_frames[0].gpu_frame_ms = 9.5;
    pass_frames[0].gpu_frame_submit = 61;
    pass_frames[0].gpu_frame_scope = .pass_sum;
    pass_frames[0].gpu_shadow_ms = 0.5;
    pass_frames[0].gpu_shadow_submit = 62;
    pass_frames[0].gpu_main_ms = 8.0;
    pass_frames[0].gpu_main_submit = 63;
    pass_frames[0].gpu_post_ms = 1.0;
    pass_frames[0].gpu_post_submit = 64;
    // Second frame carries only a newer main sample: exercises per-frame
    // gating (no shadow/post events) and the no-GPU-event tail stays
    // on the LAST frame only when it has no samples at all.
    pass_frames[1].gpu_main_ms = 7.0;
    pass_frames[1].gpu_main_submit = 65;
    try std.testing.expect(hasGpuPassData(&pass_frames));

    var pass_summary = summary;
    pass_summary.avg_gpu_frame_ms = 4.75;
    pass_summary.max_gpu_frame_ms = 9.5;
    pass_summary.avg_gpu_shadow_ms = 0.25;
    pass_summary.max_gpu_shadow_ms = 0.5;
    pass_summary.avg_gpu_main_ms = 7.5;
    pass_summary.max_gpu_main_ms = 8.0;
    pass_summary.avg_gpu_post_ms = 0.5;
    pass_summary.max_gpu_post_ms = 1.0;

    const pass_html = try generateReportHtml(&pass_frames, pass_summary, &findings, null, ally);
    defer ally.free(pass_html);
    try std.testing.expect(std.mem.indexOf(u8, pass_html, "Per-pass avg (available distinct samples)") != null);
    try std.testing.expect(std.mem.indexOf(u8, pass_html, "GPU shadow/main/post (measured)") != null);

    const pass_md = try generateReportMd(&pass_frames, pass_summary, &findings, null, ally);
    defer ally.free(pass_md);
    try std.testing.expect(std.mem.indexOf(u8, pass_md, "per-pass") != null);
    // Still no raw field names in the human prose (frame-only rule holds
    // for the new fields too).
    try std.testing.expect(std.mem.indexOf(u8, pass_md, "gpu_shadow_ms") == null);
    try std.testing.expect(std.mem.indexOf(u8, pass_md, "gpu_main_ms") == null);
    try std.testing.expect(std.mem.indexOf(u8, pass_md, "gpu_post_ms") == null);

    const pass_json = try generateTraceJson(&pass_frames, ally);
    defer ally.free(pass_json);
    try std.testing.expect(std.mem.indexOf(u8, pass_json, "\"name\": \"GPU Shadow (measured)\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, pass_json, "\"name\": \"GPU Main (measured)\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, pass_json, "\"name\": \"GPU PostFX (measured)\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, pass_json, "\"gpu_shadow_ms\": 0.500") != null);
    try std.testing.expect(std.mem.indexOf(u8, pass_json, "\"gpu_main_ms\": 8.000") != null);
    try std.testing.expect(std.mem.indexOf(u8, pass_json, "\"gpu_post_ms\": 1.000") != null);
    // CPU-submit labels are untouched by the new GPU events.
    try std.testing.expect(std.mem.indexOf(u8, pass_json, "\"Shadow Pass (CPU submit)\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, pass_json, "\"PostFX (CPU submit)\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, pass_json, ",\n  ]") == null);
    // Per-pass trace args carry submission ids; the frame event carries
    // the frame submission id and scope.
    try std.testing.expect(std.mem.indexOf(u8, pass_json, "\"submit\": 62") != null);
    try std.testing.expect(std.mem.indexOf(u8, pass_json, "\"gpu_frame_submit\": 61") != null);
    try std.testing.expect(std.mem.indexOf(u8, pass_json, "\"gpu_frame_scope\": \"pass_sum\"") != null);
}

test "Profiler reports treat valid GPU zeros as available and dedup repeats" {
    const ally = std.testing.allocator;
    // Availability is the submission id: a valid quantized zero (id != 0,
    // ms == 0) gates GPU output ON, while bare zeros stay CPU-only.
    var zero_frames = [_]FrameRecord{
        .{ .frame_index = 1, .timestamp_us = 0, .total_frame_ms = 5.0 },
        .{ .frame_index = 2, .timestamp_us = 5000, .total_frame_ms = 5.0 },
    };
    try std.testing.expect(!hasGpuData(&zero_frames));
    try std.testing.expect(!hasGpuPassData(&zero_frames));

    zero_frames[0].gpu_frame_submit = 71;
    zero_frames[0].gpu_frame_scope = .native_pass_span;
    zero_frames[0].gpu_main_submit = 72; // ms stays 0: valid zero
    // Same submissions re-polled on the next CPU frame (async: no newer
    // completion yet) — one measurement, not two.
    zero_frames[1].gpu_frame_ms = 0;
    zero_frames[1].gpu_frame_submit = 71;
    zero_frames[1].gpu_frame_scope = .native_pass_span;
    zero_frames[1].gpu_main_ms = 0;
    zero_frames[1].gpu_main_submit = 72;
    try std.testing.expect(hasGpuData(&zero_frames));
    try std.testing.expect(hasGpuPassData(&zero_frames));

    const findings = [_]DiagnosticFinding{};
    const summary: SessionSummary = .{
        .frame_count = 2,
        .avg_gpu_frame_ms = 0,
        .max_gpu_frame_ms = 0,
        .gpu_frame_samples = 1,
        .avg_gpu_main_ms = 0,
        .max_gpu_main_ms = 0,
        .gpu_main_samples = 1,
    };
    const html = try generateReportHtml(&zero_frames, summary, &findings, null, ally);
    defer ally.free(html);
    try std.testing.expect(std.mem.indexOf(u8, html, "GPU Frame (measured)") != null);

    const json = try generateTraceJson(&zero_frames, ally);
    defer ally.free(json);
    // Exactly one GPU Frame event and one GPU Main event for the repeated
    // submission id — including the zero duration.
    var frame_events: usize = 0;
    var main_events: usize = 0;
    var cursor: usize = 0;
    while (std.mem.indexOfPos(u8, json, cursor, "\"name\": \"GPU Frame (measured)\"")) |pos| {
        frame_events += 1;
        cursor = pos + 1;
    }
    cursor = 0;
    while (std.mem.indexOfPos(u8, json, cursor, "\"name\": \"GPU Main (measured)\"")) |pos| {
        main_events += 1;
        cursor = pos + 1;
    }
    try std.testing.expectEqual(@as(usize, 1), frame_events);
    try std.testing.expectEqual(@as(usize, 1), main_events);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"gpu_frame_ms\": 0.000") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"gpu_frame_submit\": 71") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"gpu_frame_scope\": \"native_pass_span\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, ",\n  ]") == null);
}
