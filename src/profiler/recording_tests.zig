const std = @import("std");
const sokol = @import("sokol");
const Profiler = @import("core.zig").Profiler;
const SceneStats = @import("../scene/stats.zig").SceneStats;

test "Profiler records buffer dynamic updates separately from textures" {
    sokol.time.setup();
    const ally = std.testing.allocator;
    var prof = Profiler.init(ally);
    defer prof.deinit();
    prof.start();
    defer prof.stop();

    // Frame 1: textures only; frame 2: buffer dynamics only.
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

test "Profiler records measured per-pass GPU times" {
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
    stats.gpu_shadow_submit = 41;
    stats.gpu_main_ms = 8.0;
    stats.gpu_main_submit = 42;
    stats.gpu_post_ms = 1.0;
    stats.gpu_post_submit = 43;
    stats.gpu_frame_ms = 9.5;
    stats.gpu_frame_submit = 44;
    stats.gpu_frame_scope = .pass_sum;
    prof.recordFrame(1, &stats);

    try std.testing.expectEqual(@as(f32, 0), prof.frames.items[0].gpu_shadow_ms);
    try std.testing.expectEqual(@as(u32, 0), prof.frames.items[0].gpu_shadow_submit);
    try std.testing.expectEqual(@as(f32, 0.5), prof.frames.items[1].gpu_shadow_ms);
    try std.testing.expectEqual(@as(f32, 8.0), prof.frames.items[1].gpu_main_ms);
    try std.testing.expectEqual(@as(f32, 1.0), prof.frames.items[1].gpu_post_ms);
    // Submission identity and scope flow through untouched.
    try std.testing.expectEqual(@as(u32, 41), prof.frames.items[1].gpu_shadow_submit);
    try std.testing.expectEqual(@as(u32, 42), prof.frames.items[1].gpu_main_submit);
    try std.testing.expectEqual(@as(u32, 43), prof.frames.items[1].gpu_post_submit);
    try std.testing.expectEqual(@as(u32, 44), prof.frames.items[1].gpu_frame_submit);

    const summary = prof.summarize();
    // Averages cover AVAILABLE DISTINCT submissions only (one per channel
    // here), not all recorded CPU frames — so full values, not halves.
    try std.testing.expectEqual(@as(f32, 0.5), summary.avg_gpu_shadow_ms);
    try std.testing.expectEqual(@as(f32, 0.5), summary.max_gpu_shadow_ms);
    try std.testing.expectEqual(@as(f32, 8.0), summary.avg_gpu_main_ms);
    try std.testing.expectEqual(@as(f32, 8.0), summary.max_gpu_main_ms);
    try std.testing.expectEqual(@as(f32, 1.0), summary.avg_gpu_post_ms);
    try std.testing.expectEqual(@as(f32, 1.0), summary.max_gpu_post_ms);
    try std.testing.expectEqual(@as(f32, 9.5), summary.avg_gpu_frame_ms);
    try std.testing.expectEqual(@as(f32, 9.5), summary.max_gpu_frame_ms);
    try std.testing.expectEqual(@as(usize, 1), summary.gpu_shadow_samples);
    try std.testing.expectEqual(@as(usize, 1), summary.gpu_main_samples);
    try std.testing.expectEqual(@as(usize, 1), summary.gpu_post_samples);
    try std.testing.expectEqual(@as(usize, 1), summary.gpu_frame_samples);
    // CPU-submit stats are untouched by the GPU fields.
    try std.testing.expectEqual(@as(f32, 9.5), summary.avg_frame_ms);
}

test "Profiler distinguishes unavailable GPU samples from valid zeros" {
    sokol.time.setup();
    const ally = std.testing.allocator;
    var prof = Profiler.init(ally);
    defer prof.deinit();
    prof.start();
    defer prof.stop();

    var stats: SceneStats = .{
        .update_ms = 1.0,
        .prepare_ms = 0.5,
        .shadow_ms = 1.0,
        .main_ms = 6.0,
        .post_ms = 1.0,
    };
    // Frame 0: timer off / phase skipped — absent (submit 0, duration 0).
    prof.recordFrame(0, &stats);
    // Frame 1: valid quantized zero — present (submit != 0, duration 0).
    stats.gpu_frame_ms = 0;
    stats.gpu_frame_submit = 7;
    stats.gpu_main_ms = 0;
    stats.gpu_main_submit = 8;
    prof.recordFrame(1, &stats);

    const summary = prof.summarize();
    // The valid zero is a real measurement: counted, averaged as 0.
    try std.testing.expectEqual(@as(usize, 1), summary.gpu_frame_samples);
    try std.testing.expectEqual(@as(f32, 0), summary.avg_gpu_frame_ms);
    try std.testing.expectEqual(@as(f32, 0), summary.max_gpu_frame_ms);
    try std.testing.expectEqual(@as(usize, 1), summary.gpu_main_samples);
    try std.testing.expectEqual(@as(f32, 0), summary.avg_gpu_main_ms);
    // Channels with no sample at all stay at 0 samples.
    try std.testing.expectEqual(@as(usize, 0), summary.gpu_shadow_samples);
    try std.testing.expectEqual(@as(usize, 0), summary.gpu_post_samples);
    try std.testing.expectEqual(@as(f32, 0), summary.avg_gpu_shadow_ms);
}

test "Profiler dedups repeated async polls of the same GPU submission" {
    sokol.time.setup();
    const ally = std.testing.allocator;
    var prof = Profiler.init(ally);
    defer prof.deinit();
    prof.start();
    defer prof.stop();

    var stats: SceneStats = .{
        .update_ms = 1.0,
        .prepare_ms = 0.5,
        .shadow_ms = 1.0,
        .main_ms = 6.0,
        .post_ms = 1.0,
        .gpu_frame_ms = 3.0,
        .gpu_frame_submit = 21,
        .gpu_main_ms = 2.0,
        .gpu_main_submit = 22,
    };
    // Same completed submission observed on three consecutive CPU frames
    // (async GPU execution hasn't completed a newer one yet).
    prof.recordFrame(0, &stats);
    prof.recordFrame(1, &stats);
    prof.recordFrame(2, &stats);
    // Then a newer distinct submission — including a valid zero — counts again.
    stats.gpu_frame_ms = 0;
    stats.gpu_frame_submit = 23;
    stats.gpu_main_ms = 5.0;
    stats.gpu_main_submit = 24;
    prof.recordFrame(3, &stats);

    const summary = prof.summarize();
    try std.testing.expectEqual(@as(usize, 2), summary.gpu_frame_samples);
    try std.testing.expectEqual(@as(f32, 1.5), summary.avg_gpu_frame_ms); // (3+0)/2
    try std.testing.expectEqual(@as(f32, 3.0), summary.max_gpu_frame_ms);
    try std.testing.expectEqual(@as(usize, 2), summary.gpu_main_samples);
    try std.testing.expectEqual(@as(f32, 3.5), summary.avg_gpu_main_ms); // (2+5)/2
    try std.testing.expectEqual(@as(f32, 5.0), summary.max_gpu_main_ms);
}

test "Profiler ring buffer O(1) wrapping and linearization preserves chronological order" {
    sokol.time.setup();
    const ally = std.testing.allocator;
    var prof = Profiler.init(ally);
    defer prof.deinit();

    prof.setMaxFrames(4);
    prof.start();

    var stats: SceneStats = .{
        .update_ms = 1.0,
        .prepare_ms = 1.0,
        .shadow_ms = 1.0,
        .main_ms = 1.0,
        .post_ms = 1.0,
    };

    // Record 6 frames into capacity 4
    for (1..7) |frame_id| {
        prof.recordFrame(frame_id, &stats);
    }

    try std.testing.expectEqual(@as(usize, 4), prof.frames.items.len);
    try std.testing.expect(prof.wrapped);

    // Stop linearizes the ring buffer
    prof.stop();
    try std.testing.expect(!prof.wrapped);

    // After linearize, oldest remaining frame should be #3, newest #6
    try std.testing.expectEqual(@as(u64, 3), prof.frames.items[0].frame_index);
    try std.testing.expectEqual(@as(u64, 4), prof.frames.items[1].frame_index);
    try std.testing.expectEqual(@as(u64, 5), prof.frames.items[2].frame_index);
    try std.testing.expectEqual(@as(u64, 6), prof.frames.items[3].frame_index);
}

test "Profiler tracks physics_ms and calculates summary physics metrics" {
    sokol.time.setup();
    const ally = std.testing.allocator;
    var prof = Profiler.init(ally);
    defer prof.deinit();
    prof.start();
    defer prof.stop();

    var stats: SceneStats = .{
        .update_ms = 2.0,
        .physics_ms = 4.0,
        .prepare_ms = 1.0,
        .shadow_ms = 1.0,
        .main_ms = 5.0,
        .post_ms = 1.0,
    };
    prof.recordFrame(1, &stats);
    stats.physics_ms = 8.0;
    prof.recordFrame(2, &stats);

    try std.testing.expectEqual(@as(f32, 4.0), prof.frames.items[0].physics_ms);
    try std.testing.expectEqual(@as(f32, 8.0), prof.frames.items[1].physics_ms);

    const summary = prof.summarize();
    try std.testing.expectEqual(@as(f32, 6.0), summary.avg_physics_ms);
    try std.testing.expectEqual(@as(f32, 8.0), summary.max_physics_ms);
}
