//! Profiler bottleneck diagnostics. Split out of `profiler.zig` (facade).
//!
//! `analyze` takes the profiler as `anytype` (a `*const Profiler` from
//! `core.zig` in practice) so this module never imports `core.zig` or the
//! facade back — same discipline as `ui/*` taking a generic canvas.
//! `core.zig` owns the `Profiler` type and forwards `analyze` here.
//! Leaf: imports `types.zig` only (`self.summarize()` resolves on the
//! caller's concrete profiler, no sibling import needed).

const std = @import("std");

const types = @import("types.zig");

const MemorySnapshot = types.MemorySnapshot;
const DiagnosticSeverity = types.DiagnosticSeverity;
const DiagnosticFinding = types.DiagnosticFinding;
const dominantPhase = types.dominantPhase;

/// Runs automated bottleneck diagnostics on recorded metrics and memory.
pub fn analyze(self: anytype, memory: ?*const MemorySnapshot, allocator: std.mem.Allocator) ![]DiagnosticFinding {
    var findings = std.ArrayListUnmanaged(DiagnosticFinding).empty;
    errdefer {
        for (findings.items) |*f| f.deinit(allocator);
        findings.deinit(allocator);
    }

    const summary = self.summarize();

    // 1. Frame Pacing / Target FPS Check (observed wall-clock intervals).
    // Phase attribution in section 2 stays CPU-submit based.
    if (summary.frame_count >= 5 and summary.max_interval_ms > 0) {
        if (summary.observed_avg_fps >= 58.0 and summary.p99_interval_ms <= 18.0 and summary.interval_hitches_over_33ms == 0) {
            try findings.append(allocator, .{
                .severity = .good,
                .title = try allocator.dupe(u8, "Stable frame pacing (60+ FPS)"),
                .details = try std.fmt.allocPrint(allocator, "Observed avg FPS (wall-clock): {d:.1}, P99 interval: {d:.1} ms, avg CPU submit: {d:.2} ms. No hitches below 30 FPS.", .{ summary.observed_avg_fps, summary.p99_interval_ms, summary.avg_frame_ms }),
                .recommendation = try allocator.dupe(u8, "Performance satisfies target frame budget (16.6 ms)."),
            });
        } else if (summary.interval_hitches_over_33ms > 0) {
            const sev: DiagnosticSeverity = if (summary.interval_hitches_over_33ms > summary.frame_count / 10 or summary.max_interval_ms > 50.0) .critical else .warning;
            try findings.append(allocator, .{
                .severity = sev,
                .title = try allocator.dupe(u8, "Framerate drop below 30 FPS"),
                .details = try std.fmt.allocPrint(allocator, "Detected {d} wall-clock intervals > 33.3 ms (worst interval: {d:.1} ms, 1% low: {d:.1} FPS). Peak frame CPU submit: {d:.1} ms.", .{ summary.interval_hitches_over_33ms, summary.max_interval_ms, summary.observed_fps_1pct_low, summary.max_frame_ms }),
                .recommendation = try allocator.dupe(u8, "Inspect Spike Frames (CPU submit time) to locate the offending phase and avoid blocking work on the main thread."),
            });
        } else if (summary.interval_hitches_over_16ms > 0) {
            try findings.append(allocator, .{
                .severity = .info,
                .title = try allocator.dupe(u8, "Intermittent framerate drops below 60 FPS"),
                .details = try std.fmt.allocPrint(allocator, "Detected {d} wall-clock intervals between 16.7 and 33.3 ms.", .{summary.interval_hitches_over_16ms}),
                .recommendation = try allocator.dupe(u8, "Optimize heavy CPU phases (Main pass, Shadows) to achieve solid 60 FPS."),
            });
        }
    }

    // 2. Worst Phase Attribution on Spikes (CPU-submit times, not GPU time)
    if (self.frames.items.len > 0) {
        var worst_frame = self.frames.items[0];
        for (self.frames.items) |f| {
            if (f.total_frame_ms > worst_frame.total_frame_ms) worst_frame = f;
        }

        if (worst_frame.total_frame_ms > 20.0) {
            const culprit = dominantPhase(worst_frame);
            if (std.mem.eql(u8, culprit.name, "Main Pass") and culprit.percent > 45.0) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Bottleneck: Main Render Pass"),
                    .details = try std.fmt.allocPrint(allocator, "In peak frame #{d} (CPU submit {d:.1} ms), Main Pass took {d:.1} ms ({d:.1}% of frame).", .{ worst_frame.frame_index, worst_frame.total_frame_ms, culprit.ms, culprit.percent }),
                    .recommendation = try allocator.dupe(u8, "Reduce draw calls using InstancedMesh, merge static meshes with identical materials, or enable occlusion culling."),
                });
            } else if (std.mem.eql(u8, culprit.name, "Shadow Pass") and culprit.percent > 35.0) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Bottleneck: CSM Shadow Pass"),
                    .details = try std.fmt.allocPrint(allocator, "In peak frame #{d}, shadow rendering took {d:.1} ms ({d:.1}% of frame).", .{ worst_frame.frame_index, culprit.ms, culprit.percent }),
                    .recommendation = try allocator.dupe(u8, "Disable cast_shadows on small meshes, reduce cascade distance, or limit point light shadows."),
                });
            } else if (std.mem.eql(u8, culprit.name, "Update") and culprit.percent > 40.0) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Bottleneck: CPU Update / Scripts"),
                    .details = try std.fmt.allocPrint(allocator, "In peak frame #{d}, update logic took {d:.1} ms ({d:.1}% of CPU submit).", .{ worst_frame.frame_index, culprit.ms, culprit.percent }),
                    .recommendation = try allocator.dupe(u8, "Optimize animations, physics, or offload heavy compute to jobs.TaskRunner."),
                });
            } else if (std.mem.eql(u8, culprit.name, "Physics / Box3D") and culprit.percent > 30.0) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Bottleneck: Physics (Box3D)"),
                    .details = try std.fmt.allocPrint(allocator, "In peak frame #{d}, physics simulation took {d:.1} ms ({d:.1}% of CPU submit).", .{ worst_frame.frame_index, culprit.ms, culprit.percent }),
                    .recommendation = try allocator.dupe(u8, "Reduce PhysicsWorld substeps, enable CCD only for fast projectiles, and use primitive colliders instead of mesh colliders."),
                });
            } else if (std.mem.eql(u8, culprit.name, "PostFX") and culprit.percent > 40.0) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Bottleneck: PostFX Stack"),
                    .details = try std.fmt.allocPrint(allocator, "In peak frame #{d}, post-processing submit took {d:.1} ms ({d:.1}% of frame).", .{ worst_frame.frame_index, culprit.ms, culprit.percent }),
                    .recommendation = try allocator.dupe(u8, "Check SSAO settings (lower sample count), reduce bloom pyramid mips, or lower render resolution."),
                });
            }

            if (worst_frame.uploaded_bytes > 4 * 1024 * 1024) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Texture streaming GPU latency"),
                    .details = try std.fmt.allocPrint(allocator, "In frame #{d}, texture streaming (UploadQueue) uploaded {d:.2} MB to the GPU.", .{ worst_frame.frame_index, @as(f32, @floatFromInt(worst_frame.uploaded_bytes)) / (1024.0 * 1024.0) }),
                    .recommendation = try allocator.dupe(u8, "Use asynchronous Scene.uploads (UploadQueue) with a max_bytes_per_frame limit."),
                });
            }

            if (worst_frame.updated_bytes > 16 * 1024 * 1024) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Heavy dynamic GPU buffer updates"),
                    .details = try std.fmt.allocPrint(allocator, "In frame #{d}, dynamic buffers (instances, morphs, particles) uploaded {d:.2} MB.", .{ worst_frame.frame_index, @as(f32, @floatFromInt(worst_frame.updated_bytes)) / (1024.0 * 1024.0) }),
                    .recommendation = try allocator.dupe(u8, "Verify instance buffer update frequency, CPU morph overhead, and active particle count."),
                });
            }
        }
    }

    // 3. Draw Calls & Pipeline Switches
    if (summary.frame_count > 0) {
        if (summary.avg_draw_calls > 1000) {
            try findings.append(allocator, .{
                .severity = .critical,
                .title = try allocator.dupe(u8, "Critical draw call count (> 1000)"),
                .details = try std.fmt.allocPrint(allocator, "Averaging {d} draw calls per frame (max {d}).", .{ summary.avg_draw_calls, summary.max_draw_calls }),
                .recommendation = try allocator.dupe(u8, "Use InstancedMesh for duplicate objects, batch static meshes, and check frustum culling."),
            });
        } else if (summary.avg_draw_calls > 400) {
            try findings.append(allocator, .{
                .severity = .warning,
                .title = try allocator.dupe(u8, "High draw call count (> 400)"),
                .details = try std.fmt.allocPrint(allocator, "Averaging {d} draw calls per frame (max {d}).", .{ summary.avg_draw_calls, summary.max_draw_calls }),
                .recommendation = try allocator.dupe(u8, "Batching and instancing are recommended to reduce driver CPU overhead."),
            });
        }

        if (summary.avg_pipeline_switches > 120) {
            try findings.append(allocator, .{
                .severity = .warning,
                .title = try allocator.dupe(u8, "Frequent shader and pipeline switches"),
                .details = try std.fmt.allocPrint(allocator, "Averaging {d} pipeline switches per frame.", .{summary.avg_pipeline_switches}),
                .recommendation = try allocator.dupe(u8, "Sort meshes by material and shader before rendering to minimize GPU state changes."),
            });
        }
    }

    // 4. Memory & Assets Checks (if snapshot available)
    if (memory) |mem| {
        if (mem.total_gpu_vram_bytes > 512 * 1024 * 1024) {
            try findings.append(allocator, .{
                .severity = .warning,
                .title = try allocator.dupe(u8, "High VRAM consumption (> 512 MB)"),
                .details = try std.fmt.allocPrint(allocator, "Total VRAM: {d:.1} MB (textures: {d:.1} MB, meshes: {d:.1} MB, render targets: {d:.1} MB).", .{
                    @as(f32, @floatFromInt(mem.total_gpu_vram_bytes)) / (1024.0 * 1024.0),
                    @as(f32, @floatFromInt(mem.textures_vram_bytes)) / (1024.0 * 1024.0),
                    @as(f32, @floatFromInt(mem.meshes_vram_bytes)) / (1024.0 * 1024.0),
                    @as(f32, @floatFromInt(mem.render_targets_vram_bytes)) / (1024.0 * 1024.0),
                }),
                .recommendation = try allocator.dupe(u8, "Apply KTX2 GPU texture compression (Basis Universal / BCn / ETC2) to reduce memory by 70-80%."),
            });
        }

        for (mem.textures) |tex| {
            if (tex.width >= 2048 and tex.height >= 2048 and tex.gpu_bytes >= 16 * 1024 * 1024 and tex.format == .RGBA8) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Uncompressed 2K+ texture"),
                    .details = try std.fmt.allocPrint(allocator, "Texture '{s}' ({d}x{d}, RGBA8) uses {d:.1} MB VRAM.", .{ tex.name, tex.width, tex.height, @as(f32, @floatFromInt(tex.gpu_bytes)) / (1024.0 * 1024.0) }),
                    .recommendation = try allocator.dupe(u8, "Convert texture to KTX2 with block compression (BC7 / BC1 / ETC1S) or reduce resolution to 1024x1024."),
                });
                break;
            }
        }

        for (mem.textures) |tex| {
            if (tex.width >= 512 and tex.height >= 512 and tex.num_mips <= 1 and !tex.is_cube) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "Missing mipmaps on large texture"),
                    .details = try std.fmt.allocPrint(allocator, "Texture '{s}' ({d}x{d}) loaded without a mipmap chain.", .{ tex.name, tex.width, tex.height }),
                    .recommendation = try allocator.dupe(u8, "Enable mipmap generation (options.mipmaps = true) to prevent distant texture aliasing and improve GPU cache hits."),
                });
                break;
            }
        }

        var wasteful_u32_count: usize = 0;
        for (mem.meshes) |m| {
            if (m.index_type == .UINT32 and m.vertex_count > 0 and m.vertex_count <= 65535) {
                wasteful_u32_count += 1;
            }
        }
        if (wasteful_u32_count > 0) {
            try findings.append(allocator, .{
                .severity = .info,
                .title = try allocator.dupe(u8, "Suboptimal 32-bit index buffers"),
                .details = try std.fmt.allocPrint(allocator, "{d} meshes use UINT32 indices for under 65536 vertices.", .{wasteful_u32_count}),
                .recommendation = try allocator.dupe(u8, "UINT16 indices save 50% index memory and reduce GPU memory bus pressure."),
            });
        }

        for (mem.meshes) |m| {
            if (m.vertex_count > 60000) {
                try findings.append(allocator, .{
                    .severity = .warning,
                    .title = try allocator.dupe(u8, "High mesh vertex density (> 60k)"),
                    .details = try std.fmt.allocPrint(allocator, "Mesh '{s}' has {d} vertices and {d} indices ({d:.1} MB VRAM).", .{ m.name, m.vertex_count, m.index_count, @as(f32, @floatFromInt(m.gpu_bytes)) / (1024.0 * 1024.0) }),
                    .recommendation = try allocator.dupe(u8, "Use meshopt simplification to generate levels of detail (LOD)."),
                });
                break;
            }
        }
    }

    // Default good condition if no warnings or criticals were produced
    var has_issues = false;
    for (findings.items) |f| {
        if (f.severity == .warning or f.severity == .critical) {
            has_issues = true;
            break;
        }
    }
    if (!has_issues and findings.items.len == 0) {
        try findings.append(allocator, .{
            .severity = .good,
            .title = try allocator.dupe(u8, "All key metrics normal"),
            .details = try allocator.dupe(u8, "No critical hitches, memory bloat, or excessive draw calls detected."),
            .recommendation = try allocator.dupe(u8, "The current scene configuration is performing within expected budgets."),
        });
    }

    return findings.toOwnedSlice(allocator);
}
