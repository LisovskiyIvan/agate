//! Standalone Markdown report generation. Split out of `report.zig` (facade).
//!
//! Owns `generateReportMd`. Imports std + `types` + the `report_gates`
//! predicates/formatter; no Scene import, so the report graph stays acyclic.

const std = @import("std");
const types = @import("types.zig");
const gates = @import("report_gates.zig");

const FrameRecord = types.FrameRecord;
const TextureMemoryRecord = types.TextureMemoryRecord;
const MeshMemoryRecord = types.MeshMemoryRecord;
const MemorySnapshot = types.MemorySnapshot;
const SessionSummary = types.SessionSummary;
const DiagnosticFinding = types.DiagnosticFinding;
const dominantPhase = types.dominantPhase;

const formatBytes = gates.formatBytes;
const hasGpuData = gates.hasGpuData;
const hasGpuPassData = gates.hasGpuPassData;

/// Generates a comprehensive Markdown report.
pub fn generateReportMd(
    frames: []const FrameRecord,
    summary: SessionSummary,
    findings: []const DiagnosticFinding,
    memory_ptr: ?*const MemorySnapshot,
    allocator: std.mem.Allocator,
) ![]u8 {
    var buf = std.ArrayListUnmanaged(u8).empty;
    errdefer buf.deinit(allocator);

    // Header & Overview
    try buf.appendSlice(allocator,
        \\# Agate Engine - Performance & Memory Profile Report
        \\
        \\## 1. Session Overview
        \\
    );

    const vram_total_str = if (memory_ptr) |m| try formatBytes(allocator, m.total_gpu_vram_bytes) else try allocator.dupe(u8, "N/A");
    defer allocator.free(vram_total_str);
    const cpu_mesh_str = if (memory_ptr) |m| try formatBytes(allocator, m.total_cpu_mesh_bytes) else try allocator.dupe(u8, "N/A");
    defer allocator.free(cpu_mesh_str);
    const has_gpu = hasGpuData(frames);
    const has_pass = hasGpuPassData(frames);

    const overview_table = try std.fmt.allocPrint(allocator,
        \\| Metric | Value | Metric | Value |
        \\| :--- | :--- | :--- | :--- |
        \\| **Total Frames** | {d} | **Duration (CPU-submit sum)** | {d:.2} s |
        \\| **Average FPS (wall clock)** | {d:.1} FPS | **1% Low FPS (wall clock)** | {d:.1} FPS |
        \\| **Average Interval (wall)** | {d:.2} ms | **P99 Interval (wall)** | {d:.2} ms |
        \\| **Average CPU-submit** | {d:.2} ms | **Max CPU-submit** | {d:.2} ms |
        \\| **CPU-submit P50 / P95 / P99** | {d:.2} / {d:.2} / {d:.2} ms | **Interval Hitches > 33.3 ms (wall)** | {d} frames (CPU-submit > 33.3: {d}) |
        \\| **Average Draw Calls** | {d} | **Max Draw Calls** | {d} |
        \\| **Average Triangles** | {d} | **Max Triangles** | {d} |
        \\| **Total VRAM** | {s} | **CPU Geometry** | {s} |
        \\
        \\### Average Frame Phases (CPU-submit, not GPU time)
        \\- **Update (CPU logic/animation):** {d:.2} ms
        \\- **Physics (Box3D simulation):** {d:.2} ms
        \\- **Prepare (queue preparation):** {d:.2} ms
        \\- **Shadow Pass (CPU-submit, not GPU):** {d:.2} ms
        \\- **Main Pass (CPU-submit, not GPU):** {d:.2} ms
        \\- **PostFX (CPU-submit, not GPU):** {d:.2} ms
        \\
    , .{
        summary.frame_count,
        summary.total_time_ms / 1000.0,
        summary.observed_avg_fps,
        summary.observed_fps_1pct_low,
        summary.avg_interval_ms,
        summary.p99_interval_ms,
        summary.avg_frame_ms,
        summary.max_frame_ms,
        summary.p50_frame_ms,
        summary.p95_frame_ms,
        summary.p99_frame_ms,
        summary.interval_hitches_over_33ms,
        summary.hitches_over_33ms,
        summary.avg_draw_calls,
        summary.max_draw_calls,
        summary.avg_triangles,
        summary.max_triangles,
        vram_total_str,
        cpu_mesh_str,
        summary.avg_update_ms,
        summary.avg_physics_ms,
        summary.avg_prepare_ms,
        summary.avg_shadow_ms,
        summary.avg_main_ms,
        summary.avg_post_ms,
    });
    defer allocator.free(overview_table);
    try buf.appendSlice(allocator, overview_table);

    // Measured GPU samples (gated: disabled path keeps the exact
    // pre-GPU overview above). Per-pass averages append only when
    // per-pass samples are available.
    if (has_gpu) {
        const gpu_overview = try std.fmt.allocPrint(allocator,
            \\| **Average GPU frame (measured)** | {d:.2} ms | **Max GPU frame** | {d:.2} ms |
            \\
            \\GPU time is the last completed sample observed after
            \\commit (async, without fixed lag; scope is backend-dependent
            \\and not interchangeable with CPU submit); phases above
            \\remain CPU submit, not GPU.
            \\
        , .{ summary.avg_gpu_frame_ms, summary.max_gpu_frame_ms });
        defer allocator.free(gpu_overview);
        try buf.appendSlice(allocator, gpu_overview);
        if (has_pass) {
            const pass_overview = try std.fmt.allocPrint(allocator,
                \\| **Average GPU per-pass (measured)** | shadow {d:.2} / main {d:.2} / post {d:.2} ms | **Max GPU per-pass** | {d:.2} / {d:.2} / {d:.2} ms |
                \\
                \\Per-pass GPU time is the last completed sample of each
                \\pass; values from different submission IDs are never
                \\summed into frame duration.
                \\
            , .{
                summary.avg_gpu_shadow_ms,
                summary.avg_gpu_main_ms,
                summary.avg_gpu_post_ms,
                summary.max_gpu_shadow_ms,
                summary.max_gpu_main_ms,
                summary.max_gpu_post_ms,
            });
            defer allocator.free(pass_overview);
            try buf.appendSlice(allocator, pass_overview);
        }
    }

    // Section: Bottleneck diagnostics
    try buf.appendSlice(allocator,
        \\## 2. Bottleneck Diagnostics
        \\
    );

    for (findings) |finding| {
        const badge = switch (finding.severity) {
            .good => "🟢 [OK]",
            .info => "ℹ️ [INFO]",
            .warning => "⚠️ [WARNING]",
            .critical => "🚨 [CRITICAL]",
        };
        const f_md = try std.fmt.allocPrint(allocator,
            \\### {s} {s}
            \\- **Details:** {s}
            \\- **Recommendation:** {s}
            \\
        , .{ badge, finding.title, finding.details, finding.recommendation });
        defer allocator.free(f_md);
        try buf.appendSlice(allocator, f_md);
    }

    // Section: Spike Frames (ranked by CPU-submit time, not GPU/wall time).
    // GPU column is the last-completed GPU sample, present only when
    // GPU samples are available.
    if (has_gpu) {
        try buf.appendSlice(allocator,
            \\## 3. Spike Frames (CPU-submit time)
            \\
            \\| Rank | Frame # | CPU-submit (ms) | GPU measured (ms) | FPS (wall) | Dominant Phase (CPU) | Draw Calls | Triangles | Textures | Dynamic |
            \\| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
            \\
        );
    } else {
        try buf.appendSlice(allocator,
            \\## 3. Spike Frames (CPU-submit time)
            \\
            \\| Rank | Frame # | CPU-submit (ms) | FPS (wall) | Dominant Phase (CPU) | Draw Calls | Triangles | Textures | Dynamic |
            \\| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
            \\
        );
    }

    if (frames.len > 0) {
        const spike_count = @min(10, frames.len);
        const sorted_frames = try allocator.dupe(FrameRecord, frames);
        defer allocator.free(sorted_frames);

        std.mem.sort(FrameRecord, sorted_frames, {}, struct {
            fn lessThan(_: void, a: FrameRecord, b: FrameRecord) bool {
                return a.total_frame_ms > b.total_frame_ms;
            }
        }.lessThan);

        for (sorted_frames[0..spike_count], 1..) |sf, rank| {
            const culprit = dominantPhase(sf);
            const upload_kb: f32 = @as(f32, @floatFromInt(sf.uploaded_bytes)) / 1024.0;
            const update_kb: f32 = @as(f32, @floatFromInt(sf.updated_bytes)) / 1024.0;
            const row = if (has_gpu) try std.fmt.allocPrint(allocator,
                \\| #{d} | {d} | **{d:.2} ms** | {d:.2} | {d:.1} | {s} ({d:.1} ms, {d:.0}%) | {d} | {d} | {d:.1} KB | {d:.1} KB |
            , .{
                rank,
                sf.frame_index,
                sf.total_frame_ms,
                sf.gpu_frame_ms,
                sf.fps,
                culprit.name,
                culprit.ms,
                culprit.percent,
                sf.draw_calls,
                sf.triangles,
                upload_kb,
                update_kb,
            }) else try std.fmt.allocPrint(allocator,
                \\| #{d} | {d} | **{d:.2} ms** | {d:.1} | {s} ({d:.1} ms, {d:.0}%) | {d} | {d} | {d:.1} KB | {d:.1} KB |
            , .{
                rank,
                sf.frame_index,
                sf.total_frame_ms,
                sf.fps,
                culprit.name,
                culprit.ms,
                culprit.percent,
                sf.draw_calls,
                sf.triangles,
                upload_kb,
                update_kb,
            });
            defer allocator.free(row);
            try buf.appendSlice(allocator, row);
            try buf.appendSlice(allocator, "\n");
        }
    }
    try buf.appendSlice(allocator, "\n");

    // Section: Memory Breakdown
    if (memory_ptr) |mem| {
        try buf.appendSlice(allocator,
            \\## 4. Memory & VRAM Breakdown
            \\
        );

        const tex_vram_str = try formatBytes(allocator, mem.textures_vram_bytes);
        defer allocator.free(tex_vram_str);
        const mesh_vram_str = try formatBytes(allocator, mem.meshes_vram_bytes);
        defer allocator.free(mesh_vram_str);
        const rt_vram_str = try formatBytes(allocator, mem.render_targets_vram_bytes);
        defer allocator.free(rt_vram_str);

        const mem_summary = try std.fmt.allocPrint(allocator,
            \\- **Textures:** {s}
            \\- **Meshes (VBO / IBO):** {s}
            \\- **Render Targets:** {s}
            \\- **Total VRAM:** {s}
            \\- **CPU Geometry:** {s}
            \\
            \\### Top Textures by Memory
            \\| Name | Resolution | Type | Mips | Format | VRAM |
            \\| :--- | :--- | :--- | :--- | :--- | :--- |
        , .{ tex_vram_str, mesh_vram_str, rt_vram_str, vram_total_str, cpu_mesh_str });
        defer allocator.free(mem_summary);
        try buf.appendSlice(allocator, mem_summary);
        try buf.appendSlice(allocator, "\n");

        const sorted_textures = try allocator.dupe(TextureMemoryRecord, mem.textures);
        defer allocator.free(sorted_textures);
        std.mem.sort(TextureMemoryRecord, sorted_textures, {}, struct {
            fn lessThan(_: void, a: TextureMemoryRecord, b: TextureMemoryRecord) bool {
                return a.gpu_bytes > b.gpu_bytes;
            }
        }.lessThan);

        const top_tex_n = @min(10, sorted_textures.len);
        for (sorted_textures[0..top_tex_n]) |tex| {
            const tex_sz = try formatBytes(allocator, tex.gpu_bytes);
            defer allocator.free(tex_sz);
            const row = try std.fmt.allocPrint(allocator,
                \\| {s} | {d}x{d} | {s} | {d} | {s} | **{s}** |
            , .{ tex.name, tex.width, tex.height, if (tex.is_cube) "Cubemap" else "2D", tex.num_mips, @tagName(tex.format), tex_sz });
            defer allocator.free(row);
            try buf.appendSlice(allocator, row);
            try buf.appendSlice(allocator, "\n");
        }
        try buf.appendSlice(allocator, "\n");

        // Top Meshes
        try buf.appendSlice(allocator,
            \\### Top Meshes by Memory
            \\| Mesh | Vertices | Indices | Format | VRAM | CPU |
            \\| :--- | :--- | :--- | :--- | :--- | :--- |
        );
        try buf.appendSlice(allocator, "\n");

        const sorted_meshes = try allocator.dupe(MeshMemoryRecord, mem.meshes);
        defer allocator.free(sorted_meshes);
        std.mem.sort(MeshMemoryRecord, sorted_meshes, {}, struct {
            fn lessThan(_: void, a: MeshMemoryRecord, b: MeshMemoryRecord) bool {
                return a.gpu_bytes > b.gpu_bytes;
            }
        }.lessThan);

        const top_mesh_n = @min(10, sorted_meshes.len);
        for (sorted_meshes[0..top_mesh_n]) |m| {
            const gpu_sz = try formatBytes(allocator, m.gpu_bytes);
            defer allocator.free(gpu_sz);
            const cpu_sz = try formatBytes(allocator, m.cpu_bytes);
            defer allocator.free(cpu_sz);
            const row = try std.fmt.allocPrint(allocator,
                \\| {s} | {d} | {d} | {s} | **{s}** | {s} |
            , .{ m.name, m.vertex_count, m.index_count, @tagName(m.index_type), gpu_sz, cpu_sz });
            defer allocator.free(row);
            try buf.appendSlice(allocator, row);
            try buf.appendSlice(allocator, "\n");
        }
        try buf.appendSlice(allocator, "\n");
    }

    return buf.toOwnedSlice(allocator);
}
