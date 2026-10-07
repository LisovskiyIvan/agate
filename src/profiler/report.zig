const std = @import("std");
const types = @import("types.zig");

const FrameRecord = types.FrameRecord;
const TextureMemoryRecord = types.TextureMemoryRecord;
const MeshMemoryRecord = types.MeshMemoryRecord;
const MemorySnapshot = types.MemorySnapshot;
const SessionSummary = types.SessionSummary;
const DiagnosticFinding = types.DiagnosticFinding;
const dominantPhase = types.dominantPhase;

// Standalone report generation over profiler data values.
// Depends only on types.zig (and std): no Scene import, so the
// profiler.zig -> report.zig -> types.zig graph stays acyclic.
// Callers pass already-computed summary/findings/memory values;
// Profiler wrapper methods in profiler.zig capture those first.

/// True when at least one frame carries an AVAILABLE GPU frame sample
/// (submission id != 0 — valid quantized zeros count). All GPU-specific
/// report output (HTML/MD/trace) is gated on this so the disabled path
/// (no ids) stays byte-identical to the pre-GPU output.
pub fn hasGpuData(frames: []const FrameRecord) bool {
    for (frames) |f| {
        if (f.gpu_frame_submit != 0) return true;
    }
    return false;
}

/// True when at least one frame carries an AVAILABLE per-pass GPU sample
/// (any phase submission id != 0, zeros included). Gated separately from
/// `hasGpuData`, so frame-scope-only sessions (no per-pass samples) and
/// the disabled path both render exactly the pre-per-pass output.
pub fn hasGpuPassData(frames: []const FrameRecord) bool {
    for (frames) |f| {
        if (f.gpu_shadow_submit != 0 or f.gpu_main_submit != 0 or f.gpu_post_submit != 0) return true;
    }
    return false;
}

/// Helper to format byte values (e.g. "12.4 MB").
pub fn formatBytes(allocator: std.mem.Allocator, bytes: usize) ![]u8 {
    if (bytes < 1024) {
        return std.fmt.allocPrint(allocator, "{d} B", .{bytes});
    } else if (bytes < 1024 * 1024) {
        return std.fmt.allocPrint(allocator, "{d:.1} KB", .{@as(f32, @floatFromInt(bytes)) / 1024.0});
    } else if (bytes < 1024 * 1024 * 1024) {
        return std.fmt.allocPrint(allocator, "{d:.2} MB", .{@as(f32, @floatFromInt(bytes)) / (1024.0 * 1024.0)});
    } else {
        return std.fmt.allocPrint(allocator, "{d:.2} GB", .{@as(f32, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0)});
    }
}

/// Generates a standalone, beautiful, dark-themed HTML report.
pub fn generateReportHtml(
    frames: []const FrameRecord,
    summary: SessionSummary,
    findings: []const DiagnosticFinding,
    memory_ptr: ?*const MemorySnapshot,
    allocator: std.mem.Allocator,
) ![]u8 {
    var buf = std.ArrayListUnmanaged(u8).empty;
    errdefer buf.deinit(allocator);

    // HTML Header & Embedded Dark Styles
    try buf.appendSlice(allocator,
        \\<!DOCTYPE html>
        \\<html lang="ru">
        \\<head>
        \\<meta charset="UTF-8">
        \\<meta name="viewport" content="width=device-width, initial-scale=1.0">
        \\<title>Agate Engine - Profile & Memory Report</title>
        \\<style>
        \\  :root {
        \\    --bg-main: #0b0f19;
        \\    --bg-card: #151e2e;
        \\    --bg-card-hover: #1c283c;
        \\    --border: #243247;
        \\    --border-subtle: #1a2536;
        \\    --text-main: #f1f5f9;
        \\    --text-muted: #94a3b8;
        \\    --color-update: #3b82f6;
        \\    --color-prepare: #06b6d4;
        \\    --color-shadow: #8b5cf6;
        \\    --color-main: #10b981;
        \\    --color-post: #f59e0b;
        \\    --color-good: #22c55e;
        \\    --color-warn: #eab308;
        \\    --color-crit: #ef4444;
        \\    --color-info: #38bdf8;
        \\  }
        \\  * { box-sizing: border-box; margin: 0; padding: 0; }
        \\  body {
        \\    background: var(--bg-main);
        \\    color: var(--text-main);
        \\    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        \\    line-height: 1.5;
        \\    padding: 24px;
        \\  }
        \\  .container { max-width: 1400px; margin: 0 auto; }
        \\  header {
        \\    display: flex;
        \\    justify-content: space-between;
        \\    align-items: center;
        \\    border-bottom: 1px solid var(--border);
        \\    padding-bottom: 20px;
        \\    margin-bottom: 24px;
        \\  }
        \\  .logo-title { display: flex; align-items: center; gap: 14px; }
        \\  .badge-logo {
        \\    background: linear-gradient(135deg, #6366f1, #a855f7);
        \\    padding: 6px 14px;
        \\    border-radius: 8px;
        \\    font-weight: 800;
        \\    font-size: 16px;
        \\    color: #fff;
        \\    letter-spacing: 1px;
        \\  }
        \\  h1 { font-size: 24px; font-weight: 700; color: #fff; }
        \\  .meta-sub { color: var(--text-muted); font-size: 13px; margin-top: 4px; }
        \\  .kpi-grid {
        \\    display: grid;
        \\    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        \\    gap: 16px;
        \\    margin-bottom: 28px;
        \\  }
        \\  .kpi-card {
        \\    background: var(--bg-card);
        \\    border: 1px solid var(--border);
        \\    border-radius: 12px;
        \\    padding: 16px;
        \\    transition: transform 0.15s ease, border-color 0.15s ease;
        \\  }
        \\  .kpi-card:hover { transform: translateY(-2px); border-color: #3b82f6; }
        \\  .kpi-label { font-size: 12px; font-weight: 600; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.5px; }
        \\  .kpi-val { font-size: 26px; font-weight: 800; margin-top: 6px; color: #fff; }
        \\  .kpi-sub { font-size: 12px; color: var(--text-muted); margin-top: 4px; }
        \\  .section-title {
        \\    font-size: 18px;
        \\    font-weight: 700;
        \\    margin-bottom: 16px;
        \\    display: flex;
        \\    align-items: center;
        \\    gap: 10px;
        \\  }
        \\  .diag-list { display: flex; flex-direction: column; gap: 12px; margin-bottom: 32px; }
        \\  .diag-card {
        \\    background: var(--bg-card);
        \\    border: 1px solid var(--border);
        \\    border-left: 5px solid var(--border);
        \\    border-radius: 10px;
        \\    padding: 16px 20px;
        \\  }
        \\  .diag-card.good { border-left-color: var(--color-good); }
        \\  .diag-card.info { border-left-color: var(--color-info); }
        \\  .diag-card.warning { border-left-color: var(--color-warn); }
        \\  .diag-card.critical { border-left-color: var(--color-crit); }
        \\  .diag-header { display: flex; align-items: center; gap: 12px; margin-bottom: 6px; }
        \\  .diag-badge {
        \\    font-size: 11px;
        \\    font-weight: 700;
        \\    text-transform: uppercase;
        \\    padding: 3px 8px;
        \\    border-radius: 4px;
        \\  }
        \\  .diag-badge.good { background: #052e16; color: #86efac; border: 1px solid #16a34a; }
        \\  .diag-badge.info { background: #082f49; color: #7dd3fc; border: 1px solid #0284c7; }
        \\  .diag-badge.warning { background: #422006; color: #fde68a; border: 1px solid #d97706; }
        \\  .diag-badge.critical { background: #450a0a; color: #fca5a5; border: 1px solid #dc2626; }
        \\  .diag-title { font-size: 15px; font-weight: 600; color: #fff; }
        \\  .diag-desc { font-size: 13px; color: var(--text-muted); margin-bottom: 8px; }
        \\  .diag-rec {
        \\    font-size: 13px;
        \\    color: #a7f3d0;
        \\    background: #064e3b33;
        \\    border: 1px solid #065f46;
        \\    border-radius: 6px;
        \\    padding: 8px 12px;
        \\  }
        \\  .chart-box {
        \\    background: var(--bg-card);
        \\    border: 1px solid var(--border);
        \\    border-radius: 12px;
        \\    padding: 20px;
        \\    margin-bottom: 32px;
        \\  }
        \\  .chart-legend {
        \\    display: flex;
        \\    flex-wrap: wrap;
        \\    gap: 18px;
        \\    font-size: 12px;
        \\    margin-top: 14px;
        \\    justify-content: center;
        \\  }
        \\  .legend-item { display: flex; align-items: center; gap: 6px; color: var(--text-muted); }
        \\  .legend-dot { width: 12px; height: 12px; border-radius: 3px; }
        \\  .timeline-svg { width: 100%; height: 260px; overflow: visible; }
        \\  .card-table {
        \\    background: var(--bg-card);
        \\    border: 1px solid var(--border);
        \\    border-radius: 12px;
        \\    overflow: hidden;
        \\    margin-bottom: 32px;
        \\  }
        \\  table { width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }
        \\  th {
        \\    background: #111927;
        \\    color: var(--text-muted);
        \\    font-weight: 600;
        \\    text-transform: uppercase;
        \\    font-size: 11px;
        \\    letter-spacing: 0.5px;
        \\    padding: 12px 16px;
        \\    border-bottom: 1px solid var(--border);
        \\  }
        \\  td { padding: 12px 16px; border-bottom: 1px solid var(--border-subtle); color: #cbd5e1; }
        \\  tr:last-child td { border-bottom: none; }
        \\  tr:hover td { background: var(--bg-card-hover); }
        \\  .num { text-align: right; font-variant-numeric: tabular-nums; }
        \\  .vram-bar {
        \\    display: flex;
        \\    height: 18px;
        \\    border-radius: 6px;
        \\    overflow: hidden;
        \\    background: #0f172a;
        \\    margin: 12px 0 20px 0;
        \\    border: 1px solid var(--border);
        \\  }
        \\  .vram-seg { height: 100%; transition: width 0.3s ease; }
        \\  footer {
        \\    text-align: center;
        \\    color: var(--text-muted);
        \\    font-size: 12px;
        \\    padding-top: 24px;
        \\    border-top: 1px solid var(--border);
        \\  }
        \\</style>
        \\</head>
        \\<body>
        \\<div class="container">
        \\<header>
        \\  <div class="logo-title">
        \\    <div class="badge-logo">AGATE</div>
        \\    <div>
        \\      <h1>Performance & Memory Profile Report</h1>
        \\      <div class="meta-sub">Engine Flight Recorder & Snapshot Inspector</div>
        \\    </div>
        \\  </div>
        \\  <div style="text-align: right;">
    );

    // Header metadata: wall-clock pacing is primary; CPU-submit sum is separate.
    const total_sec = summary.total_time_ms / 1000.0;
    const meta_str = try std.fmt.allocPrint(allocator,
        \\<div style="font-size: 14px; font-weight: 700; color: #fff;">{d} Кадров ({d:.2} с CPU-submit)</div>
        \\<div class="meta-sub">Avg {d:.1} FPS (wall) | P99 интервала: {d:.1} мс | CPU-submit avg: {d:.2} мс</div>
        \\</div></header>
    , .{ summary.frame_count, total_sec, summary.observed_avg_fps, summary.p99_interval_ms, summary.avg_frame_ms });
    defer allocator.free(meta_str);
    try buf.appendSlice(allocator, meta_str);

    // KPI Cards Grid
    const vram_total_str = if (memory_ptr) |m| try formatBytes(allocator, m.total_gpu_vram_bytes) else try allocator.dupe(u8, "N/A");
    defer allocator.free(vram_total_str);
    const cpu_mesh_str = if (memory_ptr) |m| try formatBytes(allocator, m.total_cpu_mesh_bytes) else try allocator.dupe(u8, "N/A");
    defer allocator.free(cpu_mesh_str);
    const has_gpu = hasGpuData(frames);
    const has_pass = hasGpuPassData(frames);

    const kpi_html = try std.fmt.allocPrint(allocator,
        \\<div class="kpi-grid">
        \\  <div class="kpi-card">
        \\    <div class="kpi-label">Average FPS (wall)</div>
        \\    <div class="kpi-val" style="color: {s};">{d:.1}</div>
        \\    <div class="kpi-sub">Avg interval: {d:.2} ms | CPU-submit avg: {d:.2} ms</div>
        \\  </div>
        \\  <div class="kpi-card">
        \\    <div class="kpi-label">1% Low FPS (wall)</div>
        \\    <div class="kpi-val" style="color: {s};">{d:.1}</div>
        \\    <div class="kpi-sub">0.1% low (wall): {d:.1} FPS</div>
        \\  </div>
        \\  <div class="kpi-card">
        \\    <div class="kpi-label">P99 Frame Interval (wall)</div>
        \\    <div class="kpi-val">{d:.1} <span style="font-size: 14px; font-weight: normal; color: #94a3b8;">ms</span></div>
        \\    <div class="kpi-sub">P50 interval: {d:.1} ms | CPU-submit P99: {d:.1} ms</div>
        \\  </div>
        \\  <div class="kpi-card">
        \\    <div class="kpi-label">Average Draw Calls</div>
        \\    <div class="kpi-val">{d}</div>
        \\    <div class="kpi-sub">Max: {d} | Avg Tris: {d}</div>
        \\  </div>
        \\  <div class="kpi-card">
        \\    <div class="kpi-label">GPU VRAM Total</div>
        \\    <div class="kpi-val" style="color: #38bdf8;">{s}</div>
        \\    <div class="kpi-sub">Textures + Meshes + RTs</div>
        \\  </div>
        \\  <div class="kpi-card">
        \\    <div class="kpi-label">CPU Mesh Memory</div>
        \\    <div class="kpi-val" style="color: #a855f7;">{s}</div>
        \\    <div class="kpi-sub">{d} Meshes in Scene</div>
        \\  </div>
        \\</div>
    , .{
        if (summary.observed_avg_fps >= 55.0) "#22c55e" else if (summary.observed_avg_fps >= 30.0) "#eab308" else "#ef4444",
        summary.observed_avg_fps,
        summary.avg_interval_ms,
        summary.avg_frame_ms,
        if (summary.observed_fps_1pct_low >= 45.0) "#22c55e" else if (summary.observed_fps_1pct_low >= 25.0) "#eab308" else "#ef4444",
        summary.observed_fps_1pct_low,
        summary.observed_fps_01pct_low,
        summary.p99_interval_ms,
        summary.p50_interval_ms,
        summary.p99_frame_ms,
        summary.avg_draw_calls,
        summary.max_draw_calls,
        summary.avg_triangles,
        vram_total_str,
        cpu_mesh_str,
        if (memory_ptr) |m| m.mesh_count else 0,
    });
    defer allocator.free(kpi_html);
    try buf.appendSlice(allocator, kpi_html);

    // GPU frame time card (measured on GPU via the opt-in gpu_timing
    // samples; gated so the disabled path renders exactly the pre-GPU report).
    if (has_gpu) {
        const gpu_sub = if (hasGpuPassData(frames)) try std.fmt.allocPrint(allocator,
            \\<div class="kpi-sub">Per-pass avg (available distinct samples): shadow {d:.2} / main {d:.2} / post {d:.2} ms | max: {d:.2} / {d:.2} / {d:.2} ms (backend phase spans; per-pass values are never summed into a frame)</div>
        , .{
            summary.avg_gpu_shadow_ms,
            summary.avg_gpu_main_ms,
            summary.avg_gpu_post_ms,
            summary.max_gpu_shadow_ms,
            summary.max_gpu_main_ms,
            summary.max_gpu_post_ms,
        }) else try allocator.dupe(u8, "");
        defer allocator.free(gpu_sub);
        const gpu_html = try std.fmt.allocPrint(allocator,
            \\<div class="kpi-grid">
            \\  <div class="kpi-card">
            \\    <div class="kpi-label">GPU Frame (measured)</div>
            \\    <div class="kpi-val" style="color: #f472b6;">{d:.2} <span style="font-size: 14px; font-weight: normal; color: #94a3b8;">ms</span></div>
            \\    <div class="kpi-sub">Max: {d:.2} ms | Last-completed GPU sample, observed after commit (async; backend scope varies — not interchangeable with CPU submit)</div>
            \\{s}  </div>
            \\</div>
        , .{ summary.avg_gpu_frame_ms, summary.max_gpu_frame_ms, gpu_sub });
        defer allocator.free(gpu_html);
        try buf.appendSlice(allocator, gpu_html);
    }

    // Section: "Что не так / Автоматическая диагностика"
    try buf.appendSlice(allocator,
        \\<div class="section-title">
        \\  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#eab308" stroke-width="2"><path d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>
        \\  Что не так / Автоматическая диагностика
        \\</div>
        \\<div class="diag-list">
    );

    for (findings) |finding| {
        const sev_class = switch (finding.severity) {
            .good => "good",
            .info => "info",
            .warning => "warning",
            .critical => "critical",
        };
        const sev_label = switch (finding.severity) {
            .good => "НОРМА",
            .info => "ИНФО",
            .warning => "ВНИМАНИЕ",
            .critical => "КРИТИЧНО",
        };
        const f_html = try std.fmt.allocPrint(allocator,
            \\  <div class="diag-card {s}">
            \\    <div class="diag-header">
            \\      <span class="diag-badge {s}">{s}</span>
            \\      <span class="diag-title">{s}</span>
            \\    </div>
            \\    <div class="diag-desc">{s}</div>
            \\    <div class="diag-rec"><strong>Рекомендация:</strong> {s}</div>
            \\  </div>
        , .{ sev_class, sev_class, sev_label, finding.title, finding.details, finding.recommendation });
        defer allocator.free(f_html);
        try buf.appendSlice(allocator, f_html);
    }
    try buf.appendSlice(allocator, "</div>\n");

    // Section: Interactive Stacked Frame Timeline (SVG)
    try buf.appendSlice(allocator,
        \\<div class="section-title">
        \\  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
        \\  График времени кадров по фазам (Stacked Frame Timeline)
        \\</div>
        \\<div class="chart-box">
        \\  <svg viewBox="0 0 1000 240" class="timeline-svg" preserveAspectRatio="none">
    );

    const chart_h: f32 = 190.0;
    const chart_w: f32 = 930.0;
    const chart_y_bottom: f32 = 210.0;
    const chart_x_start: f32 = 50.0;
    const max_time = @max(35.0, summary.max_frame_ms * 1.15);
    const y_scale = chart_h / max_time;

    // Horizontal guidelines: 16.6ms (60 FPS) and 33.3ms (30 FPS)
    const y_16 = chart_y_bottom - (16.67 * y_scale);
    const line16_str = try std.fmt.allocPrint(allocator,
        \\<line x1="{d:.1}" y1="{d:.1}" x2="{d:.1}" y2="{d:.1}" stroke="#22c55e" stroke-dasharray="4" stroke-width="1.5" />
        \\<text x="5" y="{d:.1}" fill="#22c55e" font-size="11" font-weight="600">16.6ms (60 FPS)</text>
    , .{ chart_x_start, y_16, chart_x_start + chart_w, y_16, y_16 + 4.0 });
    defer allocator.free(line16_str);
    try buf.appendSlice(allocator, line16_str);

    if (max_time >= 33.33) {
        const y_33 = chart_y_bottom - (33.33 * y_scale);
        const line33_str = try std.fmt.allocPrint(allocator,
            \\<line x1="{d:.1}" y1="{d:.1}" x2="{d:.1}" y2="{d:.1}" stroke="#f97316" stroke-dasharray="4" stroke-width="1.5" />
            \\<text x="5" y="{d:.1}" fill="#f97316" font-size="11" font-weight="600">33.3ms (30 FPS)</text>
        , .{ chart_x_start, y_33, chart_x_start + chart_w, y_33, y_33 + 4.0 });
        defer allocator.free(line33_str);
        try buf.appendSlice(allocator, line33_str);
    }

    // Draw stacked bars for each frame
    const frame_count = frames.len;
    if (frame_count > 0) {
        const bar_step = chart_w / @as(f32, @floatFromInt(frame_count));
        const bar_w = @max(1.0, bar_step * 0.85);

        for (frames, 0..) |f, i| {
            const x = chart_x_start + @as(f32, @floatFromInt(i)) * bar_step;

            // Stack from bottom up: update -> physics -> prepare -> shadow -> main -> post
            const h_update = f.update_ms * y_scale;
            const h_physics = f.physics_ms * y_scale;
            const h_prepare = f.prepare_ms * y_scale;
            const h_shadow = f.shadow_ms * y_scale;
            const h_main = f.main_ms * y_scale;
            const h_post = f.post_ms * y_scale;

            var cur_y = chart_y_bottom;

            // Group with native tooltip (stacked bars show CPU-submit phases, not GPU time).
            // The GPU line below is the last-completed GPU sample observed at
            // this CPU frame (async execution, no fixed lag promised);
            // emitted only when GPU samples are available so the disabled
            // path stays byte-identical. Per-pass GPU lines append only when
            // per-pass samples are available.
            const tooltip_open = if (has_pass) try std.fmt.allocPrint(allocator,
                \\<g><title>Кадр #{d}: {d:.2} мс CPU-submit (FPS wall: {d:.1}, интервал wall: {d:.2} мс)&#10;Update: {d:.2} мс&#10;Physics (Box3D): {d:.2} мс&#10;Prepare: {d:.2} мс&#10;Shadow (CPU submit): {d:.2} мс&#10;Main (CPU submit): {d:.2} мс&#10;Post (CPU submit): {d:.2} мс&#10;GPU (measured, last completed): {d:.2} мс&#10;GPU shadow/main/post (measured): {d:.2} / {d:.2} / {d:.2} мс&#10;Draw calls: {d} | Tris: {d}</title>
            , .{ f.frame_index, f.total_frame_ms, f.fps, f.frame_interval_ms, f.update_ms, f.physics_ms, f.prepare_ms, f.shadow_ms, f.main_ms, f.post_ms, f.gpu_frame_ms, f.gpu_shadow_ms, f.gpu_main_ms, f.gpu_post_ms, f.draw_calls, f.triangles }) else if (has_gpu) try std.fmt.allocPrint(allocator,
                \\<g><title>Кадр #{d}: {d:.2} мс CPU-submit (FPS wall: {d:.1}, интервал wall: {d:.2} мс)&#10;Update: {d:.2} мс&#10;Physics (Box3D): {d:.2} мс&#10;Prepare: {d:.2} мс&#10;Shadow (CPU submit): {d:.2} мс&#10;Main (CPU submit): {d:.2} мс&#10;Post (CPU submit): {d:.2} мс&#10;GPU (measured, last completed): {d:.2} мс&#10;Draw calls: {d} | Tris: {d}</title>
            , .{ f.frame_index, f.total_frame_ms, f.fps, f.frame_interval_ms, f.update_ms, f.physics_ms, f.prepare_ms, f.shadow_ms, f.main_ms, f.post_ms, f.gpu_frame_ms, f.draw_calls, f.triangles }) else try std.fmt.allocPrint(allocator,
                \\<g><title>Кадр #{d}: {d:.2} мс CPU-submit (FPS wall: {d:.1}, интервал wall: {d:.2} мс)&#10;Update: {d:.2} мс&#10;Physics (Box3D): {d:.2} мс&#10;Prepare: {d:.2} мс&#10;Shadow (CPU submit): {d:.2} мс&#10;Main (CPU submit): {d:.2} мс&#10;Post (CPU submit): {d:.2} мс&#10;Draw calls: {d} | Tris: {d}</title>
            , .{ f.frame_index, f.total_frame_ms, f.fps, f.frame_interval_ms, f.update_ms, f.physics_ms, f.prepare_ms, f.shadow_ms, f.main_ms, f.post_ms, f.draw_calls, f.triangles });
            defer allocator.free(tooltip_open);
            try buf.appendSlice(allocator, tooltip_open);

            // Update rect
            if (h_update > 0.1) {
                cur_y -= h_update;
                const r = try std.fmt.allocPrint(allocator, "<rect x=\"{d:.1}\" y=\"{d:.1}\" width=\"{d:.1}\" height=\"{d:.1}\" fill=\"#3b82f6\" />\n", .{ x, cur_y, bar_w, h_update });
                defer allocator.free(r);
                try buf.appendSlice(allocator, r);
            }

            // Physics rect
            if (h_physics > 0.1) {
                cur_y -= h_physics;
                const r = try std.fmt.allocPrint(allocator, "<rect x=\"{d:.1}\" y=\"{d:.1}\" width=\"{d:.1}\" height=\"{d:.1}\" fill=\"#10b981\" />\n", .{ x, cur_y, bar_w, h_physics });
                defer allocator.free(r);
                try buf.appendSlice(allocator, r);
            }

            // Prepare rect
            if (h_prepare > 0.1) {
                cur_y -= h_prepare;
                const r = try std.fmt.allocPrint(allocator, "<rect x=\"{d:.1}\" y=\"{d:.1}\" width=\"{d:.1}\" height=\"{d:.1}\" fill=\"#06b6d4\" />\n", .{ x, cur_y, bar_w, h_prepare });
                defer allocator.free(r);
                try buf.appendSlice(allocator, r);
            }

            // Shadow rect
            if (h_shadow > 0.1) {
                cur_y -= h_shadow;
                const r = try std.fmt.allocPrint(allocator, "<rect x=\"{d:.1}\" y=\"{d:.1}\" width=\"{d:.1}\" height=\"{d:.1}\" fill=\"#8b5cf6\" />\n", .{ x, cur_y, bar_w, h_shadow });
                defer allocator.free(r);
                try buf.appendSlice(allocator, r);
            }

            // Main pass rect
            if (h_main > 0.1) {
                cur_y -= h_main;
                const r = try std.fmt.allocPrint(allocator, "<rect x=\"{d:.1}\" y=\"{d:.1}\" width=\"{d:.1}\" height=\"{d:.1}\" fill=\"#6366f1\" />\n", .{ x, cur_y, bar_w, h_main });
                defer allocator.free(r);
                try buf.appendSlice(allocator, r);
            }

            // PostFX rect
            if (h_post > 0.1) {
                cur_y -= h_post;
                const r = try std.fmt.allocPrint(allocator, "<rect x=\"{d:.1}\" y=\"{d:.1}\" width=\"{d:.1}\" height=\"{d:.1}\" fill=\"#f59e0b\" />\n", .{ x, cur_y, bar_w, h_post });
                defer allocator.free(r);
                try buf.appendSlice(allocator, r);
            }

            try buf.appendSlice(allocator, "</g>\n");
        }
    }

    try buf.appendSlice(allocator,
        \\  </svg>
        \\  <div class="chart-legend">
        \\    <div class="legend-item"><div class="legend-dot" style="background: #3b82f6;"></div>Update</div>
        \\    <div class="legend-item"><div class="legend-dot" style="background: #10b981;"></div>Physics (Box3D)</div>
        \\    <div class="legend-item"><div class="legend-dot" style="background: #06b6d4;"></div>Prepare</div>
        \\    <div class="legend-item"><div class="legend-dot" style="background: #8b5cf6;"></div>Shadow Pass</div>
        \\    <div class="legend-item"><div class="legend-dot" style="background: #6366f1;"></div>Main Pass</div>
        \\    <div class="legend-item"><div class="legend-dot" style="background: #f59e0b;"></div>PostFX</div>
        \\  </div>
        \\</div>
    );

    // Section: Top Spike Frames Table (ranked by CPU-submit time, not GPU/wall time).
    // The GPU column is the last-completed GPU sample, appended only when
    // GPU samples are available (disabled path keeps the exact pre-GPU table).
    if (has_gpu) {
        try buf.appendSlice(allocator,
            \\<div class="section-title">
            \\  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
            \\  Топ пиковых кадров (Spike Frames, время CPU-submit)
            \\</div>
            \\<div class="card-table">
            \\<table>
            \\  <thead>
            \\    <tr>
            \\      <th>Ранг</th>
            \\      <th>Кадр #</th>
            \\      <th class="num">Время CPU-submit</th>
            \\      <th class="num">GPU (measured)</th>
            \\      <th class="num">FPS (wall)</th>
            \\      <th>Главная причина (CPU-фаза)</th>
            \\      <th class="num">Draw Calls</th>
            \\      <th class="num">Треугольники</th>
            \\      <th class="num">Pipeline Switches</th>
            \\      <th class="num">Текстуры (КБ)</th>
            \\      <th class="num">Динамика (КБ)</th>
            \\    </tr>
            \\  </thead>
            \\  <tbody>
        );
    } else {
        try buf.appendSlice(allocator,
            \\<div class="section-title">
            \\  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
            \\  Топ пиковых кадров (Spike Frames, время CPU-submit)
            \\</div>
            \\<div class="card-table">
            \\<table>
            \\  <thead>
            \\    <tr>
            \\      <th>Ранг</th>
            \\      <th>Кадр #</th>
            \\      <th class="num">Время CPU-submit</th>
            \\      <th class="num">FPS (wall)</th>
            \\      <th>Главная причина (CPU-фаза)</th>
            \\      <th class="num">Draw Calls</th>
            \\      <th class="num">Треугольники</th>
            \\      <th class="num">Pipeline Switches</th>
            \\      <th class="num">Текстуры (КБ)</th>
            \\      <th class="num">Динамика (КБ)</th>
            \\    </tr>
            \\  </thead>
            \\  <tbody>
        );
    }

    // Sort frames by total_frame_ms (CPU-submit sum) descending
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
                \\    <tr>
                \\      <td><strong>#{d}</strong></td>
                \\      <td>Кадр {d}</td>
                \\      <td class="num"><span style="color: {s}; font-weight: 700;">{d:.2} мс</span></td>
                \\      <td class="num">{d:.2} мс</td>
                \\      <td class="num">{d:.1}</td>
                \\      <td><span style="color: #fff; font-weight: 600;">{s}</span> <span style="color: var(--text-muted);">({d:.1} мс, {d:.0}%)</span></td>
                \\      <td class="num">{d}</td>
                \\      <td class="num">{d}</td>
                \\      <td class="num">{d}</td>
                \\      <td class="num">{d:.1}</td>
                \\      <td class="num">{d:.1}</td>
                \\    </tr>
            , .{
                rank,
                sf.frame_index,
                if (sf.total_frame_ms > 33.33) "#ef4444" else if (sf.total_frame_ms > 16.67) "#eab308" else "#22c55e",
                sf.total_frame_ms,
                sf.gpu_frame_ms,
                sf.fps,
                culprit.name,
                culprit.ms,
                culprit.percent,
                sf.draw_calls,
                sf.triangles,
                sf.pipeline_switches,
                upload_kb,
                update_kb,
            }) else try std.fmt.allocPrint(allocator,
                \\    <tr>
                \\      <td><strong>#{d}</strong></td>
                \\      <td>Кадр {d}</td>
                \\      <td class="num"><span style="color: {s}; font-weight: 700;">{d:.2} мс</span></td>
                \\      <td class="num">{d:.1}</td>
                \\      <td><span style="color: #fff; font-weight: 600;">{s}</span> <span style="color: var(--text-muted);">({d:.1} мс, {d:.0}%)</span></td>
                \\      <td class="num">{d}</td>
                \\      <td class="num">{d}</td>
                \\      <td class="num">{d}</td>
                \\      <td class="num">{d:.1}</td>
                \\      <td class="num">{d:.1}</td>
                \\    </tr>
            , .{
                rank,
                sf.frame_index,
                if (sf.total_frame_ms > 33.33) "#ef4444" else if (sf.total_frame_ms > 16.67) "#eab308" else "#22c55e",
                sf.total_frame_ms,
                sf.fps,
                culprit.name,
                culprit.ms,
                culprit.percent,
                sf.draw_calls,
                sf.triangles,
                sf.pipeline_switches,
                upload_kb,
                update_kb,
            });
            defer allocator.free(row);
            try buf.appendSlice(allocator, row);
        }
    }
    try buf.appendSlice(allocator, "  </tbody>\n</table>\n</div>\n");

    // Section: Memory & VRAM Breakdown
    if (memory_ptr) |mem| {
        try buf.appendSlice(allocator,
            \\<div class="section-title">
            \\  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"/><rect x="2" y="14" width="20" height="8" rx="2" ry="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>
            \\  Распределение видеопамяти (VRAM Breakdown)
            \\</div>
        );

        // VRAM segmented progress bar
        const total_vram = @max(1, mem.total_gpu_vram_bytes);
        const tex_pct = (@as(f32, @floatFromInt(mem.textures_vram_bytes)) / @as(f32, @floatFromInt(total_vram))) * 100.0;
        const mesh_pct = (@as(f32, @floatFromInt(mem.meshes_vram_bytes)) / @as(f32, @floatFromInt(total_vram))) * 100.0;
        const rt_pct = (@as(f32, @floatFromInt(mem.render_targets_vram_bytes)) / @as(f32, @floatFromInt(total_vram))) * 100.0;

        const tex_vram_str = try formatBytes(allocator, mem.textures_vram_bytes);
        defer allocator.free(tex_vram_str);
        const mesh_vram_str = try formatBytes(allocator, mem.meshes_vram_bytes);
        defer allocator.free(mesh_vram_str);
        const rt_vram_str = try formatBytes(allocator, mem.render_targets_vram_bytes);
        defer allocator.free(rt_vram_str);

        const vram_bar_html = try std.fmt.allocPrint(allocator,
            \\<div class="vram-bar">
            \\  <div class="vram-seg" style="width: {d:.1}%; background: #38bdf8;" title="Текстуры: {s} ({d:.1}%)"></div>
            \\  <div class="vram-seg" style="width: {d:.1}%; background: #a855f7;" title="Меши: {s} ({d:.1}%)"></div>
            \\  <div class="vram-seg" style="width: {d:.1}%; background: #f59e0b;" title="Render Targets: {s} ({d:.1}%)"></div>
            \\</div>
            \\<div class="chart-legend" style="margin-bottom: 24px;">
            \\  <div class="legend-item"><div class="legend-dot" style="background: #38bdf8;"></div>Текстуры: {s} ({d:.1}%)</div>
            \\  <div class="legend-item"><div class="legend-dot" style="background: #a855f7;"></div>Меши (VBO/IBO): {s} ({d:.1}%)</div>
            \\  <div class="legend-item"><div class="legend-dot" style="background: #f59e0b;"></div>Буферы кадров (RT): {s} ({d:.1}%)</div>
            \\</div>
        , .{ tex_pct, tex_vram_str, tex_pct, mesh_pct, mesh_vram_str, mesh_pct, rt_pct, rt_vram_str, rt_pct, tex_vram_str, tex_pct, mesh_vram_str, mesh_pct, rt_vram_str, rt_pct });
        defer allocator.free(vram_bar_html);
        try buf.appendSlice(allocator, vram_bar_html);

        // Top Textures Table
        try buf.appendSlice(allocator,
            \\<div class="section-title" style="font-size: 16px;">Текстуры (отсортированы по размеру VRAM)</div>
            \\<div class="card-table">
            \\<table>
            \\  <thead>
            \\    <tr>
            \\      <th>Название / Назначение</th>
            \\      <th>Разрешение</th>
            \\      <th>Тип</th>
            \\      <th>Mips</th>
            \\      <th>Формат</th>
            \\      <th class="num">VRAM</th>
            \\    </tr>
            \\  </thead>
            \\  <tbody>
        );

        const sorted_textures = try allocator.dupe(TextureMemoryRecord, mem.textures);
        defer allocator.free(sorted_textures);
        std.mem.sort(TextureMemoryRecord, sorted_textures, {}, struct {
            fn lessThan(_: void, a: TextureMemoryRecord, b: TextureMemoryRecord) bool {
                return a.gpu_bytes > b.gpu_bytes;
            }
        }.lessThan);

        for (sorted_textures) |tex| {
            const tex_sz = try formatBytes(allocator, tex.gpu_bytes);
            defer allocator.free(tex_sz);
            const row = try std.fmt.allocPrint(allocator,
                \\    <tr>
                \\      <td><strong style="color: #fff;">{s}</strong></td>
                \\      <td>{d}x{d}</td>
                \\      <td>{s}</td>
                \\      <td>{d}</td>
                \\      <td>{s}</td>
                \\      <td class="num"><strong>{s}</strong></td>
                \\    </tr>
            , .{ tex.name, tex.width, tex.height, if (tex.is_cube) "Cubemap" else "2D Texture", tex.num_mips, @tagName(tex.format), tex_sz });
            defer allocator.free(row);
            try buf.appendSlice(allocator, row);
        }
        try buf.appendSlice(allocator, "  </tbody>\n</table>\n</div>\n");

        // Top Meshes Table
        try buf.appendSlice(allocator,
            \\<div class="section-title" style="font-size: 16px;">Меши (геометрия сцены)</div>
            \\<div class="card-table">
            \\<table>
            \\  <thead>
            \\    <tr>
            \\      <th>Имя меша</th>
            \\      <th class="num">Вершины</th>
            \\      <th class="num">Индексы</th>
            \\      <th>Тип индексов</th>
            \\      <th class="num">VRAM Буферы</th>
            \\      <th class="num">CPU Память</th>
            \\    </tr>
            \\  </thead>
            \\  <tbody>
        );

        const sorted_meshes = try allocator.dupe(MeshMemoryRecord, mem.meshes);
        defer allocator.free(sorted_meshes);
        std.mem.sort(MeshMemoryRecord, sorted_meshes, {}, struct {
            fn lessThan(_: void, a: MeshMemoryRecord, b: MeshMemoryRecord) bool {
                return a.gpu_bytes > b.gpu_bytes;
            }
        }.lessThan);

        for (sorted_meshes) |m| {
            const gpu_sz = try formatBytes(allocator, m.gpu_bytes);
            defer allocator.free(gpu_sz);
            const cpu_sz = try formatBytes(allocator, m.cpu_bytes);
            defer allocator.free(cpu_sz);

            const row = try std.fmt.allocPrint(allocator,
                \\    <tr>
                \\      <td><strong style="color: #fff;">{s}</strong></td>
                \\      <td class="num">{d}</td>
                \\      <td class="num">{d}</td>
                \\      <td>{s}</td>
                \\      <td class="num"><strong>{s}</strong></td>
                \\      <td class="num" style="color: var(--text-muted);">{s}</td>
                \\    </tr>
            , .{ m.name, m.vertex_count, m.index_count, @tagName(m.index_type), gpu_sz, cpu_sz });
            defer allocator.free(row);
            try buf.appendSlice(allocator, row);
        }
        try buf.appendSlice(allocator, "  </tbody>\n</table>\n</div>\n");

        // Render Targets Table
        try buf.appendSlice(allocator,
            \\<div class="section-title" style="font-size: 16px;">Таргеты рендера (Offscreen Targets)</div>
            \\<div class="card-table">
            \\<table>
            \\  <thead>
            \\    <tr>
            \\      <th>Название таргета</th>
            \\      <th>Разрешение</th>
            \\      <th>MSAA Сэмплы</th>
            \\      <th>Формат</th>
            \\      <th class="num">VRAM</th>
            \\    </tr>
            \\  </thead>
            \\  <tbody>
        );

        for (mem.render_targets) |rt| {
            const rt_sz = try formatBytes(allocator, rt.gpu_bytes);
            defer allocator.free(rt_sz);
            const row = try std.fmt.allocPrint(allocator,
                \\    <tr>
                \\      <td><strong style="color: #fff;">{s}</strong></td>
                \\      <td>{d}x{d}</td>
                \\      <td>{d}x</td>
                \\      <td>{s}</td>
                \\      <td class="num"><strong>{s}</strong></td>
                \\    </tr>
            , .{ rt.name, rt.width, rt.height, rt.samples, @tagName(rt.format), rt_sz });
            defer allocator.free(row);
            try buf.appendSlice(allocator, row);
        }
        try buf.appendSlice(allocator, "  </tbody>\n</table>\n</div>\n");
    }

    // Footer
    try buf.appendSlice(allocator,
        \\<footer>
        \\  Сгенерировано встроенным модулем профилирования Agate Engine &bull; Совместимо с Chrome Trace & Perfetto
        \\</footer>
        \\</div>
        \\</body>
        \\</html>
    );

    return buf.toOwnedSlice(allocator);
}

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
        \\# Agate Engine - Отчет о производительности и памяти
        \\
        \\## 1. Сводка сессии (Session Overview)
        \\
    );

    const vram_total_str = if (memory_ptr) |m| try formatBytes(allocator, m.total_gpu_vram_bytes) else try allocator.dupe(u8, "N/A");
    defer allocator.free(vram_total_str);
    const cpu_mesh_str = if (memory_ptr) |m| try formatBytes(allocator, m.total_cpu_mesh_bytes) else try allocator.dupe(u8, "N/A");
    defer allocator.free(cpu_mesh_str);
    const has_gpu = hasGpuData(frames);
    const has_pass = hasGpuPassData(frames);

    const overview_table = try std.fmt.allocPrint(allocator,
        \\| Метрика | Значение | Метрика | Значение |
        \\| :--- | :--- | :--- | :--- |
        \\| **Всего кадров** | {d} | **Длительность (сумма CPU-submit)** | {d:.2} с |
        \\| **Средний FPS (wall, интервал)** | {d:.1} FPS | **1% Low FPS (wall)** | {d:.1} FPS |
        \\| **Средний интервал (wall)** | {d:.2} мс | **P99 интервал (wall)** | {d:.2} мс |
        \\| **Средний CPU-submit** | {d:.2} мс | **Макс. CPU-submit** | {d:.2} мс |
        \\| **CPU-submit P50 / P95 / P99** | {d:.2} / {d:.2} / {d:.2} мс | **Просадки интервала > 33.3 мс (wall)** | {d} кадров (CPU-submit > 33.3: {d}) |
        \\| **Средний Draw Calls** | {d} | **Макс. Draw Calls** | {d} |
        \\| **Средний треугольников** | {d} | **Макс. треугольников** | {d} |
        \\| **Суммарный VRAM** | {s} | **CPU геометрия** | {s} |
        \\
        \\### Фазы кадра в среднем (CPU-submit, не GPU-время)
        \\- **Update (CPU логика/анимация):** {d:.2} мс
        \\- **Physics (Box3D симуляция):** {d:.2} мс
        \\- **Prepare (подготовка очередей):** {d:.2} мс
        \\- **Shadow Pass (CPU submit, не GPU):** {d:.2} мс
        \\- **Main Pass (CPU submit, не GPU):** {d:.2} мс
        \\- **PostFX (CPU submit, не GPU):** {d:.2} мс
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
            \\| **Средний GPU frame (measured)** | {d:.2} мс | **Макс. GPU frame** | {d:.2} мс |
            \\
            \\GPU-время — последний завершённый семпл, наблюдённый после
            \\коммита (асинхронно, без фиксированного лага; скоуп зависит от
            \\бэкенда и не взаимозаменяем с CPU-submit); фазы выше —
            \\по-прежнему CPU-submit, не GPU.
            \\
        , .{ summary.avg_gpu_frame_ms, summary.max_gpu_frame_ms });
        defer allocator.free(gpu_overview);
        try buf.appendSlice(allocator, gpu_overview);
        if (has_pass) {
            const pass_overview = try std.fmt.allocPrint(allocator,
                \\| **Средний GPU per-pass (measured)** | shadow {d:.2} / main {d:.2} / post {d:.2} мс | **Макс. GPU per-pass** | {d:.2} / {d:.2} / {d:.2} мс |
                \\
                \\Per-pass GPU-время — последние завершённые семплы каждого
                \\прохода; значения разных submission id никогда не
                \\суммируются в длительность кадра.
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

    // Section: "Что не так"
    try buf.appendSlice(allocator,
        \\## 2. Что не так / Автоматическая диагностика узких мест
        \\
    );

    for (findings) |finding| {
        const badge = switch (finding.severity) {
            .good => "🟢 [НОРМА]",
            .info => "ℹ️ [ИНФО]",
            .warning => "⚠️ [ВНИМАНИЕ]",
            .critical => "🚨 [КРИТИЧНО]",
        };
        const f_md = try std.fmt.allocPrint(allocator,
            \\### {s} {s}
            \\- **Описание:** {s}
            \\- **Рекомендация:** {s}
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
            \\## 3. Топ пиковых кадров (Spike Frames, время CPU-submit)
            \\
            \\| Ранг | Кадр # | Время CPU-submit (мс) | GPU measured (мс) | FPS (wall) | Главная причина (CPU-фаза) | Draw Calls | Треугольники | Текстуры | Динамика |
            \\| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
            \\
        );
    } else {
        try buf.appendSlice(allocator,
            \\## 3. Топ пиковых кадров (Spike Frames, время CPU-submit)
            \\
            \\| Ранг | Кадр # | Время CPU-submit (мс) | FPS (wall) | Главная причина (CPU-фаза) | Draw Calls | Треугольники | Текстуры | Динамика |
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
                \\| #{d} | {d} | **{d:.2} мс** | {d:.2} | {d:.1} | {s} ({d:.1} мс, {d:.0}%) | {d} | {d} | {d:.1} KB | {d:.1} KB |
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
                \\| #{d} | {d} | **{d:.2} мс** | {d:.1} | {s} ({d:.1} мс, {d:.0}%) | {d} | {d} | {d:.1} KB | {d:.1} KB |
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
            \\## 4. Использование памяти и VRAM
            \\
        );

        const tex_vram_str = try formatBytes(allocator, mem.textures_vram_bytes);
        defer allocator.free(tex_vram_str);
        const mesh_vram_str = try formatBytes(allocator, mem.meshes_vram_bytes);
        defer allocator.free(mesh_vram_str);
        const rt_vram_str = try formatBytes(allocator, mem.render_targets_vram_bytes);
        defer allocator.free(rt_vram_str);

        const mem_summary = try std.fmt.allocPrint(allocator,
            \\- **Текстуры:** {s}
            \\- **Меши (VBO / IBO):** {s}
            \\- **Буферы кадра (Render Targets):** {s}
            \\- **Суммарно VRAM:** {s}
            \\- **CPU геометрия:** {s}
            \\
            \\### Топ тяжелых текстур
            \\| Название | Разрешение | Тип | Mips | Формат | VRAM |
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
            \\### Топ мешей по памяти
            \\| Меш | Вершины | Индексы | Формат | VRAM | CPU |
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

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

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
            .title = "Все основные параметры в норме",
            .details = "Критических задержек, перерасхода памяти или чрезмерного количества вызовов отрисовки не обнаружено.",
            .recommendation = "Текущая конфигурация сцены работает оптимально.",
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
    try std.testing.expect(std.mem.indexOf(u8, md, "# Agate Engine - Отчет") != null);

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
