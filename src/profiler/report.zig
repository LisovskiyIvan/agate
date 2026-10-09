// Standalone report generation over profiler data values.
// Depends only on types.zig (and std): no Scene import, so the
// profiler.zig -> report.zig -> types.zig graph stays acyclic.
// Callers pass already-computed summary/findings/memory values;
// Profiler wrapper methods in profiler.zig capture those first.
//
// Layout: `report_gates.zig` (GPU-availability gates + `formatBytes`),
// `report_html.zig` (HTML), `report_markdown.zig` (Markdown),
// `report_trace.zig` (Chrome-trace JSON). This file is a facade re-exporting
// the public API so callers are unchanged.

const gates_mod = @import("report_gates.zig");
const html_mod = @import("report_html.zig");
const markdown_mod = @import("report_markdown.zig");
const trace_mod = @import("report_trace.zig");

pub const hasGpuData = gates_mod.hasGpuData;
pub const hasGpuPassData = gates_mod.hasGpuPassData;
pub const formatBytes = gates_mod.formatBytes;
pub const generateReportHtml = html_mod.generateReportHtml;
pub const generateReportMd = markdown_mod.generateReportMd;
pub const generateTraceJson = trace_mod.generateTraceJson;
