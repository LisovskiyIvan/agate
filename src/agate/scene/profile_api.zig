//! Scene profiling/diagnostics API: thin `profiler` forwarders.
//! Split out of `scene.zig` (facade).
//!
/// Anti-cycle rule (same as `audio/*`, `profiler/*`): every function takes
/// the scene as `anytype` (a `*Scene` from `core.zig` in practice) and this
/// module never imports `core.zig` or the `scene.zig` facade back.
/// Cross-leaf helpers consumed here are `pub` in their home module but are
/// deliberately NOT re-exported by the facade.
const profiler_types = @import("../profiler/types.zig");
const MemorySnapshot = profiler_types.MemorySnapshot;

// ---- Profiling & Diagnostics API ----
//
// Ownership (actual update||render): the Profiler is render-owned.
// start/stop/reset/recordFrame/save* run on the CONTEXT thread between
// submissions (the F8 window callback shares that thread — no concurrent
// render there; no broad IO-backend refactor). recordFrame is called only
// at the end of render. captureMemorySnapshot reads LIVE registries
// (meshes/materials/textures), so it additionally requires update
// exclusion (phase lock held) — never from a worker, never concurrently
// with update. The game/update side never touches the Profiler.

/// Starts recording per-frame performance metrics.
pub fn startProfiling(self: anytype) void {
    self.profiler.start();
}

/// Stops recording per-frame performance metrics.
pub fn stopProfiling(self: anytype) void {
    self.profiler.stop();
}

/// Clears any recorded frame history.
pub fn resetProfiling(self: anytype) void {
    self.profiler.reset();
}

/// Returns true if profiling is currently active.
pub fn isProfiling(self: anytype) bool {
    return self.profiler.isRecording();
}

/// Captures a snapshot of current memory allocations (CPU objects & GPU VRAM).
/// Context thread + update excluded (phase lock): reads live registries.
pub fn captureMemorySnapshot(self: anytype) !*const MemorySnapshot {
    return self.profiler.captureMemorySnapshot(self);
}

/// Saves an interactive HTML report to `path`.
pub fn saveProfileReportHtml(self: anytype, path: []const u8) !void {
    try self.profiler.saveReportHtml(self, path);
}

/// Saves a Markdown report to `path`.
pub fn saveProfileReportMd(self: anytype, path: []const u8) !void {
    try self.profiler.saveReportMd(self, path);
}

/// Saves Chrome Trace Event JSON to `path`.
pub fn saveProfileTraceJson(self: anytype, path: []const u8) !void {
    try self.profiler.saveTraceJson(path);
}

/// Saves all reports (HTML, Markdown, Chrome Trace JSON) to `<base_path>.html`,
/// `<base_path>.md`, and `<base_path>.json`.
pub fn saveProfileReports(self: anytype, base_path: []const u8) !void {
    try self.profiler.saveReports(self, base_path);
}
