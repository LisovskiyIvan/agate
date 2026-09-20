//! Profiler state and lifecycle. Split out of `profiler.zig` (facade).
//!
//! This module owns the `Profiler` type: the flight-recorder fields, the
//! trivial lifecycle (`init`/`deinit`/`setMaxFrames`/`start`/`stop`/`reset`/
//! `isRecording`) and the report/save glue that already delegated to
//! `report.zig`. The heavy method bodies live in focused siblings and are
//! reached through thin forwarders, so `prof.recordFrame(...)`,
//! `prof.summarize()` and every other call site keep working unchanged:
//!
//! - `recording.zig` — `recordFrame` (per-frame ring-buffer insert).
//! - `snapshot.zig` — `captureMemorySnapshot` (CPU/GPU memory census).
//! - `summary.zig` — `summarize` (session statistics + pacing).
//! - `diagnostics.zig` — `analyze` (bottleneck findings).
//!
//! Anti-cycle rule (same as `scene/`, `ui/`): siblings take the profiler as
//! `anytype` and never import this module or the `profiler.zig` facade back;
//! this module passes `self` straight through. `profiler.zig` re-exports
//! `Profiler` under its historical path.

const std = @import("std");
const sokol = @import("sokol");

const types = @import("types.zig");
const report = @import("report.zig");
const recording = @import("recording.zig");
const snapshot = @import("snapshot.zig");
const summary_mod = @import("summary.zig");
const diagnostics = @import("diagnostics.zig");
const stats_mod = @import("../scene/stats.zig");
const scene_mod = @import("../scene.zig");

const SceneStats = stats_mod.SceneStats;
const Scene = scene_mod.Scene;
const FrameRecord = types.FrameRecord;
const MemorySnapshot = types.MemorySnapshot;
const SessionSummary = types.SessionSummary;
const DiagnosticFinding = types.DiagnosticFinding;

pub const Profiler = struct {
    allocator: std.mem.Allocator,
    is_recording: bool = false,
    frames: std.ArrayListUnmanaged(FrameRecord) = .empty,
    max_frames: usize = 3600, // 60 seconds @ 60 FPS default
    start_time_ticks: u64 = 0,
    last_frame_ticks: u64 = 0,
    last_memory_snapshot: ?MemorySnapshot = null,

    pub fn init(allocator: std.mem.Allocator) Profiler {
        return .{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Profiler) void {
        self.frames.deinit(self.allocator);
        if (self.last_memory_snapshot) |*snap| {
            snap.deinit(self.allocator);
            self.last_memory_snapshot = null;
        }
        self.* = undefined;
    }

    /// Sets the maximum number of frames retained in the recording ring buffer.
    pub fn setMaxFrames(self: *Profiler, max: usize) void {
        self.max_frames = @max(1, max);
    }

    /// Begins recording frames.
    /// Precondition: the app-owned sokol.time clock is already initialized
    /// exactly once before any thread reads it (main.zig init calls
    /// sokol.time.setup() ahead of the game-thread spawn). Never
    /// re-initializes the clock here: stm_setup memsets the global origin,
    /// so calling it per start would reset time.now for concurrent readers
    /// (gameLoop) and move previously taken anchors backward.
    pub fn start(self: *Profiler) void {
        self.is_recording = true;
        self.start_time_ticks = sokol.time.now();
        self.last_frame_ticks = self.start_time_ticks;
    }

    /// Stops recording frames.
    pub fn stop(self: *Profiler) void {
        self.is_recording = false;
    }

    /// Clears recorded frame history while maintaining configuration.
    pub fn reset(self: *Profiler) void {
        self.frames.clearRetainingCapacity();
        self.start_time_ticks = 0;
        self.last_frame_ticks = 0;
        if (self.last_memory_snapshot) |*old| old.deinit(self.allocator);
        self.last_memory_snapshot = null;
    }

    /// Returns true if currently recording frames.
    pub fn isRecording(self: *const Profiler) bool {
        return self.is_recording;
    }

    /// Records metrics for the completed frame (see recording.zig).
    pub fn recordFrame(self: *Profiler, frame_id: u64, stats: *const SceneStats) void {
        recording.recordFrame(self, frame_id, stats);
    }

    /// Captures a complete snapshot of CPU memory and GPU VRAM allocations
    /// in the scene (see snapshot.zig).
    pub fn captureMemorySnapshot(self: *Profiler, scene: *const Scene) !*const MemorySnapshot {
        return snapshot.captureMemorySnapshot(self, scene);
    }

    /// Computes statistical summary across all recorded frames
    /// (see summary.zig).
    pub fn summarize(self: *const Profiler) SessionSummary {
        return summary_mod.summarize(self);
    }

    /// Runs automated bottleneck diagnostics on recorded metrics and memory
    /// (see diagnostics.zig).
    pub fn analyze(self: *const Profiler, memory: ?*const MemorySnapshot, allocator: std.mem.Allocator) ![]DiagnosticFinding {
        return diagnostics.analyze(self, memory, allocator);
    }

    /// Helper to format byte values (e.g. "12.4 MB").
    pub fn formatBytes(allocator: std.mem.Allocator, bytes: usize) ![]u8 {
        return report.formatBytes(allocator, bytes);
    }

    /// Generates a standalone, beautiful, dark-themed HTML report.
    pub fn generateReportHtml(self: *const Profiler, scene: ?*const Scene, allocator: std.mem.Allocator) ![]u8 {
        // Ensure memory snapshot is captured if scene is provided
        const memory_ptr: ?*const MemorySnapshot = if (scene) |sc| blk: {
            var mut_prof = @constCast(self);
            break :blk mut_prof.captureMemorySnapshot(sc) catch null;
        } else (if (self.last_memory_snapshot) |*s| s else null);

        const summary = self.summarize();
        const findings = try self.analyze(memory_ptr, allocator);
        defer {
            for (findings) |*f| @constCast(f).deinit(allocator);
            allocator.free(findings);
        }

        return report.generateReportHtml(self.frames.items, summary, findings, memory_ptr, allocator);
    }

    /// Generates a comprehensive Markdown report.
    pub fn generateReportMd(self: *const Profiler, scene: ?*const Scene, allocator: std.mem.Allocator) ![]u8 {
        const memory_ptr: ?*const MemorySnapshot = if (scene) |sc| blk: {
            var mut_prof = @constCast(self);
            break :blk mut_prof.captureMemorySnapshot(sc) catch null;
        } else (if (self.last_memory_snapshot) |*s| s else null);

        const summary = self.summarize();
        const findings = try self.analyze(memory_ptr, allocator);
        defer {
            for (findings) |*f| @constCast(f).deinit(allocator);
            allocator.free(findings);
        }

        return report.generateReportMd(self.frames.items, summary, findings, memory_ptr, allocator);
    }

    /// Generates Chrome Trace Event JSON for chrome://tracing and ui.perfetto.dev.
    pub fn generateTraceJson(self: *const Profiler, allocator: std.mem.Allocator) ![]u8 {
        return report.generateTraceJson(self.frames.items, allocator);
    }

    /// Saves an interactive HTML report to `path`.
    pub fn saveReportHtml(self: *const Profiler, scene: ?*const Scene, path: []const u8) !void {
        const html = try self.generateReportHtml(scene, self.allocator);
        defer self.allocator.free(html);
        const io = std.Io.Threaded.global_single_threaded.io();
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = html });
    }

    /// Saves a Markdown report to `path`.
    pub fn saveReportMd(self: *const Profiler, scene: ?*const Scene, path: []const u8) !void {
        const md = try self.generateReportMd(scene, self.allocator);
        defer self.allocator.free(md);
        const io = std.Io.Threaded.global_single_threaded.io();
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = md });
    }

    /// Saves Chrome Trace Event JSON to `path`.
    pub fn saveTraceJson(self: *const Profiler, path: []const u8) !void {
        const json = try self.generateTraceJson(self.allocator);
        defer self.allocator.free(json);
        const io = std.Io.Threaded.global_single_threaded.io();
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = json });
    }

    /// Saves all reports (HTML, Markdown, and Chrome Trace JSON) to `<base_path>.html`,
    /// `<base_path>.md`, and `<base_path>.json`.
    pub fn saveReports(self: *const Profiler, scene: ?*const Scene, base_path: []const u8) !void {
        if (scene) |sc| {
            var mut_prof = @constCast(self);
            _ = try mut_prof.captureMemorySnapshot(sc);
        }
        const html_path = try std.fmt.allocPrint(self.allocator, "{s}.html", .{base_path});
        defer self.allocator.free(html_path);
        try self.saveReportHtml(null, html_path);

        const md_path = try std.fmt.allocPrint(self.allocator, "{s}.md", .{base_path});
        defer self.allocator.free(md_path);
        try self.saveReportMd(null, md_path);

        const json_path = try std.fmt.allocPrint(self.allocator, "{s}.json", .{base_path});
        defer self.allocator.free(json_path);
        try self.saveTraceJson(json_path);
    }
};

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

/// Sleep helper for pacing-sensitive tests (Zig 0.16 has no Thread.sleep).
fn testSleepMs(ms: u64) void {
    const ts = std.c.timespec{ .sec = @intCast(ms / 1000), .nsec = @intCast((ms % 1000) * 1_000_000) };
    var rem: std.c.timespec = undefined;
    _ = std.c.nanosleep(&ts, &rem);
}

test "Profiler restart preserves monotonic app clock" {
    // Timer is app-owned, initialized once before threads (main.zig init).
    // Standalone unit context: single-threaded explicit setup.
    sokol.time.setup();
    // Let the anchor sit comfortably above call overhead so a clock origin
    // reset (old Profiler.start calling setup) shows up as a deterministic
    // backward step, not timer granularity noise.
    testSleepMs(5);
    const anchor = sokol.time.now();
    try std.testing.expect(anchor > 0);

    const ally = std.testing.allocator;
    var prof = Profiler.init(ally);
    defer prof.deinit();

    prof.start();
    const t1 = sokol.time.now();
    // Source clock must not reset on start: still at/after pre-start anchor.
    try std.testing.expect(t1 >= anchor);
    try std.testing.expect(prof.start_time_ticks >= anchor);

    testSleepMs(2);
    prof.stop();
    prof.start();
    const t2 = sokol.time.now();
    try std.testing.expect(t2 >= t1);
    try std.testing.expect(t2 >= anchor);
    try std.testing.expect(prof.start_time_ticks >= t1);
    prof.stop();
}

test "Profiler start/stop never moves app clock backward for concurrent readers" {
    sokol.time.setup();
    testSleepMs(5);

    const ally = std.testing.allocator;
    var prof = Profiler.init(ally);
    defer prof.deinit();

    const Ctx = struct {
        ready: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        backward: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
        fn run(c: *@This()) void {
            var last = sokol.time.now();
            c.ready.store(true, .release);
            while (!c.stop.load(.acquire)) {
                const now = sokol.time.now();
                if (now < last) _ = c.backward.fetchAdd(1, .monotonic);
                last = now;
                std.atomic.spinLoopHint();
            }
        }
    };
    var ctx = Ctx{};
    const reader = try std.Thread.spawn(.{}, Ctx.run, .{&ctx});
    // Wait until the reader holds a pre-restart anchor so every restart
    // below races a live read (otherwise a fast restart loop can finish
    // before the thread's first read and miss the window entirely).
    while (!ctx.ready.load(.acquire)) std.atomic.spinLoopHint();
    // Only the context thread touches the Profiler; the worker only reads
    // the shared app clock, mirroring gameLoop's time.now use.
    var i: usize = 0;
    while (i < 200) : (i += 1) {
        prof.start();
        var k: usize = 0;
        while (k < 200) : (k += 1) std.atomic.spinLoopHint();
        prof.stop();
    }
    ctx.stop.store(true, .release);
    reader.join();
    try std.testing.expectEqual(@as(u32, 0), ctx.backward.load(.acquire));
}
