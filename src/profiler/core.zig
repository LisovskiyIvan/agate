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
const report_queue = @import("report_queue.zig");
const jobs = @import("../jobs.zig");
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
    ring_head: usize = 0,
    wrapped: bool = false,
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

    /// Rotates the internal ring buffer in-place so frames are in strict chronological order.
    pub fn linearize(self: *Profiler) void {
        if (!self.wrapped or self.ring_head == 0 or self.frames.items.len == 0) return;
        std.mem.rotate(FrameRecord, self.frames.items, self.ring_head);
        self.ring_head = 0;
        self.wrapped = false;
    }

    /// Stops recording frames and ensures frames are in chronological order.
    pub fn stop(self: *Profiler) void {
        self.is_recording = false;
        self.linearize();
    }

    /// Clears recorded frame history while maintaining configuration.
    pub fn reset(self: *Profiler) void {
        self.frames.clearRetainingCapacity();
        self.start_time_ticks = 0;
        self.last_frame_ticks = 0;
        self.ring_head = 0;
        self.wrapped = false;
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
        @constCast(self).linearize();
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
    /// `<base_path>.md`, and `<base_path>.json`. Synchronous: the caller
    /// holds whatever exclusion the capture needs (or none, when the
    /// registries are quiesced); file IO happens inline before return.
    pub fn saveReports(self: *const Profiler, scene: ?*const Scene, base_path: []const u8) !void {
        if (scene) |sc| {
            var mut_prof = @constCast(self);
            _ = try mut_prof.captureMemorySnapshot(sc);
        }
        var bundle = try self.generateReportsAlloc(self.allocator);
        defer bundle.deinit(self.allocator);
        const io = std.Io.Threaded.global_single_threaded.io();
        const html_path = try std.fmt.allocPrint(self.allocator, "{s}.html", .{base_path});
        defer self.allocator.free(html_path);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = html_path, .data = bundle.html });

        const md_path = try std.fmt.allocPrint(self.allocator, "{s}.md", .{base_path});
        defer self.allocator.free(md_path);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = md_path, .data = bundle.md });

        const json_path = try std.fmt.allocPrint(self.allocator, "{s}.json", .{base_path});
        defer self.allocator.free(json_path);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = json_path, .data = bundle.json });
    }

    /// Owned encoded reports (HTML + Markdown + Chrome-trace JSON).
    /// Produced by `generateReportsAlloc`; each field moves into its
    /// `ReportWriteTask` on the async path, or is written + freed inline on
    /// the sync path above.
    pub const ReportBundle = struct {
        html: []u8 = &.{},
        md: []u8 = &.{},
        json: []u8 = &.{},

        pub fn deinit(self: *ReportBundle, allocator: std.mem.Allocator) void {
            if (self.html.len > 0) allocator.free(self.html);
            if (self.md.len > 0) allocator.free(self.md);
            if (self.json.len > 0) allocator.free(self.json);
            self.* = .{};
        }
    };

    /// Which report files an async save covers. The F8 full-save selects
    /// all; the HTML-only / trace-only buttons select one.
    pub const ReportFiles = struct {
        html: bool = true,
        md: bool = true,
        json: bool = true,

        pub const all: ReportFiles = .{};
        pub const html_only: ReportFiles = .{ .md = false, .json = false };
        pub const trace_only: ReportFiles = .{ .html = false, .md = false };
    };

    /// Encodes all reports over already-frozen state (recorded frames plus
    /// the last captured memory snapshot). No live-registry reads, no
    /// capture, no file IO — safe OFF the phase lock: the game thread never
    /// touches the profiler, and only the context thread mutates it,
    /// sequentially. The caller captures first (under exclusion when live)
    /// and enqueues or writes the bundle afterwards.
    pub fn generateReportsAlloc(self: *const Profiler, allocator: std.mem.Allocator) !ReportBundle {
        const memory_ptr: ?*const MemorySnapshot = if (self.last_memory_snapshot) |*s| s else null;
        const summary = self.summarize();
        const findings = try self.analyze(memory_ptr, allocator);
        defer {
            for (findings) |*f| @constCast(f).deinit(allocator);
            allocator.free(findings);
        }
        const html = try report.generateReportHtml(self.frames.items, summary, findings, memory_ptr, allocator);
        errdefer allocator.free(html);
        const md = try report.generateReportMd(self.frames.items, summary, findings, memory_ptr, allocator);
        errdefer allocator.free(md);
        const json = try report.generateTraceJson(self.frames.items, allocator);
        errdefer allocator.free(json);
        return .{ .html = html, .md = md, .json = json };
    }

    /// Async report save: encodes (see `generateReportsAlloc`) and enqueues
    /// one `ReportWriteTask` per selected file on `runner` (in practice
    /// `Scene.io_runner`). Returns after the enqueue — no file IO happens
    /// before return, so a bounded exclusion window may call this and
    /// release the mutex immediately. Each `out` slot holds the posted task
    /// (null when its file was not selected); the caller polls `isDone()`
    /// and `deinit()`s every non-null slot. On error the not-yet-enqueued
    /// bytes are freed here; already-posted tasks complete independently
    /// and stay pollable through the filled `out` slots.
    pub fn enqueueReportWrites(
        self: *const Profiler,
        runner: *jobs.TaskRunner,
        base_path: []const u8,
        files: ReportFiles,
        out: *[3]?*report_queue.ReportWriteTask,
    ) !void {
        out.* = .{ null, null, null };
        const bundle = try self.generateReportsAlloc(self.allocator);
        // Each bundle field moves into `datas` and then into its task; the
        // bundle struct itself is never deinited after the move (that would
        // double-free). The errdefer below frees whatever has not moved yet.
        var datas = [_][]u8{ bundle.html, bundle.md, bundle.json };
        errdefer {
            for (datas) |d| if (d.len > 0) self.allocator.free(d);
        }
        const want = [_]bool{ files.html, files.md, files.json };
        const exts = [_][]const u8{ ".html", ".md", ".json" };
        for (0..3) |i| {
            if (!want[i]) {
                if (datas[i].len > 0) self.allocator.free(datas[i]);
                datas[i] = &.{};
                continue;
            }
            const path = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ base_path, exts[i] });
            errdefer self.allocator.free(path);
            out[i] = try report_queue.enqueueReportWrite(self.allocator, runner, path, datas[i]);
            self.allocator.free(path); // enqueue dupes the path
            datas[i] = &.{};
        }
    }
};
