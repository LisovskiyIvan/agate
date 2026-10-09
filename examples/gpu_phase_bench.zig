//! Shared three-phase in-window GPU A/B bench (MEASUREMENTS.md waves 3+).
//!
//! One sokol process/window walks 2..3 render-scale phases back to back —
//! the only compare that counts is same-session back-to-back. Each phase
//! skips a fixed warmup, keeps every completed GPU frame sample, counts
//! occluded/minimized frames apart (they render but never sample), and
//! reports the MEDIAN as the canonical statistic (run-to-run means spread
//! +-20% under window-manager jitter; see the harness notes in
//! ../../MEASUREMENTS.md).
//!
//! Host contract (hdr-showcase, q0-interior, ...):
//!   - `bench.noteEvent` from the sokol event callback (iconify/restore);
//!   - `bench.beginPhase(frame, frame_limit)` every frame BEFORE render:
//!     it returns the finished phase's report exactly once per boundary,
//!     flips `currentScale()` (the host applies it to
//!     `scene.post_process.render_scale` and resets TAA), and resets the
//!     warmup/sample bookkeeping;
//!   - `bench.recordSample` once per frame right AFTER `scene.render()`
//!     (the engine writes `stats.gpu_frame_*` at the end of render and the
//!     next staged begin resets `stats` — poll anywhere else and you read
//!     zeros);
//!   - `bench.finalReport()` from cleanup for the last phase.
//!
//! No allocations: the sample store is a fixed static buffer.

const std = @import("std");
const sokol = @import("sokol");
const sapp = sokol.app;

pub const max_samples: usize = 4096;

pub const Phase = struct {
    index: usize,
    scale: f32,
    n: usize,
    occluded: u32,
    mean_ms: f64,
    median_ms: f32,
    p95_ms: f32,
    min_ms: f32,
    max_ms: f32,
    /// Realized main-target size while this phase ran (the host records it
    /// via `noteTargetSize` every frame). Nonzero `target_changes` marks a
    /// mid-phase resize (window-manager settle): the median is not
    /// comparable across such a phase.
    target_w: i32 = 0,
    target_h: i32 = 0,
    target_changes: u32 = 0,
};

pub const Bench = struct {
    scales: [3]f32 = .{ 1.0, 0.75, 0.66 },
    phase_frames: u32 = 120,
    warmup: u32 = 30,

    index: usize = 0,
    store: [max_samples]f32 = undefined,
    len: usize = 0,
    warmup_seen: u32 = 0,
    occluded: u32 = 0,
    window_hidden: bool = false,
    target_w: i32 = 0,
    target_h: i32 = 0,
    /// True on the frame a phase switched (the host prints target size then).
    phase_started: bool = false,
    // Target-size stability: the window manager can settle/resize the
    // window mid-phase (macOS "zooms" fresh windows into the work area).
    // A phase whose target changed is reported UNSTABLE — its median is
    // not comparable.
    prev_w: i32 = 0,
    prev_h: i32 = 0,
    target_changes: u32 = 0,

    pub fn currentScale(self: *const Bench) f32 {
        return self.scales[self.index];
    }

    /// Phase boundary check for frame `f` (1-based). Call BEFORE render.
    /// `frame_limit` gates the whole run; when it is shorter than the full
    /// schedule the bench stays on phase 0 and the host's own scale choice
    /// (env/CLI) applies via `overrideScale`.
    pub fn beginPhase(self: *Bench, f: u32, frame_limit: u32) ?Phase {
        self.phase_started = false;
        if (frame_limit == 0) return null;
        if (f <= 1) return null;
        if ((f - 1) % self.phase_frames != 0) return null;
        if (f - 1 >= self.scales.len * self.phase_frames) return null;
        const finished = self.report(self.index);
        self.index = (f - 1) / self.phase_frames;
        self.warmup_seen = 0;
        self.len = 0;
        self.target_changes = 0;
        self.phase_started = true;
        return finished;
    }

    /// Keeps phase 0 at a host-chosen scale (env/CLI single-scale runs).
    pub fn overrideScale(self: *Bench, scale: f32) void {
        if (self.index == 0) self.scales[0] = scale;
    }

    pub fn noteEvent(self: *Bench, ev: sapp.Event) void {
        switch (ev.type) {
            .ICONIFIED, .SUSPENDED => self.window_hidden = true,
            .RESTORED, .RESUMED => self.window_hidden = false,
            else => {},
        }
    }

    /// Call EVERY frame (cheap): keeps the phase's realized size current
    /// and counts mid-phase size changes for the stability flag.
    pub fn noteTargetSize(self: *Bench, w: i32, h: i32) void {
        if (self.prev_w != 0 and (self.prev_w != w or self.prev_h != h)) self.target_changes += 1;
        self.prev_w = w;
        self.prev_h = h;
        self.target_w = w;
        self.target_h = h;
    }

    /// One completed GPU frame. `submit != 0` is availability (a valid
    /// quantized zero still counts); degenerate windows count as occluded.
    pub fn recordSample(self: *Bench, ms: f32, submit: u32) void {
        if (self.window_hidden or sapp.width() <= 1 or sapp.height() <= 1) {
            self.occluded += 1;
            return;
        }
        if (submit == 0) return;
        if (!(ms >= 0 and std.math.isFinite(ms))) return;
        if (self.warmup_seen < self.warmup) {
            self.warmup_seen += 1;
        } else if (self.len < max_samples) {
            self.store[self.len] = ms;
            self.len += 1;
        }
    }

    pub fn report(self: *Bench, idx: usize) Phase {
        const n = self.len;
        if (n == 0) {
            return .{ .index = idx, .scale = self.scales[idx], .n = 0, .occluded = self.occluded, .mean_ms = 0, .median_ms = 0, .p95_ms = 0, .min_ms = 0, .max_ms = 0, .target_w = self.target_w, .target_h = self.target_h, .target_changes = self.target_changes };
        }
        const sorted = self.store[0..n];
        std.mem.sort(f32, sorted, {}, std.sort.asc(f32));
        var sum: f64 = 0;
        for (sorted) |v| sum += v;
        return .{
            .index = idx,
            .scale = self.scales[idx],
            .n = n,
            .occluded = self.occluded,
            .mean_ms = sum / @as(f64, @floatFromInt(n)),
            .median_ms = sorted[n / 2],
            .p95_ms = sorted[@min(n - 1, (n * 95) / 100)],
            .min_ms = sorted[0],
            .max_ms = sorted[n - 1],
            .target_w = self.target_w,
            .target_h = self.target_h,
            .target_changes = self.target_changes,
        };
    }

    pub fn printPhase(p: Phase) void {
        if (p.n == 0) {
            std.debug.print("phase {}: scale={d:.2} no samples (occl {}) target={}x{} window={}x{}\n", .{ p.index, p.scale, p.occluded, p.target_w, p.target_h, sapp.width(), sapp.height() });
            return;
        }
        std.debug.print("phase {}: scale={d:.2} target={}x{} window={}x{} n={} occl={} mean={d:.3} median={d:.3} p95={d:.3} min={d:.3} max={d:.3}\n", .{
            p.index,  p.scale,    p.target_w, p.target_h,  sapp.width(), sapp.height(),
            p.n,      p.occluded, p.mean_ms,  p.median_ms, p.p95_ms,     p.min_ms,
            p.max_ms,
        });
    }
};
