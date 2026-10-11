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
//
//! Wave 5 adds `AbBench` (same file, additive): an interleaved A/B over
//! two host-provided configs in alternating A/B blocks (cycle A B B A
//! B A A B), with occlusion gating, transition-frame discard, block
//! medians, a flip check, and a SIGNIFICANT/INCONCLUSIVE verdict.

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

// ---- interleaved A/B (wave 5; additive, the phase Bench above is untouched)
//
// Why this exists: back-to-back phases (all of A, then all of B) confound
// the config delta with session drift — waves 3/4 watched the LAST phase
// run slowest at ANY scale. Alternating A/B in short BLOCKS inside one
// window/session puts both arms through the same drift, so the pooled
// delta cancels linear drift to first order. Blocks (tens of frames), not
// per-frame toggling: caches/pipelines stay warm and one TAA reset +
// a few discarded transition frames amortize the switch cost.
//
// The bench stays generic: it only deals in arms A/B. The host maps a
// block event to its config pair (q0-interior: SSGI on/off — a plain
// composite-shader bool, no target rebuild, so it is safe mid-block;
// a render_scale pair would realloc targets every block and belongs in
// the phase bench, NOT here).
//
// Verdict rule (heuristic, not a statistical test — documented in
// MEASUREMENTS.md wave 5): SIGNIFICANT iff ALL hold —
//   1. no dropped blocks (every measured block ran clean at the run's
//      modal target size; a mid-run resize drops blocks and vetoes),
//   2. >= 2 kept blocks per arm,
//   3. >= 30 pooled samples per arm,
//   4. |pooled-median(B) - pooled-median(A)| exceeds the block-to-block
//      noise floor = max(range of A block medians, range of B block medians),
//   5. the interleave flip check passes: every consecutive kept-block
//      mixed pair (AB or BA sliding window) agrees in sign with the
//      pooled delta.
// Otherwise INCONCLUSIVE with the first failing reason. A single noisy
// pair flipping sign, a mid-run resize, or drift-scale block ranges all
// force INCONCLUSIVE rather than a false positive.

/// Block-cycle position: A B B A B A A B repeating. Each adjacent pair
/// is mixed (AB or BA), and each arm appears equally often in even and
/// odd slots, so linear drift cancels to first order in the pooled delta.
pub fn abPatternIsB(slot: usize) bool {
    const table = [_]bool{ false, true, true, false, true, false, false, true };
    return table[slot % table.len];
}

pub const ab_max_blocks: usize = 64;
pub const ab_max_block_frames: usize = 256;
pub const ab_min_samples_per_arm: usize = 30;

pub const AbArm = enum { A, B };

pub const AbBlockEvent = struct {
    index: usize,
    arm: AbArm,
    /// Convenience mirror of `arm == .B`.
    is_b: bool,
};

pub const AbBlockRec = struct {
    is_b: bool,
    median_ms: f32,
    n: usize,
    /// Offset of this block's samples in its arm store (for size-filtered
    /// pooling at summary time).
    start: usize,
    /// Main-target size at block close; `size_clean` means the size never
    /// changed between block start and close.
    w: i32,
    h: i32,
    size_clean: bool,
};

pub const AbVerdict = enum { SIGNIFICANT, INCONCLUSIVE };

pub const AbSummary = struct {
    a_name: []const u8,
    b_name: []const u8,
    n_a: usize,
    n_b: usize,
    occluded_a: u32,
    occluded_b: u32,
    blocks_a: usize,
    blocks_b: usize,
    mean_a: f64,
    mean_b: f64,
    median_a: f32,
    median_b: f32,
    p95_a: f32,
    p95_b: f32,
    /// B minus A (negative = B faster).
    delta_median_ms: f32,
    delta_p95_ms: f32,
    noise_floor_ms: f32,
    flip_pairs: u32,
    flip_agree: u32,
    flip_ok: bool,
    unstable: bool,
    /// Measured blocks excluded from every statistic (wrong size or
    /// size changed mid-block). Any nonzero count vetoes SIGNIFICANT.
    dropped: u32,
    verdict: AbVerdict,
    reason: []const u8,
};

pub const AbBench = struct {
    block_frames: u32 = 60,
    /// Frames discarded at each block start (TAA reset + switch cost).
    transition_skip: u32 = 5,
    /// Frames discarded at run start (cold pipelines/history).
    first_warmup: u32 = 30,
    /// Leading blocks that run their config (pipelines warm, TAA resets)
    /// but record NOTHING: the window manager settles/resizes fresh
    /// windows in the first ~1-2 blocks (720x540 -> 728x882 on macOS),
    /// and samples at two pixel counts are not comparable. Skipping the
    /// first two slots rotates the measured schedule to pattern slots
    /// 2..9 — still 4A/4B with every adjacent pair mixed (B A | B A |
    /// A B | A B) — so the flip check keeps full strength. A resize
    /// AFTER the settle window still marks the run UNSTABLE.
    settle_blocks: u32 = 2,

    index: usize = 0,
    store_a: [max_samples]f32 = undefined,
    store_b: [max_samples]f32 = undefined,
    len_a: usize = 0,
    len_b: usize = 0,
    /// Arm-store offset where the open block started (for its median).
    arm_start: usize = 0,
    blocks: [ab_max_blocks]AbBlockRec = undefined,
    n_blocks: usize = 0,
    occluded_a: u32 = 0,
    occluded_b: u32 = 0,
    window_hidden: bool = false,
    /// True on the frame a block switched (host applies config + TAA reset).
    block_started: bool = false,
    target_w: i32 = 0,
    target_h: i32 = 0,
    prev_w: i32 = 0,
    prev_h: i32 = 0,
    target_changes: u32 = 0,
    /// Stability snapshot at the open block's start (for size_clean).
    changes_at_block_start: u32 = 0,
    start_w: i32 = 0,
    start_h: i32 = 0,
    scratch: [ab_max_block_frames]f32 = undefined,
    tmp: [max_samples]f32 = undefined,

    fn armIsB(self: *const AbBench, slot: usize) bool {
        _ = self;
        return abPatternIsB(slot);
    }

    fn skipFor(self: *const AbBench, slot: usize) u32 {
        return if (slot == 0) self.first_warmup else self.transition_skip;
    }

    /// Block boundary check for frame `f` (1-based). Call BEFORE render.
    /// Returns an event on block 0 (f == 1) and on every block switch so
    /// the host can apply its A/B config. Closes the finished block's
    /// median at each switch. Frames past `frame_limit` never start a
    /// block (the host quits there); the open block closes in
    /// `finalSummary`.
    pub fn beginBlock(self: *AbBench, f: u32, frame_limit: u32) ?AbBlockEvent {
        self.block_started = false;
        if (frame_limit == 0 or f == 0) return null;
        const slot = (f - 1) / self.block_frames;
        if (f == 1) {
            self.index = 0;
            self.arm_start = 0;
            self.block_started = true;
            self.changes_at_block_start = self.target_changes;
            self.start_w = self.target_w;
            self.start_h = self.target_h;
            const is_b = self.armIsB(0);
            return .{ .index = 0, .arm = if (is_b) .B else .A, .is_b = is_b };
        }
        if (slot == self.index) return null;
        if ((f - 1) >= frame_limit) return null;
        self.closeBlock();
        self.index = slot;
        const is_b = self.armIsB(slot);
        self.arm_start = if (is_b) self.len_b else self.len_a;
        self.changes_at_block_start = self.target_changes;
        self.start_w = self.target_w;
        self.start_h = self.target_h;
        self.block_started = true;
        return .{ .index = slot, .arm = if (is_b) .B else .A, .is_b = is_b };
    }

    pub fn noteEvent(self: *AbBench, ev: sapp.Event) void {
        switch (ev.type) {
            .ICONIFIED, .SUSPENDED => self.window_hidden = true,
            .RESTORED, .RESUMED => self.window_hidden = false,
            else => {},
        }
    }

    pub fn noteTargetSize(self: *AbBench, w: i32, h: i32) void {
        if (self.prev_w != 0 and (self.prev_w != w or self.prev_h != h)) self.target_changes += 1;
        self.prev_w = w;
        self.prev_h = h;
        self.target_w = w;
        self.target_h = h;
    }

    fn occludedNow(self: *const AbBench) bool {
        return self.window_hidden or sapp.width() <= 1 or sapp.height() <= 1;
    }

    /// One completed GPU frame for the CURRENT block's arm. Position in
    /// block derives from the frame counter (`posInBlock`), so no extra
    /// per-frame state: settle blocks, transition frames and occluded
    /// frames never sample (occluded tallies per arm — unequal occlusion
    /// is visible).
    pub fn recordSample(self: *AbBench, f: u32, ms: f32, submit: u32) void {
        if (self.index < self.settle_blocks) return;
        const is_b = self.armIsB(self.index);
        if (self.occludedNow()) {
            if (is_b) {
                self.occluded_b += 1;
            } else {
                self.occluded_a += 1;
            }
            return;
        }
        if (submit == 0) return;
        if (!(ms >= 0 and std.math.isFinite(ms))) return;
        const pos = (f - 1) % self.block_frames;
        if (pos < self.skipFor(self.index)) return;
        if (is_b) {
            if (self.len_b < max_samples) {
                self.store_b[self.len_b] = ms;
                self.len_b += 1;
            }
        } else {
            if (self.len_a < max_samples) {
                self.store_a[self.len_a] = ms;
                self.len_a += 1;
            }
        }
    }

    /// Median of the open block's samples so far; pushes one block
    /// record (empty blocks leave no record and break no pairing).
    /// Settle blocks leave no record either.
    fn closeBlock(self: *AbBench) void {
        if (self.index < self.settle_blocks) return;
        const is_b = self.armIsB(self.index);
        const store = if (is_b) self.store_b[0..self.len_b] else self.store_a[0..self.len_a];
        if (store.len <= self.arm_start) return;
        const slice = store[self.arm_start..];
        const n = @min(slice.len, self.scratch.len);
        @memcpy(self.scratch[0..n], slice[0..n]);
        const buf = self.scratch[0..n];
        std.mem.sort(f32, buf, {}, std.sort.asc(f32));
        if (self.n_blocks < ab_max_blocks) {
            self.blocks[self.n_blocks] = .{
                .is_b = is_b,
                .median_ms = buf[n / 2],
                .n = n,
                .start = self.arm_start,
                .w = self.target_w,
                .h = self.target_h,
                .size_clean = self.target_changes == self.changes_at_block_start and
                    self.target_w == self.start_w and self.target_h == self.start_h,
            };
            self.n_blocks += 1;
        }
    }

    fn pooledStats(store: []f32) struct { mean: f64, median: f32, p95: f32 } {
        const n = store.len;
        std.mem.sort(f32, store, {}, std.sort.asc(f32));
        var sum: f64 = 0;
        for (store) |v| sum += v;
        return .{
            .mean = sum / @as(f64, @floatFromInt(@max(n, 1))),
            .median = store[n / 2],
            .p95 = store[@min(n - 1, (n * 95) / 100)],
        };
    }

    /// Closes the trailing block and compares the arms. Only clean
    /// modal-size blocks enter the statistics: a block whose target size
    /// differs from the run's modal size, or changed mid-block, is
    /// dropped from every number (pooled stats gather the kept ranges;
    /// flip/noise walk the kept blocks in temporal order). Any drop sets
    /// unstable and vetoes SIGNIFICANT — the phase bench's UNSTABLE rule,
    /// generalized to interleaving. Sorts the scratch in place (end of
    /// run; block medians were taken earlier from fixed ranges).
    pub fn finalSummary(self: *AbBench, a_name: []const u8, b_name: []const u8) AbSummary {
        self.closeBlock();
        var s: AbSummary = .{
            .a_name = a_name,
            .b_name = b_name,
            .n_a = 0,
            .n_b = 0,
            .occluded_a = self.occluded_a,
            .occluded_b = self.occluded_b,
            .blocks_a = 0,
            .blocks_b = 0,
            .mean_a = 0,
            .mean_b = 0,
            .median_a = 0,
            .median_b = 0,
            .p95_a = 0,
            .p95_b = 0,
            .delta_median_ms = 0,
            .delta_p95_ms = 0,
            .noise_floor_ms = 0,
            .flip_pairs = 0,
            .flip_agree = 0,
            .flip_ok = false,
            .unstable = false,
            .dropped = 0,
            .verdict = .INCONCLUSIVE,
            .reason = "no samples",
        };
        if (self.n_blocks == 0) return s;
        // Modal main-target size over measured blocks (ties -> earliest).
        var modal_w = self.blocks[0].w;
        var modal_h = self.blocks[0].h;
        var modal_votes: usize = 0;
        for (self.blocks[0..self.n_blocks]) |b| {
            var votes: usize = 0;
            for (self.blocks[0..self.n_blocks]) |c| {
                if (c.w == b.w and c.h == b.h) votes += 1;
            }
            if (votes > modal_votes) {
                modal_votes = votes;
                modal_w = b.w;
                modal_h = b.h;
            }
        }
        const kept = self.blocks[0..self.n_blocks];
        var kept_total: usize = 0;
        for (kept) |b| {
            if (b.size_clean and b.w == modal_w and b.h == modal_h) kept_total += 1;
        }
        s.dropped = @as(u32, @intCast(self.n_blocks - kept_total));
        s.unstable = s.dropped > 0;
        // Pooled stats over the kept ranges only (tmp reused per arm).
        var na: usize = 0;
        {
            var off: usize = 0;
            for (kept) |b| {
                if (b.is_b or !(b.size_clean and b.w == modal_w and b.h == modal_h)) continue;
                const src = self.store_a[b.start .. b.start + b.n];
                @memcpy(self.tmp[off..][0..src.len], src);
                off += src.len;
            }
            na = off;
        }
        if (na == 0) {
            s.reason = "no same-size samples for arm A";
            return s;
        }
        const sa = pooledStats(self.tmp[0..na]);
        var nb: usize = 0;
        {
            var off: usize = 0;
            for (kept) |b| {
                if (!b.is_b or !(b.size_clean and b.w == modal_w and b.h == modal_h)) continue;
                const src = self.store_b[b.start .. b.start + b.n];
                @memcpy(self.tmp[off..][0..src.len], src);
                off += src.len;
            }
            nb = off;
        }
        if (nb == 0) {
            s.reason = "no same-size samples for arm B";
            return s;
        }
        const sb = pooledStats(self.tmp[0..nb]);
        s.n_a = na;
        s.n_b = nb;
        s.mean_a, s.median_a, s.p95_a = .{ sa.mean, sa.median, sa.p95 };
        s.mean_b, s.median_b, s.p95_b = .{ sb.mean, sb.median, sb.p95 };
        s.delta_median_ms = s.median_b - s.median_a;
        s.delta_p95_ms = s.p95_b - s.p95_a;

        // Block-to-block noise floor: range of KEPT block medians per arm.
        var have_a = false;
        var have_b = false;
        var min_a: f32 = 0;
        var max_a: f32 = 0;
        var min_b: f32 = 0;
        var max_b: f32 = 0;
        for (kept) |b| {
            if (!(b.size_clean and b.w == modal_w and b.h == modal_h)) continue;
            if (b.is_b) {
                s.blocks_b += 1;
                if (!have_b) {
                    min_b = b.median_ms;
                    max_b = b.median_ms;
                    have_b = true;
                } else {
                    min_b = @min(min_b, b.median_ms);
                    max_b = @max(max_b, b.median_ms);
                }
            } else {
                s.blocks_a += 1;
                if (!have_a) {
                    min_a = b.median_ms;
                    max_a = b.median_ms;
                    have_a = true;
                } else {
                    min_a = @min(min_a, b.median_ms);
                    max_a = @max(max_a, b.median_ms);
                }
            }
        }
        const range_a: f32 = if (s.blocks_a >= 2) max_a - min_a else 0;
        const range_b: f32 = if (s.blocks_b >= 2) max_b - min_b else 0;
        s.noise_floor_ms = @max(range_a, range_b);

        // Flip check over consecutive KEPT blocks (drops shift pairing but
        // the mixed-pair order-invariance test stays valid): every mixed
        // adjacent pair (AB or BA) must agree in sign with the pooled delta.
        if (s.delta_median_ms != 0) {
            var order: [ab_max_blocks]usize = undefined;
            var n_kept: usize = 0;
            for (kept, 0..) |b, i| {
                if (!(b.size_clean and b.w == modal_w and b.h == modal_h)) continue;
                order[n_kept] = i;
                n_kept += 1;
            }
            var i: usize = 0;
            var ok = true;
            var pairs: u32 = 0;
            var agree: u32 = 0;
            while (i + 1 < n_kept) : (i += 1) {
                const x = kept[order[i]];
                const y = kept[order[i + 1]];
                if (x.is_b == y.is_b) continue;
                const bmed = if (x.is_b) x.median_ms else y.median_ms;
                const amed = if (x.is_b) y.median_ms else x.median_ms;
                const d = bmed - amed;
                pairs += 1;
                if ((d > 0) == (s.delta_median_ms > 0) and d != 0) {
                    agree += 1;
                } else ok = false;
            }
            s.flip_pairs = pairs;
            s.flip_agree = agree;
            s.flip_ok = pairs > 0 and ok;
        }

        s.reason = "pending";
        if (s.unstable) {
            s.reason = "UNSTABLE target resize mid-run (blocks dropped)";
        } else if (s.blocks_a < 2 or s.blocks_b < 2) {
            s.reason = "fewer than 2 completed blocks per arm";
        } else if (s.n_a < ab_min_samples_per_arm or s.n_b < ab_min_samples_per_arm) {
            s.reason = "fewer than 30 pooled samples per arm";
        } else if (!s.flip_ok) {
            s.reason = "flip check failed (pair sign disagrees with pooled delta)";
        } else if (@abs(s.delta_median_ms) <= s.noise_floor_ms) {
            s.reason = "pooled-median delta within block-to-block noise floor";
        } else {
            s.verdict = .SIGNIFICANT;
            s.reason = "delta exceeds block-to-block noise floor, flip check holds";
        }
        return s;
    }

    pub fn printSummary(s: AbSummary) void {
        std.debug.print("ab {s} vs {s}: blocks A={} B={} n A={} B={} occl A={} B={} unstable={} dropped={} (settle blocks excluded)\n", .{
            s.a_name,     s.b_name,     s.blocks_a, s.blocks_b, s.n_a, s.n_b,
            s.occluded_a, s.occluded_b, s.unstable, s.dropped,
        });
        std.debug.print("ab A {s}: mean={d:.3} median={d:.3} p95={d:.3}\n", .{ s.a_name, s.mean_a, s.median_a, s.p95_a });
        std.debug.print("ab B {s}: mean={d:.3} median={d:.3} p95={d:.3}\n", .{ s.b_name, s.mean_b, s.median_b, s.p95_b });
        std.debug.print("ab delta(B-A): median={d:.3} p95={d:.3} noise_floor={d:.3} flip={}/{} verdict={s} ({s})\n", .{
            s.delta_median_ms, s.delta_p95_ms, s.noise_floor_ms,
            s.flip_agree,      s.flip_pairs,   @tagName(s.verdict),
            s.reason,
        });
    }

    /// Temporal block-median trace (one line): which arm each measured
    /// block ran, its median, its sample count and target size; `*` marks
    /// a block dropped from the summary (size changed or off-modal).
    pub fn printBlocks(self: *const AbBench) void {
        std.debug.print("ab blocks ({}):", .{self.n_blocks});
        for (self.blocks[0..self.n_blocks]) |b| {
            std.debug.print(" {s}:{d:.2}({})@{}x{}{s}", .{ if (b.is_b) "B" else "A", b.median_ms, b.n, b.w, b.h, if (b.size_clean) "" else "*" });
        }
        std.debug.print("\n", .{});
    }
};

// ---- verdict-rule tests (no window needed: stores/blocks are filled
// directly; recordSample's occlusion gate is covered by live runs) --------

const AbBlockSpec = struct {
    is_b: bool,
    w: i32 = 728,
    h: i32 = 882,
    clean: bool = true,
    val: f32,
    n: usize = 40,
};

fn abTestBench(specs: []const AbBlockSpec) AbBench {
    var ab: AbBench = .{};
    for (specs) |sp| {
        std.debug.assert(sp.n <= ab.scratch.len);
        const start = if (sp.is_b) ab.len_b else ab.len_a;
        for (0..sp.n) |_| {
            if (sp.is_b) {
                ab.store_b[ab.len_b] = sp.val;
                ab.len_b += 1;
            } else {
                ab.store_a[ab.len_a] = sp.val;
                ab.len_a += 1;
            }
        }
        ab.blocks[ab.n_blocks] = .{
            .is_b = sp.is_b,
            .median_ms = sp.val,
            .n = sp.n,
            .start = start,
            .w = sp.w,
            .h = sp.h,
            .size_clean = sp.clean,
        };
        ab.n_blocks += 1;
    }
    return ab;
}

test "ab pattern cycle is drift-balanced" {
    var a: usize = 0;
    var b: usize = 0;
    for (0..8) |i| {
        if (abPatternIsB(i)) {
            b += 1;
        } else a += 1;
    }
    try std.testing.expectEqual(@as(usize, 4), a);
    try std.testing.expectEqual(@as(usize, 4), b);
    // Measured window after two settle blocks (slots 2..9): still 4/4,
    // and every non-overlapping pair is mixed (order flips each pair).
    a = 0;
    b = 0;
    var i: usize = 2;
    while (i < 10) : (i += 2) {
        try std.testing.expect(abPatternIsB(i) != abPatternIsB(i + 1));
        for ([2]usize{ i, i + 1 }) |s| {
            if (abPatternIsB(s)) {
                b += 1;
            } else a += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 4), a);
    try std.testing.expectEqual(@as(usize, 4), b);
}

test "ab verdict SIGNIFICANT on separated arms" {
    var ab = abTestBench(&.{
        .{ .is_b = false, .val = 10.0 },
        .{ .is_b = true, .val = 13.0 },
        .{ .is_b = false, .val = 10.2 },
        .{ .is_b = true, .val = 13.2 },
    });
    const s = ab.finalSummary("a", "b");
    try std.testing.expectEqual(AbVerdict.SIGNIFICANT, s.verdict);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), s.delta_median_ms, 0.001);
    try std.testing.expect(s.flip_ok);
    try std.testing.expectEqual(@as(u32, 0), s.dropped);
}

test "ab verdict INCONCLUSIVE when delta within noise" {
    // Sliding pairs all agree (B above A every time) but the pooled
    // delta (+1) sits inside the block ranges (4 each).
    var ab = abTestBench(&.{
        .{ .is_b = true, .val = 11.0 },
        .{ .is_b = false, .val = 10.0 },
        .{ .is_b = true, .val = 15.0 },
        .{ .is_b = false, .val = 14.0 },
    });
    const s = ab.finalSummary("a", "b");
    try std.testing.expectEqual(AbVerdict.INCONCLUSIVE, s.verdict);
    try std.testing.expect(s.flip_ok);
    try std.testing.expect(s.noise_floor_ms >= @abs(s.delta_median_ms));
    try std.testing.expect(std.mem.eql(u8, s.reason, "pooled-median delta within block-to-block noise floor"));
}

test "ab verdict INCONCLUSIVE on flip disagreement" {
    var ab = abTestBench(&.{
        .{ .is_b = false, .val = 10.0 },
        .{ .is_b = true, .val = 12.0 },
        .{ .is_b = false, .val = 10.0 },
        .{ .is_b = true, .val = 8.0 },
    });
    const s = ab.finalSummary("a", "b");
    try std.testing.expectEqual(AbVerdict.INCONCLUSIVE, s.verdict);
    try std.testing.expect(!s.flip_ok);
}

test "ab verdict INCONCLUSIVE on mid-run resize, kept stats stay modal" {
    var ab = abTestBench(&.{
        .{ .is_b = false, .val = 10.0 },
        .{ .is_b = true, .val = 13.0 },
        .{ .is_b = false, .val = 10.2 },
        .{ .is_b = true, .val = 13.2 },
        .{ .is_b = false, .val = 7.0, .w = 481, .h = 882 },
    });
    const s = ab.finalSummary("a", "b");
    try std.testing.expectEqual(AbVerdict.INCONCLUSIVE, s.verdict);
    try std.testing.expect(s.unstable);
    try std.testing.expectEqual(@as(u32, 1), s.dropped);
    // The off-size block's fast samples must not leak into arm A.
    try std.testing.expectApproxEqAbs(@as(f32, 10.2), s.median_a, 0.001);
}

test "ab verdict INCONCLUSIVE on thin quorum or thin samples" {
    var ab = abTestBench(&.{
        .{ .is_b = false, .val = 10.0 },
        .{ .is_b = true, .val = 13.0 },
        .{ .is_b = true, .val = 13.2 },
    });
    try std.testing.expectEqual(AbVerdict.INCONCLUSIVE, ab.finalSummary("a", "b").verdict);
    var thin = abTestBench(&.{
        .{ .is_b = false, .val = 10.0, .n = 10 },
        .{ .is_b = true, .val = 13.0, .n = 40 },
        .{ .is_b = false, .val = 10.2, .n = 10 },
        .{ .is_b = true, .val = 13.2, .n = 40 },
    });
    const s = thin.finalSummary("a", "b");
    try std.testing.expectEqual(AbVerdict.INCONCLUSIVE, s.verdict);
    try std.testing.expect(s.n_a < ab_min_samples_per_arm);
}
