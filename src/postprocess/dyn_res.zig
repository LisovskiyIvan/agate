//! Dynamic resolution controller: pure decision logic, no GPU/scene imports.
//!
//! Observes frame times and moves `PostProcessOptions.render_scale` inside
//! `[min_scale, max_scale]` to hold a target frame budget. Deliberately
//! predictable, not clever:
//!
//! - EMA smoothing of the frame time (single-pole, `ema_alpha`).
//! - Asymmetric hysteresis: downscale needs `sustained_frames` over
//!   `target * downscale_margin`, upscale needs the same under
//!   `target * upscale_margin` — dropping fast, recovering slow.
//! - Per-change step limits (`max_down_step`/`max_up_step`) and a cooldown
//!   in frames between changes, so a borderline frame time cannot oscillate
//!   the scale (each change rebuilds temporal history via the one-frame
//!   `taa.camera_cut` reset — visible if it flaps).
//!
//! The proportional step aims at `scale * margin_target / ema` (so a 2×
//! overrun drops harder than a 1.1× one), then the step limit clamps it.
//!
//! Threading: none. The caller (game side, via `Scene.observeFrameTime`)
//! feeds one sample per frame; the engine applies the returned scale and
//! owns the one-frame history reset. Headless-testable by construction.

const std = @import("std");

pub const Config = struct {
    /// Frame budget the controller holds, in milliseconds (16.67 / 8.33).
    /// Non-finite or <= 0 falls back to 16.67 in `init`.
    target_frame_ms: f32 = 16.67,
    /// Scale floor. The engine-wide hard floor is `min_render_scale` (0.25);
    /// below ~0.5 the output-res upscale is mush, so the controller defaults
    /// to 0.5 and never goes there on its own.
    min_scale: f32 = 0.5,
    max_scale: f32 = 1.0,
    /// Sustained EMA over `target * downscale_margin` → downscale.
    downscale_margin: f32 = 1.10,
    /// Sustained EMA under `target * upscale_margin` → upscale.
    upscale_margin: f32 = 0.90,
    /// Inside the band `[up_margin, down_margin]` nothing changes.
    ema_alpha: f32 = 0.25,
    /// Frames ignored before the controller trusts the timings (startup,
    /// shader warmup, window settle).
    warmup_frames: u32 = 30,
    /// Frames to wait after each change before counting again.
    cooldown_frames: u32 = 30,
    /// Consecutive out-of-band frames required for one change (after
    /// cooldown). Keeps single-frame spikes (loading hitch) from moving
    /// the scale.
    sustained_frames: u32 = 10,
    /// Largest single downscale, as an absolute scale delta.
    max_down_step: f32 = 0.10,
    /// Largest single upscale, as an absolute scale delta. Recovery is
    /// deliberately slower than the drop.
    max_up_step: f32 = 0.05,
};

pub const DynamicRes = struct {
    config: Config = .{},
    /// Current scale; 1.0 until the controller changes it.
    scale: f32 = 1.0,
    ema_ms: f32 = 0.0,
    frames_seen: u32 = 0,
    cooldown_left: u32 = 0,
    overrun_run: u32 = 0,
    underrun_run: u32 = 0,
    /// Number of scale changes so far (observability).
    changes: u64 = 0,

    pub fn init(config: Config) DynamicRes {
        var cfg = config;
        if (!std.math.isFinite(cfg.target_frame_ms) or cfg.target_frame_ms <= 0.0) {
            cfg.target_frame_ms = 16.67;
        }
        if (!std.math.isFinite(cfg.ema_alpha) or cfg.ema_alpha <= 0.0) cfg.ema_alpha = 0.25;
        if (cfg.ema_alpha > 1.0) cfg.ema_alpha = 1.0;
        if (!std.math.isFinite(cfg.downscale_margin) or cfg.downscale_margin < 1.0) cfg.downscale_margin = 1.10;
        if (!std.math.isFinite(cfg.upscale_margin) or cfg.upscale_margin <= 0.0 or cfg.upscale_margin > 1.0) cfg.upscale_margin = 0.90;
        if (!std.math.isFinite(cfg.max_down_step) or cfg.max_down_step <= 0.0) cfg.max_down_step = 0.10;
        if (!std.math.isFinite(cfg.max_up_step) or cfg.max_up_step <= 0.0) cfg.max_up_step = 0.05;
        if (!std.math.isFinite(cfg.min_scale)) cfg.min_scale = 0.5;
        if (!std.math.isFinite(cfg.max_scale) or cfg.max_scale > 1.0) cfg.max_scale = 1.0;
        if (cfg.min_scale < 0.25) cfg.min_scale = 0.25; // engine hard floor
        if (cfg.min_scale > cfg.max_scale) cfg.min_scale = cfg.max_scale;

        var self = DynamicRes{ .config = cfg };
        self.scale = std.math.clamp(1.0, cfg.min_scale, cfg.max_scale);
        return self;
    }

    /// Feeds one frame time (milliseconds; GPU frame time preferred, CPU
    /// delta works). Returns the new scale exactly on frames where the
    /// controller CHANGED it — the caller applies it and resets temporal
    /// history — and null on every other frame.
    pub fn observe(self: *DynamicRes, frame_ms: f32) ?f32 {
        if (!std.math.isFinite(frame_ms) or frame_ms <= 0.0) return null;

        self.frames_seen +|= 1;
        self.ema_ms = if (self.frames_seen == 1)
            frame_ms
        else
            self.ema_ms + self.config.ema_alpha * (frame_ms - self.ema_ms);

        if (self.frames_seen <= self.config.warmup_frames) {
            self.overrun_run = 0;
            self.underrun_run = 0;
            return null;
        }
        if (self.cooldown_left > 0) {
            self.cooldown_left -= 1;
            self.overrun_run = 0;
            self.underrun_run = 0;
            return null;
        }

        const target = self.config.target_frame_ms;
        var changed: ?f32 = null;
        if (self.ema_ms > target * self.config.downscale_margin) {
            self.overrun_run +|= 1;
            self.underrun_run = 0;
            if (self.overrun_run >= self.config.sustained_frames and self.scale > self.config.min_scale) {
                const desired = self.scale * (target * self.config.downscale_margin) / self.ema_ms;
                // Step limit takes the SLOWER drop of the two.
                var candidate = @max(desired, self.scale - self.config.max_down_step);
                candidate = @max(candidate, self.config.min_scale);
                changed = self.commit(candidate);
            }
        } else if (self.ema_ms < target * self.config.upscale_margin) {
            self.underrun_run +|= 1;
            self.overrun_run = 0;
            if (self.underrun_run >= self.config.sustained_frames and self.scale < self.config.max_scale) {
                const desired = self.scale * (target * self.config.upscale_margin) / self.ema_ms;
                // Step limit takes the SLOWER rise of the two.
                var candidate = @min(desired, self.scale + self.config.max_up_step);
                candidate = @min(candidate, self.config.max_scale);
                changed = self.commit(candidate);
            }
        } else {
            self.overrun_run = 0;
            self.underrun_run = 0;
        }
        return changed;
    }

    fn commit(self: *DynamicRes, candidate: f32) ?f32 {
        const clamped = std.math.clamp(candidate, self.config.min_scale, self.config.max_scale);
        // Clamped into a no-op (e.g. desired step already at the floor):
        // keep the run alive so the controller reacts when the margin
        // widens, without emitting a no-op change.
        if (@abs(clamped - self.scale) < 1e-6) return null;
        self.scale = clamped;
        self.changes += 1;
        self.cooldown_left = self.config.cooldown_frames;
        self.overrun_run = 0;
        self.underrun_run = 0;
        return self.scale;
    }

    pub fn current(self: *const DynamicRes) f32 {
        return self.scale;
    }

    pub fn emaFrameMs(self: *const DynamicRes) f32 {
        return self.ema_ms;
    }
};

test "warmup ignores the first frames" {
    var dr = DynamicRes.init(.{ .target_frame_ms = 10.0, .warmup_frames = 3, .cooldown_frames = 0, .sustained_frames = 1 });
    try std.testing.expectEqual(@as(?f32, null), dr.observe(100.0));
    try std.testing.expectEqual(@as(?f32, null), dr.observe(100.0));
    try std.testing.expectEqual(@as(?f32, null), dr.observe(100.0));
    // Frame 4 is past warmup: sustained overrun fires immediately.
    const changed = dr.observe(100.0) orelse return error.TestUnexpectedResult;
    try std.testing.expect(changed < 1.0);
}

test "stable frame time inside the band never changes the scale" {
    var dr = DynamicRes.init(.{
        .target_frame_ms = 10.0,
        .warmup_frames = 0,
        .cooldown_frames = 0,
        .sustained_frames = 2,
    });
    for (0..200) |_| {
        try std.testing.expectEqual(@as(?f32, null), dr.observe(10.0));
    }
    try std.testing.expectEqual(@as(f32, 1.0), dr.current());
}

test "sustained overrun steps down, then cooldown, then down again" {
    var dr = DynamicRes.init(.{
        .target_frame_ms = 10.0,
        .warmup_frames = 0,
        .cooldown_frames = 2,
        .sustained_frames = 2,
        .max_down_step = 0.1,
    });
    // Two sustained overruns -> one downscale on the second frame.
    _ = dr.observe(30.0);
    const s1 = dr.observe(30.0) orelse return error.TestUnexpectedResult;
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), s1, 1e-6);
    // Cooldown swallows the next two frames even though they overrun.
    try std.testing.expectEqual(@as(?f32, null), dr.observe(30.0));
    try std.testing.expectEqual(@as(?f32, null), dr.observe(30.0));
    // Sustained again -> second step.
    _ = dr.observe(30.0);
    const s2 = dr.observe(30.0) orelse return error.TestUnexpectedResult;
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), s2, 1e-6);
}

test "sustained underrun recovers slowly and clamps at max" {
    var dr = DynamicRes.init(.{
        .target_frame_ms = 10.0,
        .warmup_frames = 0,
        .cooldown_frames = 0,
        .sustained_frames = 1,
        .max_up_step = 0.05,
        .min_scale = 0.5,
    });
    dr.scale = 0.5;
    const up = dr.observe(1.0) orelse return error.TestUnexpectedResult;
    try std.testing.expectApproxEqAbs(@as(f32, 0.55), up, 1e-6);
    // March back to the max; at 1.0 the controller goes quiet.
    var guard: usize = 0;
    while (dr.current() < dr.config.max_scale) : (guard += 1) {
        _ = dr.observe(1.0);
        if (guard > 100) return error.TestUnexpectedResult;
    }
    try std.testing.expectEqual(@as(?f32, null), dr.observe(1.0));
}

test "min clamp: sustained overrun at the floor goes quiet" {
    var dr = DynamicRes.init(.{
        .target_frame_ms = 10.0,
        .warmup_frames = 0,
        .cooldown_frames = 0,
        .sustained_frames = 1,
        .min_scale = 0.5,
    });
    dr.scale = 0.5;
    try std.testing.expectEqual(@as(?f32, null), dr.observe(100.0));
    try std.testing.expectEqual(@as(?f32, null), dr.observe(100.0));
    try std.testing.expectEqual(@as(f32, 0.5), dr.current());
}

test "invalid and non-positive samples are ignored" {
    var dr = DynamicRes.init(.{ .target_frame_ms = 10.0, .warmup_frames = 0, .sustained_frames = 1 });
    const before_frames = dr.frames_seen;
    try std.testing.expectEqual(@as(?f32, null), dr.observe(std.math.nan(f32)));
    try std.testing.expectEqual(@as(?f32, null), dr.observe(0.0));
    try std.testing.expectEqual(@as(?f32, null), dr.observe(-5.0));
    try std.testing.expectEqual(before_frames, dr.frames_seen);
}

test "init sanitizes a broken config" {
    const dr = DynamicRes.init(.{
        .target_frame_ms = -1.0,
        .min_scale = 2.0,
        .max_scale = 0.3,
        .downscale_margin = 0.5,
        .upscale_margin = 4.0,
        .ema_alpha = 0.0,
    });
    try std.testing.expectApproxEqAbs(@as(f32, 16.67), dr.config.target_frame_ms, 1e-4);
    try std.testing.expect(dr.config.min_scale <= dr.config.max_scale);
    try std.testing.expect(dr.config.downscale_margin >= 1.0);
    try std.testing.expect(dr.config.upscale_margin <= 1.0);
    try std.testing.expect(dr.config.ema_alpha > 0.0);
}
