//! GPU frame timings, v1 (Metal-only) via the vendored sokol patch.
//!
//! Upstream sokol-gfx has no GPU-timestamp mechanism (trace hooks are
//! CPU-side begin/end callbacks, not GPU time), so the engine profiler
//! historically recorded CPU-submit times only. The vendored
//! `sokol_gfx.h` carries a minimal patch (see
//! `vendor/sokol/README.agate.md`, "GPU timings patch", applied by
//! `tools/patch_sokol_gpu_timings.py`):
//!
//! - `sg_agate_set_gpu_timing_enabled(bool)` — default OFF. While on,
//!   each committed Metal command buffer is retained one extra frame.
//! - `sg_agate_query_gpu_frame_ms()` — last COMPLETED frame's
//!   `(GPUEndTime-GPUStartTime)` in ms, or -1 when unavailable
//!   (disabled, not ready yet, or a non-Metal backend).
//!
//! Semantics: the value lags one frame behind the CPU submit (async GPU
//! execution) and is a single frame-level number — per-pass GPU
//! attribution is out of scope for v1. Headless/dummy (no sg context)
//! is fail-closed: every entry point returns 0/false and never calls
//! into sokol C code outside a valid context.
//!
//! Enablement (default OFF, zero behavior change while off):
//! - programmatic: `gpu_timing.setEnabled(true)` (call before or after
//!   `sg.setup`; the C flag is applied lazily on the next poll once a
//!   context exists), or
//! - environment: `AGATE_GPU_TIMINGS=1` (also `true`/`on`/`yes`),
//!   picked up lazily on first use — no sandbox flag plumbing needed.
//!
//! Leaf module: imports std + sokol only, never the profiler or scene,
//! so `scene/frame_render.zig` can poll it without an import cycle.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

extern fn sg_agate_set_gpu_timing_enabled(enabled: bool) void;
extern fn sg_agate_query_gpu_frame_ms() f32;

/// Truth values accepted for AGATE_GPU_TIMINGS (exact match, lowercase).
pub fn parseEnvFlag(value: ?[]const u8) bool {
    const v = value orelse return false;
    return std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "true") or
        std.mem.eql(u8, v, "on") or std.mem.eql(u8, v, "yes");
}

var enabled: std.atomic.Value(bool) = .init(false);
var c_applied: std.atomic.Value(bool) = .init(false);
var env_checked: std.atomic.Value(bool) = .init(false);

fn pollEnvOnce() void {
    if (env_checked.swap(true, .acq_rel)) return;
    const raw = std.c.getenv("AGATE_GPU_TIMINGS") orelse return;
    if (parseEnvFlag(std.mem.span(raw))) enabled.store(true, .release);
}

/// Applies the Zig-side intent to the sokol C flag. Fail-closed: never
/// touches C without a valid sg context.
fn syncToC(want: bool) void {
    if (!sg.isvalid()) return;
    if (c_applied.load(.acquire) == want) return;
    sg_agate_set_gpu_timing_enabled(want);
    c_applied.store(want, .release);
}

/// Enables or disables GPU frame timings. Safe to call with no sg
/// context (the C flag is applied lazily by `pollFrameMs` once a
/// context exists); safe to call from any thread.
pub fn setEnabled(on: bool) void {
    pollEnvOnce();
    enabled.store(on, .release);
    syncToC(on);
}

/// Current enablement intent (includes the `AGATE_GPU_TIMINGS` opt-in).
pub fn isEnabled() bool {
    pollEnvOnce();
    return enabled.load(.acquire);
}

/// Per-frame hook for the render thread: call once after `sg.commit()`.
/// Returns the last completed GPU frame time in ms, or 0 when disabled,
/// headless, not ready yet, or on a backend without support (GL returns
/// -1 from C today). Never panics, never calls into C without context.
pub fn pollFrameMs() f32 {
    pollEnvOnce();
    if (!enabled.load(.acquire)) {
        syncToC(false);
        return 0;
    }
    if (!sg.isvalid()) return 0;
    syncToC(true);
    const ms = sg_agate_query_gpu_frame_ms();
    return if (ms >= 0) ms else 0;
}

// ---------------------------------------------------------------------------
// Unit tests (all headless: no sg context here, so every call must be a
// fail-closed no-op returning 0 and must never reach the C symbols).
// ---------------------------------------------------------------------------

test "gpu_timing parses AGATE_GPU_TIMINGS flag values" {
    try std.testing.expect(parseEnvFlag("1"));
    try std.testing.expect(parseEnvFlag("true"));
    try std.testing.expect(parseEnvFlag("on"));
    try std.testing.expect(parseEnvFlag("yes"));
    try std.testing.expect(!parseEnvFlag(null));
    try std.testing.expect(!parseEnvFlag(""));
    try std.testing.expect(!parseEnvFlag("0"));
    try std.testing.expect(!parseEnvFlag("false"));
    try std.testing.expect(!parseEnvFlag("TRUE"));
}

test "gpu_timing is off by default and fail-closed headless" {
    // No other test module touches this leaf, so the default holds.
    // (If this ever flakes, some test started enabling timings globally.)
    const was_enabled = isEnabled();
    defer setEnabled(was_enabled);
    setEnabled(false);

    try std.testing.expect(!sg.isvalid());
    try std.testing.expect(!isEnabled());
    try std.testing.expectEqual(@as(f32, 0), pollFrameMs());

    // Enabled intent but no sg context: still 0, no C call, no panic.
    setEnabled(true);
    try std.testing.expect(isEnabled());
    try std.testing.expectEqual(@as(f32, 0), pollFrameMs());
}
