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
//!   each committed Metal command buffer is retained one extra frame,
//!   and each engine phase (shadow/main/post) opens a `GL_TIME_ELAPSED`
//!   query on GL4.1.
//! - `sg_agate_query_gpu_frame_ms()` — last COMPLETED frame's GPU time
//!   in ms (Metal `(GPUEndTime-GPUStartTime)`; GL sum of the
//!   last-completed per-pass samples), or -1 when unavailable
//!   (disabled, not ready yet, or an unsupported backend).
//! - `sg_agate_gpu_pass_begin/end(int)` + `sg_agate_query_gpu_pass_ms`
//!   — per-pass timers (GL4.1 only; linked no-ops / -1 elsewhere).
//!
//! Semantics: the value lags one frame behind the CPU submit (async GPU
//! execution) and is a single frame-level number on Metal — per-pass GPU
//! attribution is v2: engine-driven `GL_TIME_ELAPSED` pools on GL4.1
//! (`Pass` below; Metal begin/end are linked no-ops and per-pass queries
//! are -1 there — the single Metal command buffer spans the whole frame,
//! see `vendor/sokol/README.agate.md`). Headless/dummy (no sg context)
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
extern fn sg_agate_gpu_pass_begin(pass: c_int) void;
extern fn sg_agate_gpu_pass_end(pass: c_int) void;
extern fn sg_agate_query_gpu_pass_ms(pass: c_int) f32;

/// Engine render phases with GPU timers. Ids match the sokol patch
/// (`vendor/sokol/README.agate.md`): 0=shadow, 1=main, 2=post. GL4.1
/// carries a real `GL_TIME_ELAPSED` pool per phase; Metal compiles the
/// brackets to linked no-ops (frame timer only — one command buffer per
/// frame) and every other backend is fail-closed.
pub const Pass = enum(c_int) {
    shadow = 0,
    main = 1,
    post = 2,
};

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
/// headless, not ready yet, or on a backend without support. On GL the
/// frame value is the sum of the last-completed per-pass samples (a
/// lower bound: inter-pass bubbles excluded). Never panics, never calls
/// into C without context.
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

/// Opens the GPU timer for phase `pass`. Bracket each engine phase
/// (`frame_render.zig` shadow/main/post) with begin/end; strictly
/// sequential, never nested. Fail-closed no-op when disabled or
/// headless (never calls into C without context); linked no-op on
/// Metal, so the same call sites serve every backend.
pub fn beginPass(pass: Pass) void {
    pollEnvOnce();
    if (!enabled.load(.acquire)) return;
    if (!sg.isvalid()) return;
    syncToC(true);
    sg_agate_gpu_pass_begin(@intFromEnum(pass));
}

/// Closes the GPU timer for phase `pass`. Same fail-closed contract as
/// `beginPass`; an end without begin is ignored by the C side.
pub fn endPass(pass: Pass) void {
    pollEnvOnce();
    if (!enabled.load(.acquire)) return;
    if (!sg.isvalid()) return;
    syncToC(true);
    sg_agate_gpu_pass_end(@intFromEnum(pass));
}

/// Reads the last COMPLETED sample for phase `pass` in ms (lags behind
/// the CPU submit like the frame timer), or 0 when disabled, headless,
/// not ready yet, or unsupported (Metal per-pass is always 0 — the
/// frame timer is the only Metal GPU number). Never panics, never calls
/// into C without context.
pub fn pollPassMs(pass: Pass) f32 {
    pollEnvOnce();
    if (!enabled.load(.acquire)) {
        syncToC(false);
        return 0;
    }
    if (!sg.isvalid()) return 0;
    syncToC(true);
    const ms = sg_agate_query_gpu_pass_ms(@intFromEnum(pass));
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

test "gpu_timing pass ids match the sokol patch contract" {
    // The C side indexes pools by these ids (0=shadow, 1=main, 2=post);
    // a renumber here without the matching C change would misattribute.
    try std.testing.expectEqual(@as(c_int, 0), @intFromEnum(Pass.shadow));
    try std.testing.expectEqual(@as(c_int, 1), @intFromEnum(Pass.main));
    try std.testing.expectEqual(@as(c_int, 2), @intFromEnum(Pass.post));
}

test "gpu_timing per-pass brackets are fail-closed headless" {
    const was_enabled = isEnabled();
    defer setEnabled(was_enabled);
    try std.testing.expect(!sg.isvalid());

    // Disabled: brackets are pure no-ops, polls read 0.
    setEnabled(false);
    beginPass(.shadow);
    endPass(.shadow);
    try std.testing.expectEqual(@as(f32, 0), pollPassMs(.shadow));
    try std.testing.expectEqual(@as(f32, 0), pollPassMs(.main));
    try std.testing.expectEqual(@as(f32, 0), pollPassMs(.post));

    // Enabled intent but no sg context: still no C call, no panic, 0.
    setEnabled(true);
    beginPass(.main);
    beginPass(.post);
    endPass(.main);
    endPass(.post);
    // Unbalanced end (no begin) must also stay a safe no-op.
    endPass(.shadow);
    try std.testing.expectEqual(@as(f32, 0), pollPassMs(.shadow));
    try std.testing.expectEqual(@as(f32, 0), pollPassMs(.main));
    try std.testing.expectEqual(@as(f32, 0), pollPassMs(.post));
}
