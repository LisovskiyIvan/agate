//! Agate phase mapping over sokol's engine-independent GPU timing scopes.
//!
//! Metal reports command-buffer time and counter-supported phase spans;
//! WebGPU reports native-pass/phase spans when timestamp-query was requested
//! on the device; desktop GL reports elapsed phase queries and their sum.
//! `capabilities` describes support and scope. Optional `Sample`s distinguish
//! unavailable results from valid zero-duration (quantized) measurements.
//! Sample indices identify completed sokol submissions, not Scene.frame_id.
//!
//! Enable via `setEnabled(true)` or AGATE_GPU_TIMINGS=1/true/on/yes. For
//! WebGPU also set sapp.Desc.wgpu_gpu_timing_enabled BEFORE device creation
//! (e.g. to `isEnabled()`). Unsupported devices remain usable without timers.
//! Intent is any-thread; begin/end/poll apply it only on the context thread.
//! Default OFF encodes no timestamps. sg.shutdown owns timer teardown, and
//! the next setup/poll reapplies intent without a stale context cache.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

/// Agate assigns three caller-defined sokol scopes; sokol has no phase names.
pub const Pass = enum(c_int) {
    shadow = 0,
    main = 1,
    post = 2,
};

pub const FrameScope = enum { none, command_buffer, native_pass_span, pass_sum };

pub const Capabilities = struct {
    frame: bool = false,
    passes: bool = false,
    frame_scope: FrameScope = .none,
};

/// A completed GPU measurement. Index 0 is reserved for unavailable data.
pub const Sample = struct {
    ms: f32,
    frame_index: u32,
};

/// Device support, independent of enablement. Context-thread query.
pub fn capabilities() Capabilities {
    if (!sg.isvalid()) return .{};
    const frame = sg.gpuFrameTimingSupported();
    return .{
        .frame = frame,
        .passes = sg.gpuScopeTimingSupported(),
        .frame_scope = if (frame) switch (sg.queryBackend()) {
            .METAL_MACOS, .METAL_IOS, .METAL_SIMULATOR => .command_buffer,
            .WGPU => .native_pass_span,
            .GLCORE => .pass_sum,
            else => .none,
        } else .none,
    };
}

/// Truth values accepted for AGATE_GPU_TIMINGS (exact match, lowercase).
pub fn parseEnvFlag(value: ?[]const u8) bool {
    const v = value orelse return false;
    return std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "true") or
        std.mem.eql(u8, v, "on") or std.mem.eql(u8, v, "yes");
}

var enabled: std.atomic.Value(bool) = .init(false);
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
    sg.setGpuTimingEnabled(want);
}

/// Any-thread intent only. The next context-thread begin/end/poll applies it.
pub fn setEnabled(on: bool) void {
    pollEnvOnce();
    enabled.store(on, .release);
}

/// Current enablement intent (includes the `AGATE_GPU_TIMINGS` opt-in).
pub fn isEnabled() bool {
    pollEnvOnce();
    return enabled.load(.acquire);
}

fn applyIntent() bool {
    const want = isEnabled();
    if (!sg.isvalid()) return false;
    syncToC(want);
    return want;
}

fn sampleFromC(ms: f32, frame_index: u32) ?Sample {
    if (frame_index == 0 or !std.math.isFinite(ms) or ms < 0) return null;
    return .{ .ms = ms, .frame_index = frame_index };
}

/// Context-thread poll after commit. Does not wait for the current GPU frame.
pub fn pollFrameSample() ?Sample {
    if (!applyIntent()) return null;
    const ms = sg.queryGpuFrameMs();
    return sampleFromC(ms, sg.queryGpuFrameIndex());
}

/// Legacy profiler helper: unavailable samples map to 0, not CPU time.
pub fn pollFrameMs() f32 {
    return if (pollFrameSample()) |sample| sample.ms else 0;
}

/// Context-thread phase bracket; sequential, never nested. Unsupported is a no-op.
pub fn beginPass(pass: Pass) void {
    if (!applyIntent()) return;
    sg.gpuTimingScopeBegin(@intFromEnum(pass));
}

/// Closes the GPU timer for phase `pass`. Same fail-closed contract as
/// `beginPass`; an end without begin is ignored by the C side.
pub fn endPass(pass: Pass) void {
    if (!applyIntent()) return;
    sg.gpuTimingScopeEnd(@intFromEnum(pass));
}

/// Completed phase span, possibly several native passes. Context thread only.
pub fn pollPassSample(pass: Pass) ?Sample {
    if (!applyIntent()) return null;
    const ms = sg.queryGpuScopeMs(@intFromEnum(pass));
    return sampleFromC(ms, sg.queryGpuScopeFrameIndex(@intFromEnum(pass)));
}

pub fn pollPassMs(pass: Pass) f32 {
    return if (pollPassSample(pass)) |sample| sample.ms else 0;
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

test "gpu_timing maps Agate phases to generic sokol scopes" {
    try std.testing.expectEqual(@as(c_int, 0), @intFromEnum(Pass.shadow));
    try std.testing.expectEqual(@as(c_int, 1), @intFromEnum(Pass.main));
    try std.testing.expectEqual(@as(c_int, 2), @intFromEnum(Pass.post));
    try std.testing.expect(@intFromEnum(Pass.post) < sg.max_gpu_timing_scopes);
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

test "gpu_timing optional samples and capabilities are unavailable headless" {
    const was_enabled = isEnabled();
    defer setEnabled(was_enabled);
    setEnabled(true);
    try std.testing.expectEqual(Capabilities{}, capabilities());
    try std.testing.expect(pollFrameSample() == null);
    inline for (.{ Pass.shadow, Pass.main, Pass.post }) |pass| {
        try std.testing.expect(pollPassSample(pass) == null);
    }
}

test "gpu_timing preserves valid zero and rejects invalid samples" {
    try std.testing.expectEqual(Sample{ .ms = 0, .frame_index = 7 }, sampleFromC(0, 7).?);
    try std.testing.expect(sampleFromC(-1, 7) == null);
    try std.testing.expect(sampleFromC(1, 0) == null);
    try std.testing.expect(sampleFromC(std.math.nan(f32), 7) == null);
    try std.testing.expect(sampleFromC(std.math.inf(f32), 7) == null);
}
