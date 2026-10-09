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

pub fn sampleFromC(ms: f32, frame_index: u32) ?Sample {
    if (frame_index == 0 or !std.math.isFinite(ms) or ms < 0) return null;
    return .{ .ms = ms, .frame_index = frame_index };
}

/// Context-thread poll after commit. Does not wait for the current GPU frame.
pub fn pollFrameSample() ?Sample {
    if (!applyIntent()) return null;
    const ms = sg.queryGpuFrameMs();
    return sampleFromC(ms, sg.queryGpuFrameIndex());
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
