const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const gpu_timing = @import("gpu_timing.zig");

const Pass = gpu_timing.Pass;
const Capabilities = gpu_timing.Capabilities;
const Sample = gpu_timing.Sample;
const capabilities = gpu_timing.capabilities;
const parseEnvFlag = gpu_timing.parseEnvFlag;
const setEnabled = gpu_timing.setEnabled;
const isEnabled = gpu_timing.isEnabled;
const pollFrameSample = gpu_timing.pollFrameSample;
const beginPass = gpu_timing.beginPass;
const endPass = gpu_timing.endPass;
const pollPassSample = gpu_timing.pollPassSample;
const sampleFromC = gpu_timing.sampleFromC;

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
    try std.testing.expect(pollFrameSample() == null);

    // Enabled intent but no sg context: still null, no C call, no panic.
    setEnabled(true);
    try std.testing.expect(isEnabled());
    try std.testing.expect(pollFrameSample() == null);
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

    // Disabled: brackets are pure no-ops, polls read null.
    setEnabled(false);
    beginPass(.shadow);
    endPass(.shadow);
    try std.testing.expect(pollPassSample(.shadow) == null);
    try std.testing.expect(pollPassSample(.main) == null);
    try std.testing.expect(pollPassSample(.post) == null);

    // Enabled intent but no sg context: still no C call, no panic, null.
    setEnabled(true);
    beginPass(.main);
    beginPass(.post);
    endPass(.main);
    endPass(.post);
    // Unbalanced end (no begin) must also stay a safe no-op.
    endPass(.shadow);
    try std.testing.expect(pollPassSample(.shadow) == null);
    try std.testing.expect(pollPassSample(.main) == null);
    try std.testing.expect(pollPassSample(.post) == null);
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
