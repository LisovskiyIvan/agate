const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const msaa = @import("msaa.zig");

const Inputs = msaa.Inputs;
const effectiveSampleCount = msaa.effectiveSampleCount;
const clampSampleCount = msaa.clampSampleCount;
const depthEffectsActive = msaa.depthEffectsActive;
const depthPrepassActive = msaa.depthPrepassActive;
const suppressDepthEffects = msaa.suppressDepthEffects;
const needsResolveAttachment = msaa.needsResolveAttachment;
const WarnOnce = msaa.WarnOnce;

const testing = std.testing;

test "HDR sample policy rejects WebGPU 2x and preserves other choices" {
    const web = Inputs{ .backend = .WGPU };
    try std.testing.expectEqual(@as(i32, 1), effectiveSampleCount(2, web));
    try std.testing.expectEqual(@as(i32, 1), effectiveSampleCount(3, web));
    try std.testing.expectEqual(@as(i32, 4), effectiveSampleCount(4, web));
    try std.testing.expectEqual(@as(i32, 2), effectiveSampleCount(2, .{ .backend = .METAL_MACOS }));
}

// --- GPU-free contract tests (visual AA quality cannot be unit-tested; it
// is verified with the agate smoke run: `agate --frames 120 --msaa 4`). ---

test "clampSampleCount snaps down to valid counts and caps by backend" {
    // Portable backends cap at 4 (see maxSamplesForBackend docs).
    for ([_]sg.Backend{ .METAL_MACOS, .METAL_IOS, .D3D11, .VULKAN, .GLCORE, .GLES3 }) |backend| {
        try testing.expectEqual(@as(i32, 1), clampSampleCount(backend, 1));
        try testing.expectEqual(@as(i32, 1), clampSampleCount(backend, 0));
        try testing.expectEqual(@as(i32, 1), clampSampleCount(backend, -4));
        try testing.expectEqual(@as(i32, 2), clampSampleCount(backend, 2));
        try testing.expectEqual(@as(i32, 2), clampSampleCount(backend, 3));
        try testing.expectEqual(@as(i32, 4), clampSampleCount(backend, 4));
        try testing.expectEqual(@as(i32, 4), clampSampleCount(backend, 5));
        try testing.expectEqual(@as(i32, 4), clampSampleCount(backend, 8));
        try testing.expectEqual(@as(i32, 4), clampSampleCount(backend, 16));
    }
    // Dummy backend (tests, headless tooling) allows the full table.
    try testing.expectEqual(@as(i32, 8), clampSampleCount(.DUMMY, 8));
    try testing.expectEqual(@as(i32, 8), clampSampleCount(.DUMMY, 99));
}

test "effectiveSampleCount gates only on target format support" {
    const base = Inputs{ .backend = .DUMMY };
    try testing.expectEqual(@as(i32, 4), effectiveSampleCount(4, base));
    // Runtime format gate (e.g. a backend that cannot MSAA the swapchain
    // color format): degrade to 1x rather than fail resource creation.
    try testing.expectEqual(@as(i32, 1), effectiveSampleCount(4, .{ .formats_msaa_capable = false, .backend = .DUMMY }));
    // Clamp still applies on the effective path (real-backend cap is 4;
    // the dummy backend used by base allows the full table).
    try testing.expectEqual(@as(i32, 4), effectiveSampleCount(8, .{ .backend = .METAL_MACOS }));
    try testing.expectEqual(@as(i32, 8), effectiveSampleCount(8, base));
}

test "depthEffectsActive matches the suppressed set (SSAO/SSR/DoF/Fog/MotionBlur/SSGI)" {
    // SSAO defaults on in this engine: that alone counts as active.
    try testing.expect(depthEffectsActive(true, true, false, false, false, false, false, false));
    try testing.expect(depthEffectsActive(true, false, true, false, false, false, false, false)); // debug
    try testing.expect(depthEffectsActive(true, false, false, true, false, false, false, false)); // SSR
    try testing.expect(depthEffectsActive(true, false, false, false, true, false, false, false)); // DoF
    try testing.expect(depthEffectsActive(true, false, false, false, false, true, false, false)); // Fog
    try testing.expect(depthEffectsActive(true, false, false, false, false, false, true, false)); // MotionBlur
    try testing.expect(depthEffectsActive(true, false, false, false, false, false, false, true)); // SSGI; // MotionBlur
    try testing.expect(!depthEffectsActive(true, false, false, false, false, false, false, false));
    // Without the post chain nothing runs at all.
    try testing.expect(!depthEffectsActive(false, true, true, true, true, true, true, true));
}

test "needsResolveAttachment follows the sokol resolve contract" {
    try testing.expect(!needsResolveAttachment(1));
    try testing.expect(needsResolveAttachment(2));
    try testing.expect(needsResolveAttachment(4));
}

test "WarnOnce fires exactly once" {
    var w = WarnOnce{};
    try testing.expect(w.warn("msaa: {}", .{1}));
    try testing.expect(!w.warn("msaa: {}", .{2}));
}

test "depthPrepassActive needs post, gate, and MSAA samples" {
    try testing.expect(depthPrepassActive(true, true, 4));
    try testing.expect(depthPrepassActive(true, true, 2));
    // Gate off (default): never runs — the off path stays bit-identical.
    try testing.expect(!depthPrepassActive(true, false, 4));
    // Post off: no post chain to feed.
    try testing.expect(!depthPrepassActive(false, true, 4));
    // 1x target: the main depth texture already exists, no prepass needed.
    try testing.expect(!depthPrepassActive(true, true, 1));
    try testing.expect(!depthPrepassActive(true, true, 0));
}

test "suppressDepthEffects lifts only under the prepass gate" {
    // 1x: nothing is ever suppressed.
    try testing.expect(!suppressDepthEffects(1, false));
    try testing.expect(!suppressDepthEffects(1, true));
    // MSAA without the gate: suppression applies (legacy behavior).
    try testing.expect(suppressDepthEffects(4, false));
    try testing.expect(suppressDepthEffects(2, false));
    // MSAA with the gate: the prepass feeds depth, suppression lifts.
    try testing.expect(!suppressDepthEffects(4, true));
}
