const std = @import("std");

// Regression scope: applyMotionBlur gather kernel in
// src/shaders/postprocess.glsl ONLY. The loop formula
// `t = i/(samples-1) - 0.5` must cover BOTH endpoints (i = 0..N-1)
// so the kernel is a zero-mean centered box for every configured
// sample count (2..32). Starting the loop at i = 1 drops the -0.5
// endpoint and biases the blur toward +velocity (fully one-sided
// at N = 2). No general GLSL parser: brace-matched body extract
// plus full loop-header detection (fixed vs legacy; unknown headers
// fail loudly), then numeric kernel moments from the pinned formula.
const src = @embedFile("../shaders/postprocess.glsl");

const k_signature = "vec3 applyMotionBlur(vec3 color, vec2 uv, float depth)";
const k_formula = "float(i) / float(samples - 1) - 0.5";
// Full loop headers, pinned verbatim: the fixed shape covers both
// endpoints (i = 0..N-1), the legacy shape starts at 1 and drops -0.5.
// Any other header is an unknown shape and must fail loudly, never be
// silently interpreted as legacy.
const k_loop_fixed = "for (int i = 0; i < samples; ++i)";
const k_loop_legacy = "for (int i = 1; i < samples; ++i)";
// Additive raw-scene gather inside the loop, shared by both shapes.
const k_gather = "acc += texture(sampler2D(scene_tex, smp), sample_uv).rgb;";
// Return normalization both shapes divide by: N uniform taps scaled by
// 1/N, the actual unit-gain term (not just the tap count).
const k_normalize = "return acc * (1.0 / float(samples));";
const k_acc_center = "vec3 acc = color;";
const k_acc_correction = "color - texture(sampler2D(scene_tex, smp), uv).rgb";

fn extractBody() ?[]const u8 {
    const start = std.mem.indexOf(u8, src, k_signature) orelse return null;
    const brace = std.mem.indexOfScalarPos(u8, src, start, '{') orelse return null;
    var depth: usize = 0;
    var i = brace;
    while (i < src.len) : (i += 1) {
        if (src[i] == '{') {
            depth += 1;
        } else if (src[i] == '}') {
            depth -= 1;
            if (depth == 0) return src[brace..i];
        }
    }
    return null;
}

fn tapOffset(i: u32, samples: u32) f32 {
    const fi: f32 = @floatFromInt(i);
    const denom: f32 = @floatFromInt(samples - 1);
    return fi / denom - 0.5;
}

// Kernel the CURRENT shader source actually gathers: shape is read
// from the source (full loop header + acc initializer), offsets from
// the pinned loop formula. Returns the tap slice, or null for an
// unknown loop header (callers fail; never assume legacy).
fn shaderKernel(body: []const u8, samples: u32, out: []f32) ?[]f32 {
    const fixed = std.mem.indexOf(u8, body, k_loop_fixed) != null;
    const legacy = std.mem.indexOf(u8, body, k_loop_legacy) != null;
    if (fixed == legacy) return null;
    var n: usize = 0;
    if (fixed) {
        // Fixed shape: N raw taps over both endpoints, no duplicate center.
        var i: u32 = 0;
        while (i < samples) : (i += 1) {
            out[n] = tapOffset(i, samples);
            n += 1;
        }
    } else {
        // Legacy shape: CA-carrying center (offset 0) + i = 1..N-1.
        out[n] = 0.0;
        n += 1;
        var i: u32 = 1;
        while (i < samples) : (i += 1) {
            out[n] = tapOffset(i, samples);
            n += 1;
        }
    }
    return out[0..n];
}

fn kernelMean(taps: []const f32) f64 {
    var sum: f64 = 0.0;
    for (taps) |t| sum += @as(f64, t);
    return sum / @as(f64, @floatCast(@as(f32, @floatFromInt(taps.len))));
}

test "motion blur kernel is zero-mean over all configured counts 2..32" {
    const body = extractBody() orelse return error.MissingApplyMotionBlur;
    // Numeric reference is tied to this exact formula string in the shader.
    try std.testing.expect(std.mem.indexOf(u8, body, k_formula) != null);
    var buf: [32]f32 = undefined;
    var n: u32 = 2;
    while (n <= 32) : (n += 1) {
        const taps = shaderKernel(body, n, &buf) orelse return error.UnknownMotionBlurLoop;
        try std.testing.expectEqual(@as(usize, n), taps.len);
        try std.testing.expectApproxEqAbs(@as(f64, 0.0), kernelMean(taps), 1e-6);
    }
}

test "motion blur minimum 2 taps span { -0.5, +0.5 }" {
    const body = extractBody() orelse return error.MissingApplyMotionBlur;
    var buf: [32]f32 = undefined;
    const taps = shaderKernel(body, 2, &buf) orelse return error.UnknownMotionBlurLoop;
    try std.testing.expectEqual(@as(usize, 2), taps.len);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), taps[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), taps[1], 1e-7);
}

test "motion blur kernel has unit gain and pair symmetry" {
    const body = extractBody() orelse return error.MissingApplyMotionBlur;
    // Gain is the actual shader math, not just the tap count: every tap
    // is added with weight 1 (k_gather) and the sum is scaled by 1/N
    // (k_normalize), so a uniform field returns exactly itself.
    try std.testing.expect(std.mem.indexOf(u8, body, k_gather) != null);
    try std.testing.expect(std.mem.indexOf(u8, body, k_normalize) != null);
    var buf: [32]f32 = undefined;
    var n: u32 = 2;
    while (n <= 32) : (n += 1) {
        const taps = shaderKernel(body, n, &buf) orelse return error.UnknownMotionBlurLoop;
        // Uniform weights, N taps normalized by 1/N: gain is exactly 1.
        try std.testing.expectEqual(@as(usize, n), taps.len);
        var k: usize = 0;
        while (k < taps.len / 2) : (k += 1) {
            try std.testing.expectApproxEqAbs(
                taps[k] + taps[taps.len - 1 - k],
                @as(f32, 0.0),
                1e-6,
            );
        }
    }
}

test "motion blur preserves the CA center delta and keeps all bypasses" {
    const body = extractBody() orelse return error.MissingApplyMotionBlur;
    const fixed = std.mem.indexOf(u8, body, k_loop_fixed) != null;
    const legacy = std.mem.indexOf(u8, body, k_loop_legacy) != null;
    if (fixed == legacy) return error.UnknownMotionBlurLoop;
    if (fixed) {
        // sampleScene feeds the CA-shifted center in `color`; the gather
        // taps raw scene radiance, so acc must start as the CA delta
        // (color - raw center): the /N normalize then keeps exactly delta/N,
        // the same contribution as the old acc = color shape.
        try std.testing.expect(std.mem.indexOf(u8, body, k_acc_correction) != null);
        try std.testing.expect(std.mem.indexOf(u8, body, k_acc_center) == null);
    } else {
        // Legacy pre-fix shape stays recognized so the moment tests above
        // keep failing on it (never silently reclassified).
        try std.testing.expect(std.mem.indexOf(u8, body, k_acc_center) != null);
    }
    // Non-blur / background / sub-pixel exits and velocity lanes unchanged.
    try std.testing.expect(std.mem.indexOf(u8, body, "if (motion_blur_params.x < 0.5) return color;") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "if (depth >= 1.0) return color;") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "if (speed < resolution.z * 0.75) return color;") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "vel_sample.a > 0.5") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "(uv - prev_uv) * motion_blur_params.y") != null);
}
