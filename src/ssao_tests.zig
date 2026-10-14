const std = @import("std");
const ssao = @import("ssao.zig");
const SSAOOptions = ssao.SSAOOptions;
const taa = @import("postprocess/taa.zig");
const ddgi = @import("gi/ddgi.zig");

test "ssao options clamping" {
    const custom = (SSAOOptions{ .sample_count = 100 }).clamped();
    try std.testing.expectEqual(@as(u32, 32), custom.sample_count);

    const low = (SSAOOptions{ .sample_count = 1 }).clamped();
    try std.testing.expectEqual(@as(u32, 4), low.sample_count);
}

test "ssao defaults preserve the hemisphere byte path" {
    const d = SSAOOptions{};
    try std.testing.expect(d.enabled);
    try std.testing.expectEqual(ssao.AoMode.hemisphere, d.mode);
    try std.testing.expect(!d.temporal.enabled);
    try std.testing.expectApproxEqAbs(ssao.AO_TEMPORAL_BLEND_DEFAULT, d.temporal.blend, 1e-7);
    const c = d.clamped();
    try std.testing.expectEqual(ssao.AoMode.hemisphere, c.mode);
    try std.testing.expect(!c.temporal.enabled);
    try std.testing.expectEqual(d.sample_count, c.sample_count);
    try std.testing.expectApproxEqAbs(d.radius, c.radius, 1e-7);
}

test "ssao temporal blend clamps to [0, 1]" {
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), (SSAOOptions{ .temporal = .{ .blend = 42.0 } }).clamped().temporal.blend, 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), (SSAOOptions{ .temporal = .{ .blend = -1.0 } }).clamped().temporal.blend, 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.85), ssao.aoTemporalBlend(.{ .temporal = .{ .blend = 0.85 } }), 1e-7);
}

test "ssao temporal activity gate" {
    try std.testing.expect(!ssao.aoTemporalActive(.{}));
    try std.testing.expect(ssao.aoTemporalActive(.{ .temporal = .{ .enabled = true } }));
    try std.testing.expect(!ssao.aoTemporalActive(.{ .enabled = false, .temporal = .{ .enabled = true } }));
}

test "ssao half size rounds odd dimensions up" {
    try std.testing.expectEqual(ssao.AoHalfSize{ .w = 360, .h = 270 }, ssao.aoHalfSize(720, 540));
    try std.testing.expectEqual(ssao.AoHalfSize{ .w = 361, .h = 271 }, ssao.aoHalfSize(721, 541));
    try std.testing.expectEqual(ssao.AoHalfSize{ .w = 1, .h = 1 }, ssao.aoHalfSize(1, 1));
    try std.testing.expectEqual(ssao.AoHalfSize{ .w = 0, .h = 0 }, ssao.aoHalfSize(0, 0));
}

test "ssao temporal reset mirrors the taa/ssgi conditions" {
    try std.testing.expect(ssao.aoTemporalShouldReset(.{ .first_frame = true }));
    try std.testing.expect(ssao.aoTemporalShouldReset(.{ .toggled_on = true }));
    try std.testing.expect(ssao.aoTemporalShouldReset(.{ .resized = true }));
    try std.testing.expect(ssao.aoTemporalShouldReset(.{ .camera_cut = true }));
    try std.testing.expect(ssao.aoTemporalShouldReset(.{ .explicit_reset = true }));
    try std.testing.expect(!ssao.aoTemporalShouldReset(.{}));
}

test "ssao temporal ping-pong alternates by parity" {
    try std.testing.expectEqual(@as(u8, 0), ssao.aoTemporalReadIndex(0));
    try std.testing.expectEqual(@as(u8, 1), ssao.aoTemporalWriteIndex(0));
    try std.testing.expectEqual(@as(u8, 1), ssao.aoTemporalReadIndex(7));
    try std.testing.expectEqual(@as(u8, 0), ssao.aoTemporalWriteIndex(7));
}

test "ssao history uv subtracts the velocity tap" {
    const uv = ssao.aoTemporalHistoryUv(.{ 0.5, 0.5 }, .{ 0.02, -0.01 });
    try std.testing.expectApproxEqAbs(@as(f32, 0.48), uv[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.51), uv[1], 1e-6);
}

test "ssao marker band cross-pins the taa band" {
    try std.testing.expectEqual(taa.TAA_REACTIVE_MARKER_LO, ssao.AO_TEMPORAL_MARKER_LO);
    try std.testing.expectEqual(taa.TAA_REACTIVE_MARKER_HI, ssao.AO_TEMPORAL_MARKER_HI);
    try std.testing.expect(ssao.aoTemporalIsMarker(0.25));
    try std.testing.expect(!ssao.aoTemporalIsMarker(0.0));
    try std.testing.expect(!ssao.aoTemporalIsMarker(0.6));
    try std.testing.expect(!ssao.aoTemporalIsMarker(std.math.inf(f32)));
}

test "ssao depth gate follows the bilateral threshold" {
    // radius 0.5 -> gate 0.1: 5cm blends, 50cm rejects.
    try std.testing.expect(ssao.aoTemporalDepthOk(1.0, 1.05, 0.5));
    try std.testing.expect(!ssao.aoTemporalDepthOk(1.0, 1.5, 0.5));
    try std.testing.expect(!ssao.aoTemporalDepthOk(std.math.inf(f32), 1.0, 0.5));
}

test "ssao normal gate rejects edges and degenerate normals" {
    const n: [3]f32 = .{ 0.0, 0.0, 1.0 };
    try std.testing.expect(ssao.aoTemporalNormalOk(n, n));
    try std.testing.expect(!ssao.aoTemporalNormalOk(n, .{ 0.0, 1.0, 0.0 }));
    try std.testing.expect(!ssao.aoTemporalNormalOk(.{ 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0 }));
}

test "ssao variance bounds pin the uniform neighborhood" {
    const box = ssao.aoTemporalVarianceBounds(0.7, .{0.7} ** 8, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), box.min, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), box.max, 1e-6);
    // One bright outlier widens the box but the mean stays inside.
    const box2 = ssao.aoTemporalVarianceBounds(0.5, .{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0 }, 1.0);
    try std.testing.expect(box2.min <= 0.5 and 0.5 <= box2.max);
    try std.testing.expect(box2.max < 1.0);
}

test "ssao rejection weight is 1 inside the box" {
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ssao.aoTemporalRejectionWeight(0.5, 0.4, 0.6), 1e-6);
    const w = ssao.aoTemporalRejectionWeight(2.0, 0.4, 0.6);
    try std.testing.expect(w < 1.0 and w >= 0.0);
}

test "ssao temporal resolve rejection paths return current" {
    const nb = [_]f32{0.8} ** 8;
    // Reset frame primes history.
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), ssao.aoTemporalResolve(0.8, nb, 0.2, false, 0.85, 1.0, true, true, false), 1e-6);
    // Marker / depth / normal rejection sheds history exactly.
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), ssao.aoTemporalResolve(0.8, nb, 0.2, true, 0.85, 1.0, true, true, true), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), ssao.aoTemporalResolve(0.8, nb, 0.2, true, 0.85, 1.0, false, true, false), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), ssao.aoTemporalResolve(0.8, nb, 0.2, true, 0.85, 1.0, true, false, false), 1e-6);
    // Zero blend holds the current frame.
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), ssao.aoTemporalResolve(0.8, nb, 0.79, true, 0.0, 1.0, true, true, false), 1e-6);
}

test "ssao temporal resolve converges between current and history" {
    const nb = [_]f32{ 0.78, 0.82, 0.78, 0.82, 0.78, 0.82, 0.78, 0.82 };
    const out = ssao.aoTemporalResolve(0.8, nb, 0.79, true, 0.85, 1.0, true, true, false);
    try std.testing.expect(out < 0.8 and out >= 0.79);
}

test "ssao horizon steps match the hemisphere tap budget" {
    try std.testing.expectEqual(@as(u32, 3), ssao.aoHorizonStepsPerSide(24));
    try std.testing.expectEqual(@as(u32, 2), ssao.aoHorizonStepsPerSide(4));
    try std.testing.expectEqual(@as(u32, 4), ssao.aoHorizonStepsPerSide(32));
    // 4 slices x 2 sides x steps ~= sample_count at the default.
    try std.testing.expectEqual(@as(u32, 24), 2 * ssao.AO_HORIZON_SLICES * ssao.aoHorizonStepsPerSide(24));
}

test "ssao geometry golden: flat plane is unoccluded in both modes" {
    // No tap passes the depth comparison on a flat fronto-parallel plane.
    var sum: f32 = 0.0;
    for (0..24) |i| {
        const w = ssao.aoKernelCos(ssao.aoKernelIndex(@intCast(i), 24));
        sum += ssao.aoHemisphereTap(0.0, 0.2, 0.5, false, w);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ssao.aoHemisphereResolve(sum, 24, 1.2), 1e-6);
    const flat = [_]f32{0.0} ** ssao.AO_HORIZON_SLICES;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ssao.aoHorizonResolve(flat, 1.2), 1e-6);
}

test "ssao geometry golden: saturated enclosure darkens without the historical amplification" {
    // Full enclosure: every executed tap blocked at 10cm in a 0.5m
    // radius (falloff 0.8). The cosine-weighted fraction equals the
    // falloff, so AO = 0.2^1.2 ~= 0.145: really dark, never fully
    // black, and far from the historical 11.43x-amplified 0.0 which
    // this test used to pin. This proves the fix, not a rewrite of
    // the old expectation: the denominator is now derived from the
    // executed taps' weights (mean ~0.56), not the 2.8/32 magic.
    const n: u32 = 24;
    var sum: f32 = 0.0;
    for (0..n) |i| {
        const u: u32 = @intCast(i);
        sum += ssao.aoHemisphereTap(1.0, 0.1, 0.5, true, ssao.aoKernelCos(ssao.aoKernelIndex(u, n)));
    }
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), sum / ssao.aoHemisphereDenom(n), 1e-6);
    const hemi = ssao.aoHemisphereResolve(sum, n, 1.2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.145), hemi, 1e-3);
    // Horizon maxima still saturate per direction, not per tap: the
    // per-tap elevation * falloff (1 - 0.15) * (1 - 0.1/0.5) = 0.68.
    const elev_tap: f32 = (1.0 - ssao.AO_HORIZON_OFFSET) * (1.0 - 0.1 / 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.68), elev_tap, 1e-6);
    const slice = ssao.aoHorizonSliceResolve(elev_tap, elev_tap);
    try std.testing.expectApproxEqAbs(@as(f32, 0.68), slice, 1e-6);
    const slices = [_]f32{slice} ** ssao.AO_HORIZON_SLICES;
    const horizon = ssao.aoHorizonResolve(slices, 1.2);
    // Max-filtering per direction saturates at the single-tap elevation
    // instead of summing all taps: the corner keeps ~0.255 AO. The
    // structural mode difference is retained under the new estimator.
    try std.testing.expectApproxEqAbs(@as(f32, 0.255), horizon, 1e-3);
    try std.testing.expect(horizon > hemi);
}

test "ssao geometry golden: quarter-plane keeps finite ambient" {
    // A perpendicular wall blocks roughly half the cosine-weighted
    // hemisphere (alternating taps: uniform directional coverage).
    // Ideal visibility is ~0.5; the estimator must retain finite
    // ambient here instead of the historical full black.
    const n: u32 = 24;
    var sum: f32 = 0.0;
    for (0..n) |i| {
        const u: u32 = @intCast(i);
        const blocked = (i % 2 == 0);
        sum += ssao.aoHemisphereTap(1.0, 0.1, 0.5, blocked, ssao.aoKernelCos(ssao.aoKernelIndex(u, n)));
    }
    const hemi = ssao.aoHemisphereResolve(sum, n, 1.2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.556), hemi, 2e-2);
    try std.testing.expect(hemi > 0.45);
}

test "ssao geometry golden: a lone blocked tap never blacks out" {
    // Corner-adjacent pixel where a single tap of 24 catches the wall:
    // the old 11.43x amplification turned even small minorities dark.
    // AO must stay within a few percent of unoccluded.
    const n: u32 = 24;
    const w0 = ssao.aoKernelCos(ssao.aoKernelIndex(0, n));
    const sum = ssao.aoHemisphereTap(1.0, 0.1, 0.5, true, w0);
    const hemi = ssao.aoHemisphereResolve(sum, n, 1.2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.991), hemi, 1e-3);
    try std.testing.expect(hemi > 0.9);
}

test "ssao geometry golden: grazing and rotated planes stay clean" {
    // Below the horizon offset the gate is binary: grazing hits that
    // pass the depth test still contribute nothing.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ssao.aoHemisphereTap(0.05, 0.2, 0.5, true, 0.56), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ssao.aoHemisphereTap(0.15, 0.2, 0.5, true, 0.56), 1e-7);
    // A rotated flat plane: no tap passes depth, any normal resolves 1.
    var sum: f32 = 0.0;
    for (0..16) |i| {
        const w = ssao.aoKernelCos(ssao.aoKernelIndex(@intCast(i), 16));
        sum += ssao.aoHemisphereTap(0.9, 0.2, 0.5, false, w);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ssao.aoHemisphereResolve(sum, 16, 1.2), 1e-6);
}

test "ssao estimator is monotonic in radius and power" {
    // Same blocked geometry at larger radius: falloff grows, AO falls.
    const w: f32 = 0.56;
    const near_tap = ssao.aoHemisphereTap(1.0, 0.2, 0.5, true, w);
    const far_tap = ssao.aoHemisphereTap(1.0, 0.2, 1.0, true, w);
    try std.testing.expect(far_tap > near_tap);
    const denom = ssao.aoHemisphereDenom(24);
    const ao_near = ssao.aoHemisphereResolve(denom * 0.6, 24, 1.2);
    const ao_far = ssao.aoHemisphereResolve(denom * 0.8, 24, 1.2);
    try std.testing.expect(ao_far < ao_near);
    // Same partial occlusion under higher power: AO falls.
    const partial = denom * 0.4;
    try std.testing.expect(ssao.aoHemisphereResolve(partial, 24, 2.0) < ssao.aoHemisphereResolve(partial, 24, 1.0));
}

test "ssao geometry golden: near exclusion kills self-occlusion" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ssao.aoHemisphereTap(1.0, 0.01, 0.5, true, 0.6), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ssao.aoHemisphereTap(1.0, 0.6, 0.5, true, 0.6), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ssao.aoHemisphereTap(-0.5, 0.1, 0.5, true, 0.6), 1e-7);
}

test "ssao kernel index stratifies the full kernel at every budget" {
    // N=32 is the identity: bit-identical to the historical first-N loop.
    for (0..32) |i| try std.testing.expectEqual(@as(u32, @intCast(i)), ssao.aoKernelIndex(@intCast(i), 32));
    // Shader-contract spot values (GLSL: ((2*i+1)*32)/(2*sample_count)).
    try std.testing.expectEqual(@as(u32, 0), ssao.aoKernelIndex(0, 24));
    try std.testing.expectEqual(@as(u32, 2), ssao.aoKernelIndex(1, 24));
    try std.testing.expectEqual(@as(u32, 31), ssao.aoKernelIndex(23, 24));
    // Every budget 4..32: in bounds, no duplicates (a true cycle over
    // distinct taps, never the same tap twice per frame).
    var n: u32 = 4;
    while (n <= 32) : (n += 1) {
        var seen = [_]bool{false} ** 32;
        for (0..n) |i| {
            const k = ssao.aoKernelIndex(@intCast(i), n);
            try std.testing.expect(k < 32);
            try std.testing.expect(!seen[k]);
            seen[k] = true;
        }
    }
    // No grazing/short-ray prefix: small budgets start mid-kernel and
    // span most of it (radius AND angle coverage).
    try std.testing.expectEqual(@as(u32, 4), ssao.aoKernelIndex(0, 4));
    try std.testing.expectEqual(@as(u32, 28), ssao.aoKernelIndex(3, 4));
    try std.testing.expectEqual(@as(u32, 2), ssao.aoKernelIndex(0, 8));
    try std.testing.expectEqual(@as(u32, 30), ssao.aoKernelIndex(7, 8));
}

test "ssao kernel weights average ~0.56 at every budget" {
    // The estimator denominator derives from these weights; pin the
    // mean so a future kernel rebuild cannot silently reintroduce a
    // hidden amplification.
    const budgets = [_]u32{ 4, 8, 16, 24, 32 };
    for (budgets) |n| {
        const mean = ssao.aoHemisphereDenom(n) / @as(f32, @floatFromInt(n));
        try std.testing.expectApproxEqAbs(@as(f32, 0.56), mean, 2e-2);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 32.0 * 0.56), ssao.aoHemisphereDenom(32), 1e-4);
    // Out-of-range budgets clamp, never divide by zero.
    try std.testing.expect(ssao.aoHemisphereDenom(0) > 0.0);
    try std.testing.expect(ssao.aoHemisphereDenom(100) > 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ssao.aoHemisphereResolve(0.0, 0, 1.2), 1e-7);
}

test "ssao degenerate gradients carry no surface, not a fake normal" {
    // Zero and collinear gradients are degenerate: null, so the caller
    // returns unoccluded instead of the historical normalize(-pos).
    try std.testing.expect(ssao.aoNormalFromGradients(.{ 0, 0, 0 }, .{ 0, 0, 0 }) == null);
    try std.testing.expect(ssao.aoNormalFromGradients(.{ 1, 0, 0 }, .{ 2, 0, 0 }) == null);
    try std.testing.expect(ssao.aoNormalFromGradients(.{ 1, 2, 3 }, .{ 2, 4, 6 }) == null);
    // Orthogonal gradients: unit normal out.
    const n = ssao.aoNormalFromGradients(.{ 1, 0, 0 }, .{ 0, 1, 0 }).?;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), n[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), n[1], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), n[2], 1e-7);
    // Scale-relative, not absolute: tiny but well-conditioned gradients
    // (small pixels, distant walls) still resolve — the old absolute
    // 1e-20 cutoff would have discarded this normal.
    const tiny = ssao.aoNormalFromGradients(.{ 1e-6, 0, 0 }, .{ 0, 1e-6, 0 }).?;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), tiny[2], 1e-5);
}

test "ssao oct codec round-trips unit normals" {
    const axes = [_][3]f32{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 }, .{ 0, 0, -1 }, .{ -0.57735026, 0.57735026, 0.57735026 } };
    for (axes) |n| {
        const back = ssao.aoOctDecode(ssao.aoOctEncode(n));
        for (0..3) |c| try std.testing.expectApproxEqAbs(n[c], back[c], 1e-5);
    }
    // Degenerate input hits the documented +Z sentinel, which the normal
    // gate treats like any other normal (the depth gate guards the edge).
    try std.testing.expectEqual([2]f32{ 0.5, 0.5 }, ssao.aoOctEncode(.{ 0, 0, 0 }));
    const sentinel = ssao.aoOctDecode(.{ 0.5, 0.5 });
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sentinel[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sentinel[1], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sentinel[2], 1e-7);
}

test "ssao oct codec cross-pins the ddgi convention" {
    const probes = [_][3]f32{ .{ 1, 0, 0 }, .{ 0, -1, 0 }, .{ 0.3, -0.5, -0.81 }, .{ 0, 0, -1 } };
    for (probes) |n| {
        const a = ssao.aoOctEncode(n);
        const b = ddgi.octEncode(n);
        try std.testing.expectApproxEqAbs(b[0], a[0], 1e-6);
        try std.testing.expectApproxEqAbs(b[1], a[1], 1e-6);
        const da = ssao.aoOctDecode(a);
        const db = ddgi.octDecode(b);
        for (0..3) |c| try std.testing.expectApproxEqAbs(db[c], da[c], 1e-6);
    }
}

test "ssao decoded history normal drives the rejection gate" {
    const cur: [3]f32 = .{ 0, 0, 1 };
    // Same surface next frame: history decodes to itself, gate passes.
    try std.testing.expect(ssao.aoTemporalNormalOk(cur, ssao.aoOctDecode(ssao.aoOctEncode(cur))));
    // Orthogonal surface at the reprojected UV: gate rejects.
    const side = ssao.aoOctDecode(ssao.aoOctEncode([3]f32{ 1, 0, 0 }));
    try std.testing.expect(!ssao.aoTemporalNormalOk(cur, side));
}

test "ssao reproject maps clip to history uv and expected depth" {
    // Identity previous VP: NDC passes through to UV, raw depth unchanged.
    const r = ssao.aoReproject([16]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    }, .{ 0.2, -0.4, 0.5 }).?;
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), r.uv[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), r.uv[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), r.raw_depth, 1e-6);
    // Lateral shift pushes the tap offscreen (caller primes instead).
    const shifted = ssao.aoReproject([16]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        2, 0, 0, 1,
    }, .{ 0, 0, 0.3 }).?;
    try std.testing.expect(shifted.uv[0] > 0.999 or shifted.uv[0] < 0.001);
    // Point behind the previous camera: no history, prime instead.
    const behind = ssao.aoReproject([16]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    }, .{ 0, 0, 0.5 });
    try std.testing.expect(behind != null); // identity keeps w=1
    const flipped = ssao.aoReproject([16]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, -1,
    }, .{ 0, 0, 0.5 });
    try std.testing.expect(flipped == null);
}

test "ssao temporal rotation cycles evenly over a bounded period" {
    try std.testing.expectEqual(@as(u32, 8), ssao.AO_TEMPORAL_ROTATION_PERIOD);
    // Frame 0 (reset/prime orientation) is exactly 0.0 in both modes.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ssao.aoTemporalRotation(0, .hemisphere), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ssao.aoTemporalRotation(0, .horizon), 1e-7);
    // Hemisphere even coverage: phase k holds k * 2pi/8.
    const step: f32 = 2.0 * std.math.pi / 8.0;
    var k: u32 = 1;
    while (k < 8) : (k += 1) {
        try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(k)) * step, ssao.aoTemporalRotation(k, .hemisphere), 1e-6);
    }
    // Bounded wrap: period multiples return to 0, phase 9 resumes phase 1.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ssao.aoTemporalRotation(8, .hemisphere), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ssao.aoTemporalRotation(16, .hemisphere), 1e-7);
    try std.testing.expectApproxEqAbs(ssao.aoTemporalRotation(1, .hemisphere), ssao.aoTemporalRotation(9, .hemisphere), 1e-7);
    // Distinct across one cycle (varying directions, not the same angle).
    try std.testing.expect(@abs(ssao.aoTemporalRotation(2, .hemisphere) - ssao.aoTemporalRotation(5, .hemisphere)) > 1e-3);
}

test "ssao horizon rotation micro-steps within one slice spacing" {
    // The horizon slice star (4 slices spaced pi/4 x 2 sides = 8
    // directions) is invariant under a 2pi/8 rotation — that would only
    // permute the same taps. The horizon step is (pi/4)/8 = pi/32.
    const hstep: f32 = (std.math.pi / 4.0) / 8.0;
    try std.testing.expectApproxEqAbs(hstep, ssao.aoTemporalRotation(1, .horizon), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0) * hstep, ssao.aoTemporalRotation(4, .horizon), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ssao.aoTemporalRotation(8, .horizon), 1e-7);
}

test "ssao horizon rotation stratifies 64 directions over the period" {
    // CPU mirror of the shader slice star with IGN = 0: frame k holds the
    // 8 directed samples rot(k) + j*pi/4 (j = 0..7, both slice sides).
    const period = ssao.AO_TEMPORAL_ROTATION_PERIOD;
    const two_pi: f32 = 2.0 * std.math.pi;
    var angles: [8 * 8]f32 = undefined;
    var n: usize = 0;
    var k: u32 = 0;
    while (k < period) : (k += 1) {
        const rot = ssao.aoTemporalRotation(k, .horizon);
        var j: u32 = 0;
        while (j < 8) : (j += 1) {
            const a = @mod(rot + @as(f32, @floatFromInt(j)) * (std.math.pi / 4.0), two_pi);
            angles[n] = a;
            n += 1;
        }
    }
    // All 64 directed angles pairwise distinct (min separation pi/32).
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var m: usize = i + 1;
        while (m < n) : (m += 1) {
            const d = @abs(angles[i] - angles[m]);
            const wrap = @min(d, two_pi - d);
            try std.testing.expect(wrap > 1e-4);
        }
    }
    // Every non-zero frame differs from frame 0 as new taps, not a
    // permutation of the same 8 directions.
    var f: usize = 1;
    while (f < period) : (f += 1) {
        var j: usize = 0;
        while (j < 8) : (j += 1) {
            var closest: f32 = two_pi;
            var g: usize = 0;
            while (g < 8) : (g += 1) {
                const d = @abs(angles[f * 8 + j] - angles[g]);
                closest = @min(closest, @min(d, two_pi - d));
            }
            try std.testing.expect(closest > 1e-4);
        }
    }
}

test "ssao temporal rotation fails closed to unjittered" {
    // Off by default: the gather uploads exactly 0.0 (extra_params.z) and
    // the shader adds nothing, preserving the original arithmetic.
    try std.testing.expect(!ssao.aoTemporalActive(.{}));
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ssao.aoTemporalRotation(0, .hemisphere), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ssao.aoTemporalRotation(0, .horizon), 1e-7);
    // Every fail-closed gate (multi-camera, MSAA, allocation failure with
    // no history output, history-invalid reset) gathers with rotation 0:
    // the pass only uploads aoTemporalRotation on active non-reset frames
    // with a successful ensureTemporal, and resets the phase with history.
    // Pin the reset orientation and the master-off gate here; the pass
    // headless tests pin the phase reset on those paths.
    try std.testing.expect(!ssao.aoTemporalActive(.{ .enabled = false, .temporal = .{ .enabled = true } }));
    try std.testing.expect(ssao.aoTemporalShouldReset(.{ .first_frame = true }));
}

test "ssao horizon step budget matches the configured steps" {
    // The shader caps the inner loop BEFORE any depth fetch, so the actual
    // trip count is the configured steps per side — never 4-with-zeroed
    // extras when 2 or 3 are requested.
    const cases = [_]struct { sc: u32, steps: u32 }{
        .{ .sc = 4, .steps = 2 },
        .{ .sc = 12, .steps = 2 },
        .{ .sc = 20, .steps = 2 },
        .{ .sc = 24, .steps = 3 },
        .{ .sc = 32, .steps = 4 },
    };
    for (cases) |c| {
        try std.testing.expectEqual(c.steps, ssao.aoHorizonStepsPerSide(c.sc));
        // 4 slices x 2 sides x actual steps: the executed tap count.
        const taps = 2 * ssao.AO_HORIZON_SLICES * ssao.aoHorizonStepsPerSide(c.sc);
        try std.testing.expectEqual(2 * ssao.AO_HORIZON_SLICES * c.steps, taps);
    }
    // Default budget still matches the hemisphere path exactly.
    try std.testing.expectEqual(@as(u32, 24), 2 * ssao.AO_HORIZON_SLICES * ssao.aoHorizonStepsPerSide(24));
}

test "ssao gather uniform contract pins the rotation lanes" {
    // extra_params.z (rotation) defaults to 0.0 and w stays unused (0.0):
    // temporal OFF uploads (sample_count, mode, 0, 0), identical values to
    // the pre-temporal path. Hemisphere adds the rotation only in the
    // active branch (phase != 0); the reset frame uses phase 0.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ssao.aoTemporalRotation(0, .hemisphere), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ssao.aoTemporalRotation(0, .horizon), 1e-7);
    try std.testing.expect(ssao.aoTemporalRotation(3, .hemisphere) > 1e-3);
    try std.testing.expect(ssao.aoTemporalRotation(3, .horizon) > 1e-3);
}

test "ssao depth gate keeps static history under camera dolly" {
    // Static point; previous camera 0.2 (clip units) nearer along view,
    // so the expected previous raw depth is 0.3 while the current raw
    // depth is 0.5. near=0.1 far=100, radius=0.2 -> gate 0.04.
    const near: f32 = 0.1;
    const far: f32 = 100.0;
    const lin = struct {
        fn f(d: f32) f32 {
            return (near * far) / @max(far - d * (far - near), 0.0001);
        }
    }.f;
    const prev_vp = [16]f32{
        1, 0, 0,    0,
        0, 1, 0,    0,
        0, 0, 1,    0,
        0, 0, -0.2, 1,
    };
    const r = ssao.aoReproject(prev_vp, .{ 0, 0, 0.5 }).?;
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), r.raw_depth, 1e-6);
    const expected_lin = lin(r.raw_depth);
    const hist_lin = lin(0.3); // what the previous frame stored
    try std.testing.expect(ssao.aoTemporalDepthOk(expected_lin, hist_lin, 0.2));
    // The old direct compare (current vs history) would wrongly reject:
    // |lin(0.5) - lin(0.3)| ~= 0.057 >= 0.04. Pin both sides so the
    // golden documents WHY the expectation exists.
    try std.testing.expectApproxEqAbs(@as(f32, 0.057), @abs(lin(0.5) - lin(0.3)), 1e-3);
    try std.testing.expect(!ssao.aoTemporalDepthOk(lin(0.5), hist_lin, 0.2));
}
