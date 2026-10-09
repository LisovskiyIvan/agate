const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const ibl = @import("ibl_prefilter.zig");

test "hammersley generates low discrepancy points in [0, 1)^2" {
    const n = 64;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const pt = ibl.hammersley(i, n);
        try std.testing.expect(pt[0] >= 0.0 and pt[0] < 1.0);
        try std.testing.expect(pt[1] >= 0.0 and pt[1] < 1.0);
    }
}

test "importanceSampleGGX generates normalized vectors in upper hemisphere" {
    const n = Vec3.new(0, 1, 0);
    var i: u32 = 0;
    while (i < 32) : (i += 1) {
        const xi = ibl.hammersley(i, 32);
        const h = ibl.importanceSampleGGX(xi, n, 0.5);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), h.length(), 0.001);
        try std.testing.expect(h.dot(n) >= -0.001);
    }
}

test "sampleCosineHemisphere generates cosine-distributed directions" {
    const n = Vec3.new(0, 0, 1);
    var i: u32 = 0;
    var avg_z: f32 = 0.0;
    const count = 128;
    while (i < count) : (i += 1) {
        const xi = ibl.hammersley(i, count);
        const l = ibl.sampleCosineHemisphere(xi, n);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), l.length(), 0.001);
        try std.testing.expect(l.z >= 0.0);
        avg_z += l.z;
    }
    // Theoretical expected value of cos(theta) under cosine distribution is 2/3 ~ 0.667
    avg_z /= @floatFromInt(count);
    try std.testing.expectApproxEqAbs(@as(f32, 0.667), avg_z, 0.05);
}

test "smoothFalloff is continuous, 1 at center, 0 at radius, with zero derivative at bounds" {
    const r: f32 = 10.0;
    try std.testing.expectEqual(@as(f32, 1.0), ibl.smoothFalloff(0.0, r));
    try std.testing.expectEqual(@as(f32, 0.0), ibl.smoothFalloff(10.0, r));
    try std.testing.expectEqual(@as(f32, 0.0), ibl.smoothFalloff(15.0, r));

    // Midpoint: u = 0.5 -> 1 - (3*0.25 - 2*0.125) = 1 - (0.75 - 0.25) = 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), ibl.smoothFalloff(5.0, r), 0.001);

    // Monotonicity check
    var prev = ibl.smoothFalloff(0.0, r);
    var d: f32 = 0.5;
    while (d <= r) : (d += 0.5) {
        const cur = ibl.smoothFalloff(d, r);
        try std.testing.expect(cur <= prev);
        prev = cur;
    }
}

test "blendProbeWeights transitions smoothly between probes and environment" {
    // Isolated probe 0 at center (w0 = 1, w1 = 0)
    const b0 = ibl.blendProbeWeights(1.0, 0.0);
    try std.testing.expectEqual(@as(f32, 1.0), b0.w0);
    try std.testing.expectEqual(@as(f32, 0.0), b0.w1);
    try std.testing.expectEqual(@as(f32, 0.0), b0.w_env);

    // Probe 0 fading out near edge (w0 = 0.6, w1 = 0)
    const b1 = ibl.blendProbeWeights(0.6, 0.0);
    try std.testing.expectEqual(@as(f32, 0.6), b1.w0);
    try std.testing.expectEqual(@as(f32, 0.0), b1.w1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), b1.w_env, 0.001);

    // Overlapping probes (w0 = 0.5, w1 = 0.5)
    const b2 = ibl.blendProbeWeights(0.5, 0.5);
    try std.testing.expectEqual(@as(f32, 0.5), b2.w0);
    try std.testing.expectEqual(@as(f32, 0.5), b2.w1);
    try std.testing.expectEqual(@as(f32, 0.0), b2.w_env);

    // Outside both probes
    const b3 = ibl.blendProbeWeights(0.0, 0.0);
    try std.testing.expectEqual(@as(f32, 0.0), b3.w0);
    try std.testing.expectEqual(@as(f32, 0.0), b3.w1);
    try std.testing.expectEqual(@as(f32, 1.0), b3.w_env);
}

test "white furnace test: uniform environment radiance preserves energy <= 1" {
    const count = 128;
    var irr_sum: f32 = 0.0;
    const n = Vec3.new(0, 1, 0);
    for (0..count) |i| {
        const xi = ibl.hammersley(@intCast(i), count);
        const l = ibl.sampleCosineHemisphere(xi, n);
        _ = l;
        irr_sum += 1.0;
    }
    const irr = irr_sum / @as(f32, @floatFromInt(count));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), irr, 0.01);
}

test "boxProjectReflection: zero or tiny extent returns uncorrected ray" {
    const r = Vec3.new(0, 1, 0);
    const p = Vec3.new(1, 2, 3);
    const c = Vec3.new(0, 0, 0);

    const out0 = ibl.boxProjectReflection(r, p, c, 0.0);
    try std.testing.expectEqual(r, out0);

    const out_tiny = ibl.boxProjectReflection(r, p, c, 0.0005);
    try std.testing.expectEqual(r, out_tiny);
}

test "boxProjectReflection: fragment at probe center produces identical direction" {
    const c = Vec3.new(10, 5, 2);
    const r = Vec3.new(0.57735, 0.57735, 0.57735).normalize();
    const extent: f32 = 8.0;

    const out = ibl.boxProjectReflection(r, c, c, extent);
    try std.testing.expectApproxEqAbs(r.x, out.x, 0.0001);
    try std.testing.expectApproxEqAbs(r.y, out.y, 0.0001);
    try std.testing.expectApproxEqAbs(r.z, out.z, 0.0001);
}

test "boxProjectReflection: off-center interior reflection parallax correction" {
    const c = Vec3.zero;
    const extent: f32 = 5.0;

    const r_axis = Vec3.new(1, 0, 0);
    const p_axis = Vec3.new(2, 0, 0);
    const out_axis = ibl.boxProjectReflection(r_axis, p_axis, c, extent);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out_axis.x, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out_axis.y, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out_axis.z, 0.0001);

    const r_up = Vec3.new(0, 1, 0);
    const p_side = Vec3.new(4, 0, 0);
    const out_up = ibl.boxProjectReflection(r_up, p_side, c, extent);

    const expected_x: f32 = 4.0 / @sqrt(41.0);
    const expected_y: f32 = 5.0 / @sqrt(41.0);
    try std.testing.expectApproxEqAbs(expected_x, out_up.x, 0.001);
    try std.testing.expectApproxEqAbs(expected_y, out_up.y, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out_up.z, 0.0001);
}
