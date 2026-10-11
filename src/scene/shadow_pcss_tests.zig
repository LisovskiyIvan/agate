const std = @import("std");
const pcss = @import("shadow_pcss.zig");

test "penumbraRadius matches the analytic formula" {
    // (0.6 - 0.5) * 0.02 = 0.002, inside wide clamps: raw value passes through.
    try std.testing.expectApproxEqAbs(@as(f32, 0.002), pcss.penumbraRadius(0.6, 0.5, 0.02, 0.0, 1.0), 1e-6);
    // (0.9 - 0.3) * 0.05 = 0.03.
    try std.testing.expectApproxEqAbs(@as(f32, 0.03), pcss.penumbraRadius(0.9, 0.3, 0.05, 0.0, 1.0), 1e-6);
}

test "penumbraRadius clamps below min and above max" {
    // Near-contact: (0.5001 - 0.5) * 0.02 = 2e-6 -> min.
    try std.testing.expectEqual(@as(f32, 0.0005), pcss.penumbraRadius(0.5001, 0.5, 0.02, 0.0005, 0.01));
    // Far blocker: (0.9 - 0.1) * 0.02 = 0.016 -> max.
    try std.testing.expectEqual(@as(f32, 0.01), pcss.penumbraRadius(0.9, 0.1, 0.02, 0.0005, 0.01));
}

test "penumbraRadius guards zero and negative blocker depth" {
    try std.testing.expectEqual(@as(f32, 0.01), pcss.penumbraRadius(0.5, 0.0, 0.02, 0.0005, 0.01));
    try std.testing.expectEqual(@as(f32, 0.01), pcss.penumbraRadius(0.5, -0.3, 0.02, 0.0005, 0.01));
}

test "penumbraRadius floors receiver at or behind blocker to min" {
    // Coincident surfaces: raw 0.0 -> min.
    try std.testing.expectEqual(@as(f32, 0.0005), pcss.penumbraRadius(0.4, 0.4, 0.02, 0.0005, 0.01));
    // Receiver behind the blocker (float noise): raw negative -> min.
    try std.testing.expectEqual(@as(f32, 0.0005), pcss.penumbraRadius(0.3, 0.5, 0.02, 0.0005, 0.01));
}

test "averageBlocker is null without blockers" {
    try std.testing.expect(pcss.averageBlocker(0.0, 0) == null);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), pcss.averageBlocker(1.5, 3).?, 1e-6);
}

test "averageBlocker handles zero and maximum spread" {
    // Zero spread: three identical blockers average exactly.
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), pcss.averageBlocker(1.2, 3).?, 1e-6);
    // Maximum spread: blockers at the depth range extremes average to mid.
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), pcss.averageBlocker(1.0, 2).?, 1e-6);
}

test "tapWeight normalizes the PCF disk" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0625), pcss.tapWeight(16), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), pcss.tapWeight(16) * 16.0, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), pcss.tapWeight(8) * 8.0, 1e-6);
    try std.testing.expectEqual(@as(f32, 0.0), pcss.tapWeight(0));
}

test "pcfTapCount matches the shader cascade LOD" {
    try std.testing.expectEqual(@as(u32, 16), pcss.pcfTapCount(0));
    try std.testing.expectEqual(@as(u32, 8), pcss.pcfTapCount(1));
    try std.testing.expectEqual(@as(u32, 8), pcss.pcfTapCount(2));
    try std.testing.expectEqual(@as(u32, 8), pcss.pcfTapCount(3));
}

test "resolveLit is fully lit without blockers" {
    // No blockers: fully lit even if a stale PCF value says shadowed.
    try std.testing.expectEqual(@as(f32, 1.0), pcss.resolveLit(null, 0.0));
    try std.testing.expectEqual(@as(f32, 1.0), pcss.resolveLit(null, 0.37));
    // Blockers present: the PCF result stands.
    try std.testing.expectEqual(@as(f32, 0.25), pcss.resolveLit(0.4, 0.25));
}

test "isEnabled thresholds the packed flag at one half" {
    try std.testing.expect(!pcss.isEnabled(0.0));
    try std.testing.expect(!pcss.isEnabled(0.5));
    try std.testing.expect(pcss.isEnabled(0.5001));
    try std.testing.expect(pcss.isEnabled(1.0));
}

test "spotPcssEnabled thresholds source_radius at epsilon" {
    try std.testing.expect(!pcss.spotPcssEnabled(0.0));
    try std.testing.expect(!pcss.spotPcssEnabled(pcss.spot_pcss_epsilon));
    try std.testing.expect(pcss.spotPcssEnabled(pcss.spot_pcss_epsilon + 1e-6));
    try std.testing.expect(pcss.spotPcssEnabled(0.02));
}

test "spot PCSS constants match the shader tap budgets" {
    try std.testing.expectEqual(@as(u32, 8), pcss.spot_blocker_sample_count);
    try std.testing.expectEqual(@as(u32, 8), pcss.spot_pcf_tap_count);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), pcss.spot_blocker_radius, 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0005), pcss.spot_min_penumbra, 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), pcss.spot_max_penumbra, 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0001), pcss.spot_pcss_epsilon, 1e-8);
}

test "penumbraRadius covers the spot golden values" {
    // Mid-range blocker with a sun-scale emitter: (0.6 - 0.5) * 0.02 = 0.002.
    try std.testing.expectApproxEqAbs(@as(f32, 0.002), pcss.penumbraRadius(0.6, 0.5, 0.02, pcss.spot_min_penumbra, pcss.spot_max_penumbra), 1e-6);
    // Contact -> min penumbra, distant blocker -> max penumbra.
    try std.testing.expectEqual(pcss.spot_min_penumbra, pcss.penumbraRadius(0.5001, 0.5, 0.02, pcss.spot_min_penumbra, pcss.spot_max_penumbra));
    try std.testing.expectEqual(pcss.spot_max_penumbra, pcss.penumbraRadius(0.9, 0.1, 0.02, pcss.spot_min_penumbra, pcss.spot_max_penumbra));
}

test "pointPcssEnabled thresholds source_radius at epsilon" {
    try std.testing.expect(!pcss.pointPcssEnabled(0.0));
    try std.testing.expect(!pcss.pointPcssEnabled(pcss.point_pcss_epsilon));
    try std.testing.expect(pcss.pointPcssEnabled(pcss.point_pcss_epsilon + 1e-6));
    try std.testing.expect(pcss.pointPcssEnabled(0.02));
}

test "point PCSS constants match the shader tap budgets" {
    try std.testing.expectEqual(@as(u32, 8), pcss.point_blocker_sample_count);
    try std.testing.expectEqual(@as(u32, 8), pcss.point_pcf_tap_count);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), pcss.point_blocker_radius, 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0005), pcss.point_min_penumbra, 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.01), pcss.point_max_penumbra, 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0001), pcss.point_pcss_epsilon, 1e-8);
}

test "penumbraRadius covers the point golden values" {
    // Same shared formula as the spot path, evaluated at the point clamps.
    try std.testing.expectApproxEqAbs(@as(f32, 0.002), pcss.penumbraRadius(0.6, 0.5, 0.02, pcss.point_min_penumbra, pcss.point_max_penumbra), 1e-6);
    // Contact -> min penumbra, distant blocker -> max penumbra.
    try std.testing.expectEqual(pcss.point_min_penumbra, pcss.penumbraRadius(0.5001, 0.5, 0.02, pcss.point_min_penumbra, pcss.point_max_penumbra));
    try std.testing.expectEqual(pcss.point_max_penumbra, pcss.penumbraRadius(0.9, 0.1, 0.02, pcss.point_min_penumbra, pcss.point_max_penumbra));
}

const math = @import("math");

test "pointFaceIndex selects the major-axis face with X>Y>Z tie-break" {
    const V = math.Vec3.new;
    try std.testing.expectEqual(@as(usize, 0), pcss.pointFaceIndex(V(1, 0, 0)));
    try std.testing.expectEqual(@as(usize, 1), pcss.pointFaceIndex(V(-2, 0.5, 0.5)));
    try std.testing.expectEqual(@as(usize, 2), pcss.pointFaceIndex(V(0.1, 3, 0.1)));
    try std.testing.expectEqual(@as(usize, 3), pcss.pointFaceIndex(V(0, -1, 0)));
    try std.testing.expectEqual(@as(usize, 4), pcss.pointFaceIndex(V(0, 0, 5)));
    try std.testing.expectEqual(@as(usize, 5), pcss.pointFaceIndex(V(0.2, 0.1, -4)));
    // Exact ties: X beats Y beats Z, matching the GLSL >= chain.
    try std.testing.expectEqual(@as(usize, 0), pcss.pointFaceIndex(V(1, 1, 1)));
    try std.testing.expectEqual(@as(usize, 1), pcss.pointFaceIndex(V(-1, 1, 1)));
    try std.testing.expectEqual(@as(usize, 0), pcss.pointFaceIndex(V(1, 1, 0)));
    try std.testing.expectEqual(@as(usize, 2), pcss.pointFaceIndex(V(0, 1, 1)));
    try std.testing.expectEqual(@as(usize, 3), pcss.pointFaceIndex(V(0, -1, -1)));
    // Degenerate zero direction falls into +X exactly like the shader
    // (0.0 >= 0.0 selects the x branch, 0.0 >= 0.0 the positive face).
    try std.testing.expectEqual(@as(usize, 0), pcss.pointFaceIndex(V(0, 0, 0)));
}

test "pointAtlasUv maps all 6 faces x 2 slots without overlap" {
    // Tile corners land exactly on the tile rect: local (0,0) -> min,
    // local (1,1) -> max, center -> center.
    for (0..2) |slot| {
        for (0..6) |face| {
            const b = pcss.pointTileBounds(slot, face);
            const lo = pcss.pointAtlasUv(slot, face, .{ 0.0, 0.0 });
            const hi = pcss.pointAtlasUv(slot, face, .{ 1.0, 1.0 });
            const mid = pcss.pointAtlasUv(slot, face, .{ 0.5, 0.5 });
            try std.testing.expectApproxEqAbs(b.min[0], lo[0], 1e-7);
            try std.testing.expectApproxEqAbs(b.min[1], lo[1], 1e-7);
            try std.testing.expectApproxEqAbs(b.max[0], hi[0], 1e-7);
            try std.testing.expectApproxEqAbs(b.max[1], hi[1], 1e-7);
            try std.testing.expectApproxEqAbs((b.min[0] + b.max[0]) * 0.5, mid[0], 1e-7);
            try std.testing.expectApproxEqAbs((b.min[1] + b.max[1]) * 0.5, mid[1], 1e-7);
            // Tile size: 1/6 wide, 1/2 tall (matches the shader arithmetic).
            try std.testing.expectApproxEqAbs(@as(f32, 1.0) / 6.0, b.max[0] - b.min[0], 1e-7);
            try std.testing.expectApproxEqAbs(@as(f32, 0.5), b.max[1] - b.min[1], 1e-7);
        }
    }
    // Golden spot-check: face 5, slot 1, local (0.25, 0.75).
    const g = pcss.pointAtlasUv(1, 5, .{ 0.25, 0.75 });
    try std.testing.expectApproxEqAbs(@as(f32, (5.25 / 6.0)), g[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.875), g[1], 1e-6);
}

test "pointTapSkipped never clamps: out-of-range taps carry no contribution" {
    // Interior tap passes.
    try std.testing.expect(!pcss.pointTapSkipped(1.0, 0.5, .{ 0.5, 0.5 }));
    // Exact edges are valid (the shader skips only on strict < 0 / > 1).
    try std.testing.expect(!pcss.pointTapSkipped(0.001, 0.0, .{ 0.0, 0.0 }));
    try std.testing.expect(!pcss.pointTapSkipped(0.001, 1.0, .{ 1.0, 1.0 }));
    // Behind the tap-face camera: skip.
    try std.testing.expect(pcss.pointTapSkipped(0.0, 0.5, .{ 0.5, 0.5 }));
    try std.testing.expect(pcss.pointTapSkipped(-2.0, 0.5, .{ 0.5, 0.5 }));
    // Outside the shadow near/far range: skip.
    try std.testing.expect(pcss.pointTapSkipped(1.0, -0.01, .{ 0.5, 0.5 }));
    try std.testing.expect(pcss.pointTapSkipped(1.0, 1.01, .{ 0.5, 0.5 }));
    // Off the tap tile: skip (this is the v1 face-clamp case — v2 drops
    // the tap instead of sampling the clamped edge texel).
    try std.testing.expect(pcss.pointTapSkipped(1.0, 0.5, .{ -0.01, 0.5 }));
    try std.testing.expect(pcss.pointTapSkipped(1.0, 0.5, .{ 0.5, 1.01 }));
}

test "pointBlockerGap is null for non-blockers, relative otherwise" {
    try std.testing.expect(pcss.pointBlockerGap(0.5, 0.5) == null);
    try std.testing.expect(pcss.pointBlockerGap(0.5, 0.7) == null);
    // (0.6 - 0.4) / 0.6 = 1/3.
    try std.testing.expectApproxEqAbs(@as(f32, 1.0) / 3.0, pcss.pointBlockerGap(0.6, 0.4).?, 1e-6);
    // Contact-ish blocker: gap near zero but positive.
    const g = pcss.pointBlockerGap(0.5, 0.4999).?;
    try std.testing.expect(g > 0.0 and g < 1e-3);
}

test "pointEffectiveBlocker reconstructs the single-face mean exactly" {
    // Three blockers in the receiver's own face: the gap average must
    // rebuild their arithmetic mean (this is the v1 bit-identical anchor).
    const r: f32 = 0.8;
    const blockers = [_]f32{ 0.5, 0.6, 0.7 };
    var gap_sum: f32 = 0.0;
    var gap_sum_direct: f32 = 0.0;
    for (blockers) |b| gap_sum += pcss.pointBlockerGap(r, b).?;
    gap_sum_direct = ((r - blockers[0]) + (r - blockers[1]) + (r - blockers[2])) / r;
    try std.testing.expectApproxEqAbs(gap_sum_direct, gap_sum, 1e-6);
    const avg_gap = gap_sum / 3.0;
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), pcss.pointEffectiveBlocker(r, avg_gap), 1e-6);
    // Zero gap (blocker at the receiver) reconstructs the receiver depth.
    try std.testing.expectApproxEqAbs(r, pcss.pointEffectiveBlocker(r, 0.0), 1e-7);
}

test "pointPerspectiveDepth01 matches Mat4.perspective projection" {
    // Pins the depth convention against the real matrix: projecting
    // (0, 0, -d) through Mat4.perspective(90, 1, near, far) must equal
    // the mirror for near, mid, and far distances.
    const near: f32 = 0.1;
    const far: f32 = 10.0;
    const p = math.Mat4.perspective(90.0, 1.0, near, far);
    const dists = [_]f32{ 0.1, 0.5, 2.0, 5.0, 9.999 };
    for (dists) |d| {
        const ndc_z = p.transformPoint(math.Vec3.new(0, 0, -d)).z;
        try std.testing.expectApproxEqAbs(pcss.pointPerspectiveDepth01(d, near, far), ndc_z, 1e-5);
        // Round-trip through the inverse.
        try std.testing.expectApproxEqAbs(d, pcss.pointViewDistance(ndc_z, near, far), 1e-4);
    }
    // Endpoints: near -> 0, far -> ~1.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), pcss.pointPerspectiveDepth01(near, near, far), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), pcss.pointPerspectiveDepth01(far, near, far), 1e-4);
}

test "same-ray geometry normalizes equally at a shared face edge" {
    // Blocker at 70% along the receiver ray: the view transform about the
    // light is linear, so both faces see the same depth ratio. At the
    // shared cube edge the forward distances coincide (symmetric 45-degree
    // view), hence both faces MUST produce the identical gap — this is the
    // cross-face normalization golden. Off the edge the perspective
    // nonlinearity leaks a bounded difference (documented v2 approximation).
    const near: f32 = 0.1;
    const far: f32 = 10.0;
    const s: f32 = 0.7;
    const z_edge: f32 = 5.0 * @cos(std.math.pi / 4.0);
    const r_edge = pcss.pointPerspectiveDepth01(z_edge, near, far);
    const b_edge = pcss.pointPerspectiveDepth01(s * z_edge, near, far);
    const gap_a = pcss.pointBlockerGap(r_edge, b_edge).?;
    const gap_b = pcss.pointBlockerGap(r_edge, b_edge).?;
    try std.testing.expectApproxEqAbs(gap_a, gap_b, 1e-7);
    // Same-ray pair deeper in one face: still a valid blocker gap in (0,1).
    const r_face = pcss.pointPerspectiveDepth01(5.0, near, far);
    const b_face = pcss.pointPerspectiveDepth01(s * 5.0, near, far);
    const gap_face = pcss.pointBlockerGap(r_face, b_face).?;
    try std.testing.expect(gap_face > 0.0 and gap_face < 1.0);
    // Gap is monotonic in the blocker fraction: nearer blocker -> bigger gap.
    const b_near = pcss.pointPerspectiveDepth01(0.4 * 5.0, near, far);
    try std.testing.expect(pcss.pointBlockerGap(r_face, b_near).? > gap_face);
}

test "pointWorldRadiusFromUv preserves the v1 texel footprint" {
    // Default blocker radius at 5m: 0.01 atlas-UV -> 0.6m world
    // (0.01 * 12 * 5), i.e. 15.36 texels at 2*5/256 m per texel.
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), pcss.pointWorldRadiusFromUv(0.01, 5.0), 1e-6);
    // Linear in both inputs; zero radius stays zero.
    try std.testing.expectApproxEqAbs(@as(f32, 0.12), pcss.pointWorldRadiusFromUv(0.002, 5.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.2), pcss.pointWorldRadiusFromUv(0.01, 10.0), 1e-6);
    try std.testing.expectEqual(@as(f32, 0.0), pcss.pointWorldRadiusFromUv(0.0, 5.0));
}
