const std = @import("std");
const wboit = @import("wboit_pass.zig");
const WboitPass = wboit.WboitPass;
const wboitWeight = wboit.wboitWeight;
const wboitAccumulate = wboit.wboitAccumulate;
const wboitComposite = wboit.wboitComposite;

test "wboitWeight: zero alpha yields zero weight" {
    try std.testing.expectEqual(@as(f32, 0.0), wboitWeight(10.0, 0.0));
    try std.testing.expectEqual(@as(f32, 0.0), wboitWeight(100.0, 0.0));
}

test "wboitWeight: depth falloff gives higher weight to closer surfaces" {
    const w_near = wboitWeight(2.0, 1.0);
    const w_mid = wboitWeight(20.0, 1.0);
    const w_far = wboitWeight(200.0, 1.0);
    const w_very_far = wboitWeight(800.0, 1.0);

    try std.testing.expect(w_near > w_mid);
    try std.testing.expect(w_mid > w_far);
    try std.testing.expect(w_far > w_very_far);
}

test "wboitWeight: symmetric with respect to sign of z" {
    const w_pos = wboitWeight(25.0, 0.8);
    const w_neg = wboitWeight(-25.0, 0.8);
    try std.testing.expectApproxEqAbs(w_pos, w_neg, 0.0001);
}

test "wboitWeight: clamped within finite range" {
    const w_extreme_near = wboitWeight(0.00001, 1.0);
    try std.testing.expect(w_extreme_near <= 3000.0);

    const w_extreme_far = wboitWeight(10000.0, 1.0);
    try std.testing.expect(w_extreme_far >= 0.01);
}

test "wboitAccumulate: packs weighted RGB and A" {
    const color = [3]f32{ 0.8, 0.2, 0.4 };
    const alpha: f32 = 0.5;
    const z: f32 = 10.0;
    const acc = wboitAccumulate(color, alpha, z);
    const w = wboitWeight(z, alpha);
    const aw = alpha * w;

    try std.testing.expectApproxEqAbs(color[0] * aw, acc[0], 0.0001);
    try std.testing.expectApproxEqAbs(color[1] * aw, acc[1], 0.0001);
    try std.testing.expectApproxEqAbs(color[2] * aw, acc[2], 0.0001);
    try std.testing.expectApproxEqAbs(aw, acc[3], 0.0001);
}

test "wboitComposite: untouched background when reveal is 1.0" {
    const bg = [3]f32{ 0.1, 0.2, 0.3 };
    const accum = [4]f32{ 0.0, 0.0, 0.0, 0.0 };
    const comp = wboitComposite(accum, 1.0, bg);
    try std.testing.expectEqual(bg[0], comp[0]);
    try std.testing.expectEqual(bg[1], comp[1]);
    try std.testing.expectEqual(bg[2], comp[2]);
}

test "wboitComposite: single layer bit-identically matches standard alpha blend" {
    const bg = [3]f32{ 0.2, 0.4, 0.6 };
    const fg = [3]f32{ 0.9, 0.1, 0.3 };
    const alpha: f32 = 0.6;
    const z: f32 = 15.0;

    // Single layer accumulation
    const accum = wboitAccumulate(fg, alpha, z);
    const reveal = 1.0 - alpha; // Single layer product

    const comp = wboitComposite(accum, reveal, bg);

    // Standard alpha blend: C_out = fg * alpha + bg * (1 - alpha)
    const expected_r = fg[0] * alpha + bg[0] * (1.0 - alpha);
    const expected_g = fg[1] * alpha + bg[1] * (1.0 - alpha);
    const expected_b = fg[2] * alpha + bg[2] * (1.0 - alpha);

    try std.testing.expectApproxEqAbs(expected_r, comp[0], 0.0001);
    try std.testing.expectApproxEqAbs(expected_g, comp[1], 0.0001);
    try std.testing.expectApproxEqAbs(expected_b, comp[2], 0.0001);
}

test "wboitComposite: fully opaque layer completely occludes background" {
    const bg = [3]f32{ 0.2, 0.4, 0.6 };
    const fg = [3]f32{ 0.7, 0.8, 0.9 };
    const alpha: f32 = 1.0;
    const z: f32 = 5.0;

    const accum = wboitAccumulate(fg, alpha, z);
    const reveal: f32 = 0.0; // 1 - 1.0 = 0.0

    const comp = wboitComposite(accum, reveal, bg);

    try std.testing.expectApproxEqAbs(fg[0], comp[0], 0.0001);
    try std.testing.expectApproxEqAbs(fg[1], comp[1], 0.0001);
    try std.testing.expectApproxEqAbs(fg[2], comp[2], 0.0001);
}

test "wboitComposite: two overlapping transparent layers combine smoothly" {
    const bg = [3]f32{ 0.0, 0.0, 0.0 };
    const layer1 = [3]f32{ 1.0, 0.0, 0.0 }; // Red layer at z = 5
    const alpha1: f32 = 0.5;
    const layer2 = [3]f32{ 0.0, 0.0, 1.0 }; // Blue layer at z = 10
    const alpha2: f32 = 0.5;

    const acc1 = wboitAccumulate(layer1, alpha1, 5.0);
    const acc2 = wboitAccumulate(layer2, alpha2, 10.0);
    const accum = [4]f32{
        acc1[0] + acc2[0],
        acc1[1] + acc2[1],
        acc1[2] + acc2[2],
        acc1[3] + acc2[3],
    };
    const reveal = (1.0 - alpha1) * (1.0 - alpha2); // 0.5 * 0.5 = 0.25

    const comp = wboitComposite(accum, reveal, bg);

    // Because layer 1 is closer (z = 5 vs z = 10), red should have higher weight than blue
    try std.testing.expect(comp[0] > comp[2]);
    // Both should contribute
    try std.testing.expect(comp[0] > 0.0);
    try std.testing.expect(comp[2] > 0.0);
    // Green was zero for both
    try std.testing.expectApproxEqAbs(0.0, comp[1], 0.0001);
}

test "WboitPass: headless lifecycle is clean and safe" {
    var pass = WboitPass{};
    try std.testing.expectEqual(@as(i32, 0), pass.width);
    try std.testing.expectEqual(@as(i32, 0), pass.height);
    try std.testing.expect(!pass.targetsValid());

    // Safe to call destroy/deinit on empty struct
    pass.destroyTargets();
    pass.deinit();
    try std.testing.expectEqual(@as(i32, 0), pass.width);
}
