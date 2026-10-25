const std = @import("std");
const testing = std.testing;
const lens_flare = @import("lens_flare.zig");
const options = @import("options.zig");

test "ghostVector points towards center and scales with spacing" {
    // Top-left pixel (0.1, 0.1) -> vector to center (0.5, 0.5) is (+0.4, +0.4)
    const vec = lens_flare.ghostVector(.{ 0.1, 0.1 }, 0.5);
    try testing.expectApproxEqAbs(@as(f32, 0.2), vec[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.2), vec[1], 1e-6);

    // Center pixel (0.5, 0.5) -> zero vector
    const center_vec = lens_flare.ghostVector(.{ 0.5, 0.5 }, 0.5);
    try testing.expectApproxEqAbs(@as(f32, 0.0), center_vec[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), center_vec[1], 1e-6);
}

test "ghostUv reflects across center for ghost 0 and steps along axis" {
    // Pixel at (0.3, 0.3)
    // Flipped across center (0.5, 0.5) is (0.7, 0.7)
    const uv0 = lens_flare.ghostUv(.{ 0.3, 0.3 }, 0, 0.25);
    try testing.expectApproxEqAbs(@as(f32, 0.7), uv0[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.7), uv0[1], 1e-6);

    // Ghost 1 with spacing 0.25: step is (0.5 - 0.3) * 0.25 = 0.05
    // uv1 = 0.7 + 0.05 = 0.75
    const uv1 = lens_flare.ghostUv(.{ 0.3, 0.3 }, 1, 0.25);
    try testing.expectApproxEqAbs(@as(f32, 0.75), uv1[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.75), uv1[1], 1e-6);
}

test "barrelWeight is 1.0 at center and falls off to zero at radius 0.5" {
    const w_center = lens_flare.barrelWeight(.{ 0.5, 0.5 });
    try testing.expectApproxEqAbs(@as(f32, 1.0), w_center, 1e-6);

    // At (0.5, 1.0), distance to center is 0.5, d = 1.0, 1 - d = 0
    const w_edge = lens_flare.barrelWeight(.{ 0.5, 1.0 });
    try testing.expectApproxEqAbs(@as(f32, 0.0), w_edge, 1e-6);

    // Outside (e.g. corner (0.0, 0.0), dist ~0.707) is clamped to 0
    const w_corner = lens_flare.barrelWeight(.{ 0.0, 0.0 });
    try testing.expectApproxEqAbs(@as(f32, 0.0), w_corner, 1e-6);
}

test "haloWeight peaks at halo_radius and falls off beyond thickness" {
    // Pixel at distance 0.45 from center
    const uv_on_ring = [2]f32{ 0.5 + 0.45, 0.5 };
    const w_peak = lens_flare.haloWeight(uv_on_ring, 0.45, 0.1);
    try testing.expectApproxEqAbs(@as(f32, 1.0), w_peak, 1e-6);

    // Pixel at distance 0.45 + 0.1 = 0.55 (diff = 0.1 = thickness), weight drops to 0
    const uv_beyond = [2]f32{ 0.5 + 0.55, 0.5 };
    const w_zero = lens_flare.haloWeight(uv_beyond, 0.45, 0.1);
    try testing.expectApproxEqAbs(@as(f32, 0.0), w_zero, 1e-6);
}

test "chromaticOffset scales normalized direction by dispersion" {
    const offset = lens_flare.chromaticOffset(.{ 1.0, 0.0 }, 0.015);
    try testing.expectApproxEqAbs(@as(f32, 0.015), offset[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), offset[1], 1e-6);
}

test "lensFlareParams zeroes when disabled or bloom disabled" {
    var cfg = options.PostProcessOptions{};
    cfg.lens_flares.enabled = false;
    cfg.lens_flares.intensity = 0.5;
    try testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, lens_flare.lensFlareParams(cfg));
    try testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, lens_flare.lensFlareParams2(cfg));

    cfg.lens_flares.enabled = true;
    cfg.bloom.enabled = false;
    try testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, lens_flare.lensFlareParams(cfg));
    try testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, lens_flare.lensFlareParams2(cfg));
}

test "lensFlareParams packs active values correctly" {
    var cfg = options.PostProcessOptions{};
    cfg.bloom.enabled = true;
    cfg.lens_flares.enabled = true;
    cfg.lens_flares.intensity = 0.6;
    cfg.lens_flares.ghosts = 5;
    cfg.lens_flares.ghost_spacing = 0.3;
    cfg.lens_flares.halo_radius = 0.4;
    cfg.lens_flares.halo_thickness = 0.08;
    cfg.lens_flares.chromatic_dispersion = 0.02;

    const p1 = lens_flare.lensFlareParams(cfg);
    try testing.expectApproxEqAbs(@as(f32, 1.0), p1[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.6), p1[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.4), p1[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.02), p1[3], 1e-6);

    const p2 = lens_flare.lensFlareParams2(cfg);
    try testing.expectApproxEqAbs(@as(f32, 5.0), p2[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.3), p2[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.08), p2[2], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.0), p2[3], 1e-6);
}
