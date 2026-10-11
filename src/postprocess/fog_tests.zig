const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const fog = @import("fog.zig");

test "heightDensity is smooth across delta_y == 0 and strictly attenuates at high altitude" {
    const falloff: f32 = 0.05;
    // Camera and world at y = 0
    const d0 = fog.heightDensity(0.0, 0.0, falloff);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), d0, 0.001);

    // Limit from right: cam = 0, world = 0.0005
    const d_near = fog.heightDensity(0.0, 0.0005, falloff);
    try std.testing.expectApproxEqAbs(d0, d_near, 0.001);

    // High altitude (y = 50): density should be much lower (exp(-2.5) ~ 0.082)
    const d_high = fog.heightDensity(50.0, 50.0, falloff);
    try std.testing.expect(d_high < 0.1);
    try std.testing.expect(d_high > 0.0);
}

test "fogExtinction follows Beer-Lambert: monotonic, strictly in [0, 1]" {
    try std.testing.expectEqual(@as(f32, 0.0), fog.fogExtinction(0.0));
    try std.testing.expectEqual(@as(f32, 0.0), fog.fogExtinction(-1.0));

    var prev: f32 = 0.0;
    var tau: f32 = 0.1;
    while (tau <= 10.0) : (tau += 0.5) {
        const ext = fog.fogExtinction(tau);
        try std.testing.expect(ext >= prev);
        try std.testing.expect(ext <= 1.0);
        prev = ext;
    }
    // High optical depth approaches complete fog opacity
    try std.testing.expect(fog.fogExtinction(10.0) > 0.999);
}

test "sunInscatter peaks along sun direction and vanishes backwards" {
    const sun_dir = Vec3.new(0, 0.7071, 0.7071).normalize();
    const ray_forward = sun_dir;
    const ray_perp = Vec3.new(1, 0, 0);
    const ray_back = sun_dir.scale(-1.0);

    const s_fwd = fog.sunInscatter(ray_forward, sun_dir, 1.0);
    const s_perp = fog.sunInscatter(ray_perp, sun_dir, 1.0);
    const s_back = fog.sunInscatter(ray_back, sun_dir, 1.0);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), s_fwd, 0.001);
    try std.testing.expectEqual(@as(f32, 0.0), s_perp);
    try std.testing.expectEqual(@as(f32, 0.0), s_back);
}

test "skyHorizonHaze peaks at horizon and vanishes at zenith" {
    const density: f32 = 0.02;
    // Horizon ray (y = 0)
    const h_horizon = fog.skyHorizonHaze(0.0, density);
    try std.testing.expect(h_horizon > 0.0);

    // Zenith ray (y = 1.0): 1 - 4 * 1 < 0 -> clamped to 0
    const h_zenith = fog.skyHorizonHaze(1.0, density);
    try std.testing.expectEqual(@as(f32, 0.0), h_zenith);

    // Ray above 15 degrees (y = 0.3): 1 - 4 * 0.3 = -0.2 -> 0
    const h_high = fog.skyHorizonHaze(0.3, density);
    try std.testing.expectEqual(@as(f32, 0.0), h_high);
}

test "heightDensity with zero falloff is uniform density 1.0" {
    // Zero falloff is the uniform-atmosphere limit: the analytic mean of
    // exp(-lambda*y) tends to 1.0. Without the guard the sloped branch
    // computes 0/0, which the CPU clamp maps to 5 (5x over-fog on slopes).
    const zeros = [_]f32{ 0.0, -0.0 };
    for (zeros) |zero| {
        // Rising ray.
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), fog.heightDensity(0.0, 10.0, zero), 1e-6);
        // Falling ray.
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), fog.heightDensity(10.0, 0.0, zero), 1e-6);
        // Equal-height ray (delta_y == 0 branch).
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), fog.heightDensity(5.0, 5.0, zero), 1e-6);
        // Full Beer-Lambert chain stays finite: tau = (20-5)*0.015*1.0 = 0.225.
        const tau = fog.opticalDepth(20.0, 5.0, 0.015, fog.heightDensity(0.0, 10.0, zero));
        try std.testing.expect(std.math.isFinite(tau));
        try std.testing.expectApproxEqAbs(@as(f32, 0.225), tau, 1e-6);
        const ext = fog.fogExtinction(tau);
        try std.testing.expect(std.math.isFinite(ext));
        try std.testing.expectApproxEqAbs(@as(f32, 1.0 - @exp(-@as(f32, 0.225))), ext, 1e-6);
    }
}
