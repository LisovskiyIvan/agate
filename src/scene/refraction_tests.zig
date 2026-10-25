const std = @import("std");
const testing = std.testing;
const math = @import("math");
const Vec3 = math.Vec3;
const queues_mod = @import("render_queue.zig");
const refraction = @import("refraction.zig");
const needsCapture = refraction.needsCapture;
const refractRay = refraction.refractRay;
const reflectRay = refraction.reflectRay;
const refractDispersionIors = refraction.refractDispersionIors;
const slabExitPoint = refraction.slabExitPoint;
const screenEdgeWeight = refraction.screenEdgeWeight;
const dielectricFresnel = refraction.dielectricFresnel;

test "refraction empty queues need no capture" {
    const queues = queues_mod.RenderQueues{};
    try testing.expect(!needsCapture(&queues));
}

test "refractRay follows Snell's law at normal incidence" {
    const V = Vec3.new(0, 0, 1);
    const N = Vec3.new(0, 0, 1);
    const ray = refractRay(V, N, 1.5) orelse return error.TestUnexpectedResult;

    try testing.expectApproxEqAbs(@as(f32, 0.0), ray.x, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.0), ray.y, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -1.0), ray.z, 1e-5);
}

test "refractRay bends toward normal in denser medium" {
    // 45 degree incident angle in XZ plane
    const inv_sqrt2: f32 = 1.0 / @sqrt(2.0);
    const V = Vec3.new(inv_sqrt2, 0, inv_sqrt2);
    const N = Vec3.new(0, 0, 1);
    const ray = refractRay(V, N, 1.5) orelse return error.TestUnexpectedResult;

    // By Snell's law: sin(theta_t) = sin(45) / 1.5 = (1 / sqrt(2)) / 1.5 ≈ 0.4714
    // ray.x should be -sin(theta_t) ≈ -0.4714
    // ray.z should be -cos(theta_t) ≈ -sqrt(1 - 0.4714^2) ≈ -0.8819
    try testing.expectApproxEqAbs(@as(f32, -inv_sqrt2 / 1.5), ray.x, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, -@sqrt(1.0 - (inv_sqrt2 / 1.5) * (inv_sqrt2 / 1.5))), ray.z, 1e-4);
}

test "reflectRay computes reflection vector" {
    const inv_sqrt2: f32 = 1.0 / @sqrt(2.0);
    const V = Vec3.new(inv_sqrt2, 0, inv_sqrt2);
    const N = Vec3.new(0, 0, 1);
    const r = reflectRay(V, N);

    try testing.expectApproxEqAbs(@as(f32, -inv_sqrt2), r.x, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.0), r.y, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, inv_sqrt2), r.z, 1e-5);
}

test "refractDispersionIors separates wavelengths by dispersion strength" {
    const iors = refractDispersionIors(1.5, 0.05);
    // Red has lower IOR, Green is base, Blue has higher IOR
    try testing.expect(iors[0] < iors[1]);
    try testing.expect(iors[1] < iors[2]);
    try testing.expectApproxEqAbs(@as(f32, 1.5), iors[1], 1e-6);

    const zero_disp = refractDispersionIors(1.5, 0.0);
    try testing.expectEqual(@as(f32, 1.5), zero_disp[0]);
    try testing.expectEqual(@as(f32, 1.5), zero_disp[1]);
    try testing.expectEqual(@as(f32, 1.5), zero_disp[2]);
}

test "slabExitPoint projects through thickness along refracted ray" {
    const pos = Vec3.new(1, 2, 3);
    const ray = Vec3.new(0, 0, -1);
    const N = Vec3.new(0, 0, 1);
    const thickness: f32 = 0.5;

    const exit = slabExitPoint(pos, ray, N, thickness);
    try testing.expectApproxEqAbs(@as(f32, 1.0), exit.x, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.0), exit.y, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.5), exit.z, 1e-5); // 3.0 - 0.5 = 2.5
}

test "screenEdgeWeight smoothly fades out near screen borders" {
    // Center is 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), screenEdgeWeight(.{ 0.5, 0.5 }, 0.08), 1e-4);

    // Outside is 0.0
    try testing.expectEqual(@as(f32, 0.0), screenEdgeWeight(.{ -0.05, 0.5 }, 0.08));
    try testing.expectEqual(@as(f32, 0.0), screenEdgeWeight(.{ 1.05, 0.5 }, 0.08));
    try testing.expectEqual(@as(f32, 0.0), screenEdgeWeight(.{ 0.5, -0.01 }, 0.08));
    try testing.expectEqual(@as(f32, 0.0), screenEdgeWeight(.{ 0.5, 1.01 }, 0.08));

    // Mid-vignette is strictly between 0 and 1
    const edge = screenEdgeWeight(.{ 0.04, 0.5 }, 0.08);
    try testing.expect(edge > 0.1 and edge < 0.9);
}

test "dielectricFresnel computes Schlick approximation" {
    // Normal incidence for glass (IOR 1.5): F0 = ((1.5-1)/(1.5+1))^2 = 1/25 = 0.04
    try testing.expectApproxEqAbs(@as(f32, 0.04), dielectricFresnel(1.0, 1.5), 1e-4);

    // Grazing angle: F = 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), dielectricFresnel(0.0, 1.5), 1e-4);
}
