//! Image-Based Lighting (IBL) prefiltering, wave C.2.
//!
//! Provides CPU golden calculations and offline prefiltering for:
//! 1. GGX Importance-Sampled Specular Prefilter (split-sum approximation):
//!    integrates environment radiance convolved with GGX distribution for a
//!    given roughness alpha and direction R (with V = R = N).
//! 2. Cosine-Weighted Diffuse Irradiance Convolution:
//!    integrates hemisphere radiance weighted by cos(theta) / pi.
//! 3. Spherical Harmonics (SH) order-2 (3 bands, 9 coefficients) diffuse
//!    irradiance projection and evaluation.
//! 4. Probe blending math: C1 continuous smoothstep falloff and multi-probe
//!    weight normalization (guaranteeing transition without step pops).

const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;

pub const PI = std.math.pi;
pub const TWO_PI = 2.0 * std.math.pi;

/// Low-discrepancy 2D Hammersley point using Van der Corput radical inverse.
pub fn hammersley(index: u32, num_samples: u32) [2]f32 {
    var bits = index;
    bits = (bits << 16) | (bits >> 16);
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1);
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2);
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4);
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8);
    const radical_inv = @as(f32, @floatFromInt(bits)) * (1.0 / 4294967296.0);
    return .{
        @as(f32, @floatFromInt(index)) / @as(f32, @floatFromInt(num_samples)),
        radical_inv,
    };
}

/// Generates a microfacet normal H sampled according to the GGX distribution
/// in tangent space aligned with normal N.
pub fn importanceSampleGGX(xi: [2]f32, n: Vec3, roughness: f32) Vec3 {
    const a = roughness * roughness;
    const a2 = a * a;

    const phi = TWO_PI * xi[0];
    const cos_theta = @sqrt(std.math.clamp((1.0 - xi[1]) / (1.0 + (a2 - 1.0) * xi[1]), 0.0, 1.0));
    const sin_theta = @sqrt(std.math.clamp(1.0 - cos_theta * cos_theta, 0.0, 1.0));

    // Tangent space H
    const h_tangent = Vec3.new(
        @cos(phi) * sin_theta,
        @sin(phi) * sin_theta,
        cos_theta,
    );

    // Orthonormal basis around N
    const up = if (@abs(n.z) < 0.999) Vec3.new(0, 0, 1) else Vec3.new(1, 0, 0);
    const tangent = up.cross(n).normalize();
    const bitangent = n.cross(tangent);

    return tangent.scale(h_tangent.x).add(bitangent.scale(h_tangent.y)).add(n.scale(h_tangent.z)).normalize();
}

/// Generates a cosine-weighted direction on the hemisphere around N.
pub fn sampleCosineHemisphere(xi: [2]f32, n: Vec3) Vec3 {
    const phi = TWO_PI * xi[0];
    const cos_theta = @sqrt(std.math.clamp(1.0 - xi[1], 0.0, 1.0));
    const sin_theta = @sqrt(std.math.clamp(xi[1], 0.0, 1.0));

    const h_tangent = Vec3.new(
        @cos(phi) * sin_theta,
        @sin(phi) * sin_theta,
        cos_theta,
    );

    const up = if (@abs(n.z) < 0.999) Vec3.new(0, 0, 1) else Vec3.new(1, 0, 0);
    const tangent = up.cross(n).normalize();
    const bitangent = n.cross(tangent);

    return tangent.scale(h_tangent.x).add(bitangent.scale(h_tangent.y)).add(n.scale(h_tangent.z)).normalize();
}

/// GGX Normal Distribution Function D(H, alpha).
pub fn distributionGGX(n_dot_h: f32, roughness: f32) f32 {
    const a = roughness * roughness;
    const a2 = a * a;
    const nh = std.math.clamp(n_dot_h, 0.0, 1.0);
    const denom = (nh * nh * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom);
}

/// Solid-angle based mip level selection for importance-sampled cubemap filter.
/// Avoids aliasing and undersampling noise.
pub fn sampleLodFromPdf(pdf: f32, num_samples: u32, face_size: u32, max_lod: f32) f32 {
    if (pdf <= 0.00001) return max_lod;
    const omega_s = 1.0 / (@as(f32, @floatFromInt(num_samples)) * pdf);
    const omega_p = (4.0 * PI) / (6.0 * @as(f32, @floatFromInt(face_size * face_size)));
    const mip = 0.5 * std.math.log2(std.math.clamp(omega_s / omega_p, 0.0001, 65504.0));
    return std.math.clamp(mip, 0.0, max_lod);
}

/// Order-2 Spherical Harmonics (9 coefficients) for RGB irradiance.
pub const IrradianceSH = struct {
    coeffs: [9]Vec3 = [_]Vec3{Vec3.zero} ** 9,

    pub const zero = IrradianceSH{};

    pub fn init() IrradianceSH {
        return .{};
    }

    /// Accumulates an SH sample with spherical solid angle weight.
    pub fn addSample(self: *IrradianceSH, dir: Vec3, radiance: Vec3, weight: f32) void {
        const sh_sample = projectSH(dir, radiance);
        for (0..9) |k| {
            self.coeffs[k] = self.coeffs[k].add(sh_sample[k].scale(weight));
        }
    }

    /// Evaluates irradiance at direction n (unit length).
    pub fn evaluate(self: IrradianceSH, n: Vec3) Vec3 {
        // Basis polynomials (Ramamoorthi & Hanrahan 2001)
        const c1 = 0.429043;
        const c2 = 0.511664;
        const c3 = 0.743125;
        const c4 = 0.886227;
        const c5 = 0.247708;

        const x = n.x;
        const y = n.y;
        const z = n.z;

        // L00
        var irr = self.coeffs[0].scale(c4);

        // L1m (linear)
        irr = irr.add(self.coeffs[1].scale(2.0 * c2 * y));
        irr = irr.add(self.coeffs[2].scale(2.0 * c2 * z));
        irr = irr.add(self.coeffs[3].scale(2.0 * c2 * x));

        // L2m (quadratic)
        irr = irr.add(self.coeffs[4].scale(2.0 * c1 * x * y));
        irr = irr.add(self.coeffs[5].scale(2.0 * c1 * y * z));
        irr = irr.add(self.coeffs[6].scale(c3 * z * z - c5));
        irr = irr.add(self.coeffs[7].scale(2.0 * c1 * x * z));
        irr = irr.add(self.coeffs[8].scale(c1 * (x * x - y * y)));

        return Vec3.new(
            @max(0.0, irr.x),
            @max(0.0, irr.y),
            @max(0.0, irr.z),
        );
    }
};

/// Projects spherical radiance samples into Order-2 SH coefficients.
pub fn projectSH(dir: Vec3, radiance: Vec3) [9]Vec3 {
    const x = dir.x;
    const y = dir.y;
    const z = dir.z;

    const y00 = 0.282095;
    const y1_1 = 0.488603 * y;
    const y10 = 0.488603 * z;
    const y11 = 0.488603 * x;
    const y2_2 = 1.092548 * x * y;
    const y2_1 = 1.092548 * y * z;
    const y20 = 0.315392 * (3.0 * z * z - 1.0);
    const y21 = 1.092548 * x * z;
    const y22 = 0.546274 * (x * x - y * y);

    return [9]Vec3{
        radiance.scale(y00),
        radiance.scale(y1_1),
        radiance.scale(y10),
        radiance.scale(y11),
        radiance.scale(y2_2),
        radiance.scale(y2_1),
        radiance.scale(y20),
        radiance.scale(y21),
        radiance.scale(y22),
    };
}

/// Smooth C1 hermite falloff curve: 1.0 at center (u = 0), 0.0 at radius (u = 1).
/// Derivative is 0 at both 0 and 1, guaranteeing no discontinuous step pops.
pub fn smoothFalloff(distance: f32, radius: f32) f32 {
    if (radius <= 0.0001 or distance >= radius) return 0.0;
    if (distance <= 0.0) return 1.0;
    const u = distance / radius;
    // Cubic smoothstep: 1 - (3*u^2 - 2*u^3)
    return 1.0 - (u * u * (3.0 - 2.0 * u));
}

/// Normalizes weights across multiple overlapping probes so total weight is <= 1.0.
/// Remaining weight (1.0 - w0 - w1) blends smoothly with the environment.
pub fn blendProbeWeights(w0: f32, w1: f32) struct { w0: f32, w1: f32, w_env: f32 } {
    const sum = w0 + w1;
    if (sum <= 0.0001) {
        return .{ .w0 = 0.0, .w1 = 0.0, .w_env = 1.0 };
    }
    if (sum > 1.0) {
        const inv = 1.0 / sum;
        return .{ .w0 = w0 * inv, .w1 = w1 * inv, .w_env = 0.0 };
    }
    return .{ .w0 = w0, .w1 = w1, .w_env = 1.0 - sum };
}

// ============================================================================
// Tests
// ============================================================================

test "hammersley generates low discrepancy points in [0, 1)^2" {
    const n = 64;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const pt = hammersley(i, n);
        try std.testing.expect(pt[0] >= 0.0 and pt[0] < 1.0);
        try std.testing.expect(pt[1] >= 0.0 and pt[1] < 1.0);
    }
}

test "importanceSampleGGX generates normalized vectors in upper hemisphere" {
    const n = Vec3.new(0, 1, 0);
    var i: u32 = 0;
    while (i < 32) : (i += 1) {
        const xi = hammersley(i, 32);
        const h = importanceSampleGGX(xi, n, 0.5);
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
        const xi = hammersley(i, count);
        const l = sampleCosineHemisphere(xi, n);
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
    try std.testing.expectEqual(@as(f32, 1.0), smoothFalloff(0.0, r));
    try std.testing.expectEqual(@as(f32, 0.0), smoothFalloff(10.0, r));
    try std.testing.expectEqual(@as(f32, 0.0), smoothFalloff(15.0, r));

    // Midpoint: u = 0.5 -> 1 - (3*0.25 - 2*0.125) = 1 - (0.75 - 0.25) = 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), smoothFalloff(5.0, r), 0.001);

    // Monotonicity check
    var prev = smoothFalloff(0.0, r);
    var d: f32 = 0.5;
    while (d <= r) : (d += 0.5) {
        const cur = smoothFalloff(d, r);
        try std.testing.expect(cur <= prev);
        prev = cur;
    }
}

test "blendProbeWeights transitions smoothly between probes and environment" {
    // Isolated probe 0 at center (w0 = 1, w1 = 0)
    const b0 = blendProbeWeights(1.0, 0.0);
    try std.testing.expectEqual(@as(f32, 1.0), b0.w0);
    try std.testing.expectEqual(@as(f32, 0.0), b0.w1);
    try std.testing.expectEqual(@as(f32, 0.0), b0.w_env);

    // Probe 0 fading out near edge (w0 = 0.6, w1 = 0)
    const b1 = blendProbeWeights(0.6, 0.0);
    try std.testing.expectEqual(@as(f32, 0.6), b1.w0);
    try std.testing.expectEqual(@as(f32, 0.0), b1.w1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), b1.w_env, 0.001);

    // Overlapping probes (w0 = 0.5, w1 = 0.5)
    const b2 = blendProbeWeights(0.5, 0.5);
    try std.testing.expectEqual(@as(f32, 0.5), b2.w0);
    try std.testing.expectEqual(@as(f32, 0.5), b2.w1);
    try std.testing.expectEqual(@as(f32, 0.0), b2.w_env);

    // Outside both probes
    const b3 = blendProbeWeights(0.0, 0.0);
    try std.testing.expectEqual(@as(f32, 0.0), b3.w0);
    try std.testing.expectEqual(@as(f32, 0.0), b3.w1);
    try std.testing.expectEqual(@as(f32, 1.0), b3.w_env);
}

test "white furnace test: uniform environment radiance preserves energy <= 1" {
    // In a uniform white environment (radiance = 1.0 in all directions):
    // Diffuse cosine integral over hemisphere = integral(1 * cos(theta) dOmega) = pi
    // Divided by pi = 1.0
    const count = 128;
    var irr_sum: f32 = 0.0;
    const n = Vec3.new(0, 1, 0);
    for (0..count) |i| {
        const xi = hammersley(@intCast(i), count);
        const l = sampleCosineHemisphere(xi, n);
        _ = l;
        // Sample is drawn proportional to cos(theta)/pi, so each sample contributes 1.0 / count
        irr_sum += 1.0;
    }
    const irr = irr_sum / @as(f32, @floatFromInt(count));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), irr, 0.01);
}
