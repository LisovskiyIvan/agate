const std = @import("std");
const sg = @import("sokol").gfx;
const types = @import("types.zig");

// Linear-HDR math helpers (pure CPU; no postfx service, no GPU calls except
// queryCapabilities). Mirrors the HDR GLSL lanes so headless tests pin the
// contract: finite half-range bound, exact IEC sRGB transfer, and the
// existing ACES/Reinhard curves reused once at final output. STET.

/// Finite half max: the only bound the HDR branch applies (negatives to 0).
pub const half_max: f32 = 65504.0;

/// RGBA16F capability bits for the linear-HDR main target. `msaa` is
/// informational only: the 1x HDR path never requires it.
pub const Capabilities = struct {
    sample: bool = false,
    filter: bool = false,
    render: bool = false,
    blend: bool = false,
    msaa: bool = false,

    pub fn supported(self: Capabilities) bool {
        return self.sample and self.filter and self.render and self.blend;
    }
};

/// Query the ACTUAL backend for RGBA16F. Headless (invalid context or DUMMY
/// backend) returns all false.
pub fn queryCapabilities() Capabilities {
    if (!sg.isvalid()) return .{};
    if (sg.queryBackend() == .DUMMY) return .{};
    const info = sg.queryPixelformat(.RGBA16F);
    return .{
        .sample = info.sample,
        .filter = info.filter,
        .render = info.render,
        .blend = info.blend,
        .msaa = info.msaa,
    };
}

/// Non-finite lane to 0.0; finite values pass through. Every HDR helper
/// sanitizes inputs first, so NaN/Inf can never reach the transfer curve.
pub fn sanitizeFinite(v: f32) f32 {
    if (!std.math.isFinite(v)) return 0.0;
    return v;
}

/// Exposure sanitize: non-finite exposure falls back to the neutral default
/// (1.0); negatives render black (0.0); finite values clamp to 0..65504.
/// Sanitizing BEFORE the multiply removes the 0*Inf path entirely (Inf
/// behaves as 1.0), and huge finite exposure still lands in the
/// post-exposure half bound applied by tonemap below.
pub fn sanitizeExposure(e: f32) f32 {
    if (!std.math.isFinite(e)) return 1.0;
    if (e < 0.0) return 0.0;
    return @min(e, half_max);
}

pub fn boundHdr(v: f32) f32 {
    return std.math.clamp(sanitizeFinite(v), 0.0, half_max);
}

pub fn boundHdr3(c: [3]f32) [3]f32 {
    return .{ boundHdr(c[0]), boundHdr(c[1]), boundHdr(c[2]) };
}

/// Exact IEC 61966-2-1 sRGB encode (mirrors the GLSL `linearToSrgb`).
pub fn linearToSrgb(v: f32) f32 {
    const c = sanitizeFinite(v);
    if (c <= 0.0) return 0.0;
    if (c <= 0.0031308) return 12.92 * c;
    if (c >= 1.0) return 1.0;
    return 1.055 * std.math.pow(f32, c, 1.0 / 2.4) - 0.055;
}

pub fn linearToSrgb3(c: [3]f32) [3]f32 {
    return .{ linearToSrgb(c[0]), linearToSrgb(c[1]), linearToSrgb(c[2]) };
}

fn acesFilm(v: f32) f32 {
    const a = 2.51;
    const b = 0.03;
    const c = 2.43;
    const d = 0.59;
    const e = 0.14;
    return std.math.clamp((v * (a * v + b)) / (v * (c * v + d) + e), 0.0, 1.0);
}

fn uchimuraFilm(x: f32) f32 {
    const P: f32 = 1.0;
    const a: f32 = 1.0;
    const m: f32 = 0.22;
    const l: f32 = 0.4;
    const c: f32 = 1.33;
    const b: f32 = 0.0;
    const l0 = ((P - m) * l) / a;
    const S0 = m + l0;
    const S1 = m + a * l0;
    const C2 = (a * P) / (P - S1);
    const cp = -C2 / P;

    if (x <= 0.0) return 0.0;
    if (x <= m) {
        return m * std.math.pow(f32, x / m, c) + b;
    } else if (x <= m + l0) {
        return m + a * (x - m);
    } else {
        return std.math.clamp(P - (P - S1) * @exp(cp * (x - S0)), 0.0, 1.0);
    }
}

fn pbrNeutral(c: [3]f32) [3]f32 {
    const start_compression: f32 = 0.8 - 0.04;
    const desaturation: f32 = 0.15;
    const x = @min(c[0], @min(c[1], c[2]));
    const offset = if (x < 0.08) x - 6.25 * x * x else 0.04;
    var col = [3]f32{ @max(c[0] - offset, 0.0), @max(c[1] - offset, 0.0), @max(c[2] - offset, 0.0) };
    const peak = @max(col[0], @max(col[1], col[2]));
    if (peak < start_compression) return col;
    const d = 1.0 - start_compression;
    const new_peak = 1.0 - d * d / (peak + d - start_compression);
    col[0] *= new_peak / peak;
    col[1] *= new_peak / peak;
    col[2] *= new_peak / peak;
    const g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
    return .{
        std.math.clamp(col[0] + (new_peak - col[0]) * g, 0.0, 1.0),
        std.math.clamp(col[1] + (new_peak - col[1]) * g, 0.0, 1.0),
        std.math.clamp(col[2] + (new_peak - col[2]) * g, 0.0, 1.0),
    };
}

fn agxFilm(c: [3]f32) [3]f32 {
    if (c[0] <= 0.0 and c[1] <= 0.0 and c[2] <= 0.0) return .{ 0.0, 0.0, 0.0 };

    const m00: f32 = 0.842479062253094;
    const m01: f32 = 0.0423282422610123;
    const m02: f32 = 0.0423756549057051;
    const m10: f32 = 0.0784335996993431;
    const m11: f32 = 0.878468636469772;
    const m12: f32 = 0.0784336099914461;
    const m20: f32 = 0.0792237451477422;
    const m21: f32 = 0.0791661274605434;
    const m22: f32 = 0.879142973798673;

    const inv00: f32 = 1.19687900512017;
    const inv01: f32 = -0.0528968517590771;
    const inv02: f32 = -0.0529716355084725;
    const inv10: f32 = -0.0980208811401368;
    const inv11: f32 = 1.15190312990417;
    const inv12: f32 = -0.0980434501171241;
    const inv20: f32 = -0.0990297440797205;
    const inv21: f32 = -0.098961176813784;
    const inv22: f32 = 1.15107367264185;

    const r = @max(c[0], 1e-6);
    const g = @max(c[1], 1e-6);
    const b = @max(c[2], 1e-6);

    var x = m00 * r + m01 * g + m02 * b;
    var y = m10 * r + m11 * g + m12 * b;
    var z = m20 * r + m21 * g + m22 * b;

    const min_ev: f32 = -10.0;
    const max_ev: f32 = 6.5;
    const ev_range: f32 = max_ev - min_ev;

    x = std.math.clamp((@log2(@max(x, 1e-6)) - min_ev) / ev_range, 0.0, 1.0);
    y = std.math.clamp((@log2(@max(y, 1e-6)) - min_ev) / ev_range, 0.0, 1.0);
    z = std.math.clamp((@log2(@max(z, 1e-6)) - min_ev) / ev_range, 0.0, 1.0);

    const curve = struct {
        fn f(v: f32) f32 {
            const v2 = v * v;
            const v4 = v2 * v2;
            return 15.5 * v4 * v - 40.14 * v4 + 31.96 * v2 * v - 6.868 * v2 + 0.4298 * v + 0.1191;
        }
    }.f;

    const c0: f32 = 0.1191;
    const inv_scale: f32 = 1.0 / (1.0 - c0);
    const cx = std.math.clamp((curve(x) - c0) * inv_scale, 0.0, 1.0);
    const cy = std.math.clamp((curve(y) - c0) * inv_scale, 0.0, 1.0);
    const cz = std.math.clamp((curve(z) - c0) * inv_scale, 0.0, 1.0);

    const out_r = inv00 * cx + inv01 * cy + inv02 * cz;
    const out_g = inv10 * cx + inv11 * cy + inv12 * cz;
    const out_b = inv20 * cx + inv21 * cy + inv22 * cz;

    return .{
        std.math.clamp(out_r, 0.0, 1.0),
        std.math.clamp(out_g, 0.0, 1.0),
        std.math.clamp(out_b, 0.0, 1.0),
    };
}

/// Mirror of the composite shader curves (params3.x modes): exposure scales
/// first, then the selected tonemap polynomial. Inputs are sanitized
/// and HDR-bounded AFTER exposure (boundHdr3(c*e), matching the GLSL
/// `color *= exposure; color = boundHdr(color)`), so huge exposure stays
/// finite.
pub fn tonemap(c: [3]f32, exposure: f32, mode: types.TonemappingType) [3]f32 {
    const e = sanitizeExposure(exposure);
    const hdr = boundHdr3(.{ c[0] * e, c[1] * e, c[2] * e });
    switch (mode) {
        .reinhard => return .{
            hdr[0] / (hdr[0] + 1.0),
            hdr[1] / (hdr[1] + 1.0),
            hdr[2] / (hdr[2] + 1.0),
        },
        .aces => return .{ acesFilm(hdr[0]), acesFilm(hdr[1]), acesFilm(hdr[2]) },
        .filmic => return .{ uchimuraFilm(hdr[0]), uchimuraFilm(hdr[1]), uchimuraFilm(hdr[2]) },
        .agx => return agxFilm(hdr),
        .neutral => return pbrNeutral(hdr),
        .none => return .{ @min(hdr[0], 1.0), @min(hdr[1], 1.0), @min(hdr[2], 1.0) },
    }
}

/// Final display color: exactly one output encode. sRGB targets take the
/// linear tonemapped value (hardware encodes); UNORM targets take the manual
/// piecewise encode. Exactly one of the two paths encodes, never both.
pub fn displayColor(c: [3]f32, exposure: f32, mode: types.TonemappingType, srgb_target: bool) [3]f32 {
    const t = tonemap(c, exposure, mode);
    if (srgb_target) return t;
    return linearToSrgb3(t);
}

// Auto-exposure re-exports
pub const auto_exposure = @import("auto_exposure.zig");
pub const AutoExposureOptions = auto_exposure.AutoExposureOptions;
pub const AutoExposureState = auto_exposure.AutoExposureState;
pub const LuminanceHistogram = auto_exposure.LuminanceHistogram;
pub const calcLuminance = auto_exposure.calcLuminance;
pub const calcGeometricMeanLuminance = auto_exposure.calcGeometricMeanLuminance;
pub const calcTargetExposure = auto_exposure.calcTargetExposure;
pub const adaptExposure = auto_exposure.adaptExposure;

// HDR regression tests live in `hdr_tests.zig` (same directory,
// imported below so the test registry picks them up exactly once).
