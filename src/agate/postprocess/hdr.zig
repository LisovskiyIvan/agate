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

/// Mirror of the composite shader curves (params3.x modes): exposure scales
/// first, then the EXISTING ACES/Reinhard polynomial. Inputs are sanitized
/// and HDR-bounded AFTER exposure (boundHdr3(c*e), matching the GLSL
/// `color *= exposure; color = boundHdr(color)`), so huge exposure stays
/// finite; no new curve is introduced here.
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

// HDR regression tests live in `hdr_tests.zig` (same directory,
// imported below so the test registry picks them up exactly once).

test {
    _ = @import("hdr_tests.zig");
}
