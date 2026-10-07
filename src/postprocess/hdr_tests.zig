//! Tests for `postprocess/hdr.zig` (moved verbatim from inline blocks; production code unchanged).
const std = @import("std");
const sg = @import("sokol").gfx;
const types = @import("types.zig");
const prod = @import("hdr.zig");
const half_max = prod.half_max;
const Capabilities = prod.Capabilities;
const queryCapabilities = prod.queryCapabilities;
const boundHdr = prod.boundHdr;
const sanitizeExposure = prod.sanitizeExposure;
const linearToSrgb = prod.linearToSrgb;
const linearToSrgb3 = prod.linearToSrgb3;
const tonemap = prod.tonemap;
const displayColor = prod.displayColor;

test "capabilities default to all false and headless queries stay off" {
    const def: Capabilities = .{};
    try std.testing.expect(!def.supported());
    try std.testing.expect((Capabilities{ .sample = true, .filter = true, .render = true, .blend = true }).supported());
    // msaa is informational: 1x HDR does not require it.
    try std.testing.expect((Capabilities{ .sample = true, .filter = true, .render = true, .blend = true, .msaa = false }).supported());
    // Headless unit runs have no valid context: every support bit is false.
    if (!sg.isvalid()) {
        try std.testing.expect(!queryCapabilities().supported());
    }
}

test "linearToSrgb hits exact IEC goldens" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), linearToSrgb(0.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), linearToSrgb(1.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 12.92 * 0.002), linearToSrgb(0.002), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7354), linearToSrgb(0.5), 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), linearToSrgb(-3.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), linearToSrgb(std.math.nan(f32)), 1e-6);
}

test "radiance above 1 survives until the tonemap and stays distinct" {
    const m = types.TonemappingType.aces;
    const t1 = tonemap(.{ 1.0, 1.0, 1.0 }, 1.0, m)[0];
    const t4 = tonemap(.{ 4.0, 4.0, 4.0 }, 1.0, m)[0];
    const t16 = tonemap(.{ 16.0, 16.0, 16.0 }, 1.0, m)[0];
    try std.testing.expect(t4 > t1 + 0.05);
    try std.testing.expect(t16 > t4);
    // Exposure is manual and predictable: x2 exposure brightens.
    const e2 = tonemap(.{ 1.0, 1.0, 1.0 }, 2.0, m)[0];
    try std.testing.expect(e2 > t1);
    // Zero exposure renders black on every mode.
    for ([_]types.TonemappingType{ .none, .aces, .reinhard, .filmic, .agx, .neutral }) |mode| {
        const black = tonemap(.{ 8.0, 4.0, 2.0 }, 0.0, mode);
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), black[0], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), black[1], 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), black[2], 1e-6);
    }
}

test "finite bound, half max, and exposure sanitize as documented" {
    try std.testing.expectApproxEqAbs(half_max, boundHdr(1e10), 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), boundHdr(-1.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), boundHdr(std.math.inf(f32)), 1e-6);
    // Invalid exposure falls back to the neutral default 1.0, negatives to 0.
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sanitizeExposure(std.math.nan(f32)), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sanitizeExposure(std.math.inf(f32)), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sanitizeExposure(-1.0), 1e-6);
    try std.testing.expectApproxEqAbs(half_max, sanitizeExposure(1e10), 1e-2);
    // NaN color sanitizes to 0 while NaN exposure defaults to 1 (finite rest).
    const nan_t = tonemap(.{ std.math.nan(f32), 4.0, 16.0 }, std.math.nan(f32), .aces);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), nan_t[0], 1e-6);
    try std.testing.expect(std.math.isFinite(nan_t[1]) and nan_t[1] > 0.0);
    try std.testing.expect(std.math.isFinite(nan_t[2]) and nan_t[2] > nan_t[1]);
    // 0 * Inf edge: Inf exposure clamps to half max, 0 * 65504 stays 0.
    const zero_inf = tonemap(.{ 0.0, 4.0, 16.0 }, std.math.inf(f32), .aces);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), zero_inf[0], 1e-6);
    for (zero_inf[1..]) |v| try std.testing.expect(std.math.isFinite(v));
    // Post-exposure bound: huge/Inf exposure stays finite on every mode.
    for ([_]types.TonemappingType{ .none, .aces, .reinhard, .filmic, .agx, .neutral }) |mode| {
        const huge = tonemap(.{ 4.0, 2.0, 1.0 }, 1e10, mode);
        for (huge) |v| try std.testing.expect(std.math.isFinite(v) and v >= 0.0 and v <= 1.0);
        const inf_e = tonemap(.{ 4.0, 2.0, 1.0 }, std.math.inf(f32), mode);
        for (inf_e) |v| try std.testing.expect(std.math.isFinite(v) and v >= 0.0 and v <= 1.0);
    }
}

test "exactly one output encode: linear for sRGB targets, encoded for UNORM" {
    const c = [3]f32{ 4.0, 2.0, 1.0 };
    const lin = displayColor(c, 1.0, .aces, true);
    const enc = displayColor(c, 1.0, .aces, false);
    const t = tonemap(c, 1.0, .aces);
    try std.testing.expectEqual(t, lin);
    try std.testing.expectEqual(linearToSrgb3(t), enc);
}

test "filmic, agx, and neutral curves preserve monotonic brightness and finite bounds" {
    const modes = [_]types.TonemappingType{ .filmic, .agx, .neutral };
    for (modes) |m| {
        const dark = tonemap(.{ 0.1, 0.1, 0.1 }, 1.0, m);
        const mid = tonemap(.{ 0.5, 0.5, 0.5 }, 1.0, m);
        const bright = tonemap(.{ 1.0, 1.0, 1.0 }, 1.0, m);
        const super = tonemap(.{ 5.0, 5.0, 5.0 }, 1.0, m);

        try std.testing.expect(dark[0] < mid[0]);
        try std.testing.expect(mid[0] < bright[0]);
        try std.testing.expect(bright[0] <= super[0]);
        try std.testing.expect(super[0] <= 1.0);
    }
}
