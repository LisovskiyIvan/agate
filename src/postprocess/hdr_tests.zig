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

test "local tonemap: uniform fields stay neutral under any exposure and any bloom add" {
    const contrast: f32 = 0.3;
    const intensity: f32 = 0.5;
    for ([_]f32{ 0.05, 0.5, 4.0 }) |level| {
        const flat = [3]f32{ level, level, level };
        const taps = [4][3]f32{ flat, flat, flat, flat };
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), prod.localTonemapGain(flat, taps, contrast), 1e-6);
        for ([_]f32{ 0.25, 1.0, 4.0 }) |exposure| {
            // Include zero bloom (below threshold) and a uniform bloom add.
            for ([_]f32{ 0.0, 0.35 }) |bloom_add| {
                const v = (level + bloom_add) * exposure;
                const composed = [3]f32{ v, v, v };
                const out = prod.applyLocalTonemap(composed, flat, taps, intensity, contrast);
                const tol = 1e-5 * @max(1.0, @abs(v));
                try std.testing.expectApproxEqAbs(composed[0], out[0], tol);
                try std.testing.expectApproxEqAbs(composed[1], out[1], tol);
                try std.testing.expectApproxEqAbs(composed[2], out[2], tol);
            }
        }
    }
}

test "local tonemap: real detail shifts gain inside the [0.25, 4.0] bounds" {
    const contrast: f32 = 0.3;
    const hi = [3]f32{ 2.0, 2.0, 2.0 };
    const lo = [3]f32{ 0.5, 0.5, 0.5 };
    const hi_taps = [4][3]f32{ lo, lo, lo, lo };
    const g_hi = prod.localTonemapGain(hi, hi_taps, contrast);
    try std.testing.expect(g_hi > 1.0 and g_hi <= 4.0);
    const out_hi = prod.applyLocalTonemap(hi, hi, hi_taps, 1.0, contrast);
    try std.testing.expect(out_hi[0] > hi[0]);

    const dk = [3]f32{ 0.1, 0.1, 0.1 };
    const br_taps = [4][3]f32{ hi, hi, hi, hi };
    const g_lo = prod.localTonemapGain(dk, br_taps, contrast);
    try std.testing.expect(g_lo < 1.0 and g_lo >= 0.25);
    const out_lo = prod.applyLocalTonemap(dk, dk, br_taps, 1.0, contrast);
    try std.testing.expect(out_lo[0] < dk[0]);

    // A steep contrast engages the rails; the packed default stays inside.
    const hot = [3]f32{ 100.0, 100.0, 100.0 };
    const black = [3]f32{ 0.0, 0.0, 0.0 };
    const black_taps = [4][3]f32{ black, black, black, black };
    const hot_taps = [4][3]f32{ hot, hot, hot, hot };
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), prod.localTonemapGain(hot, black_taps, 3.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), prod.localTonemapGain(black, hot_taps, 3.0), 1e-6);
    const g_hot = prod.localTonemapGain(hot, black_taps, contrast);
    try std.testing.expect(g_hot > 1.0 and g_hot <= 4.0);
    const g_black = prod.localTonemapGain(black, hot_taps, contrast);
    try std.testing.expect(g_black < 1.0 and g_black >= 0.25);

    const nan_c = [3]f32{ std.math.nan(f32), 0.5, 0.5 };
    try std.testing.expect(std.math.isFinite(prod.localTonemapGain(nan_c, hi_taps, contrast)));
    try std.testing.expectEqual(@as(f32, 0.0), prod.localLuma(.{ std.math.nan(f32), std.math.inf(f32), -1.0 }));
    try std.testing.expectApproxEqAbs(prod.localLuma(.{ half_max, half_max, half_max }), prod.localLuma(.{ 1e10, 1e10, 1e10 }), 1e-2);
}

test "local tonemap: effectively-off intensity is the exact identity" {
    const contrast: f32 = 0.3;
    const center = [3]f32{ 2.0, 2.0, 2.0 };
    const lo = [3]f32{ 0.5, 0.5, 0.5 };
    const taps = [4][3]f32{ lo, lo, lo, lo };
    const composed = [3]f32{ 1.7, 0.9, 0.4 };
    try std.testing.expect(prod.localTonemapGain(center, taps, contrast) > 1.0);
    try std.testing.expectEqual(composed, prod.applyLocalTonemap(composed, center, taps, 0.0, contrast));
    try std.testing.expectEqual(composed, prod.applyLocalTonemap(composed, center, taps, 0.001, contrast));
}

test "local tonemap source contract: no bloom dependency, matched pre-exposure domains" {
    const src = @embedFile("../shaders/postprocess.glsl");
    const fn_marker = "vec3 applyLocalTonemapping(vec3 color, vec2 uv) {";
    const start = std.mem.indexOf(u8, src, fn_marker) orelse return error.LocalTonemapFnMissing;
    const rest = src[start..];
    const end_rel = std.mem.indexOf(u8, rest, "\n}\n") orelse return error.LocalTonemapFnUnterminated;
    const body = rest[0..end_rel];
    try std.testing.expect(std.mem.indexOf(u8, body, "bloom") == null);
    try std.testing.expect(std.mem.indexOf(u8, body, "params3") == null);
    try std.testing.expectEqual(@as(usize, 5), std.mem.count(u8, body, "dot(boundRadiance(texture("));
    try std.testing.expect(std.mem.indexOf(u8, body, "dot(color") == null);
    try std.testing.expect(std.mem.indexOf(u8, body, "0.18 + l0") != null);
}
