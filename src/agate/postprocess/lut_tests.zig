//! Tests for `postprocess/lut.zig` (moved verbatim from inline blocks; production code unchanged).
const std = @import("std");
const texture_mod = @import("../texture.zig");
const options = @import("options.zig");
const prod = @import("lut.zig");
const LutFormat = prod.LutFormat;
const LutStripSample = prod.LutStripSample;
const validLutSize = prod.validLutSize;
const lutStripLayout = prod.lutStripLayout;
const lutStripUv = prod.lutStripUv;
const lutSampleUv = prod.lutSampleUv;
const applyLutStrip = prod.applyLutStrip;
const writeIdentityLutStrip = prod.writeIdentityLutStrip;
const buildIdentityLutStrip = prod.buildIdentityLutStrip;
const lutParams = prod.lutParams;

test "lut strip layout validation" {
    // The two engine-validated strips decode cleanly.
    const lut32 = try lutStripLayout(1024, 32);
    try std.testing.expectEqual(@as(u32, 32), lut32.size);
    try std.testing.expectEqual(@as(u32, 1024), lut32.width);
    const lut64 = try lutStripLayout(4096, 64);
    try std.testing.expectEqual(@as(u32, 64), lut64.size);
    const lut16 = try lutStripLayout(256, 16);
    try std.testing.expectEqual(@as(u32, 16), lut16.size);

    // Broken aspects cannot address the cube: swapped dims, non-square
    // slice, degenerate sizes, or an oversized edge.
    try std.testing.expectError(error.InvalidLutStrip, lutStripLayout(1024, 64));
    try std.testing.expectError(error.InvalidLutStrip, lutStripLayout(512, 32));
    try std.testing.expectError(error.InvalidLutStrip, lutStripLayout(33, 33));
    try std.testing.expectError(error.InvalidLutStrip, lutStripLayout(32, 32));
    try std.testing.expectError(error.InvalidLutStrip, lutStripLayout(0, 0));
    try std.testing.expectError(error.InvalidLutStrip, lutStripLayout(1, 1));
    try std.testing.expectError(error.InvalidLutStrip, lutStripLayout(16384, 128));
    try std.testing.expectError(error.InvalidLutStrip, lutStripLayout(4, 0));

    try std.testing.expect(validLutSize(2));
    try std.testing.expect(validLutSize(64));
    try std.testing.expect(!validLutSize(1));
    try std.testing.expect(!validLutSize(0));
    try std.testing.expect(!validLutSize(128));
}

test "lut strip uv golden values" {
    // Black at N=32: layer 0, half-texel inset in both slice axes. v is
    // the slice-space row coordinate (already normalized); only u gains
    // the layer offset and the extra 1/N strip scaling.
    const s000 = lutStripUv(0.0, 0.0, 0.0, 32);
    try std.testing.expectApproxEqAbs(@as(f32, 0.015625 / 32.0), s000.uv0[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.015625), s000.uv0[1], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 1.015625 / 32.0), s000.uv1[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), s000.blend, 1e-7);

    // White at N=32: last layer (31), no second layer needed.
    const s111 = lutStripUv(1.0, 1.0, 1.0, 32);
    try std.testing.expectApproxEqAbs(@as(f32, 31.984375 / 32.0), s111.uv0[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 31.5 / 32.0), s111.uv0[1], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), s111.blend, 1e-7);

    // Mid-blue at N=32 sits halfway between layers 15 and 16.
    const smid = lutStripUv(0.0, 0.0, 0.5, 32);
    try std.testing.expectApproxEqAbs(@as(f32, 15.015625 / 32.0), smid.uv0[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 16.015625 / 32.0), smid.uv1[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), smid.blend, 1e-7);
    // Green only moves v, never the layer.
    try std.testing.expectApproxEqAbs(smid.uv0[0], lutStripUv(0.0, 1.0, 0.5, 32).uv0[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 31.5 / 32.0), lutStripUv(0.0, 1.0, 0.5, 32).uv0[1], 1e-7);

    // N=64 spot check at mid-gray.
    const s64 = lutStripUv(0.5, 0.5, 0.5, 64);
    try std.testing.expectApproxEqAbs(@as(f32, 31.5 / 64.0), s64.uv0[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 32.5 / 64.0), s64.uv1[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), s64.uv0[1], 1e-7);

    // Out-of-range blue clamps to the layer range instead of wrapping.
    const sclamp = lutStripUv(0.0, 0.0, 2.0, 32);
    try std.testing.expectApproxEqAbs(@as(f32, 31.015625 / 32.0), sclamp.uv0[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sclamp.blend, 1e-7);
}

test "lut intensity mix" {
    const sample = LutStripSample{
        .uv0 = .{ 0.0, 0.0 },
        .uv1 = .{ 0.0, 0.0 },
        .blend = 0.5,
    };
    const base = [3]f32{ 0.2, 0.4, 0.6 };
    const l0 = [3]f32{ 0.0, 0.0, 0.0 };
    const l1 = [3]f32{ 1.0, 1.0, 1.0 };

    // Intensity 0 keeps the curve-graded color untouched.
    const off = applyLutStrip(base, l0, l1, sample, 0.0);
    try std.testing.expectApproxEqAbs(base[0], off[0], 1e-6);
    try std.testing.expectApproxEqAbs(base[1], off[1], 1e-6);
    try std.testing.expectApproxEqAbs(base[2], off[2], 1e-6);

    // Full trilinear between the two sampled layers.
    const full = applyLutStrip(base, l0, l1, sample, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), full[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), full[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), full[2], 1e-6);

    // Quarter blend sits a quarter of the way to the graded color.
    const quarter = applyLutStrip(base, l0, l1, sample, 0.25);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2 + (0.5 - 0.2) * 0.25), quarter[0], 1e-6);

    // Out-of-range intensity clamps, never overshoots.
    const over = applyLutStrip(base, l0, l1, sample, 4.0);
    try std.testing.expectApproxEqAbs(full[0], over[0], 1e-6);
    const under = applyLutStrip(base, l0, l1, sample, -1.0);
    try std.testing.expectApproxEqAbs(base[0], under[0], 1e-6);
}

test "lut params packing and default path" {
    // Defaults: no LUT, disabled — the pre-LUT composite path.
    const def = options.PostProcessOptions{};
    try std.testing.expect(def.lut_texture == null);
    try std.testing.expect(!def.lut_enabled);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, lutParams(def));
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, lutParams(def.clamped()));

    // A live binding enables the uniform triplet.
    var live = options.PostProcessOptions{
        .lut_texture = .{
            .image = .{ .id = 1 },
            .view = .{ .id = 7 },
            .sampler = .{ .id = 3 },
            .width = 1024,
            .height = 32,
        },
        .lut_enabled = true,
        .lut_size = 32,
        .lut_strength = 1.0,
    };
    try std.testing.expectEqual([4]f32{ 1.0, 1.0, 32.0, 0.0 }, lutParams(live));

    // Strength packs clamped.
    live.lut_strength = 3.0;
    try std.testing.expectEqual(@as(f32, 1.0), live.clamped().lut_strength);
    live.lut_strength = -0.5;
    try std.testing.expectEqual(@as(f32, 0.0), live.clamped().lut_strength);

    // clamped() drops bindings that can never sample: dead view, or strip
    // dims that do not match the cube edge.
    var dead = live;
    dead.lut_texture.?.view.id = 0;
    try std.testing.expect(dead.clamped().lut_texture == null);
    var dim_mismatch = live;
    dim_mismatch.lut_texture.?.width = 100;
    try std.testing.expect(dim_mismatch.clamped().lut_texture == null);
    // Any N inside [2, 64] with matching strip dims stays (33 is legal,
    // the strip math is generic); only out-of-range sizes are dropped.
    var keep = live;
    keep.lut_size = 33;
    keep.lut_texture.?.width = 33 * 33;
    keep.lut_texture.?.height = 33;
    try std.testing.expect(keep.clamped().lut_texture != null);
    var bogus = live;
    bogus.lut_size = 100;
    try std.testing.expect(bogus.clamped().lut_texture == null);

    // Present but disabled still packs zeros.
    var idle = live;
    idle.lut_enabled = false;
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, lutParams(idle));
}

test "color grading LUT setter, clear, and defaults" {
    const T = texture_mod.Texture;
    var cfg = options.PostProcessOptions{};
    // Defaults: OFF, full strength, 16-edge strip format, no texture —
    // the pre-LUT composite path.
    try std.testing.expect(!cfg.lut_enabled);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cfg.lut_strength, 1e-6);
    try std.testing.expectEqual(@as(u8, 16), cfg.lut_size);
    try std.testing.expectEqual(LutFormat.strip_2d, cfg.lut_format);
    try std.testing.expect(cfg.lut_texture == null);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, lutParams(cfg));

    // Null and dead bindings clear instead of binding garbage (no GPU
    // touched here — the setter only copies handles).
    cfg.setColorGradingLut(null, 16);
    try std.testing.expect(!cfg.lut_enabled);
    const dead = T{
        .image = .{ .id = 1 },
        .view = .{},
        .sampler = .{ .id = 3 },
        .width = 256,
        .height = 16,
    };
    cfg.setColorGradingLut(dead, 16);
    try std.testing.expect(cfg.lut_texture == null);
    try std.testing.expect(!cfg.lut_enabled);

    // Wrong strip dims for the size are rejected too.
    const wrong = T{
        .image = .{ .id = 1 },
        .view = .{ .id = 9 },
        .sampler = .{ .id = 3 },
        .width = 100,
        .height = 16,
    };
    cfg.setColorGradingLut(wrong, 16);
    try std.testing.expect(cfg.lut_texture == null);
    try std.testing.expect(!cfg.lut_enabled);

    // Live 16-strip binds and enables; strength 0 still packs enabled
    // (the shader-side mix is the no-op) while disabled packs zeros.
    var tex = T{
        .image = .{ .id = 1 },
        .view = .{ .id = 7 },
        .sampler = .{ .id = 3 },
        .width = 256,
        .height = 16,
    };
    cfg.setColorGradingLut(tex, 16);
    try std.testing.expect(cfg.lut_enabled);
    try std.testing.expectEqual(@as(u8, 16), cfg.lut_size);
    try std.testing.expectEqual([4]f32{ 1.0, 1.0, 16.0, 0.0 }, lutParams(cfg));
    cfg.lut_strength = 0.0;
    try std.testing.expectEqual([4]f32{ 1.0, 0.0, 16.0, 0.0 }, lutParams(cfg));
    cfg.clearColorGradingLut();
    try std.testing.expect(cfg.lut_texture == null);
    try std.testing.expect(!cfg.lut_enabled);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, lutParams(cfg));

    // 32-strip binds the same way.
    tex.width = 1024;
    tex.height = 32;
    cfg.setColorGradingLut(tex, 32);
    try std.testing.expect(cfg.lut_enabled);
    try std.testing.expectEqual([4]f32{ 1.0, 0.0, 32.0, 0.0 }, lutParams(cfg));
}

test "lut sample uv matches strip formula" {
    // 2D-strip formula: u = (b*size + r + 0.5)/(size*size),
    // v = (g + 0.5)/size, evaluated on lattice colors where no inter-slice
    // lerp applies.
    const uv = lutSampleUv(0.0, 0.0, 0.0, 16);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5 / 256.0), uv[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5 / 16.0), uv[1], 1e-7);

    // White lands on the last texel center.
    const w = lutSampleUv(1.0, 1.0, 1.0, 16);
    try std.testing.expectApproxEqAbs(@as(f32, 255.5 / 256.0), w[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 15.5 / 16.0), w[1], 1e-6);

    // Equals the floor-layer uv0 of the trilinear sample everywhere.
    const s = lutStripUv(0.3, 0.6, 0.8, 32);
    const u = lutSampleUv(0.3, 0.6, 0.8, 32);
    try std.testing.expectApproxEqAbs(s.uv0[0], u[0], 1e-7);
    try std.testing.expectApproxEqAbs(s.uv0[1], u[1], 1e-7);
}

test "identity LUT strip maps color to itself" {
    const size: u32 = 16;
    const buf = try buildIdentityLutStrip(std.testing.allocator, size);
    defer std.testing.allocator.free(buf);
    try std.testing.expectEqual(@as(usize, 16 * 16 * 16 * 4), buf.len);

    // Corners: black texel at (0, 0), white at (255, 15).
    try std.testing.expectEqual(@as(u8, 0), buf[0]);
    try std.testing.expectEqual(@as(u8, 0), buf[1]);
    try std.testing.expectEqual(@as(u8, 0), buf[2]);
    try std.testing.expectEqual(@as(u8, 255), buf[3]);
    const woff: usize = (15 * 256 + 255) * 4;
    try std.testing.expectEqual(@as(u8, 255), buf[woff]);
    try std.testing.expectEqual(@as(u8, 255), buf[woff + 1]);
    try std.testing.expectEqual(@as(u8, 255), buf[woff + 2]);
    try std.testing.expectEqual(@as(u8, 255), buf[woff + 3]);

    // Bad sizes and short buffers are hard errors, never silent garbage.
    try std.testing.expectError(error.InvalidLutStrip, buildIdentityLutStrip(std.testing.allocator, 1));
    var tiny = [_]u8{0} ** 16;
    try std.testing.expectError(error.InvalidLutStrip, writeIdentityLutStrip(&tiny, 16));

    // Round-trip sample colors through the strip bytes: quantize to the
    // lattice, read the stored texel, compare in f32. Error stays under
    // half a lattice step plus byte rounding.
    const samples = [_][3]f32{
        .{ 0.0, 0.0, 0.0 },
        .{ 1.0, 1.0, 1.0 },
        .{ 1.0, 0.0, 0.0 },
        .{ 0.5, 0.25, 0.75 },
        .{ 0.13, 0.87, 0.42 },
    };
    const step: f32 = 1.0 / 15.0;
    for (samples) |c| {
        const ri: u32 = @intFromFloat(@round(std.math.clamp(c[0], 0.0, 1.0) * 15.0));
        const gi: u32 = @intFromFloat(@round(std.math.clamp(c[1], 0.0, 1.0) * 15.0));
        const bi: u32 = @intFromFloat(@round(std.math.clamp(c[2], 0.0, 1.0) * 15.0));
        const off: usize = @as(usize, gi * 256 + bi * 16 + ri) * 4;
        const back = [3]f32{
            @as(f32, @floatFromInt(buf[off])) / 255.0,
            @as(f32, @floatFromInt(buf[off + 1])) / 255.0,
            @as(f32, @floatFromInt(buf[off + 2])) / 255.0,
        };
        try std.testing.expectApproxEqAbs(c[0], back[0], step * 0.5 + 0.003);
        try std.testing.expectApproxEqAbs(c[1], back[1], step * 0.5 + 0.003);
        try std.testing.expectApproxEqAbs(c[2], back[2], step * 0.5 + 0.003);

        // The sampling math points at the same texel: lutSampleUv of the
        // quantized color lands on the texel center.
        const q = [3]f32{
            @as(f32, @floatFromInt(ri)) / 15.0,
            @as(f32, @floatFromInt(gi)) / 15.0,
            @as(f32, @floatFromInt(bi)) / 15.0,
        };
        const uv = lutSampleUv(q[0], q[1], q[2], size);
        const cx = (@as(f32, @floatFromInt(bi * 16 + ri)) + 0.5) / 256.0;
        const cy = (@as(f32, @floatFromInt(gi)) + 0.5) / 16.0;
        try std.testing.expectApproxEqAbs(cx, uv[0], 1e-5);
        try std.testing.expectApproxEqAbs(cy, uv[1], 1e-5);
    }
}
