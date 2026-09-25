const std = @import("std");
const texture_mod = @import("../texture.zig");
const types = @import("types.zig");
const options = @import("options.zig");

pub const LutFormat = types.LutFormat;
pub const LutStripLayout = types.LutStripLayout;
pub const LutStripSample = types.LutStripSample;

// LUT color grading (2D strip, Babylon.js ColorGradingTexture parity): an
// N^3 color cube packed as an N*N wide, N tall RGBA8 image. N is the cube
// edge; 16 (256x16) and 32 (1024x32) are the recommended sizes, the math
// stays generic over 2..64. The cap keeps the blue-derived layer math
// comfortably exact in f32 and stops absurd uploads.
pub const LUT_SIZE_MIN: u32 = 2;
pub const LUT_SIZE_MAX: u32 = 64;

/// True when `size` is a supported LUT cube edge (strip is N*N x N).
pub fn validLutSize(size: u32) bool {
    return size >= LUT_SIZE_MIN and size <= LUT_SIZE_MAX;
}

/// True when `tex` can be sampled as a 2D-strip LUT of cube edge `size`:
/// live view plus exact strip dims (width == size*size, height == size).
pub fn lutTextureValid(tex: texture_mod.Texture, size: u8) bool {
    if (tex.view.id == 0) return false;
    const n: u32 = @as(u32, size);
    if (!validLutSize(n)) return false;
    return tex.width == n * n and tex.height == n;
}

/// Validate decoded LUT strip dimensions: the image must be N texels tall
/// and N*N wide (one N x N slice per blue step), with N in the supported
/// range. Any other aspect cannot address the cube, so it is a hard error
/// at load time rather than a silent broken grade.
pub fn lutStripLayout(width: u32, height: u32) !LutStripLayout {
    if (height < LUT_SIZE_MIN or height > LUT_SIZE_MAX or width != height * height) {
        return error.InvalidLutStrip;
    }
    return .{ .size = height, .width = width, .height = height };
}

/// Strip uv math for one color lookup. Mirrors applyLut in
/// postprocess.glsl so CPU tests pin the exact shader formula.
pub fn lutStripUv(r: f32, g: f32, b: f32, size: u32) LutStripSample {
    const n: f32 = @floatFromInt(size);
    const t = std.math.clamp(b, 0.0, 1.0) * (n - 1.0);
    const k = @floor(t);
    const f = t - k;
    const u_slice = (std.math.clamp(r, 0.0, 1.0) * (n - 1.0) + 0.5) / n;
    const v_slice = (std.math.clamp(g, 0.0, 1.0) * (n - 1.0) + 0.5) / n;
    const k1 = @min(k + 1.0, n - 1.0);
    return .{
        .uv0 = .{ (k + u_slice) / n, v_slice },
        .uv1 = .{ (k1 + u_slice) / n, v_slice },
        .blend = f,
    };
}

/// Single-layer strip uv for one color lookup: the per-slice coordinate of
/// the 2D-strip formula (u = (b*size + r + 0.5)/(size*size),
/// v = (g + 0.5)/size, up to the texel-center quantization used here). This
/// is uv0 of lutStripUv for the floor(b) layer; the full trilinear path
/// (lutStripUv + applyLutStrip) lerps toward the next layer explicitly
/// because hardware bilinear only interpolates r/g inside one slice.
pub fn lutSampleUv(r: f32, g: f32, b: f32, size: u32) [2]f32 {
    return lutStripUv(r, g, b, size).uv0;
}

/// Final LUT blend. Mirrors applyLut in postprocess.glsl: mix the two
/// sampled layers by the trilinear weight, then blend the graded color
/// back toward the curve-graded input by `intensity` in [0, 1].
pub fn applyLutStrip(
    color: [3]f32,
    layer0: [3]f32,
    layer1: [3]f32,
    sample: LutStripSample,
    intensity: f32,
) [3]f32 {
    const graded = [3]f32{
        layer0[0] + (layer1[0] - layer0[0]) * sample.blend,
        layer0[1] + (layer1[1] - layer0[1]) * sample.blend,
        layer0[2] + (layer1[2] - layer0[2]) * sample.blend,
    };
    const i = std.math.clamp(intensity, 0.0, 1.0);
    return .{
        color[0] + (graded[0] - color[0]) * i,
        color[1] + (graded[1] - color[1]) * i,
        color[2] + (graded[2] - color[2]) * i,
    };
}

fn quantizeLutChannel(v: u32, denom: f32) u8 {
    const f: f32 = @floatFromInt(v);
    return @intFromFloat(@round(f / denom * 255.0));
}

/// Fill `buf` with an identity 2D-strip LUT for cube edge `size`: texel
/// (column b*size + r, row g) stores the lattice color
/// (r/(size-1), g/(size-1), b/(size-1), opaque). CPU-side helper for tests
/// and the sandbox; GPU upload stays with Texture.initRaw using
/// mipmaps=false and CLAMP_TO_EDGE wraps (see the LutFormat note above).
pub fn writeIdentityLutStrip(buf: []u8, size: u32) !void {
    if (!validLutSize(size)) return error.InvalidLutStrip;
    const n: usize = @as(usize, size);
    if (buf.len != n * n * n * 4) return error.InvalidLutStrip;
    const denom: f32 = @floatFromInt(size - 1);
    var b: u32 = 0;
    while (b < size) : (b += 1) {
        var g: u32 = 0;
        while (g < size) : (g += 1) {
            var r: u32 = 0;
            while (r < size) : (r += 1) {
                const x: usize = @as(usize, b * size + r);
                const y: usize = @as(usize, g);
                const off = (y * n * n + x) * 4;
                buf[off] = quantizeLutChannel(r, denom);
                buf[off + 1] = quantizeLutChannel(g, denom);
                buf[off + 2] = quantizeLutChannel(b, denom);
                buf[off + 3] = 255;
            }
        }
    }
}

/// Allocate and fill an identity 2D-strip LUT (size*size*size*4 RGBA8
/// bytes) for tests and the sandbox. Caller owns the returned slice.
pub fn buildIdentityLutStrip(allocator: std.mem.Allocator, size: u32) ![]u8 {
    if (!validLutSize(size)) return error.InvalidLutStrip;
    const n: usize = @as(usize, size);
    const buf = try allocator.alloc(u8, n * n * n * 4);
    errdefer allocator.free(buf);
    try writeIdentityLutStrip(buf, size);
    return buf;
}

/// Pack the shader lut_params vec4: (enabled 1/0, strength, size N, 0).
/// Anything that cannot sample — no binding, dead view, bad size,
/// mismatched strip dims, disabled flag — packs all zeros, which is exactly
/// the pre-LUT uniform state, so the no-LUT path stays unchanged. Kept
/// beside the Zig LUT math so tests pin what postprocess_pass uploads.
pub fn lutParams(cfg: options.PostProcessOptions) [4]f32 {
    const tex = cfg.lut_texture orelse return .{ 0.0, 0.0, 0.0, 0.0 };
    if (!cfg.lut_enabled or !lutTextureValid(tex, cfg.lut_size)) {
        return .{ 0.0, 0.0, 0.0, 0.0 };
    }
    return .{
        1.0,
        std.math.clamp(cfg.lut_strength, 0.0, 1.0),
        @floatFromInt(cfg.lut_size),
        0.0,
    };
}

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
