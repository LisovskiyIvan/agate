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

// Lut regression tests live in `lut_tests.zig` (same directory,
// imported below so the test registry picks them up exactly once).

test {
    _ = @import("lut_tests.zig");
}
