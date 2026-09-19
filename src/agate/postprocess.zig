const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const texture_mod = @import("texture.zig");

pub const TonemappingType = enum(u32) {
    none = 0,
    aces = 1,
    reinhard = 2,
};

// Valid range for the bloom mip pyramid depth (see clampBloomMips).
pub const BLOOM_PYRAMID_MIPS_MIN: u32 = 3;
pub const BLOOM_PYRAMID_MIPS_MAX: u32 = 7;
// Hard capacity of BloomPass target arrays; always >= MAX.
// Sized as usize so passes can index target arrays directly.
pub const BLOOM_MAX_MIPS: usize = 7;

// Golden angle (radians) used by the DoF spiral gather. Shared with
// postprocess.glsl applyDoF so CPU tests mirror the shader exactly.
pub const DOF_GOLDEN_ANGLE: f32 = 2.3999632;
pub const DOF_TAPS: u32 = 14;

// LUT color grading (2D strip, Babylon.js ColorGradingTexture parity): an
// N^3 color cube packed as an N*N wide, N tall RGBA8 image. N is the cube
// edge; 16 (256x16) and 32 (1024x32) are the recommended sizes, the math
// stays generic over 2..64. The cap keeps the blue-derived layer math
// comfortably exact in f32 and stops absurd uploads.
pub const LUT_SIZE_MIN: u32 = 2;
pub const LUT_SIZE_MAX: u32 = 64;

/// LUT texture packing. Only the 2D strip is supported (an N*N x N RGBA8
/// image uploaded with mipmaps=false and CLAMP_TO_EDGE wraps); the enum
/// exists so future packings (3D texture, .cube decode) extend here rather
/// than in PostProcessOptions.
pub const LutFormat = enum(u8) {
    strip_2d = 0,
};

/// Post-processing chain knobs (exposure, tonemapping, SSAO/bloom/DOF
/// toggles and their parameters). A flat config read/written by tooling;
/// applied per frame by `Scene.postfx` (PostFXStack.renderChain).
pub const PostProcessOptions = struct {
    enabled: bool = false,
    exposure: f32 = 1.0,
    tonemapping: TonemappingType = .aces,

    // Bloom (bright pass filter + soft Gaussian halo)
    bloom_enabled: bool = true,
    bloom_threshold: f32 = 0.8,
    bloom_intensity: f32 = 0.5,
    bloom_radius: f32 = 2.0,

    // High-quality bloom pyramid (BloomPass, Karis down + tent up).
    // When false, postprocess.glsl falls back to the single-shader bloom.
    bloom_pyramid: bool = false,
    bloom_pyramid_mips: u32 = 5,

    // Vignette (cinematic lens falloff)
    vignette_enabled: bool = true,
    vignette_intensity: f32 = 0.35,
    vignette_radius: f32 = 0.8,

    // Color Grading & Optics
    saturation: f32 = 1.05,
    contrast: f32 = 1.05,
    chromatic_aberration: f32 = 0.0,

    // Depth of Field (gather blur driven by linearized depth)
    dof_enabled: bool = false,
    dof_focus_distance: f32 = 10.0,
    dof_focus_range: f32 = 5.0,
    dof_max_blur: f32 = 8.0,

    // Parametric color curves: per-channel additive lifts in [-1, 1]
    // weighted by shadows/midtones/highlights luminance zones.
    // Primary grading path (no texture dependency). A LUT texture
    // remains an opt-in parent-side extension, see report.
    grade_shadows: [3]f32 = .{ 0.0, 0.0, 0.0 },
    grade_midtones: [3]f32 = .{ 0.0, 0.0, 0.0 },
    grade_highlights: [3]f32 = .{ 0.0, 0.0, 0.0 },

    // Texture LUT color grading (Babylon.js parity), sampled after the
    // parametric curves. With lut_texture == null (the default) the shader
    // skips the LUT branch entirely and the composite path stays
    // bit-identical to pre-LUT. The handle is a by-value Texture (plain
    // GPU handles, no CPU refs), so the whole struct — including the LUT —
    // copies into SceneFrameSnapshot.post_process with a plain assignment
    // (scene.zig packFrameSnapshot) and needs no snapshot-side support.
    lut_enabled: bool = false,
    lut_strength: f32 = 1.0,
    lut_size: u8 = 16,
    lut_format: LutFormat = .strip_2d,
    lut_texture: ?texture_mod.Texture = null,

    // Anti-Aliasing (FXAA 3.11 Sub-Pixel Edge Smoothing)
    fxaa_enabled: bool = true,

    // Atmospheric Depth & Height Fog
    fog_enabled: bool = true,
    fog_density: f32 = 0.015,
    fog_height_falloff: f32 = 0.08,
    fog_start_distance: f32 = 5.0,
    fog_color: [3]f32 = .{ 0.72, 0.82, 0.92 },
    fog_sun_scattering: f32 = 0.8,

    // Screen-Space Reflections (SSR)
    ssr_enabled: bool = true,
    ssr_intensity: f32 = 0.55,
    ssr_max_distance: f32 = 25.0,
    ssr_thickness: f32 = 0.4,

    // Sharpen (post-tonemap unsharp mask)
    sharpen_enabled: bool = false,
    sharpen_amount: f32 = 0.3,

    // Film Grain (post-tonemap hash noise, luminance-masked)
    grain_enabled: bool = false,
    grain_intensity: f32 = 0.05,

    // White Balance (post-tonemap channel gains, 0 = neutral)
    temperature: f32 = 0.0,
    tint: f32 = 0.0,

    // Camera Motion Blur (screen velocity gather blur)
    motion_blur_enabled: bool = false,
    motion_blur_intensity: f32 = 0.5,
    motion_blur_max_blur_px: f32 = 32.0,

    // Return a copy with out-of-range values pulled into valid ranges.
    // Never fails; safe to apply on load or before uploading uniforms.
    pub fn clamped(self: PostProcessOptions) PostProcessOptions {
        var out = self;
        out.exposure = @max(self.exposure, 0.0);
        out.bloom_threshold = @max(self.bloom_threshold, 0.0);
        out.bloom_intensity = @max(self.bloom_intensity, 0.0);
        out.bloom_radius = @max(self.bloom_radius, 0.0);
        out.bloom_pyramid_mips = clampBloomMips(self.bloom_pyramid_mips);
        out.dof_focus_distance = @max(self.dof_focus_distance, 0.0);
        out.dof_focus_range = @max(self.dof_focus_range, 0.0);
        out.dof_max_blur = @max(self.dof_max_blur, 0.0);
        out.motion_blur_intensity = std.math.clamp(self.motion_blur_intensity, 0.0, 3.0);
        out.motion_blur_max_blur_px = std.math.clamp(self.motion_blur_max_blur_px, 1.0, 128.0);
        out.grade_shadows = clampGrade(self.grade_shadows);
        out.grade_midtones = clampGrade(self.grade_midtones);
        out.grade_highlights = clampGrade(self.grade_highlights);
        out.lut_strength = std.math.clamp(self.lut_strength, 0.0, 1.0);
        // A binding without a live view, a supported size, or matching
        // strip dims can never be sampled; drop it so the pass keeps its
        // no-LUT path instead of binding a dead handle.
        if (out.lut_texture) |tex| {
            if (!lutTextureValid(tex, out.lut_size)) out.lut_texture = null;
        }
        return out;
    }

    /// Bind a 2D-strip LUT texture (Babylon.js parity) and enable grading.
    /// The strip geometry is validated (width == size*size, height == size,
    /// live view); invalid input — including null — clears the binding and
    /// disables grading, so the shader can never sample garbage. Consumed
    /// per frame by PostFXStack.renderChain via ChainParams.post, which is
    /// copied from the render-owned SceneFrameSnapshot; that is why the
    /// setter lives on the config struct and not on PostFXStack.
    pub fn setColorGradingLut(self: *PostProcessOptions, tex: ?texture_mod.Texture, size: u8) void {
        if (tex) |t| {
            if (lutTextureValid(t, size)) {
                self.lut_texture = t;
                self.lut_size = size;
                self.lut_enabled = true;
                return;
            }
        }
        self.clearColorGradingLut();
    }

    /// Unbind the LUT and disable grading (back to the bit-identical
    /// no-LUT composite path).
    pub fn clearColorGradingLut(self: *PostProcessOptions) void {
        self.lut_texture = null;
        self.lut_enabled = false;
    }
};

// Clamp the requested pyramid depth into [3, 7].
pub fn clampBloomMips(mips: u32) u32 {
    return std.math.clamp(mips, BLOOM_PYRAMID_MIPS_MIN, BLOOM_PYRAMID_MIPS_MAX);
}

// Clamp one grade triplet into [-1, 1] per channel.
pub fn clampGrade(v: [3]f32) [3]f32 {
    return .{
        std.math.clamp(v[0], -1.0, 1.0),
        std.math.clamp(v[1], -1.0, 1.0),
        std.math.clamp(v[2], -1.0, 1.0),
    };
}

// True when `size` is a supported LUT cube edge (strip is N*N x N).
pub fn validLutSize(size: u32) bool {
    return size >= LUT_SIZE_MIN and size <= LUT_SIZE_MAX;
}

// True when `tex` can be sampled as a 2D-strip LUT of cube edge `size`:
// live view plus exact strip dims (width == size*size, height == size).
pub fn lutTextureValid(tex: texture_mod.Texture, size: u8) bool {
    if (tex.view.id == 0) return false;
    const n: u32 = @as(u32, size);
    if (!validLutSize(n)) return false;
    return tex.width == n * n and tex.height == n;
}

// Validated 2D strip geometry for one LUT.
pub const LutStripLayout = struct {
    /// Cube edge N (also the strip height).
    size: u32,
    /// Strip pixel width, always size * size.
    width: u32,
    /// Strip pixel height, always size.
    height: u32,
};

// Validate decoded LUT strip dimensions: the image must be N texels tall
// and N*N wide (one N x N slice per blue step), with N in the supported
// range. Any other aspect cannot address the cube, so it is a hard error
// at load time rather than a silent broken grade.
pub fn lutStripLayout(width: u32, height: u32) !LutStripLayout {
    if (height < LUT_SIZE_MIN or height > LUT_SIZE_MAX or width != height * height) {
        return error.InvalidLutStrip;
    }
    return .{ .size = height, .width = width, .height = height };
}

// One manual-trilinear LUT strip lookup: two layer uvs plus the blend
// weight between them.
pub const LutStripSample = struct {
    /// Strip uv inside layer floor(t) (blue axis).
    uv0: [2]f32,
    /// Strip uv inside the next layer up (same layer when b == 1).
    uv1: [2]f32,
    /// Linear blend weight from uv0's color toward uv1's color.
    blend: f32,
};

// Strip uv math for one color lookup. Mirrors applyLut in
// postprocess.glsl so CPU tests pin the exact shader formula.
//
// Derivation (N = cube edge, strip is N*N wide by N tall, v = 0 is the
// first row as decoded/uploaded):
//   1. Blue addresses the cube's third axis scaled to layer centers:
//      t = b*(N-1), so b in {0,1} lands exactly on the first/last layer.
//   2. t splits into integer layer k = floor(t) and fraction f = t - k;
//      sampling layers k and k+1 and mixing by f is the manual
//      trilinear third axis (hardware bilinear only covers r/g inside
//      one layer).
//   3. Red/green address texel centers inside one N x N slice with the
//      same center inset, (c*(N-1)+0.5)/N: c=0/1 hit the slice edge
//      centers, and the half-texel inset keeps hardware bilinear inside
//      the layer (no bleed across neighboring slices in the strip).
//   4. Layer k starts at column k*N of the strip, so
//      uv.x = (k + u_slice)/N and uv.y = v_slice.
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

// Single-layer strip uv for one color lookup: the per-slice coordinate of
// the 2D-strip formula (u = (b*size + r + 0.5)/(size*size),
// v = (g + 0.5)/size, up to the texel-center quantization used here). This
// is uv0 of lutStripUv for the floor(b) layer; the full trilinear path
// (lutStripUv + applyLutStrip) lerps toward the next layer explicitly
// because hardware bilinear only interpolates r/g inside one slice.
pub fn lutSampleUv(r: f32, g: f32, b: f32, size: u32) [2]f32 {
    return lutStripUv(r, g, b, size).uv0;
}

// Final LUT blend. Mirrors applyLut in postprocess.glsl: mix the two
// sampled layers by the trilinear weight, then blend the graded color
// back toward the curve-graded input by `intensity` in [0, 1].
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

// Fill `buf` with an identity 2D-strip LUT for cube edge `size`: texel
// (column b*size + r, row g) stores the lattice color
// (r/(size-1), g/(size-1), b/(size-1), opaque). CPU-side helper for tests
// and the sandbox; GPU upload stays with Texture.initRaw using
// mipmaps=false and CLAMP_TO_EDGE wraps (see the LutFormat note above).
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

// Pack the shader lut_params vec4: (enabled 1/0, strength, size N, 0).
// Anything that cannot sample — no binding, dead view, bad size,
// mismatched strip dims, disabled flag — packs all zeros, which is exactly
// the pre-LUT uniform state, so the no-LUT path stays unchanged. Kept
// beside the Zig LUT math so tests pin what postprocess_pass uploads.
pub fn lutParams(cfg: PostProcessOptions) [4]f32 {
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

pub const BloomMipSize = struct {
    w: i32,
    h: i32,
};

// Size of one bloom pyramid level. Level 0 is half resolution, each
// further level halves again, clamped to 1x1. Mirrors BloomPass.resize.
pub fn bloomMipSize(base_w: i32, base_h: i32, level: u32) BloomMipSize {
    var w: i32 = @max(1, @divTrunc(base_w, 2));
    var h: i32 = @max(1, @divTrunc(base_h, 2));
    var i: u32 = 0;
    while (i < level) : (i += 1) {
        w = @max(1, @divTrunc(w, 2));
        h = @max(1, @divTrunc(h, 2));
    }
    return .{ .w = w, .h = h };
}

// Linearize a [0, 1] depth buffer value. Mirrors linearize helpers in
// ssao_blur.glsl and postprocess.glsl. Returns far for degenerate input.
pub fn linearizeDepth(raw_depth: f32, near: f32, far: f32) f32 {
    if (far <= near) return far;
    const denom = far - raw_depth * (far - near);
    return (near * far) / @max(denom, 0.0001);
}

// Circle of confusion in pixels for a linearized view distance.
// 0 inside the focal plane, ramps to max_blur at focus_range away.
// Mirrors postprocess.glsl applyDoF exactly (including the epsilon guard).
pub fn circleOfConfusion(depth_linear: f32, focus_distance: f32, focus_range: f32, max_blur: f32) f32 {
    const fr = @max(focus_range, 0.0001);
    const coc = @abs(depth_linear - focus_distance) / fr;
    return @min(coc, 1.0) * @max(max_blur, 0.0);
}

// One DoF spiral tap offset in pixels for tap `index` of `taps` taps at
// gather radius `radius_px`. Mirrors postprocess.glsl applyDoF.
pub fn dofTapOffset(index: u32, taps: u32, radius_px: f32) [2]f32 {
    const fi: f32 = @floatFromInt(index);
    const ft: f32 = @floatFromInt(@max(taps, 1));
    const ang = fi * DOF_GOLDEN_ANGLE;
    const rr = (fi + 0.5) / ft * radius_px;
    return .{ @cos(ang) * rr, @sin(ang) * rr };
}

// Karis weighting for firefly suppression: bright outliers contribute
// less to the downsampled average. Mirrors bloom_down.glsl.
pub fn karisWeight(luma: f32) f32 {
    return 1.0 / (1.0 + @max(luma, 0.0));
}

// 1D tent (triangle) filter weight. Mirrors the separable form of the
// 3x3 tent kernel used by bloom_up.glsl.
pub fn tentWeight1D(x: f32) f32 {
    return @max(0.0, 1.0 - @abs(x));
}

// 3x3 tent kernel weight for integer offsets in [-1, 1], normalized so
// the kernel sums to 1 (center 4/16, edges 2/16, corners 1/16).
// Returns 0 for offsets outside the kernel.
pub fn bloomTentWeight(ix: i32, iy: i32) f32 {
    if (ix < -1 or ix > 1 or iy < -1 or iy > 1) return 0.0;
    const ax: f32 = if (ix == 0) 2.0 else 1.0;
    const ay: f32 = if (iy == 0) 2.0 else 1.0;
    return (ax * ay) / 16.0;
}

pub fn rgbLuma(c: [3]f32) f32 {
    return c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722;
}

// Parametric zone grade. Mirrors applyColorCurves in postprocess.glsl:
// each lift is weighted by its luminance zone (shadows ramp out by
// l=0.5, highlights ramp in from l=0.5, midtones peak at l=0.5).
pub fn applyGrade(color: [3]f32, shadows: [3]f32, midtones: [3]f32, highlights: [3]f32) [3]f32 {
    const l = rgbLuma(color);
    const w_s = std.math.clamp(1.0 - l * 2.0, 0.0, 1.0);
    const w_h = std.math.clamp((l - 0.5) * 2.0, 0.0, 1.0);
    const w_m = std.math.clamp(1.0 - @abs(l - 0.5) * 2.0, 0.0, 1.0);
    return .{
        color[0] + shadows[0] * w_s + midtones[0] * w_m + highlights[0] * w_h,
        color[1] + shadows[1] * w_s + midtones[1] * w_m + highlights[1] * w_h,
        color[2] + shadows[2] * w_s + midtones[2] * w_m + highlights[2] * w_h,
    };
}

test "postprocess defaults" {
    const cfg = PostProcessOptions{};
    try std.testing.expect(!cfg.enabled);
    try std.testing.expect(!cfg.bloom_pyramid);
    try std.testing.expectEqual(@as(u32, 5), cfg.bloom_pyramid_mips);
    try std.testing.expect(!cfg.dof_enabled);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), cfg.dof_focus_distance, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), cfg.dof_focus_range, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), cfg.dof_max_blur, 1e-6);
    try std.testing.expectEqual([3]f32{ 0.0, 0.0, 0.0 }, cfg.grade_shadows);
    try std.testing.expectEqual([3]f32{ 0.0, 0.0, 0.0 }, cfg.grade_midtones);
    try std.testing.expectEqual([3]f32{ 0.0, 0.0, 0.0 }, cfg.grade_highlights);
    // Legacy single-shader bloom stays on by default as the fallback path.
    try std.testing.expect(cfg.bloom_enabled);
    try std.testing.expectApproxEqAbs(@as(f32, 1.05), cfg.saturation, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.05), cfg.contrast, 1e-6);
}

test "bloom mips clamp" {
    try std.testing.expectEqual(@as(u32, 3), clampBloomMips(0));
    try std.testing.expectEqual(@as(u32, 3), clampBloomMips(3));
    try std.testing.expectEqual(@as(u32, 5), clampBloomMips(5));
    try std.testing.expectEqual(@as(u32, 7), clampBloomMips(7));
    try std.testing.expectEqual(@as(u32, 7), clampBloomMips(42));

    var cfg = PostProcessOptions{ .bloom_pyramid_mips = 99 };
    try std.testing.expectEqual(@as(u32, 7), cfg.clamped().bloom_pyramid_mips);
    cfg.bloom_pyramid_mips = 1;
    try std.testing.expectEqual(@as(u32, 3), cfg.clamped().bloom_pyramid_mips);
}

test "config clamped sanitizes new fields" {
    var cfg = PostProcessOptions{
        .exposure = -2.0,
        .dof_focus_distance = -4.0,
        .dof_focus_range = -1.0,
        .dof_max_blur = -3.0,
        .grade_shadows = .{ 2.0, -2.0, 0.5 },
    };
    const out = cfg.clamped();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.exposure, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.dof_focus_distance, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.dof_focus_range, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out.dof_max_blur, 1e-6);
    try std.testing.expectEqual([3]f32{ 1.0, -1.0, 0.5 }, out.grade_shadows);
}

test "bloom mip sizes" {
    try std.testing.expectEqual(BloomMipSize{ .w = 640, .h = 360 }, bloomMipSize(1280, 720, 0));
    try std.testing.expectEqual(BloomMipSize{ .w = 320, .h = 180 }, bloomMipSize(1280, 720, 1));
    try std.testing.expectEqual(BloomMipSize{ .w = 40, .h = 22 }, bloomMipSize(1280, 720, 4));
    // Odd dimensions round down but never below 1x1.
    try std.testing.expectEqual(BloomMipSize{ .w = 1, .h = 1 }, bloomMipSize(3, 3, 3));
    const a = bloomMipSize(1920, 1080, 6);
    try std.testing.expect(a.w >= 1 and a.h >= 1);
}

test "circle of confusion" {
    // In focus -> no blur.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), circleOfConfusion(10.0, 10.0, 5.0, 8.0), 1e-6);
    // Halfway to the ramp edge -> half of max blur.
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), circleOfConfusion(12.5, 10.0, 5.0, 8.0), 1e-5);
    // Beyond the range -> clamped to max blur (both sides).
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), circleOfConfusion(100.0, 10.0, 5.0, 8.0), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), circleOfConfusion(0.0, 10.0, 5.0, 8.0), 1e-5);
    // Degenerate range never divides by zero.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), circleOfConfusion(10.0, 10.0, 0.0, 8.0), 1e-6);
    try std.testing.expect(circleOfConfusion(11.0, 10.0, 0.0, 8.0) >= 0.0);
}

test "linearize depth" {
    // Near maps to near, far maps to far.
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), linearizeDepth(0.0, 1.0, 100.0), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), linearizeDepth(1.0, 1.0, 100.0), 1e-2);
    // Monotonic in between.
    const a = linearizeDepth(0.5, 0.1, 50.0);
    const b = linearizeDepth(0.9, 0.1, 50.0);
    try std.testing.expect(a < b);
}

test "tent weights" {
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), tentWeight1D(0.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), tentWeight1D(0.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), tentWeight1D(1.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), tentWeight1D(2.0), 1e-6);
    // 3x3 kernel shape and normalization.
    try std.testing.expectApproxEqAbs(@as(f32, 4.0 / 16.0), bloomTentWeight(0, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 16.0), bloomTentWeight(1, 0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 16.0), bloomTentWeight(0, -1), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 16.0), bloomTentWeight(1, 1), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), bloomTentWeight(2, 0), 1e-6);
    var sum: f32 = 0.0;
    var ix: i32 = -1;
    while (ix <= 1) : (ix += 1) {
        var iy: i32 = -1;
        while (iy <= 1) : (iy += 1) sum += bloomTentWeight(ix, iy);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-6);
}

test "karis weight" {
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), karisWeight(0.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), karisWeight(1.0), 1e-6);
    // Outliers are suppressed but never reach zero or negative.
    try std.testing.expect(karisWeight(10.0) < karisWeight(1.0));
    try std.testing.expect(karisWeight(10.0) > 0.0);
}

test "dof tap offsets" {
    // First tap points along +X with the expected spiral radius.
    const t0 = dofTapOffset(0, DOF_TAPS, 8.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5 / 14.0 * 8.0), t0[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), t0[1], 1e-5);
    // All taps stay inside the gather radius.
    for (0..DOF_TAPS) |i| {
        const t = dofTapOffset(@intCast(i), DOF_TAPS, 8.0);
        try std.testing.expect(@sqrt(t[0] * t[0] + t[1] * t[1]) <= 8.0 + 1e-5);
    }
    // Zero radius collapses every tap to the center pixel.
    const z = dofTapOffset(7, DOF_TAPS, 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), z[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), z[1], 1e-6);
}

test "color grade identity and shadows" {
    const mid = [3]f32{ 0.5, 0.5, 0.5 };
    const zero = [3]f32{ 0.0, 0.0, 0.0 };
    const id = applyGrade(mid, zero, zero, zero);
    try std.testing.expectApproxEqAbs(mid[0], id[0], 1e-6);
    try std.testing.expectApproxEqAbs(mid[1], id[1], 1e-6);
    try std.testing.expectApproxEqAbs(mid[2], id[2], 1e-6);
    // Shadows lift applies fully to black, not at all to white.
    const lifted = applyGrade(zero, .{ 0.2, 0.1, 0.0 }, zero, zero);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), lifted[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), lifted[1], 1e-6);
    const white = applyGrade(.{ 1.0, 1.0, 1.0 }, .{ 0.2, 0.2, 0.2 }, zero, zero);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), white[0], 1e-6);
    // Highlights lift applies fully to white, not at all to black.
    const hl = applyGrade(.{ 1.0, 1.0, 1.0 }, zero, zero, .{ 0.0, 0.3, 0.0 });
    try std.testing.expectApproxEqAbs(@as(f32, 1.3), hl[1], 1e-6);
    const hl_dark = applyGrade(zero, zero, zero, .{ 0.0, 0.3, 0.0 });
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), hl_dark[1], 1e-6);
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
    const def = PostProcessOptions{};
    try std.testing.expect(def.lut_texture == null);
    try std.testing.expect(!def.lut_enabled);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, lutParams(def));
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, lutParams(def.clamped()));

    // A live binding enables the uniform triplet.
    var live = PostProcessOptions{
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
    var cfg = PostProcessOptions{};
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
