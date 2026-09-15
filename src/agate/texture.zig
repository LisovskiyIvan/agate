const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const c = @import("c.zig").c;
const ktx2 = @import("ktx2.zig");

/// Shared RGBA8 box-filter downsample of one mip level. Dims floor at 1,
/// source coords clamp at edges (handles NPOT). Fast path for the exact
/// 2:1 case keeps power-of-two results bit-identical to the old inline
/// loops; the generic path covers odd/NPOT sizes.
/// Precondition: src.len >= src_w*src_h*4, dst.len >= dst_w*dst_h*4.
fn boxDownsampleU8(src: []const u8, src_w: u32, src_h: u32, dst: []u8, dst_w: u32, dst_h: u32) void {
    @setRuntimeSafety(false);
    if (src_w == dst_w * 2 and src_h == dst_h * 2) {
        const src_stride: usize = @as(usize, src_w) * 4;
        const dst_stride: usize = @as(usize, dst_w) * 4;
        var y: usize = 0;
        while (y < dst_h) : (y += 1) {
            const r0 = src[(y * 2) * src_stride ..];
            const r1 = src[(y * 2 + 1) * src_stride ..];
            const out_row = dst[y * dst_stride ..];
            var x: usize = 0;
            while (x < dst_w) : (x += 1) {
                const si = x * 8;
                const di = x * 4;
                inline for (0..4) |ch| {
                    const sum: u32 = @as(u32, r0[si + ch]) +
                        @as(u32, r0[si + 4 + ch]) +
                        @as(u32, r1[si + ch]) +
                        @as(u32, r1[si + 4 + ch]);
                    out_row[di + ch] = @intCast((sum + 2) >> 2);
                }
            }
        }
        return;
    }

    var y: u32 = 0;
    while (y < dst_h) : (y += 1) {
        const sy0 = @min(y * 2, src_h - 1);
        const sy1 = @min(y * 2 + 1, src_h - 1);
        var x: u32 = 0;
        while (x < dst_w) : (x += 1) {
            const sx0 = @min(x * 2, src_w - 1);
            const sx1 = @min(x * 2 + 1, src_w - 1);
            const q00 = (sy0 * src_w + sx0) * 4;
            const q10 = (sy0 * src_w + sx1) * 4;
            const q01 = (sy1 * src_w + sx0) * 4;
            const q11 = (sy1 * src_w + sx1) * 4;
            const o = (y * dst_w + x) * 4;
            inline for (0..4) |ch| {
                const sum: u32 = @as(u32, src[q00 + ch]) + @as(u32, src[q10 + ch]) + @as(u32, src[q01 + ch]) + @as(u32, src[q11 + ch]);
                dst[o + ch] = @intCast((sum + 2) >> 2);
            }
        }
    }
}

/// Comptime sRGB -> linear LUT for one u8 channel (IEC 61966-2-1). Round to
/// nearest: linear byte = trunc(linear_f32 * 255 + 0.5).
const srgb_to_linear_lut: [256]u8 = blk: {
    @setEvalBranchQuota(200000);
    var table: [256]u8 = undefined;
    for (0..256) |i| {
        const srgb: f32 = @as(f32, @floatFromInt(i)) / 255.0;
        const lin = if (srgb <= 0.04045)
            srgb / 12.92
        else
            std.math.pow(f32, (srgb + 0.055) / 1.055, 2.4);
        table[i] = @intFromFloat(std.math.clamp(lin * 255.0 + 0.5, 0.0, 255.0));
    }
    break :blk table;
};

/// Exact per-byte sRGB -> linear conversion used for LDR color textures.
pub fn srgbToLinearU8(value: u8) u8 {
    return srgb_to_linear_lut[value];
}

/// In-place sRGB -> linear on RGB lanes of an RGBA8 buffer; alpha lanes are
/// never touched. Runs BEFORE mip generation so the box filter averages in
/// linear space.
fn convertSrgbToLinearInPlace(pixels: []u8) void {
    for (pixels, 0..) |*byte, i| {
        if (i % 4 != 3) byte.* = srgb_to_linear_lut[byte.*];
    }
}

/// Radial particle-dot falloff shared by createDefaultParticleDot32 and
/// createParticleDot. Takes already-computed center/radius so both callers
/// keep their exact historic parameters and output is unchanged.
fn particleDotAlpha(x: u32, y: u32, center: f32, radius: f32) u8 {
    const dx = @as(f32, @floatFromInt(x)) - center;
    const dy = @as(f32, @floatFromInt(y)) - center;
    const dist = @sqrt(dx * dx + dy * dy);
    const norm_dist = @min(1.0, dist / radius);
    const alpha_f = (1.0 - norm_dist) * (1.0 - norm_dist);
    return @intFromFloat(std.math.clamp(alpha_f * 255.0, 0.0, 255.0));
}

/// Checked RGBA8 face size (size*size*4). The old u32 `size * size * 4`
/// wrapped to a small value for large sizes, causing undersized allocations
/// and OOB writes. Uses the file's ImageTooLarge/InvalidDimensions
/// conventions; returns usize for direct use as an alloc length.
fn checkedFaceBytes(size: u32) !usize {
    if (size == 0) return error.InvalidDimensions;
    const pixels = std.math.mul(u32, size, size) catch return error.ImageTooLarge;
    const bytes = std.math.mul(u32, pixels, 4) catch return error.ImageTooLarge;
    return @as(usize, bytes);
}

pub const Texture = struct {
    image: sg.Image,
    view: sg.View,
    sampler: sg.Sampler,
    width: u32,
    height: u32,
    /// Pixel format of the GPU image. LDR loaders leave the default RGBA8;
    /// HDR loaders set RGBA16F. Defaults keep existing call sites unchanged.
    format: sg.PixelFormat = .RGBA8,
    /// True for float (HDR) textures: no gamma correction, linear sampling.
    is_hdr: bool = false,

    pub const Options = struct {
        min_filter: sg.Filter = .LINEAR,
        mag_filter: sg.Filter = .LINEAR,
        /// Mip selection filter used when the texture has a mip chain. Has no
        /// effect on single-level textures. glTF's mipmapped min_filter
        /// variants map their mip half here (loader/materials.zig).
        mip_filter: sg.Filter = .LINEAR,
        wrap_u: sg.Wrap = .REPEAT,
        wrap_v: sg.Wrap = .REPEAT,
        /// Max anisotropy 1..16 (sokol sg_sampler_desc.max_anisotropy). sokol
        /// requires LINEAR min/mag/mip filters for anisotropy > 1 and fails
        /// sampler validation otherwise, so non-LINEAR combinations clamp
        /// back to 1 here. sokol has no lod bias; lod range defaults to
        /// 0..FLT_MAX (not exposed).
        max_anisotropy: u32 = 1,
        /// Signed distance fields must not be box-downsampled: the mip chain
        /// dilutes thin strokes and the shader edge drifts. Disable for fonts.
        mipmaps: bool = true,
        /// Convert sRGB to linear at load time, BEFORE mip generation (the
        /// box filter then averages in linear space). Turn ON for LDR color
        /// textures that feed lighting math (glTF albedo/emissive), keep OFF
        /// for data textures (normal / metallic-roughness / occlusion: they
        /// are authored linear) and for GPU-bound views where the raw bytes
        /// matter (fonts, LUTs, sprites drawn without lighting).
        srgb_to_linear: bool = false,
    };

    /// Shared sampler creation for LDR 2D textures: applies min/mag/wrap,
    /// the mip filter (only meaningful with a chain), and the anisotropy
    /// guard described on Options.max_anisotropy.
    fn makeSamplerFor(options: Options, num_mip_levels: u32) sg.Sampler {
        const mip_filter: sg.Filter = if (num_mip_levels > 1) options.mip_filter else .NEAREST;
        var aniso = options.max_anisotropy;
        if (aniso > 16) aniso = 16;
        if (aniso > 1 and (options.min_filter != .LINEAR or options.mag_filter != .LINEAR or mip_filter != .LINEAR)) {
            aniso = 1;
        }
        return sg.makeSampler(.{
            .min_filter = options.min_filter,
            .mag_filter = options.mag_filter,
            .mipmap_filter = mip_filter,
            .wrap_u = options.wrap_u,
            .wrap_v = options.wrap_v,
            .max_anisotropy = aniso,
        });
    }

    pub fn initRaw(width: u32, height: u32, rgba_pixels: []const u8, options: Options) Texture {
        var img_desc = sg.ImageDesc{
            .width = @intCast(width),
            .height = @intCast(height),
            .pixel_format = .RGBA8,
        };
        img_desc.data.mip_levels[0] = sg.asRange(rgba_pixels);

        const img = sg.makeImage(img_desc);
        const view = sg.makeView(.{
            .texture = .{ .image = img },
        });

        const smp = makeSamplerFor(options, 1);

        return .{
            .image = img,
            .view = view,
            .sampler = smp,
            .width = width,
            .height = height,
        };
    }
    /// Box-filter downsample of one RGBA8 level. Dims floor at 1, source coords
    /// clamp at edges (handles NPOT). Thin wrapper over the shared
    /// file-private `boxDownsampleU8` so 2D and cube paths stay in sync.
    fn downsampleLevel(src: []const u8, src_w: u32, src_h: u32, dst: []u8, dst_w: u32, dst_h: u32) void {
        boxDownsampleU8(src, src_w, src_h, dst, dst_w, dst_h);
    }

    pub fn mipLevelCount(width: u32, height: u32) u32 {
        var levels: u32 = 1;
        var w = width;
        var h = height;
        while ((w > 1 or h > 1) and levels < 16) {
            w = @max(1, w / 2);
            h = @max(1, h / 2);
            levels += 1;
        }
        return levels;
    }

    /// initRaw plus a full CPU mipmap chain. Uploads immediately on the
    /// calling (main) thread; use decodeMemory + fromRaw for parallel loads.
    pub fn initRawMipped(allocator: std.mem.Allocator, width: u32, height: u32, rgba_pixels: []const u8, options: Options) !Texture {
        if (mipLevelCount(width, height) == 1) return initRaw(width, height, rgba_pixels, options);

        var raw = try buildRaw(allocator, width, height, rgba_pixels, true);
        defer raw.deinit(allocator);
        return fromRaw(&raw, options);
    }

    /// CPU-decoded RGBA pixels plus an optional mip chain. Decoding and mip
    /// generation are GPU-free, so they can run on worker threads; the
    /// sg.Image must be created later with `fromRaw` on the main thread.
    pub const RawTexture = struct {
        width: u32 = 0,
        height: u32 = 0,
        num_levels: u32 = 0,
        is_srgb: bool = false,
        levels: [16]?[]u8 = @splat(null),

        pub fn deinit(self: *RawTexture, allocator: std.mem.Allocator) void {
            for (self.levels[0..self.num_levels]) |level| {
                if (level) |buf| allocator.free(buf);
            }
            self.* = .{};
        }
    };

    /// Copies `rgba_pixels` and optionally builds the box-filtered mip chain.
    /// Public so the KTX2 reader (ktx2.zig) can reuse the generator for
    /// single-level files; the decode paths above use it internally.
    pub fn buildRaw(allocator: std.mem.Allocator, width: u32, height: u32, rgba_pixels: []const u8, gen_mipmaps: bool) !RawTexture {
        var raw = RawTexture{ .width = width, .height = height, .num_levels = 1 };
        errdefer raw.deinit(allocator);

        const level0 = try allocator.alloc(u8, rgba_pixels.len);
        @memcpy(level0, rgba_pixels);
        raw.levels[0] = level0;

        if (gen_mipmaps) {
            var prev_w = width;
            var prev_h = height;
            var prev: []const u8 = level0;
            const levels = mipLevelCount(width, height);
            for (1..levels) |m| {
                const cur_w: u32 = @max(1, prev_w / 2);
                const cur_h: u32 = @max(1, prev_h / 2);
                const cur = try allocator.alloc(u8, @as(usize, cur_w) * cur_h * 4);
                raw.levels[m] = cur;
                downsampleLevel(prev, prev_w, prev_h, cur, cur_w, cur_h);
                raw.num_levels = @intCast(m + 1);
                prev_w = cur_w;
                prev_h = cur_h;
                prev = cur;
            }
        }
        return raw;
    }

    /// Creates the GPU image from CPU-decoded pixels. Main thread only.
    pub fn fromRaw(raw: *const RawTexture, options: Options) Texture {
        var img_desc = sg.ImageDesc{
            .width = @intCast(raw.width),
            .height = @intCast(raw.height),
            .pixel_format = .RGBA8,
            .num_mipmaps = @intCast(raw.num_levels),
        };
        for (0..raw.num_levels) |m| {
            if (raw.levels[m]) |level| {
                img_desc.data.mip_levels[m] = sg.asRange(level);
            }
        }

        const img = sg.makeImage(img_desc);
        const view = sg.makeView(.{
            .texture = .{ .image = img },
        });
        const smp = makeSamplerFor(options, raw.num_levels);

        return .{
            .image = img,
            .view = view,
            .sampler = smp,
            .width = raw.width,
            .height = raw.height,
        };
    }

    /// Decode-time switches for the CPU paths (GPU-free, thread-safe).
    /// srgb_to_linear converts RGB lanes before the mip chain is built.
    pub const DecodeOptions = struct {
        gen_mipmaps: bool = true,
        srgb_to_linear: bool = false,
    };

    /// Decodes an in-memory image (PNG/JPEG/KTX2/...) to RGBA without
    /// touching the GPU. Thread-safe; pair with `fromRaw`. KTX2 payloads
    /// (magic sniff) route to the ktx2 reader: only its uncompressed LDR
    /// subset decodes here — cube KTX2 files are rejected with
    /// error.UnsupportedFaceCount (use ktx2.decodeCube instead).
    pub fn decodeMemory(allocator: std.mem.Allocator, bytes: []const u8, opts: DecodeOptions) !RawTexture {
        if (ktx2.sniff(bytes)) {
            return ktx2.decode2D(allocator, bytes, .{
                .gen_mipmaps = opts.gen_mipmaps,
                // The caller's per-slot color/data decision is authoritative;
                // ktx2's format-tag auto detection applies only to its own
                // direct API (DecodeOptions.srgb_to_linear = null).
                .srgb_to_linear = opts.srgb_to_linear,
            });
        }

        var w: c_int = 0;
        var h: c_int = 0;
        var channels_in_file: c_int = 0;

        const data = c.stbi_load_from_memory(
            bytes.ptr,
            @intCast(bytes.len),
            &w,
            &h,
            &channels_in_file,
            4,
        );
        if (data == null) return error.ImageDecodeFailed;
        defer c.stbi_image_free(data);

        const width: u32 = @intCast(w);
        const height: u32 = @intCast(h);
        const size_bytes: usize = @as(usize, width) * @as(usize, height) * 4;
        // Convert before buildRaw so the box filter averages in linear space.
        if (opts.srgb_to_linear) convertSrgbToLinearInPlace(data[0..size_bytes]);
        var raw = try buildRaw(allocator, width, height, data[0..size_bytes], opts.gen_mipmaps);
        raw.is_srgb = opts.srgb_to_linear;
        return raw;
    }

    /// File variant of `decodeMemory`. Thread-safe; pair with `fromRaw`.
    /// The file is read into memory first so KTX2 files (magic sniff) take
    /// the same reader as in-memory payloads; everything else decodes via
    /// stb from the buffered bytes.
    pub fn decodeFile(allocator: std.mem.Allocator, file_path: []const u8, opts: DecodeOptions) !RawTexture {
        // Zig 0.16 removed std.fs.cwd(); read through the global
        // single-threaded Io (same pattern as CubeTexture.fromEquirectangularFile).
        const io = std.Io.Threaded.global_single_threaded.io();
        const file = try std.Io.Dir.cwd().openFile(io, file_path, .{});
        defer file.close(io);

        const file_size = try file.length(io);
        const bytes = try allocator.alloc(u8, std.math.cast(usize, file_size) orelse return error.ImageTooLarge);
        defer allocator.free(bytes);

        const read = try file.readPositionalAll(io, bytes, 0);
        if (read < bytes.len) return error.ImageDecodeFailed;
        return decodeMemory(allocator, bytes, opts);
    }

    pub fn createWhite1x1() Texture {
        const white = [_]u8{ 255, 255, 255, 255 };
        return initRaw(1, 1, &white, .{
            .min_filter = .NEAREST,
            .mag_filter = .NEAREST,
        });
    }

    pub fn createBlack1x1() Texture {
        const black = [_]u8{ 0, 0, 0, 255 };
        return initRaw(1, 1, &black, .{
            .min_filter = .NEAREST,
            .mag_filter = .NEAREST,
        });
    }

    pub fn createFlatNormal1x1() Texture {
        const flat_normal = [_]u8{ 128, 128, 255, 255 };
        return initRaw(1, 1, &flat_normal, .{
            .min_filter = .NEAREST,
            .mag_filter = .NEAREST,
        });
    }

    pub fn createCheckerboard(
        allocator: std.mem.Allocator,
        width: u32,
        height: u32,
        cell_size: u32,
        color1: [4]u8,
        color2: [4]u8,
    ) !Texture {
        // Checked arithmetic: on 64-bit, mul(usize, u32, u32) never fails,
        // so check u32 range explicitly to avoid a 16 GiB alloc attempt and
        // to turn the old `width * height` debug-mode panic into an error.
        const pixel_count = std.math.mul(u32, width, height) catch return error.ImageTooLarge;
        const byte_count = std.math.mul(u32, pixel_count, 4) catch return error.ImageTooLarge;
        const buffer = try allocator.alloc(u8, byte_count);
        defer allocator.free(buffer);

        var y: u32 = 0;
        while (y < height) : (y += 1) {
            var x: u32 = 0;
            while (x < width) : (x += 1) {
                const is_even = (((x / cell_size) + (y / cell_size)) % 2) == 0;
                const c_val = if (is_even) color1 else color2;
                const idx: usize = (@as(usize, y) * @as(usize, width) + @as(usize, x)) * 4;
                buffer[idx + 0] = c_val[0];
                buffer[idx + 1] = c_val[1];
                buffer[idx + 2] = c_val[2];
                buffer[idx + 3] = c_val[3];
            }
        }

        return initRaw(width, height, buffer, .{
            .min_filter = .NEAREST,
            .mag_filter = .NEAREST,
        });
    }

    pub fn createDefaultParticleDot32() Texture {
        var buf: [32 * 32 * 4]u8 = undefined;
        const size: usize = 32;
        const center: f32 = 15.5;
        const radius: f32 = 15.5;

        var y: usize = 0;
        while (y < size) : (y += 1) {
            var x: usize = 0;
            while (x < size) : (x += 1) {
                const alpha = particleDotAlpha(@intCast(x), @intCast(y), center, radius);

                const idx = (y * size + x) * 4;
                buf[idx + 0] = 255;
                buf[idx + 1] = 255;
                buf[idx + 2] = 255;
                buf[idx + 3] = alpha;
            }
        }

        return initRaw(32, 32, &buf, .{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });
    }

    pub fn createParticleDot(allocator: std.mem.Allocator, size: u32) !Texture {
        const byte_count = try checkedFaceBytes(size);
        const buffer = try allocator.alloc(u8, byte_count);
        defer allocator.free(buffer);

        const center: f32 = @as(f32, @floatFromInt(size)) * 0.5 - 0.5;
        const radius: f32 = @as(f32, @floatFromInt(size)) * 0.5;

        var y: u32 = 0;
        while (y < size) : (y += 1) {
            var x: u32 = 0;
            while (x < size) : (x += 1) {
                const alpha = particleDotAlpha(x, y, center, radius);

                const idx: usize = (@as(usize, y) * @as(usize, size) + @as(usize, x)) * 4;
                buffer[idx + 0] = 255;
                buffer[idx + 1] = 255;
                buffer[idx + 2] = 255;
                buffer[idx + 3] = alpha;
            }
        }

        return initRaw(size, size, buffer, .{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });
    }

    pub fn fromMemory(allocator: std.mem.Allocator, bytes: []const u8, options: Options) !Texture {
        var raw = try decodeMemory(allocator, bytes, .{
            .gen_mipmaps = options.mipmaps,
            .srgb_to_linear = options.srgb_to_linear,
        });
        defer raw.deinit(allocator);
        return fromRaw(&raw, options);
    }

    pub fn fromFile(allocator: std.mem.Allocator, file_path: []const u8, options: Options) !Texture {
        var raw = try decodeFile(allocator, file_path, .{
            .gen_mipmaps = options.mipmaps,
            .srgb_to_linear = options.srgb_to_linear,
        });
        defer raw.deinit(allocator);
        return fromRaw(&raw, options);
    }

    /// Converts one f32 channel to an IEEE-754 half-precision bit pattern.
    /// Out-of-range magnitudes become half infinity, NaN stays NaN.
    /// Content above 65504 loses detail: tone-map before upload if it matters.
    pub fn floatToHalfBits(value: f32) u16 {
        return @bitCast(@as(f16, @floatCast(value)));
    }

    /// Converts an IEEE-754 half-precision bit pattern back to f32.
    /// Used by tests and debugging; the GPU upload path never needs it.
    pub fn halfBitsToFloat(bits: u16) f32 {
        return @floatCast(@as(f16, @bitCast(bits)));
    }

    /// CPU-decoded HDR pixels in half-float RGBA. Decoding and the f32->f16
    /// conversion are GPU-free, so they can run on worker threads; the
    /// sg.Image must be created later with `fromRawHdr` on the main thread.
    /// Single mip level only: HDR images upload without mipmaps (linear
    /// min/mag, clamp wrap, no gamma correction).
    pub const RawHdrTexture = struct {
        width: u32 = 0,
        height: u32 = 0,
        /// RGBA half-float bit patterns, width*height*4 entries.
        pixels: []u16 = &.{},

        pub fn deinit(self: *RawHdrTexture, allocator: std.mem.Allocator) void {
            if (self.pixels.len > 0) allocator.free(self.pixels);
            self.* = .{};
        }
    };

    /// Copies f32 RGBA samples and converts them to half-float bits.
    /// Rejects empty dimensions and short sample buffers.
    fn buildRawHdr(allocator: std.mem.Allocator, width: u32, height: u32, samples: []const f32) !RawHdrTexture {
        const pixel_count = std.math.mul(usize, width, height) catch return error.ImageTooLarge;
        const channel_count = std.math.mul(usize, pixel_count, 4) catch return error.ImageTooLarge;
        if (channel_count == 0 or samples.len < channel_count) return error.InvalidDimensions;

        const pixels = try allocator.alloc(u16, channel_count);
        errdefer allocator.free(pixels);
        for (samples[0..channel_count], 0..) |sample, i| {
            pixels[i] = floatToHalfBits(sample);
        }
        return .{ .width = width, .height = height, .pixels = pixels };
    }

    /// Decodes an image to half-float RGBA via stbi_loadf: Radiance .hdr
    /// natively, LDR formats (PNG/JPEG/...) upconverted to float.
    /// Thread-safe; pair with `fromRawHdr`. Foreign or corrupt
    /// data yields error.ImageDecodeFailed, empty sizes error.InvalidDimensions.
    pub fn decodeHDRMemory(allocator: std.mem.Allocator, bytes: []const u8) !RawHdrTexture {
        if (bytes.len == 0) return error.ImageDecodeFailed;
        if (bytes.len > std.math.maxInt(c_int)) return error.ImageTooLarge;

        var w: c_int = 0;
        var h: c_int = 0;
        var channels_in_file: c_int = 0;

        const data = c.stbi_loadf_from_memory(
            bytes.ptr,
            @intCast(bytes.len),
            &w,
            &h,
            &channels_in_file,
            4,
        );
        if (data == null) return error.ImageDecodeFailed;
        defer c.stbi_image_free(data);
        if (w <= 0 or h <= 0) return error.InvalidDimensions;

        const width: u32 = @intCast(w);
        const height: u32 = @intCast(h);
        const channel_count = std.math.mul(usize, std.math.mul(usize, width, height) catch return error.ImageTooLarge, 4) catch return error.ImageTooLarge;
        // stb_image's flat Radiance path ignores stbi__getn's result and
        // zero-fills truncated pixel data, so short payloads are rejected
        // here: after the resolution line a flat file needs exactly
        // width*height RGBE pixels. RLE payloads are variable-length, but
        // stb errors on bad RLE runs, so they need no extra check.
        if (width < 8 or width >= 32768) {
            if (radiancePayload(bytes)) |payload| {
                if (payload.len < channel_count) return error.ImageDecodeFailed;
            }
        }
        return buildRawHdr(allocator, width, height, data[0..channel_count]);
    }

    /// Returns the pixel-data slice after the Radiance resolution line
    /// ("-Y h +X w"), or null when the header has no such line. The
    /// resolution line is the first line after the header's empty separator
    /// line; scanning beyond it could hit binary payload bytes.
    fn radiancePayload(bytes: []const u8) ?[]const u8 {
        var pos: usize = 0;
        while (pos < bytes.len) {
            const nl = std.mem.indexOfScalarPos(u8, bytes, pos, '\n') orelse return null;
            if (nl == pos) {
                pos = nl + 1;
                break;
            }
            pos = nl + 1;
        } else return null;
        const nl = std.mem.indexOfScalarPos(u8, bytes, pos, '\n') orelse return null;
        if (!std.mem.startsWith(u8, bytes[pos..nl], "-Y ")) return null;
        return bytes[nl + 1 ..];
    }

    /// File variant of `decodeHDRMemory`. Thread-safe; pair with `fromRawHdr`.
    pub fn decodeHDRFile(allocator: std.mem.Allocator, file_path: []const u8) !RawHdrTexture {
        const path_z = try allocator.dupeZ(u8, file_path);
        defer allocator.free(path_z);

        var w: c_int = 0;
        var h: c_int = 0;
        var channels_in_file: c_int = 0;

        const data = c.stbi_loadf(path_z.ptr, &w, &h, &channels_in_file, 4);
        if (data == null) return error.ImageDecodeFailed;
        defer c.stbi_image_free(data);
        if (w <= 0 or h <= 0) return error.InvalidDimensions;

        const width: u32 = @intCast(w);
        const height: u32 = @intCast(h);
        const channel_count = std.math.mul(usize, std.math.mul(usize, width, height) catch return error.ImageTooLarge, 4) catch return error.ImageTooLarge;
        return buildRawHdr(allocator, width, height, data[0..channel_count]);
    }

    /// Thin GPU uploader for RGBA half-float data. Main thread only.
    /// Precondition: rgba_f16.len == width*height*4.
    pub fn initRawHdr(width: u32, height: u32, rgba_f16: []const u16) Texture {
        var img_desc = sg.ImageDesc{
            .width = @intCast(width),
            .height = @intCast(height),
            .pixel_format = .RGBA16F,
            .num_mipmaps = 1,
        };
        img_desc.data.mip_levels[0] = sg.asRange(rgba_f16);

        const img = sg.makeImage(img_desc);
        const view = sg.makeView(.{
            .texture = .{ .image = img },
        });
        const smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .mipmap_filter = .NEAREST,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });

        return .{
            .image = img,
            .view = view,
            .sampler = smp,
            .width = width,
            .height = height,
            .format = .RGBA16F,
            .is_hdr = true,
        };
    }

    /// Creates the GPU image from CPU-decoded half-float pixels.
    /// Main thread only. Always uploads single-level RGBA16F with LINEAR
    /// min/mag and CLAMP_TO_EDGE: `options` is accepted for call symmetry
    /// but its filter/wrap/mipmap fields are ignored for HDR (no gamma
    /// correction, no mip chain).
    pub fn fromRawHdr(raw: *const RawHdrTexture, options: Options) Texture {
        _ = options;
        return initRawHdr(raw.width, raw.height, raw.pixels);
    }

    /// Decodes Radiance .hdr from memory and uploads an RGBA16F texture.
    /// Main thread only (GPU upload); use decodeHDRMemory + fromRawHdr to
    /// split worker-thread decode from main-thread upload.
    pub fn fromHDRMemory(allocator: std.mem.Allocator, bytes: []const u8, options: Options) !Texture {
        var raw = try decodeHDRMemory(allocator, bytes);
        defer raw.deinit(allocator);
        return fromRawHdr(&raw, options);
    }

    /// Loads a Radiance .hdr file and uploads an RGBA16F texture.
    /// Main thread only (GPU upload); use decodeHDRFile + fromRawHdr to
    /// split worker-thread decode from main-thread upload.
    pub fn loadHDRFile(allocator: std.mem.Allocator, file_path: []const u8, options: Options) !Texture {
        var raw = try decodeHDRFile(allocator, file_path);
        defer raw.deinit(allocator);
        return fromRawHdr(&raw, options);
    }

    pub fn deinit(self: *Texture) void {
        sg.destroyView(self.view);
        sg.destroyImage(self.image);
        sg.destroySampler(self.sampler);
    }
};

/// Gradient + sun parameters for `CubeTexture.createProceduralSkybox`.
pub const SkyboxOptions = struct {
    top_color: [4]u8 = .{ 35, 75, 155, 255 }, // Deep blue zenith
    horizon_color: [4]u8 = .{ 175, 195, 220, 255 }, // Atmospheric horizon
    bottom_color: [4]u8 = .{ 45, 42, 40, 255 }, // Ground nadir
    sun_direction: [3]f32 = .{ 0.4, 0.7, 0.5 }, // Matching scene light
    sun_color: [4]u8 = .{ 255, 250, 220, 255 },
    sun_intensity: f32 = 2.5,
    size: u32 = 128,
};

pub const CubeTexture = struct {
    image: sg.Image,
    view: sg.View,
    sampler: sg.Sampler,
    size: u32,
    num_mipmaps: u32 = 1,
    /// Pixel format of the GPU cube image. Defaults keep LDR call sites
    /// unchanged; HDR uploaders set RGBA16F.
    format: sg.PixelFormat = .RGBA8,
    /// True for float (HDR) cubes: no gamma correction, linear sampling.
    is_hdr: bool = false,

    pub fn deinit(self: *CubeTexture) void {
        sg.destroyView(self.view);
        sg.destroyImage(self.image);
        sg.destroySampler(self.sampler);
    }

    pub fn createDefault1x1(color: [4]u8) CubeTexture {
        var raw: [24]u8 = undefined;
        for (0..6) |face| {
            raw[face * 4 + 0] = color[0];
            raw[face * 4 + 1] = color[1];
            raw[face * 4 + 2] = color[2];
            raw[face * 4 + 3] = color[3];
        }

        var img_desc = sg.ImageDesc{
            .type = .CUBE,
            .width = 1,
            .height = 1,
            .num_slices = 6,
            .num_mipmaps = 1,
            .pixel_format = .RGBA8,
            .sample_count = 1,
        };
        img_desc.data.mip_levels[0] = sg.asRange(&raw);

        const img = sg.makeImage(img_desc);
        const view = sg.makeView(.{
            .texture = .{ .image = img },
        });
        const smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
            .wrap_w = .CLAMP_TO_EDGE,
        });

        return .{
            .image = img,
            .view = view,
            .sampler = smp,
            .size = 1,
            .num_mipmaps = 1,
        };
    }

    pub fn initRawFaces(allocator: std.mem.Allocator, size: u32, faces: [6][]const u8, generate_mips: bool) !CubeTexture {
        const face_bytes = try checkedFaceBytes(size);
        for (faces) |face| {
            if (face.len != face_bytes) return error.InvalidFaceBufferSize;
        }

        // Concatenate 6 faces for mip 0
        const mip0_total_bytes = std.math.mul(usize, face_bytes, 6) catch return error.ImageTooLarge;
        const mip0_buffer = try allocator.alloc(u8, mip0_total_bytes);
        defer allocator.free(mip0_buffer);

        for (faces, 0..) |face, i| {
            const dst_offset = i * face_bytes;
            @memcpy(mip0_buffer[dst_offset .. dst_offset + face_bytes], face);
        }

        var num_mips: u32 = 1;
        if (generate_mips and size > 1) {
            // Same chain length as the 2D path; identical to the old
            // divide-by-2 loop for powers of two, NPOT-safe otherwise.
            num_mips = Texture.mipLevelCount(size, size);
        }
        if (num_mips > 16) num_mips = 16;

        var img_desc = sg.ImageDesc{
            .type = .CUBE,
            .width = @intCast(size),
            .height = @intCast(size),
            .num_slices = 6,
            .num_mipmaps = @intCast(num_mips),
            .pixel_format = .RGBA8,
            .sample_count = 1,
        };
        img_desc.data.mip_levels[0] = sg.asRange(mip0_buffer);

        // Generate downsampled mipmaps
        var mip_buffers: [16]?[]u8 = @splat(null);
        defer {
            for (1..num_mips) |m| {
                if (mip_buffers[m]) |buf| allocator.free(buf);
            }
        }

        var prev_size = size;
        var prev_buf = mip0_buffer;
        for (1..num_mips) |m| {
            // @max(1, ...) keeps NPOT chains alive (e.g. 3 -> 1) instead of
            // hitting a zero-sized level; POT chains are unchanged.
            const cur_size: u32 = @max(1, prev_size / 2);
            const cur_face_checked = try checkedFaceBytes(cur_size);
            const cur_total_bytes = std.math.mul(usize, cur_face_checked, 6) catch return error.ImageTooLarge;
            const cur_buf = try allocator.alloc(u8, cur_total_bytes);
            mip_buffers[m] = cur_buf;

            const prev_face_bytes: usize = @as(usize, prev_size) * @as(usize, prev_size) * 4;
            const cur_face_bytes: usize = cur_face_checked;
            for (0..6) |face| {
                const src_face = prev_buf[face * prev_face_bytes .. (face + 1) * prev_face_bytes];
                const dst_face = cur_buf[face * cur_face_bytes .. (face + 1) * cur_face_bytes];
                // Shared NPOT-safe box filter; bit-identical to the old
                // inline loop when prev == cur*2 (all POT levels).
                boxDownsampleU8(src_face, prev_size, prev_size, dst_face, cur_size, cur_size);
            }

            img_desc.data.mip_levels[m] = sg.asRange(cur_buf);
            prev_size = cur_size;
            prev_buf = cur_buf;
        }

        const img = sg.makeImage(img_desc);
        const view = sg.makeView(.{
            .texture = .{ .image = img },
        });
        const smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .mipmap_filter = if (num_mips > 1) .LINEAR else .NEAREST,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
            .wrap_w = .CLAMP_TO_EDGE,
        });

        return .{
            .image = img,
            .view = view,
            .sampler = smp,
            .size = size,
            .num_mipmaps = num_mips,
        };
    }

    pub fn createProceduralSkybox(allocator: std.mem.Allocator, config: SkyboxOptions) !CubeTexture {
        const size = config.size;
        const face_bytes = try checkedFaceBytes(size);

        var face_slices: [6][]u8 = undefined;
        var allocated_faces: usize = 0;
        defer for (face_slices[0..allocated_faces]) |s| allocator.free(s);
        for (0..6) |i| {
            face_slices[i] = try allocator.alloc(u8, face_bytes);
            allocated_faces += 1;
        }

        // Normalize sun direction
        const s_len = @sqrt(config.sun_direction[0] * config.sun_direction[0] +
            config.sun_direction[1] * config.sun_direction[1] +
            config.sun_direction[2] * config.sun_direction[2]);
        const sun_dir: [3]f32 = if (s_len > 0.0001) .{
            config.sun_direction[0] / s_len,
            config.sun_direction[1] / s_len,
            config.sun_direction[2] / s_len,
        } else .{ 0.0, 1.0, 0.0 };

        for (0..6) |face| {
            const buf = face_slices[face];
            var py: u32 = 0;
            while (py < size) : (py += 1) {
                const v = 2.0 * (@as(f32, @floatFromInt(py)) + 0.5) / @as(f32, @floatFromInt(size)) - 1.0;
                var px: u32 = 0;
                while (px < size) : (px += 1) {
                    const u = 2.0 * (@as(f32, @floatFromInt(px)) + 0.5) / @as(f32, @floatFromInt(size)) - 1.0;

                    const dir: [3]f32 = switch (face) {
                        0 => .{ 1.0, -v, -u }, // +X
                        1 => .{ -1.0, -v, u }, // -X
                        2 => .{ u, 1.0, v }, // +Y
                        3 => .{ u, -1.0, -v }, // -Y
                        4 => .{ u, -v, 1.0 }, // +Z
                        5 => .{ -u, -v, -1.0 }, // -Z
                        else => unreachable,
                    };

                    const d_len = @sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
                    const d: [3]f32 = .{ dir[0] / d_len, dir[1] / d_len, dir[2] / d_len };

                    // Elevation gradient (d[1] is Y)
                    var color: [3]f32 = undefined;
                    if (d[1] >= 0.0) {
                        const t = std.math.pow(f32, d[1], 0.65);
                        for (0..3) |c_idx| {
                            const horiz = @as(f32, @floatFromInt(config.horizon_color[c_idx])) / 255.0;
                            const top = @as(f32, @floatFromInt(config.top_color[c_idx])) / 255.0;
                            color[c_idx] = (1.0 - t) * horiz + t * top;
                        }
                    } else {
                        const t = std.math.pow(f32, -d[1], 0.7);
                        for (0..3) |c_idx| {
                            const horiz = @as(f32, @floatFromInt(config.horizon_color[c_idx])) / 255.0;
                            const bot = @as(f32, @floatFromInt(config.bottom_color[c_idx])) / 255.0;
                            color[c_idx] = (1.0 - t) * horiz + t * bot;
                        }
                    }

                    // Sun contribution
                    const sun_dot = d[0] * sun_dir[0] + d[1] * sun_dir[1] + d[2] * sun_dir[2];
                    if (sun_dot > 0.0) {
                        const sun_disk = std.math.pow(f32, sun_dot, 512.0) * 3.5;
                        const sun_halo = std.math.pow(f32, sun_dot, 16.0) * 0.45;
                        const total_sun = (sun_disk + sun_halo) * config.sun_intensity;

                        for (0..3) |c_idx| {
                            const sc = @as(f32, @floatFromInt(config.sun_color[c_idx])) / 255.0;
                            color[c_idx] += total_sun * sc;
                        }
                    }

                    const out_idx: usize = (@as(usize, py) * @as(usize, size) + @as(usize, px)) * 4;
                    inline for (0..3) |c_idx| {
                        const byte_val = std.math.clamp(color[c_idx] * 255.0, 0.0, 255.0);
                        buf[out_idx + c_idx] = @intFromFloat(byte_val);
                    }
                    buf[out_idx + 3] = 255;
                }
            }
        }

        const const_faces = [6][]const u8{
            face_slices[0],
            face_slices[1],
            face_slices[2],
            face_slices[3],
            face_slices[4],
            face_slices[5],
        };

        return initRawFaces(allocator, size, const_faces, true);
    }

    pub fn fromFiles(allocator: std.mem.Allocator, face_paths: [6][]const u8) !CubeTexture {
        var face_data: [6][*c]u8 = undefined;
        // Single ownership scheme: `loaded` counts fully validated faces.
        // The defer frees exactly those on every exit — error returns after
        // a `try` (e.g. dupeZ OOM) or decode failure, and the success path
        // after initRawFaces has copied the pixels for GPU upload.
        var loaded: usize = 0;
        defer for (face_data[0..loaded]) |d| c.stbi_image_free(d);
        var size: u32 = 0;

        for (face_paths, 0..) |path, i| {
            const path_z = try allocator.dupeZ(u8, path);
            defer allocator.free(path_z);

            var w: c_int = 0;
            var h: c_int = 0;
            var comp: c_int = 0;
            const data = c.stbi_load(path_z.ptr, &w, &h, &comp, 4);
            if (data == null) {
                return error.ImageDecodeFailed;
            }
            if (w <= 0 or h <= 0) {
                c.stbi_image_free(data);
                return error.InvalidDimensions;
            }
            if (w != h) {
                c.stbi_image_free(data);
                return error.CubeFaceMustBeSquare;
            }
            if (i == 0) {
                size = @intCast(w);
            } else if (@as(u32, @intCast(w)) != size) {
                c.stbi_image_free(data);
                return error.CubeFacesMustHaveEqualSize;
            }
            face_data[i] = data;
            loaded += 1;
        }

        const face_bytes = try checkedFaceBytes(size);
        var const_faces: [6][]const u8 = undefined;
        for (0..6) |i| {
            const_faces[i] = face_data[i][0..face_bytes];
        }

        return initRawFaces(allocator, size, const_faces, true);
    }

    pub fn fromEquirectangular(allocator: std.mem.Allocator, panorama_bytes: []const u8, face_size: u32) !CubeTexture {
        // Same guard as the HDR converter: zero face size would divide by
        // zero in cubeTexelDirection and produce zero-byte faces.
        if (face_size == 0) return error.InvalidDimensions;
        var pw: c_int = 0;
        var ph: c_int = 0;
        var comp: c_int = 0;
        const p_data = c.stbi_load_from_memory(
            panorama_bytes.ptr,
            @intCast(panorama_bytes.len),
            &pw,
            &ph,
            &comp,
            4,
        );
        if (p_data == null) return error.ImageDecodeFailed;
        defer c.stbi_image_free(p_data);
        if (pw <= 0 or ph <= 0) return error.InvalidDimensions;

        const pano_w: u32 = @intCast(pw);
        const pano_h: u32 = @intCast(ph);
        const face_bytes = try checkedFaceBytes(face_size);

        var face_slices: [6][]u8 = undefined;
        var allocated_faces: usize = 0;
        defer for (face_slices[0..allocated_faces]) |s| allocator.free(s);
        for (0..6) |i| {
            face_slices[i] = try allocator.alloc(u8, face_bytes);
            allocated_faces += 1;
        }

        for (0..6) |face| {
            const buf = face_slices[face];
            var py: u32 = 0;
            while (py < face_size) : (py += 1) {
                var px: u32 = 0;
                while (px < face_size) : (px += 1) {
                    const d = cubeTexelDirection(face, px, py, face_size);
                    const t = panoramaTexel(d, pano_w, pano_h);
                    const p_idx: usize = (@as(usize, t.y) * @as(usize, pano_w) + @as(usize, t.x)) * 4;
                    const out_idx: usize = (@as(usize, py) * @as(usize, face_size) + @as(usize, px)) * 4;

                    @memcpy(buf[out_idx .. out_idx + 4], p_data[p_idx .. p_idx + 4]);
                }
            }
        }

        const const_faces = [6][]const u8{
            face_slices[0],
            face_slices[1],
            face_slices[2],
            face_slices[3],
            face_slices[4],
            face_slices[5],
        };

        return initRawFaces(allocator, face_size, const_faces, true);
    }

    pub fn fromEquirectangularFile(allocator: std.mem.Allocator, file_path: []const u8, face_size: u32) !CubeTexture {
        if (face_size == 0) return error.InvalidDimensions;
        // Zig 0.16 removed std.fs.cwd(); read through the global single-threaded
        // Io so the public (allocator, path) signature stays unchanged.
        const io = std.Io.Threaded.global_single_threaded.io();
        const file = try std.Io.Dir.cwd().openFile(io, file_path, .{});
        defer file.close(io);

        const file_size = try file.length(io);
        const bytes = try allocator.alloc(u8, std.math.cast(usize, file_size) orelse return error.ImageTooLarge);
        defer allocator.free(bytes);

        const read = try file.readPositionalAll(io, bytes, 0);
        if (read < bytes.len) return error.ImageDecodeFailed;
        return fromEquirectangular(allocator, bytes, face_size);
    }

    /// Normalized cube-face direction for texel (px, py). Shared by the LDR
    /// and HDR equirectangular converters so both sample the same texels.
    fn cubeTexelDirection(face: usize, px: u32, py: u32, size: u32) [3]f32 {
        const size_f: f32 = @floatFromInt(size);
        const u = 2.0 * (@as(f32, @floatFromInt(px)) + 0.5) / size_f - 1.0;
        const v = 2.0 * (@as(f32, @floatFromInt(py)) + 0.5) / size_f - 1.0;

        const dir: [3]f32 = switch (face) {
            0 => .{ 1.0, -v, -u }, // +X
            1 => .{ -1.0, -v, u }, // -X
            2 => .{ u, 1.0, v }, // +Y
            3 => .{ u, -1.0, -v }, // -Y
            4 => .{ u, -v, 1.0 }, // +Z
            5 => .{ -u, -v, -1.0 }, // -Z
            else => unreachable,
        };

        const len = @sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
        return .{ dir[0] / len, dir[1] / len, dir[2] / len };
    }

    /// Nearest panorama texel for a normalized direction (spherical mapping).
    /// Shared by the LDR and HDR equirectangular converters.
    fn panoramaTexel(d: [3]f32, pano_w: u32, pano_h: u32) struct { x: u32, y: u32 } {
        // Spherical coordinates
        const phi = std.math.atan2(d[2], d[0]); // longitude [-pi, pi]
        const theta = std.math.asin(std.math.clamp(d[1], -1.0, 1.0)); // latitude [-pi/2, pi/2]

        const u_pano = (phi / (2.0 * std.math.pi)) + 0.5;
        const v_pano = 0.5 - (theta / std.math.pi);

        const sx_f = std.math.clamp(u_pano * @as(f32, @floatFromInt(pano_w)), 0.0, @as(f32, @floatFromInt(pano_w - 1)));
        const sy_f = std.math.clamp(v_pano * @as(f32, @floatFromInt(pano_h)), 0.0, @as(f32, @floatFromInt(pano_h - 1)));

        return .{ .x = @intFromFloat(sx_f), .y = @intFromFloat(sy_f) };
    }

    /// GPU-free half-float cube faces converted from an HDR panorama.
    /// Owns six size*size*4 half-float bit patterns; free with deinit.
    /// Single mip level: `initRawFacesHdr` builds the f16 chain from this
    /// level-0 data via `buildRawFacesHdr`.
    pub const RawHdrCube = struct {
        size: u32 = 0,
        faces: [6]?[]u16 = @splat(null),

        pub fn deinit(self: *RawHdrCube, allocator: std.mem.Allocator) void {
            for (&self.faces) |*face| {
                if (face.*) |buf| allocator.free(buf);
                face.* = null;
            }
            self.size = 0;
        }
    };

    /// Converts an f32 RGBA equirectangular panorama to six half-float cube
    /// faces using the same texel math as `fromEquirectangular`. GPU-free and
    /// safe for worker threads; pair with `initRawFacesHdr`. Nearest sampling,
    /// no gamma correction. Rejects zero dimensions and short buffers.
    pub fn convertEquirectangularHDR(
        allocator: std.mem.Allocator,
        width: u32,
        height: u32,
        rgba: []const f32,
        size: u32,
    ) !RawHdrCube {
        if (size == 0 or width == 0 or height == 0) return error.InvalidDimensions;
        const pano_pixels = std.math.mul(usize, width, height) catch return error.ImageTooLarge;
        const pano_channels = std.math.mul(usize, pano_pixels, 4) catch return error.ImageTooLarge;
        if (rgba.len != pano_channels) return error.InvalidPanoramaSize;

        const face_pixels = std.math.mul(usize, std.math.mul(usize, size, size) catch return error.ImageTooLarge, 4) catch return error.ImageTooLarge;

        var raw = RawHdrCube{ .size = size };
        errdefer raw.deinit(allocator);

        for (0..6) |face| {
            const out = try allocator.alloc(u16, face_pixels);
            raw.faces[face] = out;
            var py: u32 = 0;
            while (py < size) : (py += 1) {
                var px: u32 = 0;
                while (px < size) : (px += 1) {
                    const d = cubeTexelDirection(face, px, py, size);
                    const t = panoramaTexel(d, width, height);
                    const p_idx: usize = (@as(usize, t.y) * @as(usize, width) + @as(usize, t.x)) * 4;
                    const out_idx: usize = (@as(usize, py) * @as(usize, size) + @as(usize, px)) * 4;
                    for (0..4) |ch| {
                        out[out_idx + ch] = Texture.floatToHalfBits(rgba[p_idx + ch]);
                    }
                }
            }
        }
        return raw;
    }

    /// One RGBA8-style box-filter mip step over RGBA16F (half-float) texels.
    /// Averages in f32 (decode via halfBitsToFloat, encode via
    /// floatToHalfBits); dims floor at 1, source coords clamp at edges —
    /// same contract as `boxDownsampleU8`.
    fn boxDownsampleF16(src: []const u16, src_w: u32, src_h: u32, dst: []u16, dst_w: u32, dst_h: u32) void {
        var y: u32 = 0;
        while (y < dst_h) : (y += 1) {
            const sy0 = @min(y * 2, src_h - 1);
            const sy1 = @min(y * 2 + 1, src_h - 1);
            var x: u32 = 0;
            while (x < dst_w) : (x += 1) {
                const sx0 = @min(x * 2, src_w - 1);
                const sx1 = @min(x * 2 + 1, src_w - 1);
                const q00 = (sy0 * src_w + sx0) * 4;
                const q10 = (sy0 * src_w + sx1) * 4;
                const q01 = (sy1 * src_w + sx0) * 4;
                const q11 = (sy1 * src_w + sx1) * 4;
                const o = (y * dst_w + x) * 4;
                inline for (0..4) |ch| {
                    const sum = Texture.halfBitsToFloat(src[q00 + ch]) +
                        Texture.halfBitsToFloat(src[q10 + ch]) +
                        Texture.halfBitsToFloat(src[q01 + ch]) +
                        Texture.halfBitsToFloat(src[q11 + ch]);
                    dst[o + ch] = Texture.floatToHalfBits(sum * 0.25);
                }
            }
        }
    }

    /// GPU-free f16 mip chain for a cube (six faces concatenated per level,
    /// face layout identical to the LDR path). Level 0 is a deep copy of
    /// `faces`; each following level halves the face size via
    /// `boxDownsampleF16`. Same chain length as the 2D path. Used by
    /// `initRawFacesHdr` and tests.
    pub const RawHdrCubeMips = struct {
        size: u32 = 0,
        num_levels: u32 = 0,
        /// levels[m] holds 6 * size_m * size_m * 4 half-float patterns.
        levels: [16]?[]u16 = @splat(null),

        pub fn deinit(self: *RawHdrCubeMips, allocator: std.mem.Allocator) void {
            for (self.levels[0..self.num_levels]) |level| {
                if (level) |buf| allocator.free(buf);
            }
            self.* = .{};
        }
    };

    pub fn buildRawFacesHdr(allocator: std.mem.Allocator, size: u32, faces: [6][]const u16) !RawHdrCubeMips {
        if (size == 0) return error.InvalidDimensions;
        const face_texels = std.math.mul(usize, std.math.mul(usize, size, size) catch return error.ImageTooLarge, 4) catch return error.ImageTooLarge;
        for (faces) |face| {
            if (face.len != face_texels) return error.InvalidFaceBufferSize;
        }

        var chain = RawHdrCubeMips{ .size = size, .num_levels = 1 };
        errdefer chain.deinit(allocator);

        const mip0 = try allocator.alloc(u16, std.math.mul(usize, 6, face_texels) catch return error.ImageTooLarge);
        for (faces, 0..) |face, i| {
            @memcpy(mip0[i * face_texels .. (i + 1) * face_texels], face);
        }
        chain.levels[0] = mip0;

        var prev_size = size;
        const levels = Texture.mipLevelCount(size, size);
        for (1..levels) |m| {
            const cur_size: u32 = @max(1, prev_size / 2);
            const prev_texels: usize = @as(usize, prev_size) * prev_size * 4;
            const cur_texels: usize = @as(usize, cur_size) * cur_size * 4;
            const cur = try allocator.alloc(u16, std.math.mul(usize, 6, cur_texels) catch return error.ImageTooLarge);
            chain.levels[m] = cur;
            chain.num_levels = @intCast(m + 1);

            const prev = chain.levels[m - 1].?;
            for (0..6) |face| {
                boxDownsampleF16(
                    prev[face * prev_texels .. (face + 1) * prev_texels],
                    prev_size,
                    prev_size,
                    cur[face * cur_texels .. (face + 1) * cur_texels],
                    cur_size,
                    cur_size,
                );
            }
            prev_size = cur_size;
        }
        return chain;
    }

    /// Uploads six half-float RGBA faces as an RGBA16F cube WITH a full
    /// CPU-generated mip chain (the PBR IBL path samples
    /// `textureLod(lod <= 7)`, which needs the levels). Main thread only.
    /// Each face must hold size*size*4 half patterns.
    /// Linear min/mag/mip, clamp on all axes, no gamma correction.
    pub fn initRawFacesHdr(allocator: std.mem.Allocator, size: u32, faces: [6][]const u16) !CubeTexture {
        var chain = try buildRawFacesHdr(allocator, size, faces);
        defer chain.deinit(allocator);

        var img_desc = sg.ImageDesc{
            .type = .CUBE,
            .width = @intCast(size),
            .height = @intCast(size),
            .num_slices = 6,
            .num_mipmaps = @intCast(chain.num_levels),
            .pixel_format = .RGBA16F,
            .sample_count = 1,
        };
        for (0..chain.num_levels) |m| {
            if (chain.levels[m]) |level| {
                img_desc.data.mip_levels[m] = sg.asRange(level);
            }
        }

        const img = sg.makeImage(img_desc);
        const view = sg.makeView(.{
            .texture = .{ .image = img },
        });
        const smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .mipmap_filter = if (chain.num_levels > 1) .LINEAR else .NEAREST,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
            .wrap_w = .CLAMP_TO_EDGE,
        });

        return .{
            .image = img,
            .view = view,
            .sampler = smp,
            .size = size,
            .num_mipmaps = chain.num_levels,
            .format = .RGBA16F,
            .is_hdr = true,
        };
    }

    /// Converts an in-memory f32 RGBA equirectangular panorama to an RGBA16F
    /// cube for IBL/skybox. Main thread only (GPU upload); use
    /// convertEquirectangularHDR alone to stay GPU-free (worker threads).
    pub fn fromEquirectangularHDR(
        allocator: std.mem.Allocator,
        width: u32,
        height: u32,
        rgba: []const f32,
        size: u32,
    ) !CubeTexture {
        var raw = try convertEquirectangularHDR(allocator, width, height, rgba, size);
        defer raw.deinit(allocator);
        var faces: [6][]const u16 = undefined;
        for (0..6) |i| faces[i] = raw.faces[i].?;
        return initRawFacesHdr(allocator, size, faces);
    }

    /// GPU-free pre-built LDR cube levels (six faces concatenated per level,
    /// face order +X,-X,+Y,-Y,+Z,-Z, level 0 largest). Produced by
    /// ktx2.decodeCube; uploaded with `initRawFacesMips`. Free with deinit.
    pub const RawCubeMips = struct {
        size: u32 = 0,
        num_levels: u32 = 0,
        /// levels[m] holds 6 * size_m * size_m * 4 RGBA8 bytes.
        levels: [16]?[]u8 = @splat(null),

        pub fn deinit(self: *RawCubeMips, allocator: std.mem.Allocator) void {
            for (self.levels[0..self.num_levels]) |level| {
                if (level) |buf| allocator.free(buf);
            }
            self.* = .{};
        }
    };

    /// Uploads pre-built LDR cube levels as an RGBA8 cube WITH a full mip
    /// chain as authored (unlike `initRawFaces`, which generates the chain
    /// from level 0). Main thread only. Sampler: linear min/mag/mip, clamp
    /// on all axes — the env/skybox convention of every cube path.
    pub fn initRawFacesMips(raw: *const RawCubeMips) CubeTexture {
        var img_desc = sg.ImageDesc{
            .type = .CUBE,
            .width = @intCast(raw.size),
            .height = @intCast(raw.size),
            .num_slices = 6,
            .num_mipmaps = @intCast(raw.num_levels),
            .pixel_format = .RGBA8,
            .sample_count = 1,
        };
        for (0..raw.num_levels) |m| {
            if (raw.levels[m]) |level| {
                img_desc.data.mip_levels[m] = sg.asRange(level);
            }
        }

        const img = sg.makeImage(img_desc);
        const view = sg.makeView(.{
            .texture = .{ .image = img },
        });
        const smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .mipmap_filter = if (raw.num_levels > 1) .LINEAR else .NEAREST,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
            .wrap_w = .CLAMP_TO_EDGE,
        });

        return .{
            .image = img,
            .view = view,
            .sampler = smp,
            .size = raw.size,
            .num_mipmaps = raw.num_levels,
        };
    }
};

test "mipLevelCount covers powers of two and minimums" {
    try std.testing.expectEqual(@as(u32, 1), Texture.mipLevelCount(1, 1));
    try std.testing.expectEqual(@as(u32, 2), Texture.mipLevelCount(2, 2));
    try std.testing.expectEqual(@as(u32, 10), Texture.mipLevelCount(512, 512));
    try std.testing.expectEqual(@as(u32, 3), Texture.mipLevelCount(5, 3));
}

test "downsampleLevel handles odd dimensions and averages correctly" {
    var src: [3 * 5 * 4]u8 = undefined;
    for (&src, 0..) |*b, i| b.* = @intCast(i % 251);
    var dst: [1 * 2 * 4]u8 = undefined;
    Texture.downsampleLevel(&src, 3, 5, &dst, 1, 2);
    // Top-left texel averages src bytes {0,4,12,16}: (0+4+12+16+2)/4 = 8.
    try std.testing.expectEqual(@as(u8, 8), dst[0]);
}

test "decodeMemory returns RGBA levels and owns its mip chain" {
    const png = @embedFile("assets/font_sdf.png");
    const allocator = std.testing.allocator;

    // decodeMemory logs decode timings through sokol.time; sokol_time is
    // CPU-only and needs its one-time setup (normally done at app startup).
    sokol.time.setup();

    var single = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = false });
    defer single.deinit(allocator);
    try std.testing.expectEqual(@as(u32, 512), single.width);
    try std.testing.expectEqual(@as(u32, 512), single.height);
    try std.testing.expectEqual(@as(u32, 1), single.num_levels);
    try std.testing.expectEqual(@as(usize, 512 * 512 * 4), single.levels[0].?.len);

    var mipped = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = true });
    defer mipped.deinit(allocator);
    try std.testing.expectEqual(Texture.mipLevelCount(512, 512), mipped.num_levels);
    try std.testing.expectEqual(@as(usize, 4), mipped.levels[mipped.num_levels - 1].?.len);
}

// Minimal 1x1 Radiance .hdr with flat (non-RLE) RGBE data. Width < 8 takes
// the flat decode path in stb_image. RGBE {128,128,128,128} decodes to
// 128 * 2^(128-136) = 0.5 per channel, alpha forced to 1.0.
const hdr_1x1_flat: []const u8 = "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n\x80\x80\x80\x80";

// Minimal 8x1 Radiance .hdr exercising the RLE scanline path: scanline
// header {2, 2, 0, 8} (width 8), then each of the 4 channels as one run of
// 8 pixels with value 128 (count byte 128+8 = 0x88). Same 0.5 gray texels.
const hdr_8x1_rle: []const u8 = "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 8\n\x02\x02\x00\x08\x88\x80\x88\x80\x88\x80\x88\x80";

test "decodeHDRMemory decodes minimal flat Radiance .hdr" {
    const allocator = std.testing.allocator;

    var raw = try Texture.decodeHDRMemory(allocator, hdr_1x1_flat);
    defer raw.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 1), raw.width);
    try std.testing.expectEqual(@as(u32, 1), raw.height);
    try std.testing.expectEqual(@as(usize, 4), raw.pixels.len);
    // Exact half patterns: 0.5 -> 0x3800, 1.0 -> 0x3C00.
    try std.testing.expectEqual(@as(u16, 0x3800), raw.pixels[0]);
    try std.testing.expectEqual(@as(u16, 0x3800), raw.pixels[1]);
    try std.testing.expectEqual(@as(u16, 0x3800), raw.pixels[2]);
    try std.testing.expectEqual(@as(u16, 0x3C00), raw.pixels[3]);
}

test "decodeHDRMemory decodes RLE Radiance scanlines" {
    const allocator = std.testing.allocator;

    var raw = try Texture.decodeHDRMemory(allocator, hdr_8x1_rle);
    defer raw.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 8), raw.width);
    try std.testing.expectEqual(@as(u32, 1), raw.height);
    try std.testing.expectEqual(@as(usize, 8 * 1 * 4), raw.pixels.len);
    for (0..8) |i| {
        try std.testing.expectEqual(@as(u16, 0x3800), raw.pixels[i * 4 + 0]);
        try std.testing.expectEqual(@as(u16, 0x3800), raw.pixels[i * 4 + 1]);
        try std.testing.expectEqual(@as(u16, 0x3800), raw.pixels[i * 4 + 2]);
        try std.testing.expectEqual(@as(u16, 0x3C00), raw.pixels[i * 4 + 3]);
    }
}

test "decodeHDRMemory rejects foreign, truncated and empty data" {
    const allocator = std.testing.allocator;

    try std.testing.expectError(
        error.ImageDecodeFailed,
        Texture.decodeHDRMemory(allocator, "hello world, this is not an image"),
    );
    try std.testing.expectError(
        error.ImageDecodeFailed,
        Texture.decodeHDRMemory(allocator, "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n"),
    );
    try std.testing.expectError(
        error.ImageDecodeFailed,
        Texture.decodeHDRMemory(allocator, hdr_1x1_flat[0 .. hdr_1x1_flat.len - 2]),
    );
    const empty: []const u8 = &[_]u8{};
    try std.testing.expectError(
        error.ImageDecodeFailed,
        Texture.decodeHDRMemory(allocator, empty),
    );
}

test "floatToHalfBits covers exact values, overflow and NaN" {
    try std.testing.expectEqual(@as(u16, 0x3C00), Texture.floatToHalfBits(1.0));
    try std.testing.expectEqual(@as(u16, 0x3800), Texture.floatToHalfBits(0.5));
    try std.testing.expectEqual(@as(u16, 0x0000), Texture.floatToHalfBits(0.0));
    try std.testing.expectEqual(@as(f32, 1.0), Texture.halfBitsToFloat(0x3C00));

    // Beyond f16 max (65504): documents the half-float precision ceiling.
    try std.testing.expect(std.math.isInf(Texture.halfBitsToFloat(Texture.floatToHalfBits(1.0e10))));
    try std.testing.expect(std.math.isNan(Texture.halfBitsToFloat(Texture.floatToHalfBits(std.math.nan(f32)))));
}

test "convertEquirectangularHDR maps gradient faces without NaN" {
    const allocator = std.testing.allocator;
    const pano_w: u32 = 4;
    const pano_h: u32 = 2;
    const face_size: u32 = 4;

    var pano: [4 * 2 * 4]f32 = undefined;
    for (0..pano_h) |y| {
        for (0..pano_w) |x| {
            const o = (y * pano_w + x) * 4;
            pano[o + 0] = @as(f32, @floatFromInt(x)) / @as(f32, @floatFromInt(pano_w - 1));
            pano[o + 1] = @as(f32, @floatFromInt(y)) / @as(f32, @floatFromInt(pano_h - 1));
            pano[o + 2] = 0.5;
            pano[o + 3] = 1.0;
        }
    }

    var cube = try CubeTexture.convertEquirectangularHDR(allocator, pano_w, pano_h, &pano, face_size);
    defer cube.deinit(allocator);

    try std.testing.expectEqual(face_size, cube.size);
    for (cube.faces) |maybe_face| {
        const face = maybe_face.?;
        try std.testing.expectEqual(@as(usize, face_size * face_size * 4), face.len);
        for (face) |bits| {
            const v = Texture.halfBitsToFloat(bits);
            try std.testing.expect(!std.math.isNan(v));
            try std.testing.expect(std.math.isFinite(v));
        }
    }
}

test "convertEquirectangularHDR validates dimensions and buffer size" {
    const allocator = std.testing.allocator;
    var pano: [2 * 1 * 4]f32 = [_]f32{0.0} ** (2 * 1 * 4);

    try std.testing.expectError(
        error.InvalidDimensions,
        CubeTexture.convertEquirectangularHDR(allocator, 0, 1, &pano, 2),
    );
    try std.testing.expectError(
        error.InvalidDimensions,
        CubeTexture.convertEquirectangularHDR(allocator, 2, 1, &pano, 0),
    );
    try std.testing.expectError(
        error.InvalidPanoramaSize,
        CubeTexture.convertEquirectangularHDR(allocator, 2, 1, pano[0..4], 2),
    );
}

test "decodeHDRFile reports missing files without leaking" {
    const allocator = std.testing.allocator;

    try std.testing.expectError(
        error.ImageDecodeFailed,
        Texture.decodeHDRFile(allocator, "definitely/missing/file.hdr"),
    );
}

test "fromEquirectangularFile validates input and missing files" {
    const allocator = std.testing.allocator;

    // face_size = 0 fails before any file IO.
    try std.testing.expectError(
        error.InvalidDimensions,
        CubeTexture.fromEquirectangularFile(allocator, "definitely/missing/file.png", 0),
    );
    // Missing file must fail cleanly (this also keeps the std.Io read path
    // analyzed, so future Zig API churn is caught by `zig build test`).
    try std.testing.expectError(
        error.FileNotFound,
        CubeTexture.fromEquirectangularFile(allocator, "definitely/missing/file.png", 4),
    );
}

test "boxDownsampleU8 handles NPOT cube-face steps without OOB" {
    // 3x3 -> 1x1 is the exact mip step of a size-3 NPOT cube face
    // (cur = max(1, prev / 2)). R channel holds the texel index.
    var src: [3 * 3 * 4]u8 = undefined;
    for (0..9) |i| {
        src[i * 4 + 0] = @intCast(i);
        src[i * 4 + 1] = 0;
        src[i * 4 + 2] = 0;
        src[i * 4 + 3] = 255;
    }
    var dst: [1 * 1 * 4]u8 = undefined;
    boxDownsampleU8(&src, 3, 3, &dst, 1, 1);
    // Averages top-left quad {0,1,3,4}: (0+1+3+4+2)>>2 = 2.
    try std.testing.expectEqual(@as(u8, 2), dst[0]);
    try std.testing.expectEqual(@as(u8, 255), dst[3]);
    // Same dims through the 2D wrapper must agree (shared helper).
    var dst2: [1 * 1 * 4]u8 = undefined;
    Texture.downsampleLevel(&src, 3, 3, &dst2, 1, 1);
    try std.testing.expectEqualSlices(u8, &dst, &dst2);

    // 2x1 -> 1x1 exercises edge clamping (sy1 clamps to 0): the single
    // source row is sampled twice, i.e. a plain average, no OOB read.
    var edge_src: [2 * 1 * 4]u8 = .{ 10, 0, 0, 255, 20, 0, 0, 255 };
    var edge_dst: [1 * 1 * 4]u8 = undefined;
    boxDownsampleU8(&edge_src, 2, 1, &edge_dst, 1, 1);
    // (10+20+10+20+2)>>2 = 15.
    try std.testing.expectEqual(@as(u8, 15), edge_dst[0]);
}

test "createCheckerboard overflow returns error instead of panicking" {
    const allocator = std.testing.allocator;
    const c1 = [4]u8{ 0, 0, 0, 255 };
    const c2 = [4]u8{ 255, 255, 255, 255 };
    // 100000^2 overflows u32: old `width * height` panicked in debug;
    // now returns ImageTooLarge before any allocation or GPU upload.
    try std.testing.expectError(
        error.ImageTooLarge,
        Texture.createCheckerboard(allocator, 100000, 100000, 8, c1, c2),
    );
}

// ---------------------------------------------------------------------------
// Golden tests: sRGB conversion
// ---------------------------------------------------------------------------

test "srgbToLinearU8 hits exact golden values and the endpoints" {
    // Computed with the IEC 61966-2-1 formula, rounded to nearest.
    try std.testing.expectEqual(@as(u8, 0), srgbToLinearU8(0));
    try std.testing.expectEqual(@as(u8, 255), srgbToLinearU8(255));
    try std.testing.expectEqual(@as(u8, 55), srgbToLinearU8(128));
    try std.testing.expectEqual(@as(u8, 13), srgbToLinearU8(64));
    try std.testing.expectEqual(@as(u8, 2), srgbToLinearU8(25));
    try std.testing.expectEqual(@as(u8, 147), srgbToLinearU8(200));
    // Monotonic non-decreasing over the whole table.
    var v: usize = 1;
    while (v < 256) : (v += 1) {
        try std.testing.expect(srgbToLinearU8(@intCast(v)) >= srgbToLinearU8(@intCast(v - 1)));
    }
}

test "convertSrgbToLinearInPlace converts RGB lanes and preserves alpha" {
    var pixels = [_]u8{ 200, 128, 0, 42, 255, 0, 25, 7 };
    convertSrgbToLinearInPlace(&pixels);
    try std.testing.expectEqual(@as(u8, srgbToLinearU8(200)), pixels[0]);
    try std.testing.expectEqual(@as(u8, srgbToLinearU8(128)), pixels[1]);
    try std.testing.expectEqual(@as(u8, 0), pixels[2]);
    try std.testing.expectEqual(@as(u8, 42), pixels[3]); // alpha untouched
    try std.testing.expectEqual(@as(u8, 255), pixels[4]);
    try std.testing.expectEqual(@as(u8, 0), pixels[5]);
    try std.testing.expectEqual(@as(u8, srgbToLinearU8(25)), pixels[6]);
    try std.testing.expectEqual(@as(u8, 7), pixels[7]); // alpha untouched
}

// ---------------------------------------------------------------------------
// Minimal PNG fixture synthesis (no external dependencies): PNG container +
// zlib stream with STORED deflate blocks + Adler-32. stb_image decodes these
// like any real PNG.
// ---------------------------------------------------------------------------

const TestPng = struct {
    fn crc32(tag: *const [4]u8, data: []const u8) u32 {
        var h = std.hash.crc.Crc32.init();
        h.update(tag);
        h.update(data);
        return h.final();
    }

    fn appendChunk(out: *std.Io.Writer, tag: *const [4]u8, data: []const u8) !void {
        var len_buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &len_buf, @intCast(data.len), .big);
        try out.writeAll(&len_buf);
        try out.writeAll(tag);
        try out.writeAll(data);
        var crc_buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &crc_buf, crc32(tag, data), .big);
        try out.writeAll(&crc_buf);
    }

    /// Wraps `raw` in a zlib stream that only uses stored (uncompressed)
    /// deflate blocks: header 0x78 0x01, blocks of up to 65535 bytes
    /// (BFINAL flag, LEN, NLEN), Adler-32 checksum.
    fn zlibStore(writer: *std.Io.Writer, raw: []const u8) !void {
        try writer.writeAll(&.{ 0x78, 0x01 });
        var rest = raw;
        while (true) {
            const final = rest.len <= 65535;
            const chunk_len: u16 = @intCast(@min(rest.len, 65535));
            try writer.writeByte(if (final) 0x01 else 0x00);
            var len_buf: [2]u8 = undefined;
            std.mem.writeInt(u16, &len_buf, chunk_len, .little);
            try writer.writeAll(&len_buf);
            std.mem.writeInt(u16, &len_buf, ~chunk_len, .little);
            try writer.writeAll(&len_buf);
            try writer.writeAll(rest[0..chunk_len]);
            rest = rest[chunk_len..];
            if (final) break;
        }
        var sum_buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &sum_buf, std.hash.Adler32.hash(raw), .big);
        try writer.writeAll(&sum_buf);
    }

    /// Builds a minimal PNG: signature, IHDR, optional PLTE, IDAT, IEND.
    /// `scanlines` must already contain one filter byte (0 = None) per row.
    fn build(
        allocator: std.mem.Allocator,
        width: u32,
        height: u32,
        bit_depth: u8,
        color_type: u8,
        plte: ?[]const u8,
        scanlines: []const u8,
    ) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(allocator);
        errdefer out.deinit();
        const w = &out.writer;

        try w.writeAll(&[_]u8{ 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' });

        var ihdr: [13]u8 = undefined;
        std.mem.writeInt(u32, ihdr[0..4], width, .big);
        std.mem.writeInt(u32, ihdr[4..8], height, .big);
        ihdr[8] = bit_depth;
        ihdr[9] = color_type;
        ihdr[10] = 0; // compression: deflate
        ihdr[11] = 0; // filter: adaptive filtering
        ihdr[12] = 0; // interlace: none
        try appendChunk(w, "IHDR", &ihdr);

        if (plte) |entries| try appendChunk(w, "PLTE", entries);

        var idat: std.Io.Writer.Allocating = .init(allocator);
        defer idat.deinit();
        try zlibStore(&idat.writer, scanlines);
        try appendChunk(w, "IDAT", idat.written());

        try appendChunk(w, "IEND", "");
        return out.toOwnedSlice();
    }
};

test "decodeMemory handles synthesized 8-bit grayscale PNG (gray replicated to RGB)" {
    const allocator = std.testing.allocator;
    sokol.time.setup();

    // 1x2 grayscale, values 0x00 and 0x80, filter byte 0 per row.
    const scanlines = [_]u8{ 0, 0x00, 0, 0x80 };
    const png = try TestPng.build(allocator, 1, 2, 8, 0, null, &scanlines);
    defer allocator.free(png);

    var raw = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = false });
    defer raw.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 1), raw.width);
    try std.testing.expectEqual(@as(u32, 2), raw.height);
    const px = raw.levels[0].?;
    // stb replicates the single gray channel into RGB; alpha becomes 255.
    try std.testing.expectEqual([4]u8{ 0, 0, 0, 255 }, px[0..4].*);
    try std.testing.expectEqual([4]u8{ 128, 128, 128, 255 }, px[4..8].*);
}

test "decodeMemory keeps the HIGH byte of 16-bit PNG channels" {
    const allocator = std.testing.allocator;
    sokol.time.setup();

    // 1x1 RGB 16-bit: R=0x1234, G=0xABCD, B=0x0001. stb's LDR output keeps
    // the top byte of each channel (stbi__convert_16_to_8: orig >> 8).
    var scanlines: [7]u8 = undefined;
    scanlines[0] = 0; // filter
    std.mem.writeInt(u16, scanlines[1..3], 0x1234, .big);
    std.mem.writeInt(u16, scanlines[3..5], 0xABCD, .big);
    std.mem.writeInt(u16, scanlines[5..7], 0x0001, .big);
    const png = try TestPng.build(allocator, 1, 1, 16, 2, null, &scanlines);
    defer allocator.free(png);

    var raw = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = false });
    defer raw.deinit(allocator);

    try std.testing.expectEqual([4]u8{ 0x12, 0xAB, 0x00, 255 }, raw.levels[0].?[0..4].*);
}

test "decodeMemory keeps the HIGH byte of 16-bit grayscale PNG" {
    const allocator = std.testing.allocator;
    sokol.time.setup();

    // 1x1 gray 16-bit value 0xABCD -> high byte 0xAB replicated to RGB.
    var scanlines: [3]u8 = undefined;
    scanlines[0] = 0;
    std.mem.writeInt(u16, scanlines[1..3], 0xABCD, .big);
    const png = try TestPng.build(allocator, 1, 1, 16, 0, null, &scanlines);
    defer allocator.free(png);

    var raw = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = false });
    defer raw.deinit(allocator);

    try std.testing.expectEqual([4]u8{ 0xAB, 0xAB, 0xAB, 255 }, raw.levels[0].?[0..4].*);
}

test "decodeMemory decodes synthesized palette PNG through PLTE" {
    const allocator = std.testing.allocator;
    sokol.time.setup();

    // 2x1 indexed (color type 3), palette {red, blue}; indices 0 and 1.
    const scanlines = [_]u8{ 0, 0, 1 };
    const plte = [_]u8{ 255, 0, 0, 0, 0, 255 };
    const png = try TestPng.build(allocator, 2, 1, 8, 3, &plte, &scanlines);
    defer allocator.free(png);

    var raw = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = false });
    defer raw.deinit(allocator);

    const px = raw.levels[0].?;
    try std.testing.expectEqual([4]u8{ 255, 0, 0, 255 }, px[0..4].*);
    try std.testing.expectEqual([4]u8{ 0, 0, 255, 255 }, px[4..8].*);
}

test "decodeMemory srgb_to_linear converts decoded pixels before upload" {
    const allocator = std.testing.allocator;
    sokol.time.setup();

    // 1x1 RGBA PNG with sRGB value 200 in R and a distinctive alpha 42.
    const scanlines = [_]u8{ 0, 200, 128, 25, 42 };
    const png = try TestPng.build(allocator, 1, 1, 8, 6, null, &scanlines);
    defer allocator.free(png);

    var raw = try Texture.decodeMemory(allocator, png, .{ .gen_mipmaps = false, .srgb_to_linear = true });
    defer raw.deinit(allocator);

    const px = raw.levels[0].?;
    try std.testing.expectEqual(@as(u8, 147), px[0]); // 200 -> 147 golden
    try std.testing.expectEqual(srgbToLinearU8(128), px[1]);
    try std.testing.expectEqual(srgbToLinearU8(25), px[2]);
    try std.testing.expectEqual(@as(u8, 42), px[3]); // alpha never converted
}

// ---------------------------------------------------------------------------
// Golden test: the 2D mip chain (levels, dimensions, averaged colors)
// ---------------------------------------------------------------------------

test "buildRaw mip chain has correct levels, sizes and box-filter colors" {
    const allocator = std.testing.allocator;

    // 4x4: top half pure red, bottom half pure blue, alpha 255 everywhere.
    var pixels: [4 * 4 * 4]u8 = undefined;
    for (0..4) |y| {
        for (0..4) |x| {
            const o = (y * 4 + x) * 4;
            const red = y < 2;
            pixels[o + 0] = if (red) 255 else 0;
            pixels[o + 1] = 0;
            pixels[o + 2] = if (red) 0 else 255;
            pixels[o + 3] = 255;
        }
    }

    var raw = try Texture.buildRaw(allocator, 4, 4, &pixels, true);
    defer raw.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 3), raw.num_levels);
    try std.testing.expectEqual(@as(u32, 4), raw.width);
    try std.testing.expectEqual(@as(u32, 4), raw.height);

    // Level 1 (2x2): top row red, bottom row blue, untouched by the filter.
    const l1 = raw.levels[1].?;
    try std.testing.expectEqual(@as(usize, 2 * 2 * 4), l1.len);
    try std.testing.expectEqual([4]u8{ 255, 0, 0, 255 }, l1[0..4].*);
    try std.testing.expectEqual([4]u8{ 0, 0, 255, 255 }, l1[8..12].*);

    // Level 2 (1x1): equal red/blue mix rounds to 128 via (sum+2)>>2.
    const l2 = raw.levels[2].?;
    try std.testing.expectEqual(@as(usize, 4), l2.len);
    try std.testing.expectEqual([4]u8{ 128, 0, 128, 255 }, l2[0..4].*);
}

test "buildRaw without mipmaps uploads exactly one level" {
    const allocator = std.testing.allocator;
    var pixels: [2 * 2 * 4]u8 = @splat(200);
    var raw = try Texture.buildRaw(allocator, 2, 2, &pixels, false);
    defer raw.deinit(allocator);
    try std.testing.expectEqual(@as(u32, 1), raw.num_levels);
    try std.testing.expectEqual(@as(u32, 200), raw.levels[0].?[0]);
}

// ---------------------------------------------------------------------------
// Golden test: HDR cube f16 mip chain (feeds textureLod in the IBL path)
// ---------------------------------------------------------------------------

test "buildRawFacesHdr builds the f16 mip chain with averaged values" {
    const allocator = std.testing.allocator;

    // One 2x2 face with R={1, 3, 5, 7}, G=B=0, A=1: level 1 must average
    // R to 4.0 exactly (f16-representable).
    var face: [2 * 2 * 4]u16 = undefined;
    const r_values = [_]f32{ 1, 3, 5, 7 };
    for (r_values, 0..) |rv, i| {
        face[i * 4 + 0] = Texture.floatToHalfBits(rv);
        face[i * 4 + 1] = 0;
        face[i * 4 + 2] = 0;
        face[i * 4 + 3] = Texture.floatToHalfBits(1.0);
    }
    var faces: [6][]const u16 = undefined;
    for (&faces) |*f| f.* = &face;

    var chain = try CubeTexture.buildRawFacesHdr(allocator, 2, faces);
    defer chain.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 2), chain.num_levels);
    try std.testing.expectEqual(@as(u32, 2), chain.size);

    const l1 = chain.levels[1].?;
    try std.testing.expectEqual(@as(usize, 6 * 1 * 1 * 4), l1.len);
    // Face 0 averaged R = 4.0; every face inherits the same average.
    for (0..6) |f| {
        const o = f * 4;
        try std.testing.expectEqual(@as(f32, 4.0), Texture.halfBitsToFloat(l1[o + 0]));
        try std.testing.expectEqual(@as(f32, 0.0), Texture.halfBitsToFloat(l1[o + 1]));
        try std.testing.expectEqual(@as(f32, 1.0), Texture.halfBitsToFloat(l1[o + 3]));
    }
}

test "buildRawFacesHdr validates size and face buffers" {
    const allocator = std.testing.allocator;
    const face: [2 * 2 * 4]u16 = @splat(0);
    var faces: [6][]const u16 = @splat(&face);

    try std.testing.expectError(
        error.InvalidDimensions,
        CubeTexture.buildRawFacesHdr(allocator, 0, faces),
    );
    faces[5] = face[0..2];
    try std.testing.expectError(
        error.InvalidFaceBufferSize,
        CubeTexture.buildRawFacesHdr(allocator, 2, faces),
    );
}

test "checkedFaceBytes rejects empty and overflowing sizes" {
    try std.testing.expectError(error.InvalidDimensions, checkedFaceBytes(0));
    // 100000^2 overflows u32: old `size * size * 4` wrapped to a small
    // alloc size; now ImageTooLarge before any allocation or GPU upload.
    try std.testing.expectError(error.ImageTooLarge, checkedFaceBytes(100000));
    try std.testing.expectEqual(@as(usize, 2 * 2 * 4), try checkedFaceBytes(2));
}

test "createParticleDot validates size without GPU upload" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(
        error.InvalidDimensions,
        Texture.createParticleDot(allocator, 0),
    );
    try std.testing.expectError(
        error.ImageTooLarge,
        Texture.createParticleDot(allocator, 100000),
    );
}

test "initRawFaces validates dimensions without GPU upload" {
    const allocator = std.testing.allocator;
    const empty_faces: [6][]const u8 = @splat(&.{});
    try std.testing.expectError(
        error.InvalidDimensions,
        CubeTexture.initRawFaces(allocator, 0, empty_faces, false),
    );
    // Huge size fails on checked arithmetic before face-length checks.
    try std.testing.expectError(
        error.ImageTooLarge,
        CubeTexture.initRawFaces(allocator, 100000, empty_faces, false),
    );
    // Small size with short faces fails on length validation (no GPU).
    var one_pixel: [4]u8 = .{ 1, 2, 3, 255 };
    const bad_faces: [6][]const u8 = .{
        one_pixel[0..2], &one_pixel, &one_pixel, &one_pixel, &one_pixel, &one_pixel,
    };
    try std.testing.expectError(
        error.InvalidFaceBufferSize,
        CubeTexture.initRawFaces(allocator, 1, bad_faces, false),
    );
}

test "createProceduralSkybox rejects empty size" {
    const allocator = std.testing.allocator;
    try std.testing.expectError(
        error.InvalidDimensions,
        CubeTexture.createProceduralSkybox(allocator, .{ .size = 0 }),
    );
}

test "fromFiles reports missing faces without freeing uninitialized memory" {
    const allocator = std.testing.allocator;
    // First face missing: loaded == 0, the sized defer frees nothing.
    // C stbi allocations are not tracked by std.testing.allocator; this
    // only asserts the error contract, not leak accounting.
    const missing: [6][]const u8 = @splat("definitely/missing/face.png");
    try std.testing.expectError(
        error.ImageDecodeFailed,
        CubeTexture.fromFiles(allocator, missing),
    );
}

test "fromEquirectangular rejects huge faces before allocation" {
    const allocator = std.testing.allocator;
    // 1x1 RGBA PNG through the in-file fixture builder (no GPU involved).
    const scanlines = [_]u8{ 0, 10, 20, 30, 255 };
    const png = try TestPng.build(allocator, 1, 1, 8, 6, null, &scanlines);
    defer allocator.free(png);
    try std.testing.expectError(
        error.ImageTooLarge,
        CubeTexture.fromEquirectangular(allocator, png, 100000),
    );
}
