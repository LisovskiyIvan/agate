const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const c = @import("c.zig").c;

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
        wrap_u: sg.Wrap = .REPEAT,
        wrap_v: sg.Wrap = .REPEAT,
        /// Signed distance fields must not be box-downsampled: the mip chain
        /// dilutes thin strokes and the shader edge drifts. Disable for fonts.
        mipmaps: bool = true,
    };

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

        const smp = sg.makeSampler(.{
            .min_filter = options.min_filter,
            .mag_filter = options.mag_filter,
            .wrap_u = options.wrap_u,
            .wrap_v = options.wrap_v,
        });

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
        levels: [16]?[]u8 = @splat(null),

        pub fn deinit(self: *RawTexture, allocator: std.mem.Allocator) void {
            for (self.levels[0..self.num_levels]) |level| {
                if (level) |buf| allocator.free(buf);
            }
            self.* = .{};
        }
    };

    /// Copies `rgba_pixels` and optionally builds the box-filtered mip chain.
    fn buildRaw(allocator: std.mem.Allocator, width: u32, height: u32, rgba_pixels: []const u8, gen_mipmaps: bool) !RawTexture {
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
        const smp = sg.makeSampler(.{
            .min_filter = options.min_filter,
            .mag_filter = options.mag_filter,
            .mipmap_filter = if (raw.num_levels > 1) .LINEAR else .NEAREST,
            .wrap_u = options.wrap_u,
            .wrap_v = options.wrap_v,
        });

        return .{
            .image = img,
            .view = view,
            .sampler = smp,
            .width = raw.width,
            .height = raw.height,
        };
    }

    /// Decodes an in-memory image (PNG/JPEG/...) to RGBA without touching the
    /// GPU. Thread-safe; pair with `fromRaw`.
    pub fn decodeMemory(allocator: std.mem.Allocator, bytes: []const u8, gen_mipmaps: bool) !RawTexture {
        var w: c_int = 0;
        var h: c_int = 0;
        var channels_in_file: c_int = 0;

        const t0 = sokol.time.now();
        const data = c.stbi_load_from_memory(
            bytes.ptr,
            @intCast(bytes.len),
            &w,
            &h,
            &channels_in_file,
            4,
        );
        const t1 = sokol.time.now();
        if (data == null) return error.ImageDecodeFailed;
        defer c.stbi_image_free(data);

        const width: u32 = @intCast(w);
        const height: u32 = @intCast(h);
        const size_bytes: usize = @as(usize, width) * @as(usize, height) * 4;
        const raw = try buildRaw(allocator, width, height, data[0..size_bytes], gen_mipmaps);

        const t2 = sokol.time.now();
        std.debug.print("  [TEX {d}x{d}] stbi: {d:0.1} ms | mipgen: {d:0.1} ms | total: {d:0.1} ms\n", .{
            width,                                  height,
            sokol.time.ms(sokol.time.diff(t1, t0)), sokol.time.ms(sokol.time.diff(t2, t1)),
            sokol.time.ms(sokol.time.diff(t2, t0)),
        });
        return raw;
    }

    /// File variant of `decodeMemory`. Thread-safe; pair with `fromRaw`.
    pub fn decodeFile(allocator: std.mem.Allocator, file_path: []const u8, gen_mipmaps: bool) !RawTexture {
        const path_z = try allocator.dupeZ(u8, file_path);
        defer allocator.free(path_z);

        var w: c_int = 0;
        var h: c_int = 0;
        var channels_in_file: c_int = 0;

        const t0 = sokol.time.now();
        const data = c.stbi_load(path_z.ptr, &w, &h, &channels_in_file, 4);
        const t1 = sokol.time.now();
        if (data == null) return error.ImageDecodeFailed;
        defer c.stbi_image_free(data);

        const width: u32 = @intCast(w);
        const height: u32 = @intCast(h);
        const size_bytes: usize = @as(usize, width) * @as(usize, height) * 4;
        const raw = try buildRaw(allocator, width, height, data[0..size_bytes], gen_mipmaps);

        const t2 = sokol.time.now();
        std.debug.print("  [TEX {d}x{d}] stbi: {d:0.1} ms | mipgen: {d:0.1} ms | total: {d:0.1} ms\n", .{
            width,                                  height,
            sokol.time.ms(sokol.time.diff(t1, t0)), sokol.time.ms(sokol.time.diff(t2, t1)),
            sokol.time.ms(sokol.time.diff(t2, t0)),
        });
        return raw;
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
        const pixel_count = size * size;
        const buffer = try allocator.alloc(u8, pixel_count * 4);
        defer allocator.free(buffer);

        const center: f32 = @as(f32, @floatFromInt(size)) * 0.5 - 0.5;
        const radius: f32 = @as(f32, @floatFromInt(size)) * 0.5;

        var y: u32 = 0;
        while (y < size) : (y += 1) {
            var x: u32 = 0;
            while (x < size) : (x += 1) {
                const alpha = particleDotAlpha(x, y, center, radius);

                const idx = (y * size + x) * 4;
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
        var raw = try decodeMemory(allocator, bytes, options.mipmaps);
        defer raw.deinit(allocator);
        return fromRaw(&raw, options);
    }

    pub fn fromFile(allocator: std.mem.Allocator, file_path: []const u8, options: Options) !Texture {
        var raw = try decodeFile(allocator, file_path, options.mipmaps);
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
pub const SkyboxConfig = struct {
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
        const face_bytes = size * size * 4;
        for (faces) |face| {
            if (face.len != face_bytes) return error.InvalidFaceBufferSize;
        }

        // Concatenate 6 faces for mip 0
        const mip0_total_bytes = 6 * face_bytes;
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
            const cur_total_bytes = 6 * cur_size * cur_size * 4;
            const cur_buf = try allocator.alloc(u8, cur_total_bytes);
            mip_buffers[m] = cur_buf;

            const prev_face_bytes: usize = @as(usize, prev_size) * @as(usize, prev_size) * 4;
            const cur_face_bytes: usize = @as(usize, cur_size) * @as(usize, cur_size) * 4;
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

    pub fn createProceduralSkybox(allocator: std.mem.Allocator, config: SkyboxConfig) !CubeTexture {
        const size = config.size;
        const face_bytes = size * size * 4;

        var face_slices: [6][]u8 = undefined;
        for (0..6) |i| {
            face_slices[i] = try allocator.alloc(u8, face_bytes);
        }
        defer {
            for (0..6) |i| allocator.free(face_slices[i]);
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

                    const out_idx = (py * size + px) * 4;
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
        var size: u32 = 0;

        for (face_paths, 0..) |path, i| {
            const path_z = try allocator.dupeZ(u8, path);
            defer allocator.free(path_z);

            var w: c_int = 0;
            var h: c_int = 0;
            var comp: c_int = 0;
            const data = c.stbi_load(path_z.ptr, &w, &h, &comp, 4);
            if (data == null) {
                // Free previously loaded
                for (0..i) |prev| c.stbi_image_free(face_data[prev]);
                return error.ImageDecodeFailed;
            }
            if (w != h) {
                for (0..i + 1) |prev| c.stbi_image_free(face_data[prev]);
                return error.CubeFaceMustBeSquare;
            }
            if (i == 0) {
                size = @intCast(w);
            } else if (@as(u32, @intCast(w)) != size) {
                for (0..i + 1) |prev| c.stbi_image_free(face_data[prev]);
                return error.CubeFacesMustHaveEqualSize;
            }
            face_data[i] = data;
        }
        defer {
            for (0..6) |i| c.stbi_image_free(face_data[i]);
        }

        const face_bytes = size * size * 4;
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

        const pano_w: u32 = @intCast(pw);
        const pano_h: u32 = @intCast(ph);
        const face_bytes = face_size * face_size * 4;

        var face_slices: [6][]u8 = undefined;
        for (0..6) |i| {
            face_slices[i] = try allocator.alloc(u8, face_bytes);
        }
        defer {
            for (0..6) |i| allocator.free(face_slices[i]);
        }

        for (0..6) |face| {
            const buf = face_slices[face];
            var py: u32 = 0;
            while (py < face_size) : (py += 1) {
                var px: u32 = 0;
                while (px < face_size) : (px += 1) {
                    const d = cubeTexelDirection(face, px, py, face_size);
                    const t = panoramaTexel(d, pano_w, pano_h);
                    const p_idx = (t.y * pano_w + t.x) * 4;
                    const out_idx = (py * face_size + px) * 4;

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
    /// Single mip level only: HDR cubes upload without mipmaps.
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
                    const p_idx = (t.y * width + t.x) * 4;
                    const out_idx = (py * size + px) * 4;
                    for (0..4) |ch| {
                        out[out_idx + ch] = Texture.floatToHalfBits(rgba[p_idx + ch]);
                    }
                }
            }
        }
        return raw;
    }

    /// Uploads six half-float RGBA faces as a single-level RGBA16F cube.
    /// Main thread only. Each face must hold size*size*4 half patterns.
    /// Linear min/mag, clamp on all axes, no mipmaps, no gamma correction.
    pub fn initRawFacesHdr(allocator: std.mem.Allocator, size: u32, faces: [6][]const u16) !CubeTexture {
        if (size == 0) return error.InvalidDimensions;
        const face_pixels = std.math.mul(usize, std.math.mul(usize, size, size) catch return error.ImageTooLarge, 4) catch return error.ImageTooLarge;
        for (faces) |face| {
            if (face.len != face_pixels) return error.InvalidFaceBufferSize;
        }

        // Concatenate the 6 faces: sokol expects one contiguous cube upload.
        const mip0 = try allocator.alloc(u16, std.math.mul(usize, 6, face_pixels) catch return error.ImageTooLarge);
        defer allocator.free(mip0);
        for (faces, 0..) |face, i| {
            @memcpy(mip0[i * face_pixels .. (i + 1) * face_pixels], face);
        }

        var img_desc = sg.ImageDesc{
            .type = .CUBE,
            .width = @intCast(size),
            .height = @intCast(size),
            .num_slices = 6,
            .num_mipmaps = 1,
            .pixel_format = .RGBA16F,
            .sample_count = 1,
        };
        img_desc.data.mip_levels[0] = sg.asRange(mip0);

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
            .wrap_w = .CLAMP_TO_EDGE,
        });

        return .{
            .image = img,
            .view = view,
            .sampler = smp,
            .size = size,
            .num_mipmaps = 1,
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

    var single = try Texture.decodeMemory(allocator, png, false);
    defer single.deinit(allocator);
    try std.testing.expectEqual(@as(u32, 512), single.width);
    try std.testing.expectEqual(@as(u32, 512), single.height);
    try std.testing.expectEqual(@as(u32, 1), single.num_levels);
    try std.testing.expectEqual(@as(usize, 512 * 512 * 4), single.levels[0].?.len);

    var mipped = try Texture.decodeMemory(allocator, png, true);
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
