//! 2D texture type and upload/decode paths. Split out of `texture.zig`
//! (facade).
//!
//! This module owns the `Texture` type: fields, sampler/upload paths, the
//! CPU decode/routing entries (KTX2/DDS/EXR entry points included; those
//! readers stay separate top-level modules), the raw pixel containers, and
//! thin forwarders into `color.zig` / `mip.zig` so every call site keeps
//! working unchanged.
//!
//! Anti-cycle rule (same as `camera/`, `profiler/`, `ui/`): this module
//! never imports the `texture.zig` facade back.
const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const c = @import("../c.zig").c;
const ktx2 = @import("../ktx2.zig");
const dds = @import("../dds.zig");
const exr = @import("../exr.zig");
const color = @import("color.zig");
const mip = @import("mip.zig");

pub const Texture = struct {
    image: sg.Image,
    view: sg.View,
    sampler: sg.Sampler,
    width: u32,
    height: u32,
    num_mipmaps: u32 = 1,
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
        /// Max anisotropy 1..16 (sokol sg_sampler_desc.max_anisotropy).
        /// Default 4 = Babylon's `Texture.DEFAULT_ANISOTROPIC_FILTERING_LEVEL`
        /// (babylon.js: `e.DEFAULT_ANISOTROPIC_FILTERING_LEVEL=4`, assigned to
        /// every `Texture.anisotropicFilteringLevel`; the glTF loader never
        /// overrides it, so every texture a glTF file references is sampled
        /// with 4). sokol requires LINEAR min/mag/mip filters for anisotropy
        /// > 1 and fails sampler validation otherwise, so non-LINEAR
        /// combinations clamp back to 1 here (Babylon clamps the same way,
        /// by sampling mode, in `_setAnisotropicLevel`). sokol has no lod
        /// bias; lod range defaults to 0..FLT_MAX (not exposed).
        max_anisotropy: u32 = 4,
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

    /// Mip selection filter actually handed to sokol: the authored filter
    /// only exists when there IS a mip chain, otherwise NEAREST (a
    /// single-level texture has no mip to select).
    fn samplerMipFilter(options: Options, num_mip_levels: u32) sg.Filter {
        return if (num_mip_levels > 1) options.mip_filter else .NEAREST;
    }

    /// Effective anisotropy for one sampler after the two clamps: the sokol
    /// range (1..16) and the LINEAR-filter requirement described on
    /// Options.max_anisotropy. Pure, so the clamping contract is testable
    /// without a graphics context.
    pub fn effectiveAnisotropy(options: Options, num_mip_levels: u32) u32 {
        const aniso = @min(options.max_anisotropy, 16);
        if (aniso <= 1) return aniso;
        if (options.min_filter != .LINEAR or options.mag_filter != .LINEAR or
            samplerMipFilter(options, num_mip_levels) != .LINEAR) return 1;
        return aniso;
    }

    /// Shared sampler creation for LDR 2D textures: applies min/mag/wrap,
    /// the mip filter (only meaningful with a chain), and the anisotropy
    /// guard described on Options.max_anisotropy.
    fn makeSamplerFor(options: Options, num_mip_levels: u32) sg.Sampler {
        return sg.makeSampler(.{
            .min_filter = options.min_filter,
            .mag_filter = options.mag_filter,
            .mipmap_filter = samplerMipFilter(options, num_mip_levels),
            .wrap_u = options.wrap_u,
            .wrap_v = options.wrap_v,
            .max_anisotropy = effectiveAnisotropy(options, num_mip_levels),
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

    /// Total estimated GPU memory in bytes for this texture (including all mip levels).
    pub fn getGpuMemoryBytes(self: *const Texture) usize {
        if (self.image.id == 0) return 0;
        const bpp = mip.pixelFormatBytes(self.format);
        var total: usize = 0;
        var w = @max(1, self.width);
        var h = @max(1, self.height);
        const mips = @max(1, self.num_mipmaps);
        for (0..mips) |_| {
            total += gpuLevelBytes(self.format, w, h, bpp);
            if (w == 1 and h == 1) break;
            w = @max(1, w / 2);
            h = @max(1, h / 2);
        }
        return total;
    }

    /// Exact level bytes for one mip level: block-grid math for the
    /// compressed formats fromRawBlock uploads (4x4 blocks, 8 B each for
    /// BC1, 16 B each for BC2/BC3/BC7/ETC2 RGBA8/ASTC 4x4), w*h*bpp otherwise. `bpp`
    /// is the pixelFormatBytes of the format.
    fn gpuLevelBytes(format: sg.PixelFormat, w: u32, h: u32, bpp: usize) usize {
        switch (format) {
            .BC1_RGBA => {
                const bw = (@as(usize, w) + 3) / 4;
                const bh = (@as(usize, h) + 3) / 4;
                return bw * bh * 8;
            },
            .BC2_RGBA, .BC3_RGBA, .BC3_SRGBA => {
                const bw = (@as(usize, w) + 3) / 4;
                const bh = (@as(usize, h) + 3) / 4;
                return bw * bh * 16;
            },
            .BC7_RGBA, .BC7_SRGBA, .ETC2_RGBA8, .ETC2_SRGB8A8, .ASTC_4x4_RGBA, .ASTC_4x4_SRGBA => {
                const bw = (@as(usize, w) + 3) / 4;
                const bh = (@as(usize, h) + 3) / 4;
                return bw * bh * 16;
            },
            else => return @as(usize, w) * @as(usize, h) * bpp,
        }
    }
    /// Box-filter downsample of one RGBA8 level. Dims floor at 1, source coords
    /// clamp at edges (handles NPOT). Thin wrapper over the shared
    /// file-private `boxDownsampleU8` so 2D and cube paths stay in sync.
    fn downsampleLevel(src: []const u8, src_w: u32, src_h: u32, dst: []u8, dst_w: u32, dst_h: u32) void {
        mip.boxDownsampleU8(src, src_w, src_h, dst, dst_w, dst_h);
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
        // Guard the public entry point: the mip chain treats the buffer as
        // width*height*4, and downsampleLevel runs unsafely, so a short
        // buffer would cause out-of-bounds reads. Checked arithmetic keeps
        // huge dimensions from wrapping to a small expected size.
        if (width == 0 or height == 0) return error.InvalidDimensions;
        const pixel_count = std.math.mul(u32, width, height) catch return error.ImageTooLarge;
        const expected_bytes = std.math.mul(u32, pixel_count, 4) catch return error.ImageTooLarge;
        if (rgba_pixels.len != expected_bytes) return error.InvalidDimensions;

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
            .num_mipmaps = raw.num_levels,
        };
    }

    /// Decode-time switches for the CPU paths (GPU-free, thread-safe).
    /// srgb_to_linear converts RGB lanes before the mip chain is built.
    pub const DecodeOptions = struct {
        gen_mipmaps: bool = true,
        srgb_to_linear: bool = false,
        /// Transcode target for KTX2 Basis payloads (ETC1S/UASTC); ignored
        /// by every other format. Null = desktop-first .bc7 default (the
        /// fromRawBlock backend gate still rejects unsupported backends
        /// explicitly). Loaders snapshot per load via
        /// basisTargetForCurrentThread() so worker decodes already carry the
        /// backend-accurate target.
        basis_target: ?ktx2.BasisTarget = null,
    };

    // -----------------------------------------------------------------------
    // Block-compressed textures (KTX2 BC1/BC2/BC3/BC7 / ETC2 RGBA8 / ASTC 4x4 and DDS
    // BC1/BC2/BC3/BC7, never decoded).
    //
    // Backend gate: compressed formats are uploaded only when
    // sg.queryPixelformat reports sample support for the EXACT variant
    // (UNORM vs SRGB matter — they are different GPU formats). There is no
    // CPU fallback: without a transcoder dependency an unsupported backend
    // gets error.BlockFormatNotSupportedByBackend, never silent corruption.
    //
    // sokol upload mechanism (verified against this checkout's sokol-zig):
    // sg.ImageDesc.data.mip_levels[m] carries one tightly packed surface
    // per level (for 2D: exactly sg.querySurfacePitch bytes, which for our
    // 4x4/16B blocks equals the KTX2 level size) with .num_mipmaps set to
    // the authored level count. Images stay immutable (the default): sokol
    // rejects compressed dynamic_update/attachment images, and the engine
    // never streams into textures — .data at creation is the only path.
    //
    // Partial chains: levels upload exactly as authored (no CPU mip
    // synthesis — that needs a decoder plus an encoder). The GPU clamps LOD
    // to the smallest present level, so a single-level file simply never
    // minifies past level 0. Sampler mip filtering is enabled only with >1
    // authored level (same convention as makeSamplerFor), and every filter
    // drops to NEAREST when the backend reports sample-without-filter.
    // -----------------------------------------------------------------------

    /// Backend capability snapshot for every block family the engine can
    /// upload. Fill with queryBlockSupport(); the decision helpers are pure
    /// and unit-tested. BC2 shares the BC3 (S3TC) feature bit: desktop GL
    /// exposes BC1-BC3 as one extension, and sokol has no separate BC2
    /// query target that would behave differently.
    pub const BlockSupport = struct {
        bc1_sample: bool = false,
        bc1_filter: bool = false,
        bc3_sample: bool = false,
        bc3_filter: bool = false,
        bc7_sample: bool = false,
        bc7_filter: bool = false,
        etc2_sample: bool = false,
        etc2_filter: bool = false,
        astc_sample: bool = false,
        astc_filter: bool = false,

        /// True when the backend can sample the exact sg format. Pure.
        pub fn supportsFormat(self: BlockSupport, fmt: sg.PixelFormat) bool {
            return switch (fmt) {
                .BC1_RGBA => self.bc1_sample,
                .BC2_RGBA, .BC3_RGBA, .BC3_SRGBA => self.bc3_sample,
                .BC7_RGBA, .BC7_SRGBA => self.bc7_sample,
                .ETC2_RGBA8, .ETC2_SRGB8A8 => self.etc2_sample,
                .ASTC_4x4_RGBA, .ASTC_4x4_SRGBA => self.astc_sample,
                else => false,
            };
        }

        /// Preference order BC7 → BC3 → BC1 → ASTC 4x4 → ETC2 RGBA8 (UNORM
        /// representatives). Pure. Feeds the unsupported-format log hint so
        /// authors learn which encoding the current backend prefers.
        pub fn preferred(self: BlockSupport) ?sg.PixelFormat {
            if (self.bc7_sample) return .BC7_RGBA;
            if (self.bc3_sample) return .BC3_RGBA;
            if (self.bc1_sample) return .BC1_RGBA;
            if (self.astc_sample) return .ASTC_4x4_RGBA;
            if (self.etc2_sample) return .ETC2_RGBA8;
            return null;
        }
    };

    /// Exact sg format for a block format. Pure: UNORM stays UNORM; _SRGB
    /// files upload to the sRGB GPU variant (hardware linearizes on sample
    /// — the no-decoder equivalent of the CPU sRGB→linear path). BC1/BC2
    /// have no sRGB variant in this sokol checkout and always map to UNORM
    /// (see dds.zig).
    pub fn sgPixelFormatForBlock(format: ktx2.BlockFormat) sg.PixelFormat {
        return switch (format) {
            .bc1_unorm => .BC1_RGBA,
            .bc2_unorm => .BC2_RGBA,
            .bc3_unorm => .BC3_RGBA,
            .bc3_srgb => .BC3_SRGBA,
            .bc7_unorm => .BC7_RGBA,
            .bc7_srgb => .BC7_SRGBA,
            .etc2_rgba8_unorm => .ETC2_RGBA8,
            .etc2_rgba8_srgb => .ETC2_SRGB8A8,
            .astc_4x4_unorm => .ASTC_4x4_RGBA,
            .astc_4x4_srgb => .ASTC_4x4_SRGBA,
        };
    }

    /// Live backend support for every block family (UNORM representatives;
    /// the upload gate additionally checks the exact variant). Requires a
    /// valid sg context; keep the pure logic in BlockSupport for tests.
    pub fn queryBlockSupport() BlockSupport {
        const bc1 = sg.queryPixelformat(.BC1_RGBA);
        const bc3 = sg.queryPixelformat(.BC3_RGBA);
        const bc7 = sg.queryPixelformat(.BC7_RGBA);
        const etc2 = sg.queryPixelformat(.ETC2_RGBA8);
        const astc = sg.queryPixelformat(.ASTC_4x4_RGBA);
        return .{
            .bc1_sample = bc1.sample,
            .bc1_filter = bc1.filter,
            .bc3_sample = bc3.sample,
            .bc3_filter = bc3.filter,
            .bc7_sample = bc7.sample,
            .bc7_filter = bc7.filter,
            .etc2_sample = etc2.sample,
            .etc2_filter = etc2.filter,
            .astc_sample = astc.sample,
            .astc_filter = astc.filter,
        };
    }

    /// Creates the GPU image from owned block-compressed levels. Main
    /// thread only. Precondition: raw holds ≥1 authored level from
    /// ktx2.decodeBlock2D or dds.decodeBlock2D (an empty default is a
    /// programming error). Fails with BlockFormatNotSupportedByBackend when
    /// the backend cannot sample the format — re-encode the asset, there is
    /// no decoder.
    pub fn fromRawBlock(raw: *const ktx2.RawBlockTexture, options: Options) error{BlockFormatNotSupportedByBackend}!Texture {
        std.debug.assert(raw.num_levels >= 1 and raw.num_levels <= 16);
        const sg_format = sgPixelFormatForBlock(raw.format);
        const info = sg.queryPixelformat(sg_format);
        if (!info.sample) {
            const hint = queryBlockSupport().preferred();
            std.log.err("Block-compressed texture ({s} {d}x{d}, {d} level(s)) not sampleable on this backend; closest supported block format: {?}; re-encode the asset (no CPU decoder exists)", .{
                @tagName(raw.format), raw.width, raw.height, raw.num_levels, hint,
            });
            return error.BlockFormatNotSupportedByBackend;
        }

        var effective = options;
        if (!info.filter) {
            // Sample-without-filter backends: any LINEAR filter on this
            // format is invalid, so force NEAREST everywhere (anisotropy
            // then clamps to 1 inside makeSamplerFor).
            effective.min_filter = .NEAREST;
            effective.mag_filter = .NEAREST;
            effective.mip_filter = .NEAREST;
        }

        var img_desc = sg.ImageDesc{
            .width = @intCast(raw.width),
            .height = @intCast(raw.height),
            .pixel_format = sg_format,
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
        const smp = makeSamplerFor(effective, raw.num_levels);

        return .{
            .image = img,
            .view = view,
            .sampler = smp,
            .width = raw.width,
            .height = raw.height,
            .num_mipmaps = raw.num_levels,
            .format = sg_format,
        };
    }

    /// Decodes an in-memory image (PNG/JPEG/KTX2/...) to RGBA without
    /// touching the GPU. Thread-safe; pair with `fromRaw`. KTX2 payloads
    /// (magic sniff) route to the ktx2 reader: only its uncompressed LDR
    /// subset decodes here — cube KTX2 files are rejected with
    /// error.UnsupportedFaceCount (use ktx2.decodeCube instead), and Basis
    /// payloads (ETC1S/UASTC) fail with error.BasisRequiresBlockDecode after
    /// reporting the file's own validation reason for malformed files (use
    /// decodeImageMemory or fromBasisMemory instead). DDS
    /// payloads are block-compressed and have no RGBA8 form: they fail
    /// here (malformed files with their own validation error, valid ones
    /// with DdsRequiresBlockDecode) — use decodeImageMemory or
    /// fromDdsMemory instead.
    pub fn decodeMemory(allocator: std.mem.Allocator, bytes: []const u8, opts: DecodeOptions) !RawTexture {
        if (ktx2.isBasisKtx2(bytes)) {
            // Report the file's own validation error when malformed (init or
            // subset gate), so a corrupt Basis file never degrades into a
            // generic stb failure; a valid one names the correct API instead.
            _ = try ktx2.basisInfo(bytes);
            return error.BasisRequiresBlockDecode;
        }
        if (dds.sniff(bytes)) {
            // Report the file's own validation error when malformed, so a
            // corrupt DDS never degrades into a generic stb failure; a
            // valid DDS names the correct API instead.
            var tmp = try dds.decodeBlock2D(allocator, bytes, .{ .srgb = opts.srgb_to_linear });
            tmp.deinit(allocator);
            return error.DdsRequiresBlockDecode;
        }
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
        if (opts.srgb_to_linear) color.convertSrgbToLinearInPlace(data[0..size_bytes]);
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

    /// CPU-decoded image in either engine form: RGBA8 levels or owned
    /// block-compressed levels. GPU-free; upload the payload with fromRaw
    /// (.rgba) or fromRawBlock (.block) on the main thread. Produced by
    /// decodeImageMemory/decodeImageFile; free with deinit.
    pub const DecodedImage = union(enum) {
        rgba: RawTexture,
        block: ktx2.RawBlockTexture,

        pub fn deinit(self: *DecodedImage, allocator: std.mem.Allocator) void {
            switch (self.*) {
                .rgba => |*r| r.deinit(allocator),
                .block => |*b| b.deinit(allocator),
            }
        }

        /// Exact bytes the upload hands to sg (asset byte budget).
        pub fn totalBytes(self: *const DecodedImage) usize {
            switch (self.*) {
                .rgba => |*r| {
                    var total: usize = 0;
                    for (r.levels[0..r.num_levels]) |level| {
                        if (level) |buf| total += buf.len;
                    }
                    return total;
                },
                .block => |*b| return b.totalBytes(),
            }
        }
    };

    /// decodeMemory plus block routing: KTX2 BC/ASTC and DDS BC payloads
    /// decode to .block (owned slices, mip chain as authored, no
    /// synthesis); KTX2 Basis payloads (ETC1S/UASTC) transcode to
    /// opts.basis_target (null = desktop-first .bc7) — .block for bc7/astc,
    /// .rgba for the rgba32 fallback; everything else behaves exactly like
    /// decodeMemory (.rgba). Thread-safe; pair with fromRaw/fromRawBlock. A
    /// Basis file that fails validation surfaces the reader error — never a
    /// silent RGBA8 fallback.
    /// decode_opts are RGBA8-only, except srgb_to_linear, which doubles as
    /// the DDS legacy sRGB decision (DX10/KTX2 files carry their own tag):
    /// gen_mipmaps has no effect on .block (chain and sRGB-ness ride in
    /// the authored levels and the GPU format). For Basis, srgb_to_linear is
    /// the explicit RGBA32-path decision (block targets follow the file DFD).
    pub fn decodeImageMemory(allocator: std.mem.Allocator, bytes: []const u8, opts: DecodeOptions) !DecodedImage {
        if (ktx2.isBasisKtx2(bytes)) {
            const t = opts.basis_target orelse .bc7;
            return try ktx2.decodeBasis2D(allocator, bytes, t, .{
                .gen_mipmaps = opts.gen_mipmaps,
                .srgb_to_linear = opts.srgb_to_linear,
            });
        }
        if (ktx2.isBlockKtx2(bytes)) {
            return .{ .block = try ktx2.decodeBlock2D(allocator, bytes) };
        }
        if (dds.sniff(bytes)) {
            return .{ .block = try dds.decodeBlock2D(allocator, bytes, .{ .srgb = opts.srgb_to_linear }) };
        }
        return .{ .rgba = try decodeMemory(allocator, bytes, opts) };
    }

    /// File variant of decodeImageMemory. Same buffered-read pattern as
    /// decodeFile (Zig 0.16 removed std.fs.cwd()); thread-safe, pair with
    /// fromRaw/fromRawBlock.
    pub fn decodeImageFile(allocator: std.mem.Allocator, file_path: []const u8, opts: DecodeOptions) !DecodedImage {
        const io = std.Io.Threaded.global_single_threaded.io();
        const file = try std.Io.Dir.cwd().openFile(io, file_path, .{});
        defer file.close(io);

        const file_size = try file.length(io);
        const bytes = try allocator.alloc(u8, std.math.cast(usize, file_size) orelse return error.ImageTooLarge);
        defer allocator.free(bytes);

        const read = try file.readPositionalAll(io, bytes, 0);
        if (read < bytes.len) return error.ImageDecodeFailed;
        return decodeImageMemory(allocator, bytes, opts);
    }

    /// Babylon's `BRDFTextureTools.GetEnvironmentBRDFTexture` (256x256,
    /// `gammaSpace = true`) embedded byte-identical: the environment-BRDF
    /// lookup behind `coloredEnergyConservationFactor`. The PNG holds the
    /// gamma-encoded bytes, so it goes through the LDR sRGB decode path and
    /// the shader reads linear values. No mip chain: Babylon's lookup texture
    /// has none either, so the implicit-LOD sample always hits level 0.
    pub fn createBrdfLut(allocator: std.mem.Allocator) !Texture {
        const png = @embedFile("brdf_lut.png");
        var raw = try decodeMemory(allocator, png, .{ .gen_mipmaps = false, .srgb_to_linear = true });
        defer raw.deinit(allocator);
        return fromRaw(&raw, .{ .min_filter = .LINEAR, .mag_filter = .LINEAR });
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
                const alpha = color.particleDotAlpha(@intCast(x), @intCast(y), center, radius);

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
        const byte_count = try mip.checkedFaceBytes(size);
        const buffer = try allocator.alloc(u8, byte_count);
        defer allocator.free(buffer);

        const center: f32 = @as(f32, @floatFromInt(size)) * 0.5 - 0.5;
        const radius: f32 = @as(f32, @floatFromInt(size)) * 0.5;

        var y: u32 = 0;
        while (y < size) : (y += 1) {
            var x: u32 = 0;
            while (x < size) : (x += 1) {
                const alpha = color.particleDotAlpha(x, y, center, radius);

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

    /// Resolves a null Basis target against the live backend (BC7 when
    /// sampleable, else ASTC 4x4, then ETC2 RGBA8, else universal RGBA32). Without
    /// an sg context (worker threads, CLI tools) returns null: decode entries
    /// then use the desktop-first .bc7 default and the fromRawBlock backend
    /// gate reports unsupported backends explicitly. Snapshot per load on the
    /// context thread when the accurate answer matters (the glTF loader does).
    pub fn basisTargetForCurrentThread() ?ktx2.BasisTarget {
        if (!sg.isvalid()) return null;
        return ktx2.preferredBasisTarget(queryBlockSupport());
    }

    /// Decodes a KTX2 Basis file (ETC1S/UASTC, 2D) from memory and uploads
    /// it: block targets via fromRawBlock, the RGBA32 fallback via fromRaw.
    /// Main thread only (GPU upload); use ktx2.decodeBasis2D + fromRawBlock/
    /// fromRaw to split worker-thread transcode from main-thread upload. A
    /// null target snapshots the live backend (BC7 → ASTC → ETC2 → RGBA32); an
    /// explicit target forces it (unit tests, re-encode previews).
    /// options.srgb_to_linear is the RGBA-path decision (the block targets
    /// follow the file DFD, like every other .block upload).
    pub fn fromBasisMemory(
        allocator: std.mem.Allocator,
        bytes: []const u8,
        options: Options,
        target: ?ktx2.BasisTarget,
    ) !Texture {
        const t = target orelse basisTargetForCurrentThread() orelse .bc7;
        var dec = try ktx2.decodeBasis2D(allocator, bytes, t, .{
            .gen_mipmaps = options.mipmaps,
            .srgb_to_linear = options.srgb_to_linear,
        });
        defer dec.deinit(allocator);
        return switch (dec) {
            .rgba => |*raw| fromRaw(raw, options),
            .block => |*blk| try fromRawBlock(blk, options),
        };
    }

    /// File variant of `fromBasisMemory`. Main thread only (GPU upload); the
    /// buffered read matches decodeFile so the async UploadQueue
    /// (decodeImageFile → decodeImageMemory) sees identical bytes.
    pub fn fromBasisFile(
        allocator: std.mem.Allocator,
        file_path: []const u8,
        options: Options,
        target: ?ktx2.BasisTarget,
    ) !Texture {
        const io = std.Io.Threaded.global_single_threaded.io();
        const file = try std.Io.Dir.cwd().openFile(io, file_path, .{});
        defer file.close(io);

        const file_size = try file.length(io);
        const bytes = try allocator.alloc(u8, std.math.cast(usize, file_size) orelse return error.ImageTooLarge);
        defer allocator.free(bytes);

        const read = try file.readPositionalAll(io, bytes, 0);
        if (read < bytes.len) return error.ImageDecodeFailed;
        return fromBasisMemory(allocator, bytes, options, target);
    }

    /// Decodes a block-compressed DDS file (BC1/BC2/BC3/BC7, mip chain as
    /// authored) from memory and uploads it via fromRawBlock. Main thread
    /// only (GPU upload); use dds.decodeBlock2D + fromRawBlock to split
    /// worker-thread decode from main-thread upload. options.srgb_to_linear
    /// is the legacy sRGB decision (DX10 files carry their own tag):
    /// color-slot DDS textures pass true to select the sRGB GPU variant
    /// (BC3_SRGBA/BC7_SRGBA; BC1/BC2 have no sRGB variant and stay UNORM).
    /// glTF never references .dds (the spec has no such image MIME), so
    /// there is no glTF wiring — this is the standalone entry point.
    pub fn fromDdsMemory(allocator: std.mem.Allocator, bytes: []const u8, options: Options) !Texture {
        var raw = try dds.decodeBlock2D(allocator, bytes, .{ .srgb = options.srgb_to_linear });
        defer raw.deinit(allocator);
        return fromRawBlock(&raw, options);
    }

    /// File variant of `fromDdsMemory`. Main thread only (GPU upload); the
    /// buffered read matches decodeFile so the async UploadQueue
    /// (decodeImageFile → decodeImageMemory) sees identical bytes.
    pub fn fromDdsFile(allocator: std.mem.Allocator, file_path: []const u8, options: Options) !Texture {
        const io = std.Io.Threaded.global_single_threaded.io();
        const file = try std.Io.Dir.cwd().openFile(io, file_path, .{});
        defer file.close(io);

        const file_size = try file.length(io);
        const bytes = try allocator.alloc(u8, std.math.cast(usize, file_size) orelse return error.ImageTooLarge);
        defer allocator.free(bytes);

        const read = try file.readPositionalAll(io, bytes, 0);
        if (read < bytes.len) return error.ImageDecodeFailed;
        return fromDdsMemory(allocator, bytes, options);
    }

    /// Converts one f32 channel to an IEEE-754 half-precision bit pattern.
    /// Out-of-range magnitudes become half infinity, NaN stays NaN.
    /// Content above 65504 loses detail: tone-map before upload if it matters.
    pub fn floatToHalfBits(value: f32) u16 {
        return color.floatToHalfBits(value);
    }

    /// Converts an IEEE-754 half-precision bit pattern back to f32.
    /// Used by tests and debugging; the GPU upload path never needs it.
    pub fn halfBitsToFloat(bits: u16) f32 {
        return color.halfBitsToFloat(bits);
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

    /// Decodes an image to half-float RGBA: OpenEXR (magic sniff) via the
    /// engine's scanline reader, Radiance .hdr (and LDR formats upconverted
    /// to float) via stbi_loadf. Thread-safe; pair with `fromRawHdr`.
    /// Foreign or corrupt data yields error.ImageDecodeFailed, empty sizes
    /// error.InvalidDimensions. EXR validation errors surface as their own
    /// failures (never a silent stb fallback).
    pub fn decodeHDRMemory(allocator: std.mem.Allocator, bytes: []const u8) !RawHdrTexture {
        if (exr.sniff(bytes)) {
            const dec = try exr.decode(allocator, bytes);
            // Same layout (RGBA half-float bits); adopt without copying.
            return .{ .width = dec.width, .height = dec.height, .pixels = dec.pixels };
        }
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
    /// Buffered read so EXR files take the same sniff-routed reader as
    /// in-memory payloads (stb decodes the same bytes from memory).
    pub fn decodeHDRFile(allocator: std.mem.Allocator, file_path: []const u8) !RawHdrTexture {
        const io = std.Io.Threaded.global_single_threaded.io();
        // Missing/unreadable files keep the historical ImageDecodeFailed
        // (same observable contract as the old stbi_loadf path).
        const file = std.Io.Dir.cwd().openFile(io, file_path, .{}) catch return error.ImageDecodeFailed;
        defer file.close(io);

        const file_size = try file.length(io);
        const bytes = try allocator.alloc(u8, std.math.cast(usize, file_size) orelse return error.ImageTooLarge);
        defer allocator.free(bytes);

        const read = try file.readPositionalAll(io, bytes, 0);
        if (read < bytes.len) return error.ImageDecodeFailed;
        return decodeHDRMemory(allocator, bytes);
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

    /// Decodes an OpenEXR file from memory and uploads an RGBA16F texture.
    /// Main thread only (GPU upload); use exr.decode + fromRawHdr to split
    /// worker-thread decode from main-thread upload. EXR is linear, so the
    /// upload path marks the texture HDR (no sRGB decode on sample).
    pub fn fromExrMemory(allocator: std.mem.Allocator, bytes: []const u8, options: Options) !Texture {
        var dec = try exr.decode(allocator, bytes);
        defer dec.deinit(allocator);
        var raw = RawHdrTexture{ .width = dec.width, .height = dec.height, .pixels = dec.pixels };
        dec.pixels = &.{};
        defer raw.deinit(allocator);
        return fromRawHdr(&raw, options);
    }

    /// Loads an OpenEXR file and uploads an RGBA16F texture.
    /// Main thread only (GPU upload); use decodeHDRFile + fromRawHdr to
    /// split worker-thread decode from main-thread upload. Strict: foreign
    /// data fails with the EXR reader's own error (NotExr, ...).
    pub fn fromExrFile(allocator: std.mem.Allocator, file_path: []const u8, options: Options) !Texture {
        const io = std.Io.Threaded.global_single_threaded.io();
        const file = try std.Io.Dir.cwd().openFile(io, file_path, .{});
        defer file.close(io);

        const file_size = try file.length(io);
        const bytes = try allocator.alloc(u8, std.math.cast(usize, file_size) orelse return error.ImageTooLarge);
        defer allocator.free(bytes);

        const read = try file.readPositionalAll(io, bytes, 0);
        if (read < bytes.len) return error.ImageDecodeFailed;
        return fromExrMemory(allocator, bytes, options);
    }

    pub fn deinit(self: *Texture) void {
        sg.destroyView(self.view);
        sg.destroyImage(self.image);
        sg.destroySampler(self.sampler);
    }
};

// Minimal 1x1 Radiance .hdr with flat (non-RLE) RGBE data. Width < 8 takes
// the flat decode path in stb_image. RGBE {128,128,128,128} decodes to
// 128 * 2^(128-136) = 0.5 per channel, alpha forced to 1.0.
const hdr_1x1_flat: []const u8 = "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 1\n\x80\x80\x80\x80";

// Minimal 8x1 Radiance .hdr exercising the RLE scanline path: scanline
// header {2, 2, 0, 8} (width 8), then each of the 4 channels as one run of
// 8 pixels with value 128 (count byte 128+8 = 0x88). Same 0.5 gray texels.
const hdr_8x1_rle: []const u8 = "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 8\n\x02\x02\x00\x08\x88\x80\x88\x80\x88\x80\x88\x80";

// ---------------------------------------------------------------------------
// Minimal PNG fixture synthesis (no external dependencies): PNG container +
// zlib stream with STORED deflate blocks + Adler-32. stb_image decodes these
// like any real PNG.
// ---------------------------------------------------------------------------
pub const TestPng = struct {
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
    pub fn build(
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

test "downsampleLevel handles odd dimensions and averages correctly" {
    var src: [3 * 5 * 4]u8 = undefined;
    for (&src, 0..) |*b, i| b.* = @intCast(i % 251);
    var dst: [1 * 2 * 4]u8 = undefined;
    Texture.downsampleLevel(&src, 3, 5, &dst, 1, 2);
    // Top-left texel averages src bytes {0,4,12,16}: (0+4+12+16+2)/4 = 8.
    try std.testing.expectEqual(@as(u8, 8), dst[0]);
}

// ---------------------------------------------------------------------------
// Golden test: the 2D mip chain (levels, dimensions, averaged colors)
// ---------------------------------------------------------------------------

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
    mip.boxDownsampleU8(&src, 3, 3, &dst, 1, 1);
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
    mip.boxDownsampleU8(&edge_src, 2, 1, &edge_dst, 1, 1);
    // (10+20+10+20+2)>>2 = 15.
    try std.testing.expectEqual(@as(u8, 15), edge_dst[0]);
}
