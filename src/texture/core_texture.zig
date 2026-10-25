//! Owner of the `Texture` type: fields, lifecycle, and GPU upload entries.
//! Split out of `core.zig` (now a facade); the CPU-side concerns moved to
//! leaves and are aliased back into the `Texture` namespace below, so every
//! `Texture.X` path resolves exactly as before.
//!
//! Layout: `core_sampler.zig` (Options + sampler helpers),
//! `core_raw.zig` (RawTexture + mip-chain builders), `core_block.zig`
//! (BlockSupport + format mapping), `core_decode.zig` (LDR decode routing),
//! `core_hdr.zig` (HDR decode), `core_testpng.zig` (TestPng fixture).
//! This module never imports the `texture.zig` facade back (same anti-cycle
//! rule as `camera/`, `profiler/`, `ui/`).
//!
//! The `Texture` type cannot span files: methods that construct or consume
//! GPU handles (`initRaw*`, `fromRaw*`, `from*`, factories, `deinit`) stay
//! here; GPU-free namespace members live in the leaves above.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const ktx2 = @import("../ktx2.zig");
const dds = @import("../dds.zig");
const exr = @import("../exr.zig");
const color = @import("color.zig");
const mip = @import("mip.zig");
const gpu_thread = @import("../gpu_thread.zig");
const sampler_mod = @import("core_sampler.zig");
const raw_mod = @import("core_raw.zig");
const block_mod = @import("core_block.zig");
const decode_mod = @import("core_decode.zig");
const hdr_mod = @import("core_hdr.zig");

pub const Texture = struct {
    pub const TextureColorSpace = dds.TextureColorSpace;
    pub const TextureSlot = dds.TextureSlot;
    pub const ColorSpace = dds.TextureColorSpace;
    pub const Slot = dds.TextureSlot;

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
    /// Render-target borrows share handles but never destroy their owner.
    owns_handles: bool = true,

    // Re-exported leaf members: the `Texture` type cannot span files, so
    // every namespace member that moved to a `core_*` leaf is aliased back
    // here. Both `Texture.X` and `tex.x()` call forms keep working unchanged.
    pub const Options = sampler_mod.Options;
    pub const effectiveAnisotropy = sampler_mod.effectiveAnisotropy;
    pub const RawTexture = raw_mod.RawTexture;
    pub const buildRaw = raw_mod.buildRaw;
    pub const buildRawSrgbFused = raw_mod.buildRawSrgbFused;
    pub const downsampleLevel = raw_mod.downsampleLevel;
    pub const mipLevelCount = raw_mod.mipLevelCount;
    pub const DecodeOptions = decode_mod.DecodeOptions;
    pub const DecodedImage = decode_mod.DecodedImage;
    pub const decodeMemory = decode_mod.decodeMemory;
    pub const decodeFile = decode_mod.decodeFile;
    pub const decodeImageMemory = decode_mod.decodeImageMemory;
    pub const decodeImageFile = decode_mod.decodeImageFile;
    pub const BlockSupport = block_mod.BlockSupport;
    pub const sgPixelFormatForBlock = block_mod.sgPixelFormatForBlock;
    pub const queryBlockSupport = block_mod.queryBlockSupport;
    pub const RawHdrTexture = hdr_mod.RawHdrTexture;
    pub const floatToHalfBits = hdr_mod.floatToHalfBits;
    pub const halfBitsToFloat = hdr_mod.halfBitsToFloat;
    pub const decodeHDRMemory = hdr_mod.decodeHDRMemory;
    pub const decodeHDRFile = hdr_mod.decodeHDRFile;
    pub const RawHdrLevels = ktx2.RawHdrLevels;

    const makeSamplerFor = sampler_mod.makeSamplerFor;

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

    /// initRaw plus a full CPU mipmap chain. Uploads immediately on the
    /// calling (main) thread; use decodeMemory + fromRaw for parallel loads.
    pub fn initRawMipped(allocator: std.mem.Allocator, width: u32, height: u32, rgba_pixels: []const u8, options: Options) !Texture {
        if (mipLevelCount(width, height) == 1) return initRaw(width, height, rgba_pixels, options);

        var raw = try buildRaw(allocator, width, height, rgba_pixels, true);
        defer raw.deinit(allocator);
        return fromRaw(&raw, options);
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
        return fromRaw(&raw, brdf_lut_options);
    }

    /// Sampler options for `createBrdfLut`. CLAMP, not the engine default
    /// REPEAT: Babylon's lookup texture is sampled with clamp semantics, and
    /// at v = 1.0 (roughness exactly 1.0) REPEAT blends the last row with the
    /// first, which under-reported `coloredEnergyConservationFactor`
    /// (measured g ~ 0.68 instead of ~ 0.44 — PROBE.md 10.9). Named so a test
    /// can pin it: the wrap mode is not readable from the `sg.Sampler` handle.
    pub const brdf_lut_options: Options = .{
        .min_filter = .LINEAR,
        .mag_filter = .LINEAR,
        .wrap_u = .CLAMP_TO_EDGE,
        .wrap_v = .CLAMP_TO_EDGE,
    };

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
            .srgb_to_linear = options.isColorSlot(),
            .color_space = options.color_space,
            .slot = options.slot,
        });
        defer raw.deinit(allocator);
        return fromRaw(&raw, options);
    }

    pub fn fromFile(allocator: std.mem.Allocator, file_path: []const u8, options: Options) !Texture {
        var raw = try decodeFile(allocator, file_path, .{
            .gen_mipmaps = options.mipmaps,
            .srgb_to_linear = options.isColorSlot(),
            .color_space = options.color_space,
            .slot = options.slot,
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
            .srgb_to_linear = options.isColorSlot(),
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
        var raw = try dds.decodeBlock2D(allocator, bytes, .{
            .srgb = options.isColorSlot(),
            .color_space = options.color_space,
            .slot = options.slot,
        });
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

    /// Creates the GPU image from CPU-decoded half-float mip levels (e.g. from KTX2 HDR).
    /// Main thread only. Uploads RGBA16F with LINEAR min/mag/mip and configured wrap.
    pub fn fromRawHdrLevels(raw: *const ktx2.RawHdrLevels, options: Options) Texture {
        std.debug.assert(raw.num_levels >= 1 and raw.num_levels <= 16);
        var img_desc = sg.ImageDesc{
            .width = @intCast(raw.width),
            .height = @intCast(raw.height),
            .pixel_format = .RGBA16F,
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
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
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
            .num_mipmaps = raw.num_levels,
            .format = .RGBA16F,
            .is_hdr = true,
        };
    }

    /// Decodes an uncompressed float HDR KTX2 file from memory and uploads an RGBA16F texture.
    /// Main thread only (GPU upload); use ktx2.decodeHdr2D + fromRawHdrLevels to split
    /// worker-thread decode from main-thread upload.
    pub fn fromKtx2HdrMemory(allocator: std.mem.Allocator, bytes: []const u8, options: Options) !Texture {
        var raw = try ktx2.decodeHdr2D(allocator, bytes);
        defer raw.deinit(allocator);
        return fromRawHdrLevels(&raw, options);
    }

    /// File variant of `fromKtx2HdrMemory`.
    pub fn fromKtx2HdrFile(allocator: std.mem.Allocator, file_path: []const u8, options: Options) !Texture {
        const io = std.Io.Threaded.global_single_threaded.io();
        const file = try std.Io.Dir.cwd().openFile(io, file_path, .{});
        defer file.close(io);

        const file_size = try file.length(io);
        const bytes = try allocator.alloc(u8, std.math.cast(usize, file_size) orelse return error.ImageTooLarge);
        defer allocator.free(bytes);

        const read = try file.readPositionalAll(io, bytes, 0);
        if (read < bytes.len) return error.ImageDecodeFailed;
        return fromKtx2HdrMemory(allocator, bytes, options);
    }

    /// Decodes HDR image from memory (Radiance .hdr, OpenEXR, or KTX2 HDR) and uploads an RGBA16F texture.
    /// Main thread only (GPU upload); use decodeHDRMemory + fromRawHdr to
    /// split worker-thread decode from main-thread upload.
    pub fn fromHDRMemory(allocator: std.mem.Allocator, bytes: []const u8, options: Options) !Texture {
        if (ktx2.isHdrKtx2(bytes)) {
            return fromKtx2HdrMemory(allocator, bytes, options);
        }
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
        if (!self.owns_handles) return;
        // Context-thread only when a destroy will fire (debug tripwire;
        // release unchanged): handle-less cleanup stays legal off-context.
        // Legal destroys additionally require prepared frames to have
        // stopped borrowing these handles (same rule as Scene.deinit).
        if (self.image.id != 0 or self.view.id != 0 or self.sampler.id != 0) {
            gpu_thread.assertOnContextThreadDebug();
        }
        sg.destroyView(self.view);
        sg.destroyImage(self.image);
        sg.destroySampler(self.sampler);
    }
};
