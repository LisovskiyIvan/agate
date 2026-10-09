const std = @import("std");

const ktx2 = @import("ktx2.zig");

// ---------------------------------------------------------------------------
// DDS container reader — the block-compressed subset only (BC1/BC2/BC3/BC7).
//
// Supports (all little-endian):
//   - magic "DDS " plus the 124-byte DDS_HEADER.
//   - DX10 extended header (fourCC "DX10") only: BC1, BC2, BC3, BC7.
//     The DXGI format tag is authoritative and carries unambiguous sRGB semantics.
//     Legacy FourCC DXT1/DXT2/DXT3/DXT4/DXT5 headers are rejected as part of
//     modernization (audit item 12).
//   - full file-provided mip chains (dwMipMapCount levels, contiguous,
//     largest first). No CPU mip synthesis — like the KTX2 block path,
//     levels upload exactly as authored.
//
// Deliberately rejected with explicit errors (never a silent fallback):
//   - uncompressed/float/packed DDS (no fourCC, RGB bitmasks, DXT2/DXT4,
//     unknown DXGI codes) -> UnsupportedDdsFormat,
//   - cubemaps, arrays (arraySize != 1), volumes/depth -> UnsupportedCube /
//     UnsupportedArraySize / Unsupported3D,
//   - absurd dimensions (> 16384) -> UnsupportedDimensions,
//   - level counts that exceed the 16-slot engine storage (TooManyLevels)
//     or the full halving chain for the base size (InvalidMipData),
//   - payload sizes that do not match the block-grid math exactly
//     (short -> Truncated, trailing garbage -> InvalidMipData).
//
// Decodes into ktx2.RawBlockTexture (the BlockFormat enum carries the
// BC1/BC2/BC3 codes too), so Texture.fromRawBlock, DecodedImage.block and
// the async UploadQueue serve DDS with zero new thread paths: route with
// Texture.decodeImageMemory/decodeImageFile, upload with fromRawBlock.
//
// glTF never references .dds (the spec defines no DDS image MIME type), so
// there is no loader wiring — the standalone entry points are
// Texture.fromDdsMemory/fromDdsFile.
// ---------------------------------------------------------------------------

/// The 4-byte DDS magic: "DDS " (0x20534444 little-endian).
pub const magic = [4]u8{ 'D', 'D', 'S', ' ' };

/// DDS_HEADER size after the magic; the DX10 extension adds 20 bytes.
const header_size: usize = 124;
const dx10_size: usize = 20;
/// Largest accepted dimension (D3D11-class cap); bigger means corrupt.
const max_dimension: u32 = 16384;

pub const DecodeError = error{
    NotDds,
    Truncated,
    UnsupportedDdsFormat,
    UnsupportedDimensions,
    Unsupported3D,
    UnsupportedCube,
    UnsupportedArraySize,
    TooManyLevels,
    InvalidMipData,
    OutOfMemory,
};

pub const TextureColorSpace = enum {
    linear,
    srgb,
};

pub const TextureSlot = enum {
    color, // albedo, emissive, base color (sRGB)
    data, // normal, metallic_roughness, occlusion, height, lookup (linear)

    pub fn colorSpace(self: TextureSlot) TextureColorSpace {
        return switch (self) {
            .color => .srgb,
            .data => .linear,
        };
    }
};

/// Decode switches. `srgb` is the legacy sRGB decision: legacy
/// DXT1/DXT3/DXT5 headers carry no color-space tag, so true (or slot = .color
/// or color_space = .srgb) selects the sRGB GPU variant where one exists
/// (DXT5 -> BC3_SRGBA; DXT1/DXT3 stay UNORM — sokol has no BC1_SRGBA/BC2_SRGBA).
/// DX10 files ignore it: the DXGI _SRGB code is authoritative.
pub const DecodeOptions = struct {
    srgb: bool = false,
    color_space: ?TextureColorSpace = null,
    slot: ?TextureSlot = null,

    pub fn isSrgb(self: DecodeOptions) bool {
        if (self.color_space) |cs| return cs == .srgb;
        if (self.slot) |s| return s.colorSpace() == .srgb;
        return self.srgb;
    }
};

// DDS_HEADER dwFlags bits.
pub const flag_caps: u32 = 0x1;
pub const flag_height: u32 = 0x2;
pub const flag_width: u32 = 0x4;
pub const flag_pixelformat: u32 = 0x1000;
pub const flag_mipmapcount: u32 = 0x20000;
pub const flag_depth: u32 = 0x800000;
pub const required_flags: u32 = flag_caps | flag_height | flag_width | flag_pixelformat;

// DDS_PIXELFORMAT dwFlags bits.
pub const pf_fourcc: u32 = 0x4;

// DDS_HEADER dwCaps bits.
pub const caps_texture: u32 = 0x1000;
// DDS_HEADER dwCaps2 bits: cubemap face mask + volume.
pub const caps2_cubemap_mask: u32 = 0xFE00;
pub const caps2_volume: u32 = 0x200000;

// Legacy fourCC codes (u32 little-endian of the ASCII tag).
pub const fourcc_dxt1: u32 = 0x31545844;
pub const fourcc_dxt2: u32 = 0x32545844;
pub const fourcc_dxt3: u32 = 0x33545844;
pub const fourcc_dxt4: u32 = 0x34545844;
pub const fourcc_dxt5: u32 = 0x35545844;
pub const fourcc_dx10: u32 = 0x30315844;

// DXGI_FORMAT codes (DXGI 1.0) understood by the DX10 path.
pub const dxgi_bc1_unorm: u32 = 71;
pub const dxgi_bc1_unorm_srgb: u32 = 72;
pub const dxgi_bc2_unorm: u32 = 74;
pub const dxgi_bc2_unorm_srgb: u32 = 75;
pub const dxgi_bc3_unorm: u32 = 77;
pub const dxgi_bc3_unorm_srgb: u32 = 78;
pub const dxgi_bc7_unorm: u32 = 98;
pub const dxgi_bc7_unorm_srgb: u32 = 99;

// DDS_HEADER_DXT10 resourceDimension values.
pub const dx10_dimension_texture2d: u32 = 3;
// DDS_HEADER_DXT10 miscFlag bits.
pub const dx10_misc_texturecube: u32 = 0x4;

/// True when `bytes` starts with the DDS magic. Cheap guard used to route
/// assets between the DDS reader, the KTX2 reader and stb_image.
pub fn sniff(bytes: []const u8) bool {
    return bytes.len >= magic.len and std.mem.eql(u8, bytes[0..magic.len], &magic);
}

fn readU32(bytes: []const u8, offset: usize) u32 {
    return std.mem.readInt(u32, bytes[offset..][0..4], .little);
}

/// Parsed 2D layout: the block format, the base size, the authored level
/// count and the absolute file offset of the first level.
const Layout = struct {
    format: ktx2.BlockFormat,
    width: u32,
    height: u32,
    num_levels: u32,
    data_offset: usize,
};

/// Full halving chain length for a base size (1x1-terminated, capped at
/// 16 like Texture.mipLevelCount). Local copy: the reader must not depend
/// on texture.zig (which imports this module for the upload path).
fn fullChainLen(width: u32, height: u32) u32 {
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

/// Maps a DX10 dxgiFormat code onto the block subset. BC1/BC2 _SRGB codes
/// collapse onto UNORM: sokol exposes no BC1_SRGBA/BC2_SRGBA variant, so
/// there is nothing to select (documented, not silent: the returned format
/// is exact).
fn blockFormatFromDxgi(dxgi_format: u32) DecodeError!ktx2.BlockFormat {
    return switch (dxgi_format) {
        dxgi_bc1_unorm, dxgi_bc1_unorm_srgb => .bc1_unorm,
        dxgi_bc2_unorm, dxgi_bc2_unorm_srgb => .bc2_unorm,
        dxgi_bc3_unorm => .bc3_unorm,
        dxgi_bc3_unorm_srgb => .bc3_srgb,
        dxgi_bc7_unorm => .bc7_unorm,
        dxgi_bc7_unorm_srgb => .bc7_srgb,
        else => error.UnsupportedDdsFormat,
    };
}

fn parseHeader(bytes: []const u8, opts: DecodeOptions) DecodeError!Layout {
    if (!sniff(bytes)) return error.NotDds;
    if (bytes.len < magic.len + header_size) return error.Truncated;

    const base = magic.len;
    if (readU32(bytes, base + 0) != header_size) return error.InvalidMipData;
    const flags = readU32(bytes, base + 4);
    if (flags & required_flags != required_flags) return error.InvalidMipData;
    const height = readU32(bytes, base + 8);
    const width = readU32(bytes, base + 12);
    if (flags & flag_depth != 0) return error.Unsupported3D;
    if (readU32(bytes, base + 20) > 1) return error.Unsupported3D;
    if (width == 0 or height == 0 or width > max_dimension or height > max_dimension) {
        return error.UnsupportedDimensions;
    }

    // Pixel format sub-struct at header offset 72 (32 bytes).
    const pf = base + 72;
    if (readU32(bytes, pf + 0) != 32) return error.InvalidMipData;
    if (readU32(bytes, pf + 4) & pf_fourcc == 0) return error.UnsupportedDdsFormat;
    const fourcc = readU32(bytes, pf + 8);
    if (fourcc != fourcc_dx10) {
        // Legacy FourCC DXT formats (DXT1..DXT5) are rejected.
        // Modern engine contract mandates DX10 extended headers carrying unambiguous DXGI formats.
        return error.UnsupportedDdsFormat;
    }
    _ = opts;

    if (bytes.len < magic.len + header_size + dx10_size) return error.Truncated;
    const dx = base + header_size;
    if (readU32(bytes, dx + 4) != dx10_dimension_texture2d) return error.UnsupportedDdsFormat;
    if (readU32(bytes, dx + 8) & dx10_misc_texturecube != 0) return error.UnsupportedCube;
    if (readU32(bytes, dx + 12) != 1) return error.UnsupportedArraySize;
    const data_offset: usize = base + header_size + dx10_size;
    const format = try blockFormatFromDxgi(readU32(bytes, dx + 0));

    if (readU32(bytes, base + 104) & caps_texture == 0) return error.InvalidMipData;
    const caps2 = readU32(bytes, base + 108);
    if (caps2 & caps2_cubemap_mask != 0) return error.UnsupportedCube;
    if (caps2 & caps2_volume != 0) return error.Unsupported3D;

    const mip_count = readU32(bytes, base + 24);
    const num_levels: u32 = if (flags & flag_mipmapcount != 0) @max(1, mip_count) else 1;
    if (num_levels > 16) return error.TooManyLevels;
    // Overlong chains (past the 1x1-terminated halving chain) are corrupt:
    // they would upload duplicate 1x1 levels past the smallest real mip.
    if (num_levels > fullChainLen(width, height)) return error.InvalidMipData;

    // Tight size check: the payload must hold exactly the block-grid bytes
    // of every authored level — short files are truncated, trailing bytes
    // are garbage, both rejected.
    var expected: u64 = 0;
    for (0..num_levels) |m| {
        const w: u32 = @max(1, width >> @intCast(m));
        const h: u32 = @max(1, height >> @intCast(m));
        const level_bytes = format.levelByteSize(w, h) orelse return error.InvalidMipData;
        expected = std.math.add(u64, expected, level_bytes) catch return error.InvalidMipData;
    }
    const have: u64 = @as(u64, bytes.len) - data_offset;
    if (have < expected) return error.Truncated;
    if (have > expected) return error.InvalidMipData;

    return .{
        .format = format,
        .width = width,
        .height = height,
        .num_levels = num_levels,
        .data_offset = data_offset,
    };
}

/// Reads a 2D block-compressed DDS file into owned per-level slices.
/// GPU-free and thread-safe. No mip synthesis: levels upload exactly as
/// authored via Texture.fromRawBlock. Copies (not views): the source
/// buffer is freed right after decode by the asset queue.
pub fn decodeBlock2D(allocator: std.mem.Allocator, bytes: []const u8, opts: DecodeOptions) DecodeError!ktx2.RawBlockTexture {
    const layout = try parseHeader(bytes, opts);

    var raw = ktx2.RawBlockTexture{
        .width = layout.width,
        .height = layout.height,
        .num_levels = layout.num_levels,
        .format = layout.format,
    };
    errdefer raw.deinit(allocator);

    var cursor: usize = layout.data_offset;
    for (0..layout.num_levels) |m| {
        const w: u32 = @max(1, layout.width >> @intCast(m));
        const h: u32 = @max(1, layout.height >> @intCast(m));
        // levelByteSize already validated in parseHeader; unreachable on
        // mismatch (checked arithmetic on u32-sized dims cannot fail here).
        const len: usize = @intCast(layout.format.levelByteSize(w, h) orelse return error.InvalidMipData);
        const owned = try allocator.alloc(u8, len);
        @memcpy(owned, bytes[cursor .. cursor + len]);
        raw.levels[m] = owned;
        cursor += len;
    }
    return raw;
}
