const std = @import("std");

const c = @import("c.zig").c;
const texture = @import("texture.zig");
const Texture = texture.Texture;
const CubeTexture = texture.CubeTexture;

// ---------------------------------------------------------------------------
// KTX2 container reader — the honest uncompressed subset, a no-transcode
// block-compressed subset (BC7 / ASTC 4x4 / ETC2 RGBA8), and a REAL Basis Universal
// transcode subset (ETC1S/BasisLZ + UASTC LDR 4x4 via the official transcoder
// vendored under src/agate/c/basisu/, Apache-2.0; see LICENSES.md).
//
// Supports (KTX2 spec v2.0, all little-endian):
//   - supercompressionScheme 0 (NONE) only for the two paths below. BasisLZ
//     (1) routes to the Basis transcoder; Zstandard (2) and ZLIB (3) route
//     there too when vkFormat is UNDEFINED (UASTC supercompression — the
//     vendored zstd decoder handles it). Scheme > 3 errors with
//     UnsupportedSupercompression.
//   - 8-bit UNORM/SRGB formats that map onto the engine's RGBA8 LDR upload
//     path: R8, R8G8, R8G8B8A8, B8G8R8A8, A8B8G8R8_PACK32 (same LE byte
//     order as R8G8B8A8). Everything else uncompressed (16F/32F, packed
//     16-bit, BC4-BC6, ETC1/EAC, other ASTC footprints) errors with
//     UnsupportedVkFormat.
//   - block-compressed BC1_UNORM (vk 133), BC2_UNORM (135),
//     BC3_UNORM/SRGB (vk 137/138), BC7_UNORM/SRGB (vk 145/146) and
//     ASTC_4x4_UNORM/SRGB (vk 157/158), ETC2_RGBA8_UNORM/SRGB (vk 151/152), uploaded WITHOUT decoding:
//     decodeBlock2D returns owned per-level slices for
//     Texture.fromRawBlock. No CPU mip synthesis (a decoder/encoder pair
//     would be a new dependency). Cube block files are rejected — only 2D.
//     The BC1/BC2/BC3 codes also serve the DDS reader (dds.zig), which
//     decodes into the same RawBlockTexture.
//   - Basis (vkFormat UNDEFINED == 0): ETC1S (BasisLZ supercompression) and
//     UASTC LDR 4x4, transcoded to a caller-chosen target — BC7 (desktop),
//     ASTC 4x4, ETC2 RGBA8, or RGBA32 (universal CPU fallback) — with the full
//     file-authored mip chain, per level, 2D only. sRGB follows the file DFD
//     for block targets (the sRGB GPU variant, like the no-transcode path)
//     and the caller's srgb_to_linear decision for RGBA32.
//   - full file-provided mip chains (level index, largest level first) and
//     optional chain generation for single-level files (gen_mipmaps, RGBA8
//     only — block levels upload exactly as authored).
//   - cube maps (faceCount 6, faces +X,-X,+Y,-Y,+Z,-Z — the engine's order),
//     RGBA8 only.
//
// Deliberately NOT supported (explicit errors, never silent corruption):
//   - Basis HDR/XUASTC/ASTC-LDR/XUBC7 kinds, ETC1S video (P-frames), Basis
//     cubes/arrays/3D, levelCount 0/17+ → BasisUnsupported (cubes additionally
//     hit UnsupportedFaceCount on the 2D-only entry points, matching the
//     no-transcode path). BC1/BC3/BC5 transcode targets remain omitted because
//     BC7 supersedes BC1/BC3 on the desktop-first backends; ETC2 RGBA8 is kept
//     as a mobile compressed fallback when ASTC is unavailable. The RGBA32
//     fallback always uploads on backends with no compressed target.
//   - the data format descriptor (DFD) and the key/value data are skipped by
//     offset on the non-Basis paths — the numeric vkFormat alone drives the
//     texel interpretation, which is exact for the supported subset. KTX2
//     stores no orientation; glTF/KTX2 assets are top-left like the engine's
//     other LDR loaders, so rows upload unflipped. The Basis path lets the
//     official transcoder parse DFD/KVD itself (notably the sRGB transfer
//     function).
//
// NOTE on "DXGI codes": the KTX2 header carries only vkFormat (offset 12);
// there is no DXGI field to read. The SRGB block variants (146/152/158) are
// the BC7, ETC2 RGBA8, and ASTC sRGB encodings.
// ---------------------------------------------------------------------------

/// The 12-byte KTX2 identifier: «KTX 20» + CR LF ^Z LF.
pub const magic = [12]u8{ 0xAB, 'K', 'T', 'X', ' ', '2', '0', 0xBB, 0x0D, 0x0A, 0x1A, 0x0A };

/// Fixed prefix of the file: header (48 bytes incl. identifier) + index
/// (32 bytes: dfd/kvd u32 pairs, sgd u64 pair) = 80, then the level index.
pub const header_and_index_size: usize = 80;
pub const level_index_entry_size: usize = 24;

/// Supported texture supercompression schemes; must stay 0.
const scheme_none: u32 = 0;

pub const DecodeError = error{
    NotKtx2,
    Truncated,
    UnsupportedSupercompression,
    UnsupportedVkFormat,
    UnsupportedTypeSize,
    UnsupportedDimensions,
    Unsupported3D,
    UnsupportedLayers,
    UnsupportedFaceCount,
    TooManyLevels,
    InvalidLevelData,
    OutOfMemory,
    /// Payload carries the Basis marker (vkFormat UNDEFINED) but the official
    /// transcoder rejects the header (foreign/corrupt container).
    NotBasisKtx2,
    /// A valid Basis file outside the transcoded subset: HDR/XUASTC/ASTC-LDR
    /// kinds, ETC1S video, cube/array/3D, levelCount 0/17+. Re-encode the
    /// asset (ETC1S/UASTC LDR 2D).
    BasisUnsupported,
    /// The transcoder accepted the file but a level failed to transcode
    /// (truncated/corrupt level data).
    BasisTranscodeFailed,
    /// Payload exceeds the transcoder's 32-bit size limit.
    FileTooLarge,
};

/// The engine-supported vkFormat subset. Values are the Vulkan enum numbers.
pub const Format = enum(u32) {
    r8_unorm = 9,
    r8_srgb = 15,
    rg8_unorm = 16,
    rg8_srgb = 22,
    rgba8_unorm = 37,
    rgba8_srgb = 43,
    bgra8_unorm = 44,
    bgra8_srgb = 50,
    abgr8_unorm_pack32 = 51,
    abgr8_srgb_pack32 = 57,

    /// Bytes per texel as stored in the file.
    pub fn texelBlockSize(self: Format) usize {
        return switch (self) {
            .r8_unorm, .r8_srgb => 1,
            .rg8_unorm, .rg8_srgb => 2,
            else => 4,
        };
    }

    /// True for the _SRGB variants: the file's color channels carry sRGB
    /// values (alpha is always linear).
    pub fn isSrgb(self: Format) bool {
        return switch (self) {
            .r8_srgb, .rg8_srgb, .rgba8_srgb, .bgra8_srgb, .abgr8_srgb_pack32 => true,
            else => false,
        };
    }
};

/// Maps a Vulkan vkFormat number onto the supported subset.
pub fn formatFromVk(vk_format: u32) ?Format {
    inline for (@typeInfo(Format).@"enum".fields) |field| {
        if (vk_format == field.value) return @enumFromInt(vk_format);
    }
    return null;
}

/// Block-compressed formats uploaded WITHOUT decoding (no transcoder
/// dependency). Vulkan enum numbers (Vulkan registry):
///   133 = VK_FORMAT_BC1_RGBA_UNORM_BLOCK (DXT1, 4x4, 8 B/block),
///   135 = VK_FORMAT_BC2_UNORM_BLOCK (DXT3, 4x4, 16 B/block),
///   137/138 = VK_FORMAT_BC3_UNORM_BLOCK / _SRGB_BLOCK (DXT5, 4x4, 16 B/block),
///   145/146 = VK_FORMAT_BC7_UNORM_BLOCK / _SRGB_BLOCK (4x4, 16 B/block),
///   151/152 = VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK / _SRGB_BLOCK (4x4, 16 B/block),
///   157/158 = VK_FORMAT_ASTC_4x4_UNORM_BLOCK / _SRGB_BLOCK (4x4, 16 B/block).
/// BC1/BC2 have no sRGB GPU variant in this sokol checkout (only BC3_SRGBA
/// and BC7_SRGBA exist), so their _SRGB Vulkan/DXGI counterparts upload as
/// UNORM — documented on the DDS side (dds.zig), which shares this enum.
/// BC4/BC5/BC6 stay out of scope (no engine use case yet).
pub const BlockFormat = enum(u32) {
    bc1_unorm = 133,
    bc2_unorm = 135,
    bc3_unorm = 137,
    bc3_srgb = 138,
    bc7_unorm = 145,
    bc7_srgb = 146,
    etc2_rgba8_unorm = 151,
    etc2_rgba8_srgb = 152,
    astc_4x4_unorm = 157,
    astc_4x4_srgb = 158,

    /// Texel footprint of one compression block (all families are 4x4).
    pub fn blockExtent(self: BlockFormat) struct { w: u32, h: u32 } {
        _ = self;
        return .{ .w = 4, .h = 4 };
    }

    /// Storage bytes of one compression block (BC1 packs two texels per
    /// byte: 8 B; every other family is 16 B).
    pub fn blockByteSize(self: BlockFormat) usize {
        return switch (self) {
            .bc1_unorm => 8,
            else => 16,
        };
    }

    /// True for the _SRGB variants: sampling must go through the sRGB GPU
    /// format (hardware converts to linear); no CPU conversion is possible
    /// without a decoder.
    pub fn isSrgb(self: BlockFormat) bool {
        return switch (self) {
            .bc3_srgb, .bc7_srgb, .etc2_rgba8_srgb, .astc_4x4_srgb => true,
            else => false,
        };
    }

    /// Exact tightly-packed byte size of one 2D level: ceil-divided block
    /// grid times the block size. Checked arithmetic: absurd dimensions
    /// (level size not representable in u64) report null and the caller
    /// rejects the level as InvalidLevelData.
    pub fn levelByteSize(self: BlockFormat, width: u32, height: u32) ?u64 {
        const ext = self.blockExtent();
        const bw: u64 = (@as(u64, width) + ext.w - 1) / ext.w;
        const bh: u64 = (@as(u64, height) + ext.h - 1) / ext.h;
        const blocks = std.math.mul(u64, bw, bh) catch return null;
        return std.math.mul(u64, blocks, self.blockByteSize()) catch null;
    }
};

/// Maps a Vulkan vkFormat number onto the supported block subset.
pub fn blockFormatFromVk(vk_format: u32) ?BlockFormat {
    inline for (@typeInfo(BlockFormat).@"enum".fields) |field| {
        if (vk_format == field.value) return @enumFromInt(vk_format);
    }
    return null;
}

/// True when `bytes` starts with the KTX2 identifier. Cheap guard used to
/// route assets between the KTX2 reader and stb_image.
pub fn sniff(bytes: []const u8) bool {
    return bytes.len >= magic.len and std.mem.eql(u8, bytes[0..magic.len], &magic);
}

/// True when `bytes` look like a BLOCK-compressed KTX2 (identifier plus a
/// vkFormat from the BlockFormat subset). Routing-only: supercompression,
/// dimensions and level data are validated later by decodeBlock2D, so a
/// `true` here never implies uploadability.
pub fn isBlockKtx2(bytes: []const u8) bool {
    if (!sniff(bytes)) return false;
    if (bytes.len < header_and_index_size) return false;
    return blockFormatFromVk(readU32(bytes, 12)) != null;
}

/// Decode switches. `srgb_to_linear: null` = auto: convert exactly the
/// _SRGB formats (the format tag knows better than the caller). Explicit
/// true/false force/forbid the conversion for any format, matching
/// Texture.DecodeOptions semantics (the glTF loader passes its per-slot
/// color/data flag straight through, so data slots never auto-convert).
pub const DecodeOptions = struct {
    /// Only applies when the FILE has a single level: a file-provided mip
    /// chain is always uploaded as authored (generating on top of it would
    /// silently discard authoring data).
    gen_mipmaps: bool = true,
    srgb_to_linear: ?bool = null,
};

/// Decoded but GPU-free pixel data, normalized to the engine's LDR upload
/// formats. 2D output reuses Texture.RawTexture (RGBA8 levels, largest
/// first); cube output reuses CubeTexture.RawCubeMips (six faces
/// concatenated per level, face order +X,-X,+Y,-Y,+Z,-Z).
///
/// Parse outline per the spec: 80-byte header+index, `levels[max(1,
/// levelCount)]` 24-byte level-index entries (byteOffset/byteLength/
/// uncompressedByteLength), then mipPadding-aligned level payloads. Rows
/// are tightly packed (unpack alignment 1) and faces are contiguous within
/// a level, so every level is exactly w*h*texelBlockSize*(faces) bytes and
/// the face stride is w*h*texelBlockSize.
fn parseHeader(bytes: []const u8) DecodeError!Header {
    const h = try parseHeaderFields(bytes);
    if (formatFromVk(h.vk_format) == null) return error.UnsupportedVkFormat;
    return h;
}

/// Header + index validation shared by the RGBA8 and block paths: every
/// check EXCEPT the format-subset gate, in the same order, so both paths
/// report identical errors for malformed containers (notably
/// UnsupportedSupercompression precedes UnsupportedVkFormat).
fn parseHeaderFields(bytes: []const u8) DecodeError!Header {
    if (!sniff(bytes)) return error.NotKtx2;
    if (bytes.len < header_and_index_size) return error.Truncated;

    const h = Header{
        .vk_format = readU32(bytes, 12),
        .type_size = readU32(bytes, 16),
        .pixel_width = readU32(bytes, 20),
        .pixel_height = readU32(bytes, 24),
        .pixel_depth = readU32(bytes, 28),
        .layer_count = readU32(bytes, 32),
        .face_count = readU32(bytes, 36),
        .level_count = readU32(bytes, 40),
        .supercompression_scheme = readU32(bytes, 44),
    };

    if (h.supercompression_scheme != scheme_none) return error.UnsupportedSupercompression;
    if (h.type_size != 1) return error.UnsupportedTypeSize; // byte-typed incl. block formats
    if (h.pixel_width == 0 or h.pixel_height == 0) return error.UnsupportedDimensions;
    if (h.pixel_depth > 1) return error.Unsupported3D;
    if (h.layer_count > 1) return error.UnsupportedLayers;
    if (h.face_count != 1 and h.face_count != 6) return error.UnsupportedFaceCount;
    // levelCount 0 means one level (the index holds max(1, levelCount)
    // entries); > 16 does not fit the engine's mip storage.
    if (h.level_count > 16) return error.TooManyLevels;
    return h;
}

const Header = struct {
    vk_format: u32,
    type_size: u32,
    pixel_width: u32,
    pixel_height: u32,
    pixel_depth: u32,
    layer_count: u32,
    face_count: u32,
    level_count: u32,
    supercompression_scheme: u32,

    fn effectiveLevelCount(self: Header) u32 {
        return @max(1, self.level_count);
    }
};

fn readU32(bytes: []const u8, offset: usize) u32 {
    return std.mem.readInt(u32, bytes[offset..][0..4], .little);
}

fn readU64(bytes: []const u8, offset: usize) u64 {
    return std.mem.readInt(u64, bytes[offset..][0..8], .little);
}

/// One validated level: its payload slice plus the level dimensions.
const Level = struct {
    data: []const u8,
    width: u32,
    height: u32,
};

fn levelDims(width: u32, height: u32, level: u32) struct { w: u32, h: u32 } {
    return .{
        .w = @max(1, width >> @intCast(level)),
        .h = @max(1, height >> @intCast(level)),
    };
}

/// Validates every level-index entry against the file bounds and the exact
/// uncompressed size (scheme NONE: byteLength == uncompressedByteLength ==
/// w*h*texelBlockSize*faceCount; faces are tightly packed).
fn parseLevels(allocator: std.mem.Allocator, bytes: []const u8, header: Header, format: Format) DecodeError![]Level {
    const level_count = header.effectiveLevelCount();
    const index_end = header_and_index_size + level_count * level_index_entry_size;
    if (bytes.len < index_end) return error.Truncated;

    const face_count: usize = header.face_count;
    const levels = try allocator.alloc(Level, level_count);
    errdefer allocator.free(levels);

    for (0..level_count) |m| {
        const entry = header_and_index_size + m * level_index_entry_size;
        const byte_offset = readU64(bytes, entry);
        const byte_length = readU64(bytes, entry + 8);
        const uncompressed_length = readU64(bytes, entry + 16);

        const dims = levelDims(header.pixel_width, header.pixel_height, @intCast(m));
        const expected: u64 = @as(u64, dims.w) * dims.h * format.texelBlockSize() * face_count;
        if (byte_length != uncompressed_length or uncompressed_length != expected) {
            return error.InvalidLevelData;
        }
        const start: usize = std.math.cast(usize, byte_offset) orelse return error.InvalidLevelData;
        const len: usize = std.math.cast(usize, byte_length) orelse return error.InvalidLevelData;
        if (start < index_end or @as(u64, start) + byte_length > bytes.len) return error.Truncated;
        levels[m] = .{ .data = bytes[start .. start + len], .width = dims.w, .height = dims.h };
    }
    return levels;
}

// ---------------------------------------------------------------------------
// Block-compressed upload without decoding: per-level slices for
// Texture.fromRawBlock. Levels are tightly packed block grids
// (ceil(w/4) x ceil(h/4) x 16 B); the level index byteOffset values are
// absolute file offsets, validated the same way as the RGBA8 path.
// ---------------------------------------------------------------------------

/// Owned per-level block payloads (copies, not views: the source buffer is
/// freed right after decode by the asset queue). Pair with
/// Texture.fromRawBlock on the main thread. 2D only: cube block files are
/// rejected with UnsupportedFaceCount (no block cube uploader exists).
pub const RawBlockTexture = struct {
    width: u32 = 0,
    height: u32 = 0,
    num_levels: u32 = 0,
    format: BlockFormat = .bc7_unorm,
    levels: [16]?[]u8 = @splat(null),

    pub fn deinit(self: *RawBlockTexture, allocator: std.mem.Allocator) void {
        for (self.levels[0..self.num_levels]) |level| {
            if (level) |buf| allocator.free(buf);
        }
        self.* = .{};
    }

    /// Exact bytes handed to sg on upload (drives the asset byte budget).
    pub fn totalBytes(self: *const RawBlockTexture) usize {
        var total: usize = 0;
        for (self.levels[0..self.num_levels]) |level| {
            if (level) |buf| total += buf.len;
        }
        return total;
    }
};

/// Validates every level-index entry against the file bounds and the exact
/// block-grid size (scheme NONE: byteLength == uncompressedByteLength ==
/// blockLevelByteSize * faceCount, faces tightly packed — always 1 face
/// here, enforced by decodeBlock2D).
fn parseBlockLevels(allocator: std.mem.Allocator, bytes: []const u8, header: Header, format: BlockFormat) DecodeError![]Level {
    const level_count = header.effectiveLevelCount();
    const index_end = header_and_index_size + level_count * level_index_entry_size;
    if (bytes.len < index_end) return error.Truncated;

    const face_count: usize = header.face_count;
    const levels = try allocator.alloc(Level, level_count);
    errdefer allocator.free(levels);

    for (0..level_count) |m| {
        const entry = header_and_index_size + m * level_index_entry_size;
        const byte_offset = readU64(bytes, entry);
        const byte_length = readU64(bytes, entry + 8);
        const uncompressed_length = readU64(bytes, entry + 16);

        const dims = levelDims(header.pixel_width, header.pixel_height, @intCast(m));
        const level_bytes = format.levelByteSize(dims.w, dims.h) orelse return error.InvalidLevelData;
        const expected = std.math.mul(u64, level_bytes, face_count) catch return error.InvalidLevelData;
        if (byte_length != uncompressed_length or uncompressed_length != expected) {
            return error.InvalidLevelData;
        }
        const start: usize = std.math.cast(usize, byte_offset) orelse return error.InvalidLevelData;
        const len: usize = std.math.cast(usize, byte_length) orelse return error.InvalidLevelData;
        if (start < index_end or @as(u64, start) + byte_length > bytes.len) return error.Truncated;
        levels[m] = .{ .data = bytes[start .. start + len], .width = dims.w, .height = dims.h };
    }
    return levels;
}

/// Reads a 2D block-compressed KTX2 file into owned per-level slices.
/// GPU-free and thread-safe. No mip synthesis: levels upload exactly as
/// authored (a single-level file uploads one level; the backend clamps LOD
/// to the smallest present level). Cube files are rejected — the 2D block
/// path must never silently drop faces.
pub fn decodeBlock2D(allocator: std.mem.Allocator, bytes: []const u8) DecodeError!RawBlockTexture {
    const header = try parseHeaderFields(bytes);
    const format = blockFormatFromVk(header.vk_format) orelse return error.UnsupportedVkFormat;
    if (header.face_count != 1) return error.UnsupportedFaceCount;
    const levels = try parseBlockLevels(allocator, bytes, header, format);
    defer allocator.free(levels);

    var raw = RawBlockTexture{
        .width = header.pixel_width,
        .height = header.pixel_height,
        .num_levels = @intCast(levels.len),
        .format = format,
    };
    errdefer raw.deinit(allocator);
    for (levels, 0..) |level, m| {
        const owned = try allocator.alloc(u8, level.data.len);
        @memcpy(owned, level.data);
        raw.levels[m] = owned;
    }
    return raw;
}

// ---------------------------------------------------------------------------
// Basis Universal transcoding (ETC1S/BasisLZ + UASTC LDR 4x4) through the
// official transcoder (src/agate/c/basis_glue.cpp over the vendored
// src/agate/c/basisu/). GPU-free and thread-safe: the glue builds a private
// transcoder per call (global tables init once via call_once) and the caller
// (here) owns every output buffer, so worker-thread decodes in the asset
// queue and the glTF parallel decoder need no extra synchronization.
//
// Detection: per the KTX2 spec a Basis payload carries vkFormat UNDEFINED
// (0). isBasisKtx2 is routing-only (like isBlockKtx2); basisInfo validates.
// Level data for ETC1S reports uncompressedByteLength 0 — the transcoder
// (not the level index) knows the transcoded size, so expected output sizes
// come from the level math below and the glue enforces the exact-size
// contract (any mismatch fails, never truncates).
// ---------------------------------------------------------------------------

/// Transcode output targets. Integer values match basis_glue.cpp — do not
/// reorder without updating the glue.
pub const BasisTarget = enum(i32) {
    /// Desktop compressed target (BC7_RGBA, 16 B per 4x4 block).
    bc7 = 0,
    /// Mobile compressed target (ASTC LDR 4x4 RGBA, 16 B per block).
    astc = 1,
    /// Universal CPU fallback (RGBA32 raster, R first, 4 B per pixel).
    rgba32 = 2,
    /// Mobile compressed fallback (ETC2 RGBA8, 16 B per 4x4 block).
    etc2_rgba = 3,
};

/// Basis payload kind reported by the transcoder.
pub const BasisKind = enum { etc1s, uastc };

/// File description from the official transcoder (dims/levels authoritative).
pub const BasisInfo = struct {
    width: u32,
    height: u32,
    levels: u32,
    faces: u32,
    has_alpha: bool,
    is_srgb: bool,
    kind: BasisKind,
};

/// True when `bytes` carry the Basis marker: KTX2 identifier, vkFormat
/// UNDEFINED (0), supercompression 0..3 (NONE/BasisLZ/Zstd/ZLIB — the
/// transcoder inflates UASTC supercompression itself). Routing-only:
/// kind/dims/levels are validated later by basisInfo, so `true` never
/// implies transcodability.
pub fn isBasisKtx2(bytes: []const u8) bool {
    if (!sniff(bytes)) return false;
    if (bytes.len < header_and_index_size) return false;
    if (readU32(bytes, 12) != 0) return false;
    return readU32(bytes, 44) <= 3;
}

/// Compressed-target preference from live backend caps: BC7 when sampleable
/// (desktop), else ASTC 4x4, then ETC2 RGBA8 (widely available on GLES 3
/// devices), else the universal RGBA32 fallback. Pure and unit-tested.
pub fn preferredBasisTarget(support: Texture.BlockSupport) BasisTarget {
    if (support.bc7_sample) return .bc7;
    if (support.astc_sample) return .astc;
    if (support.etc2_sample) return .etc2_rgba;
    return .rgba32;
}

/// Envelope validation shared by the Basis entries: every container check
/// EXCEPT the payload-kind gate, in parseHeaderFields order, so Basis and
/// non-Basis paths report identical errors for malformed containers
/// (notably UnsupportedSupercompression precedes UnsupportedVkFormat, and
/// the vkFormat gate here requires UNDEFINED instead of a subset member).
fn parseBasisEnvelope(bytes: []const u8) DecodeError!Header {
    if (!sniff(bytes)) return error.NotKtx2;
    if (bytes.len < header_and_index_size) return error.Truncated;
    if (bytes.len > std.math.maxInt(u32)) return error.FileTooLarge;

    const h = Header{
        .vk_format = readU32(bytes, 12),
        .type_size = readU32(bytes, 16),
        .pixel_width = readU32(bytes, 20),
        .pixel_height = readU32(bytes, 24),
        .pixel_depth = readU32(bytes, 28),
        .layer_count = readU32(bytes, 32),
        .face_count = readU32(bytes, 36),
        .level_count = readU32(bytes, 40),
        .supercompression_scheme = readU32(bytes, 44),
    };

    if (h.supercompression_scheme > 3) return error.UnsupportedSupercompression;
    if (h.vk_format != 0) return error.UnsupportedVkFormat;
    if (h.type_size != 1) return error.UnsupportedTypeSize;
    if (h.pixel_width == 0 or h.pixel_height == 0) return error.UnsupportedDimensions;
    if (h.pixel_depth > 1) return error.Unsupported3D;
    if (h.layer_count > 1) return error.UnsupportedLayers;
    if (h.face_count != 1) return error.UnsupportedFaceCount;
    if (h.level_count > 16) return error.TooManyLevels;
    // levelCount 0 (implicit single level: no encoder writes it, and the
    // transcoder's level index would be empty) is a valid Basis file outside
    // the subset — BasisUnsupported, not a container error.
    if (h.level_count == 0) return error.BasisUnsupported;
    return h;
}

/// Inspects a KTX2 Basis payload through the official transcoder. Returns
/// the file description when the payload is in the transcoded subset;
/// NotBasisKtx2 when the transcoder rejects the header (foreign/corrupt
/// container behind the UNDEFINED marker); BasisUnsupported for valid Basis
/// files outside the subset (grains: kind/cube/video/levelCount 0).
/// GPU-free and thread-safe.
pub fn basisInfo(bytes: []const u8) DecodeError!BasisInfo {
    _ = try parseBasisEnvelope(bytes);
    var out: c.agate_basis_info = undefined;
    const rc = c.agate_basis_ktx2_info(bytes.ptr, bytes.len, &out);
    if (rc == 1) {
        return .{
            .width = out.width,
            .height = out.height,
            .levels = out.levels,
            .faces = out.faces,
            .has_alpha = out.has_alpha != 0,
            .is_srgb = out.is_srgb != 0,
            .kind = if (out.kind == 0) .etc1s else .uastc,
        };
    }
    if (rc == 0) return error.NotBasisKtx2;
    return error.BasisUnsupported;
}

/// Basis decode switches. `srgb_to_linear: null` = auto from the file DFD
/// (mirrors DecodeOptions); applies to the RGBA32 target only — block
/// targets always follow the DFD (sRGB files transcode to the sRGB GPU
/// variant, exactly like the no-transcode path, so the caller's per-slot
/// color/data flag has no effect there).
pub const BasisDecodeOptions = struct {
    /// Only applies to RGBA32 when the file has a single level: a
    /// file-provided chain is always transcoded as authored (generating on
    /// top of it would silently discard authoring data). Block targets
    /// always transcode the authored chain (no CPU mip synthesis exists).
    gen_mipmaps: bool = true,
    srgb_to_linear: ?bool = null,
};

/// Transcodes a 2D KTX2 Basis file to `target`, level by level
/// (largest-first, the full authored chain). Returns a DecodedImage directly:
/// .block for bc7/astc (pair with Texture.fromRawBlock), .rgba for rgba32
/// (pair with Texture.fromRaw). Owned buffers (the asset queue frees the
/// source right after decode); free with deinit. GPU-free and thread-safe.
pub fn decodeBasis2D(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    target: BasisTarget,
    opts: BasisDecodeOptions,
) DecodeError!Texture.DecodedImage {
    const info = try basisInfo(bytes);
    return switch (target) {
        .bc7, .astc, .etc2_rgba => .{ .block = try transcodeBlockLevels(allocator, bytes, info, target) },
        .rgba32 => .{ .rgba = try transcodeRgbaLevels(allocator, bytes, info, opts) },
    };
}

/// Block-target levels: exact ceil-grid sizes, sRGB GPU variant for sRGB
/// files. The glue re-derives the same sizes from the transcoder's own level
/// description and fails on any mismatch (never truncates).
fn transcodeBlockLevels(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    info: BasisInfo,
    target: BasisTarget,
) DecodeError!RawBlockTexture {
    const format: BlockFormat = switch (target) {
        .bc7 => if (info.is_srgb) .bc7_srgb else .bc7_unorm,
        .astc => if (info.is_srgb) .astc_4x4_srgb else .astc_4x4_unorm,
        .etc2_rgba => if (info.is_srgb) .etc2_rgba8_srgb else .etc2_rgba8_unorm,
        .rgba32 => unreachable,
    };
    var raw = RawBlockTexture{
        .width = info.width,
        .height = info.height,
        .num_levels = info.levels,
        .format = format,
    };
    errdefer raw.deinit(allocator);
    for (0..info.levels) |m| {
        const dims = levelDims(info.width, info.height, @intCast(m));
        const n = format.levelByteSize(dims.w, dims.h) orelse return error.InvalidLevelData;
        const len: usize = std.math.cast(usize, n) orelse return error.InvalidLevelData;
        const buf = try allocator.alloc(u8, len);
        const ok = c.agate_basis_ktx2_transcode(
            bytes.ptr,
            bytes.len,
            @intCast(m),
            0,
            @intFromEnum(target),
            buf.ptr,
            buf.len,
        );
        if (!ok) {
            allocator.free(buf);
            return error.BasisTranscodeFailed;
        }
        raw.levels[m] = buf;
    }
    return raw;
}

/// Converts RGB lanes of an RGBA32 buffer through the shared sRGB LUT
/// (alpha untouched), matching the KTX2/PNG sRGB path exactly.
fn convertRgbaSrgbInPlace(buf: []u8) void {
    var i: usize = 0;
    while (i < buf.len) : (i += 4) {
        buf[i + 0] = texture.srgbToLinearU8(buf[i + 0]);
        buf[i + 1] = texture.srgbToLinearU8(buf[i + 1]);
        buf[i + 2] = texture.srgbToLinearU8(buf[i + 2]);
    }
}

fn transcodeRgbaLevel(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    info: BasisInfo,
    level: u32,
) DecodeError![]u8 {
    const dims = levelDims(info.width, info.height, level);
    const n = @as(u64, dims.w) * dims.h * 4;
    const len: usize = std.math.cast(usize, n) orelse return error.InvalidLevelData;
    const buf = try allocator.alloc(u8, len);
    const ok = c.agate_basis_ktx2_transcode(
        bytes.ptr,
        bytes.len,
        level,
        0,
        @intFromEnum(BasisTarget.rgba32),
        buf.ptr,
        buf.len,
    );
    if (!ok) {
        allocator.free(buf);
        return error.BasisTranscodeFailed;
    }
    return buf;
}

fn transcodeRgbaLevels(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    info: BasisInfo,
    opts: BasisDecodeOptions,
) DecodeError!Texture.RawTexture {
    const srgb = opts.srgb_to_linear orelse info.is_srgb;
    // Single-level files can feed the engine's chain generator; multi-level
    // files transcode exactly as authored.
    if (info.levels == 1 and opts.gen_mipmaps) {
        const level0 = try transcodeRgbaLevel(allocator, bytes, info, 0);
        defer allocator.free(level0);
        if (srgb) convertRgbaSrgbInPlace(level0);
        // buildRaw validates dimensions against the byte length; a mismatch
        // here means the transcoder disagrees with its own header.
        var raw = Texture.buildRaw(allocator, info.width, info.height, level0, true) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.InvalidLevelData,
        };
        raw.is_srgb = srgb;
        return raw;
    }

    var raw = Texture.RawTexture{
        .width = info.width,
        .height = info.height,
        .num_levels = info.levels,
        .is_srgb = srgb,
    };
    errdefer raw.deinit(allocator);
    for (0..info.levels) |m| {
        const buf = try transcodeRgbaLevel(allocator, bytes, info, @intCast(m));
        if (srgb) convertRgbaSrgbInPlace(buf);
        raw.levels[m] = buf;
    }
    return raw;
}
// `srgb` converts COLOR lanes (never alpha) with the shared golden LUT, so
// KTX2 color assets match the PNG sRGB path bit for bit.
// ---------------------------------------------------------------------------

/// Converts one byte through the shared sRGB LUT when `apply` is set.
fn mapChannel(v: u8, apply: bool) u8 {
    return if (apply) texture.srgbToLinearU8(v) else v;
}

fn convertTexels(format: Format, src: []const u8, dst: []u8, srgb: bool) void {
    // `srgb` is the caller's FINAL decision (auto from the format tag or an
    // explicit override); convertTexels must not re-gate it by isSrgb.
    const w = srgb;
    switch (format) {
        // Memory layouts identical to the engine upload order.
        .rgba8_unorm, .rgba8_srgb, .abgr8_unorm_pack32, .abgr8_srgb_pack32 => {
            var i: usize = 0;
            while (i < src.len) : (i += 4) {
                dst[i + 0] = mapChannel(src[i + 0], w);
                dst[i + 1] = mapChannel(src[i + 1], w);
                dst[i + 2] = mapChannel(src[i + 2], w);
                dst[i + 3] = src[i + 3];
            }
        },
        // B,G,R,A -> R,G,B,A (byte swap of lanes 0 and 2).
        .bgra8_unorm, .bgra8_srgb => {
            var i: usize = 0;
            while (i < src.len) : (i += 4) {
                dst[i + 0] = mapChannel(src[i + 2], w);
                dst[i + 1] = mapChannel(src[i + 1], w);
                dst[i + 2] = mapChannel(src[i + 0], w);
                dst[i + 3] = src[i + 3];
            }
        },
        // Single channel replicated to RGB (same convention as grayscale
        // PNG), alpha forced opaque.
        .r8_unorm, .r8_srgb => {
            for (src, 0..) |v, i| {
                const ch = mapChannel(v, w);
                dst[i * 4 + 0] = ch;
                dst[i * 4 + 1] = ch;
                dst[i * 4 + 2] = ch;
                dst[i * 4 + 3] = 255;
            }
        },
        // Two channels expanded to RG01; alpha forced opaque (documented:
        // no automatic normal reconstruction).
        .rg8_unorm, .rg8_srgb => {
            var i: usize = 0;
            while (i < src.len) : (i += 2) {
                dst[i * 2 + 0] = mapChannel(src[i + 0], w);
                dst[i * 2 + 1] = mapChannel(src[i + 1], w);
                dst[i * 2 + 2] = 0;
                dst[i * 2 + 3] = 255;
            }
        },
    }
}

/// Decodes a 2D (faceCount 1) KTX2 file into engine RGBA8 levels.
/// GPU-free and thread-safe; pair with Texture.fromRaw. Cube files are
/// rejected (use decodeCube) — the 2D path must never silently drop faces.
pub fn decode2D(allocator: std.mem.Allocator, bytes: []const u8, opts: DecodeOptions) DecodeError!Texture.RawTexture {
    const header = try parseHeader(bytes);
    if (header.face_count != 1) return error.UnsupportedFaceCount;
    const format = formatFromVk(header.vk_format).?;
    const levels = try parseLevels(allocator, bytes, header, format);
    defer allocator.free(levels);

    const srgb = opts.srgb_to_linear orelse format.isSrgb();
    return decodeLevels2D(allocator, header, format, levels, srgb, opts.gen_mipmaps);
}

fn decodeLevels2D(
    allocator: std.mem.Allocator,
    header: Header,
    format: Format,
    levels: []const Level,
    srgb: bool,
    gen_mipmaps: bool,
) DecodeError!Texture.RawTexture {
    // Single-level files can feed the engine's chain generator; multi-level
    // files are uploaded exactly as authored.
    if (levels.len == 1 and gen_mipmaps) {
        const level0 = try convertLevel(allocator, format, levels[0], srgb);
        defer allocator.free(level0);
        // buildRaw validates dimensions against the byte length; a mismatch
        // here means the file's level data disagrees with its header.
        var raw = texture.Texture.buildRaw(allocator, header.pixel_width, header.pixel_height, level0, true) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.InvalidLevelData,
        };
        raw.is_srgb = srgb;
        return raw;
    }

    // ---------------------------------------------------------------------------
    // Texel expansion: supported file formats -> engine RGBA8 (unorm bytes).

    var raw = Texture.RawTexture{
        .width = header.pixel_width,
        .height = header.pixel_height,
        .num_levels = @intCast(levels.len),
        .is_srgb = srgb,
    };
    errdefer raw.deinit(allocator);
    for (levels, 0..) |level, m| {
        raw.levels[m] = try convertLevel(allocator, format, level, srgb);
    }
    return raw;
}

/// Converts one file level to engine RGBA8 bytes.
fn convertLevel(allocator: std.mem.Allocator, format: Format, level: Level, srgb: bool) error{OutOfMemory}![]u8 {
    const dst = try allocator.alloc(u8, level.data.len / format.texelBlockSize() * 4);
    convertTexels(format, level.data, dst, srgb);
    return dst;
}

/// Decodes a cube (faceCount 6) KTX2 file into GPU-free RGBA8 cube levels
/// (six faces concatenated per level, KTX2 face order = engine order).
/// GPU-free; pair with CubeTexture.initRawFacesMips on the main thread.
pub fn decodeCube(allocator: std.mem.Allocator, bytes: []const u8, opts: DecodeOptions) DecodeError!CubeTexture.RawCubeMips {
    const header = try parseHeader(bytes);
    if (header.face_count != 6) return error.UnsupportedFaceCount;
    const format = formatFromVk(header.vk_format).?;
    const levels = try parseLevels(allocator, bytes, header, format);
    defer allocator.free(levels);

    if (header.pixel_width != header.pixel_height) return error.UnsupportedDimensions;

    const srgb = opts.srgb_to_linear orelse format.isSrgb();
    // Chain generation from level 0 is intentionally NOT offered for cubes:
    // the engine's cube generator exists for procedural/env content, while a
    // KTX2 cube always carries authored data. A single-level cube uploads
    // one level (callers needing a chain should author it).
    _ = opts.gen_mipmaps;

    var raw = CubeTexture.RawCubeMips{
        .size = header.pixel_width,
        .num_levels = @intCast(levels.len),
    };
    errdefer raw.deinit(allocator);
    for (levels, 0..) |level, m| {
        raw.levels[m] = try convertLevel(allocator, format, level, srgb);
    }
    return raw;
}

// ---------------------------------------------------------------------------
// Tests — fixtures are synthesized in-memory (header + level index + dummy
// DFD + payloads): the container is simple enough that no external .ktx2
// files are needed.
// ---------------------------------------------------------------------------
