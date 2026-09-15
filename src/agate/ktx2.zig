const std = @import("std");

const texture = @import("texture.zig");
const Texture = texture.Texture;
const CubeTexture = texture.CubeTexture;

// ---------------------------------------------------------------------------
// KTX2 container reader — the honest uncompressed subset.
//
// Supports (KTX2 spec v2.0, all little-endian):
//   - supercompressionScheme 0 (NONE) only. BasisLZ (1), Zstandard (2) and
//     ZLIB (3) need a transcoder/inflator dependency (basis_universal /
//     zstd); see TEXTURE_AUDIT.md "Честно НЕ сделано" — future work.
//   - 8-bit UNORM/SRGB formats that map onto the engine's RGBA8 LDR upload
//     path: R8, R8G8, R8G8B8A8, B8G8R8A8, A8B8G8R8_PACK32 (same LE byte
//     order as R8G8B8A8). Everything else (16F/32F, packed 16-bit, BC/ETC/
//     ASTC blocks) errors with UnsupportedVkFormat.
//   - full file-provided mip chains (level index, largest level first) and
//     optional chain generation for single-level files (gen_mipmaps).
//   - cube maps (faceCount 6, faces +X,-X,+Y,-Y,+Z,-Z — the engine's order).
//
// Deliberately NOT interpreted: the data format descriptor (DFD) and the
// key/value data are skipped by offset — the numeric vkFormat alone drives
// the texel interpretation, which is exact for the supported subset. KTX2
// stores no orientation; glTF/KTX2 assets are top-left like the engine's
// other LDR loaders, so rows upload unflipped.
// ---------------------------------------------------------------------------

/// The 12-byte KTX2 identifier: «KTX 20» + CR LF ^Z LF.
const magic = [12]u8{ 0xAB, 'K', 'T', 'X', ' ', '2', '0', 0xBB, 0x0D, 0x0A, 0x1A, 0x0A };

/// Fixed prefix of the file: header (48 bytes incl. identifier) + index
/// (32 bytes: dfd/kvd u32 pairs, sgd u64 pair) = 80, then the level index.
const header_and_index_size: usize = 80;
const level_index_entry_size: usize = 24;

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

/// True when `bytes` starts with the KTX2 identifier. Cheap guard used to
/// route assets between the KTX2 reader and stb_image.
pub fn sniff(bytes: []const u8) bool {
    return bytes.len >= magic.len and std.mem.eql(u8, bytes[0..magic.len], &magic);
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
    if (h.type_size != 1) return error.UnsupportedTypeSize; // all supported formats are byte-typed
    if (h.pixel_width == 0 or h.pixel_height == 0) return error.UnsupportedDimensions;
    if (h.pixel_depth > 1) return error.Unsupported3D;
    if (h.layer_count > 1) return error.UnsupportedLayers;
    if (h.face_count != 1 and h.face_count != 6) return error.UnsupportedFaceCount;
    // levelCount 0 means one level (the index holds max(1, levelCount)
    // entries); > 16 does not fit the engine's mip storage.
    if (h.level_count > 16) return error.TooManyLevels;
    if (formatFromVk(h.vk_format) == null) return error.UnsupportedVkFormat;
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
// Texel expansion: supported file formats -> engine RGBA8 (unorm bytes).
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
                const c = mapChannel(v, w);
                dst[i * 4 + 0] = c;
                dst[i * 4 + 1] = c;
                dst[i * 4 + 2] = c;
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

const testing = std.testing;

/// Minimal spec-shaped KTX2 builder for tests: writes the 80-byte header,
/// the level index, a small dummy DFD (never parsed by the reader, present
/// for spec fidelity) and the level payloads with correct mipPadding
/// alignment (lcm(texel_block_size, 4)).
const TestKtx2 = struct {
    vk_format: u32 = 37, // VK_FORMAT_R8G8B8A8_UNORM
    type_size: u32 = 1,
    width: u32,
    height: u32,
    depth: u32 = 0,
    layer_count: u32 = 0,
    face_count: u32 = 1,
    level_count: u32 = 1,
    supercompression: u32 = 0,
    /// Payloads largest-first; each entry covers the whole level (all faces).
    level_payloads: []const []const u8,
    /// Overwrites the identifier with garbage when true.
    corrupt_identifier: bool = false,
    /// Lies about a level's byteLength (index says one less).
    truncate_last_level: bool = false,

    fn texelBlockSize(self: TestKtx2) usize {
        const f = formatFromVk(self.vk_format) orelse return 4;
        return f.texelBlockSize();
    }

    fn build(self: TestKtx2, allocator: std.mem.Allocator) ![]u8 {
        const n_levels: usize = @max(1, self.level_count);
        const block: usize = @max(self.texelBlockSize(), 4); // lcm(1..4, 4) == 4 for our sizes

        var out: std.Io.Writer.Allocating = .init(allocator);
        errdefer out.deinit();
        const w = &out.writer;

        var identifier = magic;
        if (self.corrupt_identifier) identifier[0] = 0x00;
        try w.writeAll(&identifier);

        // Header fields (offsets 12..48), spec order.
        try w.writeInt(u32, self.vk_format, .little);
        try w.writeInt(u32, self.type_size, .little);
        try w.writeInt(u32, self.width, .little);
        try w.writeInt(u32, self.height, .little);
        try w.writeInt(u32, self.depth, .little);
        try w.writeInt(u32, self.layer_count, .little);
        try w.writeInt(u32, self.face_count, .little);
        try w.writeInt(u32, self.level_count, .little);
        try w.writeInt(u32, self.supercompression, .little);

        const dfd_offset: u64 = header_and_index_size + n_levels * level_index_entry_size;
        const dfd_len: u64 = 64; // dummy: a real DFD for these formats is 64+ bytes
        // Index (offsets 48..80): dfd/kvd offsets patched after the level
        // index is sized; sgd unused.
        try w.writeInt(u32, 0, .little); // dfdByteOffset placeholder
        try w.writeInt(u32, 0, .little); // dfdByteLength placeholder
        try w.writeInt(u32, 0, .little); // kvdByteOffset placeholder
        try w.writeInt(u32, 0, .little); // kvdByteLength placeholder
        try w.writeInt(u64, 0, .little); // sgdByteOffset
        try w.writeInt(u64, 0, .little); // sgdByteLength

        // Level index with computed mipPadding-aligned offsets. Sized past
        // the 16-level cap so bogus headers (e.g. levelCount 17, rejected by
        // the reader) still build without spilling.
        var offsets: [32]u64 = undefined;
        var cursor: u64 = dfd_offset + dfd_len;
        for (0..n_levels) |m| {
            cursor = std.mem.alignForward(u64, cursor, block);
            offsets[m] = cursor;
            const payload = self.level_payloads[@min(m, self.level_payloads.len - 1)];
            const payload_len: u64 = if (self.truncate_last_level and m == n_levels - 1)
                payload.len - 1
            else
                payload.len;
            cursor += payload_len;
            try w.writeInt(u64, offsets[m], .little);
            try w.writeInt(u64, payload_len, .little);
            try w.writeInt(u64, payload_len, .little);
        }

        // Patch the deferred index fields (dfd first, kvd after it).
        std.mem.writeInt(u32, out.written()[48..52], @intCast(dfd_offset), .little);
        std.mem.writeInt(u32, out.written()[52..56], @intCast(dfd_len), .little);
        std.mem.writeInt(u32, out.written()[56..60], @intCast(dfd_offset + dfd_len), .little);
        std.mem.writeInt(u32, out.written()[60..64], 0, .little);

        // Dummy DFD (skipped by the reader; present for spec fidelity).
        for (0..@intCast(dfd_len)) |_| try w.writeByte(0);

        // Level payloads with mipPadding zero gaps.
        for (0..n_levels) |m| {
            const here: u64 = @intCast(out.written().len);
            for (0..@intCast(offsets[m] - here)) |_| try w.writeByte(0);
            try w.writeAll(self.level_payloads[@min(m, self.level_payloads.len - 1)]);
        }
        return out.toOwnedSlice();
    }
};

test "sniff matches the KTX2 identifier and nothing else" {
    try testing.expect(sniff(&magic));
    try testing.expect(!sniff("not a ktx2 file at all"));
    try testing.expect(!sniff(magic[0..8])); // truncated identifier
    try testing.expect(!sniff(&.{}));
}

test "decode2D reads a 4x4 two-level RGBA8 file exactly as authored" {
    const allocator = testing.allocator;
    // Level 0: 16 texels with distinct values; level 1: 4 texels.
    var level0: [4 * 4 * 4]u8 = undefined;
    for (&level0, 0..) |*b, i| b.* = @intCast(i % 251);
    const level1 = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    const ktx = try (TestKtx2{
        .width = 4,
        .height = 4,
        .level_count = 2,
        .level_payloads = &.{ &level0, &level1 },
    }).build(allocator);
    defer allocator.free(ktx);

    var raw = try decode2D(allocator, ktx, .{});
    defer raw.deinit(allocator);
    try testing.expectEqual(@as(u32, 4), raw.width);
    try testing.expectEqual(@as(u32, 4), raw.height);
    try testing.expectEqual(@as(u32, 2), raw.num_levels);
    try testing.expectEqualSlices(u8, &level0, raw.levels[0].?);
    try testing.expectEqualSlices(u8, &level1, raw.levels[1].?);
}

test "decode2D generates the mip chain for single-level files" {
    const allocator = testing.allocator;
    const level0 = [_]u8{
        255, 0, 0,   255, 255, 0, 0,   255,
        255, 0, 0,   255, 255, 0, 0,   255,
        0,   0, 255, 255, 0,   0, 255, 255,
        0,   0, 255, 255, 0,   0, 255, 255,
        0,   0, 255, 255, 0,   0, 255, 255,
        0,   0, 255, 255, 0,   0, 255, 255,
        255, 0, 0,   255, 255, 0, 0,   255,
        255, 0, 0,   255, 255, 0, 0,   255,
    }; // 4x4
    const ktx = try (TestKtx2{
        .width = 4,
        .height = 4,
        .level_count = 1,
        .level_payloads = &.{&level0},
    }).build(allocator);
    defer allocator.free(ktx);

    var raw = try decode2D(allocator, ktx, .{});
    defer raw.deinit(allocator);
    try testing.expectEqual(Texture.mipLevelCount(4, 4), raw.num_levels);
    // gen_mipmaps=false keeps exactly the authored single level.
    var flat = try decode2D(allocator, ktx, .{ .gen_mipmaps = false });
    defer flat.deinit(allocator);
    try testing.expectEqual(@as(u32, 1), flat.num_levels);
}

test "decode2D auto-converts sRGB formats and honors explicit overrides" {
    const allocator = testing.allocator;
    const level0 = [_]u8{ 200, 200, 200, 42 };
    const srgb_ktx = try (TestKtx2{
        .vk_format = 43, // R8G8B8A8_SRGB
        .width = 1,
        .height = 1,
        .level_payloads = &.{&level0},
    }).build(allocator);
    defer allocator.free(srgb_ktx);

    // Auto (null): the format tag drives the conversion.
    var auto = try decode2D(allocator, srgb_ktx, .{});
    defer auto.deinit(allocator);
    try testing.expectEqual(@as(u8, texture.srgbToLinearU8(200)), auto.levels[0].?[0]);
    try testing.expectEqual(@as(u8, 42), auto.levels[0].?[3]); // alpha untouched

    // Explicit false forbids it even for SRGB formats.
    var raw8 = try decode2D(allocator, srgb_ktx, .{ .srgb_to_linear = false });
    defer raw8.deinit(allocator);
    try testing.expectEqualSlices(u8, &level0, raw8.levels[0].?);

    // Explicit true converts an UNORM file (caller-authoritative).
    const unorm_ktx = try (TestKtx2{
        .vk_format = 37,
        .width = 1,
        .height = 1,
        .level_payloads = &.{&level0},
    }).build(allocator);
    defer allocator.free(unorm_ktx);
    var forced = try decode2D(allocator, unorm_ktx, .{ .srgb_to_linear = true });
    defer forced.deinit(allocator);
    try testing.expectEqual(@as(u8, texture.srgbToLinearU8(200)), forced.levels[0].?[0]);
}

test "decode2D swizzles BGRA and expands R8/RG8" {
    const allocator = testing.allocator;
    const bgra = [_]u8{ 11, 22, 33, 44 };
    const bgra_ktx = try (TestKtx2{
        .vk_format = 44, // B8G8R8A8_UNORM
        .width = 1,
        .height = 1,
        .level_payloads = &.{&bgra},
    }).build(allocator);
    defer allocator.free(bgra_ktx);
    var raw = try decode2D(allocator, bgra_ktx, .{ .srgb_to_linear = false });
    defer raw.deinit(allocator);
    try testing.expectEqual([4]u8{ 33, 22, 11, 44 }, raw.levels[0].?[0..4].*);

    const gray = [_]u8{0x80};
    const r_ktx = try (TestKtx2{
        .vk_format = 9, // R8_UNORM
        .width = 1,
        .height = 1,
        .level_payloads = &.{&gray},
    }).build(allocator);
    defer allocator.free(r_ktx);
    var r8 = try decode2D(allocator, r_ktx, .{});
    defer r8.deinit(allocator);
    try testing.expectEqual([4]u8{ 0x80, 0x80, 0x80, 255 }, r8.levels[0].?[0..4].*);

    const rg = [_]u8{ 10, 200 };
    const rg_ktx = try (TestKtx2{
        .vk_format = 16, // R8G8_UNORM
        .width = 1,
        .height = 1,
        .level_payloads = &.{&rg},
    }).build(allocator);
    defer allocator.free(rg_ktx);
    var rg8 = try decode2D(allocator, rg_ktx, .{});
    defer rg8.deinit(allocator);
    try testing.expectEqual([4]u8{ 10, 200, 0, 255 }, rg8.levels[0].?[0..4].*);
}

test "decode2D rejects truncated payloads" {
    const allocator = testing.allocator;
    const level0 = [_]u8{ 1, 2, 3, 4 } ** 4; // 4x1... keep 4 texels
    const ktx = try (TestKtx2{
        .width = 4,
        .height = 1,
        .level_payloads = &.{&level0},
        .truncate_last_level = true,
    }).build(allocator);
    defer allocator.free(ktx);
    try testing.expectError(error.InvalidLevelData, decode2D(allocator, ktx, .{}));
}

test "decode2D rejects foreign data and corrupted identifiers" {
    const allocator = testing.allocator;
    try testing.expectError(error.NotKtx2, decode2D(allocator, "png data pretending", .{}));

    const level0 = [_]u8{0} ** 4;
    const bad = try (TestKtx2{
        .width = 1,
        .height = 1,
        .level_payloads = &.{&level0},
        .corrupt_identifier = true,
    }).build(allocator);
    defer allocator.free(bad);
    try testing.expectError(error.NotKtx2, decode2D(allocator, bad, .{}));
}

test "decode2D rejects unsupported containers and formats" {
    const allocator = testing.allocator;
    const level0 = [_]u8{0} ** 4;

    const cases = [_]struct { name: []const u8, spec: TestKtx2, expected: DecodeError }{
        .{ .name = "BasisLZ", .spec = .{ .width = 1, .height = 1, .supercompression = 1, .level_payloads = &.{&level0} }, .expected = error.UnsupportedSupercompression },
        .{ .name = "Zstandard", .spec = .{ .width = 1, .height = 1, .supercompression = 2, .level_payloads = &.{&level0} }, .expected = error.UnsupportedSupercompression },
        .{ .name = "ZLIB", .spec = .{ .width = 1, .height = 1, .supercompression = 3, .level_payloads = &.{&level0} }, .expected = error.UnsupportedSupercompression },
        .{ .name = "BC3 (block compressed)", .spec = .{ .vk_format = 135, .width = 4, .height = 4, .level_payloads = &.{&level0} }, .expected = error.UnsupportedVkFormat },
        .{ .name = "RGBA16F", .spec = .{ .vk_format = 110, .width = 1, .height = 1, .level_payloads = &.{&level0} }, .expected = error.UnsupportedVkFormat },
        .{ .name = "3D texture", .spec = .{ .width = 2, .height = 2, .depth = 2, .level_payloads = &.{&level0} }, .expected = error.Unsupported3D },
        .{ .name = "texture array", .spec = .{ .width = 1, .height = 1, .layer_count = 2, .level_payloads = &.{&level0} }, .expected = error.UnsupportedLayers },
        .{ .name = "faceCount 3", .spec = .{ .width = 1, .height = 1, .face_count = 3, .level_payloads = &.{&level0} }, .expected = error.UnsupportedFaceCount },
        .{ .name = "16-bit typeSize", .spec = .{ .width = 1, .height = 1, .type_size = 2, .level_payloads = &.{&level0} }, .expected = error.UnsupportedTypeSize },
        .{ .name = "17 levels", .spec = .{ .width = 2, .height = 2, .level_count = 17, .level_payloads = &.{&level0} }, .expected = error.TooManyLevels },
    };
    for (cases) |case| {
        const ktx = try case.spec.build(allocator);
        defer allocator.free(ktx);
        try testing.expectError(case.expected, decode2D(allocator, ktx, .{}));
    }
}

test "decodeCube maps KTX2 faces to engine order and levels" {
    const allocator = testing.allocator;
    // 2x2 cube, one level: each face's texels carry the face index so the
    // face ordering (+X,-X,+Y,-Y,+Z,-Z) is directly observable.
    const texel_count = 2 * 2;
    var level0: [6 * texel_count * 4]u8 = undefined;
    for (0..6) |face| {
        for (0..texel_count) |t| {
            const o = (face * texel_count + t) * 4;
            level0[o + 0] = @intCast(face);
            level0[o + 1] = 0;
            level0[o + 2] = 0;
            level0[o + 3] = 255;
        }
    }
    const ktx = try (TestKtx2{
        .width = 2,
        .height = 2,
        .face_count = 6,
        .level_payloads = &.{&level0},
    }).build(allocator);
    defer allocator.free(ktx);

    var cube = try decodeCube(allocator, ktx, .{});
    defer cube.deinit(allocator);
    try testing.expectEqual(@as(u32, 2), cube.size);
    try testing.expectEqual(@as(u32, 1), cube.num_levels);
    try testing.expectEqualSlices(u8, &level0, cube.levels[0].?);
    // Face 4 (+Z) starts at texel 16 with R=4.
    try testing.expectEqual(@as(u8, 4), cube.levels[0].?[16 * 4]);

    // decode2D must not silently accept the cube.
    try testing.expectError(error.UnsupportedFaceCount, decode2D(allocator, ktx, .{}));
}

test "formatFromVk covers exactly the supported subset" {
    try testing.expectEqual(Format.rgba8_unorm, formatFromVk(37).?);
    try testing.expectEqual(Format.rgba8_srgb, formatFromVk(43).?);
    try testing.expectEqual(Format.bgra8_srgb, formatFromVk(50).?);
    try testing.expectEqual(Format.abgr8_srgb_pack32, formatFromVk(57).?);
    try testing.expect(formatFromVk(0) == null); // UNDEFINED
    try testing.expect(formatFromVk(44) != null);
    try testing.expect(formatFromVk(1) == null); // packed 4-bit
    try testing.expect(Format.r8_unorm.texelBlockSize() == 1);
    try testing.expect(Format.rgba8_unorm.isSrgb() == false);
    try testing.expect(Format.r8_srgb.isSrgb() == true);
}

test "Texture.decodeMemory routes KTX2 payloads by magic sniff" {
    const allocator = testing.allocator;
    const level0 = [_]u8{ 200, 0, 0, 255 };
    const ktx = try (TestKtx2{
        .vk_format = 43, // R8G8B8A8_SRGB
        .width = 1,
        .height = 1,
        .level_payloads = &.{&level0},
    }).build(allocator);
    defer allocator.free(ktx);

    // The sniffed route forwards the caller's per-slot sRGB decision
    // (color slot -> true), unlike ktx2.decode2D's format-tag auto mode.
    var routed = try Texture.decodeMemory(allocator, ktx, .{ .srgb_to_linear = true });
    defer routed.deinit(allocator);
    try testing.expectEqual(@as(u8, texture.srgbToLinearU8(200)), routed.levels[0].?[0]);

    var raw = try Texture.decodeMemory(allocator, ktx, .{ .srgb_to_linear = false });
    defer raw.deinit(allocator);
    try testing.expectEqual([4]u8{ 200, 0, 0, 255 }, raw.levels[0].?[0..4].*);
}
