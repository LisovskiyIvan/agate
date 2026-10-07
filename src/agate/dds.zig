const std = @import("std");

const ktx2 = @import("ktx2.zig");

// ---------------------------------------------------------------------------
// DDS container reader — the block-compressed subset only (BC1/BC2/BC3/BC7).
//
// Supports (all little-endian):
//   - magic "DDS " plus the 124-byte DDS_HEADER.
//   - legacy fourCC pixel formats: DXT1 (-> BC1), DXT3 (-> BC2),
//     DXT5 (-> BC3). sRGB-ness is NOT stored in legacy headers: the caller
//     picks it via DecodeOptions.srgb, which selects the sRGB GPU variant
//     where one exists (BC3_SRGBA/BC7_SRGBA; BC1/BC2 stay UNORM — this
//     sokol checkout exposes no BC1_SRGBA/BC2_SRGBA).
//   - the DX10 extended header (fourCC "DX10") for BC7 (and the DX10
//     spellings of BC1/BC2/BC3): the DXGI _SRGB tag is authoritative and
//     the caller's srgb flag is ignored.
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
const magic = [4]u8{ 'D', 'D', 'S', ' ' };

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
const flag_caps: u32 = 0x1;
const flag_height: u32 = 0x2;
const flag_width: u32 = 0x4;
const flag_pixelformat: u32 = 0x1000;
const flag_mipmapcount: u32 = 0x20000;
const flag_depth: u32 = 0x800000;
const required_flags: u32 = flag_caps | flag_height | flag_width | flag_pixelformat;

// DDS_PIXELFORMAT dwFlags bits.
const pf_fourcc: u32 = 0x4;

// DDS_HEADER dwCaps bits.
const caps_texture: u32 = 0x1000;
// DDS_HEADER dwCaps2 bits: cubemap face mask + volume.
const caps2_cubemap_mask: u32 = 0xFE00;
const caps2_volume: u32 = 0x200000;

// Legacy fourCC codes (u32 little-endian of the ASCII tag).
const fourcc_dxt1: u32 = 0x31545844;
const fourcc_dxt2: u32 = 0x32545844;
const fourcc_dxt3: u32 = 0x33545844;
const fourcc_dxt4: u32 = 0x34545844;
const fourcc_dxt5: u32 = 0x35545844;
const fourcc_dx10: u32 = 0x30315844;

// DXGI_FORMAT codes (DXGI 1.0) understood by the DX10 path.
const dxgi_bc1_unorm: u32 = 71;
const dxgi_bc1_unorm_srgb: u32 = 72;
const dxgi_bc2_unorm: u32 = 74;
const dxgi_bc2_unorm_srgb: u32 = 75;
const dxgi_bc3_unorm: u32 = 77;
const dxgi_bc3_unorm_srgb: u32 = 78;
const dxgi_bc7_unorm: u32 = 98;
const dxgi_bc7_unorm_srgb: u32 = 99;

// DDS_HEADER_DXT10 resourceDimension values.
const dx10_dimension_texture2d: u32 = 3;
// DDS_HEADER_DXT10 miscFlag bits.
const dx10_misc_texturecube: u32 = 0x4;

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

    var data_offset: usize = base + header_size;
    const format: ktx2.BlockFormat = switch (fourcc) {
        fourcc_dxt1 => .bc1_unorm,
        fourcc_dxt3 => .bc2_unorm,
        fourcc_dxt5 => if (opts.isSrgb()) .bc3_srgb else .bc3_unorm,
        fourcc_dx10 => blk: {
            if (bytes.len < magic.len + header_size + dx10_size) return error.Truncated;
            const dx = base + header_size;
            if (readU32(bytes, dx + 4) != dx10_dimension_texture2d) return error.UnsupportedDdsFormat;
            if (readU32(bytes, dx + 8) & dx10_misc_texturecube != 0) return error.UnsupportedCube;
            if (readU32(bytes, dx + 12) != 1) return error.UnsupportedArraySize;
            data_offset += dx10_size;
            break :blk try blockFormatFromDxgi(readU32(bytes, dx + 0));
        },
        // DXT2/DXT4 are premultiplied-alpha flavors the engine never
        // agreed to interpret; uncompressed/packed fourCCs (e.g. "RGBA",
        // "G16R16") have no block decoder here.
        fourcc_dxt2, fourcc_dxt4 => return error.UnsupportedDdsFormat,
        else => return error.UnsupportedDdsFormat,
    };

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

// ---------------------------------------------------------------------------
// Tests — fixtures are synthesized in-memory (magic + DDS_HEADER [+ DX10]
// + payloads): the container is simple enough that no external .dds files
// are needed.
// ---------------------------------------------------------------------------

const testing = std.testing;

/// Minimal DDS builder for tests: 4-byte magic, 124-byte header, optional
/// 20-byte DX10 extension, then the level payloads back to back (largest
/// first). Payloads are passed per level; sizes are NOT validated here so
/// malformed fixtures (short/trailing data) can be built on purpose.
const TestDds = struct {
    fourcc: u32 = fourcc_dxt5,
    /// DX10 extension: null = legacy header; set = dxgiFormat code.
    dxgi: ?u32 = null,
    width: u32 = 4,
    height: u32 = 4,
    /// Mip levels to declare (dwMipMapCount); set_mip_flag controls
    /// DDSD_MIPMAPCOUNT. Payload count should match for valid fixtures.
    mip_count: u32 = 1,
    set_mip_flag: bool = false,
    /// Overwrites the magic with garbage when true.
    corrupt_magic: bool = false,
    /// Header dwSize override (124 = valid).
    header_size: u32 = 124,
    /// Pixel-format dwSize override (32 = valid).
    pf_size: u32 = 32,
    /// Extra required-flags bits to clear (0 = valid).
    clear_flags: u32 = 0,
    /// dwCaps2 override (0 = valid 2D texture).
    caps2: u32 = 0,
    /// Depth header field override (0 = valid 2D texture).
    depth: u32 = 0,
    /// DX10 arraySize override (1 = valid).
    array_size: u32 = 1,
    /// DX10 resourceDimension override (3 = TEXTURE2D).
    dimension: u32 = 3,
    /// DX10 miscFlag override (0 = valid).
    misc_flag: u32 = 0,
    level_payloads: []const []const u8 = &.{},
    /// Appends trailing garbage bytes after the last level when true.
    trailing_garbage: bool = false,

    fn build(self: TestDds, allocator: std.mem.Allocator) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(allocator);
        errdefer out.deinit();
        const w = &out.writer;

        if (self.corrupt_magic) {
            try w.writeAll("NOPE");
        } else {
            try w.writeAll(&magic);
        }

        var flags: u32 = required_flags;
        if (self.set_mip_flag or self.mip_count > 1) flags |= flag_mipmapcount;
        flags &= ~self.clear_flags;

        try w.writeInt(u32, self.header_size, .little);
        try w.writeInt(u32, flags, .little);
        try w.writeInt(u32, self.height, .little);
        try w.writeInt(u32, self.width, .little);
        try w.writeInt(u32, 0, .little); // dwPitchOrLinearSize (unchecked)
        try w.writeInt(u32, self.depth, .little);
        try w.writeInt(u32, self.mip_count, .little);
        for (0..11) |_| try w.writeInt(u32, 0, .little); // dwReserved1
        // DDS_PIXELFORMAT (32 bytes).
        try w.writeInt(u32, self.pf_size, .little);
        try w.writeInt(u32, pf_fourcc, .little);
        try w.writeInt(u32, self.fourcc, .little);
        try w.writeInt(u32, 0, .little); // dwRGBBitCount
        for (0..4) |_| try w.writeInt(u32, 0, .little); // bit masks
        try w.writeInt(u32, caps_texture, .little); // dwCaps
        try w.writeInt(u32, self.caps2, .little); // dwCaps2
        try w.writeInt(u32, 0, .little); // dwCaps3
        try w.writeInt(u32, 0, .little); // dwCaps4
        try w.writeInt(u32, 0, .little); // dwReserved2

        if (self.dxgi) |dxgi| {
            try w.writeInt(u32, dxgi, .little);
            try w.writeInt(u32, self.dimension, .little);
            try w.writeInt(u32, self.misc_flag, .little);
            try w.writeInt(u32, self.array_size, .little);
            try w.writeInt(u32, 0, .little); // miscFlags2
        }

        for (self.level_payloads) |payload| try w.writeAll(payload);
        if (self.trailing_garbage) try w.writeAll(&.{ 0xAA, 0xBB });
        return out.toOwnedSlice();
    }
};

test "sniff matches the DDS magic and nothing else" {
    try testing.expect(sniff(&magic));
    try testing.expect(!sniff("not a dds file!!"));
    try testing.expect(!sniff("DDS")); // truncated magic
    try testing.expect(!sniff(&.{}));
    // KTX2 files must not route here (magics are disjoint).
    try testing.expect(!sniff(&[12]u8{ 0xAB, 'K', 'T', 'X', ' ', '2', '0', 0xBB, 0x0D, 0x0A, 0x1A, 0x0A }));
}

test "decodeBlock2D reads a legacy DXT1 4x4 file exactly as authored" {
    const allocator = testing.allocator;
    var payload: [8]u8 = undefined; // 4x4 BC1 -> one 8-byte block
    for (&payload, 0..) |*b, i| b.* = @intCast(10 + i);
    const file = try (TestDds{
        .fourcc = fourcc_dxt1,
        .level_payloads = &.{&payload},
    }).build(allocator);
    defer allocator.free(file);

    var raw = try decodeBlock2D(allocator, file, .{});
    defer raw.deinit(allocator);
    try testing.expectEqual(@as(u32, 4), raw.width);
    try testing.expectEqual(@as(u32, 4), raw.height);
    try testing.expectEqual(@as(u32, 1), raw.num_levels);
    try testing.expectEqual(ktx2.BlockFormat.bc1_unorm, raw.format);
    try testing.expectEqualSlices(u8, &payload, raw.levels[0].?);
    try testing.expectEqual(@as(usize, 8), raw.totalBytes());
    // Owned copies (the asset queue frees the source right after decode).
    @memset(file, 0xFF);
    try testing.expectEqual(@as(u8, 10), raw.levels[0].?[0]);
}

test "decodeBlock2D reads a legacy DXT5 8x8 two-level file exactly as authored" {
    const allocator = testing.allocator;
    // Level 0: 8x8 -> 2x2 blocks -> 64 B; level 1: 4x4 -> 1 block -> 16 B.
    var level0: [64]u8 = undefined;
    for (&level0, 0..) |*b, i| b.* = @intCast(i % 251);
    const level1 = [_]u8{ 7, 7, 7, 7, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const file = try (TestDds{
        .fourcc = fourcc_dxt5,
        .width = 8,
        .height = 8,
        .mip_count = 2,
        .level_payloads = &.{ &level0, &level1 },
    }).build(allocator);
    defer allocator.free(file);

    var raw = try decodeBlock2D(allocator, file, .{});
    defer raw.deinit(allocator);
    try testing.expectEqual(@as(u32, 8), raw.width);
    try testing.expectEqual(@as(u32, 8), raw.height);
    try testing.expectEqual(@as(u32, 2), raw.num_levels);
    try testing.expectEqual(ktx2.BlockFormat.bc3_unorm, raw.format);
    try testing.expect(!raw.format.isSrgb());
    try testing.expectEqualSlices(u8, &level0, raw.levels[0].?);
    try testing.expectEqualSlices(u8, &level1, raw.levels[1].?);
    try testing.expectEqual(@as(usize, 64 + 16), raw.totalBytes());
}

test "decodeBlock2D reads a DX10 BC7 8x8 two-level file exactly as authored" {
    const allocator = testing.allocator;
    var level0: [64]u8 = undefined;
    for (&level0, 0..) |*b, i| b.* = @intCast(200 - i % 100);
    var level1: [16]u8 = undefined;
    for (&level1, 0..) |*b, i| b.* = @intCast(i);
    const file = try (TestDds{
        .fourcc = fourcc_dx10,
        .dxgi = dxgi_bc7_unorm,
        .width = 8,
        .height = 8,
        .mip_count = 2,
        .level_payloads = &.{ &level0, &level1 },
    }).build(allocator);
    defer allocator.free(file);

    var raw = try decodeBlock2D(allocator, file, .{});
    defer raw.deinit(allocator);
    try testing.expectEqual(ktx2.BlockFormat.bc7_unorm, raw.format);
    try testing.expect(!raw.format.isSrgb());
    try testing.expectEqual(@as(u32, 2), raw.num_levels);
    try testing.expectEqualSlices(u8, &level0, raw.levels[0].?);
    try testing.expectEqualSlices(u8, &level1, raw.levels[1].?);
}

test "format mapping covers fourCC, DXGI and the legacy sRGB decision" {
    const allocator = testing.allocator;
    // One BC1 block is enough bytes for every header-only mapping probe;
    // the payload only needs to satisfy the exact-size check.
    var bc1_payload: [8]u8 = undefined;
    @memset(&bc1_payload, 0x11);
    var bc3_payload: [16]u8 = undefined;
    @memset(&bc3_payload, 0x22);
    var bc7_payload: [16]u8 = undefined;
    @memset(&bc7_payload, 0x33);

    const Case = struct {
        name: []const u8,
        spec: TestDds,
        srgb: bool,
        expected: ktx2.BlockFormat,
    };
    const cases = [_]Case{
        .{ .name = "DXT1 legacy", .spec = .{ .fourcc = fourcc_dxt1, .level_payloads = &.{&bc1_payload} }, .srgb = false, .expected = .bc1_unorm },
        // BC1 has no sRGB GPU variant: the flag is accepted but ignored.
        .{ .name = "DXT1 legacy srgb ignored", .spec = .{ .fourcc = fourcc_dxt1, .level_payloads = &.{&bc1_payload} }, .srgb = true, .expected = .bc1_unorm },
        .{ .name = "DXT3 legacy", .spec = .{ .fourcc = fourcc_dxt3, .level_payloads = &.{&bc3_payload} }, .srgb = false, .expected = .bc2_unorm },
        .{ .name = "DXT5 legacy linear", .spec = .{ .fourcc = fourcc_dxt5, .level_payloads = &.{&bc3_payload} }, .srgb = false, .expected = .bc3_unorm },
        .{ .name = "DXT5 legacy srgb", .spec = .{ .fourcc = fourcc_dxt5, .level_payloads = &.{&bc3_payload} }, .srgb = true, .expected = .bc3_srgb },
        .{ .name = "DX10 BC1", .spec = .{ .fourcc = fourcc_dx10, .dxgi = dxgi_bc1_unorm, .level_payloads = &.{&bc1_payload} }, .srgb = false, .expected = .bc1_unorm },
        .{ .name = "DX10 BC1_SRGB collapses to UNORM", .spec = .{ .fourcc = fourcc_dx10, .dxgi = dxgi_bc1_unorm_srgb, .level_payloads = &.{&bc1_payload} }, .srgb = false, .expected = .bc1_unorm },
        .{ .name = "DX10 BC2", .spec = .{ .fourcc = fourcc_dx10, .dxgi = dxgi_bc2_unorm, .level_payloads = &.{&bc3_payload} }, .srgb = false, .expected = .bc2_unorm },
        .{ .name = "DX10 BC3", .spec = .{ .fourcc = fourcc_dx10, .dxgi = dxgi_bc3_unorm, .level_payloads = &.{&bc3_payload} }, .srgb = false, .expected = .bc3_unorm },
        .{ .name = "DX10 BC3_SRGB", .spec = .{ .fourcc = fourcc_dx10, .dxgi = dxgi_bc3_unorm_srgb, .level_payloads = &.{&bc3_payload} }, .srgb = false, .expected = .bc3_srgb },
        .{ .name = "DX10 BC7", .spec = .{ .fourcc = fourcc_dx10, .dxgi = dxgi_bc7_unorm, .level_payloads = &.{&bc7_payload} }, .srgb = false, .expected = .bc7_unorm },
        // The DXGI tag is authoritative: the caller flag cannot downgrade it.
        .{ .name = "DX10 BC7_SRGB ignores caller flag", .spec = .{ .fourcc = fourcc_dx10, .dxgi = dxgi_bc7_unorm_srgb, .level_payloads = &.{&bc7_payload} }, .srgb = false, .expected = .bc7_srgb },
    };
    for (cases) |case| {
        const file = try case.spec.build(allocator);
        defer allocator.free(file);
        var raw = try decodeBlock2D(allocator, file, .{ .srgb = case.srgb });
        defer raw.deinit(allocator);
        try testing.expectEqual(case.expected, raw.format);
    }

    // sRGB-ness observable on the format predicate.
    const srgb_file = try (TestDds{
        .fourcc = fourcc_dxt5,
        .level_payloads = &.{&bc3_payload},
    }).build(allocator);
    defer allocator.free(srgb_file);
    var srgb_raw = try decodeBlock2D(allocator, srgb_file, .{ .srgb = true });
    defer srgb_raw.deinit(allocator);
    try testing.expect(srgb_raw.format.isSrgb());

    // Explicit TextureSlot contract
    var slot_color = try decodeBlock2D(allocator, srgb_file, .{ .slot = .color });
    defer slot_color.deinit(allocator);
    try testing.expectEqual(ktx2.BlockFormat.bc3_srgb, slot_color.format);

    var slot_data = try decodeBlock2D(allocator, srgb_file, .{ .slot = .data });
    defer slot_data.deinit(allocator);
    try testing.expectEqual(ktx2.BlockFormat.bc3_unorm, slot_data.format);

    // Explicit TextureColorSpace contract
    var cs_srgb = try decodeBlock2D(allocator, srgb_file, .{ .color_space = .srgb });
    defer cs_srgb.deinit(allocator);
    try testing.expectEqual(ktx2.BlockFormat.bc3_srgb, cs_srgb.format);

    var cs_linear = try decodeBlock2D(allocator, srgb_file, .{ .color_space = .linear });
    defer cs_linear.deinit(allocator);
    try testing.expectEqual(ktx2.BlockFormat.bc3_unorm, cs_linear.format);

    // Precedence: color_space overrides slot and srgb boolean
    var override_cs = try decodeBlock2D(allocator, srgb_file, .{ .slot = .color, .color_space = .linear, .srgb = true });
    defer override_cs.deinit(allocator);
    try testing.expectEqual(ktx2.BlockFormat.bc3_unorm, override_cs.format);
}

test "decodeBlock2D rejects bad magic, sizes and malformed headers" {
    const allocator = testing.allocator;
    var payload: [16]u8 = undefined;
    @memset(&payload, 0x44);

    // Foreign data and corrupted magic.
    try testing.expectError(error.NotDds, decodeBlock2D(allocator, "png data pretending", .{}));
    const bad_magic = try (TestDds{
        .corrupt_magic = true,
        .level_payloads = &.{&payload},
    }).build(allocator);
    defer allocator.free(bad_magic);
    try testing.expectError(error.NotDds, decodeBlock2D(allocator, bad_magic, .{}));

    // Truncated header and truncated payload.
    const full = try (TestDds{
        .fourcc = fourcc_dxt5,
        .level_payloads = &.{&payload},
    }).build(allocator);
    defer allocator.free(full);
    try testing.expectError(error.Truncated, decodeBlock2D(allocator, full[0..100], .{}));
    try testing.expectError(error.Truncated, decodeBlock2D(allocator, full[0 .. full.len - 1], .{}));
    // ...but trailing garbage past the exact block-grid size is rejected too.
    const garbage = try (TestDds{
        .fourcc = fourcc_dxt5,
        .level_payloads = &.{&payload},
        .trailing_garbage = true,
    }).build(allocator);
    defer allocator.free(garbage);
    try testing.expectError(error.InvalidMipData, decodeBlock2D(allocator, garbage, .{}));

    // Malformed header fields.
    const bad_cases = [_]struct { name: []const u8, spec: TestDds, expected: DecodeError }{
        .{ .name = "dwSize", .spec = .{ .header_size = 100, .level_payloads = &.{&payload} }, .expected = error.InvalidMipData },
        .{ .name = "pixel format size", .spec = .{ .pf_size = 16, .level_payloads = &.{&payload} }, .expected = error.InvalidMipData },
        .{ .name = "missing required flags", .spec = .{ .clear_flags = flag_height, .level_payloads = &.{&payload} }, .expected = error.InvalidMipData },
        .{ .name = "zero width", .spec = .{ .width = 0, .level_payloads = &.{&payload} }, .expected = error.UnsupportedDimensions },
        .{ .name = "huge height", .spec = .{ .height = 32768, .level_payloads = &.{&payload} }, .expected = error.UnsupportedDimensions },
        .{ .name = "depth set", .spec = .{ .depth = 4, .level_payloads = &.{&payload} }, .expected = error.Unsupported3D },
        .{ .name = "cubemap caps2", .spec = .{ .caps2 = 0x200 | 0x400, .level_payloads = &.{&payload} }, .expected = error.UnsupportedCube },
        .{ .name = "volume caps2", .spec = .{ .caps2 = caps2_volume, .level_payloads = &.{&payload} }, .expected = error.Unsupported3D },
        .{ .name = "17 levels", .spec = .{ .width = 256, .height = 256, .mip_count = 17, .level_payloads = &.{&payload} }, .expected = error.TooManyLevels },
        // 4x4 has a 3-level halving chain; claiming 4 levels is corrupt.
        .{ .name = "overlong chain", .spec = .{ .mip_count = 4, .level_payloads = &.{&payload} }, .expected = error.InvalidMipData },
        // DX10 without its extension bytes.
        .{ .name = "DX10 truncated extension", .spec = .{ .fourcc = fourcc_dx10, .level_payloads = &.{&payload} }, .expected = error.Truncated },
        .{ .name = "DX10 cube flag", .spec = .{ .fourcc = fourcc_dx10, .dxgi = dxgi_bc7_unorm, .misc_flag = dx10_misc_texturecube, .level_payloads = &.{&payload} }, .expected = error.UnsupportedCube },
        .{ .name = "DX10 array", .spec = .{ .fourcc = fourcc_dx10, .dxgi = dxgi_bc7_unorm, .array_size = 6, .level_payloads = &.{&payload} }, .expected = error.UnsupportedArraySize },
        .{ .name = "DX10 1D dimension", .spec = .{ .fourcc = fourcc_dx10, .dxgi = dxgi_bc7_unorm, .dimension = 2, .level_payloads = &.{&payload} }, .expected = error.UnsupportedDdsFormat },
    };
    for (bad_cases) |case| {
        const file = try case.spec.build(allocator);
        defer allocator.free(file);
        try testing.expectError(case.expected, decodeBlock2D(allocator, file, .{}));
    }
}

test "decodeBlock2D rejects unsupported pixel formats with a clear error" {
    const allocator = testing.allocator;
    var payload: [16]u8 = undefined;
    @memset(&payload, 0x55);

    const cases = [_]struct { name: []const u8, spec: TestDds }{
        // Uncompressed RGBA (fourCC 0 would also fail the FOURCC gate via
        // the pixel-format flags, so use an explicit unknown tag here).
        .{ .name = "unknown fourCC", .spec = .{ .fourcc = 0x41424344, .level_payloads = &.{&payload} } },
        // Premultiplied-alpha flavors: no engine interpretation agreed.
        .{ .name = "DXT2", .spec = .{ .fourcc = fourcc_dxt2, .level_payloads = &.{&payload} } },
        .{ .name = "DXT4", .spec = .{ .fourcc = fourcc_dxt4, .level_payloads = &.{&payload} } },
        // Uncompressed DXGI code behind a DX10 header.
        .{ .name = "DXGI R8G8B8A8_UNORM", .spec = .{ .fourcc = fourcc_dx10, .dxgi = 28, .level_payloads = &.{&payload} } },
        // Float DXGI code behind a DX10 header.
        .{ .name = "DXGI R16G16B16A16_FLOAT", .spec = .{ .fourcc = fourcc_dx10, .dxgi = 10, .level_payloads = &.{&payload} } },
        // Neighboring block family with no engine support.
        .{ .name = "DXGI BC6H_UF16", .spec = .{ .fourcc = fourcc_dx10, .dxgi = 95, .level_payloads = &.{&payload} } },
    };
    for (cases) |case| {
        const file = try case.spec.build(allocator);
        defer allocator.free(file);
        try testing.expectError(error.UnsupportedDdsFormat, decodeBlock2D(allocator, file, .{}));
    }
}

test "decodeBlock2D validates non-multiple dimensions and partial chains" {
    const allocator = testing.allocator;
    // 5x3 DXT5: ceil to 2x1 blocks -> 32 B; mip 1 is 2x1 -> 1 block -> 16 B.
    var level0: [32]u8 = undefined;
    for (&level0, 0..) |*b, i| b.* = @intCast(100 + i);
    const level1 = [_]u8{0xAA} ** 16;
    const file = try (TestDds{
        .fourcc = fourcc_dxt5,
        .width = 5,
        .height = 3,
        .mip_count = 2,
        .level_payloads = &.{ &level0, &level1 },
    }).build(allocator);
    defer allocator.free(file);

    var raw = try decodeBlock2D(allocator, file, .{});
    defer raw.deinit(allocator);
    try testing.expectEqual(ktx2.BlockFormat.bc3_unorm, raw.format);
    try testing.expectEqual(@as(u32, 2), raw.num_levels);
    try testing.expectEqualSlices(u8, &level0, raw.levels[0].?);
    try testing.expectEqualSlices(u8, &level1, raw.levels[1].?);
    try testing.expectEqual(@as(usize, 32 + 16), raw.totalBytes());

    // A single-level file for a mippable base size is a valid partial
    // chain (no synthesis — uploads exactly as authored).
    var single: [32]u8 = undefined;
    @memset(&single, 0x77);
    const partial = try (TestDds{
        .fourcc = fourcc_dxt5,
        .width = 5,
        .height = 3,
        .level_payloads = &.{&single},
    }).build(allocator);
    defer allocator.free(partial);
    var raw_single = try decodeBlock2D(allocator, partial, .{});
    defer raw_single.deinit(allocator);
    try testing.expectEqual(@as(u32, 1), raw_single.num_levels);
}

test "Texture.decodeImageMemory routes DDS payloads to the block path" {
    // Function-local import: production code depends only on ktx2.zig;
    // only this routing test needs the Texture facade.
    const Texture = @import("texture.zig").Texture;
    const allocator = testing.allocator;

    var payload: [16]u8 = undefined;
    for (&payload, 0..) |*b, i| b.* = @intCast(i);
    const file = try (TestDds{
        .fourcc = fourcc_dxt5,
        .level_payloads = &.{&payload},
    }).build(allocator);
    defer allocator.free(file);

    // The caller's color/data decision flows into the legacy sRGB choice:
    // data slot -> BC3_RGBA, color slot -> BC3_SRGBA.
    var linear = try Texture.decodeImageMemory(allocator, file, .{ .srgb_to_linear = false });
    defer linear.deinit(allocator);
    switch (linear) {
        .block => |b| {
            try testing.expectEqual(ktx2.BlockFormat.bc3_unorm, b.format);
            try testing.expectEqualSlices(u8, &payload, b.levels[0].?);
            try testing.expectEqual(@as(usize, 16), linear.totalBytes());
        },
        .rgba => return error.TestUnexpectedResult,
    }

    var color = try Texture.decodeImageMemory(allocator, file, .{ .srgb_to_linear = true });
    defer color.deinit(allocator);
    switch (color) {
        .block => |b| try testing.expectEqual(ktx2.BlockFormat.bc3_srgb, b.format),
        .rgba => return error.TestUnexpectedResult,
    }

    var slot_color = try Texture.decodeImageMemory(allocator, file, .{ .slot = .color });
    defer slot_color.deinit(allocator);
    switch (slot_color) {
        .block => |b| try testing.expectEqual(ktx2.BlockFormat.bc3_srgb, b.format),
        .rgba => return error.TestUnexpectedResult,
    }

    var slot_data = try Texture.decodeImageMemory(allocator, file, .{ .slot = .data });
    defer slot_data.deinit(allocator);
    switch (slot_data) {
        .block => |b| try testing.expectEqual(ktx2.BlockFormat.bc3_unorm, b.format),
        .rgba => return error.TestUnexpectedResult,
    }

    // The RGBA8-only entry point names the correct API for valid files...
    try testing.expectError(error.DdsRequiresBlockDecode, Texture.decodeMemory(allocator, file, .{}));
    // ...foreign data still falls through to stb (unchanged behavior)...
    try testing.expectError(error.ImageDecodeFailed, Texture.decodeMemory(allocator, "png data pretending", .{}));
    // ...and surfaces the file's own validation error for corrupt DDS.
    const short = try (TestDds{
        .fourcc = fourcc_dxt5,
        .level_payloads = &.{&payload},
        .trailing_garbage = true,
    }).build(allocator);
    defer allocator.free(short);
    try testing.expectError(error.InvalidMipData, Texture.decodeMemory(allocator, short, .{}));
}
