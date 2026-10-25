const std = @import("std");
const testing = std.testing;
const math = @import("math");

const texture = @import("texture.zig");
const Texture = texture.Texture;
const CubeTexture = texture.CubeTexture;

const ktx2 = @import("ktx2.zig");
const magic = ktx2.magic;
const sniff = ktx2.sniff;
const decode2D = ktx2.decode2D;
const decodeCube = ktx2.decodeCube;
const formatFromVk = ktx2.formatFromVk;
const blockFormatFromVk = ktx2.blockFormatFromVk;
const BlockFormat = ktx2.BlockFormat;
const isBlockKtx2 = ktx2.isBlockKtx2;
const decodeBlock2D = ktx2.decodeBlock2D;
const hdrFormatFromVk = ktx2.hdrFormatFromVk;
const HdrFormat = ktx2.HdrFormat;
const isHdrKtx2 = ktx2.isHdrKtx2;
const decodeHdr2D = ktx2.decodeHdr2D;
const decodeHdrCube = ktx2.decodeHdrCube;
const floatToHalfBits = ktx2.floatToHalfBits;
const halfBitsToFloat = ktx2.halfBitsToFloat;
const HALF_ONE = ktx2.HALF_ONE;
const isBasisKtx2 = ktx2.isBasisKtx2;
const basisInfo = ktx2.basisInfo;
const decodeBasis2D = ktx2.decodeBasis2D;
const preferredBasisTarget = ktx2.preferredBasisTarget;
const DecodeError = ktx2.DecodeError;
const Format = ktx2.Format;
const BasisKind = ktx2.BasisKind;
const BasisTarget = ktx2.BasisTarget;
const header_and_index_size = ktx2.header_and_index_size;
const level_index_entry_size = ktx2.level_index_entry_size;

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

    fn alignForwardAny(val: u64, alignment: usize) u64 {
        if (alignment == 0) return val;
        const rem = val % alignment;
        if (rem == 0) return val;
        return val + (alignment - rem);
    }

    fn texelBlockSize(self: TestKtx2) usize {
        if (formatFromVk(self.vk_format)) |f| return f.texelBlockSize();
        if (hdrFormatFromVk(self.vk_format)) |h| return h.texelBlockSize();
        // Block payloads align to the block size rounded up to 4
        // (lcm(block, 4) per the KTX2 mipPadding rule: 8 for BC1, 16 for
        // the other families), matching real encoders (toktx/ktx).
        if (blockFormatFromVk(self.vk_format)) |b| return b.blockByteSize();
        return 4;
    }

    fn build(self: TestKtx2, allocator: std.mem.Allocator) ![]u8 {
        const n_levels: usize = @max(1, self.level_count);
        const sz = self.texelBlockSize();
        const block: usize = if (sz == 6 or sz == 12) 12 else @max(sz, 4);

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
            cursor = alignForwardAny(cursor, block);
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
    const level0 = math.repeatPattern(4, &[_]u8{ 1, 2, 3, 4 }); // 4x1... keep 4 texels
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

    const level0 = @as([4]u8, @splat(0));
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
    const level0 = @as([4]u8, @splat(0));

    const cases = [_]struct { name: []const u8, spec: TestKtx2, expected: DecodeError }{
        .{ .name = "BasisLZ", .spec = .{ .width = 1, .height = 1, .supercompression = 1, .level_payloads = &.{&level0} }, .expected = error.UnsupportedSupercompression },
        .{ .name = "Zstandard", .spec = .{ .width = 1, .height = 1, .supercompression = 2, .level_payloads = &.{&level0} }, .expected = error.UnsupportedSupercompression },
        .{ .name = "ZLIB", .spec = .{ .width = 1, .height = 1, .supercompression = 3, .level_payloads = &.{&level0} }, .expected = error.UnsupportedSupercompression },
        .{ .name = "BC2 (block compressed)", .spec = .{ .vk_format = 135, .width = 4, .height = 4, .level_payloads = &.{&level0} }, .expected = error.UnsupportedVkFormat },
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

test "blockFormatFromVk covers the BC, ETC2 RGBA8, and ASTC 4x4 subsets" {
    try testing.expectEqual(BlockFormat.bc1_unorm, blockFormatFromVk(133).?);
    try testing.expectEqual(BlockFormat.bc2_unorm, blockFormatFromVk(135).?);
    try testing.expectEqual(BlockFormat.bc3_unorm, blockFormatFromVk(137).?);
    try testing.expectEqual(BlockFormat.bc3_srgb, blockFormatFromVk(138).?);
    try testing.expectEqual(BlockFormat.bc7_unorm, blockFormatFromVk(145).?);
    try testing.expectEqual(BlockFormat.bc7_srgb, blockFormatFromVk(146).?);
    try testing.expectEqual(BlockFormat.etc2_rgba8_unorm, blockFormatFromVk(151).?);
    try testing.expectEqual(BlockFormat.etc2_rgba8_srgb, blockFormatFromVk(152).?);
    try testing.expectEqual(BlockFormat.astc_4x4_unorm, blockFormatFromVk(157).?);
    try testing.expectEqual(BlockFormat.astc_4x4_srgb, blockFormatFromVk(158).?);
    try testing.expect(blockFormatFromVk(0) == null); // UNDEFINED
    try testing.expect(blockFormatFromVk(37) == null); // RGBA8 is NOT a block format
    try testing.expect(blockFormatFromVk(131) == null); // BC1_RGB_UNORM (no-alpha flavor, out of scope)
    try testing.expect(blockFormatFromVk(134) == null); // BC1_RGB vs BC1_RGBA gap: not a listed variant
    try testing.expect(blockFormatFromVk(159) == null); // ASTC 5x4: neighboring footprint, unsupported
    try testing.expect(blockFormatFromVk(110) == null); // RGBA16F
    try testing.expect(BlockFormat.bc1_unorm.isSrgb() == false);
    try testing.expect(BlockFormat.bc2_unorm.isSrgb() == false);
    try testing.expect(BlockFormat.bc3_unorm.isSrgb() == false);
    try testing.expect(BlockFormat.bc3_srgb.isSrgb() == true);
    try testing.expect(BlockFormat.bc7_unorm.isSrgb() == false);
    try testing.expect(BlockFormat.bc7_srgb.isSrgb() == true);
    try testing.expect(BlockFormat.etc2_rgba8_unorm.isSrgb() == false);
    try testing.expect(BlockFormat.etc2_rgba8_srgb.isSrgb() == true);
    try testing.expect(BlockFormat.astc_4x4_unorm.isSrgb() == false);
    try testing.expect(BlockFormat.astc_4x4_srgb.isSrgb() == true);
    try testing.expect(BlockFormat.bc1_unorm.blockByteSize() == 8);
    try testing.expect(BlockFormat.bc2_unorm.blockByteSize() == 16);
    try testing.expect(BlockFormat.bc3_unorm.blockByteSize() == 16);
    try testing.expect(BlockFormat.bc7_unorm.blockByteSize() == 16);
    try testing.expect(BlockFormat.etc2_rgba8_unorm.blockByteSize() == 16);
    try testing.expect(BlockFormat.astc_4x4_unorm.blockByteSize() == 16);
}

test "BlockFormat.levelByteSize does ceil-divided block math" {
    const bc7 = BlockFormat.bc7_unorm;
    try testing.expectEqual(@as(?u64, 16), bc7.levelByteSize(4, 4)); // one block
    try testing.expectEqual(@as(?u64, 16), bc7.levelByteSize(1, 1)); // sub-block still one block
    try testing.expectEqual(@as(?u64, 64), bc7.levelByteSize(8, 8)); // 2x2 blocks
    try testing.expectEqual(@as(?u64, 64), bc7.levelByteSize(5, 5)); // ceil to 2x2 blocks
    try testing.expectEqual(@as(?u64, 32), bc7.levelByteSize(5, 3)); // 2x1 blocks
    try testing.expectEqual(@as(?u64, 16), bc7.levelByteSize(2, 1)); // 1x1 blocks
}

test "isBlockKtx2 routes by vkFormat without validating the container" {
    const allocator = testing.allocator;
    var block_payload: [16]u8 = undefined;
    for (&block_payload, 0..) |*b, i| b.* = @intCast(i);
    const block_ktx = try (TestKtx2{
        .vk_format = 145, // BC7_UNORM
        .width = 4,
        .height = 4,
        .level_payloads = &.{&block_payload},
    }).build(allocator);
    defer allocator.free(block_ktx);
    try testing.expect(isBlockKtx2(block_ktx));

    const rgba_payload = @as([4]u8, @splat(0));
    const rgba_ktx = try (TestKtx2{
        .width = 1,
        .height = 1,
        .level_payloads = &.{&rgba_payload},
    }).build(allocator);
    defer allocator.free(rgba_ktx);
    try testing.expect(!isBlockKtx2(rgba_ktx));

    try testing.expect(!isBlockKtx2("png data pretending"));
    try testing.expect(!isBlockKtx2(&.{}));
    try testing.expect(!isBlockKtx2(magic[0..8])); // truncated identifier
}

test "decodeBlock2D reads a BC7 8x8 two-level file exactly as authored" {
    const allocator = testing.allocator;
    // Level 0: 8x8 -> 2x2 blocks -> 64 B; level 1: 4x4 -> 1 block -> 16 B.
    var level0: [64]u8 = undefined;
    for (&level0, 0..) |*b, i| b.* = @intCast(i % 251);
    const level1 = [_]u8{ 7, 7, 7, 7, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const ktx = try (TestKtx2{
        .vk_format = 145, // BC7_UNORM_BLOCK
        .width = 8,
        .height = 8,
        .level_count = 2,
        .level_payloads = &.{ &level0, &level1 },
    }).build(allocator);
    defer allocator.free(ktx);

    var raw = try decodeBlock2D(allocator, ktx);
    defer raw.deinit(allocator);
    try testing.expectEqual(@as(u32, 8), raw.width);
    try testing.expectEqual(@as(u32, 8), raw.height);
    try testing.expectEqual(@as(u32, 2), raw.num_levels);
    try testing.expectEqual(BlockFormat.bc7_unorm, raw.format);
    try testing.expectEqualSlices(u8, &level0, raw.levels[0].?);
    try testing.expectEqualSlices(u8, &level1, raw.levels[1].?);
    try testing.expectEqual(@as(usize, 64 + 16), raw.totalBytes());
    // Owned copies (the asset queue frees the source right after decode):
    // clobbering the file leaves the decoded levels intact.
    @memset(ktx, 0xFF);
    try testing.expectEqual(@as(u8, 0), raw.levels[0].?[0]);
}

test "decodeBlock2D handles non-multiple dimensions and sRGB tags" {
    const allocator = testing.allocator;
    // 5x3 ASTC: ceil to 2x1 blocks -> 32 B; mip 1 is 2x1 -> 1 block -> 16 B.
    var level0: [32]u8 = undefined;
    for (&level0, 0..) |*b, i| b.* = @intCast(100 + i);
    const level1 = @as([16]u8, @splat(0xAA));
    const ktx = try (TestKtx2{
        .vk_format = 158, // ASTC_4x4_SRGB_BLOCK
        .width = 5,
        .height = 3,
        .level_count = 2,
        .level_payloads = &.{ &level0, &level1 },
    }).build(allocator);
    defer allocator.free(ktx);

    var raw = try decodeBlock2D(allocator, ktx);
    defer raw.deinit(allocator);
    try testing.expectEqual(BlockFormat.astc_4x4_srgb, raw.format);
    try testing.expect(raw.format.isSrgb());
    try testing.expectEqualSlices(u8, &level0, raw.levels[0].?);
    try testing.expectEqualSlices(u8, &level1, raw.levels[1].?);
    try testing.expectEqual(@as(usize, 32 + 16), raw.totalBytes());
}

test "decodeBlock2D rejects supercompressed, foreign and cube block files" {
    const allocator = testing.allocator;
    var payload: [16]u8 = undefined;
    for (&payload, 0..) |*b, i| b.* = @intCast(i);

    // Supercompression is rejected before the format gate (same order as
    // the RGBA8 path): scheme 1 on a BC7 file is UnsupportedSupercompression.
    const zstd_ktx = try (TestKtx2{
        .vk_format = 145,
        .width = 4,
        .height = 4,
        .supercompression = 1,
        .level_payloads = &.{&payload},
    }).build(allocator);
    defer allocator.free(zstd_ktx);
    try testing.expectError(error.UnsupportedSupercompression, decodeBlock2D(allocator, zstd_ktx));

    // Unknown vkFormat (ASTC 5x4 neighbors the supported 4x4 footprint).
    const footprint_ktx = try (TestKtx2{
        .vk_format = 159,
        .width = 4,
        .height = 4,
        .level_payloads = &.{&payload},
    }).build(allocator);
    defer allocator.free(footprint_ktx);
    try testing.expectError(error.UnsupportedVkFormat, decodeBlock2D(allocator, footprint_ktx));

    // 16-bit typeSize is rejected even for block formats.
    const type_ktx = try (TestKtx2{
        .vk_format = 145,
        .type_size = 2,
        .width = 4,
        .height = 4,
        .level_payloads = &.{&payload},
    }).build(allocator);
    defer allocator.free(type_ktx);
    try testing.expectError(error.UnsupportedTypeSize, decodeBlock2D(allocator, type_ktx));

    // Cube block files: the 2D block path must not silently drop faces.
    var cube_level: [16 * 6]u8 = undefined;
    for (&cube_level, 0..) |*b, i| b.* = @intCast(i % 251);
    const cube_ktx = try (TestKtx2{
        .vk_format = 145,
        .width = 4,
        .height = 4,
        .face_count = 6,
        .level_payloads = &.{&cube_level},
    }).build(allocator);
    defer allocator.free(cube_ktx);
    try testing.expectError(error.UnsupportedFaceCount, decodeBlock2D(allocator, cube_ktx));

    // Truncated level payloads disagree with the block-grid size.
    const short_ktx = try (TestKtx2{
        .vk_format = 157,
        .width = 4,
        .height = 4,
        .level_payloads = &.{&payload},
        .truncate_last_level = true,
    }).build(allocator);
    defer allocator.free(short_ktx);
    try testing.expectError(error.InvalidLevelData, decodeBlock2D(allocator, short_ktx));

    // The RGBA8 path stays closed to block formats (regression: no silent
    // widening of decode2D).
    const bc7_ktx = try (TestKtx2{
        .vk_format = 145,
        .width = 4,
        .height = 4,
        .level_payloads = &.{&payload},
    }).build(allocator);
    defer allocator.free(bc7_ktx);
    try testing.expectError(error.UnsupportedVkFormat, decode2D(allocator, bc7_ktx, .{}));
    try testing.expect(!sniff("definitely not ktx2"));
    try testing.expectError(error.NotKtx2, decodeBlock2D(allocator, "definitely not ktx2"));
}

test "Texture.decodeImageMemory routes block payloads without touching the RGBA8 path" {
    const allocator = testing.allocator;
    // Block fixture: 4x4 BC7, one 16-byte level.
    var payload: [16]u8 = undefined;
    for (&payload, 0..) |*b, i| b.* = @intCast(i);
    const block_file = try (TestKtx2{
        .vk_format = 145,
        .width = 4,
        .height = 4,
        .level_payloads = &.{&payload},
    }).build(allocator);
    defer allocator.free(block_file);

    var block_img = try Texture.decodeImageMemory(allocator, block_file, .{});
    defer block_img.deinit(allocator);
    switch (block_img) {
        .block => |b| {
            try testing.expectEqual(BlockFormat.bc7_unorm, b.format);
            try testing.expectEqualSlices(u8, &payload, b.levels[0].?);
            try testing.expectEqual(@as(usize, 16), block_img.totalBytes());
        },
        else => return error.TestUnexpectedResult,
    }

    // Uncompressed KTX2 still routes to .rgba with identical bytes and the
    // same byte tally the asset budget consumes.
    const rgba_level = [_]u8{ 200, 0, 0, 255 };
    const rgba_file = try (TestKtx2{
        .vk_format = 37,
        .width = 1,
        .height = 1,
        .level_payloads = &.{&rgba_level},
    }).build(allocator);
    defer allocator.free(rgba_file);
    var rgba_img = try Texture.decodeImageMemory(allocator, rgba_file, .{ .srgb_to_linear = false });
    defer rgba_img.deinit(allocator);
    switch (rgba_img) {
        .rgba => |r| {
            try testing.expectEqual([4]u8{ 200, 0, 0, 255 }, r.levels[0].?[0..4].*);
            try testing.expectEqual(@as(usize, 4), rgba_img.totalBytes());
        },
        else => return error.TestUnexpectedResult,
    }
}

// ---------------------------------------------------------------------------
// Basis Universal tests — REAL toktx fixtures (src/ktx2_fixtures/,
// see README.md there), not fabricated headers. Sizes: 16x16 base, 5-level
// chains (16..1): RGBA32 level bytes [1024,256,64,16,4], BC7/ASTC/ETC2 block
// bytes [256,64,16,16,16].
// ---------------------------------------------------------------------------

const fx_etc1s_rgb_mip = @embedFile("ktx2_fixtures/fx_etc1s_rgb_mip.ktx2");
const fx_etc1s_rgba_mip = @embedFile("ktx2_fixtures/fx_etc1s_rgba_mip.ktx2");
const fx_uastc_rgba_mip = @embedFile("ktx2_fixtures/fx_uastc_rgba_mip.ktx2");
const fx_uastc_rgba_zstd = @embedFile("ktx2_fixtures/fx_uastc_rgba_zstd.ktx2");
const fx_uastc_rgb_flat = @embedFile("ktx2_fixtures/fx_uastc_rgb_flat.ktx2");
const fx_uastc_rgb_flat_linear = @embedFile("ktx2_fixtures/fx_uastc_rgb_flat_linear.ktx2");
const fx_etc1s_rgb_flat_linear = @embedFile("ktx2_fixtures/fx_etc1s_rgb_flat_linear.ktx2");
const fx_uastc_cube = @embedFile("ktx2_fixtures/fx_uastc_cube.ktx2");

test "isBasisKtx2 routes by the UNDEFINED marker only" {
    for ([_][]const u8{
        fx_etc1s_rgb_mip,
        fx_etc1s_rgba_mip,
        fx_uastc_rgba_mip,
        fx_uastc_rgba_zstd,
        fx_uastc_rgb_flat,
        fx_uastc_rgb_flat_linear,
        fx_etc1s_rgb_flat_linear,
        fx_uastc_cube,
    }) |fx| try testing.expect(isBasisKtx2(fx));

    // Non-Basis KTX2 (RGBA8/BC7), foreign data, truncated identifier.
    const allocator = testing.allocator;
    const rgba_level = @as([4]u8, @splat(0));
    const rgba_ktx = try (TestKtx2{
        .width = 1,
        .height = 1,
        .level_payloads = &.{&rgba_level},
    }).build(allocator);
    defer allocator.free(rgba_ktx);
    try testing.expect(!isBasisKtx2(rgba_ktx));
    try testing.expect(!isBasisKtx2("png data pretending"));
    try testing.expect(!isBasisKtx2(fx_uastc_rgb_flat[0..8]));
    // Scheme > 3 is not a Basis payload even behind the UNDEFINED marker.
    const patched = try allocator.dupe(u8, fx_uastc_rgb_flat);
    defer allocator.free(patched);
    std.mem.writeInt(u32, patched[44..48], 4, .little);
    try testing.expect(!isBasisKtx2(patched));
}

test "basisInfo reads real ETC1S/UASTC headers" {
    const etc1s = try basisInfo(fx_etc1s_rgb_mip);
    try testing.expectEqual(BasisKind.etc1s, etc1s.kind);
    try testing.expectEqual(@as(u32, 16), etc1s.width);
    try testing.expectEqual(@as(u32, 16), etc1s.height);
    try testing.expectEqual(@as(u32, 5), etc1s.levels);
    try testing.expect(etc1s.is_srgb);
    try testing.expect(!etc1s.has_alpha);

    const etc1s_alpha = try basisInfo(fx_etc1s_rgba_mip);
    try testing.expectEqual(BasisKind.etc1s, etc1s_alpha.kind);
    try testing.expect(etc1s_alpha.has_alpha);

    const uastc = try basisInfo(fx_uastc_rgba_mip);
    try testing.expectEqual(BasisKind.uastc, uastc.kind);
    try testing.expectEqual(@as(u32, 5), uastc.levels);
    try testing.expect(uastc.is_srgb);
    try testing.expect(uastc.has_alpha);

    const linear = try basisInfo(fx_uastc_rgb_flat_linear);
    try testing.expect(!linear.is_srgb);
    try testing.expect(!linear.has_alpha);

    // The 2D-only entries reject the cube before the transcoder runs
    // (same UnsupportedFaceCount contract as decodeBlock2D).
    try testing.expectError(error.UnsupportedFaceCount, basisInfo(fx_uastc_cube));
    try testing.expectError(error.NotKtx2, basisInfo("png data pretending"));
    try testing.expectError(error.Truncated, basisInfo(fx_uastc_rgb_flat[0..40]));
    // Header-only prefix: envelope passes, the transcoder init fails.
    try testing.expectError(error.NotBasisKtx2, basisInfo(fx_uastc_rgb_flat[0..80]));
}

test "decodeBasis2D transcodes a real ETC1S mip chain to BC7" {
    const allocator = testing.allocator;
    var img = try decodeBasis2D(allocator, fx_etc1s_rgb_mip, .bc7, .{});
    defer img.deinit(allocator);
    switch (img) {
        .block => |b| {
            try testing.expectEqual(BlockFormat.bc7_srgb, b.format);
            try testing.expectEqual(@as(u32, 16), b.width);
            try testing.expectEqual(@as(u32, 5), b.num_levels);
            const want = [_]usize{ 256, 64, 16, 16, 16 };
            for (want, 0..) |n, m| try testing.expectEqual(n, b.levels[m].?.len);
            try testing.expectEqual(@as(usize, 368), b.totalBytes());
            // Non-degenerate transcode output (gradient in, gradient out).
            var all_same = true;
            for (b.levels[0].?[1..]) |byte| {
                if (byte != b.levels[0].?[0]) {
                    all_same = false;
                    break;
                }
            }
            try testing.expect(!all_same);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "decodeBasis2D transcodes a real UASTC mip chain to BC7" {
    const allocator = testing.allocator;
    var img = try decodeBasis2D(allocator, fx_uastc_rgba_mip, .bc7, .{});
    defer img.deinit(allocator);
    switch (img) {
        .block => |b| {
            try testing.expectEqual(BlockFormat.bc7_srgb, b.format);
            try testing.expectEqual(@as(u32, 5), b.num_levels);
            const want = [_]usize{ 256, 64, 16, 16, 16 };
            for (want, 0..) |n, m| try testing.expectEqual(n, b.levels[m].?.len);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "decodeBasis2D transcodes a real ETC1S mip chain to ETC2 RGBA8" {
    const allocator = testing.allocator;
    var img = try decodeBasis2D(allocator, fx_etc1s_rgba_mip, .etc2_rgba, .{});
    defer img.deinit(allocator);
    switch (img) {
        .block => |b| {
            try testing.expectEqual(BlockFormat.etc2_rgba8_srgb, b.format);
            try testing.expectEqual(@as(u32, 5), b.num_levels);
            const want = [_]usize{ 256, 64, 16, 16, 16 };
            for (want, 0..) |n, m| try testing.expectEqual(n, b.levels[m].?.len);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "decodeBasis2D transcodes zstd-supercompressed UASTC (vendored zstd)" {
    const allocator = testing.allocator;
    var img = try decodeBasis2D(allocator, fx_uastc_rgba_zstd, .bc7, .{});
    defer img.deinit(allocator);
    switch (img) {
        .block => |b| {
            try testing.expectEqual(BlockFormat.bc7_srgb, b.format);
            try testing.expectEqual(@as(u32, 1), b.num_levels);
            try testing.expectEqual(@as(usize, 256), b.levels[0].?.len);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "decodeBasis2D selects UNORM variants for linear-transfer files" {
    const allocator = testing.allocator;
    var bc7 = try decodeBasis2D(allocator, fx_etc1s_rgb_flat_linear, .bc7, .{});
    defer bc7.deinit(allocator);
    switch (bc7) {
        .block => |b| try testing.expectEqual(BlockFormat.bc7_unorm, b.format),
        else => return error.TestUnexpectedResult,
    }
    var astc = try decodeBasis2D(allocator, fx_uastc_rgb_flat_linear, .astc, .{});
    defer astc.deinit(allocator);
    switch (astc) {
        .block => |b| {
            try testing.expectEqual(BlockFormat.astc_4x4_unorm, b.format);
            try testing.expectEqual(@as(usize, 256), b.levels[0].?.len);
        },
        else => return error.TestUnexpectedResult,
    }
    // sRGB flat picks the sRGB ASTC variant with identical block math.
    var astc_srgb = try decodeBasis2D(allocator, fx_uastc_rgb_flat, .astc, .{});
    defer astc_srgb.deinit(allocator);
    switch (astc_srgb) {
        .block => |b| try testing.expectEqual(BlockFormat.astc_4x4_srgb, b.format),
        else => return error.TestUnexpectedResult,
    }
}

test "decodeBasis2D RGBA32 fallback keeps sizes, alpha and sRGB behavior" {
    const allocator = testing.allocator;
    var img = try decodeBasis2D(allocator, fx_uastc_rgba_mip, .rgba32, .{});
    defer img.deinit(allocator);
    switch (img) {
        .rgba => |r| {
            try testing.expectEqual(@as(u32, 5), r.num_levels);
            try testing.expect(r.is_srgb); // auto from the sRGB DFD
            const want = [_]usize{ 1024, 256, 64, 16, 4 };
            for (want, 0..) |n, m| try testing.expectEqual(n, r.levels[m].?.len);
            // Gradient alpha survives the round trip (not all opaque).
            var opaque_count: usize = 0;
            var i: usize = 3;
            while (i < r.levels[0].?.len) : (i += 4) {
                if (r.levels[0].?[i] == 255) opaque_count += 1;
            }
            try testing.expect(opaque_count < 256);
        },
        else => return error.TestUnexpectedResult,
    }

    // Opaque ETC1S decodes to fully opaque alpha.
    var opaque_img = try decodeBasis2D(allocator, fx_etc1s_rgb_mip, .rgba32, .{ .srgb_to_linear = false });
    defer opaque_img.deinit(allocator);
    switch (opaque_img) {
        .rgba => |r| {
            try testing.expect(!r.is_srgb); // explicit override wins over DFD
            var i: usize = 3;
            while (i < r.levels[0].?.len) : (i += 4) {
                try testing.expectEqual(@as(u8, 255), r.levels[0].?[i]);
            }
        },
        else => return error.TestUnexpectedResult,
    }

    // Single-level RGBA32 honors gen_mipmaps like every other RGBA8 path.
    var chained = try decodeBasis2D(allocator, fx_uastc_rgb_flat, .rgba32, .{});
    defer chained.deinit(allocator);
    switch (chained) {
        .rgba => |r| try testing.expectEqual(Texture.mipLevelCount(16, 16), r.num_levels),
        else => return error.TestUnexpectedResult,
    }
    var flat = try decodeBasis2D(allocator, fx_uastc_rgb_flat, .rgba32, .{ .gen_mipmaps = false });
    defer flat.deinit(allocator);
    switch (flat) {
        .rgba => |r| try testing.expectEqual(@as(u32, 1), r.num_levels),
        else => return error.TestUnexpectedResult,
    }
}

test "preferredBasisTarget ladders BC7, ASTC, ETC2, then RGBA32" {
    const bc7: Texture.BlockSupport = .{ .bc7_sample = true };
    try testing.expectEqual(BasisTarget.bc7, preferredBasisTarget(bc7));
    const astc: Texture.BlockSupport = .{ .astc_sample = true };
    try testing.expectEqual(BasisTarget.astc, preferredBasisTarget(astc));
    const etc2: Texture.BlockSupport = .{ .etc2_sample = true };
    try testing.expectEqual(BasisTarget.etc2_rgba, preferredBasisTarget(etc2));
    const mobile: Texture.BlockSupport = .{ .astc_sample = true, .etc2_sample = true };
    try testing.expectEqual(BasisTarget.astc, preferredBasisTarget(mobile));
    const both: Texture.BlockSupport = .{ .bc7_sample = true, .astc_sample = true };
    try testing.expectEqual(BasisTarget.bc7, preferredBasisTarget(both));
    const bc7_etc2: Texture.BlockSupport = .{ .bc7_sample = true, .etc2_sample = true };
    try testing.expectEqual(BasisTarget.bc7, preferredBasisTarget(bc7_etc2));
    try testing.expectEqual(BasisTarget.rgba32, preferredBasisTarget(.{}));
    // Sample-without-filter still counts (upload forces NEAREST there).
    const no_filter: Texture.BlockSupport = .{ .bc7_sample = true };
    try testing.expectEqual(BasisTarget.bc7, preferredBasisTarget(no_filter));
}

test "decodeBasis2D rejects malformed and out-of-subset containers" {
    const allocator = testing.allocator;
    try testing.expectError(error.NotKtx2, decodeBasis2D(allocator, "png data", .bc7, .{}));
    try testing.expectError(error.Truncated, decodeBasis2D(allocator, fx_uastc_rgb_flat[0..40], .bc7, .{}));
    try testing.expectError(error.UnsupportedFaceCount, decodeBasis2D(allocator, fx_uastc_cube, .bc7, .{}));

    // Non-Basis KTX2 on the Basis entry: unsupported vkFormat.
    const rgba_level = @as([4]u8, @splat(0));
    const rgba_ktx = try (TestKtx2{
        .width = 1,
        .height = 1,
        .level_payloads = &.{&rgba_level},
    }).build(allocator);
    defer allocator.free(rgba_ktx);
    try testing.expectError(error.UnsupportedVkFormat, decodeBasis2D(allocator, rgba_ktx, .bc7, .{}));

    // Scheme 4 behind the UNDEFINED marker: supercompression first.
    const sc4 = try allocator.dupe(u8, fx_uastc_rgb_flat);
    defer allocator.free(sc4);
    std.mem.writeInt(u32, sc4[44..48], 4, .little);
    try testing.expectError(error.UnsupportedSupercompression, decodeBasis2D(allocator, sc4, .bc7, .{}));

    // levelCount 0 (implicit single level, no encoder writes it): the
    // transcoder has no level index to work with — explicit BasisUnsupported.
    const lc0 = try allocator.dupe(u8, fx_uastc_rgb_flat);
    defer allocator.free(lc0);
    std.mem.writeInt(u32, lc0[40..44], 0, .little);
    try testing.expectError(error.BasisUnsupported, decodeBasis2D(allocator, lc0, .bc7, .{}));

    // Truncated payloads fail at init (the transcoder validates level bounds
    // up front): explicit NotBasisKtx2, never silent.
    try testing.expectError(error.NotBasisKtx2, decodeBasis2D(allocator, fx_uastc_rgb_flat[0 .. fx_uastc_rgb_flat.len - 8], .bc7, .{}));
    try testing.expectError(error.NotBasisKtx2, decodeBasis2D(allocator, fx_etc1s_rgb_flat_linear[0 .. fx_etc1s_rgb_flat_linear.len - 4], .bc7, .{}));
    // Corrupt slice CONTENT (valid bounds, garbage bytes): init passes, the
    // level transcode fails. fx_etc1s_rgb_flat_linear carries its 8-byte
    // slice at file offset 448.
    const corrupt = try allocator.dupe(u8, fx_etc1s_rgb_flat_linear);
    defer allocator.free(corrupt);
    corrupt[450] ^= 0xFF;
    corrupt[453] ^= 0xFF;
    try testing.expectError(error.BasisTranscodeFailed, decodeBasis2D(allocator, corrupt, .bc7, .{}));
}

test "Texture.decodeImageMemory routes real Basis files by target" {
    const allocator = testing.allocator;
    var block_img = try Texture.decodeImageMemory(allocator, fx_uastc_rgba_mip, .{});
    defer block_img.deinit(allocator);
    switch (block_img) {
        .block => |b| {
            try testing.expectEqual(BlockFormat.bc7_srgb, b.format);
            try testing.expectEqual(@as(u32, 5), b.num_levels);
        },
        else => return error.TestUnexpectedResult,
    }

    var rgba_img = try Texture.decodeImageMemory(allocator, fx_uastc_rgba_mip, .{ .basis_target = .rgba32 });
    defer rgba_img.deinit(allocator);
    switch (rgba_img) {
        .rgba => |r| try testing.expectEqual(@as(u32, 5), r.num_levels),
        else => return error.TestUnexpectedResult,
    }
}

test "Texture.decodeMemory rejects Basis with its own reason" {
    const allocator = testing.allocator;
    try testing.expectError(error.BasisRequiresBlockDecode, Texture.decodeMemory(allocator, fx_uastc_rgb_flat, .{}));
    try testing.expectError(error.Truncated, Texture.decodeMemory(allocator, fx_uastc_rgb_flat[0..40], .{}));
}

/// Builds a raw 104-byte header+single-level-index container with explicit
/// level-index fields, for malformed-input overflow cases the TestKtx2
/// builder cannot express (absurd dims, hostile offsets).
fn buildRawOverflowCase(
    allocator: std.mem.Allocator,
    width: u32,
    height: u32,
    face_count: u32,
    byte_offset: u64,
    byte_length: u64,
    uncompressed_length: u64,
) ![]u8 {
    var buf = try allocator.alloc(u8, header_and_index_size + level_index_entry_size);
    @memcpy(buf[0..magic.len], &magic);
    std.mem.writeInt(u32, buf[12..16], 37, .little); // RGBA8_UNORM
    std.mem.writeInt(u32, buf[16..20], 1, .little); // typeSize
    std.mem.writeInt(u32, buf[20..24], width, .little);
    std.mem.writeInt(u32, buf[24..28], height, .little);
    std.mem.writeInt(u32, buf[28..32], 0, .little); // depth
    std.mem.writeInt(u32, buf[32..36], 0, .little); // layers
    std.mem.writeInt(u32, buf[36..40], face_count, .little);
    std.mem.writeInt(u32, buf[40..44], 1, .little); // levelCount
    std.mem.writeInt(u32, buf[44..48], 0, .little); // scheme NONE
    std.mem.writeInt(u32, buf[48..52], 0, .little); // dfd offset
    std.mem.writeInt(u32, buf[52..56], 0, .little); // dfd length
    std.mem.writeInt(u32, buf[56..60], 0, .little); // kvd offset
    std.mem.writeInt(u32, buf[60..64], 0, .little); // kvd length
    std.mem.writeInt(u64, buf[64..72], 0, .little); // sgd offset
    std.mem.writeInt(u64, buf[72..80], 0, .little); // sgd length
    std.mem.writeInt(u64, buf[80..88], byte_offset, .little);
    std.mem.writeInt(u64, buf[88..96], byte_length, .little);
    std.mem.writeInt(u64, buf[96..104], uncompressed_length, .little);
    return buf;
}

test "decode2D rejects dimension-size multiplication overflow without panicking" {
    const allocator = testing.allocator;
    // 3e9 x 3e9 x 4 B: w*h fits u64 (9e18) but x texel size (3.6e19)
    // exceeds maxInt(u64). Unchecked math wraps/panics; checked math must
    // report InvalidLevelData.
    const ktx = try buildRawOverflowCase(allocator, 3_000_000_000, 3_000_000_000, 1, 104, 0, 0);
    defer allocator.free(ktx);
    try testing.expectEqual(@as(usize, 104), ktx.len);
    try testing.expectError(error.InvalidLevelData, decode2D(allocator, ktx, .{}));
}

test "decode2D rejects level offset+length addition overflow without panicking" {
    const allocator = testing.allocator;
    // 1x1 RGBA8 expects exactly 4 bytes; offset=maxInt(u64) makes
    // start+length overflow u64. Must report Truncated, never panic or wrap
    // into a small slice.
    const ktx = try buildRawOverflowCase(
        allocator,
        1,
        1,
        1,
        std.math.maxInt(u64),
        4,
        4,
    );
    defer allocator.free(ktx);
    const expected = if (@bitSizeOf(usize) < 64) error.InvalidLevelData else error.Truncated;
    try testing.expectError(expected, decode2D(allocator, ktx, .{}));
}

test "decodeCube rejects face-count multiplication overflow without panicking" {
    const allocator = testing.allocator;
    // 1.5e9 x 1.5e9 x 4 B fits u64 (9e18) but x 6 faces (5.4e19) does not:
    // the faces factor must be checked too.
    const ktx = try buildRawOverflowCase(allocator, 1_500_000_000, 1_500_000_000, 6, 104, 0, 0);
    defer allocator.free(ktx);
    try testing.expectError(error.InvalidLevelData, decodeCube(allocator, ktx, .{}));
}

test "decodeBlock2D rejects level offset+length addition overflow without panicking" {
    const allocator = testing.allocator;
    // 4x4 BC7 expects exactly one 16-byte block; offset=maxInt(u64) makes
    // start+length overflow u64. Must never panic or wrap into a small slice.
    // On 64-bit targets the usize cast succeeds and the checked addition
    // reports Truncated; on 32-bit targets the cast itself rejects the
    // offset as InvalidLevelData. Assert the target-specific error.
    const ktx = try buildRawOverflowCase(allocator, 4, 4, 1, std.math.maxInt(u64), 16, 16);
    defer allocator.free(ktx);
    std.mem.writeInt(u32, ktx[12..16], 145, .little); // BC7_UNORM_BLOCK
    const expected = if (@bitSizeOf(usize) < 64) error.InvalidLevelData else error.Truncated;
    try testing.expectError(expected, decodeBlock2D(allocator, ktx));
}

test "blockFormatFromVk accepts BC4/BC5/BC6H with exact block sizes" {
    const cases = [_]struct { vk: u32, fmt: ktx2.BlockFormat, block_bytes: usize }{
        .{ .vk = 139, .fmt = .bc4_unorm, .block_bytes = 8 },
        .{ .vk = 140, .fmt = .bc4_snorm, .block_bytes = 8 },
        .{ .vk = 141, .fmt = .bc5_unorm, .block_bytes = 16 },
        .{ .vk = 142, .fmt = .bc5_snorm, .block_bytes = 16 },
        .{ .vk = 143, .fmt = .bc6h_uf16, .block_bytes = 16 },
        .{ .vk = 144, .fmt = .bc6h_sf16, .block_bytes = 16 },
    };
    for (cases) |c| {
        const got = ktx2.blockFormatFromVk(c.vk) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqual(c.fmt, got);
        try std.testing.expectEqual(c.block_bytes, got.blockByteSize());
        try std.testing.expect(!got.isSrgb()); // BC4/5/6H are linear by definition
    }
    try std.testing.expectEqual(@as(?ktx2.BlockFormat, null), ktx2.blockFormatFromVk(0));
    // Level math: an 8x8 BC4 level is 2x2 blocks x 8 B = 32 bytes.
    try std.testing.expectEqual(@as(u64, 32), ktx2.BlockFormat.bc4_unorm.levelByteSize(8, 8).?);
}

test "isHdrKtx2 detects all 8 float formats and excludes non-HDR" {
    const allocator = testing.allocator;
    const hdr_vks = [_]u32{ 76, 83, 90, 97, 100, 103, 106, 109 };
    for (hdr_vks) |vk| {
        const dummy = @as([16]u8, @splat(0));
        const file = try (TestKtx2{
            .vk_format = vk,
            .type_size = 2,
            .width = 1,
            .height = 1,
            .level_payloads = &.{&dummy},
        }).build(allocator);
        defer allocator.free(file);

        try testing.expect(isHdrKtx2(file));
        try testing.expect(!isBlockKtx2(file));
        try testing.expect(!isBasisKtx2(file));
    }

    // LDR, block, and non-KTX2 data are not HDR
    try testing.expect(!isHdrKtx2("not a ktx2 file"));
    const ldr_dummy = [_]u8{ 255, 255, 255, 255 };
    const ldr_file = try (TestKtx2{
        .vk_format = 37,
        .width = 1,
        .height = 1,
        .level_payloads = &.{&ldr_dummy},
    }).build(allocator);
    defer allocator.free(ldr_file);
    try testing.expect(!isHdrKtx2(ldr_file));
}

test "halfBitsToFloat and floatToHalfBits convert standard values" {
    try testing.expectEqual(@as(u16, 0x0000), floatToHalfBits(0.0));
    try testing.expectEqual(@as(u16, HALF_ONE), floatToHalfBits(1.0));
    try testing.expectEqual(@as(u16, 0xBC00), floatToHalfBits(-1.0));
    try testing.expectEqual(@as(u16, 0x3800), floatToHalfBits(0.5));
    try testing.expectEqual(@as(u16, 0x4000), floatToHalfBits(2.0));

    try testing.expectEqual(@as(f32, 0.0), halfBitsToFloat(0x0000));
    try testing.expectEqual(@as(f32, 1.0), halfBitsToFloat(HALF_ONE));
    try testing.expectEqual(@as(f32, -1.0), halfBitsToFloat(0xBC00));
    try testing.expectEqual(@as(f32, 0.5), halfBitsToFloat(0x3800));
    try testing.expectEqual(@as(f32, 2.0), halfBitsToFloat(0x4000));
}

test "decodeHdr2D decodes RGBA16_SFLOAT exact half-float bits" {
    const allocator = testing.allocator;
    // 2x2 texture with 4 channels of u16 half-floats per pixel = 4 * 4 * 2 = 32 bytes
    var payload: [32]u8 = undefined;
    const test_vals = [_]u16{
        0x3C00, 0x4000, 0x4200, 0x4500, // pixel (0,0): 1.0, 2.0, 3.0, 5.0
        0x3800, 0x0000, 0xBC00, 0x3C00, // pixel (1,0): 0.5, 0.0, -1.0, 1.0
        0x4800, 0x4900, 0x4A00, 0x4B00, // pixel (0,1): 8.0, 9.0, 10.0, 11.0
        0x4C00, 0x4D00, 0x4E00, 0x4F00, // pixel (1,1): 12.0, 13.0, 14.0, 15.0
    };
    for (test_vals, 0..) |v, i| {
        std.mem.writeInt(u16, payload[i * 2 ..][0..2], v, .little);
    }

    const file = try (TestKtx2{
        .vk_format = 97, // VK_FORMAT_R16G16B16A16_SFLOAT
        .type_size = 2,
        .width = 2,
        .height = 2,
        .level_payloads = &.{&payload},
    }).build(allocator);
    defer allocator.free(file);

    var raw = try decodeHdr2D(allocator, file);
    defer raw.deinit(allocator);

    try testing.expectEqual(@as(u32, 2), raw.width);
    try testing.expectEqual(@as(u32, 2), raw.height);
    try testing.expectEqual(@as(u32, 1), raw.num_levels);
    try testing.expectEqual(HdrFormat.rgba16_sfloat, raw.format);
    try testing.expect(raw.levels[0] != null);
    try testing.expectEqual(@as(usize, 16), raw.levels[0].?.len);
    try testing.expectEqualSlices(u16, &test_vals, raw.levels[0].?);
    try testing.expectEqual(@as(usize, 32), raw.totalBytes());
}

test "decodeHdr2D decodes RGB16_SFLOAT expanding with alpha 1.0" {
    const allocator = testing.allocator;
    // 2x1 texture with 3 channels of u16 half-floats per pixel = 2 * 3 * 2 = 12 bytes
    var payload: [12]u8 = undefined;
    const rgb_vals = [_]u16{
        0x3C00, 0x4000, 0x4200, // pixel 0: 1.0, 2.0, 3.0
        0x3800, 0x0000, 0xBC00, // pixel 1: 0.5, 0.0, -1.0
    };
    for (rgb_vals, 0..) |v, i| {
        std.mem.writeInt(u16, payload[i * 2 ..][0..2], v, .little);
    }

    const file = try (TestKtx2{
        .vk_format = 90, // VK_FORMAT_R16G16B16_SFLOAT
        .type_size = 2,
        .width = 2,
        .height = 1,
        .level_payloads = &.{&payload},
    }).build(allocator);
    defer allocator.free(file);

    var raw = try decodeHdr2D(allocator, file);
    defer raw.deinit(allocator);

    try testing.expectEqual(@as(u32, 2), raw.width);
    try testing.expectEqual(@as(u32, 1), raw.height);
    try testing.expectEqual(@as(u32, 1), raw.num_levels);
    try testing.expectEqual(HdrFormat.rgb16_sfloat, raw.format);
    try testing.expect(raw.levels[0] != null);
    try testing.expectEqual(@as(usize, 8), raw.levels[0].?.len);

    const expected = [_]u16{
        0x3C00, 0x4000, 0x4200, HALF_ONE,
        0x3800, 0x0000, 0xBC00, HALF_ONE,
    };
    try testing.expectEqualSlices(u16, &expected, raw.levels[0].?);
}

test "decodeHdr2D decodes RG16_SFLOAT and R16_SFLOAT" {
    const allocator = testing.allocator;

    // RG16_SFLOAT (83): 1x1, 2 channels -> R, G, B=0, A=1.0
    var rg_payload: [4]u8 = undefined;
    std.mem.writeInt(u16, rg_payload[0..2], 0x3C00, .little); // 1.0
    std.mem.writeInt(u16, rg_payload[2..4], 0x4000, .little); // 2.0
    const rg_file = try (TestKtx2{
        .vk_format = 83,
        .type_size = 2,
        .width = 1,
        .height = 1,
        .level_payloads = &.{&rg_payload},
    }).build(allocator);
    defer allocator.free(rg_file);

    var rg_raw = try decodeHdr2D(allocator, rg_file);
    defer rg_raw.deinit(allocator);
    const expected_rg = [_]u16{ 0x3C00, 0x4000, 0, HALF_ONE };
    try testing.expectEqualSlices(u16, &expected_rg, rg_raw.levels[0].?);

    // R16_SFLOAT (76): 1x1, 1 channel -> R, R, R, A=1.0
    var r_payload: [2]u8 = undefined;
    std.mem.writeInt(u16, r_payload[0..2], 0x3C00, .little);
    const r_file = try (TestKtx2{
        .vk_format = 76,
        .type_size = 2,
        .width = 1,
        .height = 1,
        .level_payloads = &.{&r_payload},
    }).build(allocator);
    defer allocator.free(r_file);

    var r_raw = try decodeHdr2D(allocator, r_file);
    defer r_raw.deinit(allocator);
    const expected_r = [_]u16{ 0x3C00, 0x3C00, 0x3C00, HALF_ONE };
    try testing.expectEqualSlices(u16, &expected_r, r_raw.levels[0].?);
}

test "decodeHdr2D decodes 32-bit float formats to RGBA16F" {
    const allocator = testing.allocator;

    // RGBA32_SFLOAT (109): 1x1, 4 floats = 16 bytes
    var rgba_payload: [16]u8 = undefined;
    const floats = [_]f32{ 1.0, 2.0, -0.5, 0.25 };
    for (floats, 0..) |fl, i| {
        std.mem.writeInt(u32, rgba_payload[i * 4 ..][0..4], @bitCast(fl), .little);
    }
    const rgba_file = try (TestKtx2{
        .vk_format = 109,
        .type_size = 4,
        .width = 1,
        .height = 1,
        .level_payloads = &.{&rgba_payload},
    }).build(allocator);
    defer allocator.free(rgba_file);

    var raw = try decodeHdr2D(allocator, rgba_file);
    defer raw.deinit(allocator);
    const expected = [_]u16{
        floatToHalfBits(1.0),
        floatToHalfBits(2.0),
        floatToHalfBits(-0.5),
        floatToHalfBits(0.25),
    };
    try testing.expectEqualSlices(u16, &expected, raw.levels[0].?);

    // RGB32_SFLOAT (106): 1x1, 3 floats = 12 bytes -> A=HALF_ONE
    var rgb_payload: [12]u8 = undefined;
    const rgb_floats = [_]f32{ 4.0, 5.0, 6.0 };
    for (rgb_floats, 0..) |fl, i| {
        std.mem.writeInt(u32, rgb_payload[i * 4 ..][0..4], @bitCast(fl), .little);
    }
    const rgb_file = try (TestKtx2{
        .vk_format = 106,
        .type_size = 4,
        .width = 1,
        .height = 1,
        .level_payloads = &.{&rgb_payload},
    }).build(allocator);
    defer allocator.free(rgb_file);

    var rgb_raw = try decodeHdr2D(allocator, rgb_file);
    defer rgb_raw.deinit(allocator);
    const expected_rgb = [_]u16{
        floatToHalfBits(4.0),
        floatToHalfBits(5.0),
        floatToHalfBits(6.0),
        HALF_ONE,
    };
    try testing.expectEqualSlices(u16, &expected_rgb, rgb_raw.levels[0].?);
}

test "decodeHdr2D decodes authored multi-level mip chain" {
    const allocator = testing.allocator;
    // Level 0: 2x2 RGBA16F = 32 bytes
    var lvl0: [32]u8 = undefined;
    for (0..16) |i| {
        std.mem.writeInt(u16, lvl0[i * 2 ..][0..2], @intCast(0x1000 + i), .little);
    }
    // Level 1: 1x1 RGBA16F = 8 bytes
    var lvl1: [8]u8 = undefined;
    for (0..4) |i| {
        std.mem.writeInt(u16, lvl1[i * 2 ..][0..2], @intCast(0x2000 + i), .little);
    }

    const file = try (TestKtx2{
        .vk_format = 97,
        .type_size = 2,
        .width = 2,
        .height = 2,
        .level_count = 2,
        .level_payloads = &.{ &lvl0, &lvl1 },
    }).build(allocator);
    defer allocator.free(file);

    var raw = try decodeHdr2D(allocator, file);
    defer raw.deinit(allocator);

    try testing.expectEqual(@as(u32, 2), raw.width);
    try testing.expectEqual(@as(u32, 2), raw.height);
    try testing.expectEqual(@as(u32, 2), raw.num_levels);
    try testing.expectEqual(@as(usize, 16), raw.levels[0].?.len);
    try testing.expectEqual(@as(usize, 4), raw.levels[1].?.len);
    try testing.expectEqual(@as(u16, 0x1000), raw.levels[0].?[0]);
    try testing.expectEqual(@as(u16, 0x2000), raw.levels[1].?[0]);
}

test "decodeHdrCube decodes 6 faces in engine order" {
    const allocator = testing.allocator;
    // 6 faces of 1x1 RGBA16F = 6 * 8 bytes = 48 bytes
    var payload: [48]u8 = undefined;
    for (0..6) |face| {
        const base = face * 8;
        const val: u16 = @intCast(0x3000 + face * 0x100);
        std.mem.writeInt(u16, payload[base + 0 ..][0..2], val, .little);
        std.mem.writeInt(u16, payload[base + 2 ..][0..2], val + 1, .little);
        std.mem.writeInt(u16, payload[base + 4 ..][0..2], val + 2, .little);
        std.mem.writeInt(u16, payload[base + 6 ..][0..2], val + 3, .little);
    }

    const file = try (TestKtx2{
        .vk_format = 97,
        .type_size = 2,
        .width = 1,
        .height = 1,
        .face_count = 6,
        .level_payloads = &.{&payload},
    }).build(allocator);
    defer allocator.free(file);

    var cube = try decodeHdrCube(allocator, file);
    defer cube.deinit(allocator);

    try testing.expectEqual(@as(u32, 1), cube.size);
    try testing.expectEqual(@as(u32, 1), cube.num_levels);
    try testing.expect(cube.levels[0] != null);
    try testing.expectEqual(@as(usize, 24), cube.levels[0].?.len);

    for (0..6) |face| {
        const val: u16 = @intCast(0x3000 + face * 0x100);
        const face_data = cube.levels[0].?[face * 4 .. (face + 1) * 4];
        try testing.expectEqual(val + 0, face_data[0]);
        try testing.expectEqual(val + 1, face_data[1]);
        try testing.expectEqual(val + 2, face_data[2]);
        try testing.expectEqual(val + 3, face_data[3]);
    }
}

test "decodeHdr2D and decodeHdrCube reject malformed headers and dimensions" {
    const allocator = testing.allocator;
    const dummy = @as([64]u8, @splat(0));

    // 2D rejects face_count != 1
    const file_face6 = try (TestKtx2{
        .vk_format = 97,
        .face_count = 6,
        .width = 2,
        .height = 2,
        .level_payloads = &.{&dummy},
    }).build(allocator);
    defer allocator.free(file_face6);
    try testing.expectError(error.UnsupportedFaceCount, decodeHdr2D(allocator, file_face6));

    // Cube rejects face_count != 6
    const file_face1 = try (TestKtx2{
        .vk_format = 97,
        .face_count = 1,
        .width = 2,
        .height = 2,
        .level_payloads = &.{&dummy},
    }).build(allocator);
    defer allocator.free(file_face1);
    try testing.expectError(error.UnsupportedFaceCount, decodeHdrCube(allocator, file_face1));

    // Cube rejects non-square
    const file_nonsquare = try (TestKtx2{
        .vk_format = 97,
        .face_count = 6,
        .width = 4,
        .height = 2,
        .level_payloads = &.{&dummy},
    }).build(allocator);
    defer allocator.free(file_nonsquare);
    try testing.expectError(error.UnsupportedDimensions, decodeHdrCube(allocator, file_nonsquare));

    // Non-HDR format rejected
    const file_ldr = try (TestKtx2{
        .vk_format = 37,
        .width = 2,
        .height = 2,
        .level_payloads = &.{&dummy},
    }).build(allocator);
    defer allocator.free(file_ldr);
    try testing.expectError(error.UnsupportedVkFormat, decodeHdr2D(allocator, file_ldr));

    // Invalid type_size rejected (e.g. 8)
    const file_bad_typesize = try (TestKtx2{
        .vk_format = 97,
        .type_size = 8,
        .width = 2,
        .height = 2,
        .level_payloads = &.{&dummy},
    }).build(allocator);
    defer allocator.free(file_bad_typesize);
    try testing.expectError(error.UnsupportedTypeSize, decodeHdr2D(allocator, file_bad_typesize));
}

test "Texture.decodeImageMemory routes HDR KTX2 to .hdr" {
    const allocator = testing.allocator;
    var payload: [8]u8 = undefined;
    for (0..4) |i| std.mem.writeInt(u16, payload[i * 2 ..][0..2], @intCast(0x3C00 + i), .little);

    const file = try (TestKtx2{
        .vk_format = 97,
        .type_size = 2,
        .width = 1,
        .height = 1,
        .level_payloads = &.{&payload},
    }).build(allocator);
    defer allocator.free(file);

    var dec = try Texture.decodeImageMemory(allocator, file, .{});
    defer dec.deinit(allocator);

    switch (dec) {
        .hdr => |*h| {
            try testing.expectEqual(@as(u32, 1), h.width);
            try testing.expectEqual(@as(u32, 1), h.height);
            try testing.expectEqual(@as(u32, 1), h.num_levels);
            try testing.expectEqual(HdrFormat.rgba16_sfloat, h.format);
            try testing.expectEqual(@as(usize, 8), dec.totalBytes());
        },
        else => return error.TestUnexpectedResult,
    }
}
