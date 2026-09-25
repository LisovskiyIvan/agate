const std = @import("std");
const testing = std.testing;

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

    fn texelBlockSize(self: TestKtx2) usize {
        if (formatFromVk(self.vk_format)) |f| return f.texelBlockSize();
        // Block payloads align to the block size rounded up to 4
        // (lcm(block, 4) per the KTX2 mipPadding rule: 8 for BC1, 16 for
        // the other families), matching real encoders (toktx/ktx).
        if (blockFormatFromVk(self.vk_format)) |b| return b.blockByteSize();
        return 4;
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
    try testing.expect(blockFormatFromVk(139) == null); // BC4: single-channel, out of scope
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

    const rgba_payload = [_]u8{0} ** 4;
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
    const level1 = [_]u8{0xAA} ** 16;
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
        .rgba => return error.TestUnexpectedResult,
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
        .block => return error.TestUnexpectedResult,
    }
}

// ---------------------------------------------------------------------------
// Basis Universal tests — REAL toktx fixtures (src/agate/ktx2_fixtures/,
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
    const rgba_level = [_]u8{0} ** 4;
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
        .rgba => return error.TestUnexpectedResult,
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
        .rgba => return error.TestUnexpectedResult,
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
        .rgba => return error.TestUnexpectedResult,
    }
}

test "decodeBasis2D transcodes a real UASTC image to ETC2 RGBA8" {
    const allocator = testing.allocator;
    var img = try decodeBasis2D(allocator, fx_uastc_rgba_mip, .etc2_rgba, .{});
    defer img.deinit(allocator);
    switch (img) {
        .block => |b| {
            try testing.expectEqual(BlockFormat.etc2_rgba8_srgb, b.format);
            try testing.expectEqual(@as(usize, 256), b.levels[0].?.len);
        },
        .rgba => return error.TestUnexpectedResult,
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
        .rgba => return error.TestUnexpectedResult,
    }
}

test "decodeBasis2D selects UNORM variants for linear-transfer files" {
    const allocator = testing.allocator;
    var bc7 = try decodeBasis2D(allocator, fx_etc1s_rgb_flat_linear, .bc7, .{});
    defer bc7.deinit(allocator);
    switch (bc7) {
        .block => |b| try testing.expectEqual(BlockFormat.bc7_unorm, b.format),
        .rgba => return error.TestUnexpectedResult,
    }
    var astc = try decodeBasis2D(allocator, fx_uastc_rgb_flat_linear, .astc, .{});
    defer astc.deinit(allocator);
    switch (astc) {
        .block => |b| {
            try testing.expectEqual(BlockFormat.astc_4x4_unorm, b.format);
            try testing.expectEqual(@as(usize, 256), b.levels[0].?.len);
        },
        .rgba => return error.TestUnexpectedResult,
    }
    // sRGB flat picks the sRGB ASTC variant with identical block math.
    var astc_srgb = try decodeBasis2D(allocator, fx_uastc_rgb_flat, .astc, .{});
    defer astc_srgb.deinit(allocator);
    switch (astc_srgb) {
        .block => |b| try testing.expectEqual(BlockFormat.astc_4x4_srgb, b.format),
        .rgba => return error.TestUnexpectedResult,
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
        .block => return error.TestUnexpectedResult,
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
        .block => return error.TestUnexpectedResult,
    }

    // Single-level RGBA32 honors gen_mipmaps like every other RGBA8 path.
    var chained = try decodeBasis2D(allocator, fx_uastc_rgb_flat, .rgba32, .{});
    defer chained.deinit(allocator);
    switch (chained) {
        .rgba => |r| try testing.expectEqual(Texture.mipLevelCount(16, 16), r.num_levels),
        .block => return error.TestUnexpectedResult,
    }
    var flat = try decodeBasis2D(allocator, fx_uastc_rgb_flat, .rgba32, .{ .gen_mipmaps = false });
    defer flat.deinit(allocator);
    switch (flat) {
        .rgba => |r| try testing.expectEqual(@as(u32, 1), r.num_levels),
        .block => return error.TestUnexpectedResult,
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
    const rgba_level = [_]u8{0} ** 4;
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
        .rgba => return error.TestUnexpectedResult,
    }

    var rgba_img = try Texture.decodeImageMemory(allocator, fx_uastc_rgba_mip, .{ .basis_target = .rgba32 });
    defer rgba_img.deinit(allocator);
    switch (rgba_img) {
        .rgba => |r| try testing.expectEqual(@as(u32, 5), r.num_levels),
        .block => return error.TestUnexpectedResult,
    }
}

test "Texture.decodeMemory rejects Basis with its own reason" {
    const allocator = testing.allocator;
    try testing.expectError(error.BasisRequiresBlockDecode, Texture.decodeMemory(allocator, fx_uastc_rgb_flat, .{}));
    try testing.expectError(error.Truncated, Texture.decodeMemory(allocator, fx_uastc_rgb_flat[0..40], .{}));
}
