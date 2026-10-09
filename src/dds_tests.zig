const std = @import("std");
const testing = std.testing;
const dds = @import("dds.zig");
const ktx2 = @import("ktx2.zig");
const Texture = @import("texture.zig").Texture;

/// Minimal DDS builder for tests: 4-byte magic, 124-byte header, optional
/// 20-byte DX10 extension, then the level payloads back to back (largest
/// first). Payloads are passed per level; sizes are NOT validated here so
/// malformed fixtures (short/trailing data) can be built on purpose.
const TestDds = struct {
    fourcc: u32 = dds.fourcc_dx10,
    /// DX10 extension: dxgiFormat code.
    dxgi: ?u32 = dds.dxgi_bc7_unorm,
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
            try w.writeAll(&dds.magic);
        }

        var flags: u32 = dds.required_flags;
        if (self.set_mip_flag or self.mip_count > 1) flags |= dds.flag_mipmapcount;
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
        try w.writeInt(u32, dds.pf_fourcc, .little);
        try w.writeInt(u32, self.fourcc, .little);
        try w.writeInt(u32, 0, .little); // dwRGBBitCount
        for (0..4) |_| try w.writeInt(u32, 0, .little); // bit masks
        try w.writeInt(u32, dds.caps_texture, .little); // dwCaps
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
    try testing.expect(dds.sniff(&dds.magic));
    try testing.expect(!dds.sniff("not a dds file!!"));
    try testing.expect(!dds.sniff("DDS")); // truncated magic
    try testing.expect(!dds.sniff(&.{}));
    // KTX2 files must not route here (magics are disjoint).
    try testing.expect(!dds.sniff(&[12]u8{ 0xAB, 'K', 'T', 'X', ' ', '2', '0', 0xBB, 0x0D, 0x0A, 0x1A, 0x0A }));
}

test "decodeBlock2D rejects legacy FourCC headers (DXT1, DXT2, DXT3, DXT4, DXT5)" {
    const allocator = testing.allocator;
    var bc1_payload: [8]u8 = undefined;
    @memset(&bc1_payload, 0x11);
    var bc3_payload: [16]u8 = undefined;
    @memset(&bc3_payload, 0x22);

    const legacy_cases = [_]struct { name: []const u8, fourcc: u32, payload: []const u8 }{
        .{ .name = "DXT1", .fourcc = dds.fourcc_dxt1, .payload = &bc1_payload },
        .{ .name = "DXT2", .fourcc = dds.fourcc_dxt2, .payload = &bc3_payload },
        .{ .name = "DXT3", .fourcc = dds.fourcc_dxt3, .payload = &bc3_payload },
        .{ .name = "DXT4", .fourcc = dds.fourcc_dxt4, .payload = &bc3_payload },
        .{ .name = "DXT5", .fourcc = dds.fourcc_dxt5, .payload = &bc3_payload },
    };
    for (legacy_cases) |c| {
        const file = try (TestDds{
            .fourcc = c.fourcc,
            .dxgi = null,
            .level_payloads = &.{c.payload},
        }).build(allocator);
        defer allocator.free(file);
        try testing.expectError(error.UnsupportedDdsFormat, dds.decodeBlock2D(allocator, file, .{}));
    }
}

test "decodeBlock2D reads a DX10 BC1 4x4 file exactly as authored" {
    const allocator = testing.allocator;
    var payload: [8]u8 = undefined; // 4x4 BC1 -> one 8-byte block
    for (&payload, 0..) |*b, i| b.* = @intCast(10 + i);
    const file = try (TestDds{
        .fourcc = dds.fourcc_dx10,
        .dxgi = dds.dxgi_bc1_unorm,
        .width = 4,
        .height = 4,
        .mip_count = 1,
        .level_payloads = &.{&payload},
    }).build(allocator);
    defer allocator.free(file);

    var tex = try dds.decodeBlock2D(allocator, file, .{});
    defer tex.deinit(allocator);

    try testing.expectEqual(ktx2.BlockFormat.bc1_unorm, tex.format);
    try testing.expectEqual(@as(u32, 4), tex.width);
    try testing.expectEqual(@as(u32, 4), tex.height);
    try testing.expectEqual(@as(u32, 1), tex.num_levels);
    try testing.expectEqualSlices(u8, &payload, tex.levels[0].?);
    try testing.expectEqual(@as(?[]const u8, null), tex.levels[1]);
}

test "decodeBlock2D reads a DX10 BC3 8x8 two-level file exactly as authored" {
    const allocator = testing.allocator;
    // 8x8 -> 2x2 blocks of 16 bytes = 64 bytes.
    var m0: [64]u8 = undefined;
    @memset(&m0, 0x33);
    // 4x4 -> 1x1 block of 16 bytes = 16 bytes.
    var m1: [16]u8 = undefined;
    @memset(&m1, 0x44);

    const file = try (TestDds{
        .fourcc = dds.fourcc_dx10,
        .dxgi = dds.dxgi_bc3_unorm,
        .width = 8,
        .height = 8,
        .mip_count = 2,
        .level_payloads = &.{ &m0, &m1 },
    }).build(allocator);
    defer allocator.free(file);

    var tex = try dds.decodeBlock2D(allocator, file, .{});
    defer tex.deinit(allocator);

    try testing.expectEqual(ktx2.BlockFormat.bc3_unorm, tex.format);
    try testing.expectEqual(@as(u32, 8), tex.width);
    try testing.expectEqual(@as(u32, 8), tex.height);
    try testing.expectEqual(@as(u32, 2), tex.num_levels);
    try testing.expectEqualSlices(u8, &m0, tex.levels[0].?);
    try testing.expectEqualSlices(u8, &m1, tex.levels[1].?);
    try testing.expectEqual(@as(?[]const u8, null), tex.levels[2]);
}

test "decodeBlock2D reads a DX10 BC7 8x8 two-level file exactly as authored" {
    const allocator = testing.allocator;
    var m0: [64]u8 = undefined;
    @memset(&m0, 0x77);
    var m1: [16]u8 = undefined;
    @memset(&m1, 0x88);

    const file = try (TestDds{
        .fourcc = dds.fourcc_dx10,
        .dxgi = dds.dxgi_bc7_unorm,
        .width = 8,
        .height = 8,
        .mip_count = 2,
        .level_payloads = &.{ &m0, &m1 },
    }).build(allocator);
    defer allocator.free(file);

    var tex = try dds.decodeBlock2D(allocator, file, .{});
    defer tex.deinit(allocator);

    try testing.expectEqual(ktx2.BlockFormat.bc7_unorm, tex.format);
    try testing.expectEqual(@as(u32, 8), tex.width);
    try testing.expectEqual(@as(u32, 8), tex.height);
    try testing.expectEqual(@as(u32, 2), tex.num_levels);
    try testing.expectEqualSlices(u8, &m0, tex.levels[0].?);
    try testing.expectEqualSlices(u8, &m1, tex.levels[1].?);
}

test "format mapping covers DX10 DXGI codes" {
    const allocator = testing.allocator;
    var bc1_payload: [8]u8 = undefined;
    @memset(&bc1_payload, 0);
    var bc3_payload: [16]u8 = undefined;
    @memset(&bc3_payload, 0);
    var bc7_payload: [16]u8 = undefined;
    @memset(&bc7_payload, 0);

    const cases = [_]struct {
        name: []const u8,
        spec: TestDds,
        expected: ktx2.BlockFormat,
    }{
        .{ .name = "DX10 BC1", .spec = .{ .fourcc = dds.fourcc_dx10, .dxgi = dds.dxgi_bc1_unorm, .level_payloads = &.{&bc1_payload} }, .expected = .bc1_unorm },
        .{ .name = "DX10 BC1_SRGB collapses to UNORM", .spec = .{ .fourcc = dds.fourcc_dx10, .dxgi = dds.dxgi_bc1_unorm_srgb, .level_payloads = &.{&bc1_payload} }, .expected = .bc1_unorm },
        .{ .name = "DX10 BC2", .spec = .{ .fourcc = dds.fourcc_dx10, .dxgi = dds.dxgi_bc2_unorm, .level_payloads = &.{&bc3_payload} }, .expected = .bc2_unorm },
        .{ .name = "DX10 BC3", .spec = .{ .fourcc = dds.fourcc_dx10, .dxgi = dds.dxgi_bc3_unorm, .level_payloads = &.{&bc3_payload} }, .expected = .bc3_unorm },
        .{ .name = "DX10 BC3_SRGB", .spec = .{ .fourcc = dds.fourcc_dx10, .dxgi = dds.dxgi_bc3_unorm_srgb, .level_payloads = &.{&bc3_payload} }, .expected = .bc3_srgb },
        .{ .name = "DX10 BC7", .spec = .{ .fourcc = dds.fourcc_dx10, .dxgi = dds.dxgi_bc7_unorm, .level_payloads = &.{&bc7_payload} }, .expected = .bc7_unorm },
        .{ .name = "DX10 BC7_SRGB", .spec = .{ .fourcc = dds.fourcc_dx10, .dxgi = dds.dxgi_bc7_unorm_srgb, .level_payloads = &.{&bc7_payload} }, .expected = .bc7_srgb },
    };
    for (cases) |c| {
        const file = try c.spec.build(allocator);
        defer allocator.free(file);
        var tex = try dds.decodeBlock2D(allocator, file, .{});
        defer tex.deinit(allocator);
        try testing.expectEqual(c.expected, tex.format);
    }
}

test "decodeBlock2D rejects bad magic, sizes and malformed headers" {
    const allocator = testing.allocator;
    var payload: [16]u8 = undefined;
    @memset(&payload, 0);

    const bad_cases = [_]struct { name: []const u8, spec: TestDds, expected: dds.DecodeError }{
        .{ .name = "dwSize", .spec = .{ .header_size = 100, .level_payloads = &.{&payload} }, .expected = error.InvalidMipData },
        .{ .name = "pixel format size", .spec = .{ .pf_size = 16, .level_payloads = &.{&payload} }, .expected = error.InvalidMipData },
        .{ .name = "missing required flags", .spec = .{ .clear_flags = dds.flag_height, .level_payloads = &.{&payload} }, .expected = error.InvalidMipData },
        .{ .name = "zero width", .spec = .{ .width = 0, .level_payloads = &.{&payload} }, .expected = error.UnsupportedDimensions },
        .{ .name = "huge height", .spec = .{ .height = 32768, .level_payloads = &.{&payload} }, .expected = error.UnsupportedDimensions },
        .{ .name = "depth set", .spec = .{ .depth = 4, .level_payloads = &.{&payload} }, .expected = error.Unsupported3D },
        .{ .name = "cubemap caps2", .spec = .{ .caps2 = 0x200 | 0x400, .level_payloads = &.{&payload} }, .expected = error.UnsupportedCube },
        .{ .name = "volume caps2", .spec = .{ .caps2 = dds.caps2_volume, .level_payloads = &.{&payload} }, .expected = error.Unsupported3D },
        .{ .name = "17 levels", .spec = .{ .width = 256, .height = 256, .mip_count = 17, .level_payloads = &.{&payload} }, .expected = error.TooManyLevels },
        // 4x4 has a 3-level halving chain; claiming 4 levels is corrupt.
        .{ .name = "overlong chain", .spec = .{ .mip_count = 4, .level_payloads = &.{&payload} }, .expected = error.InvalidMipData },
        // DX10 without its extension bytes.
        .{ .name = "DX10 truncated extension", .spec = .{ .fourcc = dds.fourcc_dx10, .dxgi = null, .level_payloads = &.{&payload} }, .expected = error.Truncated },
        .{ .name = "DX10 cube flag", .spec = .{ .fourcc = dds.fourcc_dx10, .dxgi = dds.dxgi_bc7_unorm, .misc_flag = dds.dx10_misc_texturecube, .level_payloads = &.{&payload} }, .expected = error.UnsupportedCube },
        .{ .name = "DX10 array", .spec = .{ .fourcc = dds.fourcc_dx10, .dxgi = dds.dxgi_bc7_unorm, .array_size = 6, .level_payloads = &.{&payload} }, .expected = error.UnsupportedArraySize },
        .{ .name = "DX10 1D dimension", .spec = .{ .fourcc = dds.fourcc_dx10, .dxgi = dds.dxgi_bc7_unorm, .dimension = 2, .level_payloads = &.{&payload} }, .expected = error.UnsupportedDdsFormat },
    };
    for (bad_cases) |case| {
        const file = try case.spec.build(allocator);
        defer allocator.free(file);
        try testing.expectError(case.expected, dds.decodeBlock2D(allocator, file, .{}));
    }
}

test "decodeBlock2D rejects unsupported pixel formats with a clear error" {
    const allocator = testing.allocator;
    var payload: [16]u8 = undefined;
    @memset(&payload, 0);

    const bad_formats = [_]struct {
        name: []const u8,
        spec: TestDds,
    }{
        .{ .name = "DXT2", .spec = .{ .fourcc = dds.fourcc_dxt2, .level_payloads = &.{&payload} } },
        .{ .name = "DXT4", .spec = .{ .fourcc = dds.fourcc_dxt4, .level_payloads = &.{&payload} } },
        .{ .name = "DXGI R8G8B8A8_UNORM", .spec = .{ .fourcc = dds.fourcc_dx10, .dxgi = 28, .level_payloads = &.{&payload} } },
        .{ .name = "DXGI R16G16B16A16_FLOAT", .spec = .{ .fourcc = dds.fourcc_dx10, .dxgi = 10, .level_payloads = &.{&payload} } },
        .{ .name = "DXGI R9G9B9E5_SHAREDEXP", .spec = .{ .fourcc = dds.fourcc_dx10, .dxgi = 67, .level_payloads = &.{&payload} } },
    };
    for (bad_formats) |c| {
        const file = try c.spec.build(allocator);
        defer allocator.free(file);
        try testing.expectError(error.UnsupportedDdsFormat, dds.decodeBlock2D(allocator, file, .{}));
    }
}

test "decodeBlock2D validates non-multiple dimensions and partial chains" {
    const allocator = testing.allocator;
    var level0: [32]u8 = undefined;
    for (&level0, 0..) |*b, i| b.* = @intCast(100 + i);
    const level1 = [_]u8{0xAA} ** 16;
    const file = try (TestDds{
        .fourcc = dds.fourcc_dx10,
        .dxgi = dds.dxgi_bc3_unorm,
        .width = 5,
        .height = 3,
        .mip_count = 2,
        .level_payloads = &.{ &level0, &level1 },
    }).build(allocator);
    defer allocator.free(file);

    var raw = try dds.decodeBlock2D(allocator, file, .{});
    defer raw.deinit(allocator);
    try testing.expectEqual(ktx2.BlockFormat.bc3_unorm, raw.format);
    try testing.expectEqual(@as(u32, 2), raw.num_levels);
    try testing.expectEqualSlices(u8, &level0, raw.levels[0].?);
    try testing.expectEqualSlices(u8, &level1, raw.levels[1].?);
    try testing.expectEqual(@as(usize, 32 + 16), raw.totalBytes());

    var single: [32]u8 = undefined;
    @memset(&single, 0x77);
    const partial = try (TestDds{
        .fourcc = dds.fourcc_dx10,
        .dxgi = dds.dxgi_bc3_unorm,
        .width = 5,
        .height = 3,
        .level_payloads = &.{&single},
    }).build(allocator);
    defer allocator.free(partial);
    var raw_single = try dds.decodeBlock2D(allocator, partial, .{});
    defer raw_single.deinit(allocator);
    try testing.expectEqual(@as(u32, 1), raw_single.num_levels);
}

test "Texture.decodeImageMemory routes DDS payloads to the block path" {
    const allocator = testing.allocator;

    var payload: [16]u8 = undefined;
    for (&payload, 0..) |*b, i| b.* = @intCast(i);
    const file = try (TestDds{
        .fourcc = dds.fourcc_dx10,
        .dxgi = dds.dxgi_bc3_unorm,
        .level_payloads = &.{&payload},
    }).build(allocator);
    defer allocator.free(file);

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

    const srgb_file = try (TestDds{
        .fourcc = dds.fourcc_dx10,
        .dxgi = dds.dxgi_bc3_unorm_srgb,
        .level_payloads = &.{&payload},
    }).build(allocator);
    defer allocator.free(srgb_file);

    var srgb_dec = try Texture.decodeImageMemory(allocator, srgb_file, .{});
    defer srgb_dec.deinit(allocator);
    switch (srgb_dec) {
        .block => |b| try testing.expectEqual(ktx2.BlockFormat.bc3_srgb, b.format),
        .rgba => return error.TestUnexpectedResult,
    }

    try testing.expectError(error.DdsRequiresBlockDecode, Texture.decodeMemory(allocator, file, .{}));
    try testing.expectError(error.ImageDecodeFailed, Texture.decodeMemory(allocator, "png data pretending", .{}));

    const short = try (TestDds{
        .fourcc = dds.fourcc_dx10,
        .dxgi = dds.dxgi_bc3_unorm,
        .level_payloads = &.{&payload},
        .trailing_garbage = true,
    }).build(allocator);
    defer allocator.free(short);
    try testing.expectError(error.InvalidMipData, Texture.decodeMemory(allocator, short, .{}));
}

test "decodeBlock2D reads DX10 BC4/BC5/BC6H files exactly as authored" {
    const allocator = testing.allocator;

    // BC4: 4x4 -> one 8-byte block (single-channel R).
    var bc4: [8]u8 = undefined;
    for (&bc4, 0..) |*b, i| b.* = @intCast(0x40 + i);
    const f4 = try (TestDds{
        .dxgi = dds.dxgi_bc4_unorm,
        .level_payloads = &.{&bc4},
    }).build(allocator);
    defer allocator.free(f4);
    var t4 = try dds.decodeBlock2D(allocator, f4, .{});
    defer t4.deinit(allocator);
    try testing.expectEqual(ktx2.BlockFormat.bc4_unorm, t4.format);
    try testing.expectEqualSlices(u8, &bc4, t4.levels[0].?);

    // BC5 SNORM: 8x8 -> 2x2 blocks of 16 bytes = 64 bytes (two-channel RG).
    var bc5: [64]u8 = undefined;
    @memset(&bc5, 0x5A);
    const f5 = try (TestDds{
        .dxgi = dds.dxgi_bc5_snorm,
        .width = 8,
        .height = 8,
        .level_payloads = &.{&bc5},
    }).build(allocator);
    defer allocator.free(f5);
    var t5 = try dds.decodeBlock2D(allocator, f5, .{});
    defer t5.deinit(allocator);
    try testing.expectEqual(ktx2.BlockFormat.bc5_snorm, t5.format);
    try testing.expectEqualSlices(u8, &bc5, t5.levels[0].?);

    // BC6H SF16: 4x4 -> one 16-byte block (HDR RGB half float).
    var bc6: [16]u8 = undefined;
    @memset(&bc6, 0x6B);
    const f6 = try (TestDds{
        .dxgi = dds.dxgi_bc6h_sf16,
        .level_payloads = &.{&bc6},
    }).build(allocator);
    defer allocator.free(f6);
    var t6 = try dds.decodeBlock2D(allocator, f6, .{});
    defer t6.deinit(allocator);
    try testing.expectEqual(ktx2.BlockFormat.bc6h_sf16, t6.format);
    try testing.expectEqualSlices(u8, &bc6, t6.levels[0].?);

    // Overlong payload (BC4 level must be exactly 8 bytes for 4x4):
    // trailing bytes are garbage, not truncation.
    var bad: [16]u8 = undefined;
    @memset(&bad, 1);
    const fbad = try (TestDds{
        .dxgi = dds.dxgi_bc4_unorm,
        .level_payloads = &.{&bad},
    }).build(allocator);
    defer allocator.free(fbad);
    try testing.expectError(error.InvalidMipData, dds.decodeBlock2D(allocator, fbad, .{}));
}
