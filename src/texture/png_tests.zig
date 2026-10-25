const std = @import("std");
const png = @import("png.zig");
const Texture = @import("core_texture.zig").Texture;

test "png: encode 2x2 RGBA8 and verify PNG header and chunks" {
    const allocator = std.testing.allocator;
    const pixels = [_]u8{
        255, 0,   0,   255, // Red
        0,   255, 0,   255, // Green
        0,   0,   255, 255, // Blue
        255, 255, 255, 128, // Translucent white
    };

    const encoded = try png.encodeRgbaPng(allocator, 2, 2, &pixels);
    defer allocator.free(encoded);

    // PNG signature: 0x89 0x50 0x4E 0x47 0x0D 0x0A 0x1A 0x0A
    try std.testing.expect(encoded.len > 8);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' }, encoded[0..8]);

    // Decode with Texture.decodeMemory to verify round-trip accuracy
    var raw = try Texture.decodeMemory(allocator, encoded, .{ .gen_mipmaps = false, .srgb_to_linear = false });
    defer raw.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 2), raw.width);
    try std.testing.expectEqual(@as(u32, 2), raw.height);
    try std.testing.expectEqual(@as(usize, 16), raw.levels[0].?.len);
    try std.testing.expectEqualSlices(u8, &pixels, raw.levels[0].?);
}

test "png: dimension and buffer validation fail-closed" {
    const allocator = std.testing.allocator;
    const dummy = [_]u8{ 0, 0, 0, 0 };

    try std.testing.expectError(error.InvalidDimensions, png.encodeRgbaPng(allocator, 0, 10, &dummy));
    try std.testing.expectError(error.InvalidDimensions, png.encodeRgbaPng(allocator, 10, 0, &dummy));
    try std.testing.expectError(error.InvalidBufferSize, png.encodeRgbaPng(allocator, 2, 2, &dummy)); // expects 16 bytes, got 4
}

test "png: multi-block zlib (> 65535 bytes) encodes and decodes cleanly" {
    const allocator = std.testing.allocator;
    // 200x100 = 20,000 pixels = 80,000 bytes (spans at least two 65535-byte deflate blocks)
    const w: u32 = 200;
    const h: u32 = 100;
    const size = w * h * 4;
    const pixels = try allocator.alloc(u8, size);
    defer allocator.free(pixels);

    for (pixels, 0..) |*b, i| {
        b.* = @truncate(i * 17 + 5);
    }

    const encoded = try png.encodeRgbaPng(allocator, w, h, pixels);
    defer allocator.free(encoded);

    var raw = try Texture.decodeMemory(allocator, encoded, .{ .gen_mipmaps = false, .srgb_to_linear = false });
    defer raw.deinit(allocator);

    try std.testing.expectEqual(w, raw.width);
    try std.testing.expectEqual(h, raw.height);
    try std.testing.expectEqualSlices(u8, pixels, raw.levels[0].?);
}

test "png: write to file and verify readback" {
    const allocator = std.testing.allocator;
    const pixels = [_]u8{
        100, 150, 200, 255,
        50,  75,  100, 255,
    };
    const path = "zig-out/test_snapshot.png";
    const io = std.testing.io;
    std.Io.Dir.cwd().createDirPath(io, "zig-out") catch {};

    try png.writeRgbaPngFile(allocator, path, 2, 1, &pixels);
    defer std.Io.Dir.cwd().deleteFile(io, path) catch {};

    const read_bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(read_bytes);

    var raw = try Texture.decodeMemory(allocator, read_bytes, .{ .gen_mipmaps = false, .srgb_to_linear = false });
    defer raw.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 2), raw.width);
    try std.testing.expectEqual(@as(u32, 1), raw.height);
    try std.testing.expectEqualSlices(u8, &pixels, raw.levels[0].?);
}
