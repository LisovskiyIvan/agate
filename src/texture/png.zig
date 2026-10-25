//! Pure Zig PNG encoding for snapshots, screenshots, and texture tests.
//! Wraps RGBA8 pixels in a standard PNG container with stored deflate blocks,
//! Adler-32 checksum, and CRC-32 chunk validation. Fully compatible with
//! stb_image, browsers, and standard image viewers.

const std = @import("std");
const core_testpng = @import("core_testpng.zig");
const TestPng = core_testpng.TestPng;

pub const PngError = error{
    InvalidDimensions,
    ImageTooLarge,
    InvalidBufferSize,
    OutOfMemory,
};

/// Encodes an uncompressed RGBA8 pixel buffer (width * height * 4 bytes)
/// into a standard PNG file byte slice.
pub fn encodeRgbaPng(allocator: std.mem.Allocator, width: u32, height: u32, rgba_pixels: []const u8) PngError![]u8 {
    if (width == 0 or height == 0) return error.InvalidDimensions;
    const row_bytes = std.math.mul(usize, width, 4) catch return error.ImageTooLarge;
    const total_bytes = std.math.mul(usize, row_bytes, height) catch return error.ImageTooLarge;
    if (rgba_pixels.len != total_bytes) return error.InvalidBufferSize;

    const scanline_size = 1 + row_bytes;
    const total_scanline_bytes = std.math.mul(usize, scanline_size, height) catch return error.ImageTooLarge;
    const scanlines = try allocator.alloc(u8, total_scanline_bytes);
    defer allocator.free(scanlines);

    for (0..height) |y| {
        const row_out = scanlines[y * scanline_size .. (y + 1) * scanline_size];
        row_out[0] = 0; // Filter type: None
        const row_in = rgba_pixels[y * row_bytes .. (y + 1) * row_bytes];
        @memcpy(row_out[1..], row_in);
    }

    return TestPng.build(allocator, width, height, 8, 6, null, scanlines) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.WriteFailed => error.OutOfMemory,
    };
}

/// Encodes and writes an RGBA8 pixel buffer directly to a PNG file on disk.
pub fn writeRgbaPngFile(allocator: std.mem.Allocator, path: []const u8, width: u32, height: u32, rgba_pixels: []const u8) !void {
    const png_data = try encodeRgbaPng(allocator, width, height, rgba_pixels);
    defer allocator.free(png_data);

    const io = std.Io.Threaded.global_single_threaded.io();
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = png_data });
}
