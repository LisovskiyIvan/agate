//! Minimal PNG fixture synthesis for texture tests. Split out of `core.zig` (facade).
//!
//! `TestPng.build` wraps caller-provided scanlines in a PNG container with a
//! stored-deflate zlib stream, so stb decodes it like any real PNG. Used by
//! the `core`/`mip`/`cube` test suites via the `core.zig` facade re-export.
//! Leaf: `std` only.

const std = @import("std");

// ---------------------------------------------------------------------------
// Minimal PNG fixture synthesis (no external dependencies): PNG container +
// zlib stream with STORED deflate blocks + Adler-32. stb_image decodes these
// like any real PNG.
// ---------------------------------------------------------------------------
pub const TestPng = struct {
    fn crc32(tag: *const [4]u8, data: []const u8) u32 {
        var h = std.hash.Crc32.init();
        h.update(tag);
        h.update(data);
        return h.final();
    }

    fn appendChunk(out: *std.Io.Writer, tag: *const [4]u8, data: []const u8) !void {
        var len_buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &len_buf, @intCast(data.len), .big);
        try out.writeAll(&len_buf);
        try out.writeAll(tag);
        try out.writeAll(data);
        var crc_buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &crc_buf, crc32(tag, data), .big);
        try out.writeAll(&crc_buf);
    }

    /// Wraps `raw` in a zlib stream that only uses stored (uncompressed)
    /// deflate blocks: header 0x78 0x01, blocks of up to 65535 bytes
    /// (BFINAL flag, LEN, NLEN), Adler-32 checksum.
    fn zlibStore(writer: *std.Io.Writer, raw: []const u8) !void {
        try writer.writeAll(&.{ 0x78, 0x01 });
        var rest = raw;
        while (true) {
            const final = rest.len <= 65535;
            const chunk_len: u16 = @intCast(@min(rest.len, 65535));
            try writer.writeByte(if (final) 0x01 else 0x00);
            var len_buf: [2]u8 = undefined;
            std.mem.writeInt(u16, &len_buf, chunk_len, .little);
            try writer.writeAll(&len_buf);
            std.mem.writeInt(u16, &len_buf, ~chunk_len, .little);
            try writer.writeAll(&len_buf);
            try writer.writeAll(rest[0..chunk_len]);
            rest = rest[chunk_len..];
            if (final) break;
        }
        var sum_buf: [4]u8 = undefined;
        std.mem.writeInt(u32, &sum_buf, std.hash.Adler32.hash(raw), .big);
        try writer.writeAll(&sum_buf);
    }

    /// Builds a minimal PNG: signature, IHDR, optional PLTE, IDAT, IEND.
    /// `scanlines` must already contain one filter byte (0 = None) per row.
    pub fn build(
        allocator: std.mem.Allocator,
        width: u32,
        height: u32,
        bit_depth: u8,
        color_type: u8,
        plte: ?[]const u8,
        scanlines: []const u8,
    ) ![]u8 {
        var out: std.Io.Writer.Allocating = .init(allocator);
        errdefer out.deinit();
        const w = &out.writer;

        try w.writeAll(&[_]u8{ 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' });

        var ihdr: [13]u8 = undefined;
        std.mem.writeInt(u32, ihdr[0..4], width, .big);
        std.mem.writeInt(u32, ihdr[4..8], height, .big);
        ihdr[8] = bit_depth;
        ihdr[9] = color_type;
        ihdr[10] = 0; // compression: deflate
        ihdr[11] = 0; // filter: adaptive filtering
        ihdr[12] = 0; // interlace: none
        try appendChunk(w, "IHDR", &ihdr);

        if (plte) |entries| try appendChunk(w, "PLTE", entries);

        var idat: std.Io.Writer.Allocating = .init(allocator);
        defer idat.deinit();
        try zlibStore(&idat.writer, scanlines);
        try appendChunk(w, "IDAT", idat.written());

        try appendChunk(w, "IEND", "");
        return out.toOwnedSlice();
    }
};
