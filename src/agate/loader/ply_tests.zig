//! Tests for `ply.zig` (moved from `ply.zig` inline blocks).
const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const ply = @import("ply.zig");
const parse = ply.parse;

fn writeTestU8(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, v: u8) !void {
    try list.append(allocator, v);
}

fn writeTestI32(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, v: i32, endian: std.builtin.Endian) !void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(i32, &buf, v, endian);
    try list.appendSlice(allocator, &buf);
}

fn writeTestF32(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, f: f32, endian: std.builtin.Endian) !void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, @bitCast(f), endian);
    try list.appendSlice(allocator, &buf);
}

fn writeTestF64(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, f: f64, endian: std.builtin.Endian) !void {
    var buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &buf, @bitCast(f), endian);
    try list.appendSlice(allocator, &buf);
}

fn writeTestU16(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, v: u16, endian: std.builtin.Endian) !void {
    var buf: [2]u8 = undefined;
    std.mem.writeInt(u16, &buf, v, endian);
    try list.appendSlice(allocator, &buf);
}

fn writeTestI16(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, v: i16, endian: std.builtin.Endian) !void {
    var buf: [2]u8 = undefined;
    std.mem.writeInt(i16, &buf, v, endian);
    try list.appendSlice(allocator, &buf);
}

test "ply ascii quad fans into two triangles" {
    const alloc = std.testing.allocator;
    const text =
        \\ply
        \\format ascii 1.0
        \\comment single quad
        \\obj_info generated for unit test
        \\element vertex 4
        \\property float x
        \\property float y
        \\property float z
        \\property float nx
        \\property float ny
        \\property float nz
        \\property float s
        \\property float t
        \\property uchar red
        \\property uchar green
        \\property uchar blue
        \\element face 1
        \\property list uchar int vertex_indices
        \\end_header
        \\0 0 0 0 0 1 0 0 255 0 0
        \\1 0 0 0 0 1 1 0 0 255 0
        \\1 1 0 0 0 1 1 1 0 0 255
        \\0 1 0 0 0 1 0 1 255 255 255
        \\4 0 1 2 3
    ;
    var data = try parse(alloc, text);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 4), data.vertexCount());
    try std.testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 2, 0, 2, 3 }, data.indices);
    try std.testing.expect(data.has_normals);
    try std.testing.expect(data.has_uvs);
    try std.testing.expect(data.has_colors);
    try std.testing.expectApproxEqAbs(@as(f32, 1), data.normals[2], 1e-6);
    try std.testing.expectEqual(@as(f32, 1), data.uvs[2]);
    try std.testing.expectEqual(@as(f32, 0), data.uvs[3]);
    try std.testing.expectEqual(@as(f32, 1), data.uvs[4]);
    try std.testing.expectEqual(@as(f32, 1), data.uvs[5]);
    // uchar 255 -> 1.0, first vertex is pure red.
    try std.testing.expectEqual(@as(f32, 1), data.colors[0]);
    try std.testing.expectEqual(@as(f32, 0), data.colors[1]);
    try std.testing.expectEqual(@as(f32, 0), data.colors[2]);
    try std.testing.expectEqual(@as(f32, 1), data.colors[3]);
    // Last vertex is white with opaque alpha.
    try std.testing.expectEqual(@as(f32, 1), data.colors[12]);
    try std.testing.expectEqual(@as(f32, 1), data.colors[15]);
    const avg = data.averageColor();
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), avg[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), avg[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), avg[2], 1e-6);
}

test "ply binary little endian with float colors and computed normals" {
    const alloc = std.testing.allocator;
    var bytes: std.ArrayListUnmanaged(u8) = .empty;
    defer bytes.deinit(alloc);
    try bytes.appendSlice(alloc,
        \\ply
        \\format binary_little_endian 1.0
        \\element vertex 3
        \\property float x
        \\property float y
        \\property float z
        \\property float red
        \\property float green
        \\property float blue
        \\element face 1
        \\property list uchar int vertex_indices
        \\end_header
    );
    // v0 red, v1 green, v2 blue.
    for ([3][6]f32{
        .{ 0, 0, 0, 1, 0, 0 },
        .{ 1, 0, 0, 0, 1, 0 },
        .{ 0, 1, 0, 0, 0, 1 },
    }) |row| {
        for (row) |f| try writeTestF32(&bytes, alloc, f, .little);
    }
    try writeTestU8(&bytes, alloc, 3);
    try writeTestI32(&bytes, alloc, 0, .little);
    try writeTestI32(&bytes, alloc, 1, .little);
    try writeTestI32(&bytes, alloc, 2, .little);

    var data = try parse(alloc, bytes.items);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 3), data.vertexCount());
    try std.testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 2 }, data.indices);
    try std.testing.expectEqual(@as(f32, 1), data.positions[3]);
    try std.testing.expect(!data.has_normals);
    try std.testing.expect(!data.has_uvs);
    // Computed face normal (0,0,1), missing UVs default to zero.
    try std.testing.expectApproxEqAbs(@as(f32, 1), data.normals[2], 1e-6);
    try std.testing.expectEqual(@as(f32, 0), data.uvs[0]);
    try std.testing.expectEqual(@as(f32, 0), data.uvs[1]);
    try std.testing.expect(data.has_colors);
    try std.testing.expectEqual(@as(f32, 1), data.colors[0]);
    try std.testing.expectEqual(@as(f32, 1), data.colors[5]);
    try std.testing.expectEqual(@as(f32, 1), data.colors[10]);
    try std.testing.expectEqual(@as(f32, 1), data.colors[11]);
}

test "ply binary big endian with double positions" {
    const alloc = std.testing.allocator;
    var bytes: std.ArrayListUnmanaged(u8) = .empty;
    defer bytes.deinit(alloc);
    try bytes.appendSlice(alloc,
        \\ply
        \\format binary_big_endian 1.0
        \\element vertex 3
        \\property double x
        \\property double y
        \\property double z
        \\property short nx
        \\property short ny
        \\property short nz
        \\element face 1
        \\property list ushort int vertex_indices
        \\end_header
    );
    for ([3][3]f64{
        .{ 0.25, 0, 0 },
        .{ 1.5, 0, 0 },
        .{ 0, 2.5, 0 },
    }) |row| {
        for (row) |f| try writeTestF64(&bytes, alloc, f, .big);
        for ([3]i16{ 0, 0, 1 }) |n| try writeTestI16(&bytes, alloc, n, .big);
    }
    try writeTestU16(&bytes, alloc, 3, .big);
    try writeTestI32(&bytes, alloc, 0, .big);
    try writeTestI32(&bytes, alloc, 1, .big);
    try writeTestI32(&bytes, alloc, 2, .big);

    var data = try parse(alloc, bytes.items);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 3), data.vertexCount());
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), data.positions[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), data.positions[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), data.positions[7], 1e-6);
    try std.testing.expect(data.has_normals);
    try std.testing.expectApproxEqAbs(@as(f32, 1), data.normals[2], 1e-6);
    try std.testing.expect(!data.has_colors);
    try std.testing.expectEqual(@as(f32, 1), data.colors[0]);
}

test "ply skips unknown properties and elements" {
    const alloc = std.testing.allocator;
    const text =
        \\ply
        \\format ascii 1.0
        \\element vertex 3
        \\property float x
        \\property float confidence
        \\property float y
        \\property uchar quality
        \\property float z
        \\element edge 2
        \\property int vertex1
        \\property int vertex2
        \\property float crease
        \\element face 1
        \\property uchar material
        \\property list uchar uint vertex_indices
        \\property list uchar float texcoord
        \\end_header
        \\0 0.5 0 7 0
        \\1 0.25 0 3 0
        \\0 0.75 1 9 0
        \\0 1 0.1
        \\2 0 1
        \\9 3 0 1 2 2 0.5 0.5
    ;
    var data = try parse(alloc, text);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 3), data.vertexCount());
    try std.testing.expectEqualSlices(u32, &[_]u32{ 0, 1, 2 }, data.indices);
    try std.testing.expectEqual(@as(f32, 0), data.positions[0]);
    try std.testing.expectEqual(@as(f32, 0), data.positions[1]);
    try std.testing.expectEqual(@as(f32, 0), data.positions[2]);
    try std.testing.expectEqual(@as(f32, 1), data.positions[3]);
    try std.testing.expectEqual(@as(f32, 1), data.positions[7]);
    try std.testing.expectEqual(@as(f32, 0), data.positions[8]);
    try std.testing.expect(!data.has_colors);
}

test "ply truncated header is TruncatedPly" {
    const alloc = std.testing.allocator;
    try std.testing.expectError(error.TruncatedPly, parse(alloc, "ply\nformat ascii 1.0\nelement vertex 1\n"));
    try std.testing.expectError(error.TruncatedPly, parse(alloc, "ply"));
}

test "ply truncated data is TruncatedPly" {
    const alloc = std.testing.allocator;
    const ascii =
        \\ply
        \\format ascii 1.0
        \\element vertex 2
        \\property float x
        \\property float y
        \\property float z
        \\element face 1
        \\property list uchar int vertex_indices
        \\end_header
        \\0 0 0
    ;
    try std.testing.expectError(error.TruncatedPly, parse(alloc, ascii));

    var bytes: std.ArrayListUnmanaged(u8) = .empty;
    defer bytes.deinit(alloc);
    try bytes.appendSlice(alloc,
        \\ply
        \\format binary_little_endian 1.0
        \\element vertex 1
        \\property float x
        \\property float y
        \\property float z
        \\element face 0
        \\property list uchar int vertex_indices
        \\end_header
    );
    try writeTestF32(&bytes, alloc, 1.0, .little);
    try writeTestF32(&bytes, alloc, 2.0, .little);
    // Third float of the single vertex is missing.
    try std.testing.expectError(error.TruncatedPly, parse(alloc, bytes.items));
}

test "ply garbage magic is InvalidPly" {
    const alloc = std.testing.allocator;
    try std.testing.expectError(error.InvalidPly, parse(alloc, "this is not a mesh {{{ }}}"));
    try std.testing.expectError(error.InvalidPly, parse(alloc, "PLY\nformat ascii 1.0\n"));
    try std.testing.expectError(error.InvalidPly, parse(alloc, "  ply\nformat ascii 1.0\nend_header\n"));
}

test "ply empty and zero-vertex files are InvalidPly" {
    const alloc = std.testing.allocator;
    try std.testing.expectError(error.InvalidPly, parse(alloc, ""));
    const zero =
        \\ply
        \\format ascii 1.0
        \\element vertex 0
        \\property float x
        \\property float y
        \\property float z
        \\element face 0
        \\property list uchar int vertex_indices
        \\end_header
        \\
    ;
    try std.testing.expectError(error.InvalidPly, parse(alloc, zero));
}

test "ply missing positions are InvalidPly" {
    const alloc = std.testing.allocator;
    const text =
        \\ply
        \\format ascii 1.0
        \\element vertex 1
        \\property float y
        \\property float z
        \\element face 0
        \\property list uchar int vertex_indices
        \\end_header
        \\0 0
    ;
    try std.testing.expectError(error.InvalidPly, parse(alloc, text));
}

test "ply out of range face index is InvalidPly" {
    const alloc = std.testing.allocator;
    const text =
        \\ply
        \\format ascii 1.0
        \\element vertex 3
        \\property float x
        \\property float y
        \\property float z
        \\element face 1
        \\property list uchar int vertex_indices
        \\end_header
        \\0 0 0
        \\1 0 0
        \\0 1 0
        \\3 0 1 9
    ;
    try std.testing.expectError(error.InvalidPly, parse(alloc, text));
}

test "ply unsupported format is UnsupportedPlyFormat" {
    const alloc = std.testing.allocator;
    try std.testing.expectError(
        error.UnsupportedPlyFormat,
        parse(alloc, "ply\nformat binary 1.0\nend_header\n"),
    );
    try std.testing.expectError(
        error.UnsupportedPlyFormat,
        parse(alloc, "ply\nformat ascii 2.0\nend_header\n"),
    );
}

test "ply float colors clamp to 0..1" {
    const alloc = std.testing.allocator;
    const text =
        \\ply
        \\format ascii 1.0
        \\element vertex 3
        \\property float x
        \\property float y
        \\property float z
        \\property float red
        \\property float green
        \\property float blue
        \\element face 1
        \\property list uchar int vertex_indices
        \\end_header
        \\0 0 0 2.0 -1.0 0.5
        \\1 0 0 0 0 0
        \\0 1 0 0 0 0
        \\3 0 1 2
    ;
    var data = try parse(alloc, text);
    defer data.deinit(alloc);
    try std.testing.expect(data.has_colors);
    try std.testing.expectEqual(@as(f32, 1), data.colors[0]);
    try std.testing.expectEqual(@as(f32, 0), data.colors[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), data.colors[2], 1e-6);
}

test "ply zero faces returns InvalidPly without double free" {
    const alloc = std.testing.allocator;
    const text =
        \\ply
        \\format ascii 1.0
        \\element vertex 3
        \\property float x
        \\property float y
        \\property float z
        \\element face 0
        \\property list uchar int vertex_indices
        \\end_header
        \\0 0 0
        \\1 0 0
        \\0 1 0
    ;
    try std.testing.expectError(error.InvalidPly, parse(alloc, text));
}

test "ply rejects invalid magic or truncated header" {
    const alloc = std.testing.allocator;
    try std.testing.expectError(error.InvalidPly, parse(alloc, "not a ply file at all"));
    try std.testing.expectError(error.TruncatedPly, parse(alloc, "ply\n"));
    try std.testing.expectError(error.TruncatedPly, parse(alloc, "ply\nformat ascii 1.0\n"));
}
