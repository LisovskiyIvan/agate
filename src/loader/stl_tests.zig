const std = @import("std");
const stl = @import("stl.zig");
const parse = stl.parse;

fn writeF32LE(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, f: f32) !void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, @bitCast(f), .little);
    try list.appendSlice(allocator, &buf);
}

fn writeU16LE(list: *std.ArrayListUnmanaged(u8), allocator: std.mem.Allocator, v: u16) !void {
    var buf: [2]u8 = undefined;
    std.mem.writeInt(u16, &buf, v, .little);
    try list.appendSlice(allocator, &buf);
}

test "stl binary two facets with shared edge dedup" {
    const alloc = std.testing.allocator;
    var bytes: std.ArrayListUnmanaged(u8) = .empty;
    defer bytes.deinit(alloc);
    try bytes.appendNTimes(alloc, 0, 80);
    var count_buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &count_buf, 2, .little);
    try bytes.appendSlice(alloc, &count_buf);
    // facet 1: (0,0,0) (1,0,0) (0,1,0), normal (0,0,1)
    for ([3]f32{ 0, 0, 1 }) |f| try writeF32LE(&bytes, alloc, f);
    for ([9]f32{ 0, 0, 0, 1, 0, 0, 0, 1, 0 }) |f| try writeF32LE(&bytes, alloc, f);
    try writeU16LE(&bytes, alloc, 0);
    // facet 2: (1,0,0) (1,1,0) (0,1,0), same normal
    for ([3]f32{ 0, 0, 1 }) |f| try writeF32LE(&bytes, alloc, f);
    for ([9]f32{ 1, 0, 0, 1, 1, 0, 0, 1, 0 }) |f| try writeF32LE(&bytes, alloc, f);
    try writeU16LE(&bytes, alloc, 0);

    var data = try parse(alloc, bytes.items);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 4), data.vertex_count);
    try std.testing.expectEqual(@as(usize, 6), data.indices.len);
    try std.testing.expectEqual(@as(f32, 1), data.positions[3]);
    try std.testing.expectEqual(@as(f32, 1), data.normals[2]);
}

test "stl ascii single triangle" {
    const alloc = std.testing.allocator;
    const text =
        \\solid test
        \\  facet normal 0 0 1
        \\    outer loop
        \\      vertex 0 0 0
        \\      vertex 1 0 0
        \\      vertex 0 1 0
        \\    endloop
        \\  endfacet
        \\endsolid test
    ;
    var data = try parse(alloc, text);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 3), data.vertex_count);
    try std.testing.expectEqual(@as(usize, 3), data.indices.len);
    try std.testing.expectEqual(@as(f32, 1), data.positions[3]);
    try std.testing.expectEqual(@as(f32, 1), data.normals[2]);
}

test "stl garbage is InvalidFormat" {
    const alloc = std.testing.allocator;
    try std.testing.expectError(error.InvalidFormat, parse(alloc, "this is not a mesh {{{ }}}"));
}

test "stl truncated ascii is Truncated" {
    const alloc = std.testing.allocator;
    try std.testing.expectError(error.Truncated, parse(alloc, "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\n"));
}

test "stl empty is NoGeometry" {
    const alloc = std.testing.allocator;
    try std.testing.expectError(error.NoGeometry, parse(alloc, ""));
}

test "stl large coordinates do not panic" {
    const alloc = std.testing.allocator;
    const text =
        \\solid test
        \\  facet normal 0 0 1
        \\    outer loop
        \\      vertex 50000.0 0 0
        \\      vertex 50001.0 0 0
        \\      vertex 50000.0 1.0 0
        \\    endloop
        \\  endfacet
        \\endsolid test
    ;
    var data = try parse(alloc, text);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 3), data.vertex_count);
    try std.testing.expectApproxEqAbs(@as(f32, 50000.0), data.positions[0], 1e-3);
}

test "stl zero facet normal is computed from geometry" {
    const alloc = std.testing.allocator;
    const text =
        \\solid test
        \\  facet normal 0 0 0
        \\    outer loop
        \\      vertex 0 0 0
        \\      vertex 1 0 0
        \\      vertex 0 1 0
        \\    endloop
        \\  endfacet
        \\endsolid test
    ;
    var data = try parse(alloc, text);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 3), data.vertex_count);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.normals[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data.normals[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), data.normals[2], 1e-5);
}

test "stl rejects invalid binary header and zero facets" {
    const alloc = std.testing.allocator;
    const short_buf = [_]u8{0} ** 50;
    try std.testing.expectError(error.InvalidFormat, parse(alloc, &short_buf));

    const empty_buf = [_]u8{0} ** 84;
    try std.testing.expectError(error.NoGeometry, parse(alloc, &empty_buf));
}
