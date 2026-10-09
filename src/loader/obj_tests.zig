const std = @import("std");
const obj = @import("obj.zig");
const parse = obj.parse;

test "obj triangle with computed normals" {
    const alloc = std.testing.allocator;
    const text =
        \\v 0 0 0
        \\v 1 0 0
        \\v 0 1 0
        \\f 1 2 3
    ;
    var data = try parse(alloc, text);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 3), data.vertexCount());
    try std.testing.expectEqual(@as(usize, 3), data.indices.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0), data.normals[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), data.normals[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), data.normals[2], 1e-6);
    try std.testing.expectEqual(@as(f32, 0), data.uvs[0]);
}

test "obj quad fans into two triangles" {
    const alloc = std.testing.allocator;
    const text =
        \\v 0 0 0
        \\v 1 0 0
        \\v 1 1 0
        \\v 0 1 0
        \\f 1 2 3 4
    ;
    var data = try parse(alloc, text);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 4), data.vertexCount());
    try std.testing.expectEqual(@as(usize, 6), data.indices.len);
}

test "obj v/vt/vn with negative indices" {
    const alloc = std.testing.allocator;
    const text =
        \\v 0 0 0
        \\v 1 0 0
        \\v 1 1 0
        \\v 0 1 0
        \\vt 0 0
        \\vt 1 0
        \\vt 1 1
        \\vt 0 1
        \\vn 0 0 1
        \\f -4/-4/-1 -3/-3/-1 -2/-2/-1 -1/-1/-1
    ;
    var data = try parse(alloc, text);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 4), data.vertexCount());
    try std.testing.expectEqual(@as(usize, 6), data.indices.len);
    // First emitted vertex maps to v1/vt1/vn1.
    try std.testing.expectEqual(@as(f32, 0), data.positions[0]);
    try std.testing.expectEqual(@as(f32, 0), data.uvs[0]);
    try std.testing.expectEqual(@as(f32, 0), data.uvs[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 1), data.normals[2], 1e-6);
}

test "obj v//vn form keeps file normals" {
    const alloc = std.testing.allocator;
    const text =
        \\v 0 0 0
        \\v 1 0 0
        \\v 0 1 0
        \\vn 0 0 1
        \\f 1//1 2//1 3//1
    ;
    var data = try parse(alloc, text);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 3), data.vertexCount());
    try std.testing.expectApproxEqAbs(@as(f32, 1), data.normals[2], 1e-6);
}

test "obj empty is NoGeometry" {
    const alloc = std.testing.allocator;
    try std.testing.expectError(error.NoGeometry, parse(alloc, ""));
    try std.testing.expectError(error.NoGeometry, parse(alloc, "# only a comment\n\n"));
}

test "obj rejects invalid and out-of-bounds face indices" {
    const alloc = std.testing.allocator;
    const zero_idx =
        \\v 0 0 0
        \\v 1 0 0
        \\v 0 1 0
        \\f 0 1 2
    ;
    try std.testing.expectError(error.InvalidFormat, parse(alloc, zero_idx));

    const out_of_bounds =
        \\v 0 0 0
        \\v 1 0 0
        \\v 0 1 0
        \\f 1 2 999
    ;
    try std.testing.expectError(error.InvalidFormat, parse(alloc, out_of_bounds));

    const negative_out_of_bounds =
        \\v 0 0 0
        \\v 1 0 0
        \\v 0 1 0
        \\f -99 1 2
    ;
    try std.testing.expectError(error.InvalidFormat, parse(alloc, negative_out_of_bounds));
}
