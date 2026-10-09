const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Mesh = @import("../mesh.zig").Mesh;
const stl_loader = @import("../loader/stl.zig");
const stl_export = @import("stl.zig");
const writeStlAlloc = stl_export.writeStlAlloc;

fn makeTriMesh(positions: []Vec3, indices: []u32) Mesh {
    return .{
        .name = "tri",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .cpu_positions = positions,
        .cpu_indices = indices,
    };
}

test "stl ascii round-trip single triangle" {
    const alloc = std.testing.allocator;
    var positions = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(1, 0, 0), Vec3.new(0, 1, 0) };
    var indices = [_]u32{ 0, 1, 2 };
    var mesh = makeTriMesh(&positions, &indices);
    var list = [_]*Mesh{&mesh};
    const bytes = try writeStlAlloc(alloc, &list, .{});
    defer alloc.free(bytes);
    try std.testing.expect(std.mem.indexOf(u8, bytes, "solid agate\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, bytes, "endsolid agate\n") != null);
    var data = try stl_loader.parse(alloc, bytes);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 3), data.vertex_count);
    try std.testing.expectEqual(@as(usize, 3), data.indices.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1), data.positions[3], 1e-6);
}

test "stl binary round-trip with size formula and count" {
    const alloc = std.testing.allocator;
    var positions = [_]Vec3{
        Vec3.new(0, 0, 0),
        Vec3.new(1, 0, 0),
        Vec3.new(1, 1, 0),
        Vec3.new(0, 1, 0),
    };
    var indices = [_]u32{ 0, 1, 2, 0, 2, 3 };
    var mesh = Mesh{
        .name = "quad",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 6,
        .cpu_positions = &positions,
        .cpu_indices = &indices,
    };
    var list = [_]*Mesh{&mesh};
    const bytes = try writeStlAlloc(alloc, &list, .{ .binary = true });
    defer alloc.free(bytes);
    const tris: usize = 2;
    try std.testing.expectEqual(@as(usize, 84 + 50 * tris), bytes.len);
    try std.testing.expectEqual(@as(u32, @intCast(tris)), std.mem.readInt(u32, bytes[80..84], .little));
    var data = try stl_loader.parse(alloc, bytes);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 4), data.vertex_count);
    try std.testing.expectEqual(@as(usize, 6), data.indices.len);
}

test "stl empty list: ascii shell rejected, binary is header-only" {
    const alloc = std.testing.allocator;
    const empty: []const *Mesh = &.{};

    const ascii = try writeStlAlloc(alloc, empty, .{});
    defer alloc.free(ascii);
    try std.testing.expect(std.mem.indexOf(u8, ascii, "solid agate\n") != null);
    try std.testing.expectError(error.InvalidFormat, stl_loader.parse(alloc, ascii));

    const binary = try writeStlAlloc(alloc, empty, .{ .binary = true });
    defer alloc.free(binary);
    try std.testing.expectEqual(@as(usize, 84), binary.len);
    try std.testing.expectEqual(@as(u32, 0), std.mem.readInt(u32, binary[80..84], .little));
    try std.testing.expectError(error.NoGeometry, stl_loader.parse(alloc, binary));
}

test "stl skips mesh without cpu geometry" {
    const alloc = std.testing.allocator;
    var positions = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(1, 0, 0), Vec3.new(0, 1, 0) };
    var indices = [_]u32{ 0, 1, 2 };
    var mesh = makeTriMesh(&positions, &indices);
    var bare = Mesh{
        .name = "bare",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 0,
    };
    var list = [_]*Mesh{ &mesh, &bare };
    for ([2]bool{ false, true }) |binary| {
        const bytes = try writeStlAlloc(alloc, &list, .{ .binary = binary });
        defer alloc.free(bytes);
        var data = try stl_loader.parse(alloc, bytes);
        defer data.deinit(alloc);
        try std.testing.expectEqual(@as(usize, 3), data.vertex_count);
    }
}

test "stl visible_only skips hidden meshes" {
    const alloc = std.testing.allocator;
    var positions = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(1, 0, 0), Vec3.new(0, 1, 0) };
    var indices = [_]u32{ 0, 1, 2 };
    var shown = makeTriMesh(&positions, &indices);
    var hidden = Mesh{
        .name = "hidden",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .is_visible = false,
        .cpu_positions = &positions,
        .cpu_indices = &indices,
    };
    var list = [_]*Mesh{ &shown, &hidden };
    const bytes = try writeStlAlloc(alloc, &list, .{ .binary = true, .visible_only = true });
    defer alloc.free(bytes);
    try std.testing.expectEqual(@as(usize, 84 + 50), bytes.len);
    try std.testing.expectEqual(@as(u32, 1), std.mem.readInt(u32, bytes[80..84], .little));
}

test "stl world transform applies mesh translation" {
    const alloc = std.testing.allocator;
    var positions = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(1, 0, 0), Vec3.new(0, 1, 0) };
    var indices = [_]u32{ 0, 1, 2 };
    var mesh = makeTriMesh(&positions, &indices);
    mesh.position = Vec3.new(10, 0, 0);
    var list = [_]*Mesh{&mesh};

    const world = try writeStlAlloc(alloc, &list, .{ .apply_world_transform = true });
    defer alloc.free(world);
    var data_w = try stl_loader.parse(alloc, world);
    defer data_w.deinit(alloc);
    try std.testing.expectApproxEqAbs(@as(f32, 10), data_w.positions[0], 1e-5);

    const local = try writeStlAlloc(alloc, &list, .{ .apply_world_transform = false });
    defer alloc.free(local);
    var data_l = try stl_loader.parse(alloc, local);
    defer data_l.deinit(alloc);
    try std.testing.expectApproxEqAbs(@as(f32, 0), data_l.positions[0], 1e-6);
}

test "stl solid name is sanitized" {
    const alloc = std.testing.allocator;
    var positions = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(1, 0, 0), Vec3.new(0, 1, 0) };
    var indices = [_]u32{ 0, 1, 2 };
    var mesh = makeTriMesh(&positions, &indices);
    var list = [_]*Mesh{&mesh};

    const ascii = try writeStlAlloc(alloc, &list, .{ .solid_name = "my solid" });
    defer alloc.free(ascii);
    try std.testing.expect(std.mem.indexOf(u8, ascii, "solid my_solid\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, ascii, "endsolid my_solid\n") != null);

    const binary = try writeStlAlloc(alloc, &list, .{ .binary = true, .solid_name = "my solid" });
    defer alloc.free(binary);
    try std.testing.expect(std.mem.startsWith(u8, binary, "my_solid"));
}
