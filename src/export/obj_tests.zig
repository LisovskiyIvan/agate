const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Color3 = math.Color3;
const Mesh = @import("../mesh.zig").Mesh;
const obj_loader = @import("../loader/obj.zig");
const MaterialModule = @import("../material.zig");
const PBRMaterial = MaterialModule.PBRMaterial;

const obj_export = @import("obj.zig");
const writeObjAlloc = obj_export.writeObjAlloc;
const writeMtlAlloc = obj_export.writeMtlAlloc;
const sanitizeNameAlloc = obj_export.sanitizeNameAlloc;

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

test "obj export round-trip single triangle" {
    const alloc = std.testing.allocator;
    var positions = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(1, 0, 0), Vec3.new(0, 1, 0) };
    var indices = [_]u32{ 0, 1, 2 };
    var mesh = makeTriMesh(&positions, &indices);
    var list = [_]*Mesh{&mesh};
    const bytes = try writeObjAlloc(alloc, &list, .{});
    defer alloc.free(bytes);
    var data = try obj_loader.parse(alloc, bytes);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 3), data.vertexCount());
    try std.testing.expectEqual(@as(usize, 3), data.indices.len);
    try std.testing.expectApproxEqAbs(@as(f32, 1), data.positions[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), data.positions[7], 1e-6);
}

test "obj export round-trip quad keeps vertex and face counts" {
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
    const bytes = try writeObjAlloc(alloc, &list, .{});
    defer alloc.free(bytes);
    var data = try obj_loader.parse(alloc, bytes);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 4), data.vertexCount());
    try std.testing.expectEqual(@as(usize, 6), data.indices.len);
}

test "obj export skips mesh without cpu geometry" {
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
    const bytes = try writeObjAlloc(alloc, &list, .{});
    defer alloc.free(bytes);
    var data = try obj_loader.parse(alloc, bytes);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 3), data.vertexCount());
    try std.testing.expectEqual(@as(usize, 3), data.indices.len);
}

test "obj export empty list yields empty output" {
    const alloc = std.testing.allocator;
    const empty: []const *Mesh = &.{};
    const bytes = try writeObjAlloc(alloc, empty, .{});
    defer alloc.free(bytes);
    try std.testing.expectEqual(@as(usize, 0), bytes.len);
}

test "obj export visible_only skips hidden meshes" {
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
    const filtered = try writeObjAlloc(alloc, &list, .{ .visible_only = true });
    defer alloc.free(filtered);
    var data = try obj_loader.parse(alloc, filtered);
    defer data.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 3), data.vertexCount());

    const unfiltered = try writeObjAlloc(alloc, &list, .{ .visible_only = false });
    defer alloc.free(unfiltered);
    var data_all = try obj_loader.parse(alloc, unfiltered);
    defer data_all.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 6), data_all.indices.len);
}

test "obj export world transform applies mesh translation" {
    const alloc = std.testing.allocator;
    var positions = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(1, 0, 0), Vec3.new(0, 1, 0) };
    var indices = [_]u32{ 0, 1, 2 };
    var mesh = makeTriMesh(&positions, &indices);
    mesh.position = Vec3.new(10, 0, 0);
    var list = [_]*Mesh{&mesh};

    const world = try writeObjAlloc(alloc, &list, .{ .apply_world_transform = true });
    defer alloc.free(world);
    var data_w = try obj_loader.parse(alloc, world);
    defer data_w.deinit(alloc);
    try std.testing.expectApproxEqAbs(@as(f32, 10), data_w.positions[0], 1e-5);

    const local = try writeObjAlloc(alloc, &list, .{ .apply_world_transform = false });
    defer alloc.free(local);
    var data_l = try obj_loader.parse(alloc, local);
    defer data_l.deinit(alloc);
    try std.testing.expectApproxEqAbs(@as(f32, 0), data_l.positions[0], 1e-6);
}

test "obj sanitize replaces spaces and unicode bytes" {
    const alloc = std.testing.allocator;
    const ascii = try sanitizeNameAlloc(alloc, "hello world!");
    defer alloc.free(ascii);
    try std.testing.expectEqualStrings("hello_world_", ascii);

    const uni_src = "mesh Привет";
    const uni = try sanitizeNameAlloc(alloc, uni_src);
    defer alloc.free(uni);
    // 1:1 byte mapping: spaces and every UTF-8 byte become '_'.
    try std.testing.expectEqual(uni_src.len, uni.len);
    for (uni) |c| {
        try std.testing.expect(c < 128 and c != ' ');
    }

    const empty_name = try sanitizeNameAlloc(alloc, "");
    defer alloc.free(empty_name);
    try std.testing.expectEqualStrings("mesh", empty_name);

    var positions = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(1, 0, 0), Vec3.new(0, 1, 0) };
    var indices = [_]u32{ 0, 1, 2 };
    var mesh = makeTriMesh(&positions, &indices);
    mesh.name = "my mesh";
    var list = [_]*Mesh{&mesh};
    const bytes = try writeObjAlloc(alloc, &list, .{});
    defer alloc.free(bytes);
    try std.testing.expect(std.mem.indexOf(u8, bytes, "o my_mesh\n") != null);
}

test "obj export mtl carries pbr albedo as Kd" {
    const alloc = std.testing.allocator;
    var pbr = PBRMaterial.init("red_mat");
    pbr.albedo_color = Color3.new(1, 0, 0);
    var positions = [_]Vec3{ Vec3.new(0, 0, 0), Vec3.new(1, 0, 0), Vec3.new(0, 1, 0) };
    var indices = [_]u32{ 0, 1, 2 };
    var mesh = makeTriMesh(&positions, &indices);
    mesh.material = .{ .pbr = &pbr };
    var list = [_]*Mesh{&mesh};

    const obj = try writeObjAlloc(alloc, &list, .{});
    defer alloc.free(obj);
    try std.testing.expect(std.mem.indexOf(u8, obj, "mtllib scene.mtl\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, obj, "usemtl red_mat\n") != null);

    const mtl = try writeMtlAlloc(alloc, &list, .{});
    defer alloc.free(mtl);
    try std.testing.expect(std.mem.indexOf(u8, mtl, "newmtl red_mat\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, mtl, "Kd 1 0 0\n") != null);
}
