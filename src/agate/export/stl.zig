//! Mesh -> STL exporter (CPU geometry only, no Scene dependency).
//!
//! Exports `Mesh.cpu_positions` / `Mesh.cpu_indices` as ASCII or binary STL
//! (`options.binary` selects the variant). Meshes without CPU geometry
//! (empty `cpu_positions`) are skipped: glTF imports and geometry builders
//! retain it, GPU-only meshes do not.
//!
//! Normals: flat per-triangle face normals with an up fallback for
//! degenerate triangles (same convention as the OBJ exporter).
//! ASCII layout: `solid <name>`, one `facet normal` / `outer loop` / three
//! `vertex` lines per triangle, `endsolid <name>`.
//! Binary layout: 80-byte header (solid name, zero-padded), u32 LE triangle
//! count, then 50 bytes per triangle (normal + 3 vertices as f32 LE +
//! u16 attribute 0), so the size is always `84 + 50 * tris`.
//!
//! Transforms: `apply_world_transform == false` exports raw CPU coordinates.
//! When true, positions go through `mesh.getWorldMatrix()` (TRS composed
//! with base_matrix, parents and bone attachments). Skinning and morph
//! weights are NOT applied: skinned/morphed meshes export in bind/base pose,
//! and normals under non-uniform scale are direction-transformed (exact for
//! rigid and uniform transforms only).
//! Out-of-range indices drop the triangle; a trailing index tail (< 3) is
//! ignored. Meshes with empty `cpu_indices` export as a triangle soup over
//! `cpu_positions` (trailing vertices < 3 are ignored).

const std = @import("std");

const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;

const Mesh = @import("../mesh.zig").Mesh;

pub const StlExportOptions = struct {
    visible_only: bool = false,
    apply_world_transform: bool = false,
    binary: bool = false,
    solid_name: []const u8 = "agate",
};

/// Same rule as the OBJ exporter: every byte outside `[A-Za-z0-9._-]`
/// becomes `_`, empty input becomes `"mesh"`. Used for the ASCII
/// `solid` name and the binary header.
pub fn sanitizeNameAlloc(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    if (name.len == 0) return allocator.dupe(u8, "mesh");
    const out = try allocator.alloc(u8, name.len);
    errdefer allocator.free(out);
    for (name, 0..) |c, i| {
        const ok = (c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or
            (c >= '0' and c <= '9') or c == '.' or c == '_' or c == '-';
        out[i] = if (ok) c else '_';
    }
    return out;
}

const Tri = struct {
    n: Vec3,
    v: [3]Vec3,
};

/// Flat face normal with an up fallback for degenerate triangles.
fn faceNormal(a: Vec3, b: Vec3, c: Vec3) Vec3 {
    const n = b.sub(a).cross(c.sub(a));
    if (n.lengthSq() < 1e-12) return Vec3.up;
    return n.normalize();
}

fn exportPosition(mesh: *const Mesh, world: ?Mat4, index: usize) Vec3 {
    const p = mesh.cpu_positions[index];
    return if (world) |m| m.transformPoint(p) else p;
}

fn collectTris(
    allocator: std.mem.Allocator,
    meshes: []const *Mesh,
    options: StlExportOptions,
) !std.ArrayListUnmanaged(Tri) {
    var tris: std.ArrayListUnmanaged(Tri) = .empty;
    errdefer tris.deinit(allocator);
    for (meshes) |mesh| {
        if (options.visible_only and !mesh.is_visible) continue;
        if (mesh.cpu_positions.len == 0) continue;
        const world: ?Mat4 = if (options.apply_world_transform) mesh.getWorldMatrix() else null;
        if (mesh.cpu_indices.len > 0) {
            var t: usize = 0;
            while (t + 2 < mesh.cpu_indices.len) : (t += 3) {
                const a = mesh.cpu_indices[t];
                const b = mesh.cpu_indices[t + 1];
                const c = mesh.cpu_indices[t + 2];
                if (a >= mesh.cpu_positions.len or b >= mesh.cpu_positions.len or c >= mesh.cpu_positions.len) continue;
                const pa = exportPosition(mesh, world, a);
                const pb = exportPosition(mesh, world, b);
                const pc = exportPosition(mesh, world, c);
                try tris.append(allocator, .{ .n = faceNormal(pa, pb, pc), .v = .{ pa, pb, pc } });
            }
        } else {
            var v: usize = 0;
            while (v + 2 < mesh.cpu_positions.len) : (v += 3) {
                const pa = exportPosition(mesh, world, v);
                const pb = exportPosition(mesh, world, v + 1);
                const pc = exportPosition(mesh, world, v + 2);
                try tris.append(allocator, .{ .n = faceNormal(pa, pb, pc), .v = .{ pa, pb, pc } });
            }
        }
    }
    return tris;
}

/// Exports meshes as STL text (owned, caller frees with allocator).
/// Empty input yields a `solid`/`endsolid` shell with no facets (which the
/// STL importer rejects with `InvalidFormat` — there is no geometry).
pub fn writeStlAsciiAlloc(
    allocator: std.mem.Allocator,
    meshes: []const *Mesh,
    options: StlExportOptions,
) ![]u8 {
    var tris = try collectTris(allocator, meshes, options);
    defer tris.deinit(allocator);

    const clean = try sanitizeNameAlloc(allocator, options.solid_name);
    defer allocator.free(clean);

    var buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buf.deinit(allocator);

    const header = try std.fmt.allocPrint(allocator, "solid {s}\n", .{clean});
    defer allocator.free(header);
    try buf.appendSlice(allocator, header);

    for (tris.items) |tri| {
        const facet = try std.fmt.allocPrint(
            allocator,
            "  facet normal {d} {d} {d}\n    outer loop\n" ++
                "      vertex {d} {d} {d}\n" ++
                "      vertex {d} {d} {d}\n" ++
                "      vertex {d} {d} {d}\n" ++
                "    endloop\n  endfacet\n",
            .{
                tri.n.x,    tri.n.y,    tri.n.z,
                tri.v[0].x, tri.v[0].y, tri.v[0].z,
                tri.v[1].x, tri.v[1].y, tri.v[1].z,
                tri.v[2].x, tri.v[2].y, tri.v[2].z,
            },
        );
        defer allocator.free(facet);
        try buf.appendSlice(allocator, facet);
    }

    const footer = try std.fmt.allocPrint(allocator, "endsolid {s}\n", .{clean});
    defer allocator.free(footer);
    try buf.appendSlice(allocator, footer);

    return buf.toOwnedSlice(allocator);
}

/// Exports meshes as binary STL (owned, caller frees with allocator).
/// Size is always `84 + 50 * tris`; empty input yields the 84-byte
/// header with a zero count (the importer reports `NoGeometry` for it).
pub fn writeStlBinaryAlloc(
    allocator: std.mem.Allocator,
    meshes: []const *Mesh,
    options: StlExportOptions,
) ![]u8 {
    var tris = try collectTris(allocator, meshes, options);
    defer tris.deinit(allocator);
    if (tris.items.len > std.math.maxInt(u32)) return error.TooLarge;

    const clean = try sanitizeNameAlloc(allocator, options.solid_name);
    defer allocator.free(clean);

    var buf: std.ArrayListUnmanaged(u8) = .empty;
    errdefer buf.deinit(allocator);

    var header: [80]u8 = [_]u8{0} ** 80;
    @memcpy(header[0..@min(clean.len, 80)], clean[0..@min(clean.len, 80)]);
    try buf.appendSlice(allocator, &header);

    var count_buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &count_buf, @intCast(tris.items.len), .little);
    try buf.appendSlice(allocator, &count_buf);

    for (tris.items) |tri| {
        for ([4]Vec3{ tri.n, tri.v[0], tri.v[1], tri.v[2] }) |vec| {
            for ([3]f32{ vec.x, vec.y, vec.z }) |f| {
                var fb: [4]u8 = undefined;
                std.mem.writeInt(u32, &fb, @bitCast(f), .little);
                try buf.appendSlice(allocator, &fb);
            }
        }
        try buf.appendSlice(allocator, &[_]u8{ 0, 0 });
    }

    return buf.toOwnedSlice(allocator);
}

/// Exports meshes as ASCII or binary STL per `options.binary`
/// (owned, caller frees with allocator).
pub fn writeStlAlloc(
    allocator: std.mem.Allocator,
    meshes: []const *Mesh,
    options: StlExportOptions,
) ![]u8 {
    if (options.binary) return writeStlBinaryAlloc(allocator, meshes, options);
    return writeStlAsciiAlloc(allocator, meshes, options);
}

// ---- tests (GPU-free) ----

const stl_loader = @import("../loader/stl.zig");

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
