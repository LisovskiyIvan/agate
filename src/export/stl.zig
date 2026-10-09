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
