//! CPU mesh helpers. Split out of `asset_manager.zig` (facade).
//!
//! - `readFileBytesAlloc`: whole-file read used by every file-backed task.
//! - `sumSpawned`: vertex/index totals over meshes spawned into a scene.
//! - `buildCpuGeometry`: owned CPU `GeometryData` from parsed loader output
//!   (bounds + tangents mirror the `appendToScene` upload path, no GPU).
//! Leaf: no sibling imports.

const std = @import("std");

const mesh_mod = @import("../mesh.zig");
const math = @import("math");

const Mesh = mesh_mod.Mesh;
const GeometryData = mesh_mod.GeometryData;
const Vertex = mesh_mod.Vertex;
const computeTangents = mesh_mod.computeTangents;
const BoundingBox = math.BoundingBox;

pub fn readFileBytesAlloc(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const io = std.Io.Threaded.global_single_threaded.io();
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const len = try file.length(io);
    const n = std.math.cast(usize, len) orelse return error.FileTooLarge;
    const buf = try allocator.alloc(u8, n);
    errdefer allocator.free(buf);
    const read = try file.readPositionalAll(io, buf, 0);
    if (read < buf.len) return error.UnexpectedEof;
    return buf;
}

/// Sums vertex/index totals over meshes spawned into a scene.
pub fn sumSpawned(spawned: []*Mesh) struct { verts: usize, idx: usize } {
    var v: usize = 0;
    var idx: usize = 0;
    for (spawned) |m| {
        v += m.vertex_count;
        idx += m.index_count;
    }
    return .{ .verts = v, .idx = idx };
}

/// Builds owned CPU `GeometryData` from parsed loader output (positions /
/// normals / uvs / indices, white color; PLY passes its own colors). Bounds
/// + tangents mirror the `appendToScene` upload path; no GPU calls.
pub fn buildCpuGeometry(
    allocator: std.mem.Allocator,
    positions: []const f32,
    normals: []const f32,
    uvs: []const f32,
    colors: ?[]const f32,
    indices: []const u32,
) !GeometryData {
    const n = positions.len / 3;
    const vertices = try allocator.alloc(Vertex, n);
    errdefer allocator.free(vertices);
    for (0..n) |i| {
        vertices[i] = .{
            .position = .{ positions[3 * i], positions[3 * i + 1], positions[3 * i + 2] },
            .normal = .{ normals[3 * i], normals[3 * i + 1], normals[3 * i + 2] },
            .color = if (colors) |c| .{ c[4 * i], c[4 * i + 1], c[4 * i + 2], c[4 * i + 3] } else .{ 1, 1, 1, 1 },
            .uv = .{ uvs[2 * i], uvs[2 * i + 1] },
        };
    }
    var min_p = math.Vec3.new(vertices[0].position[0], vertices[0].position[1], vertices[0].position[2]);
    var max_p = min_p;
    for (vertices) |vert| {
        min_p.x = @min(min_p.x, vert.position[0]);
        min_p.y = @min(min_p.y, vert.position[1]);
        min_p.z = @min(min_p.z, vert.position[2]);
        max_p.x = @max(max_p.x, vert.position[0]);
        max_p.y = @max(max_p.y, vert.position[1]);
        max_p.z = @max(max_p.z, vert.position[2]);
    }
    const owned_idx = try allocator.dupe(u32, indices);
    errdefer allocator.free(owned_idx);
    computeTangents(vertices, owned_idx, null);
    return .{
        .vertices = vertices,
        .indices = owned_idx,
        .bounds = BoundingBox.init(min_p, max_p),
    };
}
