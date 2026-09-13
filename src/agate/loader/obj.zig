//! Minimal OBJ importer: pure parser plus an optional Scene upload helper
//! modelled on loader/mesh_spawn.zig.
//!
//! Supports v/vn/vt/f (v, v/vt, v//vn, v/vt/vn, negative indices); o/g and
//! material statements are ignored (single mesh output). Quads and n-gons
//! are fan-triangulated. Missing normals are computed from geometry,
//! missing UVs default to zero.

const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;

const math = @import("math");
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;

const Scene = @import("../scene.zig").Scene;
const Mesh = @import("../mesh.zig").Mesh;
const Vertex = @import("../mesh.zig").Vertex;
const computeTangents = @import("../mesh.zig").computeTangents;
const GeometryData = @import("../mesh.zig").GeometryData;
const uploadGeometry = @import("../mesh.zig").uploadGeometry;

/// Indexed triangle mesh. positions/normals are 3 floats per vertex, uvs are
/// 2 floats per vertex, indices are triples. All slices are owned (allocator
/// passed to parse) and freed by deinit.
pub const ObjData = struct {
    positions: []f32 = &.{},
    normals: []f32 = &.{},
    uvs: []f32 = &.{},
    indices: []u32 = &.{},

    pub fn vertexCount(self: ObjData) usize {
        return self.positions.len / 3;
    }

    pub fn deinit(self: *ObjData, allocator: std.mem.Allocator) void {
        if (self.positions.len > 0) allocator.free(self.positions);
        if (self.normals.len > 0) allocator.free(self.normals);
        if (self.uvs.len > 0) allocator.free(self.uvs);
        if (self.indices.len > 0) allocator.free(self.indices);
        self.* = .{};
    }
};

/// Hard cap on emitted triangles.
pub const max_triangles: usize = 10_000_000;

const FaceVert = struct {
    p: u32,
    t: i32, // -1 = missing
    n: i32, // -1 = missing
};

const FaceKey = struct {
    p: u32,
    t: i32,
    n: i32,
};

fn isWs(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\r';
}

fn trimLeft(s: []const u8) []const u8 {
    var i: usize = 0;
    while (i < s.len and isWs(s[i])) : (i += 1) {}
    return s[i..];
}

fn parseF32(tok: []const u8) !f32 {
    return std.fmt.parseFloat(f32, tok) catch return error.InvalidFormat;
}

fn parseI32(tok: []const u8) !i32 {
    if (tok.len == 0) return error.InvalidFormat;
    return std.fmt.parseInt(i32, tok, 10) catch return error.InvalidFormat;
}

/// Collects whitespace-separated tokens of a line into out; returns count.
fn lineTokens(line: []const u8, out: [][]const u8) usize {
    var count: usize = 0;
    var i: usize = 0;
    while (i < line.len) {
        while (i < line.len and isWs(line[i])) : (i += 1) {}
        if (i >= line.len) break;
        const start = i;
        while (i < line.len and !isWs(line[i])) : (i += 1) {}
        if (count < out.len) out[count] = line[start..i];
        count += 1;
    }
    return count;
}

/// OBJ index: 1-based positive or negative (relative to end). Zero invalid.
fn resolveIdx(raw: i32, count: usize) !u32 {
    if (raw > 0) {
        if (@as(usize, @intCast(raw)) > count) return error.InvalidFormat;
        return @intCast(raw - 1);
    }
    if (raw < 0) {
        const c: i64 = @intCast(count);
        const r: i64 = c + @as(i64, raw);
        if (r < 0) return error.InvalidFormat;
        return @intCast(r);
    }
    return error.InvalidFormat;
}

fn parseFaceVert(tok: []const u8, p_count: usize, t_count: usize, n_count: usize) !FaceVert {
    const s1 = std.mem.indexOfScalar(u8, tok, '/');
    if (s1 == null) {
        return .{ .p = try resolveIdx(try parseI32(tok), p_count), .t = -1, .n = -1 };
    }
    const p = try resolveIdx(try parseI32(tok[0..s1.?]), p_count);
    const rest = tok[s1.? + 1 ..];
    const s2 = std.mem.indexOfScalar(u8, rest, '/');
    if (s2 == null) {
        return .{ .p = p, .t = @intCast(try resolveIdx(try parseI32(rest), t_count)), .n = -1 };
    }
    const tpart = rest[0..s2.?];
    const npart = rest[s2.? + 1 ..];
    if (std.mem.indexOfScalar(u8, npart, '/') != null) return error.InvalidFormat;
    const t: i32 = if (tpart.len == 0) -1 else @intCast(try resolveIdx(try parseI32(tpart), t_count));
    const n: i32 = if (npart.len == 0) -1 else @intCast(try resolveIdx(try parseI32(npart), n_count));
    return .{ .p = p, .t = t, .n = n };
}

/// Parses OBJ text. Errors: InvalidFormat, TooLarge, NoGeometry (plus
/// allocator errors). No Truncated case: input is complete text by contract.
pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !ObjData {
    var raw_p: std.ArrayListUnmanaged(f32) = .empty;
    defer raw_p.deinit(allocator);
    var raw_n: std.ArrayListUnmanaged(f32) = .empty;
    defer raw_n.deinit(allocator);
    var raw_t: std.ArrayListUnmanaged(f32) = .empty;
    defer raw_t.deinit(allocator);

    var out_pos: std.ArrayListUnmanaged(f32) = .empty;
    errdefer out_pos.deinit(allocator);
    var out_nrm: std.ArrayListUnmanaged(f32) = .empty;
    errdefer out_nrm.deinit(allocator);
    var out_uv: std.ArrayListUnmanaged(f32) = .empty;
    errdefer out_uv.deinit(allocator);
    var out_idx: std.ArrayListUnmanaged(u32) = .empty;
    errdefer out_idx.deinit(allocator);
    var has_n: std.ArrayListUnmanaged(bool) = .empty;
    defer has_n.deinit(allocator);

    var map = std.AutoHashMap(FaceKey, u32).init(allocator);
    defer map.deinit();
    var face: std.ArrayListUnmanaged(FaceVert) = .empty;
    defer face.deinit(allocator);

    var saw_content = false;
    var need_compute = false;
    var tri_count: usize = 0;

    var start: usize = 0;
    while (start <= bytes.len) {
        var end = start;
        while (end < bytes.len and bytes[end] != '\n') : (end += 1) {}
        const line = trimLeft(bytes[start..end]);
        start = end + 1;
        if (line.len == 0 or line[0] == '#') continue;
        // Keyword ends at first whitespace; rest is the payload.
        var kw_end: usize = 0;
        while (kw_end < line.len and !isWs(line[kw_end])) : (kw_end += 1) {}
        const kw = line[0..kw_end];
        const rest = trimLeft(line[kw_end..]);

        if (std.mem.eql(u8, kw, "v") or std.mem.eql(u8, kw, "vn") or std.mem.eql(u8, kw, "vt")) {
            saw_content = true;
            var toks: [8][]const u8 = undefined;
            const ntok = lineTokens(rest, &toks);
            if (std.mem.eql(u8, kw, "vt")) {
                if (ntok < 2) return error.InvalidFormat;
                try raw_t.appendSlice(allocator, &[_]f32{ try parseF32(toks[0]), try parseF32(toks[1]) });
            } else {
                if (ntok < 3) return error.InvalidFormat;
                const vals = [_]f32{ try parseF32(toks[0]), try parseF32(toks[1]), try parseF32(toks[2]) };
                if (std.mem.eql(u8, kw, "v")) {
                    try raw_p.appendSlice(allocator, &vals);
                } else {
                    try raw_n.appendSlice(allocator, &vals);
                }
            }
        } else if (std.mem.eql(u8, kw, "f")) {
            saw_content = true;
            face.clearRetainingCapacity();
            var toks: [256][]const u8 = undefined;
            const ntok = lineTokens(rest, &toks);
            if (ntok < 3 or ntok > toks.len) return error.InvalidFormat;
            for (toks[0..ntok]) |tok| {
                try face.append(allocator, try parseFaceVert(tok, raw_p.items.len / 3, raw_t.items.len / 2, raw_n.items.len / 3));
            }
            // Fan triangulation: (0, k, k+1).
            var k: usize = 1;
            while (k + 1 < face.items.len) : (k += 1) {
                if (tri_count >= max_triangles) return error.TooLarge;
                tri_count += 1;
                for ([3]FaceVert{ face.items[0], face.items[k], face.items[k + 1] }) |fv| {
                    const key = FaceKey{ .p = fv.p, .t = fv.t, .n = fv.n };
                    const entry = try map.getOrPut(key);
                    if (!entry.found_existing) {
                        if (out_pos.items.len / 3 > std.math.maxInt(u32)) return error.TooLarge;
                        entry.value_ptr.* = @intCast(out_pos.items.len / 3);
                        try out_pos.appendSlice(allocator, raw_p.items[fv.p * 3 ..][0..3]);
                        if (fv.t >= 0) {
                            const ti: usize = @intCast(fv.t);
                            try out_uv.appendSlice(allocator, raw_t.items[ti * 2 ..][0..2]);
                        } else {
                            try out_uv.appendSlice(allocator, &[_]f32{ 0, 0 });
                        }
                        if (fv.n >= 0) {
                            const ni: usize = @intCast(fv.n);
                            try out_nrm.appendSlice(allocator, raw_n.items[ni * 3 ..][0..3]);
                            try has_n.append(allocator, true);
                        } else {
                            try out_nrm.appendSlice(allocator, &[_]f32{ 0, 0, 0 });
                            try has_n.append(allocator, false);
                            need_compute = true;
                        }
                    }
                    try out_idx.append(allocator, entry.value_ptr.*);
                }
            }
        } else {
            // o, g, s, usemtl, mtllib, unknown: ignored (single mesh).
            if (std.mem.eql(u8, kw, "o") or std.mem.eql(u8, kw, "g")) saw_content = true;
        }
    }

    if (out_idx.items.len == 0) {
        // The errdefers release the out_* lists on these error returns; do not
        // deinit them here as well (double free).
        if (!saw_content) return error.NoGeometry;
        return error.InvalidFormat;
    }

    if (need_compute) {
        var tri: usize = 0;
        while (tri + 2 < out_idx.items.len) : (tri += 3) {
            const a0 = out_idx.items[tri];
            const a1 = out_idx.items[tri + 1];
            const a2 = out_idx.items[tri + 2];
            const p0 = out_pos.items[a0 * 3 ..][0..3];
            const p1 = out_pos.items[a1 * 3 ..][0..3];
            const p2 = out_pos.items[a2 * 3 ..][0..3];
            const e1 = Vec3.new(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]);
            const e2 = Vec3.new(p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]);
            const fn_ = e1.cross(e2);
            for ([3]u32{ a0, a1, a2 }) |idx| {
                if (has_n.items[idx]) continue;
                out_nrm.items[idx * 3] += fn_.x;
                out_nrm.items[idx * 3 + 1] += fn_.y;
                out_nrm.items[idx * 3 + 2] += fn_.z;
            }
        }
        for (0..out_pos.items.len / 3) |i| {
            if (has_n.items[i]) continue;
            const n = Vec3.new(out_nrm.items[i * 3], out_nrm.items[i * 3 + 1], out_nrm.items[i * 3 + 2]);
            const fixed = if (n.lengthSq() > 1e-12) n.normalize() else Vec3.up;
            out_nrm.items[i * 3] = fixed.x;
            out_nrm.items[i * 3 + 1] = fixed.y;
            out_nrm.items[i * 3 + 2] = fixed.z;
        }
    }

    var d: ObjData = .{};
    errdefer d.deinit(allocator);
    d.positions = try out_pos.toOwnedSlice(allocator);
    d.normals = try out_nrm.toOwnedSlice(allocator);
    d.uvs = try out_uv.toOwnedSlice(allocator);
    d.indices = try out_idx.toOwnedSlice(allocator);
    return d;
}

/// Uploads parsed OBJ as one Mesh and returns the single-element spawn list.
/// The CALLER owns the returned slice and must free it with allocator.
/// GPU call — not unit tested; type-checked via ast-check.
pub fn appendToScene(scene: *Scene, allocator: std.mem.Allocator, name: []const u8, bytes: []const u8) ![]*Mesh {
    var data = try parse(allocator, bytes);
    defer data.deinit(allocator);
    const n = data.vertexCount();

    const vertices = try scene.allocator.alloc(Vertex, n);
    // Staging copy only: geometry is retained separately below.
    defer scene.allocator.free(vertices);
    for (0..n) |i| {
        vertices[i] = .{
            .position = .{ data.positions[3 * i], data.positions[3 * i + 1], data.positions[3 * i + 2] },
            .normal = .{ data.normals[3 * i], data.normals[3 * i + 1], data.normals[3 * i + 2] },
            .color = .{ 1, 1, 1, 1 },
            .uv = .{ data.uvs[2 * i], data.uvs[2 * i + 1] },
        };
    }

    var min_p = Vec3.new(vertices[0].position[0], vertices[0].position[1], vertices[0].position[2]);
    var max_p = min_p;
    for (vertices) |vert| {
        min_p.x = @min(min_p.x, vert.position[0]);
        min_p.y = @min(min_p.y, vert.position[1]);
        min_p.z = @min(min_p.z, vert.position[2]);
        max_p.x = @max(max_p.x, vert.position[0]);
        max_p.y = @max(max_p.y, vert.position[1]);
        max_p.z = @max(max_p.z, vert.position[2]);
    }

    computeTangents(vertices, data.indices, null);

    const owned_name = try scene.allocator.dupe(u8, name);
    errdefer scene.allocator.free(owned_name);

    const geom = GeometryData{
        .vertices = vertices,
        .indices = @constCast(data.indices),
        .bounds = BoundingBox.init(min_p, max_p),
    };

    const mesh_obj = try uploadGeometry(scene, owned_name, geom);
    mesh_obj.owns_name = true;
    try mesh_obj.retainMorphBase(scene.allocator, vertices);

    const out = try allocator.alloc(*Mesh, 1);
    out[0] = mesh_obj;
    return out;
}

// ---- tests (GPU-free) ----

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

test "obj appendToScene links (type check)" {
    _ = appendToScene;
}
