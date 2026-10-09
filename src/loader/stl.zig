//! Minimal STL importer (ASCII + binary): pure parser plus an optional
//! Scene upload helper modelled on loader/mesh_spawn.zig.
//!
//! Binary detection: 80-byte header + u32 LE facet count; binary iff
//! `bytes.len == 84 + count * 50`, ASCII otherwise.

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

/// Flat triangle soup with per-vertex normals, deduplicated on import.
/// All slices are owned (allocator passed to parse) and freed by deinit.
pub const StlData = struct {
    positions: []f32 = &.{},
    normals: []f32 = &.{},
    indices: []u32 = &.{},
    vertex_count: usize = 0,

    pub fn deinit(self: *StlData, allocator: std.mem.Allocator) void {
        if (self.positions.len > 0) allocator.free(self.positions);
        if (self.normals.len > 0) allocator.free(self.normals);
        if (self.indices.len > 0) allocator.free(self.indices);
        self.* = .{};
    }
};

/// Hard cap on facet count (binary header or ASCII triangles).
pub const max_facets: usize = 10_000_000;
/// Dedup quantization: 1e-5 on position and normal components.
const quant_scale: f32 = 100000.0;

const VertKey = struct {
    p: [3]i64,
    n: [3]i32,
};

fn quantizePos(x: f32) i64 {
    if (std.math.isNan(x) or std.math.isInf(x)) return 0;
    const scaled = @as(f64, x) * quant_scale;
    const clamped = std.math.clamp(scaled, -9e15, 9e15);
    return @intFromFloat(@round(clamped));
}

fn quantizeNrm(x: f32) i32 {
    if (std.math.isNan(x) or std.math.isInf(x)) return 0;
    const scaled = std.math.clamp(x, -1.0, 1.0) * quant_scale;
    return @intFromFloat(@round(scaled));
}

fn keyFor(pos: [3]f32, nrm: [3]f32) VertKey {
    return .{
        .p = .{ quantizePos(pos[0]), quantizePos(pos[1]), quantizePos(pos[2]) },
        .n = .{ quantizeNrm(nrm[0]), quantizeNrm(nrm[1]), quantizeNrm(nrm[2]) },
    };
}

const Builder = struct {
    positions: std.ArrayListUnmanaged(f32) = .empty,
    normals: std.ArrayListUnmanaged(f32) = .empty,
    indices: std.ArrayListUnmanaged(u32) = .empty,
    map: std.AutoHashMap(VertKey, u32),
    tri_count: usize = 0,

    fn init(allocator: std.mem.Allocator) Builder {
        return .{ .map = std.AutoHashMap(VertKey, u32).init(allocator) };
    }

    fn deinit(self: *Builder, allocator: std.mem.Allocator) void {
        self.positions.deinit(allocator);
        self.normals.deinit(allocator);
        self.indices.deinit(allocator);
        self.map.deinit();
    }

    fn addFacet(self: *Builder, allocator: std.mem.Allocator, n_in: [3]f32, v: [3][3]f32) !void {
        if (self.tri_count >= max_facets) return error.TooLarge;
        self.tri_count += 1;
        var n = n_in;
        if (n[0] * n[0] + n[1] * n[1] + n[2] * n[2] < 1e-12) {
            const e1 = [3]f32{ v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2] };
            const e2 = [3]f32{ v[2][0] - v[0][0], v[2][1] - v[0][1], v[2][2] - v[0][2] };
            const c = [3]f32{
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            };
            const len = @sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
            if (len > 1e-6) {
                n = .{ c[0] / len, c[1] / len, c[2] / len };
            } else {
                n = .{ 0.0, 1.0, 0.0 };
            }
        }
        for (v) |p| {
            const k = keyFor(p, n);
            const entry = try self.map.getOrPut(k);
            if (!entry.found_existing) {
                if (self.positions.items.len / 3 > std.math.maxInt(u32)) return error.TooLarge;
                entry.value_ptr.* = @intCast(self.positions.items.len / 3);
                try self.positions.appendSlice(allocator, &p);
                try self.normals.appendSlice(allocator, &n);
            }
            try self.indices.append(allocator, entry.value_ptr.*);
        }
    }

    fn toData(self: *Builder, allocator: std.mem.Allocator) !StlData {
        var d: StlData = .{};
        errdefer d.deinit(allocator);
        d.positions = try self.positions.toOwnedSlice(allocator);
        d.normals = try self.normals.toOwnedSlice(allocator);
        d.indices = try self.indices.toOwnedSlice(allocator);
        d.vertex_count = d.positions.len / 3;
        return d;
    }
};

fn isWs(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\r' or c == '\n';
}

fn parseF32(tok: []const u8) !f32 {
    return std.fmt.parseFloat(f32, tok) catch return error.InvalidFormat;
}

fn readF32LE(bytes: []const u8) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[0..4], .little));
}

/// Returns the binary facet count when the size matches exactly, else null.
fn binaryCount(bytes: []const u8) ?u32 {
    if (bytes.len < 84) return null;
    const count = std.mem.readInt(u32, bytes[80..84], .little);
    const want: u64 = 84 + @as(u64, count) * 50;
    if (@as(u64, bytes.len) == want) return count;
    return null;
}

fn parseBinary(allocator: std.mem.Allocator, bytes: []const u8, count: u32) !StlData {
    if (count == 0) return error.NoGeometry;
    if (count > max_facets) return error.TooLarge;
    var b = Builder.init(allocator);
    errdefer b.deinit(allocator);
    var off: usize = 84;
    var i: u32 = 0;
    while (i < count) : (i += 1) {
        if (off + 50 > bytes.len) return error.Truncated;
        const n: [3]f32 = .{ readF32LE(bytes[off..]), readF32LE(bytes[off + 4 ..]), readF32LE(bytes[off + 8 ..]) };
        var v: [3][3]f32 = undefined;
        var k: usize = 0;
        while (k < 3) : (k += 1) {
            const base = off + 12 + k * 12;
            v[k] = .{ readF32LE(bytes[base..]), readF32LE(bytes[base + 4 ..]), readF32LE(bytes[base + 8 ..]) };
        }
        // u16 attribute at off+48 is intentionally ignored.
        try b.addFacet(allocator, n, v);
        off += 50;
    }
    const data = try b.toData(allocator);
    b.map.deinit();
    return data;
}

const Tokenizer = struct {
    s: []const u8,
    pos: usize = 0,

    fn next(self: *Tokenizer) ?[]const u8 {
        while (self.pos < self.s.len and isWs(self.s[self.pos])) : (self.pos += 1) {}
        if (self.pos >= self.s.len) return null;
        const start = self.pos;
        while (self.pos < self.s.len and !isWs(self.s[self.pos])) : (self.pos += 1) {}
        return self.s[start..self.pos];
    }
};

fn parseAscii(allocator: std.mem.Allocator, bytes: []const u8) !StlData {
    var b = Builder.init(allocator);
    errdefer b.deinit(allocator);
    var t = Tokenizer{ .s = bytes };
    var n: [3]f32 = .{ 0, 0, 0 };
    var v: [3][3]f32 = undefined;
    var nv: usize = 0;
    var in_facet = false;
    var saw_any = false;
    // ASCII STL must open with "solid"; reaching this path without it means
    // the bytes are not STL text (e.g. a corrupt binary failed the size check).
    const first = t.next() orelse return error.NoGeometry;
    if (!std.mem.eql(u8, first, "solid")) return error.InvalidFormat;
    saw_any = true;
    while (t.next()) |tok| {
        if (std.mem.eql(u8, tok, "solid")) {
            saw_any = true;
        } else if (std.mem.eql(u8, tok, "facet")) {
            const w = t.next() orelse return error.Truncated;
            if (!std.mem.eql(u8, w, "normal")) return error.InvalidFormat;
            n[0] = try parseF32(t.next() orelse return error.Truncated);
            n[1] = try parseF32(t.next() orelse return error.Truncated);
            n[2] = try parseF32(t.next() orelse return error.Truncated);
            in_facet = true;
            nv = 0;
            saw_any = true;
        } else if (std.mem.eql(u8, tok, "vertex")) {
            if (!in_facet or nv >= 3) return error.InvalidFormat;
            v[nv][0] = try parseF32(t.next() orelse return error.Truncated);
            v[nv][1] = try parseF32(t.next() orelse return error.Truncated);
            v[nv][2] = try parseF32(t.next() orelse return error.Truncated);
            nv += 1;
            saw_any = true;
        } else if (std.mem.eql(u8, tok, "endfacet")) {
            if (!in_facet or nv != 3) return error.Truncated;
            try b.addFacet(allocator, n, v);
            in_facet = false;
        } else {
            // outer/loop/endloop/endsolid/solid names: ignored.
        }
    }
    if (in_facet) return error.Truncated;
    if (b.tri_count == 0) {
        if (!saw_any) return error.NoGeometry;
        return error.InvalidFormat;
    }
    const data = try b.toData(allocator);
    b.map.deinit();
    return data;
}

/// Parses ASCII or binary STL. Errors: InvalidFormat, Truncated, TooLarge,
/// NoGeometry (plus allocator errors).
pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !StlData {
    if (bytes.len == 0) return error.NoGeometry;
    if (binaryCount(bytes)) |count| return parseBinary(allocator, bytes, count);
    return parseAscii(allocator, bytes);
}

/// Uploads parsed STL as one Mesh (white, zero UVs) and returns the
/// single-element spawn list. The CALLER owns the returned slice and must
/// free it with allocator. GPU call — not unit tested; type-checked via
/// ast-check.
pub fn appendToScene(scene: *Scene, allocator: std.mem.Allocator, name: []const u8, bytes: []const u8) ![]*Mesh {
    var data = try parse(allocator, bytes);
    defer data.deinit(allocator);
    const n = data.vertex_count;

    const vertices = try scene.allocator.alloc(Vertex, n);
    // Staging copy only: geometry is retained separately below.
    defer scene.allocator.free(vertices);
    for (0..n) |i| {
        vertices[i] = .{
            .position = .{ data.positions[3 * i], data.positions[3 * i + 1], data.positions[3 * i + 2] },
            .normal = .{ data.normals[3 * i], data.normals[3 * i + 1], data.normals[3 * i + 2] },
            .color = .{ 1, 1, 1, 1 },
            .uv = .{ 0, 0 },
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
