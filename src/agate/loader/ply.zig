//! Minimal PLY importer (Stanford polygon format, ASCII + binary):
//! pure parser plus an optional Scene upload helper modelled on
//! loader/stl.zig and loader/obj.zig.
//!
//! Header: `ply` magic, one `format` line (`ascii 1.0`,
//! `binary_little_endian 1.0`, `binary_big_endian 1.0`), `element` and
//! `property` declarations (scalar and `property list` forms), `comment` and
//! `obj_info` lines ignored, terminated by `end_header`. Anything else in the
//! header is InvalidPly; a header cut before `end_header` is TruncatedPly.
//!
//! Geometry: the `vertex` element must carry x, y, z. Recognized optional
//! attributes are normals (nx, ny, nz), UVs (u/v or s/t, plus texture_u/v
//! aliases) and colors (red/green/blue with optional alpha; r/g/b and
//! diffuse_* aliases accepted). Integer colors scale by their type max
//! (uchar convention: 255); float colors are used as 0..1 and clamped. The
//! `face` element contributes triangles through its `vertex_indices` list
//! property (vertex_index accepted as fallback) with fan triangulation.
//! Unknown elements and properties are read and skipped, so files with extra
//! channels (confidence, intensity, ...) or sections (edge, tristrips, ...)
//! still load.
//!
//! Colors land in Vertex.color (white when the file has none); no Material
//! is created, mirroring stl.zig/obj.zig. Use PlyData.averageColor to seed a
//! PBR albedo through Scene.createPBRMaterial when needed. Missing normals
//! are computed with the same area-weighted accumulation as obj.zig.

const std = @import("std");

const math = @import("math");
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;

const Scene = @import("../scene.zig").Scene;
const Mesh = @import("../mesh.zig").Mesh;
const Vertex = @import("../mesh.zig").Vertex;
const GeometryData = @import("../mesh.zig").GeometryData;
const uploadGeometry = @import("../mesh.zig").uploadGeometry;
const computeTangents = @import("../mesh.zig").computeTangents;

/// Indexed triangle mesh parsed from PLY. positions/normals are 3 floats per
/// vertex, uvs are 2 floats per vertex, colors are RGBA (white when the file
/// has no colors), indices are triples. All slices are owned (allocator
/// passed to parse) and freed by deinit.
pub const PlyData = struct {
    positions: []f32 = &.{},
    normals: []f32 = &.{},
    uvs: []f32 = &.{},
    colors: []f32 = &.{},
    indices: []u32 = &.{},
    has_normals: bool = false,
    has_uvs: bool = false,
    has_colors: bool = false,

    pub fn vertexCount(self: PlyData) usize {
        return self.positions.len / 3;
    }

    /// Mean vertex RGB (white when empty); seed for a caller-built PBR albedo.
    pub fn averageColor(self: PlyData) [3]f32 {
        const n = self.vertexCount();
        if (n == 0 or self.colors.len < n * 4) return .{ 1, 1, 1 };
        var acc = [3]f64{ 0, 0, 0 };
        for (0..n) |i| {
            acc[0] += self.colors[4 * i];
            acc[1] += self.colors[4 * i + 1];
            acc[2] += self.colors[4 * i + 2];
        }
        const d: f64 = @floatFromInt(n);
        return .{ @floatCast(acc[0] / d), @floatCast(acc[1] / d), @floatCast(acc[2] / d) };
    }

    pub fn deinit(self: *PlyData, allocator: std.mem.Allocator) void {
        if (self.positions.len > 0) allocator.free(self.positions);
        if (self.normals.len > 0) allocator.free(self.normals);
        if (self.uvs.len > 0) allocator.free(self.uvs);
        if (self.colors.len > 0) allocator.free(self.colors);
        if (self.indices.len > 0) allocator.free(self.indices);
        self.* = .{};
    }
};

const ScalarType = enum {
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    f32,
    f64,
};

fn scalarTypeFromName(name: []const u8) ?ScalarType {
    if (std.mem.eql(u8, name, "char") or std.mem.eql(u8, name, "int8")) return .i8;
    if (std.mem.eql(u8, name, "uchar") or std.mem.eql(u8, name, "uint8")) return .u8;
    if (std.mem.eql(u8, name, "short") or std.mem.eql(u8, name, "int16")) return .i16;
    if (std.mem.eql(u8, name, "ushort") or std.mem.eql(u8, name, "uint16")) return .u16;
    if (std.mem.eql(u8, name, "int") or std.mem.eql(u8, name, "int32")) return .i32;
    if (std.mem.eql(u8, name, "uint") or std.mem.eql(u8, name, "uint32")) return .u32;
    if (std.mem.eql(u8, name, "float") or std.mem.eql(u8, name, "float32")) return .f32;
    if (std.mem.eql(u8, name, "double") or std.mem.eql(u8, name, "float64")) return .f64;
    return null;
}

fn scalarByteSize(t: ScalarType) usize {
    return switch (t) {
        .i8, .u8 => 1,
        .i16, .u16 => 2,
        .i32, .u32, .f32 => 4,
        .f64 => 8,
    };
}

/// Integer color channels scale by their type max (uchar convention: 255);
/// float channels are already 0..1 (clamped at append).
fn colorScale(t: ScalarType) f32 {
    return switch (t) {
        .u8 => 1.0 / 255.0,
        .u16 => 1.0 / 65535.0,
        .u32 => 1.0 / 4294967295.0,
        .i8 => 1.0 / 127.0,
        .i16 => 1.0 / 32767.0,
        .i32 => 1.0 / 2147483647.0,
        .f32, .f64 => 1.0,
    };
}

const Prop = struct {
    name: []const u8, // slice into the input bytes
    is_list: bool,
    scalar: ScalarType, // value type, or item type for lists
    count: ScalarType, // item-count type for lists
};

const Element = struct {
    name: []const u8, // slice into the input bytes
    count: usize,
    props: std.ArrayListUnmanaged(Prop) = .empty,
};

const Format = enum {
    ascii,
    little,
    big,
};

const Header = struct {
    format: Format = .ascii,
    has_format: bool = false,
    data_off: usize = 0,
    elements: std.ArrayListUnmanaged(Element) = .empty,

    fn deinit(self: *Header, allocator: std.mem.Allocator) void {
        for (self.elements.items) |*e| e.props.deinit(allocator);
        self.elements.deinit(allocator);
    }
};

fn isHws(c: u8) bool {
    return c == ' ' or c == '\t';
}

fn trimH(s: []const u8) []const u8 {
    var lo: usize = 0;
    var hi: usize = s.len;
    while (lo < hi and isHws(s[lo])) : (lo += 1) {}
    while (hi > lo and isHws(s[hi - 1])) : (hi -= 1) {}
    return s[lo..hi];
}

fn firstToken(line: []const u8) []const u8 {
    var end: usize = 0;
    while (end < line.len and !isHws(line[end])) : (end += 1) {}
    return line[0..end];
}

/// Splits a header line on spaces/tabs; returns the total token count (may
/// exceed out.len, in which case only the prefix is stored).
fn headerTokens(line: []const u8, out: [][]const u8) usize {
    var n: usize = 0;
    var i: usize = 0;
    while (i < line.len) {
        while (i < line.len and isHws(line[i])) : (i += 1) {}
        if (i >= line.len) break;
        const start = i;
        while (i < line.len and !isHws(line[i])) : (i += 1) {}
        if (n < out.len) out[n] = line[start..i];
        n += 1;
    }
    return n;
}

fn appendProp(allocator: std.mem.Allocator, element: *Element, p: Prop) !void {
    for (element.props.items) |old| {
        if (std.mem.eql(u8, old.name, p.name)) return error.InvalidPly;
    }
    try element.props.append(allocator, p);
}

fn parseHeader(allocator: std.mem.Allocator, bytes: []const u8) !Header {
    var h: Header = .{};
    errdefer h.deinit(allocator);
    var pos: usize = 0;
    var first = true;
    var cur: ?usize = null;
    while (true) {
        if (pos >= bytes.len) return error.TruncatedPly;
        var eol = pos;
        while (eol < bytes.len and bytes[eol] != '\n') : (eol += 1) {}
        const terminated = eol < bytes.len;
        var line = bytes[pos..eol];
        pos = if (terminated) eol + 1 else eol;
        if (line.len > 0 and line[line.len - 1] == '\r') line = line[0 .. line.len - 1];
        line = trimH(line);
        if (first) {
            first = false;
            if (!std.mem.eql(u8, line, "ply")) return error.InvalidPly;
            continue;
        }
        if (line.len == 0) continue;
        // Binary payload may start directly after "end_header" with no
        // newline, so the keyword must be detected by prefix: tokenizing the
        // line would fold payload bytes into the keyword token. The exact
        // match takes the normal path; otherwise the header ends where the
        // keyword ends (line is trimmed, so its start is eol - line.len).
        if (std.mem.startsWith(u8, line, "end_header")) {
            if (std.mem.eql(u8, line, "end_header")) {
                h.data_off = pos;
                break;
            }
            h.data_off = eol - line.len + "end_header".len;
            break;
        }
        const kw = firstToken(line);
        if (std.mem.eql(u8, kw, "comment") or std.mem.eql(u8, kw, "obj_info")) continue;
        var toks: [8][]const u8 = undefined;
        const n = headerTokens(line, &toks);
        if (std.mem.eql(u8, kw, "format")) {
            if (h.has_format) return error.InvalidPly;
            if (n != 3) return error.InvalidPly;
            if (!std.mem.eql(u8, toks[2], "1.0")) return error.UnsupportedPlyFormat;
            if (std.mem.eql(u8, toks[1], "ascii")) {
                h.format = .ascii;
            } else if (std.mem.eql(u8, toks[1], "binary_little_endian")) {
                h.format = .little;
            } else if (std.mem.eql(u8, toks[1], "binary_big_endian")) {
                h.format = .big;
            } else return error.UnsupportedPlyFormat;
            h.has_format = true;
        } else if (std.mem.eql(u8, kw, "element")) {
            if (n != 3) return error.InvalidPly;
            const count = std.fmt.parseInt(usize, toks[2], 10) catch return error.InvalidPly;
            try h.elements.append(allocator, .{ .name = toks[1], .count = count });
            cur = h.elements.items.len - 1;
        } else if (std.mem.eql(u8, kw, "property")) {
            const ei = cur orelse return error.InvalidPly;
            if (n == 5 and std.mem.eql(u8, toks[1], "list")) {
                const ct = scalarTypeFromName(toks[2]) orelse return error.InvalidPly;
                const it = scalarTypeFromName(toks[3]) orelse return error.InvalidPly;
                try appendProp(allocator, &h.elements.items[ei], .{ .name = toks[4], .is_list = true, .scalar = it, .count = ct });
            } else if (n == 3) {
                const t = scalarTypeFromName(toks[1]) orelse return error.InvalidPly;
                try appendProp(allocator, &h.elements.items[ei], .{ .name = toks[2], .is_list = false, .scalar = t, .count = .u8 });
            } else return error.InvalidPly;
        } else return error.InvalidPly;
    }
    if (!h.has_format) return error.InvalidPly;
    return h;
}

fn isAsciiWs(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\r' or c == '\n';
}

const AsciiCursor = struct {
    s: []const u8,
    pos: usize = 0,

    fn next(self: *AsciiCursor) ?[]const u8 {
        while (self.pos < self.s.len and isAsciiWs(self.s[self.pos])) : (self.pos += 1) {}
        if (self.pos >= self.s.len) return null;
        const start = self.pos;
        while (self.pos < self.s.len and !isAsciiWs(self.s[self.pos])) : (self.pos += 1) {}
        return self.s[start..self.pos];
    }
};

fn nextAsciiFloat(c: *AsciiCursor) !f32 {
    const tok = c.next() orelse return error.TruncatedPly;
    const v = std.fmt.parseFloat(f64, tok) catch return error.InvalidPly;
    if (!std.math.isFinite(v)) return error.InvalidPly;
    return @floatCast(v);
}

fn nextAsciiInt(c: *AsciiCursor) !i64 {
    const tok = c.next() orelse return error.TruncatedPly;
    return std.fmt.parseInt(i64, tok, 10) catch return error.InvalidPly;
}

const BinCursor = struct {
    s: []const u8,
    pos: usize = 0,
    endian: std.builtin.Endian = .little,

    fn take(self: *BinCursor, n: usize) ![]const u8 {
        if (self.pos > self.s.len or n > self.s.len - self.pos) return error.TruncatedPly;
        const out = self.s[self.pos..][0..n];
        self.pos += n;
        return out;
    }
};

fn readScalarF32(t: ScalarType, bytes: []const u8, endian: std.builtin.Endian) f32 {
    return switch (t) {
        .i8 => @floatFromInt(@as(i8, @bitCast(bytes[0]))),
        .u8 => @floatFromInt(bytes[0]),
        .i16 => @floatFromInt(std.mem.readInt(i16, bytes[0..2], endian)),
        .u16 => @floatFromInt(std.mem.readInt(u16, bytes[0..2], endian)),
        .i32 => @floatFromInt(std.mem.readInt(i32, bytes[0..4], endian)),
        .u32 => @floatFromInt(std.mem.readInt(u32, bytes[0..4], endian)),
        .f32 => @as(f32, @bitCast(std.mem.readInt(u32, bytes[0..4], endian))),
        .f64 => @floatCast(@as(f64, @bitCast(std.mem.readInt(u64, bytes[0..8], endian)))),
    };
}

fn readScalarU64(t: ScalarType, bytes: []const u8, endian: std.builtin.Endian) !u64 {
    const v: i64 = switch (t) {
        .i8 => @as(i8, @bitCast(bytes[0])),
        .u8 => return @as(u64, bytes[0]),
        .i16 => std.mem.readInt(i16, bytes[0..2], endian),
        .u16 => return @as(u64, std.mem.readInt(u16, bytes[0..2], endian)),
        .i32 => std.mem.readInt(i32, bytes[0..4], endian),
        .u32 => return @as(u64, std.mem.readInt(u32, bytes[0..4], endian)),
        .f32 => {
            const f: f32 = @bitCast(std.mem.readInt(u32, bytes[0..4], endian));
            if (!std.math.isFinite(f) or f < 0 or f >= 18446744073709551616.0) return error.InvalidPly;
            return @as(u64, @intFromFloat(f));
        },
        .f64 => {
            const f: f64 = @bitCast(std.mem.readInt(u64, bytes[0..8], endian));
            if (!std.math.isFinite(f) or f < 0 or f >= 18446744073709551616.0) return error.InvalidPly;
            return @as(u64, @intFromFloat(f));
        },
    };
    if (v < 0) return error.InvalidPly;
    return @intCast(v);
}

const VertRole = enum {
    skip,
    px,
    py,
    pz,
    nx,
    ny,
    nz,
    uu,
    vv,
    cr,
    cg,
    cb,
    ca,
};

fn findProp(props: []const Prop, names: []const []const u8) ?usize {
    for (names) |want| {
        for (props, 0..) |p, i| {
            if (std.mem.eql(u8, p.name, want)) return i;
        }
    }
    return null;
}

const VertRow = struct {
    p: [3]f32 = .{ 0, 0, 0 },
    n: [3]f32 = .{ 0, 0, 0 },
    uv: [2]f32 = .{ 0, 0 },
    c: [4]f32 = .{ 0, 0, 0, 1 },
};

fn readVertexAscii(c: *AsciiCursor, props: []const Prop, roles: []const VertRole, row: *VertRow) !void {
    for (props, roles) |p, role| {
        if (p.is_list) {
            // Lists on vertices are non-standard; consume and ignore.
            const n = try nextAsciiInt(c);
            if (n < 0) return error.InvalidPly;
            var k: i64 = 0;
            while (k < n) : (k += 1) {
                _ = c.next() orelse return error.TruncatedPly;
            }
            continue;
        }
        const f = try nextAsciiFloat(c);
        switch (role) {
            .skip => {},
            .px => row.p[0] = f,
            .py => row.p[1] = f,
            .pz => row.p[2] = f,
            .nx => row.n[0] = f,
            .ny => row.n[1] = f,
            .nz => row.n[2] = f,
            .uu => row.uv[0] = f,
            .vv => row.uv[1] = f,
            .cr => row.c[0] = f,
            .cg => row.c[1] = f,
            .cb => row.c[2] = f,
            .ca => row.c[3] = f,
        }
    }
}

fn readVertexBinary(c: *BinCursor, props: []const Prop, roles: []const VertRole, row: *VertRow) !void {
    for (props, roles) |p, role| {
        if (p.is_list) {
            const raw_count = try c.take(scalarByteSize(p.count));
            const n = try readScalarU64(p.count, raw_count, c.endian);
            const item = scalarByteSize(p.scalar);
            // Each item costs at least one byte: bounds the skip without overflow.
            if (n > @as(u64, @intCast((c.s.len - c.pos) / item))) return error.TruncatedPly;
            c.pos += @as(usize, @intCast(n)) * item;
            continue;
        }
        const raw = try c.take(scalarByteSize(p.scalar));
        const f = readScalarF32(p.scalar, raw, c.endian);
        switch (role) {
            .skip => {},
            .px => row.p[0] = f,
            .py => row.p[1] = f,
            .pz => row.p[2] = f,
            .nx => row.n[0] = f,
            .ny => row.n[1] = f,
            .nz => row.n[2] = f,
            .uu => row.uv[0] = f,
            .vv => row.uv[1] = f,
            .cr => row.c[0] = f,
            .cg => row.c[1] = f,
            .cb => row.c[2] = f,
            .ca => row.c[3] = f,
        }
    }
}

fn readFaceAscii(
    c: *AsciiCursor,
    allocator: std.mem.Allocator,
    props: []const Prop,
    list_idx: usize,
    vcount: usize,
    face_buf: *std.ArrayListUnmanaged(u32),
) !void {
    for (props, 0..) |p, i| {
        if (i == list_idx) {
            const n = try nextAsciiInt(c);
            if (n < 0) return error.InvalidPly;
            face_buf.clearRetainingCapacity();
            var k: i64 = 0;
            while (k < n) : (k += 1) {
                const v = try nextAsciiInt(c);
                if (v < 0 or @as(usize, @intCast(v)) >= vcount) return error.InvalidPly;
                try face_buf.append(allocator, @intCast(v));
            }
        } else if (p.is_list) {
            const n = try nextAsciiInt(c);
            if (n < 0) return error.InvalidPly;
            var k: i64 = 0;
            while (k < n) : (k += 1) {
                _ = c.next() orelse return error.TruncatedPly;
            }
        } else {
            _ = c.next() orelse return error.TruncatedPly;
        }
    }
}

fn readFaceBinary(
    c: *BinCursor,
    allocator: std.mem.Allocator,
    props: []const Prop,
    list_idx: usize,
    vcount: usize,
    face_buf: *std.ArrayListUnmanaged(u32),
) !void {
    for (props, 0..) |p, i| {
        if (i == list_idx) {
            const raw_count = try c.take(scalarByteSize(p.count));
            const n = try readScalarU64(p.count, raw_count, c.endian);
            face_buf.clearRetainingCapacity();
            var k: u64 = 0;
            while (k < n) : (k += 1) {
                const raw = try c.take(scalarByteSize(p.scalar));
                const v = try readScalarU64(p.scalar, raw, c.endian);
                if (v >= vcount) return error.InvalidPly;
                try face_buf.append(allocator, @intCast(v));
            }
        } else if (p.is_list) {
            const raw_count = try c.take(scalarByteSize(p.count));
            const n = try readScalarU64(p.count, raw_count, c.endian);
            const item = scalarByteSize(p.scalar);
            if (n > @as(u64, @intCast((c.s.len - c.pos) / item))) return error.TruncatedPly;
            c.pos += @as(usize, @intCast(n)) * item;
        } else {
            _ = try c.take(scalarByteSize(p.scalar));
        }
    }
}

fn skipRowAscii(c: *AsciiCursor, props: []const Prop) !void {
    for (props) |p| {
        if (p.is_list) {
            const n = try nextAsciiInt(c);
            if (n < 0) return error.InvalidPly;
            var k: i64 = 0;
            while (k < n) : (k += 1) {
                _ = c.next() orelse return error.TruncatedPly;
            }
        } else {
            _ = c.next() orelse return error.TruncatedPly;
        }
    }
}

fn skipRowBinary(c: *BinCursor, props: []const Prop) !void {
    for (props) |p| {
        if (p.is_list) {
            const raw_count = try c.take(scalarByteSize(p.count));
            const n = try readScalarU64(p.count, raw_count, c.endian);
            const item = scalarByteSize(p.scalar);
            if (n > @as(u64, @intCast((c.s.len - c.pos) / item))) return error.TruncatedPly;
            c.pos += @as(usize, @intCast(n)) * item;
        } else {
            _ = try c.take(scalarByteSize(p.scalar));
        }
    }
}

fn appendVertRow(
    allocator: std.mem.Allocator,
    out_pos: *std.ArrayListUnmanaged(f32),
    out_nrm: *std.ArrayListUnmanaged(f32),
    out_uv: *std.ArrayListUnmanaged(f32),
    out_col: *std.ArrayListUnmanaged(f32),
    row: VertRow,
    has_col: bool,
    col_scale: *const [4]f32,
) !void {
    try out_pos.appendSlice(allocator, &row.p);
    try out_nrm.appendSlice(allocator, &row.n);
    try out_uv.appendSlice(allocator, &row.uv);
    if (has_col) {
        const cc = [4]f32{
            std.math.clamp(row.c[0] * col_scale[0], 0, 1),
            std.math.clamp(row.c[1] * col_scale[1], 0, 1),
            std.math.clamp(row.c[2] * col_scale[2], 0, 1),
            std.math.clamp(row.c[3] * col_scale[3], 0, 1),
        };
        try out_col.appendSlice(allocator, &cc);
    } else {
        try out_col.appendSlice(allocator, &[_]f32{ 1, 1, 1, 1 });
    }
}

fn fanTriangulate(allocator: std.mem.Allocator, out_idx: *std.ArrayListUnmanaged(u32), face: []const u32) !void {
    if (face.len < 3) return;
    var k: usize = 1;
    while (k + 1 < face.len) : (k += 1) {
        try out_idx.appendSlice(allocator, &[_]u32{ face[0], face[k], face[k + 1] });
    }
}

/// Parses PLY (ASCII, binary little/big endian). Errors: InvalidPly (bad
/// magic, malformed header or rows, missing x/y/z, bad face indices, no
/// triangles), UnsupportedPlyFormat (unknown format name or version),
/// TruncatedPly (header cut before end_header or data shorter than declared),
/// plus allocator errors.
pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !PlyData {
    if (bytes.len == 0) return error.InvalidPly;
    var header = try parseHeader(allocator, bytes);
    defer header.deinit(allocator);
    const elems = header.elements.items;

    var vi: ?usize = null;
    for (elems, 0..) |e, i| {
        if (std.mem.eql(u8, e.name, "vertex")) {
            vi = i;
            break;
        }
    }
    const v_idx = vi orelse return error.InvalidPly;
    const vcount = elems[v_idx].count;
    if (vcount == 0) return error.InvalidPly;

    // Classify vertex properties; only x/y/z are mandatory.
    const vprops = elems[v_idx].props.items;
    const roles = try allocator.alloc(VertRole, vprops.len);
    defer allocator.free(roles);
    @memset(roles, .skip);
    roles[findProp(vprops, &.{"x"}) orelse return error.InvalidPly] = .px;
    roles[findProp(vprops, &.{"y"}) orelse return error.InvalidPly] = .py;
    roles[findProp(vprops, &.{"z"}) orelse return error.InvalidPly] = .pz;
    var has_normals = false;
    if (findProp(vprops, &.{"nx"})) |ix| {
        const iy = findProp(vprops, &.{"ny"}) orelse return error.InvalidPly;
        const iz = findProp(vprops, &.{"nz"}) orelse return error.InvalidPly;
        roles[ix] = .nx;
        roles[iy] = .ny;
        roles[iz] = .nz;
        has_normals = true;
    }
    var has_uvs = false;
    const iu = findProp(vprops, &.{ "u", "s", "texture_u" });
    const iv = findProp(vprops, &.{ "v", "t", "texture_v" });
    if (iu != null and iv != null) {
        roles[iu.?] = .uu;
        roles[iv.?] = .vv;
        has_uvs = true;
    }
    var has_colors = false;
    var col_scale = [4]f32{ 1, 1, 1, 1 };
    const icr = findProp(vprops, &.{ "red", "r", "diffuse_red" });
    const icg = findProp(vprops, &.{ "green", "g", "diffuse_green" });
    const icb = findProp(vprops, &.{ "blue", "b", "diffuse_blue" });
    if (icr != null and icg != null and icb != null) {
        roles[icr.?] = .cr;
        roles[icg.?] = .cg;
        roles[icb.?] = .cb;
        col_scale[0] = colorScale(vprops[icr.?].scalar);
        col_scale[1] = colorScale(vprops[icg.?].scalar);
        col_scale[2] = colorScale(vprops[icb.?].scalar);
        if (findProp(vprops, &.{"alpha"}) orelse findProp(vprops, &.{"a"})) |ia| {
            roles[ia] = .ca;
            col_scale[3] = colorScale(vprops[ia].scalar);
        }
        has_colors = true;
    }

    // Face section: first element named "face" with a vertex_indices list.
    var fi: ?usize = null;
    for (elems, 0..) |e, i| {
        if (std.mem.eql(u8, e.name, "face")) {
            fi = i;
            break;
        }
    }
    var face_list_prop: ?usize = null;
    if (fi) |fidx| {
        const fprops = elems[fidx].props.items;
        face_list_prop = findProp(fprops, &.{"vertex_indices"}) orelse findProp(fprops, &.{"vertex_index"});
        if (face_list_prop) |li| {
            if (!fprops[li].is_list) return error.InvalidPly;
        } else return error.InvalidPly;
    }

    var out_pos: std.ArrayListUnmanaged(f32) = .empty;
    errdefer out_pos.deinit(allocator);
    var out_nrm: std.ArrayListUnmanaged(f32) = .empty;
    errdefer out_nrm.deinit(allocator);
    var out_uv: std.ArrayListUnmanaged(f32) = .empty;
    errdefer out_uv.deinit(allocator);
    var out_col: std.ArrayListUnmanaged(f32) = .empty;
    errdefer out_col.deinit(allocator);
    var out_idx: std.ArrayListUnmanaged(u32) = .empty;
    errdefer out_idx.deinit(allocator);
    var face_buf: std.ArrayListUnmanaged(u32) = .empty;
    defer face_buf.deinit(allocator);

    if (header.format == .ascii) {
        var ac = AsciiCursor{ .s = bytes[header.data_off..] };
        for (elems, 0..) |e, i| {
            if (i == v_idx) {
                var r: usize = 0;
                while (r < e.count) : (r += 1) {
                    var row: VertRow = .{};
                    try readVertexAscii(&ac, vprops, roles, &row);
                    try appendVertRow(allocator, &out_pos, &out_nrm, &out_uv, &out_col, row, has_colors, &col_scale);
                }
            } else if (fi != null and i == fi.?) {
                var r: usize = 0;
                while (r < e.count) : (r += 1) {
                    try readFaceAscii(&ac, allocator, e.props.items, face_list_prop.?, vcount, &face_buf);
                    try fanTriangulate(allocator, &out_idx, face_buf.items);
                }
            } else {
                var r: usize = 0;
                while (r < e.count) : (r += 1) try skipRowAscii(&ac, e.props.items);
            }
        }
    } else {
        const endian: std.builtin.Endian = if (header.format == .little) .little else .big;
        var bc = BinCursor{ .s = bytes[header.data_off..], .endian = endian };
        for (elems, 0..) |e, i| {
            if (i == v_idx) {
                var r: usize = 0;
                while (r < e.count) : (r += 1) {
                    var row: VertRow = .{};
                    try readVertexBinary(&bc, vprops, roles, &row);
                    try appendVertRow(allocator, &out_pos, &out_nrm, &out_uv, &out_col, row, has_colors, &col_scale);
                }
            } else if (fi != null and i == fi.?) {
                var r: usize = 0;
                while (r < e.count) : (r += 1) {
                    try readFaceBinary(&bc, allocator, e.props.items, face_list_prop.?, vcount, &face_buf);
                    try fanTriangulate(allocator, &out_idx, face_buf.items);
                }
            } else {
                var r: usize = 0;
                while (r < e.count) : (r += 1) try skipRowBinary(&bc, e.props.items);
            }
        }
    }

    if (out_idx.items.len == 0) {
        return error.InvalidPly;
    }

    if (!has_normals) {
        // Area-weighted smooth normals, same accumulation as obj.zig.
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
                out_nrm.items[idx * 3] += fn_.x;
                out_nrm.items[idx * 3 + 1] += fn_.y;
                out_nrm.items[idx * 3 + 2] += fn_.z;
            }
        }
        for (0..out_pos.items.len / 3) |i| {
            const n = Vec3.new(out_nrm.items[i * 3], out_nrm.items[i * 3 + 1], out_nrm.items[i * 3 + 2]);
            const fixed = if (n.lengthSq() > 1e-12) n.normalize() else Vec3.up;
            out_nrm.items[i * 3] = fixed.x;
            out_nrm.items[i * 3 + 1] = fixed.y;
            out_nrm.items[i * 3 + 2] = fixed.z;
        }
    }

    var d: PlyData = .{};
    errdefer d.deinit(allocator);
    d.positions = try out_pos.toOwnedSlice(allocator);
    d.normals = try out_nrm.toOwnedSlice(allocator);
    d.uvs = try out_uv.toOwnedSlice(allocator);
    d.colors = try out_col.toOwnedSlice(allocator);
    d.indices = try out_idx.toOwnedSlice(allocator);
    d.has_normals = has_normals;
    d.has_uvs = has_uvs;
    d.has_colors = has_colors;
    return d;
}

/// Uploads parsed PLY as one Mesh (vertex colors when present, else white)
/// and returns the single-element spawn list. The CALLER owns the returned
/// slice and must free it with allocator. No Material is created; use
/// PlyData.averageColor with Scene.createPBRMaterial for an albedo tint.
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
            .color = .{ data.colors[4 * i], data.colors[4 * i + 1], data.colors[4 * i + 2], data.colors[4 * i + 3] },
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

test "ply appendToScene links (type check)" {
    _ = appendToScene;
}
