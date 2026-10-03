const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Vec2 = math.Vec2;
const Color4 = math.Color4;
const Mat4 = math.Mat4;
const BoundingBox = math.BoundingBox;

const types = @import("types.zig");
const Vertex = types.Vertex;
const GeometryData = types.GeometryData;
const Mesh = @import("mesh.zig").Mesh;
const uploadGeometry = @import("mesh.zig").uploadGeometry;
const Scene = @import("../scene.zig").Scene;
const tangents = @import("tangents.zig");
const builders = @import("builders.zig");

/// Vertex structure representing position, normal, UV coordinates, and color for CSG operations.
pub const CSGVertex = struct {
    pos: Vec3,
    normal: Vec3,
    uv: Vec2 = Vec2.new(0.0, 0.0),
    uv1: Vec2 = Vec2.new(0.0, 0.0),
    color: Color4 = Color4.new(1.0, 1.0, 1.0, 1.0),

    pub fn interpolate(self: CSGVertex, other: CSGVertex, t: f32) CSGVertex {
        const clamped_t = std.math.clamp(t, 0.0, 1.0);
        var norm = self.normal.lerp(other.normal, clamped_t);
        const norm_len = norm.length();
        if (norm_len > 1e-6) {
            norm = norm.scale(1.0 / norm_len);
        } else {
            norm = self.normal;
        }
        return .{
            .pos = self.pos.lerp(other.pos, clamped_t),
            .normal = norm,
            .uv = Vec2.new(
                self.uv.x + (other.uv.x - self.uv.x) * clamped_t,
                self.uv.y + (other.uv.y - self.uv.y) * clamped_t,
            ),
            .uv1 = Vec2.new(
                self.uv1.x + (other.uv1.x - self.uv1.x) * clamped_t,
                self.uv1.y + (other.uv1.y - self.uv1.y) * clamped_t,
            ),
            .color = Color4.new(
                self.color.r + (other.color.r - self.color.r) * clamped_t,
                self.color.g + (other.color.g - self.color.g) * clamped_t,
                self.color.b + (other.color.b - self.color.b) * clamped_t,
                self.color.a + (other.color.a - self.color.a) * clamped_t,
            ),
        };
    }

    pub fn flip(self: *CSGVertex) void {
        self.normal = Vec3.new(-self.normal.x, -self.normal.y, -self.normal.z);
    }
};

/// Partitioning plane with distance equation: normal . point - w = 0.
pub const CSGPlane = struct {
    normal: Vec3,
    w: f32,

    pub const EPSILON: f32 = 1e-5;

    pub fn fromPoints(a: Vec3, b: Vec3, c: Vec3) CSGPlane {
        const edge1 = b.sub(a);
        const edge2 = c.sub(a);
        var n = edge1.cross(edge2);
        const len = n.length();
        if (len > 1e-6) {
            n = n.scale(1.0 / len);
        } else {
            n = Vec3.new(0.0, 1.0, 0.0);
        }
        return .{
            .normal = n,
            .w = n.dot(a),
        };
    }

    pub fn flip(self: *CSGPlane) void {
        self.normal = Vec3.new(-self.normal.x, -self.normal.y, -self.normal.z);
        self.w = -self.w;
    }

    pub fn splitPolygon(
        self: CSGPlane,
        allocator: std.mem.Allocator,
        polygon: CSGPolygon,
        coplanar_front: *std.ArrayList(CSGPolygon),
        coplanar_back: *std.ArrayList(CSGPolygon),
        front: *std.ArrayList(CSGPolygon),
        back: *std.ArrayList(CSGPolygon),
    ) !void {
        const COPLANAR: u8 = 0;
        const FRONT: u8 = 1;
        const BACK: u8 = 2;
        const SPANNING: u8 = 3;

        var polygon_type: u8 = 0;
        var types_buf: [64]u8 = undefined;
        const vertex_types: []u8 = if (polygon.vertices.len <= types_buf.len)
            types_buf[0..polygon.vertices.len]
        else
            try allocator.alloc(u8, polygon.vertices.len);
        defer if (polygon.vertices.len > types_buf.len) allocator.free(vertex_types);

        for (polygon.vertices, 0..) |v, i| {
            const dist = self.normal.dot(v.pos) - self.w;
            const t: u8 = if (dist < -EPSILON)
                BACK
            else if (dist > EPSILON)
                FRONT
            else
                COPLANAR;
            polygon_type |= t;
            vertex_types[i] = t;
        }

        switch (polygon_type) {
            COPLANAR => {
                const dot = self.normal.dot(polygon.plane.normal);
                if (dot > 0.0) {
                    try coplanar_front.append(allocator, try polygon.clone(allocator));
                } else {
                    try coplanar_back.append(allocator, try polygon.clone(allocator));
                }
            },
            FRONT => {
                try front.append(allocator, try polygon.clone(allocator));
            },
            BACK => {
                try back.append(allocator, try polygon.clone(allocator));
            },
            SPANNING => {
                var f_buf: [16]CSGVertex = undefined;
                var b_buf: [16]CSGVertex = undefined;
                var f_count: usize = 0;
                var b_count: usize = 0;

                const n = polygon.vertices.len;
                for (0..n) |i| {
                    const j = (i + 1) % n;
                    const ti = vertex_types[i];
                    const tj = vertex_types[j];
                    const vi = polygon.vertices[i];
                    const vj = polygon.vertices[j];

                    if (ti != BACK) {
                        if (f_count < f_buf.len) {
                            f_buf[f_count] = vi;
                            f_count += 1;
                        }
                    }
                    if (ti != FRONT) {
                        if (b_count < b_buf.len) {
                            b_buf[b_count] = vi;
                            b_count += 1;
                        }
                    }

                    if ((ti | tj) == SPANNING) {
                        const denom = self.normal.dot(vj.pos.sub(vi.pos));
                        const t = if (@abs(denom) > 1e-7)
                            (self.w - self.normal.dot(vi.pos)) / denom
                        else
                            0.5;
                        const v = vi.interpolate(vj, t);
                        if (f_count < f_buf.len) {
                            f_buf[f_count] = v;
                            f_count += 1;
                        }
                        if (b_count < b_buf.len) {
                            b_buf[b_count] = v;
                            b_count += 1;
                        }
                    }
                }

                if (f_count >= 3) {
                    try front.append(allocator, try CSGPolygon.initWithPlane(allocator, f_buf[0..f_count], polygon.plane));
                }
                if (b_count >= 3) {
                    try back.append(allocator, try CSGPolygon.initWithPlane(allocator, b_buf[0..b_count], polygon.plane));
                }
            },
            else => unreachable,
        }
    }
};

/// Convex polygon composed of vertices and an embedding plane.
pub const CSGPolygon = struct {
    vertices: []CSGVertex,
    plane: CSGPlane,

    pub fn init(allocator: std.mem.Allocator, vertices: []const CSGVertex) !CSGPolygon {
        const copy = try allocator.alloc(CSGVertex, vertices.len);
        @memcpy(copy, vertices);
        const plane = if (vertices.len >= 3)
            CSGPlane.fromPoints(vertices[0].pos, vertices[1].pos, vertices[2].pos)
        else
            CSGPlane{ .normal = Vec3.new(0.0, 1.0, 0.0), .w = 0.0 };
        return .{
            .vertices = copy,
            .plane = plane,
        };
    }

    pub fn initWithPlane(allocator: std.mem.Allocator, vertices: []const CSGVertex, plane: CSGPlane) !CSGPolygon {
        const copy = try allocator.alloc(CSGVertex, vertices.len);
        @memcpy(copy, vertices);
        return .{
            .vertices = copy,
            .plane = plane,
        };
    }

    pub fn clone(self: CSGPolygon, allocator: std.mem.Allocator) !CSGPolygon {
        const copy = try allocator.alloc(CSGVertex, self.vertices.len);
        @memcpy(copy, self.vertices);
        return .{
            .vertices = copy,
            .plane = self.plane,
        };
    }

    pub fn deinit(self: *CSGPolygon, allocator: std.mem.Allocator) void {
        if (self.vertices.len > 0) {
            allocator.free(self.vertices);
            self.vertices = &.{};
        }
    }

    pub fn flip(self: *CSGPolygon) void {
        std.mem.reverse(CSGVertex, self.vertices);
        for (self.vertices) |*v| {
            v.flip();
        }
        self.plane.flip();
    }
};

/// Node in a Binary Space Partitioning (BSP) tree representing a 3D solid.
pub const CSGNode = struct {
    plane: ?CSGPlane = null,
    front: ?*CSGNode = null,
    back: ?*CSGNode = null,
    polygons: std.ArrayList(CSGPolygon),

    pub fn init(allocator: std.mem.Allocator) CSGNode {
        _ = allocator;
        return .{
            .plane = null,
            .front = null,
            .back = null,
            .polygons = .empty,
        };
    }

    pub fn invert(self: *CSGNode) void {
        if (self.plane) |*p| {
            p.flip();
        }
        for (self.polygons.items) |*poly| {
            poly.flip();
        }
        if (self.front) |f| f.invert();
        if (self.back) |b| b.invert();
        const tmp = self.front;
        self.front = self.back;
        self.back = tmp;
    }

    pub fn clipPolygons(self: *const CSGNode, allocator: std.mem.Allocator, list: []const CSGPolygon) anyerror![]CSGPolygon {
        if (self.plane == null) {
            var result = try allocator.alloc(CSGPolygon, list.len);
            for (list, 0..) |p, i| {
                result[i] = try p.clone(allocator);
            }
            return result;
        }

        const p = self.plane.?;
        var front_list: std.ArrayList(CSGPolygon) = .empty;
        try front_list.ensureTotalCapacity(allocator, list.len);
        var back_list: std.ArrayList(CSGPolygon) = .empty;
        try back_list.ensureTotalCapacity(allocator, list.len);

        for (list) |poly| {
            try p.splitPolygon(allocator, poly, &front_list, &back_list, &front_list, &back_list);
        }

        var clipped_front: []CSGPolygon = undefined;
        if (self.front) |f| {
            clipped_front = try f.clipPolygons(allocator, front_list.items);
        } else {
            clipped_front = try allocator.alloc(CSGPolygon, front_list.items.len);
            for (front_list.items, 0..) |poly, i| {
                clipped_front[i] = try poly.clone(allocator);
            }
        }

        var clipped_back: []CSGPolygon = undefined;
        if (self.back) |b| {
            clipped_back = try b.clipPolygons(allocator, back_list.items);
        } else {
            clipped_back = &.{};
        }

        const total = clipped_front.len + clipped_back.len;
        const result = try allocator.alloc(CSGPolygon, total);
        @memcpy(result[0..clipped_front.len], clipped_front);
        @memcpy(result[clipped_front.len..], clipped_back);
        return result;
    }

    pub fn clipTo(self: *CSGNode, allocator: std.mem.Allocator, bsp: *const CSGNode) anyerror!void {
        const clipped = try bsp.clipPolygons(allocator, self.polygons.items);
        self.polygons.clearRetainingCapacity();
        try self.polygons.appendSlice(allocator, clipped);

        if (self.front) |f| try f.clipTo(allocator, bsp);
        if (self.back) |b| try b.clipTo(allocator, bsp);
    }

    pub fn allPolygons(self: *const CSGNode, allocator: std.mem.Allocator, out: *std.ArrayList(CSGPolygon)) anyerror!void {
        for (self.polygons.items) |poly| {
            try out.append(allocator, try poly.clone(allocator));
        }
        if (self.front) |f| try f.allPolygons(allocator, out);
        if (self.back) |b| try b.allPolygons(allocator, out);
    }

    pub fn build(self: *CSGNode, allocator: std.mem.Allocator, list: []const CSGPolygon) anyerror!void {
        if (list.len == 0) return;

        if (self.plane == null) {
            self.plane = list[list.len / 2].plane;
        }

        const p = self.plane.?;
        var front_list: std.ArrayList(CSGPolygon) = .empty;
        try front_list.ensureTotalCapacity(allocator, list.len / 2 + 2);
        var back_list: std.ArrayList(CSGPolygon) = .empty;
        try back_list.ensureTotalCapacity(allocator, list.len / 2 + 2);

        for (list) |poly| {
            try p.splitPolygon(allocator, poly, &self.polygons, &self.polygons, &front_list, &back_list);
        }

        if (front_list.items.len > 0) {
            if (self.front == null) {
                const f = try allocator.create(CSGNode);
                f.* = CSGNode.init(allocator);
                self.front = f;
            }
            try self.front.?.build(allocator, front_list.items);
        }

        if (back_list.items.len > 0) {
            if (self.back == null) {
                const b = try allocator.create(CSGNode);
                b.* = CSGNode.init(allocator);
                self.back = b;
            }
            try self.back.?.build(allocator, back_list.items);
        }
    }

    pub fn fromPolygons(allocator: std.mem.Allocator, list: []const CSGPolygon) anyerror!*CSGNode {
        const node = try allocator.create(CSGNode);
        node.* = CSGNode.init(allocator);
        try node.build(allocator, list);
        return node;
    }
};

/// Constructive Solid Geometry container providing boolean operations:
/// Union, Subtract, Intersect.
pub const CSG = struct {
    allocator: std.mem.Allocator,
    polygons: std.ArrayList(CSGPolygon),

    pub fn init(allocator: std.mem.Allocator) CSG {
        return .{
            .allocator = allocator,
            .polygons = .empty,
        };
    }

    pub fn deinit(self: *CSG) void {
        for (self.polygons.items) |*poly| {
            poly.deinit(self.allocator);
        }
        self.polygons.deinit(self.allocator);
    }

    pub fn clone(self: *const CSG) !CSG {
        var copy = CSG.init(self.allocator);
        try copy.polygons.ensureTotalCapacity(self.allocator, self.polygons.items.len);
        for (self.polygons.items) |poly| {
            try copy.polygons.append(self.allocator, try poly.clone(self.allocator));
        }
        return copy;
    }

    /// Computes the axis-aligned bounding box enclosing this CSG solid.
    pub fn computeBounds(self: *const CSG) BoundingBox {
        var min_p = Vec3.new(std.math.inf(f32), std.math.inf(f32), std.math.inf(f32));
        var max_p = Vec3.new(-std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32));
        if (self.polygons.items.len == 0) return BoundingBox.zero;
        for (self.polygons.items) |poly| {
            for (poly.vertices) |v| {
                min_p.x = @min(min_p.x, v.pos.x);
                min_p.y = @min(min_p.y, v.pos.y);
                min_p.z = @min(min_p.z, v.pos.z);
                max_p.x = @max(max_p.x, v.pos.x);
                max_p.y = @max(max_p.y, v.pos.y);
                max_p.z = @max(max_p.z, v.pos.z);
            }
        }
        return BoundingBox.init(min_p, max_p);
    }

    /// Computes the union of two solids (A ∪ B).
    pub fn unionWith(self: *const CSG, other: *const CSG) !CSG {
        if (self.polygons.items.len == 0) return other.clone();
        if (other.polygons.items.len == 0) return self.clone();

        const box_a = self.computeBounds();
        const box_b = other.computeBounds();
        if (!box_a.intersects(box_b)) {
            // Disjoint solids: simply concatenate without clipping
            var result = CSG.init(self.allocator);
            try result.polygons.ensureTotalCapacity(self.allocator, self.polygons.items.len + other.polygons.items.len);
            for (self.polygons.items) |p| try result.polygons.append(self.allocator, try p.clone(self.allocator));
            for (other.polygons.items) |p| try result.polygons.append(self.allocator, try p.clone(self.allocator));
            return result;
        }

        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const tmp = arena.allocator();

        const a = try CSGNode.fromPolygons(tmp, self.polygons.items);
        const b = try CSGNode.fromPolygons(tmp, other.polygons.items);

        try a.clipTo(tmp, b);
        try b.clipTo(tmp, a);
        b.invert();
        try b.clipTo(tmp, a);
        b.invert();

        var b_polys: std.ArrayList(CSGPolygon) = .empty;
        try b.allPolygons(tmp, &b_polys);
        try a.build(tmp, b_polys.items);

        var out_polys: std.ArrayList(CSGPolygon) = .empty;
        try a.allPolygons(tmp, &out_polys);

        var result = CSG.init(self.allocator);
        try result.polygons.ensureTotalCapacity(self.allocator, out_polys.items.len);
        for (out_polys.items) |p| {
            try result.polygons.append(self.allocator, try p.clone(self.allocator));
        }
        return result;
    }

    /// Computes the difference of two solids (A - B).
    pub fn subtract(self: *const CSG, other: *const CSG) !CSG {
        if (self.polygons.items.len == 0) return CSG.init(self.allocator);
        if (other.polygons.items.len == 0) return self.clone();

        const box_a = self.computeBounds();
        const box_b = other.computeBounds();
        if (!box_a.intersects(box_b)) {
            // Disjoint solids: subtracting B leaves A completely unaffected
            return self.clone();
        }

        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const tmp = arena.allocator();

        const a = try CSGNode.fromPolygons(tmp, self.polygons.items);
        const b = try CSGNode.fromPolygons(tmp, other.polygons.items);

        a.invert();
        try a.clipTo(tmp, b);
        try b.clipTo(tmp, a);
        b.invert();
        try b.clipTo(tmp, a);
        b.invert();

        var b_polys: std.ArrayList(CSGPolygon) = .empty;
        try b.allPolygons(tmp, &b_polys);
        try a.build(tmp, b_polys.items);
        a.invert();

        var out_polys: std.ArrayList(CSGPolygon) = .empty;
        try a.allPolygons(tmp, &out_polys);

        var result = CSG.init(self.allocator);
        try result.polygons.ensureTotalCapacity(self.allocator, out_polys.items.len);
        for (out_polys.items) |p| {
            try result.polygons.append(self.allocator, try p.clone(self.allocator));
        }
        return result;
    }

    /// Computes the intersection of two solids (A ∩ B).
    pub fn intersect(self: *const CSG, other: *const CSG) !CSG {
        if (self.polygons.items.len == 0 or other.polygons.items.len == 0) {
            return CSG.init(self.allocator);
        }

        const box_a = self.computeBounds();
        const box_b = other.computeBounds();
        if (!box_a.intersects(box_b)) {
            // Disjoint solids: intersection is empty
            return CSG.init(self.allocator);
        }

        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const tmp = arena.allocator();

        const a = try CSGNode.fromPolygons(tmp, self.polygons.items);
        const b = try CSGNode.fromPolygons(tmp, other.polygons.items);

        a.invert();
        try b.clipTo(tmp, a);
        b.invert();
        try a.clipTo(tmp, b);
        try b.clipTo(tmp, a);

        var b_polys: std.ArrayList(CSGPolygon) = .empty;
        try b.allPolygons(tmp, &b_polys);
        try a.build(tmp, b_polys.items);
        a.invert();

        var out_polys: std.ArrayList(CSGPolygon) = .empty;
        try a.allPolygons(tmp, &out_polys);

        var result = CSG.init(self.allocator);
        try result.polygons.ensureTotalCapacity(self.allocator, out_polys.items.len);
        for (out_polys.items) |p| {
            try result.polygons.append(self.allocator, try p.clone(self.allocator));
        }
        return result;
    }

    /// Creates a CSG object from GeometryData with an optional transformation matrix.
    pub fn fromGeometryData(allocator: std.mem.Allocator, data: GeometryData, transform: ?Mat4) !CSG {
        var csg = CSG.init(allocator);
        errdefer csg.deinit();

        const has_indices = data.indices.len >= 3;
        const tri_count = if (has_indices) data.indices.len / 3 else data.vertices.len / 3;
        try csg.polygons.ensureTotalCapacity(allocator, tri_count);

        for (0..tri_count) |t| {
            const idx0 = if (has_indices) data.indices[t * 3 + 0] else t * 3 + 0;
            const idx1 = if (has_indices) data.indices[t * 3 + 1] else t * 3 + 1;
            const idx2 = if (has_indices) data.indices[t * 3 + 2] else t * 3 + 2;

            if (idx0 >= data.vertices.len or idx1 >= data.vertices.len or idx2 >= data.vertices.len) continue;

            const v0_src = data.vertices[idx0];
            const v1_src = data.vertices[idx1];
            const v2_src = data.vertices[idx2];

            var p0 = Vec3.new(v0_src.position[0], v0_src.position[1], v0_src.position[2]);
            var p1 = Vec3.new(v1_src.position[0], v1_src.position[1], v1_src.position[2]);
            var p2 = Vec3.new(v2_src.position[0], v2_src.position[1], v2_src.position[2]);

            var n0 = Vec3.new(v0_src.normal[0], v0_src.normal[1], v0_src.normal[2]);
            var n1 = Vec3.new(v1_src.normal[0], v1_src.normal[1], v1_src.normal[2]);
            var n2 = Vec3.new(v2_src.normal[0], v2_src.normal[1], v2_src.normal[2]);

            if (transform) |mat| {
                p0 = mat.transformPoint(p0);
                p1 = mat.transformPoint(p1);
                p2 = mat.transformPoint(p2);
                n0 = mat.transformDirection(n0).normalize();
                n1 = mat.transformDirection(n1).normalize();
                n2 = mat.transformDirection(n2).normalize();
            }

            // Verify triangle is non-degenerate
            const edge1 = p1.sub(p0);
            const edge2 = p2.sub(p0);
            if (edge1.cross(edge2).lengthSq() < 1e-10) continue;

            const verts = [3]CSGVertex{
                .{
                    .pos = p0,
                    .normal = n0,
                    .uv = Vec2.new(v0_src.uv[0], v0_src.uv[1]),
                    .uv1 = Vec2.new(v0_src.uv1[0], v0_src.uv1[1]),
                    .color = Color4.new(v0_src.color[0], v0_src.color[1], v0_src.color[2], v0_src.color[3]),
                },
                .{
                    .pos = p1,
                    .normal = n1,
                    .uv = Vec2.new(v1_src.uv[0], v1_src.uv[1]),
                    .uv1 = Vec2.new(v1_src.uv1[0], v1_src.uv1[1]),
                    .color = Color4.new(v1_src.color[0], v1_src.color[1], v1_src.color[2], v1_src.color[3]),
                },
                .{
                    .pos = p2,
                    .normal = n2,
                    .uv = Vec2.new(v2_src.uv[0], v2_src.uv[1]),
                    .uv1 = Vec2.new(v2_src.uv1[0], v2_src.uv1[1]),
                    .color = Color4.new(v2_src.color[0], v2_src.color[1], v2_src.color[2], v2_src.color[3]),
                },
            };

            const poly = try CSGPolygon.init(allocator, &verts);
            try csg.polygons.append(allocator, poly);
        }

        return csg;
    }

    /// Convenient constructor for Box geometry.
    pub fn fromBox(allocator: std.mem.Allocator, options: builders.BoxOptions, transform: ?Mat4) !CSG {
        var data = try builders.buildBoxData(allocator, options);
        defer data.deinit(allocator);
        return fromGeometryData(allocator, data, transform);
    }

    /// Convenient constructor for Sphere geometry.
    pub fn fromSphere(allocator: std.mem.Allocator, options: builders.SphereOptions, transform: ?Mat4) !CSG {
        var data = try builders.buildSphereData(allocator, options);
        defer data.deinit(allocator);
        return fromGeometryData(allocator, data, transform);
    }

    /// Convenient constructor for Cylinder geometry.
    pub fn fromCylinder(allocator: std.mem.Allocator, options: builders.CylinderOptions, transform: ?Mat4) !CSG {
        var data = try builders.buildCylinderData(allocator, options);
        defer data.deinit(allocator);
        return fromGeometryData(allocator, data, transform);
    }

    /// Creates a CSG solid from an existing Mesh using its CPU geometry and world matrix.
    pub fn fromMesh(allocator: std.mem.Allocator, mesh: *const Mesh) !CSG {
        const mat = mesh.computeWorldMatrix();
        if (mesh.morph_base.len > 0 and mesh.cpu_indices.len >= 3) {
            const geom = GeometryData{
                .vertices = mesh.morph_base,
                .indices = mesh.cpu_indices,
                .bounds = mesh.local_bounding_box,
            };
            return fromGeometryData(allocator, geom, mat);
        } else if (mesh.cpu_positions.len >= 3 and mesh.cpu_indices.len >= 3) {
            var verts = try allocator.alloc(Vertex, mesh.cpu_positions.len);
            defer allocator.free(verts);
            for (mesh.cpu_positions, 0..) |p, i| {
                verts[i] = .{
                    .position = p.toArray(),
                    .normal = .{ 0.0, 1.0, 0.0 },
                    .uv = .{ 0.0, 0.0 },
                    // CPU-position fallback genuinely lacks UVs: keep both sets zeroed.
                    .uv1 = .{ 0.0, 0.0 },
                    .color = .{ 1.0, 1.0, 1.0, 1.0 },
                };
            }
            const tri_count = mesh.cpu_indices.len / 3;
            for (0..tri_count) |t| {
                const idx0 = mesh.cpu_indices[t * 3 + 0];
                const idx1 = mesh.cpu_indices[t * 3 + 1];
                const idx2 = mesh.cpu_indices[t * 3 + 2];
                if (idx0 < verts.len and idx1 < verts.len and idx2 < verts.len) {
                    const p0 = mesh.cpu_positions[idx0];
                    const p1 = mesh.cpu_positions[idx1];
                    const p2 = mesh.cpu_positions[idx2];
                    const n = p1.sub(p0).cross(p2.sub(p0)).normalize();
                    verts[idx0].normal = n.toArray();
                    verts[idx1].normal = n.toArray();
                    verts[idx2].normal = n.toArray();
                }
            }
            const geom = GeometryData{
                .vertices = verts,
                .indices = mesh.cpu_indices,
                .bounds = mesh.local_bounding_box,
            };
            return fromGeometryData(allocator, geom, mat);
        } else {
            return error.MeshLacksCpuGeometry;
        }
    }

    /// Converts the CSG solid into triangulated GeometryData with recalculated tangents and bounding box.
    pub fn toGeometryData(self: *const CSG, allocator: std.mem.Allocator) !GeometryData {
        var tri_count: usize = 0;
        for (self.polygons.items) |poly| {
            if (poly.vertices.len >= 3) {
                tri_count += poly.vertices.len - 2;
            }
        }

        const vert_count = tri_count * 3;
        var vertices = try allocator.alloc(Vertex, vert_count);
        errdefer allocator.free(vertices);
        var indices = try allocator.alloc(u32, vert_count);
        errdefer allocator.free(indices);

        var idx: usize = 0;
        var min_p = Vec3.new(std.math.inf(f32), std.math.inf(f32), std.math.inf(f32));
        var max_p = Vec3.new(-std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32));

        for (self.polygons.items) |poly| {
            if (poly.vertices.len < 3) continue;
            const v0 = poly.vertices[0];
            for (1..poly.vertices.len - 1) |i| {
                const v1 = poly.vertices[i];
                const v2 = poly.vertices[i + 1];

                const tri = [3]CSGVertex{ v0, v1, v2 };
                for (tri) |tv| {
                    min_p.x = @min(min_p.x, tv.pos.x);
                    min_p.y = @min(min_p.y, tv.pos.y);
                    min_p.z = @min(min_p.z, tv.pos.z);
                    max_p.x = @max(max_p.x, tv.pos.x);
                    max_p.y = @max(max_p.y, tv.pos.y);
                    max_p.z = @max(max_p.z, tv.pos.z);

                    vertices[idx] = .{
                        .position = tv.pos.toArray(),
                        .normal = tv.normal.toArray(),
                        .uv = .{ tv.uv.x, tv.uv.y },
                        .uv1 = .{ tv.uv1.x, tv.uv1.y },
                        .color = tv.color.toArray(),
                        .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
                    };
                    indices[idx] = @intCast(idx);
                    idx += 1;
                }
            }
        }

        if (vert_count == 0) {
            min_p = Vec3.zero;
            max_p = Vec3.zero;
        }

        tangents.computeTangents(vertices, indices, null);

        return GeometryData{
            .vertices = vertices,
            .indices = indices,
            .bounds = BoundingBox.init(min_p, max_p),
        };
    }

    /// Uploads the CSG solid as a GPU Mesh ready to render in the scene.
    pub fn toMesh(self: *const CSG, scene: *Scene, name: []const u8) !*Mesh {
        var data = try self.toGeometryData(scene.allocator);
        defer data.deinit(scene.allocator);
        return uploadGeometry(scene, name, data);
    }
};

test "CSG: interpolate splits uv and uv1 independently" {
    const a = CSGVertex{
        .pos = Vec3.new(0, 0, 0),
        .normal = Vec3.new(0, 0, 1),
        .uv = Vec2.new(0, 0),
        .uv1 = Vec2.new(5, 7),
    };
    const b = CSGVertex{
        .pos = Vec3.new(1, 0, 0),
        .normal = Vec3.new(0, 0, 1),
        .uv = Vec2.new(1, 1),
        .uv1 = Vec2.new(15, 17),
    };
    const m = a.interpolate(b, 0.5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), m.uv.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), m.uv.y, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), m.uv1.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), m.uv1.y, 1e-6);
}

test "CSG: multiUV uv1 survives fromGeometryData/toGeometryData roundtrip" {
    const ally = std.testing.allocator;
    var quad_verts = [_]Vertex{
        .{ .position = .{ -1, -1, 0 }, .normal = .{ 0, 0, 1 }, .color = .{ 1, 1, 1, 1 }, .uv = .{ 0, 0 }, .uv1 = .{ 5, 7 } },
        .{ .position = .{ 1, -1, 0 }, .normal = .{ 0, 0, 1 }, .color = .{ 1, 1, 1, 1 }, .uv = .{ 1, 0 }, .uv1 = .{ 15, 7 } },
        .{ .position = .{ 1, 1, 0 }, .normal = .{ 0, 0, 1 }, .color = .{ 1, 1, 1, 1 }, .uv = .{ 1, 1 }, .uv1 = .{ 15, 17 } },
        .{ .position = .{ -1, 1, 0 }, .normal = .{ 0, 0, 1 }, .color = .{ 1, 1, 1, 1 }, .uv = .{ 0, 1 }, .uv1 = .{ 5, 17 } },
    };
    var quad_idx = [_]u32{ 0, 1, 2, 0, 2, 3 };
    const quad = GeometryData{
        .vertices = &quad_verts,
        .indices = &quad_idx,
        .bounds = BoundingBox.init(Vec3.new(-1, -1, 0), Vec3.new(1, 1, 0)),
    };
    var csg = try CSG.fromGeometryData(ally, quad, null);
    defer csg.deinit();
    var out = try csg.toGeometryData(ally);
    defer out.deinit(ally);

    try std.testing.expectEqual(@as(usize, 6), out.vertices.len);
    // Fan triangulation preserves the input triangles exactly; each output
    // uv1 must match one of the distinct input uv1 corners.
    const corners = [4][2]f32{ .{ 5, 7 }, .{ 15, 7 }, .{ 15, 17 }, .{ 5, 17 } };
    for (out.vertices) |v| {
        var hit = false;
        for (corners) |c| {
            if (@abs(v.uv1[0] - c[0]) < 1e-5 and @abs(v.uv1[1] - c[1]) < 1e-5) hit = true;
        }
        try std.testing.expect(hit);
        // Distinct from uv0 (uv in [0,1], uv1 in [5,15]x[7,17]).
        try std.testing.expect(v.uv1[0] > 1.0 and v.uv1[1] > 1.0);
    }
}

test "CSG: multiUV uv1 survives real boolean union" {
    const ally = std.testing.allocator;
    var box_a = try builders.buildBoxData(ally, .{ .width = 2.0, .height = 2.0, .depth = 2.0 });
    defer box_a.deinit(ally);
    var box_b = try builders.buildBoxData(ally, .{ .width = 2.0, .height = 2.0, .depth = 2.0 });
    defer box_b.deinit(ally);
    // Paint distinct channels: uv stays builder-provided, uv1 is offset far away.
    for (box_a.vertices) |*v| v.uv1 = .{ v.uv[0] * 10.0 + 5.0, v.uv[1] * 10.0 + 7.0 };
    for (box_b.vertices) |*v| v.uv1 = .{ v.uv[0] * 10.0 + 5.0, v.uv[1] * 10.0 + 7.0 };

    var csg_a = try CSG.fromGeometryData(ally, box_a, null);
    defer csg_a.deinit();
    var csg_b = try CSG.fromGeometryData(ally, box_b, Mat4.translation(Vec3.new(1, 0, 0)));
    defer csg_b.deinit();

    var joined = try csg_a.unionWith(&csg_b);
    defer joined.deinit();
    var out = try joined.toGeometryData(ally);
    defer out.deinit(ally);

    try std.testing.expect(out.vertices.len > 0);
    // Split planes interpolate new verts: every output uv1 must lie within
    // the painted input range and keep the 10x+offset split from uv0.
    var saw_nonzero_uv1 = false;
    for (out.vertices) |v| {
        try std.testing.expect(v.uv1[0] >= 5.0 - 1e-3 and v.uv1[0] <= 15.0 + 1e-3);
        try std.testing.expect(v.uv1[1] >= 7.0 - 1e-3 and v.uv1[1] <= 17.0 + 1e-3);
        try std.testing.expectApproxEqAbs(10.0 * v.uv[0] + 5.0, v.uv1[0], 1e-3);
        try std.testing.expectApproxEqAbs(10.0 * v.uv[1] + 7.0, v.uv1[1], 1e-3);
        if (v.uv1[0] != 0.0 or v.uv1[1] != 0.0) saw_nonzero_uv1 = true;
    }
    try std.testing.expect(saw_nonzero_uv1);
}
