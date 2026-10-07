//! Flat and revolved parametric surfaces (see `mesh/builders.zig`): plane,
//! torus, torus-knot, and disc plus their option structs. Imports the
//! `common` sibling only; never the `builders.zig` facade.
const std = @import("std");
const math = @import("math");
const Vec2 = math.Vec2;
const Vec3 = math.Vec3;
const Color4 = math.Color4;
const BoundingBox = math.BoundingBox;
const Vertex = @import("../types.zig").Vertex;
const GeometryData = @import("../types.zig").GeometryData;
const common = @import("common.zig");
const appendGridQuad = common.appendGridQuad;
const buildTrigTable = common.buildTrigTable;
const trigEntry = common.trigEntry;
const resolveFrameSeed = common.resolveFrameSeed;

pub const PlaneOptions = struct {
    width: f32 = 1.0,
    height: f32 = 1.0,
    subdivisions_x: u32 = 1, // quads along X (clamped to >= 1)
    subdivisions_y: u32 = 1, // quads along Y (clamped to >= 1)
    uv_scale: Vec2 = Vec2.one, // UV multiplier applied per axis
    color: Color4 = Color4.white,
};

pub fn buildPlaneData(allocator: std.mem.Allocator, options: PlaneOptions) !GeometryData {
    const sx = @max(1, options.subdivisions_x);
    const sy = @max(1, options.subdivisions_y);
    const half_w = options.width * 0.5;
    const half_h = options.height * 0.5;
    const color = options.color.toArray();

    const cols = sx + 1;
    const rows = sy + 1;
    const vertices = try allocator.alloc(Vertex, cols * rows);
    errdefer allocator.free(vertices);
    const indices = try allocator.alloc(u32, sx * sy * 6);
    errdefer allocator.free(indices);

    const sx_f: f32 = @floatFromInt(sx);
    const sy_f: f32 = @floatFromInt(sy);
    var vi: usize = 0;
    for (0..rows) |iy| {
        const fy = @as(f32, @floatFromInt(iy)) / sy_f;
        const y = -half_h + fy * options.height;
        for (0..cols) |ix| {
            const fx = @as(f32, @floatFromInt(ix)) / sx_f;
            const x = -half_w + fx * options.width;
            vertices[vi] = .{
                .position = .{ x, y, 0.0 },
                .normal = .{ 0.0, 0.0, 1.0 },
                .color = color,
                .uv = .{ fx * options.uv_scale.x, fy * options.uv_scale.y },
                .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
            };
            vi += 1;
        }
    }

    var ii: usize = 0;
    for (0..sy) |iy| {
        for (0..sx) |ix| {
            appendGridQuad(indices, ii, cols, iy, ix);
            ii += 6;
        }
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(
            Vec3.new(-half_w, -half_h, 0.0),
            Vec3.new(half_w, half_h, 0.0),
        ),
    };
}

pub const TorusOptions = struct {
    diameter: f32 = 1.0,
    thickness: f32 = 0.5,
    tessellation: u32 = 24, // segments around both the ring and the tube
    color: Color4 = Color4.white,
};

pub fn buildTorusData(allocator: std.mem.Allocator, options: TorusOptions) !GeometryData {
    const tess = @max(3, options.tessellation);
    const ring_radius = @max(0.0, options.diameter) * 0.5;
    const tube_radius = @max(0.0, options.thickness) * 0.5;
    const color = options.color.toArray();

    const side = tess + 1;
    const vertices = try allocator.alloc(Vertex, side * side);
    errdefer allocator.free(vertices);
    const indices = try allocator.alloc(u32, tess * tess * 6);
    errdefer allocator.free(indices);

    const trig = try buildTrigTable(allocator, tess);
    defer allocator.free(trig);
    const ring_tab = trig;
    const tube_tab = trig;

    var vi: usize = 0;
    for (0..side) |j| {
        const cos_u = ring_tab[j].cos;
        const sin_u = ring_tab[j].sin;
        const uu = ring_tab[j].f;
        for (0..side) |i| {
            const cos_v = tube_tab[i].cos;
            const sin_v = tube_tab[i].sin;
            const cx = ring_radius + tube_radius * cos_v;
            vertices[vi] = .{
                .position = .{ cx * cos_u, tube_radius * sin_v, cx * sin_u },
                .normal = .{ cos_v * cos_u, sin_v, cos_v * sin_u },
                .color = color,
                .uv = .{ uu, tube_tab[i].f },
                .tangent = .{ -sin_u, 0.0, cos_u, 1.0 },
            };
            vi += 1;
        }
    }

    var ii: usize = 0;
    for (0..tess) |j| {
        for (0..tess) |i| {
            appendGridQuad(indices, ii, side, j, i);
            ii += 6;
        }
    }

    const extent = ring_radius + tube_radius;
    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(
            Vec3.new(-extent, -tube_radius, -extent),
            Vec3.new(extent, tube_radius, extent),
        ),
    };
}

pub const TorusKnotOptions = struct {
    radius: f32 = 1.0, // overall scale of the knot center curve
    tube: f32 = 0.4,
    radial_segments: u32 = 16, // segments around the tube
    tubular_segments: u32 = 128, // segments along the knot
    p: u32 = 2,
    q: u32 = 3,
    color: Color4 = Color4.white,
};

fn torusKnotCenter(p: u32, q: u32, scale: f32, t: f32) Vec3 {
    const pf = @as(f32, @floatFromInt(p));
    const qf = @as(f32, @floatFromInt(q));
    const k = 2.0 + @cos(qf * t);
    return Vec3.new(
        k * @cos(pf * t) * scale,
        @sin(qf * t) * scale,
        k * @sin(pf * t) * scale,
    );
}

pub fn buildTorusKnotData(allocator: std.mem.Allocator, options: TorusKnotOptions) !GeometryData {
    const radial = @max(3, options.radial_segments);
    const tubular = @max(3, options.tubular_segments);
    const p = @max(1, options.p);
    const q = @max(1, options.q);
    const tube_radius = @max(0.0, options.tube);
    const scale = @max(0.0, options.radius) / 3.0;
    const color = options.color.toArray();

    const ring = radial + 1;
    const rows = tubular + 1;
    const vertices = try allocator.alloc(Vertex, rows * ring);
    errdefer allocator.free(vertices);
    const indices = try allocator.alloc(u32, tubular * radial * 6);
    errdefer allocator.free(indices);

    const pi = std.math.pi;
    const Frame = struct {
        center: Vec3,
        tangent: Vec3,
        n: Vec3,
        b: Vec3,
    };
    const frames = try allocator.alloc(Frame, rows);
    defer allocator.free(frames);
    const tubular_f: f32 = @floatFromInt(tubular);
    for (0..rows) |i| {
        const t = @as(f32, @floatFromInt(i)) / tubular_f * 2.0 * pi;
        frames[i].center = torusKnotCenter(p, q, scale, t);
    }

    for (0..rows) |i| {
        const prev = frames[(i + rows - 1) % rows].center;
        const next = frames[(i + 1) % rows].center;
        const d = next.sub(prev);
        frames[i].tangent = if (d.lengthSq() > 1e-12) d.normalize() else Vec3.forward;
    }

    const ref = resolveFrameSeed(frames[0].tangent, Vec3.up);
    var n0 = ref.sub(frames[0].tangent.scale(frames[0].tangent.dot(ref)));
    if (n0.lengthSq() < 1e-12) n0 = Vec3.forward;
    n0 = n0.normalize();
    frames[0].n = n0;
    frames[0].b = frames[0].tangent.cross(n0);
    for (1..rows) |i| {
        const t = frames[i].tangent;
        var n = frames[i - 1].n.sub(t.scale(t.dot(frames[i - 1].n)));
        if (n.lengthSq() < 1e-12) {
            n = frames[i - 1].n;
        } else {
            n = n.normalize();
        }
        frames[i].n = n;
        frames[i].b = t.cross(n);
    }

    const tube_tab = try buildTrigTable(allocator, radial);
    defer allocator.free(tube_tab);

    var min_v = Vec3.new(std.math.inf(f32), std.math.inf(f32), std.math.inf(f32));
    var max_v = Vec3.new(-std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32));
    var vi: usize = 0;
    for (0..rows) |i| {
        const fr = frames[i];
        const uu = @as(f32, @floatFromInt(i)) / tubular_f;
        for (0..ring) |j| {
            const cos_v = tube_tab[j].cos;
            const sin_v = tube_tab[j].sin;
            const offset = fr.n.scale(cos_v).add(fr.b.scale(sin_v));
            const pos = fr.center.add(offset.scale(tube_radius));
            vertices[vi] = .{
                .position = pos.toArray(),
                .normal = offset.toArray(),
                .color = color,
                .uv = .{ uu, tube_tab[j].f },
                .tangent = .{ fr.tangent.x, fr.tangent.y, fr.tangent.z, 1.0 },
            };
            min_v = Vec3.new(@min(min_v.x, pos.x), @min(min_v.y, pos.y), @min(min_v.z, pos.z));
            max_v = Vec3.new(@max(max_v.x, pos.x), @max(max_v.y, pos.y), @max(max_v.z, pos.z));
            vi += 1;
        }
    }

    var ii: usize = 0;
    for (0..tubular) |i| {
        for (0..radial) |j| {
            appendGridQuad(indices, ii, ring, i, j);
            ii += 6;
        }
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(min_v, max_v),
    };
}

pub const DiscOptions = struct {
    radius: f32 = 0.5,
    tessellation: u32 = 32, // segments around the rim
    color: Color4 = Color4.white,
};

pub fn buildDiscData(allocator: std.mem.Allocator, options: DiscOptions) !GeometryData {
    const tess = @max(3, options.tessellation);
    const radius = @max(0.0, options.radius);
    const color = options.color.toArray();

    const vertices = try allocator.alloc(Vertex, tess + 2);
    errdefer allocator.free(vertices);
    const indices = try allocator.alloc(u32, tess * 3);
    errdefer allocator.free(indices);

    vertices[0] = .{
        .position = .{ 0.0, 0.0, 0.0 },
        .normal = .{ 0.0, 1.0, 0.0 },
        .color = color,
        .uv = .{ 0.5, 0.5 },
        .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
    };

    for (0..tess + 1) |j| {
        const e = trigEntry(tess, j);
        const x = e.sin;
        const z = e.cos;
        vertices[1 + j] = .{
            .position = .{ x * radius, 0.0, z * radius },
            .normal = .{ 0.0, 1.0, 0.0 },
            .color = color,
            .uv = .{ 0.5 + x * 0.5, 0.5 + z * 0.5 },
            .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
        };
    }

    for (0..tess) |j| {
        indices[j * 3 + 0] = 0;
        indices[j * 3 + 1] = @intCast(1 + j);
        indices[j * 3 + 2] = @intCast(1 + j + 1);
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(
            Vec3.new(-radius, 0.0, -radius),
            Vec3.new(radius, 0.0, radius),
        ),
    };
}
