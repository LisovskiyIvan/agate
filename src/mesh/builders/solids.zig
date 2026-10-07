//! Closed primitive solids (see `mesh/builders.zig`): box, ground, terrain,
//! sphere, cylinder, and capsule plus their option structs. Imports the
//! `common` sibling only; never the `builders.zig` facade.
const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Color4 = math.Color4;
const BoundingBox = math.BoundingBox;
const Vertex = @import("../types.zig").Vertex;
const GeometryData = @import("../types.zig").GeometryData;
const common = @import("common.zig");
const storeQuad = common.storeQuad;
const appendGridQuad = common.appendGridQuad;
const appendGridQuadFlipped = common.appendGridQuadFlipped;
const buildTrigTable = common.buildTrigTable;

pub const BoxOptions = struct {
    size: f32 = 1.0,
    width: ?f32 = null,
    height: ?f32 = null,
    depth: ?f32 = null,
    face_colors: ?[6]Color4 = null,
};

pub fn buildBoxData(allocator: std.mem.Allocator, options: BoxOptions) !GeometryData {
    const w = (options.width orelse options.size) * 0.5;
    const h = (options.height orelse options.size) * 0.5;
    const d = (options.depth orelse options.size) * 0.5;

    const colors = options.face_colors orelse [6]Color4{
        Color4.new(1.0, 0.2, 0.2, 1.0), // Front: Red
        Color4.new(0.2, 1.0, 0.2, 1.0), // Back: Green
        Color4.new(0.2, 0.4, 1.0, 1.0), // Left: Blue
        Color4.new(1.0, 0.6, 0.1, 1.0), // Right: Orange
        Color4.new(0.9, 0.2, 0.8, 1.0), // Top: Magenta
        Color4.new(0.2, 0.8, 0.9, 1.0), // Bottom: Cyan
    };

    const vertices = try allocator.alloc(Vertex, 24);
    errdefer allocator.free(vertices);
    const indices = try allocator.alloc(u32, 36);
    errdefer allocator.free(indices);

    const Quad = struct {
        p: [4][3]f32,
        n: [3]f32,
        t: [4]f32,
        c: [4]f32,
    };

    const quads = [_]Quad{
        // Front (+Z)
        .{
            .p = .{ .{ -w, -h, d }, .{ w, -h, d }, .{ w, h, d }, .{ -w, h, d } },
            .n = .{ 0, 0, 1 },
            .t = .{ 1, 0, 0, 1 },
            .c = colors[0].toArray(),
        },
        // Back (-Z)
        .{
            .p = .{ .{ w, -h, -d }, .{ -w, -h, -d }, .{ -w, h, -d }, .{ w, h, -d } },
            .n = .{ 0, 0, -1 },
            .t = .{ -1, 0, 0, 1 },
            .c = colors[1].toArray(),
        },
        // Left (-X)
        .{
            .p = .{ .{ -w, -h, -d }, .{ -w, -h, d }, .{ -w, h, d }, .{ -w, h, -d } },
            .n = .{ -1, 0, 0 },
            .t = .{ 0, 0, 1, 1 },
            .c = colors[2].toArray(),
        },
        // Right (+X)
        .{
            .p = .{ .{ w, -h, d }, .{ w, -h, -d }, .{ w, h, -d }, .{ w, h, d } },
            .n = .{ 1, 0, 0 },
            .t = .{ 0, 0, -1, 1 },
            .c = colors[3].toArray(),
        },
        // Top (+Y)
        .{
            .p = .{ .{ -w, h, d }, .{ w, h, d }, .{ w, h, -d }, .{ -w, h, -d } },
            .n = .{ 0, 1, 0 },
            .t = .{ 1, 0, 0, 1 },
            .c = colors[4].toArray(),
        },
        // Bottom (-Y)
        .{
            .p = .{ .{ -w, -h, -d }, .{ w, -h, -d }, .{ w, -h, d }, .{ -w, -h, d } },
            .n = .{ 0, -1, 0 },
            .t = .{ 1, 0, 0, 1 },
            .c = colors[5].toArray(),
        },
    };

    const uvs = [4][2]f32{
        .{ 0.0, 0.0 },
        .{ 1.0, 0.0 },
        .{ 1.0, 1.0 },
        .{ 0.0, 1.0 },
    };

    var vi: usize = 0;
    for (quads) |q| {
        for (0..4) |i| {
            vertices[vi] = .{
                .position = q.p[i],
                .normal = q.n,
                .color = q.c,
                .uv = uvs[i],
                .tangent = q.t,
            };
            vi += 1;
        }
    }

    var ii: usize = 0;
    for (0..6) |face| {
        const base: u32 = @intCast(face * 4);
        storeQuad(indices, ii, base, base + 1, base + 2, base + 3);
        ii += 6;
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(
            Vec3.new(-w, -h, -d),
            Vec3.new(w, h, d),
        ),
    };
}

pub const GroundOptions = struct {
    width: f32 = 10.0,
    height: f32 = 10.0,
    subdivisions: u32 = 1,
    color: Color4 = Color4.white,
};

pub fn buildGroundData(allocator: std.mem.Allocator, options: GroundOptions) !GeometryData {
    const half_w = options.width * 0.5;
    const half_h = options.height * 0.5;
    const subs = @max(1, options.subdivisions);

    const vert_count = (subs + 1) * (subs + 1);
    const index_count = subs * subs * 6;

    const vertices = try allocator.alloc(Vertex, vert_count);
    errdefer allocator.free(vertices);
    const indices = try allocator.alloc(u32, index_count);
    errdefer allocator.free(indices);

    const color = options.color.toArray();
    var vi: usize = 0;
    for (0..subs + 1) |iz| {
        const fz = @as(f32, @floatFromInt(iz)) / @as(f32, @floatFromInt(subs));
        const z = -half_h + fz * options.height;
        for (0..subs + 1) |ix| {
            const fx = @as(f32, @floatFromInt(ix)) / @as(f32, @floatFromInt(subs));
            const x = -half_w + fx * options.width;
            vertices[vi] = .{
                .position = .{ x, 0.0, z },
                .normal = .{ 0.0, 1.0, 0.0 },
                .color = color,
                .uv = .{ fx, fz },
                .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
            };
            vi += 1;
        }
    }

    var ii: usize = 0;
    const row_stride = subs + 1;
    for (0..subs) |iz| {
        for (0..subs) |ix| {
            appendGridQuadFlipped(indices, ii, row_stride, iz, ix);
            ii += 6;
        }
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(
            Vec3.new(-half_w, 0.0, -half_h),
            Vec3.new(half_w, 0.0, half_h),
        ),
    };
}

pub const TerrainOptions = struct {
    /// x/z cell size and y height multiplier (matches HeightFieldOptions).
    scale: Vec3 = Vec3.new(1.0, 1.0, 1.0),
    color: Color4 = Color4.white,
};

pub fn buildTerrainData(
    allocator: std.mem.Allocator,
    heights: []const f32,
    count_x: u32,
    count_z: u32,
    options: TerrainOptions,
) !GeometryData {
    const expected = @as(usize, count_x) * count_z;
    if (count_x < 2 or count_z < 2 or heights.len != expected) {
        return error.InvalidTerrainDimensions;
    }

    const vert_count = expected;
    const index_count = (count_x - 1) * (count_z - 1) * 6;
    const sx = options.scale.x;
    const sy = options.scale.y;
    const sz = options.scale.z;

    const vertices = try allocator.alloc(Vertex, vert_count);
    errdefer allocator.free(vertices);
    const indices = try allocator.alloc(u32, index_count);
    errdefer allocator.free(indices);

    var min_y: f32 = heights[0] * sy;
    var max_y: f32 = heights[0] * sy;
    const color = options.color.toArray();
    const u_denom: f32 = @floatFromInt(count_x - 1);
    const v_denom: f32 = @floatFromInt(count_z - 1);

    var vi: usize = 0;
    for (0..count_z) |row| {
        for (0..count_x) |col| {
            const x = @as(f32, @floatFromInt(col)) * sx;
            const z = @as(f32, @floatFromInt(row)) * sz;
            const h = heights[row * count_x + col] * sy;
            min_y = @min(min_y, h);
            max_y = @max(max_y, h);

            // Central-difference normal.
            const col_l = if (col > 0) col - 1 else col;
            const col_r = if (col + 1 < count_x) col + 1 else col;
            const row_u = if (row > 0) row - 1 else row;
            const row_d = if (row + 1 < count_z) row + 1 else row;
            const dh_dx = (heights[row * count_x + col_r] - heights[row * count_x + col_l]) * sy;
            const dh_dz = (heights[row_d * count_x + col] - heights[row_u * count_x + col]) * sy;
            const dx = @as(f32, @floatFromInt(col_r - col_l)) * sx;
            const dz = @as(f32, @floatFromInt(row_d - row_u)) * sz;
            const tx = Vec3.new(if (dx > 0.0) dx else 1.0, dh_dx, 0.0);
            const tz = Vec3.new(0.0, dh_dz, if (dz > 0.0) dz else 1.0);
            const normal = Vec3.new(
                tz.y * tx.z - tz.z * tx.y,
                tz.z * tx.x - tz.x * tx.z,
                tz.x * tx.y - tz.y * tx.x,
            ).normalize();

            vertices[vi] = .{
                .position = .{ x, h, z },
                .normal = .{ normal.x, normal.y, normal.z },
                .color = color,
                .uv = .{
                    @as(f32, @floatFromInt(col)) / u_denom,
                    @as(f32, @floatFromInt(row)) / v_denom,
                },
                .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
            };
            vi += 1;
        }
    }

    var ii: usize = 0;
    const row_stride = count_x;
    for (0..count_z - 1) |row| {
        for (0..count_x - 1) |col| {
            appendGridQuadFlipped(indices, ii, row_stride, row, col);
            ii += 6;
        }
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(
            Vec3.new(0.0, min_y, 0.0),
            Vec3.new(@as(f32, @floatFromInt(count_x - 1)) * sx, max_y, @as(f32, @floatFromInt(count_z - 1)) * sz),
        ),
    };
}

pub const SphereOptions = struct {
    diameter: f32 = 1.0,
    segments: u32 = 24,
    color: Color4 = Color4.white,
};

pub fn buildSphereData(allocator: std.mem.Allocator, options: SphereOptions) !GeometryData {
    const radius = options.diameter * 0.5;
    const segs = @max(4, options.segments);
    const rings = segs;
    const slices = segs;

    const vert_count = (rings + 1) * (slices + 1);
    const index_count = rings * slices * 6;

    const vertices = try allocator.alloc(Vertex, vert_count);
    errdefer allocator.free(vertices);
    const indices = try allocator.alloc(u32, index_count);
    errdefer allocator.free(indices);

    const pi = std.math.pi;
    const color = options.color.toArray();
    const slice_tab = try buildTrigTable(allocator, slices);
    defer allocator.free(slice_tab);

    var vi: usize = 0;
    for (0..rings + 1) |r| {
        const v = @as(f32, @floatFromInt(r)) / @as(f32, @floatFromInt(rings));
        const phi = -pi * 0.5 + pi * v;
        const cos_phi = @cos(phi);
        const sin_phi = @sin(phi);

        for (0..slices + 1) |s| {
            const u = slice_tab[s].f;
            const cos_theta = slice_tab[s].cos;
            const sin_theta = slice_tab[s].sin;

            const nx = cos_phi * sin_theta;
            const ny = sin_phi;
            const nz = cos_phi * cos_theta;

            vertices[vi] = .{
                .position = .{ nx * radius, ny * radius, nz * radius },
                .normal = .{ nx, ny, nz },
                .color = color,
                .uv = .{ u, v },
                .tangent = .{ cos_theta, 0.0, -sin_theta, 1.0 },
            };
            vi += 1;
        }
    }

    var ii: usize = 0;
    const slice_stride = slices + 1;
    for (0..rings) |r| {
        for (0..slices) |s| {
            appendGridQuad(indices, ii, slice_stride, r, s);
            ii += 6;
        }
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(
            Vec3.new(-radius, -radius, -radius),
            Vec3.new(radius, radius, radius),
        ),
    };
}

pub const CylinderOptions = struct {
    height: f32 = 2.0,
    diameter: f32 = 1.0,
    tessellation: u32 = 24,
    color: Color4 = Color4.white,
};

pub fn buildCylinderData(allocator: std.mem.Allocator, options: CylinderOptions) !GeometryData {
    const radius = options.diameter * 0.5;
    const half_h = options.height * 0.5;
    const tess = @max(3, options.tessellation);

    const side_verts = (tess + 1) * 2;
    const cap_verts = (tess + 2) * 2;
    const total_verts = side_verts + cap_verts;

    const side_indices = tess * 6;
    const cap_indices = tess * 3 * 2;
    const total_indices = side_indices + cap_indices;

    const vertices = try allocator.alloc(Vertex, total_verts);
    errdefer allocator.free(vertices);
    const indices = try allocator.alloc(u32, total_indices);
    errdefer allocator.free(indices);

    const color = options.color.toArray();
    const ring_tab = try buildTrigTable(allocator, tess);
    defer allocator.free(ring_tab);
    var vi: usize = 0;

    // 1. Sides
    for (0..tess + 1) |i| {
        const u = ring_tab[i].f;
        const nx = ring_tab[i].sin;
        const nz = ring_tab[i].cos;

        // Bottom
        vertices[vi] = .{
            .position = .{ nx * radius, -half_h, nz * radius },
            .normal = .{ nx, 0.0, nz },
            .color = color,
            .uv = .{ u, 0.0 },
            .tangent = .{ nz, 0.0, -nx, 1.0 },
        };
        vi += 1;

        // Top
        vertices[vi] = .{
            .position = .{ nx * radius, half_h, nz * radius },
            .normal = .{ nx, 0.0, nz },
            .color = color,
            .uv = .{ u, 1.0 },
            .tangent = .{ nz, 0.0, -nx, 1.0 },
        };
        vi += 1;
    }

    var ii: usize = 0;
    for (0..tess) |i| {
        const b0: u32 = @intCast(i * 2);
        const t0: u32 = @intCast(i * 2 + 1);
        const b1: u32 = @intCast((i + 1) * 2);
        const t1: u32 = @intCast((i + 1) * 2 + 1);

        storeQuad(indices, ii, b0, b1, t1, t0);
        ii += 6;
    }

    // 2. Top cap
    const top_center_idx: u32 = @intCast(vi);
    vertices[vi] = .{
        .position = .{ 0.0, half_h, 0.0 },
        .normal = .{ 0.0, 1.0, 0.0 },
        .color = color,
        .uv = .{ 0.5, 0.5 },
        .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
    };
    vi += 1;

    const top_ring_start: u32 = @intCast(vi);
    for (0..tess + 1) |i| {
        const x = ring_tab[i].sin;
        const z = ring_tab[i].cos;
        vertices[vi] = .{
            .position = .{ x * radius, half_h, z * radius },
            .normal = .{ 0.0, 1.0, 0.0 },
            .color = color,
            .uv = .{ 0.5 + x * 0.5, 0.5 + z * 0.5 },
            .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
        };
        vi += 1;
    }

    for (0..tess) |i| {
        indices[ii + 0] = top_center_idx;
        indices[ii + 1] = top_ring_start + @as(u32, @intCast(i));
        indices[ii + 2] = top_ring_start + @as(u32, @intCast(i + 1));
        ii += 3;
    }

    // 3. Bottom cap
    const bot_center_idx: u32 = @intCast(vi);
    vertices[vi] = .{
        .position = .{ 0.0, -half_h, 0.0 },
        .normal = .{ 0.0, -1.0, 0.0 },
        .color = color,
        .uv = .{ 0.5, 0.5 },
        .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
    };
    vi += 1;

    const bot_ring_start: u32 = @intCast(vi);
    for (0..tess + 1) |i| {
        const x = ring_tab[i].sin;
        const z = ring_tab[i].cos;
        vertices[vi] = .{
            .position = .{ x * radius, -half_h, z * radius },
            .normal = .{ 0.0, -1.0, 0.0 },
            .color = color,
            .uv = .{ 0.5 + x * 0.5, 0.5 + z * 0.5 },
            .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
        };
        vi += 1;
    }

    for (0..tess) |i| {
        indices[ii + 0] = bot_center_idx;
        indices[ii + 1] = bot_ring_start + @as(u32, @intCast(i + 1));
        indices[ii + 2] = bot_ring_start + @as(u32, @intCast(i));
        ii += 3;
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(
            Vec3.new(-radius, -half_h, -radius),
            Vec3.new(radius, half_h, radius),
        ),
    };
}

pub const CapsuleOptions = struct {
    radius: f32 = 0.5,
    height: f32 = 2.0, // total height including hemisphere caps
    tessellation: u32 = 16, // radial slices
    cap_subdivisions: u32 = 8, // rings per cap
    color: Color4 = Color4.white,
};

pub fn buildCapsuleData(allocator: std.mem.Allocator, options: CapsuleOptions) !GeometryData {
    const radius = options.radius;
    const total_height = @max(options.height, radius * 2.0);
    const half_h = (total_height - radius * 2.0) * 0.5;
    const slices = @max(4, options.tessellation);
    const cap_rings = @max(2, options.cap_subdivisions);

    const total_rings = 2 * cap_rings + 2;
    const vert_count = total_rings * (slices + 1);
    const quad_rows = total_rings - 1;
    const index_count = quad_rows * slices * 6;

    const vertices = try allocator.alloc(Vertex, vert_count);
    errdefer allocator.free(vertices);
    const indices = try allocator.alloc(u32, index_count);
    errdefer allocator.free(indices);

    const pi = std.math.pi;
    const color = options.color.toArray();
    const slice_tab = try buildTrigTable(allocator, slices);
    defer allocator.free(slice_tab);
    var vi: usize = 0;

    for (0..total_rings) |r| {
        var phi: f32 = 0.0;
        var center_y: f32 = 0.0;

        if (r <= cap_rings) {
            // Bottom hemisphere: phi in [-pi/2, 0]
            const t = @as(f32, @floatFromInt(r)) / @as(f32, @floatFromInt(cap_rings));
            phi = -pi * 0.5 + t * (pi * 0.5);
            center_y = -half_h;
        } else if (r == cap_rings + 1) {
            // Top of cylinder body: phi = 0
            phi = 0.0;
            center_y = half_h;
        } else {
            // Top hemisphere: phi in [0, pi/2]
            const t = @as(f32, @floatFromInt(r - cap_rings - 1)) / @as(f32, @floatFromInt(cap_rings));
            phi = t * (pi * 0.5);
            center_y = half_h;
        }

        const cos_phi = @cos(phi);
        const sin_phi = @sin(phi);
        const v = @as(f32, @floatFromInt(r)) / @as(f32, @floatFromInt(quad_rows));

        for (0..slices + 1) |s| {
            const u = slice_tab[s].f;
            const cos_theta = slice_tab[s].cos;
            const sin_theta = slice_tab[s].sin;

            const nx = cos_phi * sin_theta;
            const ny = sin_phi;
            const nz = cos_phi * cos_theta;

            vertices[vi] = .{
                .position = .{ nx * radius, center_y + ny * radius, nz * radius },
                .normal = .{ nx, ny, nz },
                .color = color,
                .uv = .{ u, v },
                .tangent = .{ cos_theta, 0.0, -sin_theta, 1.0 },
            };
            vi += 1;
        }
    }

    var ii: usize = 0;
    const slice_stride: usize = @intCast(slices + 1);
    for (0..quad_rows) |r| {
        for (0..slices) |s| {
            appendGridQuad(indices, ii, slice_stride, r, s);
            ii += 6;
        }
    }

    const total_half = total_height * 0.5;
    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(
            Vec3.new(-radius, -total_half, -radius),
            Vec3.new(radius, total_half, radius),
        ),
    };
}
