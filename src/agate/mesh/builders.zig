const std = @import("std");
const math = @import("math");
const Vec2 = math.Vec2;
const Vec3 = math.Vec3;
const Color4 = math.Color4;
const BoundingBox = math.BoundingBox;

const Vertex = @import("types.zig").Vertex;
const GeometryData = @import("types.zig").GeometryData;
const pickOrthogonal = @import("tangents.zig").pickOrthogonal;
const orthogonal_dot_threshold = @import("tangents.zig").orthogonal_dot_threshold;

pub const BoxOptions = struct {
    size: f32 = 1.0,
    width: ?f32 = null,
    height: ?f32 = null,
    depth: ?f32 = null,
    face_colors: ?[6]Color4 = null,
};

pub const SphereOptions = struct {
    diameter: f32 = 1.0,
    segments: u32 = 24,
    color: Color4 = Color4.white,
};

pub const GroundOptions = struct {
    width: f32 = 10.0,
    height: f32 = 10.0,
    subdivisions: u32 = 1,
    color: Color4 = Color4.white,
};

pub const TerrainOptions = struct {
    /// x/z cell size and y height multiplier (matches HeightFieldOptions).
    scale: Vec3 = Vec3.new(1.0, 1.0, 1.0),
    color: Color4 = Color4.white,
};

pub const CylinderOptions = struct {
    height: f32 = 2.0,
    diameter: f32 = 1.0,
    tessellation: u32 = 24,
    color: Color4 = Color4.white,
};

pub const CapsuleOptions = struct {
    radius: f32 = 0.5,
    height: f32 = 2.0, // total height including hemisphere caps
    tessellation: u32 = 16, // radial slices
    cap_subdivisions: u32 = 8, // rings per cap
    color: Color4 = Color4.white,
};

pub const TorusOptions = struct {
    diameter: f32 = 1.0,
    thickness: f32 = 0.5,
    tessellation: u32 = 24, // segments around both the ring and the tube
    color: Color4 = Color4.white,
};

pub const TorusKnotOptions = struct {
    radius: f32 = 1.0, // overall scale of the knot center curve
    tube: f32 = 0.4,
    radial_segments: u32 = 16, // segments around the tube
    tubular_segments: u32 = 128, // segments along the knot
    p: u32 = 2,
    q: u32 = 3,
    color: Color4 = Color4.white,
};

pub const DiscOptions = struct {
    radius: f32 = 0.5,
    tessellation: u32 = 32, // segments around the rim
    color: Color4 = Color4.white,
};

pub const RibbonOptions = struct {
    paths: []const []const Vec3, // at least 2 paths with equal point counts
    close_path: bool = false, // connect last point back to first within each path
    close_array: bool = false, // connect last path back to first path
    color: Color4 = Color4.white,
};

pub const LatheOptions = struct {
    shape: []const Vec3, // profile revolved around Y; x is radius, y is height
    tessellation: u32 = 24, // segments around the Y axis
    color: Color4 = Color4.white,
};

pub const PlaneOptions = struct {
    width: f32 = 1.0,
    height: f32 = 1.0,
    subdivisions_x: u32 = 1, // quads along X (clamped to >= 1)
    subdivisions_y: u32 = 1, // quads along Y (clamped to >= 1)
    uv_scale: Vec2 = Vec2.one, // UV multiplier applied per axis
    color: Color4 = Color4.white,
};

pub const TubeOptions = struct {
    path: []const Vec3, // at least 2 points
    radius: f32 = 0.1, // uniform radius, used when radii is null
    radii: ?[]const f32 = null, // per-point radius override (must match path length)
    tessellation: u32 = 8, // segments around the tube (clamped to >= 3)
    capped: bool = false, // flat end caps (ignored when closed)
    closed: bool = false, // connect the last point back to the first
    color: Color4 = Color4.white,
};

pub const LinesOptions = struct {
    points: []const Vec3, // at least 2 points
    width: f32 = 0.1, // ribbon width in world units
    colors: ?[]const Color4 = null, // per-point color override (must match points length)
    closed: bool = false, // connect the last point back to the first
    up: Vec3 = Vec3.up, // reference hint for the ribbon plane orientation
    color: Color4 = Color4.white,
};

pub const ExtrudeOptions = struct {
    profile: []const Vec2, // closed 2D outline in XY (CCW preferred, CW is normalized)
    depth: f32 = 1.0, // extrusion distance along +Z, from z = 0 to z = depth
    capped: bool = true, // front/back caps triangulated with ear clipping
    uv_scale: Vec2 = Vec2.one, // UV multiplier (arclength/depth on sides, XY on caps)
    color: Color4 = Color4.white,
};

pub const PolygonSideOrientation = enum {
    front,
    back,
    double_sided,
};

pub const PolygonPlane = enum {
    xz, // Ground plane, depth along +Y (default)
    xy, // Upright plane, depth along +Z
};

pub const PolygonOptions = struct {
    /// 2D outer perimeter contour in CCW order (CW is automatically normalized).
    shape: []const Vec2,
    /// Optional inner hole contours in CW order (CCW is automatically normalized).
    holes: []const []const Vec2 = &.{},
    /// Extrusion depth. 0.0 creates a flat 2D planar polygon; > 0.0 extrudes into a 3D prism.
    depth: f32 = 0.0,
    /// Reference plane for the polygon: .xz (ground) or .xy (upright).
    plane: PolygonPlane = .xz,
    /// Side orientation for flat polygon faces.
    side_orientation: PolygonSideOrientation = .front,
    /// UV multiplier.
    uv_scale: Vec2 = Vec2.one,
    color: Color4 = Color4.white,
};

// One table entry per unique revolution angle: sin/cos plus the normalized
// coordinate reused for UVs. Tables keep results bit-identical to per-vertex
// trig while computing each angle once per builder call.
pub const TrigEntry = struct {
    cos: f32,
    sin: f32,
    f: f32,
};

// Single revolution entry: f = index / count, angle = f * 2π.
pub inline fn trigEntry(count: u32, index: usize) TrigEntry {
    const count_f: f32 = @floatFromInt(count);
    const f = @as(f32, @floatFromInt(index)) / count_f;
    const a = f * 2.0 * std.math.pi;
    return .{ .cos = @cos(a), .sin = @sin(a), .f = f };
}

// One shared revolution table (count + 1 entries) for revolving builders.
pub fn buildTrigTable(allocator: std.mem.Allocator, tessellation: u32) ![]TrigEntry {
    const tab = try allocator.alloc(TrigEntry, tessellation + 1);
    for (0..tessellation + 1) |j| tab[j] = trigEntry(tessellation, j);
    return tab;
}

// Standard grid quad (a, b, c) + (a, c, d) with
// a = row * stride + col, b = a + 1, c = (row + 1) * stride + col + 1,
// d = (row + 1) * stride + col.
// Deliberately generic in the index buffer: it is genuinely called with both
// `[]u32` (GeometryData builders below) and `[]u16` (narrow-index mesh paths
// and mesh/tests.zig, which pins the u16 narrowing behavior). `@intCast`
// narrows the u32 corner ids into the slice's element type.
pub inline fn storeQuad(indices: anytype, ii: usize, a: u32, b: u32, c: u32, d: u32) void {
    indices[ii + 0] = @intCast(a);
    indices[ii + 1] = @intCast(b);
    indices[ii + 2] = @intCast(c);
    indices[ii + 3] = @intCast(a);
    indices[ii + 4] = @intCast(c);
    indices[ii + 5] = @intCast(d);
}

// Flipped grid quad (a, c, b) + (a, d, c): the Ground/Terrain winding for the
// +Y normal. Same deliberate []u16/[]u32 slice genericity as storeQuad.
pub inline fn storeQuadFlipped(indices: anytype, ii: usize, a: u32, b: u32, c: u32, d: u32) void {
    indices[ii + 0] = @intCast(a);
    indices[ii + 1] = @intCast(c);
    indices[ii + 2] = @intCast(b);
    indices[ii + 3] = @intCast(a);
    indices[ii + 4] = @intCast(d);
    indices[ii + 5] = @intCast(c);
}

// Computes the four corner ids of grid cell (row, col) and forwards to
// storeQuad (same []u16/[]u32 slice genericity).
pub inline fn appendGridQuad(indices: anytype, ii: usize, stride: usize, row: usize, col: usize) void {
    const s: u32 = @intCast(stride);
    const r: u32 = @intCast(row);
    const c: u32 = @intCast(col);
    storeQuad(indices, ii, r * s + c, r * s + c + 1, (r + 1) * s + c + 1, (r + 1) * s + c);
}

// Flipped variant of appendGridQuad (Ground/Terrain winding).
pub inline fn appendGridQuadFlipped(indices: anytype, ii: usize, stride: usize, row: usize, col: usize) void {
    const s: u32 = @intCast(stride);
    const r: u32 = @intCast(row);
    const c: u32 = @intCast(col);
    storeQuadFlipped(indices, ii, r * s + c, r * s + c + 1, (r + 1) * s + c + 1, (r + 1) * s + c);
}

pub inline fn resolveFrameSeed(tangent: Vec3, hint: Vec3) Vec3 {
    var ref = hint;
    if (ref.lengthSq() < 1e-12) ref = Vec3.up;
    if (@abs(tangent.dot(ref)) > orthogonal_dot_threshold) {
        ref = if (@abs(tangent.x) < orthogonal_dot_threshold) Vec3.right else Vec3.forward;
    }
    return ref;
}

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

pub fn buildRibbonData(allocator: std.mem.Allocator, options: RibbonOptions) !GeometryData {
    const paths = options.paths;
    if (paths.len < 2) return error.InvalidRibbon;
    const count = paths[0].len;
    if (count < 2) return error.InvalidRibbon;
    for (paths[1..]) |path| {
        if (path.len != count) return error.InvalidRibbon;
    }

    const num_paths = paths.len;
    const color = options.color.toArray();
    const close_path = options.close_path;
    const close_array = options.close_array;
    const row_quads = if (close_array) num_paths else num_paths - 1;
    const col_quads = if (close_path) count else count - 1;
    const u_denom: f32 = if (close_array) @floatFromInt(num_paths) else @floatFromInt(num_paths - 1);
    const v_denom: f32 = if (close_path) @floatFromInt(count) else @floatFromInt(count - 1);

    const vertices = try allocator.alloc(Vertex, num_paths * count);
    errdefer allocator.free(vertices);
    const indices = try allocator.alloc(u32, row_quads * col_quads * 6);
    errdefer allocator.free(indices);

    var min_v = paths[0][0];
    var max_v = paths[0][0];
    for (0..num_paths) |pi_| {
        const uu = @as(f32, @floatFromInt(pi_)) / u_denom;
        for (0..count) |ci| {
            const pos = paths[pi_][ci];
            min_v = Vec3.new(@min(min_v.x, pos.x), @min(min_v.y, pos.y), @min(min_v.z, pos.z));
            max_v = Vec3.new(@max(max_v.x, pos.x), @max(max_v.y, pos.y), @max(max_v.z, pos.z));
            vertices[pi_ * count + ci] = .{
                .position = pos.toArray(),
                .normal = .{ 0.0, 0.0, 0.0 },
                .color = color,
                .uv = .{ uu, @as(f32, @floatFromInt(ci)) / v_denom },
                .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
            };
        }
    }

    var ii: usize = 0;
    for (0..row_quads) |pi_| {
        const qi = if (close_array) (pi_ + 1) % num_paths else pi_ + 1;
        for (0..col_quads) |ci| {
            const cj = if (close_path) (ci + 1) % count else ci + 1;
            storeQuad(
                indices,
                ii,
                @intCast(pi_ * count + ci),
                @intCast(pi_ * count + cj),
                @intCast(qi * count + cj),
                @intCast(qi * count + ci),
            );
            ii += 6;
        }
    }

    var tri: usize = 0;
    while (tri < indices.len) : (tri += 3) {
        const p0 = vertices[indices[tri]].position;
        const p1 = vertices[indices[tri + 1]].position;
        const p2 = vertices[indices[tri + 2]].position;
        const e1 = Vec3.new(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]);
        const e2 = Vec3.new(p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]);
        const fn_ = e1.cross(e2);
        for ([3]u32{ indices[tri], indices[tri + 1], indices[tri + 2] }) |idx| {
            vertices[idx].normal[0] += fn_.x;
            vertices[idx].normal[1] += fn_.y;
            vertices[idx].normal[2] += fn_.z;
        }
    }

    for (0..num_paths) |pi_| {
        const path = paths[pi_];
        for (0..count) |ci| {
            const v = &vertices[pi_ * count + ci];
            var n = Vec3.new(v.normal[0], v.normal[1], v.normal[2]);
            if (n.lengthSq() < 1e-12) {
                n = Vec3.up;
            } else {
                n = n.normalize();
            }
            v.normal = n.toArray();
            const c0 = if (close_path) (ci + count - 1) % count else if (ci > 0) ci - 1 else 0;
            const c1 = if (close_path) (ci + 1) % count else @min(ci + 1, count - 1);
            var t = path[c1].sub(path[c0]);
            t = t.sub(n.scale(n.dot(t)));
            if (t.lengthSq() < 1e-12) {
                t = pickOrthogonal(n);
            } else {
                t = t.normalize();
            }
            v.tangent = .{ t.x, t.y, t.z, 1.0 };
        }
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(min_v, max_v),
    };
}

pub fn buildLatheData(allocator: std.mem.Allocator, options: LatheOptions) !GeometryData {
    const shape = options.shape;
    if (shape.len < 2) return error.InvalidLathe;
    const tess = @max(3, options.tessellation);
    const color = options.color.toArray();
    const eps = 1e-6;

    var max_r: f32 = 0.0;
    var min_y = shape[0].y;
    var max_y = shape[0].y;
    for (shape) |pt| {
        max_r = @max(max_r, @max(0.0, pt.x));
        min_y = @min(min_y, pt.y);
        max_y = @max(max_y, pt.y);
    }
    const mid_y = (min_y + max_y) * 0.5;

    var quads: usize = 0;
    for (0..shape.len - 1) |i| {
        if (@abs(shape[i].x) < eps and @abs(shape[i + 1].x) < eps) continue;
        quads += 1;
    }

    const side = tess + 1;
    const vertices = try allocator.alloc(Vertex, shape.len * side);
    errdefer allocator.free(vertices);
    const indices = try allocator.alloc(u32, quads * tess * 6);
    errdefer allocator.free(indices);

    const trig = try buildTrigTable(allocator, tess);
    defer allocator.free(trig);
    const v_denom: f32 = @floatFromInt(shape.len - 1);

    for (shape, 0..) |pt, i| {
        const prev = shape[if (i > 0) i - 1 else 0];
        const next = shape[if (i + 1 < shape.len) i + 1 else shape.len - 1];
        const dr = next.x - prev.x;
        const dy = next.y - prev.y;
        const t_len = @sqrt(dr * dr + dy * dy);
        const degenerate = t_len < eps;
        const ndr: f32 = if (degenerate) 0.0 else dr / t_len;
        const ndy: f32 = if (degenerate) 0.0 else dy / t_len;
        const cap_n: Vec3 = if (pt.y >= mid_y) Vec3.up else Vec3.down;
        const radius = @max(0.0, pt.x);
        const vv = @as(f32, @floatFromInt(i)) / v_denom;
        for (0..side) |j| {
            const sin_t = trig[j].sin;
            const cos_t = trig[j].cos;
            const n = if (degenerate) cap_n else Vec3.new(ndy * sin_t, -ndr, ndy * cos_t);
            vertices[i * side + j] = .{
                .position = .{ radius * sin_t, pt.y, radius * cos_t },
                .normal = n.toArray(),
                .color = color,
                .uv = .{ trig[j].f, vv },
                .tangent = .{ cos_t, 0.0, -sin_t, 1.0 },
            };
        }
    }

    var ii: usize = 0;
    for (0..shape.len - 1) |i| {
        if (@abs(shape[i].x) < eps and @abs(shape[i + 1].x) < eps) continue;
        for (0..tess) |j| {
            appendGridQuad(indices, ii, side, i, j);
            ii += 6;
        }
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(
            Vec3.new(-max_r, min_y, -max_r),
            Vec3.new(max_r, max_y, max_r),
        ),
    };
}

fn computePathTangents(points: []const Vec3, closed: bool, out: []Vec3) void {
    const n = points.len;
    for (0..n) |i| {
        const prev_idx = if (closed) (i + n - 1) % n else if (i > 0) i - 1 else i;
        const next_idx = if (closed) (i + 1) % n else if (i + 1 < n) i + 1 else i;
        const d = points[next_idx].sub(points[prev_idx]);
        if (d.lengthSq() > 1e-12) {
            out[i] = d.normalize();
        } else if (i > 0) {
            out[i] = out[i - 1];
        } else {
            out[i] = Vec3.forward;
        }
    }
}

fn parallelTransportFrames(tangents: []const Vec3, closed: bool, up_hint: Vec3, normals: []Vec3, binormals: []Vec3) void {
    const n = tangents.len;
    const ref = resolveFrameSeed(tangents[0], up_hint);
    var n0 = ref.sub(tangents[0].scale(tangents[0].dot(ref)));
    if (n0.lengthSq() < 1e-12) n0 = Vec3.forward;
    normals[0] = n0.normalize();
    binormals[0] = tangents[0].cross(normals[0]);
    for (1..n) |i| {
        const t = tangents[i];
        var nn = normals[i - 1].sub(t.scale(t.dot(normals[i - 1])));
        if (nn.lengthSq() < 1e-12) {
            nn = normals[i - 1];
        } else {
            nn = nn.normalize();
        }
        normals[i] = nn;
        binormals[i] = t.cross(nn);
    }
    if (closed and n > 1) {
        const t0 = tangents[0];
        var wrap = normals[n - 1].sub(t0.scale(t0.dot(normals[n - 1])));
        if (wrap.lengthSq() > 1e-12) {
            wrap = wrap.normalize();
            const cos_a = std.math.clamp(wrap.dot(normals[0]), -1.0, 1.0);
            const sin_a = t0.dot(wrap.cross(normals[0]));
            const twist = std.math.atan2(sin_a, cos_a);
            if (@abs(twist) > 1e-6) {
                const n_f: f32 = @floatFromInt(n);
                for (1..n) |i| {
                    const a = twist * (@as(f32, @floatFromInt(i)) / n_f);
                    const c = @cos(a);
                    const s = @sin(a);
                    const nn2 = normals[i];
                    const tt = tangents[i];
                    const rotated = nn2.scale(c).add(tt.cross(nn2).scale(s));
                    normals[i] = rotated;
                    binormals[i] = tt.cross(rotated);
                }
            }
        }
    }
}

fn computePathLengths(points: []const Vec3, closed: bool, cumulative: []f32) f32 {
    const n = points.len;
    cumulative[0] = 0.0;
    for (1..n) |i| {
        cumulative[i] = cumulative[i - 1] + points[i].distance(points[i - 1]);
    }
    var total = cumulative[n - 1];
    if (closed) total += points[0].distance(points[n - 1]);
    if (total < 1e-9) total = 1.0;
    return total;
}

pub fn buildTubeData(allocator: std.mem.Allocator, options: TubeOptions) !GeometryData {
    const points = options.path;
    if (points.len < 2) return error.InvalidTube;
    if (options.radii) |radii| {
        if (radii.len != points.len) return error.InvalidTube;
    }
    const tess = @max(3, options.tessellation);
    const closed = options.closed;
    const capped = options.capped and !closed;
    const color = options.color.toArray();

    const n = points.len;
    const ring = tess + 1;

    const point_radii = try allocator.alloc(f32, n);
    defer allocator.free(point_radii);
    if (options.radii) |radii| {
        for (radii, 0..) |r, i| point_radii[i] = @max(0.0, r);
    } else {
        @memset(point_radii, @max(0.0, options.radius));
    }

    const tangents = try allocator.alloc(Vec3, n);
    defer allocator.free(tangents);
    computePathTangents(points, closed, tangents);
    const normals = try allocator.alloc(Vec3, n);
    defer allocator.free(normals);
    const binormals = try allocator.alloc(Vec3, n);
    defer allocator.free(binormals);
    parallelTransportFrames(tangents, closed, Vec3.up, normals, binormals);

    const cumulative = try allocator.alloc(f32, n);
    defer allocator.free(cumulative);
    const total_len = computePathLengths(points, closed, cumulative);

    const tube_tab = try buildTrigTable(allocator, tess);
    defer allocator.free(tube_tab);

    const cap_verts = if (capped) 2 * ring + 2 else 0;
    const vertices = try allocator.alloc(Vertex, n * ring + cap_verts);
    errdefer allocator.free(vertices);
    const side_quads = if (closed) n else n - 1;
    const cap_indices = if (capped) 2 * tess * 3 else 0;
    const indices = try allocator.alloc(u32, side_quads * tess * 6 + cap_indices);
    errdefer allocator.free(indices);

    var min_v = Vec3.new(std.math.inf(f32), std.math.inf(f32), std.math.inf(f32));
    var max_v = Vec3.new(-std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32));
    var vi: usize = 0;
    for (0..n) |i| {
        const uu = cumulative[i] / total_len;
        for (0..ring) |j| {
            const dir = normals[i].scale(tube_tab[j].cos).add(binormals[i].scale(tube_tab[j].sin));
            const pos = points[i].add(dir.scale(point_radii[i]));
            vertices[vi] = .{
                .position = pos.toArray(),
                .normal = dir.toArray(),
                .color = color,
                .uv = .{ uu, tube_tab[j].f },
                .tangent = .{ tangents[i].x, tangents[i].y, tangents[i].z, 1.0 },
            };
            min_v = Vec3.new(@min(min_v.x, pos.x), @min(min_v.y, pos.y), @min(min_v.z, pos.z));
            max_v = Vec3.new(@max(max_v.x, pos.x), @max(max_v.y, pos.y), @max(max_v.z, pos.z));
            vi += 1;
        }
    }

    var ii: usize = 0;
    for (0..side_quads) |i| {
        const ni = if (closed) (i + 1) % n else i + 1;
        for (0..tess) |j| {
            storeQuad(
                indices,
                ii,
                @intCast(i * ring + j),
                @intCast(i * ring + (j + 1)),
                @intCast(ni * ring + (j + 1)),
                @intCast(ni * ring + j),
            );
            ii += 6;
        }
    }

    if (capped) {
        const cap_uv_c: [2]f32 = .{ 0.5, 0.5 };
        for ([2]usize{ 0, 1 }) |cap| {
            const row = if (cap == 0) 0 else n - 1;
            const axial = if (cap == 0) tangents[0].scale(-1.0) else tangents[n - 1];
            const center_idx: u32 = @intCast(vi);
            const cap_center = points[row];
            vertices[vi] = .{
                .position = cap_center.toArray(),
                .normal = axial.toArray(),
                .color = color,
                .uv = cap_uv_c,
                .tangent = .{ normals[row].x, normals[row].y, normals[row].z, 1.0 },
            };
            vi += 1;
            const ring_start: u32 = @intCast(vi);
            for (0..ring) |j| {
                const dir = normals[row].scale(tube_tab[j].cos).add(binormals[row].scale(tube_tab[j].sin));
                const pos = cap_center.add(dir.scale(point_radii[row]));
                vertices[vi] = .{
                    .position = pos.toArray(),
                    .normal = axial.toArray(),
                    .color = color,
                    .uv = .{ 0.5 + tube_tab[j].cos * 0.5, 0.5 + tube_tab[j].sin * 0.5 },
                    .tangent = .{ normals[row].x, normals[row].y, normals[row].z, 1.0 },
                };
                vi += 1;
            }
            for (0..tess) |j| {
                const j0: u32 = @intCast(j);
                const j1: u32 = @intCast(j + 1);
                if (cap == 0) {
                    indices[ii + 0] = center_idx;
                    indices[ii + 1] = ring_start + j1;
                    indices[ii + 2] = ring_start + j0;
                } else {
                    indices[ii + 0] = center_idx;
                    indices[ii + 1] = ring_start + j0;
                    indices[ii + 2] = ring_start + j1;
                }
                ii += 3;
            }
        }
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(min_v, max_v),
    };
}

pub fn buildLinesData(allocator: std.mem.Allocator, options: LinesOptions) !GeometryData {
    const points = options.points;
    if (points.len < 2) return error.InvalidLines;
    if (options.colors) |cols| {
        if (cols.len != points.len) return error.InvalidLines;
    }
    const half_w = @max(0.0, options.width) * 0.5;
    const closed = options.closed;
    const uniform = options.color.toArray();

    const n = points.len;
    const tangents = try allocator.alloc(Vec3, n);
    defer allocator.free(tangents);
    computePathTangents(points, closed, tangents);
    const sides = try allocator.alloc(Vec3, n);
    defer allocator.free(sides);
    const face_normals = try allocator.alloc(Vec3, n);
    defer allocator.free(face_normals);
    parallelTransportFrames(tangents, closed, options.up, sides, face_normals);

    const cumulative = try allocator.alloc(f32, n);
    defer allocator.free(cumulative);
    const total_len = computePathLengths(points, closed, cumulative);

    const vertices = try allocator.alloc(Vertex, 2 * n);
    errdefer allocator.free(vertices);
    const segs = if (closed) n else n - 1;
    const indices = try allocator.alloc(u32, segs * 6);
    errdefer allocator.free(indices);

    var min_v = Vec3.new(std.math.inf(f32), std.math.inf(f32), std.math.inf(f32));
    var max_v = Vec3.new(-std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32));
    for (0..n) |i| {
        const uu = cumulative[i] / total_len;
        const col = if (options.colors) |cols| cols[i].toArray() else uniform;
        const left = points[i].add(sides[i].scale(half_w));
        const right = points[i].sub(sides[i].scale(half_w));
        const normal = face_normals[i].toArray();
        const tangent = [4]f32{ tangents[i].x, tangents[i].y, tangents[i].z, 1.0 };
        vertices[2 * i] = .{
            .position = left.toArray(),
            .normal = normal,
            .color = col,
            .uv = .{ uu, 1.0 },
            .tangent = tangent,
        };
        vertices[2 * i + 1] = .{
            .position = right.toArray(),
            .normal = normal,
            .color = col,
            .uv = .{ uu, 0.0 },
            .tangent = tangent,
        };
        for ([2]Vec3{ left, right }) |p| {
            min_v = Vec3.new(@min(min_v.x, p.x), @min(min_v.y, p.y), @min(min_v.z, p.z));
            max_v = Vec3.new(@max(max_v.x, p.x), @max(max_v.y, p.y), @max(max_v.z, p.z));
        }
    }

    var ii: usize = 0;
    for (0..segs) |i| {
        const ni = if (closed) (i + 1) % n else i + 1;
        const a: u32 = @intCast(2 * i);
        const b: u32 = @intCast(2 * i + 1);
        const c: u32 = @intCast(2 * ni);
        const d: u32 = @intCast(2 * ni + 1);
        indices[ii + 0] = a;
        indices[ii + 1] = b;
        indices[ii + 2] = c;
        indices[ii + 3] = b;
        indices[ii + 4] = d;
        indices[ii + 5] = c;
        ii += 6;
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(min_v, max_v),
    };
}

inline fn orient2d(a: Vec2, b: Vec2, c: Vec2) f32 {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

fn pointInTriangle2d(p: Vec2, a: Vec2, b: Vec2, c: Vec2) bool {
    const eps: f32 = 1e-9;
    return orient2d(a, b, p) >= -eps and orient2d(b, c, p) >= -eps and orient2d(c, a, p) >= -eps;
}

fn isEarTip(poly: []const Vec2, ip: usize, ic: usize, inext: usize, live: []const usize) bool {
    const a = poly[ip];
    const b = poly[ic];
    const c = poly[inext];
    if (orient2d(a, b, c) <= 1e-9) return false;
    for (live) |vi| {
        if (vi == ip or vi == ic or vi == inext) continue;
        if (pointInTriangle2d(poly[vi], a, b, c)) return false;
    }
    return true;
}

fn segmentsCross2d(a: Vec2, b: Vec2, c: Vec2, d: Vec2) bool {
    const o1 = orient2d(a, b, c);
    const o2 = orient2d(a, b, d);
    const o3 = orient2d(c, d, a);
    const o4 = orient2d(c, d, b);
    return ((o1 > 0 and o2 < 0) or (o1 < 0 and o2 > 0)) and
        ((o3 > 0 and o4 < 0) or (o3 < 0 and o4 > 0));
}

pub fn buildExtrudeData(allocator: std.mem.Allocator, options: ExtrudeOptions) !GeometryData {
    const src = options.profile;
    if (src.len < 3) return error.InvalidExtrude;

    const clean = try allocator.alloc(Vec2, src.len);
    defer allocator.free(clean);
    var m: usize = 0;
    for (src) |p| {
        if (m > 0) {
            const dx = p.x - clean[m - 1].x;
            const dy = p.y - clean[m - 1].y;
            if (dx * dx + dy * dy < 1e-12) continue;
        }
        clean[m] = p;
        m += 1;
    }
    if (m > 1) {
        const dx = clean[m - 1].x - clean[0].x;
        const dy = clean[m - 1].y - clean[0].y;
        if (dx * dx + dy * dy < 1e-12) m -= 1;
    }
    if (m < 3) return error.InvalidExtrude;
    const poly = clean[0..m];

    var area2: f32 = 0.0;
    for (0..m) |i| {
        const a = poly[i];
        const b = poly[(i + 1) % m];
        area2 += a.x * b.y - b.x * a.y;
    }
    if (@abs(area2) < 1e-9) return error.InvalidExtrude;
    if (area2 < 0.0) {
        var lo: usize = 0;
        var hi: usize = m - 1;
        while (lo < hi) {
            const tmp = poly[lo];
            poly[lo] = poly[hi];
            poly[hi] = tmp;
            lo += 1;
            hi -= 1;
        }
        area2 = -area2;
    }

    for (0..m) |i| {
        const a0 = poly[i];
        const a1 = poly[(i + 1) % m];
        for (i + 1..m) |j| {
            if (j == i + 1) continue;
            if (i == 0 and j == m - 1) continue;
            if (segmentsCross2d(a0, a1, poly[j], poly[(j + 1) % m])) {
                return error.InvalidExtrude;
            }
        }
    }

    const order = try allocator.alloc(usize, m);
    defer allocator.free(order);
    for (0..m) |k| order[k] = k;
    const cap_tris = try allocator.alloc([3]u32, m - 2);
    defer allocator.free(cap_tris);

    var live_count = m;
    var tri_count: usize = 0;
    var scan_idx: usize = 0;
    var loops_without_clip: usize = 0;
    while (live_count > 3) {
        const ip = order[(scan_idx + live_count - 1) % live_count];
        const ic = order[scan_idx % live_count];
        const inext = order[(scan_idx + 1) % live_count];
        if (isEarTip(poly, ip, ic, inext, order[0..live_count])) {
            cap_tris[tri_count] = .{ @intCast(ip), @intCast(ic), @intCast(inext) };
            tri_count += 1;
            const remove_at = scan_idx % live_count;
            var shift = remove_at;
            while (shift + 1 < live_count) : (shift += 1) {
                order[shift] = order[shift + 1];
            }
            live_count -= 1;
            loops_without_clip = 0;
        } else {
            scan_idx = (scan_idx + 1) % live_count;
            loops_without_clip += 1;
            if (loops_without_clip > live_count) return error.InvalidExtrude;
        }
    }
    cap_tris[tri_count] = .{ @intCast(order[0]), @intCast(order[1]), @intCast(order[2]) };
    tri_count += 1;

    const depth = options.depth;
    const color = options.color.toArray();
    const capped = options.capped;

    const side_verts = m * 4;
    const cap_verts = if (capped) m * 2 else 0;
    const total_verts = side_verts + cap_verts;

    const side_indices = m * 6;
    const cap_indices = if (capped) (m - 2) * 3 * 2 else 0;
    const total_indices = side_indices + cap_indices;

    const vertices = try allocator.alloc(Vertex, total_verts);
    errdefer allocator.free(vertices);
    const indices = try allocator.alloc(u32, total_indices);
    errdefer allocator.free(indices);

    var min_x = poly[0].x;
    var max_x = poly[0].x;
    var min_y = poly[0].y;
    var max_y = poly[0].y;
    for (poly[1..]) |p| {
        min_x = @min(min_x, p.x);
        max_x = @max(max_x, p.x);
        min_y = @min(min_y, p.y);
        max_y = @max(max_y, p.y);
    }

    var arclen: f32 = 0.0;
    var vi: usize = 0;
    var ii: usize = 0;
    for (0..m) |i| {
        const a = poly[i];
        const b = poly[(i + 1) % m];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const len = @sqrt(dx * dx + dy * dy);
        const inv = 1.0 / @max(len, 1e-9);
        const nx = dy * inv;
        const ny = -dx * inv;
        const us = arclen * options.uv_scale.x;
        arclen += len;
        const ue = arclen * options.uv_scale.x;
        const vs: f32 = 0.0;
        const ve = depth * options.uv_scale.y;
        const normal = [3]f32{ nx, ny, 0.0 };
        const tangent = [4]f32{ 0.0, 0.0, 1.0, 1.0 };
        const base: u32 = @intCast(vi);
        vertices[vi + 0] = .{ .position = .{ a.x, a.y, 0.0 }, .normal = normal, .color = color, .uv = .{ us, vs }, .tangent = tangent };
        vertices[vi + 1] = .{ .position = .{ b.x, b.y, 0.0 }, .normal = normal, .color = color, .uv = .{ ue, vs }, .tangent = tangent };
        vertices[vi + 2] = .{ .position = .{ b.x, b.y, depth }, .normal = normal, .color = color, .uv = .{ ue, ve }, .tangent = tangent };
        vertices[vi + 3] = .{ .position = .{ a.x, a.y, depth }, .normal = normal, .color = color, .uv = .{ us, ve }, .tangent = tangent };
        vi += 4;
        storeQuad(indices, ii, base, base + 1, base + 2, base + 3);
        ii += 6;
    }

    if (capped) {
        const front_base: u32 = @intCast(vi);
        for (poly) |p| {
            vertices[vi] = .{
                .position = .{ p.x, p.y, depth },
                .normal = .{ 0.0, 0.0, 1.0 },
                .color = color,
                .uv = .{ p.x * options.uv_scale.x, p.y * options.uv_scale.y },
                .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
            };
            vi += 1;
        }
        const back_base: u32 = @intCast(vi);
        for (poly) |p| {
            vertices[vi] = .{
                .position = .{ p.x, p.y, 0.0 },
                .normal = .{ 0.0, 0.0, -1.0 },
                .color = color,
                .uv = .{ p.x * options.uv_scale.x, p.y * options.uv_scale.y },
                .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
            };
            vi += 1;
        }
        for (cap_tris[0..tri_count]) |tri| {
            indices[ii + 0] = front_base + tri[0];
            indices[ii + 1] = front_base + tri[1];
            indices[ii + 2] = front_base + tri[2];
            ii += 3;
        }
        for (cap_tris[0..tri_count]) |tri| {
            indices[ii + 0] = back_base + tri[0];
            indices[ii + 1] = back_base + tri[2];
            indices[ii + 2] = back_base + tri[1];
            ii += 3;
        }
    }

    return .{
        .vertices = vertices,
        .indices = indices,
        .bounds = BoundingBox.init(
            Vec3.new(min_x, min_y, @min(0.0, depth)),
            Vec3.new(max_x, max_y, @max(0.0, depth)),
        ),
    };
}

fn cleanPolygonContour(allocator: std.mem.Allocator, src: []const Vec2) ![]Vec2 {
    if (src.len < 3) return error.InvalidPolygon;
    const clean = try allocator.alloc(Vec2, src.len);
    errdefer allocator.free(clean);
    var m: usize = 0;
    for (src) |p| {
        if (m > 0) {
            const dx = p.x - clean[m - 1].x;
            const dy = p.y - clean[m - 1].y;
            if (dx * dx + dy * dy < 1e-12) continue;
        }
        clean[m] = p;
        m += 1;
    }
    if (m > 1) {
        const dx = clean[m - 1].x - clean[0].x;
        const dy = clean[m - 1].y - clean[0].y;
        if (dx * dx + dy * dy < 1e-12) m -= 1;
    }
    if (m < 3) return error.InvalidPolygon;
    return clean[0..m];
}

fn polygonSignedArea2(poly: []const Vec2) f32 {
    var area2: f32 = 0.0;
    const m = poly.len;
    for (0..m) |i| {
        const a = poly[i];
        const b = poly[(i + 1) % m];
        area2 += a.x * b.y - b.x * a.y;
    }
    return area2;
}

fn reversePolygonContour(poly: []Vec2) void {
    if (poly.len < 2) return;
    var lo: usize = 0;
    var hi: usize = poly.len - 1;
    while (lo < hi) {
        const tmp = poly[lo];
        poly[lo] = poly[hi];
        poly[hi] = tmp;
        lo += 1;
        hi -= 1;
    }
}

pub fn buildPolygonData(allocator: std.mem.Allocator, options: PolygonOptions) !GeometryData {
    // 1. Clean and normalize outer boundary to CCW
    const outer_clean = try cleanPolygonContour(allocator, options.shape);
    defer allocator.free(outer_clean);
    const outer_area2 = polygonSignedArea2(outer_clean);
    if (@abs(outer_area2) < 1e-9) return error.InvalidPolygon;
    if (outer_area2 < 0.0) reversePolygonContour(outer_clean);

    // 2. Clean and normalize holes to CW
    var holes_clean: std.ArrayListUnmanaged([]Vec2) = .empty;
    defer {
        for (holes_clean.items) |h| allocator.free(h);
        holes_clean.deinit(allocator);
    }
    for (options.holes) |h_src| {
        const h = cleanPolygonContour(allocator, h_src) catch continue;
        const h_area2 = polygonSignedArea2(h);
        if (@abs(h_area2) < 1e-9) {
            allocator.free(h);
            continue;
        }
        if (h_area2 > 0.0) reversePolygonContour(h); // Ensure CW for holes
        try holes_clean.append(allocator, h);
    }

    // 3. Merge holes into outer contour via bridge edges
    var merged: std.ArrayListUnmanaged(Vec2) = .empty;
    defer merged.deinit(allocator);
    try merged.appendSlice(allocator, outer_clean);

    for (holes_clean.items) |hole| {
        // Find vertex in hole with maximum X
        var h_max_idx: usize = 0;
        var max_hx = hole[0].x;
        for (hole[1..], 1..) |p, i| {
            if (p.x > max_hx) {
                max_hx = p.x;
                h_max_idx = i;
            }
        }
        const h_pt = hole[h_max_idx];

        // Shoot horizontal ray from h_pt to the right (+X direction)
        var best_edge_idx: ?usize = null;
        var min_intersect_x: f32 = std.math.inf(f32);
        const m_len = merged.items.len;
        for (0..m_len) |ei| {
            const a = merged.items[ei];
            const b = merged.items[(ei + 1) % m_len];
            if ((a.y <= h_pt.y and b.y > h_pt.y) or (b.y <= h_pt.y and a.y > h_pt.y)) {
                const dy = b.y - a.y;
                if (@abs(dy) > 1e-7) {
                    const t_param = (h_pt.y - a.y) / dy;
                    const ix = a.x + t_param * (b.x - a.x);
                    if (ix >= h_pt.x and ix < min_intersect_x) {
                        min_intersect_x = ix;
                        best_edge_idx = ei;
                    }
                }
            }
        }

        if (best_edge_idx) |ei| {
            const a = merged.items[ei];
            const b = merged.items[(ei + 1) % m_len];
            var v_mut_idx: usize = if (a.x >= b.x) ei else (ei + 1) % m_len;

            const inter_pt = Vec2.new(min_intersect_x, h_pt.y);
            const cand_pt = merged.items[v_mut_idx];
            var min_slope: f32 = std.math.inf(f32);
            for (merged.items, 0..) |v, vi| {
                if (vi == v_mut_idx) continue;
                if (pointInTriangle2d(v, h_pt, inter_pt, cand_pt)) {
                    const dx = v.x - h_pt.x;
                    const dy = @abs(v.y - h_pt.y);
                    const slope = dy / @max(dx, 1e-6);
                    if (slope < min_slope) {
                        min_slope = slope;
                        v_mut_idx = vi;
                    }
                }
            }

            // Splice hole into merged polygon at v_mut_idx
            var splice: std.ArrayListUnmanaged(Vec2) = .empty;
            defer splice.deinit(allocator);
            for (h_max_idx..hole.len) |hi| try splice.append(allocator, hole[hi]);
            for (0..h_max_idx + 1) |hi| try splice.append(allocator, hole[hi]);
            try splice.append(allocator, merged.items[v_mut_idx]);
            try merged.insertSlice(allocator, v_mut_idx + 1, splice.items);
        }
    }

    // 4. Triangulate the merged contour via ear clipping
    const poly = merged.items;
    const m = poly.len;
    if (m < 3) return error.InvalidPolygon;

    const order = try allocator.alloc(usize, m);
    defer allocator.free(order);
    for (0..m) |k| order[k] = k;

    const cap_tris = try allocator.alloc([3]u32, m - 2);
    defer allocator.free(cap_tris);

    var live_count = m;
    var tri_count: usize = 0;
    var scan_idx: usize = 0;
    var loops_without_clip: usize = 0;
    while (live_count > 3) {
        const ip = order[(scan_idx + live_count - 1) % live_count];
        const ic = order[scan_idx % live_count];
        const inext = order[(scan_idx + 1) % live_count];
        if (isEarTip(poly, ip, ic, inext, order[0..live_count])) {
            cap_tris[tri_count] = .{ @intCast(ip), @intCast(ic), @intCast(inext) };
            tri_count += 1;
            const remove_at = scan_idx % live_count;
            var shift = remove_at;
            while (shift + 1 < live_count) : (shift += 1) {
                order[shift] = order[shift + 1];
            }
            live_count -= 1;
            loops_without_clip = 0;
        } else {
            scan_idx += 1;
            loops_without_clip += 1;
            if (loops_without_clip > live_count * 2) {
                cap_tris[tri_count] = .{ @intCast(ip), @intCast(ic), @intCast(inext) };
                tri_count += 1;
                const remove_at = scan_idx % live_count;
                var shift = remove_at;
                while (shift + 1 < live_count) : (shift += 1) {
                    order[shift] = order[shift + 1];
                }
                live_count -= 1;
                loops_without_clip = 0;
            }
        }
    }
    if (live_count == 3) {
        cap_tris[tri_count] = .{ @intCast(order[0]), @intCast(order[1]), @intCast(order[2]) };
        tri_count += 1;
    }

    // 5. Compute 2D bounds
    var min_x: f32 = outer_clean[0].x;
    var max_x: f32 = outer_clean[0].x;
    var min_y: f32 = outer_clean[0].y;
    var max_y: f32 = outer_clean[0].y;
    for (outer_clean[1..]) |p| {
        min_x = @min(min_x, p.x);
        max_x = @max(max_x, p.x);
        min_y = @min(min_y, p.y);
        max_y = @max(max_y, p.y);
    }
    const span_x = @max(max_x - min_x, 1e-4);
    const span_y = @max(max_y - min_y, 1e-4);

    const is_3d = options.depth > 1e-5;
    const depth = options.depth;
    const is_xz = (options.plane == .xz);

    if (!is_3d) {
        // Flat 2D polygon
        const double_sided = (options.side_orientation == .double_sided);
        const total_indices = tri_count * 3 * (if (double_sided) @as(usize, 2) else 1);
        const vertices = try allocator.alloc(Vertex, m);
        const indices = try allocator.alloc(u32, total_indices);

        const color_arr = options.color.toArray();
        for (poly, 0..) |p, i| {
            const u = (p.x - min_x) / span_x * options.uv_scale.x;
            const v = (p.y - min_y) / span_y * options.uv_scale.y;
            if (is_xz) {
                vertices[i] = .{
                    .position = .{ p.x, 0.0, p.y },
                    .normal = .{ 0.0, 1.0, 0.0 },
                    .color = color_arr,
                    .uv = .{ u, v },
                    .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
                };
            } else {
                vertices[i] = .{
                    .position = .{ p.x, p.y, 0.0 },
                    .normal = .{ 0.0, 0.0, 1.0 },
                    .color = color_arr,
                    .uv = .{ u, v },
                    .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
                };
            }
        }

        var ii: usize = 0;
        if (is_xz) {
            for (cap_tris[0..tri_count]) |tri| {
                indices[ii + 0] = tri[0];
                indices[ii + 1] = tri[2];
                indices[ii + 2] = tri[1];
                ii += 3;
            }
            if (double_sided) {
                for (cap_tris[0..tri_count]) |tri| {
                    indices[ii + 0] = tri[0];
                    indices[ii + 1] = tri[1];
                    indices[ii + 2] = tri[2];
                    ii += 3;
                }
            }
        } else {
            for (cap_tris[0..tri_count]) |tri| {
                indices[ii + 0] = tri[0];
                indices[ii + 1] = tri[1];
                indices[ii + 2] = tri[2];
                ii += 3;
            }
            if (double_sided) {
                for (cap_tris[0..tri_count]) |tri| {
                    indices[ii + 0] = tri[0];
                    indices[ii + 1] = tri[2];
                    indices[ii + 2] = tri[1];
                    ii += 3;
                }
            }
        }

        const b_min = if (is_xz) Vec3.new(min_x, -0.01, min_y) else Vec3.new(min_x, min_y, -0.01);
        const b_max = if (is_xz) Vec3.new(max_x, 0.01, max_y) else Vec3.new(max_x, max_y, 0.01);
        return .{
            .vertices = vertices,
            .indices = indices,
            .bounds = BoundingBox.init(b_min, b_max),
        };
    } else {
        // Extruded 3D prism: top cap, bottom cap, and side walls for outer + holes
        var total_side_edges: usize = outer_clean.len;
        for (holes_clean.items) |h| total_side_edges += h.len;

        const cap_vert_count = 2 * m;
        const side_vert_count = total_side_edges * 4;
        const total_verts = cap_vert_count + side_vert_count;

        const cap_index_count = 2 * tri_count * 3;
        const side_index_count = total_side_edges * 6;
        const total_indices = cap_index_count + side_index_count;

        const vertices = try allocator.alloc(Vertex, total_verts);
        const indices = try allocator.alloc(u32, total_indices);

        var vi: usize = 0;
        var ii: usize = 0;
        const color_arr = options.color.toArray();

        // Top Cap
        const top_base: u32 = @intCast(vi);
        for (poly) |p| {
            const u = (p.x - min_x) / span_x * options.uv_scale.x;
            const v = (p.y - min_y) / span_y * options.uv_scale.y;
            if (is_xz) {
                vertices[vi] = .{
                    .position = .{ p.x, depth, p.y },
                    .normal = .{ 0.0, 1.0, 0.0 },
                    .color = color_arr,
                    .uv = .{ u, v },
                    .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
                };
            } else {
                vertices[vi] = .{
                    .position = .{ p.x, p.y, depth },
                    .normal = .{ 0.0, 0.0, 1.0 },
                    .color = color_arr,
                    .uv = .{ u, v },
                    .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
                };
            }
            vi += 1;
        }

        // Bottom Cap
        const bot_base: u32 = @intCast(vi);
        for (poly) |p| {
            const u = (p.x - min_x) / span_x * options.uv_scale.x;
            const v = (p.y - min_y) / span_y * options.uv_scale.y;
            if (is_xz) {
                vertices[vi] = .{
                    .position = .{ p.x, 0.0, p.y },
                    .normal = .{ 0.0, -1.0, 0.0 },
                    .color = color_arr,
                    .uv = .{ u, v },
                    .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
                };
            } else {
                vertices[vi] = .{
                    .position = .{ p.x, p.y, 0.0 },
                    .normal = .{ 0.0, 0.0, -1.0 },
                    .color = color_arr,
                    .uv = .{ u, v },
                    .tangent = .{ 1.0, 0.0, 0.0, 1.0 },
                };
            }
            vi += 1;
        }

        if (is_xz) {
            // Top cap indices (facing +Y)
            for (cap_tris[0..tri_count]) |tri| {
                indices[ii + 0] = top_base + tri[0];
                indices[ii + 1] = top_base + tri[2];
                indices[ii + 2] = top_base + tri[1];
                ii += 3;
            }

            // Bottom cap indices (facing -Y)
            for (cap_tris[0..tri_count]) |tri| {
                indices[ii + 0] = bot_base + tri[0];
                indices[ii + 1] = bot_base + tri[1];
                indices[ii + 2] = bot_base + tri[2];
                ii += 3;
            }
        } else {
            // Top cap indices (facing +Z, CCW)
            for (cap_tris[0..tri_count]) |tri| {
                indices[ii + 0] = top_base + tri[0];
                indices[ii + 1] = top_base + tri[1];
                indices[ii + 2] = top_base + tri[2];
                ii += 3;
            }

            // Bottom cap indices (facing -Z, reversed winding)
            for (cap_tris[0..tri_count]) |tri| {
                indices[ii + 0] = bot_base + tri[0];
                indices[ii + 1] = bot_base + tri[2];
                indices[ii + 2] = bot_base + tri[1];
                ii += 3;
            }
        }

        // Side walls helper
        const emitSideWalls = struct {
            fn emit(
                ring: []const Vec2,
                verts: []Vertex,
                inds: []u32,
                cur_vi: *usize,
                cur_ii: *usize,
                d: f32,
                xz_mode: bool,
                col: [4]f32,
                uv_s: Vec2,
            ) void {
                const r_len = ring.len;
                var dist_accum: f32 = 0.0;
                for (0..r_len) |edge_i| {
                    const p0 = ring[edge_i];
                    const p1 = ring[(edge_i + 1) % r_len];
                    const dx = p1.x - p0.x;
                    const dy = p1.y - p0.y;
                    const edge_len = @max(@sqrt(dx * dx + dy * dy), 1e-6);

                    const nx = dy / edge_len;
                    const ny = -dx / edge_len;
                    const tx = dx / edge_len;
                    const ty = dy / edge_len;

                    const uv_u0 = dist_accum * uv_s.x;
                    const uv_u1 = (dist_accum + edge_len) * uv_s.x;
                    dist_accum += edge_len;

                    const base_v: u32 = @intCast(cur_vi.*);

                    if (xz_mode) {
                        verts[cur_vi.* + 0] = .{
                            .position = .{ p0.x, 0.0, p0.y },
                            .normal = .{ nx, 0.0, ny },
                            .color = col,
                            .uv = .{ uv_u0, 0.0 },
                            .tangent = .{ tx, 0.0, ty, 1.0 },
                        };
                        verts[cur_vi.* + 1] = .{
                            .position = .{ p1.x, 0.0, p1.y },
                            .normal = .{ nx, 0.0, ny },
                            .color = col,
                            .uv = .{ uv_u1, 0.0 },
                            .tangent = .{ tx, 0.0, ty, 1.0 },
                        };
                        verts[cur_vi.* + 2] = .{
                            .position = .{ p0.x, d, p0.y },
                            .normal = .{ nx, 0.0, ny },
                            .color = col,
                            .uv = .{ uv_u0, d * uv_s.y },
                            .tangent = .{ tx, 0.0, ty, 1.0 },
                        };
                        verts[cur_vi.* + 3] = .{
                            .position = .{ p1.x, d, p1.y },
                            .normal = .{ nx, 0.0, ny },
                            .color = col,
                            .uv = .{ uv_u1, d * uv_s.y },
                            .tangent = .{ tx, 0.0, ty, 1.0 },
                        };
                    } else {
                        verts[cur_vi.* + 0] = .{
                            .position = .{ p0.x, p0.y, 0.0 },
                            .normal = .{ nx, ny, 0.0 },
                            .color = col,
                            .uv = .{ uv_u0, 0.0 },
                            .tangent = .{ tx, ty, 0.0, 1.0 },
                        };
                        verts[cur_vi.* + 1] = .{
                            .position = .{ p1.x, p1.y, 0.0 },
                            .normal = .{ nx, ny, 0.0 },
                            .color = col,
                            .uv = .{ uv_u1, 0.0 },
                            .tangent = .{ tx, ty, 0.0, 1.0 },
                        };
                        verts[cur_vi.* + 2] = .{
                            .position = .{ p0.x, p0.y, d },
                            .normal = .{ nx, ny, 0.0 },
                            .color = col,
                            .uv = .{ uv_u0, d * uv_s.y },
                            .tangent = .{ tx, ty, 0.0, 1.0 },
                        };
                        verts[cur_vi.* + 3] = .{
                            .position = .{ p1.x, p1.y, d },
                            .normal = .{ nx, ny, 0.0 },
                            .color = col,
                            .uv = .{ uv_u1, d * uv_s.y },
                            .tangent = .{ tx, ty, 0.0, 1.0 },
                        };
                    }
                    cur_vi.* += 4;

                    if (xz_mode) {
                        inds[cur_ii.* + 0] = base_v + 0;
                        inds[cur_ii.* + 1] = base_v + 2;
                        inds[cur_ii.* + 2] = base_v + 1;

                        inds[cur_ii.* + 3] = base_v + 1;
                        inds[cur_ii.* + 4] = base_v + 2;
                        inds[cur_ii.* + 5] = base_v + 3;
                    } else {
                        inds[cur_ii.* + 0] = base_v + 0;
                        inds[cur_ii.* + 1] = base_v + 1;
                        inds[cur_ii.* + 2] = base_v + 2;

                        inds[cur_ii.* + 3] = base_v + 1;
                        inds[cur_ii.* + 4] = base_v + 3;
                        inds[cur_ii.* + 5] = base_v + 2;
                    }
                    cur_ii.* += 6;
                }
            }
        }.emit;

        // Outer contour side walls
        emitSideWalls(outer_clean, vertices, indices, &vi, &ii, depth, is_xz, color_arr, options.uv_scale);

        // Hole contours side walls
        for (holes_clean.items) |hole| {
            emitSideWalls(hole, vertices, indices, &vi, &ii, depth, is_xz, color_arr, options.uv_scale);
        }

        const b_min = if (is_xz) Vec3.new(min_x, 0.0, min_y) else Vec3.new(min_x, min_y, 0.0);
        const b_max = if (is_xz) Vec3.new(max_x, depth, max_y) else Vec3.new(max_x, max_y, depth);
        return .{
            .vertices = vertices,
            .indices = indices,
            .bounds = BoundingBox.init(b_min, b_max),
        };
    }
}
