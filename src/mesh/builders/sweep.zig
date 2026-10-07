//! Path-swept and lofted surfaces (see `mesh/builders.zig`): ribbon, lathe,
//! tube, and lines plus their option structs and the shared
//! parallel-transport frame machinery used by the tube/lines paths.
//! Imports the `common` sibling only; never the `builders.zig` facade.
const std = @import("std");
const math = @import("math");
const Vec2 = math.Vec2;
const Vec3 = math.Vec3;
const Color4 = math.Color4;
const BoundingBox = math.BoundingBox;
const Vertex = @import("../types.zig").Vertex;
const GeometryData = @import("../types.zig").GeometryData;
const pickOrthogonal = @import("../tangents.zig").pickOrthogonal;
const common = @import("common.zig");
const storeQuad = common.storeQuad;
const appendGridQuad = common.appendGridQuad;
const buildTrigTable = common.buildTrigTable;
const resolveFrameSeed = common.resolveFrameSeed;

pub const RibbonOptions = struct {
    paths: []const []const Vec3, // at least 2 paths with equal point counts
    close_path: bool = false, // connect last point back to first within each path
    close_array: bool = false, // connect last path back to first path
    color: Color4 = Color4.white,
};

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

pub const LatheOptions = struct {
    shape: []const Vec3, // profile revolved around Y; x is radius, y is height
    tessellation: u32 = 24, // segments around the Y axis
    color: Color4 = Color4.white,
};

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

pub const TubeOptions = struct {
    path: []const Vec3, // at least 2 points
    radius: f32 = 0.1, // uniform radius, used when radii is null
    radii: ?[]const f32 = null, // per-point radius override (must match path length)
    tessellation: u32 = 8, // segments around the tube (clamped to >= 3)
    capped: bool = false, // flat end caps (ignored when closed)
    closed: bool = false, // connect the last point back to the first
    color: Color4 = Color4.white,
};

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

pub const LinesOptions = struct {
    points: []const Vec3, // at least 2 points
    width: f32 = 0.1, // ribbon width in world units
    colors: ?[]const Color4 = null, // per-point color override (must match points length)
    closed: bool = false, // connect the last point back to the first
    up: Vec3 = Vec3.up, // reference hint for the ribbon plane orientation
    color: Color4 = Color4.white,
};

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
