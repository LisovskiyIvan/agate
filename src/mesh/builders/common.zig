//! Shared helpers for the mesh builders leaves (see `mesh/builders.zig`).
//!
//! - Trig tables (`TrigEntry`, `trigEntry`, `buildTrigTable`) keep revolving
//!   builders bit-identical to per-vertex trig while computing each angle
//!   once per builder call.
//! - Grid-quad index writers (`storeQuad`, `storeQuadFlipped`,
//!   `appendGridQuad`, `appendGridQuadFlipped`) stay generic over the index
//!   slice element type (called with both `[]u32` and `[]u16`).
//! - `resolveFrameSeed` picks a stable reference for swept frames.
//! - 2D ear-clipping predicates (`orient2d`, `pointInTriangle2d`,
//!   `isEarTip`, `segmentsCross2d`) are shared by the extrude and polygon
//!   leaves. They are `pub` for those siblings but deliberately NOT
//!   re-exported by the `builders.zig` facade, so the public surface is
//!   identical to the pre-split file.
//!
//! Documented anti-cycle rule: this module is imported BY the leaves, never
//! the reverse, and no leaf may import the `builders.zig` facade.
const std = @import("std");
const math = @import("math");
const Vec2 = math.Vec2;
const Vec3 = math.Vec3;
const orthogonal_dot_threshold = @import("../tangents.zig").orthogonal_dot_threshold;

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

pub inline fn orient2d(a: Vec2, b: Vec2, c: Vec2) f32 {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

pub fn pointInTriangle2d(p: Vec2, a: Vec2, b: Vec2, c: Vec2) bool {
    const eps: f32 = 1e-9;
    return orient2d(a, b, p) >= -eps and orient2d(b, c, p) >= -eps and orient2d(c, a, p) >= -eps;
}

pub fn isEarTip(poly: []const Vec2, ip: usize, ic: usize, inext: usize, live: []const usize) bool {
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

pub fn segmentsCross2d(a: Vec2, b: Vec2, c: Vec2, d: Vec2) bool {
    const o1 = orient2d(a, b, c);
    const o2 = orient2d(a, b, d);
    const o3 = orient2d(c, d, a);
    const o4 = orient2d(c, d, b);
    return ((o1 > 0 and o2 < 0) or (o1 < 0 and o2 > 0)) and
        ((o3 > 0 and o4 < 0) or (o3 < 0 and o4 > 0));
}
