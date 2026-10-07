//! CPU-side debug wireframe helpers for physics bodies.
//! Extracted verbatim from `physics.zig`; behavior unchanged.
const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const c = @import("../c.zig").c;
const convert = @import("convert.zig");
const types = @import("types.zig");
const body_mod = @import("body.zig");
const ColliderType = types.ColliderType;
const DebugLine = types.DebugLine;
const ChildShape = body_mod.ChildShape;
const fromB3Vec = convert.fromB3Vec;

// Debug wireframe colors: dynamic bodies green, static/kinematic
// (mass <= 0) white, sensors yellow. The sensor tint wins over the other two.
pub const debug_dynamic_color: [3]f32 = .{ 0.1, 0.9, 0.3 };
pub const debug_static_color: [3]f32 = .{ 0.9, 0.9, 0.9 };
pub const debug_sensor_color: [3]f32 = .{ 0.95, 0.8, 0.15 };

// Default segments per debug circle (sphere great circles, capsule cap
// rings); override per world with PhysicsWorld.debug_circle_segments.
pub const default_debug_circle_segments: usize = 24;
pub const min_debug_circle_segments: usize = 4;
// Unit-circle samples precomputed on the stack per appendDebugLines call.
// Larger counts use per-segment trig instead (same angles, same points).
pub const debug_unit_stack_max: usize = 128;

// 12 edges of a box, shared by the scaled-box and world-AABB wireframes.
pub const debug_box_edges: [12][2]usize = .{
    .{ 0, 1 }, .{ 1, 2 }, .{ 2, 3 }, .{ 3, 0 },
    .{ 4, 5 }, .{ 5, 6 }, .{ 6, 7 }, .{ 7, 4 },
    .{ 0, 4 }, .{ 1, 5 }, .{ 2, 6 }, .{ 3, 7 },
};

pub const DebugCirclePlane = enum { xy, xz, yz };

// One unit-circle sample shared by every debug ring in a single
// appendDebugLines call; scaled by the shape radius when emitting.
pub const DebugUnit = struct { c: f32, s: f32 };

// Ring source for the circle helpers: the precomputed unit table when the
// segment count fits the stack buffer, otherwise direct per-segment trig.
// Both evaluate the same angles, so the emitted points match exactly.
pub const DebugRings = struct {
    segs: usize,
    unit: ?[]const DebugUnit,
};

// Local Y-capsule dims in body units, mirroring RigidBody.shapeVolume:
// radius is the horizontal extent, half height is the leftover Y extent.
pub fn debugCapsuleRadius(base_extents: Vec3) f32 {
    return @min(base_extents.x, base_extents.z);
}

pub fn debugCapsuleHalfHeight(base_extents: Vec3) f32 {
    return @max(0.0, base_extents.y - @min(base_extents.x, base_extents.z));
}

// Appends the 12 edges of the box (center, half extents) in body units,
// transformed to world space by the mesh world matrix (which carries the
// body scale, so callers pass unscaled dims). Allocation-free: the caller
// reserves debugLineCount() entries up front.
pub fn appendDebugBoxLines(
    out: *std.ArrayListUnmanaged(DebugLine),
    wm: Mat4,
    center: Vec3,
    half_extents: Vec3,
    color: [3]f32,
) void {
    const hx = half_extents.x;
    const hy = half_extents.y;
    const hz = half_extents.z;
    const corners = [8]Vec3{
        Vec3.new(center.x - hx, center.y - hy, center.z - hz),
        Vec3.new(center.x + hx, center.y - hy, center.z - hz),
        Vec3.new(center.x + hx, center.y + hy, center.z - hz),
        Vec3.new(center.x - hx, center.y + hy, center.z - hz),
        Vec3.new(center.x - hx, center.y - hy, center.z + hz),
        Vec3.new(center.x + hx, center.y - hy, center.z + hz),
        Vec3.new(center.x + hx, center.y + hy, center.z + hz),
        Vec3.new(center.x - hx, center.y + hy, center.z + hz),
    };
    for (debug_box_edges) |e| {
        out.appendAssumeCapacity(.{
            .a = wm.transformPoint(corners[e[0]]),
            .b = wm.transformPoint(corners[e[1]]),
            .color = color,
        });
    }
}

pub fn debugCirclePoint(center: Vec3, radius: f32, plane: DebugCirclePlane, angle: f32) Vec3 {
    const cx = @cos(angle) * radius;
    const sx = @sin(angle) * radius;
    return switch (plane) {
        .xy => Vec3.new(center.x + cx, center.y + sx, center.z),
        .xz => Vec3.new(center.x + cx, center.y, center.z + sx),
        .yz => Vec3.new(center.x, center.y + cx, center.z + sx),
    };
}

// Same math as debugCirclePoint but from a precomputed unit sample, so
// shared ring endpoints reuse one cos/sin pair instead of recomputing it.
pub fn debugCirclePointUnit(center: Vec3, radius: f32, plane: DebugCirclePlane, u: DebugUnit) Vec3 {
    const cx = u.c * radius;
    const sx = u.s * radius;
    return switch (plane) {
        .xy => Vec3.new(center.x + cx, center.y + sx, center.z),
        .xz => Vec3.new(center.x + cx, center.y, center.z + sx),
        .yz => Vec3.new(center.x, center.y + cx, center.z + sx),
    };
}

// Appends one ring of segs lines in the given local plane, transformed to
// world space by the mesh world matrix. Allocation-free (see above).
pub fn appendDebugCircleLines(
    out: *std.ArrayListUnmanaged(DebugLine),
    wm: Mat4,
    center: Vec3,
    radius: f32,
    plane: DebugCirclePlane,
    color: [3]f32,
    rings: DebugRings,
) void {
    if (rings.unit) |unit| {
        var i: usize = 0;
        while (i < rings.segs) : (i += 1) {
            out.appendAssumeCapacity(.{
                .a = wm.transformPoint(debugCirclePointUnit(center, radius, plane, unit[i])),
                .b = wm.transformPoint(debugCirclePointUnit(center, radius, plane, unit[i + 1])),
                .color = color,
            });
        }
        return;
    }
    var i: usize = 0;
    while (i < rings.segs) : (i += 1) {
        const t0 = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(rings.segs)) * 2.0 * std.math.pi;
        const t1 = @as(f32, @floatFromInt(i + 1)) / @as(f32, @floatFromInt(rings.segs)) * 2.0 * std.math.pi;
        out.appendAssumeCapacity(.{
            .a = wm.transformPoint(debugCirclePoint(center, radius, plane, t0)),
            .b = wm.transformPoint(debugCirclePoint(center, radius, plane, t1)),
            .color = color,
        });
    }
}

// 3 orthogonal great circles: 3 * segs lines.
pub fn appendDebugSphereLines(
    out: *std.ArrayListUnmanaged(DebugLine),
    wm: Mat4,
    center: Vec3,
    radius: f32,
    color: [3]f32,
    rings: DebugRings,
) void {
    appendDebugCircleLines(out, wm, center, radius, .xy, color, rings);
    appendDebugCircleLines(out, wm, center, radius, .xz, color, rings);
    appendDebugCircleLines(out, wm, center, radius, .yz, color, rings);
}

// 4 side lines joining the capsule cap rings at the cardinal points. Kept
// on the exact k * pi/2 angles (not the unit table) so points stay
// bit-identical to the historical output.
pub fn appendDebugCapsuleSides(
    out: *std.ArrayListUnmanaged(DebugLine),
    wm: Mat4,
    center: Vec3,
    radius: f32,
    half_height: f32,
    color: [3]f32,
) void {
    var k: usize = 0;
    while (k < 4) : (k += 1) {
        const t = @as(f32, @floatFromInt(k)) * 0.5 * std.math.pi;
        const x = @cos(t) * radius;
        const z = @sin(t) * radius;
        out.appendAssumeCapacity(.{
            .a = wm.transformPoint(Vec3.new(center.x + x, center.y + half_height, center.z + z)),
            .b = wm.transformPoint(Vec3.new(center.x + x, center.y - half_height, center.z + z)),
            .color = color,
        });
    }
}

// Y-capsule: top/bottom cap rings plus 4 side lines joining them at the
// cardinal points: 2 * segs + 4 lines.
pub fn appendDebugCapsuleLines(
    out: *std.ArrayListUnmanaged(DebugLine),
    wm: Mat4,
    center: Vec3,
    radius: f32,
    half_height: f32,
    color: [3]f32,
    rings: DebugRings,
) void {
    const top = Vec3.new(center.x, center.y + half_height, center.z);
    const bottom = Vec3.new(center.x, center.y - half_height, center.z);
    appendDebugCircleLines(out, wm, top, radius, .xz, color, rings);
    appendDebugCircleLines(out, wm, bottom, radius, .xz, color, rings);
    appendDebugCapsuleSides(out, wm, center, radius, half_height, color);
}

// Wireframe box of a shape's world AABB (already world space, no transform).
// Invalid shapes are skipped.
pub fn appendDebugAabbLines(
    out: *std.ArrayListUnmanaged(DebugLine),
    shape_id: c.b3ShapeId,
    color: [3]f32,
) void {
    if (shape_id.index1 == 0) return;
    const aabb = c.b3Shape_GetAABB(shape_id);
    const lo = fromB3Vec(aabb.lowerBound);
    const hi = fromB3Vec(aabb.upperBound);
    const corners = [8]Vec3{
        Vec3.new(lo.x, lo.y, lo.z),
        Vec3.new(hi.x, lo.y, lo.z),
        Vec3.new(hi.x, hi.y, lo.z),
        Vec3.new(lo.x, hi.y, lo.z),
        Vec3.new(lo.x, lo.y, hi.z),
        Vec3.new(hi.x, lo.y, hi.z),
        Vec3.new(hi.x, hi.y, hi.z),
        Vec3.new(lo.x, hi.y, hi.z),
    };
    for (debug_box_edges) |e| {
        out.appendAssumeCapacity(.{ .a = corners[e[0]], .b = corners[e[1]], .color = color });
    }
}

pub fn primaryDebugLineCount(collider: ColliderType, segs: usize) usize {
    return switch (collider) {
        .box => 12,
        .sphere => 3 * segs,
        .capsule => 2 * segs + 4,
        .hull, .mesh, .heightfield => 12,
    };
}

pub fn childDebugLineCount(kind: ChildShape.Kind, segs: usize) usize {
    return switch (kind) {
        .box => 12,
        .sphere => 3 * segs,
        .capsule => 2 * segs + 4,
        // No cheap wireframe; skipped gracefully.
        .hull => 0,
    };
}
