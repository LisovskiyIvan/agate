//! Debug wireframe entry points as free functions.
//! Extracted from `PhysicsWorld` methods in `physics.zig`; behavior unchanged.
//! Functions take a concrete `*PhysicsWorld` (the type lives in the neutral
//! `world.zig`, so importing it here breaks no import cycle). The geometry
//! helpers live in `debug_geo.zig`.
const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const types = @import("types.zig");
const debug_geo = @import("debug_geo.zig");
const world_mod = @import("world.zig");
const PhysicsWorld = world_mod.PhysicsWorld;
const DebugLine = types.DebugLine;
const min_debug_circle_segments = debug_geo.min_debug_circle_segments;
const debug_unit_stack_max = debug_geo.debug_unit_stack_max;
const DebugUnit = debug_geo.DebugUnit;
const DebugRings = debug_geo.DebugRings;
const debug_dynamic_color = debug_geo.debug_dynamic_color;
const debug_static_color = debug_geo.debug_static_color;
const debug_sensor_color = debug_geo.debug_sensor_color;
const debugCapsuleRadius = debug_geo.debugCapsuleRadius;
const debugCapsuleHalfHeight = debug_geo.debugCapsuleHalfHeight;
const appendDebugBoxLines = debug_geo.appendDebugBoxLines;
const appendDebugSphereLines = debug_geo.appendDebugSphereLines;
const appendDebugCapsuleLines = debug_geo.appendDebugCapsuleLines;
const appendDebugAabbLines = debug_geo.appendDebugAabbLines;
const primaryDebugLineCount = debug_geo.primaryDebugLineCount;
const childDebugLineCount = debug_geo.childDebugLineCount;

/// Clamped ring segment count used by the debug wireframe helpers.
pub fn debugCircleSegments(world: *const PhysicsWorld) usize {
    return @max(min_debug_circle_segments, world.debug_circle_segments);
}

/// Builds a CPU-side wireframe for every enabled body and appends it to
/// `out` (never cleared) for an external/debug line renderer. Local
/// wireframes live in body units; the mesh world matrix
/// (`Mesh.getWorldMatrix`, carrying scale/rotation/translation) moves
/// them to world space, so uniform scales land exactly on the collider
/// dims (non-uniform scales approximate circles as ellipses).
/// Hull/mesh/heightfield colliders fall back to the shape's world AABB
/// (`b3Shape_GetAABB`); hull child shapes are skipped. Colors: dynamic
/// green, static/kinematic (mass <= 0) white, sensors yellow (sensor
/// wins). No global state; allocation failures surface as
/// `error.OutOfMemory`.
/// Line counts: box 12, sphere 3 * debugCircleSegments() (72 at the
/// default 24), capsule 2 * debugCircleSegments() + 4 (52 at default),
/// hull/mesh/heightfield 12 (AABB box). Capacity for the exact count is
/// reserved once up front, so steady-state appends never reallocate;
/// circle trig is computed once per call into a shared unit table.
pub fn appendDebugLines(world: *const PhysicsWorld, allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(DebugLine)) !void {
    // Reserve once; every helper below uses appendAssumeCapacity.
    try out.ensureUnusedCapacity(allocator, debugLineCount(world));
    const segs = debugCircleSegments(world);
    var rings = DebugRings{ .segs = segs, .unit = null };
    // segs + 1 samples so consecutive segments share endpoints.
    var unit_buf: [debug_unit_stack_max + 1]DebugUnit = undefined;
    if (segs <= debug_unit_stack_max) {
        var j: usize = 0;
        while (j <= segs) : (j += 1) {
            const t = @as(f32, @floatFromInt(j)) / @as(f32, @floatFromInt(segs)) * 2.0 * std.math.pi;
            unit_buf[j] = .{ .c = @cos(t), .s = @sin(t) };
        }
        rings.unit = unit_buf[0 .. segs + 1];
    }
    for (world.bodies.items) |body| {
        if (!body.enabled) continue;
        const color: [3]f32 = if (body.is_sensor)
            debug_sensor_color
        else if (body.mass <= 0.0)
            debug_static_color
        else
            debug_dynamic_color;
        const wm = body.mesh.getWorldMatrix();
        switch (body.collider) {
            .box => appendDebugBoxLines(out, wm, Vec3.zero, body.base_extents, color),
            .sphere => appendDebugSphereLines(out, wm, Vec3.zero, body.base_radius, color, rings),
            .capsule => appendDebugCapsuleLines(
                out,
                wm,
                Vec3.zero,
                debugCapsuleRadius(body.base_extents),
                debugCapsuleHalfHeight(body.base_extents),
                color,
                rings,
            ),
            .hull, .mesh, .heightfield => appendDebugAabbLines(out, body.shape_id, color),
        }
        for (body.child_shapes.items) |child| {
            switch (child.kind) {
                .box => |he| appendDebugBoxLines(out, wm, child.offset, he, color),
                .sphere => |r| appendDebugSphereLines(out, wm, child.offset, r, color, rings),
                .capsule => |cp| appendDebugCapsuleLines(out, wm, child.offset, cp.radius, cp.half_height, color, rings),
                .hull => {},
            }
        }
    }
}

/// Exact line count `appendDebugLines` would add (no allocation).
pub fn debugLineCount(world: *const PhysicsWorld) usize {
    const segs = debugCircleSegments(world);
    var n: usize = 0;
    for (world.bodies.items) |body| {
        if (!body.enabled) continue;
        n += primaryDebugLineCount(body.collider, segs);
        for (body.child_shapes.items) |child| {
            n += childDebugLineCount(child.kind, segs);
        }
    }
    return n;
}
