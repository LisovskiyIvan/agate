//! Spatial queries and ray/sphere casts as free functions.
//! Extracted from `PhysicsWorld` methods in `physics.zig`; behavior unchanged.
//! Functions take a concrete `*PhysicsWorld` (the type lives in the neutral
//! `world.zig`, so importing it here breaks no import cycle).
const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const c = @import("../c.zig").c;
const convert = @import("convert.zig");
const types = @import("types.zig");
const body_mod = @import("body.zig");
const world_mod = @import("world.zig");
const RigidBody = body_mod.RigidBody;
const CollisionFilter = types.CollisionFilter;
const PhysicsRayHit = body_mod.PhysicsRayHit;
const PhysicsWorld = world_mod.PhysicsWorld;
const toB3Vec = convert.toB3Vec;
const fromB3Vec = convert.fromB3Vec;
const toB3Pos = convert.toB3Pos;
const fromB3Pos = convert.fromB3Pos;

pub fn raycast(world: *PhysicsWorld, origin: Vec3, direction: Vec3, max_distance: f32) PhysicsRayHit {
    return raycastWithFilter(world, origin, direction, max_distance, .{});
}

/// Same as raycast but only accepts shapes matching the filter mask.
pub fn raycastWithFilter(
    world: *PhysicsWorld,
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
    filter: CollisionFilter,
) PhysicsRayHit {
    if (max_distance <= 0.0) return .{};
    const len = direction.length();
    if (len < 1e-6) return .{};

    const translation = direction.scale(max_distance / len);
    var query_filter = c.b3DefaultQueryFilter();
    query_filter.categoryBits = filter.category_bits;
    query_filter.maskBits = filter.mask_bits;
    const result = c.b3World_CastRayClosest(world.world_id, toB3Pos(origin), toB3Vec(translation), query_filter);
    if (!result.hit) return .{};

    return .{
        .hit = true,
        .point = fromB3Pos(result.point),
        .normal = fromB3Vec(result.normal),
        .distance = max_distance * result.fraction,
        .body = findBodyByShape(world, result.shapeId),
    };
}

/// Maps a `CollisionFilter` onto a Box3D query filter, exactly like
/// `raycastWithFilter` does (category/mask bits; recorder id/name stay default).
pub fn toB3QueryFilter(f: CollisionFilter) c.b3QueryFilter {
    var q = c.b3DefaultQueryFilter();
    q.categoryBits = f.category_bits;
    q.maskBits = f.mask_bits;
    return q;
}

/// Shared context for the overlap-query callbacks below.
const OverlapCollectCtx = struct {
    world: *PhysicsWorld,
    results: *std.ArrayListUnmanaged(*RigidBody),
    /// When set (queryPoint), only accept shapes whose world AABB contains this point.
    point: ?Vec3 = null,
    /// Last reported body: Box3D often reports a compound body's shapes
    /// back to back, so re-check it before the full body scan.
    last: ?*RigidBody = null,
    /// Set when `results.append` runs out of memory; the query is aborted
    /// (callback returns false) and the caller converts this to `error.OutOfMemory`.
    oom: bool = false,
};

/// `b3OverlapResultFcn`: maps each reported shape to its body, skips
/// untracked shapes (e.g. the internal ground plane) and `!enabled`
/// bodies, and appends each body at most once (compound bodies report
/// one shape per child).
fn overlapCollectFcn(shape_id: c.b3ShapeId, context: ?*anyopaque) callconv(.c) bool {
    const ctx_ptr = context orelse return true;
    const ctx: *OverlapCollectCtx = @ptrCast(@alignCast(ctx_ptr));
    // Fast path for consecutive shapes of one compound body; the
    // ownsShape check returns the same body findBodyByShape would.
    var cached: ?*RigidBody = null;
    if (ctx.last) |last| {
        if (ownsShape(last, shape_id)) cached = last;
    }
    const body = cached orelse findBodyByShape(ctx.world, shape_id) orelse return true;
    ctx.last = body;
    if (!body.enabled) return true;
    if (ctx.point) |p| {
        const aabb = c.b3Shape_GetAABB(shape_id);
        if (p.x < aabb.lowerBound.x or p.x > aabb.upperBound.x or
            p.y < aabb.lowerBound.y or p.y > aabb.upperBound.y or
            p.z < aabb.lowerBound.z or p.z > aabb.upperBound.z) return true;
    }
    // Fast path: a repeat report lands at the end of the list, which the
    // scan below would find anyway; same ordering and contents.
    if (ctx.results.items.len > 0 and ctx.results.items[ctx.results.items.len - 1] == body) return true;
    for (ctx.results.items) |b| {
        if (b == body) return true;
    }
    ctx.results.append(ctx.world.allocator, body) catch {
        ctx.oom = true;
        return false;
    };
    return true;
}

/// Appends every enabled body with a shape potentially overlapping the
/// box `[min, max]` (broadphase `b3World_OverlapAABB`). Each body is
/// appended at most once. Results are APPENDED, not cleared.
pub fn queryAABB(
    world: *PhysicsWorld,
    min: Vec3,
    max: Vec3,
    results: *std.ArrayListUnmanaged(*RigidBody),
) !void {
    return queryAABBWithFilter(world, min, max, .{}, results);
}

/// Same as `queryAABB` but only accepts shapes matching the filter mask
/// (mapped exactly like `raycastWithFilter`).
pub fn queryAABBWithFilter(
    world: *PhysicsWorld,
    min: Vec3,
    max: Vec3,
    filter: CollisionFilter,
    results: *std.ArrayListUnmanaged(*RigidBody),
) !void {
    const aabb = c.b3AABB{
        .lowerBound = toB3Vec(Vec3.new(@min(min.x, max.x), @min(min.y, max.y), @min(min.z, max.z))),
        .upperBound = toB3Vec(Vec3.new(@max(min.x, max.x), @max(min.y, max.y), @max(min.z, max.z))),
    };
    var ctx = OverlapCollectCtx{ .world = world, .results = results };
    _ = c.b3World_OverlapAABB(world.world_id, aabb, toB3QueryFilter(filter), &overlapCollectFcn, &ctx);
    if (ctx.oom) return error.OutOfMemory;
}

/// Appends every enabled body overlapping the sphere (`center`, `radius`)
/// via an exact `b3World_OverlapShape` query. The sphere proxy is a single
/// point with a non-zero radius (see `b3ShapeCastInput` docs), so this is
/// precise for all collider types — no AABB approximation. Each body is
/// appended at most once. Results are APPENDED, not cleared.
pub fn querySphere(
    world: *PhysicsWorld,
    center: Vec3,
    radius: f32,
    results: *std.ArrayListUnmanaged(*RigidBody),
) !void {
    return querySphereWithFilter(world, center, radius, .{}, results);
}

/// Same as `querySphere` but only accepts shapes matching the filter mask
/// (mapped exactly like `raycastWithFilter`).
pub fn querySphereWithFilter(
    world: *PhysicsWorld,
    center: Vec3,
    radius: f32,
    filter: CollisionFilter,
    results: *std.ArrayListUnmanaged(*RigidBody),
) !void {
    if (!(radius > 0.0)) return;
    // Proxy points are relative to `origin`, so a sphere is one
    // origin-centered point plus the radius.
    var point = c.b3Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    var proxy = c.b3ShapeProxy{ .points = &point, .count = 1, .radius = radius };
    var ctx = OverlapCollectCtx{ .world = world, .results = results };
    _ = c.b3World_OverlapShape(world.world_id, toB3Pos(center), &proxy, toB3QueryFilter(filter), &overlapCollectFcn, &ctx);
    if (ctx.oom) return error.OutOfMemory;
}

/// Appends every enabled body whose shape world AABB (`b3Shape_GetAABB`)
/// contains `point`. Broadphase is a zero-extent `b3World_OverlapAABB`
/// query; the per-shape AABB check is the precise test, so rotated/thin
/// shapes report AABB containment, not exact surface containment. Each
/// body is appended at most once. Results are APPENDED, not cleared.
pub fn queryPoint(
    world: *PhysicsWorld,
    point: Vec3,
    results: *std.ArrayListUnmanaged(*RigidBody),
) !void {
    return queryPointWithFilter(world, point, .{}, results);
}

/// Same as `queryPoint` but only accepts shapes matching the filter mask
/// (mapped exactly like `raycastWithFilter`).
pub fn queryPointWithFilter(
    world: *PhysicsWorld,
    point: Vec3,
    filter: CollisionFilter,
    results: *std.ArrayListUnmanaged(*RigidBody),
) !void {
    const p = toB3Vec(point);
    const aabb = c.b3AABB{ .lowerBound = p, .upperBound = p };
    var ctx = OverlapCollectCtx{ .world = world, .results = results, .point = point };
    _ = c.b3World_OverlapAABB(world.world_id, aabb, toB3QueryFilter(filter), &overlapCollectFcn, &ctx);
    if (ctx.oom) return error.OutOfMemory;
}

/// Shared context for the sphere-cast callback below.
const SphereCastCtx = struct {
    world: *PhysicsWorld,
    travel: f32,
    best_fraction: f32 = std.math.floatMax(f32),
    body: ?*RigidBody = null,
    point: Vec3 = Vec3.zero,
    normal: Vec3 = Vec3.up,
    /// Last reported body (same consecutive-shape fast path as above).
    last: ?*RigidBody = null,
};

/// `b3CastResultFcn`: keeps the closest accepted hit, clipping the cast to
/// the best fraction found so far to enable BVH tree pruning.
/// Untracked shapes and `!enabled` bodies are ignored (return -1.0).
fn sphereCastCollectFcn(
    shape_id: c.b3ShapeId,
    point: c.b3Pos,
    normal: c.b3Vec3,
    fraction: f32,
    _: u64,
    _: c_int,
    _: c_int,
    context: ?*anyopaque,
) callconv(.c) f32 {
    const ctx_ptr = context orelse return 1.0;
    const ctx: *SphereCastCtx = @ptrCast(@alignCast(ctx_ptr));
    var cached: ?*RigidBody = null;
    if (ctx.last) |last| {
        if (ownsShape(last, shape_id)) cached = last;
    }
    const body = cached orelse findBodyByShape(ctx.world, shape_id) orelse return -1.0;
    ctx.last = body;
    if (!body.enabled) return -1.0;
    if (fraction < ctx.best_fraction) {
        ctx.best_fraction = fraction;
        ctx.body = body;
        ctx.point = fromB3Pos(point);
        ctx.normal = fromB3Vec(normal);
        return fraction;
    }
    return ctx.best_fraction;
}

/// Sweeps a sphere (`origin`, `radius`) along `translation` and returns
/// the closest hit, or null on a miss. Implemented with `b3World_CastShape`
/// and a single-point sphere proxy (same construction as `querySphere`).
pub fn spherecast(world: *PhysicsWorld, origin: Vec3, radius: f32, translation: Vec3) ?PhysicsRayHit {
    return spherecastWithFilter(world, origin, radius, translation, .{});
}

/// Same as `spherecast` but only accepts shapes matching the filter mask
/// (mapped exactly like `raycastWithFilter`).
pub fn spherecastWithFilter(
    world: *PhysicsWorld,
    origin: Vec3,
    radius: f32,
    translation: Vec3,
    filter: CollisionFilter,
) ?PhysicsRayHit {
    const travel = translation.length();
    if (!(radius > 0.0) or travel < 1e-6) return null;
    var point = c.b3Vec3{ .x = 0.0, .y = 0.0, .z = 0.0 };
    var proxy = c.b3ShapeProxy{ .points = &point, .count = 1, .radius = radius };
    var ctx = SphereCastCtx{ .world = world, .travel = travel };
    _ = c.b3World_CastShape(world.world_id, toB3Pos(origin), &proxy, toB3Vec(translation), toB3QueryFilter(filter), &sphereCastCollectFcn, &ctx);
    const body = ctx.body orelse return null;
    return .{
        .hit = true,
        // Box3D reports the surface point on the TARGET shape; report the
        // cast-sphere CENTER at first contact instead, so point and distance
        // share the same convention (point == origin + translation * t,
        // where distance == travel * t).
        .point = origin.add(translation.scale(ctx.best_fraction)),
        .normal = ctx.normal,
        .distance = travel * ctx.best_fraction,
        .body = body,
    };
}

pub fn findBodyByShape(world: *PhysicsWorld, shape_id: c.b3ShapeId) ?*RigidBody {
    for (world.bodies.items) |b| {
        if (ownsShape(b, shape_id)) {
            return b;
        }
    }
    return null;
}

/// True when the shape is the body's primary shape or one of its children.
/// Takes no world: the original `_: *PhysicsWorld` parameter was unused.
pub fn ownsShape(body: *RigidBody, shape_id: c.b3ShapeId) bool {
    if (body.shape_id.index1 == shape_id.index1 and body.shape_id.generation == shape_id.generation) {
        return true;
    }
    for (body.child_shapes.items) |ch| {
        if (ch.shape_id.index1 == shape_id.index1 and ch.shape_id.generation == shape_id.generation) {
            return true;
        }
    }
    return false;
}
