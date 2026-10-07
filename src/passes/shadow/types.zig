//! Shared shadow vocabulary: atlas layouts, per-light render infos, pipeline
//! buckets and the point-atlas helpers. Split out of `shadow_pass.zig`
//! (facade).
//!
//! This module owns the free-standing public items: the CSM/spot/point atlas
//! constants, `SpotShadowRenderInfo`/`PointShadowRenderInfo`,
//! `pointFaceForDir`/`pointTileOrigin`, the pipeline `Bucket` vocabulary with
//! `bucket_order`, the `BinResult` ranges and the pure `bucketFor`
//! classifier. The `ShadowPass` owner lives in `core.zig`.
//!
//! Leaf: imports `math` + `mesh` only, never a sibling or the facade (same
//! anti-cycle rule as `particles/*`, `profiler/*`). `bucketFor` is `pub` for
//! the `binning`/`prepare` siblings but is deliberately NOT re-exported from
//! the `shadow_pass.zig` facade, so the public surface is identical to the
//! pre-split file.
const std = @import("std");
const math = @import("math");
const mesh_mod = @import("../../mesh.zig");
const Mesh = mesh_mod.Mesh;

// Resolution of the shadow atlas texture (square). Must match the
// SHADOW_ATLAS_SIZE fallback in shaders/{standard,pbr,instanced,skinned_pbr}.glsl:
// sokol-shdc --defines only supports valueless macros, so the value cannot be
// injected from the build and lives in these two places by convention.
pub const SHADOW_ATLAS_SIZE: u32 = 2048;
pub const SPOT_SHADOW_SLOTS: usize = 2;
pub const SPOT_SHADOW_MAP_WIDTH: u32 = 1024;
pub const SPOT_SHADOW_MAP_HEIGHT: u32 = 512;
pub const SPOT_SHADOW_RES: i32 = 512;

// Point-light shadow atlas: 2 shadow slots x 6 cube faces as 256px tiles in
// one 2D depth texture, sampled with the same 2D compare + PCF path as the
// spot atlas (no cube textures anywhere).
//
// Final atlas budget (all atlases are separate depth textures):
//   CSM atlas   2048x2048: 4 cascades of 1024x1024 (2x2 grid, unchanged).
//   Spot atlas  1024x512:  2 tiles of 512x512 side by side (unchanged).
//   Point atlas 1536x512:  12 tiles of 256x256 — 6 faces per row, one row
//     per shadow slot (slot 0: y 0..256, slot 1: y 256..512).
// SHADOW_ATLAS_SIZE stays 2048: the CSM layout is untouched, so no shader
// fallback needs updating.
pub const POINT_SHADOW_SLOTS: usize = 2;
pub const POINT_SHADOW_FACES: usize = 6;
pub const POINT_SHADOW_RES: i32 = 256;
pub const POINT_SHADOW_MAP_WIDTH: u32 = 1536;
pub const POINT_SHADOW_MAP_HEIGHT: u32 = 512;

pub const SpotShadowRenderInfo = struct {
    spot_index: usize = 0,
    tile_x: i32 = 0,
    tile_y: i32 = 0,
    view_proj: Mat4 = Mat4.identity,
};

pub const PointShadowRenderInfo = struct {
    tile_x: i32 = 0,
    tile_y: i32 = 0,
    view_proj: Mat4 = Mat4.identity,
};

const Mat4 = math.Mat4;

/// Pixel origin of a spot-shadow atlas page / tile: side-by-side tiles.
pub fn spotTileOrigin(slot: usize) struct { x: i32, y: i32 } {
    return .{
        .x = @as(i32, @intCast(slot % SPOT_SHADOW_SLOTS)) * SPOT_SHADOW_RES,
        .y = 0,
    };
}

/// Cube-face index for a light-space direction, mirroring the GLSL
/// pointFaceIndex in the forward shaders: major axis wins, ties prefer
/// X over Y over Z, the sign picks the positive/negative face
/// (order +X, -X, +Y, -Y, +Z, -Z, matching PointLight face order).
pub fn pointFaceForDir(d: math.Vec3) usize {
    const ax = @abs(d.x);
    const ay = @abs(d.y);
    const az = @abs(d.z);
    if (ax >= ay and ax >= az) return if (d.x >= 0.0) 0 else 1;
    if (ay >= ax and ay >= az) return if (d.y >= 0.0) 2 else 3;
    return if (d.z >= 0.0) 4 else 5;
}

/// Pixel origin of a point-shadow tile: faces run left to right, one row
/// per shadow slot.
pub fn pointTileOrigin(slot: usize, face: usize) struct { x: i32, y: i32 } {
    return .{
        .x = @as(i32, @intCast(face % POINT_SHADOW_FACES)) * POINT_SHADOW_RES,
        .y = @as(i32, @intCast(slot % POINT_SHADOW_SLOTS)) * POINT_SHADOW_RES,
    };
}

// Pipeline buckets in fixed order; the grouping key is defined once per
// render so each cascade issues at most one applyPipeline per bucket.
pub const Bucket = enum { regular_u16, regular_u32, inst_u16, inst_u32, skinned_u16, skinned_u32 };
pub const bucket_order: [6]Bucket = .{ .regular_u16, .regular_u32, .inst_u16, .inst_u32, .skinned_u16, .skinned_u32 };

pub const BinResult = struct {
    counts: [6]usize,
    offsets: [6]usize,
};

// Matches Scene.render's pipeline pick: instanced wins over skinned.
pub fn bucketFor(mesh: *const Mesh) Bucket {
    const is_32 = mesh.index_type == .UINT32;
    if (mesh.instances.items.len > 0) return if (is_32) .inst_u32 else .inst_u16;
    if (mesh.skeleton != null) return if (is_32) .skinned_u32 else .skinned_u16;
    return if (is_32) .regular_u32 else .regular_u16;
}

// ---- CSM cascade size-culling + shadow LOD policy. ----
//
// Minimum world-space AABB max-extent (meters) for a caster to be drawn
// into each CSM cascade. Cascade 0 (near camera) never culls by size, so
// near-camera shadows stay bit-identical; distant cascades drop
// sub-texel casters (historical 0.12/0.35/0.75 policy, now named).
pub const CASCADE_MIN_DIM: [4]f32 = .{ 0.0, 0.12, 0.35, 0.75 };

/// First CSM cascade allowed to draw the low-poly shadow LOD stand-in
/// instead of the full-res mesh. Cascades 0..1 (near field) always draw
/// full-res; cascades 2..3 (distant) draw the LOD when one was snapshotted.
/// Spot/point paths pass `null` (no cascade) and always draw full-res.
pub const SHADOW_LOD_FIRST_CASCADE: usize = 2;

/// Pure size-culling predicate behind `buckets.shadowItemCulled`.
/// `cascade_idx == null` (spot/point paths) never culls by size.
pub fn cascadeSizeCulled(cascade_idx: ?usize, max_dim: f32) bool {
    const c = cascade_idx orelse return false;
    if (c >= CASCADE_MIN_DIM.len) return false;
    return max_dim < CASCADE_MIN_DIM[c];
}

/// Pure shadow LOD gate behind `buckets.renderBuckets`: draw the snapshotted
/// low-poly geometry only for distant CSM cascades with a valid stand-in.
pub fn shadowLodActive(has_shadow_lod: bool, cascade_idx: ?usize) bool {
    const c = cascade_idx orelse return false;
    return has_shadow_lod and c >= SHADOW_LOD_FIRST_CASCADE;
}

/// Resolves the genuinely-simplified LOD child with the fewest indices that
/// is usable as a shadow stand-in for `mesh`, or null when no safe stand-in
/// exists — the caller must then fail safe to the high-poly mesh. Rejects:
/// - skinned sources (LOD children carry no skeleton, and the skinned
///   shadow pipeline needs bone matrices);
/// - morph sources (the LOD child would freeze the unmorphed shape);
/// - empty chains and chains of pure cull markers (`mesh == null`);
/// - self-aliases and non-decimated children
///   (`index_count >= source`), so a misconfigured LOD never silently
///   replaces the mesh with itself or a heavier copy;
/// - index-type mismatches (the bucket pipeline is picked from the source);
/// - children still awaiting GPU upload or without live handle ids
///   (`gpu_pending`, zero vertex/index buffers).
///
/// The returned geometry is expected to come from the QEM simplifier
/// (`mesh/simplify.zig`: `simplifyGeometry`/`generateLODLevels`). The
/// vendored meshoptimizer is decoder-only (index/vertex codecs + filters);
/// its `meshopt_simplify*` implementations are NOT vendored, so it cannot
/// serve as the simplifier here.
pub fn shadowLodMesh(mesh: *const Mesh) ?*const Mesh {
    if (mesh.skeleton != null) return null;
    if (mesh.morph_targets.len > 0) return null;
    if (mesh.lod_levels.items.len == 0) return null;
    var coarsest: ?*const Mesh = null;
    for (mesh.lod_levels.items) |lvl| {
        const lm = lvl.mesh orelse continue;
        if (@intFromPtr(lm) == @intFromPtr(mesh)) continue;
        if (lm.gpu_pending or lm.vertex_buffer.id == 0 or lm.index_buffer.id == 0 or lm.index_count == 0) continue;
        if (lm.index_type != mesh.index_type or lm.index_count >= mesh.index_count) continue;
        if (coarsest == null or lm.index_count < coarsest.?.index_count) coarsest = lm;
    }
    const lod = coarsest orelse return null;
    return lod;
}

// ---- Point-light shadow atlas: face math + tile bookkeeping. ----

test "pointFaceForDir selects the major-axis face with X>Y>Z tie-break" {
    const V = math.Vec3.new;
    try std.testing.expectEqual(@as(usize, 0), pointFaceForDir(V(1, 0, 0)));
    try std.testing.expectEqual(@as(usize, 1), pointFaceForDir(V(-2, 0.5, 0.5)));
    try std.testing.expectEqual(@as(usize, 2), pointFaceForDir(V(0.1, 3, 0.1)));
    try std.testing.expectEqual(@as(usize, 3), pointFaceForDir(V(0, -1, 0)));
    try std.testing.expectEqual(@as(usize, 4), pointFaceForDir(V(0, 0, 5)));
    try std.testing.expectEqual(@as(usize, 5), pointFaceForDir(V(0.2, 0.1, -4)));
    // Ties prefer X, then Y, then Z (mirrors the GLSL pointFaceIndex).
    try std.testing.expectEqual(@as(usize, 0), pointFaceForDir(V(1, 1, 0)));
    try std.testing.expectEqual(@as(usize, 1), pointFaceForDir(V(-1, 1, 1)));
    try std.testing.expectEqual(@as(usize, 2), pointFaceForDir(V(0, 1, 1)));
    try std.testing.expectEqual(@as(usize, 3), pointFaceForDir(V(0, -1, -1)));
    try std.testing.expectEqual(@as(usize, 4), pointFaceForDir(V(0, 0, 1)));
}

test "pointTileOrigin tiles 2 slots x 6 faces inside the atlas without overlap" {
    try std.testing.expectEqual(@as(u32, 6 * POINT_SHADOW_RES), POINT_SHADOW_MAP_WIDTH);
    try std.testing.expectEqual(@as(u32, 2 * POINT_SHADOW_RES), POINT_SHADOW_MAP_HEIGHT);
    var seen: [POINT_SHADOW_SLOTS * POINT_SHADOW_FACES]@TypeOf(pointTileOrigin(0, 0)) = undefined;
    var n: usize = 0;
    for (0..POINT_SHADOW_SLOTS) |slot| {
        for (0..POINT_SHADOW_FACES) |face| {
            const o = pointTileOrigin(slot, face);
            try std.testing.expectEqual(@as(i32, @intCast(face)) * POINT_SHADOW_RES, o.x);
            try std.testing.expectEqual(@as(i32, @intCast(slot)) * POINT_SHADOW_RES, o.y);
            // Tile stays inside the atlas.
            try std.testing.expect(o.x >= 0 and o.x + POINT_SHADOW_RES <= POINT_SHADOW_MAP_WIDTH);
            try std.testing.expect(o.y >= 0 and o.y + POINT_SHADOW_RES <= POINT_SHADOW_MAP_HEIGHT);
            for (seen[0..n]) |prev| try std.testing.expect(prev.x != o.x or prev.y != o.y);
            seen[n] = o;
            n += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 12), n);
}

test "cascade size culling keeps the near cascade and culls small AABBs far" {
    // Cascade 0 never culls by size: near-camera shadows stay exact.
    try std.testing.expect(!cascadeSizeCulled(0, 0.0));
    try std.testing.expect(!cascadeSizeCulled(0, 0.001));
    // Spot/point paths (null cascade) never cull by size either.
    try std.testing.expect(!cascadeSizeCulled(null, 0.001));
    // Out-of-range cascade indices fail safe to no culling.
    try std.testing.expect(!cascadeSizeCulled(4, 0.001));
    // Per-cascade thresholds are strict less-than: equality still draws.
    try std.testing.expect(cascadeSizeCulled(1, 0.119));
    try std.testing.expect(!cascadeSizeCulled(1, 0.12));
    try std.testing.expect(cascadeSizeCulled(2, 0.349));
    try std.testing.expect(!cascadeSizeCulled(2, 0.35));
    try std.testing.expect(cascadeSizeCulled(3, 0.749));
    try std.testing.expect(!cascadeSizeCulled(3, 0.75));
    // Monotonic: a caster culled in cascade N is also culled further out.
    try std.testing.expect(cascadeSizeCulled(2, 0.1));
    try std.testing.expect(cascadeSizeCulled(3, 0.1));
    try std.testing.expect(!cascadeSizeCulled(0, 0.1));
}

test "shadow LOD gate draws full-res near and low-poly far" {
    // No stand-in: always full-res, even in the far cascade.
    try std.testing.expect(!shadowLodActive(false, 0));
    try std.testing.expect(!shadowLodActive(false, 3));
    try std.testing.expect(!shadowLodActive(false, null));
    // Near cascades always full-res, even with a stand-in.
    try std.testing.expect(!shadowLodActive(true, 0));
    try std.testing.expect(!shadowLodActive(true, 1));
    // Distant CSM cascades use the stand-in.
    try std.testing.expect(shadowLodActive(true, 2));
    try std.testing.expect(shadowLodActive(true, 3));
    // Spot/point paths (null cascade) always full-res.
    try std.testing.expect(!shadowLodActive(true, null));
}

test "shadowLodMesh fails safe to high-poly without a valid stand-in" {
    const ally = std.testing.allocator;
    var plain = Mesh{
        .name = "lod_plain",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 300,
    };
    // No LOD levels at all.
    try std.testing.expect(shadowLodMesh(&plain) == null);

    // Pure cull marker (null mesh): nothing to draw.
    try plain.addLODLevel(ally, 100.0, null);
    defer plain.lod_levels.deinit(ally);
    try std.testing.expect(shadowLodMesh(&plain) == null);
}

test "shadowLodMesh picks the coarsest decimated child" {
    const ally = std.testing.allocator;
    var src = Mesh{
        .name = "lod_src",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 300,
    };
    defer src.lod_levels.deinit(ally);
    var mid = Mesh{
        .name = "lod_mid",
        .vertex_buffer = .{ .id = 11 },
        .index_buffer = .{ .id = 12 },
        .index_count = 150,
    };
    var coarse = Mesh{
        .name = "lod_coarse",
        .vertex_buffer = .{ .id = 13 },
        .index_buffer = .{ .id = 14 },
        .index_count = 60,
    };
    try src.addLODLevel(ally, 20.0, &mid);
    try src.addLODLevel(ally, 50.0, &coarse);
    const picked = shadowLodMesh(&src) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@intFromPtr(&coarse), @intFromPtr(picked));

    // A heavier-or-equal "LOD" is ignored in favor of another valid child.
    coarse.index_count = 300;
    try std.testing.expectEqual(@intFromPtr(&mid), @intFromPtr(shadowLodMesh(&src).?));
    coarse.index_count = 60;

    // A pending (not yet uploaded) child is ignored; keep a valid fallback.
    coarse.gpu_pending = true;
    try std.testing.expectEqual(@intFromPtr(&mid), @intFromPtr(shadowLodMesh(&src).?));
    coarse.gpu_pending = false;

    // Index-type mismatch would draw garbage through the source bucket
    // pipeline, so keep the valid mid child instead.
    coarse.index_type = .UINT32;
    try std.testing.expectEqual(@intFromPtr(&mid), @intFromPtr(shadowLodMesh(&src).?));
    coarse.index_type = .UINT16;
    try std.testing.expect(shadowLodMesh(&src) != null);

    // Missing GPU handles are not usable stand-ins.
    coarse.index_buffer = .{};
    try std.testing.expectEqual(@intFromPtr(&mid), @intFromPtr(shadowLodMesh(&src).?));

    // A morph source would freeze the unmorphed shape in the stand-in:
    // fall back to high-poly.
    var dummy_morph = [_]mesh_mod.MorphTarget{.{}};
    src.morph_targets = &dummy_morph;
    try std.testing.expect(shadowLodMesh(&src) == null);
    src.morph_targets = &.{};
    try std.testing.expect(shadowLodMesh(&src) != null);
}
