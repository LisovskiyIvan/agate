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
    view_proj: Mat4 = Mat4.identity,
};

pub const PointShadowRenderInfo = struct {
    tile_x: i32 = 0,
    tile_y: i32 = 0,
    view_proj: Mat4 = Mat4.identity,
};

const Mat4 = math.Mat4;

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
