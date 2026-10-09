const std = @import("std");
const math = @import("math");
const mesh_mod = @import("../../mesh.zig");
const Mesh = mesh_mod.Mesh;
const types = @import("types.zig");

const POINT_SHADOW_SLOTS = types.POINT_SHADOW_SLOTS;
const POINT_SHADOW_FACES = types.POINT_SHADOW_FACES;
const POINT_SHADOW_RES = types.POINT_SHADOW_RES;
const POINT_SHADOW_MAP_WIDTH = types.POINT_SHADOW_MAP_WIDTH;
const POINT_SHADOW_MAP_HEIGHT = types.POINT_SHADOW_MAP_HEIGHT;
const pointFaceForDir = types.pointFaceForDir;
const pointTileOrigin = types.pointTileOrigin;
const cascadeSizeCulled = types.cascadeSizeCulled;
const shadowLodActive = types.shadowLodActive;
const shadowLodMesh = types.shadowLodMesh;

test "pointFaceForDir selects the major-axis face with X>Y>Z tie-break" {
    const V = math.Vec3.new;
    try std.testing.expectEqual(@as(usize, 0), pointFaceForDir(V(1, 0, 0)));
    try std.testing.expectEqual(@as(usize, 1), pointFaceForDir(V(-2, 0.5, 0.5)));
    try std.testing.expectEqual(@as(usize, 2), pointFaceForDir(V(0.1, 3, 0.1)));
    try std.testing.expectEqual(@as(usize, 3), pointFaceForDir(V(0, -1, 0)));
    try std.testing.expectEqual(@as(usize, 4), pointFaceForDir(V(0, 0, 5)));
    try std.testing.expectEqual(@as(usize, 5), pointFaceForDir(V(0.2, 0.1, -4)));
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
    try std.testing.expect(!cascadeSizeCulled(0, 0.0));
    try std.testing.expect(!cascadeSizeCulled(0, 0.001));
    try std.testing.expect(!cascadeSizeCulled(null, 0.001));
    try std.testing.expect(!cascadeSizeCulled(4, 0.001));
    try std.testing.expect(cascadeSizeCulled(1, 0.119));
    try std.testing.expect(!cascadeSizeCulled(1, 0.12));
    try std.testing.expect(cascadeSizeCulled(2, 0.349));
    try std.testing.expect(!cascadeSizeCulled(2, 0.35));
    try std.testing.expect(cascadeSizeCulled(3, 0.749));
    try std.testing.expect(!cascadeSizeCulled(3, 0.75));
    try std.testing.expect(cascadeSizeCulled(2, 0.1));
    try std.testing.expect(cascadeSizeCulled(3, 0.1));
    try std.testing.expect(!cascadeSizeCulled(0, 0.1));
}

test "shadow LOD gate draws full-res near and low-poly far" {
    try std.testing.expect(!shadowLodActive(false, 0));
    try std.testing.expect(!shadowLodActive(false, 3));
    try std.testing.expect(!shadowLodActive(false, null));
    try std.testing.expect(!shadowLodActive(true, 0));
    try std.testing.expect(!shadowLodActive(true, 1));
    try std.testing.expect(shadowLodActive(true, 2));
    try std.testing.expect(shadowLodActive(true, 3));
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
    try std.testing.expect(shadowLodMesh(&plain) == null);

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

    coarse.index_count = 300;
    try std.testing.expectEqual(@intFromPtr(&mid), @intFromPtr(shadowLodMesh(&src).?));
    coarse.index_count = 60;

    coarse.gpu_pending = true;
    try std.testing.expectEqual(@intFromPtr(&mid), @intFromPtr(shadowLodMesh(&src).?));
    coarse.gpu_pending = false;

    coarse.index_type = .UINT32;
    try std.testing.expectEqual(@intFromPtr(&mid), @intFromPtr(shadowLodMesh(&src).?));
    coarse.index_type = .UINT16;
    try std.testing.expect(shadowLodMesh(&src) != null);

    coarse.index_buffer = .{};
    try std.testing.expectEqual(@intFromPtr(&mid), @intFromPtr(shadowLodMesh(&src).?));

    var dummy_morph = [_]mesh_mod.MorphTarget{.{}};
    src.morph_targets = &dummy_morph;
    try std.testing.expect(shadowLodMesh(&src) == null);
    src.morph_targets = &.{};
    try std.testing.expect(shadowLodMesh(&src) != null);
}
