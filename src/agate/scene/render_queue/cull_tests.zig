//! Presented-velocity commit gates for `cull.zig`.
const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const Mesh = @import("../../mesh.zig").Mesh;
const skeleton_mod = @import("../../animation/skeleton.zig");
const Skeleton = skeleton_mod.Skeleton;
const items = @import("items.zig");
const RenderQueues = items.RenderQueues;
const cull = @import("cull.zig");
const morphNeedsDepthFallback = cull.morphNeedsDepthFallback;
const commitPresentedVelocityQueue = cull.commitPresentedVelocityQueue;
const resetPresentedVelocity = cull.resetPresentedVelocity;

test "morphNeedsDepthFallback gates on enabled flag and nonzero weights" {
    const morph_gpu_mod = @import("../../mesh/morph_gpu.zig");
    const off = morph_gpu_mod.VsUniforms{
        .weights0 = .{ 0, 0, 0, 0 },
        .weights1 = .{ 0, 0, 0, 0 },
        .params = .{ 0, 1, 1, 0 },
    };
    try std.testing.expect(!morphNeedsDepthFallback(off));
    // Enabled but all-zero weights: no displacement, rigid vector is exact.
    const armed_idle = morph_gpu_mod.VsUniforms{
        .weights0 = .{ 0, 0, 0, 0 },
        .weights1 = .{ 0, 0, 0, 0 },
        .params = .{ 1, 4, 4, 0 },
    };
    try std.testing.expect(!morphNeedsDepthFallback(armed_idle));
    const active = morph_gpu_mod.VsUniforms{
        .weights0 = .{ 0.5, 0, 0, 0 },
        .weights1 = .{ 0, 0, 0, 0 },
        .params = .{ 1, 4, 4, 0 },
    };
    try std.testing.expect(morphNeedsDepthFallback(active));
    const active_hi = morph_gpu_mod.VsUniforms{
        .weights0 = .{ 0, 0, 0, 0 },
        .weights1 = .{ 0, 0, 0, -1.0 },
        .params = .{ 1, 4, 4, 0 },
    };
    try std.testing.expect(morphNeedsDepthFallback(active_hi));
}

test "presented velocity commit stamps models/skins, reset zeroes newcomers" {
    const ally = std.testing.allocator;

    const skel = try Skeleton.init(ally, 1);
    defer skel.deinit();
    skel.bones[0].local_position = Vec3.new(4, 0, 0);
    skel.update();

    var mesh = Mesh{
        .name = "vel",
        .vertex_buffer = .{},
        .index_buffer = .{},
        .index_count = 3,
        .local_bounding_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1)),
    };
    mesh.skeleton = skel;
    const uid = mesh.ensureUid();
    const meshes = [_]*Mesh{&mesh};

    // Presented payload: model at x=10, skin copy with bone at x=7.
    var presented_skins = [_][items.MAX_BONES]Mat4{[_]Mat4{Mat4.identity} ** items.MAX_BONES};
    presented_skins[0][0] = Mat4.translation(Vec3.new(7, 0, 0));
    var presented_items = [_]items.RenderMeshItem{.{
        .model = Mat4.translation(Vec3.new(10, 0, 0)),
        .distance_sq = 0,
        .is_pbr = true,
        .texture_id = 1,
        .mesh_index = 0,
        .source_uid = uid,
        .skin_index = 0,
        .skin_source_uid = uid,
    }};

    var queues = RenderQueues{};
    defer queues.deinit(ally);
    try queues.items.appendSlice(ally, &presented_items);
    try queues.skin_storage.appendSlice(ally, &presented_skins);

    resetPresentedVelocity(&meshes, 9);
    commitPresentedVelocityQueue(&meshes, &queues, 9);
    try std.testing.expectEqual(@as(u64, 9), mesh.vel_presented_frame);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), mesh.prev_matrix.m[12], 1e-6);
    try std.testing.expectEqual(@as(u64, 9), skel.vel_presented_frame);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), skel.getPrevSkinMatrices()[0].m[12], 1e-6);

    // Repeating the same front commit is idempotent.
    skel.bones[0].local_position = Vec3.new(99, 0, 0);
    skel.update();
    resetPresentedVelocity(&meshes, 9);
    commitPresentedVelocityQueue(&meshes, &queues, 9);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), mesh.prev_matrix.m[12], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), skel.getPrevSkinMatrices()[0].m[12], 1e-6);

    // Removal between build and commit: the item's entity is gone from the
    // list — nothing is stamped, nothing panics. (A stale INDEX with a
    // still-listed uid resolves via the scan fallback; covered below.)
    const empty = [_]*Mesh{};
    resetPresentedVelocity(&empty, 10);
    commitPresentedVelocityQueue(&empty, &queues, 10);
    try std.testing.expectEqual(@as(u64, 9), mesh.vel_presented_frame);
    try std.testing.expectEqual(@as(u64, 9), skel.vel_presented_frame);

    // Reset returns every mesh to the never-presented (zero-motion) state.
    resetPresentedVelocity(&meshes, 11);
    try std.testing.expectEqual(std.math.maxInt(u64), mesh.vel_presented_frame);
    // Unpresented skeletons alias the current slot: zero motion by construction.
    const unpresented = Skeleton{ .allocator = ally, .bones = &.{} };
    try std.testing.expect(unpresented.getPrevSkinMatrices() == unpresented.getRenderSkinMatrices());
}

test "commit resolves by uid across same-length shuffles and removals" {
    const ally = std.testing.allocator;
    var ma = Mesh{ .name = "a", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 1 };
    var mb = Mesh{ .name = "b", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 1 };
    const ua = ma.ensureUid();
    const ub = mb.ensureUid();
    try std.testing.expect(ua != ub);

    var queues = RenderQueues{};
    defer queues.deinit(ally);
    // Build-time order [a, b]: item 0 is a's model, item 1 is b's.
    try queues.items.appendSlice(ally, &[_]items.RenderMeshItem{
        .{ .model = Mat4.translation(Vec3.new(1, 0, 0)), .distance_sq = 0, .is_pbr = true, .texture_id = 1, .mesh_index = 0, .source_uid = ua },
        .{ .model = Mat4.translation(Vec3.new(2, 0, 0)), .distance_sq = 0, .is_pbr = true, .texture_id = 1, .mesh_index = 1, .source_uid = ub },
    });

    // Same-length shuffle at commit time: [b, a]. Bare indices would
    // cross-stamp; uid resolve must land each model on its own entity.
    var shuffled = [_]*Mesh{ &mb, &ma };
    resetPresentedVelocity(&shuffled, 5);
    commitPresentedVelocityQueue(&shuffled, &queues, 5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ma.prev_matrix.m[12], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), mb.prev_matrix.m[12], 1e-6);
    try std.testing.expectEqual(@as(u64, 5), ma.vel_presented_frame);
    try std.testing.expectEqual(@as(u64, 5), mb.vel_presented_frame);

    // Removal: only b remains, at a different index. a's item resolves to
    // nothing (uid unknown) and is skipped; b's own item still finds b via
    // the scan fallback and stamps it — uid resolve, not index luck.
    var only_b = [_]*Mesh{&mb};
    resetPresentedVelocity(&only_b, 6);
    commitPresentedVelocityQueue(&only_b, &queues, 6);
    try std.testing.expectEqual(@as(u64, 5), ma.vel_presented_frame);
    try std.testing.expectEqual(@as(u64, 6), mb.vel_presented_frame);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), mb.prev_matrix.m[12], 1e-6);
}

test "commit stamps the LOD proxy skeleton, never the entity's" {
    const ally = std.testing.allocator;
    const skel_entity = try Skeleton.init(ally, 1);
    defer skel_entity.deinit();
    const skel_proxy = try Skeleton.init(ally, 1);
    defer skel_proxy.deinit();

    var entity = Mesh{ .name = "ent", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 1 };
    entity.skeleton = skel_entity;
    const entity_uid = entity.ensureUid();
    var proxy = Mesh{ .name = "lod", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 1 };
    proxy.skeleton = skel_proxy;
    const proxy_uid = proxy.ensureUid();
    const meshes = [_]*Mesh{ &entity, &proxy };

    var skins = [_][items.MAX_BONES]Mat4{[_]Mat4{Mat4.identity} ** items.MAX_BONES};
    skins[0][0] = Mat4.translation(Vec3.new(3, 0, 0));
    var queues = RenderQueues{};
    defer queues.deinit(ally);
    try queues.items.appendSlice(ally, &[_]items.RenderMeshItem{.{
        .model = Mat4.identity,
        .distance_sq = 0,
        .is_pbr = true,
        .texture_id = 1,
        .mesh_index = 0,
        .source_uid = entity_uid,
        .skin_index = 0,
        .skin_source_uid = proxy_uid,
    }});
    try queues.skin_storage.appendSlice(ally, &skins);

    resetPresentedVelocity(&meshes, 7);
    commitPresentedVelocityQueue(&meshes, &queues, 7);
    // Entity model stamped (source_uid), proxy skeleton stamped (skin uid).
    try std.testing.expectEqual(@as(u64, 7), entity.vel_presented_frame);
    try std.testing.expectEqual(std.math.maxInt(u64), skel_entity.vel_presented_frame);
    try std.testing.expectEqual(@as(u64, 7), skel_proxy.vel_presented_frame);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), skel_proxy.getPrevSkinMatrices()[0].m[12], 1e-6);

    // Absent-skeleton history: drop the proxy from the list. The proxy
    // skeleton object is unreachable through the list, so the reset cannot
    // clear it — it keeps its last committed generation. That staleness is
    // still safe: the draw-time generation gate refuses any prev whose
    // generation is not the last rendered one (see usePresentedPrev), so a
    // reattached skeleton can never replay ancient motion.
    const only_entity = [_]*Mesh{&entity};
    resetPresentedVelocity(&only_entity, 8);
    try std.testing.expectEqual(@as(u64, 7), skel_proxy.vel_presented_frame);
    try std.testing.expectEqual(std.math.maxInt(u64), entity.vel_presented_frame);
}

pub fn checkUidResolve(resolve: anytype) !void {
    var ma = Mesh{ .name = "a", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 1 };
    var mb = Mesh{ .name = "b", .vertex_buffer = .{}, .index_buffer = .{}, .index_count = 1 };
    const ua = ma.ensureUid();
    const ub = mb.ensureUid();
    const meshes = [_]*Mesh{ &ma, &mb };
    // Hint hit.
    try std.testing.expect(resolve(&meshes, 0, ua) == &ma);
    try std.testing.expect(resolve(&meshes, 1, ub) == &mb);
    // Hint miss, scan hit (shuffled).
    try std.testing.expect(resolve(&meshes, 0, ub) == &mb);
    // Unknown uid, OOB hint, and legacy zero uid.
    try std.testing.expect(resolve(&meshes, 0, 0xDEAD) == null);
    try std.testing.expect(resolve(&meshes, 99, ub) == &mb);
    try std.testing.expect(resolve(&meshes, 1, 0) == &mb);
    try std.testing.expect(resolve(&meshes, 99, 0) == null);
}
