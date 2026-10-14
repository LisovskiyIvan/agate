const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;

const material = @import("../../material.zig");
const Material = material.Material;
const items_mod = @import("items.zig");
const RenderMeshItem = items_mod.RenderMeshItem;
const materialIsTransparent = items_mod.materialIsTransparent;
const materialIsCutout = items_mod.materialIsCutout;
const materialIsDoubleSided = items_mod.materialIsDoubleSided;
const sortRenderItems = items_mod.sortRenderItems;
const lessRenderItems = items_mod.lessRenderItems;
const sortOpaqueItemsIndexed = items_mod.sortOpaqueItemsIndexed;
const blendDescFor = items_mod.blendDescFor;
const skinAt = items_mod.skinAt;
const coatAt = items_mod.coatAt;
const MAX_BONES = items_mod.MAX_BONES;

test "cutout stays opaque, blend stays transparent" {
    try std.testing.expect(!materialIsTransparent(null));
    try std.testing.expect(!materialIsCutout(null));
    try std.testing.expect(!materialIsDoubleSided(null));

    var pbr_mat = material.PBRMaterial.init("p");

    try std.testing.expect(!materialIsTransparent(.{ .pbr = &pbr_mat }));
    try std.testing.expect(!materialIsCutout(.{ .pbr = &pbr_mat }));

    pbr_mat.alpha_mode = .cutout;
    try std.testing.expect(!materialIsTransparent(.{ .pbr = &pbr_mat }));
    try std.testing.expect(materialIsCutout(.{ .pbr = &pbr_mat }));

    pbr_mat.alpha_mode = .blend;
    try std.testing.expect(materialIsTransparent(.{ .pbr = &pbr_mat }));
    try std.testing.expect(!materialIsCutout(.{ .pbr = &pbr_mat }));

    try std.testing.expect(!materialIsDoubleSided(.{ .pbr = &pbr_mat }));
    pbr_mat.double_sided = true;
    try std.testing.expect(materialIsDoubleSided(.{ .pbr = &pbr_mat }));
}

test "transparent classification follows material alpha mode" {
    try std.testing.expect(!materialIsTransparent(null));
    var pbr_mat = material.PBRMaterial.init("p");
    try std.testing.expect(!materialIsTransparent(Material{ .pbr = &pbr_mat }));
    pbr_mat.alpha_mode = .blend;
    try std.testing.expect(materialIsTransparent(Material{ .pbr = &pbr_mat }));
}

test "opaque sort unchanged: state groups, front-to-back" {
    var items = [_]RenderMeshItem{
        .{ .model = Mat4.identity, .distance_sq = 9.0, .is_pbr = false, .texture_id = 2 },
        .{ .model = Mat4.identity, .distance_sq = 1.0, .is_pbr = false, .texture_id = 2 },
        .{ .model = Mat4.identity, .distance_sq = 5.0, .is_pbr = true, .texture_id = 1 },
        .{ .model = Mat4.identity, .distance_sq = 3.0, .is_pbr = false, .texture_id = 1 },
    };
    try std.testing.expect(!items[0].transparent);
    std.mem.sort(RenderMeshItem, &items, {}, sortRenderItems);
    try std.testing.expect(!items[0].is_pbr and items[0].texture_id == 1 and items[0].distance_sq == 3.0);
    try std.testing.expect(!items[1].is_pbr and items[1].texture_id == 2 and items[1].distance_sq == 1.0);
    try std.testing.expect(!items[2].is_pbr and items[2].texture_id == 2 and items[2].distance_sq == 9.0);
    try std.testing.expect(items[3].is_pbr and items[3].distance_sq == 5.0);
}

test "blendDescFor enables alpha blending without depth write" {
    const base = sg.PipelineDesc{
        .shader = .{},
        .index_type = .UINT32,
        .depth = .{ .compare = .LESS_EQUAL, .write_enabled = true },
        .cull_mode = .BACK,
        .face_winding = .CCW,
    };
    const blended = blendDescFor(base);
    try std.testing.expect(blended.colors[0].blend.enabled);
    try std.testing.expect(blended.colors[0].blend.src_factor_rgb == .SRC_ALPHA);
    try std.testing.expect(blended.colors[0].blend.dst_factor_rgb == .ONE_MINUS_SRC_ALPHA);
    try std.testing.expect(!blended.depth.write_enabled);
    try std.testing.expect(blended.depth.compare == .LESS_EQUAL);
    try std.testing.expect(blended.index_type == .UINT32);
    try std.testing.expect(blended.cull_mode == .BACK);
    try std.testing.expect(!base.colors[0].blend.enabled);
    try std.testing.expect(base.depth.write_enabled);
}

test "P4: skinAt resolves owned copies without stale reads" {
    var empty: [0][MAX_BONES]Mat4 = .{};
    try std.testing.expect(skinAt(&empty, null) == null);
    try std.testing.expect(skinAt(&empty, 0) == null);
    try std.testing.expect(skinAt(&empty, std.math.maxInt(u32)) == null);

    var one: [1][MAX_BONES]Mat4 = [_][MAX_BONES]Mat4{[_]Mat4{Mat4.identity} ** MAX_BONES};
    one[0][3] = Mat4.translation(Vec3.new(2, 0, 0));
    const got = skinAt(&one, 0) orelse return error.TestUnexpectedResult;
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), got[3].m[12], 1e-6);
    try std.testing.expect(skinAt(&one, 1) == null);
    try std.testing.expect(skinAt(&one, std.math.maxInt(u32)) == null);
}

test "coatAt resolves owned copies without stale reads" {
    const CoatParams = material.CoatParams;
    var empty: [0]CoatParams = .{};
    try std.testing.expect(coatAt(&empty, null) == null);
    try std.testing.expect(coatAt(&empty, 0) == null);

    var one = [_]CoatParams{.{ .clearcoat_factors = .{ 0.8, 0.12, 0, 0 } }};
    const got = coatAt(&one, 0) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualSlices(f32, &.{ 0.8, 0.12, 0, 0 }, &got.clearcoat_factors);
    try std.testing.expect(coatAt(&one, 1) == null);
    try std.testing.expect(coatAt(&one, null) == null);
    const fallback = coatAt(&one, null) orelse &CoatParams.neutral;
    try std.testing.expectEqual(@as(f32, 0.0), fallback.clearcoat_factors[0]);
    try std.testing.expectEqual(@as(f32, 0.0), fallback.sheen_factors[0]);
}

fn benchItem(i: usize, state: *u64) RenderMeshItem {
    // Deterministic LCG: small texture_id range forces key ties, distances
    // spread, is_pbr mixes (full-tie stress for the unstable sort comes
    // from the forced equal run in the test body below).
    state.* = state.* *% 6364136223846793005 +% 1442695040888963407;
    const r = state.*;
    const tex: u32 = @intCast((r >> 17) % 5);
    const dist: f32 = @floatFromInt((r >> 33) % 100000);
    const pbr = ((r >> 7) & 1) == 0;
    return RenderMeshItem{
        .model = Mat4.identity,
        .distance_sq = dist,
        .is_pbr = pbr,
        .texture_id = tex,
        .mesh_index = @intCast(i),
    };
}

fn isSortedOpaque(list: []const RenderMeshItem) bool {
    for (0..list.len) |i| {
        for (i + 1..list.len) |j| {
            if (lessRenderItems(&list[j], &list[i])) return false;
        }
    }
    return true;
}

test "indexed opaque sort matches direct sort bit-identically" {
    const ally = std.testing.allocator;
    const RenderQueues = items_mod.RenderQueues;
    const n: usize = 1500;

    var seed: u64 = 0x12345678;
    const src = try ally.alloc(RenderMeshItem, n);
    defer ally.free(src);
    for (src, 0..) |*it, i| it.* = benchItem(i, &seed);
    // Force a run of fully-equal keys (same state group + distance).
    for (src[100..120]) |*it| {
        it.is_pbr = false;
        it.texture_id = 3;
        it.distance_sq = 42.0;
    }

    const direct = try ally.dupe(RenderMeshItem, src);
    defer ally.free(direct);
    std.mem.sort(RenderMeshItem, direct, {}, sortRenderItems);

    var queues = RenderQueues{};
    defer queues.deinit(ally);
    try queues.items.appendSlice(ally, src);
    sortOpaqueItemsIndexed(&queues, ally);
    const got = queues.items.items;

    try std.testing.expectEqual(direct.len, got.len);
    // Bit-identical order (mesh_index distinguishes full ties).
    for (direct, got) |a, b| {
        try std.testing.expectEqualDeep(a, b);
        try std.testing.expectEqual(a.mesh_index, b.mesh_index);
        try std.testing.expectEqual(a.texture_id, b.texture_id);
        try std.testing.expectEqual(a.distance_sq, b.distance_sq);
        try std.testing.expectEqual(a.is_pbr, b.is_pbr);
    }
    try std.testing.expect(isSortedOpaque(got));

    // Deterministic + scratch reuse: second run over fresh input matches.
    queues.reset();
    try queues.items.appendSlice(ally, src);
    const cap_before = queues.sort_idx.capacity;
    sortOpaqueItemsIndexed(&queues, ally);
    try std.testing.expect(queues.sort_idx.capacity >= cap_before);
    for (direct, queues.items.items) |a, b| {
        try std.testing.expectEqual(a.mesh_index, b.mesh_index);
    }

    // Empty + singleton are no-ops (scratch untouched, no crash).
    queues.reset();
    sortOpaqueItemsIndexed(&queues, ally);
    try std.testing.expectEqual(@as(usize, 0), queues.items.items.len);
    try queues.items.append(ally, src[0]);
    sortOpaqueItemsIndexed(&queues, ally);
    try std.testing.expectEqual(@as(usize, 1), queues.items.items.len);
    try std.testing.expectEqual(src[0].mesh_index, queues.items.items[0].mesh_index);
}

test "indexed opaque sort OOM falls back to sorted output" {
    const ally = std.testing.allocator;
    const RenderQueues = items_mod.RenderQueues;
    var queues = RenderQueues{};
    defer queues.deinit(ally);
    var seed: u64 = 0xabcdef;
    for (0..64) |i| try queues.items.append(ally, benchItem(i, &seed));
    // Failing allocator forces the direct-sort fallback path.
    var failing = std.testing.FailingAllocator.init(ally, .{ .fail_index = 0 });
    sortOpaqueItemsIndexed(&queues, failing.allocator());
    try std.testing.expect(isSortedOpaque(queues.items.items));
    try std.testing.expectEqual(@as(usize, 64), queues.items.items.len);
}
