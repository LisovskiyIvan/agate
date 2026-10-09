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
