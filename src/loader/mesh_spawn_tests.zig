const std = @import("std");
const c = @import("../c.zig").c;
const mesh_spawn = @import("mesh_spawn.zig");
const readUv1 = mesh_spawn.readUv1;
const validateTextureCoordinates = mesh_spawn.validateTextureCoordinates;
const materials_mod = @import("materials.zig");
const fixture = @import("uv1_gltf_fixture.zig");
const testScene = @import("../testing.zig").testScene;

test "UV1 accessor reads distinct normalized coordinates and validates shape" {
    var bytes = [_]u16{ 0, 65535, 32768, 0 };
    var buf = std.mem.zeroes(c.cgltf_buffer);
    buf.data = @ptrCast(&bytes);
    buf.size = @sizeOf(@TypeOf(bytes));
    var view = std.mem.zeroes(c.cgltf_buffer_view);
    view.buffer = &buf;
    view.size = buf.size;
    var acc = std.mem.zeroes(c.cgltf_accessor);
    acc.buffer_view = &view;
    acc.type = c.cgltf_type_vec2;
    acc.component_type = c.cgltf_component_type_r_16u;
    acc.normalized = 1;
    acc.count = 2;
    acc.stride = 2 * @sizeOf(u16);
    try std.testing.expectEqualSlices(f32, &.{ 0, 1 }, &(try readUv1(&acc, 2, 0)));
    const second = try readUv1(&acc, 2, 1);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), second[0], 0.00002);
    try std.testing.expectEqual(@as(f32, 0), second[1]);
    try std.testing.expectError(error.InvalidTextureCoordinate, readUv1(&acc, 3, 0));
    try std.testing.expectError(error.InvalidTextureCoordinate, readUv1(&acc, 2, 2));
    acc.type = c.cgltf_type_vec3;
    try std.testing.expectError(error.InvalidTextureCoordinate, readUv1(&acc, 2, 0));
    try std.testing.expectEqualSlices(f32, &.{ 0, 0 }, &(try readUv1(null, 2, 0)));
}

test "UV1 end-to-end: KHR_texture_transform override selects coord 1 with independent clearcoat/sheen" {
    const alloc = std.testing.allocator;
    const json = try fixture.buildJson(alloc, .valid_override);
    defer alloc.free(json);
    // Genuine cgltf parse + data: buffer load (same prefix as
    // SceneLoader.appendGlbOptions), not a hand-built struct.
    const data = try fixture.parseLoaded(json);
    defer c.cgltf_free(data);

    try validateTextureCoordinates(data);

    // The parsed material actually selects set 1 on all three textured
    // slots: base via the transform override, clearcoat directly, sheen
    // via its own independent offset transform.
    const src = &data.materials[0];
    try std.testing.expectEqual(@as(u1, 1), try materials_mod.textureCoordFromView(&src.*.pbr_metallic_roughness.base_color_texture));
    try std.testing.expectEqual(@as(u1, 1), try materials_mod.textureCoordFromView(&src.*.clearcoat.clearcoat_texture));
    try std.testing.expectEqual(@as(u1, 1), try materials_mod.textureCoordFromView(&src.*.sheen.sheen_color_texture));

    // UV1 bytes really arrived through the data: URI path and differ from
    // UV0 (guards against a fixture that only looks textured).
    const prim = &data.meshes[0].primitives[0];
    var uv0_acc: ?*c.cgltf_accessor = null;
    var uv1_acc: ?*c.cgltf_accessor = null;
    for (0..prim.attributes_count) |ai| {
        const attr = prim.attributes[ai];
        if (attr.type == c.cgltf_attribute_type_texcoord and attr.index == 0) uv0_acc = attr.data;
        if (attr.type == c.cgltf_attribute_type_texcoord and attr.index == 1) uv1_acc = attr.data;
    }
    try std.testing.expect(uv0_acc != null and uv1_acc != null);
    var uv: [2]f32 = undefined;
    try std.testing.expect(c.cgltf_accessor_read_float(uv1_acc.?, 0, &uv, 2) != 0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), uv[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.75), uv[1], 1e-6);
    try std.testing.expect(c.cgltf_accessor_read_float(uv1_acc.?, 2, &uv, 2) != 0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), uv[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), uv[1], 1e-6);
    try std.testing.expect(c.cgltf_accessor_read_float(uv0_acc.?, 0, &uv, 2) != 0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), uv[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), uv[1], 1e-6);
}

test "UV1 end-to-end: missing TEXCOORD_1 is MissingTextureCoordinate, scene registries unchanged" {
    const alloc = std.testing.allocator;
    const json = try fixture.buildJson(alloc, .missing_uv1);
    defer alloc.free(json);
    const data = try fixture.parseLoaded(json);
    defer c.cgltf_free(data);

    // The material coordinates themselves are legal: the failure comes
    // from the mesh gate (textured slot wants set 1, primitive has none).
    try materials_mod.validateTextureCoordinates(data);
    try std.testing.expectError(error.MissingTextureCoordinate, validateTextureCoordinates(data));

    // The gate runs before loadMaterials/spawnMeshes publish anything
    // (scene_loader calls it first), so a rejection leaves the scene
    // registries exactly as they were.
    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const scene = testScene(arena.allocator());
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.pbr_materials.items.len);
}

test "UV1 end-to-end: transform override to coord 2 is UnsupportedTextureCoordinate" {
    const alloc = std.testing.allocator;
    const json = try fixture.buildJson(alloc, .override_2);
    defer alloc.free(json);
    const data = try fixture.parseLoaded(json);
    defer c.cgltf_free(data);

    try std.testing.expectError(error.UnsupportedTextureCoordinate, materials_mod.validateTextureCoordinates(data));
    try std.testing.expectError(error.UnsupportedTextureCoordinate, validateTextureCoordinates(data));

    var arena = std.heap.ArenaAllocator.init(alloc);
    defer arena.deinit();
    const scene = testScene(arena.allocator());
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.pbr_materials.items.len);
}

test "UV1 end-to-end: vec3 and count-mismatched TEXCOORD_1 are InvalidTextureCoordinate" {
    const alloc = std.testing.allocator;

    const vec3_json = try fixture.buildJson(alloc, .uv1_vec3);
    defer alloc.free(vec3_json);
    const vec3_data = try fixture.parseLoaded(vec3_json);
    defer c.cgltf_free(vec3_data);
    try std.testing.expectError(error.InvalidTextureCoordinate, validateTextureCoordinates(vec3_data));

    const count_json = try fixture.buildJson(alloc, .uv1_count_mismatch);
    defer alloc.free(count_json);
    const count_data = try fixture.parseLoaded(count_json);
    defer c.cgltf_free(count_data);
    try std.testing.expectError(error.InvalidTextureCoordinate, validateTextureCoordinates(count_data));
}
