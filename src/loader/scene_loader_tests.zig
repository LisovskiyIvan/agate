const std = @import("std");
const scene_loader_mod = @import("scene_loader.zig");
const isExtensionSupported = scene_loader_mod.isExtensionSupported;
const scene_registry = @import("../scene/registry.zig");
const testScene = @import("../testing.zig").testScene;
const testMesh = @import("../testing.zig").testMesh;
const Mesh = @import("../mesh.zig").Mesh;
const Texture = @import("../texture.zig").Texture;
const PBRMaterial = @import("../material.zig").PBRMaterial;

/// Headless stand-in for one loadGlb result (a real load needs a sg context
/// for the buffer spawn): the exact aliasing shape the glTF loader produces
/// — one image backing several material slots as copies of the same handle
/// set, meshes sharing one material. owns_handles=false keeps
/// Texture.deinit from touching sg (no context here); the live VRAM half of
/// the contract is proven by the wave-3 scenario `mem-churn-stable` check
/// (WAVE3_SCENARIO_2026-10-23.md).
fn fakeTex(id: u32) Texture {
    var t = std.mem.zeroes(Texture);
    t.image = .{ .id = id };
    t.view = .{ .id = id + 1 };
    t.sampler = .{ .id = id + 2 };
    t.owns_handles = false;
    return t;
}

test "GltfAsset.deinit frees meshes, materials and deduplicates shared textures" {
    const alloc = std.testing.allocator;
    var scene = testScene(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.pbr_materials.deinit(alloc);

    const m1 = try scene.createPBRMaterial("m1");
    const m2 = try scene.createPBRMaterial("m2");
    const shared = fakeTex(7);
    m1.albedo_texture = shared;
    m1.normal_texture = shared; // same handles in another slot
    m2.emissive_texture = shared; // same handles in another material
    m2.albedo_texture = fakeTex(17);

    const mesh_a = try alloc.create(Mesh);
    mesh_a.* = testMesh("a");
    mesh_a.material = .{ .pbr = m1 };
    const mesh_b = try alloc.create(Mesh);
    mesh_b.* = testMesh("b");
    mesh_b.material = .{ .pbr = m1 };
    const mesh_c = try alloc.create(Mesh);
    mesh_c.* = testMesh("c");
    mesh_c.material = .{ .pbr = m2 };
    try scene.meshes.append(alloc, mesh_a);
    try scene.meshes.append(alloc, mesh_b);
    try scene.meshes.append(alloc, mesh_c);

    const meshes = try alloc.dupe(*Mesh, &.{ mesh_a, mesh_b, mesh_c });
    const mats = try alloc.dupe(*PBRMaterial, &.{ m1, m2 });
    const asset = try alloc.create(scene_loader_mod.SceneLoader.GltfAsset);
    asset.* = .{ .scene = &scene, .meshes = meshes, .materials = mats };
    try std.testing.expectEqual(@as(usize, 2), scene.pbr_materials.items.len);
    try std.testing.expectEqual(@as(usize, 3), scene.meshes.items.len);

    // deinit frees the slices, the meshes, the loader materials (each
    // distinct one exactly once even when shared) and the textures (each
    // distinct handle set exactly once even when aliased across
    // slots/materials).
    asset.deinit();
    try std.testing.expectEqual(@as(usize, 0), scene.pbr_materials.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);

    // Second cycle: counts stay flat — load/unload cycles must not grow
    // scene inventory.
    const m3 = try scene.createPBRMaterial("m3");
    m3.albedo_texture = shared;
    const mesh_d = try alloc.create(Mesh);
    mesh_d.* = testMesh("d");
    mesh_d.material = .{ .pbr = m3 };
    try scene.meshes.append(alloc, mesh_d);
    const asset2 = try alloc.create(scene_loader_mod.SceneLoader.GltfAsset);
    asset2.* = .{
        .scene = &scene,
        .meshes = try alloc.dupe(*Mesh, &.{mesh_d}),
        .materials = try alloc.dupe(*PBRMaterial, &.{m3}),
    };
    asset2.deinit();
    try std.testing.expectEqual(@as(usize, 0), scene.pbr_materials.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
}

test "Scene.deinit unloads registered assets the caller forgot" {
    const alloc = std.testing.allocator;
    var scene = testScene(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.pbr_materials.deinit(alloc);

    const m1 = try scene.createPBRMaterial("m1");
    m1.albedo_texture = fakeTex(7);
    const mesh_a = try alloc.create(Mesh);
    mesh_a.* = testMesh("a");
    mesh_a.material = .{ .pbr = m1 };
    try scene.meshes.append(alloc, mesh_a);

    const asset = try alloc.create(scene_loader_mod.SceneLoader.GltfAsset);
    asset.* = .{
        .scene = &scene,
        .meshes = try alloc.dupe(*Mesh, &.{mesh_a}),
        .materials = try alloc.dupe(*PBRMaterial, &.{m1}),
    };
    // Registered assets are unloaded by the scene teardown path even when
    // the caller never calls deinit (the "forgetting must not leak" rule).
    const Thunk = struct {
        fn run(ctx: *anyopaque) void {
            const self: *scene_loader_mod.SceneLoader.GltfAsset = @ptrCast(@alignCast(ctx));
            self.deinit();
        }
    };
    try scene.registerLoadedAsset(asset, Thunk.run);
    try std.testing.expectEqual(@as(usize, 1), scene.loaded_assets.items.len);

    scene_registry.deinitLoadedAssets(&scene);
    try std.testing.expectEqual(@as(usize, 0), scene.loaded_assets.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.pbr_materials.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);
}

test "isExtensionSupported accepts engine extensions and rejects unsupported" {
    try std.testing.expect(isExtensionSupported("EXT_meshopt_compression"));
    try std.testing.expect(isExtensionSupported("KHR_mesh_quantization"));
    try std.testing.expect(isExtensionSupported("KHR_lights_punctual"));
    try std.testing.expect(isExtensionSupported("KHR_texture_transform"));
    try std.testing.expect(isExtensionSupported("KHR_materials_unlit"));
    try std.testing.expect(isExtensionSupported("KHR_materials_clearcoat"));
    try std.testing.expect(isExtensionSupported("KHR_materials_sheen"));
    try std.testing.expect(isExtensionSupported("KHR_materials_transmission"));
    try std.testing.expect(isExtensionSupported("KHR_materials_ior"));
    // KTX2 images ride on KHR_texture_basisu: required-and-supported
    // files load through the basisu resolve path (see
    // materials.textureImage), so the gate must accept it.
    try std.testing.expect(isExtensionSupported("KHR_texture_basisu"));

    // Unsupported extensions that must be rejected when required:
    try std.testing.expect(!isExtensionSupported("KHR_draco_mesh_compression"));
    try std.testing.expect(!isExtensionSupported("KHR_materials_volume"));
    try std.testing.expect(!isExtensionSupported("KHR_materials_specular"));
    try std.testing.expect(!isExtensionSupported("KHR_texture_basisu_extra"));
    try std.testing.expect(!isExtensionSupported("UNKNOWN_extension"));
}
