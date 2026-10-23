const std = @import("std");
const scene_loader_mod = @import("scene_loader.zig");
const isExtensionSupported = scene_loader_mod.isExtensionSupported;
const testScene = @import("../testing.zig").testScene;
const testMesh = @import("../testing.zig").testMesh;
const Mesh = @import("../mesh.zig").Mesh;
const Texture = @import("../texture.zig").Texture;

test "releaseGlb frees meshes, materials and deduplicates shared textures" {
    const alloc = std.testing.allocator;
    var scene = testScene(alloc);
    defer scene.meshes.deinit(alloc);
    defer scene.pbr_materials.deinit(alloc);

    // Headless stand-in for one appendGlb load (a real load needs a sg
    // context for the buffer spawn): the exact aliasing shape the glTF
    // loader produces — one image backing several material slots as
    // copies of the same handle set, meshes sharing one material.
    // owns_handles=false keeps Texture.deinit from touching sg (no context
    // here); the live VRAM half of the contract is proven by the wave-3
    // scenario `mem-churn-stable` check (WAVE3_SCENARIO_2026-10-23.md).
    const makeTex = struct {
        fn f(id: u32) Texture {
            var t = std.mem.zeroes(Texture);
            t.image = .{ .id = id };
            t.view = .{ .id = id + 1 };
            t.sampler = .{ .id = id + 2 };
            t.owns_handles = false;
            return t;
        }
    }.f;

    const m1 = try scene.createPBRMaterial("m1");
    const m2 = try scene.createPBRMaterial("m2");
    const shared = makeTex(7);
    m1.albedo_texture = shared;
    m1.normal_texture = shared; // same handles in another slot
    m2.emissive_texture = shared; // same handles in another material
    m2.albedo_texture = makeTex(17);

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
    try std.testing.expectEqual(@as(usize, 2), scene.pbr_materials.items.len);
    try std.testing.expectEqual(@as(usize, 3), scene.meshes.items.len);

    // releaseGlb frees the slice itself, the meshes, the loader materials
    // (each distinct one exactly once even when shared) and the textures
    // (each distinct handle set exactly once even when aliased across
    // slots/materials). A second load+release cycle must leave every count
    // at baseline — the no-monotonic-growth contract of load/unload churn.
    scene_loader_mod.SceneLoader.releaseGlb(&scene, meshes);
    try std.testing.expectEqual(@as(usize, 0), scene.pbr_materials.items.len);
    try std.testing.expectEqual(@as(usize, 0), scene.meshes.items.len);

    const m3 = try scene.createPBRMaterial("m3");
    m3.albedo_texture = shared;
    const mesh_d = try alloc.create(Mesh);
    mesh_d.* = testMesh("d");
    mesh_d.material = .{ .pbr = m3 };
    try scene.meshes.append(alloc, mesh_d);
    const second = try alloc.dupe(*Mesh, &.{mesh_d});
    scene_loader_mod.SceneLoader.releaseGlb(&scene, second);
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
