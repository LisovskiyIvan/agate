const std = @import("std");
const scene_loader_mod = @import("scene_loader.zig");
const isExtensionSupported = scene_loader_mod.isExtensionSupported;
const LoadTimings = scene_loader_mod.LoadTimings;
const SceneLoader = scene_loader_mod.SceneLoader;

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

test "LoadTimings defaults to zero and LoadOptions stays default-compatible" {
    const t = LoadTimings{};
    try std.testing.expectEqual(@as(f64, 0), t.parse_ms);
    try std.testing.expectEqual(@as(f64, 0), t.total_ms);
    const o = SceneLoader.LoadOptions{};
    try std.testing.expect(o.timings == null);
}
