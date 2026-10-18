const std = @import("std");
const scene_loader_mod = @import("scene_loader.zig");
const isExtensionSupported = scene_loader_mod.isExtensionSupported;

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
