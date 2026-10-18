const std = @import("std");
const sky_layer = @import("sky_layer.zig");
const SkyboxLayer = sky_layer.SkyboxLayer;
const CubeTexture = @import("../texture.zig").CubeTexture;

test "skybox layer state toggles with setSkybox" {
    var sky: SkyboxLayer = .{ .pass = undefined };
    try std.testing.expect(!sky.enabled);
    try std.testing.expect(sky.texture == null);

    const cube: CubeTexture = undefined;
    sky.setSkybox(cube);
    try std.testing.expect(sky.enabled);
    try std.testing.expect(sky.texture != null);
}
