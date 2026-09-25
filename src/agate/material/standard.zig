const std = @import("std");
const math = @import("math");
const Color3 = math.Color3;
const Texture = @import("../texture.zig").Texture;
const types = @import("types.zig");
const AlphaMode = types.AlphaMode;
const UvTransform = types.UvTransform;

pub const StandardMaterial = struct {
    name: []const u8 = "StandardMaterial",
    diffuse_color: Color3 = Color3.white,
    alpha: f32 = 1.0,
    alpha_mode: AlphaMode = .@"opaque",
    /// Alpha-test threshold used only when alpha_mode == .cutout.
    /// The fragment shader discards fragments with alpha < alpha_cutoff.
    /// For opaque/blend the draw path uploads 0.0 so the test never fires.
    alpha_cutoff: f32 = 0.5,
    /// When true the mesh renders with face culling disabled (a cull-off
    /// pipeline twin). Both opaque and blend twins exist.
    double_sided: bool = false,
    /// When true, lighting calculations are bypassed: renders pure base color + emissive.
    unlit: bool = false,
    emissive_color: Color3 = Color3.black,
    diffuse_texture: ?Texture = null,
    /// KHR_texture_transform-style UV map for the diffuse slot (identity =
    /// unchanged sampling). glTF never produces StandardMaterials, so this
    /// is a manual-material API only.
    diffuse_uv_transform: UvTransform = .{},

    pub fn init(name: []const u8) StandardMaterial {
        return .{
            .name = name,
            .diffuse_color = Color3.white,
            .alpha = 1.0,
        };
    }

    pub fn getDiffuseColor4(self: StandardMaterial) [4]f32 {
        return .{ self.diffuse_color.r, self.diffuse_color.g, self.diffuse_color.b, self.alpha };
    }

    pub fn isTransparent(self: StandardMaterial) bool {
        return self.alpha_mode == .blend;
    }

    /// Cutout is NOT transparent: it stays in the opaque queue with depth
    /// writes on; only sub-cutoff fragments are discarded.
    pub fn isCutout(self: StandardMaterial) bool {
        return self.alpha_mode == .cutout;
    }
};
