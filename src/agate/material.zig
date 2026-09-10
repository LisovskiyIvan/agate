const sokol = @import("sokol");
const sg = sokol.gfx;
const std = @import("std");
const math = @import("math");
const Color3 = math.Color3;
const Color4 = math.Color4;
const Texture = @import("texture.zig").Texture;
const CubeTexture = @import("texture.zig").CubeTexture;

/// How a material blends with the framebuffer.
/// .@"opaque" keeps the legacy behavior (depth write on, no blending).
/// .blend enables standard alpha blending (SRC_ALPHA, ONE_MINUS_SRC_ALPHA)
/// with depth test on and depth write off; such meshes are drawn after all
/// opaque geometry, sorted back-to-front.
// Note: `opaque` is a Zig keyword, so the variant is spelled @"opaque".
pub const AlphaMode = enum {
    @"opaque",
    blend,
};

pub const StandardMaterial = struct {
    name: []const u8 = "StandardMaterial",
    diffuse_color: Color3 = Color3.white,
    alpha: f32 = 1.0,
    alpha_mode: AlphaMode = .@"opaque",
    diffuse_texture: ?Texture = null,

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
};

pub const PBRMaterial = struct {
    name: []const u8 = "PBRMaterial",
    albedo_color: Color3 = Color3.white,
    alpha: f32 = 1.0,
    alpha_mode: AlphaMode = .@"opaque",
    metallic: f32 = 0.0,
    roughness: f32 = 0.5,
    albedo_texture: ?Texture = null,

    normal_texture: ?Texture = null,
    metallic_roughness_texture: ?Texture = null,
    emissive_texture: ?Texture = null,
    emissive_color: Color3 = Color3.black,
    occlusion_texture: ?Texture = null,
    occlusion_strength: f32 = 1.0,

    environment_texture: ?CubeTexture = null,
    environment_intensity: f32 = 1.0,

    pub fn init(name: []const u8) PBRMaterial {
        return .{
            .name = name,
        };
    }

    pub fn getAlbedoColor4(self: PBRMaterial) [4]f32 {
        return .{ self.albedo_color.r, self.albedo_color.g, self.albedo_color.b, self.alpha };
    }

    pub fn getEmissiveColor4(self: PBRMaterial) [4]f32 {
        return .{ self.emissive_color.r, self.emissive_color.g, self.emissive_color.b, 1.0 };
    }

    pub fn isTransparent(self: PBRMaterial) bool {
        return self.alpha_mode == .blend;
    }
};

pub const Material = union(enum) {
    standard: *StandardMaterial,
    pbr: *PBRMaterial,

    pub fn isTransparent(self: Material) bool {
        return switch (self) {
            .standard => |s| s.isTransparent(),
            .pbr => |p| p.isTransparent(),
        };
    }
};

test "alpha_mode defaults to opaque (back-compat)" {
    const std_mat = StandardMaterial.init("m");
    try std.testing.expect(std_mat.alpha_mode == .@"opaque");
    try std.testing.expect(!std_mat.isTransparent());
    try std.testing.expect(std_mat.alpha == 1.0);

    const pbr_mat = PBRMaterial.init("p");
    try std.testing.expect(pbr_mat.alpha_mode == .@"opaque");
    try std.testing.expect(!pbr_mat.isTransparent());
    try std.testing.expect(pbr_mat.alpha == 1.0);
}

test "Material.isTransparent follows alpha_mode" {
    var std_mat = StandardMaterial.init("m");
    var pbr_mat = PBRMaterial.init("p");

    const m_std: Material = .{ .standard = &std_mat };
    const m_pbr: Material = .{ .pbr = &pbr_mat };
    try std.testing.expect(!m_std.isTransparent());
    try std.testing.expect(!m_pbr.isTransparent());

    std_mat.alpha_mode = .blend;
    pbr_mat.alpha_mode = .blend;
    try std.testing.expect(m_std.isTransparent());
    try std.testing.expect(m_pbr.isTransparent());
}
