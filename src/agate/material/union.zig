const std = @import("std");
const math = @import("math");
const Color3 = math.Color3;
const Texture = @import("../texture.zig").Texture;
const types = @import("types.zig");
const AlphaMode = types.AlphaMode;
const CoatParams = types.CoatParams;
const StandardMaterial = @import("standard.zig").StandardMaterial;
const PBRMaterial = @import("pbr.zig").PBRMaterial;
const ShaderMaterial = @import("shader_mat.zig").ShaderMaterial;

pub const Material = union(enum) {
    standard: *StandardMaterial,
    pbr: *PBRMaterial,
    shader_material: *ShaderMaterial,

    pub fn isTransparent(self: Material) bool {
        return switch (self) {
            inline else => |m| m.isTransparent(),
        };
    }

    pub fn isCutout(self: Material) bool {
        return switch (self) {
            inline else => |m| m.isCutout(),
        };
    }

    pub fn isDoubleSided(self: Material) bool {
        return switch (self) {
            inline else => |m| m.double_sided,
        };
    }

    pub fn isUnlit(self: Material) bool {
        return switch (self) {
            .standard => |s| s.unlit,
            .pbr => |p| p.unlit,
            .shader_material => false,
        };
    }

    pub fn setUnlit(self: *Material, unlit_val: bool) void {
        switch (self.*) {
            .standard => |s| s.unlit = unlit_val,
            .pbr => |p| p.unlit = unlit_val,
            .shader_material => {},
        }
    }

    /// Raw per-material cutoff (defaults to 0.5). The draw path gates this:
    /// only cutout materials upload a live cutoff, otherwise 0.0 (disabled).
    /// See scene/uniforms.zig alphaCutoffFor.
    pub fn alphaCutoff(self: Material) f32 {
        return switch (self) {
            inline else => |m| m.alpha_cutoff,
        };
    }

    /// Display/registration name. Every variant must carry `name` — the
    /// `inline else` enforces the convention at compile time when a new
    /// variant is added.
    pub fn name(self: Material) []const u8 {
        return switch (self) {
            inline else => |m| m.name,
        };
    }

    /// The single scalar alpha. Same compile-time convention as `name`.
    pub fn alpha(self: Material) f32 {
        return switch (self) {
            inline else => |m| m.alpha,
        };
    }

    /// Alpha blending mode. Same compile-time convention as `name`.
    pub fn alphaMode(self: Material) AlphaMode {
        return switch (self) {
            inline else => |m| m.alpha_mode,
        };
    }

    /// Semantic base color of the workflow: diffuse_color (standard /
    /// specular-glossiness), albedo_color (metallic-roughness), tint_color
    /// (shader material multiply). The storage names stay glTF-faithful per
    /// workflow; the mapping lives here so call sites never hand-switch.
    pub fn baseColor3(self: Material) Color3 {
        return switch (self) {
            .standard => |s| s.diffuse_color,
            .pbr => |p| p.albedo_color,
            .shader_material => |sm| sm.tint_color,
        };
    }

    /// The material's primary texture (diffuse/albedo slot), or null.
    pub fn primaryTexture(self: Material) ?Texture {
        return switch (self) {
            .standard => |s| s.diffuse_texture,
            .pbr => |p| p.albedo_texture,
            .shader_material => |sm| sm.texture,
        };
    }

    /// Tint * alpha as uploaded to the albedo color uniform.
    pub fn tintColor4(self: Material) [4]f32 {
        return switch (self) {
            .standard => |s| s.getDiffuseColor4(),
            .pbr => |p| p.getAlbedoColor4(),
            .shader_material => |sm| sm.getTintColor4(),
        };
    }
};

/// Builds the side-table snapshot for a material: non-null only for PBR
/// materials with an enabled layer (any intensity/factor/strength > 0).
/// Materials without these features (including every non-PBR material)
/// yield null and draw from CoatParams.neutral. Bound coat/sheen textures
/// alone do NOT opt in (a zero intensity multiplies the mask to zero, so
/// the legacy path stays exact and cheaper). Pure function (no GPU calls).
pub fn coatParamsFor(mat: ?Material) ?CoatParams {
    const m = mat orelse return null;
    if (m != .pbr) return null;
    const p = m.pbr;
    if (p.clearcoat.intensity <= 0 and p.sheen.intensity <= 0 and
        p.anisotropy.intensity <= 0 and p.transmission.factor <= 0 and
        p.subsurface.strength <= 0) return null;
    return .{
        .clearcoat_factors = .{ p.clearcoat.intensity, p.clearcoat.roughness, 0, 0 },
        .clearcoat_color = .{ p.clearcoat.color.r, p.clearcoat.color.g, p.clearcoat.color.b, 1.0 },
        .sheen_factors = .{ p.sheen.intensity, p.sheen.roughness, 0, 0 },
        .sheen_color = .{ p.sheen.color.r, p.sheen.color.g, p.sheen.color.b, 1.0 },
        .anisotropy_factors = .{ p.anisotropy.intensity, p.anisotropy.rotation, 0, 0 },
        .transmission_factors = .{ p.transmission.factor, 0, 0, 0 },
        .transmission_color = .{ p.transmission.color.r, p.transmission.color.g, p.transmission.color.b, 1.0 },
        .sss_factors = .{ p.subsurface.strength, 0, 0, 0 },
        .sss_color = .{ p.subsurface.color.r, p.subsurface.color.g, p.subsurface.color.b, 1.0 },
        .clearcoat_uv_matrix = (p.clearcoat.uv_transform orelse p.albedo_uv_transform).matrixRows(),
        .clearcoat_uv_offset = (p.clearcoat.uv_transform orelse p.albedo_uv_transform).offsetPacked(),
        .sheen_uv_matrix = (p.sheen.uv_transform orelse p.albedo_uv_transform).matrixRows(),
        .sheen_uv_offset = (p.sheen.uv_transform orelse p.albedo_uv_transform).offsetPacked(),
        .refraction_factors = p.transmission.refractionPacked(),
    };
}
