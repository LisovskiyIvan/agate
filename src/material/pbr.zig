const std = @import("std");
const math = @import("math");
const Color3 = math.Color3;
const Texture = @import("../texture.zig").Texture;
const CubeTexture = @import("../texture.zig").CubeTexture;
const types = @import("types.zig");
const AlphaMode = types.AlphaMode;
const Channel = types.Channel;
const UvTransform = types.UvTransform;
const Clearcoat = types.Clearcoat;
const Sheen = types.Sheen;
const Anisotropy = types.Anisotropy;
const Transmission = types.Transmission;
const Subsurface = types.Subsurface;

/// Blinn-Phong `specularPower` -> GGX roughness (lobe match): the
/// normalized Blinn-Phong lobe `(n+2)/2 * (N.H)^n` matches the GGX
/// distribution with `alpha = sqrt(2/(n+2))`. Used to migrate legacy
/// Standard-material shininess to PBR; clamped to the engine's practical
/// roughness floor (0.05, same as glass) so huge exponents stay renderable.
pub fn roughnessFromSpecularPower(power: f32) f32 {
    return @max(0.05, @min(1.0, @sqrt(2.0 / (@max(1.0, power) + 2.0))));
}

const file_roughness = roughnessFromSpecularPower;

pub const PBRMaterial = struct {
    pub const roughnessFromSpecularPower = file_roughness;

    name: []const u8 = "PBRMaterial",
    albedo_color: Color3 = Color3.white,
    alpha: f32 = 1.0,
    alpha_mode: AlphaMode = .@"opaque",
    /// Alpha-test threshold used only when alpha_mode == .cutout.
    /// The fragment shader discards fragments with alpha < alpha_cutoff.
    /// For opaque/blend the draw path uploads 0.0 so the test never fires.
    alpha_cutoff: f32 = 0.5,
    /// When true the mesh renders with face culling disabled (a cull-off
    /// pipeline twin). Both opaque and blend twins exist.
    double_sided: bool = false,
    /// Babylon's `twoSidedLighting` (default FALSE, like Babylon's own
    /// `_twoSidedLighting`). Babylon defines `TWOSIDEDLIGHTING` only when the
    /// material ALSO has back-face culling off, and then flips the shading
    /// normal on back faces (`normalW = gl_FrontFacing ? normalW : -normalW`,
    /// after the normal-map perturbation), so the inside of a double-sided
    /// surface is lit like its outside. agate's `double_sided` only turns
    /// culling off, so back faces were shaded with a normal pointing away and
    /// came out black. Rides the `emissive_*.w` lane (unused elsewhere).
    two_sided_lighting: bool = false,
    /// When true, lighting calculations (direct lights, shadows, SSAO, IBL)
    /// are bypassed: renders pure albedo + emissive.
    unlit: bool = false,
    ior: f32 = 1.5,
    metallic: f32 = 0.0,
    roughness: f32 = 0.5,
    albedo_texture: ?Texture = null,

    normal_texture: ?Texture = null,
    /// Tangent-space normal map xy scale (glTF normalTexture.scale). 1.0
    /// keeps the map as authored; 0.0 flattens it to the geometric normal.
    /// Scaled in the PBR fragment shaders (uniform normal_scale).
    normal_scale: f32 = 1.0,
    metallic_roughness_texture: ?Texture = null,
    emissive_texture: ?Texture = null,
    emissive_color: Color3 = Color3.black,
    occlusion_texture: ?Texture = null,
    occlusion_strength: f32 = 1.0,

    // KHR_texture_transform per-slot UV maps (identity = unchanged
    // sampling). glTF texture views with `has_transform` load into these.
    albedo_uv_transform: UvTransform = .{},
    normal_uv_transform: UvTransform = .{},
    metallic_roughness_uv_transform: UvTransform = .{},
    emissive_uv_transform: UvTransform = .{},
    occlusion_uv_transform: UvTransform = .{},

    // Manual channel selection (see Channel): glTF fixes AO=R, roughness=G,
    // metallic=B and the defaults reproduce that; re-point the lanes for
    // hand-authored ORM-style maps. Uploaded in the channel_selectors
    // uniform and applied with constant branches in the PBR shaders.
    occlusion_channel: Channel = .r,
    roughness_channel: Channel = .g,
    metallic_channel: Channel = .b,

    environment_texture: ?CubeTexture = null,
    environment_intensity: f32 = 1.0,

    /// Babylon's `enableSpecularAntiAliasing` (`SPECULARAA`): widen the
    /// specular roughness of every analytic lobe by the screen-space normal
    /// variation, so a sub-pixel lobe is not point-sampled into aliasing
    /// (`getAARoughnessFactors` → `max(info.roughness,
    /// geometricRoughnessFactor)`; see `aaRoughnessFactor` in
    /// common/pbr_brdf.glsl). Rides the `channel_selectors.w` lane.
    ///
    /// Default matches a hand-built Babylon `PBRMaterial` (`false`); the glTF
    /// loaders force it ON for every material they create — that is what
    /// `babylonjs.loaders.js`'s `PBRMaterialLoadingAdapter` constructor does
    /// (`this._material.enableSpecularAntiAliasing = true`), and it is why a
    /// glTF helmet and a hand-made ground plane in the same Babylon scene
    /// compile different shaders. `SceneLoader` follows suit below.
    specular_anti_aliasing: bool = false,

    /// Clearcoat coat layer (default OFF: intensity 0 = lobe disabled,
    /// neutral tint/roughness). Optional R-mask texture (null = scalar).
    clearcoat: Clearcoat = .{},
    /// Sheen fabric lobe (default OFF: intensity 0 = lobe disabled,
    /// neutral tint/roughness). Optional rgb tint texture (null = scalar).
    sheen: Sheen = .{},
    /// Anisotropic specular stretch (default OFF: intensity 0 = isotropic,
    /// legacy GGX path bit-identical).
    anisotropy: Anisotropy = .{},
    /// Thin-slab transmission approx (default OFF: factor 0 = opaque,
    /// legacy path bit-identical). No refraction target (non-goal).
    transmission: Transmission = .{},
    /// Wrap/back-scatter SSS approx (default OFF: strength 0 = no term,
    /// legacy path bit-identical). Not a BSSRDF (non-goal).
    subsurface: Subsurface = .{},

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
        return self.alpha_mode == .blend or self.transmission.isRefractive();
    }

    /// Cutout is NOT transparent: it stays in the opaque queue with depth
    /// writes on; only sub-cutoff fragments are discarded.
    pub fn isCutout(self: PBRMaterial) bool {
        return self.alpha_mode == .cutout;
    }
};
