const std = @import("std");
const math = @import("math");
const Color3 = math.Color3;
const Color4 = math.Color4;
const Texture = @import("../texture.zig").Texture;

/// How a material blends with the framebuffer.
/// .@"opaque" keeps the legacy behavior (depth write on, no blending).
/// .cutout is alpha-tested opaque: fragments with material alpha below
/// alpha_cutoff are discarded in the fragment shader, everything else
/// renders exactly like opaque (depth write on, opaque queue, no sorting).
/// .blend enables standard alpha blending (SRC_ALPHA, ONE_MINUS_SRC_ALPHA)
/// with depth test on and depth write off; such meshes are drawn after all
/// opaque geometry, sorted back-to-front.
// Note: `opaque` is a Zig keyword, so the variant is spelled @"opaque".
pub const AlphaMode = enum {
    @"opaque",
    cutout,
    blend,
};

/// Selects one RGBA lane of a sampled texture. Data slots (occlusion /
/// roughness / metallic) only: glTF fixes the KHR_materials_occlusion_roughness-
/// metallic conventions (AO=R, roughness=G, metallic=B) and provides no
/// channel override, so this is an API for MANUAL materials — e.g. an ORM
/// map in a different layout, or packing three scalar maps into one RGBA
/// texture. The defaults reproduce the glTF behavior exactly.
pub const Channel = enum(u2) {
    r = 0,
    g = 1,
    b = 2,
    a = 3,

    /// Index uploaded to the shaders' channel_selectors uniform. The GLSL
    /// side picks the lane via constant branches (SPIRV-Cross cannot
    /// flatten dynamic component indexing, same constraint as morphWeight).
    pub fn selector(self: Channel) f32 {
        return @floatFromInt(@intFromEnum(self));
    }
};

/// Per-slot texture UV transform (KHR_texture_transform subset):
///   uv' = R(rotation) * (scale .* uv) + offset
/// with counter-clockwise `rotation` in radians around the UV origin, per
/// the extension spec. Applied in the fragment shaders from two vec4
/// uniforms per slot ([m00, m01, m10, m11] rows + offset) so every slot can
/// carry its own transform without touching the vertex contract.
///
/// Documented limitation: glTF `texCoord` > 0 (a second UV set) is NOT
/// supported — the transform applies to texcoord0 regardless. Multi-UV
/// support would need a v_uv2 varying across the five forward shaders
/// (roadmap).
pub const UvTransform = struct {
    offset: [2]f32 = .{ 0, 0 },
    rotation: f32 = 0,
    scale: [2]f32 = .{ 1, 1 },

    pub const identity: UvTransform = .{};

    pub fn isIdentity(self: UvTransform) bool {
        return self.offset[0] == 0 and self.offset[1] == 0 and
            self.rotation == 0 and
            self.scale[0] == 1 and self.scale[1] == 1;
    }

    /// The 2x2 rotation*scale matrix rows packed for the shaders' uv_matrix
    /// uniform: [m00, m01, m10, m11], i.e.
    ///   u' = m00*u + m01*v + offset.x
    ///   v' = m10*u + m11*v + offset.y
    pub fn matrixRows(self: UvTransform) [4]f32 {
        const cos_r = @cos(self.rotation);
        const sin_r = @sin(self.rotation);
        // M = R(rotation) * diag(scale.x, scale.y): the scale applies to the
        // raw uv components BEFORE the rotation (extension spec:
        // uv' = offset + R * (scale .* uv)).
        return .{
            cos_r * self.scale[0], -sin_r * self.scale[1],
            sin_r * self.scale[0], cos_r * self.scale[1],
        };
    }

    /// The offset packed for the shaders' uv_offset uniform ([x, y, 0, 0]).
    pub fn offsetPacked(self: UvTransform) [4]f32 {
        return .{ self.offset[0], self.offset[1], 0, 0 };
    }
};

/// Scalar-only clearcoat layer (Babylon parity, OpenPBR-adjacent subset).
/// A dielectric coat (car paint, lacquered wood) over the base PBR layer:
/// separate GGX specular lobe with its own roughness, F0 = 0.04 tinted by
/// `color`, and an energy-conserving (1 - F_cc) attenuation of the BASE
/// specular (direct + IBL). No clearcoat roughness/normal maps and no
/// second normal layer — the coat reuses the base N.
///
/// `mask_texture` (v1, optional): R lane multiplies `intensity` (white
/// fallback = identity, so null renders exactly like a 1.0 mask).
/// Sampled with the ALBEDO uv transform (no per-slot coat transform yet).
/// The mask modulates — it never enables: `intensity == 0` keeps the lobe
/// off even with a texture bound (null side-table entry, legacy path).
///
/// Gating: `intensity == 0` (the default) disables the lobe. Every shader
/// term is multiplied by the intensity and the base attenuation becomes
/// exactly (1 - 0), so disabled materials render bit-identically to before.
/// Babylon mapping: intensity = clearCoat.intensity (Babylon default 1 when
/// the coat is enabled; here 0 = off replaces the isEnabled flag),
/// roughness = clearCoat.roughness, color = clearCoat.tintColor (white =
/// untinted, matching glTF KHR_materials_clearcoat clearcoatColorFactor).
pub const Clearcoat = struct {
    intensity: f32 = 0.0,
    roughness: f32 = 0.03,
    color: Color3 = Color3.white,
    mask_texture: ?Texture = null,
};

/// Scalar-only sheen layer (Babylon parity, OpenPBR-adjacent subset).
/// A view-dependent fabric/fuzz lobe (Charlie distribution + Neubelt
/// visibility, Karis-style grazing response), additive over the base layer.
///
/// `color_texture` (v1, optional): rgb multiplies `color` (white fallback
/// = identity, so null renders exactly like a 1.0 tint). Sampled with the
/// ALBEDO uv transform (no per-slot sheen transform yet). The map
/// modulates — it never enables: `intensity == 0` keeps the lobe off even
/// with a texture bound (null side-table entry, legacy path).
///
/// Gating: `intensity == 0` (the default) disables the lobe — every shader
/// term is multiplied by the intensity, so disabled materials render
/// bit-identically to before. Babylon mapping: intensity = sheen.intensity,
/// roughness = sheen.roughness (0 = mirror-smooth fuzz, 1 = fully rough),
/// color = sheen.color (the retroreflective tint; white = untinted).
pub const Sheen = struct {
    color: Color3 = Color3.white,
    intensity: f32 = 0.0,
    roughness: f32 = 0.5,
    color_texture: ?Texture = null,
};

/// Anisotropic specular v1 (GGX-Heitz-style stretch of the base-lobe NDF
/// along the mesh tangent frame). NOT full OpenPBR anisotropy (no
/// per-light anisotropic roughness maps, isotropic geometry G, isotropic
/// IBL): the specular highlight elongates along T/B with `intensity`.
/// The frame is the EXISTING vertex tangent attribute (Gram-Schmidt
/// orthogonalized in the vertex shader) steered in the tangent plane by
/// `rotation` — no new vertex attribute. Meshes without authored tangents
/// (glTF default +X, see loader/mesh_spawn.zig) get a uniform fallback
/// direction: documented approximation, not an error.
///
/// Gating: `intensity == 0` (the default) = isotropic: the shader
/// early-returns to the legacy GGX NDF, bit-identical to before.
pub const Anisotropy = struct {
    intensity: f32 = 0.0,
    /// Tangent-plane steering angle in radians (0 = mesh tangents as-is).
    rotation: f32 = 0.0,
};

/// Cheap transmission v1 (NOT refraction: no refraction target, no IOR,
/// no thickness — porting a full refraction pass is an explicit non-goal).
/// Thin-slab approximation that keeps the mesh in its current queue
/// (alpha/queue untouched: real see-through glass still uses
/// alpha_mode.blend): the diffuse albedo scales by (1 - factor) AFTER F0
/// (metals keep their F0) and an additive sun+ambient back-light term
/// tinted by `color` fakes throughput. Punctual point/spot/area/clustered
/// lights do NOT contribute to the transmitted term (v1 scope).
///
/// Gating: `factor == 0` (the default) = off: the albedo scale is skipped
/// and the additive term branches off, bit-identical to before.
pub const Transmission = struct {
    factor: f32 = 0.0,
    color: Color3 = Color3.white,
    ior: f32 = 1.5,
};

/// Cheap subsurface scattering v1 (NOT a BSSRDF / random-walk SSS:
/// physical SSS is an explicit non-goal). Wrap-diffuse + back-scatter glow
/// driven by the sun + ambient, tinted by `color` and the albedo, added
/// over the base layer. Punctual point/spot/area/clustered lights do NOT
/// contribute (v1 scope).
///
/// Gating: `strength == 0` (the default) = off: the additive term branches
/// off, bit-identical to before.
pub const Subsurface = struct {
    strength: f32 = 0.0,
    color: Color3 = Color3.white,
};

/// Render-owned, GPU-ready clearcoat + sheen factors plus the v1 PBR layer
/// pack (anisotropy / transmission / subsurface). Lives in the render-queue
/// side table (RenderQueues.coat_storage, resolved by index at draw time
/// like skin_storage), NOT in MaterialDrawRecord: per-draw factors would
/// bust the P4 size guard on RenderMeshItem, so only draws with an enabled
/// layer occupy a slot. Every other draw resolves to `neutral` at draw time
/// and shades bit-identically to before.
///
/// Coat/sheen TEXTURES are not here: texture views stage into
/// MaterialDrawRecord (a handful of view ids, like every other map) while
/// this table carries the scalar/color pack. A bound texture never enables
/// a lobe on its own — the scalar gate below decides.
pub const CoatParams = struct {
    clearcoat_factors: [4]f32 = .{ 0, 0.03, 0, 0 }, // x: intensity, y: roughness
    clearcoat_color: [4]f32 = .{ 1, 1, 1, 1 }, // rgb tint, w unused
    sheen_factors: [4]f32 = .{ 0, 0.5, 0, 0 }, // x: intensity, y: roughness
    sheen_color: [4]f32 = .{ 1, 1, 1, 1 }, // rgb tint, w unused
    // APPENDED LAST (pbr-layers v1): the first four lanes keep their
    // offsets; disabled (all zero) uploads bit-identical shading.
    anisotropy_factors: [4]f32 = .{ 0, 0, 0, 0 }, // x: intensity (0 = isotropic), y: rotation rad
    transmission_factors: [4]f32 = .{ 0, 0, 0, 0 }, // x: factor (0 = off)
    transmission_color: [4]f32 = .{ 1, 1, 1, 1 }, // rgb throughput tint, w unused
    sss_factors: [4]f32 = .{ 0, 0, 0, 0 }, // x: strength (0 = off)
    sss_color: [4]f32 = .{ 1, 1, 1, 1 }, // rgb scatter tint, w unused

    pub const neutral: CoatParams = .{};
};

/// CPU mirror of the shader's anisotropy axis computation (Heitz-style):
/// (ax, ay) roughnesses along T/B. intensity <= 0 returns the isotropic
/// pair (roughness^2, roughness^2); the shader then skips the stretch and
/// evaluates the legacy GGX NDF instead. Unit-tested below.
pub fn anisotropyAxes(roughness: f32, intensity: f32) [2]f32 {
    const r2 = roughness * roughness;
    if (intensity <= 0.0) return .{ r2, r2 };
    const clamped = @min(@max(intensity, 0.0), 1.0);
    const aspect = @sqrt(@max(1.0 - 0.9 * clamped, 0.01));
    return .{ @max(r2 / aspect, 0.001), @max(r2 * aspect, 0.001) };
}

/// CPU mirror of the SSS wrap term: wraps NdotL toward 1 as strength
/// grows (strength <= 0 returns the input unchanged). Unit-tested below.
pub fn wrapNdotL(ndotl: f32, strength: f32) f32 {
    if (strength <= 0.0) return ndotl;
    const wrap = strength * 0.5;
    return @min(@max((ndotl + wrap) / (1.0 + wrap), 0.0), 1.0);
}
