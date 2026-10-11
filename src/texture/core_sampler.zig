//! Sampler configuration for 2D textures. Split out of `core.zig` (facade).
//!
//! `Options` (filter/wrap/anisotropy/color-space contract) plus the pure
//! sampler helpers (`samplerMipFilter`, `effectiveAnisotropy`) and the shared
//! sokol sampler constructor (`makeSamplerFor`). Leaf: `sg` + `dds` only, so
//! the texture graph stays acyclic.

const sokol = @import("sokol");
const sg = sokol.gfx;
const dds = @import("../dds.zig");

pub const Options = struct {
    min_filter: sg.Filter = .LINEAR,
    mag_filter: sg.Filter = .LINEAR,
    /// Mip selection filter used when the texture has a mip chain. Has no
    /// effect on single-level textures. glTF's mipmapped min_filter
    /// variants map their mip half here (loader/materials.zig).
    mip_filter: sg.Filter = .LINEAR,
    wrap_u: sg.Wrap = .REPEAT,
    wrap_v: sg.Wrap = .REPEAT,
    /// Max anisotropy 1..16 (sokol sg_sampler_desc.max_anisotropy).
    /// Default 4 = Babylon's `Texture.DEFAULT_ANISOTROPIC_FILTERING_LEVEL`
    /// (babylon.js: `e.DEFAULT_ANISOTROPIC_FILTERING_LEVEL=4`, assigned to
    /// every `Texture.anisotropicFilteringLevel`; the glTF loader never
    /// overrides it, so every texture a glTF file references is sampled
    /// with 4). sokol requires LINEAR min/mag/mip filters for anisotropy
    /// > 1 and fails sampler validation otherwise, so non-LINEAR
    /// combinations clamp back to 1 here (Babylon clamps the same way,
    /// by sampling mode, in `_setAnisotropicLevel`). sokol has no lod
    /// bias; lod range defaults to 0..FLT_MAX (not exposed).
    max_anisotropy: u32 = 4,
    /// Signed distance fields must not be box-downsampled: the mip chain
    /// dilutes thin strokes and the shader edge drifts. Disable for fonts.
    mipmaps: bool = true,
    /// Explicit color space contract. Takes precedence over `slot` and `srgb_to_linear`.
    color_space: ?dds.TextureColorSpace = null,
    /// Semantic asset slot contract:
    /// - `.color`: albedo / emissive / base color (sRGB space)
    /// - `.data`: normal / metallic_roughness / occlusion / height / lookup (linear space)
    slot: ?dds.TextureSlot = null,
    /// Convert sRGB to linear at load time, BEFORE mip generation (the
    /// box filter then averages in linear space). Turn ON for LDR color
    /// textures that feed lighting math (glTF albedo/emissive), keep OFF
    /// for data textures (normal / metallic-roughness / occlusion: they
    /// are authored linear) and for GPU-bound views where the raw bytes
    /// matter (fonts, LUTs, sprites drawn without lighting).
    /// Kept for backward compatibility; prefer explicit `color_space` or `slot`.
    srgb_to_linear: bool = false,

    pub fn isColorSlot(self: Options) bool {
        if (self.color_space) |cs| return cs == .srgb;
        if (self.slot) |s| return s.colorSpace() == .srgb;
        return self.srgb_to_linear;
    }
};
/// Mip selection filter actually handed to sokol: the authored filter
/// only exists when there IS a mip chain, otherwise NEAREST (a
/// single-level texture has no mip to select).
fn samplerMipFilter(options: Options, num_mip_levels: u32) sg.Filter {
    return if (num_mip_levels > 1) options.mip_filter else .NEAREST;
}
/// Effective anisotropy for one sampler after the two clamps: the sokol
/// range (1..16) and the LINEAR-filter requirement described on
/// Options.max_anisotropy. Pure, so the clamping contract is testable
/// without a graphics context.
pub fn effectiveAnisotropy(options: Options, num_mip_levels: u32) u32 {
    const aniso = @min(options.max_anisotropy, 16);
    if (aniso <= 1) return aniso;
    if (options.min_filter != .LINEAR or options.mag_filter != .LINEAR or
        samplerMipFilter(options, num_mip_levels) != .LINEAR) return 1;
    return aniso;
}
/// Shared sampler creation for LDR 2D textures: applies min/mag/wrap,
/// the mip filter (only meaningful with a chain), and the anisotropy
/// guard described on Options.max_anisotropy.
pub fn makeSamplerFor(options: Options, num_mip_levels: u32) sg.Sampler {
    return sg.makeSampler(.{
        .min_filter = options.min_filter,
        .mag_filter = options.mag_filter,
        .mipmap_filter = samplerMipFilter(options, num_mip_levels),
        .wrap_u = options.wrap_u,
        .wrap_v = options.wrap_v,
        .max_anisotropy = effectiveAnisotropy(options, num_mip_levels),
    });
}
