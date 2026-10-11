//! Block-compressed format mapping and backend gates. Split out of `core.zig` (facade).
//!
//! `BlockSupport` (pure capability snapshot + `supportsFormat`/`preferred`),
//! the exact `sg.PixelFormat` mapping (`sgPixelFormatForBlock`), and the live
//! backend query (`queryBlockSupport`). The upload gate itself (`fromRawBlock`)
//! stays with the `Texture` owner; this leaf only answers "what/where".
//! Leaf: `sg` + `ktx2` only.

const sokol = @import("sokol");
const sg = sokol.gfx;
const ktx2 = @import("../ktx2.zig");

// -----------------------------------------------------------------------
// Block-compressed textures (KTX2 BC1/BC2/BC3/BC7 / ETC2 RGBA8 / ASTC 4x4 and DDS
// BC1/BC2/BC3/BC7, never decoded).
//
// Backend gate: compressed formats are uploaded only when
// sg.queryPixelformat reports sample support for the EXACT variant
// (UNORM vs SRGB matter — they are different GPU formats). There is no
// CPU fallback: without a transcoder dependency an unsupported backend
// gets error.BlockFormatNotSupportedByBackend, never silent corruption.
//
// sokol upload mechanism (verified against this checkout's sokol-zig):
// sg.ImageDesc.data.mip_levels[m] carries one tightly packed surface
// per level (for 2D: exactly sg.querySurfacePitch bytes, which for our
// 4x4/16B blocks equals the KTX2 level size) with .num_mipmaps set to
// the authored level count. Images stay immutable (the default): sokol
// rejects compressed dynamic_update/attachment images, and the engine
// never streams into textures — .data at creation is the only path.
//
// Partial chains: levels upload exactly as authored (no CPU mip
// synthesis — that needs a decoder plus an encoder). The GPU clamps LOD
// to the smallest present level, so a single-level file simply never
// minifies past level 0. Sampler mip filtering is enabled only with >1
// authored level (same convention as makeSamplerFor), and every filter
// drops to NEAREST when the backend reports sample-without-filter.
// -----------------------------------------------------------------------
/// Backend capability snapshot for every block family the engine can
/// upload. Fill with queryBlockSupport(); the decision helpers are pure
/// and unit-tested. BC2 shares the BC3 (S3TC) feature bit: desktop GL
/// exposes BC1-BC3 as one extension, and sokol has no separate BC2
/// query target that would behave differently.
pub const BlockSupport = struct {
    bc1_sample: bool = false,
    bc1_filter: bool = false,
    bc3_sample: bool = false,
    bc3_filter: bool = false,
    bc45_sample: bool = false,
    bc45_filter: bool = false,
    bc6_sample: bool = false,
    bc6_filter: bool = false,
    bc7_sample: bool = false,
    bc7_filter: bool = false,
    etc2_sample: bool = false,
    etc2_filter: bool = false,
    astc_sample: bool = false,
    astc_filter: bool = false,

    /// True when the backend can sample the exact sg format. Pure.
    pub fn supportsFormat(self: BlockSupport, fmt: sg.PixelFormat) bool {
        return switch (fmt) {
            .BC1_RGBA => self.bc1_sample,
            .BC2_RGBA, .BC3_RGBA, .BC3_SRGBA => self.bc3_sample,
            .BC4_R, .BC4_RSN, .BC5_RG, .BC5_RGSN => self.bc45_sample,
            .BC6H_RGBF, .BC6H_RGBUF => self.bc6_sample,
            .BC7_RGBA, .BC7_SRGBA => self.bc7_sample,
            .ETC2_RGBA8, .ETC2_SRGB8A8 => self.etc2_sample,
            .ASTC_4x4_RGBA, .ASTC_4x4_SRGBA => self.astc_sample,
            else => false,
        };
    }

    /// Preference order BC7 → BC3 → BC1 → ASTC 4x4 → ETC2 RGBA8 (UNORM
    /// representatives). Pure. Feeds the unsupported-format log hint so
    /// authors learn which encoding the current backend prefers.
    pub fn preferred(self: BlockSupport) ?sg.PixelFormat {
        if (self.bc7_sample) return .BC7_RGBA;
        if (self.bc3_sample) return .BC3_RGBA;
        if (self.bc1_sample) return .BC1_RGBA;
        if (self.astc_sample) return .ASTC_4x4_RGBA;
        if (self.etc2_sample) return .ETC2_RGBA8;
        return null;
    }
};
/// Exact sg format for a block format. Pure: UNORM stays UNORM; _SRGB
/// files upload to the sRGB GPU variant (hardware linearizes on sample
/// — the no-decoder equivalent of the CPU sRGB→linear path). BC1/BC2
/// have no sRGB variant in this sokol checkout and always map to UNORM
/// (see dds.zig).
pub fn sgPixelFormatForBlock(format: ktx2.BlockFormat) sg.PixelFormat {
    return switch (format) {
        .bc1_unorm => .BC1_RGBA,
        .bc2_unorm => .BC2_RGBA,
        .bc3_unorm => .BC3_RGBA,
        .bc3_srgb => .BC3_SRGBA,
        .bc4_unorm => .BC4_R,
        .bc4_snorm => .BC4_RSN,
        .bc5_unorm => .BC5_RG,
        .bc5_snorm => .BC5_RGSN,
        .bc6h_uf16 => .BC6H_RGBUF,
        .bc6h_sf16 => .BC6H_RGBF,
        .bc7_unorm => .BC7_RGBA,
        .bc7_srgb => .BC7_SRGBA,
        .etc2_rgba8_unorm => .ETC2_RGBA8,
        .etc2_rgba8_srgb => .ETC2_SRGB8A8,
        .astc_4x4_unorm => .ASTC_4x4_RGBA,
        .astc_4x4_srgb => .ASTC_4x4_SRGBA,
    };
}
/// Live backend support for every block family (UNORM representatives;
/// the upload gate additionally checks the exact variant). Requires a
/// valid sg context; keep the pure logic in BlockSupport for tests.
pub fn queryBlockSupport() BlockSupport {
    const bc1 = sg.queryPixelformat(.BC1_RGBA);
    const bc3 = sg.queryPixelformat(.BC3_RGBA);
    const bc45 = sg.queryPixelformat(.BC5_RG);
    const bc6 = sg.queryPixelformat(.BC6H_RGBUF);
    const bc7 = sg.queryPixelformat(.BC7_RGBA);
    const etc2 = sg.queryPixelformat(.ETC2_RGBA8);
    const astc = sg.queryPixelformat(.ASTC_4x4_RGBA);
    return .{
        .bc1_sample = bc1.sample,
        .bc1_filter = bc1.filter,
        .bc3_sample = bc3.sample,
        .bc3_filter = bc3.filter,
        .bc45_sample = bc45.sample,
        .bc45_filter = bc45.filter,
        .bc6_sample = bc6.sample,
        .bc6_filter = bc6.filter,
        .bc7_sample = bc7.sample,
        .bc7_filter = bc7.filter,
        .etc2_sample = etc2.sample,
        .etc2_filter = etc2.filter,
        .astc_sample = astc.sample,
        .astc_filter = astc.filter,
    };
}
