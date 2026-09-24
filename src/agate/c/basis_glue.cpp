// Basis Universal KTX2 transcoding glue (BinomialLLC/basis_universal,
// Apache-2.0, vendored verbatim under src/agate/c/basisu/; see LICENSES.md).
//
// Why this file is C++: ktx2_transcoder is a C++ class. Everything else in
// c_impl.c stays C99; this translation unit only exposes the extern "C"
// entry points declared in agate_c_impl.h.
//
// Thread safety: basisu_transcoder_init() must run exactly once per process
// (it fills global lookup tables and is NOT safe under concurrent first
// calls), so it is guarded by std::call_once. Every entry point below then
// constructs its own ktx2_transcoder on the stack: no shared mutable state,
// no retained pointers into the caller's bytes after return. Concurrent
// transcodes of different files (the UploadQueue worker path) are safe.
//
// Ownership: the caller (Zig) owns every buffer. info() only reads; transcode()
// writes into a caller-allocated, caller-sized buffer and never allocates.
#include <stddef.h>
#include <stdint.h>
#include <mutex>

#include "basisu/transcoder/basisu_transcoder.h"

static std::once_flag g_basis_init_flag;

static void ensure_basisu_initialized()
{
    std::call_once(g_basis_init_flag, []() { basist::basisu_transcoder_init(); });
}

// Mirrors agate_basis_info in agate_c_impl.h (C ABI: fixed layout).
struct agate_basis_info_cpp {
    uint32_t width;
    uint32_t height;
    uint32_t levels;
    uint32_t faces;
    uint32_t layers;
    uint32_t has_alpha;
    uint32_t is_srgb;
    int32_t kind; // 0 = ETC1S, 1 = UASTC LDR 4x4
};

extern "C" {

// Tri-state: 1 = supported (out filled), 0 = not-a-Basis/corrupt (init
// fails), -1 = valid Basis file outside the supported subset.
int agate_basis_ktx2_info(const uint8_t* data, size_t size, void* out_info)
{
    if (!data || size == 0 || size > 0xFFFFFFFFu || !out_info)
        return 0;
    ensure_basisu_initialized();

    basist::ktx2_transcoder t;
    if (!t.init(data, (uint32_t)size))
        return 0;
    // Supported subset, truthfully narrow:
    //   - ETC1S (BasisLZ) and UASTC LDR 4x4 only. HDR/XUASTC/ASTC-LDR/XUBC7
    //     sources report -1 (caller maps to BasisUnsupported).
    //   - 2D, single layer/face. Cubes/arrays/3D report -1; the engine has
    //     no block-cube upload path, so this must never silently drop faces.
    //   - ETC1S video (P-frames) reports -1: it requires strictly ordered
    //     first-to-last transcoding which the per-level API cannot promise.
    //   - levelCount 0 (implicit single level: rejected up front by the Zig
    //     envelope as BasisUnsupported; the glue also guards it) or > 16.
    const bool etc1s = t.is_etc1s();
    const bool uastc = t.is_uastc();
    if (!etc1s && !uastc)
        return -1;
    if (t.get_header().m_pixel_depth != 0)
        return -1;
    if (t.get_faces() != 1 || t.get_layers() != 0)
        return -1;
    if (t.get_levels() == 0 || t.get_levels() > 16)
        return -1;
    if (t.get_width() == 0 || t.get_height() == 0)
        return -1;
    if (!t.start_transcoding())
        return 0;
    if (t.is_video())
        return -1;

    agate_basis_info_cpp* out = (agate_basis_info_cpp*)out_info;
    out->width = t.get_width();
    out->height = t.get_height();
    out->levels = t.get_levels();
    out->faces = t.get_faces();
    out->layers = t.get_layers();
    out->has_alpha = t.get_has_alpha() ? 1u : 0u;
    out->is_srgb = t.is_srgb() ? 1u : 0u;
    out->kind = etc1s ? 0 : 1;
    return 1;
}

bool agate_basis_ktx2_transcode(const uint8_t* data, size_t size,
    uint32_t level, uint32_t face, int32_t target,
    uint8_t* out, size_t out_len)
{
    if (!data || size == 0 || size > 0xFFFFFFFFu || !out || out_len == 0)
        return false;
    ensure_basisu_initialized();

    basist::transcoder_texture_format fmt;
    size_t block_or_pixel_bytes;
    switch (target) {
    case 0: // BC7 RGBA, 16 B per 4x4 block
        fmt = basist::transcoder_texture_format::cTFBC7_RGBA;
        block_or_pixel_bytes = 16;
        break;
    case 1: // ASTC LDR 4x4 RGBA, 16 B per block
        fmt = basist::transcoder_texture_format::cTFASTC_LDR_4x4_RGBA;
        block_or_pixel_bytes = 16;
        break;
    case 2: // RGBA32, 4 B per pixel in raster order (R first, A last)
        fmt = basist::transcoder_texture_format::cTFRGBA32;
        block_or_pixel_bytes = 4;
        break;
    default:
        return false;
    }

    basist::ktx2_transcoder t;
    if (!t.init(data, (uint32_t)size))
        return false;
    if (!t.is_etc1s() && !t.is_uastc())
        return false;
    if (!t.start_transcoding())
        return false;
    if (t.is_video())
        return false;
    if (level >= t.get_levels() || face >= t.get_faces())
        return false;

    basist::ktx2_image_level_info li;
    if (!t.get_image_level_info(li, level, 0, face))
        return false;

    // Expected output size from the transcoder's own level description (not
    // from the KTX2 header: ETC1S levels report uncompressedByteLength 0 and
    // UASTC physical dims round up to the block grid).
    uint64_t units;
    if (block_or_pixel_bytes == 16) {
        units = (uint64_t)li.m_total_blocks;
    } else {
        units = (uint64_t)li.m_orig_width * (uint64_t)li.m_orig_height;
    }
    const uint64_t expected = units * (uint64_t)block_or_pixel_bytes;
    if (expected == 0 || expected > (uint64_t)out_len)
        return false;
    // Exact-size contract: the caller sizes the buffer from the same level
    // math, so a mismatch is a programming error, never silent truncation.
    if ((uint64_t)out_len != expected)
        return false;

    return t.transcode_image_level(level, 0, face,
        out, (uint32_t)units, fmt);
}

} // extern "C"
