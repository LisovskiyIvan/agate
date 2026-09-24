/* Declarations for definitions that live in c_impl.c, so the Zig side can
   call them through @cImport (src/agate/c.zig includes this after cgltf.h). */
#ifndef AGATE_C_IMPL_H
#define AGATE_C_IMPL_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "cgltf.h"

/* Decodes every buffer view carrying EXT_meshopt_compression in place:
   view->data is replaced with freshly allocated (options->memory) decoded
   bytes, owned and freed by cgltf_free. No-op for files without the
   extension. Call only after cgltf_load_buffers succeeded. */
cgltf_result agate_cgltf_decode_meshopt(const cgltf_options* options, cgltf_data* data);

#ifdef __cplusplus
extern "C" {
#endif

/* Basis Universal KTX2 transcoding (src/agate/c/basis_glue.cpp, C++).
   Thread-safe: global tables init once via call_once, each call uses its own
   transcoder; the caller owns all buffers. */

/* agate_basis_info layout shared with basis_glue.cpp (C ABI). */
typedef struct agate_basis_info {
    uint32_t width;
    uint32_t height;
    uint32_t levels;
    uint32_t faces;
    uint32_t layers;
    uint32_t has_alpha;
    uint32_t is_srgb;
    int32_t kind; /* 0 = ETC1S (BasisLZ), 1 = UASTC LDR 4x4 */
} agate_basis_info;

/* Parses a KTX2 Basis payload (vkFormat UNDEFINED): inspects the file when it
   is in the supported subset (ETC1S/UASTC LDR, 2D, single layer/face,
   1..16 levels, not ETC1S video). Returns 1 and fills `out_info` on success,
   0 when the payload is not a Basis file or is corrupt (transcoder init
   fails), -1 when it IS a Basis file outside the supported subset
   (HDR/XUASTC/ASTC-LDR/XUBC7 kind, cube/array/3D, video, levelCount 0/17+). */
int agate_basis_ktx2_info(const uint8_t* data, size_t size, agate_basis_info* out_info);

/* Transcodes one (level, face) of a KTX2 Basis file inspected by
   agate_basis_ktx2_info. `target`: 0 = BC7 RGBA, 1 = ASTC LDR 4x4 RGBA,
   2 = RGBA32 raster. `out` must be EXACTLY the level size (blocks*16 for
   block targets, orig_w*orig_h*4 for RGBA32); any mismatch fails. */
bool agate_basis_ktx2_transcode(const uint8_t* data, size_t size,
    uint32_t level, uint32_t face, int32_t target,
    uint8_t* out, size_t out_len);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* AGATE_C_IMPL_H */
