/* Declarations for definitions that live in c_impl.c, so the Zig side can
   call them through @cImport (src/agate/c.zig includes this after cgltf.h). */
#ifndef AGATE_C_IMPL_H
#define AGATE_C_IMPL_H

#include "cgltf.h"

/* Decodes every buffer view carrying EXT_meshopt_compression in place:
   view->data is replaced with freshly allocated (options->memory) decoded
   bytes, owned and freed by cgltf_free. No-op for files without the
   extension. Call only after cgltf_load_buffers succeeded. */
cgltf_result agate_cgltf_decode_meshopt(const cgltf_options* options, cgltf_data* data);

#endif /* AGATE_C_IMPL_H */
