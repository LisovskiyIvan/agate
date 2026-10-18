// Umbrella header for agate's Zig-visible C declarations.
//
// Zig 0.17 removed @cImport; build.zig runs the official translate-c
// package over this file instead (module "c_translate", imported by
// src/c.zig). The include set mirrors the old @cImport block exactly:
// implementations still compile once through src/c/c_impl.c (wired in
// build.zig), this header only provides declarations to Zig.
//
// "box3d/box3d.h" resolves through the translator's include path
// src/c/box3d/include (same as mod_agate).
#ifndef AGATE_C_IMPORT_H
#define AGATE_C_IMPORT_H

#include "stb_image.h"
#include "cgltf.h"
#include "agate_c_impl.h"
#include "box3d/box3d.h"

#endif // AGATE_C_IMPORT_H
