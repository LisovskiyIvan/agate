// Umbrella header for the vendored audio decoder declarations.
//
// Zig 0.17 removed @cImport; build.zig runs the official translate-c
// package over this file instead (module "c_audio_translate", imported by
// src/audio/decode.zig). Mirrors the old @cImport block: dr_mp3 plus
// stb_vorbis in declarations-only mode (STB_VORBIS_HEADER_ONLY); the
// implementations still compile exactly once through src/c/c_impl.c.
#ifndef AGATE_AUDIO_C_IMPORT_H
#define AGATE_AUDIO_C_IMPORT_H

#include "dr_mp3.h"

#define STB_VORBIS_HEADER_ONLY 1
#include "stb_vorbis.c"

#endif // AGATE_AUDIO_C_IMPORT_H
