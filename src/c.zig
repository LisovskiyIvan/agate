// C declarations visible to Zig. Zig 0.17 removed @cImport; build.zig runs
// the official translate-c package over src/c/agate_c_import.h (module
// "c_translate"), which mirrors the include set this file's @cImport had:
// stb_image + cgltf + agate's own glue + box3d. Implementations compile
// exactly once through src/c/c_impl.c (wired in build.zig).
pub const c = @import("c_translate");
