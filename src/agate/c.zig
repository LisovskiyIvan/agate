pub const c = @cImport({
    @cInclude("stb_image.h");
    @cInclude("cgltf.h");
    // agate's own C glue (c_impl.c); relies on cgltf.h types.
    @cInclude("agate_c_impl.h");
    @cInclude("box3d/box3d.h");
});
