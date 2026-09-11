const sokol = @import("sokol");
const sg = sokol.gfx;

const math = @import("math");
const Mat4 = math.Mat4;
const Mesh = @import("../mesh.zig").Mesh;
const Material = @import("../material.zig").Material;

pub const RenderMeshItem = struct {
    mesh: *Mesh,
    model: Mat4,
    distance_sq: f32,
    is_pbr: bool,
    texture_id: u32,
    // True when the mesh material uses .blend alpha mode. Set at queue
    // build time; defaults to false so opaque behavior is unchanged.
    transparent: bool = false,
    // True when the mesh material is double-sided (face culling disabled).
    // Set at queue build time from materialIsDoubleSided; when the flag is
    // stale/missing, pipeline selection re-derives it from mesh.material.
    // Defaults to false so single-sided behavior is unchanged.
    double_sided: bool = false,
};

pub fn sortRenderItems(_: void, a: RenderMeshItem, b: RenderMeshItem) bool {
    if (a.is_pbr != b.is_pbr) {
        return !a.is_pbr;
    }
    if (a.texture_id != b.texture_id) {
        return a.texture_id < b.texture_id;
    }
    return a.distance_sq < b.distance_sq;
}

// Transparent items sort strictly back-to-front (by squared camera
// distance) so alpha blending composites in the correct order.
pub fn sortTransparentBackToFront(_: void, a: RenderMeshItem, b: RenderMeshItem) bool {
    return a.distance_sq > b.distance_sq;
}

// A mesh is transparent when its material opts into .blend alpha mode.
// Meshes without a material render opaque (legacy behavior).
// Cutout (.cutout) is NOT transparent: it renders in the opaque queue
// with depth writes on, discarding only sub-cutoff fragments in-shader.
pub fn materialIsTransparent(mat: ?Material) bool {
    if (mat) |m| return m.isTransparent();
    return false;
}

// A mesh is cutout when its material uses .cutout alpha mode.
// Cutout meshes stay in the opaque queue (see materialIsTransparent).
// Shadow depth passes ignore the alpha test and render full quads
// (documented limitation: cutout holes still cast solid shadows).
pub fn materialIsCutout(mat: ?Material) bool {
    if (mat) |m| return m.isCutout();
    return false;
}

// A mesh is double-sided when its material sets double_sided.
// Such meshes select the cull-off pipeline twins (opaque or blend).
// Meshes without a material render single-sided (legacy behavior).
pub fn materialIsDoubleSided(mat: ?Material) bool {
    if (mat) |m| return m.isDoubleSided();
    return false;
}

// Derives a transparent twin pipeline desc from an opaque base desc:
// same shader/layout/depth test, but SRC_ALPHA/ONE_MINUS_SRC_ALPHA
// blending with depth write disabled. Pure function (no GPU calls).
pub fn blendDescFor(base: sg.PipelineDesc) sg.PipelineDesc {
    var desc = base;
    desc.depth.write_enabled = false;
    desc.colors[0].blend = .{
        .enabled = true,
        .src_factor_rgb = .SRC_ALPHA,
        .dst_factor_rgb = .ONE_MINUS_SRC_ALPHA,
        .src_factor_alpha = .ONE,
        .dst_factor_alpha = .ONE_MINUS_SRC_ALPHA,
    };
    return desc;
}

test "cutout stays opaque, blend stays transparent" {
    const std = @import("std");
    const material = @import("../material.zig");

    try std.testing.expect(!materialIsTransparent(null));
    try std.testing.expect(!materialIsCutout(null));
    try std.testing.expect(!materialIsDoubleSided(null));

    var std_mat = material.StandardMaterial.init("m");
    var pbr_mat = material.PBRMaterial.init("p");

    // Opaque default: opaque queue, not cutout, single-sided.
    try std.testing.expect(!materialIsTransparent(.{ .standard = &std_mat }));
    try std.testing.expect(!materialIsTransparent(.{ .pbr = &pbr_mat }));
    try std.testing.expect(!materialIsCutout(.{ .standard = &std_mat }));
    try std.testing.expect(!materialIsCutout(.{ .pbr = &pbr_mat }));

    // Cutout: still NOT in the transparent queue, but flagged cutout.
    std_mat.alpha_mode = .cutout;
    pbr_mat.alpha_mode = .cutout;
    try std.testing.expect(!materialIsTransparent(.{ .standard = &std_mat }));
    try std.testing.expect(!materialIsTransparent(.{ .pbr = &pbr_mat }));
    try std.testing.expect(materialIsCutout(.{ .standard = &std_mat }));
    try std.testing.expect(materialIsCutout(.{ .pbr = &pbr_mat }));

    // Blend: transparent queue, never cutout.
    std_mat.alpha_mode = .blend;
    pbr_mat.alpha_mode = .blend;
    try std.testing.expect(materialIsTransparent(.{ .standard = &std_mat }));
    try std.testing.expect(materialIsTransparent(.{ .pbr = &pbr_mat }));
    try std.testing.expect(!materialIsCutout(.{ .standard = &std_mat }));
    try std.testing.expect(!materialIsCutout(.{ .pbr = &pbr_mat }));

    // Double-sided is orthogonal to the alpha mode.
    try std.testing.expect(!materialIsDoubleSided(.{ .standard = &std_mat }));
    std_mat.double_sided = true;
    pbr_mat.double_sided = true;
    try std.testing.expect(materialIsDoubleSided(.{ .standard = &std_mat }));
    try std.testing.expect(materialIsDoubleSided(.{ .pbr = &pbr_mat }));
}
