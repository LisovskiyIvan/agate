const sokol = @import("sokol");
const sg = sokol.gfx;
const std = @import("std");
const math = @import("math");
const Color3 = math.Color3;
const Color4 = math.Color4;
const Texture = @import("texture.zig").Texture;
const CubeTexture = @import("texture.zig").CubeTexture;

/// How a material blends with the framebuffer.
/// .@"opaque" keeps the legacy behavior (depth write on, no blending).
/// .cutout is alpha-tested opaque: fragments with material alpha below
/// alpha_cutoff are discarded in the fragment shader, everything else
/// renders exactly like opaque (depth write on, opaque queue, no sorting).
/// .blend enables standard alpha blending (SRC_ALPHA, ONE_MINUS_SRC_ALPHA)
/// with depth test on and depth write off; such meshes are drawn after all
/// opaque geometry, sorted back-to-front.
// Note: `opaque` is a Zig keyword, so the variant is spelled @"opaque".
pub const AlphaMode = enum {
    @"opaque",
    cutout,
    blend,
};

pub const StandardMaterial = struct {
    name: []const u8 = "StandardMaterial",
    diffuse_color: Color3 = Color3.white,
    alpha: f32 = 1.0,
    alpha_mode: AlphaMode = .@"opaque",
    /// Alpha-test threshold used only when alpha_mode == .cutout.
    /// The fragment shader discards fragments with alpha < alpha_cutoff.
    /// For opaque/blend the draw path uploads 0.0 so the test never fires.
    alpha_cutoff: f32 = 0.5,
    /// When true the mesh renders with face culling disabled (a cull-off
    /// pipeline twin). Both opaque and blend twins exist.
    double_sided: bool = false,
    diffuse_texture: ?Texture = null,

    pub fn init(name: []const u8) StandardMaterial {
        return .{
            .name = name,
            .diffuse_color = Color3.white,
            .alpha = 1.0,
        };
    }

    pub fn getDiffuseColor4(self: StandardMaterial) [4]f32 {
        return .{ self.diffuse_color.r, self.diffuse_color.g, self.diffuse_color.b, self.alpha };
    }

    pub fn isTransparent(self: StandardMaterial) bool {
        return self.alpha_mode == .blend;
    }

    /// Cutout is NOT transparent: it stays in the opaque queue with depth
    /// writes on; only sub-cutoff fragments are discarded.
    pub fn isCutout(self: StandardMaterial) bool {
        return self.alpha_mode == .cutout;
    }
};

pub const PBRMaterial = struct {
    name: []const u8 = "PBRMaterial",
    albedo_color: Color3 = Color3.white,
    alpha: f32 = 1.0,
    alpha_mode: AlphaMode = .@"opaque",
    /// Alpha-test threshold used only when alpha_mode == .cutout.
    /// The fragment shader discards fragments with alpha < alpha_cutoff.
    /// For opaque/blend the draw path uploads 0.0 so the test never fires.
    alpha_cutoff: f32 = 0.5,
    /// When true the mesh renders with face culling disabled (a cull-off
    /// pipeline twin). Both opaque and blend twins exist.
    double_sided: bool = false,
    metallic: f32 = 0.0,
    roughness: f32 = 0.5,
    albedo_texture: ?Texture = null,

    normal_texture: ?Texture = null,
    metallic_roughness_texture: ?Texture = null,
    emissive_texture: ?Texture = null,
    emissive_color: Color3 = Color3.black,
    occlusion_texture: ?Texture = null,
    occlusion_strength: f32 = 1.0,

    environment_texture: ?CubeTexture = null,
    environment_intensity: f32 = 1.0,

    pub fn init(name: []const u8) PBRMaterial {
        return .{
            .name = name,
        };
    }

    pub fn getAlbedoColor4(self: PBRMaterial) [4]f32 {
        return .{ self.albedo_color.r, self.albedo_color.g, self.albedo_color.b, self.alpha };
    }

    pub fn getEmissiveColor4(self: PBRMaterial) [4]f32 {
        return .{ self.emissive_color.r, self.emissive_color.g, self.emissive_color.b, 1.0 };
    }

    pub fn isTransparent(self: PBRMaterial) bool {
        return self.alpha_mode == .blend;
    }

    /// Cutout is NOT transparent: it stays in the opaque queue with depth
    /// writes on; only sub-cutoff fragments are discarded.
    pub fn isCutout(self: PBRMaterial) bool {
        return self.alpha_mode == .cutout;
    }
};

pub const Material = union(enum) {
    standard: *StandardMaterial,
    pbr: *PBRMaterial,

    pub fn isTransparent(self: Material) bool {
        return switch (self) {
            .standard => |s| s.isTransparent(),
            .pbr => |p| p.isTransparent(),
        };
    }

    pub fn isCutout(self: Material) bool {
        return switch (self) {
            .standard => |s| s.isCutout(),
            .pbr => |p| p.isCutout(),
        };
    }

    pub fn isDoubleSided(self: Material) bool {
        return switch (self) {
            .standard => |s| s.double_sided,
            .pbr => |p| p.double_sided,
        };
    }

    /// Raw per-material cutoff (defaults to 0.5). The draw path gates this:
    /// only cutout materials upload a live cutoff, otherwise 0.0 (disabled).
    /// See scene/uniforms.zig alphaCutoffFor.
    pub fn alphaCutoff(self: Material) f32 {
        return switch (self) {
            .standard => |s| s.alpha_cutoff,
            .pbr => |p| p.alpha_cutoff,
        };
    }
};

test "alpha_mode defaults to opaque (back-compat)" {
    const std_mat = StandardMaterial.init("m");
    try std.testing.expect(std_mat.alpha_mode == .@"opaque");
    try std.testing.expect(!std_mat.isTransparent());
    try std.testing.expect(std_mat.alpha == 1.0);

    const pbr_mat = PBRMaterial.init("p");
    try std.testing.expect(pbr_mat.alpha_mode == .@"opaque");
    try std.testing.expect(!pbr_mat.isTransparent());
    try std.testing.expect(pbr_mat.alpha == 1.0);
}

test "Material.isTransparent follows alpha_mode" {
    var std_mat = StandardMaterial.init("m");
    var pbr_mat = PBRMaterial.init("p");

    const m_std: Material = .{ .standard = &std_mat };
    const m_pbr: Material = .{ .pbr = &pbr_mat };
    try std.testing.expect(!m_std.isTransparent());
    try std.testing.expect(!m_pbr.isTransparent());

    std_mat.alpha_mode = .blend;
    pbr_mat.alpha_mode = .blend;
    try std.testing.expect(m_std.isTransparent());
    try std.testing.expect(m_pbr.isTransparent());
}

test "cutout classifies as opaque, never transparent" {
    var std_mat = StandardMaterial.init("m");
    var pbr_mat = PBRMaterial.init("p");
    std_mat.alpha_mode = .cutout;
    pbr_mat.alpha_mode = .cutout;

    try std.testing.expect(std_mat.isCutout());
    try std.testing.expect(pbr_mat.isCutout());
    // Cutout stays in the opaque queue: not transparent.
    try std.testing.expect(!std_mat.isTransparent());
    try std.testing.expect(!pbr_mat.isTransparent());

    const m_std: Material = .{ .standard = &std_mat };
    const m_pbr: Material = .{ .pbr = &pbr_mat };
    try std.testing.expect(m_std.isCutout());
    try std.testing.expect(m_pbr.isCutout());
    try std.testing.expect(!m_std.isTransparent());
    try std.testing.expect(!m_pbr.isTransparent());

    // Every mode on both material kinds: only blend is transparent,
    // only cutout is cutout.
    for ([_]AlphaMode{ .@"opaque", .cutout, .blend }) |mode| {
        std_mat.alpha_mode = mode;
        pbr_mat.alpha_mode = mode;
        try std.testing.expectEqual(mode == .blend, std_mat.isTransparent());
        try std.testing.expectEqual(mode == .blend, pbr_mat.isTransparent());
        try std.testing.expectEqual(mode == .cutout, std_mat.isCutout());
        try std.testing.expectEqual(mode == .cutout, pbr_mat.isCutout());
    }
}

test "alpha_cutoff and double_sided defaults are back-compatible" {
    const std_mat = StandardMaterial.init("m");
    try std.testing.expectEqual(@as(f32, 0.5), std_mat.alpha_cutoff);
    try std.testing.expect(!std_mat.double_sided);

    const pbr_mat = PBRMaterial.init("p");
    try std.testing.expectEqual(@as(f32, 0.5), pbr_mat.alpha_cutoff);
    try std.testing.expect(!pbr_mat.double_sided);

    // Union-level accessors mirror the concrete materials.
    var std_mut = std_mat;
    var pbr_mut = pbr_mat;
    const m_std: Material = .{ .standard = &std_mut };
    const m_pbr: Material = .{ .pbr = &pbr_mut };
    try std.testing.expectEqual(@as(f32, 0.5), m_std.alphaCutoff());
    try std.testing.expectEqual(@as(f32, 0.5), m_pbr.alphaCutoff());
    try std.testing.expect(!m_std.isDoubleSided());
    try std.testing.expect(!m_pbr.isDoubleSided());

    std_mut.alpha_cutoff = 0.25;
    std_mut.double_sided = true;
    pbr_mut.alpha_cutoff = 0.75;
    pbr_mut.double_sided = true;
    try std.testing.expectEqual(@as(f32, 0.25), m_std.alphaCutoff());
    try std.testing.expectEqual(@as(f32, 0.75), m_pbr.alphaCutoff());
    try std.testing.expect(m_std.isDoubleSided());
    try std.testing.expect(m_pbr.isDoubleSided());
}
