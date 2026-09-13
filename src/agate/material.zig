const sokol = @import("sokol");
const sg = sokol.gfx;
const std = @import("std");
const math = @import("math");
const Color3 = math.Color3;
const Color4 = math.Color4;
const Texture = @import("texture.zig").Texture;
const CubeTexture = @import("texture.zig").CubeTexture;
const shader_material = @import("shader_material.zig");

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

/// Custom-shader material: renders with a REGISTERED shader (build-time hook
/// material from build.zig's `user_shader_materials`, or runtime-registered
/// via shader_material.registerRuntime) instead of the built-in shaders.
///
/// The draw path resolves `entry_index` through shader_material.entry(),
/// lazily creates the pipeline set on first use (cached by the registration
/// key) and uploads the packed `uniforms` block plus `tint`/alpha state.
/// Registering by name:
///   const mat = try scene.createShaderMaterial("fx", "ramp_wave");
///   mesh.material = .{ .shader_material = mat };
pub const ShaderMaterial = struct {
    name: []const u8 = "ShaderMaterial",
    /// Registration index from shader_material.indexForName/registerRuntime.
    /// shader_material.invalid_index (default) makes the draw path skip the
    /// mesh instead of rendering with an unregistered shader.
    entry_index: u32 = shader_material.invalid_index,
    /// Cached registration name (diagnostics; resolution is index-based).
    shader_name: []const u8 = "",

    /// Multiplied into the shader's albedo hook input (diffuse_color for the
    /// standard base, base_color_factor for the pbr base).
    tint_color: Color3 = Color3.white,
    alpha: f32 = 1.0,
    alpha_mode: AlphaMode = .@"opaque",
    /// Alpha-test threshold used only when alpha_mode == .cutout (mirrors
    /// StandardMaterial/PBRMaterial semantics).
    alpha_cutoff: f32 = 0.5,
    /// Face culling disabled (a cull-off pipeline twin) — engine-template
    /// materials only; runtime-registered sources must handle winding.
    double_sided: bool = false,
    /// Bound to the material's primary texture slot (diffuse_tex /
    /// albedo_tex for engine templates; view slot 0 for runtime sources).
    texture: ?Texture = null,

    /// Packed user uniform block (8x vec4). Initialized from the declared
    /// param defaults when the registration is resolved (initForShader);
    /// raw-zero when built with defaults.
    uniforms: shader_material.UniformStorage = .{.{ 0, 0, 0, 0 }} ** shader_material.merge.user_slot_count,

    /// Creates a material pointing at a registered shader by name. Null when
    /// the name is not registered (static registry or registerRuntime).
    pub fn initForShader(name: []const u8, material_name: []const u8) ?ShaderMaterial {
        const index = shader_material.indexForName(name) orelse return null;
        const entry = shader_material.entry(index).?;
        return .{
            .name = material_name,
            .entry_index = index,
            .shader_name = entry.name,
            .uniforms = shader_material.defaultUniformStorage(entry.params),
        };
    }

    pub fn init(name: []const u8) ShaderMaterial {
        return .{ .name = name };
    }

    /// Re-resolves the registration and resets the uniform storage to its
    /// declared defaults (also usable after changing entry_index directly).
    pub fn resetUniformDefaults(self: *ShaderMaterial) void {
        if (shader_material.entry(self.entry_index)) |entry| {
            self.uniforms = shader_material.defaultUniformStorage(entry.params);
        }
    }

    /// Packs one named user uniform value (declarative table lookup; hard
    /// error for unknown names / component mismatches — a typo guard).
    pub fn setUniform(self: *ShaderMaterial, name: []const u8, value: shader_material.UniformValue) shader_material.SetUniformError!void {
        const entry = shader_material.entry(self.entry_index) orelse return error.UnknownParam;
        return shader_material.setUniform(&self.uniforms, entry.params, name, value);
    }

    pub fn getTintColor4(self: ShaderMaterial) [4]f32 {
        return .{ self.tint_color.r, self.tint_color.g, self.tint_color.b, self.alpha };
    }

    pub fn isTransparent(self: ShaderMaterial) bool {
        return self.alpha_mode == .blend;
    }

    pub fn isCutout(self: ShaderMaterial) bool {
        return self.alpha_mode == .cutout;
    }
};

pub const Material = union(enum) {
    standard: *StandardMaterial,
    pbr: *PBRMaterial,
    shader_material: *ShaderMaterial,

    pub fn isTransparent(self: Material) bool {
        return switch (self) {
            .standard => |s| s.isTransparent(),
            .pbr => |p| p.isTransparent(),
            .shader_material => |sm| sm.isTransparent(),
        };
    }

    pub fn isCutout(self: Material) bool {
        return switch (self) {
            .standard => |s| s.isCutout(),
            .pbr => |p| p.isCutout(),
            .shader_material => |sm| sm.isCutout(),
        };
    }

    pub fn isDoubleSided(self: Material) bool {
        return switch (self) {
            .standard => |s| s.double_sided,
            .pbr => |p| p.double_sided,
            .shader_material => |sm| sm.double_sided,
        };
    }

    /// Raw per-material cutoff (defaults to 0.5). The draw path gates this:
    /// only cutout materials upload a live cutoff, otherwise 0.0 (disabled).
    /// See scene/uniforms.zig alphaCutoffFor.
    pub fn alphaCutoff(self: Material) f32 {
        return switch (self) {
            .standard => |s| s.alpha_cutoff,
            .pbr => |p| p.alpha_cutoff,
            .shader_material => |sm| sm.alpha_cutoff,
        };
    }

    /// The material's primary texture (diffuse/albedo slot), or null.
    pub fn primaryTexture(self: Material) ?Texture {
        return switch (self) {
            .standard => |s| s.diffuse_texture,
            .pbr => |p| p.albedo_texture,
            .shader_material => |sm| sm.texture,
        };
    }

    /// Tint * alpha as uploaded to the albedo color uniform.
    pub fn tintColor4(self: Material) [4]f32 {
        return switch (self) {
            .standard => |s| s.getDiffuseColor4(),
            .pbr => |p| p.getAlbedoColor4(),
            .shader_material => |sm| sm.getTintColor4(),
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

test "ShaderMaterial defaults mirror the legacy material contract" {
    var sm = ShaderMaterial.init("fx");
    try std.testing.expect(!sm.isTransparent());
    try std.testing.expect(!sm.isCutout());
    try std.testing.expect(!sm.double_sided);
    try std.testing.expectEqual(@as(f32, 0.5), sm.alpha_cutoff);
    try std.testing.expectEqual(@as(f32, 1.0), sm.alpha);
    // Unresolved registration: draw paths skip the material.
    try std.testing.expectEqual(shader_material.invalid_index, sm.entry_index);

    // initForShader resolves through the registry and seeds declared
    // defaults; unknown names fail softly (null).
    try std.testing.expect(ShaderMaterial.initForShader("no_such_shader_xyz", "fx") == null);
    const registered = ShaderMaterial.initForShader("ramp_wave", "fx") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("ramp_wave", registered.shader_name);
    sm.resetUniformDefaults();
    _ = &sm;
}

test "Material union exposes shader_material classification and tint" {
    var sm = ShaderMaterial.init("fx");
    const m: Material = .{ .shader_material = &sm };

    try std.testing.expect(!m.isTransparent());
    try std.testing.expect(!m.isCutout());
    try std.testing.expect(!m.isDoubleSided());
    try std.testing.expectEqual(@as(f32, 0.5), m.alphaCutoff());
    try std.testing.expect(m.primaryTexture() == null);
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1 }, &m.tintColor4());

    sm.alpha_mode = .blend;
    sm.alpha_cutoff = 0.25;
    sm.double_sided = true;
    sm.tint_color = Color3.new(0.25, 0.5, 0.75);
    sm.alpha = 0.5;
    sm.texture = .{
        .image = .{ .id = 1 },
        .view = .{ .id = 2 },
        .sampler = .{ .id = 3 },
        .width = 4,
        .height = 4,
    };
    try std.testing.expect(m.isTransparent());
    try std.testing.expect(!m.isCutout());
    try std.testing.expect(m.isDoubleSided());
    try std.testing.expectEqual(@as(f32, 0.25), m.alphaCutoff());
    try std.testing.expectEqual(@as(u32, 2), m.primaryTexture().?.view.id);
    try std.testing.expectEqualSlices(f32, &.{ 0.25, 0.5, 0.75, 0.5 }, &m.tintColor4());

    // Cutout classification across all modes, matching standard/pbr.
    sm.alpha_mode = .cutout;
    try std.testing.expect(m.isCutout());
    try std.testing.expect(!m.isTransparent());
}

test "ShaderMaterial.setUniform packs through the registration table" {
    // ramp_wave is registered by build.zig (user_shader_materials).
    const name = "ramp_wave";
    const index = shader_material.indexForName(name) orelse {
        std.debug.print("ramp_wave not registered (static registry empty?)\n", .{});
        return error.TestUnexpectedResult;
    };
    var sm = ShaderMaterial.initForShader(name, "fx") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(index, sm.entry_index);

    // Declared defaults landed in the packed storage.
    const entry = shader_material.entry(index).?;
    for (entry.params) |p| {
        const slot = p.offset / 4;
        if (p.comps == 1) {
            try std.testing.expectEqual(p.default[0], sm.uniforms[slot][p.offset % 4]);
        } else {
            try std.testing.expectEqualSlices(f32, &p.default, &sm.uniforms[slot]);
        }
    }

    // Named writes hit the declared offsets (GLSL defines map 1:1).
    try sm.setUniform("u_wave_speed", .{ .scalar = 9.0 });
    try sm.setUniform("u_ramp_high", .{ .vector = .{ 1, 0, 0, 1 } });
    const speed = entry.params[0];
    try std.testing.expectEqualStrings("u_wave_speed", speed.name);
    try std.testing.expectEqual(@as(f32, 9.0), sm.uniforms[speed.offset / 4][speed.offset % 4]);
    const ramp = shader_material.findParam(entry.params, "u_ramp_high").?;
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 0, 1 }, &sm.uniforms[ramp.offset / 4]);

    // Typo guard: unknown names are hard errors.
    try std.testing.expectError(error.UnknownParam, sm.setUniform("u_nope", .{ .scalar = 1 }));
}
