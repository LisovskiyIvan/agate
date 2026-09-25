const std = @import("std");
const math = @import("math");
const Color3 = math.Color3;
const Texture = @import("../texture.zig").Texture;
const types = @import("types.zig");
const AlphaMode = types.AlphaMode;
const shader_material = @import("../shader_material.zig");

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
    /// Optional second texture for runtime-registered sources
    /// (engine_template = false): bound to view/sampler slot 1 (the
    /// primary occupies slot 0). Ignored by engine-template materials,
    /// whose texture contract is fixed by the template. Null falls back
    /// to the default white texture at snapshot build time.
    texture1: ?Texture = null,

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
