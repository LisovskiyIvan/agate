//! Material library v1: named presets over ShaderMaterial hook registrations.
//!
//! Four ready-made materials — Sky, Gradient, Grid, TriPlanar — as reusable
//! value constructors on top of the existing shader-material infrastructure
//! (build.zig `user_shader_materials` hook snippets under
//! examples/shader_materials/matlib_*.glsl, runtime packing via
//! shader_material.zig). A preset is an ORDINARY ShaderMaterial value:
//! nothing here touches the GPU, allocates, or reads live render state, so
//! the staging discipline is inherited, not re-implemented (freeze with
//! material.buildShaderSnapshot at queue-build time like any other shader
//! material).
//!
//! Scene wiring (no registry changes needed — presets compose with the
//! existing CRUD):
//!
//! ```zig
//! const agate = @import("agate");
//! // Value form (pure, headless-safe):
//! var sm = agate.material_library.sky("sky", .{}) orelse return null;
//! // Scene-owned form: create through the registry, then apply options:
//! const slot = scene.createShaderMaterial("sky", "matlib_sky") orelse return null;
//! agate.material_library.applySky(slot, .{});
//! mesh.material = .{ .shader_material = slot };
//! ```
//!
//! Fail-closed: every constructor returns null when its shader name is not
//! registered ( shader_material.invalid_index materials are skipped by the
//! draw path) or when uniform packing fails — never a half-built material.
//!
//! Out of v1 scope (documented non-goals): node-graph authoring,
//! physically-based sky/atmosphere models, asset textures (all presets are
//! procedural; TriPlanar takes an OPTIONAL texture through the existing
//! ShaderMaterial.texture slot).

const std = @import("std");
const math = @import("math");
const Color3 = math.Color3;
const material = @import("material.zig");
const ShaderMaterial = material.ShaderMaterial;
const AlphaMode = material.AlphaMode;
const Texture = @import("texture.zig").Texture;
const shader_material = @import("shader_material.zig");

/// Shader registry name backing the Sky preset.
pub const sky_shader_name = "matlib_sky";
/// Shader registry name backing the Gradient preset.
pub const gradient_shader_name = "matlib_gradient";
/// Shader registry name backing the Grid preset.
pub const grid_shader_name = "matlib_grid";
/// Shader registry name backing the TriPlanar preset.
pub const triplanar_shader_name = "matlib_triplanar";

/// Preset identity for discovery (UI lists, sandbox pickers).
pub const PresetKind = enum {
    sky,
    gradient,
    grid,
    triplanar,
};

/// Static descriptor for one preset: registry key + human-facing metadata.
pub const PresetInfo = struct {
    kind: PresetKind,
    /// Name in shader_material's registry (Wyhash key source).
    shader_name: []const u8,
    /// Short display label.
    label: []const u8,
    /// One-line description.
    description: []const u8,
};

/// All v1 presets in creation order. Append-only: indices are not persisted
/// anywhere, but UI code may iterate this for picker lists.
pub const presets: [4]PresetInfo = .{
    .{
        .kind = .sky,
        .shader_name = sky_shader_name,
        .label = "Sky",
        .description = "Three-stop vertical gradient sky dome (unlit, procedural).",
    },
    .{
        .kind = .gradient,
        .shader_name = gradient_shader_name,
        .label = "Gradient",
        .description = "Two-stop height gradient tint multiplied into lit albedo.",
    },
    .{
        .kind = .grid,
        .shader_name = grid_shader_name,
        .label = "Grid",
        .description = "Procedural world-space XZ grid (no textures).",
    },
    .{
        .kind = .triplanar,
        .shader_name = triplanar_shader_name,
        .label = "TriPlanar",
        .description = "Three-axis procedural blend, optional texture via the primary slot.",
    },
};

/// Looks up a preset by its shader registry name. Null for unknown names
/// (discoverability helper; never fails closed rendering — it only gates
/// picker/UI code paths).
pub fn presetForName(shader_name: []const u8) ?PresetInfo {
    for (presets) |p| {
        if (std.mem.eql(u8, p.shader_name, shader_name)) return p;
    }
    return null;
}

/// Looks up a preset by kind.
pub fn presetForKind(kind: PresetKind) PresetInfo {
    for (presets) |p| {
        if (p.kind == kind) return p;
    }
    unreachable;
}

/// Options for the Sky preset. Defaults mirror the snippet's `// @param`
/// defaults in examples/shader_materials/matlib_sky.glsl — keep in sync.
pub const SkyOptions = struct {
    top: Color3 = Color3.new(0.20, 0.45, 0.80),
    horizon: Color3 = Color3.new(0.75, 0.85, 0.95),
    bottom: Color3 = Color3.new(0.10, 0.10, 0.12),
    exponent: f32 = 1.5,
    alpha: f32 = 1.0,
    alpha_mode: AlphaMode = .@"opaque",
    double_sided: bool = false,
};

/// Options for the Gradient preset. Defaults mirror
/// examples/shader_materials/matlib_gradient.glsl — keep in sync.
pub const GradientOptions = struct {
    top: Color3 = Color3.new(0.90, 0.95, 1.00),
    bottom: Color3 = Color3.new(0.15, 0.20, 0.30),
    scale: f32 = 0.5,
    offset: f32 = 0.5,
    alpha: f32 = 1.0,
    alpha_mode: AlphaMode = .@"opaque",
    double_sided: bool = false,
};

/// Options for the Grid preset. Defaults mirror
/// examples/shader_materials/matlib_grid.glsl — keep in sync.
/// `scale` is cells per world unit; `width` is line width in cell units
/// (sane range 0..0.5).
pub const GridOptions = struct {
    line: Color3 = Color3.white,
    fill: Color3 = Color3.new(0.10, 0.10, 0.10),
    scale: f32 = 1.0,
    width: f32 = 0.05,
    alpha: f32 = 1.0,
    alpha_mode: AlphaMode = .@"opaque",
    double_sided: bool = false,
};

/// Options for the TriPlanar preset. Defaults mirror
/// examples/shader_materials/matlib_triplanar.glsl — keep in sync.
/// Axis colors tint each plane; bind ShaderMaterial.texture for a textured
/// blend (unbound = white fallback = flat axis colors).
pub const TriPlanarOptions = struct {
    color_x: Color3 = Color3.new(1.00, 0.35, 0.35),
    color_y: Color3 = Color3.new(0.35, 1.00, 0.35),
    color_z: Color3 = Color3.new(0.35, 0.35, 1.00),
    scale: f32 = 0.5,
    sharpness: f32 = 2.0,
    alpha: f32 = 1.0,
    alpha_mode: AlphaMode = .@"opaque",
    double_sided: bool = false,
};

fn packColor(sm: *ShaderMaterial, param: []const u8, c: Color3) shader_material.SetUniformError!void {
    try sm.setUniform(param, .{ .vector = .{ c.r, c.g, c.b, 1.0 } });
}

/// Applies Sky options onto a live ShaderMaterial (scene-owned or value).
/// The material must already resolve to the matlib_sky registration;
/// unknown-param errors propagate (fail-closed: the caller keeps the old
/// material state on error — setUniform writes one param at a time).
pub fn applySky(sm: *ShaderMaterial, opts: SkyOptions) shader_material.SetUniformError!void {
    try packColor(sm, "u_sky_top", opts.top);
    try packColor(sm, "u_sky_horizon", opts.horizon);
    try packColor(sm, "u_sky_bottom", opts.bottom);
    try sm.setUniform("u_sky_exponent", .{ .scalar = opts.exponent });
    sm.alpha = opts.alpha;
    sm.alpha_mode = opts.alpha_mode;
    sm.double_sided = opts.double_sided;
}

/// Applies Gradient options onto a live ShaderMaterial.
pub fn applyGradient(sm: *ShaderMaterial, opts: GradientOptions) shader_material.SetUniformError!void {
    try packColor(sm, "u_grad_top", opts.top);
    try packColor(sm, "u_grad_bottom", opts.bottom);
    try sm.setUniform("u_grad_scale", .{ .scalar = opts.scale });
    try sm.setUniform("u_grad_offset", .{ .scalar = opts.offset });
    sm.alpha = opts.alpha;
    sm.alpha_mode = opts.alpha_mode;
    sm.double_sided = opts.double_sided;
}

/// Applies Grid options onto a live ShaderMaterial.
pub fn applyGrid(sm: *ShaderMaterial, opts: GridOptions) shader_material.SetUniformError!void {
    try packColor(sm, "u_grid_line", opts.line);
    try packColor(sm, "u_grid_fill", opts.fill);
    try sm.setUniform("u_grid_scale", .{ .scalar = opts.scale });
    try sm.setUniform("u_grid_width", .{ .scalar = opts.width });
    sm.alpha = opts.alpha;
    sm.alpha_mode = opts.alpha_mode;
    sm.double_sided = opts.double_sided;
}

/// Applies TriPlanar options onto a live ShaderMaterial.
pub fn applyTriPlanar(sm: *ShaderMaterial, opts: TriPlanarOptions) shader_material.SetUniformError!void {
    try packColor(sm, "u_tri_x", opts.color_x);
    try packColor(sm, "u_tri_y", opts.color_y);
    try packColor(sm, "u_tri_z", opts.color_z);
    try sm.setUniform("u_tri_scale", .{ .scalar = opts.scale });
    try sm.setUniform("u_tri_sharp", .{ .scalar = opts.sharpness });
    sm.alpha = opts.alpha;
    sm.alpha_mode = opts.alpha_mode;
    sm.double_sided = opts.double_sided;
}

/// Value constructor: Sky preset. Null when matlib_sky is not registered
/// or packing fails (fail-closed).
pub fn sky(material_name: []const u8, opts: SkyOptions) ?ShaderMaterial {
    var sm = ShaderMaterial.initForShader(sky_shader_name, material_name) orelse return null;
    applySky(&sm, opts) catch return null;
    return sm;
}

/// Value constructor: Gradient preset. Null when unregistered / packing fails.
pub fn gradient(material_name: []const u8, opts: GradientOptions) ?ShaderMaterial {
    var sm = ShaderMaterial.initForShader(gradient_shader_name, material_name) orelse return null;
    applyGradient(&sm, opts) catch return null;
    return sm;
}

/// Value constructor: Grid preset. Null when unregistered / packing fails.
pub fn grid(material_name: []const u8, opts: GridOptions) ?ShaderMaterial {
    var sm = ShaderMaterial.initForShader(grid_shader_name, material_name) orelse return null;
    applyGrid(&sm, opts) catch return null;
    return sm;
}

/// Value constructor: TriPlanar preset. Null when unregistered / packing fails.
pub fn triPlanar(material_name: []const u8, opts: TriPlanarOptions) ?ShaderMaterial {
    var sm = ShaderMaterial.initForShader(triplanar_shader_name, material_name) orelse return null;
    applyTriPlanar(&sm, opts) catch return null;
    return sm;
}

// ---------------------------------------------------------------------------
// Pure CPU mirrors of the snippet math (unit-testable without GPU).
// ---------------------------------------------------------------------------

/// CPU mirror of the sky vertical blend: 0 = horizon color, 1 = pole color.
/// Matches `mix(horizon, pole, pow(clamp(|h|,0,1), exponent))` in
/// matlib_sky.glsl.
pub fn skyBlendT(h: f32, exponent: f32) f32 {
    const t = @min(@max(@abs(h), 0.0), 1.0);
    return std.math.pow(f32, t, exponent);
}

/// CPU mirror of the gradient height factor.
/// Matches `clamp(y * scale + offset, 0, 1)` in matlib_gradient.glsl.
pub fn gradientT(y: f32, scale: f32, offset: f32) f32 {
    return @min(@max(y * scale + offset, 0.0), 1.0);
}

/// CPU mirror of the grid line mask (1 = line, 0 = fill).
/// Matches the fract/min/smoothstep chain in matlib_grid.glsl
/// (smoothstep(0, width, d) with the standard cubic easing).
pub fn gridLineMask(world_x: f32, world_z: f32, scale: f32, width: f32) f32 {
    const fx = world_x * scale - @floor(world_x * scale);
    const fz = world_z * scale - @floor(world_z * scale);
    const dx = @min(fx, 1.0 - fx);
    const dz = @min(fz, 1.0 - fz);
    const d = @min(dx, dz);
    const t = @min(@max(d / @max(width, 1e-6), 0.0), 1.0);
    const s = t * t * (3.0 - 2.0 * t);
    return 1.0 - s;
}

/// CPU mirror of the triplanar axis weights.
/// Matches `pow(abs(N), sharp) / sum` in matlib_triplanar.glsl.
pub fn triplanarWeights(n: [3]f32, sharpness: f32) [3]f32 {
    const wx = std.math.pow(f32, @abs(n[0]), sharpness);
    const wy = std.math.pow(f32, @abs(n[1]), sharpness);
    const wz = std.math.pow(f32, @abs(n[2]), sharpness);
    const sum = @max(wx + wy + wz, 0.0001);
    return .{ wx / sum, wy / sum, wz / sum };
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

test "presets table covers all four kinds with unique registry names" {
    try std.testing.expectEqual(@as(usize, 4), presets.len);
    // Unique shader names (registry keys must not collide).
    for (presets, 0..) |a, i| {
        for (presets, 0..) |b, j| {
            if (i != j) try std.testing.expect(!std.mem.eql(u8, a.shader_name, b.shader_name));
        }
        // Round-trips through both lookup directions.
        const by_name = presetForName(a.shader_name) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqual(a.kind, by_name.kind);
        try std.testing.expectEqual(a, presetForKind(a.kind));
        // Registry key contract: the name hashes deterministically.
        try std.testing.expectEqual(shader_material.keyForName(a.shader_name), shader_material.keyForName(a.shader_name));
    }
    try std.testing.expect(presetForName("does_not_exist") == null);
    try std.testing.expect(presetForName("") == null);
}

test "constructors resolve registrations with fail-closed defaults" {
    // Unregistered names fail softly (null) — the headless fail-closed path.
    try std.testing.expect(ShaderMaterial.initForShader("matlib_nope_xyz", "x") == null);

    var s = sky("s", .{}) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings(sky_shader_name, s.shader_name);
    try std.testing.expect(s.entry_index != shader_material.invalid_index);
    try std.testing.expect(!s.isTransparent() and !s.isCutout());

    const g = gradient("g", .{}) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings(gradient_shader_name, g.shader_name);

    const gr = grid("gr", .{}) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings(grid_shader_name, gr.shader_name);

    const t = triPlanar("t", .{}) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings(triplanar_shader_name, t.shader_name);

    // Default options land in the packed storage at the declared offsets
    // (snippet @param defaults mirrored by the Options structs).
    const sky_entry = shader_material.entry(s.entry_index).?;
    const top = shader_material.findParam(sky_entry.params, "u_sky_top").?;
    try std.testing.expectEqualSlices(f32, &.{ 0.20, 0.45, 0.80, 1.0 }, &s.uniforms[top.offset / 4]);
    const exp = shader_material.findParam(sky_entry.params, "u_sky_exponent").?;
    try std.testing.expectEqual(@as(f32, 1.5), s.uniforms[exp.offset / 4][exp.offset % 4]);
}

test "options packing round-trips through the declarative offsets" {
    var s = sky("s", .{
        .top = Color3.new(1, 0, 0),
        .exponent = 3.0,
        .alpha = 0.5,
        .alpha_mode = .blend,
        .double_sided = true,
    }) orelse return error.TestUnexpectedResult;
    const entry = shader_material.entry(s.entry_index).?;
    const top = shader_material.findParam(entry.params, "u_sky_top").?;
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 0, 1 }, &s.uniforms[top.offset / 4]);
    const exp = shader_material.findParam(entry.params, "u_sky_exponent").?;
    try std.testing.expectEqual(@as(f32, 3.0), s.uniforms[exp.offset / 4][exp.offset % 4]);
    try std.testing.expect(s.isTransparent());
    try std.testing.expect(s.double_sided);
    // Tint is untouched by the preset options (top color lives in the
    // uniform block); only alpha flows into the tint upload.
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 0.5 }, &s.getTintColor4());

    var gr = grid("gr", .{ .scale = 4.0, .width = 0.1 }) orelse return error.TestUnexpectedResult;
    const gentry = shader_material.entry(gr.entry_index).?;
    const sc = shader_material.findParam(gentry.params, "u_grid_scale").?;
    try std.testing.expectEqual(@as(f32, 4.0), gr.uniforms[sc.offset / 4][sc.offset % 4]);

    // apply* on a live (scene-owned style) material rewrites the same lanes.
    try applyGrid(&gr, .{ .scale = 8.0 });
    try std.testing.expectEqual(@as(f32, 8.0), gr.uniforms[sc.offset / 4][sc.offset % 4]);
}

test "apply on a wrong-registration material is a hard error (typo guard)" {
    // A ShaderMaterial bound to a DIFFERENT registration (ramp_wave) has no
    // matlib params: applySky must fail instead of silently writing lanes.
    var other = ShaderMaterial.initForShader("ramp_wave", "fx") orelse return error.TestUnexpectedResult;
    try std.testing.expectError(error.UnknownParam, applySky(&other, .{}));
    try std.testing.expectError(error.UnknownParam, applyGradient(&other, .{}));
    try std.testing.expectError(error.UnknownParam, applyGrid(&other, .{}));
    try std.testing.expectError(error.UnknownParam, applyTriPlanar(&other, .{}));
}

test "preset snapshots freeze like any shader material (staging discipline)" {
    var t = triPlanar("t", .{}) orelse return error.TestUnexpectedResult;
    const white = Texture{ .image = .{}, .view = .{ .id = 11 }, .sampler = .{ .id = 12 }, .width = 4, .height = 4 };
    const snap = material.buildShaderSnapshot(.{ .shader_material = &t }, &white) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(t.entry_index, snap.entry_index);
    // Live mutation after the snapshot leaves the frozen copy untouched.
    const sharp = shader_material.findParam(shader_material.entry(t.entry_index).?.params, "u_tri_sharp").?;
    const before = snap.uniforms[sharp.offset / 4][sharp.offset % 4];
    t.uniforms[sharp.offset / 4][sharp.offset % 4] = before + 10.0;
    try std.testing.expectEqual(before, snap.uniforms[sharp.offset / 4][sharp.offset % 4]);
}

test "skyBlendT mirrors the snippet mix factor" {
    try std.testing.expectEqual(@as(f32, 0.0), skyBlendT(0.0, 1.5));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), skyBlendT(1.0, 1.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), skyBlendT(-1.0, 1.5), 1e-6);
    // pow curve: exponent 2 at |h| = 0.5 gives 0.25.
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), skyBlendT(0.5, 2.0), 1e-6);
    // Clamped past the poles.
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), skyBlendT(3.0, 1.5), 1e-6);
}

test "gradientT mirrors the snippet height factor" {
    try std.testing.expectEqual(@as(f32, 0.5), gradientT(0.0, 0.5, 0.5));
    try std.testing.expectEqual(@as(f32, 1.0), gradientT(1.0, 0.5, 0.5));
    try std.testing.expectEqual(@as(f32, 0.0), gradientT(-1.0, 0.5, 0.5));
    // Clamped past the ends.
    try std.testing.expectEqual(@as(f32, 1.0), gradientT(100.0, 0.5, 0.5));
    try std.testing.expectEqual(@as(f32, 0.0), gradientT(-100.0, 0.5, 0.5));
}

test "gridLineMask is 1 on lines and 0 in cell centers" {
    // Cell corners/edges are lines (fract ~ 0).
    try std.testing.expect(gridLineMask(0.0, 0.0, 1.0, 0.05) > 0.99);
    try std.testing.expect(gridLineMask(2.0, 3.0, 1.0, 0.05) > 0.99);
    // Cell center is fill.
    try std.testing.expect(gridLineMask(0.5, 0.5, 1.0, 0.05) < 0.01);
    // Scale multiplies the frequency: x = 0.25 at scale 4 sits on a line.
    try std.testing.expect(gridLineMask(0.25, 0.25, 4.0, 0.05) > 0.99);
    // Mask stays in [0, 1] everywhere.
    var x: f32 = 0.0;
    while (x < 1.0) : (x += 0.07) {
        const m = gridLineMask(x, 0.33, 2.0, 0.1);
        try std.testing.expect(m >= 0.0 and m <= 1.0);
    }
}

test "triplanarWeights mirror the snippet axis blend" {
    // Face-on normals select their axis fully.
    const wy = triplanarWeights(.{ 0, 1, 0 }, 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), wy[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), wy[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), wy[2], 1e-6);
    // Diagonal normals split evenly; weights always sum to 1.
    const diag = triplanarWeights(.{ 1, 1, 1 }, 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), diag[0] + diag[1] + diag[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), diag[0], 1e-6);
    // Sharpness 0 spreads uniformly (pow(x, 0) = 1 per axis).
    const flat = triplanarWeights(.{ 0.2, 0.9, 0.1 }, 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), flat[1], 1e-6);
    // Sign-insensitive (abs in the snippet).
    const neg = triplanarWeights(.{ 0, -1, 0 }, 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), neg[1], 1e-6);
}
