const std = @import("std");
const math = @import("math");
const Color3 = math.Color3;
const Color4 = math.Color4;
const Texture = @import("../texture.zig").Texture;
const CubeTexture = @import("../texture.zig").CubeTexture;
const shader_material = @import("../shader_material.zig");

const types = @import("types.zig");
const AlphaMode = types.AlphaMode;
const Channel = types.Channel;
const UvTransform = types.UvTransform;
const Clearcoat = types.Clearcoat;
const Sheen = types.Sheen;
const Anisotropy = types.Anisotropy;
const Transmission = types.Transmission;
const Subsurface = types.Subsurface;
const CoatParams = types.CoatParams;
const anisotropyAxes = types.anisotropyAxes;
const wrapNdotL = types.wrapNdotL;

const standard = @import("standard.zig");
const StandardMaterial = standard.StandardMaterial;

const pbr = @import("pbr.zig");
const PBRMaterial = pbr.PBRMaterial;

const shader_mat = @import("shader_mat.zig");
const ShaderMaterial = shader_mat.ShaderMaterial;

const union_mod = @import("union.zig");
const Material = union_mod.Material;
const coatParamsFor = union_mod.coatParamsFor;

const draw_record = @import("draw_record.zig");
const MaterialDrawRecord = draw_record.MaterialDrawRecord;
const ShaderDrawSnapshot = draw_record.ShaderDrawSnapshot;
const buildShaderSnapshot = draw_record.buildShaderSnapshot;
const buildDrawRecord = draw_record.buildDrawRecord;

test "refraction is opt-in, staged and classified as transparent" {
    var mat = PBRMaterial.init("glass");
    mat.transmission.factor = 1;
    try std.testing.expect(!mat.isTransparent());
    mat.transmission.refract = true;
    mat.transmission.thickness = 0.7;
    try std.testing.expect(mat.isTransparent());
    const cp = coatParamsFor(.{ .pbr = &mat }).?;
    try std.testing.expectEqualSlices(f32, &.{ 1, 0.7, 1.5, 0 }, &cp.refraction_factors);
    mat.transmission.ior = 0;
    try std.testing.expectEqual(@as(f32, 1), mat.transmission.refractionPacked()[2]);
    mat.transmission.factor = 0;
    try std.testing.expect(!mat.isTransparent());
}

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

test "buildShaderSnapshot freezes handles, uniforms and the second texture" {
    // Headless: handles are plain ids, no sg calls involved.
    const white = Texture{ .image = .{ .id = 10 }, .view = .{ .id = 11 }, .sampler = .{ .id = 12 }, .width = 4, .height = 4 };
    var sm = ShaderMaterial.initForShader("ramp_wave", "fx") orelse return error.TestUnexpectedResult;
    sm.texture = .{ .image = .{ .id = 1 }, .view = .{ .id = 2 }, .sampler = .{ .id = 3 }, .width = 4, .height = 4 };
    sm.texture1 = .{ .image = .{ .id = 4 }, .view = .{ .id = 5 }, .sampler = .{ .id = 6 }, .width = 4, .height = 4 };
    try sm.setUniform("u_wave_speed", .{ .scalar = 7.0 });
    sm.tint_color = Color3.new(0.5, 0.25, 0.125);
    sm.alpha = 0.75;
    sm.double_sided = true;

    const m: Material = .{ .shader_material = &sm };
    const snap = buildShaderSnapshot(m, &white) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(sm.entry_index, snap.entry_index);
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 0.25, 0.125, 0.75 }, &snap.tint);
    try std.testing.expectEqual(@as(u32, 2), snap.tex_view.id);
    try std.testing.expectEqual(@as(u32, 3), snap.tex_sampler.id);
    try std.testing.expectEqual(@as(u32, 5), snap.tex1_view.id);
    try std.testing.expectEqual(@as(u32, 6), snap.tex1_sampler.id);
    const speed = shader_material.findParam(shader_material.entry(sm.entry_index).?.params, "u_wave_speed").?;
    try std.testing.expectEqual(@as(f32, 7.0), snap.uniforms[speed.offset / 4][speed.offset % 4]);
    try std.testing.expect(snap.double_sided);

    // Live mutation after the snapshot leaves the frozen copy untouched
    // (staging discipline: the draw path must never see this write).
    sm.texture1.?.view.id = 50;
    sm.uniforms[speed.offset / 4][speed.offset % 4] = 1.0;
    try std.testing.expectEqual(@as(u32, 5), snap.tex1_view.id);
    try std.testing.expectEqual(@as(f32, 7.0), snap.uniforms[speed.offset / 4][speed.offset % 4]);

    // Null textures fall back to the default (white) handles.
    var bare = ShaderMaterial.initForShader("ramp_wave", "bare") orelse return error.TestUnexpectedResult;
    const bare_mat: Material = .{ .shader_material = &bare };
    const bare_snap = buildShaderSnapshot(bare_mat, &white) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u32, 11), bare_snap.tex_view.id);
    try std.testing.expectEqual(@as(u32, 11), bare_snap.tex1_view.id);

    // Non-shader materials and null build no snapshot.
    try std.testing.expect(buildShaderSnapshot(null, &white) == null);
}

test "UvTransform packs the KHR_texture_transform matrix rows" {
    const ident = UvTransform.identity;
    try std.testing.expect(ident.isIdentity());
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 0, 1 }, &ident.matrixRows());
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0 }, &ident.offsetPacked());

    // Scale only: 2x on u, 3x on v.
    const scaled = UvTransform{ .scale = .{ 2, 3 } };
    try std.testing.expect(!scaled.isIdentity());
    try std.testing.expectEqualSlices(f32, &.{ 2, 0, 0, 3 }, &scaled.matrixRows());

    // 90 degrees CCW rotation with unit scale: (u,v) -> (-v, u).
    const quarter = std.math.pi / 2.0;
    const rotated = UvTransform{ .rotation = quarter };
    try std.testing.expectApproxEqAbs(@as(f32, 0), rotated.matrixRows()[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -1), rotated.matrixRows()[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), rotated.matrixRows()[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), rotated.matrixRows()[3], 1e-6);

    // Spec formula spot-check: u' = cos*sx*u - sin*sy*v + ox,
    // v' = sin*sx*u + cos*sy*v + oy for rotation 45deg, scale (2, 4).
    const t = UvTransform{ .offset = .{ 0.25, -0.5 }, .rotation = quarter / 2.0, .scale = .{ 2, 4 } };
    const m = t.matrixRows();
    const cos_h: f32 = @cos(quarter / 2.0);
    const sin_h: f32 = @sin(quarter / 2.0);
    const u: f32 = 0.75;
    const v: f32 = 1.5;
    const u_prime = m[0] * u + m[1] * v + t.offset[0];
    const v_prime = m[2] * u + m[3] * v + t.offset[1];
    try std.testing.expectApproxEqAbs(cos_h * 2 * u - sin_h * 4 * v + 0.25, u_prime, 1e-5);
    try std.testing.expectApproxEqAbs(sin_h * 2 * u + cos_h * 4 * v - 0.5, v_prime, 1e-5);
}

test "PBRMaterial clearcoat/sheen defaults are disabled and neutral" {
    const mat = PBRMaterial.init("m");
    // Disabled: intensity 0 keeps every shader term at zero.
    try std.testing.expectEqual(@as(f32, 0.0), mat.clearcoat.intensity);
    try std.testing.expectEqual(@as(f32, 0.0), mat.sheen.intensity);
    // Neutral companions: untinted colors, sane roughnesses.
    try std.testing.expectEqual(Color3.white, mat.clearcoat.color);
    try std.testing.expectEqual(Color3.white, mat.sheen.color);
}

test "CoatParams snapshot is present when enabled, neutral when disabled" {
    // Disabled (default): no side-table entry — draws use CoatParams.neutral.
    var off = PBRMaterial.init("off");
    try std.testing.expect(coatParamsFor(.{ .pbr = &off }) == null);
    try std.testing.expect(coatParamsFor(null) == null);
    var std_mat = StandardMaterial.init("s");
    try std.testing.expect(coatParamsFor(.{ .standard = &std_mat }) == null);

    // Neutral fallback: intensities zero, companion defaults.
    const n = CoatParams.neutral;
    try std.testing.expectEqualSlices(f32, &.{ 0, 0.03, 0, 0 }, &n.clearcoat_factors);
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1 }, &n.clearcoat_color);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0.5, 0, 0 }, &n.sheen_factors);
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1 }, &n.sheen_color);

    // Enabled (either lobe): material values flow through verbatim.
    var on = PBRMaterial.init("on");
    on.clearcoat = .{ .intensity = 0.8, .roughness = 0.12, .color = Color3.new(0.9, 0.8, 0.7) };
    on.sheen = .{ .color = Color3.new(0.2, 0.4, 0.6), .intensity = 0.5, .roughness = 0.35 };
    const cp = coatParamsFor(.{ .pbr = &on }) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualSlices(f32, &.{ 0.8, 0.12, 0, 0 }, &cp.clearcoat_factors);
    try std.testing.expectEqualSlices(f32, &.{ 0.9, 0.8, 0.7, 1 }, &cp.clearcoat_color);
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 0.35, 0, 0 }, &cp.sheen_factors);
    try std.testing.expectEqualSlices(f32, &.{ 0.2, 0.4, 0.6, 1 }, &cp.sheen_color);

    // One lobe alone still opts in.
    var half = PBRMaterial.init("half");
    half.sheen.intensity = 0.25;
    try std.testing.expect(coatParamsFor(.{ .pbr = &half }) != null);
}

test "PBR layers v1 defaults are disabled and neutral" {
    const mat = PBRMaterial.init("m");
    // Scalars: every new layer defaults to off.
    try std.testing.expectEqual(@as(f32, 0.0), mat.anisotropy.intensity);
    try std.testing.expectEqual(@as(f32, 0.0), mat.anisotropy.rotation);
    try std.testing.expectEqual(@as(f32, 0.0), mat.transmission.factor);
    try std.testing.expectEqual(Color3.white, mat.transmission.color);
    try std.testing.expectEqual(@as(f32, 0.0), mat.subsurface.strength);
    try std.testing.expectEqual(Color3.white, mat.subsurface.color);
    // Textures: null slots keep the scalar path (white-fallback staging).
    try std.testing.expect(mat.clearcoat.mask_texture == null);
    try std.testing.expect(mat.sheen.color_texture == null);
    // The extended neutral snapshot stays all-zero on factors.
    const n = CoatParams.neutral;
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0 }, &n.anisotropy_factors);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0 }, &n.transmission_factors);
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1 }, &n.transmission_color);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0 }, &n.sss_factors);
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1 }, &n.sss_color);
}

test "coatParamsFor gates on any layer, textures alone stay null" {
    // Default material: null (legacy path, no side-table slot).
    var off = PBRMaterial.init("off");
    try std.testing.expect(coatParamsFor(.{ .pbr = &off }) == null);

    // A bound mask/tint texture with zero intensity does NOT opt in: the
    // mask multiplies a zero intensity, so null stays bit-identical.
    const tex = Texture{ .image = .{}, .view = .{ .id = 9 }, .sampler = .{ .id = 10 }, .width = 4, .height = 4 };
    off.clearcoat.mask_texture = tex;
    off.sheen.color_texture = tex;
    try std.testing.expect(coatParamsFor(.{ .pbr = &off }) == null);
    off.clearcoat.mask_texture = null;
    off.sheen.color_texture = null;

    // Each new layer alone opts in (negative values stay off).
    var aniso = PBRMaterial.init("aniso");
    aniso.anisotropy = .{ .intensity = 0.6, .rotation = 0.5 };
    try std.testing.expect(coatParamsFor(.{ .pbr = &aniso }) != null);
    var transm = PBRMaterial.init("transm");
    transm.transmission = .{ .factor = 0.4, .color = Color3.new(0.8, 0.9, 1.0) };
    try std.testing.expect(coatParamsFor(.{ .pbr = &transm }) != null);
    var sss = PBRMaterial.init("sss");
    sss.subsurface = .{ .strength = 0.7, .color = Color3.new(1.0, 0.4, 0.3) };
    try std.testing.expect(coatParamsFor(.{ .pbr = &sss }) != null);
    var neg = PBRMaterial.init("neg");
    neg.anisotropy.intensity = -1.0;
    neg.transmission.factor = -0.5;
    neg.subsurface.strength = -2.0;
    try std.testing.expect(coatParamsFor(.{ .pbr = &neg }) == null);
}

test "coatParamsFor packs anisotropy/transmission/sss verbatim" {
    var mat = PBRMaterial.init("m");
    mat.anisotropy = .{ .intensity = 0.6, .rotation = 1.25 };
    mat.transmission = .{ .factor = 0.4, .color = Color3.new(0.8, 0.9, 1.0) };
    mat.subsurface = .{ .strength = 0.7, .color = Color3.new(1.0, 0.4, 0.3) };
    const cp = coatParamsFor(.{ .pbr = &mat }) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualSlices(f32, &.{ 0.6, 1.25, 0, 0 }, &cp.anisotropy_factors);
    try std.testing.expectEqualSlices(f32, &.{ 0.4, 0, 0, 0 }, &cp.transmission_factors);
    try std.testing.expectEqualSlices(f32, &.{ 0.8, 0.9, 1.0, 1 }, &cp.transmission_color);
    try std.testing.expectEqualSlices(f32, &.{ 0.7, 0, 0, 0 }, &cp.sss_factors);
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 0.4, 0.3, 1 }, &cp.sss_color);
    // Legacy lanes keep packing alongside the new ones.
    try std.testing.expectEqualSlices(f32, &.{ 0, 0.03, 0, 0 }, &cp.clearcoat_factors);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0.5, 0, 0 }, &cp.sheen_factors);
}

test "anisotropyAxes mirrors the shader formula" {
    // intensity <= 0: isotropic pair (shader early-returns to legacy GGX).
    try std.testing.expectEqual([2]f32{ 0.25, 0.25 }, anisotropyAxes(0.5, 0.0));
    try std.testing.expectEqual([2]f32{ 0.25, 0.25 }, anisotropyAxes(0.5, -1.0));
    // intensity 1, roughness 1: aspect = sqrt(0.1).
    const full = anisotropyAxes(1.0, 1.0);
    const aspect: f32 = @sqrt(0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0) / aspect, full[0], 1e-6);
    try std.testing.expectApproxEqAbs(aspect, full[1], 1e-6);
    // Stretch grows monotonically; ax >= ay always; clamp floors hold.
    const mid = anisotropyAxes(0.5, 0.5);
    try std.testing.expect(mid[0] >= mid[1]);
    try std.testing.expect(mid[0] * mid[1] > 0.0);
    const tiny = anisotropyAxes(0.001, 1.0);
    try std.testing.expect(tiny[0] >= 0.001 and tiny[1] >= 0.001);
    // Over-range intensity clamps to the full-stretch pair.
    const over = anisotropyAxes(0.5, 5.0);
    const exact = anisotropyAxes(0.5, 1.0);
    try std.testing.expectEqual(exact, over);
}

test "wrapNdotL keeps the legacy value when off" {
    try std.testing.expectEqual(@as(f32, 0.3), wrapNdotL(0.3, 0.0));
    try std.testing.expectEqual(@as(f32, -0.2), wrapNdotL(-0.2, -1.0));
    // strength 1: (ndotl + 0.5) / 1.5.
    try std.testing.expectApproxEqAbs(@as(f32, (0.25 + 0.5) / 1.5), wrapNdotL(0.25, 1.0), 1e-6);
    // Grazing-back fragments wrap toward light instead of staying black.
    try std.testing.expect(wrapNdotL(-0.25, 1.0) > 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25 / 1.5), wrapNdotL(-0.25, 1.0), 1e-6);
    // Fully lit stays fully lit.
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), wrapNdotL(1.0, 0.8), 1e-6);
}

test "MaterialDrawRecord stages coat texture views with white fallback" {
    var pbr_mat = PBRMaterial.init("test_pbr");
    const def_pbr = PBRMaterial.init("def");
    const dummy_tex = Texture{ .image = .{}, .view = .{ .id = 42 }, .sampler = .{ .id = 43 }, .width = 1, .height = 1 };
    const dummy_cube = CubeTexture{ .image = .{}, .view = .{ .id = 44 }, .sampler = .{ .id = 45 }, .size = 1 };

    // Null slots stage the white fallback (identity sampling).
    const rec = buildDrawRecord(.{ .pbr = &pbr_mat }, &def_pbr, &dummy_tex, &dummy_tex, &dummy_cube, &dummy_tex, null, 1.0);
    try std.testing.expectEqual(@as(u32, 42), rec.clearcoat_view.id);
    try std.testing.expectEqual(@as(u32, 42), rec.sheen_view.id);

    // Bound slots stage verbatim; later live mutation leaves the record
    // frozen (staging discipline: the draw never sees the live write).
    pbr_mat.clearcoat.mask_texture = .{ .image = .{}, .view = .{ .id = 51 }, .sampler = .{ .id = 52 }, .width = 4, .height = 4 };
    pbr_mat.sheen.color_texture = .{ .image = .{}, .view = .{ .id = 61 }, .sampler = .{ .id = 62 }, .width = 4, .height = 4 };
    const rec2 = buildDrawRecord(.{ .pbr = &pbr_mat }, &def_pbr, &dummy_tex, &dummy_tex, &dummy_cube, &dummy_tex, null, 1.0);
    try std.testing.expectEqual(@as(u32, 51), rec2.clearcoat_view.id);
    try std.testing.expectEqual(@as(u32, 61), rec2.sheen_view.id);
    pbr_mat.clearcoat.mask_texture.?.view.id = 99;
    pbr_mat.sheen.color_texture.?.view.id = 99;
    try std.testing.expectEqual(@as(u32, 51), rec2.clearcoat_view.id);
    try std.testing.expectEqual(@as(u32, 61), rec2.sheen_view.id);
}

test "PBRMaterial slot defaults reproduce the glTF conventions" {
    const mat = PBRMaterial.init("m");
    // Channels: glTF fixed conventions (AO=R, roughness=G, metallic=B).
    try std.testing.expectEqual(Channel.r, mat.occlusion_channel);
    try std.testing.expectEqual(Channel.g, mat.roughness_channel);
    try std.testing.expectEqual(Channel.b, mat.metallic_channel);
    try std.testing.expectEqual(@as(f32, 0), mat.occlusion_channel.selector());
    try std.testing.expectEqual(@as(f32, 1), mat.roughness_channel.selector());
    try std.testing.expectEqual(@as(f32, 2), mat.metallic_channel.selector());
    try std.testing.expectEqual(@as(f32, 3), Channel.a.selector());
    // Transforms: identity, so existing materials sample unchanged.
    try std.testing.expect(mat.albedo_uv_transform.isIdentity());
    try std.testing.expect(mat.normal_uv_transform.isIdentity());
    try std.testing.expect(mat.metallic_roughness_uv_transform.isIdentity());
    try std.testing.expect(mat.emissive_uv_transform.isIdentity());
    try std.testing.expect(mat.occlusion_uv_transform.isIdentity());
}

test "MaterialDrawRecord routes specular anti-aliasing into channel_selectors.w" {
    var pbr_mat = PBRMaterial.init("aa_pbr");
    const def_pbr = PBRMaterial.init("def");
    const dummy_tex = Texture{ .image = .{}, .view = .{ .id = 42 }, .sampler = .{ .id = 43 }, .width = 1, .height = 1 };
    const dummy_cube = CubeTexture{ .image = .{}, .view = .{ .id = 44 }, .sampler = .{ .id = 45 }, .size = 1 };

    // Hand-built PBRMaterial default = Babylon's own default (SPECULARAA off):
    // the w lane must stay 0 so the shader takes the legacy roughness path.
    try std.testing.expect(!pbr_mat.specular_anti_aliasing);
    const rec_off = buildDrawRecord(.{ .pbr = &pbr_mat }, &def_pbr, &dummy_tex, &dummy_tex, &dummy_cube, &dummy_tex, null, 1.0);
    try std.testing.expectEqual(@as(f32, 0.0), rec_off.channel_selectors[3]);
    // The three lane selectors are untouched by the flag (same uniform).
    try std.testing.expectEqual(@as(f32, 0.0), rec_off.channel_selectors[0]);
    try std.testing.expectEqual(@as(f32, 1.0), rec_off.channel_selectors[1]);
    try std.testing.expectEqual(@as(f32, 2.0), rec_off.channel_selectors[2]);

    pbr_mat.specular_anti_aliasing = true;
    pbr_mat.occlusion_channel = .a;
    pbr_mat.roughness_channel = .r;
    pbr_mat.metallic_channel = .a;
    const rec_on = buildDrawRecord(.{ .pbr = &pbr_mat }, &def_pbr, &dummy_tex, &dummy_tex, &dummy_cube, &dummy_tex, null, 1.0);
    try std.testing.expectEqual(@as(f32, 1.0), rec_on.channel_selectors[3]);
    try std.testing.expectEqual(@as(f32, 3.0), rec_on.channel_selectors[0]);
    try std.testing.expectEqual(@as(f32, 0.0), rec_on.channel_selectors[1]);
    try std.testing.expectEqual(@as(f32, 3.0), rec_on.channel_selectors[2]);
}

test "MaterialDrawRecord routes StandardMaterial via equivalent PBR matte" {
    const def_pbr = PBRMaterial.init("def");
    const dummy_tex = Texture{ .image = .{}, .view = .{ .id = 42 }, .sampler = .{ .id = 43 }, .width = 1, .height = 1 };
    const dummy_cube = CubeTexture{ .image = .{}, .view = .{ .id = 44 }, .sampler = .{ .id = 45 }, .size = 1 };

    // StandardMaterial maps specularPower to equivalent PBR roughness via
    // PBRMaterial.roughnessFromSpecularPower, with metallic 0.0 (dielectric matte).
    var std_mat = StandardMaterial.init("spec");
    const rec_def = buildDrawRecord(.{ .standard = &std_mat }, &def_pbr, &dummy_tex, &dummy_tex, &dummy_cube, &dummy_tex, null, 1.0);
    const r_def = PBRMaterial.roughnessFromSpecularPower(64.0);
    try std.testing.expectEqual([4]f32{ 0.0, r_def, 1.0, 1.0 }, rec_def.pbr_factors);

    std_mat.specular_power = 8.0;
    const rec_set = buildDrawRecord(.{ .standard = &std_mat }, &def_pbr, &dummy_tex, &dummy_tex, &dummy_cube, &dummy_tex, null, 1.0);
    const r_set = PBRMaterial.roughnessFromSpecularPower(8.0);
    try std.testing.expectEqual([4]f32{ 0.0, r_set, 1.0, 1.0 }, rec_set.pbr_factors);

    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, rec_def.emissive_color);
    std_mat.emissive_color = Color3.new(0.2, 0.1, 0.05);
    const rec_emis = buildDrawRecord(.{ .standard = &std_mat }, &def_pbr, &dummy_tex, &dummy_tex, &dummy_cube, &dummy_tex, null, 1.0);
    try std.testing.expectEqual([4]f32{ 0.2, 0.1, 0.05, 0.0 }, rec_emis.emissive_color);
}

test "MaterialDrawRecord routes two-sided lighting into the emissive w lane" {
    // Babylon defines TWOSIDEDLIGHTING only for `backFaceCulling == false &&
    // twoSidedLighting == true` and then flips the shading normal on back
    // faces. The flag rides the emissive lane's w (unused by the vec3 emissive
    // upload), and it must reach BOTH families' shaders through the same lane.
    const def_pbr = PBRMaterial.init("def");
    const dummy_tex = Texture{ .image = .{}, .view = .{ .id = 42 }, .sampler = .{ .id = 43 }, .width = 1, .height = 1 };
    const dummy_cube = CubeTexture{ .image = .{}, .view = .{ .id = 44 }, .sampler = .{ .id = 45 }, .size = 1 };

    var std_mat = StandardMaterial.init("ts_std");
    var pbr_mat = PBRMaterial.init("ts_pbr");
    try std.testing.expect(!std_mat.two_sided_lighting);
    try std.testing.expect(!pbr_mat.two_sided_lighting);

    const std_off = buildDrawRecord(.{ .standard = &std_mat }, &def_pbr, &dummy_tex, &dummy_tex, &dummy_cube, &dummy_tex, null, 1.0);
    try std.testing.expectEqual(@as(f32, 0.0), std_off.emissive_color[3]);
    const pbr_off = buildDrawRecord(.{ .pbr = &pbr_mat }, &def_pbr, &dummy_tex, &dummy_tex, &dummy_cube, &dummy_tex, null, 1.0);
    try std.testing.expectEqual(@as(f32, 0.0), pbr_off.emissive_color[3]);

    std_mat.two_sided_lighting = true;
    pbr_mat.two_sided_lighting = true;
    const std_on = buildDrawRecord(.{ .standard = &std_mat }, &def_pbr, &dummy_tex, &dummy_tex, &dummy_cube, &dummy_tex, null, 1.0);
    try std.testing.expectEqual(@as(f32, 1.0), std_on.emissive_color[3]);
    const pbr_on = buildDrawRecord(.{ .pbr = &pbr_mat }, &def_pbr, &dummy_tex, &dummy_tex, &dummy_cube, &dummy_tex, null, 1.0);
    try std.testing.expectEqual(@as(f32, 1.0), pbr_on.emissive_color[3]);
    // The emissive colour itself is untouched by the flag.
    try std.testing.expectEqual(@as(f32, 0.0), std_on.emissive_color[0]);
    try std.testing.expectEqual(@as(f32, 0.0), pbr_on.emissive_color[0]);
}

test "MaterialDrawRecord builds correctly from PBRMaterial" {
    var pbr_mat = PBRMaterial.init("test_pbr");
    pbr_mat.metallic = 0.8;
    pbr_mat.roughness = 0.2;
    pbr_mat.alpha_cutoff = 0.4;
    pbr_mat.alpha_mode = .cutout;

    const def_pbr = PBRMaterial.init("def");
    const dummy_tex = Texture{ .image = .{}, .view = .{ .id = 42 }, .sampler = .{ .id = 43 }, .width = 1, .height = 1 };
    const dummy_cube = CubeTexture{ .image = .{}, .view = .{ .id = 44 }, .sampler = .{ .id = 45 }, .size = 1 };

    const rec = buildDrawRecord(
        .{ .pbr = &pbr_mat },
        &def_pbr,
        &dummy_tex,
        &dummy_tex,
        &dummy_cube,
        &dummy_tex,
        null,
        1.5,
    );

    try std.testing.expectEqual(@as(u32, 42), rec.albedo_view.id);
    try std.testing.expectEqual(@as(f32, 0.8), rec.pbr_factors[0]);
    try std.testing.expectEqual(@as(f32, 0.2), rec.pbr_factors[1]);
    try std.testing.expectEqual(@as(f32, 0.4), rec.alpha_cutoff);
}

test "Material unlit mode properly routes to DrawRecord" {
    var pbr_mat = PBRMaterial.init("unlit_pbr");
    pbr_mat.unlit = true;

    var std_mat = StandardMaterial.init("unlit_std");
    std_mat.unlit = true;

    const def_pbr = PBRMaterial.init("def");
    const dummy_tex = Texture{ .image = .{}, .view = .{ .id = 42 }, .sampler = .{ .id = 43 }, .width = 1, .height = 1 };
    const dummy_cube = CubeTexture{ .image = .{}, .view = .{ .id = 44 }, .sampler = .{ .id = 45 }, .size = 1 };

    var mat_pbr = Material{ .pbr = &pbr_mat };
    try std.testing.expect(mat_pbr.isUnlit());

    var mat_std = Material{ .standard = &std_mat };
    try std.testing.expect(mat_std.isUnlit());

    const rec_pbr = buildDrawRecord(mat_pbr, &def_pbr, &dummy_tex, &dummy_tex, &dummy_cube, &dummy_tex, null, 1.0);
    try std.testing.expectEqual(@as(f32, 1.0), rec_pbr.uv_offsets[0][2]);

    const rec_std = buildDrawRecord(mat_std, &def_pbr, &dummy_tex, &dummy_tex, &dummy_cube, &dummy_tex, null, 1.0);
    try std.testing.expectEqual(@as(f32, 1.0), rec_std.uv_offsets[0][2]);
}

test "MaterialDrawRecord falls back to default PBR material when mat is null" {
    var def_pbr = PBRMaterial.init("def");
    def_pbr.metallic = 0.5;
    def_pbr.roughness = 0.75;
    def_pbr.albedo_color = Color3.new(0.3, 0.4, 0.5);

    const dummy_tex = Texture{ .image = .{}, .view = .{ .id = 42 }, .sampler = .{ .id = 43 }, .width = 1, .height = 1 };
    const dummy_cube = CubeTexture{ .image = .{}, .view = .{ .id = 44 }, .sampler = .{ .id = 45 }, .size = 1 };

    const rec = buildDrawRecord(null, &def_pbr, &dummy_tex, &dummy_tex, &dummy_cube, &dummy_tex, null, 1.0);
    try std.testing.expectEqual(@as(u32, 42), rec.albedo_view.id);
    try std.testing.expectEqual(@as(f32, 0.5), rec.pbr_factors[0]);
    try std.testing.expectEqual(@as(f32, 0.75), rec.pbr_factors[1]);
    try std.testing.expectEqual(@as(f32, 0.3), rec.base_color[0]);
    try std.testing.expectEqual(@as(f32, 0.4), rec.base_color[1]);
    try std.testing.expectEqual(@as(f32, 0.5), rec.base_color[2]);
}

test "P4: buildShaderSnapshot copies hook material CPU state" {
    var sm = ShaderMaterial.init("hook");
    sm.entry_index = 3;
    sm.tint_color = Color3.new(0.1, 0.2, 0.3);
    sm.alpha = 0.5;
    sm.double_sided = true;
    sm.texture = Texture{ .image = .{}, .view = .{ .id = 42 }, .sampler = .{ .id = 43 }, .width = 4, .height = 4 };
    sm.uniforms[0] = .{ 1, 2, 3, 4 };

    const dummy_tex = Texture{ .image = .{}, .view = .{ .id = 7 }, .sampler = .{ .id = 8 }, .width = 1, .height = 1 };
    const snap = buildShaderSnapshot(.{ .shader_material = &sm }, &dummy_tex) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u32, 3), snap.entry_index);
    try std.testing.expectEqual([4]f32{ 0.1, 0.2, 0.3, 0.5 }, snap.tint);
    try std.testing.expectEqual(@as(u32, 42), snap.tex_view.id);
    try std.testing.expectEqual([4]f32{ 1, 2, 3, 4 }, snap.uniforms[0]);
    try std.testing.expect(snap.double_sided);

    // Мутация живого материала после снимка: снимок неизменен.
    sm.tint_color = Color3.new(9, 9, 9);
    sm.alpha = 0.0;
    sm.double_sided = false;
    sm.texture = null;
    sm.uniforms[0] = .{ 9, 9, 9, 9 };
    sm.entry_index = 9;
    try std.testing.expectEqual([4]f32{ 0.1, 0.2, 0.3, 0.5 }, snap.tint);
    try std.testing.expectEqual(@as(u32, 42), snap.tex_view.id);
    try std.testing.expectEqual([4]f32{ 1, 2, 3, 4 }, snap.uniforms[0]);
    try std.testing.expectEqual(@as(u32, 3), snap.entry_index);
    try std.testing.expect(snap.double_sided);

    // Без текстуры — дефолт из prepare-фазы, а не живой указатель.
    const fallback = buildShaderSnapshot(.{ .shader_material = &sm }, &dummy_tex) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u32, 7), fallback.tex_view.id);

    // Не-hook материалы снимка не дают.
    var std_mat = StandardMaterial.init("s");
    try std.testing.expect(buildShaderSnapshot(.{ .standard = &std_mat }, &dummy_tex) == null);
    try std.testing.expect(buildShaderSnapshot(null, &dummy_tex) == null);
}

test "roughnessFromSpecularPower lobe-match goldens" {
    // alpha = sqrt(2/(n+2)), floored at 0.05.
    try std.testing.expectApproxEqAbs(@as(f32, 0.1741), pbr.roughnessFromSpecularPower(64.0), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.4472), pbr.roughnessFromSpecularPower(8.0), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8165), pbr.roughnessFromSpecularPower(1.0), 0.001);
    // Huge exponents clamp to the practical floor, never zero.
    try std.testing.expectEqual(@as(f32, 0.05), pbr.roughnessFromSpecularPower(1_000_000.0));
    // Degenerate exponents clamp to power 1 (roughness stays in range).
    try std.testing.expectApproxEqAbs(@as(f32, 0.8165), pbr.roughnessFromSpecularPower(-4.0), 0.001);
    try std.testing.expect(pbr.roughnessFromSpecularPower(0.0) <= 1.0);
}
