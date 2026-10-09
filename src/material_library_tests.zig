const std = @import("std");
const math = @import("math");
const Color3 = math.Color3;
const material = @import("material.zig");
const ShaderMaterial = material.ShaderMaterial;
const Texture = @import("texture.zig").Texture;
const matlib = @import("material_library.zig");
const shader_material = @import("shader_material.zig");

const presets = matlib.presets;
const presetForName = matlib.presetForName;
const presetForKind = matlib.presetForKind;
const sky_shader_name = matlib.sky_shader_name;
const gradient_shader_name = matlib.gradient_shader_name;
const grid_shader_name = matlib.grid_shader_name;
const triplanar_shader_name = matlib.triplanar_shader_name;
const sky = matlib.sky;
const gradient = matlib.gradient;
const grid = matlib.grid;
const triPlanar = matlib.triPlanar;
const applySky = matlib.applySky;
const applyGradient = matlib.applyGradient;
const applyGrid = matlib.applyGrid;
const applyTriPlanar = matlib.applyTriPlanar;
const SkyOptions = matlib.SkyOptions;
const GradientOptions = matlib.GradientOptions;
const GridOptions = matlib.GridOptions;
const TriPlanarOptions = matlib.TriPlanarOptions;
const skyBlendT = matlib.skyBlendT;
const gradientT = matlib.gradientT;
const gridLineMask = matlib.gridLineMask;
const triplanarWeights = matlib.triplanarWeights;

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
