const std = @import("std");
const merge_mod = @import("merge.zig");
const merge = merge_mod.merge;
const Error = merge_mod.Error;

const test_template =
    \\void main() {
    \\    vec3 base = vec3(1.0);
    \\    // @hook(albedo)
    \\    // @endhook
    \\    // @hook(post_lighting)
    \\    // @endhook
    \\    frag = vec4(base, 1.0);
    \\}
    \\
;

test "merge without snippet is byte-identical to the template" {
    const res = try merge(std.testing.allocator, .{ .template = test_template });
    defer std.testing.allocator.free(res.glsl);
    defer std.testing.allocator.free(res.params);
    try std.testing.expectEqualSlices(u8, test_template, res.glsl);
    try std.testing.expectEqual(@as(usize, 0), res.params.len);
}

test "merge injects snippet body at the hook marker and leaves other hooks" {
    const snippet =
        "// base: standard\n" ++
        "// @hook(albedo)\n" ++
        "base.rgb = vec3(0.0, 1.0, 0.0);\n";
    const res = try merge(std.testing.allocator, .{ .template = test_template, .snippet = snippet });
    defer std.testing.allocator.free(res.glsl);
    defer std.testing.allocator.free(res.params);

    const marker = "// @hook(albedo)\n";
    const marker_end = comptime std.mem.indexOf(u8, test_template, marker).? + marker.len;
    const expected = test_template[0..marker_end] ++ "base.rgb = vec3(0.0, 1.0, 0.0);\n" ++ test_template[marker_end..];
    try std.testing.expectEqualSlices(u8, expected, res.glsl);
    // The post_lighting hook stays empty (default path).
    try std.testing.expect(std.mem.indexOf(u8, res.glsl, "// @hook(post_lighting)") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.glsl, "// @endhook") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.glsl, "final_rgb") == null);
}

test "merge rejects unknown hooks and unterminated blocks" {
    const bad_snippet = "// @hook(nonexistent)\nfoo();\n";
    try std.testing.expectError(Error.UnknownHook, merge(std.testing.allocator, .{ .template = test_template, .snippet = bad_snippet }));

    const unterminated = "void f() {\n// @hook(albedo)\n}\n";
    try std.testing.expectError(Error.HookNotTerminated, merge(std.testing.allocator, .{ .template = unterminated }));
}

test "param decls assign f32 offsets and generate the UB block" {
    const snippet =
        "// base: standard\n" ++
        "// @param u_speed float = 2.5\n" ++
        "// @param u_tint vec4 = 1 0.5 0 1\n" ++
        "// @param u_amp float\n" ++
        "// @hook(albedo)\n" ++
        "base.rgb *= u_speed;\n";
    const tmpl = "// @hook(decls)\n// @endhook\n" ++ test_template;
    const res = try merge(std.testing.allocator, .{ .template = tmpl, .snippet = snippet });
    defer std.testing.allocator.free(res.glsl);
    defer std.testing.allocator.free(res.params);

    try std.testing.expectEqual(@as(usize, 3), res.params.len);
    // float -> word 0; vec4 -> words 4..7 (slot 1); float -> word 8 (slot 2).
    try std.testing.expectEqual(@as(u8, 0), res.params[0].offset);
    try std.testing.expectEqual(@as(u8, 1), res.params[0].comps);
    try std.testing.expectEqual(@as(u8, 4), res.params[1].offset);
    try std.testing.expectEqual(@as(u8, 4), res.params[1].comps);
    try std.testing.expectEqual(@as(u8, 8), res.params[2].offset);
    try std.testing.expectEqual(@as(f32, 2.5), res.params[0].default[0]);
    try std.testing.expectEqual(@as(f32, 0.5), res.params[1].default[1]);

    // The generated block + defines landed inside the decls hook.
    try std.testing.expect(std.mem.indexOf(u8, res.glsl, "layout(binding = 3) uniform sm_user_params {") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.glsl, "#define u_speed (sm_user_0.x)") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.glsl, "#define u_tint (sm_user_1)") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.glsl, "#define u_amp (sm_user_2.x)") != null);
    // Snippet body landed at the albedo hook.
    try std.testing.expect(std.mem.indexOf(u8, res.glsl, "base.rgb *= u_speed;") != null);
}

test "snippet base header mismatch fails the build" {
    const snippet = "// base: custom\n// @hook(albedo)\nbase.rgb = vec3(0.0);\n";
    try std.testing.expectError(Error.BaseMismatch, merge(std.testing.allocator, .{ .template = test_template, .base = "pbr", .snippet = snippet }));
}

test "snippet base standard alias accepted under pbr base" {
    const snippet = "// base: standard\n// @hook(albedo)\nbase.rgb = vec3(0.0);\n";
    const res = try merge(std.testing.allocator, .{ .template = test_template, .base = "pbr", .snippet = snippet });
    defer std.testing.allocator.free(res.glsl);
    defer std.testing.allocator.free(res.params);
    try std.testing.expect(std.mem.indexOf(u8, res.glsl, "base.rgb = vec3(0.0);") != null);
}

test "real engine templates merge byte-identical and accept an albedo hook" {
    const pbr_tmpl = @embedFile("../shaders/pbr.glsl");

    // No snippet: exact identity for PBR template.
    const r1 = try merge(std.testing.allocator, .{ .template = pbr_tmpl });
    defer std.testing.allocator.free(r1.glsl);
    defer std.testing.allocator.free(r1.params);
    try std.testing.expectEqualSlices(u8, pbr_tmpl, r1.glsl);

    // A snippet overriding every pbr.glsl hook merges cleanly.
    const snippet =
        "// @hook(vertex)\nmorphed_pos.z += 0.1;\n" ++
        "// @hook(albedo)\nbase.rgb = base.rgb.bgr;\n" ++
        "// @hook(post_lighting)\nfinal_rgb = pow(final_rgb, vec3(2.2));\n";
    const r2 = try merge(std.testing.allocator, .{ .template = pbr_tmpl, .snippet = snippet });
    defer std.testing.allocator.free(r2.glsl);
    defer std.testing.allocator.free(r2.params);
    try std.testing.expect(std.mem.indexOf(u8, r2.glsl, "morphed_pos.z += 0.1;") != null);
    try std.testing.expect(std.mem.indexOf(u8, r2.glsl, "base.rgb = base.rgb.bgr;") != null);
    try std.testing.expect(std.mem.indexOf(u8, r2.glsl, "final_rgb = pow(final_rgb, vec3(2.2));") != null);
    // Injection points kept their surrounding lines.
    try std.testing.expect(std.mem.indexOf(u8, r2.glsl, "applyMorphDeltas(morphed_pos, morphed_nrm, morphed_tan, gl_VertexIndex);") != null);
    try std.testing.expect(std.mem.indexOf(u8, r2.glsl, "frag_color = linearOutputColor(final_color, albedo_rgba.a);") != null);
}

test "reserved param names are rejected" {
    const tmpl = "// @hook(decls)\n// @endhook\n";
    const snippet = "// @param roughness float = 0.5\n";
    try std.testing.expectError(error.BadParamDecl, merge(std.testing.allocator, .{ .template = tmpl, .snippet = snippet }));
}
