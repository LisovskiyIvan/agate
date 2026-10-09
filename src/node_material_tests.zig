const std = @import("std");
const node_material = @import("node_material.zig");
const Graph = node_material.Graph;
const Compiled = node_material.Compiled;
const merge = @import("shader_material/merge.zig");
const shader_material = @import("shader_material.zig");

test "minimal graph compiles to the golden snippet" {
    const alloc = std.testing.allocator;
    var g = Graph.init(alloc);
    defer g.deinit();
    const red = try g.addNode(.const_color, .{ .color = .{ 0.8, 0.1, 0.1 } });
    const out = try g.addNode(.output, .{});
    g.connect(out, 0, red);

    var c = try g.compile(alloc, "flat_red");
    defer c.deinit(alloc);

    const expected =
        \\// agate node material: flat_red (generated — do not edit)
        \\// base: standard
        \\// @hook(albedo)
        \\vec3 _n0 = vec3(0.8, 0.1, 0.1);
        \\base.rgb = _n0;
        \\
    ;
    try std.testing.expectEqualStrings(expected, c.snippet);
    try std.testing.expectEqual(@as(usize, 0), c.params.len);
}

test "compile is deterministic across runs" {
    const alloc = std.testing.allocator;
    var g = Graph.init(alloc);
    defer g.deinit();
    const uv = try g.addNode(.uv, .{});
    const tex = try g.addNode(.texture_sample, .{});
    g.connect(tex, 0, uv);
    const tint = try g.addNode(.const_color, .{ .color = .{ 1, 0.5, 0.25 }, .param = "u_tint" });
    const mul = try g.addNode(.multiply, .{});
    g.connect(mul, 0, tex);
    g.connect(mul, 1, tint);
    const out = try g.addNode(.output, .{});
    g.connect(out, 0, mul);

    var a = try g.compile(alloc, "textured");
    defer a.deinit(alloc);
    var b = try g.compile(alloc, "textured");
    defer b.deinit(alloc);
    try std.testing.expectEqualStrings(a.snippet, b.snippet);
    // Param packing: vec4 4-aligned at word 0 with rgb + 1.0 pad.
    try std.testing.expectEqual(@as(usize, 1), a.params.len);
    try std.testing.expectEqualStrings("u_tint", a.params[0].name);
    try std.testing.expectEqual(@as(u8, 0), a.params[0].offset);
    try std.testing.expectEqual(@as(u8, 4), a.params[0].comps);
    try std.testing.expectEqualSlices(f32, &.{ 1, 0.5, 0.25, 1.0 }, &a.params[0].default);
    // Golden body spot-checks.
    try std.testing.expect(std.mem.indexOf(u8, a.snippet, "// @param u_tint vec4 = 1 0.5 0.25 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, a.snippet, "vec2 _n0 = v_uv;") != null);
    try std.testing.expect(std.mem.indexOf(u8, a.snippet, "texture(sampler2D(diffuse_tex, smp), _n0).rgb;") != null);
    try std.testing.expect(std.mem.indexOf(u8, a.snippet, "vec3 _n2 = u_tint.rgb;") != null);
    try std.testing.expect(std.mem.indexOf(u8, a.snippet, "vec3 _n3 = (_n1 * _n2);") != null);
    try std.testing.expect(std.mem.indexOf(u8, a.snippet, "base.rgb = _n3;") != null);
    // Unwired alpha leaves base.a alone.
    try std.testing.expect(std.mem.indexOf(u8, a.snippet, "base.a") == null);
}

test "time/mix/alpha graph wires uniforms and the alpha port" {
    const alloc = std.testing.allocator;
    var g = Graph.init(alloc);
    defer g.deinit();
    const t = try g.addNode(.time, .{});
    const speed = try g.addNode(.const_float, .{ .float = 2.0, .param = "u_speed" });
    const rate = try g.addNode(.multiply, .{});
    g.connect(rate, 0, t);
    g.connect(rate, 1, speed);
    const wave = try g.addNode(.sin, .{});
    g.connect(wave, 0, rate);
    const lo = try g.addNode(.const_color, .{});
    const hi = try g.addNode(.const_color, .{ .color = .{ 0, 0, 1 } });
    const m = try g.addNode(.mix, .{});
    g.connect(m, 0, lo);
    g.connect(m, 1, hi);
    // Reuse the wave as the mix factor AND the output alpha (fan-out).
    g.connect(m, 2, wave);
    const out = try g.addNode(.output, .{});
    g.connect(out, 0, m);
    g.connect(out, 1, wave);

    var c = try g.compile(alloc, "pulse");
    defer c.deinit(alloc);
    // Auto u_time first, then u_speed in node-id order.
    try std.testing.expectEqual(@as(usize, 2), c.params.len);
    try std.testing.expectEqualStrings("u_time", c.params[0].name);
    try std.testing.expectEqualStrings("u_speed", c.params[1].name);
    try std.testing.expectEqual(@as(u8, 1), c.params[1].offset);
    try std.testing.expect(std.mem.indexOf(u8, c.snippet, "float _n0 = u_time;") != null);
    try std.testing.expect(std.mem.indexOf(u8, c.snippet, "float _n3 = sin(_n2);") != null);
    try std.testing.expect(std.mem.indexOf(u8, c.snippet, "base.a *= _n3;") != null);
}

test "compiled params drive shader_material packing (runtime updates)" {
    const alloc = std.testing.allocator;
    var g = Graph.init(alloc);
    defer g.deinit();
    const speed = try g.addNode(.const_float, .{ .float = 4.0, .param = "u_speed" });
    const white = try g.addNode(.const_color, .{});
    const mx = try g.addNode(.mix, .{});
    g.connect(mx, 0, white);
    g.connect(mx, 1, white);
    g.connect(mx, 2, speed);
    const out = try g.addNode(.output, .{});
    g.connect(out, 0, mx);
    var c = try g.compile(alloc, "params_only");
    defer c.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 1), c.params.len);

    // Compiled params are merge-level declarations; convert to the runtime
    // table view (identical layout, asserted below) for uniform packing.
    comptime {
        const M = @import("shader_material/merge.zig").Param;
        const S = shader_material.Param;
        std.debug.assert(@sizeOf(M) == @sizeOf(S));
        std.debug.assert(@offsetOf(M, "name") == @offsetOf(S, "name"));
        std.debug.assert(@offsetOf(M, "offset") == @offsetOf(S, "offset"));
        std.debug.assert(@offsetOf(M, "comps") == @offsetOf(S, "comps"));
        std.debug.assert(@offsetOf(M, "default") == @offsetOf(S, "default"));
    }
    const sm_params = try alloc.alloc(shader_material.Param, c.params.len);
    defer alloc.free(sm_params);
    for (c.params, 0..) |p, i| {
        sm_params[i] = .{ .name = p.name, .offset = p.offset, .comps = p.comps, .default = p.default };
    }

    var storage = shader_material.defaultUniformStorage(sm_params);
    try std.testing.expectEqual(@as(f32, 4.0), storage[0][0]);
    // Runtime update without any rebuild — the v1 "no engine rebuild" path.
    try shader_material.setUniform(&storage, sm_params, "u_speed", .{ .scalar = 9.0 });
    try std.testing.expectEqual(@as(f32, 9.0), storage[0][0]);
    try std.testing.expectError(error.UnknownParam, shader_material.setUniform(&storage, sm_params, "u_nope", .{ .scalar = 1 }));
    try std.testing.expectEqual(@as(usize, 128), shader_material.uniformBytes(&storage).len);
}

test "generated snippet merges through the hook merger" {
    const alloc = std.testing.allocator;
    var g = Graph.init(alloc);
    defer g.deinit();
    const k = try g.addNode(.const_float, .{ .float = 0.5, .param = "u_k" });
    const c0 = try g.addNode(.const_color, .{});
    // Scale via mix against black (float * vec3 would be a type error).
    const black = try g.addNode(.const_color, .{ .color = .{ 0, 0, 0 } });
    const mx = try g.addNode(.mix, .{});
    g.connect(mx, 0, black);
    g.connect(mx, 1, c0);
    g.connect(mx, 2, k);
    const out = try g.addNode(.output, .{});
    g.connect(out, 0, mx);
    var c = try g.compile(alloc, "merge_probe");
    defer c.deinit(alloc);

    const tmpl =
        \\// @hook(decls)
        \\// @endhook
        \\void main() {
        \\    vec3 base = vec3(1.0);
        \\    // @hook(albedo)
        \\    // @endhook
        \\}
        \\
    ;
    const res = try merge.merge(alloc, .{ .template = tmpl, .snippet = c.snippet, .base = "standard", .material_name = "merge_probe" });
    defer alloc.free(res.glsl);
    defer alloc.free(res.params);
    try std.testing.expectEqual(@as(usize, 1), res.params.len);
    try std.testing.expectEqualStrings("u_k", res.params[0].name);
    try std.testing.expect(std.mem.indexOf(u8, res.glsl, "base.rgb = ") != null);
    try std.testing.expect(std.mem.indexOf(u8, res.glsl, "#define u_k (sm_user_0.x)") != null);
}

test "validation rejects cycles, mismatches and dangling inputs" {
    const alloc = std.testing.allocator;
    // Cycle: a feeds b feeds a (via output root).
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        const a = try g.addNode(.add, .{});
        const b = try g.addNode(.add, .{});
        const f = try g.addNode(.const_float, .{});
        g.connect(a, 0, b);
        g.connect(a, 1, f);
        g.connect(b, 0, a);
        g.connect(b, 1, f);
        const out = try g.addNode(.output, .{});
        const w = try g.addNode(.const_color, .{});
        g.connect(out, 0, w);
        // Output is fine but a/b are unreachable — unreachable cycles are
        // dropped, so this compiles. Wire the cycle to the output instead.
        _ = try g.addNode(.output, .{});
        try std.testing.expectError(error.MultipleOutputs, g.compile(alloc, "x"));
    }
    // True cycle through the output.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        const a = try g.addNode(.add, .{});
        const f = try g.addNode(.const_float, .{});
        g.connect(a, 0, a); // self-loop
        g.connect(a, 1, f);
        const w = try g.addNode(.const_color, .{});
        const mx = try g.addNode(.mix, .{});
        g.connect(mx, 0, w);
        g.connect(mx, 1, w);
        g.connect(mx, 2, a);
        const out = try g.addNode(.output, .{});
        g.connect(out, 0, mx);
        try std.testing.expectError(error.Cycle, g.compile(alloc, "x"));
    }
    // Type mismatch: float into a vec3 color port.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        const f = try g.addNode(.const_float, .{});
        const out = try g.addNode(.output, .{});
        g.connect(out, 0, f);
        try std.testing.expectError(error.TypeMismatch, g.compile(alloc, "x"));
    }
    // Dangling: add with only one input wired.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        const a = try g.addNode(.add, .{});
        const f = try g.addNode(.const_float, .{});
        g.connect(a, 0, f);
        const w = try g.addNode(.const_color, .{});
        const mx = try g.addNode(.mix, .{});
        const k = try g.addNode(.const_float, .{});
        g.connect(mx, 0, w);
        g.connect(mx, 1, w);
        g.connect(mx, 2, k);
        const out = try g.addNode(.output, .{});
        g.connect(out, 0, mx);
        // `a` is unreachable: compiles fine (dead code dropped).
        var c = try g.compile(alloc, "dead_ok");
        defer c.deinit(alloc);
        try std.testing.expect(std.mem.indexOf(u8, c.snippet, "_n0") == null);
    }
    // Dangling on the REACHABLE path is an error.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        const a = try g.addNode(.add, .{});
        const f = try g.addNode(.const_float, .{});
        g.connect(a, 0, f);
        const w = try g.addNode(.const_color, .{});
        const mx = try g.addNode(.mix, .{});
        g.connect(mx, 0, w);
        g.connect(mx, 1, w);
        g.connect(mx, 2, a);
        const out = try g.addNode(.output, .{});
        g.connect(out, 0, mx);
        try std.testing.expectError(error.DanglingInput, g.compile(alloc, "x"));
    }
    // Missing output / unknown hookups.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        _ = try g.addNode(.const_float, .{});
        try std.testing.expectError(error.MissingOutput, g.compile(alloc, "x"));
    }
}

test "param names are guarded (reserved, duplicate, malformed, slots)" {
    const alloc = std.testing.allocator;
    // Reserved GLSL-ish name from merge.reserved_param_names, left
    // unreachable: dead code (and its names) never reach the snippet.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        _ = try g.addNode(.const_float, .{ .param = "roughness" });
        const w = try g.addNode(.const_color, .{});
        const out = try g.addNode(.output, .{});
        g.connect(out, 0, w);
        _ = try g.addNode(.const_float, .{});
        _ = try g.addNode(.mix, .{});
        var c = try g.compile(alloc, "ok");
        defer c.deinit(alloc);
    }
    // Reserved name ON the reachable path is a hard error.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        const k = try g.addNode(.const_float, .{ .param = "albedo" });
        const w = try g.addNode(.const_color, .{});
        const mx = try g.addNode(.mix, .{});
        g.connect(mx, 0, w);
        g.connect(mx, 1, w);
        g.connect(mx, 2, k);
        const out = try g.addNode(.output, .{});
        g.connect(out, 0, mx);
        try std.testing.expectError(error.BadParamName, g.compile(alloc, "x"));
    }
    // Duplicate user params.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        const a = try g.addNode(.const_float, .{ .param = "u_dup" });
        _ = try g.addNode(.const_float, .{ .param = "u_dup" });
        const w = try g.addNode(.const_color, .{});
        const mx = try g.addNode(.mix, .{});
        g.connect(mx, 0, w);
        g.connect(mx, 1, w);
        g.connect(mx, 2, a);
        const out = try g.addNode(.output, .{});
        g.connect(out, 0, mx);
        try std.testing.expectError(error.DuplicateParam, g.compile(alloc, "x"));
    }
    // Malformed identifiers.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        try std.testing.expectError(error.BadParamName, g.addNode(.const_float, .{ .param = "this name has spaces and is way too long for the buffer limit ok" }));
        const w = try g.addNode(.const_color, .{});
        const out = try g.addNode(.output, .{});
        g.connect(out, 0, w);
        const bad = try g.addNode(.const_float, .{ .param = "9lives" });
        // Reachable bad names fail: wire it in as the mix factor.
        const mx = try g.addNode(.mix, .{});
        g.connect(mx, 0, w);
        g.connect(mx, 1, w);
        g.connect(mx, 2, bad);
        g.connect(out, 0, mx);
        try std.testing.expectError(error.BadParamName, g.compile(alloc, "x"));
    }
    // Non-zero texture slot.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        const uv = try g.addNode(.uv, .{});
        const tx = try g.addNode(.texture_sample, .{ .tex_slot = 1 });
        g.connect(tx, 0, uv);
        const out = try g.addNode(.output, .{});
        g.connect(out, 0, tx);
        try std.testing.expectError(error.UnsupportedTextureSlot, g.compile(alloc, "x"));
    }
    // texture_sample fed a float is a mismatch.
    {
        var g = Graph.init(alloc);
        defer g.deinit();
        const f = try g.addNode(.const_float, .{});
        const tx = try g.addNode(.texture_sample, .{});
        g.connect(tx, 0, f);
        const out = try g.addNode(.output, .{});
        g.connect(out, 0, tx);
        try std.testing.expectError(error.TypeMismatch, g.compile(alloc, "x"));
    }
}

test "NodeMaterial v2: extended math nodes and JSON serialization round-trip" {
    const alloc = std.testing.allocator;
    var g = Graph.init(alloc);
    defer g.deinit();

    const speed = try g.addNode(.const_float, .{ .float = 0.5, .param = "pan_speed" });
    const base_col = try g.addNode(.const_color, .{ .color = .{ 0.2, 0.4, 0.9 }, .param = "rim_col" });
    const pwr = try g.addNode(.const_float, .{ .float = 3.0 });
    const fr = try g.addNode(.fresnel, .{});
    g.connect(fr, 0, pwr);

    const cs = try g.addNode(.cos, .{});
    g.connect(cs, 0, speed);

    const pw = try g.addNode(.pow, .{});
    g.connect(pw, 0, cs);
    g.connect(pw, 1, fr);

    const clmp = try g.addNode(.clamp, .{});
    g.connect(clmp, 0, pw);
    g.connect(clmp, 1, speed);
    g.connect(clmp, 2, pwr);

    const mx = try g.addNode(.mix, .{});
    g.connect(mx, 0, base_col);
    g.connect(mx, 1, base_col);
    g.connect(mx, 2, clmp);

    const out = try g.addNode(.output, .{});
    g.connect(out, 0, mx);

    var compiled = try g.compile(alloc, "v2_test");
    defer compiled.deinit(alloc);

    try std.testing.expect(std.mem.indexOf(u8, compiled.snippet, "cos") != null);
    try std.testing.expect(std.mem.indexOf(u8, compiled.snippet, "pow") != null);
    try std.testing.expect(std.mem.indexOf(u8, compiled.snippet, "clamp") != null);

    // JSON serialization
    const json = try g.serializeJson(alloc);
    defer alloc.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"kind\": \"fresnel\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"kind\": \"cos\"") != null);

    // JSON deserialization
    var g2 = try Graph.deserializeJson(alloc, json);
    defer g2.deinit();

    try std.testing.expectEqual(g.nodeCount(), g2.nodeCount());

    var compiled2 = try g2.compile(alloc, "v2_test");
    defer compiled2.deinit(alloc);
    try std.testing.expectEqualStrings(compiled.snippet, compiled2.snippet);
}
