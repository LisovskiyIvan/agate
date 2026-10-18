const std = @import("std");
const inc = @import("include.zig");
const parseIncludeDirective = inc.parseIncludeDirective;
const expand = inc.expand;
const Error = inc.Error;
const FileSystem = inc.FileSystem;

const MemFs = struct {
    files: std.StringHashMapUnmanaged([]const u8) = .empty,

    fn fs(self: *MemFs) FileSystem {
        return .{ .ctx = self, .readFn = read };
    }
    fn read(ctx: *anyopaque, allocator: std.mem.Allocator, path: []const u8) FileSystem.ReadError![]u8 {
        const self: *MemFs = @ptrCast(@alignCast(ctx));
        const content = self.files.get(path) orelse return error.NotFound;
        return allocator.dupe(u8, content) catch error.OutOfMemory;
    }
};

test "directive parsing: exact form only" {
    try std.testing.expectEqualStrings("common/x.glsl", (try parseIncludeDirective("// @include \"common/x.glsl\"")).?);
    try std.testing.expectEqualStrings("common/x.glsl", (try parseIncludeDirective("   // @include \"common/x.glsl\"  ")).?);
    try std.testing.expect((try parseIncludeDirective("// just a comment")) == null);
    try std.testing.expect((try parseIncludeDirective("float x; // @include \"a\"")) == null);
    try std.testing.expectError(Error.BadIncludeDirective, parseIncludeDirective("// @include"));
    try std.testing.expectError(Error.BadIncludeDirective, parseIncludeDirective("// @include common/x.glsl"));
    try std.testing.expectError(Error.BadIncludeDirective, parseIncludeDirective("// @include \"\""));
    try std.testing.expectError(Error.BadIncludeDirective, parseIncludeDirective("// @include \"a"));
    try std.testing.expectError(Error.BadIncludeDirective, parseIncludeDirective("// @includes \"a\""));
}

test "expansion without directives is identity (modulo trailing newline)" {
    var mem = MemFs{};
    const src = "line one\nline two\n";
    const out = try expand(std.testing.allocator, src, "main.glsl", "root", mem.fs());
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings(src, out);
}

test "single include splices verbatim" {
    var mem = MemFs{};
    defer mem.files.deinit(std.testing.allocator);
    try mem.files.put(std.testing.allocator, "root/common/x.glsl", "A\nB\n");
    const src = "head\n// @include \"common/x.glsl\"\ntail\n";
    const out = try expand(std.testing.allocator, src, "main.glsl", "root", mem.fs());
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("head\nA\nB\ntail\n", out);
}

test "nested includes expand recursively" {
    var mem = MemFs{};
    defer mem.files.deinit(std.testing.allocator);
    try mem.files.put(std.testing.allocator, "root/inner.glsl", "INNER\n");
    try mem.files.put(std.testing.allocator, "root/outer.glsl", "O1\n// @include \"inner.glsl\"\nO2\n");
    const src = "// @include \"outer.glsl\"\n";
    const out = try expand(std.testing.allocator, src, "main.glsl", "root", mem.fs());
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualStrings("O1\nINNER\nO2\n", out);
}

test "include cycles are an error" {
    var mem = MemFs{};
    defer mem.files.deinit(std.testing.allocator);
    try mem.files.put(std.testing.allocator, "root/a.glsl", "// @include \"b.glsl\"\n");
    try mem.files.put(std.testing.allocator, "root/b.glsl", "// @include \"a.glsl\"\n");
    try std.testing.expectError(Error.IncludeCycle, expand(std.testing.allocator, "// @include \"a.glsl\"\n", "main.glsl", "root", mem.fs()));
    // Self-include is a cycle too.
    try std.testing.expectError(Error.IncludeCycle, expand(std.testing.allocator, "// @include \"main.glsl\"\n", "main.glsl", "root", mem.fs()));
}

test "missing include and root escape are errors" {
    var mem = MemFs{};
    defer mem.files.deinit(std.testing.allocator);
    try std.testing.expectError(Error.IncludeNotFound, expand(std.testing.allocator, "// @include \"nope.glsl\"\n", "main.glsl", "root", mem.fs()));
    try std.testing.expectError(Error.IncludeEscapesRoot, expand(std.testing.allocator, "// @include \"../evil.glsl\"\n", "main.glsl", "root", mem.fs()));
}

test "converted shaders expand fully: no directives remain, shared chunks present" {
    // Post-refactor invariant: every committed `// @include` resolves
    // (typos fail loudly in expand) and the shared chunk it names lands
    // in the output. memFsFromEmbeds wires the real checked-in files.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var mem = MemFs{};
    const inc_files = [_]struct { name: []const u8, text: []const u8 }{
        .{ .name = "shaders/common/fullscreen_vs.glsl", .text = @embedFile("../shaders/common/fullscreen_vs.glsl") },
        .{ .name = "shaders/common/cluster.glsl", .text = @embedFile("../shaders/common/cluster.glsl") },
        .{ .name = "shaders/common/shadow_pcf.glsl", .text = @embedFile("../shaders/common/shadow_pcf.glsl") },
        .{ .name = "shaders/common/uv_apply.glsl", .text = @embedFile("../shaders/common/uv_apply.glsl") },
        .{ .name = "shaders/common/refraction.glsl", .text = @embedFile("../shaders/common/refraction.glsl") },
        .{ .name = "shaders/common/pbr_brdf.glsl", .text = @embedFile("../shaders/common/pbr_brdf.glsl") },
        .{ .name = "shaders/common/channel_select.glsl", .text = @embedFile("../shaders/common/channel_select.glsl") },
        .{ .name = "shaders/common/hemi.glsl", .text = @embedFile("../shaders/common/hemi.glsl") },
        .{ .name = "shaders/common/hemi_pbr.glsl", .text = @embedFile("../shaders/common/hemi_pbr.glsl") },
        .{ .name = "shaders/common/specular_aa.glsl", .text = @embedFile("../shaders/common/specular_aa.glsl") },
        .{ .name = "shaders/common/linear_output.glsl", .text = @embedFile("../shaders/common/linear_output.glsl") },
        .{ .name = "shaders/common/box_project.glsl", .text = @embedFile("../shaders/common/box_project.glsl") },
        .{ .name = "shaders/common/frame_view.glsl", .text = @embedFile("../shaders/common/frame_view.glsl") },
        .{ .name = "shaders/common/frame_mat.glsl", .text = @embedFile("../shaders/common/frame_mat.glsl") },
        .{ .name = "shaders/common/morph_helpers.glsl", .text = @embedFile("../shaders/common/morph_helpers.glsl") },
        .{ .name = "shaders/common/pbr_fs_bindings.glsl", .text = @embedFile("../shaders/common/pbr_fs_bindings.glsl") },
        .{ .name = "shaders/common/pbr_shadow_consts.glsl", .text = @embedFile("../shaders/common/pbr_shadow_consts.glsl") },
        .{ .name = "shaders/common/ssr_trace.glsl", .text = @embedFile("../shaders/common/ssr_trace.glsl") },
        .{ .name = "shaders/common/screen_uv.glsl", .text = @embedFile("../shaders/common/screen_uv.glsl") },
        .{ .name = "shaders/common/pbr_surface.glsl", .text = @embedFile("../shaders/common/pbr_surface.glsl") },
    };
    for (inc_files) |f| try mem.files.put(alloc, f.name, f.text);
    const shaders = [_][]const u8{
        @embedFile("../shaders/bloom_down.glsl"),
        @embedFile("../shaders/bloom_up.glsl"),
        @embedFile("../shaders/glow_extract.glsl"),
        @embedFile("../shaders/glow_blur.glsl"),
        @embedFile("../shaders/ssao.glsl"),
        @embedFile("../shaders/ssao_blur.glsl"),
        @embedFile("../shaders/volumetric_blur.glsl"),
        @embedFile("../shaders/volumetric_raymarch.glsl"),
        @embedFile("../shaders/postprocess.glsl"),
        @embedFile("../shaders/pbr.glsl"),
        @embedFile("../shaders/instanced_pbr.glsl"),
        @embedFile("../shaders/skinned_pbr.glsl"),
        @embedFile("../shaders/ssr.glsl"),
    };
    for (shaders) |src| {
        const out = try expand(alloc, src, "x.glsl", "shaders", mem.fs());
        try std.testing.expect(std.mem.indexOf(u8, out, "@include") == null);
    }
    // Spot-check: the fullscreen body and the shadow block arrive intact.
    const bloom = try expand(alloc, @embedFile("../shaders/bloom_down.glsl"), "x.glsl", "shaders", mem.fs());
    try std.testing.expect(std.mem.indexOf(u8, bloom, "v_uv = vec2(texcoord0.x, 1.0 - texcoord0.y);") != null);
    const pbr = try expand(alloc, @embedFile("../shaders/pbr.glsl"), "x.glsl", "shaders", mem.fs());
    try std.testing.expect(std.mem.indexOf(u8, pbr, "float pcssBlockerAverage(") != null);
    try std.testing.expect(std.mem.indexOf(u8, pbr, "float distributionGGX(") != null);
    try std.testing.expect(std.mem.indexOf(u8, pbr, "float channelSelect(") != null);
}

// ---------------------------------------------------------------------------
// Option C drift checks: deliberately-duplicated spans that must evolve
// together. Each test pins the documented invariant from
// ../shaders/common/README.md so a future edit touching only one copy
// fails loudly. Cheap, deterministic, no file writes.
// ---------------------------------------------------------------------------

fn countOccurrences(haystack: []const u8, needle: []const u8) usize {
    var n: usize = 0;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, haystack, i, needle)) |pos| {
        n += 1;
        i = pos + needle.len;
    }
    return n;
}

test "drift: fs_view/fs_mat parity across the three forward shaders" {
    // Post-split invariant: no forward shader declares raw fs lanes
    // anymore — each wraps the two shared chunks exactly once, and the
    // per-draw probe lanes live exactly once in the shared material
    // chunk (layout parity across PBR variants is load-bearing — see
    // build.zig and lights/probe uploads).
    const shaders = [_][]const u8{
        @embedFile("../shaders/pbr.glsl"),
        @embedFile("../shaders/instanced_pbr.glsl"),
        @embedFile("../shaders/skinned_pbr.glsl"),
    };
    for (shaders) |src| {
        try std.testing.expectEqual(@as(usize, 1), countOccurrences(src, "@include \"common/frame_view.glsl\""));
        try std.testing.expectEqual(@as(usize, 1), countOccurrences(src, "@include \"common/frame_mat.glsl\""));
        try std.testing.expectEqual(@as(usize, 0), countOccurrences(src, "vec4 probe_params;"));
        try std.testing.expect(std.mem.indexOf(u8, src, "uniform fs_params") == null);
    }
    const mat = @embedFile("../shaders/common/frame_mat.glsl");
    try std.testing.expectEqual(@as(usize, 1), countOccurrences(mat, "vec4 probe_params;"));
    try std.testing.expectEqual(@as(usize, 1), countOccurrences(mat, "vec4 probe_box[2];"));
    const view = @embedFile("../shaders/common/frame_view.glsl");
    try std.testing.expectEqual(@as(usize, 1), countOccurrences(view, "mat4 cascade_view_proj[4];"));
    try std.testing.expectEqual(@as(usize, 1), countOccurrences(view, "mat4 refraction_view_proj;"));
}

test "drift: shadow depth-only programs keep their entry points" {
    // shadow.glsl keeps the rigid/instanced/skinned depth-only entry
    // points (the MSAA depth-prepass twin was removed with the gate).
    const shadow = @embedFile("../shaders/shadow.glsl");
    const programs = [_][]const u8{
        "@program shadow vs fs",
        "@program shadow_instanced vs_inst fs",
        "@program shadow_skinned vs_skinned fs",
    };
    for (programs) |p| try std.testing.expect(std.mem.indexOf(u8, shadow, p) != null);
    // Rigid stage stays position-only.
    try std.testing.expect(std.mem.indexOf(u8, shadow, "in vec3 position;") != null);
}

test "drift: morph helpers live once in common (PBR variants)" {
    // Post-dedup invariant: the GPU-morph helpers (morphTexel /
    // morphWeight / applyMorphDeltas) are defined exactly once, in
    // common/morph_helpers.glsl, and both morph-capable vertex shaders
    // include that chunk instead of carrying inline copies. The arity
    // (tan_xyz lane) is pinned on the single definition; a signature
    // change without updating both call sites fails loudly here.
    // instanced_pbr has no morph path and must not include the chunk.
    const pbr = @embedFile("../shaders/pbr.glsl");
    const skinned = @embedFile("../shaders/skinned_pbr.glsl");
    const instanced = @embedFile("../shaders/instanced_pbr.glsl");
    const chunk = @embedFile("../shaders/common/morph_helpers.glsl");
    for ([2][]const u8{ pbr, skinned }) |src| {
        try std.testing.expectEqual(@as(usize, 1), countOccurrences(src, "@include \"common/morph_helpers.glsl\""));
        try std.testing.expectEqual(@as(usize, 0), countOccurrences(src, "vec3 morphTexel(float idx) {"));
        try std.testing.expectEqual(@as(usize, 0), countOccurrences(src, "float morphWeight(int t) {"));
    }
    try std.testing.expectEqual(@as(usize, 0), countOccurrences(instanced, "morph_helpers"));
    try std.testing.expectEqual(@as(usize, 1), countOccurrences(chunk, "void applyMorphDeltas(inout vec3 pos, inout vec3 nrm, inout vec3 tan_xyz, int vertex_id) {"));
    try std.testing.expectEqual(@as(usize, 1), countOccurrences(chunk, "vec3 morphTexel(float idx) {"));
    try std.testing.expectEqual(@as(usize, 1), countOccurrences(chunk, "float morphWeight(int t) {"));
}

test "drift: PBR fs bindings and shadow consts live once in common" {
    // Post-dedup invariant: the three PBR forward fragment shaders share
    // their texture/sampler view bindings (common/pbr_fs_bindings.glsl)
    // and their shadow-sampling constants (common/pbr_shadow_consts.glsl)
    // by include. No variant declares a conflicting inline copy: the
    // sampler/view layout is load-bearing (see scene/draw.zig), so a
    // binding added to one variant but not the chunk fails loudly here.
    const shaders = [_][]const u8{
        @embedFile("../shaders/pbr.glsl"),
        @embedFile("../shaders/instanced_pbr.glsl"),
        @embedFile("../shaders/skinned_pbr.glsl"),
    };
    for (shaders) |src| {
        try std.testing.expectEqual(@as(usize, 1), countOccurrences(src, "@include \"common/pbr_fs_bindings.glsl\""));
        try std.testing.expectEqual(@as(usize, 1), countOccurrences(src, "@include \"common/pbr_shadow_consts.glsl\""));
        try std.testing.expectEqual(@as(usize, 0), countOccurrences(src, "uniform texture2D albedo_tex;"));
        try std.testing.expectEqual(@as(usize, 0), countOccurrences(src, "const vec2 POISSON_DISK[16]"));
        try std.testing.expectEqual(@as(usize, 0), countOccurrences(src, "const vec2 CASCADE_OFFSETS[4]"));
        try std.testing.expectEqual(@as(usize, 0), countOccurrences(src, "#define PCSS_BLOCKER_SAMPLES"));
    }
    const bindings = @embedFile("../shaders/common/pbr_fs_bindings.glsl");
    try std.testing.expectEqual(@as(usize, 1), countOccurrences(bindings, "layout(binding = 10) uniform texture2D point_shadow_tex;"));
    try std.testing.expectEqual(@as(usize, 1), countOccurrences(bindings, "layout(binding = 21) uniform texture2D point_shadow_depth_tex;"));
    const consts = @embedFile("../shaders/common/pbr_shadow_consts.glsl");
    try std.testing.expectEqual(@as(usize, 1), countOccurrences(consts, "const vec2 POISSON_DISK[16]"));
    try std.testing.expectEqual(@as(usize, 1), countOccurrences(consts, "#define POINT_PCSS_MAX_PENUMBRA 0.01"));
}

test "drift: ssr reconstruction helpers live once in common" {
    // Post-dedup invariant: the depth-derived world-pos/normal helpers
    // (reconstructWorldPos/reconstructWorldNormal) are defined exactly
    // once, in common/ssr_trace.glsl, and both the fullscreen composite
    // (postprocess.glsl applySSR) and the half-res trace (ssr.glsl)
    // include that chunk instead of carrying inline copies. The march
    // loops stay per-shader on purpose: inline mixes while the trace
    // outputs (reflection, weight), and unifying them would rephrase the
    // v1 arithmetic — only the pure helpers are shared.
    const composite = @embedFile("../shaders/postprocess.glsl");
    const trace = @embedFile("../shaders/ssr.glsl");
    const chunk = @embedFile("../shaders/common/ssr_trace.glsl");
    for ([2][]const u8{ composite, trace }) |src| {
        try std.testing.expectEqual(@as(usize, 1), countOccurrences(src, "@include \"common/ssr_trace.glsl\""));
        try std.testing.expectEqual(@as(usize, 0), countOccurrences(src, "vec3 reconstructWorldPos(vec2 uv, float depth) {"));
        try std.testing.expectEqual(@as(usize, 0), countOccurrences(src, "vec3 reconstructWorldNormal(vec2 uv, vec3 world_pos) {"));
    }
    try std.testing.expectEqual(@as(usize, 1), countOccurrences(chunk, "vec3 reconstructWorldPos(vec2 uv, float depth) {"));
    try std.testing.expectEqual(@as(usize, 1), countOccurrences(chunk, "vec3 reconstructWorldNormal(vec2 uv, vec3 world_pos) {"));
}

test "drift: probe_mip stays excluded from the fullscreen include" {
    // probe_mip.glsl has no Y-flip (cube-face texel centers, not screen
    // UVs) and must NOT adopt common/fullscreen_vs.glsl. If someone adds
    // the flip "for consistency", probe faces sample mirrored.
    const probe = @embedFile("../shaders/probe_mip.glsl");
    try std.testing.expect(std.mem.indexOf(u8, probe, "1.0 - texcoord0.y") == null);
    try std.testing.expect(std.mem.indexOf(u8, probe, "@include") == null);
}

test "skinned/instanced share one PBR surface body (rigid keeps its hooks)" {
    const chunk = @embedFile("../shaders/common/pbr_surface.glsl");
    try std.testing.expect(std.mem.indexOf(u8, chunk, "vec3 pbrSurfaceLighting(vec4 albedo_rgba, vec3 N_geom)") != null);
    for ([_][]const u8{
        @embedFile("../shaders/skinned_pbr.glsl"),
        @embedFile("../shaders/instanced_pbr.glsl"),
    }) |src| {
        try std.testing.expectEqual(@as(usize, 1), countOccurrences(src, "@include \"common/pbr_surface.glsl\""));
        try std.testing.expectEqual(@as(usize, 1), countOccurrences(src, "pbrSurfaceLighting(albedo_rgba, normalize(v_normal))"));
        // The old duplicated tail must not reappear.
        try std.testing.expectEqual(@as(usize, 0), countOccurrences(src, "float aa_rough = aaRoughnessFactor(N);"));
    }
    // The rigid shader has hook-scoped locals and intentionally keeps its copy.
    try std.testing.expectEqual(@as(usize, 0), countOccurrences(@embedFile("../shaders/pbr.glsl"), "@include \"common/pbr_surface.glsl\""));
}
