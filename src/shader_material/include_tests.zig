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

test "drift: fs_params probe lane parity across the three forward shaders" {
    // Every forward fs_params block carries `vec4 probe_params;` exactly
    // once (per-draw probe state appended last; layout parity across PBR
    // variants is load-bearing — see build.zig and lights/probe uploads).
    const shaders = [_][]const u8{
        @embedFile("../shaders/pbr.glsl"),
        @embedFile("../shaders/instanced_pbr.glsl"),
        @embedFile("../shaders/skinned_pbr.glsl"),
    };
    for (shaders) |src| {
        try std.testing.expectEqual(@as(usize, 1), countOccurrences(src, "vec4 probe_params;"));
        try std.testing.expectEqual(@as(usize, 1), countOccurrences(src, "vec4 probe_box[2];"));
    }
}

test "drift: depth-only twin programs (shadow vs msaa_depth)" {
    // shadow.glsl and msaa_depth.glsl are documented twins (same
    // attribute layout, same rigid/instanced/skinned entry points, same
    // depth LESS_EQUAL/write pipelines). Adding an entry point to one
    // but not the other is drift: pin the @program sets equal.
    const shadow = @embedFile("../shaders/shadow.glsl");
    const msaa = @embedFile("../shaders/msaa_depth.glsl");
    const programs = [_][]const u8{
        "@program shadow vs fs",
        "@program shadow_instanced vs_inst fs",
        "@program shadow_skinned vs_skinned fs",
        "@program msaa_depth vs fs",
        "@program msaa_depth_instanced vs_inst fs",
        "@program msaa_depth_skinned vs_skinned fs",
    };
    for (programs[0..3]) |p| try std.testing.expect(std.mem.indexOf(u8, shadow, p) != null);
    for (programs[3..]) |p| try std.testing.expect(std.mem.indexOf(u8, msaa, p) != null);
    // Twin invariant: position-only vertex input on both rigid stages.
    try std.testing.expect(std.mem.indexOf(u8, shadow, "in vec3 position;") != null);
    try std.testing.expect(std.mem.indexOf(u8, msaa, "in vec3 position;") != null);
}

test "drift: morph helper arity (PBR variants)" {
    const pbr = @embedFile("../shaders/pbr.glsl");
    const skinned = @embedFile("../shaders/skinned_pbr.glsl");
    try std.testing.expect(std.mem.indexOf(u8, pbr, "void applyMorphDeltas(inout vec3 pos, inout vec3 nrm, inout vec3 tan_xyz, int vertex_id)") != null);
    try std.testing.expect(std.mem.indexOf(u8, skinned, "void applyMorphDeltas(inout vec3 pos, inout vec3 nrm, inout vec3 tan_xyz, int vertex_id)") != null);
}

test "drift: probe_mip stays excluded from the fullscreen include" {
    // probe_mip.glsl has no Y-flip (cube-face texel centers, not screen
    // UVs) and must NOT adopt common/fullscreen_vs.glsl. If someone adds
    // the flip "for consistency", probe faces sample mirrored.
    const probe = @embedFile("../shaders/probe_mip.glsl");
    try std.testing.expect(std.mem.indexOf(u8, probe, "1.0 - texcoord0.y") == null);
    try std.testing.expect(std.mem.indexOf(u8, probe, "@include") == null);
}
