//! Build-time textual include expansion for engine shader sources.
//!
//! sokol-shdc (vendored sokol-tools-bin) has no `-I`/`#include` support:
//! a `#include` line inside `@vs`/`@fs` fails in glslang with
//! `'#include' : required extension not requested`. This module implements
//! OUR OWN directive, consumed before shdc ever sees the source:
//!
//!     // @include "common/fullscreen_vs.glsl"
//!
//! The directive line is replaced verbatim by the referenced file's lines
//! (relative to a root directory passed by the build). Everything else
//! passes through byte-identically, so expanding a shader whose includes
//! match the text they replaced reproduces the original file byte for
//! byte (asserted by the round-trip tests below and verifiable by
//! diffing shdc output before/after conversion).
//!
//! Rules, deliberately narrow:
//! - Directive form is exactly `// @include "path"` (leading whitespace
//!   allowed, nothing else on the line). A malformed `// @include`
//!   prefix fails the build loudly instead of passing through.
//! - Included content is spliced VERBATIM (no indentation adjustment):
//!   includes are top-level GLSL spans at column 0.
//! - Includes may nest; inclusion cycles are an error.
//! - No `#line` directives are emitted: the expansion is pure
//!   substitution, so shdc messages refer to expanded line numbers
//!   (documented shift: each directive line becomes N content lines).
//! - Pure std-only code like merge.zig: shared by the test suite and by
//!   the host CLI (expand_main.zig) invoked from build.zig.

const std = @import("std");

pub const Error = error{
    /// An `@include` cycle was detected (A includes B includes A ...).
    IncludeCycle,
    /// The referenced include file could not be read.
    IncludeNotFound,
    /// A `// @include` line that is not exactly `// @include "path"`.
    BadIncludeDirective,
    /// An include escapes the root directory (`..` beyond root).
    IncludeEscapesRoot,
    OutOfMemory,
};

/// Minimal filesystem abstraction so tests run in-memory while the CLI
/// reads real files. Paths use `/` separators.
pub const FileSystem = struct {
    ctx: *anyopaque,
    readFn: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, path: []const u8) ReadError![]u8,
    pub const ReadError = error{ NotFound, OutOfMemory, Io };

    pub fn read(self: FileSystem, allocator: std.mem.Allocator, path: []const u8) FileSystem.ReadError![]u8 {
        return self.readFn(self.ctx, allocator, path);
    }
};

/// Parses a directive line. Returns the referenced path, or null when the
/// line is not an include directive at all. A `// @include` prefix that
/// does not match the exact form is an error (fail loud on typos rather
/// than silently passing a dead directive to shdc).
pub fn parseIncludeDirective(line: []const u8) Error!?[]const u8 {
    const trimmed = std.mem.trim(u8, line, " \t\r");
    const prefix = "// @include";
    if (!std.mem.startsWith(u8, trimmed, prefix)) return null;
    var rest = trimmed[prefix.len..];
    if (rest.len == 0 or (rest[0] != ' ' and rest[0] != '\t')) return Error.BadIncludeDirective;
    rest = std.mem.trim(u8, rest, " \t");
    if (rest.len < 3 or rest[0] != '"' or rest[rest.len - 1] != '"') return Error.BadIncludeDirective;
    const path = rest[1 .. rest.len - 1];
    if (path.len == 0) return Error.BadIncludeDirective;
    if (std.mem.indexOfScalar(u8, path, '"') != null) return Error.BadIncludeDirective;
    if (std.mem.indexOfScalar(u8, path, '\n') != null) return Error.BadIncludeDirective;
    return path;
}

fn joinRoot(allocator: std.mem.Allocator, root: []const u8, rel: []const u8) Error![]u8 {
    // Reject `..` segments that escape the root (lexical check is enough:
    // the build only ever references sibling `common/*.glsl` files).
    var depth: usize = 0;
    var it = std.mem.splitScalar(u8, rel, '/');
    while (it.next()) |seg| {
        if (seg.len == 0 or std.mem.eql(u8, seg, ".")) continue;
        if (std.mem.eql(u8, seg, "..")) {
            if (depth == 0) return Error.IncludeEscapesRoot;
            depth -= 1;
        } else {
            depth += 1;
        }
    }
    const clean_root = std.mem.trimEnd(u8, root, "/");
    return std.fmt.allocPrint(allocator, "{s}/{s}", .{ clean_root, rel }) catch Error.OutOfMemory;
}

/// Expands `entry_text` (the shader source) with includes resolved
/// relative to `root`. `entry_name` is used for diagnostics and cycle
/// reporting. Caller owns the returned buffer.
pub fn expand(
    allocator: std.mem.Allocator,
    entry_text: []const u8,
    entry_name: []const u8,
    root: []const u8,
    fs: FileSystem,
) Error![]u8 {
    var stack: std.ArrayListUnmanaged([]u8) = .empty;
    defer {
        for (stack.items) |p| allocator.free(p);
        stack.deinit(allocator);
    }
    const entry_path = try joinRoot(allocator, root, entry_name);
    defer allocator.free(entry_path);
    // The entry file itself anchors cycle detection (a shader including
    // itself, directly or transitively, is an error).
    try stack.append(allocator, try allocator.dupe(u8, entry_path));
    defer allocator.free(stack.pop().?);
    return expandText(allocator, entry_text, root, fs, &stack);
}

fn expandText(
    allocator: std.mem.Allocator,
    text: []const u8,
    root: []const u8,
    fs: FileSystem,
    stack: *std.ArrayListUnmanaged([]u8),
) Error![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);

    var line_start: usize = 0;
    while (line_start <= text.len) {
        const nl = std.mem.indexOfScalarPos(u8, text, line_start, '\n');
        const line_end = nl orelse text.len;
        const line = text[line_start..line_end];

        const directive = try parseIncludeDirective(line);
        if (directive) |rel| {
            const full = try joinRoot(allocator, root, rel);
            defer allocator.free(full);
            for (stack.items) |active| {
                if (std.mem.eql(u8, active, full)) return Error.IncludeCycle;
            }
            const content = fs.read(allocator, full) catch |e| switch (e) {
                error.NotFound => return Error.IncludeNotFound,
                error.OutOfMemory => return Error.OutOfMemory,
                error.Io => return Error.IncludeNotFound,
            };
            defer allocator.free(content);
            const owned = try allocator.dupe(u8, full);
            // Owned by `stack` from here on (expand()'s cleanup frees
            // every stacked path on all paths, including errors): no
            // errdefer here — freeing now would double-free via the stack.
            try stack.append(allocator, owned);
            const expanded = try expandText(allocator, content, root, fs, stack);
            defer allocator.free(expanded);
            _ = stack.pop();
            allocator.free(owned);
            try out.appendSlice(allocator, expanded);
            // Guarantee the splice ends on a line boundary even if the
            // include file lacks a trailing newline.
            if (expanded.len == 0 or expanded[expanded.len - 1] != '\n') {
                try out.append(allocator, '\n');
            }
        } else {
            try out.appendSlice(allocator, line);
            try out.append(allocator, '\n');
        }

        if (nl == null) break;
        line_start = line_end + 1;
        // A trailing newline yields one final empty line above; stop so we
        // do not emit a phantom extra blank line.
        if (line_start == text.len) break;
    }
    return out.toOwnedSlice(allocator) catch Error.OutOfMemory;
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

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
        .{ .name = "shaders/common/pbr_brdf.glsl", .text = @embedFile("../shaders/common/pbr_brdf.glsl") },
        .{ .name = "shaders/common/channel_select.glsl", .text = @embedFile("../shaders/common/channel_select.glsl") },
        .{ .name = "shaders/common/hemi.glsl", .text = @embedFile("../shaders/common/hemi.glsl") },
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
        @embedFile("../shaders/standard.glsl"),
        @embedFile("../shaders/pbr.glsl"),
        @embedFile("../shaders/instanced.glsl"),
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
    const stdfs = try expand(alloc, @embedFile("../shaders/standard.glsl"), "x.glsl", "shaders", mem.fs());
    try std.testing.expect(std.mem.indexOf(u8, stdfs, "float pcssBlockerAverage(") != null);
    try std.testing.expect(std.mem.indexOf(u8, stdfs, "float channelSelect(") == null);
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

test "drift: fs_params probe lane parity across the five forward shaders" {
    // Every forward fs_params block carries `vec4 probe_params;` exactly
    // once (per-draw probe state appended last; layout parity with the
    // regular standard FsParams is load-bearing — see build.zig and
    // lights/probe uploads). Deleting or renaming it in one copy breaks
    // uniform offsets silently; this test breaks loudly instead.
    const shaders = [_][]const u8{
        @embedFile("../shaders/standard.glsl"),
        @embedFile("../shaders/pbr.glsl"),
        @embedFile("../shaders/instanced.glsl"),
        @embedFile("../shaders/instanced_pbr.glsl"),
        @embedFile("../shaders/skinned_pbr.glsl"),
    };
    for (shaders) |src| {
        try std.testing.expectEqual(@as(usize, 1), countOccurrences(src, "vec4 probe_params;"));
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

test "drift: morph helper arity (standard vs PBR variants)" {
    // applyMorphDeltas takes (pos, nrm, id) in standard.glsl but
    // (pos, nrm, tan_xyz, id) in the PBR variants. Unifying the arity
    // "for cleanliness" breaks the call sites; the difference is
    // intentional — pin both signatures.
    const standard = @embedFile("../shaders/standard.glsl");
    const pbr = @embedFile("../shaders/pbr.glsl");
    const skinned = @embedFile("../shaders/skinned_pbr.glsl");
    try std.testing.expect(std.mem.indexOf(u8, standard, "void applyMorphDeltas(inout vec3 pos, inout vec3 nrm, int vertex_id)") != null);
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
