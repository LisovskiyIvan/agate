//! Build-time hook injection for shader materials.
//!
//! Engine shader templates (src/agate/shaders/*.glsl) contain comment-only
//! hook blocks:
//!
//!     // @hook(albedo)
//!     // @endhook
//!
//! A user snippet overrides a hook by providing a section with the same
//! name; its GLSL statements replace the (empty) default block content. When
//! a hook is NOT overridden the merged output is byte-identical to the
//! template, so the un-hooked shader path is untouched by construction.
//!
//! Snippet format (plain GLSL file):
//!
//!     // agate shader material: lava
//!     // base: standard
//!     // @param u_speed float = 2.0
//!     // @param u_tint vec4 = 1.0 0.5 0.0 1.0
//!
//!     // @hook(albedo)
//!     base.rgb = mix(u_tint.rgb, base.rgb, 0.5);
//!
//! `// @param` declares a user uniform packed into a fixed 8x vec4 UB block
//! (`sm_user_params`, binding 3) that the merger generates into the
//! template's `@hook(decls)` block together with `#define` shortcuts.
//!
//! Pure std-only code: shared by the engine test suite (golden tests) and by
//! the host CLI tool (src/agate/shader_material/tool_main.zig) invoked from
//! build.zig.

const std = @import("std");

/// Fixed capacity of the user uniform block: 8 vec4 slots = 128 bytes.
pub const user_slot_count: usize = 8;
/// f32 words across the user uniform block (32 floats).
pub const user_word_count: usize = user_slot_count * 4;
/// Fixed UB bindings for the generated user uniform blocks (UB slots 0..2
/// are used by the engine templates: vs_params, fs_params, vs_morph).
pub const fs_params_binding: u32 = 3;
pub const vs_params_binding: u32 = 4;

/// One declared user uniform: an f32-aligned window inside the 32-float
/// storage. `offset` is in f32 units (slot = offset / 4, lane = offset % 4).
pub const Param = struct {
    name: []const u8,
    offset: u8,
    /// 1 (float) or 4 (vec4)
    comps: u8,
    default: [4]f32 = .{ 0, 0, 0, 0 },
};

pub const Error = error{
    /// `// @hook(...)` without a matching `// @endhook`.
    HookNotTerminated,
    /// Snippet overrides a hook the template does not define.
    UnknownHook,
    /// More params than the 32-float user block can hold.
    ParamSpaceOverflow,
    /// Malformed `// @param` declaration.
    BadParamDecl,
    /// Snippet declares params but the template has no `@hook(decls)` block.
    MissingDeclsHook,
    /// Snippet `// base:` header disagrees with the requested base template.
    BaseMismatch,
    OutOfMemory,
};

pub const Options = struct {
    template: []const u8,
    /// Null (or empty) merges to a byte-identical copy of the template.
    snippet: ?[]const u8 = null,
    /// Expected base template name ("standard" / "pbr"); checked against the
    /// snippet's `// base:` header when that header is present.
    base: []const u8 = "pbr",
    /// For diagnostics in error messages.
    material_name: []const u8 = "",
};

pub const Result = struct {
    /// Merged GLSL source (caller owns).
    glsl: []u8,
    /// Declared user params in declaration order (caller owns).
    params: []Param,
};

const HookBlock = struct {
    name: []const u8,
    stage: Stage,
    /// Line indices into the template's line list (open marker inclusive,
    /// close marker exclusive bounds of the *content*).
    content_start: usize,
    content_end: usize,
};

const Line = struct {
    /// Slice into the source text, without the trailing newline.
    text: []const u8,
    start: usize,
};

fn splitLines(allocator: std.mem.Allocator, text: []const u8) ![]Line {
    var lines: std.ArrayListUnmanaged(Line) = .empty;
    errdefer lines.deinit(allocator);
    var i: usize = 0;
    while (i < text.len) {
        const nl = std.mem.indexOfScalarPos(u8, text, i, '\n') orelse text.len;
        try lines.append(allocator, .{ .start = i, .text = text[i..nl] });
        i = nl + 1;
    }
    // Tolerate a missing trailing newline (the split above already emitted
    // the final line) but also a trailing newline followed by nothing.
    return lines.toOwnedSlice(allocator);
}

fn hookNameOnLine(line_text: []const u8) ?[]const u8 {
    const trimmed = std.mem.trim(u8, line_text, " \t\r");
    const prefix = "// @hook(";
    if (!std.mem.startsWith(u8, trimmed, prefix)) return null;
    const rest = trimmed[prefix.len..];
    const close = std.mem.indexOfScalar(u8, rest, ')') orelse return null;
    if (rest[close + 1 ..].len != 0) return null; // trailing junk: not a marker
    if (close == 0) return null;
    return rest[0..close];
}

fn isEndhookLine(line_text: []const u8) bool {
    return std.mem.eql(u8, std.mem.trim(u8, line_text, " \t\r"), "// @endhook");
}

/// Shader stage a template hook block lives in (sokol-shdc @vs / @fs
/// sections). Drives which generated uniform block a `decls` hook receives.
pub const Stage = enum { vs, fs };

fn sectionOnLine(line_text: []const u8) ?Stage {
    const trimmed = std.mem.trim(u8, line_text, " \t\r");
    // Section headers carry the program name: "@vs vs" / "@fs fs".
    if (std.mem.startsWith(u8, trimmed, "@vs")) return .vs;
    if (std.mem.startsWith(u8, trimmed, "@fs")) return .fs;
    return null;
}

/// Scans hook blocks: pairs of `// @hook(name)` ... `// @endhook` lines.
/// Repeated names are allowed and intentional for the `decls` hook (one per
/// shader stage): a snippet section (or generated params block) splices into
/// every block of that name.
fn parseTemplateHooks(allocator: std.mem.Allocator, lines: []const Line) Error![]HookBlock {
    var hooks: std.ArrayListUnmanaged(HookBlock) = .empty;
    errdefer hooks.deinit(allocator);
    var stage: Stage = .fs;
    var i: usize = 0;
    while (i < lines.len) : (i += 1) {
        if (sectionOnLine(lines[i].text)) |st| {
            stage = st;
            continue;
        }
        const name = hookNameOnLine(lines[i].text) orelse continue;
        var j = i + 1;
        while (j < lines.len and !isEndhookLine(lines[j].text)) : (j += 1) {
            if (hookNameOnLine(lines[j].text) != null) return Error.HookNotTerminated;
        }
        if (j >= lines.len) return Error.HookNotTerminated;
        try hooks.append(allocator, .{ .name = name, .stage = stage, .content_start = i + 1, .content_end = j });
        i = j;
    }
    return hooks.toOwnedSlice(allocator);
}

const SnippetHook = struct {
    name: []const u8,
    /// Trimmed section body (view into `alloc`).
    body: []const u8,
    /// Full allocation backing `body` (freed by freeSnippetHooks).
    alloc: []u8 = &.{},
};

/// Frees a parseSnippet result (bodies + tables).
fn freeSnippetHooks(allocator: std.mem.Allocator, hooks: []SnippetHook) void {
    for (hooks) |h| {
        if (h.alloc.len > 0) allocator.free(h.alloc);
    }
    allocator.free(hooks);
}

fn trimBlankEdges(s: []const u8) []const u8 {
    return std.mem.trim(u8, s, " \t\r\n");
}

/// Parses a snippet into hook sections + declared params. Line-based:
/// `// @hook(name)` starts a section that runs until the next marker-ish
/// line (`// @hook`, `// @param`, `// base:`) or EOF.
fn parseSnippet(allocator: std.mem.Allocator, snippet: []const u8, opts: Options) Error!struct {
    hooks: []SnippetHook,
    params: []Param,
} {
    const lines = try splitLines(allocator, snippet);
    defer allocator.free(lines);

    var hooks: std.ArrayListUnmanaged(SnippetHook) = .empty;
    var params: std.ArrayListUnmanaged(Param) = .empty;
    errdefer {
        freeSnippetHooks(allocator, hooks.items);
        params.deinit(allocator);
    }

    var next_word_offset: u8 = 0;

    var i: usize = 0;
    while (i < lines.len) : (i += 1) {
        const line = std.mem.trim(u8, lines[i].text, " \t\r");
        if (std.mem.startsWith(u8, line, "// base:")) {
            const declared = std.mem.trim(u8, line["// base:".len..], " \t");
            const norm_decl = if (std.mem.eql(u8, declared, "standard")) "pbr" else declared;
            const norm_base = if (std.mem.eql(u8, opts.base, "standard")) "pbr" else opts.base;
            if (norm_decl.len > 0 and !std.mem.eql(u8, norm_decl, norm_base)) {
                return Error.BaseMismatch;
            }
            continue;
        }
        if (std.mem.startsWith(u8, line, "// @param")) {
            const p = (try parseParamDecl(line, next_word_offset)) orelse return Error.BadParamDecl;
            // Continue from the ALIGNED offset (vec4 params round up to a
            // slot boundary inside parseParamDecl).
            next_word_offset = p.offset + p.comps;
            if (next_word_offset > user_word_count) return Error.ParamSpaceOverflow;
            try params.append(allocator, p);
            continue;
        }
        if (hookNameOnLine(line)) |name| {
            // Collect the body until the next marker-ish line or EOF. The
            // buffer's storage is OWNED by the returned hook (toOwnedSlice):
            // it lives as long as the hooks slice freed by the caller.
            var body: std.ArrayListUnmanaged(u8) = .empty;
            var j = i + 1;
            while (j < lines.len) : (j += 1) {
                const inner = std.mem.trim(u8, lines[j].text, " \t\r");
                if (hookNameOnLine(inner) != null or
                    std.mem.startsWith(u8, inner, "// @param") or
                    std.mem.startsWith(u8, inner, "// base:")) break;
                try body.appendSlice(allocator, lines[j].text);
                try body.append(allocator, '\n');
            }
            const full = try body.toOwnedSlice(allocator);
            try hooks.append(allocator, .{ .name = name, .body = trimBlankEdges(full), .alloc = full });
            i = j - 1; // outer loop's += 1 lands on the next marker line
        }
    }

    return .{
        .hooks = try hooks.toOwnedSlice(allocator),
        .params = try params.toOwnedSlice(allocator),
    };
}

pub const reserved_param_names = [_][]const u8{
    "time",  "roughness",   "metallic",     "albedo",       "normal",      "position",
    "color", "gl_Position", "gl_FragColor", "gl_FragCoord", "v_world_pos", "v_normal",
    "v_uv",  "v_color",     "v_tangent",    "v_bitangent",
};

// Parses `// @param <name> <float|vec4> [= d0 d1 d2 d3]` at the given
// f32 allocation cursor. Returns null for comment headers that are not
// actual declarations (caller maps that to BadParamDecl where required).
fn parseParamDecl(line: []const u8, cursor: u8) Error!?Param {
    const trimmed = std.mem.trim(u8, line, " \t\r");
    const prefix = "// @param";
    if (!std.mem.startsWith(u8, trimmed, prefix)) return null;
    var rest = std.mem.trim(u8, trimmed[prefix.len..], " \t");
    if (rest.len == 0 or rest[0] == '=') return Error.BadParamDecl;
    // name
    const sp1 = std.mem.indexOfAny(u8, rest, " \t") orelse return Error.BadParamDecl;
    const name = rest[0..sp1];
    for (reserved_param_names) |res| {
        if (std.mem.eql(u8, name, res)) return Error.BadParamDecl;
    }
    rest = std.mem.trimStart(u8, rest[sp1..], " \t");
    // type
    const sp2 = std.mem.indexOfAny(u8, rest, " \t") orelse rest.len;
    const type_str = rest[0..sp2];
    rest = if (sp2 < rest.len) std.mem.trimStart(u8, rest[sp2..], " \t") else "";
    var comps: u8 = undefined;
    var default: [4]f32 = .{ 0, 0, 0, 0 };
    if (std.mem.eql(u8, type_str, "float")) {
        comps = 1;
    } else if (std.mem.eql(u8, type_str, "vec4")) {
        comps = 4;
    } else {
        return Error.BadParamDecl;
    }
    // optional defaults after '='
    if (rest.len > 0) {
        if (rest[0] != '=') return Error.BadParamDecl;
        var values = std.mem.tokenizeAny(u8, rest[1..], " \t");
        var got: usize = 0;
        while (values.next()) |tok| {
            const v = std.fmt.parseFloat(f32, tok) catch return Error.BadParamDecl;
            if (got >= 4) return Error.BadParamDecl;
            default[got] = v;
            got += 1;
        }
        if (got != comps) return Error.BadParamDecl;
    }
    // vec4 params occupy a full slot (4-aligned); floats take one word.
    var offset = cursor;
    if (comps == 4) offset = @intCast(std.mem.alignForward(u32, cursor, 4));
    if (@as(u32, offset) + comps > user_word_count) return Error.ParamSpaceOverflow;
    return .{ .name = name, .offset = offset, .comps = comps, .default = default };
}

/// GLSL text injected into a template `@hook(decls)` block when the snippet
/// declares params: the fixed 8x vec4 UB plus `#define` shortcuts.
///
/// sokol-shdc requires unique uniform block names across the program, so the
/// two stages get distinct blocks when BOTH are needed (snippet uses params
/// in @hook(vertex)):
///   fs: binding 3, block `sm_user_params`,   members `sm_user_N`
///   vs: binding 4, block `sm_user_params_vs`, members `sm_vu_N`
/// With fs-only params there is a single block (fs, binding 3). The runtime
/// uploads the same 128-byte storage to whichever blocks exist.
pub fn generateUserParamsGlsl(allocator: std.mem.Allocator, stage: Stage, material_name: []const u8, params: []const Param) ![]u8 {
    const block_name: []const u8 = switch (stage) {
        .fs => "sm_user_params",
        .vs => "sm_user_params_vs",
    };
    const member_prefix: []const u8 = switch (stage) {
        .fs => "sm_user_",
        .vs => "sm_vu_",
    };
    const binding: u32 = switch (stage) {
        .fs => fs_params_binding,
        .vs => vs_params_binding,
    };
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);
    try out.print(allocator, "// --- user uniform block generated for shader material '{s}' (do not edit) ---\n", .{material_name});
    try out.print(allocator, "layout(binding = {d}) uniform {s} {{\n", .{ binding, block_name });
    var i: usize = 0;
    while (i < user_slot_count) : (i += 1) {
        try out.print(allocator, "    vec4 {s}{d};\n", .{ member_prefix, i });
    }
    try out.appendSlice(allocator, "};\n");
    for (params) |p| {
        // Vec4 params reference the whole slot member; floats add the lane.
        const lane_suffix: []const u8 = switch (p.comps) {
            4 => "",
            else => laneDot(p.offset % 4),
        };
        try out.print(allocator, "#define {s} ({s}{d}{s})\n", .{ p.name, member_prefix, p.offset / 4, lane_suffix });
    }
    return out.toOwnedSlice(allocator);
}

fn laneDot(lane: u8) []const u8 {
    return switch (lane) {
        0 => ".x",
        1 => ".y",
        2 => ".z",
        3 => ".w",
        else => unreachable,
    };
}

/// Merges a user snippet into an engine shader template.
/// With no overriding hooks the output is byte-identical to the template.
pub fn merge(allocator: std.mem.Allocator, opts: Options) Error!Result {
    const template_lines = try splitLines(allocator, opts.template);
    defer allocator.free(template_lines);
    const template_hooks = try parseTemplateHooks(allocator, template_lines);
    defer allocator.free(template_hooks);

    var snippet_hooks: []SnippetHook = &.{};
    var params: []Param = &.{};
    var have_snippet = false;
    if (opts.snippet) |s| {
        if (trimBlankEdges(s).len > 0) {
            const parsed = try parseSnippet(allocator, s, opts);
            snippet_hooks = parsed.hooks;
            params = parsed.params;
            have_snippet = true;
        }
    }
    defer if (have_snippet) {
        freeSnippetHooks(allocator, snippet_hooks);
        allocator.free(params);
    };

    // Unknown snippet hooks are almost certainly typos: fail the build.
    for (snippet_hooks) |sh| {
        var found = false;
        for (template_hooks) |th| {
            if (std.mem.eql(u8, th.name, sh.name)) {
                found = true;
                break;
            }
        }
        if (!found) return Error.UnknownHook;
    }

    // Params need a decls hook to land in (fs stage is mandatory, the vs
    // block is only generated when the snippet's vertex hook uses them).
    var have_fs_decls = false;
    var have_vs_decls = false;
    var vertex_overridden = false;
    for (template_hooks) |th| {
        if (std.mem.eql(u8, th.name, "decls")) {
            switch (th.stage) {
                .fs => have_fs_decls = true,
                .vs => have_vs_decls = true,
            }
        }
    }
    for (snippet_hooks) |sh| {
        if (std.mem.eql(u8, sh.name, "vertex")) vertex_overridden = true;
    }
    if (params.len > 0 and !have_fs_decls) return Error.MissingDeclsHook;
    if (params.len > 0 and vertex_overridden and !have_vs_decls) return Error.MissingDeclsHook;

    // Generate each stage's user params block up front (referenced during
    // the splice pass below; freed at function exit).
    var fs_params_glsl: ?[]u8 = null;
    var vs_params_glsl: ?[]u8 = null;
    if (params.len > 0) {
        fs_params_glsl = try generateUserParamsGlsl(allocator, .fs, opts.material_name, params);
        if (vertex_overridden) {
            vs_params_glsl = try generateUserParamsGlsl(allocator, .vs, opts.material_name, params);
        }
    }
    defer if (fs_params_glsl) |g| allocator.free(g);
    defer if (vs_params_glsl) |g| allocator.free(g);

    // Single pass: skip content lines of overridden hooks, splice snippet
    // bodies (and generated params GLSL) after their marker lines.
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(allocator);

    var line_idx: usize = 0;
    while (line_idx < template_lines.len) : (line_idx += 1) {
        const line = template_lines[line_idx];

        // Inside an overridden hook block: drop the template content lines
        // (empty default blocks keep nothing; identity holds for un-hooked
        // merges because nothing is skipped and nothing is spliced).
        var skip_content = false;
        for (template_hooks) |th| {
            var overridden = false;
            if (params.len > 0 and std.mem.eql(u8, th.name, "decls")) {
                // Each stage's decls block is replaced by its stage's
                // generated block (vs only when the vertex hook runs).
                overridden = switch (th.stage) {
                    .fs => true,
                    .vs => vertex_overridden,
                };
            }
            for (snippet_hooks) |sh| {
                if (std.mem.eql(u8, sh.name, th.name)) overridden = true;
            }
            if (overridden and line_idx >= th.content_start and line_idx < th.content_end) skip_content = true;
        }
        if (skip_content) continue;

        try out.appendSlice(allocator, line.text);
        try out.append(allocator, '\n');

        if (hookNameOnLine(line.text)) |name| {
            for (snippet_hooks) |sh| {
                if (std.mem.eql(u8, sh.name, name) and sh.body.len > 0) {
                    try out.appendSlice(allocator, sh.body);
                    if (sh.body[sh.body.len - 1] != '\n') try out.append(allocator, '\n');
                }
            }
            if (std.mem.eql(u8, name, "decls")) {
                // Match the stage of THIS decls marker: hooks carry their
                // stage, so find the block starting at this line.
                for (template_hooks) |th| {
                    if (!std.mem.eql(u8, th.name, "decls")) continue;
                    if (th.content_start - 1 != line_idx) continue;
                    const gen: ?[]u8 = switch (th.stage) {
                        .fs => fs_params_glsl,
                        .vs => vs_params_glsl,
                    };
                    if (gen) |g| try out.appendSlice(allocator, g);
                }
            }
        }
    }

    return .{ .glsl = try out.toOwnedSlice(allocator), .params = try dupParams(allocator, params) };
}

fn dupParams(allocator: std.mem.Allocator, params: []const Param) ![]Param {
    const out = try allocator.alloc(Param, params.len);
    @memcpy(out, params);
    return out;
}
