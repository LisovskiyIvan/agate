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
