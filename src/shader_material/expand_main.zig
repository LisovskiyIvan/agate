//! Host CLI for build-time shader include expansion. Invoked from
//! build.zig as a native executable step (NOT part of the agate library),
//! mirroring shader_material/tool_main.zig.
//!
//! Usage:
//!   expand_shader_includes --root <shaders-dir> --input <file.glsl>
//!                          --output <file.glsl> --depout <file.d>
//!                          [--include-root <tracked-dir>]
//!
//! Reads the input shader, expands `// @include "relative/path.glsl"`
//! directives (resolved against --root) via include.zig, writes the
//! result. The output feeds sokol-shdc (which cannot resolve includes
//! itself). Any expansion error fails the build loudly.
//!
//! `--depout` receives a gcc-style depfile listing every file read during
//! expansion. This is what makes chunk edits invalidate the cached
//! expansion: the build system folds depfile entries into the step's input
//! set, so a chunk-only edit reruns the expansion on the next build (the
//! original silent-stale bug: without tracking, a chunk edit left the
//! shader with the OLD chunk — `touch` cannot help, the cache is content
//! addressed).
//!
//! `--include-root` is accepted (and its value ignored): build.zig passes
//! the chunk directory through it so the directory enters the build graph
//! as a declared input of this Run step. The tool itself never reads it.

const std = @import("std");
const include = @import("include.zig");

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("shader include expansion error: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}

const Ctx = struct {
    io: std.Io,
    cwd: *const std.Io.Dir,
    allocator: std.mem.Allocator,
    /// Every file path read through the FileSystem callback (owned dupes).
    deps: std.ArrayListUnmanaged([]const u8) = .empty,
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const cwd = std.Io.Dir.cwd();
    const allocator = init.arena.allocator();

    var root: []const u8 = "";
    var input: []const u8 = "";
    var output: []const u8 = "";
    var depout: []const u8 = "";
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer it.deinit();
    _ = it.next(); // argv0
    while (it.next()) |arg| {
        const value = it.next() orelse fail("missing value for '{s}'", .{arg});
        if (std.mem.eql(u8, arg, "--root")) root = value;
        if (std.mem.eql(u8, arg, "--input")) input = value;
        if (std.mem.eql(u8, arg, "--output")) output = value;
        if (std.mem.eql(u8, arg, "--depout")) depout = value;
        // Build-graph tracking only; the value is deliberately unused.
        if (std.mem.eql(u8, arg, "--include-root")) continue;
    }
    if (root.len == 0) fail("--root is required", .{});
    if (input.len == 0) fail("--input is required", .{});
    if (output.len == 0) fail("--output is required", .{});

    const src = cwd.readFileAlloc(io, input, allocator, .limited(4 << 20)) catch |e|
        fail("cannot read input '{s}': {s}", .{ input, @errorName(e) });

    var ctx = Ctx{ .io = io, .cwd = &cwd, .allocator = allocator };
    const fs = include.FileSystem{ .ctx = @ptrCast(&ctx), .readFn = readReal };
    const out = include.expand(allocator, src, input, root, fs) catch |e|
        fail("cannot expand '{s}': {s}", .{ input, @errorName(e) });

    cwd.writeFile(io, .{ .sub_path = output, .data = out }) catch |e|
        fail("cannot write '{s}': {s}", .{ output, @errorName(e) });

    if (depout.len != 0) writeDepfile(&ctx, &cwd, io, depout, output);
}

/// gcc-style depfile: `<output>: <dep> <dep> ...`. Dependencies are the
/// absolute include paths recorded by `readReal`; the target is the output
/// path exactly as the step passed it (same cwd, so it resolves).
fn writeDepfile(ctx: *Ctx, cwd: *const std.Io.Dir, io: std.Io, depout: []const u8, output: []const u8) void {
    var buf: std.Io.Writer.Allocating = .init(ctx.allocator);
    buf.writer.print("{s}:", .{output}) catch fail("depfile OOM", .{});
    for (ctx.deps.items) |dep| {
        buf.writer.print(" {s}", .{escapeSpaces(ctx.allocator, dep)}) catch fail("depfile OOM", .{});
    }
    buf.writer.writeAll("\n") catch fail("depfile OOM", .{});
    var file = cwd.createFile(io, depout, .{}) catch |e|
        fail("cannot create depfile '{s}': {s}", .{ depout, @errorName(e) });
    defer file.close(io);
    file.writeStreamingAll(io, buf.written()) catch |e|
        fail("cannot write depfile '{s}': {s}", .{ depout, @errorName(e) });
}

fn escapeSpaces(allocator: std.mem.Allocator, s: []const u8) []const u8 {
    if (std.mem.indexOfScalar(u8, s, ' ') == null) return s;
    var out = allocator.alloc(u8, s.len + 8) catch fail("depfile OOM", .{});
    var n: usize = 0;
    for (s) |c| {
        if (c == ' ') {
            out[n] = '\\';
            n += 1;
        }
        out[n] = c;
        n += 1;
    }
    return out[0..n];
}

fn readReal(ctx: *anyopaque, allocator: std.mem.Allocator, path: []const u8) include.FileSystem.ReadError![]u8 {
    const c: *Ctx = @ptrCast(@alignCast(ctx));
    // Record the dependency before reading so the depfile covers it even
    // when the read itself fails (build then aborts anyway, but the list
    // stays truthful).
    c.deps.append(c.allocator, c.allocator.dupe(u8, path) catch return error.OutOfMemory) catch
        return error.OutOfMemory;
    return c.cwd.readFileAlloc(c.io, path, allocator, .limited(4 << 20)) catch |e| switch (e) {
        error.FileNotFound => return error.NotFound,
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.Io,
    };
}
