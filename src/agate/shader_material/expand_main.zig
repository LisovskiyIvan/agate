//! Host CLI for build-time shader include expansion. Invoked from
//! build.zig as a native executable step (NOT part of the agate library),
//! mirroring shader_material/tool_main.zig.
//!
//! Usage:
//!   expand_shader_includes --root <shaders-dir> --input <file.glsl>
//!                          --output <file.glsl>
//!
//! Reads the input shader, expands `// @include "relative/path.glsl"`
//! directives (resolved against --root) via include.zig, writes the
//! result. The output feeds sokol-shdc (which cannot resolve includes
//! itself). Any expansion error fails the build loudly.

const std = @import("std");
const include = @import("include.zig");

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("shader include expansion error: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const cwd = std.Io.Dir.cwd();
    const allocator = init.arena.allocator();

    var root: []const u8 = "";
    var input: []const u8 = "";
    var output: []const u8 = "";
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer it.deinit();
    _ = it.next(); // argv0
    while (it.next()) |arg| {
        const value = it.next() orelse fail("missing value for '{s}'", .{arg});
        if (std.mem.eql(u8, arg, "--root")) root = value;
        if (std.mem.eql(u8, arg, "--input")) input = value;
        if (std.mem.eql(u8, arg, "--output")) output = value;
    }
    if (root.len == 0) fail("--root is required", .{});
    if (input.len == 0) fail("--input is required", .{});
    if (output.len == 0) fail("--output is required", .{});

    const src = cwd.readFileAlloc(io, input, allocator, .limited(4 << 20)) catch |e|
        fail("cannot read input '{s}': {s}", .{ input, @errorName(e) });

    var ctx = Ctx{ .io = io, .cwd = &cwd };
    const fs = include.FileSystem{ .ctx = @ptrCast(&ctx), .readFn = readReal };
    const out = include.expand(allocator, src, input, root, fs) catch |e|
        fail("cannot expand '{s}': {s}", .{ input, @errorName(e) });

    cwd.writeFile(io, .{ .sub_path = output, .data = out }) catch |e|
        fail("cannot write '{s}': {s}", .{ output, @errorName(e) });
}

const Ctx = struct {
    io: std.Io,
    cwd: *const std.Io.Dir,
};

fn readReal(ctx: *anyopaque, allocator: std.mem.Allocator, path: []const u8) include.FileSystem.ReadError![]u8 {
    const c: *const Ctx = @ptrCast(@alignCast(ctx));
    return c.cwd.readFileAlloc(c.io, path, allocator, .limited(4 << 20)) catch |e| switch (e) {
        error.FileNotFound => return error.NotFound,
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.Io,
    };
}
