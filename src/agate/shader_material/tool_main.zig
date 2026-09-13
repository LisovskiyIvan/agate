//! Host CLI for the build-time shader material hook merger. Invoked from
//! build.zig as a native executable step (NOT part of the agate library).
//!
//! Usage:
//!   merge_shader_material --template <file.glsl> --snippet <file.glsl>
//!                         --out-glsl <file.glsl> --out-params <file.zig>
//!                         --base <standard|pbr> --name <material_name>
//!
//! Outputs:
//!   out-glsl   merged GLSL fed to sokol-shdc
//!   out-params generated Zig file with the declarative param table:
//!              `pub const params = [_]Param{...}` (Param defined locally
//!              with shape-compatible fields; the generated registry module
//!              converts them into the runtime registry type at comptime).

const std = @import("std");
const merge = @import("merge.zig");

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("shader-material merge error: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}

const Args = struct {
    template: []const u8 = "",
    snippet: []const u8 = "",
    out_glsl: []const u8 = "",
    out_params: []const u8 = "",
    base: []const u8 = "standard",
    name: []const u8 = "",
};

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const cwd = std.Io.Dir.cwd();
    const allocator = init.arena.allocator();

    var parsed = Args{};
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer it.deinit();
    _ = it.next(); // argv0
    while (it.next()) |arg| {
        const value = it.next() orelse fail("missing value for '{s}'", .{arg});
        if (std.mem.eql(u8, arg, "--template")) parsed.template = value;
        if (std.mem.eql(u8, arg, "--snippet")) parsed.snippet = value;
        if (std.mem.eql(u8, arg, "--out-glsl")) parsed.out_glsl = value;
        if (std.mem.eql(u8, arg, "--out-params")) parsed.out_params = value;
        if (std.mem.eql(u8, arg, "--base")) parsed.base = value;
        if (std.mem.eql(u8, arg, "--name")) parsed.name = value;
    }
    if (parsed.template.len == 0) fail("--template is required", .{});
    if (parsed.snippet.len == 0) fail("--snippet is required", .{});
    if (parsed.out_glsl.len == 0) fail("--out-glsl is required", .{});
    if (parsed.out_params.len == 0) fail("--out-params is required", .{});
    if (parsed.name.len == 0) fail("--name is required", .{});

    const template = cwd.readFileAlloc(io, parsed.template, allocator, .limited(4 << 20)) catch |e|
        fail("cannot read template '{s}': {s}", .{ parsed.template, @errorName(e) });
    const snippet = cwd.readFileAlloc(io, parsed.snippet, allocator, .limited(4 << 20)) catch |e|
        fail("cannot read snippet '{s}': {s}", .{ parsed.snippet, @errorName(e) });

    const res = merge.merge(allocator, .{
        .template = template,
        .snippet = snippet,
        .base = parsed.base,
        .material_name = parsed.name,
    }) catch |e| fail("material '{s}': {s}", .{ parsed.name, @errorName(e) });

    cwd.writeFile(io, .{ .sub_path = parsed.out_glsl, .data = res.glsl }) catch |e|
        fail("cannot write '{s}': {s}", .{ parsed.out_glsl, @errorName(e) });

    const params_src = try generateParamsZig(allocator, parsed.name, res.params);
    cwd.writeFile(io, .{ .sub_path = parsed.out_params, .data = params_src }) catch |e|
        fail("cannot write '{s}': {s}", .{ parsed.out_params, @errorName(e) });
}

fn generateParamsZig(allocator: std.mem.Allocator, name: []const u8, params: []const merge.Param) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    try out.print(allocator, "// GENERATED for shader material '{s}' by merge_shader_material. Do not edit.\n", .{name});
    try out.appendSlice(allocator,
        \\// Declarative user-uniform table. The registry module converts this
        \\// shape-compatible struct into the runtime registry Param at comptime.
        \\pub const Param = struct {
        \\    name: []const u8,
        \\    offset: u8,
        \\    comps: u8,
        \\    default: [4]f32 = .{ 0, 0, 0, 0 },
        \\};
        \\pub const params = [_]Param{
        \\
    );
    for (params) |p| {
        try out.print(allocator, "    .{{ .name = \"{s}\", .offset = {d}, .comps = {d}, .default = .{{ {d:.6}, {d:.6}, {d:.6}, {d:.6} }} }},\n", .{
            p.name, p.offset, p.comps, p.default[0], p.default[1], p.default[2], p.default[3],
        });
    }
    try out.appendSlice(allocator, "};\n");
    return out.items;
}
