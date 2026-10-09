//! Host CLI for the test-registry staleness gate. Invoked from build.zig
//! as a native executable step (NOT part of the agate library): compares
//! the expected registry bytes (a WriteFiles output, fed as --expected)
//! against the tracked src/tests.zig (--actual) and fails loudly with the
//! regeneration command when they differ.
//!
//! Usage:
//!   check_test_registry --expected <expected_tests.zig> --actual <src/tests.zig>
//!
//! Exit 0 when the registry is fresh; otherwise prints the actionable
//! message to stderr and exits 1. Never writes to the source tree (a test
//! run must not dirty the working tree): regeneration stays an explicit
//! `zig build update-tests`.

const std = @import("std");

fn fail(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("test registry check error: " ++ fmt ++ "\n", args);
    std.process.exit(2);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const cwd = std.Io.Dir.cwd();
    const allocator = init.arena.allocator();

    var expected_path: []const u8 = "";
    var actual_path: []const u8 = "";
    var it = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer it.deinit();
    _ = it.next(); // argv0
    while (it.next()) |arg| {
        const value = it.next() orelse fail("missing value for '{s}'", .{arg});
        if (std.mem.eql(u8, arg, "--expected")) expected_path = value;
        if (std.mem.eql(u8, arg, "--actual")) actual_path = value;
    }
    if (expected_path.len == 0) fail("--expected is required", .{});
    if (actual_path.len == 0) fail("--actual is required", .{});

    const expected = cwd.readFileAlloc(io, expected_path, allocator, .limited(20 << 20)) catch |e|
        fail("cannot read expected registry '{s}': {s}", .{ expected_path, @errorName(e) });
    const actual = cwd.readFileAlloc(io, actual_path, allocator, .limited(20 << 20)) catch |e|
        fail("cannot read test registry '{s}': {s}", .{ actual_path, @errorName(e) });

    if (std.mem.eql(u8, expected, actual)) return;

    std.debug.print("src/tests.zig is stale: run `zig build update-tests`\n", .{});
    std.process.exit(1);
}
