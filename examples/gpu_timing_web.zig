//! Web entry exports one C main only (no duplicate __main_argc_argv).
const std = @import("std");
const app = @import("gpu_timing.zig");

fn entryMain(_: c_int, _: ?[*]?[*:0]const u8) callconv(.c) c_int {
    app.startApp();
    return 0;
}

comptime {
    @export(&entryMain, .{ .name = "main" });
}

pub export fn sysctlbyname(_: [*:0]const u8, _: ?*anyopaque, _: ?*usize, _: ?*anyopaque, _: usize) callconv(.c) c_int {
    return -1;
}

pub const os = struct {
    pub const heap = struct {
        pub const page_allocator = std.heap.c_allocator;
    };
};
