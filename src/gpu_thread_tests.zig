const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const gpu_thread = @import("gpu_thread.zig");

test "unmarked headless is a CPU-phase inline decision, not GPU authorization" {
    const saved = gpu_thread.context_thread_id;
    defer gpu_thread.context_thread_id = saved;
    gpu_thread.context_thread_id = null;

    const me = std.Thread.getCurrentId();
    try std.testing.expect(gpu_thread.matches(null, me, false));
    try std.testing.expect(!gpu_thread.matches(null, me, true));
    if (!sg.isvalid()) try std.testing.expect(gpu_thread.isOnContextThread());
}

test "marked owner passes, foreign fails even headless" {
    const saved = gpu_thread.context_thread_id;
    defer gpu_thread.context_thread_id = saved;

    gpu_thread.markContextThread();
    const owner = gpu_thread.context_thread_id.?;
    const me = std.Thread.getCurrentId();
    try std.testing.expect(gpu_thread.matches(owner, me, sg.isvalid()));
    try std.testing.expect(gpu_thread.isOnContextThread());
    const Probe = struct {
        fn run(out: *bool) void {
            out.* = gpu_thread.isOnContextThread();
        }
    };
    var foreign: bool = true;
    const t = try std.Thread.spawn(.{}, Probe.run, .{&foreign});
    t.join();
    try std.testing.expect(!foreign);
}

test "pure policy covers registered, unregistered, and foreign live" {
    const me = std.Thread.getCurrentId();
    const Probe = struct {
        fn run(out: *std.Thread.Id) void {
            out.* = std.Thread.getCurrentId();
        }
    };
    var child: std.Thread.Id = me;
    const t = try std.Thread.spawn(.{}, Probe.run, .{&child});
    t.join();
    try std.testing.expect(gpu_thread.matches(me, me, false));
    try std.testing.expect(gpu_thread.matches(me, me, true));
    try std.testing.expect(!gpu_thread.matches(me, child, false));
    try std.testing.expect(!gpu_thread.matches(me, child, true));
    try std.testing.expect(gpu_thread.matches(child, child, true));
    try std.testing.expect(gpu_thread.matches(null, me, false));
    try std.testing.expect(!gpu_thread.matches(null, me, true));
}
