const std = @import("std");
const sg = @import("sokol").gfx;
const gpu_thread = @import("../gpu_thread.zig");
const ParticleSystem = @import("system.zig").ParticleSystem;

fn expectDeferredInit() !void {
    const ps = try ParticleSystem.init(std.testing.allocator, "headless", 8);
    defer std.testing.allocator.destroy(ps);
    defer ps.deinit();
    try std.testing.expectEqual(sg.Buffer{}, ps.instance_buffer);
    try std.testing.expect(ps.instance_buffer_pending);
    ps.emitOne();
    try std.testing.expectEqual(@as(usize, 1), ps.active_count);
}

test "headless particles defer GPU creation on an unregistered CPU thread" {
    try std.testing.expect(!sg.isvalid());
    gpu_thread.resetContextThreadForTest();
    defer gpu_thread.resetContextThreadForTest();
    try std.testing.expect(gpu_thread.isOnContextThread());
    try expectDeferredInit();
}

test "headless particles defer GPU creation even on the marked owner" {
    try std.testing.expect(!sg.isvalid());
    gpu_thread.markContextThread();
    defer gpu_thread.resetContextThreadForTest();
    try std.testing.expect(gpu_thread.isOnContextThread());
    try expectDeferredInit();
}
