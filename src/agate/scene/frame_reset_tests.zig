const std = @import("std");
const Scene = @import("../scene.zig").Scene;

test "resetTaa publishes coalesced intent without mutating render-owned history" {
    const alloc = std.testing.allocator;
    var scene = @import("../testing.zig").testScene(alloc);
    defer scene.lights.deinit(alloc);
    defer scene.cameras.deinit(alloc);
    defer scene.draws.deinit(alloc);
    defer scene.gpu_retire.deinit(alloc);
    defer scene.profiler.deinit();
    scene.postfx.taa_explicit_reset = false;

    const Producer = struct {
        fn request(target: *Scene) void {
            target.resetTaa();
            target.resetTaa();
        }
    };
    const worker = try std.Thread.spawn(.{}, Producer.request, .{&scene});
    worker.join();

    try std.testing.expect(!scene.postfx.taa_explicit_reset);
    try std.testing.expect(scene.taa_reset_requested.swap(false, .acq_rel));
    try std.testing.expect(!scene.taa_reset_requested.swap(false, .acq_rel));
    scene.resetTaa();
    try std.testing.expect(scene.taa_reset_requested.swap(false, .acq_rel));
}
