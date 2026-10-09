const std = @import("std");
const queues_mod = @import("render_queue.zig");
const refraction = @import("refraction.zig");
const needsCapture = refraction.needsCapture;

test "refraction empty queues need no capture" {
    const queues = queues_mod.RenderQueues{};
    try std.testing.expect(!needsCapture(&queues));
}
