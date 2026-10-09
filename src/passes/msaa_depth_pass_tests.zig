const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;
const msaa_mod = @import("msaa_depth_pass.zig");
const MsaaDepthPass = msaa_mod.MsaaDepthPass;
const itemContributesDepth = msaa_mod.itemContributesDepth;
const RenderQueues = @import("../scene/render_queue.zig").RenderQueues;
const SceneStats = @import("../scene/stats.zig").SceneStats;

test "itemContributesDepth pins the prepass skip matrix" {
    // Opaque regular geometry: the prepass payload.
    try std.testing.expect(itemContributesDepth(false, false, false));
    // Transparent/decal write no depth in the main pass: nothing to mirror.
    try std.testing.expect(!itemContributesDepth(true, false, false));
    try std.testing.expect(!itemContributesDepth(false, true, false));
    // Hook materials need their custom vertex stage (v1 non-goal).
    try std.testing.expect(!itemContributesDepth(false, false, true));
    try std.testing.expect(!itemContributesDepth(true, true, true));
}

test "render is fail-closed without a GPU context" {
    // Headless (no sg context): zeroed pass draws nothing, stats clean.
    var pass = std.mem.zeroes(MsaaDepthPass);
    var queues = RenderQueues{};
    defer queues.deinit(std.testing.allocator);
    var stats = SceneStats{};
    pass.render(Mat4.identity, &queues, &.{}, &stats);
    try std.testing.expectEqual(@as(u32, 0), stats.draw_calls);
    try std.testing.expectEqual(@as(u32, 0), stats.main_draw_calls);
    try std.testing.expect(pass.depthTexView().id == 0);
}
