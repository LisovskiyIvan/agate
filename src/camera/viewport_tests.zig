const std = @import("std");
const viewport_mod = @import("viewport.zig");
const Viewport = viewport_mod.Viewport;

test "Viewport toPixelRect and aspect" {
    const vp = Viewport{ .x = 0.5, .y = 0.25, .width = 0.5, .height = 0.75 };
    const rect = vp.toPixelRect(1920, 1080);
    try std.testing.expectEqual(@as(i32, 960), rect.x);
    try std.testing.expectEqual(@as(i32, 270), rect.y);
    try std.testing.expectEqual(@as(i32, 960), rect.width);
    try std.testing.expectEqual(@as(i32, 810), rect.height);
    try std.testing.expectApproxEqAbs(@as(f32, 960.0 / 810.0), rect.aspect(), 1e-5);
}
