const std = @import("std");
const depth_pyramid_mod = @import("depth_pyramid_pass.zig");
const DepthPyramidPass = depth_pyramid_mod.DepthPyramidPass;
const pp = @import("../postprocess.zig");

test "depth pyramid pass fail-closes headless with no state touched" {
    var pass = DepthPyramidPass{};
    // Render with empty depth view returns empty view without panic or sg calls
    const res = pass.render(.{}, 1920, 1080);
    try std.testing.expectEqual(@as(u32, 0), res.id);

    // Render with 0 dimensions returns empty view
    const res2 = pass.render(.{ .id = 42 }, 0, 0);
    try std.testing.expectEqual(@as(u32, 0), res2.id);

    // Out of range mip query returns empty view
    const m = pass.mipView(10);
    try std.testing.expectEqual(@as(u32, 0), m.id);

    // deinit on unallocated pass is completely safe
    pass.deinit();
}

test "depth pyramid mip count and halving bounds" {
    try std.testing.expectEqual(@as(u32, 0), pp.computeMipCount(0, 0));
    try std.testing.expectEqual(@as(u32, 8), pp.computeMipCount(1920, 1080));
    try std.testing.expectEqual(@as(u32, 7), pp.computeMipCount(128, 64));

    const s0 = pp.depthPyramidMipSize(800, 600, 0);
    try std.testing.expectEqual(@as(i32, 400), s0.w);
    try std.testing.expectEqual(@as(i32, 300), s0.h);
}
