const std = @import("std");
const dp = @import("depth_pyramid.zig");
const depthPyramidMipSize = dp.depthPyramidMipSize;
const computeMipCount = dp.computeMipCount;
const reduceConservativeDepth = dp.reduceConservativeDepth;

test "depth pyramid mip sizing halving down to 1x1" {
    const s0 = depthPyramidMipSize(1920, 1080, 0);
    try std.testing.expectEqual(@as(i32, 960), s0.w);
    try std.testing.expectEqual(@as(i32, 540), s0.h);

    const s1 = depthPyramidMipSize(1920, 1080, 1);
    try std.testing.expectEqual(@as(i32, 480), s1.w);
    try std.testing.expectEqual(@as(i32, 270), s1.h);

    const s7 = depthPyramidMipSize(1920, 1080, 7);
    try std.testing.expectEqual(@as(i32, 7), s7.w);
    try std.testing.expectEqual(@as(i32, 4), s7.h);

    const count = computeMipCount(1920, 1080);
    try std.testing.expectEqual(@as(u32, 8), count);
}

test "conservative depth reduction takes maximum depth" {
    const d = reduceConservativeDepth(0.3, 0.7, 0.5, 0.2);
    try std.testing.expectEqual(@as(f32, 0.7), d);
}
