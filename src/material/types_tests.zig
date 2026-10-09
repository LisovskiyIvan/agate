const std = @import("std");
const types = @import("types.zig");
const UvTransform = types.UvTransform;

test "UV1 selection preserves transform matrix and unlit lane" {
    const uv = UvTransform{ .tex_coord = 1, .offset = .{ 0.25, -0.5 } };
    try std.testing.expect(!uv.isIdentity());
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 0, 1 }, &uv.matrixRows());
    try std.testing.expectEqualSlices(f32, &.{ 0.25, -0.5, 0, 1 }, &uv.offsetPacked());
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0 }, &UvTransform.identity.offsetPacked());
}
