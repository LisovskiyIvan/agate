const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const target_shape = @import("target_shape.zig");
const TargetShape = target_shape.TargetShape;

test "TargetShape equality and hashing" {
    const s1 = TargetShape{
        .color_format = .RGBA16F,
        .depth_format = .DEPTH,
        .stencil_format = .NONE,
        .sample_count = 1,
    };
    const s2 = TargetShape.init(.RGBA16F, .DEPTH, 1);
    try std.testing.expect(s1.eql(s2));
    try std.testing.expectEqual(s1.hash(), s2.hash());

    const s_msaa = TargetShape.init(.RGBA16F, .DEPTH, 4);
    try std.testing.expect(!s1.eql(s_msaa));
    try std.testing.expect(s1.hash() != s_msaa.hash());

    const s_stencil = TargetShape.init(.RGBA16F, .DEPTH_STENCIL, 1);
    try std.testing.expectEqual(sg.PixelFormat.DEPTH_STENCIL, s_stencil.stencil_format);
    try std.testing.expect(!s1.eql(s_stencil));
    try std.testing.expect(s1.hash() != s_stencil.hash());
}

test "TargetShape resolveEnvironment" {
    const s_default = TargetShape{
        .color_format = .DEFAULT,
        .depth_format = .DEFAULT,
        .stencil_format = .NONE,
        .sample_count = 2,
    };
    const resolved = s_default.resolveEnvironment();
    try std.testing.expectEqual(sg.PixelFormat.RGBA16F, resolved.color_format);
    try std.testing.expect(resolved.depth_format != .DEFAULT);
    try std.testing.expectEqual(@as(i32, 2), resolved.sample_count);
}
