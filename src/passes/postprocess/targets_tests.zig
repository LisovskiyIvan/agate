//! Target/TAA lifecycle tests exercise the unchanged PostProcessPass API.
//! All headless-safe: pure field reads plus fail-closed guards (no sg calls
//! without live handles).
const std = @import("std");
const testing = std.testing;
const sg = @import("sokol").gfx;
const PostProcessPass = @import("../postprocess_pass.zig").PostProcessPass;

test "taa ping-pong selects read slot and opposite write slot" {
    const read0 = sg.View{ .id = 11 };
    const read1 = sg.View{ .id = 12 };
    const att0 = sg.View{ .id = 21 };
    const att1 = sg.View{ .id = 22 };
    var pass = PostProcessPass{
        .taa_tex_views = .{ read0, read1 },
        .taa_att_views = .{ att0, att1 },
    };
    pass.taa_read = 0;
    try testing.expectEqual(read0.id, pass.taaReadView().id);
    try testing.expectEqual(att1.id, pass.taaWriteAttView().id);
    pass.taa_read = 1;
    try testing.expectEqual(read1.id, pass.taaReadView().id);
    try testing.expectEqual(att0.id, pass.taaWriteAttView().id);
}

test "depthSampleView prefers the 1x depth texture, else the resolve view" {
    const depth = sg.View{ .id = 31 };
    const resolve = sg.View{ .id = 32 };
    const with_depth = PostProcessPass{
        .offscreen_depth_tex_view = depth,
        .offscreen_resolve_tex_view = resolve,
    };
    try testing.expectEqual(depth.id, with_depth.depthSampleView().id);
    // MSAA shape: no depth texture view, resolve view is the placeholder.
    const msaa = PostProcessPass{
        .offscreen_resolve_tex_view = resolve,
    };
    try testing.expectEqual(resolve.id, msaa.depthSampleView().id);
}

test "taaReset clears history metadata headless without touching sg" {
    var pass = PostProcessPass{
        .taa_width = 64,
        .taa_height = 64,
        .taa_format = .RGBA16F,
        .taa_read = 1,
    };
    pass.taaReset();
    try testing.expectEqual(@as(i32, 0), pass.taa_width);
    try testing.expectEqual(@as(i32, 0), pass.taa_height);
    try testing.expectEqual(sg.PixelFormat.DEFAULT, pass.taa_format);
    try testing.expectEqual(@as(u8, 0), pass.taa_read);
    try testing.expectEqual(@as(u32, 0), pass.taaReadView().id);
    try testing.expect(!pass.taaAvailable());
}
