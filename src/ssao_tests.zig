const std = @import("std");
const ssao = @import("ssao.zig");
const SSAOOptions = ssao.SSAOOptions;

test "ssao options clamping" {
    const custom = (SSAOOptions{ .sample_count = 100 }).clamped();
    try std.testing.expectEqual(@as(u32, 32), custom.sample_count);

    const low = (SSAOOptions{ .sample_count = 1 }).clamped();
    try std.testing.expectEqual(@as(u32, 4), low.sample_count);
}
