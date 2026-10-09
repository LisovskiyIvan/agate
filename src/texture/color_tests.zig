const std = @import("std");
const color_mod = @import("color.zig");
const srgbToLinearU8 = color_mod.srgbToLinearU8;
const convertSrgbToLinearInPlace = color_mod.convertSrgbToLinearInPlace;

test "srgbToLinearU8 hits exact golden values and the endpoints" {
    // Computed with the IEC 61966-2-1 formula, rounded to nearest.
    try std.testing.expectEqual(@as(u8, 0), srgbToLinearU8(0));
    try std.testing.expectEqual(@as(u8, 255), srgbToLinearU8(255));
    try std.testing.expectEqual(@as(u8, 55), srgbToLinearU8(128));
    try std.testing.expectEqual(@as(u8, 13), srgbToLinearU8(64));
    try std.testing.expectEqual(@as(u8, 2), srgbToLinearU8(25));
    try std.testing.expectEqual(@as(u8, 147), srgbToLinearU8(200));
    // Monotonic non-decreasing over the whole table.
    var v: usize = 1;
    while (v < 256) : (v += 1) {
        try std.testing.expect(srgbToLinearU8(@intCast(v)) >= srgbToLinearU8(@intCast(v - 1)));
    }
}

test "convertSrgbToLinearInPlace converts RGB lanes and preserves alpha" {
    var pixels = [_]u8{ 200, 128, 0, 42, 255, 0, 25, 7 };
    convertSrgbToLinearInPlace(&pixels);
    try std.testing.expectEqual(@as(u8, srgbToLinearU8(200)), pixels[0]);
    try std.testing.expectEqual(@as(u8, srgbToLinearU8(128)), pixels[1]);
    try std.testing.expectEqual(@as(u8, 0), pixels[2]);
    try std.testing.expectEqual(@as(u8, 42), pixels[3]); // alpha untouched
    try std.testing.expectEqual(@as(u8, 255), pixels[4]);
    try std.testing.expectEqual(@as(u8, 0), pixels[5]);
    try std.testing.expectEqual(@as(u8, srgbToLinearU8(25)), pixels[6]);
    try std.testing.expectEqual(@as(u8, 7), pixels[7]); // alpha untouched
}
