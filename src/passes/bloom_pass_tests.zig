const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const bloom_mod = @import("bloom_pass.zig");
const BloomPass = bloom_mod.BloomPass;

test "bloom pass fail-closes headless with no state touched" {
    // Zero-initialized pass (never init'ed: no sg context headless) must
    // return an empty view before any sg.* call.
    var pass: BloomPass = .{};
    const empty = pass.render(.{}, 1.0, 5, 2.0, 1280, 720);
    try std.testing.expectEqual(@as(u32, 0), empty.id);
    // Empty source, degenerate size, and missing pipelines all fail closed
    // the same way (guard order: pipelines first, then source, then size).
    try std.testing.expectEqual(@as(u32, 0), pass.render(.{ .id = 9 }, 1.0, 5, 2.0, 1280, 720).id);
    try std.testing.expectEqual(@as(u32, 0), pass.render(.{ .id = 9 }, 1.0, 5, 2.0, 0, 720).id);
    // The sole-contract format is pinned headless (no sg calls).
    try std.testing.expectEqual(sg.PixelFormat.RGBA16F, BloomPass.bloomPixelFormat());
    // Base size untouched: no resize happened.
    try std.testing.expectEqual(@as(i32, 0), pass.base_width);
    try std.testing.expectEqual(@as(i32, 0), pass.base_height);
}

test "sanitizeBloomRadius clamps finite, neutral on non-finite" {
    // Pure CPU (no sg calls): identity inside the range, 1.0 preserved.
    try std.testing.expectEqual(@as(f32, 1.0), BloomPass.sanitizeBloomRadius(1.0));
    try std.testing.expectEqual(@as(f32, 2.0), BloomPass.sanitizeBloomRadius(2.0));
    // 0 is valid: every upsample tap lands on the center (point upscale).
    try std.testing.expectEqual(@as(f32, 0.0), BloomPass.sanitizeBloomRadius(0.0));
    // Finite clamps at both ends.
    try std.testing.expectEqual(@as(f32, 0.0), BloomPass.sanitizeBloomRadius(-3.0));
    try std.testing.expectEqual(@as(f32, 16.0), BloomPass.sanitizeBloomRadius(99.0));
    // Non-finite falls back to the neutral single-texel tent.
    try std.testing.expectEqual(@as(f32, 1.0), BloomPass.sanitizeBloomRadius(std.math.nan(f32)));
    try std.testing.expectEqual(@as(f32, 1.0), BloomPass.sanitizeBloomRadius(std.math.inf(f32)));
}
