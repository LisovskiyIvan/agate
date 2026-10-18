const std = @import("std");
const builtin = @import("builtin");
const compute = @import("compute.zig");

test "groupCount is ceiling division" {
    try std.testing.expectEqual(@as(usize, 0), compute.groupCount(0, 64));
    try std.testing.expectEqual(@as(usize, 1), compute.groupCount(1, 64));
    try std.testing.expectEqual(@as(usize, 1), compute.groupCount(64, 64));
    try std.testing.expectEqual(@as(usize, 2), compute.groupCount(65, 64));
    try std.testing.expectEqual(@as(usize, 16), compute.groupCount(1024, 64));
    try std.testing.expectEqual(@as(usize, 17), compute.groupCount(1025, 64));
    // Non-default workgroup sizes stay exact.
    try std.testing.expectEqual(@as(usize, 3), compute.groupCount(97, 40));
}

test "static backend matrix matches the sokol documentation" {
    try std.testing.expect(compute.backendSupportsCompute(.METAL_MACOS));
    try std.testing.expect(compute.backendSupportsCompute(.METAL_IOS));
    try std.testing.expect(compute.backendSupportsCompute(.METAL_SIMULATOR));
    try std.testing.expect(compute.backendSupportsCompute(.D3D11));
    try std.testing.expect(compute.backendSupportsCompute(.WGPU));
    // Desktop GL: 4.3+ everywhere except macOS (capped at 4.1).
    try std.testing.expectEqual(builtin.os.tag != .macos, compute.backendSupportsCompute(.GLCORE));
    try std.testing.expect(compute.backendSupportsCompute(.GLES3));
    try std.testing.expect(!compute.backendSupportsCompute(.VULKAN));
    try std.testing.expect(!compute.backendSupportsCompute(.DUMMY));
}
