const std = @import("std");
const morph_gpu = @import("morph_gpu.zig");

test "supported reports false without an sg context" {
    // Headless unit-test environment has no sokol context: the capability
    // gate must fail closed (never claim RGBA32F support it cannot verify).
    try std.testing.expect(!morph_gpu.supported());
}
