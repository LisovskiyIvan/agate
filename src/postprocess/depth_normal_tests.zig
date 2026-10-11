const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const depthNormalFromGradients = @import("depth_normal.zig").depthNormalFromGradients;

test "depth normal keeps vertical faces vertical at every pixel footprint" {
    // At common resolutions a pixel spans millimetres/centimetres, so the
    // squared cross product is far below the old absolute 1e-6 cutoff.
    // Falling back to up made SSR incorrectly reflect on the whole wall.
    for ([_]f32{ 0.1, 0.01, 0.001, 0.0001 }) |pixel_size| {
        const normal = depthNormalFromGradients(Vec3.new(pixel_size, 0, 0), Vec3.new(0, pixel_size, 0));
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), normal.y, 1e-6);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), normal.z, 1e-6);
        try std.testing.expect(normal.y < 0.45); // SSR's horizontal-surface gate
        const floor = depthNormalFromGradients(Vec3.new(pixel_size, 0, 0), Vec3.new(0, 0, -pixel_size));
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), floor.y, 1e-6);
    }
}

test "depth normal rejects degenerate gradients without inventing a floor" {
    try std.testing.expectEqual(Vec3.zero, depthNormalFromGradients(Vec3.zero, Vec3.up));
    try std.testing.expectEqual(Vec3.zero, depthNormalFromGradients(Vec3.up, Vec3.up.scale(2.0)));
}
