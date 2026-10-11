const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const depthNormalFromGradients = @import("depth_normal.zig").depthNormalFromGradients;
const depthNormalFromNeighbors = @import("depth_normal.zig").depthNormalFromNeighbors;

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

test "depth normal preserves silhouette neighbors under camera rotation" {
    const eye = Vec3.new(3, -2, 5);
    const pixel: f32 = 0.001;
    for ([_]Vec3{ Vec3.new(0, 0, -1), Vec3.new(1, 0, 0), Vec3.new(0.4, 0.7, -0.3).normalize() }) |forward| {
        const base_right = forward.cross(Vec3.up).normalize();
        const base_up = base_right.cross(forward);
        for ([_]f32{ 0.0, 0.7 }) |roll| {
            const right = base_right.scale(@cos(roll)).add(base_up.scale(@sin(roll)));
            const up = base_up.scale(@cos(roll)).sub(base_right.scale(@sin(roll)));
            const vp = math.Mat4.mul(math.Mat4.perspective(60, 16.0 / 9.0, 0.1, 100), math.Mat4.lookAt(eye, eye.add(forward), up));
            for ([_]f32{ -1.0, 1.0 }) |side| {
                // A front-facing wall at depth 10 meets background depth 20.
                // All taps lie on adjacent screen rays. At 90-degree yaw,
                // world-Z differences can favor the background instead.
                const sx = side * 2.0 * pixel;
                const sy = side * 2.0 * pixel;
                const center = eye.add(forward.add(right.scale(sx)).add(up.scale(sy)).scale(10));
                const positive_depth: f32 = if (side > 0) 10 else 20;
                const negative_depth: f32 = if (side > 0) 20 else 10;
                const p_r = eye.add(forward.add(right.scale(sx + pixel)).add(up.scale(sy)).scale(positive_depth));
                const p_l = eye.add(forward.add(right.scale(sx - pixel)).add(up.scale(sy)).scale(negative_depth));
                const p_u = eye.add(forward.add(right.scale(sx)).add(up.scale(sy + pixel)).scale(positive_depth));
                const p_d = eye.add(forward.add(right.scale(sx)).add(up.scale(sy - pixel)).scale(negative_depth));
                const normal = depthNormalFromNeighbors(vp, center, p_r, p_l, p_u, p_d);
                try std.testing.expect(normal.dot(forward.scale(-1)) > 0.9999);
            }
        }
    }
}
