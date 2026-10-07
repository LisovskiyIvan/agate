const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const BoundingBox = math.BoundingBox;
const HiZBuffer = @import("hiz_buffer.zig").HiZBuffer;
const SoftwareRasterizer = @import("rasterizer.zig").SoftwareRasterizer;
const OcclusionCuller = @import("culler.zig").OcclusionCuller;

test "HiZBuffer: initialization and pyramid structure" {
    var hiz = HiZBuffer.init();
    try std.testing.expectEqual(@as(u32, 256), HiZBuffer.WIDTH);
    try std.testing.expectEqual(@as(u32, 128), HiZBuffer.HEIGHT);
    try std.testing.expectEqual(@as(usize, 9), HiZBuffer.NUM_MIPS);

    // Initial clear value should be 1.0
    try std.testing.expectEqual(@as(f32, 1.0), hiz.sampleLevel(0, 0, 0));
    try std.testing.expectEqual(@as(f32, 1.0), hiz.sampleLevel(0, 255, 127));

    // Write a pixel at (10, 10) with depth 0.25
    hiz.writePixel(10, 10, 0.25);
    try std.testing.expectEqual(@as(f32, 0.25), hiz.sampleLevel(0, 10, 10));

    // Writing a deeper value should NOT overwrite (depth test passes only if closer)
    hiz.writePixel(10, 10, 0.75);
    try std.testing.expectEqual(@as(f32, 0.25), hiz.sampleLevel(0, 10, 10));

    // Writing a closer value SHOULD overwrite
    hiz.writePixel(10, 10, 0.15);
    try std.testing.expectEqual(@as(f32, 0.15), hiz.sampleLevel(0, 10, 10));
}

test "HiZBuffer: pyramid conservative downsampling" {
    var hiz = HiZBuffer.init();

    // Fill an entire 2x2 block at (0, 0) in Level 0
    hiz.writePixel(0, 0, 0.2);
    hiz.writePixel(1, 0, 0.3);
    hiz.writePixel(0, 1, 0.5);
    hiz.writePixel(1, 1, 0.4);

    hiz.buildPyramid();

    // In Level 1, (0, 0) covers Level 0 [(0,0), (1,0), (0,1), (1,1)]
    // Conservative downsample must be max(0.2, 0.3, 0.5, 0.4) = 0.5
    const l1_val = hiz.sampleLevel(1, 0, 0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), l1_val, 0.0001);

    // Another block at (2, 0) has only 1 occluded pixel (0.1), others empty (1.0)
    hiz.writePixel(2, 0, 0.1);
    hiz.buildPyramid();
    const l1_part = hiz.sampleLevel(1, 1, 0);
    // Conservative max with empty (1.0) must remain 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), l1_part, 0.0001);
}

test "OcclusionCuller: wall occludes object behind it" {
    // Camera at (0, 0, 10) looking towards origin (0, 0, 0)
    const eye = Vec3.new(0.0, 0.0, 10.0);
    const target = Vec3.new(0.0, 0.0, 0.0);
    const up = Vec3.new(0.0, 1.0, 0.0);
    const view = Mat4.lookAt(eye, target, up);
    const proj = Mat4.perspective(60.0, 16.0 / 9.0, 0.1, 100.0);
    const view_proj = proj.mul(view);

    var culler = OcclusionCuller.init();
    culler.beginFrame(view_proj);

    // Large occluder wall at z = 5.0 (between camera at z=10 and origin at z=0)
    // Spans from x: -6..6, y: -6..6, z: 4.8..5.2
    const wall_box = BoundingBox.init(
        Vec3.new(-6.0, -6.0, 4.8),
        Vec3.new(6.0, 6.0, 5.2),
    );
    culler.rasterizeOccluderBox(wall_box, Mat4.identity);
    culler.endOccluders();

    try std.testing.expect(culler.occluder_count > 0);
    try std.testing.expect(culler.triangles_rasterized > 0);

    // Object A: BEHIND the wall at origin z = 0.0, size 2x2x2 (z: -1..1)
    const obj_behind = BoundingBox.init(
        Vec3.new(-1.0, -1.0, -1.0),
        Vec3.new(1.0, 1.0, 1.0),
    );
    const occluded_behind = culler.isOccluded(obj_behind);
    try std.testing.expect(occluded_behind);

    // Object B: IN FRONT of the wall at z = 7.5, size 1x1x1 (z: 7..8)
    const obj_in_front = BoundingBox.init(
        Vec3.new(-0.5, -0.5, 7.0),
        Vec3.new(0.5, 0.5, 8.0),
    );
    const occluded_in_front = culler.isOccluded(obj_in_front);
    try std.testing.expect(!occluded_in_front);

    // Object C: BESIDE the wall at x = 12.0 (outside wall span of x: -6..6)
    const obj_beside = BoundingBox.init(
        Vec3.new(11.0, -1.0, 0.0),
        Vec3.new(13.0, 1.0, 2.0),
    );
    const occluded_beside = culler.isOccluded(obj_beside);
    try std.testing.expect(!occluded_beside);
}

test "OcclusionCuller: partially hidden object is never falsely occluded" {
    const eye = Vec3.new(0.0, 0.0, 10.0);
    const target = Vec3.new(0.0, 0.0, 0.0);
    const up = Vec3.new(0.0, 1.0, 0.0);
    const view_proj = Mat4.perspective(60.0, 1.0, 0.1, 100.0).mul(Mat4.lookAt(eye, target, up));

    var culler = OcclusionCuller.init();
    culler.beginFrame(view_proj);

    // Wall covers left half of the view: x: -5..0, y: -5..5, z: 4.8..5.2
    const wall_left = BoundingBox.init(
        Vec3.new(-5.0, -5.0, 4.8),
        Vec3.new(0.0, 5.0, 5.2),
    );
    culler.rasterizeOccluderBox(wall_left, Mat4.identity);
    culler.endOccluders();

    // Object spans across x = 0 (from x: -1 to x: 2) behind wall at z = 0
    // Half is behind the wall, half is sticking out in the open!
    const obj_crossing = BoundingBox.init(
        Vec3.new(-1.0, -1.0, -1.0),
        Vec3.new(2.0, 1.0, 1.0),
    );

    // Must be VISIBLE (false) because it sticks out past the wall!
    const is_occluded = culler.isOccluded(obj_crossing);
    try std.testing.expect(!is_occluded);
}

test "OcclusionCuller: zero occluders fast path" {
    const view_proj = Mat4.identity;
    var culler = OcclusionCuller.init();
    culler.beginFrame(view_proj);
    culler.endOccluders();

    const any_box = BoundingBox.init(Vec3.new(-1, -1, -1), Vec3.new(1, 1, 1));
    // When no occluders submitted, isOccluded returns false immediately
    try std.testing.expect(!culler.isOccluded(any_box));
}

test "OcclusionCuller: performance benchmark 1000 AABB queries" {
    const eye = Vec3.new(0.0, 0.0, 15.0);
    const target = Vec3.new(0.0, 0.0, 0.0);
    const up = Vec3.new(0.0, 1.0, 0.0);
    const view_proj = Mat4.perspective(60.0, 1.0, 0.1, 100.0).mul(Mat4.lookAt(eye, target, up));

    var culler = OcclusionCuller.init();
    culler.beginFrame(view_proj);

    // Wall in the center
    const wall = BoundingBox.init(Vec3.new(-5, -5, 4.9), Vec3.new(5, 5, 5.1));
    culler.rasterizeOccluderBox(wall, Mat4.identity);
    culler.endOccluders();

    var occluded_count: usize = 0;
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        const fi = @as(f32, @floatFromInt(i));
        const offset_x = @mod(fi * 0.17, 12.0) - 6.0;
        const offset_y = @mod(fi * 0.23, 12.0) - 6.0;
        const box = BoundingBox.init(
            Vec3.new(offset_x - 0.5, offset_y - 0.5, -2.0),
            Vec3.new(offset_x + 0.5, offset_y + 0.5, -1.0),
        );
        if (culler.isOccluded(box)) {
            occluded_count += 1;
        }
    }

    // Since wall is [-5, 5], many queries inside [-5, 5] should be occluded
    try std.testing.expect(occluded_count > 200);
}
