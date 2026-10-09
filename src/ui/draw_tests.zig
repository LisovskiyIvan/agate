const std = @import("std");
const draw = @import("draw.zig");
const lineCorners = draw.lineCorners;
const batchUploadBytes = draw.batchUploadBytes;
const UIVertex = draw.UIVertex;

test "UICanvas lineCorners" {
    // Horizontal segment: thickness extends along +/-Y.
    const h = lineCorners(0, 0, 10, 0, 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), h[0][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), h[0][1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), h[1][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), h[1][1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), h[2][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), h[2][1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), h[3][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), h[3][1], 1e-5);

    // Vertical segment: thickness extends along +/-X.
    const v = lineCorners(0, 0, 0, 8, 4.0);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), v[0][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), v[0][1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), v[1][0], 1e-5);
    // p1-right = end + half-normal; the normal is (-1, 0) here, mirroring
    // the horizontal case (p1-right keeps the +Y side there).
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), v[2][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), v[2][1], 1e-5);

    // Quad width matches thickness (diagonal case).
    const d = lineCorners(0, 0, 3, 4, 2.0);
    const w0x = d[1][0] - d[0][0];
    const w0y = d[1][1] - d[0][1];
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), @sqrt(w0x * w0x + w0y * w0y), 1e-5);

    // Degenerate segment collapses to the endpoints.
    const z = lineCorners(5, 5, 5, 5, 2.0);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), z[0][0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), z[2][0], 1e-5);
}

test "batchUploadBytes keeps the u16 vertex cap in usize arithmetic" {
    // 65535 vertices x 48 B ~= 3.1 MB: u16 arithmetic would already trap at
    // 1366 vertices, so the helper must compute in usize in every build mode.
    const verts: usize = std.math.maxInt(u16);
    const idx: usize = std.math.maxInt(u16);
    const expected = verts * @sizeOf(UIVertex) + idx * @sizeOf(u16);
    try std.testing.expectEqual(expected, batchUploadBytes(verts, idx));
}
