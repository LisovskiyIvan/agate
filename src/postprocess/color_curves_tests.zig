const std = @import("std");
const color_curves = @import("color_curves.zig");
const applyGrade = color_curves.applyGrade;

test "color grade identity and shadows" {
    const mid = [3]f32{ 0.5, 0.5, 0.5 };
    const zero = [3]f32{ 0.0, 0.0, 0.0 };
    const id = applyGrade(mid, zero, zero, zero);
    try std.testing.expectApproxEqAbs(mid[0], id[0], 1e-6);
    try std.testing.expectApproxEqAbs(mid[1], id[1], 1e-6);
    try std.testing.expectApproxEqAbs(mid[2], id[2], 1e-6);
    // Shadows lift applies fully to black, not at all to white.
    const lifted = applyGrade(zero, .{ 0.2, 0.1, 0.0 }, zero, zero);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), lifted[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), lifted[1], 1e-6);
    const white = applyGrade(.{ 1.0, 1.0, 1.0 }, .{ 0.2, 0.2, 0.2 }, zero, zero);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), white[0], 1e-6);
    // Highlights lift applies fully to white, not at all to black.
    const hl = applyGrade(.{ 1.0, 1.0, 1.0 }, zero, zero, .{ 0.0, 0.3, 0.0 });
    try std.testing.expectApproxEqAbs(@as(f32, 1.3), hl[1], 1e-6);
    const hl_dark = applyGrade(zero, zero, zero, .{ 0.0, 0.3, 0.0 });
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), hl_dark[1], 1e-6);
}
