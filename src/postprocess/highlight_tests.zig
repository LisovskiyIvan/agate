const std = @import("std");
const options = @import("options.zig");
const glow = @import("glow.zig");
const highlight = @import("highlight.zig");
const highlightActive = highlight.highlightActive;
const highlightParams = highlight.highlightParams;
const highlightInnerGlow = highlight.highlightInnerGlow;
const highlightComposite = highlight.highlightComposite;

test "highlight active gating and params packing" {
    // Zero highlights (or post off) runs zero passes: the composite keeps
    // its no-highlight path (bit-identical to pre-highlight).
    try std.testing.expect(!highlightActive(true, 0));
    try std.testing.expect(!highlightActive(false, 1));
    try std.testing.expect(!highlightActive(false, 8));
    try std.testing.expect(highlightActive(true, 1));
    try std.testing.expect(highlightActive(true, 8));

    // Disabled packs all zeros (the shader returns before sampling).
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, highlightParams(false));
    // Enabled packs the baked global scale 1.0: per-item intensity already
    // folded into the mask at draw time (exact under the normalized blur).
    try std.testing.expectEqual([4]f32{ 1.0, 1.0, 0.0, 0.0 }, highlightParams(true));

    // Independent of glow/bloom: toggling those never changes the
    // highlight decision (and vice versa).
    const glow_on = options.PostProcessOptions{ .glow_enabled = true };
    try std.testing.expect(!highlightActive(true, 0));
    try std.testing.expect(glow.glowActive(true, glow_on));
    try std.testing.expect(highlightActive(true, 3));
}

test "highlight composite is an inner glow, not a flat fill" {
    // Mesh interior (mask == blurred, whatever the folded intensity):
    // zero contribution — this is what the old flat-fill got wrong.
    try std.testing.expectEqual([3]f32{ 0.5, 0.5, 0.5 }, highlightComposite(.{ 0.5, 0.5, 0.5 }, .{ 0.4, 0.2, 0.8 }, .{ 0.4, 0.2, 0.8 }));
    // Interior stays dark at full intensity too (intensity-independent).
    try std.testing.expectEqual([3]f32{ 0.1, 0.1, 0.1 }, highlightComposite(.{ 0.1, 0.1, 0.1 }, .{ 1.0, 1.0, 1.0 }, .{ 1.0, 1.0, 1.0 }));
    // Silhouette edge (blur ~half coverage): x2 restores the full mask
    // color on top of the scene color.
    try std.testing.expectEqual([3]f32{ 0.5 + 0.4, 0.5 + 0.2, 0.5 + 0.8 }, highlightComposite(.{ 0.5, 0.5, 0.5 }, .{ 0.4, 0.2, 0.8 }, .{ 0.2, 0.1, 0.4 }));
    // Outside the mesh (raw mask 0, blurred spill > 0): clamped to 0 —
    // inner-only glow, no out-of-mesh halo.
    try std.testing.expectEqual([3]f32{ 0.2, 0.4, 0.6 }, highlightComposite(.{ 0.2, 0.4, 0.6 }, .{ 0.0, 0.0, 0.0 }, .{ 0.3, 0.1, 0.2 }));
    // Zero mask + zero blur is identity (what the disabled path computes
    // without sampling).
    try std.testing.expectEqual([3]f32{ 0.2, 0.4, 0.6 }, highlightComposite(.{ 0.2, 0.4, 0.6 }, .{ 0.0, 0.0, 0.0 }, .{ 0.0, 0.0, 0.0 }));
}

test "highlight inner glow edge term" {
    // Interior: mask == blurred -> exactly zero at any intensity.
    try std.testing.expectEqual([3]f32{ 0.0, 0.0, 0.0 }, highlightInnerGlow(.{ 0.7, 0.1, 0.0 }, .{ 0.7, 0.1, 0.0 }));
    // Edge: (mask - blurred) * 2 per channel.
    const edge = highlightInnerGlow(.{ 0.4, 0.2, 0.8 }, .{ 0.2, 0.1, 0.4 });
    try std.testing.expectApproxEqAbs(@as(f32, 0.4), edge[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), edge[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), edge[2], 1e-6);
    // Outside: negative floors per channel, never bleeds across channels.
    try std.testing.expectEqual([3]f32{ 0.0, 0.0, 0.0 }, highlightInnerGlow(.{ 0.0, 0.0, 0.0 }, .{ 0.3, 0.1, 0.2 }));
    const mixed = highlightInnerGlow(.{ 0.5, 0.0, 0.25 }, .{ 0.1, 0.4, 0.25 });
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), mixed[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mixed[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), mixed[2], 1e-6);
}
