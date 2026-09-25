const std = @import("std");
const options = @import("options.zig");
const glow = @import("glow.zig");

// --- Highlight layer v1 (per-mesh inner glow) ---
//
// Composite order (see postprocess.glsl main): ... BLOOM -> GLOW ->
// HIGHLIGHT -> contrast -> saturation -> curves -> LUT -> vignette ->
// grain. The highlight halo composites right after the glow block so
// either toggle leaves the other's contribution bit-identical, and before
// the grading chain so per-mesh colors grade with the same LDR the
// bloom/glow halos use.
//
// Pass order (see scene/postfx_stack.zig renderChain): PASS 2.5 SSAO, PASS
// 2.75 bloom pyramid, PASS 2.8 glow (extract + H + V), PASS 2.85 highlight
// (mask fills + H + V = HIGHLIGHT_BLUR_DRAWS fullscreen draws), PASS 3
// fullscreen composite. Like glow the highlight pass samples no depth, so
// the MSAA depth-suppression never applies to it. Like glow it is
// uniform-only past its mask binds (no sg.updateBuffer, hence no
// gpu_upload_meter records) and resize-idempotent, so renderReuse replays
// stay upload-free by construction.
//
// Per-item intensity folds into the mask fill color at draw time (exact:
// the blur kernel is normalized), so the composite carries a single baked
// global scale of 1.0 below. The frame blur sigma is the max over the
// staged items (documented v1 approximation in
// passes/highlight_pass.zig). The composite is an inner glow, not a flat
// fill: highlightInnerGlow/highlightComposite mirror the shader block
// (raw mask minus blurred halo, floored at zero, x2) — see
// highlightInnerGlow for why the difference form is exact under folded
// intensity.

/// Pass-construction decision (pure; PostFXStack.renderChain gates the GPU
/// passes on this). Zero staged items (or post off) runs zero passes and
/// binds the placeholder (bit-identical composite).
pub fn highlightActive(post_enabled: bool, highlight_count: usize) bool {
    return post_enabled and highlight_count > 0;
}

/// Pack the composite highlight_params vec4: (enabled 1/0, baked global
/// scale 1.0, 0, 0). Disabled packs all zeros, which keeps the composite
/// bit-identical to the pre-highlight path (the shader returns before
/// sampling highlight_tex).
pub fn highlightParams(active: bool) [4]f32 {
    if (!active) return .{ 0.0, 0.0, 0.0, 0.0 };
    return .{ 1.0, 1.0, 0.0, 0.0 };
}

/// Inner-glow edge term. Mirrors the highlight block in postprocess.glsl:
/// per-channel max(mask - blurred, 0) * 2.
pub fn highlightInnerGlow(mask: [3]f32, blurred: [3]f32) [3]f32 {
    return .{
        @max(mask[0] - blurred[0], 0.0) * 2.0,
        @max(mask[1] - blurred[1], 0.0) * 2.0,
        @max(mask[2] - blurred[2], 0.0) * 2.0,
    };
}

/// Additive inner-glow composite. Mirrors the highlight block in
/// postprocess.glsl: the raw per-item mask minus its blurred halo,
/// floored at zero (inner-only) and added straight onto the scene color.
pub fn highlightComposite(color: [3]f32, mask: [3]f32, blurred: [3]f32) [3]f32 {
    const inner = highlightInnerGlow(mask, blurred);
    return .{ color[0] + inner[0], color[1] + inner[1], color[2] + inner[2] };
}

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
