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
