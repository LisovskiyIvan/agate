//! Highlight layer v1 (per-mesh colored "inner glow", Babylon.js
//! HighlightLayer parity item).
//!
//! Design (bounded, additive, OFF by default):
//! - At most `max_highlights` (8) highlighted meshes per scene;
//!   `addHighlightMesh` past the cap is a hard `error.TooManyHighlights`
//!   (never a silent clamp or replacement), mirroring the reflection-probe
//!   cap in `probe_layer.zig` and the 3D-GUI panel cap in `gui3d_layer.zig`.
//! - Each entry carries its own `HighlightOptions` (color, blur radius,
//!   intensity): unlike the single-color inverse-hull outline
//!   (`Scene.outline_meshes` + `PostFXStack.outline_color`), highlights are
//!   per-mesh. The staged `HighlightDrawItem` (see
//!   `passes/highlight_pass.zig`) freezes the world matrix, the GPU buffer
//!   handles, and the options at prepare time, so `Scene.render` never
//!   dereferences a live `Mesh` (P4/P7/P8 discipline, same as
//!   `OutlineDrawItem`).
//! - Rendering with zero highlights is bit-identical to before this wave:
//!   `PostFXStack.renderChain` gates the whole mask/blur/composite chain on
//!   `postprocess.highlightActive` (post on AND at least one staged item),
//!   and the composite shader keeps its no-highlight path when the pass
//!   feeds an empty view.
//! - Lazy targets: the mask + H/V blur RTs allocate on the first ACTIVE
//!   `HighlightPass.render` only — `resizeAll`/`beginMainPass` never size
//!   them, so zero-highlight frames hold no highlight VRAM and the VRAM
//!   census (gated on the pass base size) honestly reports zero. A resize
//!   during OFF cannot break the first active frame: `render` resizes to
//!   the current base size before drawing.
//! - Render approach (documented choice): mask-RT inner glow. During prepare
//!   (phase-locked) the world matrices + proxy geometry handles of the
//!   highlighted meshes are staged into the frame slot; a render pass draws
//!   them flat-colored (per-item color x intensity) into a half-resolution
//!   mask RT, blurs it with the glow-style separable Gaussian (same
//!   `glow_blur` shader module and kernel math as `GlowPass`), and the
//!   fullscreen composite adds the inner glow after the glow block, before
//!   the grading chain. The composite reads the RAW mask minus its blurred
//!   halo, floored at zero per channel and doubled
//!   (`highlightInnerGlow` in `postprocess.zig`, mirrored in
//!   `shaders/postprocess.glsl`): mesh interiors (mask ~= blurred)
//!   contribute ~0, the silhouette edge (blur ~= half coverage) restores
//!   the full per-item color, and outside the mesh the raw mask is 0 so
//!   the blurred spill clamps to 0. The difference form (rather than
//!   mask x (1 - blurred)) is exact under the folded per-item intensity;
//!   the visual is deliberately inner-only with NO out-of-mesh halo (the
//!   halo variant would add max(blurred - mask, 0) instead). The
//!   inverse-hull fallback (per-item outline colors) was rejected: it
//!   cannot produce the soft inner-glow falloff the roadmap item asks
//!   for, and reusing the outline shader for the flat mask fill keeps
//!   this pass at zero new GLSL files.
//!
//! Explicit non-goals (v1):
//! - Skinned/animated highlighted meshes: no skin matrix staging exists, so
//!   skinned meshes are skipped at capture time (fail-closed, never drawn
//!   from a bind pose).
//! - Instanced meshes: only the template proxy is drawn (one draw with the
//!   mesh world matrix, no per-instance matrices); no per-instance
//!   highlight.
//! - No occlusion-aware highlight: the mask has no depth attachment and the
//!   composite is purely additive, so a highlight bleeds through foreground
//!   geometry (same documented limit as the global glow layer).
//! - Alpha-cutout cards draw their quad proxy (no alpha test in the mask
//!   path), so the halo follows the quad, not the leaf silhouette.
//! - Multi-camera: the mask draws and composites from the primary view
//!   only, under the primary camera's pixel viewport mapped onto the
//!   half-res mask target (PIP-aware; secondary views never contribute).
//! - Blur radius is frame-global (max over the staged items): per-item blur
//!   values are validated and staged, but one separable blur runs per
//!   frame, so items with a smaller radius get the frame's widest halo.
//!   Per-item color/intensity stay exact (folded into the mask at draw).
//!
//! Threading: the layer is owned by `Scene` (game side for
//! add/remove/clear under update-vs-prepare exclusion, like the probe and
//! 3D-GUI layers). Entries own no GPU resources, so removal needs no retire
//! queue: the mesh's own buffers retire through the existing `retireMesh`
//! path, and already-staged items fail closed on dead handles at draw time
//! (epoch discipline + `queryBufferState` guards, outline precedent).
//!
//! Serialization: transient (like bloom/glow config and the outline mesh
//! list) — entries reference live `*Mesh` and are never written to
//! `SceneState`; a loaded scene starts with zero highlights.
//!
//! Headless behavior: pure CPU bookkeeping, no `sg.*` calls anywhere in
//! this file, so every function below is directly unit-testable without a
//! GPU context.

const std = @import("std");
const Mesh = @import("../mesh.zig").Mesh;

/// Fixed capacity: at most this many highlighted meshes per scene. Bakes
/// into the layer as a plain array (no per-frame allocation, no deinit).
pub const max_highlights: usize = 8;

/// Default highlight color (linear white, opaque). Intensity carries the
/// brightness; the color stays chromaticity-only in [0, 1] per channel.
pub const HIGHLIGHT_COLOR_DEFAULT: [4]f32 = .{ 1.0, 1.0, 1.0, 1.0 };
/// Default blur sigma in highlight-target texels (mirrors the glow layer's
/// wider-than-bloom radius).
pub const HIGHLIGHT_BLUR_DEFAULT: f32 = 4.0;
/// Default additive composite scale (mirrors the glow intensity default).
pub const HIGHLIGHT_INTENSITY_DEFAULT: f32 = 0.5;

/// Per-mesh highlight knobs. A flat config owned by value in the layer
/// entry and frozen into the staged draw item at prepare time.
pub const HighlightOptions = struct {
    /// Linear RGB + alpha, each channel in [0, 1]. The mask fill uses
    /// rgb x intensity; alpha is staged but currently unused downstream
    /// (the composite adds rgb only).
    color: [4]f32 = HIGHLIGHT_COLOR_DEFAULT,
    /// Blur sigma in highlight-target texels; >= 0. Frame-global max wins
    /// at render (see module docs).
    blur: f32 = HIGHLIGHT_BLUR_DEFAULT,
    /// Additive composite scale; >= 0. Folded into the mask color exactly.
    intensity: f32 = HIGHLIGHT_INTENSITY_DEFAULT,
};

/// Strict range check for the highlight knobs. Finite out-of-range values
/// are hard errors here (glow precedent: `validateGlow` returns
/// `InvalidGlowOptions` for negatives instead of silently clamping; the
/// silent `clamped()` domain lives on load paths, and highlights have no
/// load path — they are transient). Non-finite values (NaN/Inf) can never
/// sanitize meaningfully, so they are the same hard error.
pub fn validateHighlightOptions(options: HighlightOptions) !void {
    for (options.color) |c| {
        if (!std.math.isFinite(c)) return error.InvalidHighlightOptions;
    }
    if (!std.math.isFinite(options.blur)) return error.InvalidHighlightOptions;
    if (!std.math.isFinite(options.intensity)) return error.InvalidHighlightOptions;
    for (options.color) |c| {
        if (c < 0.0 or c > 1.0) return error.InvalidHighlightOptions;
    }
    if (options.blur < 0.0) return error.InvalidHighlightOptions;
    if (options.intensity < 0.0) return error.InvalidHighlightOptions;
}

/// One highlighted mesh: the borrowed mesh pointer plus its frozen-at-add
/// options and the mesh uid at add time (identity for the staged item).
pub const HighlightEntry = struct {
    mesh: *Mesh,
    options: HighlightOptions = .{},
    uid: u64 = 0,
};

/// Owns the scene's highlight entries. Fixed-size array + count (probe
/// precedent): steady state allocates nothing and needs no deinit.
pub const HighlightLayer = struct {
    entries: [max_highlights]HighlightEntry = undefined,
    count: usize = 0,

    /// Adds `mesh` with `options`; returns its index (id). Past
    /// `max_highlights` this is a hard error (never a silent clamp or
    /// replacement), mirroring `ProbeLayer.add`. Invalid options are a
    /// hard `InvalidHighlightOptions` error (never a silent clamp).
    /// Headless-safe: only CPU state, no GPU work (targets are pass-owned
    /// and lazy).
    pub fn add(self: *HighlightLayer, mesh: *Mesh, options: HighlightOptions) error{ TooManyHighlights, InvalidHighlightOptions }!usize {
        try validateHighlightOptions(options);
        if (self.count >= max_highlights) return error.TooManyHighlights;
        _ = mesh.ensureUid();
        const idx = self.count;
        self.entries[idx] = .{ .mesh = mesh, .options = options, .uid = mesh.uid };
        self.count += 1;
        return idx;
    }

    /// Removes entry `index`. Order-preserving: higher indices shift down,
    /// so callers must not cache indices across removals. Out-of-range
    /// indices are a no-op (same contract as `Scene.removeCamera`). No
    /// retire queue: entries own no GPU resources (the mesh's own buffers
    /// retire through `retireMesh`; staged items fail closed on handles).
    pub fn remove(self: *HighlightLayer, index: usize) void {
        if (index >= self.count) return;
        for (index..self.count - 1) |k| self.entries[k] = self.entries[k + 1];
        self.count -= 1;
    }

    /// Drops every entry referencing `mesh` (the `destroyMesh` path:
    /// mirrors the `outline_meshes` swap-remove there, order-preserving
    /// variant here to match `remove`). Runs synchronously on the game
    /// side for both the sync and the off-context (epoch-retired) destroy
    /// branches; already-staged draw items fail closed on their borrowed
    /// handles at draw time.
    pub fn removeForMesh(self: *HighlightLayer, mesh: *Mesh) void {
        var i: usize = 0;
        while (i < self.count) {
            if (self.entries[i].mesh == mesh) {
                self.remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Drops every entry (back to the bit-identical no-highlight path).
    pub fn clear(self: *HighlightLayer) void {
        self.count = 0;
    }

    /// Live entry state. Null when out of range.
    pub fn get(self: *HighlightLayer, index: usize) ?*HighlightEntry {
        if (index >= self.count) return null;
        return &self.entries[index];
    }

    pub fn highlightCount(self: *const HighlightLayer) usize {
        return self.count;
    }
};
