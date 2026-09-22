const std = @import("std");
const sg = @import("sokol").gfx;

// ---------------------------------------------------------------------------
// MSAA policy for the offscreen main render target (PASS 2 in Scene.render).
//
// This module holds the *decisions*; every subsystem (forward pipelines,
// postprocess targets, skybox/particles/outline/debug passes) pulls its
// sample count from the effective count computed here, so pipeline and
// attachment sample counts can never drift apart (sokol validation rejects
// sg_apply_pipeline when pipeline.sample_count differs from any attachment
// image's sample_count, including the depth attachment).
//
// Backend matrix (verified against the vendored sokol_gfx.h):
//
//   backend        offscreen MSAA  color resolve  MSAA depth     postfx reads depth
//   -------------  --------------  -------------  -------------  ------------------
//   Metal          yes             yes            required(1)    no(2)
//   D3D11          yes             yes            required(1)    no(2)
//   Vulkan         yes             yes            required(1)    no(2)
//   WebGPU         yes             yes            required(1)    no(2)
//   GL 4.1 macOS   yes (3)         yes (3)        required(1)    no (4)
//   GL 4.3+ Linux  yes             yes            required(1)    no(2)
//   GLES3/WebGL2   yes             yes            required(1)    no (4)
//
//   (1) sokol validation requires depth_stencil attachment sample_count ==
//       color attachment sample_count; a mixed 4x color + 1x depth pass is
//       rejected (VALIDATE_APIP_ATTACHMENT_SAMPLE_COUNT).
//   (2) Reading requires a `.multisampled` shader view and per-sample
//       texture2D_ms fetches; the postfx shaders use plain sampler2D, so an
//       MSAA depth attachment is effectively write-only everywhere.
//   (3) GL resolves MSAA renderbuffers via glBlitFramebuffer (color only).
//   (4) sg_features.msaa_texture_bindings == false on GL-on-macOS and
//       GLES/WebGL2; on GL 4.3+ binding is allowed but sampling still needs
//       sampler2DMS, which the shaders don't use.
//
// There is NO depth resolve in this sokol version on ANY backend: the pass
// attachment struct carries colors[] + resolves[] + one depth_stencil view;
// resolve views are color-only (Metal's depthAttachment has no resolve
// texture, GL blits COLOR_BUFFER_BIT only). Therefore a resolved 1x depth
// texture for SSAO/SSR/DoF cannot be produced from the MSAA pass at all.
//
// Chosen strategy ("MSAA wins", the task's option (a) adapted to reality):
// when the effective sample count is > 1, the depth-consuming post effects
// are SUPPRESSED for the frame (SSAO render is skipped; SSR/DoF flags are
// forced off in the composite), with a warn-once. Rationale:
//   - The inverse ("option b": drop to 1x whenever SSAO/SSR/DoF is on)
//     would make --msaa a silent no-op in any scene with SSAO enabled -
//     and SSAO defaults to on in this engine.
//   - Suppression is graceful: no SSAO output = slightly brighter corners;
//     no SSR/DoF = the exact pre-MSAA look for those terms.
//   - Full quality for both remains available by leaving
//     Scene.msaa_sample_count at 1.
// Shadows are untouched: they render into their own 1x depth atlas with
// sample_count-1 pipelines.
// ---------------------------------------------------------------------------

/// Sample counts the engine understands. Anything else is snapped down to
/// the nearest entry (3 -> 2, 5..7 -> 4, 9+ -> 8-then-capped by backend).
pub const valid_sample_counts = [_]i32{ 1, 2, 4, 8 };

/// Conservative per-backend MSAA caps. 4x is the only level every sokol
/// backend guarantees for the main-target formats (BGRA8/RGBA8 color,
/// DEPTH/DEPTH_STENCIL depth); sg_limits exposes no max-sample-count query,
/// so a device that could do 8x cannot be detected at runtime and we clamp
/// to the portable maximum instead of risking pipeline/attachment creation
/// failures (a failed MSAA target would black-screen the frame).
pub fn maxSamplesForBackend(backend: sg.Backend) i32 {
    return switch (backend) {
        // Tests and headless tooling build against the dummy backend; allow
        // the full table there so clamping stays observable in unit tests.
        .DUMMY => 8,
        // Real backends (Metal macOS/iOS/simulator, GLCore, GLES3, D3D11,
        // WebGPU, Vulkan): portable 4x maximum.
        else => 4,
    };
}

/// Snap a requested sample count to the largest valid count <= requested,
/// then cap it at the backend maximum. 1 and below means "off".
pub fn clampSampleCount(backend: sg.Backend, requested: i32) i32 {
    if (requested < 2) return 1;
    const max = maxSamplesForBackend(backend);
    var chosen: i32 = 1;
    for (valid_sample_counts) |v| {
        if (v <= requested and v <= max) chosen = v;
    }
    return chosen;
}

/// Inputs for effectiveSampleCount. Everything is a plain bool so the
/// decision stays unit-testable without a GPU.
pub const Inputs = struct {
    /// Post-processing chain on? MSAA only applies to the offscreen main
    /// target; with post off the main pass IS the swapchain, whose sample
    /// count is fixed by sokol_app at startup (agate runs it at 1).
    post_enabled: bool,
    /// Both main-target attachment formats support MSAA at runtime
    /// (sg.queryPixelformat(fmt).msaa for the swapchain color and depth
    /// formats). False forces 1x with a warn at the call site.
    formats_msaa_capable: bool = true,
    /// Backend reported by sg.queryBackend().
    backend: sg.Backend = .DUMMY,
};

/// The sample count the main render target uses this frame. Pure.
///
/// The depth-effect suppression (SSAO/SSR/DoF off while this returns > 1)
/// is NOT part of this function on purpose: it changes what runs in the
/// post chain, not the target's sample count, and lives in PostFXStack
/// (which owns the warn-once state).
pub fn effectiveSampleCount(requested: i32, in: Inputs) i32 {
    if (!in.post_enabled) return 1;
    if (!in.formats_msaa_capable) return 1;
    return clampSampleCount(in.backend, requested);
}

/// True when any depth-texture-consuming post effect would run this frame.
/// While the main target is MSAA these are suppressed (see module docs).
pub fn depthEffectsActive(post_enabled: bool, ssao_enabled: bool, ssao_debug: bool, ssr_enabled: bool, dof_enabled: bool, fog_enabled: bool) bool {
    return post_enabled and (ssao_enabled or ssao_debug or ssr_enabled or dof_enabled or fog_enabled);
}

/// Single-sample depth-prepass gate (MSAA depth-resolve design v1).
///
/// sokol has no depth resolve on any backend (see module docs), so an MSAA
/// main target cannot feed a depth texture to the post chain by itself.
/// When `gate` (Scene.msaa_depth_prepass) is on, the renderer draws the
/// opaque primary-view geometry a second time with depth-only pipelines
/// into a 1x depth texture (PASS 1.7, before the main pass); the post chain
/// then reads that texture instead of the write-only MSAA depth.
///
/// Pure so the toggle matrix stays unit-testable without a GPU. The gate
/// applies only on top of an MSAA main target: with samples <= 1 the 1x
/// main depth texture already exists and no prepass is needed; with post
/// off there is no post chain to feed.
pub fn depthPrepassActive(post_enabled: bool, gate: bool, main_samples: i32) bool {
    return post_enabled and gate and main_samples > 1;
}

/// True when the MSAA depth-effect suppression applies this frame: MSAA is
/// active but no 1x depth is available (prepass gate off). With the prepass
/// on, SSAO/SSR/DoF/Fog/MotionBlur read the prepass texture and run
/// normally; TAA stays suppressed regardless (non-goal v1: TAA under MSAA).
pub fn suppressDepthEffects(main_samples: i32, gate: bool) bool {
    return main_samples > 1 and !gate;
}

/// True when a resolve attachment must exist for the main color target.
pub fn needsResolveAttachment(sample_count: i32) bool {
    return sample_count > 1;
}

/// One-shot warning latch for runtime policy decisions. GPU-free; the flag
/// lives with the owner (Scene / PostFXStack) so it resets with them.
pub const WarnOnce = struct {
    fired: bool = false,

    /// Warns the first call only; returns true when a warning was emitted
    /// (handy for tests).
    pub fn warn(self: *WarnOnce, comptime fmt: []const u8, args: anytype) bool {
        if (self.fired) return false;
        self.fired = true;
        std.log.warn(fmt, args);
        return true;
    }
};

/// True when the swapchain color AND depth formats both support MSAA at
/// runtime (the main target mirrors them; see postprocess_pass.resize).
/// sg.queryPixelformat is the sokol-provided backend gate.
pub fn mainTargetFormatsMsaaCapable() bool {
    const env_def = sg.queryDesc().environment.defaults;
    const color_fmt: sg.PixelFormat = if (env_def.color_format != .DEFAULT and env_def.color_format != .NONE) env_def.color_format else .BGRA8;
    const depth_fmt: sg.PixelFormat = if (env_def.depth_format != .DEFAULT and env_def.depth_format != .NONE) env_def.depth_format else .DEPTH;
    return sg.queryPixelformat(color_fmt).msaa and sg.queryPixelformat(depth_fmt).msaa;
}

// --- GPU-free contract tests (visual AA quality cannot be unit-tested; it
// is verified with the agate smoke run: `agate --frames 120 --msaa 4`). ---

const testing = std.testing;

test "clampSampleCount snaps down to valid counts and caps by backend" {
    // Portable backends cap at 4 (see maxSamplesForBackend docs).
    for ([_]sg.Backend{ .METAL_MACOS, .METAL_IOS, .D3D11, .VULKAN, .WGPU, .GLCORE, .GLES3 }) |backend| {
        try testing.expectEqual(@as(i32, 1), clampSampleCount(backend, 1));
        try testing.expectEqual(@as(i32, 1), clampSampleCount(backend, 0));
        try testing.expectEqual(@as(i32, 1), clampSampleCount(backend, -4));
        try testing.expectEqual(@as(i32, 2), clampSampleCount(backend, 2));
        try testing.expectEqual(@as(i32, 2), clampSampleCount(backend, 3));
        try testing.expectEqual(@as(i32, 4), clampSampleCount(backend, 4));
        try testing.expectEqual(@as(i32, 4), clampSampleCount(backend, 5));
        try testing.expectEqual(@as(i32, 4), clampSampleCount(backend, 8));
        try testing.expectEqual(@as(i32, 4), clampSampleCount(backend, 16));
    }
    // Dummy backend (tests, headless tooling) allows the full table.
    try testing.expectEqual(@as(i32, 8), clampSampleCount(.DUMMY, 8));
    try testing.expectEqual(@as(i32, 8), clampSampleCount(.DUMMY, 99));
}

test "effectiveSampleCount gates on post path and format support" {
    const base = Inputs{ .post_enabled = true, .backend = .DUMMY };
    try testing.expectEqual(@as(i32, 4), effectiveSampleCount(4, base));
    // Post off: main pass is the 1x swapchain, MSAA never applies.
    try testing.expectEqual(@as(i32, 1), effectiveSampleCount(4, .{ .post_enabled = false, .backend = .DUMMY }));
    // Runtime format gate (e.g. a backend that cannot MSAA the swapchain
    // color format): degrade to 1x rather than fail resource creation.
    try testing.expectEqual(@as(i32, 1), effectiveSampleCount(4, .{ .post_enabled = true, .formats_msaa_capable = false, .backend = .DUMMY }));
    // Clamp still applies on the effective path (real-backend cap is 4;
    // the dummy backend used by base allows the full table).
    try testing.expectEqual(@as(i32, 4), effectiveSampleCount(8, .{ .post_enabled = true, .backend = .METAL_MACOS }));
    try testing.expectEqual(@as(i32, 8), effectiveSampleCount(8, base));
}

test "depthEffectsActive matches the suppressed set (SSAO/SSR/DoF/Fog)" {
    // SSAO defaults on in this engine: that alone counts as active.
    try testing.expect(depthEffectsActive(true, true, false, false, false, false));
    try testing.expect(depthEffectsActive(true, false, true, false, false, false)); // debug
    try testing.expect(depthEffectsActive(true, false, false, true, false, false)); // SSR
    try testing.expect(depthEffectsActive(true, false, false, false, true, false)); // DoF
    try testing.expect(depthEffectsActive(true, false, false, false, false, true)); // Fog
    try testing.expect(!depthEffectsActive(true, false, false, false, false, false));
    // Without the post chain nothing runs at all.
    try testing.expect(!depthEffectsActive(false, true, true, true, true, true));
}

test "needsResolveAttachment follows the sokol resolve contract" {
    try testing.expect(!needsResolveAttachment(1));
    try testing.expect(needsResolveAttachment(2));
    try testing.expect(needsResolveAttachment(4));
}

test "WarnOnce fires exactly once" {
    var w = WarnOnce{};
    try testing.expect(w.warn("msaa: {}", .{1}));
    try testing.expect(!w.warn("msaa: {}", .{2}));
}

test "depthPrepassActive needs post, gate, and MSAA samples" {
    try testing.expect(depthPrepassActive(true, true, 4));
    try testing.expect(depthPrepassActive(true, true, 2));
    // Gate off (default): never runs — the off path stays bit-identical.
    try testing.expect(!depthPrepassActive(true, false, 4));
    // Post off: no post chain to feed.
    try testing.expect(!depthPrepassActive(false, true, 4));
    // 1x target: the main depth texture already exists, no prepass needed.
    try testing.expect(!depthPrepassActive(true, true, 1));
    try testing.expect(!depthPrepassActive(true, true, 0));
}

test "suppressDepthEffects lifts only under the prepass gate" {
    // 1x: nothing is ever suppressed.
    try testing.expect(!suppressDepthEffects(1, false));
    try testing.expect(!suppressDepthEffects(1, true));
    // MSAA without the gate: suppression applies (legacy behavior).
    try testing.expect(suppressDepthEffects(4, false));
    try testing.expect(suppressDepthEffects(2, false));
    // MSAA with the gate: the prepass feeds depth, suppression lifts.
    try testing.expect(!suppressDepthEffects(4, true));
}
