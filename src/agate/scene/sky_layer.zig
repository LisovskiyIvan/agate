const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const Camera = @import("../camera.zig").Camera;
const CubeTexture = @import("../texture.zig").CubeTexture;
const SkyboxOptions = @import("../texture.zig").SkyboxOptions;
const passes = @import("../passes/mod.zig");
const stats_mod = @import("stats.zig");
const SceneStats = stats_mod.SceneStats;

/// Skybox state and its render pass: the optional environment cubemap, its
/// exposure, and the IBL intensity fed to PBR materials. Scene keeps only
/// the subsystem; the fallback cubemap (used when no skybox is set) is
/// passed in by Scene at render time because it belongs to the shared
/// default-texture set.
pub const SkyboxLayer = struct {
    texture: ?CubeTexture = null,
    enabled: bool = false,
    exposure: f32 = 1.0,
    // Environment (IBL) intensity multiplied into PBR materials' env term.
    ibl_intensity: f32 = 1.0,

    pass: passes.SkyboxPass,

    // Variant pass (pipeline sample count must match the main target).
    // Lazily created on the first non-base frame, recreated on shape changes.
    pass_msaa: ?passes.SkyboxPass = null,

    pub fn init() SkyboxLayer {
        return .{ .pass = passes.SkyboxPass.init(1, .RGBA16F) };
    }

    pub fn deinit(self: *SkyboxLayer) void {
        // Scene owns the skybox cubemap lifetime via this layer.
        if (self.texture) |*c| c.deinit();
        self.texture = null;
        self.pass.deinit();
        if (self.pass_msaa) |*p| p.deinit();
        self.pass_msaa = null;
    }

    pub fn setSkybox(self: *SkyboxLayer, cube: CubeTexture) void {
        self.texture = cube;
        self.enabled = true;
    }

    pub fn createDefault(self: *SkyboxLayer, allocator: std.mem.Allocator, config: SkyboxOptions) !void {
        const cube = try CubeTexture.createProceduralSkybox(allocator, config);
        self.setSkybox(cube);
    }

    /// Renders the skybox inside the main pass from CAPTURED params only:
    /// `enabled`/`cube_tex`/`exposure` travel in the frame snapshot (the
    /// caller resolves `cube_tex` as snapshot sky texture orelse the
    /// snapshot's render-owned default copy — never the game-mutatable
    /// shared default), so a concurrent update mutating the live layer
    /// cannot race the draw. `samples`/`color_format` pin the exact
    /// main-target shape. Headless-safe: no `sg.*` without a context,
    /// and the counters bump only when the pass actually draws.
    pub fn renderPrepared(
        self: *SkyboxLayer,
        enabled: bool,
        camera: Camera,
        aspect: f32,
        cube_tex: CubeTexture,
        exposure: f32,
        samples: i32,
        color_format: sg.PixelFormat,
        stats: *SceneStats,
    ) void {
        if (!enabled) return;
        // Context first (before touching the pass: headless fixtures may
        // hold an uninitialized pass, and variant passes create GPU objects).
        if (!sg.isvalid()) return;
        const pass = self.passFor(samples, color_format);
        // Consumability guard (FAILED pipeline): the pass itself early-outs
        // there, so check first to keep counters exact.
        if (sg.queryPipelineState(pass.pipeline) != .VALID) return;
        pass.render(camera, aspect, cube_tex, @import("../postprocess/hdr.zig").sanitizeExposure(exposure));
        stats.main_draw_calls += 1;
        stats.draw_calls += 1;
        stats.triangles += 12;
    }

    /// Renders the skybox inside the main pass. `fallback` is the shared
    /// default cubemap used when no custom skybox texture is set.
    ///
    /// Live-state path: Scene.render no longer calls this — it draws
    /// from the snapshot via renderPrepared. Kept for standalone/tooling.
    pub fn render(self: *SkyboxLayer, camera: Camera, aspect: f32, fallback: CubeTexture, samples: i32, color_format: sg.PixelFormat, stats: *SceneStats) void {
        if (!self.enabled) return;
        self.renderPrepared(true, camera, aspect, self.texture orelse fallback, self.exposure, samples, color_format, stats);
    }

    /// Pass variant matching the exact target shape (sample count + color
    /// format). The single twin slot serves every non-base shape, keyed by
    /// both; the base shape stays on the base pass.
    fn passFor(self: *SkyboxLayer, samples: i32, color_format: sg.PixelFormat) *passes.SkyboxPass {
        if (samples == self.pass.sample_count and color_format == self.pass.color_format) return &self.pass;
        if (self.pass_msaa == null or self.pass_msaa.?.sample_count != samples or self.pass_msaa.?.color_format != color_format) {
            if (self.pass_msaa) |*p| p.deinit();
            self.pass_msaa = passes.SkyboxPass.init(samples, color_format);
        }
        return &self.pass_msaa.?;
    }
};

// GPU-free state test; the pass itself is covered by passes/ tests.
test "skybox layer state toggles with setSkybox" {
    var sky: SkyboxLayer = .{ .pass = undefined };
    try std.testing.expect(!sky.enabled);
    try std.testing.expect(sky.texture == null);

    const cube: CubeTexture = undefined;
    sky.setSkybox(cube);
    try std.testing.expect(sky.enabled);
    try std.testing.expect(sky.texture != null);

    sky.exposure = 2.5;
    sky.ibl_intensity = 0.5;
    try std.testing.expectEqual(@as(f32, 2.5), sky.exposure);
    try std.testing.expectEqual(@as(f32, 0.5), sky.ibl_intensity);
}
