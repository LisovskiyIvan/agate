const std = @import("std");
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

    // MSAA twin pass (pipeline sample count must match the main target).
    // Lazily created on the first MSAA frame, recreated on count changes.
    pass_msaa: ?passes.SkyboxPass = null,
    pass_msaa_samples: i32 = 0,

    pub fn init() SkyboxLayer {
        return .{ .pass = passes.SkyboxPass.init() };
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

    /// Renders the skybox inside the main pass. `fallback` is the shared
    /// default cubemap used when no custom skybox texture is set; `samples`
    /// is the effective main-target sample count (scene/msaa.zig).
    pub fn render(self: *SkyboxLayer, camera: Camera, aspect: f32, fallback: CubeTexture, samples: i32, stats: *SceneStats) void {
        if (!self.enabled) return;
        const pass = self.passFor(samples);
        pass.render(camera, aspect, self.texture orelse fallback, self.exposure);
        stats.main_draw_calls += 1;
        stats.draw_calls += 1;
        stats.triangles += 12;
    }

    /// Pass variant matching the target sample count.
    fn passFor(self: *SkyboxLayer, samples: i32) *passes.SkyboxPass {
        if (samples <= 1) return &self.pass;
        if (self.pass_msaa == null or self.pass_msaa_samples != samples) {
            if (self.pass_msaa) |*p| p.deinit();
            self.pass_msaa = passes.SkyboxPass.initSampled(samples);
            self.pass_msaa_samples = samples;
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
