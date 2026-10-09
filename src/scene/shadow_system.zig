const std = @import("std");
const math = @import("math");
const Mat4 = math.Mat4;
const Color3 = math.Color3;

const Camera = @import("../camera.zig").Camera;
const passes = @import("../passes/mod.zig");
const scene_cascades = @import("cascades.zig");
const scene_uniforms = @import("uniforms.zig");

/// Shadow mapping state for the CSM sun (16-sample Poisson PCF by default,
/// optional PCSS) plus the GPU shadow depth pass. Cascaded matrices are
/// recomputed per frame from the resolved sun direction; the fragment
/// uniform side is exposed via uniformState() so the draw path never pokes
/// at individual bias fields.
pub const ShadowSystem = struct {
    enabled: bool = true,
    bias: f32 = 0.0012,
    normal_bias: f32 = 0.02,
    intensity: f32 = 0.75,
    softness: f32 = 1.5,
    debug_cascades: bool = false,
    // PCSS (percentage-closer soft shadows) for the CSM sun. Disabled keeps
    // the legacy 16x Poisson PCF path bit-identical.
    pcss_enabled: bool = false,
    pcss_light_size: f32 = 0.02,
    pcss_blocker_radius: f32 = 0.01,
    splits: [4]f32 = .{ 10.0, 26.0, 65.0, 150.0 },
    // Last computed cascade view-projections (exposed for debugging/tooling).
    matrices: [4]Mat4 = [_]Mat4{Mat4.identity} ** 4,

    // GPU shadow depth pass (CSM atlas + spot atlas).
    pass: passes.ShadowPass,

    pub fn init(allocator: std.mem.Allocator) ShadowSystem {
        return .{ .pass = passes.ShadowPass.init(allocator) };
    }

    pub fn deinit(self: *ShadowSystem) void {
        self.pass.deinit();
    }

    /// Computes the 4 cascade view-projections for `norm_light_dir` (a
    /// resolveSunDirection output) and caches them in `matrices`.
    pub fn computeCascades(self: *ShadowSystem, camera: Camera, aspect: f32, norm_light_dir: math.Vec3) [4]Mat4 {
        self.matrices = scene_cascades.computeCascades(camera, aspect, norm_light_dir, self.splits);
        return self.matrices;
    }

    /// Scene-level fragment uniform inputs (without the per-mesh
    /// receive_shadows flag, which the draw path patches per item). PCSS
    /// min/max penumbra stay at the ShadowState defaults, matching the
    /// legacy Scene.frameUniforms literals.
    pub fn uniformState(self: *const ShadowSystem, ground_color: Color3) scene_uniforms.ShadowState {
        return .{
            .ground_color = ground_color,
            .enable_shadows = self.enabled,
            .mesh_receive_shadows = true,
            .bias = self.bias,
            .intensity = self.intensity,
            .normal_bias = self.normal_bias,
            .softness = self.softness,
            .debug_cascades = self.debug_cascades,
            .splits = self.splits,
            .pcss_enabled = self.pcss_enabled,
            .pcss_light_size = self.pcss_light_size,
            .pcss_blocker_radius = self.pcss_blocker_radius,
        };
    }
};
