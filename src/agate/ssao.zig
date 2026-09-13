const std = @import("std");

/// Screen-space ambient occlusion pass settings. Lives flat on `Scene.ssao`
/// (tooling reads/writes it directly); consumed by the SSAO + blur passes.
pub const SSAOConfig = struct {
    enabled: bool = true,
    radius: f32 = 0.5,
    bias: f32 = 0.035,
    intensity: f32 = 1.1,
    power: f32 = 1.2,
    debug_mode: bool = false,
};
