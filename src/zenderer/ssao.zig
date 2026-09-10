const std = @import("std");

pub const SSAOConfig = struct {
    enabled: bool = true,
    radius: f32 = 0.5,
    bias: f32 = 0.025,
    intensity: f32 = 1.6,
    power: f32 = 1.5,
    debug_mode: bool = false,
};
