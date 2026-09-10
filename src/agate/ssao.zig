const std = @import("std");

pub const SSAOConfig = struct {
    enabled: bool = true,
    radius: f32 = 0.5,
    bias: f32 = 0.035,
    intensity: f32 = 1.1,
    power: f32 = 1.2,
    debug_mode: bool = false,
};
