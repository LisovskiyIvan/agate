const std = @import("std");

/// Screen-space ambient occlusion pass settings. Lives flat on `Scene.ssao`
/// (tooling reads/writes it directly); consumed by the SSAO + blur passes.
pub const SSAOOptions = struct {
    enabled: bool = true,
    radius: f32 = 0.5,
    bias: f32 = 0.035,
    intensity: f32 = 1.1,
    power: f32 = 1.2,
    sample_count: u32 = 24,
    debug_mode: bool = false,

    pub fn clamped(self: SSAOOptions) SSAOOptions {
        var out = self;
        out.radius = @max(self.radius, 0.0);
        out.bias = @max(self.bias, 0.0);
        out.intensity = @max(self.intensity, 0.0);
        out.power = @max(self.power, 0.0);
        out.sample_count = std.math.clamp(self.sample_count, 4, 32);
        return out;
    }
};

test "ssao options default and clamping" {
    const def = SSAOOptions{};
    try std.testing.expectEqual(@as(u32, 24), def.sample_count);

    const custom = (SSAOOptions{ .sample_count = 100 }).clamped();
    try std.testing.expectEqual(@as(u32, 32), custom.sample_count);

    const low = (SSAOOptions{ .sample_count = 1 }).clamped();
    try std.testing.expectEqual(@as(u32, 4), low.sample_count);
}

