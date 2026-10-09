const std = @import("std");
const PostProcessOptions = @import("options.zig").PostProcessOptions;

pub const CONTACT_SHADOWS_STEPS_MIN: u32 = 4;
pub const CONTACT_SHADOWS_STEPS_MAX: u32 = 32;
pub const CONTACT_SHADOWS_DISTANCE_MIN: f32 = 0.01;
pub const CONTACT_SHADOWS_THICKNESS_MIN: f32 = 0.001;

pub fn contactShadowsActive(config: PostProcessOptions) bool {
    return config.contact_shadows_enabled and config.contact_shadows_intensity > 0.001;
}

pub fn contactShadowParams(config: PostProcessOptions) [4]f32 {
    if (!contactShadowsActive(config)) {
        return .{ 0.0, 0.0, 0.0, 0.0 };
    }
    return .{
        1.0,
        config.contact_shadows_intensity,
        config.contact_shadows_distance,
        config.contact_shadows_thickness,
    };
}

/// Packed second uniform lane: (steps, 0, 0, 0). Zeros when inactive, so the
/// composite stays bit-identical to the no-contact-shadow path (the shader
/// early-outs on contactShadowParams.x before reading the steps lane).
pub fn contactShadowStepsParams(config: PostProcessOptions) [4]f32 {
    if (!contactShadowsActive(config)) {
        return .{ 0.0, 0.0, 0.0, 0.0 };
    }
    return .{ @floatFromInt(config.contact_shadows_steps), 0.0, 0.0, 0.0 };
}

/// Compute shadow attenuation factor: 1.0 = fully lit, 0.0 = fully in shadow.
pub fn calcContactShadowAttenuation(occlusion: f32, intensity: f32, n_dot_l: f32) f32 {
    if (n_dot_l <= 0.0) return 1.0;
    const factor = occlusion * intensity * n_dot_l;
    return std.math.clamp(1.0 - factor, 0.0, 1.0);
}
