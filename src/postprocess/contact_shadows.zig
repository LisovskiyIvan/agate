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

/// Compute shadow attenuation factor: 1.0 = fully lit, 0.0 = fully in shadow.
pub fn calcContactShadowAttenuation(occlusion: f32, intensity: f32, n_dot_l: f32) f32 {
    if (n_dot_l <= 0.0) return 1.0;
    const factor = occlusion * intensity * n_dot_l;
    return std.math.clamp(1.0 - factor, 0.0, 1.0);
}

test "contactShadowsActive returns true only when enabled and intensity > 0" {
    var cfg = PostProcessOptions{};
    try std.testing.expect(!contactShadowsActive(cfg));

    cfg.contact_shadows_enabled = true;
    cfg.contact_shadows_intensity = 0.5;
    try std.testing.expect(contactShadowsActive(cfg));

    cfg.contact_shadows_intensity = 0.0;
    try std.testing.expect(!contactShadowsActive(cfg));
}

test "contactShadowParams packs uniforms or zeros when disabled" {
    var cfg = PostProcessOptions{
        .contact_shadows_enabled = true,
        .contact_shadows_intensity = 0.75,
        .contact_shadows_distance = 0.4,
        .contact_shadows_thickness = 0.06,
    };
    const params = contactShadowParams(cfg);
    try std.testing.expectEqual(@as(f32, 1.0), params[0]);
    try std.testing.expectEqual(@as(f32, 0.75), params[1]);
    try std.testing.expectEqual(@as(f32, 0.4), params[2]);
    try std.testing.expectEqual(@as(f32, 0.06), params[3]);

    cfg.contact_shadows_enabled = false;
    const disabled_params = contactShadowParams(cfg);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, disabled_params);
}

test "calcContactShadowAttenuation scales by n_dot_l and intensity" {
    // Surface facing away from light receives no contact shadow darkening
    try std.testing.expectEqual(@as(f32, 1.0), calcContactShadowAttenuation(1.0, 1.0, 0.0));
    try std.testing.expectEqual(@as(f32, 1.0), calcContactShadowAttenuation(1.0, 1.0, -0.5));

    // Surface perpendicular to light (n_dot_l = 1.0), 100% occlusion, 50% intensity
    const atten = calcContactShadowAttenuation(1.0, 0.5, 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), atten, 1e-6);

    // Glancing angle (n_dot_l = 0.2)
    const glancing = calcContactShadowAttenuation(1.0, 0.5, 0.2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), glancing, 1e-6);
}
