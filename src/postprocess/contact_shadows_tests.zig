const std = @import("std");
const PostProcessOptions = @import("options.zig").PostProcessOptions;
const cs = @import("contact_shadows.zig");
const contactShadowsActive = cs.contactShadowsActive;
const contactShadowParams = cs.contactShadowParams;
const calcContactShadowAttenuation = cs.calcContactShadowAttenuation;

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

test "contactShadowStepsParams packs steps or zeros when disabled" {
    const cfg = PostProcessOptions{
        .contact_shadows_enabled = true,
        .contact_shadows_intensity = 0.5,
        .contact_shadows_steps = 20,
    };
    try std.testing.expectEqual([4]f32{ 20.0, 0.0, 0.0, 0.0 }, cs.contactShadowStepsParams(cfg));

    // Disabled or ~zero intensity: all-zero lane (composite bit-identical).
    const off = PostProcessOptions{ .contact_shadows_enabled = false, .contact_shadows_steps = 20 };
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, cs.contactShadowStepsParams(off));
    const quiet = PostProcessOptions{ .contact_shadows_enabled = true, .contact_shadows_intensity = 0.0 };
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, cs.contactShadowStepsParams(quiet));
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
