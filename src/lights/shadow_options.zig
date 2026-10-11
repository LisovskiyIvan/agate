//! Shared shadow-casting options for point and spot lights. Leaf of the
//! `lights.zig` facade; see the facade header for the module map and the
//! anti-cycle rule.

pub const LightShadowOptions = struct {
    cast_shadows: bool = false,
    shadow_bias: f32 = 0.002,
    shadow_normal_bias: f32 = 0.005,
    shadow_near: f32 = 0.1,
    /// Emitter radius for PCSS (percentage-closer soft shadows):
    /// doubles as the light_size of the sun-style penumbra formula
    /// (see scene/shadow_pcss.zig penumbraRadius). 0.0 (default) disables
    /// PCSS and keeps the legacy fixed 4-tap PCF bit-identical; any value
    /// above the epsilon enables the blocker search + variable penumbra.
    /// Rides the `point/spot_shadow_params[i].w` lane (packed in
    /// light_rig.zig).
    source_radius: f32 = 0.0,
};
