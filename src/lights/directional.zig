//! Directional light (sun + shadowless fills). Leaf of the `lights.zig`
//! facade; see the facade header for the module map and the anti-cycle rule.
const math = @import("math");
const Vec3 = math.Vec3;
const Color3 = math.Color3;

pub const DirectionalLightOptions = struct {
    direction: Vec3 = Vec3.new(0.5, 1.0, 0.5),
    diffuse: Color3 = Color3.white,
    intensity: f32 = 1.0,
};

/// Maximum simultaneous directional lights (Babylon.js parity): index 0 is
/// the shadow-casting sun (CSM unchanged), indices 1..3 are shadowless
/// fills. See LightRig.addDirectionalLight for the creation cap.
pub const max_directional_lights: usize = 4;
/// Fills beyond the primary sun (slots 1..3).
pub const max_fill_directionals: usize = max_directional_lights - 1;

pub const DirectionalLight = struct {
    name: []const u8 = "DirectionalLight",
    /// True when `name` was heap-allocated by the glTF loader; the scene frees
    /// it on deinit/replacement. Programmatic lights keep string literals.
    owns_name: bool = false,
    direction: Vec3 = Vec3.new(0.5, 1.0, 0.5),
    diffuse: Color3 = Color3.white,
    intensity: f32 = 1.0,
    /// Disabled lights pack as zeroed slots (no contribution) and the sun
    /// resolvers below treat them as absent (hemispheric fallback).
    is_enabled: bool = true,

    pub fn init(name: []const u8, options: DirectionalLightOptions) DirectionalLight {
        return .{
            .name = name,
            .direction = options.direction.normalize(),
            .diffuse = options.diffuse,
            .intensity = options.intensity,
        };
    }
};
