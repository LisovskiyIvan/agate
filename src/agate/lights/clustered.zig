//! Clustered forward-pool point light (unshadowed value type). Leaf of the
//! `lights.zig` facade; see the facade header for the module map and the
//! anti-cycle rule.
const math = @import("math");
const Vec3 = math.Vec3;
const Color3 = math.Color3;

/// Maximum EXTRA point lights in the clustered forward pool (wave 30, v1).
/// These ride OUTSIDE the legacy 4-slot top-k lanes (LightRig.point_slots):
/// every owned clustered light packs verbatim into LightRig.FramePack and
/// the 2D screen-tile build culls them per tile. See
/// LightRig.addClusteredPointLight for the creation cap and
/// scene/clustered_lights.zig for the tiling design. No shadows in v1
/// (unshadowed by design, documented there).
pub const max_clustered_lights: usize = 64;

pub const ClusteredPointLightOptions = struct {
    color: Color3 = Color3.white,
    intensity: f32 = 1.0,
    /// Influence radius in world units (<= 0 packs as absent: the tile
    /// build skips the light and the shader range-gates it like a legacy
    /// lane with zero range).
    radius: f32 = 10.0,
    enabled: bool = true,
};

/// One extra forward point light for the clustered pool. Value type (no
/// name, no heap, no shadows in v1): LightRig owns a fixed array of these
/// plus a count, Scene exposes index-based add/remove/get/count, and the
/// GPU tile build reads the staged FramePack copy (1-frame lag).
pub const ClusteredPointLight = struct {
    position: Vec3 = Vec3.zero,
    color: Color3 = Color3.white,
    intensity: f32 = 1.0,
    radius: f32 = 10.0,
    /// Disabled lights pack as zeroed lanes (no contribution), mirroring
    /// the directional-fill contract.
    is_enabled: bool = true,

    pub fn init(position: Vec3, options: ClusteredPointLightOptions) ClusteredPointLight {
        return .{
            .position = position,
            .color = options.color,
            .intensity = options.intensity,
            .radius = options.radius,
            .is_enabled = options.enabled,
        };
    }
};
