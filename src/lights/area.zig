//! Rect area light. Leaf of the `lights.zig` facade; see the facade header
//! for the module map and the anti-cycle rule.
const math = @import("math");
const Vec3 = math.Vec3;
const Color3 = math.Color3;

pub const AreaLightOptions = struct {
    center: Vec3 = Vec3.zero,
    /// Local +X half-extent vector: direction AND half-width encoded in one
    /// vector (corner = center +/- right +/- up). Must be non-zero and
    /// non-parallel to `up`; degenerate inputs emit nothing (see normal()).
    right: Vec3 = Vec3.new(0.5, 0.0, 0.0),
    /// Local +Y half-extent vector (direction and half-height).
    up: Vec3 = Vec3.new(0.0, 0.5, 0.0),
    color: Color3 = Color3.white,
    intensity: f32 = 1.0,
    is_enabled: bool = true,
};

/// Maximum simultaneous rect area lights (wave 26, v1). See
/// LightRig.addAreaLight for the creation cap.
pub const max_area_lights: usize = 2;

pub const AreaLight = struct {
    name: []const u8 = "AreaLight",
    /// True when `name` was heap-allocated; the scene frees it on
    /// deinit/removal. Programmatic lights keep string literals.
    owns_name: bool = false,
    center: Vec3 = Vec3.zero,
    right: Vec3 = Vec3.new(0.5, 0.0, 0.0),
    up: Vec3 = Vec3.new(0.0, 0.5, 0.0),
    color: Color3 = Color3.white,
    intensity: f32 = 1.0,
    /// Disabled lights pack as zeroed lanes (no contribution).
    is_enabled: bool = true,

    pub fn init(name: []const u8, options: AreaLightOptions) AreaLight {
        return .{
            .name = name,
            .center = options.center,
            .right = options.right,
            .up = options.up,
            .color = options.color,
            .intensity = options.intensity,
            .is_enabled = options.is_enabled,
        };
    }

    /// Emitting-face normal: normalize(cross(right, up)). Zero when the
    /// rect is degenerate (zero-area or parallel axes) — the shader then
    /// contributes nothing (area gate), and CPU code must treat zero as
    /// "no emission" rather than normalizing it into NaN.
    pub fn normal(self: AreaLight) Vec3 {
        const n = self.right.cross(self.up);
        if (n.lengthSq() <= 1e-12) return Vec3.zero;
        return n.normalize();
    }

    /// Emitting area (4 * |right x up|); zero for degenerate rects.
    pub fn area(self: AreaLight) f32 {
        return 4.0 * self.right.cross(self.up).length();
    }
};
