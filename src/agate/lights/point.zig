//! Point light (omnidirectional, optional cube-face shadows). Leaf of the
//! `lights.zig` facade; see the facade header for the module map and the
//! anti-cycle rule.
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Color3 = math.Color3;

pub const PointLightOptions = struct {
    position: Vec3 = Vec3.zero,
    color: Color3 = Color3.white,
    intensity: f32 = 1.0,
    range: f32 = 10.0,
    cast_shadows: bool = false,
    shadow_bias: f32 = 0.002,
    shadow_normal_bias: f32 = 0.005,
    shadow_near: f32 = 0.1,
};

pub const PointLight = struct {
    name: []const u8 = "PointLight",
    /// See DirectionalLight.owns_name.
    owns_name: bool = false,
    position: Vec3 = Vec3.zero,
    color: Color3 = Color3.white,
    intensity: f32 = 1.0,
    range: f32 = 10.0,
    is_enabled: bool = true,
    cast_shadows: bool = false,
    shadow_bias: f32 = 0.002,
    shadow_normal_bias: f32 = 0.005,
    shadow_near: f32 = 0.1,

    pub fn init(name: []const u8, options: PointLightOptions) PointLight {
        return .{
            .name = name,
            .position = options.position,
            .color = options.color,
            .intensity = options.intensity,
            .range = options.range,
            .cast_shadows = options.cast_shadows,
            .shadow_bias = options.shadow_bias,
            .shadow_normal_bias = options.shadow_normal_bias,
            .shadow_near = options.shadow_near,
        };
    }

    /// Cube face order for point shadow tiles (matches the shader
    /// pointFaceIndex and ShadowPass.pointFaceForDir): +X, -X, +Y, -Y, +Z, -Z.
    pub const shadow_face_count: usize = 6;

    /// Computes the light view-projection matrix for one cube face of the
    /// point shadow atlas (90-degree perspective, aspect 1). The tile layout
    /// lives in passes/shadow_pass.zig (pointTileOrigin).
    pub fn getShadowFaceViewProj(self: PointLight, face: usize) Mat4 {
        const eye = self.position;
        const dirs = [_]Vec3{
            Vec3.new(1, 0, 0),
            Vec3.new(-1, 0, 0),
            Vec3.new(0, 1, 0),
            Vec3.new(0, -1, 0),
            Vec3.new(0, 0, 1),
            Vec3.new(0, 0, -1),
        };
        const ups = [_]Vec3{
            Vec3.up,
            Vec3.up,
            Vec3.new(0, 0, -1),
            Vec3.new(0, 0, 1),
            Vec3.up,
            Vec3.up,
        };
        const f = face % shadow_face_count;
        const view = Mat4.lookAt(eye, eye.add(dirs[f]), ups[f]);
        const near = @max(self.shadow_near, 0.05);
        const far = @max(self.range, near + 0.1);
        const proj = Mat4.perspective(90.0, 1.0, near, far);
        return Mat4.mul(proj, view);
    }
};
