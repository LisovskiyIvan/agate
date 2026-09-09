pub const vec = @import("math/vec.zig");
pub const Vec2 = vec.Vec2;
pub const Vec3 = vec.Vec3;
pub const Vec4 = vec.Vec4;

pub const color = @import("math/color.zig");
pub const Color3 = color.Color3;
pub const Color4 = color.Color4;

pub const mat4 = @import("math/mat4.zig");
pub const Mat4 = mat4.Mat4;

pub const bounding_box = @import("math/bounding_box.zig");
pub const BoundingBox = bounding_box.BoundingBox;

pub const frustum = @import("math/frustum.zig");
pub const Frustum = frustum.Frustum;
pub const FrustumPlane = frustum.FrustumPlane;

pub const ray = @import("math/ray.zig");
pub const Ray = ray.Ray;
pub const RayHit = ray.RayHit;
pub const TriangleHit = ray.TriangleHit;

pub inline fn lerp(a: f32, b: f32, t: f32) f32 {
    return a + (b - a) * t;
}
