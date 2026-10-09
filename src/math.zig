pub const vec = @import("math/vec.zig");
pub const Vec2 = vec.Vec2;
pub const Vec3 = vec.Vec3;
pub const Vec4 = vec.Vec4;

pub const color = @import("math/color.zig");
pub const Color3 = color.Color3;
pub const Color4 = color.Color4;

pub const mat4 = @import("math/mat4.zig");
pub const Mat4 = mat4.Mat4;

pub const quat = @import("math/quat.zig");
pub const Quat = quat.Quat;

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

// math is its own build module (name "math" in build.zig), so the engine's
// tests.zig cannot import these files by path; they are aggregated here and
// reach the suite when tests.zig imports the module inside a test block.
test {
    _ = @import("math/vec.zig");
    _ = @import("math/vec_tests.zig");
    _ = @import("math/color.zig");
    _ = @import("math/color_tests.zig");
    _ = @import("math/mat4.zig");
    _ = @import("math/mat4_tests.zig");
    _ = @import("math/quat.zig");
    _ = @import("math/quat_tests.zig");
    _ = @import("math/bounding_box.zig");
    _ = @import("math/bounding_box_tests.zig");
    _ = @import("math/frustum.zig");
    _ = @import("math/frustum_tests.zig");
    _ = @import("math/ray.zig");
    _ = @import("math/ray_tests.zig");
}
