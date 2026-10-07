//! Chase camera: eases behind a target mesh or position. Split out of
//! `camera.zig` (facade): `camera.zig` re-exports the options and camera
//! types so the public API is unchanged. Leaf: imports `viewport` and the
//! mesh type only — never the facade and never sibling camera leaves.

const std = @import("std");
const sokol = @import("sokol");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const viewport_mod = @import("viewport.zig");
const Viewport = viewport_mod.Viewport;
const Mesh = @import("../mesh.zig").Mesh;

pub const FollowCameraOptions = struct {
    target_position: Vec3 = Vec3.zero,
    radius: f32 = 5.0,
    height_offset: f32 = 2.0,
    rotation_offset_deg: f32 = 0.0,
    fov_deg: f32 = 60.0,
    near: f32 = 0.1,
    far: f32 = 100.0,
    lerp_speed: f32 = 8.0,
    culling_mask: u32 = 0xFFFFFFFF,
    viewport: Viewport = .{},
};

pub const FollowCamera = struct {
    name: []const u8 = "FollowCamera",
    target_mesh: ?*Mesh = null,
    target_position: Vec3 = Vec3.zero,
    position: Vec3 = Vec3.zero,
    radius: f32 = 5.0,
    height_offset: f32 = 2.0,
    rotation_offset_deg: f32 = 0.0,
    fov_deg: f32 = 60.0,
    near: f32 = 0.1,
    far: f32 = 100.0,
    lerp_speed: f32 = 8.0, // 0 = instant snap
    culling_mask: u32 = 0xFFFFFFFF,
    viewport: Viewport = .{},

    pub fn init(name: []const u8, options: FollowCameraOptions) FollowCamera {
        var self = FollowCamera{
            .name = name,
            .target_position = options.target_position,
            .radius = options.radius,
            .height_offset = options.height_offset,
            .rotation_offset_deg = options.rotation_offset_deg,
            .fov_deg = options.fov_deg,
            .near = options.near,
            .far = options.far,
            .lerp_speed = options.lerp_speed,
            .culling_mask = options.culling_mask,
            .viewport = options.viewport,
        };
        self.position = self.desiredPosition();
        return self;
    }

    pub fn setTarget(self: *FollowCamera, mesh: ?*Mesh) void {
        self.target_mesh = mesh;
        if (mesh) |m| self.target_position = m.position;
    }

    fn desiredPosition(self: FollowCamera) Vec3 {
        const a = self.rotation_offset_deg * std.math.pi / 180.0;
        return Vec3.new(
            self.target_position.x + @sin(a) * self.radius,
            self.target_position.y + self.height_offset,
            self.target_position.z + @cos(a) * self.radius,
        );
    }

    pub fn handleEvent(self: *FollowCamera, ev: [*c]const sokol.app.Event) void {
        _ = self;
        _ = ev;
    }

    pub fn update(self: *FollowCamera, dt: f32) void {
        if (self.target_mesh) |m| self.target_position = m.position;
        const desired = self.desiredPosition();
        if (self.lerp_speed <= 0.0) {
            self.position = desired;
        } else {
            const t = @min(1.0, self.lerp_speed * dt);
            self.position = self.position.lerp(desired, t);
        }
    }

    pub fn getPosition(self: FollowCamera) Vec3 {
        return self.position;
    }

    pub fn getForward(self: FollowCamera) Vec3 {
        const fwd = self.target_position.sub(self.position);
        if (fwd.length() > 0.0001) return fwd.normalize();
        return Vec3.new(0, 0, -1);
    }

    pub fn getViewMatrix(self: FollowCamera) Mat4 {
        return Mat4.lookAt(self.position, self.target_position, Vec3.up);
    }

    pub fn getProjectionMatrix(self: FollowCamera, aspect: f32) Mat4 {
        return Mat4.perspective(self.fov_deg, aspect, self.near, self.far);
    }

    pub fn getViewProjection(self: FollowCamera, aspect: f32) Mat4 {
        return Mat4.mul(self.getProjectionMatrix(aspect), self.getViewMatrix());
    }

    pub fn getNear(self: FollowCamera) f32 {
        return self.near;
    }

    pub fn getFar(self: FollowCamera) f32 {
        return self.far;
    }

    pub fn getFovDeg(self: FollowCamera) f32 {
        return self.fov_deg;
    }

    pub fn getViewport(self: FollowCamera) Viewport {
        return self.viewport;
    }

    pub fn setViewport(self: *FollowCamera, vp: Viewport) void {
        self.viewport = vp;
    }

    pub fn getCullingMask(self: FollowCamera) u32 {
        return self.culling_mask;
    }

    pub fn setCullingMask(self: *FollowCamera, mask: u32) void {
        self.culling_mask = mask;
    }
};

test "FollowCamera snaps behind the target with lerp_speed 0" {
    var cam = FollowCamera.init("test", .{
        .target_position = Vec3.new(10.0, 0.0, 0.0),
        .radius = 5.0,
        .height_offset = 2.0,
        .lerp_speed = 0.0,
    });
    cam.position = Vec3.zero;
    cam.update(0.016);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), cam.position.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), cam.position.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), cam.position.z, 1e-5);
    const fwd = cam.getForward();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.x, 1e-5);
    try std.testing.expect(fwd.y < 0.0 and fwd.z < 0.0);
}
