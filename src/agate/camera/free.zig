//! WASD + mouse-drag free camera. Split out of `camera.zig` (facade):
//! `camera.zig` re-exports the options and camera types so the public API
//! is unchanged. Leaf: imports `viewport` only — never the facade and
//! never sibling camera leaves.
//!
//! Rotation convention (degrees): x = pitch (positive looks up),
//! y = yaw around Y, z unused. Yaw 0 + pitch 0 faces -Z.

const std = @import("std");
const sokol = @import("sokol");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const viewport_mod = @import("viewport.zig");
const Viewport = viewport_mod.Viewport;

pub const FreeCameraOptions = struct {
    position: Vec3 = Vec3.zero,
    rotation: Vec3 = Vec3.zero,
    fov_deg: f32 = 60.0,
    near: f32 = 0.1,
    far: f32 = 100.0,
    speed: f32 = 6.0,
    angular_sensitivity: f32 = 0.25,
    inertia: f32 = 0.0, // Damping factor (0.0 = instant, e.g. 0.85 = smooth damping)
    culling_mask: u32 = 0xFFFFFFFF,
    viewport: Viewport = .{},
};

pub const FreeCamera = struct {
    name: []const u8 = "FreeCamera",
    position: Vec3 = Vec3.zero,
    rotation: Vec3 = Vec3.zero,
    fov_deg: f32 = 60.0,
    near: f32 = 0.1,
    far: f32 = 100.0,
    speed: f32 = 6.0,
    angular_sensitivity: f32 = 0.25, // Degrees per pixel
    inertia: f32 = 0.0,
    inertial_rotation_x: f32 = 0.0,
    inertial_rotation_y: f32 = 0.0,
    culling_mask: u32 = 0xFFFFFFFF,
    viewport: Viewport = .{},

    move_forward: bool = false,
    move_back: bool = false,
    move_left: bool = false,
    move_right: bool = false,
    move_up: bool = false,
    move_down: bool = false,

    is_dragging: bool = false,
    last_mouse_x: f32 = 0.0,
    last_mouse_y: f32 = 0.0,

    pub fn init(name: []const u8, options: FreeCameraOptions) FreeCamera {
        return .{
            .name = name,
            .position = options.position,
            .rotation = options.rotation,
            .fov_deg = options.fov_deg,
            .near = options.near,
            .far = options.far,
            .speed = options.speed,
            .angular_sensitivity = options.angular_sensitivity,
            .inertia = options.inertia,
            .culling_mask = options.culling_mask,
            .viewport = options.viewport,
        };
    }

    fn setMoveFlag(self: *FreeCamera, key_code: sokol.app.Keycode, pressed: bool) void {
        switch (key_code) {
            .W, .UP => self.move_forward = pressed,
            .S, .DOWN => self.move_back = pressed,
            .A, .LEFT => self.move_left = pressed,
            .D, .RIGHT => self.move_right = pressed,
            .SPACE, .E => self.move_up = pressed,
            .LEFT_CONTROL, .Q => self.move_down = pressed,
            else => {},
        }
    }

    pub fn handleEvent(self: *FreeCamera, ev: [*c]const sokol.app.Event) void {
        switch (ev.*.type) {
            .MOUSE_DOWN => {
                if (ev.*.mouse_button == .LEFT) {
                    self.is_dragging = true;
                    self.last_mouse_x = ev.*.mouse_x;
                    self.last_mouse_y = ev.*.mouse_y;
                }
            },
            .MOUSE_UP => {
                if (ev.*.mouse_button == .LEFT) {
                    self.is_dragging = false;
                }
            },
            .MOUSE_MOVE => {
                if (self.is_dragging) {
                    const dx = ev.*.mouse_x - self.last_mouse_x;
                    const dy = ev.*.mouse_y - self.last_mouse_y;
                    self.last_mouse_x = ev.*.mouse_x;
                    self.last_mouse_y = ev.*.mouse_y;

                    if (self.inertia <= 0.0) {
                        self.rotation.y -= dx * self.angular_sensitivity;
                        self.rotation.x -= dy * self.angular_sensitivity;
                        self.rotation.x = std.math.clamp(self.rotation.x, -89.0, 89.0);
                    } else {
                        self.inertial_rotation_y += dx * self.angular_sensitivity;
                        self.inertial_rotation_x += dy * self.angular_sensitivity;
                    }
                }
            },
            .KEY_DOWN => self.setMoveFlag(ev.*.key_code, true),
            .KEY_UP => self.setMoveFlag(ev.*.key_code, false),
            else => {},
        }
    }

    pub fn update(self: *FreeCamera, dt: f32) void {
        if (self.inertia > 0.0) {
            self.rotation.y -= self.inertial_rotation_y;
            self.rotation.x -= self.inertial_rotation_x;
            self.rotation.x = std.math.clamp(self.rotation.x, -89.0, 89.0);

            const clamped_inertia = std.math.clamp(self.inertia, 0.0, 0.999);
            const decay = std.math.pow(f32, clamped_inertia, dt * 60.0);
            self.inertial_rotation_x *= decay;
            self.inertial_rotation_y *= decay;

            if (@abs(self.inertial_rotation_x) < 0.0001) self.inertial_rotation_x = 0.0;
            if (@abs(self.inertial_rotation_y) < 0.0001) self.inertial_rotation_y = 0.0;
        }

        const yaw_rad = self.rotation.y * std.math.pi / 180.0;
        const fwd = Vec3.new(-@sin(yaw_rad), 0.0, -@cos(yaw_rad));
        const right = fwd.cross(Vec3.up).normalize();

        const fb: f32 = (if (self.move_forward) @as(f32, 1.0) else 0.0) - (if (self.move_back) @as(f32, 1.0) else 0.0);
        const rl: f32 = (if (self.move_right) @as(f32, 1.0) else 0.0) - (if (self.move_left) @as(f32, 1.0) else 0.0);
        const ud: f32 = (if (self.move_up) @as(f32, 1.0) else 0.0) - (if (self.move_down) @as(f32, 1.0) else 0.0);

        var move = fwd.scale(fb).add(right.scale(rl)).add(Vec3.up.scale(ud));
        if (move.length() > 0.0001) {
            move = move.normalize().scale(self.speed * dt);
            self.position = self.position.add(move);
        }
    }

    pub fn getPosition(self: FreeCamera) Vec3 {
        return self.position;
    }

    pub fn getForward(self: FreeCamera) Vec3 {
        const yaw_rad = self.rotation.y * std.math.pi / 180.0;
        const pitch_rad = self.rotation.x * std.math.pi / 180.0;
        const cp = @cos(pitch_rad);
        const fwd = Vec3.new(-@sin(yaw_rad) * cp, @sin(pitch_rad), -@cos(yaw_rad) * cp);
        if (fwd.length() > 0.0001) return fwd.normalize();
        return Vec3.new(0, 0, -1);
    }

    pub fn getViewMatrix(self: FreeCamera) Mat4 {
        return Mat4.lookAt(self.position, self.position.add(self.getForward()), Vec3.up);
    }

    pub fn getProjectionMatrix(self: FreeCamera, aspect: f32) Mat4 {
        return Mat4.perspective(self.fov_deg, aspect, self.near, self.far);
    }

    pub fn getViewProjection(self: FreeCamera, aspect: f32) Mat4 {
        return Mat4.mul(self.getProjectionMatrix(aspect), self.getViewMatrix());
    }

    pub fn getNear(self: FreeCamera) f32 {
        return self.near;
    }

    pub fn getFar(self: FreeCamera) f32 {
        return self.far;
    }

    pub fn getFovDeg(self: FreeCamera) f32 {
        return self.fov_deg;
    }

    pub fn getViewport(self: FreeCamera) Viewport {
        return self.viewport;
    }

    pub fn setViewport(self: *FreeCamera, vp: Viewport) void {
        self.viewport = vp;
    }

    pub fn getCullingMask(self: FreeCamera) u32 {
        return self.culling_mask;
    }

    pub fn setCullingMask(self: *FreeCamera, mask: u32) void {
        self.culling_mask = mask;
    }
};

test "FreeCamera faces -Z at zero yaw and pitch" {
    const cam = FreeCamera.init("test", .{});
    const fwd = cam.getForward();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), fwd.z, 1e-5);
}

test "FreeCamera view matrix maps its position to the origin" {
    const cam = FreeCamera.init("test", .{ .position = Vec3.new(1.0, 2.0, 5.0) });
    const p = cam.getViewMatrix().transformPoint(cam.getPosition());
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), p.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), p.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), p.z, 1e-4);
}

test "FreeCamera projects the look-at point to the screen center" {
    const cam = FreeCamera.init("test", .{ .position = Vec3.new(0.0, 0.0, 5.0) });
    const vp = cam.getViewProjection(800.0 / 600.0);
    const p = vp.projectPoint(Vec3.zero, 800.0, 600.0) orelse return error.PointBehindCamera;
    try std.testing.expectApproxEqAbs(@as(f32, 400.0), p.x, 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 300.0), p.y, 1e-2);
}

test "FreeCamera moves forward relative to yaw" {
    var cam = FreeCamera.init("test", .{});
    cam.move_forward = true;
    cam.update(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cam.position.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cam.position.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -6.0), cam.position.z, 1e-5);

    var strafe = FreeCamera.init("strafe", .{ .rotation = Vec3.new(0.0, 90.0, 0.0) });
    strafe.move_right = true;
    strafe.update(1.0);
    // Yaw 90 faces -X, so its right hand points toward -Z.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), strafe.position.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), strafe.position.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -6.0), strafe.position.z, 1e-4);
}

test "FreeCamera inertia enables smooth rotation damping" {
    var cam = FreeCamera.init("free", .{ .inertia = 0.85 });
    var down_ev = sokol.app.Event{
        .type = .MOUSE_DOWN,
        .mouse_button = .LEFT,
        .mouse_x = 200.0,
        .mouse_y = 200.0,
    };
    cam.handleEvent(&down_ev);

    var move_ev = sokol.app.Event{
        .type = .MOUSE_MOVE,
        .mouse_x = 220.0,
        .mouse_y = 210.0,
    };
    cam.handleEvent(&move_ev);

    try std.testing.expectEqual(@as(f32, 0.0), cam.rotation.x);
    try std.testing.expectEqual(@as(f32, 0.0), cam.rotation.y);
    try std.testing.expect(cam.inertial_rotation_y != 0.0);

    cam.update(1.0 / 60.0);
    try std.testing.expect(cam.rotation.y != 0.0);
    const rot1 = cam.rotation.y;
    cam.update(1.0 / 60.0);
    try std.testing.expect(cam.rotation.y < rot1);
}
