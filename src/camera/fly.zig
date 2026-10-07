//! Six-degrees-of-freedom flight camera with roll. Split out of
//! `camera.zig` (facade): `camera.zig` re-exports the options and camera
//! types so the public API is unchanged. Leaf: imports `viewport` only —
//! never the facade and never sibling camera leaves.
//!
//! Rotation convention (degrees): x = pitch (positive looks up, clamped
//! to [-89, 89]), y = yaw around Y, z = roll around the view axis.
//! Yaw 0 + pitch 0 + roll 0 faces -Z with up +Y (matches FreeCamera).
//! Unlike FreeCamera, flight follows the true 3D forward (pitch included)
//! and the strafe/up axes tilt with the roll.

const std = @import("std");
const sokol = @import("sokol");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const viewport_mod = @import("viewport.zig");
const Viewport = viewport_mod.Viewport;

pub const FlyCameraOptions = struct {
    position: Vec3 = Vec3.zero,
    rotation: Vec3 = Vec3.zero,
    fov_deg: f32 = 60.0,
    near: f32 = 0.1,
    far: f32 = 100.0,
    speed: f32 = 6.0,
    boost_multiplier: f32 = 4.0,
    angular_sensitivity: f32 = 0.25,
    roll_speed_deg: f32 = 90.0, // Degrees per second while Q/E held
    inertia: f32 = 0.0, // Damping factor (0.0 = instant, e.g. 0.85 = smooth damping)
    culling_mask: u32 = 0xFFFFFFFF,
    viewport: Viewport = .{},
};

pub const FlyCamera = struct {
    name: []const u8 = "FlyCamera",
    position: Vec3 = Vec3.zero,
    rotation: Vec3 = Vec3.zero,
    fov_deg: f32 = 60.0,
    near: f32 = 0.1,
    far: f32 = 100.0,
    speed: f32 = 6.0,
    boost_multiplier: f32 = 4.0,
    angular_sensitivity: f32 = 0.25, // Degrees per pixel
    roll_speed_deg: f32 = 90.0,
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
    roll_left: bool = false,
    roll_right: bool = false,
    boost_held: bool = false,

    is_dragging: bool = false,
    last_mouse_x: f32 = 0.0,
    last_mouse_y: f32 = 0.0,

    pub fn init(name: []const u8, options: FlyCameraOptions) FlyCamera {
        return .{
            .name = name,
            .position = options.position,
            .rotation = options.rotation,
            .fov_deg = options.fov_deg,
            .near = options.near,
            .far = options.far,
            .speed = options.speed,
            .boost_multiplier = options.boost_multiplier,
            .angular_sensitivity = options.angular_sensitivity,
            .roll_speed_deg = options.roll_speed_deg,
            .inertia = options.inertia,
            .culling_mask = options.culling_mask,
            .viewport = options.viewport,
        };
    }

    fn setMoveFlag(self: *FlyCamera, key_code: sokol.app.Keycode, pressed: bool) void {
        switch (key_code) {
            .W, .UP => self.move_forward = pressed,
            .S, .DOWN => self.move_back = pressed,
            .A, .LEFT => self.move_left = pressed,
            .D, .RIGHT => self.move_right = pressed,
            .R => self.move_up = pressed,
            .F => self.move_down = pressed,
            .Q => self.roll_left = pressed,
            .E => self.roll_right = pressed,
            .LEFT_SHIFT, .RIGHT_SHIFT => self.boost_held = pressed,
            else => {},
        }
    }

    pub fn handleEvent(self: *FlyCamera, ev: [*c]const sokol.app.Event) void {
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

    pub fn update(self: *FlyCamera, dt: f32) void {
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

        const roll_in: f32 = (if (self.roll_right) @as(f32, 1.0) else 0.0) - (if (self.roll_left) @as(f32, 1.0) else 0.0);
        self.rotation.z += roll_in * self.roll_speed_deg * dt;

        const fb: f32 = (if (self.move_forward) @as(f32, 1.0) else 0.0) - (if (self.move_back) @as(f32, 1.0) else 0.0);
        const rl: f32 = (if (self.move_right) @as(f32, 1.0) else 0.0) - (if (self.move_left) @as(f32, 1.0) else 0.0);
        const ud: f32 = (if (self.move_up) @as(f32, 1.0) else 0.0) - (if (self.move_down) @as(f32, 1.0) else 0.0);

        var move = self.getForward().scale(fb).add(self.getRight().scale(rl)).add(self.getUp().scale(ud));
        if (move.length() > 0.0001) {
            const eff_speed = self.speed * (if (self.boost_held) self.boost_multiplier else 1.0);
            move = move.normalize().scale(eff_speed * dt);
            self.position = self.position.add(move);
        }
    }

    pub fn getPosition(self: FlyCamera) Vec3 {
        return self.position;
    }

    // Roll-invariant by construction: rolling spins around this axis.
    pub fn getForward(self: FlyCamera) Vec3 {
        const yaw_rad = self.rotation.y * std.math.pi / 180.0;
        const pitch_rad = self.rotation.x * std.math.pi / 180.0;
        const cp = @cos(pitch_rad);
        const fwd = Vec3.new(-@sin(yaw_rad) * cp, @sin(pitch_rad), -@cos(yaw_rad) * cp);
        if (fwd.length() > 0.0001) return fwd.normalize();
        return Vec3.new(0, 0, -1);
    }

    fn flatRight(self: FlyCamera) Vec3 {
        const fwd = self.getForward();
        var right = fwd.cross(Vec3.up);
        if (right.length() < 0.0001) {
            // Pitch at exactly +-90 (only possible via init): fall back to
            // the yaw-only right hand, which never degenerates.
            const yaw_rad = self.rotation.y * std.math.pi / 180.0;
            right = Vec3.new(-@sin(yaw_rad), 0.0, -@cos(yaw_rad)).cross(Vec3.up);
        }
        return right.normalize();
    }

    pub fn getRight(self: FlyCamera) Vec3 {
        const roll_rad = self.rotation.z * std.math.pi / 180.0;
        const rf = self.flatRight();
        const up0 = rf.cross(self.getForward()).normalize();
        // Rotation of (right, up) in their own plane around the forward
        // axis; preserves orthonormality and handedness (right x up = -fwd).
        return rf.scale(@cos(roll_rad)).add(up0.scale(@sin(roll_rad)));
    }

    pub fn getUp(self: FlyCamera) Vec3 {
        const roll_rad = self.rotation.z * std.math.pi / 180.0;
        const rf = self.flatRight();
        const up0 = rf.cross(self.getForward()).normalize();
        return up0.scale(@cos(roll_rad)).sub(rf.scale(@sin(roll_rad)));
    }

    pub fn getViewMatrix(self: FlyCamera) Mat4 {
        return Mat4.lookAt(self.position, self.position.add(self.getForward()), self.getUp());
    }

    pub fn getProjectionMatrix(self: FlyCamera, aspect: f32) Mat4 {
        return Mat4.perspective(self.fov_deg, aspect, self.near, self.far);
    }

    pub fn getViewProjection(self: FlyCamera, aspect: f32) Mat4 {
        return Mat4.mul(self.getProjectionMatrix(aspect), self.getViewMatrix());
    }

    pub fn getNear(self: FlyCamera) f32 {
        return self.near;
    }

    pub fn getFar(self: FlyCamera) f32 {
        return self.far;
    }

    pub fn getFovDeg(self: FlyCamera) f32 {
        return self.fov_deg;
    }

    pub fn getViewport(self: FlyCamera) Viewport {
        return self.viewport;
    }

    pub fn setViewport(self: *FlyCamera, vp: Viewport) void {
        self.viewport = vp;
    }

    pub fn getCullingMask(self: FlyCamera) u32 {
        return self.culling_mask;
    }

    pub fn setCullingMask(self: *FlyCamera, mask: u32) void {
        self.culling_mask = mask;
    }
};

test "FlyCamera forward is roll-invariant, pitch 90 looks straight up" {
    const cam = FlyCamera.init("fly", .{ .rotation = Vec3.new(90.0, 90.0, 90.0) });
    const fwd = cam.getForward();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), fwd.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.z, 1e-4);
    // Yaw 90 alone faces -X, whatever the roll is.
    const yawed = FlyCamera.init("yawed", .{ .rotation = Vec3.new(0.0, 90.0, 45.0) });
    const f2 = yawed.getForward();
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), f2.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), f2.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), f2.z, 1e-5);
}

test "FlyCamera roll of 90 degrees swaps the up/right basis" {
    const cam = FlyCamera.init("fly", .{ .rotation = Vec3.new(0.0, 0.0, 90.0) });
    const fwd = cam.getForward();
    const up = cam.getUp();
    const right = cam.getRight();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), fwd.z, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), up.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), up.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), up.z, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), right.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), right.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), right.z, 1e-5);
    // Orthonormal right-handed basis is preserved under roll.
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), right.dot(up), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), right.length(), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), up.length(), 1e-5);
    const handed = right.cross(up);
    try std.testing.expectApproxEqAbs(-fwd.x, handed.x, 1e-5);
    try std.testing.expectApproxEqAbs(-fwd.y, handed.y, 1e-5);
    try std.testing.expectApproxEqAbs(-fwd.z, handed.z, 1e-5);
}

test "FlyCamera WASD flight follows forward and rolled right" {
    var cam = FlyCamera.init("fly", .{});
    cam.move_forward = true;
    cam.update(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cam.position.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cam.position.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -6.0), cam.position.z, 1e-5);

    // With 90 degrees of roll the strafe axis points along +Y.
    var rolled = FlyCamera.init("rolled", .{ .rotation = Vec3.new(0.0, 0.0, 90.0) });
    rolled.move_right = true;
    rolled.update(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), rolled.position.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), rolled.position.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), rolled.position.z, 1e-4);

    // R/F climbs along the (rolled) up axis; boost scales the speed.
    var climb = FlyCamera.init("climb", .{});
    climb.move_up = true;
    climb.boost_held = true;
    climb.update(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 24.0), climb.position.y, 1e-4);
}

test "FlyCamera inertia enables smooth rotation damping" {
    var cam = FlyCamera.init("fly", .{ .inertia = 0.85 });
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
