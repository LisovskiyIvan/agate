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
