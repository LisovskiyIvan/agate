const std = @import("std");
const sokol = @import("sokol");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const Mesh = @import("mesh.zig").Mesh;

pub const ArcRotateCameraOptions = struct {
    alpha: f32 = 0.0,
    beta: f32 = std.math.pi / 3.0,
    radius: f32 = 5.0,
    target: Vec3 = Vec3.zero,
    fov_deg: f32 = 60.0,
    near: f32 = 0.1,
    far: f32 = 100.0,
};

pub const ArcRotateCamera = struct {
    name: []const u8 = "ArcRotateCamera",
    alpha: f32 = 0.0, // Radians, around Y
    beta: f32 = std.math.pi / 3.0, // Radians, from top (0) to bottom (pi)
    radius: f32 = 5.0,
    target: Vec3 = Vec3.zero,

    fov_deg: f32 = 60.0,
    near: f32 = 0.1,
    far: f32 = 100.0,

    // Управление мышью / трекпадом
    angular_sensitivity: f32 = 0.006,
    wheel_precision: f32 = 0.5,
    lower_radius_limit: f32 = 0.5,
    upper_radius_limit: f32 = 100.0,
    lower_beta_limit: f32 = 0.01,
    upper_beta_limit: f32 = std.math.pi - 0.01,

    is_dragging: bool = false,
    last_mouse_x: f32 = 0.0,
    last_mouse_y: f32 = 0.0,

    pub fn init(name: []const u8, options: ArcRotateCameraOptions) ArcRotateCamera {
        return .{
            .name = name,
            .alpha = options.alpha,
            .beta = options.beta,
            .radius = options.radius,
            .target = options.target,
            .fov_deg = options.fov_deg,
            .near = options.near,
            .far = options.far,
        };
    }

    pub fn handleEvent(self: *ArcRotateCamera, ev: [*c]const sokol.app.Event) void {
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

                    self.alpha -= dx * self.angular_sensitivity;
                    self.beta -= dy * self.angular_sensitivity;
                    self.beta = std.math.clamp(self.beta, self.lower_beta_limit, self.upper_beta_limit);
                }
            },
            .MOUSE_SCROLL => {
                self.radius -= ev.*.scroll_y * self.wheel_precision;
                self.radius = std.math.clamp(self.radius, self.lower_radius_limit, self.upper_radius_limit);
            },
            else => {},
        }
    }

    pub fn getPosition(self: ArcRotateCamera) Vec3 {
        // Clamp beta to prevent gimbal lock at exact poles
        const clamped_beta = std.math.clamp(self.beta, 0.001, std.math.pi - 0.001);
        const sin_beta = @sin(clamped_beta);
        const cos_beta = @cos(clamped_beta);
        const sin_alpha = @sin(self.alpha);
        const cos_alpha = @cos(self.alpha);

        return Vec3.new(
            self.target.x + self.radius * sin_beta * cos_alpha,
            self.target.y + self.radius * cos_beta,
            self.target.z + self.radius * sin_beta * sin_alpha,
        );
    }

    pub fn getViewMatrix(self: ArcRotateCamera) Mat4 {
        const pos = self.getPosition();
        return Mat4.lookAt(pos, self.target, Vec3.up);
    }

    pub fn getProjectionMatrix(self: ArcRotateCamera, aspect: f32) Mat4 {
        return Mat4.perspective(self.fov_deg, aspect, self.near, self.far);
    }

    pub fn getViewProjection(self: ArcRotateCamera, aspect: f32) Mat4 {
        const view = self.getViewMatrix();
        const proj = self.getProjectionMatrix(aspect);
        return Mat4.mul(proj, view);
    }

    pub fn getForward(self: ArcRotateCamera) Vec3 {
        const fwd = self.target.sub(self.getPosition());
        if (fwd.length() > 0.0001) return fwd.normalize();
        return Vec3.new(0, 0, -1);
    }

    pub fn getNear(self: ArcRotateCamera) f32 {
        return self.near;
    }

    pub fn getFar(self: ArcRotateCamera) f32 {
        return self.far;
    }

    pub fn getFovDeg(self: ArcRotateCamera) f32 {
        return self.fov_deg;
    }
};

// Rotation convention (degrees): x = pitch (positive looks up),
// y = yaw around Y, z unused. Yaw 0 + pitch 0 faces -Z.
pub const FreeCameraOptions = struct {
    position: Vec3 = Vec3.zero,
    rotation: Vec3 = Vec3.zero,
    fov_deg: f32 = 60.0,
    near: f32 = 0.1,
    far: f32 = 100.0,
    speed: f32 = 6.0,
    angular_sensitivity: f32 = 0.25,
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

                    self.rotation.y -= dx * self.angular_sensitivity;
                    self.rotation.x -= dy * self.angular_sensitivity;
                    self.rotation.x = std.math.clamp(self.rotation.x, -89.0, 89.0);
                }
            },
            .KEY_DOWN => self.setMoveFlag(ev.*.key_code, true),
            .KEY_UP => self.setMoveFlag(ev.*.key_code, false),
            else => {},
        }
    }

    pub fn update(self: *FreeCamera, dt: f32) void {
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
};

pub const FollowCameraOptions = struct {
    target_position: Vec3 = Vec3.zero,
    radius: f32 = 5.0,
    height_offset: f32 = 2.0,
    rotation_offset_deg: f32 = 0.0,
    fov_deg: f32 = 60.0,
    near: f32 = 0.1,
    far: f32 = 100.0,
    lerp_speed: f32 = 8.0,
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
};

pub const TargetCameraOptions = struct {
    position: Vec3 = Vec3.zero,
    target: Vec3 = Vec3.zero,
    up: Vec3 = Vec3.up,
    fov_deg: f32 = 60.0,
    near: f32 = 0.1,
    far: f32 = 100.0,
    smoothing: f32 = 8.0, // Lerp speed 1/s towards the goals; 0 = instant snap
};

// Observer camera: fixed lookAt(position, target, up), no mouse input.
// Motion happens only through desired goals + update(dt), which eases the
// current position/target towards them when smoothing > 0.
pub const TargetCamera = struct {
    name: []const u8 = "TargetCamera",
    position: Vec3 = Vec3.zero,
    target: Vec3 = Vec3.zero,
    desired_position: ?Vec3 = null,
    desired_target: ?Vec3 = null,
    up: Vec3 = Vec3.up,

    fov_deg: f32 = 60.0,
    near: f32 = 0.1,
    far: f32 = 100.0,
    smoothing: f32 = 8.0,

    pub fn init(name: []const u8, options: TargetCameraOptions) TargetCamera {
        return .{
            .name = name,
            .position = options.position,
            .target = options.target,
            .up = options.up,
            .fov_deg = options.fov_deg,
            .near = options.near,
            .far = options.far,
            .smoothing = options.smoothing,
        };
    }

    pub fn setTarget(self: *TargetCamera, target: Vec3) void {
        if (self.smoothing <= 0.0) {
            self.target = target;
            self.desired_target = null;
        } else {
            self.desired_target = target;
        }
    }

    pub fn setDesiredPosition(self: *TargetCamera, position: Vec3) void {
        if (self.smoothing <= 0.0) {
            self.position = position;
            self.desired_position = null;
        } else {
            self.desired_position = position;
        }
    }

    pub fn clearGoals(self: *TargetCamera) void {
        self.desired_position = null;
        self.desired_target = null;
    }

    pub fn handleEvent(self: *TargetCamera, ev: [*c]const sokol.app.Event) void {
        _ = self;
        _ = ev;
    }

    pub fn update(self: *TargetCamera, dt: f32) void {
        if (self.smoothing <= 0.0) {
            if (self.desired_position) |p| self.position = p;
            if (self.desired_target) |t| self.target = t;
            self.clearGoals();
            return;
        }
        const t = @min(1.0, self.smoothing * dt);
        if (self.desired_position) |p| {
            self.position = self.position.lerp(p, t);
            if (self.position.sub(p).length() < 0.0001) {
                self.position = p;
                self.desired_position = null;
            }
        }
        if (self.desired_target) |goal| {
            self.target = self.target.lerp(goal, t);
            if (self.target.sub(goal).length() < 0.0001) {
                self.target = goal;
                self.desired_target = null;
            }
        }
    }

    pub fn getPosition(self: TargetCamera) Vec3 {
        return self.position;
    }

    pub fn getForward(self: TargetCamera) Vec3 {
        const fwd = self.target.sub(self.position);
        if (fwd.length() > 0.0001) return fwd.normalize();
        return Vec3.new(0, 0, -1);
    }

    // Up vector actually used for the view matrix: Gram-Schmidt fallback
    // when the configured up is (nearly) parallel to the view direction,
    // e.g. looking straight down with up = +Y.
    fn effectiveUp(self: TargetCamera) Vec3 {
        const fwd = self.getForward();
        var u = self.up;
        if (u.length() < 0.0001) u = Vec3.up;
        u = u.normalize();
        if (@abs(fwd.dot(u)) > 0.999) {
            // Pick the world axis least aligned with fwd, then orthonormalize.
            const ax = @abs(fwd.x);
            const ay = @abs(fwd.y);
            const az = @abs(fwd.z);
            var helper = Vec3.right;
            if (ax <= ay and ax <= az) {
                helper = Vec3.right;
            } else if (ay <= az) {
                helper = Vec3.up;
            } else {
                helper = Vec3.new(0, 0, 1);
            }
            u = helper.sub(fwd.scale(fwd.dot(helper))).normalize();
        }
        return u;
    }

    pub fn getViewMatrix(self: TargetCamera) Mat4 {
        if (self.target.sub(self.position).length() < 0.000001) return Mat4.identity;
        return Mat4.lookAt(self.position, self.target, self.effectiveUp());
    }

    pub fn getProjectionMatrix(self: TargetCamera, aspect: f32) Mat4 {
        return Mat4.perspective(self.fov_deg, aspect, self.near, self.far);
    }

    pub fn getViewProjection(self: TargetCamera, aspect: f32) Mat4 {
        return Mat4.mul(self.getProjectionMatrix(aspect), self.getViewMatrix());
    }

    pub fn getNear(self: TargetCamera) f32 {
        return self.near;
    }

    pub fn getFar(self: TargetCamera) f32 {
        return self.far;
    }

    pub fn getFovDeg(self: TargetCamera) f32 {
        return self.fov_deg;
    }
};

// Rotation convention (degrees): x = pitch (positive looks up, clamped
// to [-89, 89]), y = yaw around Y, z = roll around the view axis.
// Yaw 0 + pitch 0 + roll 0 faces -Z with up +Y (matches FreeCamera).
// Unlike FreeCamera, flight follows the true 3D forward (pitch included)
// and the strafe/up axes tilt with the roll.
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

                    self.rotation.y -= dx * self.angular_sensitivity;
                    self.rotation.x -= dy * self.angular_sensitivity;
                    self.rotation.x = std.math.clamp(self.rotation.x, -89.0, 89.0);
                }
            },
            .KEY_DOWN => self.setMoveFlag(ev.*.key_code, true),
            .KEY_UP => self.setMoveFlag(ev.*.key_code, false),
            else => {},
        }
    }

    pub fn update(self: *FlyCamera, dt: f32) void {
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
};

pub const Camera = union(enum) {
    arc_rotate: ArcRotateCamera,
    free: FreeCamera,
    follow: FollowCamera,
    target: TargetCamera,
    fly: FlyCamera,

    pub fn getPosition(self: Camera) Vec3 {
        return switch (self) {
            .arc_rotate => |c| c.getPosition(),
            .free => |c| c.getPosition(),
            .follow => |c| c.getPosition(),
            .target => |c| c.getPosition(),
            .fly => |c| c.getPosition(),
        };
    }

    pub fn getViewMatrix(self: Camera) Mat4 {
        return switch (self) {
            .arc_rotate => |c| c.getViewMatrix(),
            .free => |c| c.getViewMatrix(),
            .follow => |c| c.getViewMatrix(),
            .target => |c| c.getViewMatrix(),
            .fly => |c| c.getViewMatrix(),
        };
    }

    pub fn getProjectionMatrix(self: Camera, aspect: f32) Mat4 {
        return switch (self) {
            .arc_rotate => |c| c.getProjectionMatrix(aspect),
            .free => |c| c.getProjectionMatrix(aspect),
            .follow => |c| c.getProjectionMatrix(aspect),
            .target => |c| c.getProjectionMatrix(aspect),
            .fly => |c| c.getProjectionMatrix(aspect),
        };
    }

    pub fn getViewProjection(self: Camera, aspect: f32) Mat4 {
        return switch (self) {
            .arc_rotate => |c| c.getViewProjection(aspect),
            .free => |c| c.getViewProjection(aspect),
            .follow => |c| c.getViewProjection(aspect),
            .target => |c| c.getViewProjection(aspect),
            .fly => |c| c.getViewProjection(aspect),
        };
    }

    pub fn handleEvent(self: *Camera, ev: [*c]const sokol.app.Event) void {
        switch (self.*) {
            .arc_rotate => |*c| c.handleEvent(ev),
            .free => |*c| c.handleEvent(ev),
            .follow => |*c| c.handleEvent(ev),
            .target => |*c| c.handleEvent(ev),
            .fly => |*c| c.handleEvent(ev),
        }
    }

    pub fn update(self: *Camera, dt: f32) void {
        switch (self.*) {
            .arc_rotate => {},
            .free => |*c| c.update(dt),
            .follow => |*c| c.update(dt),
            .target => |*c| c.update(dt),
            .fly => |*c| c.update(dt),
        }
    }

    pub fn getForward(self: Camera) Vec3 {
        return switch (self) {
            .arc_rotate => |c| c.getForward(),
            .free => |c| c.getForward(),
            .follow => |c| c.getForward(),
            .target => |c| c.getForward(),
            .fly => |c| c.getForward(),
        };
    }

    pub fn getNear(self: Camera) f32 {
        return switch (self) {
            .arc_rotate => |c| c.getNear(),
            .free => |c| c.getNear(),
            .follow => |c| c.getNear(),
            .target => |c| c.getNear(),
            .fly => |c| c.getNear(),
        };
    }

    pub fn getFar(self: Camera) f32 {
        return switch (self) {
            .arc_rotate => |c| c.getFar(),
            .free => |c| c.getFar(),
            .follow => |c| c.getFar(),
            .target => |c| c.getFar(),
            .fly => |c| c.getFar(),
        };
    }

    pub fn getFovDeg(self: Camera) f32 {
        return switch (self) {
            .arc_rotate => |c| c.getFovDeg(),
            .free => |c| c.getFovDeg(),
            .follow => |c| c.getFovDeg(),
            .target => |c| c.getFovDeg(),
            .fly => |c| c.getFovDeg(),
        };
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

test "Camera union dispatches getPosition" {
    const free: Camera = .{ .free = FreeCamera.init("test", .{ .position = Vec3.new(1.0, 2.0, 3.0) }) };
    const p = free.getPosition();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), p.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), p.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), p.z, 1e-5);

    const arc: Camera = .{ .arc_rotate = ArcRotateCamera.init("orbit", .{}) };
    try std.testing.expectApproxEqAbs(@as(f32, 60.0), arc.getFovDeg(), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), arc.getNear(), 1e-6);
}

test "TargetCamera view looks at the target" {
    const cam = TargetCamera.init("watcher", .{
        .position = Vec3.new(0.0, 0.0, 5.0),
        .target = Vec3.zero,
    });
    const fwd = cam.getForward();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), fwd.z, 1e-5);
    // The look-at point projects to the screen center.
    const vp = cam.getViewProjection(800.0 / 600.0);
    const p = vp.projectPoint(Vec3.zero, 800.0, 600.0) orelse return error.PointBehindCamera;
    try std.testing.expectApproxEqAbs(@as(f32, 400.0), p.x, 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 300.0), p.y, 1e-2);
}

test "TargetCamera top-down view does not degenerate" {
    const cam = TargetCamera.init("top", .{
        .position = Vec3.new(0.0, 5.0, 0.0),
        .target = Vec3.zero,
        .up = Vec3.up, // Parallel to the view direction: must be handled.
    });
    const fwd = cam.getForward();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), fwd.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.z, 1e-5);
    const view = cam.getViewMatrix();
    for (view.m) |v| try std.testing.expect(std.math.isFinite(v));
    // Target still projects to the screen center despite the up fallback.
    const vp = cam.getViewProjection(800.0 / 600.0);
    const p = vp.projectPoint(Vec3.zero, 800.0, 600.0) orelse return error.PointBehindCamera;
    try std.testing.expectApproxEqAbs(@as(f32, 400.0), p.x, 1e-2);
    try std.testing.expectApproxEqAbs(@as(f32, 300.0), p.y, 1e-2);
}

test "TargetCamera smoothing converges to the goals" {
    var cam = TargetCamera.init("watcher", .{
        .position = Vec3.new(0.0, 0.0, 10.0),
        .target = Vec3.new(5.0, 0.0, 0.0),
        .smoothing = 8.0,
    });
    cam.setDesiredPosition(Vec3.new(0.0, 0.0, 5.0));
    cam.setTarget(Vec3.zero);
    var i: usize = 0;
    while (i < 600) : (i += 1) cam.update(1.0 / 60.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cam.position.x, 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cam.position.y, 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), cam.position.z, 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cam.target.x, 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cam.target.y, 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cam.target.z, 1e-3);

    // Zero smoothing snaps instantly.
    var snap = TargetCamera.init("snap", .{ .smoothing = 0.0 });
    snap.setDesiredPosition(Vec3.new(1.0, 2.0, 3.0));
    snap.setTarget(Vec3.new(4.0, 5.0, 6.0));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), snap.position.x, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), snap.target.z, 1e-6);
}

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

test "Camera union dispatches target and fly views and lens params" {
    const aspect = 800.0 / 600.0;
    const arc: Camera = .{ .arc_rotate = ArcRotateCamera.init("orbit", .{}) };
    const free: Camera = .{ .free = FreeCamera.init("free", .{ .position = Vec3.new(0.0, 0.0, 5.0) }) };
    const follow: Camera = .{ .follow = FollowCamera.init("follow", .{ .target_position = Vec3.zero }) };
    const target: Camera = .{ .target = TargetCamera.init("watcher", .{
        .position = Vec3.new(0.0, 0.0, 5.0),
        .target = Vec3.zero,
        .fov_deg = 50.0,
        .near = 0.5,
        .far = 200.0,
    }) };
    const fly: Camera = .{ .fly = FlyCamera.init("fly", .{
        .position = Vec3.new(0.0, 0.0, 5.0),
        .fov_deg = 70.0,
        .near = 0.2,
        .far = 300.0,
    }) };
    // Every variant produces a usable view-projection: the point in front
    // of each camera projects inside the viewport.
    for ([_]Camera{ arc, free, follow, target, fly }) |cam| {
        const eye = cam.getPosition();
        const ahead = eye.add(cam.getForward().scale(5.0));
        const vp = cam.getViewProjection(aspect);
        const p = vp.projectPoint(ahead, 800.0, 600.0) orelse return error.PointBehindCamera;
        try std.testing.expect(p.x >= 0.0 and p.x <= 800.0);
        try std.testing.expect(p.y >= 0.0 and p.y <= 600.0);
    }
    try std.testing.expectApproxEqAbs(@as(f32, 50.0), target.getFovDeg(), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), target.getNear(), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 200.0), target.getFar(), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 70.0), fly.getFovDeg(), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), fly.getNear(), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 300.0), fly.getFar(), 1e-5);
}
