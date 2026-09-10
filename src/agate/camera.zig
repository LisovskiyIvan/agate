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

pub const Camera = union(enum) {
    arc_rotate: ArcRotateCamera,
    free: FreeCamera,
    follow: FollowCamera,

    pub fn getPosition(self: Camera) Vec3 {
        return switch (self) {
            .arc_rotate => |c| c.getPosition(),
            .free => |c| c.getPosition(),
            .follow => |c| c.getPosition(),
        };
    }

    pub fn getViewMatrix(self: Camera) Mat4 {
        return switch (self) {
            .arc_rotate => |c| c.getViewMatrix(),
            .free => |c| c.getViewMatrix(),
            .follow => |c| c.getViewMatrix(),
        };
    }

    pub fn getProjectionMatrix(self: Camera, aspect: f32) Mat4 {
        return switch (self) {
            .arc_rotate => |c| c.getProjectionMatrix(aspect),
            .free => |c| c.getProjectionMatrix(aspect),
            .follow => |c| c.getProjectionMatrix(aspect),
        };
    }

    pub fn getViewProjection(self: Camera, aspect: f32) Mat4 {
        return switch (self) {
            .arc_rotate => |c| c.getViewProjection(aspect),
            .free => |c| c.getViewProjection(aspect),
            .follow => |c| c.getViewProjection(aspect),
        };
    }

    pub fn handleEvent(self: *Camera, ev: [*c]const sokol.app.Event) void {
        switch (self.*) {
            .arc_rotate => |*c| c.handleEvent(ev),
            .free => |*c| c.handleEvent(ev),
            .follow => |*c| c.handleEvent(ev),
        }
    }

    pub fn update(self: *Camera, dt: f32) void {
        switch (self.*) {
            .arc_rotate => {},
            .free => |*c| c.update(dt),
            .follow => |*c| c.update(dt),
        }
    }

    pub fn getForward(self: Camera) Vec3 {
        return switch (self) {
            .arc_rotate => |c| c.getForward(),
            .free => |c| c.getForward(),
            .follow => |c| c.getForward(),
        };
    }

    pub fn getNear(self: Camera) f32 {
        return switch (self) {
            .arc_rotate => |c| c.getNear(),
            .free => |c| c.getNear(),
            .follow => |c| c.getNear(),
        };
    }

    pub fn getFar(self: Camera) f32 {
        return switch (self) {
            .arc_rotate => |c| c.getFar(),
            .free => |c| c.getFar(),
            .follow => |c| c.getFar(),
        };
    }

    pub fn getFovDeg(self: Camera) f32 {
        return switch (self) {
            .arc_rotate => |c| c.getFovDeg(),
            .free => |c| c.getFovDeg(),
            .follow => |c| c.getFovDeg(),
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
