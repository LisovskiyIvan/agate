//! Orbit camera: position derived from spherical coordinates around a
//! target. Split out of `camera.zig` (facade): `camera.zig` re-exports the
//! options and camera types so the public API is unchanged. Leaf: imports
//! `viewport` only — never the facade and never sibling camera leaves.

const std = @import("std");
const sokol = @import("sokol");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const viewport_mod = @import("viewport.zig");
const Viewport = viewport_mod.Viewport;

pub const ArcRotateCameraOptions = struct {
    alpha: f32 = 0.0,
    beta: f32 = std.math.pi / 3.0,
    radius: f32 = 5.0,
    target: Vec3 = Vec3.zero,
    fov_deg: f32 = 60.0,
    near: f32 = 0.1,
    far: f32 = 100.0,
    inertia: f32 = 0.0, // Damping factor (0.0 = instant, e.g. 0.9 = smooth damping)
    culling_mask: u32 = 0xFFFFFFFF,
    viewport: Viewport = .{},
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
    culling_mask: u32 = 0xFFFFFFFF,
    viewport: Viewport = .{},

    // Управление мышью / трекпадом
    angular_sensitivity: f32 = 0.006,
    wheel_precision: f32 = 0.5,
    lower_radius_limit: f32 = 0.5,
    upper_radius_limit: f32 = 100.0,
    lower_beta_limit: f32 = 0.01,
    upper_beta_limit: f32 = std.math.pi - 0.01,

    // Inertia & input smoothing (Babylon parity: 0.9 = smooth damping, 0.0 = instant)
    inertia: f32 = 0.0,
    inertial_alpha_offset: f32 = 0.0,
    inertial_beta_offset: f32 = 0.0,
    inertial_radius_offset: f32 = 0.0,

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
            .inertia = options.inertia,
            .culling_mask = options.culling_mask,
            .viewport = options.viewport,
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

                    if (self.inertia <= 0.0) {
                        self.alpha -= dx * self.angular_sensitivity;
                        self.beta -= dy * self.angular_sensitivity;
                        self.beta = std.math.clamp(self.beta, self.lower_beta_limit, self.upper_beta_limit);
                    } else {
                        self.inertial_alpha_offset += dx * self.angular_sensitivity;
                        self.inertial_beta_offset += dy * self.angular_sensitivity;
                    }
                }
            },
            .MOUSE_SCROLL => {
                if (self.inertia <= 0.0) {
                    self.radius -= ev.*.scroll_y * self.wheel_precision;
                    self.radius = std.math.clamp(self.radius, self.lower_radius_limit, self.upper_radius_limit);
                } else {
                    self.inertial_radius_offset += ev.*.scroll_y * self.wheel_precision;
                }
            },
            else => {},
        }
    }

    pub fn update(self: *ArcRotateCamera, dt: f32) void {
        if (self.inertia <= 0.0) return;
        if (self.inertial_alpha_offset == 0.0 and self.inertial_beta_offset == 0.0 and self.inertial_radius_offset == 0.0) return;

        self.alpha -= self.inertial_alpha_offset;
        self.beta -= self.inertial_beta_offset;
        self.beta = std.math.clamp(self.beta, self.lower_beta_limit, self.upper_beta_limit);

        self.radius -= self.inertial_radius_offset;
        self.radius = std.math.clamp(self.radius, self.lower_radius_limit, self.upper_radius_limit);

        // Frame-rate independent exponential decay
        const clamped_inertia = std.math.clamp(self.inertia, 0.0, 0.999);
        const decay = std.math.pow(f32, clamped_inertia, dt * 60.0);
        self.inertial_alpha_offset *= decay;
        self.inertial_beta_offset *= decay;
        self.inertial_radius_offset *= decay;

        if (@abs(self.inertial_alpha_offset) < 0.00001) self.inertial_alpha_offset = 0.0;
        if (@abs(self.inertial_beta_offset) < 0.00001) self.inertial_beta_offset = 0.0;
        if (@abs(self.inertial_radius_offset) < 0.00001) self.inertial_radius_offset = 0.0;
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

    pub fn getViewport(self: ArcRotateCamera) Viewport {
        return self.viewport;
    }

    pub fn setViewport(self: *ArcRotateCamera, vp: Viewport) void {
        self.viewport = vp;
    }

    pub fn getCullingMask(self: ArcRotateCamera) u32 {
        return self.culling_mask;
    }

    pub fn setCullingMask(self: *ArcRotateCamera, mask: u32) void {
        self.culling_mask = mask;
    }
};

test "ArcRotateCamera inertia disabled behaves instantly" {
    var cam = ArcRotateCamera.init("orbit", .{
        .alpha = 0.0,
        .beta = std.math.pi / 2.0,
        .radius = 10.0,
        .inertia = 0.0,
    });
    var down_ev = sokol.app.Event{
        .type = .MOUSE_DOWN,
        .mouse_button = .LEFT,
        .mouse_x = 100.0,
        .mouse_y = 100.0,
    };
    cam.handleEvent(&down_ev);

    var move_ev = sokol.app.Event{
        .type = .MOUSE_MOVE,
        .mouse_x = 110.0,
        .mouse_y = 105.0,
    };
    cam.handleEvent(&move_ev);

    // With inertia == 0, alpha and beta move immediately
    try std.testing.expect(cam.alpha != 0.0);
    try std.testing.expect(cam.beta != std.math.pi / 2.0);
    try std.testing.expectEqual(@as(f32, 0.0), cam.inertial_alpha_offset);
    try std.testing.expectEqual(@as(f32, 0.0), cam.inertial_beta_offset);
}

test "ArcRotateCamera inertia enables smooth damping across update ticks" {
    var cam = ArcRotateCamera.init("orbit", .{
        .alpha = 0.0,
        .beta = std.math.pi / 2.0,
        .radius = 10.0,
        .inertia = 0.9,
    });
    var down_ev = sokol.app.Event{
        .type = .MOUSE_DOWN,
        .mouse_button = .LEFT,
        .mouse_x = 100.0,
        .mouse_y = 100.0,
    };
    cam.handleEvent(&down_ev);

    var move_ev = sokol.app.Event{
        .type = .MOUSE_MOVE,
        .mouse_x = 120.0,
        .mouse_y = 110.0,
    };
    cam.handleEvent(&move_ev);

    // With inertia > 0, angle does not change immediately during drag
    try std.testing.expectEqual(@as(f32, 0.0), cam.alpha);
    try std.testing.expectEqual(@as(f32, std.math.pi / 2.0), cam.beta);
    try std.testing.expect(cam.inertial_alpha_offset != 0.0);
    try std.testing.expect(cam.inertial_beta_offset != 0.0);

    // Step 1: update moves the camera towards the target and decays offsets
    cam.update(1.0 / 60.0);
    const alpha_step1 = cam.alpha;
    try std.testing.expect(alpha_step1 < 0.0); // Moved
    const offset_step1 = cam.inertial_alpha_offset;
    try std.testing.expect(offset_step1 < 20.0 * cam.angular_sensitivity); // Decayed

    // Step 2: continues to move smoothly
    cam.update(1.0 / 60.0);
    try std.testing.expect(cam.alpha < alpha_step1);
    try std.testing.expect(cam.inertial_alpha_offset < offset_step1);
}
