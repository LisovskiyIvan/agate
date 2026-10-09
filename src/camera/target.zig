//! Observer camera: fixed lookAt(position, target, up), no mouse input.
//! Motion happens only through desired goals + update(dt), which eases the
//! current position/target towards them when smoothing > 0. Split out of
//! `camera.zig` (facade): `camera.zig` re-exports the options and camera
//! types so the public API is unchanged. Leaf: imports `viewport` only —
//! never the facade and never sibling camera leaves.

const std = @import("std");
const sokol = @import("sokol");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const viewport_mod = @import("viewport.zig");
const Viewport = viewport_mod.Viewport;

pub const TargetCameraOptions = struct {
    position: Vec3 = Vec3.zero,
    target: Vec3 = Vec3.zero,
    up: Vec3 = Vec3.up,
    fov_deg: f32 = 60.0,
    near: f32 = 0.1,
    far: f32 = 100.0,
    smoothing: f32 = 8.0, // Lerp speed 1/s towards the goals; 0 = instant snap
    culling_mask: u32 = 0xFFFFFFFF,
    viewport: Viewport = .{},
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
    culling_mask: u32 = 0xFFFFFFFF,
    viewport: Viewport = .{},

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
            .culling_mask = options.culling_mask,
            .viewport = options.viewport,
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

    pub fn getViewport(self: TargetCamera) Viewport {
        return self.viewport;
    }

    pub fn setViewport(self: *TargetCamera, vp: Viewport) void {
        self.viewport = vp;
    }

    pub fn getCullingMask(self: TargetCamera) u32 {
        return self.culling_mask;
    }

    pub fn setCullingMask(self: *TargetCamera, mask: u32) void {
        self.culling_mask = mask;
    }
};
