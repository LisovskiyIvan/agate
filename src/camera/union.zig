//! Polymorphic camera: tagged union over the five camera types with a
//! uniform lens/input interface. Split out of `camera.zig` (facade):
//! `camera.zig` re-exports `Camera` so the public API is unchanged.
//!
//! Documented anti-cycle rule: this module imports the sibling leaves
//! (`viewport`, `arc_rotate`, `free`, `follow`, `target`, `fly`) — never
//! the `camera.zig` facade. The leaves never import this module back, so
//! the dispatch edge points one way only.

const std = @import("std");
const sokol = @import("sokol");
const math = @import("math");
const Vec3 = math.Vec3;
const Mat4 = math.Mat4;
const viewport_mod = @import("viewport.zig");
const Viewport = viewport_mod.Viewport;
const arc_rotate_mod = @import("arc_rotate.zig");
const ArcRotateCamera = arc_rotate_mod.ArcRotateCamera;
const free_mod = @import("free.zig");
const FreeCamera = free_mod.FreeCamera;
const follow_mod = @import("follow.zig");
const FollowCamera = follow_mod.FollowCamera;
const target_mod = @import("target.zig");
const TargetCamera = target_mod.TargetCamera;
const fly_mod = @import("fly.zig");
const FlyCamera = fly_mod.FlyCamera;

pub const Camera = union(enum) {
    arc_rotate: ArcRotateCamera,
    free: FreeCamera,
    follow: FollowCamera,
    target: TargetCamera,
    fly: FlyCamera,

    pub fn getName(self: Camera) []const u8 {
        return switch (self) {
            .arc_rotate => |c| c.name,
            .free => |c| c.name,
            .follow => |c| c.name,
            .target => |c| c.name,
            .fly => |c| c.name,
        };
    }

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
            .arc_rotate => |*c| c.update(dt),
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

    pub fn getCullingMask(self: Camera) u32 {
        return switch (self) {
            inline else => |c| c.culling_mask,
        };
    }

    pub fn setCullingMask(self: *Camera, mask: u32) void {
        switch (self.*) {
            inline else => |*c| c.culling_mask = mask,
        }
    }

    pub fn getViewport(self: Camera) Viewport {
        return switch (self) {
            inline else => |c| c.viewport,
        };
    }

    pub fn setViewport(self: *Camera, vp: Viewport) void {
        switch (self.*) {
            inline else => |*c| c.viewport = vp,
        }
    }

    pub fn getRight(self: Camera) Vec3 {
        return switch (self) {
            .fly => |c| c.getRight(),
            inline else => {
                const v = self.getViewMatrix();
                const r = Vec3.new(v.m[0], v.m[4], v.m[8]);
                const len = r.length();
                return if (len > 0.0001) r.scale(1.0 / len) else Vec3.right;
            },
        };
    }

    pub fn getUp(self: Camera) Vec3 {
        return switch (self) {
            .fly => |c| c.getUp(),
            inline else => {
                const v = self.getViewMatrix();
                const u = Vec3.new(v.m[1], v.m[5], v.m[9]);
                const len = u.length();
                return if (len > 0.0001) u.scale(1.0 / len) else Vec3.up;
            },
        };
    }

    pub fn setPosition(self: *Camera, pos: Vec3) void {
        switch (self.*) {
            .target => |*c| {
                c.position = pos;
                c.desired_position = null;
            },
            .free => |*c| {
                c.position = pos;
            },
            .fly => |*c| {
                c.position = pos;
            },
            .follow => |*c| {
                c.position = pos;
            },
            .arc_rotate => |*c| {
                const diff = pos.sub(c.target);
                c.radius = diff.length();
                if (c.radius > 0.0001) {
                    c.beta = std.math.acos(std.math.clamp(diff.y / c.radius, -1.0, 1.0));
                    c.alpha = std.math.atan2(diff.z, diff.x);
                }
            },
        }
    }

    pub fn setLookAt(self: *Camera, pos: Vec3, target: Vec3, up: ?Vec3) void {
        switch (self.*) {
            .target => |*c| {
                c.position = pos;
                c.target = target;
                if (up) |u| c.up = u;
                c.clearGoals();
            },
            .free => |*c| {
                c.position = pos;
                const fwd = target.sub(pos);
                if (fwd.length() > 0.0001) {
                    const norm = fwd.normalize();
                    const pitch = std.math.asin(std.math.clamp(norm.y, -1.0, 1.0)) * 180.0 / std.math.pi;
                    const yaw = std.math.atan2(-norm.x, -norm.z) * 180.0 / std.math.pi;
                    c.rotation.x = pitch;
                    c.rotation.y = yaw;
                }
            },
            .fly => |*c| {
                c.position = pos;
                const fwd = target.sub(pos);
                if (fwd.length() > 0.0001) {
                    const norm = fwd.normalize();
                    const pitch = std.math.asin(std.math.clamp(norm.y, -1.0, 1.0)) * 180.0 / std.math.pi;
                    const yaw = std.math.atan2(-norm.x, -norm.z) * 180.0 / std.math.pi;
                    c.rotation.x = pitch;
                    c.rotation.y = yaw;
                    c.rotation.z = 0.0;
                }
            },
            .arc_rotate => |*c| {
                c.target = target;
                const diff = pos.sub(target);
                c.radius = diff.length();
                if (c.radius > 0.0001) {
                    c.beta = std.math.acos(std.math.clamp(diff.y / c.radius, -1.0, 1.0));
                    c.alpha = std.math.atan2(diff.z, diff.x);
                }
            },
            .follow => |*c| {
                c.position = pos;
                c.target_position = target;
            },
        }
    }
};
