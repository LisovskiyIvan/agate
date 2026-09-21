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

test "Camera mask and viewport dispatch" {
    var cam: Camera = .{ .free = FreeCamera.init("free", .{}) };
    try std.testing.expectEqual(@as(u32, 0xFFFFFFFF), cam.getCullingMask());
    cam.setCullingMask(0x00000004);
    try std.testing.expectEqual(@as(u32, 0x00000004), cam.getCullingMask());

    cam.setViewport(.{ .x = 0.1, .y = 0.2, .width = 0.3, .height = 0.4 });
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), cam.getViewport().x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), cam.getViewport().width, 1e-5);
}

test "Camera union dispatches update to arc_rotate with inertia" {
    var cam: Camera = .{ .arc_rotate = ArcRotateCamera.init("orbit", .{ .inertia = 0.9 }) };
    var down_ev = sokol.app.Event{
        .type = .MOUSE_DOWN,
        .mouse_button = .LEFT,
        .mouse_x = 50.0,
        .mouse_y = 50.0,
    };
    cam.handleEvent(&down_ev);

    var move_ev = sokol.app.Event{
        .type = .MOUSE_MOVE,
        .mouse_x = 70.0,
        .mouse_y = 60.0,
    };
    cam.handleEvent(&move_ev);

    const pos0 = cam.getPosition();
    cam.update(1.0 / 60.0);
    const pos1 = cam.getPosition();
    try std.testing.expect(pos0.x != pos1.x or pos0.z != pos1.z);
}

test "Camera union getRight and getUp" {
    const free: Camera = .{ .free = FreeCamera.init("free", .{ .position = Vec3.new(0, 0, 5), .rotation = Vec3.zero }) };
    const r = free.getRight();
    const u = free.getUp();
    const f = free.getForward();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), r.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), r.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), r.z, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), u.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), u.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), u.z, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), f.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), f.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), f.z, 1e-4);
}

test "Camera union setPosition and setLookAt" {
    var target_cam: Camera = .{ .target = TargetCamera.init("target", .{
        .position = Vec3.new(0, 0, 10),
        .target = Vec3.zero,
        .smoothing = 0.0,
    }) };
    target_cam.setPosition(Vec3.new(1, 2, 3));
    const p = target_cam.getPosition();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), p.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), p.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), p.z, 1e-5);

    target_cam.setLookAt(Vec3.new(0, 0, 5), Vec3.new(0, 0, 0), Vec3.up);
    const fwd = target_cam.getForward();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), fwd.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), fwd.z, 1e-5);
}

