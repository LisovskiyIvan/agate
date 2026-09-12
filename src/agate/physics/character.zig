const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const c = @import("../c.zig").c;
const convert = @import("convert.zig");
const toB3Pos = convert.toB3Pos;
const toB3Vec = convert.toB3Vec;
const fromB3Vec = convert.fromB3Vec;

/// Kinematic capsule character controller built on Box3D's mover API
/// (collide + plane solve + velocity clip). It is NOT a rigid body: it
/// slides around dynamic bodies without pushing them and is invisible to
/// sensors. `position` is the feet point (capsule bottom).
pub const CharacterController = struct {
    position: Vec3 = Vec3.zero,
    velocity: Vec3 = Vec3.zero,
    radius: f32 = 0.3,
    height: f32 = 1.5, // total capsule height
    move_speed: f32 = 4.5,
    jump_speed: f32 = 6.0,
    gravity: f32 = 18.0,
    slope_limit_cos: f32 = 0.7,
    is_grounded: bool = false,
    ground_normal: Vec3 = Vec3.up,

    const max_planes = 16;

    pub const PlaneCollector = struct {
        planes: []c.b3CollisionPlane,
        count: usize = 0,
    };

    pub fn init(position: Vec3, radius: f32, height: f32) CharacterController {
        return .{ .position = position, .radius = radius, .height = height };
    }

    /// Advances the controller. `wish_dir` is the desired horizontal move
    /// direction (y is ignored, longer than 1 is normalized). `jump_pressed`
    /// is an edge trigger consumed while grounded.
    pub fn move(self: *CharacterController, world: anytype, wish_dir: Vec3, jump_pressed: bool, dt: f32) void {
        if (dt <= 0.0) return;
        world.syncWorldParams();
        const h = @min(dt, 1.0 / 30.0);

        var plane_buf: [max_planes]c.b3CollisionPlane = undefined;
        var collector = PlaneCollector{ .planes = &plane_buf };
        var mover = c.b3Capsule{
            .center1 = .{ .x = 0.0, .y = self.radius, .z = 0.0 },
            .center2 = .{ .x = 0.0, .y = @max(self.height - self.radius, self.radius), .z = 0.0 },
            .radius = self.radius,
        };
        const filter = c.b3DefaultQueryFilter();
        c.b3World_CollideMover(world.world_id, toB3Pos(self.position), &mover, filter, &planeCollectFcn, &collector);

        self.is_grounded = false;
        self.ground_normal = Vec3.up;
        var best_up: f32 = self.slope_limit_cos;
        for (plane_buf[0..collector.count]) |cp| {
            const n = fromB3Vec(cp.plane.normal);
            if (n.y > best_up) {
                best_up = n.y;
                self.ground_normal = n;
                self.is_grounded = true;
            }
        }

        var vx = wish_dir.x;
        var vz = wish_dir.z;
        const wl = @sqrt(vx * vx + vz * vz);
        if (wl > 1.0) {
            vx /= wl;
            vz /= wl;
        }
        vx *= self.move_speed;
        vz *= self.move_speed;

        var vy = self.velocity.y;
        if (self.is_grounded) {
            if (jump_pressed) {
                vy = self.jump_speed;
                self.is_grounded = false;
            } else {
                vy = -2.0; // stick to the ground on slopes and ledges
            }
        } else {
            vy = @max(vy - self.gravity * h, -30.0);
        }
        self.velocity = Vec3.new(vx, vy, vz);

        const target = toB3Vec(Vec3.new(vx * h, vy * h, vz * h));
        const solved = c.b3SolvePlanes(target, &plane_buf, @intCast(collector.count));
        self.position = self.position.add(fromB3Vec(solved.delta));

        const clipped = c.b3ClipVector(toB3Vec(self.velocity), &plane_buf, @intCast(collector.count));
        self.velocity = fromB3Vec(clipped);
    }
};

fn planeCollectFcn(
    shape_id: c.b3ShapeId,
    planes: [*c]const c.b3PlaneResult,
    plane_count: c_int,
    context: ?*anyopaque,
) callconv(.c) bool {
    _ = shape_id;
    const ctx = context orelse return false;
    const collector: *CharacterController.PlaneCollector = @ptrCast(@alignCast(ctx));
    var i: c_int = 0;
    while (i < plane_count) : (i += 1) {
        if (collector.count >= collector.planes.len) return false;
        const pr = planes[@intCast(i)];
        collector.planes[collector.count] = .{
            .plane = pr.plane,
            .pushLimit = std.math.floatMax(f32),
            .push = 0.0,
            .clipVelocity = true,
        };
        collector.count += 1;
    }
    return true;
}
