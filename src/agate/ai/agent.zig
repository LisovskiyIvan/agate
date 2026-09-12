const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const NavMesh = @import("navmesh.zig").NavMesh;
const Pathfinding = @import("pathfinding.zig").Pathfinding;

pub const NavAgent = struct {
    allocator: std.mem.Allocator,
    nav_mesh: *const NavMesh,
    position: Vec3,
    velocity: Vec3 = Vec3.zero,
    target_pos: ?Vec3 = null,
    waypoints: []Vec3 = &.{},
    current_waypoint_idx: usize = 0,
    speed: f32 = 3.5,
    rotation_speed: f32 = 8.0,
    stopping_distance: f32 = 0.15,
    yaw: f32 = 0.0,
    arrived: bool = true,
    snap_to_mesh: bool = true,

    pub fn init(allocator: std.mem.Allocator, nav_mesh: *const NavMesh, start_pos: Vec3) NavAgent {
        const clamped_pos = if (nav_mesh.nodes.len > 0) nav_mesh.clampToMesh(start_pos) else start_pos;
        return .{
            .allocator = allocator,
            .nav_mesh = nav_mesh,
            .position = clamped_pos,
            .velocity = Vec3.zero,
            .target_pos = null,
            .waypoints = &.{},
            .current_waypoint_idx = 0,
            .arrived = true,
        };
    }

    pub fn deinit(self: *NavAgent) void {
        if (self.waypoints.len > 0) {
            self.allocator.free(self.waypoints);
            self.waypoints = &.{};
        }
    }

    pub fn setDestination(self: *NavAgent, target: Vec3) !bool {
        if (self.waypoints.len > 0) {
            self.allocator.free(self.waypoints);
            self.waypoints = &.{};
        }

        const path = try Pathfinding.findPath(self.nav_mesh, self.position, target, self.allocator);
        if (path.len == 0) {
            self.arrived = true;
            self.target_pos = null;
            return false;
        }

        self.waypoints = path;
        self.target_pos = target;
        // Waypoint 0 is start_pos, so head straight toward waypoint 1 if available
        self.current_waypoint_idx = if (path.len > 1) 1 else 0;
        self.arrived = false;
        return true;
    }

    pub fn stop(self: *NavAgent) void {
        self.arrived = true;
        self.velocity = Vec3.zero;
        if (self.waypoints.len > 0) {
            self.allocator.free(self.waypoints);
            self.waypoints = &.{};
        }
    }

    pub fn teleport(self: *NavAgent, pos: Vec3) void {
        self.stop();
        self.position = if (self.snap_to_mesh) self.nav_mesh.clampToMesh(pos) else pos;
    }

    pub fn update(self: *NavAgent, dt: f32) void {
        if (self.arrived or self.waypoints.len == 0 or dt <= 0.0) {
            self.velocity = Vec3.zero;
            return;
        }

        // Advance past waypoints that have already been reached
        while (self.current_waypoint_idx < self.waypoints.len) {
            const wp = self.waypoints[self.current_waypoint_idx];
            const dx = wp.x - self.position.x;
            const dz = wp.z - self.position.z;
            const dist_xz_sq = dx * dx + dz * dz;
            if (dist_xz_sq <= self.stopping_distance * self.stopping_distance) {
                self.current_waypoint_idx += 1;
            } else {
                break;
            }
        }

        if (self.current_waypoint_idx >= self.waypoints.len) {
            self.arrived = true;
            self.velocity = Vec3.zero;
            return;
        }

        const target_wp = self.waypoints[self.current_waypoint_idx];
        const dir = target_wp.sub(self.position);
        const dist_xz = @sqrt(dir.x * dir.x + dir.z * dir.z);

        if (dist_xz > 1e-4) {
            // Desired yaw towards waypoint
            const target_yaw = std.math.atan2(dir.x, dir.z);
            var yaw_diff = target_yaw - self.yaw;
            // Wrap angle to [-PI, PI]
            while (yaw_diff > std.math.pi) yaw_diff -= 2.0 * std.math.pi;
            while (yaw_diff < -std.math.pi) yaw_diff += 2.0 * std.math.pi;

            const max_rot = self.rotation_speed * dt;
            if (@abs(yaw_diff) <= max_rot) {
                self.yaw = target_yaw;
            } else if (yaw_diff > 0.0) {
                self.yaw += max_rot;
            } else {
                self.yaw -= max_rot;
            }

            // Move step
            const move_step = @min(self.speed * dt, dist_xz);
            const move_ratio = move_step / dist_xz;
            self.position.x += dir.x * move_ratio;
            self.position.z += dir.z * move_ratio;

            // Elevation step
            if (@abs(dir.y) > 1e-4) {
                self.position.y += dir.y * move_ratio;
            }

            self.velocity = Vec3.new(
                (dir.x / dist_xz) * self.speed,
                0.0,
                (dir.z / dist_xz) * self.speed,
            );
        } else {
            self.velocity = Vec3.zero;
        }

        if (self.snap_to_mesh) {
            self.position = self.nav_mesh.clampToMesh(self.position);
        }
    }
};
