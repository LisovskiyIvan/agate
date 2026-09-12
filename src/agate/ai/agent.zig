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
    acceleration: f32 = 8.0,
    rotation_speed: f32 = 8.0,
    stopping_distance: f32 = 0.2,
    waypoint_radius: f32 = 0.8,
    slowdown_distance: f32 = 1.5,
    elevation_speed: f32 = 14.0,
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
            // Smoothly decelerate to zero when idle or stopping
            const damp = @min(self.acceleration * 2.0 * dt, 1.0);
            self.velocity = self.velocity.lerp(Vec3.zero, damp);
            if (self.velocity.lengthSq() < 1e-4) self.velocity = Vec3.zero;
            return;
        }

        // Advance waypoints: check if within radius or passed waypoint plane
        while (self.current_waypoint_idx < self.waypoints.len) {
            const wp = self.waypoints[self.current_waypoint_idx];
            const dx = wp.x - self.position.x;
            const dz = wp.z - self.position.z;
            const dist_xz_sq = dx * dx + dz * dz;

            const is_last = (self.current_waypoint_idx + 1 >= self.waypoints.len);
            if (is_last) {
                if (dist_xz_sq <= self.stopping_distance * self.stopping_distance) {
                    self.current_waypoint_idx += 1;
                } else {
                    break;
                }
            } else {
                const radius = @max(self.waypoint_radius, self.stopping_distance);
                var advance = (dist_xz_sq <= radius * radius);
                if (!advance) {
                    // Check if agent passed the waypoint plane towards the next waypoint
                    const next_wp = self.waypoints[self.current_waypoint_idx + 1];
                    const seg_x = next_wp.x - wp.x;
                    const seg_z = next_wp.z - wp.z;
                    const past_x = self.position.x - wp.x;
                    const past_z = self.position.z - wp.z;
                    if (past_x * seg_x + past_z * seg_z > 0.0) {
                        advance = true;
                    }
                }
                if (advance) {
                    self.current_waypoint_idx += 1;
                } else {
                    break;
                }
            }
        }

        if (self.current_waypoint_idx >= self.waypoints.len) {
            self.arrived = true;
            self.velocity = Vec3.zero;
            return;
        }

        const target_wp = self.waypoints[self.current_waypoint_idx];
        const is_last_wp = (self.current_waypoint_idx + 1 >= self.waypoints.len);
        const dir = target_wp.sub(self.position);
        const dist_xz = @sqrt(dir.x * dir.x + dir.z * dir.z);

        if (dist_xz > 1e-4) {
            const dir_norm_x = dir.x / dist_xz;
            const dir_norm_z = dir.z / dist_xz;

            // Arrival braking near final destination
            var target_speed = self.speed;
            if (is_last_wp and self.slowdown_distance > 1e-3) {
                if (dist_xz < self.slowdown_distance) {
                    const brake_ratio = dist_xz / self.slowdown_distance;
                    target_speed = self.speed * @max(brake_ratio, 0.2);
                }
            }

            // Desired planar velocity
            const desired_v = Vec3.new(dir_norm_x * target_speed, 0.0, dir_norm_z * target_speed);

            // First-order acceleration smoothing (inertial steering)
            const accel_blend = @min(self.acceleration * dt, 1.0);
            self.velocity = self.velocity.lerp(desired_v, accel_blend);

            // Update planar position
            self.position.x += self.velocity.x * dt;
            self.position.z += self.velocity.z * dt;

            // Elevation step
            if (@abs(dir.y) > 1e-4) {
                const fwd_speed = @sqrt(self.velocity.x * self.velocity.x + self.velocity.z * self.velocity.z);
                const move_step = fwd_speed * dt;
                const y_ratio = if (dist_xz > 1e-4) @min(move_step / dist_xz, 1.0) else 1.0;
                self.position.y += dir.y * y_ratio;
            }

            // Desired yaw: aligns with velocity direction when moving, or waypoint direction if starting
            const fwd_speed_sq = self.velocity.x * self.velocity.x + self.velocity.z * self.velocity.z;
            const target_yaw = if (fwd_speed_sq > 0.04)
                std.math.atan2(self.velocity.x, self.velocity.z)
            else
                std.math.atan2(dir.x, dir.z);

            var yaw_diff = target_yaw - self.yaw;
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
        } else {
            const damp = @min(self.acceleration * 2.0 * dt, 1.0);
            self.velocity = self.velocity.lerp(Vec3.zero, damp);
        }

        if (self.snap_to_mesh) {
            const clamped = self.nav_mesh.clampToMesh(self.position);
            // Smooth vertical elevation adjustment so triangle seams don't pop
            const vert_blend = @min(self.elevation_speed * dt, 1.0);
            self.position.y += (clamped.y - self.position.y) * vert_blend;
            self.position.x = clamped.x;
            self.position.z = clamped.z;
        }
    }
};
