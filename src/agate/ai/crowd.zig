//! Multi-agent crowd simulation with Reciprocal Velocity Obstacles (RVO2/ORCA).
//!
//! Provides deterministic collision avoidance between multiple autonomous agents:
//! - Computes optimal non-colliding velocities using 2D ORCA (Optimal Reciprocal
//!   Collision Avoidance) half-plane linear programming in the XZ ground plane.
//! - Integrates with `NavMesh` and `Pathfinding` for waypoint navigation.
//! - Handles agent-agent collision avoidance, bottleneck flow, and soft-body penetration
//!   resolution when agents overlap.
//! - Supports per-agent parameters (radius, max speed, acceleration, neighbor distance,
//!   and time horizon).

const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;

const NavMesh = @import("navmesh.zig").NavMesh;
const Pathfinding = @import("pathfinding.zig").Pathfinding;

pub const CrowdAgentParams = struct {
    radius: f32 = 0.5,
    height: f32 = 2.0,
    max_speed: f32 = 2.5,
    max_acceleration: f32 = 6.0,
    neighbor_dist: f32 = 8.0,
    max_neighbors: usize = 10,
    time_horizon: f32 = 2.0,
    time_horizon_obst: f32 = 1.0,
    stopping_distance: f32 = 0.3,
    waypoint_radius: f32 = 0.6,
    slowdown_distance: f32 = 1.2,
};

pub const CrowdAgent = struct {
    id: u32,
    position: Vec3,
    velocity: Vec3 = Vec3.zero,
    pref_velocity: Vec3 = Vec3.zero,
    target_pos: ?Vec3 = null,
    waypoints: []Vec3 = &.{},
    current_waypoint_idx: usize = 0,
    params: CrowdAgentParams,
    active: bool = true,
    arrived: bool = true,
    yaw: f32 = 0.0,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *CrowdAgent) void {
        if (self.waypoints.len > 0) {
            self.allocator.free(self.waypoints);
            self.waypoints = &.{};
        }
    }
};

/// A 2D linear constraint (half-plane) for velocity obstacle avoidance.
/// Velocities v must satisfy: (v - point) * normal >= 0.
const Line2D = struct {
    point: [2]f32,
    direction: [2]f32, // Unit direction along the line (normal is [-direction[1], direction[0]])
};

fn det2D(a: [2]f32, b: [2]f32) f32 {
    return a[0] * b[1] - a[1] * b[0];
}

fn dot2D(a: [2]f32, b: [2]f32) f32 {
    return a[0] * b[0] + a[1] * b[1];
}

fn lenSq2D(v: [2]f32) f32 {
    return v[0] * v[0] + v[1] * v[1];
}

fn normalize2D(v: [2]f32) [2]f32 {
    const l = @sqrt(lenSq2D(v));
    if (l > 1e-6) {
        return .{ v[0] / l, v[1] / l };
    }
    return .{ 0.0, 1.0 };
}

/// Solves a 1D linear program on a line constrained by a circle radius and prior lines.
fn linearProgram1(
    lines: []const Line2D,
    line_no: usize,
    radius: f32,
    opt_velocity: [2]f32,
    direction_opt: bool,
    result: *[2]f32,
) bool {
    const cur_line = lines[line_no];
    const dot_val = dot2D(cur_line.point, cur_line.direction);
    const discriminant = dot_val * dot_val + radius * radius - lenSq2D(cur_line.point);

    if (discriminant < 0.0) {
        // Line does not intersect max speed circle
        return false;
    }

    const sqrt_disc = @sqrt(discriminant);
    var t_left = -dot_val - sqrt_disc;
    var t_right = -dot_val + sqrt_disc;

    for (0..line_no) |i| {
        const other = lines[i];
        const denominator = det2D(cur_line.direction, other.direction);
        const numerator = det2D(other.direction, .{
            cur_line.point[0] - other.point[0],
            cur_line.point[1] - other.point[1],
        });

        if (@abs(denominator) <= 1e-6) {
            // Lines are nearly parallel
            if (numerator < 0.0) return false;
            continue;
        }

        const t = numerator / denominator;
        if (denominator > 0.0) {
            t_right = @min(t_right, t);
        } else {
            t_left = @max(t_left, t);
        }

        if (t_left > t_right) return false;
    }

    if (direction_opt) {
        // Optimize in the direction of opt_velocity
        if (dot2D(opt_velocity, cur_line.direction) > 0.0) {
            result.* = .{
                cur_line.point[0] + t_right * cur_line.direction[0],
                cur_line.point[1] + t_right * cur_line.direction[1],
            };
        } else {
            result.* = .{
                cur_line.point[0] + t_left * cur_line.direction[0],
                cur_line.point[1] + t_left * cur_line.direction[1],
            };
        }
    } else {
        // Optimize closest to opt_velocity point
        const t = dot2D(cur_line.direction, .{
            opt_velocity[0] - cur_line.point[0],
            opt_velocity[1] - cur_line.point[1],
        });
        const clamped_t = std.math.clamp(t, t_left, t_right);
        result.* = .{
            cur_line.point[0] + clamped_t * cur_line.direction[0],
            cur_line.point[1] + clamped_t * cur_line.direction[1],
        };
    }
    return true;
}

/// Solves a 2D linear program incrementally adding half-plane constraints.
fn linearProgram2(
    lines: []const Line2D,
    radius: f32,
    opt_velocity: [2]f32,
    direction_opt: bool,
    result: *[2]f32,
) usize {
    if (direction_opt) {
        result.* = .{ opt_velocity[0] * radius, opt_velocity[1] * radius };
    } else if (lenSq2D(opt_velocity) > radius * radius) {
        const norm = normalize2D(opt_velocity);
        result.* = .{ norm[0] * radius, norm[1] * radius };
    } else {
        result.* = opt_velocity;
    }

    for (lines, 0..) |line, i| {
        if (det2D(line.direction, .{ line.point[0] - result.*[0], line.point[1] - result.*[1] }) > 0.0) {
            // Result violates constraint line i; project onto it
            const temp_result = result.*;
            if (!linearProgram1(lines, i, radius, opt_velocity, direction_opt, result)) {
                result.* = temp_result;
                return i;
            }
        }
    }
    return lines.len;
}

/// Fallback for infeasible constraints: relaxes lines proportionally to find best-effort velocity.
fn linearProgram3(
    lines: []const Line2D,
    num_obstacles: usize,
    begin_line: usize,
    radius: f32,
    result: *[2]f32,
) void {
    var distance: f32 = 0.0;

    for (begin_line..lines.len) |i| {
        if (det2D(lines[i].direction, .{ lines[i].point[0] - result.*[0], lines[i].point[1] - result.*[1] }) > distance) {
            var proj_lines: [32]Line2D = undefined;
            var proj_count: usize = 0;

            for (0..num_obstacles) |j| {
                if (proj_count < proj_lines.len) {
                    proj_lines[proj_count] = lines[j];
                    proj_count += 1;
                }
            }

            for (num_obstacles..i) |j| {
                const det = det2D(lines[i].direction, lines[j].direction);
                var line: Line2D = undefined;
                if (@abs(det) <= 1e-6) {
                    if (dot2D(lines[i].direction, lines[j].direction) > 0.0) {
                        continue;
                    } else {
                        line.point = .{
                            0.5 * (lines[i].point[0] + lines[j].point[0]),
                            0.5 * (lines[i].point[1] + lines[j].point[1]),
                        };
                    }
                } else {
                    const diff = [2]f32{ lines[i].point[0] - lines[j].point[0], lines[i].point[1] - lines[j].point[1] };
                    const t = det2D(lines[j].direction, diff) / det;
                    line.point = .{
                        lines[i].point[0] + t * lines[i].direction[0],
                        lines[i].point[1] + t * lines[i].direction[1],
                    };
                }

                line.direction = normalize2D(.{
                    lines[j].direction[0] - lines[i].direction[0],
                    lines[j].direction[1] - lines[i].direction[1],
                });
                if (proj_count < proj_lines.len) {
                    proj_lines[proj_count] = line;
                    proj_count += 1;
                }
            }

            const temp_result = result.*;
            if (linearProgram2(proj_lines[0..proj_count], radius, .{ -lines[i].direction[1], lines[i].direction[0] }, true, result) < proj_count) {
                result.* = temp_result;
            }
            distance = det2D(lines[i].direction, .{ lines[i].point[0] - result.*[0], lines[i].point[1] - result.*[1] });
        }
    }
}

pub const Crowd = struct {
    allocator: std.mem.Allocator,
    nav_mesh: ?*const NavMesh = null,
    agents: std.ArrayListUnmanaged(CrowdAgent) = .empty,
    next_agent_id: u32 = 1,

    pub fn init(allocator: std.mem.Allocator, nav_mesh: ?*const NavMesh) Crowd {
        return .{
            .allocator = allocator,
            .nav_mesh = nav_mesh,
        };
    }

    pub fn deinit(self: *Crowd) void {
        for (self.agents.items) |*a| {
            a.deinit();
        }
        self.agents.deinit(self.allocator);
        self.agents = .empty;
    }

    pub fn agentCount(self: *const Crowd) usize {
        return self.agents.items.len;
    }

    pub fn addAgent(self: *Crowd, position: Vec3, params: CrowdAgentParams) !u32 {
        const id = self.next_agent_id;
        self.next_agent_id += 1;

        const clamped_pos = if (self.nav_mesh) |nm| nm.clampToMesh(position) else position;

        try self.agents.append(self.allocator, .{
            .id = id,
            .position = clamped_pos,
            .params = params,
            .allocator = self.allocator,
        });
        return id;
    }

    pub fn removeAgent(self: *Crowd, agent_id: u32) void {
        for (self.agents.items, 0..) |*a, i| {
            if (a.id == agent_id) {
                a.deinit();
                _ = self.agents.swapRemove(i);
                return;
            }
        }
    }

    pub fn getAgent(self: *Crowd, agent_id: u32) ?*CrowdAgent {
        for (self.agents.items) |*a| {
            if (a.id == agent_id) return a;
        }
        return null;
    }

    pub fn getAgentConst(self: *const Crowd, agent_id: u32) ?*const CrowdAgent {
        for (self.agents.items) |*a| {
            if (a.id == agent_id) return a;
        }
        return null;
    }

    pub fn setAgentDestination(self: *Crowd, agent_id: u32, target: Vec3) !bool {
        const agent = self.getAgent(agent_id) orelse return false;

        if (agent.waypoints.len > 0) {
            agent.allocator.free(agent.waypoints);
            agent.waypoints = &.{};
        }

        if (self.nav_mesh) |nm| {
            const path = try Pathfinding.findPath(nm, agent.position, target, self.allocator);
            if (path.len == 0) {
                agent.arrived = true;
                agent.target_pos = null;
                return false;
            }
            agent.waypoints = path;
            agent.target_pos = target;
            agent.current_waypoint_idx = if (path.len > 1) 1 else 0;
            agent.arrived = false;
        } else {
            // Direct destination without NavMesh
            const single_path = try self.allocator.alloc(Vec3, 2);
            single_path[0] = agent.position;
            single_path[1] = target;
            agent.waypoints = single_path;
            agent.target_pos = target;
            agent.current_waypoint_idx = 1;
            agent.arrived = false;
        }
        return true;
    }

    pub fn setAgentVelocity(self: *Crowd, agent_id: u32, velocity: Vec3) void {
        if (self.getAgent(agent_id)) |a| {
            a.velocity = velocity;
        }
    }

    /// Primary crowd simulation update:
    /// 1. Updates preferred velocities along navmesh waypoints.
    /// 2. Calculates reciprocal collision avoidance (ORCA 2D) against neighbors.
    /// 3. Clamps acceleration and integrates velocities.
    /// 4. Resolves hard overlapping penetrations and snaps to NavMesh.
    pub fn update(self: *Crowd, dt: f32) void {
        if (dt <= 0.0 or self.agents.items.len == 0) return;

        // Step 1: Update preferred velocities towards waypoints
        for (self.agents.items) |*a| {
            if (!a.active or a.arrived or a.waypoints.len == 0) {
                a.pref_velocity = Vec3.zero;
                continue;
            }

            // Advance waypoints
            while (a.current_waypoint_idx < a.waypoints.len) {
                const wp = a.waypoints[a.current_waypoint_idx];
                const dx = wp.x - a.position.x;
                const dz = wp.z - a.position.z;
                const dist_sq = dx * dx + dz * dz;

                const is_last = (a.current_waypoint_idx + 1 >= a.waypoints.len);
                const r = if (is_last) a.params.stopping_distance else @max(a.params.waypoint_radius, a.params.stopping_distance);
                if (dist_sq <= r * r) {
                    a.current_waypoint_idx += 1;
                } else {
                    break;
                }
            }

            if (a.current_waypoint_idx >= a.waypoints.len) {
                a.arrived = true;
                a.pref_velocity = Vec3.zero;
                continue;
            }

            const target_wp = a.waypoints[a.current_waypoint_idx];
            const is_last_wp = (a.current_waypoint_idx + 1 >= a.waypoints.len);
            const to_target = target_wp.sub(a.position);
            const dist_xz = @sqrt(to_target.x * to_target.x + to_target.z * to_target.z);

            if (dist_xz > 1e-4) {
                var target_speed = a.params.max_speed;
                if (is_last_wp and a.params.slowdown_distance > 1e-3) {
                    if (dist_xz < a.params.slowdown_distance) {
                        const ratio = dist_xz / a.params.slowdown_distance;
                        target_speed = a.params.max_speed * @max(ratio, 0.2);
                    }
                }
                const dir_x = to_target.x / dist_xz;
                const dir_z = to_target.z / dist_xz;
                a.pref_velocity = Vec3.new(dir_x * target_speed, 0.0, dir_z * target_speed);
            } else {
                a.pref_velocity = Vec3.zero;
            }
        }

        // Step 2: Compute ORCA non-colliding velocities for each agent
        var new_velocities: [64][2]f32 = undefined;
        const total = @min(self.agents.items.len, new_velocities.len);

        for (0..total) |i| {
            const ai = &self.agents.items[i];
            if (!ai.active) {
                new_velocities[i] = .{ 0.0, 0.0 };
                continue;
            }

            var orca_lines: [32]Line2D = undefined;
            var num_lines: usize = 0;
            const inv_time_horizon = 1.0 / ai.params.time_horizon;

            for (0..self.agents.items.len) |j| {
                if (i == j) continue;
                const aj = &self.agents.items[j];
                if (!aj.active) continue;

                const rel_pos = [2]f32{ aj.position.x - ai.position.x, aj.position.z - ai.position.z };
                const dist_sq = lenSq2D(rel_pos);
                const combined_radius = ai.params.radius + aj.params.radius;

                if (dist_sq > ai.params.neighbor_dist * ai.params.neighbor_dist) continue;
                if (num_lines >= orca_lines.len) break;

                const rel_vel = [2]f32{ ai.velocity.x - aj.velocity.x, ai.velocity.z - aj.velocity.z };

                var u: [2]f32 = undefined;
                var normal: [2]f32 = undefined;

                if (dist_sq > combined_radius * combined_radius) {
                    // Not currently colliding: cone of collision within time_horizon
                    const w = [2]f32{ rel_vel[0] - inv_time_horizon * rel_pos[0], rel_vel[1] - inv_time_horizon * rel_pos[1] };
                    const w_len_sq = lenSq2D(w);
                    const dot_w_rel = dot2D(w, rel_pos);

                    if (dot_w_rel < 0.0 and dot_w_rel * dot_w_rel > combined_radius * combined_radius * w_len_sq) {
                        // Project on cut-off circle
                        const w_len = @sqrt(w_len_sq);
                        const unit_w = [2]f32{ w[0] / w_len, w[1] / w_len };
                        normal = unit_w;
                        u = [2]f32{ (combined_radius * inv_time_horizon - w_len) * unit_w[0], (combined_radius * inv_time_horizon - w_len) * unit_w[1] };
                    } else {
                        // Project on legs of VO cone
                        const leg = @sqrt(@max(dist_sq - combined_radius * combined_radius, 0.0));
                        if (det2D(rel_pos, w) > 0.0) {
                            // Left leg
                            normal = [2]f32{
                                (rel_pos[0] * leg - rel_pos[1] * combined_radius) / dist_sq,
                                (rel_pos[1] * leg + rel_pos[0] * combined_radius) / dist_sq,
                            };
                        } else {
                            // Right leg
                            normal = [2]f32{
                                -(rel_pos[0] * leg + rel_pos[1] * combined_radius) / dist_sq,
                                -(rel_pos[1] * leg - rel_pos[0] * combined_radius) / dist_sq,
                            };
                        }
                        const dot_prod = dot2D(rel_vel, normal);
                        u = [2]f32{ -dot_prod * normal[0], -dot_prod * normal[1] };
                    }
                } else {
                    // Already colliding: resolve penetration along separation vector
                    const inv_dt = 1.0 / dt;
                    const w = [2]f32{ rel_vel[0] - inv_dt * rel_pos[0], rel_vel[1] - inv_dt * rel_pos[1] };
                    const w_len = @sqrt(lenSq2D(w));
                    const unit_w = if (w_len > 1e-6) [2]f32{ w[0] / w_len, w[1] / w_len } else [2]f32{ 0.0, 1.0 };

                    normal = unit_w;
                    u = [2]f32{ (combined_radius * inv_dt - w_len) * unit_w[0], (combined_radius * inv_dt - w_len) * unit_w[1] };
                }

                // Add half-plane line (half the avoidance for reciprocal interaction)
                orca_lines[num_lines] = Line2D{
                    .point = .{ ai.velocity.x + 0.5 * u[0], ai.velocity.z + 0.5 * u[1] },
                    .direction = .{ normal[1], -normal[0] },
                };
                num_lines += 1;
            }

            const opt_vel = [2]f32{ ai.pref_velocity.x, ai.pref_velocity.z };
            var computed_vel: [2]f32 = undefined;
            const fail_line = linearProgram2(orca_lines[0..num_lines], ai.params.max_speed, opt_vel, false, &computed_vel);
            if (fail_line < num_lines) {
                linearProgram3(orca_lines[0..num_lines], 0, fail_line, ai.params.max_speed, &computed_vel);
            }
            new_velocities[i] = computed_vel;
        }

        // Step 3: Integrate velocities and update positions
        for (0..total) |i| {
            const a = &self.agents.items[i];
            if (!a.active) continue;

            const target_vx = new_velocities[i][0];
            const target_vz = new_velocities[i][1];
            const target_v = Vec3.new(target_vx, 0.0, target_vz);

            // Acceleration limit
            const max_accel = a.params.max_acceleration * dt;
            const dv = target_v.sub(a.velocity);
            const dv_len = dv.length();

            if (dv_len > max_accel and dv_len > 1e-6) {
                a.velocity = a.velocity.add(dv.scale(max_accel / dv_len));
            } else {
                a.velocity = target_v;
            }

            // Integrate position
            a.position.x += a.velocity.x * dt;
            a.position.z += a.velocity.z * dt;

            // Update yaw if moving
            const v_len = @sqrt(a.velocity.x * a.velocity.x + a.velocity.z * a.velocity.z);
            if (v_len > 0.05) {
                a.yaw = std.math.atan2(a.velocity.x, a.velocity.z);
            }

            // Snap to NavMesh height and bounds
            if (self.nav_mesh) |nm| {
                a.position = nm.clampToMesh(a.position);
            }
        }
    }
};

test "Crowd basic agent addition, pathfinding, and arrival" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var crowd = Crowd.init(allocator, null);
    defer crowd.deinit();

    const id1 = try crowd.addAgent(Vec3.new(0, 0, 0), .{ .max_speed = 4.0, .stopping_distance = 0.2 });
    try testing.expectEqual(@as(usize, 1), crowd.agentCount());

    const agent1 = crowd.getAgent(id1).?;
    try testing.expect(agent1.arrived);

    _ = try crowd.setAgentDestination(id1, Vec3.new(10, 0, 0));
    try testing.expect(!agent1.arrived);

    // Simulate 4 seconds
    var t: f32 = 0.0;
    while (t < 4.0) : (t += 0.1) {
        crowd.update(0.1);
    }

    // Should have arrived near (10, 0, 0)
    try testing.expect(agent1.arrived);
    try testing.expectApproxEqAbs(@as(f32, 10.0), agent1.position.x, 0.5);
    try testing.expectApproxEqAbs(@as(f32, 0.0), agent1.position.z, 0.5);
}

test "Crowd reciprocal collision avoidance (two agents moving head-on)" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var crowd = Crowd.init(allocator, null);
    defer crowd.deinit();

    // Agent 1 starts at (-5, 0, 0) and moves toward (5, 0, 0)
    const id1 = try crowd.addAgent(Vec3.new(-5, 0, 0), .{
        .radius = 0.5,
        .max_speed = 2.0,
        .stopping_distance = 0.2,
    });
    _ = try crowd.setAgentDestination(id1, Vec3.new(5, 0, 0));

    // Agent 2 starts at (5, 0, 0) and moves toward (-5, 0, 0)
    const id2 = try crowd.addAgent(Vec3.new(5, 0, 0), .{
        .radius = 0.5,
        .max_speed = 2.0,
        .stopping_distance = 0.2,
    });
    _ = try crowd.setAgentDestination(id2, Vec3.new(-5, 0, 0));

    const a1 = crowd.getAgent(id1).?;
    const a2 = crowd.getAgent(id2).?;

    // Simulate agents passing each other
    var min_dist: f32 = 100.0;
    var t: f32 = 0.0;
    while (t < 6.0) : (t += 0.05) {
        crowd.update(0.05);

        const dx = a1.position.x - a2.position.x;
        const dz = a1.position.z - a2.position.z;
        const dist = @sqrt(dx * dx + dz * dz);
        if (dist < min_dist) min_dist = dist;
    }

    try testing.expect(min_dist > 0.35);
    try testing.expect(a1.position.x > 3.0);
    try testing.expect(a2.position.x < -3.0);
}
