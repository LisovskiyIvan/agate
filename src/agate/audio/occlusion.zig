const std = @import("std");
const math = @import("math");
const Vec3 = math.Vec3;
const BusId = @import("types.zig").BusId;

/// Configuration for audio occlusion (acoustical obstruction of sound by geometry).
pub const AudioOcclusionConfig = struct {
    /// Maximum volume factor at 100% occlusion (0.0 = complete silence, 0.25 = -12 dB, 1.0 = no volume loss).
    min_volume: f32 = 0.25,
    /// Lowpass filter cutoff frequency at 100% occlusion (Hz).
    min_cutoff: f32 = 500.0,
    /// Lowpass filter cutoff frequency when completely unoccluded (Hz).
    max_cutoff: f32 = 20000.0,
    /// Number of rays cast: 1 = direct line of sight; 3..5 = multi-tap rays with spread around emitter.
    num_rays: u8 = 1,
    /// Spread radius in meters for multi-tap raycasts around the emitter.
    spread_radius: f32 = 0.6,
    /// Smoothing time constant in seconds for temporal interpolation (prevents audio pops/clicks).
    smooth_time: f32 = 0.15,
};

/// Generic raycast function pointer for occlusion queries.
/// Returns `true` if an obstacle intersects the ray between `origin` and `origin + direction * max_distance`.
pub const RaycastFn = *const fn (origin: Vec3, direction: Vec3, max_distance: f32, user_data: ?*anyopaque) bool;

/// Evaluates raw occlusion factor (0.0 = fully clear line-of-sight, 1.0 = completely occluded).
/// If `config.num_rays <= 1`, performs a single direct raycast.
/// If `config.num_rays > 1`, performs a multi-tap pattern (center + orthogonal taps) to simulate edge diffraction.
pub fn evaluateRaycastOcclusion(
    listener_pos: Vec3,
    emitter_pos: Vec3,
    config: AudioOcclusionConfig,
    raycast_fn: RaycastFn,
    user_data: ?*anyopaque,
) f32 {
    const to = emitter_pos.sub(listener_pos);
    const dist = to.length();
    if (dist < 1e-4) return 0.0;

    const dir = to.scale(1.0 / dist);

    if (config.num_rays <= 1) {
        return if (raycast_fn(listener_pos, dir, dist, user_data)) 1.0 else 0.0;
    }

    // Multi-tap raycast: compute camera-independent orthogonal basis
    var side = dir.cross(Vec3.up);
    if (side.length() < 1e-3) {
        side = dir.cross(Vec3.forward);
    }
    side = side.normalize();
    const up = dir.cross(side).normalize();

    const radius = @max(config.spread_radius, 0.01);
    const taps = [5]Vec3{
        emitter_pos, // center
        emitter_pos.add(side.scale(radius)), // right
        emitter_pos.sub(side.scale(radius)), // left
        emitter_pos.add(up.scale(radius)), // up
        emitter_pos.sub(up.scale(radius)), // down
    };

    const count: usize = std.math.clamp(@as(usize, config.num_rays), 1, 5);
    var blocked: usize = 0;
    for (taps[0..count]) |tap| {
        const to_tap = tap.sub(listener_pos);
        const tap_dist = to_tap.length();
        if (tap_dist < 1e-4) continue;
        const tap_dir = to_tap.scale(1.0 / tap_dist);
        if (raycast_fn(listener_pos, tap_dir, tap_dist, user_data)) {
            blocked += 1;
        }
    }

    return @as(f32, @floatFromInt(blocked)) / @as(f32, @floatFromInt(count));
}

/// Temporal smoother for occlusion values to avoid sudden steps and audio clicks.
pub const AudioOcclusionTracker = struct {
    current: f32 = 0.0,
    target: f32 = 0.0,

    pub fn init(initial_value: f32) AudioOcclusionTracker {
        const clamped = std.math.clamp(initial_value, 0.0, 1.0);
        return .{ .current = clamped, .target = clamped };
    }

    pub fn update(self: *AudioOcclusionTracker, target: f32, dt: f32, smooth_time: f32) f32 {
        self.target = std.math.clamp(target, 0.0, 1.0);
        if (smooth_time <= 1e-4 or dt <= 1e-4) {
            self.current = self.target;
            return self.current;
        }
        // Exponential smoothing: 1 - exp(-dt / smooth_time)
        const alpha = 1.0 - @exp(-dt / @max(smooth_time, 0.001));
        self.current += (self.target - self.current) * std.math.clamp(alpha, 0.0, 1.0);
        return self.current;
    }

    pub fn reset(self: *AudioOcclusionTracker, value: f32) void {
        self.current = std.math.clamp(value, 0.0, 1.0);
        self.target = self.current;
    }
};

/// High-level 3D audio emitter with built-in position, velocity, bus routing,
/// and smooth occlusion tracking.
pub const AudioEmitter = struct {
    position: Vec3 = Vec3.zero,
    velocity: Vec3 = Vec3.zero,
    bus: ?BusId = null,
    occlusion_config: AudioOcclusionConfig = .{},
    tracker: AudioOcclusionTracker = .{},
    active: bool = true,

    pub fn init(position: Vec3, bus: ?BusId) AudioEmitter {
        return .{
            .position = position,
            .bus = bus,
        };
    }

    pub fn setPosition(self: *AudioEmitter, pos: Vec3) void {
        self.position = pos;
    }

    pub fn setVelocity(self: *AudioEmitter, vel: Vec3) void {
        self.velocity = vel;
    }

    pub fn updateOcclusion(
        self: *AudioEmitter,
        listener_pos: Vec3,
        dt: f32,
        raycast_fn: RaycastFn,
        user_data: ?*anyopaque,
    ) f32 {
        const raw = evaluateRaycastOcclusion(listener_pos, self.position, self.occlusion_config, raycast_fn, user_data);
        return self.tracker.update(raw, dt, self.occlusion_config.smooth_time);
    }

    pub fn getOcclusion(self: *const AudioEmitter) f32 {
        return self.tracker.current;
    }
};

test "AudioOcclusionTracker smooth transition" {
    var tracker = AudioOcclusionTracker.init(0.0);
    try std.testing.expectEqual(@as(f32, 0.0), tracker.current);

    // After 0.15s with smooth_time = 0.15s: 1 - 1/e ≈ 0.632
    const v = tracker.update(1.0, 0.15, 0.15);
    try std.testing.expect(v > 0.55 and v < 0.70);

    // After 5 more seconds: reaches target ~1.0
    _ = tracker.update(1.0, 5.0, 0.15);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), tracker.current, 1e-3);

    tracker.reset(0.0);
    try std.testing.expectEqual(@as(f32, 0.0), tracker.current);
}

test "evaluateRaycastOcclusion single and multi-tap" {
    const MockWall = struct {
        wall_x: f32 = 5.0,

        fn raycast(origin: Vec3, direction: Vec3, max_dist: f32, user_data: ?*anyopaque) bool {
            const self: *const @This() = @ptrCast(@alignCast(user_data orelse return false));
            // Ray: origin + t * direction. Check if ray crosses plane x = wall_x within max_dist
            if (@abs(direction.x) < 1e-6) return false;
            const t = (self.wall_x - origin.x) / direction.x;
            return t >= 0.0 and t <= max_dist;
        }
    };

    var wall = MockWall{ .wall_x = 5.0 };
    const listener = Vec3.new(0.0, 0.0, 0.0);
    const emitter_behind = Vec3.new(10.0, 0.0, 0.0);
    const emitter_in_front = Vec3.new(2.0, 0.0, 0.0);

    // Single ray behind wall -> 1.0 (occluded)
    const occ_blocked = evaluateRaycastOcclusion(listener, emitter_behind, .{}, MockWall.raycast, &wall);
    try std.testing.expectEqual(@as(f32, 1.0), occ_blocked);

    // Single ray in front of wall -> 0.0 (clear)
    const occ_clear = evaluateRaycastOcclusion(listener, emitter_in_front, .{}, MockWall.raycast, &wall);
    try std.testing.expectEqual(@as(f32, 0.0), occ_clear);

    // Multi-tap raycast
    const multi_cfg = AudioOcclusionConfig{ .num_rays = 5, .spread_radius = 1.0 };
    const occ_multi = evaluateRaycastOcclusion(listener, emitter_behind, multi_cfg, MockWall.raycast, &wall);
    try std.testing.expectEqual(@as(f32, 1.0), occ_multi);
}
