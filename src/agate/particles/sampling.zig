//! Spawn sampling and pure visual helpers. Split out of `particles.zig`
//! (facade).
//!
//! `sampleSpawn` takes the system as `anytype` (a `*ParticleSystem` from
//! `system.zig` in practice) so this module never imports `system.zig` or the
//! facade back — same discipline as `profiler/*`. The PRNG call order
//! documented on `sampleSpawn` is part of the CPU behaviour contract — do not
//! reorder. Moved tests reach `system.zig` helpers through block-scoped
//! imports that exist only in test builds.

const std = @import("std");
const math = @import("math");

const Vec3 = math.Vec3;
const Mat4 = math.Mat4;

/// Wraps a degree angle into [0, 360).
pub fn normalizeAngleDeg(angle: f32) f32 {
    var a = @mod(angle, 360.0);
    if (a < 0.0) a += 360.0;
    // @mod of negative multiples (e.g. -360) already yields +0.0; guard -0.0.
    if (a == 0.0) return 0.0;
    return a;
}

/// Degrees -> radians for the instance rotation field.
pub fn rotationToRadians(angle_deg: f32) f32 {
    return angle_deg * std.math.pi / 180.0;
}

/// Guards a zero grid dimension to 1 (avoids div-by-zero; 0 means "no sheet").
inline fn sanitizedGrid(columns: u32, rows: u32) struct { cols: u32, rows: u32 } {
    return .{
        .cols = if (columns == 0) 1 else columns,
        .rows = if (rows == 0) 1 else rows,
    };
}

/// Total frame count of a columns x rows sheet.
pub fn spritesheetFrameCount(columns: u32, rows: u32) u32 {
    const g = sanitizedGrid(columns, rows);
    return g.cols * g.rows;
}

/// Frame index for a particle age: floor(age_norm * loops * frames) % frames,
/// with age_norm = clamp(age / lifetime, 0, 1). At exactly age == lifetime the
/// index wraps to 0 when loops * frames is integral (particle dies there anyway).
pub fn spritesheetFrameForAge(age: f32, lifetime: f32, columns: u32, rows: u32, loops: f32) u32 {
    const frames = spritesheetFrameCount(columns, rows);
    if (frames <= 1) return 0;
    if (!(lifetime > 0.0)) return 0;
    const age_norm = std.math.clamp(age / lifetime, 0.0, 1.0);
    const pos = @floor(age_norm * loops * @as(f32, @floatFromInt(frames)));
    const wrapped = @mod(pos, @as(f32, @floatFromInt(frames)));
    return @intFromFloat(wrapped);
}

/// UV sub-rect for a frame as [offset_u, offset_v, scale_u, scale_v].
/// Frames run left-to-right, bottom-to-top in UV space (frame 0 = UV origin cell).
pub fn spritesheetUvRect(frame: u32, columns: u32, rows: u32) [4]f32 {
    const g = sanitizedGrid(columns, rows);
    const frames = g.cols * g.rows;
    const f = if (frames > 0) frame % frames else 0;
    const col = f % g.cols;
    const row = (f / g.cols) % g.rows;
    const su: f32 = 1.0 / @as(f32, @floatFromInt(g.cols));
    const sv: f32 = 1.0 / @as(f32, @floatFromInt(g.rows));
    return .{
        @as(f32, @floatFromInt(col)) * su,
        @as(f32, @floatFromInt(row)) * sv,
        su,
        sv,
    };
}

/// Transforms a local-space point to world space with an emitter matrix.
pub fn localToWorld(matrix: Mat4, point: Vec3) Vec3 {
    return matrix.transformPoint(point);
}

/// Approximate uniform scale of a TRS matrix: mean length of the basis columns.
/// Exact for uniform scales (factor 1 for identity); heuristic for non-uniform.
pub fn worldScaleFactor(matrix: Mat4) f32 {
    const sx = Vec3.new(matrix.m[0], matrix.m[1], matrix.m[2]).length();
    const sy = Vec3.new(matrix.m[4], matrix.m[5], matrix.m[6]).length();
    const sz = Vec3.new(matrix.m[8], matrix.m[9], matrix.m[10]).length();
    return (sx + sy + sz) / 3.0;
}

inline fn randomRange(rnd: std.Random, min_val: f32, max_val: f32) f32 {
    return min_val + rnd.float(f32) * (max_val - min_val);
}

/// One sampled particle spawn, shared verbatim by both simulation paths so
/// identical seeds produce identical particles regardless of mode. The PRNG
/// call order (box xyz, direction xyz, speed, lifetime, rotation, angular
/// velocity) is part of the CPU behaviour contract — do not reorder.
pub const SpawnSample = struct {
    position: Vec3,
    velocity: Vec3,
    lifetime: f32,
    /// Degrees, normalized to [0, 360).
    rotation_deg: f32,
    /// Degrees/second.
    angular_velocity: f32,
};

pub fn sampleSpawn(self: anytype, rnd: std.Random) SpawnSample {
    // In local_space mode this offset is stored verbatim (emitter-local);
    // the emitter world matrix is applied only at instance-fill time.
    const spawn_pos = Vec3.new(
        self.emitter_position.x + randomRange(rnd, self.emitter_box_min.x, self.emitter_box_max.x),
        self.emitter_position.y + randomRange(rnd, self.emitter_box_min.y, self.emitter_box_max.y),
        self.emitter_position.z + randomRange(rnd, self.emitter_box_min.z, self.emitter_box_max.z),
    );

    const dir = Vec3.new(
        randomRange(rnd, self.direction_min.x, self.direction_max.x),
        randomRange(rnd, self.direction_min.y, self.direction_max.y),
        randomRange(rnd, self.direction_min.z, self.direction_max.z),
    );
    const speed = randomRange(rnd, self.speed_min, self.speed_max);
    const dir_len = dir.length();
    const vel = if (dir_len > 0.0001) dir.scale(speed / dir_len) else Vec3.new(0, speed, 0);

    const lifetime = randomRange(rnd, self.lifetime_min, self.lifetime_max);

    return .{
        .position = spawn_pos,
        .velocity = vel,
        .lifetime = if (lifetime > 0.0001) lifetime else 0.0001,
        .rotation_deg = normalizeAngleDeg(randomRange(rnd, self.rotation_min, self.rotation_max)),
        .angular_velocity = randomRange(rnd, self.angular_velocity_min, self.angular_velocity_max),
    };
}

// --- Tests: spritesheet, rotation and transform helpers (headless) ---

test "spritesheet frames across ages and loops" {
    // 2x2 sheet, lifetime 4s, single loop: one frame per second.
    try std.testing.expectEqual(@as(u32, 0), spritesheetFrameForAge(0.0, 4.0, 2, 2, 1.0));
    try std.testing.expectEqual(@as(u32, 1), spritesheetFrameForAge(1.0, 4.0, 2, 2, 1.0));
    try std.testing.expectEqual(@as(u32, 2), spritesheetFrameForAge(2.0, 4.0, 2, 2, 1.0));
    try std.testing.expectEqual(@as(u32, 3), spritesheetFrameForAge(3.0, 4.0, 2, 2, 1.0));
    // Double loop: age 1s (norm 0.25) -> floor(0.25*2*4) = 2.
    try std.testing.expectEqual(@as(u32, 2), spritesheetFrameForAge(1.0, 4.0, 2, 2, 2.0));
    try std.testing.expectEqual(@as(u32, 1), spritesheetFrameForAge(0.5, 4.0, 2, 2, 2.0));
    // UV rects run left-to-right, bottom-to-top.
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.5, 0.5 }, spritesheetUvRect(0, 2, 2));
    try std.testing.expectEqual([4]f32{ 0.5, 0.0, 0.5, 0.5 }, spritesheetUvRect(1, 2, 2));
    try std.testing.expectEqual([4]f32{ 0.0, 0.5, 0.5, 0.5 }, spritesheetUvRect(2, 2, 2));
    try std.testing.expectEqual([4]f32{ 0.5, 0.5, 0.5, 0.5 }, spritesheetUvRect(3, 2, 2));
}

test "spritesheet boundaries and 1x1 default" {
    // 1x1 (default) always yields frame 0 / full-texture UV.
    try std.testing.expectEqual(@as(u32, 0), spritesheetFrameForAge(0.0, 1.0, 1, 1, 1.0));
    try std.testing.expectEqual(@as(u32, 0), spritesheetFrameForAge(0.99, 1.0, 1, 1, 5.0));
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 1.0, 1.0 }, spritesheetUvRect(0, 1, 1));
    // Boundary: exactly age == lifetime wraps per spec (floor(loops*frames) % frames).
    try std.testing.expectEqual(@as(u32, 0), spritesheetFrameForAge(4.0, 4.0, 2, 2, 1.0));
    try std.testing.expectEqual(@as(u32, 3), spritesheetFrameForAge(3.999, 4.0, 2, 2, 1.0));
    // Zero grid dimensions are guarded to 1 (no div-by-zero).
    try std.testing.expectEqual(@as(u32, 0), spritesheetFrameForAge(0.5, 1.0, 0, 0, 1.0));
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 1.0, 1.0 }, spritesheetUvRect(7, 0, 0));
    try std.testing.expectEqual(@as(u32, 4), spritesheetFrameCount(2, 2));
    try std.testing.expectEqual(@as(u32, 1), spritesheetFrameCount(0, 0));
}

test "rotation integrates angular velocity" {
    const sys = @import("system.zig");
    var ps = try sys.makeTestSystem(std.testing.allocator, 4);
    defer sys.freeTestSystem(&ps);
    ps.direction_min = Vec3.zero;
    ps.direction_max = Vec3.zero;
    ps.speed_min = 0.0;
    ps.speed_max = 0.0;
    ps.gravity = Vec3.zero;
    ps.lifetime_min = 10.0;
    ps.lifetime_max = 10.0;
    ps.rotation_min = 0.0;
    ps.rotation_max = 0.0;
    ps.angular_velocity_min = 90.0;
    ps.angular_velocity_max = 90.0;
    ps.emitOne();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ps.particles[0].rotation, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 90.0), ps.particles[0].angular_velocity, 1e-5);
    ps.updateCpu(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 90.0), ps.particles[0].rotation, 1e-4);
    // Instance carries radians: 90deg = pi/2.
    try std.testing.expectApproxEqAbs(
        std.math.pi / 2.0,
        ps.instances[0].rotation_misc[0],
        1e-5,
    );
    // Wrap-around: 350deg + 20deg/s * 1s = 10deg.
    var ps2 = try sys.makeTestSystem(std.testing.allocator, 4);
    defer sys.freeTestSystem(&ps2);
    ps2.direction_min = Vec3.zero;
    ps2.direction_max = Vec3.zero;
    ps2.speed_min = 0.0;
    ps2.speed_max = 0.0;
    ps2.gravity = Vec3.zero;
    ps2.lifetime_min = 10.0;
    ps2.lifetime_max = 10.0;
    ps2.rotation_min = 350.0;
    ps2.rotation_max = 350.0;
    ps2.angular_velocity_min = 20.0;
    ps2.angular_velocity_max = 20.0;
    ps2.emitOne();
    ps2.updateCpu(1.0);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), ps2.particles[0].rotation, 1e-4);
}

test "angle normalization" {
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), normalizeAngleDeg(0.0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), normalizeAngleDeg(360.0), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), normalizeAngleDeg(-360.0), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), normalizeAngleDeg(370.0), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 350.0), normalizeAngleDeg(-10.0), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 180.0), normalizeAngleDeg(540.0), 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), rotationToRadians(0.0), 1e-6);
    try std.testing.expectApproxEqAbs(std.math.pi, rotationToRadians(180.0), 1e-5);
}

test "localToWorld helper with rotation and scale" {
    const m = Mat4.fromRotationTranslationScale(
        Vec3.new(10.0, 0.0, 0.0),
        Vec3.new(0.0, 0.0, 90.0),
        Vec3.new(2.0, 2.0, 2.0),
    );
    // (1,0,0) -> scaled (2,0,0) -> rotZ90 -> (0,2,0) -> translated (10,2,0).
    const w = localToWorld(m, Vec3.new(1.0, 0.0, 0.0));
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), w.x, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), w.y, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), w.z, 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), worldScaleFactor(m), 1e-5);
    // Identity is exact.
    const id = Mat4.identity;
    const p = localToWorld(id, Vec3.new(1.0, 2.0, 3.0));
    try std.testing.expectEqual(Vec3.new(1.0, 2.0, 3.0), p);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), worldScaleFactor(id), 1e-6);
}
