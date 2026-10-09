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
