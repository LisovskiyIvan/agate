const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Vec3 = math.Vec3;
const Vec4 = math.Vec4;
const Color4 = math.Color4;
const Mat4 = math.Mat4;
const Texture = @import("texture.zig").Texture;
const Mesh = @import("mesh.zig").Mesh;

pub const ParticleBlendMode = enum {
    additive,
    alpha_blend,
};

pub const Particle = struct {
    position: Vec3,
    velocity: Vec3,
    size: f32,
    size_end: f32,
    color: Color4,
    color_end: Color4,
    age: f32,
    lifetime: f32,
    /// Z-rotation of the billboard in degrees, normalized to [0, 360).
    rotation: f32 = 0.0,
    /// Spin speed in degrees per second (integrated into rotation by update).
    angular_velocity: f32 = 0.0,
};

pub const ParticleInstanceData = extern struct {
    pos_size: [4]f32,
    color: [4]f32,
    /// Spritesheet sub-rect: xy = UV offset, zw = UV scale (1/columns, 1/rows).
    /// Default (0,0,1,1) reproduces the old full-texture sampling bit-for-bit.
    uv_offset_scale: [4]f32 = .{ 0.0, 0.0, 1.0, 1.0 },
    /// x = billboard rotation in radians, yzw reserved (must stay 0).
    rotation_misc: [4]f32 = .{ 0.0, 0.0, 0.0, 0.0 },
};

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

pub const ParticleSystem = struct {
    name: []const u8,
    allocator: std.mem.Allocator,
    particles: []Particle,
    instances: []ParticleInstanceData,
    capacity: usize,
    active_count: usize = 0,

    instance_buffer: sg.Buffer,
    texture: ?Texture = null,
    blend_mode: ParticleBlendMode = .additive,

    // Emitter shape & origin
    emitter_position: Vec3 = Vec3.zero,
    emitter_box_min: Vec3 = Vec3.zero,
    emitter_box_max: Vec3 = Vec3.zero,

    // Emission rate
    emit_rate: f32 = 100.0,
    is_emitting: bool = false,
    emit_accumulator: f32 = 0.0,

    // Velocity & physics
    direction_min: Vec3 = Vec3.new(-0.2, 1.0, -0.2),
    direction_max: Vec3 = Vec3.new(0.2, 2.0, 0.2),
    speed_min: f32 = 1.0,
    speed_max: f32 = 2.0,
    gravity: Vec3 = Vec3.zero,

    // Visual attributes over lifetime
    color_start: Color4 = Color4.new(1.0, 1.0, 1.0, 1.0),
    color_end: Color4 = Color4.new(1.0, 1.0, 1.0, 0.0),
    size_start: f32 = 0.2,
    size_end: f32 = 0.0,
    lifetime_min: f32 = 1.0,
    lifetime_max: f32 = 2.0,

    // Billboard rotation (degrees) and spin (degrees/second), randomized per particle.
    rotation_min: f32 = 0.0,
    rotation_max: f32 = 0.0,
    angular_velocity_min: f32 = 0.0,
    angular_velocity_max: f32 = 0.0,

    // Spritesheet animation: grid columns x rows, looped `spritesheet_loops`
    // times over each particle's life. 1x1 reproduces the old behavior exactly.
    spritesheet_columns: u32 = 1,
    spritesheet_rows: u32 = 1,
    spritesheet_loops: f32 = 1.0,

    // Local-space simulation: when true, particle positions are stored and
    // integrated in the emitter's local frame; the emitter world matrix is
    // applied only when filling instance data for rendering (see
    // resolveEmitterMatrix/localToWorld). `emitter_position` + box act as a
    // local offset in this mode. Default false = legacy world-space behavior.
    local_space: bool = false,
    emitter_mesh: ?*Mesh = null,

    prng: std.Random.DefaultPrng,

    pub fn init(allocator: std.mem.Allocator, name: []const u8, capacity: usize) !*ParticleSystem {
        const ps = try allocator.create(ParticleSystem);
        errdefer allocator.destroy(ps);

        const particles = try allocator.alloc(Particle, capacity);
        errdefer allocator.free(particles);

        const instances = try allocator.alloc(ParticleInstanceData, capacity);
        errdefer allocator.free(instances);

        const buf = sg.makeBuffer(.{
            .usage = .{ .vertex_buffer = true, .dynamic_update = true },
            .size = capacity * @sizeOf(ParticleInstanceData),
        });

        ps.* = .{
            .name = name,
            .allocator = allocator,
            .particles = particles,
            .instances = instances,
            .capacity = capacity,
            .active_count = 0,
            .instance_buffer = buf,
            .prng = std.Random.DefaultPrng.init(1337),
        };
        return ps;
    }

    pub fn deinit(self: *ParticleSystem) void {
        if (self.instance_buffer.id != 0) {
            sg.destroyBuffer(self.instance_buffer);
            self.instance_buffer = .{};
        }
        if (self.texture) |*t| {
            t.deinit();
            self.texture = null;
        }
        self.allocator.free(self.particles);
        self.allocator.free(self.instances);
    }

    pub fn start(self: *ParticleSystem) void {
        self.is_emitting = true;
    }

    pub fn stop(self: *ParticleSystem) void {
        self.is_emitting = false;
    }

    pub fn reset(self: *ParticleSystem) void {
        self.active_count = 0;
        self.emit_accumulator = 0.0;
    }

    inline fn randomRange(rnd: std.Random, min_val: f32, max_val: f32) f32 {
        return min_val + rnd.float(f32) * (max_val - min_val);
    }

    pub fn emitOne(self: *ParticleSystem) void {
        if (self.active_count >= self.capacity) return;
        const rnd = self.prng.random();

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

        self.particles[self.active_count] = .{
            .position = spawn_pos,
            .velocity = vel,
            .size = self.size_start,
            .size_end = self.size_end,
            .color = self.color_start,
            .color_end = self.color_end,
            .age = 0.0,
            .lifetime = if (lifetime > 0.0001) lifetime else 0.0001,
            .rotation = normalizeAngleDeg(randomRange(rnd, self.rotation_min, self.rotation_max)),
            .angular_velocity = randomRange(rnd, self.angular_velocity_min, self.angular_velocity_max),
        };
        self.active_count += 1;
    }

    pub fn burst(self: *ParticleSystem, count: usize) void {
        var n: usize = 0;
        while (n < count and self.active_count < self.capacity) : (n += 1) {
            self.emitOne();
        }
    }

    /// Emitter world matrix for local-space rendering, or null when world-space
    /// rendering applies (local_space == false or no emitter_mesh bound).
    /// With local_space == true but emitter_mesh == null the local coordinates
    /// pass through unchanged (identity).
    pub fn resolveEmitterMatrix(self: *const ParticleSystem) ?Mat4 {
        if (!self.local_space) return null;
        const mesh = self.emitter_mesh orelse return null;
        return mesh.getWorldMatrix();
    }

    /// GPU-free simulation step: emission, physics, rotation integration and
    /// instance-data fill. `update` calls this and then uploads to the GPU.
    pub fn updateCpu(self: *ParticleSystem, dt: f32) void {
        if (self.is_emitting and self.emit_rate > 0.0) {
            self.emit_accumulator += dt * self.emit_rate;
            while (self.emit_accumulator >= 1.0 and self.active_count < self.capacity) {
                self.emitOne();
                self.emit_accumulator -= 1.0;
            }
        }

        const emitter_matrix = self.resolveEmitterMatrix();
        const emitter_scale = if (emitter_matrix) |m| worldScaleFactor(m) else 1.0;
        const grav_dt = self.gravity.scale(dt);
        var i: usize = 0;
        while (i < self.active_count) {
            var p = &self.particles[i];
            p.age += dt;
            if (p.age >= p.lifetime) {
                // Swap with last active particle
                self.active_count -= 1;
                if (i < self.active_count) {
                    self.particles[i] = self.particles[self.active_count];
                    continue;
                } else {
                    break;
                }
            }

            // Physics update (hoisted gravity delta + scaled velocity).
            // In local_space mode gravity/velocity integrate in emitter-local
            // units; the world transform applies below at instance-fill time.
            p.velocity = p.velocity.add(grav_dt);
            p.position = p.position.add(p.velocity.scale(dt));
            p.rotation = normalizeAngleDeg(p.rotation + p.angular_velocity * dt);

            const t = p.age / p.lifetime;
            const current_size = p.size + (p.size_end - p.size) * t;
            const current_color = Color4.lerp(p.color, p.color_end, t);
            const frame = spritesheetFrameForAge(
                p.age,
                p.lifetime,
                self.spritesheet_columns,
                self.spritesheet_rows,
                self.spritesheet_loops,
            );
            const uv = spritesheetUvRect(frame, self.spritesheet_columns, self.spritesheet_rows);

            var render_pos = p.position;
            var render_size = current_size;
            if (emitter_matrix) |m| {
                render_pos = localToWorld(m, p.position);
                render_size = current_size * emitter_scale;
            }

            self.instances[i] = .{
                .pos_size = .{ render_pos.x, render_pos.y, render_pos.z, render_size },
                .color = current_color.toArray(),
                .uv_offset_scale = uv,
                .rotation_misc = .{ rotationToRadians(p.rotation), 0.0, 0.0, 0.0 },
            };
            i += 1;
        }
    }

    pub fn update(self: *ParticleSystem, dt: f32) void {
        self.updateCpu(dt);

        if (self.active_count > 0) {
            sg.updateBuffer(self.instance_buffer, sg.asRange(self.instances[0..self.active_count]));
        }
    }
};

// --- GPU-free test helpers & tests (no sg.* calls below this line) ---

fn makeTestSystem(allocator: std.mem.Allocator, capacity: usize) !ParticleSystem {
    const parts = try allocator.alloc(Particle, capacity);
    errdefer allocator.free(parts);
    const insts = try allocator.alloc(ParticleInstanceData, capacity);
    errdefer allocator.free(insts);
    return ParticleSystem{
        .name = "test",
        .allocator = allocator,
        .particles = parts,
        .instances = insts,
        .capacity = capacity,
        .instance_buffer = .{},
        .prng = std.Random.DefaultPrng.init(42),
    };
}

fn freeTestSystem(ps: *ParticleSystem) void {
    ps.allocator.free(ps.particles);
    ps.allocator.free(ps.instances);
}

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
    var ps = try makeTestSystem(std.testing.allocator, 4);
    defer freeTestSystem(&ps);
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
    var ps2 = try makeTestSystem(std.testing.allocator, 4);
    defer freeTestSystem(&ps2);
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

test "defaults keep world positions bit-identical" {
    var ps = try makeTestSystem(std.testing.allocator, 4);
    defer freeTestSystem(&ps);
    // Defaults: local_space=false, emitter_mesh=null, 1x1 sheet, zero rotation.
    try std.testing.expectEqual(false, ps.local_space);
    try std.testing.expectEqual(@as(?*Mesh, null), ps.emitter_mesh);
    ps.emitter_position = Vec3.new(1.0, 2.0, 3.0);
    ps.direction_min = Vec3.zero;
    ps.direction_max = Vec3.zero;
    ps.speed_min = 0.0;
    ps.speed_max = 0.0;
    ps.gravity = Vec3.zero;
    ps.lifetime_min = 10.0;
    ps.lifetime_max = 10.0;
    ps.emitOne();
    ps.updateCpu(0.0);
    const p = ps.particles[0];
    const inst = ps.instances[0];
    try std.testing.expectEqual(p.position.x, inst.pos_size[0]);
    try std.testing.expectEqual(p.position.y, inst.pos_size[1]);
    try std.testing.expectEqual(p.position.z, inst.pos_size[2]);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 1.0, 1.0 }, inst.uv_offset_scale);
    try std.testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, inst.rotation_misc);
    try std.testing.expectEqual(@as(?Mat4, null), ps.resolveEmitterMatrix());
}

test "local_space spawn stays relative, instances follow emitter" {
    var ps = try makeTestSystem(std.testing.allocator, 4);
    defer freeTestSystem(&ps);
    var emitter: Mesh = std.mem.zeroes(Mesh);
    emitter.position = Vec3.new(5.0, 0.0, 0.0);
    emitter.rotation = Vec3.zero;
    emitter.scaling = Vec3.one;
    emitter.base_matrix = Mat4.identity;

    ps.local_space = true;
    ps.emitter_mesh = &emitter;
    ps.emitter_position = Vec3.new(1.0, 2.0, 3.0);
    ps.direction_min = Vec3.zero;
    ps.direction_max = Vec3.zero;
    ps.speed_min = 0.0;
    ps.speed_max = 0.0;
    ps.gravity = Vec3.zero;
    ps.lifetime_min = 10.0;
    ps.lifetime_max = 10.0;
    ps.emitOne();

    // Stored coordinates are emitter-local, not world.
    try std.testing.expectEqual(Vec3.new(1.0, 2.0, 3.0), ps.particles[0].position);

    ps.updateCpu(0.0);
    // Instance = world matrix applied: (1,2,3) + emitter offset (5,0,0).
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), ps.instances[0].pos_size[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), ps.instances[0].pos_size[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), ps.instances[0].pos_size[2], 1e-5);

    // Moving the emitter moves rendered instances, stored locals stay put.
    emitter.position = Vec3.new(10.0, 0.0, 0.0);
    ps.updateCpu(0.0);
    try std.testing.expectEqual(Vec3.new(1.0, 2.0, 3.0), ps.particles[0].position);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), ps.instances[0].pos_size[0], 1e-5);
}
