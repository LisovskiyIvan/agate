const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Vec3 = math.Vec3;
const Vec4 = math.Vec4;
const Color4 = math.Color4;
const Mat4 = math.Mat4;
const Texture = @import("texture.zig").Texture;

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
};

pub const ParticleInstanceData = extern struct {
    pos_size: [4]f32,
    color: [4]f32,
};

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
        };
        self.active_count += 1;
    }

    pub fn burst(self: *ParticleSystem, count: usize) void {
        var n: usize = 0;
        while (n < count and self.active_count < self.capacity) : (n += 1) {
            self.emitOne();
        }
    }

    pub fn update(self: *ParticleSystem, dt: f32) void {
        if (self.is_emitting and self.emit_rate > 0.0) {
            self.emit_accumulator += dt * self.emit_rate;
            while (self.emit_accumulator >= 1.0 and self.active_count < self.capacity) {
                self.emitOne();
                self.emit_accumulator -= 1.0;
            }
        }

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

            // Physics update (hoisted gravity delta + scaled velocity)
            p.velocity = p.velocity.add(grav_dt);
            p.position = p.position.add(p.velocity.scale(dt));

            const t = p.age / p.lifetime;
            const current_size = p.size + (p.size_end - p.size) * t;
            const current_color = Color4.lerp(p.color, p.color_end, t);

            self.instances[i] = .{
                .pos_size = .{ p.position.x, p.position.y, p.position.z, current_size },
                .color = current_color.toArray(),
            };
            i += 1;
        }

        if (self.active_count > 0) {
            sg.updateBuffer(self.instance_buffer, sg.asRange(self.instances[0..self.active_count]));
        }
    }
};
