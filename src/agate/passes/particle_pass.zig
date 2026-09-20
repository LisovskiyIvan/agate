const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const part_shd = @import("particle_shader");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const Camera = @import("../camera.zig").Camera;
const particles = @import("../particles.zig");
const ParticleSystem = particles.ParticleSystem;
const Texture = @import("../texture.zig").Texture;

pub const ParticlePass = struct {
    /// One prepared particle draw: PLAIN render-owned record (values only —
    /// no *ParticleSystem, no game-texture pointer). The prepared frame in
    /// scene/particle_layer.zig retains a list of these; render selects the
    /// pass variant (MSAA) and draws the list without touching live game
    /// state. GPU handles below are BORROWED values: the owning
    /// ParticleSystem (instance/gpu-slot buffers, texture) must outlive the
    /// prepared frame — in practice the layer owns every system until
    /// context teardown, and captureFrame runs after flushGpuUploads so
    /// deferred buffer creations are already visible as real ids.
    /// No full CPU geometry copy: the instance/slot bytes already sit in
    /// the borrowed GPU buffers after the prepare flush.
    pub const ParticleDraw = struct {
        active_count: usize = 0,
        simulation_mode: particles.SimulationMode = .cpu,
        instance_buffer: sg.Buffer = .{},
        gpu_slot_buffer: sg.Buffer = .{},
        /// Compute-baked draw instances (`ParticleInstanceData` layout,
        /// written by the compute pass): drawn through the EXISTING cpu
        /// billboard pipeline, so blend/texture/soft-particle behavior is
        /// identical across modes with no new render pipeline.
        compute_draw_buffer: sg.Buffer = .{},
        blend_mode: particles.ParticleBlendMode = .additive,
        /// Borrowed texture view; null selects the pass default dot.
        texture_view: ?sg.View = null,
        clock_seconds: f32 = 0.0,
        drag: f32 = 0.0,
        gravity: Vec3 = Vec3.zero,
        spritesheet_columns: u32 = 1,
        spritesheet_rows: u32 = 1,
        spritesheet_loops: f32 = 1.0,

        /// Pure snapshot of a live system: copies VALUES only (no sg.* calls,
        /// GPU-free, safe to run in prepare or in tests without a context).
        pub fn fromSystem(ps: *const ParticleSystem) ParticleDraw {
            return .{
                .active_count = ps.active_count,
                .simulation_mode = ps.simulation_mode,
                .instance_buffer = ps.instance_buffer,
                .gpu_slot_buffer = ps.gpu_slot_buffer,
                .compute_draw_buffer = ps.compute_draw_buffer,
                .blend_mode = ps.blend_mode,
                .texture_view = if (ps.texture) |t| t.view else null,
                .clock_seconds = ps.clock_seconds,
                .drag = ps.drag,
                .gravity = ps.gravity,
                .spritesheet_columns = ps.spritesheet_columns,
                .spritesheet_rows = ps.spritesheet_rows,
                .spritesheet_loops = ps.spritesheet_loops,
            };
        }

        /// Buffer the draw binds: mode-selected mirror of the legacy
        /// `if (gpu) ps.gpu_slot_buffer else ps.instance_buffer` choice.
        /// `.compute` binds its baked draw buffer through the cpu pipeline
        /// (same `ParticleInstanceData` stride), so the analytic gpu branch
        /// below stays `.gpu`-only. Pure (no sg.*), so fixtures can assert
        /// the selection headless.
        pub fn drawBuffer(self: ParticleDraw) sg.Buffer {
            if (self.simulation_mode == .gpu) return self.gpu_slot_buffer;
            if (self.simulation_mode == .compute) return self.compute_draw_buffer;
            return self.instance_buffer;
        }
    };

    /// Legacy billboard stats in draw units, computed from snapshot counts
    /// (active_count > 0 counts one draw call + two triangles per particle —
    /// the exact legacy semantics, including draws whose buffer id is still
    /// zero). Pure: shared by the immediate and the prepared stats paths.
    pub const DrawStats = struct {
        draw_calls: u32 = 0,
        triangles: u32 = 0,
    };

    pub fn statsForDraws(draws: []const ParticleDraw) DrawStats {
        var s = DrawStats{};
        for (draws) |d| {
            if (d.active_count > 0) {
                s.draw_calls += 1;
                s.triangles += 2 * @as(u32, @intCast(d.active_count));
            }
        }
        return s;
    }

    pipeline_additive: sg.Pipeline,
    pipeline_alphablend: sg.Pipeline,
    // GPU-simulation variants (shader program particle_gpu): same blend
    // states, but per-instance attributes carry spawn slots instead of
    // integrated state (see particles.GpuParticleSlot).
    pipeline_gpu_additive: sg.Pipeline,
    pipeline_gpu_alphablend: sg.Pipeline,
    shader_cpu: sg.Shader = .{},
    shader_gpu: sg.Shader = .{},

    /// Main-target sample count the render pipelines were built for
    /// (scene/msaa.zig).
    sample_count: i32 = 1,
    quad_vb: sg.Buffer,
    quad_ib: sg.Buffer,
    sampler: sg.Sampler,
    default_texture: Texture,

    /// Builds one pipeline variant per blend mode. `stride`/`attrs` differ
    /// between the CPU path (integrated instance data) and the GPU path
    /// (spawn-slot data); quad geometry, depth and blend setup are shared,
    /// matching the historical pipeline configs bit-for-bit. `sample_count`
    /// must match the main render target (compute pipelines are exempt:
    /// they run in attachment-less compute passes).
    fn makePipeline(shader: sg.Shader, blend: sg.BlendState, slot_stride: usize, gpu: bool, sample_count: i32) sg.Pipeline {
        var desc = sg.PipelineDesc{
            .shader = shader,
            .index_type = .UINT16,
            .depth = .{
                .compare = .LESS_EQUAL,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = sample_count,
        };
        // Buffer 0: Unit quad
        desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        desc.layout.attrs[part_shd.ATTR_particle_position] = .{
            .buffer_index = 0,
            .format = .FLOAT2,
            .offset = 0,
        };
        desc.layout.attrs[part_shd.ATTR_particle_texcoord0] = .{
            .buffer_index = 0,
            .format = .FLOAT2,
            .offset = 2 * @sizeOf(f32),
        };

        // Buffer 1: per-instance data (CPU integrated state or GPU spawn slot).
        desc.layout.buffers[1] = .{
            .step_func = .PER_INSTANCE,
            .step_rate = 1,
            .stride = @intCast(slot_stride),
        };
        if (gpu) {
            inline for (0..5) |i| {
                desc.layout.attrs[part_shd.ATTR_particle_gpu_gpu_slot0 + i] = .{
                    .buffer_index = 1,
                    .format = .FLOAT4,
                    .offset = i * @sizeOf([4]f32),
                };
            }
        } else {
            desc.layout.attrs[part_shd.ATTR_particle_inst_pos_size] = .{
                .buffer_index = 1,
                .format = .FLOAT4,
                .offset = 0,
            };
            desc.layout.attrs[part_shd.ATTR_particle_inst_color] = .{
                .buffer_index = 1,
                .format = .FLOAT4,
                .offset = 4 * @sizeOf(f32),
            };
            desc.layout.attrs[part_shd.ATTR_particle_inst_uv_rect] = .{
                .buffer_index = 1,
                .format = .FLOAT4,
                .offset = 8 * @sizeOf(f32),
            };
            desc.layout.attrs[part_shd.ATTR_particle_inst_rotation] = .{
                .buffer_index = 1,
                .format = .FLOAT4,
                .offset = 12 * @sizeOf(f32),
            };
        }

        desc.colors[0].blend = blend;
        return sg.makePipeline(desc);
    }

    pub fn init() ParticlePass {
        return initSampled(1);
    }

    /// Same pass at a different main-target sample count: sokol requires
    /// pipeline.sample_count to match the attachments of the pass it draws
    /// into.
    pub fn initSampled(sample_count: i32) ParticlePass {
        const particle_quad_vertices = [_]f32{
            // x,     y,     u,   v
            -0.5, -0.5, 0.0, 0.0,
            0.5,  -0.5, 1.0, 0.0,
            0.5,  0.5,  1.0, 1.0,
            -0.5, 0.5,  0.0, 1.0,
        };
        const particle_quad_indices = [_]u16{
            0, 1, 2,
            0, 2, 3,
        };

        const vb = sg.makeBuffer(.{
            .data = sg.asRange(&particle_quad_vertices),
        });
        const ib = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(&particle_quad_indices),
        });

        const smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });

        const shader_cpu = sg.makeShader(part_shd.particleShaderDesc(sg.queryBackend()));
        const shader_gpu = sg.makeShader(part_shd.particleGpuShaderDesc(sg.queryBackend()));

        const blend_additive = sg.BlendState{
            .enabled = true,
            .src_factor_rgb = .SRC_ALPHA,
            .dst_factor_rgb = .ONE,
            .src_factor_alpha = .ONE,
            .dst_factor_alpha = .ONE,
        };
        const blend_alpha = sg.BlendState{
            .enabled = true,
            .src_factor_rgb = .SRC_ALPHA,
            .dst_factor_rgb = .ONE_MINUS_SRC_ALPHA,
            .src_factor_alpha = .ONE,
            .dst_factor_alpha = .ONE_MINUS_SRC_ALPHA,
        };

        return .{
            .pipeline_additive = makePipeline(
                shader_cpu,
                blend_additive,
                @sizeOf(particles.ParticleInstanceData),
                false,
                sample_count,
            ),
            .pipeline_alphablend = makePipeline(
                shader_cpu,
                blend_alpha,
                @sizeOf(particles.ParticleInstanceData),
                false,
                sample_count,
            ),
            .pipeline_gpu_additive = makePipeline(
                shader_gpu,
                blend_additive,
                @sizeOf(particles.GpuParticleSlot),
                true,
                sample_count,
            ),
            .pipeline_gpu_alphablend = makePipeline(
                shader_gpu,
                blend_alpha,
                @sizeOf(particles.GpuParticleSlot),
                true,
                sample_count,
            ),
            .shader_cpu = shader_cpu,
            .shader_gpu = shader_gpu,
            .quad_vb = vb,
            .quad_ib = ib,
            .sampler = smp,
            .default_texture = Texture.createDefaultParticleDot32(),
            .sample_count = sample_count,
        };
    }

    /// Single shared draw algorithm for one prepared record (render side
    /// only: issues sg.* against the pass-owned pipelines/quad/sampler).
    /// Both `render` (immediate, from live systems) and `renderDraws`
    /// (prepared, from the retained frame) funnel through here so the
    /// pipeline/bind/uniform sequence exists exactly once.
    fn drawRecord(
        self: *ParticlePass,
        draw: ParticleDraw,
        view_proj: Mat4,
        cam_right: Vec3,
        cam_up: Vec3,
        current_pipeline: *sg.Pipeline,
    ) void {
        // `.compute` draws its baked instances through the cpu pipeline
        // (gpu == false here); only `.gpu` takes the analytic branch.
        const gpu = draw.simulation_mode == .gpu;
        const instance_buf = draw.drawBuffer();
        if (draw.active_count == 0 or instance_buf.id == 0) return;

        const pip_id = switch (draw.blend_mode) {
            .additive => if (gpu) self.pipeline_gpu_additive.id else self.pipeline_additive.id,
            .alpha_blend => if (gpu) self.pipeline_gpu_alphablend.id else self.pipeline_alphablend.id,
        };
        if (pip_id == 0) return;
        if (current_pipeline.id != pip_id) {
            current_pipeline.id = pip_id;
            sg.applyPipeline(.{ .id = pip_id });
        }

        var bind = sg.Bindings{};
        bind.vertex_buffers[0] = self.quad_vb;
        bind.vertex_buffers[1] = instance_buf;
        bind.index_buffer = self.quad_ib;

        bind.views[part_shd.VIEW_particle_tex] = draw.texture_view orelse self.default_texture.view;
        bind.samplers[part_shd.SMP_smp] = self.sampler;
        sg.applyBindings(bind);

        const vs_params = part_shd.VsParams{
            .view_proj = view_proj,
            .camera_right = .{ cam_right.x, cam_right.y, cam_right.z, 0.0 },
            .camera_up = .{ cam_up.x, cam_up.y, cam_up.z, 0.0 },
        };
        sg.applyUniforms(part_shd.UB_vs_params, sg.asRange(&vs_params));

        if (gpu) {
            // Analytic-simulation parameters (uniform block gpu_params).
            // Slot spawn times and the clock share the system epoch, so
            // both stay small float32 magnitudes.
            const gpu_params = part_shd.GpuParams{
                .time_drag = .{ draw.clock_seconds, draw.drag, 0.0, 0.0 },
                .gravity = .{ draw.gravity.x, draw.gravity.y, draw.gravity.z, 0.0 },
                .sprite = .{
                    @floatFromInt(if (draw.spritesheet_columns == 0) 1 else draw.spritesheet_columns),
                    @floatFromInt(if (draw.spritesheet_rows == 0) 1 else draw.spritesheet_rows),
                    draw.spritesheet_loops,
                    0.0,
                },
            };
            sg.applyUniforms(part_shd.UB_gpu_params, sg.asRange(&gpu_params));
        }

        // active_count is the exact live count on CPU and the written-slot
        // high-water mark on GPU (dead slots cull in the vertex shader).
        sg.draw(0, 6, @intCast(draw.active_count));
    }

    pub fn render(
        self: *ParticlePass,
        systems: []const *ParticleSystem,
        camera: Camera,
        aspect: f32,
    ) void {
        const view_proj = camera.getViewProjection(aspect);
        const view_mat = camera.getViewMatrix();
        const cam_right = Vec3.new(view_mat.m[0], view_mat.m[4], view_mat.m[8]);
        const cam_up = Vec3.new(view_mat.m[1], view_mat.m[5], view_mat.m[9]);

        var current_pipeline: sg.Pipeline = .{};

        for (systems) |ps| {
            self.drawRecord(ParticleDraw.fromSystem(ps), view_proj, cam_right, cam_up, &current_pipeline);
        }
    }

    /// Prepared-frame draw: renders ONLY the retained `ParticleDraw` records
    /// (never live systems), upload-free — the prepare flush already moved
    /// every staged byte into the borrowed buffers.
    pub fn renderDraws(
        self: *ParticlePass,
        draws: []const ParticleDraw,
        camera: Camera,
        aspect: f32,
    ) void {
        const view_proj = camera.getViewProjection(aspect);
        const view_mat = camera.getViewMatrix();
        const cam_right = Vec3.new(view_mat.m[0], view_mat.m[4], view_mat.m[8]);
        const cam_up = Vec3.new(view_mat.m[1], view_mat.m[5], view_mat.m[9]);

        var current_pipeline: sg.Pipeline = .{};

        for (draws) |draw| {
            self.drawRecord(draw, view_proj, cam_right, cam_up, &current_pipeline);
        }
    }
    pub fn deinit(self: *ParticlePass) void {
        sg.destroyPipeline(self.pipeline_additive);
        sg.destroyPipeline(self.pipeline_alphablend);
        sg.destroyPipeline(self.pipeline_gpu_additive);
        sg.destroyPipeline(self.pipeline_gpu_alphablend);
        if (self.shader_cpu.id != 0) sg.destroyShader(self.shader_cpu);
        if (self.shader_gpu.id != 0) sg.destroyShader(self.shader_gpu);
        self.shader_cpu = .{};
        self.shader_gpu = .{};
        sg.destroyBuffer(self.quad_vb);
        sg.destroyBuffer(self.quad_ib);
        sg.destroySampler(self.sampler);
        self.default_texture.deinit();
    }
};

// --- GPU-free prepared-record tests (no sg.* calls below this line;
// fake borrowed handle ids only) ---

fn makePassTestSystem(allocator: std.mem.Allocator, capacity: usize) !ParticleSystem {
    const parts = try allocator.alloc(particles.Particle, capacity);
    errdefer allocator.free(parts);
    const insts = try allocator.alloc(particles.ParticleInstanceData, capacity);
    errdefer allocator.free(insts);
    const scratch = try allocator.alloc(u8, capacity);
    errdefer allocator.free(scratch);
    return ParticleSystem{
        .name = "test",
        .allocator = allocator,
        .particles = parts,
        .instances = insts,
        .alive_scratch = scratch,
        .capacity = capacity,
        .instance_buffer = .{ .id = 11 },
        .prng = std.Random.DefaultPrng.init(42),
    };
}

fn freePassTestSystem(ps: *ParticleSystem) void {
    if (ps.gpu_slots.len > 0) ps.allocator.free(ps.gpu_slots);
    ps.allocator.free(ps.particles);
    ps.allocator.free(ps.instances);
    if (ps.alive_scratch.len > 0) ps.allocator.free(ps.alive_scratch);
}

test "ParticleDraw.fromSystem copies values, never live references" {
    const t = std.testing;
    const math_mod = @import("math");
    var ps = try makePassTestSystem(t.allocator, 4);
    defer freePassTestSystem(&ps);
    ps.active_count = 3;
    ps.simulation_mode = .gpu;
    ps.instance_buffer = .{ .id = 11 };
    ps.gpu_slot_buffer = .{ .id = 12 };
    ps.blend_mode = .alpha_blend;
    var tex: Texture = std.mem.zeroes(Texture);
    tex.view = .{ .id = 77 };
    ps.texture = tex;
    ps.clock_seconds = 1.5;
    ps.drag = 2.0;
    ps.gravity = math_mod.Vec3.new(1.0, -2.0, 3.0);
    ps.spritesheet_columns = 4;
    ps.spritesheet_rows = 2;
    ps.spritesheet_loops = 3.0;

    const draw = ParticlePass.ParticleDraw.fromSystem(&ps);
    try t.expectEqual(@as(usize, 3), draw.active_count);
    try t.expectEqual(particles.SimulationMode.gpu, draw.simulation_mode);
    try t.expectEqual(@as(u32, 11), draw.instance_buffer.id);
    try t.expectEqual(@as(u32, 12), draw.gpu_slot_buffer.id);
    try t.expectEqual(particles.ParticleBlendMode.alpha_blend, draw.blend_mode);
    try t.expect(draw.texture_view != null);
    try t.expectEqual(@as(u32, 77), draw.texture_view.?.id);
    try t.expectEqual(@as(f32, 1.5), draw.clock_seconds);
    try t.expectEqual(@as(f32, 2.0), draw.drag);
    try t.expectEqual(math_mod.Vec3.new(1.0, -2.0, 3.0), draw.gravity);
    try t.expectEqual(@as(u32, 4), draw.spritesheet_columns);
    try t.expectEqual(@as(u32, 2), draw.spritesheet_rows);
    try t.expectEqual(@as(f32, 3.0), draw.spritesheet_loops);

    // No texture -> null (pass substitutes its default dot at draw).
    ps.texture = null;
    try t.expectEqual(@as(?sg.View, null), ParticlePass.ParticleDraw.fromSystem(&ps).texture_view);

    // Mutating the live system leaves the earlier snapshot untouched.
    ps.active_count = 0;
    ps.clock_seconds = 9.0;
    try t.expectEqual(@as(usize, 3), draw.active_count);
    try t.expectEqual(@as(f32, 1.5), draw.clock_seconds);
}

test "ParticleDraw.drawBuffer follows the simulation mode" {
    const t = std.testing;
    var cpu = ParticlePass.ParticleDraw{
        .simulation_mode = .cpu,
        .instance_buffer = .{ .id = 11 },
        .gpu_slot_buffer = .{ .id = 12 },
    };
    try t.expectEqual(@as(u32, 11), cpu.drawBuffer().id);
    cpu.simulation_mode = .gpu;
    try t.expectEqual(@as(u32, 12), cpu.drawBuffer().id);
}

test "statsForDraws preserves the legacy count semantics" {
    const t = std.testing;
    // Zero-count draws contribute nothing (legacy render skips them and the
    // stats loop only counts active_count > 0).
    const draws = [_]ParticlePass.ParticleDraw{
        .{ .active_count = 0 },
        .{ .active_count = 3 },
        .{ .active_count = 5 },
    };
    const s = ParticlePass.statsForDraws(&draws);
    try t.expectEqual(@as(u32, 2), s.draw_calls);
    try t.expectEqual(@as(u32, 2 * (3 + 5)), s.triangles);
    const empty: []const ParticlePass.ParticleDraw = &[_]ParticlePass.ParticleDraw{};
    try t.expectEqual(ParticlePass.DrawStats{}, ParticlePass.statsForDraws(empty));
}

test "ParticleDraw.compute mode binds the baked buffer, keeps cpu visuals" {
    const t = std.testing;
    const math_mod = @import("math");
    var ps = try makePassTestSystem(t.allocator, 4);
    defer freePassTestSystem(&ps);
    ps.simulation_mode = .compute;
    ps.active_count = 3;
    ps.instance_buffer = .{ .id = 11 };
    ps.gpu_slot_buffer = .{ .id = 12 };
    ps.compute_draw_buffer = .{ .id = 13 };
    ps.blend_mode = .alpha_blend;
    ps.color_start = math_mod.Color4.new(1.0, 0.0, 0.0, 1.0);
    ps.size_start = 0.5;

    const draw = ParticlePass.ParticleDraw.fromSystem(&ps);
    try t.expectEqual(particles.SimulationMode.compute, draw.simulation_mode);
    try t.expectEqual(@as(u32, 13), draw.compute_draw_buffer.id);
    // Mode-selected buffer: compute binds its baked instances (cpu-pipeline
    // stride), cpu/gpu selections unchanged.
    try t.expectEqual(@as(u32, 13), draw.drawBuffer().id);
    var cpu_draw = draw;
    cpu_draw.simulation_mode = .cpu;
    try t.expectEqual(@as(u32, 11), cpu_draw.drawBuffer().id);
    var gpu_draw = draw;
    gpu_draw.simulation_mode = .gpu;
    try t.expectEqual(@as(u32, 12), gpu_draw.drawBuffer().id);
    // Stats keep the legacy count semantics in every mode.
    const s = ParticlePass.statsForDraws(&[_]ParticlePass.ParticleDraw{draw});
    try t.expectEqual(@as(u32, 1), s.draw_calls);
    try t.expectEqual(@as(u32, 2 * 3), s.triangles);
}
