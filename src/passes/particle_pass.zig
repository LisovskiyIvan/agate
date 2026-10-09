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
pub const TargetShape = @import("../target_shape.zig").TargetShape;
pub const defaultDepthFormat = @import("../target_shape.zig").defaultDepthFormat;

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

        /// Buffer the draw binds: mode-selected mirror of the
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

    /// Billboard stats in draw units, computed from snapshot counts
    /// (active_count > 0 counts one draw call + two triangles per particle,
    /// including draws whose buffer id is still zero). Pure: shared by the
    /// immediate and the prepared stats paths.
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

    shape: TargetShape = .{},
    /// Main-target sample count the render pipelines were built for.
    sample_count: i32 = 1,
    /// Main-target color format the render pipelines were built for.
    color_format: sg.PixelFormat = .RGBA16F,
    quad_vb: sg.Buffer,
    quad_ib: sg.Buffer,
    sampler: sg.Sampler,
    default_texture: Texture,

    /// Builds one pipeline variant per blend mode. `stride`/`attrs` differ
    /// between the CPU path (integrated instance data) and the GPU path
    /// (spawn-slot data); quad geometry, depth and blend setup are shared.
    /// `target_shape` pins the exact main-target shape.
    fn makePipeline(shader: sg.Shader, blend: sg.BlendState, slot_stride: usize, gpu: bool, target_shape: TargetShape) sg.Pipeline {
        const resolved = target_shape.resolveEnvironment();
        var desc = sg.PipelineDesc{
            .shader = shader,
            .index_type = .UINT16,
            .depth = .{
                .pixel_format = resolved.depth_format,
                .compare = .LESS_EQUAL,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = resolved.sample_count,
        };
        desc.colors[0].pixel_format = resolved.color_format;
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

    pub fn init(sample_count: i32, color_format: sg.PixelFormat) ParticlePass {
        return initForShape(.{
            .sample_count = sample_count,
            .color_format = color_format,
            .depth_format = defaultDepthFormat(),
        });
    }

    pub fn initForShape(target_shape: TargetShape) ParticlePass {
        const resolved = target_shape.resolveEnvironment();
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
                resolved,
            ),
            .pipeline_alphablend = makePipeline(
                shader_cpu,
                blend_alpha,
                @sizeOf(particles.ParticleInstanceData),
                false,
                resolved,
            ),
            .pipeline_gpu_additive = makePipeline(
                shader_gpu,
                blend_additive,
                @sizeOf(particles.GpuParticleSlot),
                true,
                resolved,
            ),
            .pipeline_gpu_alphablend = makePipeline(
                shader_gpu,
                blend_alpha,
                @sizeOf(particles.GpuParticleSlot),
                true,
                resolved,
            ),
            .shader_cpu = shader_cpu,
            .shader_gpu = shader_gpu,
            .quad_vb = vb,
            .quad_ib = ib,
            .sampler = smp,
            .default_texture = Texture.createDefaultParticleDot32(),
            .shape = resolved,
            .sample_count = resolved.sample_count,
            .color_format = resolved.color_format,
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
