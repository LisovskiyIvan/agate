const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const part_shd = @import("particle_shader");
const part_compute_shd = @import("particle_compute_shader");
const compute = @import("../compute.zig");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const Camera = @import("../camera.zig").Camera;
const particles = @import("../particles.zig");
const ParticleSystem = particles.ParticleSystem;
const Texture = @import("../texture.zig").Texture;

pub const ParticlePass = struct {
    pipeline_additive: sg.Pipeline,
    pipeline_alphablend: sg.Pipeline,
    // GPU-simulation variants (shader program particle_gpu): same blend
    // states, but per-instance attributes carry spawn slots instead of
    // integrated state (see particles.GpuParticleSlot).
    pipeline_gpu_additive: sg.Pipeline,
    pipeline_gpu_alphablend: sg.Pipeline,
    // Compute-simulation variants (shader programs particle_compute_sim /
    // particle_compute in shaders/particle_compute.glsl): a compute pipeline
    // integrates the state storage buffer, the render pipelines below draw
    // billboards reading that buffer per instance. All four are .{} when the
    // backend lacks compute (see compute.zig); systems then fall back.
    compute_sim_pipeline: sg.Pipeline = .{},
    pipeline_compute_additive: sg.Pipeline = .{},
    pipeline_compute_alphablend: sg.Pipeline = .{},

    shader_cpu: sg.Shader = .{},
    shader_gpu: sg.Shader = .{},
    shader_sim: sg.Shader = .{},
    shader_render: sg.Shader = .{},

    /// Main-target sample count the render pipelines were built for
    /// (scene/msaa.zig). Compute-sim pipeline is sample-count independent.
    sample_count: i32 = 1,
    compute_supported: bool = false,
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

    /// Render pipeline for the compute path: the quad stays vertex buffer 0,
    /// per-instance state comes from the storage-buffer view (no instance
    /// vertex attributes). Blend/depth setup matches makePipeline verbatim.
    fn makeComputeRenderPipeline(shader: sg.Shader, blend: sg.BlendState, sample_count: i32) sg.Pipeline {
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
        desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        desc.layout.attrs[part_compute_shd.ATTR_particle_compute_position] = .{
            .buffer_index = 0,
            .format = .FLOAT2,
            .offset = 0,
        };
        desc.layout.attrs[part_compute_shd.ATTR_particle_compute_texcoord0] = .{
            .buffer_index = 0,
            .format = .FLOAT2,
            .offset = 2 * @sizeOf(f32),
        };
        desc.colors[0].blend = blend;
        return sg.makePipeline(desc);
    }

    pub fn init() ParticlePass {
        return initSampled(1);
    }

    /// Same pass at a different main-target sample count: sokol requires
    /// pipeline.sample_count to match the attachments of the pass it draws
    /// into. The compute-simulation pipeline is unaffected (compute passes
    /// have no attachments) and keeps the default sample count.
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

        // Compute support is a backend property (Metal / D3D11 / GL 4.3+ /
        // WebGPU; see compute.zig). When missing, the compute pipelines stay
        // invalid and `.compute` systems sticky-fall back on their next
        // update (ParticleSystem.update gates on compute.supported()).
        const compute_ok = compute.supported();
        var compute_sim_pipeline: sg.Pipeline = .{};
        var pipeline_compute_additive: sg.Pipeline = .{};
        var pipeline_compute_alphablend: sg.Pipeline = .{};
        var shader_sim: sg.Shader = .{};
        var shader_render: sg.Shader = .{};
        if (compute_ok) {
            shader_sim = sg.makeShader(part_compute_shd.particleComputeSimShaderDesc(sg.queryBackend()));
            compute_sim_pipeline = compute.makePipeline(shader_sim, "particle-compute-sim");
            shader_render = sg.makeShader(part_compute_shd.particleComputeShaderDesc(sg.queryBackend()));
            pipeline_compute_additive = makeComputeRenderPipeline(shader_render, blend_additive, sample_count);
            pipeline_compute_alphablend = makeComputeRenderPipeline(shader_render, blend_alpha, sample_count);
        }

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
            .compute_sim_pipeline = compute_sim_pipeline,
            .pipeline_compute_additive = pipeline_compute_additive,
            .pipeline_compute_alphablend = pipeline_compute_alphablend,
            .shader_cpu = shader_cpu,
            .shader_gpu = shader_gpu,
            .shader_sim = shader_sim,
            .shader_render = shader_render,
            .compute_supported = compute_ok,
            .quad_vb = vb,
            .quad_ib = ib,
            .sampler = smp,
            .default_texture = Texture.createDefaultParticleDot32(),
            .sample_count = sample_count,
        };
    }

    /// Runs the per-frame compute simulation for every `.compute` system in
    /// one shared compute pass. Called from ParticleLayer.update — i.e.
    /// outside any render pass and before the frame's render passes, so the
    /// render stage reads freshly integrated state.
    pub fn runComputeSimulations(self: *ParticlePass, systems: []const *ParticleSystem, dt: f32) void {
        if (!self.compute_supported or self.compute_sim_pipeline.id == 0) return;
        var any = false;
        for (systems) |ps| {
            if (ps.simulation_mode == .compute and ps.compute_state_buffer.id != 0) {
                any = true;
                break;
            }
        }
        if (!any) return;

        sg.beginPass(.{ .compute = true, .label = "particle-compute-sim" });
        sg.applyPipeline(self.compute_sim_pipeline);
        for (systems) |ps| {
            if (ps.simulation_mode != .compute) continue;
            const params = ps.computeFrameParams(dt) orelse continue;
            if (params.num_slots == 0) continue;

            var bind = sg.Bindings{};
            bind.views[part_compute_shd.VIEW_cs_slots] = ps.compute_slot_view;
            bind.views[part_compute_shd.VIEW_cs_state] = ps.compute_state_view;
            sg.applyBindings(bind);

            const cs_params = part_compute_shd.CsParams{
                .sim = .{
                    params.dt,
                    params.drag,
                    @floatFromInt(params.spawn_start),
                    @floatFromInt(params.spawn_count),
                },
                .misc = .{
                    @floatFromInt(params.num_slots),
                    if (params.init_all) 1.0 else 0.0,
                    0.0,
                    0.0,
                },
                .gravity = .{ params.gravity.x, params.gravity.y, params.gravity.z, 0.0 },
            };
            sg.applyUniforms(part_compute_shd.UB_cs_params, sg.asRange(&cs_params));

            sg.dispatch(
                @intCast(compute.groupCount(params.num_slots, compute.default_workgroup_size)),
                1,
                1,
            );

            // The init_all window is exactly one dispatch after creation.
            ps.compute_init_pending = false;
        }
        sg.endPass();
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
            // Compute-simulated systems draw from the state storage buffer.
            if (ps.simulation_mode == .compute) {
                self.renderComputeSystem(ps, view_proj, cam_right, cam_up, &current_pipeline);
                continue;
            }
            const gpu = ps.simulation_mode == .gpu;
            const instance_buf = if (gpu) ps.gpu_slot_buffer else ps.instance_buffer;
            if (ps.active_count == 0 or instance_buf.id == 0) continue;

            const pip_id = switch (ps.blend_mode) {
                .additive => if (gpu) self.pipeline_gpu_additive.id else self.pipeline_additive.id,
                .alpha_blend => if (gpu) self.pipeline_gpu_alphablend.id else self.pipeline_alphablend.id,
            };
            if (pip_id == 0) continue;
            if (current_pipeline.id != pip_id) {
                current_pipeline.id = pip_id;
                sg.applyPipeline(.{ .id = pip_id });
            }

            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = self.quad_vb;
            bind.vertex_buffers[1] = instance_buf;
            bind.index_buffer = self.quad_ib;

            const tex_view = if (ps.texture) |*t| t.view else self.default_texture.view;
            bind.views[part_shd.VIEW_particle_tex] = tex_view;
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
                    .time_drag = .{ ps.clock_seconds, ps.drag, 0.0, 0.0 },
                    .gravity = .{ ps.gravity.x, ps.gravity.y, ps.gravity.z, 0.0 },
                    .sprite = .{
                        @floatFromInt(if (ps.spritesheet_columns == 0) 1 else ps.spritesheet_columns),
                        @floatFromInt(if (ps.spritesheet_rows == 0) 1 else ps.spritesheet_rows),
                        ps.spritesheet_loops,
                        0.0,
                    },
                };
                sg.applyUniforms(part_shd.UB_gpu_params, sg.asRange(&gpu_params));
            }

            // active_count is the exact live count on CPU and the written-slot
            // high-water mark on GPU (dead slots cull in the vertex shader).
            sg.draw(0, 6, @intCast(ps.active_count));
        }
    }

    /// Draws one `.compute` system: billboards fed by the live-state storage
    /// buffer (no instance vertex attributes; storage view instead).
    fn renderComputeSystem(
        self: *ParticlePass,
        ps: *ParticleSystem,
        view_proj: Mat4,
        cam_right: Vec3,
        cam_up: Vec3,
        current_pipeline: *sg.Pipeline,
    ) void {
        if (ps.active_count == 0 or ps.compute_state_view.id == 0) return;
        const pip = switch (ps.blend_mode) {
            .additive => self.pipeline_compute_additive,
            .alpha_blend => self.pipeline_compute_alphablend,
        };
        if (pip.id == 0) return;
        if (current_pipeline.id != pip.id) {
            current_pipeline.* = pip;
            sg.applyPipeline(pip);
        }

        var bind = sg.Bindings{};
        bind.vertex_buffers[0] = self.quad_vb;
        bind.index_buffer = self.quad_ib;
        const tex_view = if (ps.texture) |*t| t.view else self.default_texture.view;
        bind.views[part_compute_shd.VIEW_particle_tex] = tex_view;
        bind.views[part_compute_shd.VIEW_vs_state] = ps.compute_state_view;
        bind.samplers[part_compute_shd.SMP_smp] = self.sampler;
        sg.applyBindings(bind);

        const vs_params = part_compute_shd.VsParams{
            .view_proj = view_proj,
            .camera_right = .{ cam_right.x, cam_right.y, cam_right.z, 0.0 },
            .camera_up = .{ cam_up.x, cam_up.y, cam_up.z, 0.0 },
            .sprite = .{
                @floatFromInt(if (ps.spritesheet_columns == 0) 1 else ps.spritesheet_columns),
                @floatFromInt(if (ps.spritesheet_rows == 0) 1 else ps.spritesheet_rows),
                ps.spritesheet_loops,
                0.0,
            },
        };
        sg.applyUniforms(part_compute_shd.UB_vs_params, sg.asRange(&vs_params));

        // active_count is the written-slot high-water mark (dead slots cull
        // in the vertex stage), same contract as the analytic path.
        sg.draw(0, 6, @intCast(ps.active_count));
    }

    pub fn deinit(self: *ParticlePass) void {
        sg.destroyPipeline(self.pipeline_additive);
        sg.destroyPipeline(self.pipeline_alphablend);
        sg.destroyPipeline(self.pipeline_gpu_additive);
        sg.destroyPipeline(self.pipeline_gpu_alphablend);
        if (self.compute_sim_pipeline.id != 0) sg.destroyPipeline(self.compute_sim_pipeline);
        if (self.pipeline_compute_additive.id != 0) sg.destroyPipeline(self.pipeline_compute_additive);
        if (self.pipeline_compute_alphablend.id != 0) sg.destroyPipeline(self.pipeline_compute_alphablend);
        if (self.shader_cpu.id != 0) sg.destroyShader(self.shader_cpu);
        if (self.shader_gpu.id != 0) sg.destroyShader(self.shader_gpu);
        if (self.shader_sim.id != 0) sg.destroyShader(self.shader_sim);
        if (self.shader_render.id != 0) sg.destroyShader(self.shader_render);
        self.shader_cpu = .{};
        self.shader_gpu = .{};
        self.shader_sim = .{};
        self.shader_render = .{};
        sg.destroyBuffer(self.quad_vb);
        sg.destroyBuffer(self.quad_ib);
        sg.destroySampler(self.sampler);
        self.default_texture.deinit();
    }
};
