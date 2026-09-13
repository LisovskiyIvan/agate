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
    pipeline_additive: sg.Pipeline,
    pipeline_alphablend: sg.Pipeline,
    // GPU-simulation variants (shader program particle_gpu): same blend
    // states, but per-instance attributes carry spawn slots instead of
    // integrated state (see particles.GpuParticleSlot).
    pipeline_gpu_additive: sg.Pipeline,
    pipeline_gpu_alphablend: sg.Pipeline,
    quad_vb: sg.Buffer,
    quad_ib: sg.Buffer,
    sampler: sg.Sampler,
    default_texture: Texture,

    /// Builds one pipeline variant per blend mode. `stride`/`attrs` differ
    /// between the CPU path (integrated instance data) and the GPU path
    /// (spawn-slot data); quad geometry, depth and blend setup are shared,
    /// matching the historical pipeline configs bit-for-bit.
    fn makePipeline(shader: sg.Shader, blend: sg.BlendState, slot_stride: usize, gpu: bool) sg.Pipeline {
        var desc = sg.PipelineDesc{
            .shader = shader,
            .index_type = .UINT16,
            .depth = .{
                .compare = .LESS_EQUAL,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
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
            ),
            .pipeline_alphablend = makePipeline(
                shader_cpu,
                blend_alpha,
                @sizeOf(particles.ParticleInstanceData),
                false,
            ),
            .pipeline_gpu_additive = makePipeline(
                shader_gpu,
                blend_additive,
                @sizeOf(particles.GpuParticleSlot),
                true,
            ),
            .pipeline_gpu_alphablend = makePipeline(
                shader_gpu,
                blend_alpha,
                @sizeOf(particles.GpuParticleSlot),
                true,
            ),
            .quad_vb = vb,
            .quad_ib = ib,
            .sampler = smp,
            .default_texture = Texture.createDefaultParticleDot32(),
        };
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

    pub fn deinit(self: *ParticlePass) void {
        sg.destroyPipeline(self.pipeline_additive);
        sg.destroyPipeline(self.pipeline_alphablend);
        sg.destroyPipeline(self.pipeline_gpu_additive);
        sg.destroyPipeline(self.pipeline_gpu_alphablend);
        sg.destroyBuffer(self.quad_vb);
        sg.destroyBuffer(self.quad_ib);
        sg.destroySampler(self.sampler);
        self.default_texture.deinit();
    }
};
