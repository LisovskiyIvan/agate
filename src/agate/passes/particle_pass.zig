const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const part_shd = @import("particle_shader");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const ArcRotateCamera = @import("../camera.zig").ArcRotateCamera;
const particles = @import("../particles.zig");
const ParticleSystem = particles.ParticleSystem;
const Texture = @import("../texture.zig").Texture;

pub const ParticlePass = struct {
    pipeline_additive: sg.Pipeline,
    pipeline_alphablend: sg.Pipeline,
    quad_vb: sg.Buffer,
    quad_ib: sg.Buffer,
    sampler: sg.Sampler,
    default_texture: Texture,

    pub fn init() ParticlePass {
        const particle_quad_vertices = [_]f32{
            // x,     y,     u,   v
            -0.5, -0.5,  0.0, 0.0,
             0.5, -0.5,  1.0, 0.0,
             0.5,  0.5,  1.0, 1.0,
            -0.5,  0.5,  0.0, 1.0,
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

        var part_desc = sg.PipelineDesc{
            .shader = sg.makeShader(part_shd.particleShaderDesc(sg.queryBackend())),
            .index_type = .UINT16,
            .depth = .{
                .compare = .LESS_EQUAL,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
        };
        // Buffer 0: Unit quad
        part_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        part_desc.layout.attrs[part_shd.ATTR_particle_position] = .{
            .buffer_index = 0,
            .format = .FLOAT2,
            .offset = 0,
        };
        part_desc.layout.attrs[part_shd.ATTR_particle_texcoord0] = .{
            .buffer_index = 0,
            .format = .FLOAT2,
            .offset = 2 * @sizeOf(f32),
        };

        // Buffer 1: Dynamic per-instance data
        part_desc.layout.buffers[1] = .{
            .step_func = .PER_INSTANCE,
            .step_rate = 1,
            .stride = @sizeOf(particles.ParticleInstanceData),
        };
        part_desc.layout.attrs[part_shd.ATTR_particle_inst_pos_size] = .{
            .buffer_index = 1,
            .format = .FLOAT4,
            .offset = 0,
        };
        part_desc.layout.attrs[part_shd.ATTR_particle_inst_color] = .{
            .buffer_index = 1,
            .format = .FLOAT4,
            .offset = 4 * @sizeOf(f32),
        };

        // Additive pipeline
        part_desc.colors[0].blend = .{
            .enabled = true,
            .src_factor_rgb = .SRC_ALPHA,
            .dst_factor_rgb = .ONE,
            .src_factor_alpha = .ONE,
            .dst_factor_alpha = .ONE,
        };
        const pip_add = sg.makePipeline(part_desc);

        // AlphaBlend pipeline
        part_desc.colors[0].blend = .{
            .enabled = true,
            .src_factor_rgb = .SRC_ALPHA,
            .dst_factor_rgb = .ONE_MINUS_SRC_ALPHA,
            .src_factor_alpha = .ONE,
            .dst_factor_alpha = .ONE_MINUS_SRC_ALPHA,
        };
        const pip_alpha = sg.makePipeline(part_desc);

        return .{
            .pipeline_additive = pip_add,
            .pipeline_alphablend = pip_alpha,
            .quad_vb = vb,
            .quad_ib = ib,
            .sampler = smp,
            .default_texture = Texture.createDefaultParticleDot32(),
        };
    }

    pub fn render(
        self: *ParticlePass,
        systems: []const *ParticleSystem,
        camera: ArcRotateCamera,
        aspect: f32,
    ) void {
        const view_proj = camera.getViewProjection(aspect);
        const view_mat = camera.getViewMatrix();
        const cam_right = Vec3.new(view_mat.m[0], view_mat.m[4], view_mat.m[8]);
        const cam_up = Vec3.new(view_mat.m[1], view_mat.m[5], view_mat.m[9]);

        var current_pipeline: sg.Pipeline = .{};

        for (systems) |ps| {
            if (ps.active_count == 0 or ps.instance_buffer.id == 0) continue;

            const pip_id = if (ps.blend_mode == .additive) self.pipeline_additive.id else self.pipeline_alphablend.id;
            if (pip_id == 0) continue;
            if (current_pipeline.id != pip_id) {
                current_pipeline.id = pip_id;
                sg.applyPipeline(.{ .id = pip_id });
            }

            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = self.quad_vb;
            bind.vertex_buffers[1] = ps.instance_buffer;
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

            sg.draw(0, 6, @intCast(ps.active_count));
        }
    }

    pub fn deinit(self: *ParticlePass) void {
        sg.destroyPipeline(self.pipeline_additive);
        sg.destroyPipeline(self.pipeline_alphablend);
        sg.destroyBuffer(self.quad_vb);
        sg.destroyBuffer(self.quad_ib);
        sg.destroySampler(self.sampler);
        self.default_texture.deinit();
    }
};
