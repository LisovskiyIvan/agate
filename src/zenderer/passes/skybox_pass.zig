const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const skybox_shd = @import("skybox_shader");
const math = @import("math");
const Mat4 = math.Mat4;
const ArcRotateCamera = @import("../camera.zig").ArcRotateCamera;
const CubeTexture = @import("../texture.zig").CubeTexture;

pub const SkyboxPass = struct {
    pipeline: sg.Pipeline,
    mesh_vb: sg.Buffer,
    mesh_ib: sg.Buffer,
    sampler: sg.Sampler,

    pub fn init() SkyboxPass {
        const skybox_positions = [_][3]f32{
            .{ -1.0, -1.0, -1.0 }, // 0
            .{  1.0, -1.0, -1.0 }, // 1
            .{  1.0,  1.0, -1.0 }, // 2
            .{ -1.0,  1.0, -1.0 }, // 3
            .{ -1.0, -1.0,  1.0 }, // 4
            .{  1.0, -1.0,  1.0 }, // 5
            .{  1.0,  1.0,  1.0 }, // 6
            .{ -1.0,  1.0,  1.0 }, // 7
        };

        const skybox_indices = [_]u16{
            // Front (-Z)
            0, 2, 1,  0, 3, 2,
            // Back (+Z)
            4, 5, 6,  4, 6, 7,
            // Left (-X)
            0, 4, 7,  0, 7, 3,
            // Right (+X)
            1, 2, 6,  1, 6, 5,
            // Top (+Y)
            3, 7, 6,  3, 6, 2,
            // Bottom (-Y)
            0, 1, 5,  0, 5, 4,
        };

        const vb = sg.makeBuffer(.{
            .data = sg.asRange(&skybox_positions),
        });
        const ib = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(&skybox_indices),
        });

        const smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
            .wrap_w = .CLAMP_TO_EDGE,
        });

        var pip_desc = sg.PipelineDesc{
            .shader = sg.makeShader(skybox_shd.skyboxShaderDesc(sg.queryBackend())),
            .index_type = .UINT16,
            .depth = .{
                .compare = .EQUAL, // Skybox rendered only at maximum depth (depth = 1.0)
                .write_enabled = false,
            },
            .cull_mode = .NONE,
        };
        pip_desc.layout.buffers[0] = .{ .stride = 3 * @sizeOf(f32) };
        pip_desc.layout.attrs[skybox_shd.ATTR_skybox_position] = .{
            .format = .FLOAT3,
            .offset = 0,
        };

        const pip = sg.makePipeline(pip_desc);

        return .{
            .pipeline = pip,
            .mesh_vb = vb,
            .mesh_ib = ib,
            .sampler = smp,
        };
    }

    pub fn render(self: *SkyboxPass, camera: ArcRotateCamera, aspect: f32, cube_tex: CubeTexture, exposure: f32) void {
        const view = camera.getViewMatrix();
        var rot_view = view;
        rot_view.m[12] = 0.0;
        rot_view.m[13] = 0.0;
        rot_view.m[14] = 0.0;

        const proj = camera.getProjectionMatrix(aspect);
        const view_proj = Mat4.mul(proj, rot_view);

        sg.applyPipeline(self.pipeline);

        var bind = sg.Bindings{};
        bind.vertex_buffers[0] = self.mesh_vb;
        bind.index_buffer = self.mesh_ib;
        bind.views[skybox_shd.VIEW_sky_tex] = cube_tex.view;
        bind.samplers[skybox_shd.SMP_smp] = self.sampler;
        sg.applyBindings(bind);

        const vs_params = skybox_shd.VsParams{
            .view_proj = view_proj,
        };
        sg.applyUniforms(skybox_shd.UB_vs_params, sg.asRange(&vs_params));

        const fs_params = skybox_shd.FsParams{
            .params = .{ exposure, 0.0, 0.0, 0.0 },
        };
        sg.applyUniforms(skybox_shd.UB_fs_params, sg.asRange(&fs_params));

        sg.draw(0, 36, 1);
    }

    pub fn deinit(self: *SkyboxPass) void {
        sg.destroyPipeline(self.pipeline);
        sg.destroyBuffer(self.mesh_vb);
        sg.destroyBuffer(self.mesh_ib);
        sg.destroySampler(self.sampler);
    }
};
