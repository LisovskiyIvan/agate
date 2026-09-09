const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const shadow_shd = @import("shadow_shader");
const math = @import("math");
const Mat4 = math.Mat4;
const mesh_mod = @import("../mesh.zig");
const Mesh = mesh_mod.Mesh;
const Vertex = mesh_mod.Vertex;

pub const ShadowPass = struct {
    image: sg.Image,
    attachment_view: sg.View,
    texture_view: sg.View,
    sampler: sg.Sampler,
    pipeline_u16: sg.Pipeline,
    pipeline_u32: sg.Pipeline,
    inst_pipeline_u16: sg.Pipeline,
    inst_pipeline_u32: sg.Pipeline,
    skinned_pipeline_u16: sg.Pipeline,
    skinned_pipeline_u32: sg.Pipeline,

    pub fn init() ShadowPass {
        // 2048 atlas holding 4x 1024 cascades (2x2). Was 4096/2048: same look
        // for near geometry, 4x fewer depth texels rasterized per frame.
        const depth_img = sg.makeImage(.{
            .usage = .{ .depth_stencil_attachment = true },
            .pixel_format = .DEPTH,
            .width = 2048,
            .height = 2048,
            .sample_count = 1,
        });
        const att_view = sg.makeView(.{
            .depth_stencil_attachment = .{ .image = depth_img },
        });

        const tex_view = sg.makeView(.{
            .texture = .{ .image = depth_img },
        });

        const smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
            .compare = .LESS_EQUAL,
        });

        // 1. Shadow Depth pipelines (regular meshes)
        var shadow_pip_desc = sg.PipelineDesc{
            .shader = sg.makeShader(shadow_shd.shadowShaderDesc(sg.queryBackend())),
            .index_type = .UINT16,
            .sample_count = 1,
            .depth = .{
                .pixel_format = .DEPTH,
                .compare = .LESS_EQUAL,
                .write_enabled = true,
                .bias = 1.0,
                .bias_slope_scale = 1.0,
            },
            .cull_mode = .BACK,
            .face_winding = .CCW,
        };
        shadow_pip_desc.colors[0].pixel_format = .NONE;
        shadow_pip_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
        shadow_pip_desc.layout.attrs[shadow_shd.ATTR_shadow_position] = .{
            .format = .FLOAT3,
            .offset = @offsetOf(Vertex, "position"),
        };

        const pip_u16 = sg.makePipeline(shadow_pip_desc);
        shadow_pip_desc.index_type = .UINT32;
        const pip_u32 = sg.makePipeline(shadow_pip_desc);

        // 2. Shadow Depth pipelines (instanced meshes)
        var shadow_inst_desc = sg.PipelineDesc{
            .shader = sg.makeShader(shadow_shd.shadowInstancedShaderDesc(sg.queryBackend())),
            .index_type = .UINT16,
            .sample_count = 1,
            .depth = .{
                .pixel_format = .DEPTH,
                .compare = .LESS_EQUAL,
                .write_enabled = true,
                .bias = 1.0,
                .bias_slope_scale = 1.0,
            },
            .cull_mode = .BACK,
            .face_winding = .CCW,
        };
        shadow_inst_desc.colors[0].pixel_format = .NONE;
        shadow_inst_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
        shadow_inst_desc.layout.attrs[shadow_shd.ATTR_shadow_instanced_position] = .{
            .buffer_index = 0,
            .format = .FLOAT3,
            .offset = @offsetOf(Vertex, "position"),
        };
        shadow_inst_desc.layout.buffers[1] = .{
            .step_func = .PER_INSTANCE,
            .step_rate = 1,
            .stride = @sizeOf(Mat4),
        };
        shadow_inst_desc.layout.attrs[shadow_shd.ATTR_shadow_instanced_inst_mat0] = .{ .buffer_index = 1, .offset = 0, .format = .FLOAT4 };
        shadow_inst_desc.layout.attrs[shadow_shd.ATTR_shadow_instanced_inst_mat1] = .{ .buffer_index = 1, .offset = 16, .format = .FLOAT4 };
        shadow_inst_desc.layout.attrs[shadow_shd.ATTR_shadow_instanced_inst_mat2] = .{ .buffer_index = 1, .offset = 32, .format = .FLOAT4 };
        shadow_inst_desc.layout.attrs[shadow_shd.ATTR_shadow_instanced_inst_mat3] = .{ .buffer_index = 1, .offset = 48, .format = .FLOAT4 };

        const inst_pip_u16 = sg.makePipeline(shadow_inst_desc);
        shadow_inst_desc.index_type = .UINT32;
        const inst_pip_u32 = sg.makePipeline(shadow_inst_desc);

        // 3. Skinned Shadow pipelines
        var shadow_skinned_desc = sg.PipelineDesc{
            .shader = sg.makeShader(shadow_shd.shadowSkinnedShaderDesc(sg.queryBackend())),
            .index_type = .UINT16,
            .sample_count = 1,
            .depth = .{
                .pixel_format = .DEPTH,
                .compare = .LESS_EQUAL,
                .write_enabled = true,
                .bias = 1.0,
                .bias_slope_scale = 1.0,
            },
            .cull_mode = .BACK,
            .face_winding = .CCW,
        };
        shadow_skinned_desc.colors[0].pixel_format = .NONE;
        shadow_skinned_desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
        shadow_skinned_desc.layout.attrs[shadow_shd.ATTR_shadow_skinned_position] = .{
            .format = .FLOAT3,
            .offset = @offsetOf(Vertex, "position"),
        };
        shadow_skinned_desc.layout.attrs[shadow_shd.ATTR_shadow_skinned_joints] = .{
            .format = .FLOAT4,
            .offset = @offsetOf(Vertex, "joints"),
        };
        shadow_skinned_desc.layout.attrs[shadow_shd.ATTR_shadow_skinned_weights] = .{
            .format = .FLOAT4,
            .offset = @offsetOf(Vertex, "weights"),
        };
        const skinned_pip_u16 = sg.makePipeline(shadow_skinned_desc);
        shadow_skinned_desc.index_type = .UINT32;
        const skinned_pip_u32 = sg.makePipeline(shadow_skinned_desc);

        return .{
            .image = depth_img,
            .attachment_view = att_view,
            .texture_view = tex_view,
            .sampler = smp,
            .pipeline_u16 = pip_u16,
            .pipeline_u32 = pip_u32,
            .inst_pipeline_u16 = inst_pip_u16,
            .inst_pipeline_u32 = inst_pip_u32,
            .skinned_pipeline_u16 = skinned_pip_u16,
            .skinned_pipeline_u32 = skinned_pip_u32,
        };
    }

    pub fn render(
        self: *ShadowPass,
        meshes: []const *Mesh,
        frame_id: u64,
        cascades: [4]Mat4,
    ) void {
        var shadow_action = sg.PassAction{};
        shadow_action.depth = .{
            .load_action = .CLEAR,
            .clear_value = 1.0,
        };
        var shadow_pass = sg.Pass{
            .action = shadow_action,
        };
        shadow_pass.attachments.depth_stencil = self.attachment_view;
        sg.beginPass(shadow_pass);

        const CASCADE_RES: i32 = 1024;

        for (0..4) |c_idx| {
            const light_view_proj = cascades[c_idx];
            const vx: i32 = if (c_idx % 2 == 1) CASCADE_RES else 0;
            const vy: i32 = if (c_idx >= 2) CASCADE_RES else 0;

            sg.applyViewport(vx, vy, CASCADE_RES, CASCADE_RES, false);
            sg.applyScissorRect(vx, vy, CASCADE_RES, CASCADE_RES, false);

            const c_frustum = math.Frustum.fromViewProjection(light_view_proj);

            for (meshes) |mesh| {
                if (!mesh.cast_shadows) continue;

                if (mesh.instances.items.len > 0) {
                    if (mesh.visible_instance_count == 0 or mesh.instance_buffer.id == 0) continue;

                    const pip_id = if (mesh.index_type == .UINT32) self.inst_pipeline_u32.id else self.inst_pipeline_u16.id;
                    if (pip_id == 0) continue;
                    sg.applyPipeline(.{ .id = pip_id });

                    var bind = sg.Bindings{};
                    bind.vertex_buffers[0] = mesh.vertex_buffer;
                    bind.vertex_buffers[1] = mesh.instance_buffer;
                    bind.index_buffer = mesh.index_buffer;
                    sg.applyBindings(bind);

                    const inst_vs = shadow_shd.VsInstParams{
                        .light_view_proj = light_view_proj,
                    };
                    sg.applyUniforms(shadow_shd.UB_vs_inst_params, sg.asRange(&inst_vs));
                    sg.draw(0, mesh.index_count, mesh.visible_instance_count);
                } else {
                    if (!mesh.is_visible) continue;
                    // Scene.render() already cached world matrix + AABB this frame;
                    // fall back to direct computation if it didn't (stale or external call).
                    const aabb_w = if (mesh.cached_frame == frame_id) mesh.cached_aabb else mesh.getWorldBoundingBox();
                    if (!c_frustum.intersectsAABB(aabb_w)) continue;

                    const pip_id = if (mesh.skeleton != null)
                        (if (mesh.index_type == .UINT32) self.skinned_pipeline_u32.id else self.skinned_pipeline_u16.id)
                    else
                        (if (mesh.index_type == .UINT32) self.pipeline_u32.id else self.pipeline_u16.id);
                    if (pip_id == 0) continue;
                    sg.applyPipeline(.{ .id = pip_id });

                    var bind = sg.Bindings{};
                    bind.vertex_buffers[0] = mesh.vertex_buffer;
                    bind.index_buffer = mesh.index_buffer;
                    sg.applyBindings(bind);

                    const model = if (mesh.cached_frame == frame_id) mesh.cached_matrix else mesh.getWorldMatrix();
                    const shadow_vs = shadow_shd.VsParams{
                        .mvp = Mat4.mul(light_view_proj, model),
                    };
                    sg.applyUniforms(shadow_shd.UB_vs_params, sg.asRange(&shadow_vs));

                    if (mesh.skeleton) |skel| {
                        const vs_skin = shadow_shd.VsSkin{
                            .bones = skel.skin_matrices,
                        };
                        sg.applyUniforms(shadow_shd.UB_vs_skin, sg.asRange(&vs_skin));
                    }

                    sg.draw(0, mesh.index_count, 1);
                }
            }
        }

        sg.endPass();
    }

    pub fn deinit(self: *ShadowPass) void {
        sg.destroyPipeline(self.pipeline_u16);
        sg.destroyPipeline(self.pipeline_u32);
        sg.destroyPipeline(self.inst_pipeline_u16);
        sg.destroyPipeline(self.inst_pipeline_u32);
        sg.destroyPipeline(self.skinned_pipeline_u16);
        sg.destroyPipeline(self.skinned_pipeline_u32);
        sg.destroyView(self.attachment_view);
        sg.destroyView(self.texture_view);
        sg.destroySampler(self.sampler);
        sg.destroyImage(self.image);
    }
};
