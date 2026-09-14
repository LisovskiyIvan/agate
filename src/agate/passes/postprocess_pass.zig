const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const sglue = sokol.glue;
const post_shd = @import("postprocess_shader");
const postprocess = @import("../postprocess.zig");
const PostProcessOptions = postprocess.PostProcessOptions;
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const Color3 = math.Color3;

pub const PostProcessPass = struct {
    offscreen_color_image: sg.Image = .{},
    offscreen_color_att_view: sg.View = .{},
    offscreen_depth_image: sg.Image = .{},
    offscreen_depth_att_view: sg.View = .{},
    offscreen_depth_tex_view: sg.View = .{},
    offscreen_resolve_image: sg.Image = .{},
    offscreen_resolve_att_view: sg.View = .{},
    offscreen_resolve_tex_view: sg.View = .{},
    postprocess_sampler: sg.Sampler = .{},
    depth_sampler: sg.Sampler = .{},
    // Dedicated LUT sampler: bilinear inside the strip with LOD pinned to
    // the base level, so a mipped LUT upload can never smear cube slices
    // through box-filtered mips.
    lut_sampler: sg.Sampler = .{},
    postprocess_pipeline: sg.Pipeline = .{},
    postprocess_quad_vb: sg.Buffer = .{},
    postprocess_quad_ib: sg.Buffer = .{},
    // Optional BloomPass result (pyramid glow, half resolution). Set via
    // setBloomTexture(); empty by default, in which case the shader falls
    // back to the legacy single-shader bloom and this binds the scene view
    // as a harmless placeholder.
    bloom_tex_view: sg.View = .{},
    width: i32 = 0,
    height: i32 = 0,
    sample_count: i32 = 1,
    pub fn init() PostProcessPass {
        // Fullscreen Quad (XY, UV)
        const quad_vertices = [_]f32{
            // x,     y,    u,   v
            -1.0, -1.0, 0.0, 0.0,
            1.0,  -1.0, 1.0, 0.0,
            1.0,  1.0,  1.0, 1.0,
            -1.0, 1.0,  0.0, 1.0,
        };
        const quad_indices = [_]u16{
            0, 1, 2,
            0, 2, 3,
        };

        const vb = sg.makeBuffer(.{
            .data = sg.asRange(&quad_vertices),
        });
        const ib = sg.makeBuffer(.{
            .usage = .{ .index_buffer = true },
            .data = sg.asRange(&quad_indices),
        });

        const smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });

        const depth_smp = sg.makeSampler(.{
            .min_filter = .NEAREST,
            .mag_filter = .NEAREST,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });

        const lut_smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
            // Sample only the base level: LUT strips must not pick up
            // (corrupting) mips even when uploaded with a mip chain.
            .min_lod = 0.0,
            .max_lod = 0.0,
        });

        var pp_desc = sg.PipelineDesc{
            .shader = sg.makeShader(post_shd.postprocessShaderDesc(sg.queryBackend())),
            .index_type = .UINT16,
            .depth = .{
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
        };
        pp_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        pp_desc.layout.attrs[post_shd.ATTR_postprocess_position] = .{
            .format = .FLOAT2,
            .offset = 0,
        };
        pp_desc.layout.attrs[post_shd.ATTR_postprocess_texcoord0] = .{
            .format = .FLOAT2,
            .offset = 2 * @sizeOf(f32),
        };
        const pip = sg.makePipeline(pp_desc);

        return .{
            .postprocess_sampler = smp,
            .depth_sampler = depth_smp,
            .lut_sampler = lut_smp,
            .postprocess_pipeline = pip,
            .postprocess_quad_vb = vb,
            .postprocess_quad_ib = ib,
        };
    }

    /// Destroys the render-target images/views (not samplers/pipelines).
    /// MSAA shape (sample_count > 1): MSAA color + MSAA depth attachments,
    /// a 1x resolve image the color resolves into at end of pass, and NO
    /// depth texture view (sokol has no depth resolve; an MSAA depth image
    /// is not samplable as a plain texture). 1x shape: plain color+depth,
    /// "resolve" views alias the color image.
    fn destroyTargets(self: *PostProcessPass) void {
        if (self.offscreen_color_image.id != 0) sg.destroyImage(self.offscreen_color_image);
        if (self.offscreen_color_att_view.id != 0) sg.destroyView(self.offscreen_color_att_view);
        if (self.offscreen_resolve_image.id != 0) sg.destroyImage(self.offscreen_resolve_image);
        if (self.offscreen_resolve_att_view.id != 0) sg.destroyView(self.offscreen_resolve_att_view);
        if (self.offscreen_resolve_tex_view.id != 0) sg.destroyView(self.offscreen_resolve_tex_view);
        if (self.offscreen_depth_image.id != 0) sg.destroyImage(self.offscreen_depth_image);
        if (self.offscreen_depth_att_view.id != 0) sg.destroyView(self.offscreen_depth_att_view);
        if (self.offscreen_depth_tex_view.id != 0) sg.destroyView(self.offscreen_depth_tex_view);
        self.offscreen_color_image = .{};
        self.offscreen_color_att_view = .{};
        self.offscreen_resolve_image = .{};
        self.offscreen_resolve_att_view = .{};
        self.offscreen_resolve_tex_view = .{};
        self.offscreen_depth_image = .{};
        self.offscreen_depth_att_view = .{};
        self.offscreen_depth_tex_view = .{};
    }

    pub fn resize(self: *PostProcessPass, width: i32, height: i32, sample_count: i32) void {
        if (width <= 0 or height <= 0) return;
        const samples: i32 = if (sample_count < 1) 1 else sample_count;
        if (self.width == width and self.height == height and self.sample_count == samples) return;

        self.destroyTargets();

        const sw = sglue.swapchain();
        const color_fmt: sg.PixelFormat = if (sw.color_format != .DEFAULT and sw.color_format != .NONE) sw.color_format else .BGRA8;
        const depth_fmt: sg.PixelFormat = if (sw.depth_format != .DEFAULT and sw.depth_format != .NONE) sw.depth_format else .DEPTH_STENCIL;

        // Color: attachment at the full sample count; when resolving, a
        // separate 1x resolve image (usage.resolve_attachment) receives the
        // MSAA resolve at end of pass and carries the texture view.
        const col_img = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = width,
            .height = height,
            .pixel_format = color_fmt,
            .sample_count = samples,
        });
        const col_att = sg.makeView(.{
            .color_attachment = .{ .image = col_img },
        });
        self.offscreen_color_image = col_img;
        self.offscreen_color_att_view = col_att;

        if (samples > 1) {
            const res_img = sg.makeImage(.{
                .usage = .{ .resolve_attachment = true },
                .width = width,
                .height = height,
                .pixel_format = color_fmt,
                .sample_count = 1,
            });
            self.offscreen_resolve_image = res_img;
            self.offscreen_resolve_att_view = sg.makeView(.{
                .resolve_attachment = .{ .image = res_img },
            });
            self.offscreen_resolve_tex_view = sg.makeView(.{
                .texture = .{ .image = res_img },
            });
        } else {
            // Legacy 1x shape: postfx samples the color image directly.
            self.offscreen_resolve_tex_view = sg.makeView(.{
                .texture = .{ .image = col_img },
            });
        }

        // Depth: same sample count as color (sokol validation requires the
        // match). Only the 1x depth gets a texture view; MSAA depth is
        // write-only for the post chain (scene/msaa.zig suppresses the
        // depth-consuming effects instead).
        const depth_img = sg.makeImage(.{
            .usage = .{ .depth_stencil_attachment = true },
            .width = width,
            .height = height,
            .pixel_format = depth_fmt,
            .sample_count = samples,
        });
        self.offscreen_depth_image = depth_img;
        self.offscreen_depth_att_view = sg.makeView(.{
            .depth_stencil_attachment = .{ .image = depth_img },
        });
        if (samples == 1) {
            self.offscreen_depth_tex_view = sg.makeView(.{
                .texture = .{ .image = depth_img },
            });
        }

        self.width = width;
        self.height = height;
        self.sample_count = samples;
    }

    /// Valid texture view for slots that semantically want scene depth.
    /// 1x target: the depth texture itself. MSAA target: no depth texture
    /// exists, so the resolved color view serves as a valid placeholder —
    /// the depth-consuming shader branches (SSR/DoF) are flag-gated off
    /// while MSAA is active (scene/postfx_stack.zig), the binding only has
    /// to exist for sokol's apply-bindings validation.
    pub fn depthSampleView(self: *const PostProcessPass) sg.View {
        if (self.offscreen_depth_tex_view.id != 0) return self.offscreen_depth_tex_view;
        return self.offscreen_resolve_tex_view;
    }

    pub fn render(
        self: *PostProcessPass,
        config: PostProcessOptions,
        ssao_tex: sg.View,
        ssao_enabled: bool,
        ssao_debug: bool,
        ssao_intensity: f32,
        cur_w: i32,
        cur_h: i32,
        view_proj: Mat4,
        inv_view_proj: Mat4,
        camera_pos: Vec3,
        sun_dir: Vec3,
        sun_color: Color3,
        near_z: f32,
        far_z: f32,
    ) void {
        if (self.postprocess_pipeline.id == 0) return;
        sg.applyPipeline(self.postprocess_pipeline);
        var post_bind = sg.Bindings{};
        post_bind.vertex_buffers[0] = self.postprocess_quad_vb;
        post_bind.index_buffer = self.postprocess_quad_ib;
        post_bind.views[post_shd.VIEW_scene_tex] = self.offscreen_resolve_tex_view;
        post_bind.views[post_shd.VIEW_ssao_tex] = ssao_tex;
        post_bind.views[post_shd.VIEW_depth_tex] = self.depthSampleView();
        // Pyramid glow when the parent fed a BloomPass result; otherwise a
        // valid placeholder the shader never samples (pyramid flag off).
        post_bind.views[post_shd.VIEW_bloom_tex] = if (self.bloom_tex_view.id != 0)
            self.bloom_tex_view
        else
            self.offscreen_resolve_tex_view;
        // LUT when the config carries a live binding; otherwise the resolved
        // scene view as a valid placeholder the shader never samples
        // (lut_params.x = 0 gates the LUT branch off).
        const lut_view: sg.View = if (config.lut) |l| l.view else .{};
        post_bind.views[post_shd.VIEW_lut_tex] = if (lut_view.id != 0)
            lut_view
        else
            self.offscreen_resolve_tex_view;
        post_bind.samplers[post_shd.SMP_smp] = self.postprocess_sampler;
        post_bind.samplers[post_shd.SMP_depth_smp] = self.depth_sampler;
        post_bind.samplers[post_shd.SMP_lut_smp] = self.lut_sampler;
        sg.applyBindings(post_bind);

        const pp_params = post_shd.FsParams{
            .params1 = .{
                config.exposure,
                config.bloom_threshold,
                config.bloom_intensity,
                config.bloom_radius,
            },
            .params2 = .{
                config.vignette_intensity,
                config.vignette_radius,
                config.saturation,
                config.contrast,
            },
            .params3 = .{
                @floatFromInt(@intFromEnum(config.tonemapping)),
                config.chromatic_aberration,
                if (config.bloom_enabled) 1.0 else 0.0,
                if (config.vignette_enabled) 1.0 else 0.0,
            },
            .params4 = .{
                if (ssao_enabled) 1.0 else 0.0,
                if (ssao_debug) 1.0 else 0.0,
                ssao_intensity,
                if (config.fxaa_enabled) 1.0 else 0.0,
            },
            .resolution = .{
                @floatFromInt(cur_w),
                @floatFromInt(cur_h),
                1.0 / @as(f32, @floatFromInt(cur_w)),
                1.0 / @as(f32, @floatFromInt(cur_h)),
            },
            .camera_params = .{
                near_z,
                far_z,
                0.0,
                0.0,
            },
            .camera_pos = .{
                camera_pos.x,
                camera_pos.y,
                camera_pos.z,
                0.0,
            },
            .sun_dir = .{
                sun_dir.x,
                sun_dir.y,
                sun_dir.z,
                0.0,
            },
            .sun_color = .{
                sun_color.r,
                sun_color.g,
                sun_color.b,
                0.0,
            },
            .fog_params = .{
                if (config.fog_enabled) 1.0 else 0.0,
                config.fog_density,
                config.fog_height_falloff,
                config.fog_start_distance,
            },
            .fog_color = .{
                config.fog_color[0],
                config.fog_color[1],
                config.fog_color[2],
                config.fog_sun_scattering,
            },
            .ssr_params = .{
                if (config.ssr_enabled) 1.0 else 0.0,
                config.ssr_intensity,
                config.ssr_thickness,
                config.ssr_max_distance,
            },
            .params5 = .{
                if (config.sharpen_enabled) config.sharpen_amount else 0.0,
                if (config.grain_enabled) config.grain_intensity else 0.0,
                config.temperature,
                config.tint,
            },
            .dof_params = .{
                if (config.dof_enabled) 1.0 else 0.0,
                config.dof_focus_distance,
                config.dof_focus_range,
                config.dof_max_blur,
            },
            .bloom_pyramid = .{
                if (config.bloom_pyramid and self.bloom_tex_view.id != 0) 1.0 else 0.0,
                @floatFromInt(config.bloom_pyramid_mips),
                0.0,
                0.0,
            },
            .grade_shadows = .{
                config.grade_shadows[0],
                config.grade_shadows[1],
                config.grade_shadows[2],
                0.0,
            },
            .grade_midtones = .{
                config.grade_midtones[0],
                config.grade_midtones[1],
                config.grade_midtones[2],
                0.0,
            },
            .grade_highlights = .{
                config.grade_highlights[0],
                config.grade_highlights[1],
                config.grade_highlights[2],
                0.0,
            },
            // (enabled 1/0, intensity, size N, 0); zeros when no LUT, which
            // keeps the composite identical to the pre-LUT path.
            .lut_params = postprocess.lutParams(config),
            .view_proj = view_proj,
            .inv_view_proj = inv_view_proj,
        };
        sg.applyUniforms(post_shd.UB_fs_params, sg.asRange(&pp_params));
        sg.draw(0, 6, 1);
    }

    // Feed the BloomPass pyramid result into the composite. Call every frame
    // before render() once the parent owns a BloomPass; pass .{} to detach
    // and return to the legacy single-shader bloom path.
    pub fn setBloomTexture(self: *PostProcessPass, view: sg.View) void {
        self.bloom_tex_view = view;
    }

    pub fn deinit(self: *PostProcessPass) void {
        self.destroyTargets();
        sg.destroySampler(self.postprocess_sampler);
        sg.destroySampler(self.depth_sampler);
        sg.destroySampler(self.lut_sampler);
        sg.destroyPipeline(self.postprocess_pipeline);
        sg.destroyBuffer(self.postprocess_quad_vb);
        sg.destroyBuffer(self.postprocess_quad_ib);
    }
};
