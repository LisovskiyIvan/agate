const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const ssao_shd = @import("ssao_shader");
const blur_shd = @import("ssao_blur_shader");
const SSAOConfig = @import("../ssao.zig").SSAOConfig;
const ArcRotateCamera = @import("../camera.zig").ArcRotateCamera;

pub const SSAOPass = struct {
    ssao_raw_image: sg.Image = .{},
    ssao_raw_att_view: sg.View = .{},
    ssao_raw_tex_view: sg.View = .{},

    ssao_blur_image: sg.Image = .{},
    ssao_blur_att_view: sg.View = .{},
    ssao_blur_tex_view: sg.View = .{},

    noise_image: sg.Image = .{},
    noise_tex_view: sg.View = .{},
    noise_sampler: sg.Sampler = .{},

    depth_sampler: sg.Sampler = .{},
    blur_sampler: sg.Sampler = .{},

    ssao_pipeline: sg.Pipeline = .{},
    ssao_blur_pipeline: sg.Pipeline = .{},

    quad_vb: sg.Buffer = .{},
    quad_ib: sg.Buffer = .{},

    kernel_samples: [32][4]f32 = undefined,

    width: i32 = 0,
    height: i32 = 0,

    pub fn init() SSAOPass {
        // Fullscreen Quad (XY, UV)
        const quad_vertices = [_]f32{
            // x,     y,    u,   v
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
             1.0,  1.0,  1.0, 1.0,
            -1.0,  1.0,  0.0, 1.0,
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

        // 1. Generate 32 hemisphere samples
        var kernel: [32][4]f32 = undefined;
        var prng = std.Random.DefaultPrng.init(1337);
        const rand = prng.random();

        for (0..32) |i| {
            var sample = Vec3.new(
                rand.float(f32) * 2.0 - 1.0,
                rand.float(f32) * 2.0 - 1.0,
                rand.float(f32) * 0.85 + 0.15, // z > 0 hemisphere
            ).normalize();

            // Accelerating scale factor towards origin
            var scale = @as(f32, @floatFromInt(i)) / 32.0;
            scale = std.math.lerp(0.1, 1.0, scale * scale);
            sample = sample.scale(scale);

            kernel[i] = .{ sample.x, sample.y, sample.z, 0.0 };
        }

        // 2. Generate 4x4 random rotation noise texture (16 pixels)
        var noise_pixels: [16 * 4]u8 = undefined;
        for (0..16) |i| {
            const rx = rand.float(f32) * 2.0 - 1.0;
            const ry = rand.float(f32) * 2.0 - 1.0;
            const len = @sqrt(rx * rx + ry * ry);
            const nx = if (len > 0.001) rx / len else 1.0;
            const ny = if (len > 0.001) ry / len else 0.0;

            noise_pixels[i * 4 + 0] = @intFromFloat((nx * 0.5 + 0.5) * 255.0);
            noise_pixels[i * 4 + 1] = @intFromFloat((ny * 0.5 + 0.5) * 255.0);
            noise_pixels[i * 4 + 2] = 128;
            noise_pixels[i * 4 + 3] = 255;
        }

        var noise_img_desc = sg.ImageDesc{
            .width = 4,
            .height = 4,
            .pixel_format = .RGBA8,
        };
        noise_img_desc.data.mip_levels[0] = sg.asRange(&noise_pixels);
        const noise_img = sg.makeImage(noise_img_desc);
        const noise_view = sg.makeView(.{
            .texture = .{ .image = noise_img },
        });

        const noise_smp = sg.makeSampler(.{
            .min_filter = .NEAREST,
            .mag_filter = .NEAREST,
            .wrap_u = .REPEAT,
            .wrap_v = .REPEAT,
        });

        const depth_smp = sg.makeSampler(.{
            .min_filter = .NEAREST,
            .mag_filter = .NEAREST,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });

        const blur_smp = sg.makeSampler(.{
            .min_filter = .NEAREST,
            .mag_filter = .NEAREST,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });

        // 3. SSAO pipeline
        var ssao_pip_desc = sg.PipelineDesc{
            .shader = sg.makeShader(ssao_shd.ssaoShaderDesc(sg.queryBackend())),
            .index_type = .UINT16,
            .depth = .{
                .pixel_format = .NONE,
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = 1,
        };
        ssao_pip_desc.colors[0].pixel_format = .RGBA8;
        ssao_pip_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        ssao_pip_desc.layout.attrs[ssao_shd.ATTR_ssao_position] = .{ .format = .FLOAT2, .offset = 0 };
        ssao_pip_desc.layout.attrs[ssao_shd.ATTR_ssao_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const ssao_pip = sg.makePipeline(ssao_pip_desc);

        // 4. SSAO Blur pipeline
        var blur_pip_desc = sg.PipelineDesc{
            .shader = sg.makeShader(blur_shd.ssaoBlurShaderDesc(sg.queryBackend())),
            .index_type = .UINT16,
            .depth = .{
                .pixel_format = .NONE,
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = 1,
        };
        blur_pip_desc.colors[0].pixel_format = .RGBA8;
        blur_pip_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        blur_pip_desc.layout.attrs[blur_shd.ATTR_ssao_blur_position] = .{ .format = .FLOAT2, .offset = 0 };
        blur_pip_desc.layout.attrs[blur_shd.ATTR_ssao_blur_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const blur_pip = sg.makePipeline(blur_pip_desc);

        return .{
            .noise_image = noise_img,
            .noise_tex_view = noise_view,
            .noise_sampler = noise_smp,
            .depth_sampler = depth_smp,
            .blur_sampler = blur_smp,
            .ssao_pipeline = ssao_pip,
            .ssao_blur_pipeline = blur_pip,
            .quad_vb = vb,
            .quad_ib = ib,
            .kernel_samples = kernel,
        };
    }

    pub fn resize(self: *SSAOPass, width: i32, height: i32) void {
        if (width <= 0 or height <= 0) return;
        if (self.width == width and self.height == height) return;

        if (self.ssao_raw_image.id != 0) {
            sg.destroyImage(self.ssao_raw_image);
            sg.destroyView(self.ssao_raw_att_view);
            sg.destroyView(self.ssao_raw_tex_view);

            sg.destroyImage(self.ssao_blur_image);
            sg.destroyView(self.ssao_blur_att_view);
            sg.destroyView(self.ssao_blur_tex_view);
        }

        // SSAO Raw Target (RGBA8)
        const raw_img = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = width,
            .height = height,
            .pixel_format = .RGBA8,
            .sample_count = 1,
        });
        const raw_att = sg.makeView(.{
            .color_attachment = .{ .image = raw_img },
        });
        const raw_tex = sg.makeView(.{
            .texture = .{ .image = raw_img },
        });

        // SSAO Blur Target (RGBA8)
        const blur_img = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = width,
            .height = height,
            .pixel_format = .RGBA8,
            .sample_count = 1,
        });
        const blur_att = sg.makeView(.{
            .color_attachment = .{ .image = blur_img },
        });
        const blur_tex = sg.makeView(.{
            .texture = .{ .image = blur_img },
        });

        self.width = width;
        self.height = height;
        self.ssao_raw_image = raw_img;
        self.ssao_raw_att_view = raw_att;
        self.ssao_raw_tex_view = raw_tex;
        self.ssao_blur_image = blur_img;
        self.ssao_blur_att_view = blur_att;
        self.ssao_blur_tex_view = blur_tex;
    }

    pub fn render(
        self: *SSAOPass,
        camera: ArcRotateCamera,
        aspect: f32,
        depth_tex_view: sg.View,
        config: SSAOConfig,
        cur_w: i32,
        cur_h: i32,
    ) void {
        if (!config.enabled) return;
        if (self.ssao_pipeline.id == 0 or self.ssao_blur_pipeline.id == 0) return;
        if (depth_tex_view.id == 0) return;

        self.resize(cur_w, cur_h);

        const proj = camera.getProjectionMatrix(aspect);
        const inv_proj = proj.invert() orelse return;

        // ---------------------------------------------
        // PASS 1: Generate SSAO into ssao_raw_image
        // ---------------------------------------------
        var ssao_pass = sg.Pass{
            .action = .{
                .colors = [_]sg.ColorAttachmentAction{
                    .{ .load_action = .DONTCARE },
                } ++ [_]sg.ColorAttachmentAction{.{}} ** 7,
            },
        };
        ssao_pass.attachments.colors[0] = self.ssao_raw_att_view;
        sg.beginPass(ssao_pass);

        sg.applyPipeline(self.ssao_pipeline);
        var bind1 = sg.Bindings{};
        bind1.vertex_buffers[0] = self.quad_vb;
        bind1.index_buffer = self.quad_ib;
        bind1.views[ssao_shd.VIEW_depth_tex] = depth_tex_view;
        bind1.views[ssao_shd.VIEW_noise_tex] = self.noise_tex_view;
        bind1.samplers[ssao_shd.SMP_smp_depth] = self.depth_sampler;
        bind1.samplers[ssao_shd.SMP_smp_noise] = self.noise_sampler;
        sg.applyBindings(bind1);

        const fs_params = ssao_shd.FsParams{
            .projection = proj,
            .inv_projection = inv_proj,
            .kernel_samples = self.kernel_samples,
            .params = .{ config.radius, config.bias, config.intensity, config.power },
            .resolution = .{
                @floatFromInt(cur_w),
                @floatFromInt(cur_h),
                1.0 / @as(f32, @floatFromInt(cur_w)),
                1.0 / @as(f32, @floatFromInt(cur_h)),
            },
        };
        sg.applyUniforms(ssao_shd.UB_fs_params, sg.asRange(&fs_params));
        sg.draw(0, 6, 1);
        sg.endPass();

        // ---------------------------------------------
        // PASS 2: Bilateral Blur into ssao_blur_image
        // ---------------------------------------------
        var blur_pass = sg.Pass{
            .action = .{
                .colors = [_]sg.ColorAttachmentAction{
                    .{ .load_action = .DONTCARE },
                } ++ [_]sg.ColorAttachmentAction{.{}} ** 7,
            },
        };
        blur_pass.attachments.colors[0] = self.ssao_blur_att_view;
        sg.beginPass(blur_pass);

        sg.applyPipeline(self.ssao_blur_pipeline);
        var bind2 = sg.Bindings{};
        bind2.vertex_buffers[0] = self.quad_vb;
        bind2.index_buffer = self.quad_ib;
        bind2.views[blur_shd.VIEW_ssao_tex] = self.ssao_raw_tex_view;
        bind2.views[blur_shd.VIEW_depth_tex] = depth_tex_view;
        bind2.samplers[blur_shd.SMP_smp] = self.blur_sampler;
        sg.applyBindings(bind2);

        const blur_params = blur_shd.FsParams{
            .resolution = .{
                @floatFromInt(cur_w),
                @floatFromInt(cur_h),
                1.0 / @as(f32, @floatFromInt(cur_w)),
                1.0 / @as(f32, @floatFromInt(cur_h)),
            },
        };
        sg.applyUniforms(blur_shd.UB_fs_params, sg.asRange(&blur_params));
        sg.draw(0, 6, 1);
        sg.endPass();
    }

    pub fn deinit(self: *SSAOPass) void {
        if (self.ssao_raw_image.id != 0) {
            sg.destroyImage(self.ssao_raw_image);
            sg.destroyView(self.ssao_raw_att_view);
            sg.destroyView(self.ssao_raw_tex_view);

            sg.destroyImage(self.ssao_blur_image);
            sg.destroyView(self.ssao_blur_att_view);
            sg.destroyView(self.ssao_blur_tex_view);
        }
        if (self.noise_image.id != 0) {
            sg.destroyImage(self.noise_image);
            sg.destroyView(self.noise_tex_view);
        }
        sg.destroySampler(self.noise_sampler);
        sg.destroySampler(self.depth_sampler);
        sg.destroySampler(self.blur_sampler);
        sg.destroyPipeline(self.ssao_pipeline);
        sg.destroyPipeline(self.ssao_blur_pipeline);
        sg.destroyBuffer(self.quad_vb);
        sg.destroyBuffer(self.quad_ib);
    }
};
