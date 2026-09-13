const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const ssao_shd = @import("ssao_shader");
const blur_shd = @import("ssao_blur_shader");
const SSAOOptions = @import("../ssao.zig").SSAOOptions;
const Camera = @import("../camera.zig").Camera;

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

    cached_proj: Mat4 = Mat4.identity,
    cached_inv_proj: Mat4 = Mat4.identity,
    last_aspect: f32 = 0.0,
    last_fov: f32 = 0.0,
    last_near: f32 = 0.0,
    last_far: f32 = 0.0,

    pub fn init() SSAOPass {
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

        // 1. Generate 32 hemisphere samples using Fermat spiral (uniform solid angle)
        var kernel: [32][4]f32 = undefined;
        const golden_angle: f32 = 2.0 * std.math.pi * (1.0 - 0.618033988749895);

        for (0..32) |i| {
            const fi = @as(f32, @floatFromInt(i));
            const theta = fi * golden_angle;
            // Distribute z evenly in hemisphere [0.12, 1.0]
            const z = (fi + 0.5) / 32.0 * 0.88 + 0.12;
            const r = @sqrt(@max(0.0, 1.0 - z * z));
            const x = r * @cos(theta);
            const y = r * @sin(theta);

            // Accelerating scale factor towards origin
            const norm_i = (fi + 1.0) / 32.0;
            const scale = std.math.lerp(0.12, 1.0, norm_i * norm_i);

            kernel[i] = .{ x * scale, y * scale, z * scale, 0.0 };
        }

        // 2. Generate 8x8 random rotation noise texture (64 pixels)
        var noise_pixels: [64 * 4]u8 = undefined;
        var prng = std.Random.DefaultPrng.init(1337);
        const rand = prng.random();
        for (0..64) |i| {
            const angle = rand.float(f32) * 2.0 * std.math.pi;
            const nx = @cos(angle);
            const ny = @sin(angle);

            noise_pixels[i * 4 + 0] = @intFromFloat((nx * 0.5 + 0.5) * 255.0);
            noise_pixels[i * 4 + 1] = @intFromFloat((ny * 0.5 + 0.5) * 255.0);
            noise_pixels[i * 4 + 2] = 128;
            noise_pixels[i * 4 + 3] = 255;
        }

        var noise_img_desc = sg.ImageDesc{
            .width = 8,
            .height = 8,
            .pixel_format = .RGBA8,
        };
        noise_img_desc.data.mip_levels[0] = sg.asRange(&noise_pixels);
        const noise_img = sg.makeImage(noise_img_desc);
        const noise_view = sg.makeView(.{
            .texture = .{ .image = noise_img },
        });

        const noise_smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
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

        // Trace at half resolution; the bilateral pass upsamples to full size.
        const raw_img = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = @divTrunc(width, 2) + @mod(width, 2),
            .height = @divTrunc(height, 2) + @mod(height, 2),
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
        camera: Camera,
        aspect: f32,
        depth_tex_view: sg.View,
        config: SSAOOptions,
        cur_w: i32,
        cur_h: i32,
    ) void {
        if (!config.enabled and !config.debug_mode) return;
        if (cur_w <= 0 or cur_h <= 0) return;
        if (self.ssao_pipeline.id == 0 or self.ssao_blur_pipeline.id == 0) return;
        if (depth_tex_view.id == 0) return;

        self.resize(cur_w, cur_h);

        const fov = camera.getFovDeg();
        const near_z = camera.getNear();
        const far_z = camera.getFar();
        if (aspect != self.last_aspect or fov != self.last_fov or near_z != self.last_near or far_z != self.last_far) {
            const proj = camera.getProjectionMatrix(aspect);
            const inv = proj.invert() orelse return;
            self.cached_proj = proj;
            self.cached_inv_proj = inv;
            self.last_aspect = aspect;
            self.last_fov = fov;
            self.last_near = near_z;
            self.last_far = far_z;
        }

        const proj = self.cached_proj;
        const inv_proj = self.cached_inv_proj;

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

        const ao_w: f32 = @floatFromInt(@divTrunc(cur_w, 2) + @mod(cur_w, 2));
        const ao_h: f32 = @floatFromInt(@divTrunc(cur_h, 2) + @mod(cur_h, 2));
        const blur_params = blur_shd.FsParams{
            .resolution = .{
                ao_w,
                ao_h,
                1.0 / ao_w,
                1.0 / ao_h,
            },
            .camera_params = .{
                camera.getNear(),
                camera.getFar(),
                config.radius,
                0.0,
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
