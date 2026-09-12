const sokol = @import("sokol");
const sg = sokol.gfx;
const down_shd = @import("bloom_down_shader");
const up_shd = @import("bloom_up_shader");
const pp = @import("../postprocess.zig");

// High-quality bloom: Karis-average downsample pyramid + tent-filtered
// upsample composite. Owns its offscreen targets; never touches the
// PostProcessPass targets. The result is an unscaled HDR-ish glow texture
// (half resolution); intensity/exposure scaling and tonemapping happen in
// the final postprocess composite so this pass needs no intensity input.
//
// Typical parent (scene.zig) usage, between the SSAO pass and the
// fullscreen postprocess pass:
//
//   var bloom_view = empty_view; // e.g. default white texture view
//   if (self.post_process.enabled and self.post_process.bloom_enabled and
//       self.post_process.bloom_pyramid) {
//       bloom_view = self.bloom_pass.render(
//           self.postprocess_pass.offscreen_resolve_tex_view,
//           self.post_process.bloom_threshold,
//           self.post_process.bloom_pyramid_mips,
//           cur_w, cur_h,
//       );
//       self.stats.draw_calls += ...;
//   }
//   self.postprocess_pass.setBloomTexture(bloom_view);
pub const BloomPass = struct {
    down_images: [pp.BLOOM_MAX_MIPS]sg.Image = [_]sg.Image{.{}} ** pp.BLOOM_MAX_MIPS,
    down_att_views: [pp.BLOOM_MAX_MIPS]sg.View = [_]sg.View{.{}} ** pp.BLOOM_MAX_MIPS,
    down_tex_views: [pp.BLOOM_MAX_MIPS]sg.View = [_]sg.View{.{}} ** pp.BLOOM_MAX_MIPS,

    up_images: [pp.BLOOM_MAX_MIPS]sg.Image = [_]sg.Image{.{}} ** pp.BLOOM_MAX_MIPS,
    up_att_views: [pp.BLOOM_MAX_MIPS]sg.View = [_]sg.View{.{}} ** pp.BLOOM_MAX_MIPS,
    up_tex_views: [pp.BLOOM_MAX_MIPS]sg.View = [_]sg.View{.{}} ** pp.BLOOM_MAX_MIPS,

    sampler: sg.Sampler = .{},
    down_pipeline: sg.Pipeline = .{},
    up_pipeline: sg.Pipeline = .{},
    quad_vb: sg.Buffer = .{},
    quad_ib: sg.Buffer = .{},

    base_width: i32 = 0,
    base_height: i32 = 0,

    pub fn init() BloomPass {
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

        var down_desc = sg.PipelineDesc{
            .shader = sg.makeShader(down_shd.bloomDownShaderDesc(sg.queryBackend())),
            .index_type = .UINT16,
            .depth = .{
                .pixel_format = .NONE,
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = 1,
        };
        down_desc.colors[0].pixel_format = .RGBA8;
        down_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        down_desc.layout.attrs[down_shd.ATTR_bloom_down_position] = .{ .format = .FLOAT2, .offset = 0 };
        down_desc.layout.attrs[down_shd.ATTR_bloom_down_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const down_pip = sg.makePipeline(down_desc);

        var up_desc = sg.PipelineDesc{
            .shader = sg.makeShader(up_shd.bloomUpShaderDesc(sg.queryBackend())),
            .index_type = .UINT16,
            .depth = .{
                .pixel_format = .NONE,
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = 1,
        };
        up_desc.colors[0].pixel_format = .RGBA8;
        up_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        up_desc.layout.attrs[up_shd.ATTR_bloom_up_position] = .{ .format = .FLOAT2, .offset = 0 };
        up_desc.layout.attrs[up_shd.ATTR_bloom_up_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const up_pip = sg.makePipeline(up_desc);

        return .{
            .sampler = smp,
            .down_pipeline = down_pip,
            .up_pipeline = up_pip,
            .quad_vb = vb,
            .quad_ib = ib,
        };
    }

    pub fn resize(self: *BloomPass, width: i32, height: i32) void {
        if (width <= 0 or height <= 0) return;
        if (self.base_width == width and self.base_height == height) return;

        self.destroyTargets();

        // Down chain: level 0 is half resolution, each level halves again.
        // Up chain mirrors down sizes so every upsample lands 1:1.
        for (0..pp.BLOOM_MAX_MIPS) |i| {
            const size = pp.bloomMipSize(width, height, @intCast(i));
            const down_img = sg.makeImage(.{
                .usage = .{ .color_attachment = true },
                .width = size.w,
                .height = size.h,
                .pixel_format = .RGBA8,
                .sample_count = 1,
            });
            self.down_images[i] = down_img;
            self.down_att_views[i] = sg.makeView(.{
                .color_attachment = .{ .image = down_img },
            });
            self.down_tex_views[i] = sg.makeView(.{
                .texture = .{ .image = down_img },
            });

            const up_img = sg.makeImage(.{
                .usage = .{ .color_attachment = true },
                .width = size.w,
                .height = size.h,
                .pixel_format = .RGBA8,
                .sample_count = 1,
            });
            self.up_images[i] = up_img;
            self.up_att_views[i] = sg.makeView(.{
                .color_attachment = .{ .image = up_img },
            });
            self.up_tex_views[i] = sg.makeView(.{
                .texture = .{ .image = up_img },
            });
        }

        self.base_width = width;
        self.base_height = height;
    }

    // Build the pyramid from `src_tex` and return the composited glow view
    // (half-resolution up_tex[0]). Returns an empty view when inactive so
    // the caller can fall back to the single-shader bloom path.
    pub fn render(
        self: *BloomPass,
        src_tex: sg.View,
        threshold: f32,
        mip_count: u32,
        base_w: i32,
        base_h: i32,
    ) sg.View {
        if (self.down_pipeline.id == 0 or self.up_pipeline.id == 0) return .{};
        if (src_tex.id == 0) return .{};
        if (base_w <= 0 or base_h <= 0) return .{};

        self.resize(base_w, base_h);
        if (self.down_images[0].id == 0) return .{};

        const n: usize = @intCast(pp.clampBloomMips(mip_count));

        // Down chain: first level prefilters with the bright-pass threshold,
        // deeper levels only Karis-average (threshold disabled).
        var src_view = src_tex;
        var src_w: f32 = @floatFromInt(base_w);
        var src_h: f32 = @floatFromInt(base_h);
        for (0..n) |i| {
            var pass = sg.Pass{
                .action = .{
                    .colors = [_]sg.ColorAttachmentAction{
                        .{ .load_action = .DONTCARE },
                    } ++ [_]sg.ColorAttachmentAction{.{}} ** 7,
                },
            };
            pass.attachments.colors[0] = self.down_att_views[i];
            sg.beginPass(pass);

            sg.applyPipeline(self.down_pipeline);
            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = self.quad_vb;
            bind.index_buffer = self.quad_ib;
            bind.views[down_shd.VIEW_src_tex] = src_view;
            bind.samplers[down_shd.SMP_smp] = self.sampler;
            sg.applyBindings(bind);

            const fs_params = down_shd.FsParams{
                .src_texel = .{ 1.0 / src_w, 1.0 / src_h, 0.0, 0.0 },
                .params = .{ if (i == 0) threshold else -1.0, 0.0, 0.0, 0.0 },
            };
            sg.applyUniforms(down_shd.UB_fs_params, sg.asRange(&fs_params));
            sg.draw(0, 6, 1);
            sg.endPass();

            src_view = self.down_tex_views[i];
            const size = pp.bloomMipSize(base_w, base_h, @intCast(i));
            src_w = @floatFromInt(size.w);
            src_h = @floatFromInt(size.h);
        }

        // Up chain: tent-upsample the coarse level, add the finer level.
        // Intensity scaling stays in the postprocess composite (blend = 1).
        var current = self.down_tex_views[n - 1];
        var i: usize = n - 1;
        while (i > 0) {
            i -= 1;
            const low_size = pp.bloomMipSize(base_w, base_h, @intCast(i + 1));

            var pass = sg.Pass{
                .action = .{
                    .colors = [_]sg.ColorAttachmentAction{
                        .{ .load_action = .DONTCARE },
                    } ++ [_]sg.ColorAttachmentAction{.{}} ** 7,
                },
            };
            pass.attachments.colors[0] = self.up_att_views[i];
            sg.beginPass(pass);

            sg.applyPipeline(self.up_pipeline);
            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = self.quad_vb;
            bind.index_buffer = self.quad_ib;
            bind.views[up_shd.VIEW_high_tex] = self.down_tex_views[i];
            bind.views[up_shd.VIEW_low_tex] = current;
            bind.samplers[up_shd.SMP_smp] = self.sampler;
            sg.applyBindings(bind);

            const low_w: f32 = @floatFromInt(low_size.w);
            const low_h: f32 = @floatFromInt(low_size.h);
            const fs_params = up_shd.FsParams{
                .texel = .{ 1.0 / low_w, 1.0 / low_h, 0.0, 0.0 },
                .params = .{ 1.0, 0.0, 0.0, 0.0 },
            };
            sg.applyUniforms(up_shd.UB_fs_params, sg.asRange(&fs_params));
            sg.draw(0, 6, 1);
            sg.endPass();

            current = self.up_tex_views[i];
        }

        return current;
    }

    fn destroyTargets(self: *BloomPass) void {
        if (self.down_images[0].id == 0) return;
        for (0..pp.BLOOM_MAX_MIPS) |i| {
            sg.destroyImage(self.down_images[i]);
            sg.destroyView(self.down_att_views[i]);
            sg.destroyView(self.down_tex_views[i]);
            sg.destroyImage(self.up_images[i]);
            sg.destroyView(self.up_att_views[i]);
            sg.destroyView(self.up_tex_views[i]);
            self.down_images[i] = .{};
            self.down_att_views[i] = .{};
            self.down_tex_views[i] = .{};
            self.up_images[i] = .{};
            self.up_att_views[i] = .{};
            self.up_tex_views[i] = .{};
        }
        self.base_width = 0;
        self.base_height = 0;
    }

    pub fn deinit(self: *BloomPass) void {
        self.destroyTargets();
        sg.destroySampler(self.sampler);
        sg.destroyPipeline(self.down_pipeline);
        sg.destroyPipeline(self.up_pipeline);
        sg.destroyBuffer(self.quad_vb);
        sg.destroyBuffer(self.quad_ib);
    }
};
