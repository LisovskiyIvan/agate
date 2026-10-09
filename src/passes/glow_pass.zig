const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const extract_shd = @import("glow_extract_shader");
const blur_shd = @import("glow_blur_shader");
const pp = @import("../postprocess.zig");

// Glow layer v1: threshold extract + separable (H then V) Gaussian blur.
// Owns its half-resolution offscreen targets; never touches the
// PostProcessPass targets or the BloomPass pyramid. The result is an
// unscaled halo texture; intensity/tint scaling and the additive composite
// happen in the final postprocess composite (after bloom's block) so this
// pass needs no intensity input and toggling glow never changes bloom's
// output (and vice versa).
//
// Typical parent (scene/postfx_stack.zig) usage, between the bloom pyramid
// and the fullscreen postprocess pass:
//
//   var glow_view = empty_view; // e.g. the resolved scene view placeholder
//   if (postprocess.glowActive(post.enabled, post)) {
//       glow_view = self.glow_pass.render(
//           self.postprocess_pass.offscreen_resolve_tex_view,
//           post.glow_threshold,
//           post.glow_radius,
//           cur_w, cur_h,
//       );
//       self.stats.draw_calls += postprocess.GLOW_PASS_DRAWS;
//   }
//   self.postprocess_pass.setGlowTexture(glow_view);
//
// Headless behavior mirrors BloomPass exactly: zero handles fail closed
// (render returns an empty view before any sg.* call), no sg.isvalid gates.
// The pass is uniform-only (applyUniforms, never sg.updateBuffer), so it
// records nothing into gpu_upload_meter and renderReuse replays stay
// upload-free by construction. Target destruction is immediate (same as
// BloomPass.destroyTargets, context thread only): the retire queue stays
// for game-side removals that can happen off-context (probes, ui3d panels,
// meshes), not for pass-owned ping-pong RTs recreated on resize.
// Per-mesh glow weights are a documented non-goal for v1 (pipeline changes).
pub const GlowPass = struct {
    extract_image: sg.Image = .{},
    extract_att_view: sg.View = .{},
    extract_tex_view: sg.View = .{},

    blur_images: [2]sg.Image = [_]sg.Image{.{}} ** 2,
    blur_att_views: [2]sg.View = [_]sg.View{.{}} ** 2,
    blur_tex_views: [2]sg.View = [_]sg.View{.{}} ** 2,

    sampler: sg.Sampler = .{},
    extract_pipeline: sg.Pipeline = .{},
    blur_pipeline: sg.Pipeline = .{},
    extract_shader: sg.Shader = .{},
    blur_shader: sg.Shader = .{},
    quad_vb: sg.Buffer = .{},
    quad_ib: sg.Buffer = .{},

    base_width: i32 = 0,
    base_height: i32 = 0,

    /// Sole-contract render-target pixel format (always RGBA16F; the
    /// backend capability is validated once at Scene startup). Pure
    /// (no sg calls, headless-safe).
    pub fn glowPixelFormat() sg.PixelFormat {
        return .RGBA16F;
    }

    /// Bytes per pixel of the pass targets (always 8: RGBA16F). Pure
    /// (no sg calls, headless-safe).
    pub fn glowBytesPerPixel() usize {
        return if (glowPixelFormat() == .RGBA16F) 8 else 4;
    }

    /// VRAM census helper (pure byte math, no GPU calls, headless-safe):
    /// three half-resolution targets (extract + H/V ping-pong) at
    /// `bytes_per_pixel` (see glowBytesPerPixel). Single source of truth
    /// for profiler/snapshot.zig.
    pub fn targetBytes(width: i32, height: i32, bytes_per_pixel: usize) usize {
        const hw: usize = @intCast(@max(1, @divTrunc(width, 2)));
        const hh: usize = @intCast(@max(1, @divTrunc(height, 2)));
        return 3 * hw * hh * bytes_per_pixel;
    }

    pub fn init() GlowPass {
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

        const glow_fmt = glowPixelFormat();

        const extract_shd_handle = sg.makeShader(extract_shd.glowExtractShaderDesc(sg.queryBackend()));
        var extract_desc = sg.PipelineDesc{
            .shader = extract_shd_handle,
            .index_type = .UINT16,
            .depth = .{
                .pixel_format = .NONE,
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = 1,
        };
        extract_desc.colors[0].pixel_format = glow_fmt;
        extract_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        extract_desc.layout.attrs[extract_shd.ATTR_glow_extract_position] = .{ .format = .FLOAT2, .offset = 0 };
        extract_desc.layout.attrs[extract_shd.ATTR_glow_extract_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const extract_pip = sg.makePipeline(extract_desc);

        const blur_shd_handle = sg.makeShader(blur_shd.glowBlurShaderDesc(sg.queryBackend()));
        var blur_desc = sg.PipelineDesc{
            .shader = blur_shd_handle,
            .index_type = .UINT16,
            .depth = .{
                .pixel_format = .NONE,
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = 1,
        };
        blur_desc.colors[0].pixel_format = glow_fmt;
        blur_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        blur_desc.layout.attrs[blur_shd.ATTR_glow_blur_position] = .{ .format = .FLOAT2, .offset = 0 };
        blur_desc.layout.attrs[blur_shd.ATTR_glow_blur_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const blur_pip = sg.makePipeline(blur_desc);

        return .{
            .sampler = smp,
            .extract_pipeline = extract_pip,
            .blur_pipeline = blur_pip,
            .extract_shader = extract_shd_handle,
            .blur_shader = blur_shd_handle,
            .quad_vb = vb,
            .quad_ib = ib,
        };
    }

    pub fn resize(self: *GlowPass, width: i32, height: i32) void {
        if (width <= 0 or height <= 0) return;
        if (!sg.isvalid()) return;
        if (self.base_width == width and self.base_height == height) return;

        self.destroyTargets();

        const glow_fmt = glowPixelFormat();
        const size = pp.bloomMipSize(width, height, 0);

        // Store-first rollback: every handle lands in the struct before its
        // state check, so a single destroyTargets frees the failed handle
        // plus all partial handles (views before images). A failed resize
        // leaves zero ids and a zero base size, and render() reports empty.
        self.extract_image = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = size.w,
            .height = size.h,
            .pixel_format = glow_fmt,
            .sample_count = 1,
        });
        if (sg.queryImageState(self.extract_image) != .VALID) {
            self.destroyTargets();
            return;
        }
        self.extract_att_view = sg.makeView(.{
            .color_attachment = .{ .image = self.extract_image },
        });
        if (sg.queryViewState(self.extract_att_view) != .VALID) {
            self.destroyTargets();
            return;
        }
        self.extract_tex_view = sg.makeView(.{
            .texture = .{ .image = self.extract_image },
        });
        if (sg.queryViewState(self.extract_tex_view) != .VALID) {
            self.destroyTargets();
            return;
        }

        for (0..2) |i| {
            self.blur_images[i] = sg.makeImage(.{
                .usage = .{ .color_attachment = true },
                .width = size.w,
                .height = size.h,
                .pixel_format = glow_fmt,
                .sample_count = 1,
            });
            if (sg.queryImageState(self.blur_images[i]) != .VALID) {
                self.destroyTargets();
                return;
            }
            self.blur_att_views[i] = sg.makeView(.{
                .color_attachment = .{ .image = self.blur_images[i] },
            });
            if (sg.queryViewState(self.blur_att_views[i]) != .VALID) {
                self.destroyTargets();
                return;
            }
            self.blur_tex_views[i] = sg.makeView(.{
                .texture = .{ .image = self.blur_images[i] },
            });
            if (sg.queryViewState(self.blur_tex_views[i]) != .VALID) {
                self.destroyTargets();
                return;
            }
        }

        self.base_width = width;
        self.base_height = height;
    }

    fn blurStage(
        self: *GlowPass,
        src_view: sg.View,
        direction: f32,
        sigma: f32,
        texel_w: f32,
        texel_h: f32,
        slot: usize,
    ) void {
        var pass = sg.Pass{
            .action = .{
                .colors = [_]sg.ColorAttachmentAction{
                    .{ .load_action = .DONTCARE },
                } ++ [_]sg.ColorAttachmentAction{.{}} ** 7,
            },
        };
        pass.attachments.colors[0] = self.blur_att_views[slot];
        sg.beginPass(pass);

        sg.applyPipeline(self.blur_pipeline);
        var bind = sg.Bindings{};
        bind.vertex_buffers[0] = self.quad_vb;
        bind.index_buffer = self.quad_ib;
        bind.views[blur_shd.VIEW_src_tex] = src_view;
        bind.samplers[blur_shd.SMP_smp] = self.sampler;
        sg.applyBindings(bind);

        const fs_params = blur_shd.FsParams{
            .texel = .{ texel_w, texel_h, 0.0, 0.0 },
            .params = .{ direction, sigma, 0.0, 0.0 },
        };
        sg.applyUniforms(blur_shd.UB_fs_params, sg.asRange(&fs_params));
        sg.draw(0, 6, 1);
        sg.endPass();
    }

    // Build the halo from `src_tex` and return the blurred glow view
    // (half-resolution blur slot 1). Returns an empty view when inactive so
    // the caller can fall back to the placeholder (bit-identical composite).
    pub fn render(
        self: *GlowPass,
        src_tex: sg.View,
        threshold: f32,
        radius: f32,
        base_w: i32,
        base_h: i32,
    ) sg.View {
        if (self.extract_pipeline.id == 0 or self.blur_pipeline.id == 0) return .{};
        if (src_tex.id == 0) return .{};
        if (base_w <= 0 or base_h <= 0) return .{};

        self.resize(base_w, base_h);
        if (self.base_width != base_w or self.base_height != base_h) return .{};
        if (self.extract_image.id == 0) return .{};

        // Stage 1: threshold extract into the extract target.
        var pass = sg.Pass{
            .action = .{
                .colors = [_]sg.ColorAttachmentAction{
                    .{ .load_action = .DONTCARE },
                } ++ [_]sg.ColorAttachmentAction{.{}} ** 7,
            },
        };
        pass.attachments.colors[0] = self.extract_att_view;
        sg.beginPass(pass);

        sg.applyPipeline(self.extract_pipeline);
        var extract_bind = sg.Bindings{};
        extract_bind.vertex_buffers[0] = self.quad_vb;
        extract_bind.index_buffer = self.quad_ib;
        extract_bind.views[extract_shd.VIEW_src_tex] = src_tex;
        extract_bind.samplers[extract_shd.SMP_smp] = self.sampler;
        sg.applyBindings(extract_bind);

        const src_w: f32 = @floatFromInt(base_w);
        const src_h: f32 = @floatFromInt(base_h);
        const extract_params = extract_shd.FsParams{
            .src_texel = .{ 1.0 / src_w, 1.0 / src_h, 0.0, 0.0 },
            .params = .{ threshold, 0.0, 0.0, 0.0 },
        };
        sg.applyUniforms(extract_shd.UB_fs_params, sg.asRange(&extract_params));
        sg.draw(0, 6, 1);
        sg.endPass();

        // Stages 2-3: separable blur, H into slot 0 then V into slot 1.
        // Intensity/tint scaling stays in the postprocess composite.
        const size = pp.bloomMipSize(base_w, base_h, 0);
        const texel_w: f32 = 1.0 / @as(f32, @floatFromInt(size.w));
        const texel_h: f32 = 1.0 / @as(f32, @floatFromInt(size.h));
        self.blurStage(self.extract_tex_view, 0.0, radius, texel_w, texel_h, 0);
        self.blurStage(self.blur_tex_views[0], 1.0, radius, texel_w, texel_h, 1);

        return self.blur_tex_views[1];
    }

    fn destroyTargets(self: *GlowPass) void {
        if (self.extract_att_view.id != 0) sg.destroyView(self.extract_att_view);
        if (self.extract_tex_view.id != 0) sg.destroyView(self.extract_tex_view);
        if (self.extract_image.id != 0) sg.destroyImage(self.extract_image);
        self.extract_image = .{};
        self.extract_att_view = .{};
        self.extract_tex_view = .{};
        for (0..2) |i| {
            if (self.blur_att_views[i].id != 0) sg.destroyView(self.blur_att_views[i]);
            if (self.blur_tex_views[i].id != 0) sg.destroyView(self.blur_tex_views[i]);
            if (self.blur_images[i].id != 0) sg.destroyImage(self.blur_images[i]);
            self.blur_images[i] = .{};
            self.blur_att_views[i] = .{};
            self.blur_tex_views[i] = .{};
        }
        self.base_width = 0;
        self.base_height = 0;
    }

    pub fn deinit(self: *GlowPass) void {
        self.destroyTargets();
        sg.destroySampler(self.sampler);
        sg.destroyPipeline(self.extract_pipeline);
        sg.destroyPipeline(self.blur_pipeline);
        if (self.extract_shader.id != 0) sg.destroyShader(self.extract_shader);
        if (self.blur_shader.id != 0) sg.destroyShader(self.blur_shader);
        self.extract_shader = .{};
        self.blur_shader = .{};
        sg.destroyBuffer(self.quad_vb);
        sg.destroyBuffer(self.quad_ib);
    }
};
