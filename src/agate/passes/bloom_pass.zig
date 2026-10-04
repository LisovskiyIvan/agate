const std = @import("std");
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
//   if (self.post_process.enabled and self.post_process.bloom_enabled) {
//       bloom_view = self.bloom_pass.render(
//           self.postprocess_pass.offscreen_resolve_tex_view,
//           self.post_process.bloom_threshold,
//           self.post_process.bloom_pyramid_mips,
//           self.post_process.bloom_radius,
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
    down_shader: sg.Shader = .{},
    up_shader: sg.Shader = .{},
    quad_vb: sg.Buffer = .{},
    quad_ib: sg.Buffer = .{},

    base_width: i32 = 0,
    base_height: i32 = 0,

    /// Sole-contract render-target pixel format (always RGBA16F; the
    /// backend capability is validated once at Scene startup). Pure
    /// (no sg calls, headless-safe).
    pub fn bloomPixelFormat() sg.PixelFormat {
        return .RGBA16F;
    }

    /// Sanitizes the upsample tent radius (pure, no sg calls): finite
    /// clamped to 0..16; non-finite falls back to 1.0 (the single-texel
    /// tent). 0 collapses every tap onto the center (point upscale).
    pub fn sanitizeBloomRadius(r: f32) f32 {
        if (!std.math.isFinite(r)) return 1.0;
        return @min(@max(r, 0.0), 16.0);
    }

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

        const bloom_fmt = bloomPixelFormat();

        const down_shd_handle = sg.makeShader(down_shd.bloomDownShaderDesc(sg.queryBackend()));
        var down_desc = sg.PipelineDesc{
            .shader = down_shd_handle,
            .index_type = .UINT16,
            .depth = .{
                .pixel_format = .NONE,
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = 1,
        };
        down_desc.colors[0].pixel_format = bloom_fmt;
        down_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        down_desc.layout.attrs[down_shd.ATTR_bloom_down_position] = .{ .format = .FLOAT2, .offset = 0 };
        down_desc.layout.attrs[down_shd.ATTR_bloom_down_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const down_pip = sg.makePipeline(down_desc);

        const up_shd_handle = sg.makeShader(up_shd.bloomUpShaderDesc(sg.queryBackend()));
        var up_desc = sg.PipelineDesc{
            .shader = up_shd_handle,
            .index_type = .UINT16,
            .depth = .{
                .pixel_format = .NONE,
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = 1,
        };
        up_desc.colors[0].pixel_format = bloom_fmt;
        up_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        up_desc.layout.attrs[up_shd.ATTR_bloom_up_position] = .{ .format = .FLOAT2, .offset = 0 };
        up_desc.layout.attrs[up_shd.ATTR_bloom_up_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const up_pip = sg.makePipeline(up_desc);

        return .{
            .sampler = smp,
            .down_pipeline = down_pip,
            .up_pipeline = up_pip,
            .down_shader = down_shd_handle,
            .up_shader = up_shd_handle,
            .quad_vb = vb,
            .quad_ib = ib,
        };
    }

    pub fn resize(self: *BloomPass, width: i32, height: i32) void {
        if (width <= 0 or height <= 0) return;
        if (!sg.isvalid()) return;
        if (self.base_width == width and self.base_height == height) return;

        self.destroyTargets();

        const bloom_fmt = bloomPixelFormat();

        // Down chain: level 0 is half resolution, each level halves again.
        // Up chain mirrors down sizes so every upsample lands 1:1.
        // Store-first rollback: every handle lands in the struct before its
        // state check, so a single destroyTargets frees the failed handle
        // plus all partial handles (views before images). A failed resize
        // leaves zero ids and a zero base size, and render() reports empty.
        for (0..pp.BLOOM_MAX_MIPS) |i| {
            const size = pp.bloomMipSize(width, height, @intCast(i));
            self.down_images[i] = sg.makeImage(.{
                .usage = .{ .color_attachment = true },
                .width = size.w,
                .height = size.h,
                .pixel_format = bloom_fmt,
                .sample_count = 1,
            });
            if (sg.queryImageState(self.down_images[i]) != .VALID) {
                self.destroyTargets();
                return;
            }
            self.down_att_views[i] = sg.makeView(.{
                .color_attachment = .{ .image = self.down_images[i] },
            });
            if (sg.queryViewState(self.down_att_views[i]) != .VALID) {
                self.destroyTargets();
                return;
            }
            self.down_tex_views[i] = sg.makeView(.{
                .texture = .{ .image = self.down_images[i] },
            });
            if (sg.queryViewState(self.down_tex_views[i]) != .VALID) {
                self.destroyTargets();
                return;
            }

            self.up_images[i] = sg.makeImage(.{
                .usage = .{ .color_attachment = true },
                .width = size.w,
                .height = size.h,
                .pixel_format = bloom_fmt,
                .sample_count = 1,
            });
            if (sg.queryImageState(self.up_images[i]) != .VALID) {
                self.destroyTargets();
                return;
            }
            self.up_att_views[i] = sg.makeView(.{
                .color_attachment = .{ .image = self.up_images[i] },
            });
            if (sg.queryViewState(self.up_att_views[i]) != .VALID) {
                self.destroyTargets();
                return;
            }
            self.up_tex_views[i] = sg.makeView(.{
                .texture = .{ .image = self.up_images[i] },
            });
            if (sg.queryViewState(self.up_tex_views[i]) != .VALID) {
                self.destroyTargets();
                return;
            }
        }

        self.base_width = width;
        self.base_height = height;
    }

    // Build the pyramid from `src_tex` and return the composited glow view
    // (half-resolution up_tex[0]). `radius` scales the upsample tent
    // footprint in coarse-mip texel units (1.0 = the single-texel tent, 0 =
    // point upscale); the downsample Karis kernel is untouched, so radius
    // controls the upsample halo only. Returns an empty view when inactive
    // or when the targets failed, so the composite skips the bloom block.
    pub fn render(
        self: *BloomPass,
        src_tex: sg.View,
        threshold: f32,
        mip_count: u32,
        radius: f32,
        base_w: i32,
        base_h: i32,
    ) sg.View {
        if (self.down_pipeline.id == 0 or self.up_pipeline.id == 0) return .{};
        if (src_tex.id == 0) return .{};
        if (base_w <= 0 or base_h <= 0) return .{};

        self.resize(base_w, base_h);
        if (self.base_width != base_w or self.base_height != base_h) return .{};
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
        // The packed texel carries the sanitized tent radius: offsets are
        // measured in coarse-mip (source) texels, so the radius widens the
        // halo without touching the downsample kernel.
        const tent_radius = sanitizeBloomRadius(radius);
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
                .texel = .{ tent_radius / low_w, tent_radius / low_h, 0.0, 0.0 },
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
        for (0..pp.BLOOM_MAX_MIPS) |i| {
            if (self.down_att_views[i].id != 0) sg.destroyView(self.down_att_views[i]);
            if (self.down_tex_views[i].id != 0) sg.destroyView(self.down_tex_views[i]);
            if (self.up_att_views[i].id != 0) sg.destroyView(self.up_att_views[i]);
            if (self.up_tex_views[i].id != 0) sg.destroyView(self.up_tex_views[i]);
            if (self.down_images[i].id != 0) sg.destroyImage(self.down_images[i]);
            if (self.up_images[i].id != 0) sg.destroyImage(self.up_images[i]);
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
        if (self.down_shader.id != 0) sg.destroyShader(self.down_shader);
        if (self.up_shader.id != 0) sg.destroyShader(self.up_shader);
        self.down_shader = .{};
        self.up_shader = .{};
        sg.destroyBuffer(self.quad_vb);
        sg.destroyBuffer(self.quad_ib);
    }
};

test "bloom pass fail-closes headless with no state touched" {
    // Zero-initialized pass (never init'ed: no sg context headless) must
    // return an empty view before any sg.* call.
    var pass: BloomPass = .{};
    const empty = pass.render(.{}, 1.0, 5, 2.0, 1280, 720);
    try std.testing.expectEqual(@as(u32, 0), empty.id);
    // Empty source, degenerate size, and missing pipelines all fail closed
    // the same way (guard order: pipelines first, then source, then size).
    try std.testing.expectEqual(@as(u32, 0), pass.render(.{ .id = 9 }, 1.0, 5, 2.0, 1280, 720).id);
    try std.testing.expectEqual(@as(u32, 0), pass.render(.{ .id = 9 }, 1.0, 5, 2.0, 0, 720).id);
    // The sole-contract format is pinned headless (no sg calls).
    try std.testing.expectEqual(sg.PixelFormat.RGBA16F, BloomPass.bloomPixelFormat());
    // Base size untouched: no resize happened.
    try std.testing.expectEqual(@as(i32, 0), pass.base_width);
    try std.testing.expectEqual(@as(i32, 0), pass.base_height);
}

test "sanitizeBloomRadius clamps finite, neutral on non-finite" {
    // Pure CPU (no sg calls): identity inside the range, 1.0 preserved.
    try std.testing.expectEqual(@as(f32, 1.0), BloomPass.sanitizeBloomRadius(1.0));
    try std.testing.expectEqual(@as(f32, 2.0), BloomPass.sanitizeBloomRadius(2.0));
    // 0 is valid: every upsample tap lands on the center (point upscale).
    try std.testing.expectEqual(@as(f32, 0.0), BloomPass.sanitizeBloomRadius(0.0));
    // Finite clamps at both ends.
    try std.testing.expectEqual(@as(f32, 0.0), BloomPass.sanitizeBloomRadius(-3.0));
    try std.testing.expectEqual(@as(f32, 16.0), BloomPass.sanitizeBloomRadius(99.0));
    // Non-finite falls back to the neutral single-texel tent.
    try std.testing.expectEqual(@as(f32, 1.0), BloomPass.sanitizeBloomRadius(std.math.nan(f32)));
    try std.testing.expectEqual(@as(f32, 1.0), BloomPass.sanitizeBloomRadius(std.math.inf(f32)));
}
