const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const trace_shd = @import("ssr_shader");
const blur_shd = @import("ssr_blur_shader");
const pp = @import("../postprocess.zig");
const PostProcessOptions = pp.PostProcessOptions;

// Screen-space reflections v2: half-resolution trace + bilateral
// (depth-aware) denoise. The final depth-aware upscale lives in the
// fullscreen composite (sampleSSRPass 2x2 gather), so this pass stays at
// 2 draws and 2 half-res targets with no full-res intermediate.
//
// Per active frame (parent gates on postprocess.ssrPassActive, which is
// the master enabled + intensity floor + half_res_pass mode; MSAA
// suppression clears ssr.enabled so pass mode joins it for free):
//  1. Trace: half-res port of the v1 inline raymarch (same floor gate,
//     clip-space march, thickness/max-distance, edge/distance/fresnel
//     weights). Outputs (reflection rgb, blend weight) with NO intensity
//     baked in; intensity scaling stays in the postprocess composite.
//  2. Blur: 3x3 bilateral over the trace output with the full-res scene
//     depth as the edge gate, so neighboring denoised texels never cross
//     silhouettes.
//  3. Composite: the parent feeds the denoised view into the fullscreen
//     postprocess shader, which reconstructs full res with a depth-aware
//     2x2 gather (bilinear would smear across silhouettes) and mixes by
//     intensity * weight at the v1 stack position.
//
// Headless behavior mirrors SsgiPass: zero handles fail closed (render
// returns an empty view before any sg.* call). The pass is uniform-only
// past its input binds (applyUniforms, never sg.updateBuffer). Targets
// allocate lazily on the first active render, so post-on with pass mode
// off holds no SSR VRAM. Target destruction is immediate (context thread
// only).
pub const SsrPass = struct {
    trace_image: sg.Image = .{},
    trace_att_view: sg.View = .{},
    trace_tex_view: sg.View = .{},

    blur_image: sg.Image = .{},
    blur_att_view: sg.View = .{},
    blur_tex_view: sg.View = .{},

    sampler: sg.Sampler = .{},
    depth_sampler: sg.Sampler = .{},

    trace_pipeline: sg.Pipeline = .{},
    blur_pipeline: sg.Pipeline = .{},
    trace_shader: sg.Shader = .{},
    blur_shader: sg.Shader = .{},

    quad_vb: sg.Buffer = .{},
    quad_ib: sg.Buffer = .{},

    /// Full EFFECTIVE (render-scale) base size the half-res targets were
    /// derived from. 0 until the first active render (lazy allocation).
    width: i32 = 0,
    height: i32 = 0,

    /// Sole-contract render-target pixel format (always RGBA16F: HDR
    /// reflection + weight; validated once at Scene startup). Pure.
    pub fn ssrPixelFormat() sg.PixelFormat {
        return .RGBA16F;
    }

    pub fn ssrBytesPerPixel() usize {
        return if (ssrPixelFormat() == .RGBA16F) 8 else 4;
    }

    /// VRAM census helper (pure byte math): two half-res targets
    /// (trace + denoised) at `bytes_per_pixel`.
    pub fn targetBytes(base_w: i32, base_h: i32, bytes_per_pixel: usize) usize {
        return pp.ssrTargetBytes(base_w, base_h, bytes_per_pixel);
    }

    pub fn init() SsrPass {
        const quad_vertices = [_]f32{
            -1.0, -1.0, 0.0, 0.0,
            1.0,  -1.0, 1.0, 0.0,
            1.0,  1.0,  1.0, 1.0,
            -1.0, 1.0,  0.0, 1.0,
        };
        const quad_indices = [_]u16{ 0, 1, 2, 0, 2, 3 };

        const vb = sg.makeBuffer(.{ .data = sg.asRange(&quad_vertices) });
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

        const fmt = ssrPixelFormat();

        const trace_shd_handle = sg.makeShader(trace_shd.ssrShaderDesc(sg.queryBackend()));
        var trace_desc = sg.PipelineDesc{
            .shader = trace_shd_handle,
            .index_type = .UINT16,
            .depth = .{ .pixel_format = .NONE, .compare = .ALWAYS, .write_enabled = false },
            .cull_mode = .NONE,
            .sample_count = 1,
        };
        trace_desc.colors[0].pixel_format = fmt;
        trace_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        trace_desc.layout.attrs[trace_shd.ATTR_ssr_position] = .{ .format = .FLOAT2, .offset = 0 };
        trace_desc.layout.attrs[trace_shd.ATTR_ssr_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const trace_pip = sg.makePipeline(trace_desc);

        const blur_shd_handle = sg.makeShader(blur_shd.ssrBlurShaderDesc(sg.queryBackend()));
        var blur_desc = sg.PipelineDesc{
            .shader = blur_shd_handle,
            .index_type = .UINT16,
            .depth = .{ .pixel_format = .NONE, .compare = .ALWAYS, .write_enabled = false },
            .cull_mode = .NONE,
            .sample_count = 1,
        };
        blur_desc.colors[0].pixel_format = fmt;
        blur_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        blur_desc.layout.attrs[blur_shd.ATTR_ssr_blur_position] = .{ .format = .FLOAT2, .offset = 0 };
        blur_desc.layout.attrs[blur_shd.ATTR_ssr_blur_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const blur_pip = sg.makePipeline(blur_desc);

        return .{
            .sampler = smp,
            .depth_sampler = depth_smp,
            .trace_pipeline = trace_pip,
            .blur_pipeline = blur_pip,
            .trace_shader = trace_shd_handle,
            .blur_shader = blur_shd_handle,
            .quad_vb = vb,
            .quad_ib = ib,
        };
    }

    /// (Re)allocates both half-res targets from the full EFFECTIVE base
    /// size (ceil-half per axis, the ssrTargetSize rule). No-op on
    /// degenerate or unchanged sizes. Context thread only.
    pub fn resize(self: *SsrPass, base_w: i32, base_h: i32) void {
        if (base_w <= 0 or base_h <= 0) return;
        if (!sg.isvalid()) return;
        if (self.width == base_w and self.height == base_h) return;

        self.destroyTargets();

        const size = pp.ssrTargetSize(base_w, base_h);
        const fmt = ssrPixelFormat();

        self.trace_image = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = size.w,
            .height = size.h,
            .pixel_format = fmt,
            .sample_count = 1,
        });
        if (sg.queryImageState(self.trace_image) != .VALID) {
            self.destroyTargets();
            return;
        }
        self.trace_att_view = sg.makeView(.{ .color_attachment = .{ .image = self.trace_image } });
        if (sg.queryViewState(self.trace_att_view) != .VALID) {
            self.destroyTargets();
            return;
        }
        self.trace_tex_view = sg.makeView(.{ .texture = .{ .image = self.trace_image } });
        if (sg.queryViewState(self.trace_tex_view) != .VALID) {
            self.destroyTargets();
            return;
        }

        self.blur_image = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = size.w,
            .height = size.h,
            .pixel_format = fmt,
            .sample_count = 1,
        });
        if (sg.queryImageState(self.blur_image) != .VALID) {
            self.destroyTargets();
            return;
        }
        self.blur_att_view = sg.makeView(.{ .color_attachment = .{ .image = self.blur_image } });
        if (sg.queryViewState(self.blur_att_view) != .VALID) {
            self.destroyTargets();
            return;
        }
        self.blur_tex_view = sg.makeView(.{ .texture = .{ .image = self.blur_image } });
        if (sg.queryViewState(self.blur_tex_view) != .VALID) {
            self.destroyTargets();
            return;
        }

        self.width = base_w;
        self.height = base_h;
    }

    pub const RenderArgs = struct {
        scene_view: sg.View = .{},
        depth_view: sg.View = .{},
        view_proj: Mat4 = Mat4.identity,
        inv_view_proj: Mat4 = Mat4.identity,
        camera_pos: Vec3 = Vec3.zero,
        config: PostProcessOptions = .{},
        near_z: f32 = 0.1,
        far_z: f32 = 100.0,
        base_w: i32 = 0,
        base_h: i32 = 0,
    };

    /// Traces + denoises the SSR reflection and returns the denoised
    /// half-res view (bilinear-upsampled by the composite). Returns an
    /// empty view when inactive so the caller falls back to the
    /// placeholder (bit-identical composite). Costs 2 draws.
    pub fn render(self: *SsrPass, args: RenderArgs) sg.View {
        if (self.trace_pipeline.id == 0 or self.blur_pipeline.id == 0) return .{};
        if (args.scene_view.id == 0) return .{};
        if (args.depth_view.id == 0) return .{};
        if (args.base_w <= 0 or args.base_h <= 0) return .{};

        const cfg = args.config.clamped();
        if (!pp.ssrPassActive(cfg)) return .{};

        self.resize(args.base_w, args.base_h);
        if (self.width != args.base_w or self.height != args.base_h) return .{};
        if (self.trace_image.id == 0) return .{};

        const size = pp.ssrTargetSize(args.base_w, args.base_h);

        var trace_pass = sg.Pass{
            .action = .{
                .colors = [_]sg.ColorAttachmentAction{
                    .{ .load_action = .DONTCARE },
                } ++ [_]sg.ColorAttachmentAction{.{}} ** 7,
            },
        };
        trace_pass.attachments.colors[0] = self.trace_att_view;
        sg.beginPass(trace_pass);

        sg.applyPipeline(self.trace_pipeline);
        var trace_bind = sg.Bindings{};
        trace_bind.vertex_buffers[0] = self.quad_vb;
        trace_bind.index_buffer = self.quad_ib;
        trace_bind.views[trace_shd.VIEW_scene_tex] = args.scene_view;
        trace_bind.views[trace_shd.VIEW_depth_tex] = args.depth_view;
        trace_bind.samplers[trace_shd.SMP_smp] = self.sampler;
        trace_bind.samplers[trace_shd.SMP_depth_smp] = self.depth_sampler;
        sg.applyBindings(trace_bind);

        const fw: f32 = @floatFromInt(args.base_w);
        const fh: f32 = @floatFromInt(args.base_h);
        const trace_params = trace_shd.FsParams{
            .view_proj = args.view_proj,
            .inv_view_proj = args.inv_view_proj,
            .camera_pos = .{ args.camera_pos.x, args.camera_pos.y, args.camera_pos.z, 0.0 },
            .resolution = .{ fw, fh, 1.0 / fw, 1.0 / fh },
            .ssr_params = .{ 1.0, cfg.ssr.intensity, pp.ssr.SSR_THICKNESS_DEFAULT, pp.ssr.SSR_MAX_DISTANCE_DEFAULT },
            .ssr_params2 = .{ @floatFromInt(cfg.ssr.steps), 0.0, 0.0, 0.0 },
        };
        sg.applyUniforms(trace_shd.UB_fs_params, sg.asRange(&trace_params));
        sg.draw(0, 6, 1);
        sg.endPass();

        var blur_pass = sg.Pass{
            .action = .{
                .colors = [_]sg.ColorAttachmentAction{
                    .{ .load_action = .DONTCARE },
                } ++ [_]sg.ColorAttachmentAction{.{}} ** 7,
            },
        };
        blur_pass.attachments.colors[0] = self.blur_att_view;
        sg.beginPass(blur_pass);

        sg.applyPipeline(self.blur_pipeline);
        var blur_bind = sg.Bindings{};
        blur_bind.vertex_buffers[0] = self.quad_vb;
        blur_bind.index_buffer = self.quad_ib;
        blur_bind.views[blur_shd.VIEW_ssr_tex] = self.trace_tex_view;
        blur_bind.views[blur_shd.VIEW_depth_tex] = args.depth_view;
        blur_bind.samplers[blur_shd.SMP_smp] = self.depth_sampler;
        sg.applyBindings(blur_bind);

        const hw: f32 = @floatFromInt(size.w);
        const hh: f32 = @floatFromInt(size.h);
        const blur_params = blur_shd.FsParams{
            .resolution = .{ hw, hh, 1.0 / hw, 1.0 / hh },
            .camera_params = .{ args.near_z, args.far_z, 0.0, 0.0 },
        };
        sg.applyUniforms(blur_shd.UB_fs_params, sg.asRange(&blur_params));
        sg.draw(0, 6, 1);
        sg.endPass();

        return self.blur_tex_view;
    }

    fn destroyTargets(self: *SsrPass) void {
        if (self.trace_att_view.id != 0) sg.destroyView(self.trace_att_view);
        if (self.trace_tex_view.id != 0) sg.destroyView(self.trace_tex_view);
        if (self.trace_image.id != 0) sg.destroyImage(self.trace_image);
        self.trace_image = .{};
        self.trace_att_view = .{};
        self.trace_tex_view = .{};
        if (self.blur_att_view.id != 0) sg.destroyView(self.blur_att_view);
        if (self.blur_tex_view.id != 0) sg.destroyView(self.blur_tex_view);
        if (self.blur_image.id != 0) sg.destroyImage(self.blur_image);
        self.blur_image = .{};
        self.blur_att_view = .{};
        self.blur_tex_view = .{};
        self.width = 0;
        self.height = 0;
    }

    pub fn deinit(self: *SsrPass) void {
        self.destroyTargets();
        sg.destroySampler(self.sampler);
        sg.destroySampler(self.depth_sampler);
        sg.destroyPipeline(self.trace_pipeline);
        sg.destroyPipeline(self.blur_pipeline);
        if (self.trace_shader.id != 0) sg.destroyShader(self.trace_shader);
        if (self.blur_shader.id != 0) sg.destroyShader(self.blur_shader);
        self.trace_shader = .{};
        self.blur_shader = .{};
        sg.destroyBuffer(self.quad_vb);
        sg.destroyBuffer(self.quad_ib);
    }
};
