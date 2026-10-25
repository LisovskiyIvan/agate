const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const trace_shd = @import("ssr_shader");
const blur_shd = @import("ssr_blur_shader");
const temporal_shd = @import("ssr_temporal_shader");
const pp = @import("../postprocess.zig");
const PostProcessOptions = pp.PostProcessOptions;

// Screen-space reflections v2: half-resolution trace + temporal
// accumulation (v2.1, opt-in) + bilateral (depth-aware) denoise. The final
// depth-aware upscale lives in the fullscreen composite (sampleSSRPass 2x2
// gather), so this pass stays at half-res with no full-res intermediate.
//
// Per active frame (parent gates on postprocess.ssrPassActive, which is
// the master enabled + intensity floor + half_res_pass mode; MSAA
// suppression clears ssr.enabled so pass mode joins it for free):
//  1. Trace: half-res port of the v1 inline raymarch (floor gate,
//     clip-space march, thickness/max-distance, edge/distance/fresnel
//     weights). Outputs (reflection rgb, blend weight) with NO intensity
//     baked in; intensity scaling stays in the postprocess composite.
//  2. Temporal (opt-in via ssr.temporal.enabled, default OFF): reprojects
//     the trace output against the velocity buffer (camera-depth fallback
//     without velocity) into the history ping-pong pair, with 4-channel
//     variance clipping + depth/normal/reactive-marker rejection. History
//     holds the accumulated signal (no intensity); OFF skips this stage and
//     denoise samples the trace directly (bit-identical to pre-temporal).
//  3. Blur: 3x3 bilateral over the temporal (or raw trace) output with the
//     full-res scene depth as the edge gate, so neighboring denoised texels
//     never cross silhouettes.
//  4. Composite: the parent feeds the denoised view into the fullscreen
//     postprocess shader, which reconstructs full res with a depth-aware
//     2x2 gather (sampleSSRPass in postprocess.glsl) and mixes by
//     intensity * weight at the v1 stack position.
pub const SsrPass = struct {
    trace_image: sg.Image = .{},
    trace_att_view: sg.View = .{},
    trace_tex_view: sg.View = .{},

    blur_image: sg.Image = .{},
    blur_att_view: sg.View = .{},
    blur_tex_view: sg.View = .{},

    temp_a_image: sg.Image = .{},
    temp_a_att_view: sg.View = .{},
    temp_a_tex_view: sg.View = .{},
    temp_b_image: sg.Image = .{},
    temp_b_att_view: sg.View = .{},
    temp_b_tex_view: sg.View = .{},

    /// Temporal history cursor: read/write slots alternate by parity.
    temp_frame: u64 = 0,
    temp_has_history: bool = false,
    temp_enabled_prev: bool = false,

    sampler: sg.Sampler = .{},
    depth_sampler: sg.Sampler = .{},

    trace_pipeline: sg.Pipeline = .{},
    blur_pipeline: sg.Pipeline = .{},
    temporal_pipeline: sg.Pipeline = .{},
    trace_shader: sg.Shader = .{},
    blur_shader: sg.Shader = .{},
    temporal_shader: sg.Shader = .{},

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

    /// VRAM census helper for the temporal history pair: two half-res
    /// RGBA16F targets at `bytes_per_pixel`. Counted only while temporalSized().
    pub fn temporalTargetBytes(base_w: i32, base_h: i32, bytes_per_pixel: usize) usize {
        return pp.ssrTemporalTargetBytes(base_w, base_h, bytes_per_pixel);
    }

    pub fn temporalSized(self: *const SsrPass) bool {
        return self.temp_a_image.id != 0;
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

        const temporal_shd_handle = sg.makeShader(temporal_shd.ssrTemporalShaderDesc(sg.queryBackend()));
        var temporal_desc = sg.PipelineDesc{
            .shader = temporal_shd_handle,
            .index_type = .UINT16,
            .depth = .{ .pixel_format = .NONE, .compare = .ALWAYS, .write_enabled = false },
            .cull_mode = .NONE,
            .sample_count = 1,
        };
        temporal_desc.colors[0].pixel_format = fmt;
        temporal_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        temporal_desc.layout.attrs[temporal_shd.ATTR_ssr_temporal_position] = .{ .format = .FLOAT2, .offset = 0 };
        temporal_desc.layout.attrs[temporal_shd.ATTR_ssr_temporal_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const temporal_pip = sg.makePipeline(temporal_desc);

        return .{
            .sampler = smp,
            .depth_sampler = depth_smp,
            .trace_pipeline = trace_pip,
            .blur_pipeline = blur_pip,
            .temporal_pipeline = temporal_pip,
            .trace_shader = trace_shd_handle,
            .blur_shader = blur_shd_handle,
            .temporal_shader = temporal_shd_handle,
            .quad_vb = vb,
            .quad_ib = ib,
        };
    }

    /// (Re)allocates both half-res targets from the full EFFECTIVE base
    /// size (ceil-half per axis, the ssrTargetSize rule). Destroys the
    /// temporal pair as well (size change resets history).
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
        velocity_view: sg.View = .{},
        velocity_valid: bool = false,
        view_proj: Mat4 = Mat4.identity,
        inv_view_proj: Mat4 = Mat4.identity,
        prev_view_proj: Mat4 = Mat4.identity,
        camera_pos: Vec3 = Vec3.zero,
        config: PostProcessOptions = .{},
        camera_cut: bool = false,
        explicit_reset: bool = false,
        multi_camera: bool = false,
        near_z: f32 = 0.1,
        far_z: f32 = 100.0,
        base_w: i32 = 0,
        base_h: i32 = 0,
    };

    pub const RenderResult = struct {
        view: sg.View = .{},
        temporal_executed: bool = false,
    };

    pub fn resetTemporalHistory(self: *SsrPass) void {
        self.destroyTemporal();
        self.temp_frame = 0;
        self.temp_has_history = false;
        self.temp_enabled_prev = false;
    }

    fn noteTemporalInactive(self: *SsrPass) void {
        self.temp_has_history = false;
        self.temp_enabled_prev = false;
    }

    fn tempReadTexView(self: *const SsrPass) sg.View {
        return if (pp.ssr.ssrTemporalReadIndex(self.temp_frame) == 0) self.temp_a_tex_view else self.temp_b_tex_view;
    }

    fn tempWriteAttView(self: *const SsrPass) sg.View {
        return if (pp.ssr.ssrTemporalWriteIndex(self.temp_frame) == 0) self.temp_a_att_view else self.temp_b_att_view;
    }

    fn tempWriteTexView(self: *const SsrPass) sg.View {
        return if (pp.ssr.ssrTemporalWriteIndex(self.temp_frame) == 0) self.temp_a_tex_view else self.temp_b_tex_view;
    }

    fn ensureTemporal(self: *SsrPass, base_w: i32, base_h: i32) bool {
        if (base_w <= 0 or base_h <= 0) return false;
        if (!sg.isvalid()) return false;
        if (self.temporalSized()) return false;
        const size = pp.ssrTargetSize(base_w, base_h);
        const fmt = ssrPixelFormat();
        var imgs: [2]sg.Image = .{ .{}, .{} };
        var atts: [2]sg.View = .{ .{}, .{} };
        var texs: [2]sg.View = .{ .{}, .{} };
        for (0..2) |i| {
            imgs[i] = sg.makeImage(.{
                .usage = .{ .color_attachment = true },
                .width = size.w,
                .height = size.h,
                .pixel_format = fmt,
                .sample_count = 1,
            });
            if (sg.queryImageState(imgs[i]) != .VALID) {
                if (imgs[i].id != 0) sg.destroyImage(imgs[i]);
                if (i == 1) {
                    sg.destroyView(atts[0]);
                    sg.destroyView(texs[0]);
                    sg.destroyImage(imgs[0]);
                }
                return false;
            }
            atts[i] = sg.makeView(.{ .color_attachment = .{ .image = imgs[i] } });
            texs[i] = sg.makeView(.{ .texture = .{ .image = imgs[i] } });
            if (sg.queryViewState(atts[i]) != .VALID or sg.queryViewState(texs[i]) != .VALID) {
                if (texs[i].id != 0) sg.destroyView(texs[i]);
                if (atts[i].id != 0) sg.destroyView(atts[i]);
                sg.destroyImage(imgs[i]);
                if (i == 1) {
                    sg.destroyView(atts[0]);
                    sg.destroyView(texs[0]);
                    sg.destroyImage(imgs[0]);
                }
                return false;
            }
        }
        self.temp_a_image = imgs[0];
        self.temp_a_att_view = atts[0];
        self.temp_a_tex_view = texs[0];
        self.temp_b_image = imgs[1];
        self.temp_b_att_view = atts[1];
        self.temp_b_tex_view = texs[1];
        return true;
    }

    fn destroyTemporal(self: *SsrPass) void {
        if (self.temp_a_att_view.id != 0) sg.destroyView(self.temp_a_att_view);
        if (self.temp_a_tex_view.id != 0) sg.destroyView(self.temp_a_tex_view);
        if (self.temp_a_image.id != 0) sg.destroyImage(self.temp_a_image);
        self.temp_a_image = .{};
        self.temp_a_att_view = .{};
        self.temp_a_tex_view = .{};
        if (self.temp_b_att_view.id != 0) sg.destroyView(self.temp_b_att_view);
        if (self.temp_b_tex_view.id != 0) sg.destroyView(self.temp_b_tex_view);
        if (self.temp_b_image.id != 0) sg.destroyImage(self.temp_b_image);
        self.temp_b_image = .{};
        self.temp_b_att_view = .{};
        self.temp_b_tex_view = .{};
    }

    /// Traces + optionally temporally accumulates + denoises the SSR
    /// reflection. Returns the denoised half-res view.
    pub fn render(self: *SsrPass, args: RenderArgs) RenderResult {
        if (self.trace_pipeline.id == 0 or self.blur_pipeline.id == 0) return .{};
        if (args.scene_view.id == 0) return .{};
        if (args.depth_view.id == 0) return .{};
        if (args.base_w <= 0 or args.base_h <= 0) return .{};

        const cfg = args.config.clamped();
        if (!pp.ssrPassActive(cfg)) {
            self.noteTemporalInactive();
            return .{};
        }

        self.resize(args.base_w, args.base_h);
        if (self.width != args.base_w or self.height != args.base_h) return .{};
        if (self.trace_image.id == 0) return .{};

        const size = pp.ssrTargetSize(args.base_w, args.base_h);
        var result = RenderResult{};

        // Stage 1: Half-res trace
        var trace_pass = sg.Pass{
            .action = .{
                .colors = [_]sg.ColorAttachmentAction{
                    .{ .load_action = .DONTCARE },
                } ++ @as([7]sg.ColorAttachmentAction, @splat(.{})),
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

        var denoise_src: sg.View = self.trace_tex_view;

        // Stage 2: Temporal accumulation (v2.1, opt-in)
        if (pp.ssrTemporalActive(cfg)) {
            if (args.multi_camera) {
                self.noteTemporalInactive();
            } else {
                const resized = self.ensureTemporal(args.base_w, args.base_h);
                if (self.temporal_pipeline.id != 0 and self.temporalSized()) {
                    const toggled_on = !self.temp_enabled_prev;
                    const should_reset = pp.ssr.ssrTemporalShouldReset(.{
                        .first_frame = !self.temp_has_history,
                        .toggled_on = toggled_on,
                        .resized = resized,
                        .camera_cut = args.camera_cut,
                        .explicit_reset = args.explicit_reset,
                    });
                    const history_valid = !should_reset and self.temp_has_history;
                    const read_view = self.tempReadTexView();
                    const velocity_live = args.velocity_valid and args.velocity_view.id != 0;

                    var temporal_pass = sg.Pass{
                        .action = .{
                            .colors = [_]sg.ColorAttachmentAction{
                                .{ .load_action = .DONTCARE },
                            } ++ @as([7]sg.ColorAttachmentAction, @splat(.{})),
                        },
                    };
                    temporal_pass.attachments.colors[0] = self.tempWriteAttView();
                    sg.beginPass(temporal_pass);

                    sg.applyPipeline(self.temporal_pipeline);
                    var temporal_bind = sg.Bindings{};
                    temporal_bind.vertex_buffers[0] = self.quad_vb;
                    temporal_bind.index_buffer = self.quad_ib;
                    temporal_bind.views[temporal_shd.VIEW_trace_tex] = self.trace_tex_view;
                    temporal_bind.views[temporal_shd.VIEW_history_tex] = if (history_valid) read_view else self.trace_tex_view;
                    temporal_bind.views[temporal_shd.VIEW_depth_tex] = args.depth_view;
                    temporal_bind.views[temporal_shd.VIEW_velocity_tex] = if (velocity_live) args.velocity_view else self.trace_tex_view;
                    temporal_bind.samplers[temporal_shd.SMP_smp] = self.sampler;
                    temporal_bind.samplers[temporal_shd.SMP_depth_smp] = self.depth_sampler;
                    temporal_bind.samplers[temporal_shd.SMP_velocity_smp] = self.depth_sampler;
                    sg.applyBindings(temporal_bind);

                    const hw: f32 = @floatFromInt(size.w);
                    const hh: f32 = @floatFromInt(size.h);
                    const temporal_params = temporal_shd.FsParams{
                        .inv_view_proj = args.inv_view_proj,
                        .view_proj = args.view_proj,
                        .prev_view_proj = args.prev_view_proj,
                        .resolution = .{ hw, hh, 1.0 / hw, 1.0 / hh },
                        .full_resolution = .{ fw, fh, 1.0 / fw, 1.0 / fh },
                        .camera_params = .{ args.near_z, args.far_z, 0.0, 0.0 },
                        .temporal_params = .{
                            if (history_valid) 1.0 else 0.0,
                            if (velocity_live) 1.0 else 0.0,
                            pp.ssr.ssrTemporalBlend(cfg),
                            pp.ssr.SSR_TEMPORAL_GAMMA,
                        },
                    };
                    sg.applyUniforms(temporal_shd.UB_fs_params, sg.asRange(&temporal_params));
                    sg.draw(0, 6, 1);
                    sg.endPass();

                    denoise_src = self.tempWriteTexView();
                    self.temp_frame += 1;
                    self.temp_has_history = true;
                    self.temp_enabled_prev = true;
                    result.temporal_executed = true;
                }
            }
        } else {
            self.destroyTemporal();
            self.noteTemporalInactive();
        }

        // Stage 3: Bilateral denoise
        var blur_pass = sg.Pass{
            .action = .{
                .colors = [_]sg.ColorAttachmentAction{
                    .{ .load_action = .DONTCARE },
                } ++ @as([7]sg.ColorAttachmentAction, @splat(.{})),
            },
        };
        blur_pass.attachments.colors[0] = self.blur_att_view;
        sg.beginPass(blur_pass);

        sg.applyPipeline(self.blur_pipeline);
        var blur_bind = sg.Bindings{};
        blur_bind.vertex_buffers[0] = self.quad_vb;
        blur_bind.index_buffer = self.quad_ib;
        blur_bind.views[blur_shd.VIEW_ssr_tex] = denoise_src;
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

        result.view = self.blur_tex_view;
        return result;
    }

    fn destroyTargets(self: *SsrPass) void {
        self.destroyTemporal();
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
        sg.destroyPipeline(self.temporal_pipeline);
        if (self.trace_shader.id != 0) sg.destroyShader(self.trace_shader);
        if (self.blur_shader.id != 0) sg.destroyShader(self.blur_shader);
        if (self.temporal_shader.id != 0) sg.destroyShader(self.temporal_shader);
        self.trace_shader = .{};
        self.blur_shader = .{};
        self.temporal_shader = .{};
        sg.destroyBuffer(self.quad_vb);
        sg.destroyBuffer(self.quad_ib);
    }
};
