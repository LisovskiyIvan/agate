const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const gather_shd = @import("ssgi_shader");
const blur_shd = @import("ssgi_blur_shader");
const temporal_shd = @import("ssgi_temporal_shader");
const pp = @import("../postprocess.zig");
const PostProcessOptions = pp.PostProcessOptions;

// Screen-space GI v2: half-resolution jittered-disk gather + temporal
// accumulation (v2.1, opt-in) + bilateral (depth-aware) denoise.
//
// Per active frame (parent gates on postprocess.ssgiPassActive, which is
// the master enabled + intensity floor + half_res_pass mode; MSAA
// suppression clears ssgi_enabled so pass mode joins it for free):
//  1. Gather: fullscreen gather at half of the EFFECTIVE (render-scale)
//     size over the HDR scene color + full-res depth (world-space disk,
//     v1-identical footprint: `resolution` carries the full effective
//     size). Outputs normalized bleed * coverage with NO intensity baked
//     in; intensity scaling stays in the postprocess composite.
//  2. Temporal (opt-in via ssgi_temporal_enabled, default OFF): reproject
//     the gather output against the velocity buffer (camera-depth fallback
//     without velocity) into the history ping-pong pair, with variance
//     clipping + depth/normal/reactive-marker rejection. History holds the
//     RAW accumulated signal (no intensity); OFF skips this stage and the
//     denoise samples the gather directly (bit-identical to pre-temporal).
//  3. Blur: 3x3 bilateral over the temporal (or raw gather) output with
//     the full-res scene depth as the edge gate, so bleed never crosses
//     silhouettes.
//  4. Composite: the parent feeds the denoised view into the fullscreen
//     postprocess shader, which samples it (bilinear upscale) and adds
//     bleed * intensity before fog, at the v1 stack position.
//
// Typical parent (scene/postfx_stack.zig) usage, before the fullscreen
// postprocess pass:
//
//   var ssgi_view: sg.View = .{};
//   if (postprocess.ssgiPassActive(post)) {
//       ssgi_view = self.ssgi_pass.render(.{
//           .scene_view = ...,
//           .depth_view = ...,
//           .velocity_view = ..., // frame velocity (render-scale), may be empty
//           .velocity_valid = ...,
//           .view_proj = ...,
//           .inv_view_proj = ...,
//           .prev_view_proj = ..., // camera-fallback reprojection
//           .camera_pos = ...,
//           .config = post,
//           .camera_cut = post.taa_camera_cut, // shared one-frame cut flag
//           .near_z = ...,
//           .far_z = ...,
//           .base_w = fx_w,
//           .base_h = fx_h,
//       });
//       self.stats.draw_calls += if (postprocess.ssgiTemporalActive(post)) 3 else 2;
//   }
//   self.postprocess_pass.setSsgiTexture(ssgi_view);
//
// Headless behavior mirrors GlowPass exactly: zero handles fail closed
// (render returns an empty view before any sg.* call), no sg.isvalid
// gates on the fail-closed path. The pass is uniform-only past its input
// binds (applyUniforms, never sg.updateBuffer), so it records nothing
// into gpu_upload_meter and renderReuse replays stay upload-free by
// construction. Target destruction is immediate (same as
// GlowPass.destroyTargets, context thread only). The targets allocate
// lazily on the first active render, so post-on with pass mode off holds
// no SSGI VRAM. The temporal pair additionally frees itself on the first
// pass-active render with the flag off, so disabling temporal releases
// its VRAM without waiting for a resize.
pub const SsgiPass = struct {
    gather_image: sg.Image = .{},
    gather_att_view: sg.View = .{},
    gather_tex_view: sg.View = .{},

    blur_image: sg.Image = .{},
    blur_att_view: sg.View = .{},
    blur_tex_view: sg.View = .{},

    temp_a_image: sg.Image = .{},
    temp_a_att_view: sg.View = .{},
    temp_a_tex_view: sg.View = .{},
    temp_b_image: sg.Image = .{},
    temp_b_att_view: sg.View = .{},
    temp_b_tex_view: sg.View = .{},

    /// Temporal history cursor: read/write slots alternate by parity (CPU
    /// mirror: ssgiTemporalReadIndex/WriteIndex). Incremented per
    /// temporal-active render, including reset frames (the reset output
    /// primes the just-written slot for the next frame).
    temp_frame: u64 = 0,
    /// False until the first temporal-active render completes; the parent
    /// inactivity path clears it, so a mode toggle or disable/enable
    /// cycle always replays the reset frame.
    temp_has_history: bool = false,
    /// Last frame's temporal enable state; a false->true edge is the
    /// toggle reset (mirrors the TAA toggled_on condition).
    temp_enabled_prev: bool = false,

    sampler: sg.Sampler = .{},
    depth_sampler: sg.Sampler = .{},

    gather_pipeline: sg.Pipeline = .{},
    blur_pipeline: sg.Pipeline = .{},
    temporal_pipeline: sg.Pipeline = .{},
    gather_shader: sg.Shader = .{},
    blur_shader: sg.Shader = .{},
    temporal_shader: sg.Shader = .{},

    quad_vb: sg.Buffer = .{},
    quad_ib: sg.Buffer = .{},

    /// Full EFFECTIVE (render-scale) base size the half-res targets were
    /// derived from. 0 until the first active render (lazy allocation, so
    /// resizeAll only touches a sized pass).
    width: i32 = 0,
    height: i32 = 0,

    /// Sole-contract render-target pixel format (always RGBA16F: HDR
    /// bleed; the backend capability is validated once at Scene startup).
    /// Pure (no sg calls, headless-safe).
    pub fn ssgiPixelFormat() sg.PixelFormat {
        return .RGBA16F;
    }

    /// Bytes per pixel of the pass targets (always 8: RGBA16F). Pure
    /// (no sg calls, headless-safe).
    pub fn ssgiBytesPerPixel() usize {
        return if (ssgiPixelFormat() == .RGBA16F) 8 else 4;
    }

    /// VRAM census helper (pure byte math, no GPU calls, headless-safe):
    /// two half-res targets (gather + denoised) at `bytes_per_pixel`.
    /// Single source of truth for profiler/snapshot.zig alongside
    /// postprocess.ssgiTargetBytes (same formula, kept as a method for
    /// the GlowPass.targetBytes call-site shape).
    pub fn targetBytes(base_w: i32, base_h: i32, bytes_per_pixel: usize) usize {
        return pp.ssgiTargetBytes(base_w, base_h, bytes_per_pixel);
    }

    /// VRAM census helper for the temporal history pair (pure byte math):
    /// two half-res RGBA16F targets at `bytes_per_pixel`. Counted by the
    /// census only while temporalSized().
    pub fn temporalTargetBytes(base_w: i32, base_h: i32, bytes_per_pixel: usize) usize {
        return pp.ssgiTemporalTargetBytes(base_w, base_h, bytes_per_pixel);
    }

    /// True while the history pair is allocated. Pure handle check (no sg
    /// calls, headless-safe); the census gate.
    pub fn temporalSized(self: *const SsgiPass) bool {
        return self.temp_a_image.id != 0;
    }

    pub fn init() SsgiPass {
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

        const ssgi_fmt = ssgiPixelFormat();

        const gather_shd_handle = sg.makeShader(gather_shd.ssgiShaderDesc(sg.queryBackend()));
        var gather_desc = sg.PipelineDesc{
            .shader = gather_shd_handle,
            .index_type = .UINT16,
            .depth = .{
                .pixel_format = .NONE,
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = 1,
        };
        gather_desc.colors[0].pixel_format = ssgi_fmt;
        gather_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        gather_desc.layout.attrs[gather_shd.ATTR_ssgi_position] = .{ .format = .FLOAT2, .offset = 0 };
        gather_desc.layout.attrs[gather_shd.ATTR_ssgi_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const gather_pip = sg.makePipeline(gather_desc);

        const blur_shd_handle = sg.makeShader(blur_shd.ssgiBlurShaderDesc(sg.queryBackend()));
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
        blur_desc.colors[0].pixel_format = ssgi_fmt;
        blur_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        blur_desc.layout.attrs[blur_shd.ATTR_ssgi_blur_position] = .{ .format = .FLOAT2, .offset = 0 };
        blur_desc.layout.attrs[blur_shd.ATTR_ssgi_blur_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const blur_pip = sg.makePipeline(blur_desc);

        const temporal_shd_handle = sg.makeShader(temporal_shd.ssgiTemporalShaderDesc(sg.queryBackend()));
        var temporal_desc = sg.PipelineDesc{
            .shader = temporal_shd_handle,
            .index_type = .UINT16,
            .depth = .{
                .pixel_format = .NONE,
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = 1,
        };
        temporal_desc.colors[0].pixel_format = ssgi_fmt;
        temporal_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        temporal_desc.layout.attrs[temporal_shd.ATTR_ssgi_temporal_position] = .{ .format = .FLOAT2, .offset = 0 };
        temporal_desc.layout.attrs[temporal_shd.ATTR_ssgi_temporal_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const temporal_pip = sg.makePipeline(temporal_desc);

        return .{
            .sampler = smp,
            .depth_sampler = depth_smp,
            .gather_pipeline = gather_pip,
            .blur_pipeline = blur_pip,
            .temporal_pipeline = temporal_pip,
            .gather_shader = gather_shd_handle,
            .blur_shader = blur_shd_handle,
            .temporal_shader = temporal_shd_handle,
            .quad_vb = vb,
            .quad_ib = ib,
        };
    }

    /// (Re)allocates both half-res targets from the full EFFECTIVE base
    /// size (ceil-half per axis, the ssgiTargetSize rule). No-op on
    /// degenerate or unchanged sizes. Destroys the temporal pair as well:
    /// a resize is a history reset (fresh targets carry no valid history).
    /// Context thread only.
    pub fn resize(self: *SsgiPass, base_w: i32, base_h: i32) void {
        if (base_w <= 0 or base_h <= 0) return;
        if (!sg.isvalid()) return;
        if (self.width == base_w and self.height == base_h) return;

        self.destroyTargets();

        const size = pp.ssgiTargetSize(base_w, base_h);
        const ssgi_fmt = ssgiPixelFormat();

        self.gather_image = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = size.w,
            .height = size.h,
            .pixel_format = ssgi_fmt,
            .sample_count = 1,
        });
        if (sg.queryImageState(self.gather_image) != .VALID) {
            self.destroyTargets();
            return;
        }
        self.gather_att_view = sg.makeView(.{
            .color_attachment = .{ .image = self.gather_image },
        });
        if (sg.queryViewState(self.gather_att_view) != .VALID) {
            self.destroyTargets();
            return;
        }
        self.gather_tex_view = sg.makeView(.{
            .texture = .{ .image = self.gather_image },
        });
        if (sg.queryViewState(self.gather_tex_view) != .VALID) {
            self.destroyTargets();
            return;
        }

        self.blur_image = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = size.w,
            .height = size.h,
            .pixel_format = ssgi_fmt,
            .sample_count = 1,
        });
        if (sg.queryImageState(self.blur_image) != .VALID) {
            self.destroyTargets();
            return;
        }
        self.blur_att_view = sg.makeView(.{
            .color_attachment = .{ .image = self.blur_image },
        });
        if (sg.queryViewState(self.blur_att_view) != .VALID) {
            self.destroyTargets();
            return;
        }
        self.blur_tex_view = sg.makeView(.{
            .texture = .{ .image = self.blur_image },
        });
        if (sg.queryViewState(self.blur_tex_view) != .VALID) {
            self.destroyTargets();
            return;
        }

        self.width = base_w;
        self.height = base_h;
    }

    /// Per-frame gather inputs. All views are render-owned snapshots (the
    /// HDR resolve + scene depth); matrices and camera come from the
    /// frame; config carries the SSGI knobs (clamped inside render, so
    /// callers pass the raw frame config); base_w/h is the full
    /// EFFECTIVE (render-scale) size, never the swapchain size. The
    /// temporal stage additionally consumes the frame velocity view
    /// (render-scale, may be empty: the shader falls back to camera-depth
    /// reprojection), the previous view-projection for that fallback, and
    /// the shared one-frame camera-cut flag (taa_camera_cut doubles as
    /// the SSGI history reset).
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
        near_z: f32 = 0.1,
        far_z: f32 = 100.0,
        base_w: i32 = 0,
        base_h: i32 = 0,
    };

    /// Marks the temporal history invalid without touching GPU handles
    /// (pure bool writes, headless-safe). Called whenever the pass does
    /// not run (inline mode, disabled, intensity floor): a mode toggle or
    /// re-enable always replays the reset frame. Missing-handle/size
    /// early returns leave even this untouched (fail-closed shape).
    fn noteTemporalInactive(self: *SsgiPass) void {
        self.temp_has_history = false;
        self.temp_enabled_prev = false;
    }

    fn tempReadTexView(self: *const SsgiPass) sg.View {
        return if (pp.ssgi.ssgiTemporalReadIndex(self.temp_frame) == 0) self.temp_a_tex_view else self.temp_b_tex_view;
    }

    fn tempWriteAttView(self: *const SsgiPass) sg.View {
        return if (pp.ssgi.ssgiTemporalWriteIndex(self.temp_frame) == 0) self.temp_a_att_view else self.temp_b_att_view;
    }

    fn tempWriteTexView(self: *const SsgiPass) sg.View {
        return if (pp.ssgi.ssgiTemporalWriteIndex(self.temp_frame) == 0) self.temp_a_tex_view else self.temp_b_tex_view;
    }

    /// Allocates the history ping-pong pair at the pass's half-res size.
    /// Returns true when (re)created — the caller's reset trigger for the
    /// frame. No-op (false) when already sized. Context thread only; fails
    /// closed (false) without publishing partial state.
    fn ensureTemporal(self: *SsgiPass, base_w: i32, base_h: i32) bool {
        if (base_w <= 0 or base_h <= 0) return false;
        if (!sg.isvalid()) return false;
        if (self.temporalSized()) return false;
        const size = pp.ssgiTargetSize(base_w, base_h);
        const ssgi_fmt = ssgiPixelFormat();
        var imgs: [2]sg.Image = .{ .{}, .{} };
        var atts: [2]sg.View = .{ .{}, .{} };
        var texs: [2]sg.View = .{ .{}, .{} };
        for (0..2) |i| {
            imgs[i] = sg.makeImage(.{
                .usage = .{ .color_attachment = true },
                .width = size.w,
                .height = size.h,
                .pixel_format = ssgi_fmt,
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

    /// Gathers (+ optionally temporally accumulates) + denoises the SSGI
    /// bleed and returns the denoised half-res view (bilinear-upsampled by
    /// the composite). Returns an empty view when inactive so the caller
    /// falls back to the placeholder (bit-identical composite). Costs 2
    /// draws, 3 with temporal accumulation active.
    pub fn render(self: *SsgiPass, args: RenderArgs) sg.View {
        if (self.gather_pipeline.id == 0 or self.blur_pipeline.id == 0) return .{};
        if (args.scene_view.id == 0) return .{};
        if (args.depth_view.id == 0) return .{};
        if (args.base_w <= 0 or args.base_h <= 0) return .{};

        const cfg = args.config.clamped();
        if (!pp.ssgiPassActive(cfg)) {
            self.noteTemporalInactive();
            return .{};
        }

        self.resize(args.base_w, args.base_h);
        if (self.width != args.base_w or self.height != args.base_h) return .{};
        if (self.gather_image.id == 0) return .{};

        const size = pp.ssgiTargetSize(args.base_w, args.base_h);

        // Stage 1: jittered-disk gather into the half-res target. The
        // `resolution` lane carries the FULL effective size so the disk
        // footprint matches the v1 inline gather exactly.
        var gather_pass = sg.Pass{
            .action = .{
                .colors = [_]sg.ColorAttachmentAction{
                    .{ .load_action = .DONTCARE },
                } ++ [_]sg.ColorAttachmentAction{.{}} ** 7,
            },
        };
        gather_pass.attachments.colors[0] = self.gather_att_view;
        sg.beginPass(gather_pass);

        sg.applyPipeline(self.gather_pipeline);
        var gather_bind = sg.Bindings{};
        gather_bind.vertex_buffers[0] = self.quad_vb;
        gather_bind.index_buffer = self.quad_ib;
        gather_bind.views[gather_shd.VIEW_scene_tex] = args.scene_view;
        gather_bind.views[gather_shd.VIEW_depth_tex] = args.depth_view;
        gather_bind.samplers[gather_shd.SMP_smp] = self.sampler;
        gather_bind.samplers[gather_shd.SMP_depth_smp] = self.depth_sampler;
        sg.applyBindings(gather_bind);

        const fw: f32 = @floatFromInt(args.base_w);
        const fh: f32 = @floatFromInt(args.base_h);
        // The gather lane shape mirrors the composite ssgi_params
        // (enabled, intensity, radius, steps), but packed directly: this
        // frame already passed ssgiPassActive, while ssgiParams() gates
        // on INLINE mode (it zeros the composite gather branch in pass
        // mode so the effect never double-adds).
        const gather_params = gather_shd.FsParams{
            .inv_view_proj = args.inv_view_proj,
            .view_proj = args.view_proj,
            .camera_pos = .{ args.camera_pos.x, args.camera_pos.y, args.camera_pos.z, 0.0 },
            .resolution = .{ fw, fh, 1.0 / fw, 1.0 / fh },
            .ssgi_params = .{ 1.0, cfg.ssgi_intensity, cfg.ssgi_radius, @floatFromInt(cfg.ssgi_steps) },
        };
        sg.applyUniforms(gather_shd.UB_fs_params, sg.asRange(&gather_params));
        sg.draw(0, 6, 1);
        sg.endPass();

        // Stage 2 (opt-in): temporal accumulation of the gather output.
        // OFF (default): the denoise samples the gather target directly —
        // the exact pre-temporal frame. ON: reproject + variance-clip +
        // reject into the history write slot; the denoise samples that.
        var denoise_src = self.gather_tex_view;
        if (pp.ssgiTemporalActive(cfg)) {
            if (self.temporal_pipeline.id != 0) {
                const recreated = self.ensureTemporal(args.base_w, args.base_h);
                if (self.temporalSized()) {
                    const reset = pp.ssgiTemporalShouldReset(.{
                        .first_frame = !self.temp_has_history,
                        .toggled_on = !self.temp_enabled_prev,
                        .resized = recreated,
                        .camera_cut = args.camera_cut,
                        .explicit_reset = false,
                    });
                    const read_view = self.tempReadTexView();
                    const history_valid = !reset and read_view.id != 0;
                    const velocity_live = args.velocity_valid and args.velocity_view.id != 0;

                    var temporal_pass = sg.Pass{
                        .action = .{
                            .colors = [_]sg.ColorAttachmentAction{
                                .{ .load_action = .DONTCARE },
                            } ++ [_]sg.ColorAttachmentAction{.{}} ** 7,
                        },
                    };
                    temporal_pass.attachments.colors[0] = self.tempWriteAttView();
                    sg.beginPass(temporal_pass);

                    sg.applyPipeline(self.temporal_pipeline);
                    var temporal_bind = sg.Bindings{};
                    temporal_bind.vertex_buffers[0] = self.quad_vb;
                    temporal_bind.index_buffer = self.quad_ib;
                    temporal_bind.views[temporal_shd.VIEW_gather_tex] = self.gather_tex_view;
                    temporal_bind.views[temporal_shd.VIEW_history_tex] = if (history_valid) read_view else self.gather_tex_view;
                    temporal_bind.views[temporal_shd.VIEW_depth_tex] = args.depth_view;
                    // Format-correct placeholder (RGBA16F, never sampled:
                    // the shader gates velocity on temporal_params.y, set
                    // consistently with this bind).
                    temporal_bind.views[temporal_shd.VIEW_velocity_tex] = if (velocity_live) args.velocity_view else self.gather_tex_view;
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
                        .camera_params = .{ args.near_z, args.far_z, cfg.ssgi_radius, 0.0 },
                        .temporal_params = .{
                            if (history_valid) 1.0 else 0.0,
                            if (velocity_live) 1.0 else 0.0,
                            pp.ssgiTemporalBlend(cfg),
                            pp.ssgi.SSGI_TEMPORAL_GAMMA,
                        },
                    };
                    sg.applyUniforms(temporal_shd.UB_fs_params, sg.asRange(&temporal_params));
                    sg.draw(0, 6, 1);
                    sg.endPass();

                    // Capture the just-written slot BEFORE advancing the
                    // cursor: the parity helpers flip with temp_frame.
                    denoise_src = self.tempWriteTexView();
                    self.temp_frame += 1;
                    self.temp_has_history = true;
                    self.temp_enabled_prev = true;
                }
            }
        } else {
            // Temporal off: release the pair so disabled temporal holds no
            // VRAM (one-time destroy on the toggle-off frame; the census
            // gate temporalSized() goes quiet with it).
            self.destroyTemporal();
            self.noteTemporalInactive();
        }

        // Stage 3: bilateral (depth-aware) denoise into the second
        // half-res target. Intensity stays in the composite.
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
        blur_bind.views[blur_shd.VIEW_ssgi_tex] = denoise_src;
        blur_bind.views[blur_shd.VIEW_depth_tex] = args.depth_view;
        blur_bind.samplers[blur_shd.SMP_smp] = self.depth_sampler;
        sg.applyBindings(blur_bind);

        const hw: f32 = @floatFromInt(size.w);
        const hh: f32 = @floatFromInt(size.h);
        const blur_params = blur_shd.FsParams{
            .resolution = .{ hw, hh, 1.0 / hw, 1.0 / hh },
            .camera_params = .{ args.near_z, args.far_z, cfg.ssgi_radius, 0.0 },
        };
        sg.applyUniforms(blur_shd.UB_fs_params, sg.asRange(&blur_params));
        sg.draw(0, 6, 1);
        sg.endPass();

        return self.blur_tex_view;
    }

    fn destroyTemporal(self: *SsgiPass) void {
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

    fn destroyTargets(self: *SsgiPass) void {
        if (self.gather_att_view.id != 0) sg.destroyView(self.gather_att_view);
        if (self.gather_tex_view.id != 0) sg.destroyView(self.gather_tex_view);
        if (self.gather_image.id != 0) sg.destroyImage(self.gather_image);
        self.gather_image = .{};
        self.gather_att_view = .{};
        self.gather_tex_view = .{};
        if (self.blur_att_view.id != 0) sg.destroyView(self.blur_att_view);
        if (self.blur_tex_view.id != 0) sg.destroyView(self.blur_tex_view);
        if (self.blur_image.id != 0) sg.destroyImage(self.blur_image);
        self.blur_image = .{};
        self.blur_att_view = .{};
        self.blur_tex_view = .{};
        self.destroyTemporal();
        self.temp_frame = 0;
        self.temp_has_history = false;
        self.temp_enabled_prev = false;
        self.width = 0;
        self.height = 0;
    }

    pub fn deinit(self: *SsgiPass) void {
        self.destroyTargets();
        sg.destroySampler(self.sampler);
        sg.destroySampler(self.depth_sampler);
        sg.destroyPipeline(self.gather_pipeline);
        sg.destroyPipeline(self.blur_pipeline);
        sg.destroyPipeline(self.temporal_pipeline);
        if (self.gather_shader.id != 0) sg.destroyShader(self.gather_shader);
        if (self.blur_shader.id != 0) sg.destroyShader(self.blur_shader);
        if (self.temporal_shader.id != 0) sg.destroyShader(self.temporal_shader);
        self.gather_shader = .{};
        self.blur_shader = .{};
        self.temporal_shader = .{};
        sg.destroyBuffer(self.quad_vb);
        sg.destroyBuffer(self.quad_ib);
    }
};
