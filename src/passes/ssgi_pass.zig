const sokol = @import("sokol");
const sg = sokol.gfx;
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const gather_shd = @import("ssgi_shader");
const blur_shd = @import("ssgi_blur_shader");
const pp = @import("../postprocess.zig");
const PostProcessOptions = pp.PostProcessOptions;

// Screen-space GI v2: half-resolution jittered-disk gather + bilateral
// (depth-aware) denoise.
//
// Per active frame (parent gates on postprocess.ssgiPassActive, which is
// the master enabled + intensity floor + half_res_pass mode; MSAA
// suppression clears ssgi_enabled so pass mode joins it for free):
//  1. Gather: fullscreen gather at half of the EFFECTIVE (render-scale)
//     size over the HDR scene color + full-res depth (world-space disk,
//     v1-identical footprint: `resolution` carries the full effective
//     size). Outputs normalized bleed * coverage with NO intensity baked
//     in; intensity scaling stays in the postprocess composite.
//  2. Blur: 3x3 bilateral over the gather output with the full-res scene
//     depth as the edge gate, so bleed never crosses silhouettes.
//  3. Composite: the parent feeds the denoised view into the fullscreen
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
//           .view_proj = ...,
//           .inv_view_proj = ...,
//           .camera_pos = ...,
//           .config = post,
//           .near_z = ...,
//           .far_z = ...,
//           .base_w = fx_w,
//           .base_h = fx_h,
//       });
//       self.stats.draw_calls += 2; // gather + denoise
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
// no SSGI VRAM.
//
// Temporal accumulation is explicitly OUT of scope (v2.1 item): the
// per-pixel jitter hands the denoise to the bilateral pass (and TAA
// downstream), same division of labor as v1.
pub const SsgiPass = struct {
    gather_image: sg.Image = .{},
    gather_att_view: sg.View = .{},
    gather_tex_view: sg.View = .{},

    blur_image: sg.Image = .{},
    blur_att_view: sg.View = .{},
    blur_tex_view: sg.View = .{},

    sampler: sg.Sampler = .{},
    depth_sampler: sg.Sampler = .{},

    gather_pipeline: sg.Pipeline = .{},
    blur_pipeline: sg.Pipeline = .{},
    gather_shader: sg.Shader = .{},
    blur_shader: sg.Shader = .{},

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

        return .{
            .sampler = smp,
            .depth_sampler = depth_smp,
            .gather_pipeline = gather_pip,
            .blur_pipeline = blur_pip,
            .gather_shader = gather_shd_handle,
            .blur_shader = blur_shd_handle,
            .quad_vb = vb,
            .quad_ib = ib,
        };
    }

    /// (Re)allocates both half-res targets from the full EFFECTIVE base
    /// size (ceil-half per axis, the ssgiTargetSize rule). No-op on
    /// degenerate or unchanged sizes. Context thread only.
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
    /// EFFECTIVE (render-scale) size, never the swapchain size.
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

    /// Gathers + denoises the SSGI bleed and returns the denoised
    /// half-res view (bilinear-upsampled by the composite). Returns an
    /// empty view when inactive so the caller falls back to the
    /// placeholder (bit-identical composite). Costs 2 draws.
    pub fn render(self: *SsgiPass, args: RenderArgs) sg.View {
        if (self.gather_pipeline.id == 0 or self.blur_pipeline.id == 0) return .{};
        if (args.scene_view.id == 0) return .{};
        if (args.depth_view.id == 0) return .{};
        if (args.base_w <= 0 or args.base_h <= 0) return .{};

        const cfg = args.config.clamped();
        if (!pp.ssgiPassActive(cfg)) return .{};

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

        // Stage 2: bilateral (depth-aware) denoise into the second
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
        blur_bind.views[blur_shd.VIEW_ssgi_tex] = self.gather_tex_view;
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
        self.width = 0;
        self.height = 0;
    }

    pub fn deinit(self: *SsgiPass) void {
        self.destroyTargets();
        sg.destroySampler(self.sampler);
        sg.destroySampler(self.depth_sampler);
        sg.destroyPipeline(self.gather_pipeline);
        sg.destroyPipeline(self.blur_pipeline);
        if (self.gather_shader.id != 0) sg.destroyShader(self.gather_shader);
        if (self.blur_shader.id != 0) sg.destroyShader(self.blur_shader);
        self.gather_shader = .{};
        self.blur_shader = .{};
        sg.destroyBuffer(self.quad_vb);
        sg.destroyBuffer(self.quad_ib);
    }
};
