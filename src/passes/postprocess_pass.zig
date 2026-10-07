const std = @import("std");
const builtin = @import("builtin");
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
    // Nearest sampler for the velocity buffer (motion vectors + geometry
    // mask): bilinear filtering would smear vectors and the alpha gate
    // across silhouettes. Mirrors depth_sampler.
    velocity_sampler: sg.Sampler = .{},
    default_zero_image: sg.Image = .{},
    default_zero_view: sg.View = .{},
    // Dedicated LUT sampler: bilinear inside the strip with LOD pinned to
    // the base level, so a mipped LUT upload can never smear cube slices
    // through box-filtered mips.
    lut_sampler: sg.Sampler = .{},
    postprocess_pipeline: sg.Pipeline = .{},
    postprocess_quad_vb: sg.Buffer = .{},
    postprocess_quad_ib: sg.Buffer = .{},
    // Optional BloomPass result (pyramid glow, half resolution). Set via
    // setBloomTexture(); empty by default, in which case the composite
    // skips the bloom block and this binds the scene view as a harmless
    // placeholder.
    bloom_tex_view: sg.View = .{},
    // Optional GlowPass result (global halo, half resolution). Set via
    // setGlowTexture(); empty by default, in which case the shader skips the
    // glow block and this binds the scene view as a harmless placeholder
    // (bit-identical composite).
    glow_tex_view: sg.View = .{},
    // Optional HighlightPass result (per-mesh halo, half resolution). Set
    // via setHighlightTexture(); empty by default, in which case the shader
    // skips the highlight block and this binds the scene view as a harmless
    // placeholder (bit-identical composite). Unlike glow (whose gate reads
    // the config flag), the highlight gate reads THIS view: the parent feeds
    // the real blurred view when staged items exist and .{} otherwise.
    highlight_tex_view: sg.View = .{},
    // Optional HighlightPass raw mask (unblurred per-item fills, same
    // half-res shape as the blurred halo). Set via
    // setHighlightMaskTexture() alongside setHighlightTexture(); empty by
    // default, bound to the scene-view placeholder the shader never
    // samples while the highlight block is gated off. The inner-glow
    // composite reads raw minus blurred, so both views travel together:
    // the parent feeds the real pair only when staged items exist.
    highlight_mask_tex_view: sg.View = .{},
    // Optional VolumetricPass result (sun-CSM shaft radiance, half/quarter
    // res). Set via setShaftTexture(); empty by default, in which case the
    // shader skips the shaft block and this binds the scene view as a
    // harmless placeholder (bit-identical composite).
    shaft_tex_view: sg.View = .{},
    width: i32 = 0,
    height: i32 = 0,
    sample_count: i32 = 1,
    // Main-target depth format (env default or DEPTH): the velocity pass
    // borrows the main depth attachment and must build its pipelines with
    // this exact format. Set on every successful resize; reset with the
    // targets (unknown while unbuilt).
    depth_format: sg.PixelFormat = .DEFAULT,
    // Main-target color format metadata (diagnostics alongside
    // width/height/sample_count, never a config mode). `.DEFAULT` = never
    // built; after a successful resize this is RGBA16F, the sole contract
    // format the color and resolve images are created with.
    color_format: sg.PixelFormat = .DEFAULT,
    postprocess_shader: sg.Shader = .{},
    // TAA history ping-pong (full-size RGBA16F color targets, the sole
    // main-target format). Created on the context thread via
    // ensureTaaHistory(); destroyed with the main targets on resize (an
    // automatic reset trigger). taa_read selects the slot the composite
    // samples; the capture draw writes 1 - taa_read.
    taa_images: [2]sg.Image = [_]sg.Image{.{}} ** 2,
    taa_att_views: [2]sg.View = [_]sg.View{.{}} ** 2,
    taa_tex_views: [2]sg.View = [_]sg.View{.{}} ** 2,
    taa_width: i32 = 0,
    taa_height: i32 = 0,
    // History color format metadata (diagnostics, never a config mode:
    // always RGBA16F once built, the sole capture format).
    taa_format: sg.PixelFormat = .DEFAULT,
    taa_read: u8 = 0,
    // Fullscreen history-capture pipeline: same shader/layout/quad as the
    // display pipeline, but an explicit 1x target shape (sample_count = 1,
    // depth NONE, color RGBA16F). Never the swapchain display pipeline
    // (whose sample/depth shape mismatched the history target). Lazy:
    // created with the history slots, destroyed with them.
    taa_pipeline: sg.Pipeline = .{},
    // Shared fullscreen-quad pipeline descriptor (pure: no sg calls). The
    // display path passes the swapchain shape (color DEFAULT, samples 0,
    // depth DEFAULT); the history capture path passes the explicit 1x
    // shape (color RGBA16F, samples 1, depth NONE).
    pub fn fullscreenPipelineDesc(
        shader: sg.Shader,
        color_fmt: sg.PixelFormat,
        sample_count: i32,
        depth_fmt: sg.PixelFormat,
    ) sg.PipelineDesc {
        var desc = sg.PipelineDesc{
            .shader = shader,
            .index_type = .UINT16,
            .depth = .{
                .pixel_format = depth_fmt,
                .compare = .ALWAYS,
                .write_enabled = false,
            },
            .cull_mode = .NONE,
            .sample_count = sample_count,
        };
        desc.colors[0].pixel_format = color_fmt;
        desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        desc.layout.attrs[post_shd.ATTR_postprocess_position] = .{
            .format = .FLOAT2,
            .offset = 0,
        };
        desc.layout.attrs[post_shd.ATTR_postprocess_texcoord0] = .{
            .format = .FLOAT2,
            .offset = 2 * @sizeOf(f32),
        };
        return desc;
    }

    /// Tiny backbuffer sRGB predicate (local: render_target imports would
    /// cycle here, and postprocess/hdr.zig owns capability math, not target
    /// shape). True exactly for the hardware-sRGB swapchain variants.
    pub fn isSrgbBackbuffer(fmt: sg.PixelFormat) bool {
        return fmt == .SRGB8A8 or fmt == .SBGR8A8;
    }

    /// Pure output-params packing (no sg calls): x = 1 iff the backbuffer
    /// default format needs the manual display encode (UNORM — the exact
    /// piecewise sRGB encode runs once at the end; sRGB targets encode in
    /// hardware); y/z/w = 0. The main target is always RGBA16F, so no
    /// target-format lane exists.
    pub fn outputParamsFor(backbuffer_fmt: sg.PixelFormat) [4]f32 {
        // On WebGPU, the canvas compositor expects linear output and applies
        // display transfer automatically; manual shader encode causes double gamma / washed out screen.
        if (builtin.cpu.arch.isWasm()) return .{ 0.0, 0.0, 0.0, 0.0 };
        const x: f32 = if (isSrgbBackbuffer(backbuffer_fmt)) 0.0 else 1.0;
        return .{ x, 0.0, 0.0, 0.0 };
    }
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

        const velocity_smp = sg.makeSampler(.{
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

        const shd = sg.makeShader(post_shd.postprocessShaderDesc(sg.queryBackend()));
        // Display pipeline keeps the unchanged swapchain-default shape
        // (color DEFAULT, default sample count, depth DEFAULT).
        const pip = sg.makePipeline(fullscreenPipelineDesc(shd, .DEFAULT, 0, .DEFAULT));

        // 1x1 zero dummy texture (clear velocity fallback)
        const zero_pixels = [_]f16{ 0.0, 0.0, 0.0, 0.0 };
        var zero_data = sg.ImageData{};
        zero_data.mip_levels[0] = sg.asRange(&zero_pixels);
        const zero_img = sg.makeImage(.{
            .width = 1,
            .height = 1,
            .pixel_format = .RGBA16F,
            .data = zero_data,
        });
        const zero_view = sg.makeView(.{
            .texture = .{ .image = zero_img },
        });

        return .{
            .postprocess_sampler = smp,
            .depth_sampler = depth_smp,
            .velocity_sampler = velocity_smp,
            .lut_sampler = lut_smp,
            .default_zero_image = zero_img,
            .default_zero_view = zero_view,
            .postprocess_pipeline = pip,
            .postprocess_quad_vb = vb,
            .postprocess_quad_ib = ib,
            .postprocess_shader = shd,
        };
    }

    /// Destroys the render-target images/views (not samplers/pipelines).
    /// MSAA shape (sample_count > 1): MSAA color + MSAA depth attachments,
    /// a 1x resolve image the color resolves into at end of pass, and NO
    /// depth texture view (sokol has no depth resolve; an MSAA depth image
    /// is not samplable as a plain texture). 1x shape: plain color+depth,
    /// "resolve" views alias the color image.
    fn destroyTargets(self: *PostProcessPass) void {
        if (self.offscreen_color_att_view.id != 0) sg.destroyView(self.offscreen_color_att_view);
        if (self.offscreen_resolve_att_view.id != 0) sg.destroyView(self.offscreen_resolve_att_view);
        if (self.offscreen_resolve_tex_view.id != 0) sg.destroyView(self.offscreen_resolve_tex_view);
        if (self.offscreen_depth_att_view.id != 0) sg.destroyView(self.offscreen_depth_att_view);
        if (self.offscreen_depth_tex_view.id != 0) sg.destroyView(self.offscreen_depth_tex_view);
        if (self.offscreen_color_image.id != 0) sg.destroyImage(self.offscreen_color_image);
        if (self.offscreen_resolve_image.id != 0) sg.destroyImage(self.offscreen_resolve_image);
        if (self.offscreen_depth_image.id != 0) sg.destroyImage(self.offscreen_depth_image);
        self.offscreen_color_image = .{};
        self.offscreen_color_att_view = .{};
        self.offscreen_resolve_image = .{};
        self.offscreen_resolve_att_view = .{};
        self.offscreen_resolve_tex_view = .{};
        self.offscreen_depth_image = .{};
        self.offscreen_depth_att_view = .{};
        self.offscreen_depth_tex_view = .{};
        self.width = 0;
        self.height = 0;
        self.sample_count = 1;
        self.color_format = .DEFAULT;
        self.depth_format = .DEFAULT;
        self.destroyTaaHistory();
    }

    fn destroyTaaHistory(self: *PostProcessPass) void {
        for (0..2) |i| {
            if (self.taa_att_views[i].id != 0) sg.destroyView(self.taa_att_views[i]);
            if (self.taa_tex_views[i].id != 0) sg.destroyView(self.taa_tex_views[i]);
            if (self.taa_images[i].id != 0) sg.destroyImage(self.taa_images[i]);
            self.taa_images[i] = .{};
            self.taa_att_views[i] = .{};
            self.taa_tex_views[i] = .{};
        }
        if (self.taa_pipeline.id != 0) sg.destroyPipeline(self.taa_pipeline);
        self.taa_pipeline = .{};
        self.taa_width = 0;
        self.taa_height = 0;
        self.taa_format = .DEFAULT;
        self.taa_read = 0;
    }

    /// Explicit TAA history reset. Context thread only (destroys GPU
    /// targets immediately); headless-safe (no-ops when nothing is
    /// allocated). The update-thread reset path is the snapshot-carried
    /// PostProcessOptions.taa_camera_cut flag (one frame).
    pub fn taaReset(self: *PostProcessPass) void {
        self.destroyTaaHistory();
    }

    /// Ensures both history slots exist at `width`x`height` in RGBA16F
    /// (pre-exposure HDR capture, the sole main-target format). Returns
    /// true when (re)created — the caller's reset trigger for the frame
    /// (fresh targets carry no valid history). Size changes recreate; main
    /// destroyTargets resets history, so a main resize is also a reset.
    /// Lazily creates the 1x explicit-format capture pipeline from the
    /// same shader/layout as the display pipeline. On failure rolls back
    /// all TAA handles and returns false (disable TAA for the frame — see
    /// taaAvailable). No sg calls outside a valid context.
    pub fn ensureTaaHistory(self: *PostProcessPass, width: i32, height: i32) bool {
        if (width <= 0 or height <= 0) return false;
        if (!sg.isvalid()) return false;
        const hist_fmt: sg.PixelFormat = .RGBA16F;
        if (self.taa_width == width and self.taa_height == height and self.taa_format == hist_fmt and
            self.taa_images[0].id != 0 and self.taa_images[1].id != 0 and self.taa_pipeline.id != 0) return false;

        self.destroyTaaHistory();

        // Store-first rollback: sokol make* returns nonzero FAILED handles,
        // so every handle lands in the struct before its state check and a
        // single destroyTaaHistory frees the failed handle plus all partial
        // handles. Views are destroyed before images.
        for (0..2) |i| {
            self.taa_images[i] = sg.makeImage(.{
                .usage = .{ .color_attachment = true },
                .width = width,
                .height = height,
                .pixel_format = hist_fmt,
                .sample_count = 1,
            });
            if (sg.queryImageState(self.taa_images[i]) != .VALID) {
                self.destroyTaaHistory();
                return false;
            }
            self.taa_att_views[i] = sg.makeView(.{
                .color_attachment = .{ .image = self.taa_images[i] },
            });
            if (sg.queryViewState(self.taa_att_views[i]) != .VALID) {
                self.destroyTaaHistory();
                return false;
            }
            self.taa_tex_views[i] = sg.makeView(.{
                .texture = .{ .image = self.taa_images[i] },
            });
            if (sg.queryViewState(self.taa_tex_views[i]) != .VALID) {
                self.destroyTaaHistory();
                return false;
            }
        }
        // Explicit 1x capture shape matched to the history format. The
        // display pipeline targets the swapchain shape and must not render
        // into the history targets.
        self.taa_pipeline = sg.makePipeline(fullscreenPipelineDesc(self.postprocess_shader, hist_fmt, 1, .NONE));
        if (sg.queryPipelineState(self.taa_pipeline) != .VALID) {
            self.destroyTaaHistory();
            return false;
        }
        self.taa_width = width;
        self.taa_height = height;
        self.taa_format = hist_fmt;
        self.taa_read = 0;
        return true;
    }

    /// True when both history slots plus the capture pipeline are fully
    /// VALID (state queries, not id-only). Callers disable TAA for the
    /// frame when this is false; an empty read view then means the caller
    /// binds its dummy placeholder, never a dead handle.
    pub fn taaAvailable(self: *const PostProcessPass) bool {
        if (self.taa_width <= 0 or self.taa_height <= 0) return false;
        if (!sg.isvalid()) return false;
        if (sg.queryPipelineState(self.taa_pipeline) != .VALID) return false;
        for (0..2) |i| {
            if (sg.queryImageState(self.taa_images[i]) != .VALID) return false;
            if (sg.queryViewState(self.taa_att_views[i]) != .VALID) return false;
            if (sg.queryViewState(self.taa_tex_views[i]) != .VALID) return false;
        }
        return true;
    }

    /// History texture view the composite samples (slot taa_read). Empty
    /// until ensureTaaHistory succeeds; callers fall back to a valid
    /// placeholder view when it is empty.
    pub fn taaReadView(self: *const PostProcessPass) sg.View {
        return self.taa_tex_views[self.taa_read];
    }

    /// History color-attachment view the capture draw renders into (the
    /// slot the composite does NOT sample this frame).
    pub fn taaWriteAttView(self: *const PostProcessPass) sg.View {
        return self.taa_att_views[1 - self.taa_read];
    }

    /// True when every main-target image/view is VALID (state queries, not
    /// id-only). No sg calls outside a valid context (reports false).
    pub fn targetsValid(self: *const PostProcessPass) bool {
        if (self.width <= 0 or self.height <= 0) return false;
        if (!sg.isvalid()) return false;
        if (sg.queryImageState(self.offscreen_color_image) != .VALID) return false;
        if (sg.queryViewState(self.offscreen_color_att_view) != .VALID) return false;
        if (sg.queryViewState(self.offscreen_resolve_tex_view) != .VALID) return false;
        if (sg.queryImageState(self.offscreen_depth_image) != .VALID) return false;
        if (sg.queryViewState(self.offscreen_depth_att_view) != .VALID) return false;
        if (self.sample_count > 1) {
            if (sg.queryImageState(self.offscreen_resolve_image) != .VALID) return false;
            if (sg.queryViewState(self.offscreen_resolve_att_view) != .VALID) return false;
        } else {
            if (sg.queryViewState(self.offscreen_depth_tex_view) != .VALID) return false;
        }
        return true;
    }

    /// Rebuilds the main target at `width`x`height` in the sole contract
    /// format RGBA16F for color + resolve. Rebuilds on samples +
    /// dimensions. Returns true only when every image/view is VALID (state
    /// queries, not id-only). Creation failure rolls back all partial
    /// handles, resets width/height to 0, and returns false — disable post
    /// before any pass. No sg resource calls outside a valid context.
    /// MSAA policy preserved: 1x samples the color image directly
    /// ("resolve" views alias it); >1x adds a 1x resolve image and the
    /// MSAA depth stays write-only (no depth texture view — sokol has no
    /// depth resolve). Depth tracks the color sample count.
    pub fn resize(
        self: *PostProcessPass,
        width: i32,
        height: i32,
        sample_count: i32,
    ) bool {
        if (width <= 0 or height <= 0) return false;
        if (!sg.isvalid()) return false;
        const samples: i32 = if (sample_count < 1) 1 else sample_count;
        const env_def = sg.queryDesc().environment.defaults;
        const actual_fmt: sg.PixelFormat = .RGBA16F;
        if (self.width == width and self.height == height and
            self.sample_count == samples and self.color_format == actual_fmt) return self.targetsValid();

        self.destroyTargets();

        const depth_fmt: sg.PixelFormat = if (env_def.depth_format != .DEFAULT and env_def.depth_format != .NONE) env_def.depth_format else .DEPTH;
        self.depth_format = depth_fmt;

        // Store-first rollback throughout: sokol make* returns nonzero
        // FAILED handles, so every handle lands in the struct before its
        // state check and a single destroyTargets frees the failed handle
        // plus all partial handles. Views are destroyed before images.
        //
        // Color: attachment at the full sample count; when resolving, a
        // separate 1x resolve image (usage.resolve_attachment) receives the
        // MSAA resolve at end of pass and carries the texture view.
        self.offscreen_color_image = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = width,
            .height = height,
            .pixel_format = actual_fmt,
            .sample_count = samples,
        });
        if (sg.queryImageState(self.offscreen_color_image) != .VALID) {
            self.destroyTargets();
            return false;
        }
        self.offscreen_color_att_view = sg.makeView(.{
            .color_attachment = .{ .image = self.offscreen_color_image },
        });
        if (sg.queryViewState(self.offscreen_color_att_view) != .VALID) {
            self.destroyTargets();
            return false;
        }

        if (samples > 1) {
            self.offscreen_resolve_image = sg.makeImage(.{
                .usage = .{ .resolve_attachment = true },
                .width = width,
                .height = height,
                .pixel_format = actual_fmt,
                .sample_count = 1,
            });
            if (sg.queryImageState(self.offscreen_resolve_image) != .VALID) {
                self.destroyTargets();
                return false;
            }
            self.offscreen_resolve_att_view = sg.makeView(.{
                .resolve_attachment = .{ .image = self.offscreen_resolve_image },
            });
            if (sg.queryViewState(self.offscreen_resolve_att_view) != .VALID) {
                self.destroyTargets();
                return false;
            }
            self.offscreen_resolve_tex_view = sg.makeView(.{
                .texture = .{ .image = self.offscreen_resolve_image },
            });
            if (sg.queryViewState(self.offscreen_resolve_tex_view) != .VALID) {
                self.destroyTargets();
                return false;
            }
        } else {
            // 1x shape: postfx samples the color image directly.
            self.offscreen_resolve_tex_view = sg.makeView(.{
                .texture = .{ .image = self.offscreen_color_image },
            });
            if (sg.queryViewState(self.offscreen_resolve_tex_view) != .VALID) {
                self.destroyTargets();
                return false;
            }
        }

        // Depth: same sample count as color (sokol validation requires the
        // match). Only the 1x depth gets a texture view; MSAA depth is
        // write-only for the post chain (depth-consuming effects stay gated
        // off while MSAA is active).
        self.offscreen_depth_image = sg.makeImage(.{
            .usage = .{ .depth_stencil_attachment = true },
            .width = width,
            .height = height,
            .pixel_format = depth_fmt,
            .sample_count = samples,
        });
        if (sg.queryImageState(self.offscreen_depth_image) != .VALID) {
            self.destroyTargets();
            return false;
        }
        self.offscreen_depth_att_view = sg.makeView(.{
            .depth_stencil_attachment = .{ .image = self.offscreen_depth_image },
        });
        if (sg.queryViewState(self.offscreen_depth_att_view) != .VALID) {
            self.destroyTargets();
            return false;
        }
        if (samples == 1) {
            self.offscreen_depth_tex_view = sg.makeView(.{
                .texture = .{ .image = self.offscreen_depth_image },
            });
            if (sg.queryViewState(self.offscreen_depth_tex_view) != .VALID) {
                self.destroyTargets();
                return false;
            }
        }

        self.width = width;
        self.height = height;
        self.sample_count = samples;
        self.color_format = actual_fmt;
        if (!self.targetsValid()) {
            self.destroyTargets();
            return false;
        }
        return true;
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
        prev_view_proj: Mat4,
        camera_pos: Vec3,
        sun_dir: Vec3,
        sun_color: Color3,
        near_z: f32,
        far_z: f32,
        // Scene-depth texture sampled by the SSR/fog/motion-blur/DoF/TAA
        // blocks (BIND depth_tex). The caller resolves it (PostFXStack
        // depthSampleView): the 1x main depth, the PASS 1.7 prepass depth
        // under MSAA + gate, or the resolve-color placeholder while the
        // depth branches are flag-gated off (the binding only has to
        // exist for sokol's apply-bindings validation).
        depth_view: sg.View,
        // TAA resolve inputs: history texture sampled reprojected (bilinear
        // smp) plus the validity latch. Empty view + false keeps the
        // disabled/first-frame path (the shader early-outs before sampling).
        // capture_only re-enters the shader to store the TAA resolve output
        // into the history slot (PostFXStack capture draw): pre-exposure
        // HDR radiance in the RGBA16F history target.
        taa_history_view: sg.View,
        taa_history_valid: bool,
        taa_capture_only: bool,
        velocity_view: sg.View,
    ) void {
        if (!sg.isvalid()) return;
        if (sg.queryPipelineState(self.postprocess_pipeline) != .VALID) return;
        if (taa_capture_only) {
            // History capture renders into the 1x explicit-format history
            // target: select the matched capture pipeline, never the
            // swapchain display pipeline. Abort when the history allocation
            // failed (bind a dummy placeholder instead of a dead handle
            // and disable TAA for the frame).
            if (!self.taaAvailable()) return;
            if (self.taa_pipeline.id == 0) return;
            sg.applyPipeline(self.taa_pipeline);
        } else {
            sg.applyPipeline(self.postprocess_pipeline);
        }
        var post_bind = sg.Bindings{};
        post_bind.vertex_buffers[0] = self.postprocess_quad_vb;
        post_bind.index_buffer = self.postprocess_quad_ib;
        post_bind.views[post_shd.VIEW_scene_tex] = self.offscreen_resolve_tex_view;
        post_bind.views[post_shd.VIEW_ssao_tex] = ssao_tex;
        post_bind.views[post_shd.VIEW_depth_tex] = depth_view;
        // Pyramid glow when the parent fed a BloomPass result; otherwise a
        // valid placeholder the shader never samples (params3.z = 0 gates
        // the bloom block off).
        post_bind.views[post_shd.VIEW_bloom_tex] = if (self.bloom_tex_view.id != 0)
            self.bloom_tex_view
        else
            self.offscreen_resolve_tex_view;
        // Glow halo when the parent fed a GlowPass result; otherwise a valid
        // placeholder the shader never samples (glow_params.x = 0 gates the
        // glow block off, so the off path is bit-identical to pre-glow).
        post_bind.views[post_shd.VIEW_glow_tex] = if (self.glow_tex_view.id != 0)
            self.glow_tex_view
        else
            self.offscreen_resolve_tex_view;
        // Highlight halo when the parent fed a HighlightPass result;
        // otherwise a valid placeholder the shader never samples
        // (highlight_params.x = 0 gates the highlight block off, so the
        // off path is bit-identical to pre-highlight).
        post_bind.views[post_shd.VIEW_highlight_tex] = if (self.highlight_tex_view.id != 0)
            self.highlight_tex_view
        else
            self.offscreen_resolve_tex_view;
        // Highlight raw mask (inner-glow minuend) when the parent fed it;
        // otherwise the same harmless placeholder (unread while gated off).
        post_bind.views[post_shd.VIEW_highlight_mask_tex] = if (self.highlight_mask_tex_view.id != 0)
            self.highlight_mask_tex_view
        else
            self.offscreen_resolve_tex_view;
        // Shaft radiance when the parent fed a VolumetricPass result;
        // otherwise a valid placeholder the shader never samples
        // (shaft_params.x = 0 gates the shaft block off, so the off path
        // is bit-identical to pre-shaft).
        post_bind.views[post_shd.VIEW_shaft_tex] = if (self.shaft_tex_view.id != 0)
            self.shaft_tex_view
        else
            self.offscreen_resolve_tex_view;
        // LUT when the config carries a live binding; otherwise the resolved
        // scene view as a valid placeholder the shader never samples
        // (lut_params.x = 0 gates the LUT branch off).
        const lut_view: sg.View = if (config.lut_texture) |t| t.view else .{};
        post_bind.views[post_shd.VIEW_lut_tex] = if (lut_view.id != 0)
            lut_view
        else
            self.offscreen_resolve_tex_view;
        // TAA history (bilinear smp in-shader); a valid placeholder the
        // shader never samples while history is invalid or TAA is off
        // (taa_params.x/taa_state.x gate the branch off).
        post_bind.views[post_shd.VIEW_history_tex] = if (taa_history_view.id != 0)
            taa_history_view
        else
            self.offscreen_resolve_tex_view;
        // Screen-space velocity buffer: per-object motion vector + alpha mask;
        // binds default_zero_view when no velocity view was fed so the shader
        // falls back to depth reprojection cleanly.
        post_bind.views[post_shd.VIEW_velocity_tex] = if (velocity_view.id != 0)
            velocity_view
        else
            self.default_zero_view;
        post_bind.samplers[post_shd.SMP_smp] = self.postprocess_sampler;
        post_bind.samplers[post_shd.SMP_depth_smp] = self.depth_sampler;
        post_bind.samplers[post_shd.SMP_velocity_smp] = self.velocity_sampler;
        post_bind.samplers[post_shd.SMP_lut_smp] = self.lut_sampler;
        sg.applyBindings(post_bind);

        const pp_params = post_shd.FsParams{
            .params1 = .{
                config.exposure,
                config.bloom_threshold,
                config.bloom_intensity,
                0.0,
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
                if (config.bloom_enabled and self.bloom_tex_view.id != 0) 1.0 else 0.0,
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
                @floatFromInt(config.ssr_steps),
                @floatFromInt(config.contact_shadows_steps),
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
            // (enabled 1/0, intensity, 0, 0); zeros when glow is off, which
            // keeps the composite identical to the pre-glow path.
            .glow_params = postprocess.glowParams(config),
            // (tint rgb, 0); zeros when glow is off (unread while disabled).
            .glow_tint = postprocess.glowTintParams(config),
            // (enabled 1/0, baked global scale 1.0, 0, 0); zeros when the
            // parent fed no highlight view, which keeps the composite
            // identical to the pre-highlight path.
            .highlight_params = postprocess.highlightParams(self.highlight_tex_view.id != 0),
            // (enabled 1/0, intensity, 0, 0); zeros when the shaft is off,
            // which keeps the composite identical to the pre-shaft path.
            .shaft_params = postprocess.shaftParams(config, self.shaft_tex_view.id != 0),
            // (enabled 1/0, intensity, distance, thickness); zeros when off
            .contact_shadow_params = postprocess.contactShadowParams(config),
            // (enabled 1/0, intensity, contrast, 0); zeros when off
            .local_tonemap_params = postprocess.localTonemapParams(config),
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
            .reproj_mat = Mat4.mul(prev_view_proj, inv_view_proj),
            .motion_blur_params = .{
                if (config.motion_blur_enabled) 1.0 else 0.0,
                config.motion_blur_intensity,
                config.motion_blur_max_blur_px,
                @floatFromInt(config.motion_blur_samples),
            },
            // (enabled 1/0, blend, clamp, sharpness); zeros when TAA is off,
            // which keeps the composite identical to the pre-TAA path.
            .taa_params = postprocess.taaParams(config),
            // (history_valid 1/0, capture_only 1/0).
            .taa_state = postprocess.taaState(taa_history_valid, taa_capture_only),
            // Output lane (appended last): x = manual display-encode flag
            // iff the backbuffer default is NOT sRGB (UNORM encodes the
            // exact piecewise sRGB once at the end; sRGB targets encode in
            // hardware); y/z/w = 0. The main target is always RGBA16F.
            .output_params = outputParamsFor(
                sg.queryDesc().environment.defaults.color_format,
            ),
        };
        sg.applyUniforms(post_shd.UB_fs_params, sg.asRange(&pp_params));
        sg.draw(0, 6, 1);
    }

    // Feed the BloomPass pyramid result into the composite. Call every frame
    // before render() once the parent owns a BloomPass; pass .{} to detach
    // and skip the bloom block (params3.z = 0).
    pub fn setBloomTexture(self: *PostProcessPass, view: sg.View) void {
        self.bloom_tex_view = view;
    }

    // Feed the GlowPass halo result into the composite. Call every frame
    // before render() once the parent owns a GlowPass; pass .{} to detach
    // and return to the no-glow composite path (bit-identical to pre-glow).
    pub fn setGlowTexture(self: *PostProcessPass, view: sg.View) void {
        self.glow_tex_view = view;
    }

    // Feed the HighlightPass halo result into the composite. Call every
    // frame before render() once the parent owns a HighlightPass; pass .{}
    // to detach and return to the no-highlight composite path
    // (bit-identical to pre-highlight). The composite gate reads this view
    // (nonzero = active), not a config flag: the parent must pass the real
    // blurred view only when staged highlight items exist.
    pub fn setHighlightTexture(self: *PostProcessPass, view: sg.View) void {
        self.highlight_tex_view = view;
    }

    // Feed the HighlightPass raw mask into the composite (inner-glow
    // minuend: the shader computes raw minus blurred). Call every frame
    // alongside setHighlightTexture() with the same active/empty
    // discipline: the real mask view when staged items exist, .{} to
    // detach. Never read while the highlight block is gated off, so the
    // off path stays bit-identical.
    pub fn setHighlightMaskTexture(self: *PostProcessPass, view: sg.View) void {
        self.highlight_mask_tex_view = view;
    }

    // Feed the VolumetricPass shaft result into the composite. Call every
    // frame before render() once the parent owns a VolumetricPass; pass
    // .{} to detach and return to the no-shaft composite path
    // (bit-identical to pre-shaft).
    pub fn setShaftTexture(self: *PostProcessPass, view: sg.View) void {
        self.shaft_tex_view = view;
    }

    pub fn deinit(self: *PostProcessPass) void {
        // destroyTargets is the sole owner of main + TAA history targets,
        // including the lazy capture pipeline (via destroyTaaHistory).
        self.destroyTargets();
        if (self.postprocess_sampler.id != 0) sg.destroySampler(self.postprocess_sampler);
        if (self.depth_sampler.id != 0) sg.destroySampler(self.depth_sampler);
        if (self.velocity_sampler.id != 0) sg.destroySampler(self.velocity_sampler);
        if (self.lut_sampler.id != 0) sg.destroySampler(self.lut_sampler);
        if (self.default_zero_view.id != 0) sg.destroyView(self.default_zero_view);
        if (self.default_zero_image.id != 0) sg.destroyImage(self.default_zero_image);
        if (self.postprocess_pipeline.id != 0) sg.destroyPipeline(self.postprocess_pipeline);
        if (self.postprocess_shader.id != 0) sg.destroyShader(self.postprocess_shader);
        self.postprocess_shader = .{};
        if (self.postprocess_quad_vb.id != 0) sg.destroyBuffer(self.postprocess_quad_vb);
        if (self.postprocess_quad_ib.id != 0) sg.destroyBuffer(self.postprocess_quad_ib);
    }
};

// --- headless pure tests (no sg calls, no context): output-encode
// packing, descriptor identity, and empty guards. Live pixels stay a
// real-run concern. ---

const testing = @import("std").testing;

test "outputParamsFor packs manual-encode flag, yzw zero" {
    // UNORM backbuffer: manual encode on.
    try testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 0.0 }, PostProcessPass.outputParamsFor(.BGRA8));
    try testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 0.0 }, PostProcessPass.outputParamsFor(.RGBA8));
    try testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 0.0 }, PostProcessPass.outputParamsFor(.RGBA16F));
    try testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 0.0 }, PostProcessPass.outputParamsFor(.DEFAULT));
    try testing.expectEqual([4]f32{ 1.0, 0.0, 0.0, 0.0 }, PostProcessPass.outputParamsFor(.NONE));
    // sRGB backbuffer: hardware encodes, no manual pass.
    try testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, PostProcessPass.outputParamsFor(.SRGB8A8));
    try testing.expectEqual([4]f32{ 0.0, 0.0, 0.0, 0.0 }, PostProcessPass.outputParamsFor(.SBGR8A8));
}

test "isSrgbBackbuffer flags exactly the hardware-sRGB swapchain variants" {
    try testing.expect(PostProcessPass.isSrgbBackbuffer(.SRGB8A8));
    try testing.expect(PostProcessPass.isSrgbBackbuffer(.SBGR8A8));
    try testing.expect(!PostProcessPass.isSrgbBackbuffer(.RGBA8));
    try testing.expect(!PostProcessPass.isSrgbBackbuffer(.BGRA8));
    try testing.expect(!PostProcessPass.isSrgbBackbuffer(.RGBA16F));
    try testing.expect(!PostProcessPass.isSrgbBackbuffer(.DEFAULT));
    try testing.expect(!PostProcessPass.isSrgbBackbuffer(.NONE));
}

test "fullscreenPipelineDesc keeps display shape default, capture shape explicit 1x" {
    const shd: sg.Shader = .{};
    const display = PostProcessPass.fullscreenPipelineDesc(shd, .DEFAULT, 0, .DEFAULT);
    try testing.expectEqual(sg.PixelFormat.DEFAULT, display.colors[0].pixel_format);
    try testing.expectEqual(@as(i32, 0), display.sample_count);
    try testing.expectEqual(sg.PixelFormat.DEFAULT, display.depth.pixel_format);
    try testing.expect(display.depth.compare == .ALWAYS);
    try testing.expect(!display.depth.write_enabled);
    try testing.expect(display.cull_mode == .NONE);
    try testing.expect(display.index_type == .UINT16);

    const capture = PostProcessPass.fullscreenPipelineDesc(shd, .RGBA16F, 1, .NONE);
    try testing.expectEqual(sg.PixelFormat.RGBA16F, capture.colors[0].pixel_format);
    try testing.expectEqual(@as(i32, 1), capture.sample_count);
    try testing.expectEqual(sg.PixelFormat.NONE, capture.depth.pixel_format);
    // Same quad layout on both (shared helper, no duplicate layout).
    try testing.expectEqual(display.layout.buffers[0].stride, capture.layout.buffers[0].stride);
    try testing.expectEqual(
        display.layout.attrs[@import("postprocess_shader").ATTR_postprocess_position].format,
        capture.layout.attrs[@import("postprocess_shader").ATTR_postprocess_position].format,
    );
}

test "zero pass reports no valid targets and default shape metadata" {
    const p = PostProcessPass{};
    try testing.expectEqual(@as(i32, 0), p.width);
    try testing.expectEqual(@as(i32, 0), p.height);
    try testing.expectEqual(@as(i32, 1), p.sample_count);
    try testing.expectEqual(sg.PixelFormat.DEFAULT, p.color_format);
    try testing.expectEqual(sg.PixelFormat.DEFAULT, p.taa_format);
    try testing.expectEqual(@as(u32, 0), p.taa_pipeline.id);
    // Empty read view: callers bind a dummy placeholder, never this.
    try testing.expectEqual(@as(u32, 0), p.taaReadView().id);
    // Headless guards fail closed without touching sg resources.
    try testing.expect(!p.targetsValid());
    try testing.expect(!p.taaAvailable());
    var q = PostProcessPass{};
    try testing.expect(!q.resize(64, 64, 1));
    try testing.expect(!q.ensureTaaHistory(64, 64));
    try testing.expect(!q.taaAvailable());
}
