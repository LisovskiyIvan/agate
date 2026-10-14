//! Owns main HDR targets, TAA history and GPU submission. Pure composite
//! bindings/uniform packing live in postprocess/composite.zig and main
//! target/TAA lifecycle in postprocess/targets.zig; public methods forward
//! there without changing resource ownership or the pass API.
const std = @import("std");
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
const composite = @import("postprocess/composite.zig");
const targets = @import("postprocess/targets.zig");

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
    // Optional SsgiPass result (v2 denoised bleed, half res). Set via
    // setSsgiTexture(); empty by default, in which case the shader skips
    // the pass-sample branch and this binds the scene view as a harmless
    // placeholder (bit-identical composite).
    ssgi_tex_view: sg.View = .{},
    // Optional SsrPass result (v2 denoised reflection, half res). Set via
    // setSsrTexture(); empty by default, in which case the shader skips
    // the pass-sample branch and this binds the scene view as a harmless
    // placeholder (bit-identical composite).
    ssr_tex_view: sg.View = .{},
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
        return targets.fullscreenPipelineDesc(shader, color_fmt, sample_count, depth_fmt);
    }

    pub fn fullscreenPipelineDescForShape(
        shader: sg.Shader,
        shape: @import("../target_shape.zig").TargetShape,
    ) sg.PipelineDesc {
        return targets.fullscreenPipelineDescForShape(shader, shape);
    }

    /// Tiny backbuffer sRGB predicate (local: render_target imports would
    /// cycle here, and postprocess/hdr.zig owns capability math, not target
    /// shape). True exactly for the hardware-sRGB swapchain variants.
    pub fn isSrgbBackbuffer(fmt: sg.PixelFormat) bool {
        return composite.isSrgbBackbuffer(fmt);
    }

    /// Pure output-params packing (no sg calls): x = 1 iff the backbuffer
    /// default format needs the manual display encode (UNORM — the exact
    /// piecewise sRGB encode runs once at the end; sRGB targets encode in
    /// hardware); y/z/w = 0. The main target is always RGBA16F, so no
    /// target-format lane exists.
    pub fn outputParamsFor(backbuffer_fmt: sg.PixelFormat) [4]f32 {
        return composite.outputParamsFor(backbuffer_fmt);
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
        targets.destroyTargets(self);
    }

    fn destroyTaaHistory(self: *PostProcessPass) void {
        targets.destroyTaaHistory(self);
    }

    /// Explicit TAA history reset. Context thread only (destroys GPU
    /// targets immediately); headless-safe (no-ops when nothing is
    /// allocated). The update-thread reset path is the snapshot-carried
    /// PostProcessOptions.taa.camera_cut flag (one frame).
    pub fn taaReset(self: *PostProcessPass) void {
        targets.taaReset(self);
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
        return targets.ensureTaaHistory(self, width, height);
    }

    /// True when both history slots plus the capture pipeline are fully
    /// VALID (state queries, not id-only). Callers disable TAA for the
    /// frame when this is false; an empty read view then means the caller
    /// binds its dummy placeholder, never a dead handle.
    pub fn taaAvailable(self: *const PostProcessPass) bool {
        return targets.taaAvailable(self);
    }

    /// History texture view the composite samples (slot taa_read). Empty
    /// until ensureTaaHistory succeeds; callers fall back to a valid
    /// placeholder view when it is empty.
    pub fn taaReadView(self: *const PostProcessPass) sg.View {
        return targets.taaReadView(self);
    }

    /// History color-attachment view the capture draw renders into (the
    /// slot the composite does NOT sample this frame).
    pub fn taaWriteAttView(self: *const PostProcessPass) sg.View {
        return targets.taaWriteAttView(self);
    }

    /// True when every main-target image/view is VALID (state queries, not
    /// id-only). No sg calls outside a valid context (reports false).
    pub fn targetsValid(self: *const PostProcessPass) bool {
        return targets.targetsValid(self);
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
        return targets.resize(self, width, height, sample_count);
    }

    /// Valid texture view for slots that semantically want scene depth.
    /// 1x target: the depth texture itself. MSAA target: no depth texture
    /// exists, so the resolved color view serves as a valid placeholder —
    /// the depth-consuming shader branches (SSR/DoF) are flag-gated off
    /// while MSAA is active (scene/postfx_stack.zig), the binding only has
    /// to exist for sokol's apply-bindings validation.
    pub fn depthSampleView(self: *const PostProcessPass) sg.View {
        return targets.depthSampleView(self);
    }

    pub fn render(
        self: *PostProcessPass,
        config: PostProcessOptions,
        ssao_tex: sg.View,
        ssao_enabled: bool,
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
        // depthSampleView): the 1x main depth, or the resolve-color
        // placeholder while the depth branches are flag-gated off (the
        // binding only has to exist for sokol's apply-bindings validation).
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
        sg.applyBindings(self.buildCompositeBindings(config, ssao_tex, depth_view, taa_history_view, velocity_view));
        const pp_params = self.packCompositeParams(
            config,
            ssao_enabled,
            ssao_intensity,
            cur_w,
            cur_h,
            view_proj,
            inv_view_proj,
            prev_view_proj,
            camera_pos,
            sun_dir,
            sun_color,
            near_z,
            far_z,
            taa_history_valid,
            taa_capture_only,
            // Display-transfer lane input: the swapchain default color
            // format (UNORM needs the manual shader encode, sRGB targets
            // encode in hardware). Queried here so packCompositeParams
            // stays a pure, headless-testable pack with no sg calls.
            sg.queryDesc().environment.defaults.color_format,
        );
        sg.applyUniforms(post_shd.UB_fs_params, sg.asRange(&pp_params));
        sg.draw(0, 6, 1);
    }

    /// View/texture binding assembly for the composite (pure: fills an
    /// sg.Bindings struct, no sg calls). Every optional effect view falls
    /// back to a valid placeholder the shader never samples while its lane
    /// gates the block off, so the off paths stay bit-identical to their
    /// pre-effect composites; the velocity slot falls back to the zero
    /// mask so the shader takes the depth-reprojection path.
    pub fn buildCompositeBindings(
        self: *const PostProcessPass,
        config: PostProcessOptions,
        ssao_tex: sg.View,
        depth_view: sg.View,
        taa_history_view: sg.View,
        velocity_view: sg.View,
    ) sg.Bindings {
        return composite.buildCompositeBindings(self, config, ssao_tex, depth_view, taa_history_view, velocity_view);
    }

    /// Composite uniform packing (pure: no sg calls — the caller passes the
    /// swapchain default color format in, see render()). Builds the exact
    /// FsParams literal render() used to inline: option lanes via the
    /// postprocess named-lane helpers, camera/SSR lanes, matrices, and the
    /// TAA/output state.
    pub fn packCompositeParams(
        self: *const PostProcessPass,
        config: PostProcessOptions,
        ssao_enabled: bool,
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
        taa_history_valid: bool,
        taa_capture_only: bool,
        backbuffer_fmt: sg.PixelFormat,
    ) post_shd.FsParams {
        return composite.packCompositeParams(
            self,
            config,
            ssao_enabled,
            ssao_intensity,
            cur_w,
            cur_h,
            view_proj,
            inv_view_proj,
            prev_view_proj,
            camera_pos,
            sun_dir,
            sun_color,
            near_z,
            far_z,
            taa_history_valid,
            taa_capture_only,
            backbuffer_fmt,
        );
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

    // Feed the SsgiPass denoised bleed into the composite. Call every
    // frame before render() once the parent owns an SsgiPass; pass .{}
    // to detach and return to the no-pass composite path (bit-identical
    // to pre-SSGI-pass; the v1 inline branch keys off its own lane).
    pub fn setSsgiTexture(self: *PostProcessPass, view: sg.View) void {
        self.ssgi_tex_view = view;
    }

    // Feed the SsrPass denoised reflection into the composite. Call every
    // frame before render() once the parent owns an SsrPass; pass .{}
    // to detach and return to the no-pass composite path (bit-identical
    // to pre-SSR-pass; the v1 inline branch keys off its own lane).
    pub fn setSsrTexture(self: *PostProcessPass, view: sg.View) void {
        self.ssr_tex_view = view;
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
