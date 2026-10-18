const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const sglue = sokol.glue;

const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const Color3 = math.Color3;
const Color4 = math.Color4;

const Camera = @import("../camera.zig").Camera;
const Viewport = @import("../camera.zig").Viewport;
const passes = @import("../passes/mod.zig");
const scene_render_queue = @import("render_queue.zig");
const postprocess = @import("../postprocess.zig");
const PostProcessOptions = postprocess.PostProcessOptions;
const ssao_mod = @import("../ssao.zig");
const SSAOOptions = ssao_mod.SSAOOptions;
const msaa = @import("msaa.zig");
const ui_frame_mod = @import("ui_frame.zig");
const UiFrame = ui_frame_mod.UiFrame;
const stats_mod = @import("stats.zig");
const SceneStats = stats_mod.SceneStats;

/// Screen-space post-processing stack: the offscreen HDR main target, the
/// fullscreen display pass, SSAO, the bloom pyramid, the glow layer, the
/// highlight layer, volumetric shafts, and the inverse-hull outline state.
/// Scene feeds per-frame configs through ChainParams; everything GPU-owned
/// lives here. Highlighted mesh state stays on Scene.
pub const PostFXStack = struct {
    postprocess_pass: passes.PostProcessPass,
    ssao_pass: passes.SSAOPass,
    ssgi_pass: passes.SsgiPass,
    ssr_pass: passes.SsrPass,
    bloom_pass: passes.BloomPass,
    glow_pass: passes.GlowPass,
    highlight_pass: passes.HighlightPass,
    volumetric_pass: passes.VolumetricPass = .{},
    velocity_pass: passes.VelocityPass = .{},
    depth_pyramid_pass: passes.DepthPyramidPass = .{},
    outline_pass: passes.OutlinePass,

    outline_msaa: ?passes.OutlinePass = null,

    main_samples: i32 = 1,
    main_color_format: sg.PixelFormat = .RGBA16F,

    /// Last clamped `PostProcessOptions.render_scale` the main target was
    /// prepared/resized with. resizeOffscreen (window resize) reuses it so
    /// the offscreen targets stay at the scaled size between frames; the
    /// per-frame prepareMainTargets call refreshes it from the config.
    render_scale: f32 = 1.0,

    warn_depth_effects: msaa.WarnOnce = .{},
    warn_taa_target: msaa.WarnOnce = .{},
    warn_ssgi_temporal: msaa.WarnOnce = .{},
    warn_ao_temporal: msaa.WarnOnce = .{},

    outline_enabled: bool = false,
    outline_color: Color4 = Color4.new(1.0, 0.5, 0.0, 1.0),
    outline_width_px: f32 = 2.0,

    prev_view_proj: Mat4 = Mat4.identity,
    has_prev_view_proj: bool = false,

    taa_frame: u64 = 0,
    taa_has_history: bool = false,
    taa_enabled_prev: bool = false,
    taa_explicit_reset: bool = false,

    auto_exposure: postprocess.AutoExposureState = .{},

    warn_taa_msaa: msaa.WarnOnce = .{},

    pub fn init() PostFXStack {
        return .{
            .postprocess_pass = passes.PostProcessPass.init(),
            .ssao_pass = passes.SSAOPass.init(),
            .ssgi_pass = passes.SsgiPass.init(),
            .ssr_pass = passes.SsrPass.init(),
            .bloom_pass = passes.BloomPass.init(),
            .glow_pass = passes.GlowPass.init(),
            .highlight_pass = passes.HighlightPass.init(),
            .volumetric_pass = passes.VolumetricPass.init(),
            .velocity_pass = passes.VelocityPass.init(),
            .depth_pyramid_pass = passes.DepthPyramidPass.init(),
            .outline_pass = passes.OutlinePass.init(1, .RGBA16F),
        };
    }

    pub fn deinit(self: *PostFXStack) void {
        self.postprocess_pass.deinit();
        self.ssao_pass.deinit();
        self.ssgi_pass.deinit();
        self.ssr_pass.deinit();
        self.bloom_pass.deinit();
        self.glow_pass.deinit();
        self.highlight_pass.deinit();
        self.volumetric_pass.deinit();
        self.velocity_pass.deinit();
        self.depth_pyramid_pass.deinit();
        self.outline_pass.deinit();
        if (self.outline_msaa) |*op| op.deinit();
        self.outline_msaa = null;
    }

    /// Window-resize path: the main target is mandatory, auxiliary targets
    /// only when already allocated (avoids provisioning VRAM for effects
    /// that never ran). Highlight/shaft targets stay lazy under render().
    /// Applies the current `render_scale`: every viewport-sized target here
    /// belongs to the scaled scene/effect resolution, never the swapchain.
    pub fn resizeAll(self: *PostFXStack, width: i32, height: i32) void {
        if (width <= 0 or height <= 0) return;
        if (!sg.isvalid()) return;
        const size = postprocess.scaledRenderSize(width, height, self.render_scale);
        _ = self.postprocess_pass.resize(size.w, size.h, self.main_samples);
        if (self.ssao_pass.width != 0) self.ssao_pass.resize(size.w, size.h);
        if (self.ssgi_pass.width != 0) self.ssgi_pass.resize(size.w, size.h);
        if (self.ssr_pass.width != 0) self.ssr_pass.resize(size.w, size.h);
        if (self.bloom_pass.base_width != 0) self.bloom_pass.resize(size.w, size.h);
        if (self.glow_pass.base_width != 0) self.glow_pass.resize(size.w, size.h);
        if (self.velocity_pass.width != 0) _ = self.velocity_pass.ensure(size.w, size.h);
        if (self.depth_pyramid_pass.base_width != 0) self.depth_pyramid_pass.resize(size.w, size.h);
        passes.OutlinePass.resize(size.w, size.h);
    }

    /// Actual size of the allocated HDR main target — the render-scale
    /// resolution the scene pass, effect passes and history targets run
    /// at. 0 until prepareMainTargets succeeds; callers fall back to the
    /// swapchain size. Single source of truth: never recompute the scale
    /// math in the render path.
    pub fn renderWidth(self: *const PostFXStack) i32 {
        return self.postprocess_pass.width;
    }

    pub fn renderHeight(self: *const PostFXStack) i32 {
        return self.postprocess_pass.height;
    }

    pub fn depthSampleView(self: *const PostFXStack) sg.View {
        return self.postprocess_pass.depthSampleView();
    }

    pub fn taaReset(self: *PostFXStack) void {
        self.taa_explicit_reset = true;
        self.postprocess_pass.taaReset();
    }

    /// Actual MSAA count for the main target, including the runtime format
    /// gate. Headless-safe: reports 1x without touching resources.
    pub fn targetSampleCount(requested: i32) i32 {
        if (!sg.isvalid()) return 1;
        return msaa.effectiveSampleCount(requested, .{
            .formats_msaa_capable = msaa.mainTargetFormatsMsaaCapable(),
            .backend = sg.queryBackend(),
        });
    }

    /// Allocates the HDR main target (RGBA16F, sole format) plus the TAA
    /// history when applicable. `width`/`height` are the SWAPCHAIN size;
    /// the target is allocated at the render scale from `config` (native
    /// size at 1.0). Fails closed with a clear error before any pass
    /// begins; never provisions an alternate color path.
    pub fn prepareMainTargets(
        self: *PostFXStack,
        config: *PostProcessOptions,
        requested: i32,
        width: i32,
        height: i32,
    ) error{ UnsupportedHDR, TargetAllocationFailed }!i32 {
        if (!sg.isvalid()) return error.UnsupportedHDR;
        if (!postprocess.hdr.queryCapabilities().supported()) return error.UnsupportedHDR;
        if (width <= 0 or height <= 0) return error.TargetAllocationFailed;
        // The TAA history is display-sized, so a render-scale change no
        // longer resizes it. Reset explicitly: blending across two scene
        // resolutions is the classic stale-history artifact.
        const scale_changed = self.render_scale != config.render_scale;
        self.render_scale = config.render_scale;
        const size = postprocess.scaledRenderSize(width, height, self.render_scale);
        const samples = targetSampleCount(requested);
        if (!self.postprocess_pass.resize(size.w, size.h, samples)) return error.TargetAllocationFailed;
        self.main_samples = samples;
        self.main_color_format = .RGBA16F;
        if (config.taa.enabled and samples == 1) {
            if (scale_changed) self.taa_explicit_reset = true;
            // History is allocated at the DISPLAY size, not the scaled main
            // size: the composite resolves TAA at output pixels and
            // reprojects with the low-res depth/velocity buffers, so render
            // scale gets a temporal reconstruction instead of a bilinear
            // upscale. At scale 1.0 this is bit-identical.
            if (self.postprocess_pass.ensureTaaHistory(width, height)) self.taa_explicit_reset = true;
            if (!self.postprocess_pass.taaAvailable()) {
                _ = self.warn_taa_target.warn("TAA: history allocation failed; continuing with spatial AA", .{});
                config.taa.enabled = false;
                config.fxaa.enabled = true;
                self.taa_has_history = false;
            }
        } else if (samples > 1) {
            self.taa_has_history = false;
        }
        return samples;
    }

    /// Begins the main scene pass into the prepared HDR target. No
    /// allocation here; returns false when the prepared shape does not
    /// match so the frame can fail closed before sg.beginPass.
    pub fn beginMainPass(
        self: *PostFXStack,
        main_pass_action: sg.PassAction,
        samples: i32,
        width: i32,
        height: i32,
    ) bool {
        if (!sg.isvalid()) return false;
        if (width <= 0 or height <= 0) return false;
        const pp = &self.postprocess_pass;
        if (pp.width != width or pp.height != height) return false;
        if (pp.sample_count != samples) return false;
        if (pp.color_format != .RGBA16F) return false;
        if (!pp.targetsValid()) return false;
        var main_pass = sg.Pass{ .action = main_pass_action };
        main_pass.attachments.colors[0] = pp.offscreen_color_att_view;
        main_pass.attachments.depth_stencil = pp.offscreen_depth_att_view;
        if (samples > 1) {
            main_pass.attachments.resolves[0] = pp.offscreen_resolve_att_view;
            main_pass.action.colors[0].store_action = .DONTCARE;
            main_pass.action.depth.store_action = .DONTCARE;
        }
        sg.beginPass(main_pass);
        passes.OutlinePass.resize(width, height);
        return true;
    }

    /// Canonical outline draw for staged items; the pipeline variant
    /// matches the main-target shape (sample count + color format).
    pub fn renderOutlineItems(
        self: *PostFXStack,
        view_proj: Mat4,
        eye: Vec3,
        outline_items: []const passes.OutlineDrawItem,
        outline_skins: []const [scene_render_queue.MAX_BONES]math.Mat4,
        samples: i32,
        color_format: sg.PixelFormat,
        stats: *SceneStats,
        enabled: bool,
        color: Color4,
        width_px: f32,
    ) void {
        if (enabled and outline_items.len > 0) {
            const pass = self.outlinePassFor(samples, color_format);
            pass.renderItems(view_proj, eye, outline_items, outline_skins, color, width_px);
            const outline_count: u32 = @intCast(outline_items.len);
            stats.main_draw_calls += outline_count;
            stats.draw_calls += outline_count;
        }
    }

    fn outlinePassFor(self: *PostFXStack, samples: i32, color_format: sg.PixelFormat) *passes.OutlinePass {
        if (samples <= 1 and color_format == .RGBA16F) return &self.outline_pass;
        if (self.outline_msaa == null or self.outline_msaa.?.sample_count != samples or self.outline_msaa.?.color_format != color_format) {
            if (self.outline_msaa) |*op| op.deinit();
            self.outline_msaa = passes.OutlinePass.init(samples, color_format);
        }
        return &self.outline_msaa.?;
    }

    pub const ChainParams = struct {
        post: PostProcessOptions,
        ssao: SSAOOptions,
        camera: Camera,
        aspect: f32,
        view_proj: Mat4,
        /// The UNJITTERED view-projection: the reprojection domain for every
        /// temporal consumer (TAA history, velocity motion, SSAO/SSGI
        /// temporal). Jitter belongs to the raster only — a jittered
        /// reprojection drags the accumulated history by the per-frame
        /// jitter delta (the still-camera "shaking" bug), so reprojection
        /// must use this matrix while depth reconstruction keeps
        /// `view_proj` (the matrix the depth was rasterized with).
        view_proj_unjittered: Mat4,
        eye: Vec3,
        sun_dir: Vec3,
        sun_color: Color3,
        default_white_view: sg.View,
        stats: *SceneStats,
        main_samples: i32 = 1,
        highlight_items: []const passes.HighlightDrawItem = &.{},
        highlight_viewport: Viewport = .{},
        shaft_shadow_view: sg.View = .{},
        shaft_cascades: [4]Mat4 = @as([4]Mat4, @splat(Mat4.identity)),
        shaft_splits: [4]f32 = .{ 0, 0, 0, 0 },
        shaft_shadow_bias: f32 = 0.0,
        shadows_enabled: bool = false,
        ui: ?*const UiFrame = null,
        /// True only when this frame's velocity pass actually rendered into
        /// a valid target (ensure succeeded). Gates the composite's velocity
        /// view: a failed ensure must feed the zero fallback, never a
        /// stale-sized target.
        velocity_valid: bool = false,
        /// True when more than one camera view renders this frame: the AO
        /// temporal history must never blend across views, so the pass
        /// fails closed to the gather-direct path (default false keeps
        /// every existing call site on the single-view path).
        multi_camera: bool = false,
    };

    /// Effect stages plus the display pass. The display pass always runs;
    /// effect stages run only while the master switch and their own flag
    /// are on. Empty views (.{} ) mark inactive/failed effect results; the
    /// display pass binds its own placeholder for those slots.
    ///
    /// Render-scale split: `cur_w`/`cur_h` is the SWAPCHAIN (display) size;
    /// every scene-side size (effect targets, TAA history, and the
    /// composite's SOURCE texel size) is the allocated main-target size
    /// (`fx_*`, see PostFXStack.renderWidth). At scale 1.0 the two are
    /// equal and the chain is bit-identical to the native path; below 1.0
    /// the main target is smaller and the display pass UV-maps (bilinear)
    /// upscale it, while UI draws after the composite at full resolution.
    pub fn renderChain(self: *PostFXStack, params: ChainParams, cur_w: i32, cur_h: i32) void {
        var post = params.post.forFrame();
        const fx_w = if (self.postprocess_pass.width > 0) self.postprocess_pass.width else cur_w;
        const fx_h = if (self.postprocess_pass.height > 0) self.postprocess_pass.height else cur_h;
        if (post.auto_exposure.enabled) {
            post.exposure = self.auto_exposure.adapted_exposure;
        }
        var ssao = params.ssao;
        if (!params.post.enabled) {
            ssao.enabled = false;
        }
        const msaa_active = params.main_samples > 1;
        if (msaa.suppressDepthEffects(params.main_samples)) {
            if (msaa.depthEffectsActive(true, ssao.enabled, post.ssr.enabled, post.dof.enabled, post.fog.enabled, post.motion_blur.enabled, post.ssgi.enabled) or post.shafts.enabled) {
                _ = self.warn_depth_effects.warn(
                    "msaa: SSAO/SSR/DoF/Fog/MotionBlur/SSGI/Shaft disabled this session: MSAA x{} main target has no depth resolve",
                    .{params.main_samples},
                );
            }
            ssao.enabled = false;
            post.ssr.enabled = false;
            post.dof.enabled = false;
            post.fog.enabled = false;
            post.motion_blur.enabled = false;
            post.fxaa.enabled = false;
            post.shafts.enabled = false;
            post.ssgi.enabled = false;
        }
        if (msaa_active and post.taa.enabled) {
            _ = self.warn_taa_msaa.warn(
                "msaa: TAA disabled this session: MSAA x{} main target has no depth resolve",
                .{params.main_samples},
            );
            post.taa.enabled = false;
        }
        const taa_active = post.taa.enabled;
        const ao_explicit_reset = self.taa_explicit_reset;
        if (!taa_active) {
            self.taa_has_history = false;
            self.taa_enabled_prev = false;
            self.taa_explicit_reset = false;
        }

        var ssao_view = params.default_white_view;
        const ssao_active = ssao.enabled;
        // Reprojection basis shared by the SSAO temporal stage and the
        // SSGI/composite stages below: hoisted above the SSAO block so the
        // AO history reprojects against the same target (unsuppressed — on
        // cut frames its history resets, so stale vectors never blend).
        // The velocity view is valid only when this frame's pass rendered
        // into it (params.velocity_valid, gated on the ensure result in
        // frame_render): otherwise the zero fallback, so a failed ensure
        // can never feed a stale-sized target to the temporal stage.
        const inv_view_proj = params.view_proj.invert() orelse Mat4.identity;
        // Unjittered reprojection pair: identity under a still camera, pure
        // camera motion otherwise. `prev_view_proj` stores the unjittered
        // matrix (set below), so history lookups land in the unjittered
        // accumulation space the resolves write.
        const inv_view_proj_unjittered = params.view_proj_unjittered.invert() orelse Mat4.identity;
        const prev_vp = if (!self.has_prev_view_proj) params.view_proj_unjittered else self.prev_view_proj;
        const reproj_mat_unjittered = Mat4.mul(prev_vp, inv_view_proj_unjittered);
        const vel_view_in = if (params.velocity_valid and self.velocity_pass.width != 0)
            self.velocity_pass.velocityTexView()
        else
            sg.View{};
        // SSGI albedo rides the same validity gate as the velocity view:
        // the pass wrote both targets this frame, or neither.
        const ssgi_albedo_view_in = if (params.velocity_valid and self.velocity_pass.width != 0)
            self.velocity_pass.albedoTexView()
        else
            sg.View{};
        if (post.enabled and ssao_active) {
            const msaa_active_ssao = params.main_samples > 1;
            if (ssao.temporal.enabled and params.multi_camera) {
                _ = self.warn_ao_temporal.warn("ssao: temporal history bypassed (multi-camera); using gather-direct", .{});
            }
            const ssao_res = self.ssao_pass.renderWithContext(.{
                .camera = params.camera,
                .aspect = params.aspect,
                .depth_tex_view = self.depthSampleView(),
                .config = ssao,
                .cur_w = fx_w,
                .cur_h = fx_h,
                .view_proj = params.view_proj,
                .inv_view_proj = inv_view_proj,
                .prev_view_proj = prev_vp,
                .velocity_view = vel_view_in,
                .velocity_valid = vel_view_in.id != 0,
                .camera_cut = post.taa.camera_cut or ao_explicit_reset,
                .multi_camera = params.multi_camera,
                .msaa_active = msaa_active_ssao,
            });
            const out = self.ssao_pass.ssao_blur_tex_view;
            if (out.id != 0) {
                ssao_view = out;
                // Count only draws that ran: an allocation failure or a
                // fail-closed gate must not count a temporal draw that
                // never happened (see SSAOPass.RenderResult).
                const ssao_draws: u32 = 2 + (if (ssao_res.temporal_executed) @as(u32, 1) else 0);
                params.stats.post_draw_calls += ssao_draws;
                params.stats.draw_calls += ssao_draws;
                params.stats.triangles += 2 * ssao_draws;
            }
        } else {
            // Skipped this frame (master switch off, SSAO off, or
            // MSAA-suppressed): render() never ran, so reset history here
            // — a re-enable always replays the reset frame instead of
            // blending across the gap.
            self.ssao_pass.resetTemporalHistory();
        }

        var depth_pyramid_view: sg.View = .{};
        if (postprocess.depthPyramidActive(post.enabled, post)) {
            const v = self.depth_pyramid_pass.render(
                self.depthSampleView(),
                fx_w,
                fx_h,
            );
            if (v.id != 0) {
                depth_pyramid_view = v;
                params.stats.post_draw_calls += self.depth_pyramid_pass.mip_count;
                params.stats.draw_calls += self.depth_pyramid_pass.mip_count;
                params.stats.triangles += 2 * self.depth_pyramid_pass.mip_count;
            }
        }

        var bloom_view: sg.View = .{};
        if (postprocess.bloomPyramidActive(post.enabled, post)) {
            const v = self.bloom_pass.render(
                self.postprocess_pass.offscreen_resolve_tex_view,
                post.bloom.threshold,
                post.bloom.mips,
                post.bloom.radius,
                fx_w,
                fx_h,
            );
            if (v.id != 0) {
                bloom_view = v;
                const mips = postprocess.clampBloomMips(post.bloom.mips);
                const bloom_draws = 2 * @as(u32, mips) - 1;
                params.stats.post_draw_calls += bloom_draws;
                params.stats.draw_calls += bloom_draws;
            }
        }
        self.postprocess_pass.setBloomTexture(bloom_view);

        var glow_view: sg.View = .{};
        if (postprocess.glowActive(post.enabled, post)) {
            const v = self.glow_pass.render(
                self.postprocess_pass.offscreen_resolve_tex_view,
                post.glow.threshold,
                post.glow.radius,
                fx_w,
                fx_h,
            );
            if (v.id != 0) {
                glow_view = v;
                params.stats.post_draw_calls += postprocess.GLOW_PASS_DRAWS;
                params.stats.draw_calls += postprocess.GLOW_PASS_DRAWS;
                params.stats.triangles += 2 * postprocess.GLOW_PASS_DRAWS;
            }
        }
        self.postprocess_pass.setGlowTexture(glow_view);

        var highlight_view: sg.View = .{};
        var highlight_mask_view: sg.View = .{};
        if (postprocess.highlightActive(post.enabled, params.highlight_items.len)) {
            const hl_rect = params.highlight_viewport.toPixelRect(fx_w, fx_h);
            const hl = self.highlight_pass.render(
                params.view_proj,
                params.highlight_items,
                fx_w,
                fx_h,
                hl_rect,
            );
            if (hl.view.id != 0) {
                highlight_view = hl.view;
                highlight_mask_view = hl.mask_view;
                params.stats.post_draw_calls += hl.mask_draws + passes.highlight_pass.HIGHLIGHT_BLUR_DRAWS;
                params.stats.draw_calls += hl.mask_draws + passes.highlight_pass.HIGHLIGHT_BLUR_DRAWS;
                params.stats.triangles += hl.mask_tris + 2 * passes.highlight_pass.HIGHLIGHT_BLUR_DRAWS;
            }
        }
        self.postprocess_pass.setHighlightTexture(highlight_view);
        self.postprocess_pass.setHighlightMaskTexture(highlight_mask_view);

        var shaft_view: sg.View = .{};
        if (postprocess.shaftActive(post.enabled, post, params.shadows_enabled)) {
            const shaft_inv_view_proj = params.view_proj.invert() orelse Mat4.identity;
            const v = self.volumetric_pass.render(.{
                .depth_view = self.depthSampleView(),
                .shadow_view = params.shaft_shadow_view,
                .inv_view_proj = shaft_inv_view_proj,
                .camera_pos = params.eye,
                .sun_dir = params.sun_dir,
                .sun_color = params.sun_color,
                .splits = params.shaft_splits,
                .cascades = params.shaft_cascades,
                .shadow_bias = params.shaft_shadow_bias,
                .config = post,
                .base_w = fx_w,
                .base_h = fx_h,
            });
            if (v.id != 0) {
                shaft_view = v;
                params.stats.post_draw_calls += postprocess.SHAFT_PASS_DRAWS;
                params.stats.draw_calls += postprocess.SHAFT_PASS_DRAWS;
                params.stats.triangles += 2 * postprocess.SHAFT_PASS_DRAWS;
            }
        }
        self.postprocess_pass.setShaftTexture(shaft_view);

        // (inv_view_proj / prev_vp / vel_view_in are hoisted above the
        // SSAO block: the AO temporal stage, the SSGI stage below, and
        // the composite all reproject against the same target.)

        // SSGI v2 (half-res pass): gather + optional temporal accumulation
        // + bilateral denoise before the composite, mirroring the SSAO
        // stage above. Runs only in pass mode (ssgiPassActive: master gate
        // + half_res_pass); the inline v1 lane packs zeros in this mode so
        // the effect never double-adds. Sizing is render-scale-aware: the
        // pass derives half of the EFFECTIVE size (fx_*, never the
        // swapchain size). MSAA suppression clears ssgi_enabled upstream,
        // so pass mode joins the existing suppression with no extra branch
        // here. The temporal flag in inline mode warns once and is ignored
        // (ssgiTemporalActive gates the pass stage itself).
        var ssgi_view: sg.View = .{};
        if (post.enabled and postprocess.ssgiPassActive(post)) {
            if (post.ssgi.temporal.enabled and post.ssgi.mode != .half_res_pass) {
                _ = self.warn_ssgi_temporal.warn("ssgi: ssgi_temporal_enabled needs ssgi_mode=half_res_pass; ignoring temporal", .{});
            }
            const v = self.ssgi_pass.render(.{
                .scene_view = self.postprocess_pass.offscreen_resolve_tex_view,
                .depth_view = self.depthSampleView(),
                .velocity_view = vel_view_in,
                .velocity_valid = vel_view_in.id != 0,
                .view_proj = params.view_proj,
                .inv_view_proj = inv_view_proj,
                .prev_view_proj = prev_vp,
                .camera_pos = params.eye,
                .config = post,
                .camera_cut = post.taa.camera_cut,
                .near_z = params.camera.getNear(),
                .far_z = params.camera.getFar(),
                .base_w = fx_w,
                .base_h = fx_h,
            });
            if (v.id != 0) {
                ssgi_view = v;
                const ssgi_draws: u32 = if (postprocess.ssgiTemporalActive(post)) 3 else 2;
                params.stats.post_draw_calls += ssgi_draws;
                params.stats.draw_calls += ssgi_draws;
                params.stats.triangles += 2 * ssgi_draws;
            }
        }
        self.postprocess_pass.setSsgiTexture(ssgi_view);

        // SSR v2 (half-res pass): trace + bilateral denoise before the
        // composite, mirroring the SSGI stage above. The final depth-aware
        // upscale is a 2x2 gather inside the composite (sampleSSRPass), so
        // the pass stays at 2 draws and 2 half-res targets. Runs only in
        // pass mode (ssrPassActive: master gate + half_res_pass); the
        // inline v1 lanes pack zeros in this mode so the effect never
        // double-adds.
        // Sizing is render-scale-aware: the pass derives half of the
        // EFFECTIVE size (fx_*, never the swapchain size). MSAA suppression
        // clears ssr.enabled upstream, so pass mode joins the existing
        // suppression with no extra branch here.
        var ssr_view: sg.View = .{};
        if (post.enabled and postprocess.ssrPassActive(post)) {
            const v = self.ssr_pass.render(.{
                .scene_view = self.postprocess_pass.offscreen_resolve_tex_view,
                .depth_view = self.depthSampleView(),
                .view_proj = params.view_proj,
                .inv_view_proj = inv_view_proj,
                .camera_pos = params.eye,
                .config = post,
                .near_z = params.camera.getNear(),
                .far_z = params.camera.getFar(),
                .base_w = fx_w,
                .base_h = fx_h,
            });
            if (v.id != 0) {
                ssr_view = v;
                params.stats.post_draw_calls += 2;
                params.stats.draw_calls += 2;
                params.stats.triangles += 4;
            }
        }
        self.postprocess_pass.setSsrTexture(ssr_view);

        var taa_history_view = self.postprocess_pass.offscreen_resolve_tex_view;
        var taa_history_valid = false;
        var taa_capture = false;
        var taa_reset_now = false;
        if (taa_active) {
            const recreated = self.postprocess_pass.ensureTaaHistory(cur_w, cur_h);
            self.postprocess_pass.taa_read = postprocess.taaReadIndex(self.taa_frame);
            const reset = postprocess.taaShouldReset(.{
                .first_frame = !self.taa_has_history,
                .toggled_on = !self.taa_enabled_prev,
                .resized = recreated,
                .camera_cut = post.taa.camera_cut,
                .explicit_reset = self.taa_explicit_reset,
            });
            taa_reset_now = reset;
            if (self.postprocess_pass.taaReadView().id != 0) {
                taa_history_view = self.postprocess_pass.taaReadView();
            }
            taa_capture = self.postprocess_pass.taaWriteAttView().id != 0;
            taa_history_valid = !reset and taa_capture and taa_history_view.id != 0;
            if (!taa_capture) taa_history_valid = false;
        }

        // Cut-frame temporal suppression: on a TAA reset (first frame,
        // toggle, resize, camera cut, explicit reset) the velocity target
        // still holds pre-cut vectors, so bind the zero mask and skip the
        // blur gather for exactly this composite — TAA already discards
        // history via taa_history_valid, the blur needs the same one-frame
        // gate. Under blur-only (TAA off) a camera cut suppresses the same
        // way. Steady frames are untouched (no per-frame reset).
        const suppress_velocity = passes.velocity_pass.cutSuppressesVelocity(
            taa_reset_now,
            post.motion_blur.enabled,
            post.taa.camera_cut,
        );
        if (suppress_velocity) post.motion_blur.enabled = false;
        // Cut-frame gate over the hoisted velocity view (see above): a TAA
        // reset or a blur-only camera cut binds the zero fallback and skips
        // the blur gather for exactly this composite.
        const vel_view = if (suppress_velocity) sg.View{} else vel_view_in;

        var swap_action = sg.PassAction{};
        swap_action.colors[0] = .{ .load_action = .DONTCARE };
        sg.beginPass(.{
            .action = swap_action,
            .swapchain = sglue.swapchain(),
        });

        self.postprocess_pass.render(
            post,
            ssao_view,
            ssao.enabled,
            ssao.intensity,
            fx_w,
            fx_h,
            params.view_proj,
            inv_view_proj,
            prev_vp,
            params.eye,
            params.sun_dir,
            params.sun_color,
            params.camera.getNear(),
            params.camera.getFar(),
            self.depthSampleView(),
            taa_history_view,
            taa_history_valid,
            false,
            vel_view,
            reproj_mat_unjittered,
            ssgi_albedo_view_in,
            params.velocity_valid,
        );
        self.prev_view_proj = params.view_proj_unjittered;
        self.has_prev_view_proj = true;
        params.stats.post_draw_calls += 1;
        params.stats.draw_calls += 1;
        params.stats.triangles += 2;

        if (params.ui) |frame| {
            frame.drawPrepared();
            params.stats.post_draw_calls += 1;
            params.stats.draw_calls += 1;
        }

        sg.endPass();

        if (taa_active and taa_capture) {
            var cap_action = sg.PassAction{};
            cap_action.colors[0] = .{ .load_action = .DONTCARE };
            var cap_pass = sg.Pass{ .action = cap_action };
            cap_pass.attachments.colors[0] = self.postprocess_pass.taaWriteAttView();
            sg.beginPass(cap_pass);
            self.postprocess_pass.render(
                post,
                ssao_view,
                ssao.enabled,
                ssao.intensity,
                fx_w,
                fx_h,
                params.view_proj,
                inv_view_proj,
                prev_vp,
                params.eye,
                params.sun_dir,
                params.sun_color,
                params.camera.getNear(),
                params.camera.getFar(),
                self.depthSampleView(),
                taa_history_view,
                taa_history_valid,
                true,
                vel_view,
                reproj_mat_unjittered,
                ssgi_albedo_view_in,
                params.velocity_valid,
            );
            sg.endPass();
            self.taa_frame += 1;
            self.taa_has_history = true;
            params.stats.post_draw_calls += 1;
            params.stats.draw_calls += 1;
            params.stats.triangles += 2;
        }
        if (taa_active) {
            if (!taa_capture) self.taa_has_history = false;
            self.taa_enabled_prev = true;
            self.taa_explicit_reset = false;
        }
    }

    pub fn updateAutoExposure(self: *PostFXStack, avg_lum: f32, opts: postprocess.AutoExposureOptions, dt: f32) f32 {
        return self.auto_exposure.update(avg_lum, opts, dt);
    }

    pub fn updateAutoExposureFromBuffer(self: *PostFXStack, buffer: []const [3]f32, weights: ?[]const f32, opts: postprocess.AutoExposureOptions, dt: f32) f32 {
        return self.auto_exposure.updateFromBuffer(buffer, weights, opts, dt);
    }

    pub fn updateAutoExposureFromHistogram(self: *PostFXStack, hist: *const postprocess.LuminanceHistogram, opts: postprocess.AutoExposureOptions, dt: f32) f32 {
        return self.auto_exposure.updateFromHistogram(hist, opts, dt);
    }

    pub fn getAdaptedExposure(self: *const PostFXStack) f32 {
        return self.auto_exposure.adapted_exposure;
    }

    pub fn resetAutoExposure(self: *PostFXStack) void {
        self.auto_exposure.reset();
    }

    pub fn depthPyramidView(self: *const PostFXStack) sg.View {
        return self.depth_pyramid_pass.mipView(0);
    }

    pub fn depthPyramidMip(self: *const PostFXStack, level: usize) sg.View {
        return self.depth_pyramid_pass.mipView(level);
    }
};
