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
    bloom_pass: passes.BloomPass,
    glow_pass: passes.GlowPass,
    highlight_pass: passes.HighlightPass,
    volumetric_pass: passes.VolumetricPass = .{},
    outline_pass: passes.OutlinePass,

    outline_msaa: ?passes.OutlinePass = null,
    msaa_depth: ?passes.MsaaDepthPass = null,

    main_samples: i32 = 1,
    main_color_format: sg.PixelFormat = .RGBA16F,

    warn_depth_effects: msaa.WarnOnce = .{},
    warn_taa_target: msaa.WarnOnce = .{},

    outline_enabled: bool = false,
    outline_color: Color4 = Color4.new(1.0, 0.5, 0.0, 1.0),
    outline_width_px: f32 = 2.0,

    prev_view_proj: Mat4 = Mat4.identity,
    has_prev_view_proj: bool = false,

    taa_frame: u64 = 0,
    taa_has_history: bool = false,
    taa_enabled_prev: bool = false,
    taa_explicit_reset: bool = false,

    warn_taa_msaa: msaa.WarnOnce = .{},

    pub fn init() PostFXStack {
        return .{
            .postprocess_pass = passes.PostProcessPass.init(),
            .ssao_pass = passes.SSAOPass.init(),
            .bloom_pass = passes.BloomPass.init(),
            .glow_pass = passes.GlowPass.init(),
            .highlight_pass = passes.HighlightPass.init(),
            .volumetric_pass = passes.VolumetricPass.init(),
            .outline_pass = passes.OutlinePass.init(1, .RGBA16F),
        };
    }

    pub fn deinit(self: *PostFXStack) void {
        self.postprocess_pass.deinit();
        self.ssao_pass.deinit();
        self.bloom_pass.deinit();
        self.glow_pass.deinit();
        self.highlight_pass.deinit();
        self.volumetric_pass.deinit();
        self.outline_pass.deinit();
        if (self.outline_msaa) |*op| op.deinit();
        self.outline_msaa = null;
        self.destroyMsaaDepth();
    }

    /// Window-resize path: the main target is mandatory, auxiliary targets
    /// only when already allocated (avoids provisioning VRAM for effects
    /// that never ran). Highlight/shaft targets stay lazy under render().
    pub fn resizeAll(self: *PostFXStack, width: i32, height: i32) void {
        if (width <= 0 or height <= 0) return;
        if (!sg.isvalid()) return;
        _ = self.postprocess_pass.resize(width, height, self.main_samples);
        if (self.ssao_pass.width != 0) self.ssao_pass.resize(width, height);
        if (self.bloom_pass.base_width != 0) self.bloom_pass.resize(width, height);
        if (self.glow_pass.base_width != 0) self.glow_pass.resize(width, height);
        passes.OutlinePass.resize(width, height);
    }

    pub fn destroyMsaaDepth(self: *PostFXStack) void {
        if (self.msaa_depth) |*md| md.deinit();
        self.msaa_depth = null;
    }

    pub fn renderMsaaDepthPrepass(
        self: *PostFXStack,
        view_proj: Mat4,
        queues: *const scene_render_queue.RenderQueues,
        skins: []const [scene_render_queue.MAX_BONES]Mat4,
        viewport: Viewport,
        cur_w: i32,
        cur_h: i32,
        stats: *SceneStats,
    ) void {
        if (!sg.isvalid()) return;
        if (self.msaa_depth == null) self.msaa_depth = passes.MsaaDepthPass.init();
        const md = &self.msaa_depth.?;
        _ = md.ensure(cur_w, cur_h);
        const rect = viewport.toPixelRect(cur_w, cur_h);
        sg.applyViewport(rect.x, rect.y, rect.width, rect.height, true);
        sg.applyScissorRect(rect.x, rect.y, rect.width, rect.height, true);
        md.render(view_proj, queues, skins, stats);
        sg.applyViewport(0, 0, cur_w, cur_h, true);
        sg.applyScissorRect(0, 0, cur_w, cur_h, true);
    }

    pub fn depthSampleView(self: *const PostFXStack, prepass_active: bool) sg.View {
        if (prepass_active) {
            if (self.msaa_depth) |*md| {
                if (md.depthTexView().id != 0) return md.depthTexView();
            }
        }
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
    /// history when applicable. Fails closed with a clear error before any
    /// pass begins; never provisions an alternate color path.
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
        const samples = targetSampleCount(requested);
        if (!self.postprocess_pass.resize(width, height, samples)) return error.TargetAllocationFailed;
        self.main_samples = samples;
        self.main_color_format = .RGBA16F;
        if (config.taa_enabled and samples == 1) {
            if (self.postprocess_pass.ensureTaaHistory(width, height)) self.taa_explicit_reset = true;
            if (!self.postprocess_pass.taaAvailable()) {
                _ = self.warn_taa_target.warn("TAA: history allocation failed; continuing with spatial AA", .{});
                config.taa_enabled = false;
                config.fxaa_enabled = true;
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
        eye: Vec3,
        sun_dir: Vec3,
        sun_color: Color3,
        default_white_view: sg.View,
        stats: *SceneStats,
        main_samples: i32 = 1,
        msaa_depth_prepass: bool = false,
        highlight_items: []const passes.HighlightDrawItem = &.{},
        highlight_viewport: Viewport = .{},
        shaft_shadow_view: sg.View = .{},
        shaft_cascades: [4]Mat4 = [_]Mat4{Mat4.identity} ** 4,
        shaft_splits: [4]f32 = .{ 0, 0, 0, 0 },
        shaft_shadow_bias: f32 = 0.0,
        shadows_enabled: bool = false,
        ui: ?*const UiFrame = null,
    };

    /// Effect stages plus the display pass. The display pass always runs;
    /// effect stages run only while the master switch and their own flag
    /// are on. Empty views (.{} ) mark inactive/failed effect results; the
    /// display pass binds its own placeholder for those slots.
    pub fn renderChain(self: *PostFXStack, params: ChainParams, cur_w: i32, cur_h: i32) void {
        var post = params.post.forFrame();
        var ssao = params.ssao;
        if (!params.post.enabled) {
            ssao.enabled = false;
            ssao.debug_mode = false;
        }
        const msaa_active = params.main_samples > 1;
        const depth_prepass = msaa.depthPrepassActive(post.enabled, params.msaa_depth_prepass, params.main_samples);
        if (msaa.suppressDepthEffects(params.main_samples, params.msaa_depth_prepass)) {
            if (msaa.depthEffectsActive(true, ssao.enabled, ssao.debug_mode, post.ssr_enabled, post.dof_enabled, post.fog_enabled, post.motion_blur_enabled) or post.shaft_enabled) {
                _ = self.warn_depth_effects.warn(
                    "msaa: SSAO/SSR/DoF/Fog/MotionBlur/Shaft disabled this session: MSAA x{} main target has no depth resolve",
                    .{params.main_samples},
                );
            }
            ssao.enabled = false;
            ssao.debug_mode = false;
            post.ssr_enabled = false;
            post.dof_enabled = false;
            post.fog_enabled = false;
            post.motion_blur_enabled = false;
            post.fxaa_enabled = false;
            post.shaft_enabled = false;
        }
        if (msaa_active and post.taa_enabled) {
            _ = self.warn_taa_msaa.warn(
                "msaa: TAA disabled this session: MSAA x{} main target has no depth resolve",
                .{params.main_samples},
            );
            post.taa_enabled = false;
        }
        const taa_active = post.taa_enabled;
        if (!taa_active) {
            self.taa_has_history = false;
            self.taa_enabled_prev = false;
            self.taa_explicit_reset = false;
        }

        var ssao_view = params.default_white_view;
        const ssao_active = ssao.enabled or ssao.debug_mode;
        if (post.enabled and ssao_active) {
            self.ssao_pass.render(
                params.camera,
                params.aspect,
                self.depthSampleView(depth_prepass),
                ssao,
                cur_w,
                cur_h,
            );
            const out = self.ssao_pass.ssao_blur_tex_view;
            if (out.id != 0) {
                ssao_view = out;
                params.stats.post_draw_calls += 2;
                params.stats.draw_calls += 2;
                params.stats.triangles += 4;
            }
        }

        var bloom_view: sg.View = .{};
        if (postprocess.bloomPyramidActive(post.enabled, post)) {
            const v = self.bloom_pass.render(
                self.postprocess_pass.offscreen_resolve_tex_view,
                post.bloom_threshold,
                post.bloom_pyramid_mips,
                post.bloom_radius,
                cur_w,
                cur_h,
            );
            if (v.id != 0) {
                bloom_view = v;
                const mips = postprocess.clampBloomMips(post.bloom_pyramid_mips);
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
                post.glow_threshold,
                post.glow_radius,
                cur_w,
                cur_h,
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
            const hl_rect = params.highlight_viewport.toPixelRect(cur_w, cur_h);
            const hl = self.highlight_pass.render(
                params.view_proj,
                params.highlight_items,
                cur_w,
                cur_h,
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
            const inv_view_proj = params.view_proj.invert() orelse Mat4.identity;
            const v = self.volumetric_pass.render(.{
                .depth_view = self.depthSampleView(depth_prepass),
                .shadow_view = params.shaft_shadow_view,
                .inv_view_proj = inv_view_proj,
                .camera_pos = params.eye,
                .sun_dir = params.sun_dir,
                .sun_color = params.sun_color,
                .splits = params.shaft_splits,
                .cascades = params.shaft_cascades,
                .shadow_bias = params.shaft_shadow_bias,
                .config = post,
                .base_w = cur_w,
                .base_h = cur_h,
            });
            if (v.id != 0) {
                shaft_view = v;
                params.stats.post_draw_calls += postprocess.SHAFT_PASS_DRAWS;
                params.stats.draw_calls += postprocess.SHAFT_PASS_DRAWS;
                params.stats.triangles += 2 * postprocess.SHAFT_PASS_DRAWS;
            }
        }
        self.postprocess_pass.setShaftTexture(shaft_view);

        const inv_view_proj = params.view_proj.invert() orelse Mat4.identity;
        const prev_vp = if (!self.has_prev_view_proj) params.view_proj else self.prev_view_proj;

        var taa_history_view = self.postprocess_pass.offscreen_resolve_tex_view;
        var taa_history_valid = false;
        var taa_capture = false;
        if (taa_active) {
            const recreated = self.postprocess_pass.ensureTaaHistory(cur_w, cur_h);
            self.postprocess_pass.taa_read = postprocess.taaReadIndex(self.taa_frame);
            const reset = postprocess.taaShouldReset(.{
                .first_frame = !self.taa_has_history,
                .toggled_on = !self.taa_enabled_prev,
                .resized = recreated,
                .camera_cut = post.taa_camera_cut,
                .explicit_reset = self.taa_explicit_reset,
            });
            if (self.postprocess_pass.taaReadView().id != 0) {
                taa_history_view = self.postprocess_pass.taaReadView();
            }
            taa_capture = self.postprocess_pass.taaWriteAttView().id != 0;
            taa_history_valid = !reset and taa_capture and taa_history_view.id != 0;
            if (!taa_capture) taa_history_valid = false;
        }

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
            ssao.debug_mode,
            ssao.intensity,
            cur_w,
            cur_h,
            params.view_proj,
            inv_view_proj,
            prev_vp,
            params.eye,
            params.sun_dir,
            params.sun_color,
            params.camera.getNear(),
            params.camera.getFar(),
            self.depthSampleView(depth_prepass),
            taa_history_view,
            taa_history_valid,
            false,
        );
        self.prev_view_proj = params.view_proj;
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
                ssao.debug_mode,
                ssao.intensity,
                cur_w,
                cur_h,
                params.view_proj,
                inv_view_proj,
                prev_vp,
                params.eye,
                params.sun_dir,
                params.sun_color,
                params.camera.getNear(),
                params.camera.getFar(),
                self.depthSampleView(depth_prepass),
                taa_history_view,
                taa_history_valid,
                true,
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
};
