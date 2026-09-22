const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const sglue = sokol.glue;

const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const Color3 = math.Color3;
const Color4 = math.Color4;

const Mesh = @import("../mesh.zig").Mesh;
const Camera = @import("../camera.zig").Camera;
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

/// Screen-space post-processing stack: the offscreen main-pass target, the
/// fullscreen composite pass, SSAO, the bloom mip pyramid, the glow layer,
/// and the inverse-hull outline settings. Scene feeds the per-frame configs
/// (post_process / ssao stay public Scene config fields) through
/// ChainParams; everything GPU-owned lives here. The list of highlighted
/// meshes stays on Scene (content registry, mock-constructed in tests).
pub const PostFXStack = struct {
    postprocess_pass: passes.PostProcessPass,
    ssao_pass: passes.SSAOPass,
    bloom_pass: passes.BloomPass,
    glow_pass: passes.GlowPass,
    highlight_pass: passes.HighlightPass,
    outline_pass: passes.OutlinePass,

    // MSAA twin of the outline pass (its pipelines must match the main
    // target's sample count). Lazily created on the first MSAA frame,
    // recreated if the count changes; null keeps the 1x-only memory shape.
    outline_msaa: ?passes.OutlinePass = null,

    // Effective main-target sample count of the current/last frame. Kept so
    // resizeAll (window resize path outside render()) can keep the target
    // shape stable between frames.
    main_samples: i32 = 1,

    // One-shot policy warnings (see scene/msaa.zig for the reasoning).
    warn_depth_effects: msaa.WarnOnce = .{},

    // Inverse-hull outline (highlight layer) settings.
    outline_enabled: bool = false,
    outline_color: Color4 = Color4.new(1.0, 0.5, 0.0, 1.0),
    outline_width_px: f32 = 2.0,

    // Previous frame view_proj matrix for camera motion blur reprojection
    // (and the TAA history reprojection, which reuses this exact plumbing:
    // renderChain stores the jittered current VP here every composite, so
    // prev is always the previous frame's jittered matrix).
    prev_view_proj: Mat4 = Mat4.identity,
    has_prev_view_proj: bool = false,

    // TAA resolve state. History storage lives in PostProcessPass; this is
    // the frame counter (ping-pong slots via taaReadIndex/taaWriteIndex, so
    // skipped/reused snapshot ids can never alias read/write), the validity
    // latch, and the explicit-reset request. Reset triggers (see
    // postprocess.zig header docs): resize, taaReset(), taa_camera_cut,
    // first frame, off->on toggle. MSAA forces TAA off (no depth resolve).
    taa_frame: u64 = 0,
    taa_has_history: bool = false,
    taa_enabled_prev: bool = false,
    taa_explicit_reset: bool = false,

    // One-shot TAA policy warning (MSAA active => TAA suppressed).
    warn_taa_msaa: msaa.WarnOnce = .{},

    pub fn init() PostFXStack {
        return .{
            .postprocess_pass = passes.PostProcessPass.init(),
            .ssao_pass = passes.SSAOPass.init(),
            .bloom_pass = passes.BloomPass.init(),
            .glow_pass = passes.GlowPass.init(),
            .highlight_pass = passes.HighlightPass.init(),
            .outline_pass = passes.OutlinePass.init(),
        };
    }

    pub fn deinit(self: *PostFXStack) void {
        self.postprocess_pass.deinit();
        self.ssao_pass.deinit();
        self.bloom_pass.deinit();
        self.glow_pass.deinit();
        self.highlight_pass.deinit();
        self.outline_pass.deinit();
        if (self.outline_msaa) |*op| op.deinit();
        self.outline_msaa = null;
    }

    /// Resizes every viewport-sized offscreen target (window resize path).
    pub fn resizeAll(self: *PostFXStack, width: i32, height: i32) void {
        self.postprocess_pass.resize(width, height, self.main_samples);
        self.ssao_pass.resize(width, height);
        self.bloom_pass.resize(width, height);
        self.glow_pass.resize(width, height);
        self.highlight_pass.resize(width, height);
        passes.OutlinePass.resize(width, height);
    }

    /// Explicit TAA history reset for the next composite. Context thread
    /// only (it destroys the history targets via PostProcessPass.taaReset).
    /// Update-thread camera cuts use PostProcessOptions.taa_camera_cut
    /// (one frame, snapshot-carried) instead.
    pub fn taaReset(self: *PostFXStack) void {
        self.taa_explicit_reset = true;
        self.postprocess_pass.taaReset();
    }

    /// Begins PASS 2 (the main scene pass) either into the offscreen target
    /// (post-processing on) or straight into the swapchain. `samples` is the
    /// effective main-target sample count from scene/msaa.zig (1 = legacy
    /// shape). With samples > 1 the pass carries a resolve attachment: sokol
    /// resolves MSAA color into it at end of pass, so the MSAA color content
    /// itself is DONTCARE-stored, as is the MSAA depth (write-only for the
    /// post chain by design).
    pub fn beginMainPass(self: *PostFXStack, main_pass_action: sg.PassAction, post_enabled: bool, samples: i32, width: i32, height: i32) void {
        if (post_enabled) {
            self.main_samples = samples;
            self.postprocess_pass.resize(width, height, samples);
            self.bloom_pass.resize(width, height);
            self.glow_pass.resize(width, height);
            self.highlight_pass.resize(width, height);
            var offscreen_pass = sg.Pass{
                .action = main_pass_action,
            };
            offscreen_pass.attachments.colors[0] = self.postprocess_pass.offscreen_color_att_view;
            offscreen_pass.attachments.depth_stencil = self.postprocess_pass.offscreen_depth_att_view;
            if (samples > 1) {
                offscreen_pass.attachments.resolves[0] = self.postprocess_pass.offscreen_resolve_att_view;
                offscreen_pass.action.colors[0].store_action = .DONTCARE;
                offscreen_pass.action.depth.store_action = .DONTCARE;
            }
            sg.beginPass(offscreen_pass);
        } else {
            self.main_samples = 1;
            sg.beginPass(.{
                .action = main_pass_action,
                .swapchain = sglue.swapchain(),
            });
        }

        // Viewport for pixel-width outline expansion (static, shared).
        passes.OutlinePass.resize(width, height);
    }

    /// Inverse-hull outline for highlighted meshes: inside the main pass,
    /// depth-tested, no depth write, drawn after all surface geometry. The
    /// pass variant must match the main target's sample count.
    pub fn renderOutline(self: *PostFXStack, view_proj: Mat4, eye: Vec3, outline_meshes: []const *Mesh, samples: i32, stats: *SceneStats) void {
        self.renderOutlineExplicit(view_proj, eye, outline_meshes, samples, stats, self.outline_enabled, self.outline_color, self.outline_width_px);
    }

    pub fn renderOutlineExplicit(
        self: *PostFXStack,
        view_proj: Mat4,
        eye: Vec3,
        outline_meshes: []const *Mesh,
        samples: i32,
        stats: *SceneStats,
        enabled: bool,
        color: Color4,
        width_px: f32,
    ) void {
        if (enabled and outline_meshes.len > 0) {
            const pass = self.outlinePassFor(samples);
            pass.render(view_proj, eye, outline_meshes, color, width_px);
            const outline_count: u32 = @intCast(outline_meshes.len);
            stats.main_draw_calls += outline_count;
            stats.draw_calls += outline_count;
        }
    }

    pub fn renderOutlineItems(
        self: *PostFXStack,
        view_proj: Mat4,
        eye: Vec3,
        outline_items: []const passes.OutlineDrawItem,
        outline_skins: []const [scene_render_queue.MAX_BONES]math.Mat4,
        samples: i32,
        stats: *SceneStats,
        enabled: bool,
        color: Color4,
        width_px: f32,
    ) void {
        if (enabled and outline_items.len > 0) {
            const pass = self.outlinePassFor(samples);
            pass.renderItems(view_proj, eye, outline_items, outline_skins, color, width_px);
            const outline_count: u32 = @intCast(outline_items.len);
            stats.main_draw_calls += outline_count;
            stats.draw_calls += outline_count;
        }
    }

    /// Outline pipeline set matching the target sample count; the MSAA twin
    /// is created lazily (and recreated on count changes, e.g. when a
    /// device-specific clamp narrows a requested 8x to 4x).
    fn outlinePassFor(self: *PostFXStack, samples: i32) *passes.OutlinePass {
        if (samples <= 1) return &self.outline_pass;
        if (self.outline_msaa == null or self.outline_msaa.?.sample_count != samples) {
            if (self.outline_msaa) |*op| op.deinit();
            self.outline_msaa = passes.OutlinePass.initSampled(samples);
        }
        return &self.outline_msaa.?;
    }

    // Per-frame inputs for the post chain. Configs travel with the params
    // because they stay public Scene fields (tooling reads/writes them).
    pub const ChainParams = struct {
        post: PostProcessOptions,
        ssao: SSAOOptions,
        camera: Camera,
        aspect: f32,
        view_proj: Mat4,
        eye: Vec3,
        sun_dir: Vec3,
        sun_color: Color3,
        // Placeholder SSAO view when SSAO is off (shared 1x1 white).
        default_white_view: sg.View,
        stats: *SceneStats,
        // Main-target sample count; see scene/msaa.zig for the clamp policy.
        main_samples: i32 = 1,
        // Staged per-mesh highlight items (render-owned snapshots from the
        // frame slot — never live Scene fields — passed through by
        // frame_render from the pinned front slot, same P7 shape as the
        // outline items-view_render consumes inside the main pass). Empty
        // by default: zero items gate the whole PASS 2.85 chain off.
        highlight_items: []const passes.HighlightDrawItem = &.{},
        // Optional 2D overlay drawn on top of the post-processed swapchain.
        // P6: the prepared render-owned frame (upload-free draw), never the
        // live canvas. Intentional low-level break: `?*UICanvas` became
        // `?*const UiFrame` (P6 migration); Scene/UICanvas methods stay
        // stable, only this internal chain signature moves with the frame.
        ui: ?*const UiFrame = null,
    };

    /// PASS 2.5 (SSAO) + PASS 2.75 (bloom pyramid) + PASS 2.8 (glow layer) +
    /// PASS 2.85 (highlight layer) +
    /// PASS 3 (fullscreen composite and UI overlay onto the swapchain). The
    /// composite pass itself only runs when post-processing is enabled —
    /// SSAO/bloom/glow/highlight then just refresh their inputs (legacy behavior kept
    /// verbatim).
    pub fn renderChain(self: *PostFXStack, params: ChainParams, cur_w: i32, cur_h: i32) void {
        // Depth-consuming effects are incompatible with the MSAA main
        // target (no depth resolve in sokol — see scene/msaa.zig); degrade
        // them for the frame on a local copy of the configs.
        const msaa_active = params.main_samples > 1;
        var post = params.post;
        var ssao = params.ssao;
        if (msaa_active) {
            if (msaa.depthEffectsActive(true, ssao.enabled, ssao.debug_mode, post.ssr_enabled, post.dof_enabled, post.fog_enabled)) {
                _ = self.warn_depth_effects.warn(
                    "msaa: SSAO/SSR/DoF/Fog disabled this session: MSAA x{} main target has no depth resolve",
                    .{params.main_samples},
                );
            }
            ssao.enabled = false;
            ssao.debug_mode = false;
            post.ssr_enabled = false;
            post.dof_enabled = false;
            post.fog_enabled = false;
            post.fxaa_enabled = false;
        }
        // TAA needs the 1x depth texture for its reprojection velocity; with
        // an MSAA main target depth is write-only (no resolve in sokol), so
        // TAA is forced off for the frame like the other depth effects.
        if (msaa_active and post.taa_enabled) {
            _ = self.warn_taa_msaa.warn(
                "msaa: TAA disabled this session: MSAA x{} main target has no depth resolve",
                .{params.main_samples},
            );
            post.taa_enabled = false;
        }
        const taa_active = params.post.enabled and post.taa_enabled;
        if (!taa_active) {
            self.taa_has_history = false;
            self.taa_enabled_prev = false;
            self.taa_explicit_reset = false;
        }

        // ==============================================
        // PASS 2.5: SCREEN-SPACE AMBIENT OCCLUSION (SSAO)
        // ==============================================
        var ssao_view = params.default_white_view;
        const ssao_active = ssao.enabled or ssao.debug_mode;
        if (params.post.enabled and ssao_active) {
            self.ssao_pass.render(
                params.camera,
                params.aspect,
                self.postprocess_pass.depthSampleView(),
                ssao,
                cur_w,
                cur_h,
            );
            ssao_view = self.ssao_pass.ssao_blur_tex_view;
            params.stats.post_draw_calls += 2;
            params.stats.draw_calls += 2;
            params.stats.triangles += 4;
        }

        // ==============================================
        // PASS 2.75: BLOOM MIP PYRAMID (optional)
        // ==============================================
        // Multi-pass glow; when disabled the composite shader keeps its legacy
        // in-shader bloom and this binds the resolved scene view placeholder.
        var bloom_view = self.postprocess_pass.offscreen_resolve_tex_view;
        if (postprocess.bloomPyramidActive(params.post.enabled, post)) {
            bloom_view = self.bloom_pass.render(
                self.postprocess_pass.offscreen_resolve_tex_view,
                post.bloom_threshold,
                post.bloom_pyramid_mips,
                cur_w,
                cur_h,
            );
            const mips = postprocess.clampBloomMips(post.bloom_pyramid_mips);
            const bloom_draws = 2 * @as(u32, mips) - 1;
            params.stats.post_draw_calls += bloom_draws;
            params.stats.draw_calls += bloom_draws;
        }
        self.postprocess_pass.setBloomTexture(bloom_view);

        // ==============================================
        // PASS 2.8: GLOW LAYER v1 (optional global halo)
        // ==============================================
        // Threshold extract + separable H/V blur (GLOW_PASS_DRAWS draws);
        // when disabled the composite shader skips the glow block and this
        // binds the resolved scene view placeholder (bit-identical). Runs
        // AFTER the bloom pyramid and composites after bloom's block, so
        // either toggle leaves the other's output unchanged. Samples the
        // resolved scene color (no depth), hence no MSAA suppression. Like
        // bloom it is uniform-only (no sg.updateBuffer, no upload-meter
        // records) and resize-idempotent, so renderReuse replays replay it
        // upload-free.
        var glow_view = self.postprocess_pass.offscreen_resolve_tex_view;
        if (postprocess.glowActive(params.post.enabled, post)) {
            glow_view = self.glow_pass.render(
                self.postprocess_pass.offscreen_resolve_tex_view,
                post.glow_threshold,
                post.glow_radius,
                cur_w,
                cur_h,
            );
            params.stats.post_draw_calls += postprocess.GLOW_PASS_DRAWS;
            params.stats.draw_calls += postprocess.GLOW_PASS_DRAWS;
            params.stats.triangles += 2 * postprocess.GLOW_PASS_DRAWS;
        }
        self.postprocess_pass.setGlowTexture(glow_view);

        // ==============================================
        // PASS 2.85: HIGHLIGHT LAYER v1 (per-mesh inner glow)
        // ==============================================
        // Flat-colored mask fills (one per staged item) + separable H/V
        // blur (HIGHLIGHT_BLUR_DRAWS draws); when inactive (post off or
        // zero staged items) the composite shader skips the highlight block
        // and this binds the empty views (the pass binds the resolved scene
        // view placeholder for both — bit-identical). The composite reads
        // raw mask minus blurred halo (inner-only edge glow, x2 — see
        // highlightInnerGlow in postprocess.zig), so the raw mask view
        // travels with the blurred view under the same active/empty
        // discipline. Runs AFTER the glow
        // stage and composites after glow's block, so either toggle leaves
        // the other's output unchanged. Samples no depth (the mask has no
        // depth attachment), hence no MSAA suppression. Like glow it is
        // uniform-only past its mask binds (no sg.updateBuffer, no
        // upload-meter records) and resize-idempotent, so renderReuse
        // replays replay it upload-free. The mask composites from the
        // primary view only (v1 non-goal: multi-camera highlights follow
        // the primary camera).
        var highlight_view: sg.View = .{};
        var highlight_mask_view: sg.View = .{};
        if (postprocess.highlightActive(params.post.enabled, params.highlight_items.len)) {
            const hl = self.highlight_pass.render(
                params.view_proj,
                params.highlight_items,
                cur_w,
                cur_h,
            );
            highlight_view = hl.view;
            highlight_mask_view = hl.mask_view;
            params.stats.post_draw_calls += hl.mask_draws + passes.highlight_pass.HIGHLIGHT_BLUR_DRAWS;
            params.stats.draw_calls += hl.mask_draws + passes.highlight_pass.HIGHLIGHT_BLUR_DRAWS;
            params.stats.triangles += hl.mask_tris + 2 * passes.highlight_pass.HIGHLIGHT_BLUR_DRAWS;
        }
        self.postprocess_pass.setHighlightTexture(highlight_view);
        self.postprocess_pass.setHighlightMaskTexture(highlight_mask_view);

        // ==============================================
        // PASS 3: FULLSCREEN POST-PROCESSING PASS
        // ==============================================
        if (params.post.enabled) {
            const inv_view_proj = params.view_proj.invert() orelse Mat4.identity;
            const prev_vp = if (!self.has_prev_view_proj) params.view_proj else self.prev_view_proj;

            // TAA resolve inputs. ensureTaaHistory (re)creates the ping-pong
            // pair; any recreation, first frame, off->on toggle, camera-cut
            // flag, or explicit taaReset() invalidates history for the frame
            // (the shader then returns current without sampling history).
            // The read slot follows the internal frame counter, never the
            // snapshot id, so reused/skipped frames cannot alias read/write.
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
                // Degenerate sizes fail ensure: nothing to capture, frame
                // composites with history invalid (retried next frame).
                if (!taa_capture) taa_history_valid = false;
            }

            var swap_action = sg.PassAction{};
            swap_action.colors[0] = .{
                .load_action = .DONTCARE,
            };
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
                taa_history_view,
                taa_history_valid,
                false,
            );
            self.prev_view_proj = params.view_proj;
            self.has_prev_view_proj = true;
            params.stats.post_draw_calls += 1;
            params.stats.draw_calls += 1;
            params.stats.triangles += 2;

            // Render 2D UI overlay on top of post-processed swapchain.
            // Counter semantics unchanged (see Scene.render direct path).
            if (params.ui) |frame| {
                frame.drawPrepared();
                params.stats.post_draw_calls += 1;
                params.stats.draw_calls += 1;
            }

            sg.endPass();

            // TAA history capture, after the swapchain pass closed (and after
            // UI, so HUD pixels never feed the history): re-draw the same
            // composite into the write slot with capture-only set, which
            // stores exactly the post-TAA early-LDR color the main draw fed
            // into the DoF/grade chain. The disabled path never reaches here
            // (taa_active false), so off stays a single pass.
            if (taa_active and taa_capture) {
                var cap_action = sg.PassAction{};
                cap_action.colors[0] = .{
                    .load_action = .DONTCARE,
                };
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
                // Uncaptured frames (degenerate size) stay invalid but latch
                // the toggle so the next healthy frame does not reset-loop.
                if (!taa_capture) self.taa_has_history = false;
                self.taa_enabled_prev = true;
                self.taa_explicit_reset = false;
            }
        }
    }
};
