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
const postprocess = @import("../postprocess.zig");
const PostProcessOptions = postprocess.PostProcessOptions;
const ssao_mod = @import("../ssao.zig");
const SSAOOptions = ssao_mod.SSAOOptions;
const msaa = @import("msaa.zig");
const ui = @import("../ui.zig");
const UICanvas = ui.UICanvas;
const stats_mod = @import("stats.zig");
const SceneStats = stats_mod.SceneStats;

/// Screen-space post-processing stack: the offscreen main-pass target, the
/// fullscreen composite pass, SSAO, the bloom mip pyramid, and the
/// inverse-hull outline settings. Scene feeds the per-frame configs
/// (post_process / ssao stay public Scene config fields) through
/// ChainParams; everything GPU-owned lives here. The list of highlighted
/// meshes stays on Scene (content registry, mock-constructed in tests).
pub const PostFXStack = struct {
    postprocess_pass: passes.PostProcessPass,
    ssao_pass: passes.SSAOPass,
    bloom_pass: passes.BloomPass,
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

    pub fn init() PostFXStack {
        return .{
            .postprocess_pass = passes.PostProcessPass.init(),
            .ssao_pass = passes.SSAOPass.init(),
            .bloom_pass = passes.BloomPass.init(),
            .outline_pass = passes.OutlinePass.init(),
        };
    }

    pub fn deinit(self: *PostFXStack) void {
        self.postprocess_pass.deinit();
        self.ssao_pass.deinit();
        self.bloom_pass.deinit();
        self.outline_pass.deinit();
        if (self.outline_msaa) |*op| op.deinit();
        self.outline_msaa = null;
    }

    /// Resizes every viewport-sized offscreen target (window resize path).
    pub fn resizeAll(self: *PostFXStack, width: i32, height: i32) void {
        self.postprocess_pass.resize(width, height, self.main_samples);
        self.ssao_pass.resize(width, height);
        self.bloom_pass.resize(width, height);
        passes.OutlinePass.resize(width, height);
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
        if (self.outline_enabled and outline_meshes.len > 0) {
            const pass = self.outlinePassFor(samples);
            pass.render(view_proj, eye, outline_meshes, self.outline_color, self.outline_width_px);
            const outline_count: u32 = @intCast(outline_meshes.len);
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
        // Optional 2D overlay drawn on top of the post-processed swapchain.
        ui: ?*UICanvas = null,
    };

    /// PASS 2.5 (SSAO) + PASS 2.75 (bloom pyramid) + PASS 3 (fullscreen
    /// composite and UI overlay onto the swapchain). The composite pass
    /// itself only runs when post-processing is enabled — SSAO/bloom then
    /// just refresh their inputs (legacy behavior kept verbatim).
    pub fn renderChain(self: *PostFXStack, params: ChainParams, cur_w: i32, cur_h: i32) void {
        // Depth-consuming effects are incompatible with the MSAA main
        // target (no depth resolve in sokol — see scene/msaa.zig); degrade
        // them for the frame on a local copy of the configs.
        const msaa_active = params.main_samples > 1;
        var post = params.post;
        var ssao = params.ssao;
        if (msaa_active) {
            if (msaa.depthEffectsActive(true, ssao.enabled, ssao.debug_mode, post.ssr_enabled, post.dof_enabled)) {
                _ = self.warn_depth_effects.warn(
                    "msaa: SSAO/SSR/DoF disabled this session: MSAA x{} main target has no depth resolve",
                    .{params.main_samples},
                );
            }
            ssao.enabled = false;
            ssao.debug_mode = false;
            post.ssr_enabled = false;
            post.dof_enabled = false;
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
        if (params.post.enabled and post.bloom_enabled and post.bloom_pyramid) {
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
        // PASS 3: FULLSCREEN POST-PROCESSING PASS
        // ==============================================
        if (params.post.enabled) {
            var swap_action = sg.PassAction{};
            swap_action.colors[0] = .{
                .load_action = .DONTCARE,
            };
            sg.beginPass(.{
                .action = swap_action,
                .swapchain = sglue.swapchain(),
            });

            const inv_view_proj = params.view_proj.invert() orelse Mat4.identity;

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
                params.eye,
                params.sun_dir,
                params.sun_color,
                params.camera.getNear(),
                params.camera.getFar(),
            );
            params.stats.post_draw_calls += 1;
            params.stats.draw_calls += 1;
            params.stats.triangles += 2;

            // Render 2D UI overlay on top of post-processed swapchain
            if (params.ui) |ui_c| {
                ui_c.render(@floatFromInt(cur_w), @floatFromInt(cur_h));
                params.stats.post_draw_calls += 1;
                params.stats.draw_calls += 1;
            }

            sg.endPass();
        }
    }
};
