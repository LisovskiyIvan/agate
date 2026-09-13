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
const PostProcessConfig = postprocess.PostProcessConfig;
const ssao_mod = @import("../ssao.zig");
const SSAOConfig = ssao_mod.SSAOConfig;
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
    }

    /// Resizes every viewport-sized offscreen target (window resize path).
    pub fn resizeAll(self: *PostFXStack, width: i32, height: i32) void {
        self.postprocess_pass.resize(width, height);
        self.ssao_pass.resize(width, height);
        self.bloom_pass.resize(width, height);
        passes.OutlinePass.resize(width, height);
    }

    /// Begins PASS 2 (the main scene pass) either into the offscreen
    /// MSAA target (post-processing on) or straight into the swapchain.
    /// Then refreshes the outline pass viewport, exactly in legacy order.
    pub fn beginMainPass(self: *PostFXStack, main_pass_action: sg.PassAction, post_enabled: bool, width: i32, height: i32) void {
        if (post_enabled) {
            self.postprocess_pass.resize(width, height);
            self.bloom_pass.resize(width, height);
            var offscreen_pass = sg.Pass{
                .action = main_pass_action,
            };
            offscreen_pass.attachments.colors[0] = self.postprocess_pass.offscreen_color_att_view;
            offscreen_pass.attachments.depth_stencil = self.postprocess_pass.offscreen_depth_att_view;
            if (self.postprocess_pass.sample_count > 1) {
                offscreen_pass.attachments.resolves[0] = self.postprocess_pass.offscreen_resolve_att_view;
            }
            sg.beginPass(offscreen_pass);
        } else {
            sg.beginPass(.{
                .action = main_pass_action,
                .swapchain = sglue.swapchain(),
            });
        }

        // Viewport for pixel-width outline expansion (static, shared).
        passes.OutlinePass.resize(width, height);
    }

    /// Inverse-hull outline for highlighted meshes: inside the main pass,
    /// depth-tested, no depth write, drawn after all surface geometry.
    pub fn renderOutline(self: *PostFXStack, view_proj: Mat4, eye: Vec3, outline_meshes: []const *Mesh, stats: *SceneStats) void {
        if (self.outline_enabled and outline_meshes.len > 0) {
            self.outline_pass.render(view_proj, eye, outline_meshes, self.outline_color, self.outline_width_px);
            const outline_count: u32 = @intCast(outline_meshes.len);
            stats.main_draw_calls += outline_count;
            stats.draw_calls += outline_count;
        }
    }

    // Per-frame inputs for the post chain. Configs travel with the params
    // because they stay public Scene fields (tooling reads/writes them).
    pub const ChainParams = struct {
        post: PostProcessConfig,
        ssao: SSAOConfig,
        camera: Camera,
        aspect: f32,
        view_proj: Mat4,
        eye: Vec3,
        sun_dir: Vec3,
        sun_color: Color3,
        // Placeholder SSAO view when SSAO is off (shared 1x1 white).
        default_white_view: sg.View,
        // Optional 2D overlay drawn on top of the post-processed swapchain.
        ui: ?*UICanvas = null,
        stats: *SceneStats,
    };

    /// PASS 2.5 (SSAO) + PASS 2.75 (bloom pyramid) + PASS 3 (fullscreen
    /// composite and UI overlay onto the swapchain). The composite pass
    /// itself only runs when post-processing is enabled — SSAO/bloom then
    /// just refresh their inputs (legacy behavior kept verbatim).
    pub fn renderChain(self: *PostFXStack, params: ChainParams, cur_w: i32, cur_h: i32) void {
        // ==============================================
        // PASS 2.5: SCREEN-SPACE AMBIENT OCCLUSION (SSAO)
        // ==============================================
        var ssao_view = params.default_white_view;
        const ssao_active = params.ssao.enabled or params.ssao.debug_mode;
        if (params.post.enabled and ssao_active) {
            self.ssao_pass.render(
                params.camera,
                params.aspect,
                self.postprocess_pass.offscreen_depth_tex_view,
                params.ssao,
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
        if (params.post.enabled and params.post.bloom_enabled and params.post.bloom_pyramid) {
            bloom_view = self.bloom_pass.render(
                self.postprocess_pass.offscreen_resolve_tex_view,
                params.post.bloom_threshold,
                params.post.bloom_pyramid_mips,
                cur_w,
                cur_h,
            );
            const mips = postprocess.clampBloomMips(params.post.bloom_pyramid_mips);
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
                params.post,
                ssao_view,
                params.ssao.enabled,
                params.ssao.debug_mode,
                params.ssao.intensity,
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
