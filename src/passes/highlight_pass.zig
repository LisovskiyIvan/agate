const std = @import("std");
const sokol = @import("sokol");
const sg = sokol.gfx;
const outline_shd = @import("outline_shader");
const blur_shd = @import("glow_blur_shader");
const pp = @import("../postprocess.zig");
const glow_mod = @import("glow_pass.zig");
const math = @import("math");
const Mat4 = math.Mat4;
const Vec3 = math.Vec3;
const Mesh = @import("../mesh.zig").Mesh;
const Vertex = @import("../mesh.zig").Vertex;
const camera_mod = @import("../camera.zig");
const Viewport = camera_mod.Viewport;
const HighlightOptions = @import("../scene/highlight_layer.zig").HighlightOptions;

// Highlight layer v1 render pass: mask-RT inner glow (see
// scene/highlight_layer.zig for the layer design and the v1 non-goals).
//
// Per frame (all staged, never live meshes):
//  1. Mask: the staged items are drawn flat-colored (per-item color x
//     intensity, zero hull expansion) into a half-resolution mask RT under
//     the primary camera's pixel viewport/scissor mapped onto the mask
//     target (PIP-aware: fullscreen viewports map to the full target, so
//     the fullscreen mapping is exact; the blur stages stay
//     fullscreen on their own targets — sokol's beginPass resets
//     viewport+scissor to the full framebuffer). The
//     pipelines reuse the rigid `outline` shader program with front-face
//     culling DISABLED (silhouette-exact for open geometry too) and no
//     depth attachment — the mask is additive, never depth-tested (v1
//     non-goal), so no depth buffer is allocated. Zero new GLSL files.
//  2. Blur: the glow-style separable Gaussian (H then V, same
//     `glow_blur` shader module, sigma texel math, and kernel as GlowPass)
//     widens the mask into the halo. One blur runs per frame with sigma =
//     max over the staged items (documented approximation); per-item
//     color/intensity stay exact (folded into the mask at draw).
//  3. Composite: the parent feeds the blurred view AND the raw mask view
//     into the fullscreen postprocess composite, which adds the inner
//     glow (raw minus blurred, floored at zero, x2 — see
//     highlightInnerGlow in postprocess.zig) after the glow block and
//     before the grading chain (placeholder + zeroed uniforms when
//     inactive, so the off path is bit-identical).
//
// Headless behavior mirrors GlowPass exactly: zero handles fail closed
// (render returns an empty view before any sg.* call), no sg.isvalid gates
// on the fail-closed path. The pass is uniform-only past the mask binds
// (applyUniforms, never sg.updateBuffer), so it records nothing into
// gpu_upload_meter and renderReuse replays stay upload-free by
// construction. Target destruction is immediate (same as
// GlowPass.destroyTargets, context thread only): the retire queue stays
// for game-side removals that can happen off-context (meshes), not for
// pass-owned mask/blur RTs recreated on resize.

// Fullscreen draws one highlight frame issues past the mask fills: the H
// and V blur stages (the mask fills are per-item mesh draws, counted with
// their real draw/tris numbers in HighlightResult).
pub const HIGHLIGHT_BLUR_DRAWS: u32 = 2;

/// Self-contained per-item payload for highlight rendering. Render-owned
/// snapshot only (model matrix, buffer handles, frozen options): no live
/// `*Mesh` escapes the prepare phase. Skinned meshes never produce an
/// item (no skin matrix staging in v1); instanced meshes stage the
/// template proxy only (no per-instance matrices); cutout cards stage the
/// quad proxy (no alpha test in the mask path).
pub const HighlightDrawItem = struct {
    vertex_buffer: sg.Buffer = .{},
    index_buffer: sg.Buffer = .{},
    index_count: u32 = 0,
    model: Mat4 = Mat4.identity,
    /// Staged highlight color (linear RGB + alpha, [0, 1] per channel).
    color: [4]f32 = .{ 1.0, 1.0, 1.0, 1.0 },
    /// Staged blur sigma in highlight-target texels (>= 0).
    blur: f32 = 0.0,
    /// Staged additive scale (>= 0, folded into the mask at draw).
    intensity: f32 = 0.0,
    is_u32: bool = false,
    gpu_pending: bool = false,
    is_visible: bool = true,
    source_uid: u64 = 0,
    source_mesh: u32 = 0,
};

/// Builds a HighlightDrawItem from the live mesh (prepare phase only).
/// Returns null for skinned meshes (v1 non-goal: no skin matrix staging —
/// fail-closed skip, never a bind-pose draw) and for meshes that cannot
/// draw (invisible, gpu-pending, or empty — outline pre-filter precedent).
/// Dead buffer handles stage FINE (outline precedent: makeOutlineDrawItem
/// likewise snapshots first); the mask draw skips them fail-closed
/// (handle-zero + `queryBufferState` epoch guards in renderMask).
/// Instanced meshes stage the template proxy: the mesh world matrix with a
/// single draw, no instance buffer. Pure CPU (no allocation, no GPU
/// calls); the caller resolves `source_mesh` to the mesh-list index
/// (outline stage-2B identity domain) and skips OOB referents before
/// calling (highlights have no latch patch stage, so — unlike outline
/// items — a dead referent never reaches this function).
pub fn makeHighlightDrawItem(mesh: *Mesh, options: HighlightOptions, source_mesh: u32) ?HighlightDrawItem {
    if (mesh.skeleton != null) return null;
    if (!mesh.is_visible or mesh.gpu_pending or mesh.index_count == 0) return null;
    _ = mesh.ensureUid();
    return HighlightDrawItem{
        .vertex_buffer = mesh.vertex_buffer,
        .index_buffer = mesh.index_buffer,
        .index_count = mesh.index_count,
        .model = mesh.getWorldMatrix(),
        .color = options.color,
        .blur = options.blur,
        .intensity = options.intensity,
        .is_u32 = (mesh.index_type == .UINT32),
        .gpu_pending = mesh.gpu_pending,
        .is_visible = mesh.is_visible,
        .source_uid = mesh.uid,
        .source_mesh = source_mesh,
    };
}

/// Mask fill color for one staged item: rgb x intensity (exact — the
/// separable blur kernel is normalized, so folding here equals scaling at
/// composite). Alpha passes through (staged, currently unused by the
/// composite which adds rgb only).
pub fn highlightMaskColor(item: HighlightDrawItem) [4]f32 {
    return .{
        item.color[0] * item.intensity,
        item.color[1] * item.intensity,
        item.color[2] * item.intensity,
        item.color[3],
    };
}

/// Frame blur sigma: max over the staged items (one separable blur runs
/// per frame — documented v1 approximation). Empty set folds to 0.
pub fn highlightFrameSigma(items: []const HighlightDrawItem) f32 {
    var sigma: f32 = 0.0;
    for (items) |it| sigma = @max(sigma, it.blur);
    return sigma;
}

/// Maps a full-resolution primary-camera pixel rect onto the
/// half-resolution mask target (`bloomMipSize` level 0, the exact target
/// `resize` allocates). Pure integer math, headless-safe. Scales with the
/// real target-over-base ratio (truncating, 1px floor) so odd frame sizes
/// stay consistent with the allocated target; degenerate bases fall back
/// to the full target rect (never a zero viewport).
pub fn highlightMaskViewport(rect: Viewport.PixelRect, base_w: i32, base_h: i32) Viewport.PixelRect {
    const size = pp.bloomMipSize(base_w, base_h, 0);
    if (base_w <= 0 or base_h <= 0) return .{ .x = 0, .y = 0, .width = size.w, .height = size.h };
    const bw: i64 = @intCast(base_w);
    const bh: i64 = @intCast(base_h);
    const sw: i64 = @intCast(size.w);
    const sh: i64 = @intCast(size.h);
    return .{
        .x = @intCast(@divTrunc(@as(i64, @intCast(rect.x)) * sw, bw)),
        .y = @intCast(@divTrunc(@as(i64, @intCast(rect.y)) * sh, bh)),
        .width = @max(1, @as(i32, @intCast(@divTrunc(@as(i64, @intCast(rect.width)) * sw, bw)))),
        .height = @max(1, @as(i32, @intCast(@divTrunc(@as(i64, @intCast(rect.height)) * sh, bh)))),
    };
}

/// Shared mask pipeline state: opaque overwrite fill (no blending — the
/// last overlapping item wins the mask, documented), no depth attachment
/// (additive, never depth-tested), no culling (silhouette-exact for open
/// geometry, unlike the inverse-hull outline which culls front faces).
/// Vertex layout mirrors configureOutlineDesc (rigid Mesh vertex buffer,
/// position/normal bound; the hull expansion stays zero via width 0).
pub fn configureHighlightMaskDesc(desc: *sg.PipelineDesc) void {
    desc.depth = .{
        .pixel_format = .NONE,
        .compare = .ALWAYS,
        .write_enabled = false,
    };
    desc.cull_mode = .NONE;
    desc.colors[0].blend.enabled = false;
    desc.layout.buffers[0] = .{ .stride = @sizeOf(Vertex) };
    desc.layout.attrs[outline_shd.ATTR_outline_position] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "position"),
    };
    desc.layout.attrs[outline_shd.ATTR_outline_normal] = .{
        .format = .FLOAT3,
        .offset = @offsetOf(Vertex, "normal"),
    };
}

/// Result of one highlight frame: the blurred halo view for the composite
/// (empty when inactive/failed-closed), the raw unblurred mask view for
/// the inner-glow minuend (same active/empty discipline — the composite
/// reads raw minus blurred), plus the exact mask-stage draw accounting
/// for stats. Fail-closed renders return all zeros.
pub const HighlightResult = struct {
    view: sg.View = .{},
    mask_view: sg.View = .{},
    mask_draws: u32 = 0,
    mask_tris: u32 = 0,
};

pub const HighlightPass = struct {
    mask_image: sg.Image = .{},
    mask_att_view: sg.View = .{},
    mask_tex_view: sg.View = .{},

    blur_images: [2]sg.Image = [_]sg.Image{.{}} ** 2,
    blur_att_views: [2]sg.View = [_]sg.View{.{}} ** 2,
    blur_tex_views: [2]sg.View = [_]sg.View{.{}} ** 2,

    sampler: sg.Sampler = .{},
    mask_pipeline_u16: sg.Pipeline = .{},
    mask_pipeline_u32: sg.Pipeline = .{},
    blur_pipeline: sg.Pipeline = .{},
    mask_shader: sg.Shader = .{},
    blur_shader: sg.Shader = .{},
    /// Fullscreen-quad buffers for the two blur stages (pass-owned, like
    /// GlowPass's quad; the mask stage binds mesh buffers instead).
    blur_quad_vb: sg.Buffer = .{},
    blur_quad_ib: sg.Buffer = .{},

    base_width: i32 = 0,
    base_height: i32 = 0,

    /// VRAM census helper (pure byte math, no GPU calls, headless-safe):
    /// three half-resolution targets (mask + H/V ping-pong) — the same
    /// shape as GlowPass, so the byte math reuses GlowPass.targetBytes
    /// (single formula for both passes; see profiler/snapshot.zig).
    pub fn targetBytes(width: i32, height: i32, bytes_per_pixel: usize) usize {
        return glow_mod.GlowPass.targetBytes(width, height, bytes_per_pixel);
    }

    pub fn init() HighlightPass {
        const smp = sg.makeSampler(.{
            .min_filter = .LINEAR,
            .mag_filter = .LINEAR,
            .wrap_u = .CLAMP_TO_EDGE,
            .wrap_v = .CLAMP_TO_EDGE,
        });

        const mask_fmt = glow_mod.GlowPass.glowPixelFormat();

        const mask_shd_handle = sg.makeShader(outline_shd.outlineShaderDesc(sg.queryBackend()));
        var mask_u16 = sg.PipelineDesc{ .shader = mask_shd_handle, .index_type = .UINT16, .sample_count = 1 };
        configureHighlightMaskDesc(&mask_u16);
        mask_u16.colors[0].pixel_format = mask_fmt;
        const mask_pip_u16 = sg.makePipeline(mask_u16);

        var mask_u32 = sg.PipelineDesc{ .shader = mask_shd_handle, .index_type = .UINT32, .sample_count = 1 };
        configureHighlightMaskDesc(&mask_u32);
        mask_u32.colors[0].pixel_format = mask_fmt;
        const mask_pip_u32 = sg.makePipeline(mask_u32);

        const blur_shd_handle = sg.makeShader(blur_shd.glowBlurShaderDesc(sg.queryBackend()));
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
        blur_desc.colors[0].pixel_format = mask_fmt;
        blur_desc.layout.buffers[0] = .{ .stride = 4 * @sizeOf(f32) };
        // Fullscreen-quad bindings for the blur stages (XY, UV). The mask
        // stage binds mesh vertex buffers instead — never this quad.
        const quad_vertices = [_]f32{
            -1.0, -1.0, 0.0, 0.0,
            1.0,  -1.0, 1.0, 0.0,
            1.0,  1.0,  1.0, 1.0,
            -1.0, 1.0,  0.0, 1.0,
        };
        const quad_indices = [_]u16{ 0, 1, 2, 0, 2, 3 };
        const quad_vb = sg.makeBuffer(.{ .data = sg.asRange(&quad_vertices) });
        const quad_ib = sg.makeBuffer(.{ .usage = .{ .index_buffer = true }, .data = sg.asRange(&quad_indices) });
        blur_desc.layout.attrs[blur_shd.ATTR_glow_blur_position] = .{ .format = .FLOAT2, .offset = 0 };
        blur_desc.layout.attrs[blur_shd.ATTR_glow_blur_texcoord0] = .{ .format = .FLOAT2, .offset = 2 * @sizeOf(f32) };
        const blur_pip = sg.makePipeline(blur_desc);

        return .{
            .sampler = smp,
            .mask_pipeline_u16 = mask_pip_u16,
            .mask_pipeline_u32 = mask_pip_u32,
            .blur_pipeline = blur_pip,
            .mask_shader = mask_shd_handle,
            .blur_shader = blur_shd_handle,
            // The blur quad buffers are pass-owned like GlowPass's; the
            // mask stage never touches them.
            .blur_quad_vb = quad_vb,
            .blur_quad_ib = quad_ib,
        };
    }

    pub fn resize(self: *HighlightPass, width: i32, height: i32) void {
        if (width <= 0 or height <= 0) return;
        if (!sg.isvalid()) return;
        if (self.base_width == width and self.base_height == height) return;

        self.destroyTargets();

        const mask_fmt = glow_mod.GlowPass.glowPixelFormat();
        const size = pp.bloomMipSize(width, height, 0);

        // Store-first rollback: every handle lands in the struct before its
        // state check, so a single destroyTargets frees the failed handle
        // plus all partial handles (views before images). A failed resize
        // leaves zero ids and a zero base size, and render() reports empty.
        self.mask_image = sg.makeImage(.{
            .usage = .{ .color_attachment = true },
            .width = size.w,
            .height = size.h,
            .pixel_format = mask_fmt,
            .sample_count = 1,
        });
        if (sg.queryImageState(self.mask_image) != .VALID) {
            self.destroyTargets();
            return;
        }
        self.mask_att_view = sg.makeView(.{
            .color_attachment = .{ .image = self.mask_image },
        });
        if (sg.queryViewState(self.mask_att_view) != .VALID) {
            self.destroyTargets();
            return;
        }
        self.mask_tex_view = sg.makeView(.{
            .texture = .{ .image = self.mask_image },
        });
        if (sg.queryViewState(self.mask_tex_view) != .VALID) {
            self.destroyTargets();
            return;
        }

        for (0..2) |i| {
            self.blur_images[i] = sg.makeImage(.{
                .usage = .{ .color_attachment = true },
                .width = size.w,
                .height = size.h,
                .pixel_format = mask_fmt,
                .sample_count = 1,
            });
            if (sg.queryImageState(self.blur_images[i]) != .VALID) {
                self.destroyTargets();
                return;
            }
            self.blur_att_views[i] = sg.makeView(.{
                .color_attachment = .{ .image = self.blur_images[i] },
            });
            if (sg.queryViewState(self.blur_att_views[i]) != .VALID) {
                self.destroyTargets();
                return;
            }
            self.blur_tex_views[i] = sg.makeView(.{
                .texture = .{ .image = self.blur_images[i] },
            });
            if (sg.queryViewState(self.blur_tex_views[i]) != .VALID) {
                self.destroyTargets();
                return;
            }
        }

        self.base_width = width;
        self.base_height = height;
    }

    fn blurStage(
        self: *HighlightPass,
        src_view: sg.View,
        direction: f32,
        sigma: f32,
        texel_w: f32,
        texel_h: f32,
        slot: usize,
    ) void {
        var pass = sg.Pass{
            .action = .{
                .colors = [_]sg.ColorAttachmentAction{
                    .{ .load_action = .DONTCARE },
                } ++ [_]sg.ColorAttachmentAction{.{}} ** 7,
            },
        };
        pass.attachments.colors[0] = self.blur_att_views[slot];
        sg.beginPass(pass);

        sg.applyPipeline(self.blur_pipeline);
        var bind = sg.Bindings{};
        bind.vertex_buffers[0] = self.blur_quad_vb;
        bind.index_buffer = self.blur_quad_ib;
        bind.views[blur_shd.VIEW_src_tex] = src_view;
        bind.samplers[blur_shd.SMP_smp] = self.sampler;
        sg.applyBindings(bind);

        const fs_params = blur_shd.FsParams{
            .texel = .{ texel_w, texel_h, 0.0, 0.0 },
            .params = .{ direction, sigma, 0.0, 0.0 },
        };
        sg.applyUniforms(blur_shd.UB_fs_params, sg.asRange(&fs_params));
        sg.draw(0, 6, 1);
        sg.endPass();
    }

    /// Draws the staged items flat-colored into the mask target (zero hull
    /// expansion: width 0 collapses the outline offset exactly). `mask_rect`
    /// is the primary-camera pixel rect mapped onto this half-res target
    /// (see highlightMaskViewport): the mask draws under the same
    /// viewport/scissor the main pass used, so PIP/sub-viewports stay
    /// aligned with the scene color. No restore needed: sokol's beginPass
    /// resets viewport+scissor to the full framebuffer (sokol_gfx.h), so
    /// the following blur passes (separate beginPass calls) are fullscreen
    /// automatically. Returns
    /// the mask-stage draw accounting. Skips fail-closed items (invisible,
    /// gpu-pending, empty, dead handles — outline renderItems precedent,
    /// plus the `queryBufferState` epoch guard when a context is live).
    fn renderMask(self: *HighlightPass, view_proj: Mat4, items: []const HighlightDrawItem, mask_rect: Viewport.PixelRect) HighlightResult {
        var out = HighlightResult{ .view = self.mask_tex_view, .mask_view = self.mask_tex_view };
        var pass = sg.Pass{
            .action = .{
                .colors = [_]sg.ColorAttachmentAction{
                    .{ .load_action = .CLEAR, .clear_value = .{ .r = 0, .g = 0, .b = 0, .a = 0 } },
                } ++ [_]sg.ColorAttachmentAction{.{}} ** 7,
            },
        };
        pass.attachments.colors[0] = self.mask_att_view;
        sg.beginPass(pass);
        // Viewport-align with the primary camera (frame_render applies the
        // same rect, fullscreen, to the main pass): mask pixels land where
        // the mesh's scene pixels are, so the inner-glow composite samples
        // aligned raw/blurred/scene uvs.
        sg.applyViewport(mask_rect.x, mask_rect.y, mask_rect.width, mask_rect.height, true);
        sg.applyScissorRect(mask_rect.x, mask_rect.y, mask_rect.width, mask_rect.height, true);

        for (items) |item| {
            if (!item.is_visible or item.gpu_pending or item.index_count == 0) continue;
            if (item.vertex_buffer.id == 0 or item.index_buffer.id == 0) continue;
            const pip = if (item.is_u32) self.mask_pipeline_u32 else self.mask_pipeline_u16;
            if (pip.id == 0) continue;
            if (sg.isvalid()) {
                if (sg.queryBufferState(item.vertex_buffer) != .VALID) continue;
                if (sg.queryBufferState(item.index_buffer) != .VALID) continue;
            }

            var bind = sg.Bindings{};
            bind.vertex_buffers[0] = item.vertex_buffer;
            bind.index_buffer = item.index_buffer;
            sg.applyPipeline(pip);
            sg.applyBindings(bind);

            const mvp = Mat4.mul(view_proj, item.model);
            const fill = highlightMaskColor(item);
            const vs_params = outline_shd.VsParams{
                .mvp = mvp,
                .model = item.model,
                .color = fill,
                // Zero width: the hull offset collapses exactly (the mask
                // is the unexpanded silhouette); 1x1 viewport is exact for
                // the same reason and can never divide by zero.
                .params = .{ 0.0, 1.0, 1.0, 0.0 },
            };
            sg.applyUniforms(outline_shd.UB_vs_params, sg.asRange(&vs_params));
            sg.draw(0, item.index_count, 1);
            out.mask_draws += 1;
            out.mask_tris += item.index_count / 3;
        }
        sg.endPass();
        return out;
    }

    // Builds the halo for the staged items and returns the blurred view
    // plus mask-stage accounting (empty/zero when inactive so the caller
    // falls back to the placeholder — bit-identical composite). Guard
    // order mirrors GlowPass.render: pipelines first, then items, then
    // size — all before any sg.* call or resize. `viewport` is the
    // full-resolution primary-camera pixel rect (frame_render's rect for
    // the primary view); the mask pass maps it onto its half-res target
    // while the blur stages stay fullscreen on theirs.
    pub fn render(
        self: *HighlightPass,
        view_proj: Mat4,
        items: []const HighlightDrawItem,
        base_w: i32,
        base_h: i32,
        viewport: Viewport.PixelRect,
    ) HighlightResult {
        if (self.mask_pipeline_u16.id == 0 or self.mask_pipeline_u32.id == 0 or self.blur_pipeline.id == 0) return .{};
        if (items.len == 0) return .{};
        if (base_w <= 0 or base_h <= 0) return .{};

        self.resize(base_w, base_h);
        if (self.base_width != base_w or self.base_height != base_h) return .{};
        if (self.mask_image.id == 0) return .{};

        var out = self.renderMask(view_proj, items, highlightMaskViewport(viewport, base_w, base_h));

        // Separable blur, H into slot 0 then V into slot 1 (GlowPass
        // precedent). Frame-global sigma = max over the staged items.
        const size = pp.bloomMipSize(base_w, base_h, 0);
        const texel_w: f32 = 1.0 / @as(f32, @floatFromInt(size.w));
        const texel_h: f32 = 1.0 / @as(f32, @floatFromInt(size.h));
        const sigma = highlightFrameSigma(items);
        self.blurStage(self.mask_tex_view, 0.0, sigma, texel_w, texel_h, 0);
        self.blurStage(self.blur_tex_views[0], 1.0, sigma, texel_w, texel_h, 1);

        out.view = self.blur_tex_views[1];
        out.mask_view = self.mask_tex_view;
        return out;
    }

    fn destroyTargets(self: *HighlightPass) void {
        if (self.mask_att_view.id != 0) sg.destroyView(self.mask_att_view);
        if (self.mask_tex_view.id != 0) sg.destroyView(self.mask_tex_view);
        if (self.mask_image.id != 0) sg.destroyImage(self.mask_image);
        self.mask_image = .{};
        self.mask_att_view = .{};
        self.mask_tex_view = .{};
        for (0..2) |i| {
            if (self.blur_att_views[i].id != 0) sg.destroyView(self.blur_att_views[i]);
            if (self.blur_tex_views[i].id != 0) sg.destroyView(self.blur_tex_views[i]);
            if (self.blur_images[i].id != 0) sg.destroyImage(self.blur_images[i]);
            self.blur_images[i] = .{};
            self.blur_att_views[i] = .{};
            self.blur_tex_views[i] = .{};
        }
        self.base_width = 0;
        self.base_height = 0;
    }

    pub fn deinit(self: *HighlightPass) void {
        self.destroyTargets();
        sg.destroySampler(self.sampler);
        sg.destroyPipeline(self.mask_pipeline_u16);
        sg.destroyPipeline(self.mask_pipeline_u32);
        sg.destroyPipeline(self.blur_pipeline);
        if (self.mask_shader.id != 0) sg.destroyShader(self.mask_shader);
        if (self.blur_shader.id != 0) sg.destroyShader(self.blur_shader);
        self.mask_shader = .{};
        self.blur_shader = .{};
        sg.destroyBuffer(self.blur_quad_vb);
        sg.destroyBuffer(self.blur_quad_ib);
    }
};
